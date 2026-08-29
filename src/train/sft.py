"""v6 SFT — instruct backbone + Gemma-4-it chat template + natural-language
instruction prompt. Matches EAST's actual training format (Fu et al. 2025 Fig 18).

Key differences from src/train/sft.py:
  - Backbone: gemma-4-E2B-it (instruction-tuned, NOT base)
  - Tokenizer: _archive/results/v6b_gemma_2b/tokenizer-extended-v6 (EOR + EOW only; latency
    tokens dropped since latency is now natural language in the user turn)
  - Prompt format: chat template with system + user instruction + assistant
      user: "Translate the following text from {SRC_NAME} into {TGT_NAME}
             with {LATENCY} latency."
      assistant: interleaved EAST-format chunks (src<EOR>tgt<EOW>...)
  - Loss masking: only assistant-turn positions get gradient (matches
    standard instruct-SFT). System + user prompts are input-only.
  - Latency at inference gets 2 free interpolated points via natural-language
    interpolation: prompt "low-medium" or "medium-high".

Consumes: sft_dataset_multilingual_v5.json (base rows only — dropping
augmented rows since the corpus is already balanced at ~7K/direction).

Same recipe as v5: bf16, lr 2e-5, effective batch 64, 1 epoch, descriptive
init on EOR/EOW, Test B α=5 on EAST-special label positions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
import torch.nn.functional as F

from src.annotator.east_format import (
    END_OF_READ, END_OF_WRITE,
    LATENCY_NL, DEFAULT_SYSTEM_PROMPT,
    build_user_instruction, build_assistant_body, build_chat_prompt_v6,
    EastRow, parse_row, lang_name,
)
from src.constants import REPO_ROOT

DEFAULT_MODEL = "/g/data/po67/dipankar/models/gemma-4-E2B-it"
DEFAULT_TOKENIZER = REPO_ROOT / "results" / "phase2" / "tokenizer-extended-v6"

# Splice placeholder: a rare ASCII sentinel unique enough to survive chat-
# template rendering unchanged, and easy to split on. Not fed to the model.
_SPLICE_PLACEHOLDER = "PLACEHOLDER_ASSISTANT_BODY"


def load_rows(corpus_file: Path, use_augmentation: bool = False) -> List[Dict]:
    with open(corpus_file) as f:
        rows = json.load(f)
    if not use_augmentation:
        n_before = len(rows)
        rows = [r for r in rows if not ((r.get("_annotator_meta") or {}).get("augmented_from_base") or False)]
        print(f"Filtered augmentation: kept {len(rows)}/{n_before} base rows.", flush=True)
    return rows


def render_chat_open_close_ids(tokenizer, src_lang: str, tgt_lang: str, latency: str):
    """Render the chat template with a placeholder assistant body, split
    the string on the placeholder, and tokenize each half independently.

    Returns (prefix_ids, suffix_ids) — the sequence around the assistant body.
    Bypasses the string-round-trip artifact: the assistant body will be
    spliced in as chunk_ids directly, byte-exact from the annotator.
    """
    user_instr = build_user_instruction(src_lang, tgt_lang, latency)
    messages_with_sys = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_instr},
        {"role": "assistant", "content": _SPLICE_PLACEHOLDER},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages_with_sys, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        messages_no_sys = [
            {"role": "user", "content": DEFAULT_SYSTEM_PROMPT + "\n\n" + user_instr},
            {"role": "assistant", "content": _SPLICE_PLACEHOLDER},
        ]
        text = tokenizer.apply_chat_template(
            messages_no_sys, tokenize=False, add_generation_prompt=False
        )
    if _SPLICE_PLACEHOLDER not in text:
        raise RuntimeError(
            f"chat template rendering stripped placeholder for {src_lang}->{tgt_lang} @{latency}; "
            "cannot recover splice boundary"
        )
    open_str, close_str = text.split(_SPLICE_PLACEHOLDER)
    prefix_ids = tokenizer(open_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(close_str, add_special_tokens=False)["input_ids"]
    return prefix_ids, suffix_ids


def build_row_ids(row: Dict, tokenizer, max_length: int,
                   eor_id: int, eow_id: int,
                   prefix_cache: dict, suffix_cache: dict):
    """Build input_ids + labels for one row via direct-ids splice.

    input_ids = prefix_ids + Σ (source_chunk_ids[k] + [EOR] + target_chunk_ids[k] + [EOW]) + suffix_ids
    labels    = [-100] * len(prefix_ids) + <rest>

    This bypasses the string round-trip that dropped 40-47% of AR/VI rows
    (leading-space retokenization changes segmentation for RTL and non-Latin
    scripts). Since the chunk_ids come from the annotator's canonical
    tokenization of the full source/target, the resulting sequence is
    byte-identical to what streaming inference will produce word-by-word.
    """
    key = (row["src_lang"], row["tgt_lang"], row["latency"])
    if key not in prefix_cache:
        prefix_ids, suffix_ids = render_chat_open_close_ids(
            tokenizer, row["src_lang"], row["tgt_lang"], row["latency"]
        )
        prefix_cache[key] = prefix_ids
        suffix_cache[key] = suffix_ids
    prefix_ids = prefix_cache[key]
    suffix_ids = suffix_cache[key]

    body_ids: List[int] = []
    for src_ids, tgt_ids in zip(row["source_chunk_ids"], row["target_chunk_ids"]):
        body_ids.extend(src_ids)
        body_ids.append(eor_id)
        body_ids.extend(tgt_ids)
        body_ids.append(eow_id)

    input_ids = prefix_ids + body_ids + suffix_ids
    if len(input_ids) > max_length:
        # Truncate from the RIGHT (drop tail of body / suffix). Prefix must
        # survive intact — otherwise the model doesn't see the direction/latency
        # instruction. Also we truncate at a chunk boundary if possible.
        keep = max_length
        input_ids = input_ids[:keep]
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def snapshot_special_token_embeddings(model, tokenizer):
    out = {}
    embed = model.get_input_embeddings()
    for tok in [END_OF_READ, END_OF_WRITE]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        out[tok] = embed.weight[tok_id].detach().float().cpu().clone()
    return out


def apply_descriptive_init(model, tokenizer, noise_sigma: float = 0.01):
    """Init EOR/EOW embeddings as mean of descriptive-word embeddings + noise.
    Skips latency tokens — they're natural language in v6, not vocab entries."""
    embed = model.get_input_embeddings()
    ref = {
        END_OF_READ:  ["end", "of", "read"],
        END_OF_WRITE: ["end", "of", "write"],
    }
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        anchor = []
    else:
        anchor = [eos_id]
    print(f"\nDescriptive-init on 2 EAST-special embeddings (noise σ={noise_sigma}, anchor_ids={anchor})", flush=True)
    for special, words in ref.items():
        tid = tokenizer.convert_tokens_to_ids(special)
        ref_ids = []
        for w in words:
            wids = tokenizer(w, add_special_tokens=False)["input_ids"]
            if wids:
                ref_ids.append(wids[0])
        ref_ids = ref_ids + anchor
        with torch.no_grad():
            ref_embs = embed.weight[torch.tensor(ref_ids)]
            mean_emb = ref_embs.mean(0)
            noise = torch.randn_like(mean_emb) * noise_sigma
            embed.weight[tid] = (mean_emb + noise).to(embed.weight.dtype)
        print(f"  {special!r} (id {tid}) <- mean of ref ids {ref_ids} + N(0, {noise_sigma}^2)", flush=True)


class WeightedSFTTrainerV6:
    """Wraps HF Trainer with special-token loss upweighting (Test B α)."""
    pass  # placeholder; using trl.SFTTrainer subclass below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_file", type=Path, required=True)
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--use_augmentation", action="store_true",
                    help="Include aug2/aug4 rows. Default OFF — the 69K base rows are balanced.")
    ap.add_argument("--per_device_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--num_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--early_stopping_patience", type=int, default=999999)
    ap.add_argument("--descriptive_init", action="store_true")
    ap.add_argument("--special_token_loss_weight", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_smoke_rows", type=int, default=-1,
                    help="If > 0, only train on this many rows (fast smoke).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_dir}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments

    print("Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    print(f"  vocab size: {len(tok):,}", flush=True)
    assert tok.chat_template is not None, "tokenizer missing chat_template — wrong dir?"

    print(f"Loading model from {args.model_path} ...", flush=True)
    cfg = AutoConfig.from_pretrained(args.model_path)
    mtype = getattr(cfg, "model_type", "")
    if mtype == "gemma3n":
        from transformers import Gemma3nForCausalLM
        model = Gemma3nForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    orig_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"  loaded; original embed rows: {orig_vocab:,}", flush=True)

    if len(tok) > orig_vocab:
        print(f"Resizing embeddings: +{len(tok) - orig_vocab} rows (mean-cov init)", flush=True)
        model.resize_token_embeddings(len(tok))

    if args.descriptive_init:
        apply_descriptive_init(model, tok)

    before = snapshot_special_token_embeddings(model, tok)

    print("\nLoading + rendering training rows ...", flush=True)
    rows = load_rows(args.corpus_file, use_augmentation=args.use_augmentation)
    if args.n_smoke_rows > 0:
        rows = rows[: args.n_smoke_rows]
        print(f"  SMOKE mode: capped to {len(rows)} rows", flush=True)
    print(f"  {len(rows):,} rows to train on", flush=True)

    # Build input_ids + labels for each row via DIRECT-IDS SPLICE
    # (2026-08-22 v6b fix: bypass the string round-trip that dropped 40-47%
    # of AR/VI rows via the leading-space retokenization mismatch).
    print("Building input_ids from chunk_ids (direct-splice; no string round-trip) ...", flush=True)
    from datasets import Dataset
    eor_id = tok.convert_tokens_to_ids(END_OF_READ)
    eow_id = tok.convert_tokens_to_ids(END_OF_WRITE)
    prefix_cache: Dict = {}
    suffix_cache: Dict = {}
    features = []
    for i, r in enumerate(rows):
        feat = build_row_ids(
            r, tok, args.max_length,
            eor_id=eor_id, eow_id=eow_id,
            prefix_cache=prefix_cache, suffix_cache=suffix_cache,
        )
        feat["index"] = r["index"]
        features.append(feat)
        if i < 2:
            decoded = tok.decode(feat["input_ids"], skip_special_tokens=False)
            print(f"  Row {i} preview (first 300 chars of decoded):", flush=True)
            print(f"    {decoded[:300]!r}", flush=True)
            print(f"    n_ids={len(feat['input_ids'])}, "
                  f"labels_masked={sum(1 for x in feat['labels'] if x == -100)}", flush=True)
    ds = Dataset.from_list(features)

    # Group split by index to avoid leakage
    train_ds = ds
    eval_ds = None
    if args.val_frac > 0 and len(ds) > 20:
        import random as _random
        all_indices = sorted(set(ds["index"]))
        _rng = _random.Random(args.seed)
        _rng.shuffle(all_indices)
        n_eval_idx = max(1, int(len(all_indices) * args.val_frac))
        eval_idx_set = set(all_indices[:n_eval_idx])
        train_idx_set = set(all_indices[n_eval_idx:])
        train_ds = ds.filter(lambda r: r["index"] in train_idx_set)
        eval_ds = ds.filter(lambda r: r["index"] in eval_idx_set)
        print(f"  Split: {len(all_indices)} unique idx -> "
              f"train {len(train_idx_set)} idx ({len(train_ds)} rows), "
              f"eval {len(eval_idx_set)} idx ({len(eval_ds)} rows)", flush=True)

    # Drop the index column from the datasets we feed to the trainer
    train_ds = train_ds.remove_columns(["index"])
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns(["index"])

    # Trainer
    # 2026-08-22: wire best-model-at-end selection (HF requires save_strategy ==
    # eval_strategy and save_steps % eval_steps == 0 for `load_best_model_at_end`).
    # We evaluate + save at the same cadence and let HF surface the best-by-eval-loss
    # checkpoint at the end of training. save_total_limit=1 keeps only best after
    # cleanup (last is discarded once best is loaded). Intermediate checkpoint-*
    # dirs are then removed post-train.
    from transformers import Trainer
    if eval_ds is not None:
        # Align eval + save cadence for load_best_model_at_end
        eval_and_save_steps = args.eval_steps
        strategy_kwargs = dict(
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_and_save_steps,
            save_steps=eval_and_save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        strategy_kwargs = dict(
            eval_strategy="no",
            save_strategy="steps",
            save_steps=200,
        )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        seed=args.seed,
        report_to=[],
        save_total_limit=1,
        remove_unused_columns=False,
        **strategy_kwargs,
    )

    def collate(batch):
        # Pad to longest in batch
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id or tok.eos_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # Test B — special-token loss upweighting
    class WeightedTrainer(Trainer):
        def __init__(self, *targs, special_ids=None, alpha=1.0, **kw):
            super().__init__(*targs, **kw)
            self._special_ids = torch.tensor(list(special_ids or []), dtype=torch.long)
            self._alpha = float(alpha)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # standard next-token shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)
            # weight: alpha on positions where the LABEL is EOR/EOW; 1 elsewhere
            if self._alpha != 1.0 and self._special_ids.numel() > 0:
                sid = self._special_ids.to(flat_labels.device)
                is_special = torch.isin(flat_labels, sid)
                weights = torch.ones_like(per_token)
                weights[is_special] = self._alpha
                mask = (flat_labels != -100).float()
                weights = weights * mask
                loss = (per_token * weights).sum() / mask.sum().clamp(min=1.0)
            else:
                mask = (flat_labels != -100).float()
                loss = (per_token * mask).sum() / mask.sum().clamp(min=1.0)
            return (loss, outputs) if return_outputs else loss

    special_ids = [tok.convert_tokens_to_ids(END_OF_READ), tok.convert_tokens_to_ids(END_OF_WRITE)]
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        special_ids=special_ids,
        alpha=args.special_token_loss_weight,
    )

    print("\nStarting training ...", flush=True)
    t0 = time.time()
    train_result = trainer.train()
    wall = time.time() - t0

    print(f"\nTraining complete in {wall:.1f}s")
    print(f"  final train loss: {train_result.training_loss:.4f}")
    # With load_best_model_at_end=True, HF has already reloaded the best-by-
    # eval-loss checkpoint into `model`. Log the surfacing so we know which
    # step was picked.
    if eval_ds is not None:
        try:
            best_ckpt = trainer.state.best_model_checkpoint
            best_metric = trainer.state.best_metric
            print(f"  best checkpoint (eval_loss): {best_ckpt} @ eval_loss={best_metric}", flush=True)
        except Exception as e:
            print(f"  (couldn't read best_model_checkpoint: {e})", flush=True)

    # Save best-selected model as `final/`
    final_dir = args.output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))

    # Clean up all intermediate checkpoint-* dirs — we only need `final/`.
    import shutil as _shutil
    for ckpt_dir in args.output_dir.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            print(f"  cleanup: rm -rf {ckpt_dir}", flush=True)
            _shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Embedding deltas
    after = snapshot_special_token_embeddings(model, tok)
    deltas = {t: float((before[t] - after[t]).norm().item()) for t in before}

    # Deep-copy config dict (vars(args) shares __dict__ with args → mutating
    # would clobber args.output_dir which we still need for the write path).
    config_copy = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "config": config_copy,
        "train_metrics": {
            "wall_time_sec": wall,
            "train_loss": float(train_result.training_loss),
        },
        "n_train_rows": len(train_ds),
        "n_eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "special_token_embedding_deltas": deltas,
    }
    summary_path = args.output_dir / "sft_v6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {args.output_dir}/sft_v6_summary.json", flush=True)


if __name__ == "__main__":
    main()
