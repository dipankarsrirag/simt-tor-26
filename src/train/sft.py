"""
SFT wrapper for Phase 2 (EXPERIMENTS.md §Primary result).

Trains Gemma-4-E2B on EAST-interleaved training strings using trl.SFTTrainer.
Applies to both matched conditions:
  * Condition A (baseline)   — chunks from the shipped SiMT-660K GPT-4 tags.
  * Condition B (ours)       — chunks from our annotator's commit points.

The only thing that differs between the two conditions is where the
source_chunks / target_chunks come from. Everything downstream — the
interleave format, tokenizer, loss recipe, hyperparameters — is identical.

Loss recipe (EAST §3.2). Cross-entropy over ALL sequence positions —
source tokens, target tokens, and special tokens (<|end-of-read|>,
<|end-of-write|>, latency indicators). This is an intentional break from
Wang et al. 2024's target-only masking, because the read/write decision
is itself what we train the model to predict. Implemented by NOT using
`DataCollatorForCompletionOnlyLM` and NOT setting `completion_only_loss`.

Tokenizer. Must be the extended one from
`scripts/phase2_prepare_tokenizer.py` — has the 5 EAST special tokens
added at ids 262144..262148. Loading from `MODEL_BASE/gemma-4-E2B/`
directly will tokenize the special tokens as multi-piece garbage.
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

from src.annotator.east_format import SPECIAL_TOKENS, EastRow, interleave
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"
EXTENDED_TOKENIZER_DIR = REPO_ROOT / "results" / "phase2" / "tokenizer-extended"


def build_east_string(row: Dict[str, Any]) -> str:
    """From a corpus row (which contains source/target/source_chunks/target_chunks/latency)
    produce the EAST-interleaved training string ready for tokenisation."""
    er = EastRow(
        source=row["source"],
        target=row["target"],
        src_lang=row["src_lang"],
        tgt_lang=row["tgt_lang"],
        latency=row["latency"],
        source_chunks=list(row["source_chunks"]),
        target_chunks=list(row["target_chunks"]),
    )
    return interleave(er)


def pick_latency_balanced(rows, n_total: int, seed: int, max_src_tokens: int, tokenizer):
    """Latency-balanced sample matching EAST Fig. 6 protocol. Same seed
    used across annotation and SFT keeps A-vs-B comparable at every step."""
    import random
    rng = random.Random(seed)
    by_lat: Dict[str, List[Dict]] = {}
    for r in rows:
        by_lat.setdefault(r["latency"], []).append(r)
    per = n_total // 3
    picked = []
    for lat in ["low", "medium", "high"]:
        pool = by_lat.get(lat, [])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    remainder = n_total - len(picked)
    if remainder > 0:
        picked.extend(by_lat["medium"][per : per + remainder])
    # Length filter (matches phase1_tau_sweep protocol so future OT
    # annotation on the same indices is guaranteed to accept them).
    kept = []
    for r in picked:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src <= max_src_tokens:
            kept.append(r)
    return kept


def load_corpus_subset(
    n_sentences: int, seed: int, max_src_tokens: int,
    tokenizer, indices_file: Optional[Path] = None,
    corpus_file: Optional[Path] = None,
) -> List[Dict]:
    """Load and filter the training subset. Same protocol as phase1_tau_sweep
    so annotation and SFT operate on the same sentences (matched A vs B).

    If `corpus_file` is set, load from that path instead of the shipped
    SiMT-660K corpus — used by cond-B SFT to consume our-annotator chunks
    (built by scripts/phase2_build_condB_dataset.py).
    """
    corpus_path = corpus_file or CORPUS
    print(f"Loading corpus from {corpus_path}", flush=True)
    with open(corpus_path) as f:
        rows = json.load(f)
    print(f"Corpus: {len(rows):,} rows", flush=True)

    if indices_file is not None:
        spec = json.loads(indices_file.read_text())
        wanted = set(spec["indices"])
        by_idx = {r["index"]: r for r in rows}
        picks = [by_idx[i] for i in sorted(wanted) if i in by_idx]
        print(f"Using {len(picks)} indices from {indices_file}", flush=True)
    else:
        picks = pick_latency_balanced(
            rows, n_sentences, seed, max_src_tokens, tokenizer
        )

    kept = []
    for r in picks:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src > max_src_tokens:
            continue
        # Chunk-count mismatch would trip interleave() — pre-skip.
        if len(r.get("source_chunks", [])) != len(r.get("target_chunks", [])):
            continue
        if not r.get("source_chunks"):
            continue
        kept.append(r)
    print(f"Kept {len(kept)} / {len(picks)} after length + chunk-count filters", flush=True)
    return kept


def build_dataset(rows: List[Dict]) -> "Dataset":
    from datasets import Dataset
    return Dataset.from_dict({
        "text": [build_east_string(r) for r in rows],
        "index": [r["index"] for r in rows],
        "latency": [r["latency"] for r in rows],
    })


def load_model_and_tokenizer(model_path: str, tokenizer_dir: Path, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading extended tokenizer from {tokenizer_dir} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        # Use EOS as pad — standard for causal LM SFT.
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab size: {len(tokenizer):,}  (loaded in {time.time()-t0:.1f}s)", flush=True)

    print(f"Loading model from {model_path} ...", flush=True)
    t0 = time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch_dtype, low_cpu_mem_usage=True
    )
    orig_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"  loaded in {time.time()-t0:.1f}s; original embedding rows: {orig_vocab:,}", flush=True)

    # Resize embeddings only if the tokenizer added tokens. Rely on
    # transformers's default `mean_resizing=True`: new rows are drawn from a
    # multivariate-normal with the mean and covariance of the existing rows.
    # DO NOT set them all to the plain mean — that collapses all 5 EAST
    # tokens to an identical starting point in embedding space, and the LM
    # head can't distinguish them (bug fix 2026-08-16; previous version
    # produced special-token loss ~11.9 nats after 1 epoch on 2K rows).
    n_new = len(tokenizer) - orig_vocab
    if n_new > 0:
        print(f"Resizing embeddings: +{n_new} rows (mean-covariance init via transformers default)", flush=True)
        model.resize_token_embeddings(len(tokenizer))
    elif n_new < 0:
        raise RuntimeError(f"tokenizer smaller than model embeddings ({len(tokenizer)} < {orig_vocab})")
    else:
        print(f"  (no resize needed; sizes already match)", flush=True)
    return model, tokenizer


def snapshot_special_token_embeddings(model, tokenizer):
    """Return a dict {token: (input_emb_snapshot, output_emb_snapshot)} for the
    5 EAST special tokens. Used to prove they moved during the toy SFT."""
    snaps = {}
    in_emb = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings().weight
    same = out_emb is in_emb
    for t in SPECIAL_TOKENS:
        idx = tokenizer.encode(t, add_special_tokens=False)[0]
        snaps[t] = {
            "id": int(idx),
            "in_emb": in_emb[idx].detach().float().cpu().clone(),
            "out_emb": None if same else out_emb[idx].detach().float().cpu().clone(),
        }
    return snaps


def compare_special_token_embeddings(before, after, tokenizer):
    """Print per-token L2 delta on input/output embeddings — if all ~0, the
    special tokens didn't train (loss-mask problem)."""
    print("\nSpecial-token embedding movement (L2 norm of delta):", flush=True)
    for t, b in before.items():
        a = after[t]
        in_delta = (a["in_emb"] - b["in_emb"]).norm().item()
        out_delta_str = "tied"
        if b["out_emb"] is not None:
            out_delta = (a["out_emb"] - b["out_emb"]).norm().item()
            out_delta_str = f"{out_delta:.4f}"
        print(f"  {t!r:>22s} (id {b['id']}): in_emb Δ={in_delta:.4f}  out_emb Δ={out_delta_str}", flush=True)


def sample_generations(model, tokenizer, rows: List[Dict], n: int, max_new_tokens: int):
    """Post-training smoke: given a raw source prefix, generate and print
    whether the model emits any EAST special tokens in the continuation."""
    model.eval()
    print(f"\nPost-train generations ({n} samples, max_new_tokens={max_new_tokens}):", flush=True)
    for r in rows[:n]:
        # Feed the latency indicator + partial source; see if the model emits <|eor|>.
        from src.annotator.east_format import LATENCY_TOKENS
        prompt = f"{LATENCY_TOKENS[r['latency']]} {r['source']}"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
        has_eor = "<|end-of-read|>" in gen
        has_eow = "<|end-of-write|>" in gen
        print(f"\n  idx={r['index']} lat={r['latency']} eor={has_eor} eow={has_eow}", flush=True)
        print(f"    prompt: {prompt[:80]!r}{'...' if len(prompt) > 80 else ''}", flush=True)
        print(f"    gen:    {gen[:250]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--tokenizer_dir", type=Path, default=EXTENDED_TOKENIZER_DIR)
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Where trainer writes checkpoints + logs.")
    ap.add_argument("--n_sentences", type=int, default=2000)
    ap.add_argument("--indices_file", type=Path, default=None)
    ap.add_argument("--corpus_file", type=Path, default=None,
                    help="Override the shipped SiMT-660K corpus. Used for "
                         "cond-B SFT to consume our-annotator chunks (see "
                         "scripts/phase2_build_condB_dataset.py).")
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=1024)

    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="If > 0, overrides num_epochs. Used by toy runs.")
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=200)

    # Early-stopping — enabled by default at scale (>= 5K rows).
    # Split off val_frac of rows for eval; run eval every eval_steps; stop
    # if eval_loss doesn't improve by early_stopping_threshold within
    # early_stopping_patience consecutive evals. Loads best-eval checkpoint
    # at end (so `final/` is the best model, not the last).
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of loaded rows held out for eval. 0 → no eval / no early stopping.")
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--early_stopping_threshold", type=float, default=1e-3)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--sample_generations", type=int, default=3,
                    help="Post-train: run this many greedy generations to sanity-check tag emission.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_dir, args.dtype)
    model.to(device)

    print("\nBuilding training subset ...", flush=True)
    rows = load_corpus_subset(
        args.n_sentences, args.seed, args.max_src_tokens, tokenizer,
        indices_file=args.indices_file,
        corpus_file=args.corpus_file,
    )
    ds = build_dataset(rows)
    print(f"Full dataset: {len(ds)} rows", flush=True)

    # Optional train/eval split for early stopping.
    train_ds = ds
    eval_ds = None
    if args.val_frac > 0 and len(ds) > 20:
        split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"  train: {len(train_ds)}  eval: {len(eval_ds)}  "
              f"(val_frac={args.val_frac})", flush=True)

    # Save indices used — lets condition-B annotation match exactly.
    def _idx_list(rows_in):
        return sorted(r["index"] for r in rows_in)
    train_idx_set = set(train_ds["index"])
    eval_idx_set = set(eval_ds["index"]) if eval_ds is not None else set()
    (args.output_dir / "train_indices.json").write_text(json.dumps({
        "n_kept": len(rows),
        "seed": args.seed,
        "n_sentences_requested": args.n_sentences,
        "max_src_tokens": args.max_src_tokens,
        "indices_file": str(args.indices_file) if args.indices_file else None,
        "corpus_file": str(args.corpus_file) if args.corpus_file else None,
        "val_frac": args.val_frac,
        "n_train": len(train_idx_set),
        "n_eval": len(eval_idx_set),
        "train_indices": sorted(train_idx_set),
        "eval_indices": sorted(eval_idx_set),
        "by_latency": {
            lat: sorted(r["index"] for r in rows if r["latency"] == lat)
            for lat in ["low", "medium", "high"]
        },
    }, indent=2))
    print(f"Saved train indices to {args.output_dir}/train_indices.json", flush=True)
    print(f"Example text (row 0, first 250 chars):", flush=True)
    print(f"  {ds[0]['text'][:250]!r}", flush=True)

    # Snapshot special-token embeddings so we can prove they moved.
    before = snapshot_special_token_embeddings(model, tokenizer)

    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    use_eval = eval_ds is not None
    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_length=args.max_length,
        # completion_only_loss=False keeps default full-sequence CE loss — EAST §3.2.
        completion_only_loss=False,
        packing=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps if use_eval else args.save_steps,
        save_strategy="steps" if use_eval else "steps",
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.eval_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        seed=args.seed,
        dataset_text_field="text",
        report_to=[],
        # load_best_model_at_end needs >= 2 checkpoints to keep the best plus the current.
        save_total_limit=2 if use_eval else 1,
    )

    callbacks = []
    if use_eval:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
        print(f"Early stopping ENABLED — patience={args.early_stopping_patience}, "
              f"threshold={args.early_stopping_threshold}, eval_steps={args.eval_steps}",
              flush=True)
    else:
        print(f"Early stopping DISABLED (val_frac=0 or too few rows)", flush=True)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print(f"\nStarting training ...", flush=True)
    t0 = time.time()
    train_result = trainer.train()
    dt = time.time() - t0
    print(f"Training done in {dt:.1f}s ({dt/60:.1f} min)", flush=True)
    print(f"  metrics: {train_result.metrics}", flush=True)

    # Snapshot after training + compare.
    after = snapshot_special_token_embeddings(trainer.model, tokenizer)
    compare_special_token_embeddings(before, after, tokenizer)

    # Save final.
    print(f"\nSaving final model to {args.output_dir}/final ...", flush=True)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(args.output_dir / "final")

    # Sanity generations.
    if args.sample_generations > 0:
        sample_generations(trainer.model, tokenizer, rows, args.sample_generations, max_new_tokens=200)

    # Summary JSON for LOG.md paste.
    summary = {
        "config": vars(args) | {
            "model_path": str(args.model_path),
            "tokenizer_dir": str(args.tokenizer_dir),
            "output_dir": str(args.output_dir),
            "indices_file": str(args.indices_file) if args.indices_file else None,
            "corpus_file": str(args.corpus_file) if args.corpus_file else None,
        },
        "train_metrics": train_result.metrics,
        "wall_time_sec": dt,
        "n_rows_trained": len(ds),
        "special_token_embedding_deltas": {
            t: {
                "in_emb_delta": (after[t]["in_emb"] - before[t]["in_emb"]).norm().item(),
                "out_emb_delta": (
                    None if before[t]["out_emb"] is None
                    else (after[t]["out_emb"] - before[t]["out_emb"]).norm().item()
                ),
            } for t in SPECIAL_TOKENS
        },
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": os.popen("git -C " + str(REPO_ROOT) + " rev-parse HEAD").read().strip(),
        },
    }
    (args.output_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir}/sft_summary.json", flush=True)
    print("PHASE 2 SFT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
