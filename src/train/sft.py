"""
SFT wrapper for Phase 2 (EXPERIMENTS.md §Primary result).

Trains a backbone LLM (Gemma-4-E2B / E4B / Qwen3.5-2B) on EAST-interleaved
training strings using trl.SFTTrainer. Consumes an OT-annotated dataset
produced by `scripts/phase2_build_sft_dataset.py` (chunks derived from the
backbone's own per-token OT-convergence criterion, τ=0.30 primary + fallback
ladder to escape single-chunk collapses).

Single-arm setup since 2026-08-18 late — cond-A (GPT-4 chunks) and Cond-C
(within-framework wait-k chunking) were both removed after the decision to
compare against past-work published numbers verbatim rather than run our
own matched baselines. See `../LOG.md` for the deprecation entries.

Loss recipe (EAST §3.2). Cross-entropy over ALL sequence positions —
source tokens, target tokens, and special tokens (<|end-of-read|>,
<|end-of-write|>, latency indicators). This is an intentional break from
Wang et al. 2024's target-only masking, because the read/write decision
is itself what we train the model to predict. Implemented by NOT using
`DataCollatorForCompletionOnlyLM` and NOT setting `completion_only_loss`.

Tokenizer. Must be the extended one from
`scripts/phase2_prepare_tokenizer.py` — has the 5 EAST special tokens
added at ids 262144..262148 (E2B); 262400..262404 (E4B); 248077..248081
(Qwen). Loading from `MODEL_BASE/gemma-4-E2B/` directly will tokenize the
special tokens as multi-piece garbage.
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

from src.annotator.east_format import (
    END_OF_READ, END_OF_WRITE, LATENCY_TOKENS, SPECIAL_TOKENS,
    EastRow, interleave,
)
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"
EXTENDED_TOKENIZER_DIR = REPO_ROOT / "results" / "phase2" / "tokenizer-extended"

# Descriptive-init map (v3 experiment, 2026-08-19):
# each new EAST-special embedding = uniform mean over embeddings of
# reference tokens (descriptive words + optional semantic anchor from
# an existing special token). Small noise added to break latency-token
# symmetry. See LOG.md `[RUN] 2026-08-19 — v2 check_argmax VERDICT`
# for the mechanism read that motivated this.
#
# EOR/EOW anchor on <eos> because "end of read/write phase" is
# semantically nearest to "end of sequence" among Gemma's existing
# special tokens. Latency tokens have no such existing anchor.
DESCRIPTIVE_INIT_WORDS = {
    END_OF_READ:               ["end", "of", "read"],
    END_OF_WRITE:              ["end", "of", "write"],
    LATENCY_TOKENS["low"]:     ["low", "latency"],
    LATENCY_TOKENS["medium"]:  ["medium", "latency"],
    LATENCY_TOKENS["high"]:    ["high", "latency"],
}
# Existing-token anchor for boundary semantics. Populated at init time
# once we have a tokenizer; empty list = no anchor.
def _boundary_anchor_ids(tokenizer):
    """Return the EOS id as the semantic anchor for boundary tokens."""
    return [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []


def build_input_ids_direct(row: Dict[str, Any], bos_id: int, lat_ids: Dict[str, int],
                            eor_id: int, eow_id: int) -> list:
    """v4 fix (2026-08-19): build the EAST input_ids sequence directly from
    the annotator's per-chunk BPE ids stored in the dataset, avoiding the
    decode+retokenize round-trip that misaligns chunk-final punctuation
    tokens (`.` id 236761 in the source vs `▁.` id 783 after decode+strip).

    Layout: [<BOS>, <latency>, *src_chunk_ids_1, <EOR>, *tgt_chunk_ids_1,
             <EOW>, *src_chunk_ids_2, <EOR>, *tgt_chunk_ids_2, <EOW>, ...]

    Requires the row to have `source_chunk_ids` / `target_chunk_ids` fields
    (added by phase2_build_sft_dataset.py 2026-08-19+). Rows built by the
    pre-fix builder lack these; caller must fall back to string interleave.
    """
    ids = [bos_id, lat_ids[row["latency"]]]
    for src_c, tgt_c in zip(row["source_chunk_ids"], row["target_chunk_ids"]):
        ids.extend(src_c)
        ids.append(eor_id)
        ids.extend(tgt_c)
        ids.append(eow_id)
    return ids


def build_east_string(row: Dict[str, Any], fixed_tokenization: bool = False) -> str:
    """From a corpus row (which contains source/target/source_chunks/target_chunks/latency)
    produce the EAST-interleaved training string ready for tokenisation.

    fixed_tokenization=True (v4+) uses the leading-space-per-chunk +
    empty-join interleave that eliminates the standalone `▁` separator
    training/inference mismatch. See east_format.interleave() docstring.
    """
    er = EastRow(
        source=row["source"],
        target=row["target"],
        src_lang=row["src_lang"],
        tgt_lang=row["tgt_lang"],
        latency=row["latency"],
        source_chunks=list(row["source_chunks"]),
        target_chunks=list(row["target_chunks"]),
    )
    return interleave(er, fixed_tokenization=fixed_tokenization)


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
    SiMT-660K corpus — used to consume our OT-annotator chunks (built by
    scripts/phase2_build_sft_dataset.py).
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
    elif corpus_file is not None:
        # corpus_file is pre-filtered (e.g. by phase2_build_sft_dataset.py),
        # already-latency-balanced by construction. Use every row — capping to
        # --n_sentences here would silently drop 80% of a 10K cond-B dataset.
        picks = rows
        print(f"Using ALL {len(picks)} rows from corpus_file {corpus_file}", flush=True)
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


def build_dataset(rows: List[Dict], fixed_tokenization: bool = False,
                   tokenizer=None) -> "Dataset":
    """Build the HF Dataset for SFT.

    Three modes (from safest → most correct):
      1. Legacy string interleave (fixed_tokenization=False): produces `text`
         column; TRL tokenizes at collation time. Introduces phantom `▁`
         separators before EAST specials.
      2. Fixed-string interleave (fixed_tokenization=True, no *_chunk_ids in
         rows): produces `text` column with corrected chunk_sep; removes
         phantom ▁ separators BUT still has the chunk-final punctuation
         artifact (decode+strip inserts space before `.` etc.).
      3. Direct-ids (fixed_tokenization=True, *_chunk_ids present in rows):
         produces `input_ids` column; NO decode+retokenize round-trip;
         perfect byte-alignment with what streaming inference will feed.
    """
    from datasets import Dataset
    has_chunk_ids = fixed_tokenization and rows and "source_chunk_ids" in rows[0]
    if has_chunk_ids:
        # Direct-ids path (v4 fix). Requires tokenizer to know special-token ids.
        assert tokenizer is not None, "tokenizer required for direct-ids path"
        from src.annotator.east_format import END_OF_READ, END_OF_WRITE, LATENCY_TOKENS
        eor_id = tokenizer(END_OF_READ, add_special_tokens=False).input_ids[0]
        eow_id = tokenizer(END_OF_WRITE, add_special_tokens=False).input_ids[0]
        lat_ids = {k: tokenizer(v, add_special_tokens=False).input_ids[0]
                   for k, v in LATENCY_TOKENS.items()}
        bos_id = tokenizer.bos_token_id
        all_ids = [build_input_ids_direct(r, bos_id, lat_ids, eor_id, eow_id)
                   for r in rows]
        return Dataset.from_dict({
            "input_ids": all_ids,
            "labels": all_ids,  # causal LM: labels = input_ids
            "index": [r["index"] for r in rows],
            "latency": [r["latency"] for r in rows],
        })
    return Dataset.from_dict({
        "text": [build_east_string(r, fixed_tokenization=fixed_tokenization) for r in rows],
        "index": [r["index"] for r in rows],
        "latency": [r["latency"] for r in rows],
    })


def load_model_and_tokenizer(model_path: str, tokenizer_dir: Path, dtype: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    # Gemma-3n (E4B+) ships as multimodal with a vision tower that needs `timm`;
    # AutoModelForCausalLM instantiates `Gemma3nForConditionalGeneration` which
    # tries to load the vision tower. We only want the text CausalLM. Route
    # explicitly to `Gemma3nForCausalLM` when the config declares gemma3n.
    cfg = AutoConfig.from_pretrained(model_path)
    if getattr(cfg, "model_type", None) == "gemma3n":
        from transformers import Gemma3nForCausalLM
        print(f"  (model_type=gemma3n; loading text-only Gemma3nForCausalLM)", flush=True)
        model = Gemma3nForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, low_cpu_mem_usage=True
        )
    else:
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
        # Some backbones (Qwen3.5-2B: 248320 model rows vs 248077 base vocab)
        # ship padded embedding matrices with reserved-but-unused rows for
        # alignment. Our EAST tokens (ids past base vocab) land INSIDE this
        # reserved region, so no resize is needed — the model already has
        # embedding rows initialised for those ids (transformers default).
        # We do NOT re-init them: the existing values are as valid as
        # mean-covariance init would be.
        print(f"  (tokenizer {len(tokenizer):,} < model embeddings {orig_vocab:,}; "
              f"{orig_vocab - len(tokenizer)} reserved rows exist. Using EAST-token "
              f"positions within the reserved region — no resize.)", flush=True)
    else:
        print(f"  (no resize needed; sizes already match)", flush=True)
    return model, tokenizer


def apply_descriptive_init(model, tokenizer, noise_std: float = 0.01, seed: int = 42):
    """v3 experiment (2026-08-19): re-initialise each EAST-special embedding
    row as the mean of embeddings of descriptive words + a semantic anchor
    (EOS for boundary tokens), plus small Gaussian noise to break symmetry.

    Rationale (see LOG.md v2 verdict entry): after 700 SFT steps under
    mean-covariance random init, EOR's embedding moved only 0.077 L2 —
    well-populated tokens typically move 0.5-2.0. With no semantic prior
    the token starts at a random point of the embedding manifold and the
    ~5-15% loss-label share of specials-per-row is not enough to organise
    a coherent commit distribution during READ. Descriptive init gives
    each new token a starting basin near the semantically-closest region.
    """
    in_emb = model.get_input_embeddings()
    dtype = in_emb.weight.dtype
    device = in_emb.weight.device
    anchor_ids = _boundary_anchor_ids(tokenizer)

    print(f"\nDescriptive-init on {len(DESCRIPTIVE_INIT_WORDS)} EAST-special embeddings "
          f"(noise σ={noise_std}, anchor_ids={anchor_ids})", flush=True)

    g = torch.Generator(device="cpu").manual_seed(seed)
    for tok_str, words in DESCRIPTIVE_INIT_WORDS.items():
        # Ref token IDs = tokenization of each descriptive word (any subword
        # count OK — averaged uniformly) + optional boundary anchor (EOS)
        # for EOR/EOW only.
        ref_ids = []
        for w in words:
            ids = tokenizer(w, add_special_tokens=False).input_ids
            ref_ids.extend(ids)
        if tok_str in (END_OF_READ, END_OF_WRITE):
            ref_ids.extend(anchor_ids)
        if not ref_ids:
            print(f"  {tok_str!r}: NO reference tokens — skipping", flush=True)
            continue
        with torch.no_grad():
            ref_emb = in_emb.weight[torch.tensor(ref_ids, device=device)].float()
            mean_vec = ref_emb.mean(dim=0)
            noise = torch.randn(mean_vec.shape, generator=g).to(device) * noise_std
            new_vec = (mean_vec + noise).to(dtype)
            new_id = tokenizer(tok_str, add_special_tokens=False).input_ids[0]
            in_emb.weight[new_id].copy_(new_vec)
        print(f"  {tok_str!r} (id {new_id}) <- mean of ref ids {ref_ids} + N(0, {noise_std}^2)",
              flush=True)


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


def post_train_smoke(model, tokenizer, n_sents: int, device: str,
                      dev_src: str = "/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/newstest2013.de",
                      dev_ref: str = "/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/newstest2013.en") -> Dict:
    """v3+ (2026-08-19): tiny streaming check_argmax on newstest2013 to catch
    degenerate adaptivity BEFORE spending ~50 min on a full 3000-sent eval.

    Reuses `stream_translate` from the eval harness; runs on the model still
    resident in GPU memory (no reload). Reports chunks/sent + src-exhausted
    fraction. Verdict `fire-full-eval` if chunks/sent mean > 1.05 (adaptivity
    signal emerging); `null` otherwise. Verdict grep-able from log; JSON
    written to `<output_dir>/post_train_smoke.json`.
    """
    from src.eval.extrinsic import stream_translate, load_dev_pairs
    import statistics as s

    model.eval()
    pairs = load_dev_pairs(Path(dev_src), Path(dev_ref))[:n_sents]
    chunks = []
    src_exh = 0
    for p in pairs:
        trace = stream_translate(
            model, tokenizer, p["src"], "medium", device,
            policy="check_argmax",
        )
        chunks.append(trace.chunks_committed)
        if trace.source_exhausted_without_eor:
            src_exh += 1

    mean_ch = s.mean(chunks)
    med_ch = s.median(chunks)
    verdict = "fire-full-eval" if mean_ch > 1.05 else "null"
    print(f"\n=== POST-TRAIN SMOKE (n={len(pairs)}, check_argmax, latency=medium) ===",
          flush=True)
    print(f"  chunks/sent mean/median: {mean_ch:.2f} / {med_ch:.1f}", flush=True)
    print(f"  src-exhausted-w/o-eor:   {src_exh}/{len(pairs)}", flush=True)
    print(f"  ADAPTIVITY_VERDICT:      {verdict}", flush=True)
    return {
        "n_sents": len(pairs),
        "policy": "check_argmax",
        "latency": "medium",
        "chunks_per_sent_mean": mean_ch,
        "chunks_per_sent_median": med_ch,
        "n_source_exhausted_without_eor": src_exh,
        "verdict": verdict,
    }


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
                         "scripts/phase2_build_sft_dataset.py).")
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
    ap.add_argument("--use_augmentation", action="store_true",
                    help="Include latency-augmented rows (aug2, aug4) in train + eval. "
                         "Default OFF — corpus_file's base rows are typically balanced enough "
                         "and augmentation risks duplicate-index leakage across the split "
                         "(2026-08-20: fixed via group-split by index, but disabling aug "
                         "avoids the risk entirely and keeps the training set cleaner).")
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--early_stopping_threshold", type=float, default=1e-3)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--sample_generations", type=int, default=3,
                    help="Post-train: run this many greedy generations to sanity-check tag emission.")
    ap.add_argument("--keep_checkpoints", action="store_true",
                    help="Retain intermediate checkpoint-*/ dirs after training. "
                    "Default: delete them once final/ is saved (saves 50-300 GB per run). "
                    "See HOUSEKEEPING.md §Post-job hygiene.")

    # v3 experiment flags (2026-08-19). Address v2's degenerate check_argmax
    # via two orthogonal interventions on top of the annotator-time fixes:
    # (a) descriptive_init — meaningful embedding starting basins;
    # (b) special_token_loss_weight — class-imbalance fix on the SFT loss.
    ap.add_argument("--descriptive_init", action="store_true",
                    help="Initialize each EAST-special embedding as the mean of "
                         "embeddings of descriptive words (+ <eos> anchor for EOR/EOW) "
                         "plus small noise. See apply_descriptive_init() for the map.")
    ap.add_argument("--special_token_loss_weight", type=float, default=1.0,
                    help="If > 1, multiply the per-token loss by this value at EAST-special "
                         "label positions (Test B). Sweep α ∈ {3, 5, 10}; default 1 = off.")
    ap.add_argument("--post_train_smoke_sents", type=int, default=0,
                    help="If > 0, run a mini streaming check_argmax on this many sentences "
                         "of newstest2013 immediately after SFT, using model in memory. "
                         "Prints chunks/sent + adaptivity verdict — saves ~50 min of full "
                         "eval if the checkpoint is null-degenerate.")
    ap.add_argument("--fixed_tokenization", action="store_true",
                    help="Use east_format.interleave(fixed_tokenization=True) — leading "
                         "space per chunk + empty join, eliminating standalone ▁ separators "
                         "before EAST specials. Fixes the training/inference token-stream "
                         "mismatch that caused v1/v2/v3 to null on check_argmax adaptivity. "
                         "See src/annotator/east_format.py::interleave docstring.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_dir, args.dtype)
    model.to(device)

    if args.descriptive_init:
        apply_descriptive_init(model, tokenizer, noise_std=0.01, seed=args.seed)

    print("\nBuilding training subset ...", flush=True)
    rows = load_corpus_subset(
        args.n_sentences, args.seed, args.max_src_tokens, tokenizer,
        indices_file=args.indices_file,
        corpus_file=args.corpus_file,
    )
    ds = build_dataset(rows, fixed_tokenization=args.fixed_tokenization,
                        tokenizer=tokenizer)
    print(f"Full dataset: {len(ds)} rows", flush=True)

    # Optional aug-filter BEFORE the split. When --use_augmentation is OFF (default),
    # drop any row whose _annotator_meta.augmented_from_base is True (i.e. aug2/aug4
    # copies). Keeps only base rows — the balanced corpus is typically enough, and
    # this eliminates the duplicate-index leakage risk entirely.
    if not args.use_augmentation:
        n_before = len(ds)
        ds = ds.filter(lambda r: not ((r.get("_annotator_meta") or {}).get("augmented_from_base") or False))
        print(f"  Augmentation filter OFF: kept {len(ds)}/{n_before} base rows "
              f"(dropped {n_before - len(ds)} aug rows).", flush=True)
    else:
        print(f"  Augmentation filter ON: keeping all {len(ds)} rows (base + aug).", flush=True)

    # Optional train/eval split for early stopping.
    # GROUP SPLIT BY `index` (defence in depth): if aug rows are included, they
    # share the base row's `index` field. Splitting by row would put a base in
    # eval and its augmented copies in train — the model would see each source
    # sentence at multiple chunk granularities → optimistic eval loss + BLEU.
    # Group-split fixes this; with --use_augmentation OFF the group split is a
    # no-op since every index is unique.
    train_ds = ds
    eval_ds = None
    if args.val_frac > 0 and len(ds) > 20:
        import random as _random
        all_indices = sorted(set(ds["index"]))
        _rng = _random.Random(args.seed)
        _rng.shuffle(all_indices)
        n_eval_idx = max(1, int(len(all_indices) * args.val_frac))
        eval_idx_set = set(all_indices[:n_eval_idx])
        train_idx_set_pre = set(all_indices[n_eval_idx:])
        train_ds = ds.filter(lambda r: r["index"] in train_idx_set_pre)
        eval_ds = ds.filter(lambda r: r["index"] in eval_idx_set)
        print(f"  Split by base index (leak-free):  {len(all_indices)} unique indices "
              f"→ train {len(train_idx_set_pre)} idx ({len(train_ds)} rows), "
              f"eval {len(eval_idx_set)} idx ({len(eval_ds)} rows)", flush=True)
        leak = set(train_ds["index"]) & set(eval_ds["index"])
        assert not leak, f"leak-check failed: {len(leak)} indices in both splits"

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
    if "text" in ds.column_names:
        print(f"Example text (row 0, first 250 chars):", flush=True)
        print(f"  {ds[0]['text'][:250]!r}", flush=True)
    else:
        # fixed_tokenization path: rows carry pre-built input_ids/labels only.
        example_ids = ds[0]["input_ids"]
        print(f"Example input_ids (row 0, first 40 of {len(example_ids)}):", flush=True)
        print(f"  {list(example_ids[:40])}", flush=True)
        print(f"  decoded: {tokenizer.decode(example_ids[:60])!r}", flush=True)

    # Snapshot special-token embeddings so we can prove they moved.
    before = snapshot_special_token_embeddings(model, tokenizer)

    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback
    import torch.nn.functional as F

    # Test B (2026-08-19): weighted-loss trainer that upweights EAST-special
    # label positions by alpha. Rest of the training pipeline is unchanged.
    class WeightedSFTTrainer(SFTTrainer):
        def __init__(self, *targs, special_ids=None, alpha=1.0, **tkwargs):
            super().__init__(*targs, **tkwargs)
            self._special_ids = torch.tensor(list(special_ids or []), dtype=torch.long)
            self._alpha = float(alpha)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self._alpha == 1.0 or len(self._special_ids) == 0:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            # Custom weighted CE: αx at label positions matching any EAST-special id.
            labels = inputs.get("labels")
            if labels is None:
                # SFTTrainer default: labels = input_ids with prompt masked (which we don't
                # do here since completion_only_loss=False). Fall back to input_ids.
                labels = inputs["input_ids"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none", ignore_index=-100,
            )
            flat_labels = shift_labels.view(-1)
            special_ids = self._special_ids.to(flat_labels.device)
            is_special = torch.isin(flat_labels, special_ids)
            weights = torch.where(is_special,
                                  torch.tensor(self._alpha, device=flat_labels.device),
                                  torch.tensor(1.0, device=flat_labels.device))
            mask = (flat_labels != -100).to(per_tok.dtype)
            weights = weights.to(per_tok.dtype) * mask
            loss = (per_tok * weights).sum() / weights.sum().clamp_min(1.0)
            outputs.loss = loss
            return (loss, outputs) if return_outputs else loss

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
        # dataset_text_field only used when ds has a `text` column; direct-ids
        # mode (v4 fix) writes `input_ids`+`labels` columns directly and TRL
        # skips tokenization when `text` is absent.
        dataset_text_field="text" if "text" in ds.column_names else None,
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

    if args.special_token_loss_weight > 1.0:
        special_ids = [tokenizer(t, add_special_tokens=False).input_ids[0]
                       for t in SPECIAL_TOKENS]
        print(f"Test B ENABLED — α={args.special_token_loss_weight} on label ids "
              f"{special_ids} ({SPECIAL_TOKENS})", flush=True)
        trainer = WeightedSFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            special_ids=special_ids,
            alpha=args.special_token_loss_weight,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    # Walltime-kill resume: if intermediate `checkpoint-*/` dirs exist from a
    # prior partial run AND `final/` doesn't exist (train didn't complete),
    # tell HF Trainer to resume from the latest checkpoint. HF handles
    # everything: model+optimizer+scheduler+RNG state, dataloader position,
    # global step count. Combined with `--keep_checkpoints`, this makes SFT
    # robust to walltime kills — just resubmit the same PBS.
    resume_from = None
    final_dir = args.output_dir / "final"
    if not final_dir.exists() or not (final_dir / "config.json").exists():
        checkpoint_dirs = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0
        )
        if checkpoint_dirs:
            resume_from = str(checkpoint_dirs[-1])
            print(f"RESUME detected: {len(checkpoint_dirs)} intermediate checkpoint(s) present; "
                  f"resuming from {resume_from}", flush=True)
        else:
            print(f"No prior checkpoints; starting training from scratch.", flush=True)

    print(f"\nStarting training ...", flush=True)
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
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

    # Hygiene: after `final/` is durable, delete intermediate `checkpoint-*/`
    # dirs. Trainer keeps them for resumption; once training is complete we
    # only need `final/` (best model per load_best_model_at_end=True) +
    # `sft_summary.json`. Each intermediate checkpoint is 3-30 GB on this
    # project — leaving them accumulates ~50-300 GB per SFT run. Gated by
    # `--keep_checkpoints` for the rare case someone wants to inspect the
    # training trajectory. See HOUSEKEEPING.md §"Post-job hygiene".
    if not args.keep_checkpoints:
        import shutil
        removed = []
        for p in sorted(args.output_dir.glob("checkpoint-*")):
            if p.is_dir():
                shutil.rmtree(p)
                removed.append(p.name)
        if removed:
            print(f"Deleted intermediate checkpoints: {', '.join(removed)}", flush=True)

    # Sanity generations.
    if args.sample_generations > 0:
        sample_generations(trainer.model, tokenizer, rows, args.sample_generations, max_new_tokens=200)

    # Post-train smoke (v3+): mini streaming check_argmax on newstest2013 to
    # get an adaptivity verdict without waiting the ~50 min for a full 3000-sent
    # eval. Reuses the model in memory (no reload). Prints
    # `ADAPTIVITY_VERDICT: fire-full-eval | null` — grep-able from the log.
    if args.post_train_smoke_sents > 0:
        smoke_verdict = post_train_smoke(
            trainer.model, tokenizer,
            n_sents=args.post_train_smoke_sents,
            device=device,
        )
        (args.output_dir / "post_train_smoke.json").write_text(
            json.dumps(smoke_verdict, indent=2, ensure_ascii=False)
        )

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
