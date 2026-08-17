"""
Extrinsic eval harness for Phase-2 SFT models (Gate 3 first cut).

Layers, mirroring the advisor spec (2026-08-17):
  1. offline   — full-source generation, BLEU only. Sanity: does the model
                 translate at all? Both A and B must have offline BLEU > 10
                 before streaming numbers are worth reading.
  2. streaming — state-machine READ/WRITE with KV-cache preservation, per
                 EAST §3.2 inference protocol. BLEU should stay within a
                 few points of offline; if it collapses the protocol is
                 wrong. AL (word units) tracked here.
  3. AL-CA     — torch.cuda.Event per emitted target token; warmup discard.
                 (Punted from first cut — first-cut ships offline + streaming
                 BLEU + AL only.)

The two conditions being compared:
  cond-A = trained on shipped GPT-4 chunks (SiMT-660K's source_chunks).
  cond-B = trained on our OT-annotator chunks (results/phase2/annot_ot_*).

Both use the SAME extended tokenizer (results/phase2/tokenizer-extended)
and the same 5 EAST special tokens at ids 262144-262148. Any drift here
poisons every downstream number — the tokenizer_dir CLI flag is
non-optional.

Dev-first discipline (advisor): report numbers on newstest2013 to validate
the pipeline. newstest2015 is the test set and gets touched exactly once,
after A-vs-B behaviour on the dev set is understood.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch

from src.annotator.east_format import (
    END_OF_READ,
    END_OF_WRITE,
    LATENCY_TOKENS,
    SPECIAL_TOKENS,
)


# ─── I/O ───────────────────────────────────────────────────────────────

def load_dev_pairs(src_path: Path, ref_path: Path) -> List[Dict[str, str]]:
    """Load parallel .de/.en, one sentence per line. Requires 1:1 alignment."""
    src = src_path.read_text().splitlines()
    ref = ref_path.read_text().splitlines()
    if len(src) != len(ref):
        raise ValueError(f"src/ref line count mismatch: {len(src)} vs {len(ref)}")
    return [{"src": s, "ref": r} for s, r in zip(src, ref)]


# ─── Layer 1: offline ──────────────────────────────────────────────────

@torch.no_grad()
def generate_offline(
    model,
    tok,
    src: str,
    latency: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """Full-source greedy generation as the degenerate "one giant chunk" case
    of the streaming protocol. Prompt closes the READ phase explicitly with
    <|end-of-read|> so the model sees a valid EAST training shape:

        <|latency|> src_1 ... src_n <|end-of-read|>  → generate target tokens

    (Without the trailing EOR, the input `<|latency|> src` is not a shape the
    model ever saw during SFT — behaviour there is undefined and BLEU is not
    interpretable. cond-B in particular has real single-chunk training rows
    where the whole source is one READ.)

    Returns the raw decoded string with all EAST special tokens + BOS/EOS
    stripped by `skip_special_tokens=True`.
    """
    prompt = f"{LATENCY_TOKENS[latency]} {src} {END_OF_READ}"
    inp = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_len = inp.shape[1]
    out = model.generate(
        input_ids=inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen_ids = out[0][prompt_len:].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


# ─── Runner ────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    model_dir: str
    tokenizer_dir: str
    dev_src: str
    dev_ref: str
    latency: str          # low | medium | high (uniform prompt latency for now)
    n_sentences: int      # -1 for all
    max_new_tokens: int
    mode: str             # offline | streaming
    output: str


def run(cfg: RunConfig) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sacrebleu

    print(f"[extrinsic] mode={cfg.mode} latency={cfg.latency}", flush=True)
    print(f"[extrinsic] loading tokenizer {cfg.tokenizer_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    print(f"[extrinsic] loading model {cfg.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Sanity: verify our special tokens survived tokenizer save/load.
    for t in SPECIAL_TOKENS:
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"special token {t!r} splits into {len(ids)} ids — tokenizer drift")

    pairs = load_dev_pairs(Path(cfg.dev_src), Path(cfg.dev_ref))
    if cfg.n_sentences > 0:
        pairs = pairs[: cfg.n_sentences]
    print(f"[extrinsic] scoring {len(pairs)} sentences", flush=True)

    hyps: List[str] = []
    refs: List[str] = [p["ref"] for p in pairs]
    t0 = time.time()
    for i, p in enumerate(pairs):
        if cfg.mode == "offline":
            h = generate_offline(model, tok, p["src"], cfg.latency,
                                 cfg.max_new_tokens, device)
        else:
            raise NotImplementedError(
                f"mode={cfg.mode!r} not implemented yet — Layer 1 only. "
                "Streaming (Layer 2) added once offline BLEU passes sanity."
            )
        hyps.append(h)
        if i < 3 or (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pairs)}] src={p['src'][:70]!r}", flush=True)
            print(f"           hyp={h[:120]!r}", flush=True)
    wall = time.time() - t0
    print(f"[extrinsic] generation wall: {wall:.1f}s ({wall/len(pairs):.2f}s/sent)", flush=True)

    # sacrebleu 2.6: `.signature` moved off the score object; get it from the metric.
    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(hyps, [refs])
    signature = str(metric.get_signature())
    print(f"[extrinsic] BLEU = {bleu.score:.2f} ({signature})", flush=True)

    result = {
        "config": asdict(cfg),
        "n_sentences": len(pairs),
        "bleu": bleu.score,
        "bleu_signature": signature,
        "wall_time_sec": wall,
        "sec_per_sentence": wall / len(pairs),
        "hyps": hyps,
        "refs": refs,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[extrinsic] wrote {cfg.output}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="e.g. results/phase2/sft_condA_n10k/final")
    ap.add_argument("--tokenizer_dir",
                    default="/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended")
    ap.add_argument("--dev_src", required=True,
                    help="Parallel source file, one sentence per line.")
    ap.add_argument("--dev_ref", required=True,
                    help="Parallel reference file, one sentence per line.")
    ap.add_argument("--latency", choices=list(LATENCY_TOKENS), default="medium",
                    help="Latency prompt token — uniform per run in Layer 1. "
                         "Layer 2 will support per-sentence latency.")
    ap.add_argument("--n_sentences", type=int, default=-1,
                    help="-1 for all sentences in dev_src.")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--mode", choices=["offline", "streaming"], default="offline",
                    help="Layer 1 is offline. Streaming pending Layer 2.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = RunConfig(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        dev_src=args.dev_src,
        dev_ref=args.dev_ref,
        latency=args.latency,
        n_sentences=args.n_sentences,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        output=args.output,
    )
    run(cfg)


if __name__ == "__main__":
    main()
