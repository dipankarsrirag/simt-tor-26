"""
Post-SFT smoke test — does the trained model emit EAST special tokens at
plausible source positions when fed a streaming-style prompt?

Different from `sample_generations` inside `sft.py` (which feeds the WHOLE
source): here we feed the LATENCY token + a source prefix (first N source
words) and check whether the model emits `<|end-of-read|>` in the next few
tokens. If yes at reasonable positions, the SFT worked and Gate 2 passes.

For Gate 3 (extrinsic BLEU/COMET vs AL-CA on newstest2015) we'll build a
proper streaming inference loop; this smoke is deliberately quicker.

Usage:
    python scripts/phase2_inference_smoke.py \\
        --model_dir results/phase2/sft_condA_n2k/final \\
        --tokenizer_dir results/phase2/tokenizer-extended \\
        --n_probes 20 --prefix_words 3 --max_new_tokens 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch

from src.annotator.east_format import LATENCY_TOKENS, SPECIAL_TOKENS
from src.constants import DATA_ROOT, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--tokenizer_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "tokenizer-extended")
    ap.add_argument("--indices_file", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "phase2_n2k_indices.json",
                    help="If set, pick from these indices (evaluates on IN-DIST heldout — "
                         "not test set); otherwise samples random rows from the corpus.")
    ap.add_argument("--n_probes", type=int, default=20)
    ap.add_argument("--prefix_words", type=int, default=3,
                    help="Feed the first N source words (whitespace split) + latency token.")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=142,  # different from SFT seed
                    help="Sampling seed for probe selection; DIFFERENT from SFT seed "
                         "so we sample rows NOT trained on (approximation to heldout).")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {args.tokenizer_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"Loading model from {args.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print(f"Loading corpus", flush=True)
    with open(CORPUS) as f:
        rows = json.load(f)
    by_idx = {r["index"]: r for r in rows}

    # Choose probes — prefer rows NOT in the training indices file.
    if args.indices_file and args.indices_file.exists():
        train_idx = set(json.loads(args.indices_file.read_text())["indices"])
    else:
        train_idx = set()
    all_indices = [r["index"] for r in rows if r["index"] not in train_idx]
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    probes = [by_idx[i] for i in all_indices[: args.n_probes]]
    print(f"Selected {len(probes)} probes from {len(all_indices):,} heldout indices\n", flush=True)

    special_token_ids = {t: tok.encode(t, add_special_tokens=False)[0] for t in SPECIAL_TOKENS}
    eor_id = special_token_ids["<|end-of-read|>"]
    eow_id = special_token_ids["<|end-of-write|>"]

    n_emit_eor, n_emit_eow, n_valid_alternation = 0, 0, 0
    positions_eor = []
    per_probe = []

    for p in probes:
        latency_tok = LATENCY_TOKENS[p["latency"]]
        src_words = p["source"].split()
        prefix_words = " ".join(src_words[: args.prefix_words])
        prompt = f"{latency_tok} {prefix_words}"

        input_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_ids = out[0][prompt_len:].tolist()
        gen_str = tok.decode(gen_ids, skip_special_tokens=False)
        has_eor = eor_id in gen_ids
        has_eow = eow_id in gen_ids
        # Alternation: does EOR come before EOW? (proper streaming pattern)
        alt_ok = False
        if has_eor and has_eow:
            first_eor = gen_ids.index(eor_id)
            first_eow = gen_ids.index(eow_id)
            alt_ok = first_eor < first_eow
        if has_eor:
            n_emit_eor += 1
            positions_eor.append(gen_ids.index(eor_id))
        if has_eow:
            n_emit_eow += 1
        if alt_ok:
            n_valid_alternation += 1

        per_probe.append({
            "index": p["index"], "latency": p["latency"],
            "prompt": prompt, "gen": gen_str,
            "has_eor": has_eor, "has_eow": has_eow,
            "eor_pos": gen_ids.index(eor_id) if has_eor else None,
            "alt_ok": alt_ok,
        })
        # print first 5 in detail
        if len(per_probe) <= 5:
            print(f"idx={p['index']} lat={p['latency']} eor={has_eor} eow={has_eow} alt={alt_ok}", flush=True)
            print(f"  prompt: {prompt!r}", flush=True)
            print(f"  gen:    {gen_str[:200]!r}", flush=True)

    print(f"\n=== Summary ({len(probes)} probes) ===")
    print(f"  emitted <|end-of-read|>:       {n_emit_eor}/{len(probes)} ({100*n_emit_eor/len(probes):.1f}%)")
    print(f"  emitted <|end-of-write|>:      {n_emit_eow}/{len(probes)} ({100*n_emit_eow/len(probes):.1f}%)")
    print(f"  proper EOR-before-EOW pattern: {n_valid_alternation}/{len(probes)} ({100*n_valid_alternation/len(probes):.1f}%)")
    if positions_eor:
        p_sorted = sorted(positions_eor)
        print(f"  EOR positions in generation (median / min / max): {p_sorted[len(p_sorted)//2]} / {p_sorted[0]} / {p_sorted[-1]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "config": {
                "model_dir": str(args.model_dir),
                "tokenizer_dir": str(args.tokenizer_dir),
                "n_probes": args.n_probes,
                "prefix_words": args.prefix_words,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "summary": {
                "n_probes": len(probes),
                "n_emit_eor": n_emit_eor,
                "n_emit_eow": n_emit_eow,
                "n_valid_alternation": n_valid_alternation,
            },
            "per_probe": per_probe,
        }, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
