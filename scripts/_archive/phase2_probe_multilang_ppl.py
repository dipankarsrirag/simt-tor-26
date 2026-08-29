"""Perplexity probe: does Gemma-4-E2B know enough of each language pair
for the OT annotator to produce meaningful commit signals?

Uses FLORES-200 devtest (997 lines, held out, multiway-aligned across all
5 target languages: ar, de, ru, zh, vi + en). Same 100 source sentences
translated to every language → controlled comparison.

For each direction (10 total: 5 pairs × bidirectional):
  Build [BOS, src_ids, sep, tgt_ids] where sep = '\\n' or `<|end-of-read|>`.
  Compute teacher-forced per-token loss on target positions only.
  Report per-direction PPL over 100 sentences.

Verdict per direction:
  PPL < 10:  ✅ backbone excellent — clean OT signals expected
  10 ≤ PPL < 30:  ✅ adequate — OT annotator will work
  30 ≤ PPL < 100:  ⚠️ marginal — annotator may be noisy
  PPL ≥ 100:  ❌ backbone insufficient — need to step up (E4B or Qwen)

Runs on 1 H200, ~10 min. Output: results/phase2/probe_multilang_ppl.json
"""
import json
import sys
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "results/phase2/sft_n10k_v4/final"  # our v4 SFT checkpoint
TOK_DIR = REPO / "results/phase2/tokenizer-extended"
FLORES_DIR = Path("/g/data/ba39/dipankar/simul-mt/data/raw/flores200/flores200_dataset/devtest")
OUT = REPO / "results/phase2/probe_multilang_ppl.json"

N_SENTS = 100

# Map project language codes to FLORES-200 filenames
LANG_TO_FILE = {
    "en": "eng_Latn.devtest",
    "de": "deu_Latn.devtest",
    "ar": "arb_Arab.devtest",
    "ru": "rus_Cyrl.devtest",
    "zh": "zho_Hans.devtest",   # simplified
    "vi": "vie_Latn.devtest",
}

# 10 directions: 5 pairs × bidirectional
PAIRS = [("de", "en"), ("en", "de"),
         ("ar", "en"), ("en", "ar"),
         ("ru", "en"), ("en", "ru"),
         ("zh", "en"), ("en", "zh"),
         ("vi", "en"), ("en", "vi")]


def load_flores_lines(lang: str, n: int):
    path = FLORES_DIR / LANG_TO_FILE[lang]
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()[:n]]
    return lines


def measure_pair_ppl(model, tok, src_lang: str, tgt_lang: str, sep: str = "\n"):
    src_lines = load_flores_lines(src_lang, N_SENTS)
    tgt_lines = load_flores_lines(tgt_lang, N_SENTS)
    total_loss = 0.0
    total_tokens = 0
    per_sent = []
    for src, tgt in zip(src_lines, tgt_lines):
        src_ids = tok(src, add_special_tokens=False).input_ids
        # separator: newline
        sep_ids = tok(sep, add_special_tokens=False).input_ids
        tgt_ids = tok(tgt, add_special_tokens=False).input_ids
        if not tgt_ids:
            continue
        bos_id = tok.bos_token_id
        input_ids = [bos_id] + src_ids + sep_ids + tgt_ids
        input_t = torch.tensor([input_ids], device="cuda", dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_t)
            logits = out.logits[0]  # (T, V)
        # Target positions in the full sequence: last len(tgt_ids) positions
        # We predict tgt_ids[k] from logits[len(prefix) - 1 + k]
        prefix_len = 1 + len(src_ids) + len(sep_ids)
        # positions of interest for computing loss on tgt tokens:
        #   for k in range(len(tgt_ids)): logit_idx = prefix_len - 1 + k, label = tgt_ids[k]
        # careful: logit at position i predicts token at i+1
        query_positions = list(range(prefix_len - 1, prefix_len - 1 + len(tgt_ids)))
        labels = torch.tensor(tgt_ids, device="cuda", dtype=torch.long)
        target_logits = logits[query_positions]  # (n_tgt, V)
        loss = F.cross_entropy(target_logits.float(), labels, reduction="sum")
        total_loss += float(loss.item())
        total_tokens += len(tgt_ids)
        per_sent.append({"loss_sum": float(loss.item()), "n_tokens": len(tgt_ids)})
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # cap ppl at exp(20) to avoid overflow
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_sents": len(per_sent),
        "total_tokens": total_tokens,
        "avg_nll": avg_loss,
        "ppl": ppl,
    }


def verdict(ppl: float) -> str:
    if ppl < 10:
        return "✅ excellent — clean OT signals"
    if ppl < 30:
        return "✅ adequate — OT annotator will work"
    if ppl < 100:
        return "⚠️ marginal — annotator may be noisy"
    return "❌ backbone insufficient — step up"


def main():
    print(f"Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    print(f"Loading model from {MODEL_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"  loaded", flush=True)

    print(f"\nProbing {len(PAIRS)} directions × {N_SENTS} sents from FLORES-200 devtest ...\n", flush=True)
    results = {"config": {"model": str(MODEL_DIR), "n_sents": N_SENTS,
                          "flores_dir": str(FLORES_DIR)}, "per_direction": []}
    print(f"{'direction':<10} {'n_sents':>8} {'total_toks':>10} {'avg_nll':>10} {'PPL':>10}    verdict")
    print("-" * 90)
    for src, tgt in PAIRS:
        r = measure_pair_ppl(model, tok, src, tgt)
        r["verdict"] = verdict(r["ppl"])
        results["per_direction"].append(r)
        print(f"{src+'→'+tgt:<10} {r['n_sents']:>8} {r['total_tokens']:>10} {r['avg_nll']:>10.4f} {r['ppl']:>10.2f}    {r['verdict']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
