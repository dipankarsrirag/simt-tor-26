"""Sanity check: does averaging latency-token embeddings produce intermediate p(EOR)?

Phase A v2 (176750746) showed feed-both-tokens is BROKEN: `<|low|><|medium|>`
falls approximately AT `<|low|>`, not between low and medium. This probe tests
the alternative EAST-implied mechanism: average the two latency embeddings and
inject as the SINGLE latency-position embedding at inference.

Setup:
  * Load model, extract E_low, E_medium, E_high from `model.get_input_embeddings()`
  * Compute E_low_med = (E_low + E_medium) / 2, E_med_high = (E_medium + E_high) / 2
  * For 30 rows (10 per bucket), for each of 5 latency variants
      {low, low_med_avg, medium, med_high_avg, high}
    build inputs_embeds by:
      - Standard embedding lookup for BOS + source words
      - Replace the latency-position embedding with the target (single or averaged)
    forward pass, extract p(EOR) at word 1 and word 3
  * Aggregate: does E_low_med produce p(EOR)@w1 between p(low) and p(medium)?

Output: results/phase2/probe_v4_avg_interpolation.json + summary table.
~5 min on 1 H200.
"""
import json
import sys
from pathlib import Path
import statistics as st

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eval.extrinsic import tokenize_source_by_words  # noqa: E402
from annotator.east_format import END_OF_READ, LATENCY_TOKENS  # noqa: E402

MODEL_DIR = REPO / "results/phase2/sft_n10k_v4/final"
TOK_DIR = REPO / "results/phase2/tokenizer-extended"
DATASET = REPO / "results/phase2/sft_dataset_n10k_v4.json"
OUT = REPO / "results/phase2/probe_v4_avg_interpolation.json"

N_PER_BUCKET = 10


def main():
    print(f"Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    eor_id = tok.convert_tokens_to_ids(END_OF_READ)
    lat_ids = {k: tok.convert_tokens_to_ids(v) for k, v in LATENCY_TOKENS.items()}
    bos_id = tok.bos_token_id
    print(f"  EOR={eor_id}  lat_ids={lat_ids}  BOS={bos_id}", flush=True)

    print(f"Loading model from {MODEL_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    embed_layer = model.get_input_embeddings()
    print(f"  input_embeddings shape: {embed_layer.weight.shape}", flush=True)

    # Extract the 3 latency embeddings + compute averages
    with torch.no_grad():
        E_low = embed_layer.weight[lat_ids["low"]].clone()
        E_med = embed_layer.weight[lat_ids["medium"]].clone()
        E_high = embed_layer.weight[lat_ids["high"]].clone()
    E_low_med = (E_low + E_med) / 2
    E_med_high = (E_med + E_high) / 2

    variant_embeds = {
        "low":         E_low,
        "low_med_avg": E_low_med,
        "medium":      E_med,
        "med_high_avg": E_med_high,
        "high":        E_high,
    }
    variant_labels = list(variant_embeds.keys())

    # Sanity: cos-sim between adjacent latency embeddings (skip low↔high — not used
    # for interpolation; only low↔medium and medium↔high matter for the 5-point curve).
    def cos(a, b): return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()
    print(f"\n  Cosine similarities between adjacent latency embeddings:")
    print(f"    low↔medium:  {cos(E_low, E_med):.4f}")
    print(f"    medium↔high: {cos(E_med, E_high):.4f}", flush=True)

    print(f"\nLoading dataset ...", flush=True)
    with open(DATASET) as f:
        rows = json.load(f)
    bucket_rows = {"low": [], "medium": [], "high": []}
    for r in rows:
        if r["latency"] in bucket_rows and len(bucket_rows[r["latency"]]) < N_PER_BUCKET:
            bucket_rows[r["latency"]].append(r)
    all_rows = bucket_rows["low"] + bucket_rows["medium"] + bucket_rows["high"]
    print(f"  Sampled {len(all_rows)} rows (10 per bucket)", flush=True)

    results = {"config": {
        "n_per_bucket": N_PER_BUCKET,
        "variants": variant_labels,
        "cos_low_med": cos(E_low, E_med),
        "cos_med_high": cos(E_med, E_high),
    }, "per_row": []}

    print(f"\nRunning forward passes ({len(all_rows)} rows × {len(variant_labels)} variants) ...", flush=True)
    # Gemma-4 rejects both (input_ids, inputs_embeds) and needs input_ids for per-layer inputs.
    # Workaround: for each variant, temporarily overwrite the embedding matrix row for the
    # placeholder latency slot with the target embedding (single or averaged), then feed input_ids
    # as normal. Restore after. This is safe under model.eval() + torch.no_grad().
    # Placeholder slot: use LOW for {low, low_med_avg}, MEDIUM for {medium, med_high_avg}, HIGH for {high}.
    variant_to_slot = {
        "low":          "low",           # placeholder token = low; overwrite with E_low (no-op) or already-in-place
        "low_med_avg":  "low",           # overwrite low row with averaged
        "medium":       "medium",
        "med_high_avg": "medium",        # overwrite medium row with averaged
        "high":         "high",
    }

    # Save originals
    with torch.no_grad():
        E_orig = {
            "low":    embed_layer.weight[lat_ids["low"]].clone(),
            "medium": embed_layer.weight[lat_ids["medium"]].clone(),
            "high":   embed_layer.weight[lat_ids["high"]].clone(),
        }

    for row in all_rows:
        src_text = " ".join(row["source_chunks"])
        full_ids, spans = tokenize_source_by_words(tok, src_text)
        n_src_words = len(spans)
        qp_by_word = []
        cursor = 2
        for k in range(n_src_words):
            cursor += len(spans[k])
            qp_by_word.append(cursor - 1)

        row_result = {"index": row["index"], "annotator_bucket": row["latency"], "n_src_words": n_src_words}
        for variant_label, E_var in variant_embeds.items():
            slot = variant_to_slot[variant_label]
            slot_id = lat_ids[slot]
            template_ids = [bos_id, slot_id] + full_ids
            template_t = torch.tensor([template_ids], device="cuda", dtype=torch.long)

            with torch.no_grad():
                # Temporarily overwrite the slot's embedding row with the target
                embed_layer.weight[slot_id] = E_var.to(embed_layer.weight.dtype)
                try:
                    out = model(input_ids=template_t)
                    logits = out.logits[0]
                    probs = F.softmax(logits.float(), dim=-1)
                    p_eor = probs[qp_by_word, eor_id].cpu().tolist()
                finally:
                    # Restore original embedding row
                    embed_layer.weight[slot_id] = E_orig[slot]
            row_result[f"p_eor_{variant_label}"] = p_eor
        results["per_row"].append(row_result)

    # Aggregate: mean p(EOR)@w1 and @w3 per variant, per bucket
    print("\n  p(EOR) means by annotator bucket × latency variant:")
    for pos_label, pos_key in [("w1", 0), ("w3", 2)]:
        print(f"\n  --- position: {pos_label} ---")
        print(f"    {'bucket':<8} " + " ".join(f"{vl:>14}" for vl in variant_labels))
        for bucket in ["low", "medium", "high"]:
            row_strs = []
            for vl in variant_labels:
                vals = [r[f"p_eor_{vl}"][pos_key] for r in results["per_row"]
                        if r["annotator_bucket"] == bucket and len(r[f"p_eor_{vl}"]) > pos_key]
                row_strs.append(f"{st.mean(vals):>14.4f}" if vals else f"{'--':>14}")
            print(f"    {bucket:<8} " + " ".join(row_strs))

    # Interpolation validation for averaging
    print("\n  --- interpolation validation (averaging) ---")
    lm_pass = mh_pass = True
    lm_fracs, mh_fracs = [], []
    for r in results["per_row"]:
        if len(r.get("p_eor_low", [])) < 1:
            continue
        p_low = r["p_eor_low"][0]
        p_med = r["p_eor_medium"][0]
        p_high = r["p_eor_high"][0]
        p_lm = r["p_eor_low_med_avg"][0]
        p_mh = r["p_eor_med_high_avg"][0]
        lo, hi = sorted([p_low, p_med])
        if not (lo - 0.02 <= p_lm <= hi + 0.02):
            lm_pass = False
        lo, hi = sorted([p_med, p_high])
        if not (lo - 0.02 <= p_mh <= hi + 0.02):
            mh_pass = False
        # fractions
        denom = p_low - p_med
        if abs(denom) > 1e-4:
            lm_fracs.append((p_low - p_lm) / denom)
        denom = p_med - p_high
        if abs(denom) > 1e-4:
            mh_fracs.append((p_med - p_mh) / denom)

    print(f"    low_med_avg WITHIN [low,med] range: {'✅' if lm_pass else '❌'}")
    print(f"    med_high_avg WITHIN [med,high] range: {'✅' if mh_pass else '❌'}")
    if lm_fracs:
        print(f"    low_med_avg interpolation fraction: mean {st.mean(lm_fracs):.3f}  (0=at low, 1=at medium)")
    if mh_fracs:
        print(f"    med_high_avg interpolation fraction: mean {st.mean(mh_fracs):.3f}  (0=at medium, 1=at high)", flush=True)

    results["diagnosis"] = {
        "avg_low_med_within_range": lm_pass,
        "avg_med_high_within_range": mh_pass,
        "avg_low_med_frac_mean": st.mean(lm_fracs) if lm_fracs else None,
        "avg_med_high_frac_mean": st.mean(mh_fracs) if mh_fracs else None,
    }

    print("\nSummary verdict:")
    if lm_pass and mh_pass:
        print("  ✅ Averaging embeddings PRODUCES INTERMEDIATE p(EOR).")
        print("     This is a viable mechanism for the EAST-style interpolated 5-point curve.")
    else:
        print("  ❌ Averaging embeddings DOES NOT produce clean interpolation.")
        print("     Skip interpolation; ship 3-point curve.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
