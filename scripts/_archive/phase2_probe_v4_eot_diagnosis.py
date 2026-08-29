"""EOR/EOW over-eagerness diagnosis probe for v4 checkpoint (v2 — extended).

Follows the advisor-sequenced plan (LOG.md 2026-08-20):

v1 results (176750010) confirmed:
  - Byte-compare 20/20 pass → tokenization clean
  - p(EOR)@commit=0.51 vs @noncommit=0.22 (ratio 2.3×) → model learned commit
  - Δp(EOR)@w1 low-high = 0.38 → latency conditioning works
  - p(EOW) essentially perfect

v2 (this rewrite) adds:
  - 5 latency variants: low, medium, high, low+medium, medium+high
    (interpolation via feed-both-tokens: [BOS, lat_A, lat_B, src...])
  - Rows from all 3 buckets (10 medium + 10 low + 10 high = 30 rows)
    so we know if latency conditioning works across all buckets, not just medium
  - Retains the byte-compare / positional / EOW subtests but now cross-latency

Sample size 30 sents × 5 latency variants; single H200 forward pass batch.
~15 min.

Output: results/phase2/probe_v4_eot_diagnosis_v2.json + printed summary table.
"""
import json
import sys
from pathlib import Path
import statistics as st
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from train.sft import build_input_ids_direct  # noqa: E402
from eval.extrinsic import tokenize_source_by_words  # noqa: E402
from annotator.east_format import END_OF_READ, END_OF_WRITE, LATENCY_TOKENS  # noqa: E402


MODEL_DIR = REPO / "results/phase2/sft_n10k_v4/final"
TOK_DIR = REPO / "results/phase2/tokenizer-extended"
DATASET = REPO / "results/phase2/sft_dataset_n10k_v4.json"
OUT = REPO / "results/phase2/probe_v4_eot_diagnosis_v2.json"

N_BYTECOMPARE = 20
N_PER_BUCKET = 10  # 10 rows × 3 buckets = 30 rows total for positional/latency tests
LATENCY_TARGET = "medium"

# Latency variants for testing: single tokens + interpolated pairs.
# Interpolation via feed-both-tokens: prepend both to the prompt.
LATENCY_VARIANTS = [
    ("low",         ["low"]),
    ("low+medium",  ["low", "medium"]),
    ("medium",      ["medium"]),
    ("medium+high", ["medium", "high"]),
    ("high",        ["high"]),
]


def main():
    print(f"Loading tokenizer from {TOK_DIR} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))

    eor_id = tok.convert_tokens_to_ids(END_OF_READ)
    eow_id = tok.convert_tokens_to_ids(END_OF_WRITE)
    lat_ids = {k: tok.convert_tokens_to_ids(v) for k, v in LATENCY_TOKENS.items()}
    bos_id = tok.bos_token_id
    print(f"  EOR={eor_id}  EOW={eow_id}  lat_ids={lat_ids}  BOS={bos_id}", flush=True)

    print(f"Loading model from {MODEL_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"  loaded", flush=True)

    print(f"Loading dataset {DATASET} ...", flush=True)
    with open(DATASET) as f:
        rows = json.load(f)
    med_rows = [r for r in rows if r["latency"] == LATENCY_TARGET]
    print(f"  medium-bucket rows: {len(med_rows)}", flush=True)

    # Deterministic sample by index
    med_rows = sorted(med_rows, key=lambda r: r["index"])

    results = {"config": {
        "model": str(MODEL_DIR), "dataset": str(DATASET),
        "n_bytecompare": N_BYTECOMPARE, "n_per_bucket": N_PER_BUCKET,
        "eor_id": eor_id, "eow_id": eow_id, "lat_ids": lat_ids,
    }}

    # ========================================================
    # TEST 1: byte-compare training vs inference tokenization
    # ========================================================
    print("\n" + "="*60)
    print("TEST 1: byte-compare training vs inference tokenization")
    print("="*60, flush=True)
    bc_pass = 0
    bc_fail_details = []
    for i, row in enumerate(med_rows[:N_BYTECOMPARE]):
        train_ids = build_input_ids_direct(row, bos_id, lat_ids, eor_id, eow_id)
        # inference-path: build [<BOS>, <LAT>] + inference-tokenized source words
        # Compare only the SOURCE part (up to first EOR). Both paths must agree
        # on how source is tokenized after <|latency|>.
        first_eor_idx = train_ids.index(eor_id)
        train_source_prefix = train_ids[:first_eor_idx]  # [BOS, LAT, src_ids...]

        # inference re-tokenizes the source words
        full_ids, spans = tokenize_source_by_words(tok, row["source_chunks"][0])
        inf_source_prefix = [bos_id, lat_ids[row["latency"]]] + full_ids

        if train_source_prefix == inf_source_prefix:
            bc_pass += 1
        else:
            bc_fail_details.append({
                "index": row["index"],
                "train_prefix_len": len(train_source_prefix),
                "inf_prefix_len": len(inf_source_prefix),
                "first_diff_pos": next(
                    (i for i, (a, b) in enumerate(zip(train_source_prefix, inf_source_prefix))
                     if a != b), None
                ),
                "train_first_10": train_source_prefix[:10],
                "inf_first_10": inf_source_prefix[:10],
            })
    results["test1_bytecompare"] = {
        "n_tested": N_BYTECOMPARE, "n_pass": bc_pass, "n_fail": N_BYTECOMPARE - bc_pass,
        "fail_details": bc_fail_details[:5],
    }
    print(f"  {bc_pass}/{N_BYTECOMPARE} rows byte-identical for [BOS,LAT,src_chunk_1]", flush=True)
    if bc_fail_details:
        for d in bc_fail_details[:3]:
            print(f"  FAIL row {d['index']}: first_diff_pos={d['first_diff_pos']}", flush=True)
            print(f"    train[:10]={d['train_first_10']}")
            print(f"    inf[:10]  ={d['inf_first_10']}")

    # ========================================================
    # TEST 2 (v2): Multi-latency positional p(EOR) sweep
    # ========================================================
    print("\n" + "="*60)
    print("TEST 2 (v2): Multi-latency positional p(EOR)")
    print("  30 rows (10 per bucket) × 5 latency variants")
    print("  Variants: low, low+medium, medium, medium+high, high")
    print("="*60, flush=True)

    # Sample 10 rows per bucket
    bucket_rows = {"low": [], "medium": [], "high": []}
    for r in rows:
        if r["latency"] in bucket_rows and len(bucket_rows[r["latency"]]) < N_PER_BUCKET:
            bucket_rows[r["latency"]].append(r)
    all_rows = bucket_rows["low"] + bucket_rows["medium"] + bucket_rows["high"]
    print(f"  Sampled {len(bucket_rows['low'])} low + {len(bucket_rows['medium'])} medium + {len(bucket_rows['high'])} high = {len(all_rows)} rows", flush=True)

    def variant_prompt_ids(variant_names):
        """Build [BOS, *lat_ids_for_variant]."""
        return [bos_id] + [lat_ids[n] for n in variant_names]

    multi_lat_results = []
    for row in all_rows:
        src_text = " ".join(row["source_chunks"])
        src_words = src_text.split()
        n_src_words = len(src_words)
        # Annotator commit positions (word units)
        commit_word_positions = []
        cum = 0
        for sc in row["source_chunks"]:
            cum += len(sc.split())
            commit_word_positions.append(cum)
        # Tokenize source once
        full_ids, spans = tokenize_source_by_words(tok, src_text)

        row_result = {
            "index": row["index"],
            "annotator_bucket": row["latency"],
            "n_src_words": n_src_words,
            "commit_word_positions": commit_word_positions,
        }
        for variant_label, variant_names in LATENCY_VARIANTS:
            prompt_ids = variant_prompt_ids(variant_names)
            n_prompt = len(prompt_ids)
            input_ids = prompt_ids + full_ids
            # Query position for word k: len(prompt) + sum(len(spans[:k+1])) - 1
            qp_by_word = []
            cursor = n_prompt
            for k in range(n_src_words):
                cursor += len(spans[k])
                qp_by_word.append(cursor - 1)
            with torch.no_grad():
                input_t = torch.tensor([input_ids], device="cuda", dtype=torch.long)
                out = model(input_t)
                logits = out.logits[0]
                probs = F.softmax(logits.float(), dim=-1)
            p_eor = probs[qp_by_word, eor_id].cpu().tolist()
            argmax = probs[qp_by_word].argmax(dim=-1).cpu().tolist()
            row_result[f"p_eor_{variant_label}"] = p_eor
            row_result[f"argmax_{variant_label}"] = argmax
        multi_lat_results.append(row_result)

    # Aggregate across latency variants, split by annotator bucket
    def agg_at_position(bucket, variant_label, position_key):
        """Aggregate p_eor at a given position across rows in `bucket`.
        position_key: 'w1' → word 1 (idx 0); 'commit' → first commit position (1-indexed → idx-1)."""
        vals = []
        for r in multi_lat_results:
            if r["annotator_bucket"] != bucket:
                continue
            p_eor = r[f"p_eor_{variant_label}"]
            if position_key == "w1":
                if len(p_eor) >= 1:
                    vals.append(p_eor[0])
            elif position_key == "w3":
                if len(p_eor) >= 3:
                    vals.append(p_eor[2])
            elif position_key == "commit1":
                c = r["commit_word_positions"][0]
                if c - 1 < len(p_eor):
                    vals.append(p_eor[c - 1])
        return vals

    variant_labels = [v[0] for v in LATENCY_VARIANTS]
    agg_table = {}
    for bucket in ["low", "medium", "high"]:
        for pos_key in ["w1", "w3", "commit1"]:
            agg_table[f"{bucket}|{pos_key}"] = {}
            for vl in variant_labels:
                vals = agg_at_position(bucket, vl, pos_key)
                agg_table[f"{bucket}|{pos_key}"][vl] = st.mean(vals) if vals else None

    # Print big table
    print(f"\n  p(EOR) means by annotator bucket × latency variant × query position:")
    for pos_key in ["w1", "w3", "commit1"]:
        print(f"\n  --- position: {pos_key} ---")
        print(f"    {'bucket':<8} " + " ".join(f"{vl:>13}" for vl in variant_labels))
        for bucket in ["low", "medium", "high"]:
            key = f"{bucket}|{pos_key}"
            row_strs = [f"{agg_table[key][vl]:>13.4f}" if agg_table[key][vl] is not None else f"{'--':>13}" for vl in variant_labels]
            print(f"    {bucket:<8} " + " ".join(row_strs))

    # Interpolation validation: is p_eor(low+medium) between p_eor(low) and p_eor(medium)?
    print("\n  --- interpolation validation (avg over all rows) ---")
    interp_pass = {"low+medium": True, "medium+high": True}
    for row in multi_lat_results:
        n = len(row.get("p_eor_low", []))
        if n < 1:
            continue
        p_low = row["p_eor_low"][0]
        p_med = row["p_eor_medium"][0]
        p_high = row["p_eor_high"][0]
        p_lm = row["p_eor_low+medium"][0]
        p_mh = row["p_eor_medium+high"][0]
        # low+medium should be BETWEEN low and medium (order-independent)
        lo, hi = sorted([p_low, p_med])
        if not (lo - 0.02 <= p_lm <= hi + 0.02):  # small tolerance
            interp_pass["low+medium"] = False
        lo, hi = sorted([p_med, p_high])
        if not (lo - 0.02 <= p_mh <= hi + 0.02):
            interp_pass["medium+high"] = False

    # Mean of interpolated vs bracketing
    lm_frac = []
    mh_frac = []
    for row in multi_lat_results:
        if not row.get("p_eor_low"):
            continue
        p_low, p_med, p_high = row["p_eor_low"][0], row["p_eor_medium"][0], row["p_eor_high"][0]
        p_lm, p_mh = row["p_eor_low+medium"][0], row["p_eor_medium+high"][0]
        denom = p_low - p_med
        if abs(denom) > 1e-4:
            lm_frac.append((p_low - p_lm) / denom)  # 0=at low, 1=at medium
        denom = p_med - p_high
        if abs(denom) > 1e-4:
            mh_frac.append((p_med - p_mh) / denom)

    print(f"    low+medium: {'✅ WITHIN' if interp_pass['low+medium'] else '❌ OUT OF'} [low, medium] range for all rows")
    print(f"    medium+high: {'✅ WITHIN' if interp_pass['medium+high'] else '❌ OUT OF'} [medium, high] range for all rows")
    if lm_frac:
        print(f"    low+medium interpolation fraction: mean {st.mean(lm_frac):.3f}  (0=at low, 1=at medium)")
    if mh_frac:
        print(f"    medium+high interpolation fraction: mean {st.mean(mh_frac):.3f}  (0=at medium, 1=at high)")

    results["test2_multilatency"] = {
        "n_rows_per_bucket": N_PER_BUCKET,
        "variants": variant_labels,
        "per_row": multi_lat_results,
        "aggregate_table": agg_table,
        "interpolation_within_range": interp_pass,
        "interpolation_fraction_low_medium_mean": st.mean(lm_frac) if lm_frac else None,
        "interpolation_fraction_medium_high_mean": st.mean(mh_frac) if mh_frac else None,
    }

    # ========================================================
    # TEST 3 (legacy compat for diagnosis): rebuild positional_results
    # from multi_lat medium rows, so diagnosis code below still works.
    # ========================================================
    positional_results = []
    for r_ml in multi_lat_results:
        if r_ml["annotator_bucket"] != "medium":
            continue
        # p_eor at MEDIUM latency variant only (legacy behaviour)
        p_eor = r_ml["p_eor_medium"]
        argmax = r_ml["argmax_medium"]
        positional_results.append({
            "index": r_ml["index"],
            "n_src_words": r_ml["n_src_words"],
            "commit_word_positions": r_ml["commit_word_positions"],
            "p_eor_by_word": p_eor,
            "argmax_by_word": argmax,
        })

    # Aggregate: p(EOR) at word 1 vs at annotator commit positions
    p_eor_at_w1 = [p["p_eor_by_word"][0] for p in positional_results]
    p_eor_at_commits = []  # p(EOR) at first annotator commit position
    p_eor_at_noncommits = []  # p(EOR) at non-commit positions
    for p in positional_results:
        first_commit = p["commit_word_positions"][0]
        if first_commit <= len(p["p_eor_by_word"]):
            p_eor_at_commits.append(p["p_eor_by_word"][first_commit - 1])
        # non-commit = all word positions NOT in commit_word_positions (1-indexed)
        commit_set = set(p["commit_word_positions"])
        for wi in range(1, len(p["p_eor_by_word"]) + 1):
            if wi not in commit_set:
                p_eor_at_noncommits.append(p["p_eor_by_word"][wi - 1])

    n_argmax_eor_at_w1 = sum(1 for p in positional_results if p["argmax_by_word"][0] == eor_id)
    results["test2_positional"] = {
        "per_row": positional_results,
        "aggregate": {
            "p_eor_at_word1_mean": st.mean(p_eor_at_w1),
            "p_eor_at_word1_median": st.median(p_eor_at_w1),
            "p_eor_at_annotator_commit_mean": st.mean(p_eor_at_commits) if p_eor_at_commits else None,
            "p_eor_at_annotator_commit_median": st.median(p_eor_at_commits) if p_eor_at_commits else None,
            "p_eor_at_noncommit_mean": st.mean(p_eor_at_noncommits) if p_eor_at_noncommits else None,
            "p_eor_at_noncommit_median": st.median(p_eor_at_noncommits) if p_eor_at_noncommits else None,
            "n_argmax_eor_at_word1": n_argmax_eor_at_w1,
            "n_rows": len(positional_results),
        },
    }
    ag = results["test2_positional"]["aggregate"]
    print(f"  p(EOR) at word 1:              mean {ag['p_eor_at_word1_mean']:.4f}  median {ag['p_eor_at_word1_median']:.4f}")
    print(f"  p(EOR) at annotator commit:    mean {ag['p_eor_at_annotator_commit_mean']:.4f}  median {ag['p_eor_at_annotator_commit_median']:.4f}")
    print(f"  p(EOR) at non-commit position: mean {ag['p_eor_at_noncommit_mean']:.4f}  median {ag['p_eor_at_noncommit_median']:.4f}")
    print(f"  argmax == EOR at word 1: {n_argmax_eor_at_w1}/{len(positional_results)}", flush=True)

    # ========================================================
    # TEST 3 SKIPPED: latency-swap now subsumed by TEST 2 v2's multi-latency
    # cross-bucket sweep. Build the same aggregates from multi_lat_results
    # for the diagnosis code below.
    # ========================================================
    p_eor_w1_by_lat = {"low": [], "medium": [], "high": []}
    p_eor_w3_by_lat = {"low": [], "medium": [], "high": []}
    for r in multi_lat_results:
        for lat in ["low", "medium", "high"]:
            p_eor = r[f"p_eor_{lat}"]
            if len(p_eor) >= 1:
                p_eor_w1_by_lat[lat].append(p_eor[0])
            if len(p_eor) >= 3:
                p_eor_w3_by_lat[lat].append(p_eor[2])
    results["test3_latency_swap"] = {
        "aggregate": {
            "p_eor_at_word1_mean": {lat: st.mean(p_eor_w1_by_lat[lat]) for lat in ["low", "medium", "high"]},
            "p_eor_at_word3_mean": {lat: st.mean(p_eor_w3_by_lat[lat]) if p_eor_w3_by_lat[lat] else None for lat in ["low", "medium", "high"]},
            "delta_p_eor_word1_low_minus_high": st.mean(p_eor_w1_by_lat["low"]) - st.mean(p_eor_w1_by_lat["high"]),
            "delta_p_eor_word3_low_minus_high": (
                st.mean(p_eor_w3_by_lat["low"]) - st.mean(p_eor_w3_by_lat["high"])
                if p_eor_w3_by_lat["low"] else None
            ),
        },
    }

    # ========================================================
    # TEST 4: EOW probe (write-mode symmetric) — unchanged, medium rows only
    # ========================================================
    print("\n" + "="*60)
    print("TEST 4: EOW probe on write positions from training input")
    print("="*60, flush=True)
    print("  Feed training-shape input up to just past EOR + 1 target token")
    print("  For each write position, record p(EOW), argmax, rank(EOW)", flush=True)

    eow_results = []
    for row in bucket_rows["medium"]:
        train_ids = build_input_ids_direct(row, bos_id, lat_ids, eor_id, eow_id)
        # Find first EOR and first EOW positions
        first_eor = train_ids.index(eor_id)
        first_eow = train_ids.index(eow_id)
        # Target chunk lies between first_eor+1 and first_eow-1
        # For each position in [first_eor+1, first_eow], record p(EOW) at logit index-1
        # (i.e., predict token at position i from prefix ending at i-1)
        tgt_positions = list(range(first_eor + 1, first_eow + 1))  # positions of target tokens + EOW

        with torch.no_grad():
            input_t = torch.tensor([train_ids], device="cuda", dtype=torch.long)
            out = model(input_t)
            logits = out.logits[0]
            probs = F.softmax(logits.float(), dim=-1)

        # Query position (0-indexed) for predicting target token at position tp: tp - 1
        p_eow_at_tgt_pos = []
        argmax_at_tgt_pos = []
        for tp in tgt_positions:
            qp = tp - 1
            p_eow_at_tgt_pos.append(float(probs[qp, eow_id].item()))
            argmax_at_tgt_pos.append(int(probs[qp].argmax().item()))

        # Where should EOW appear? At position first_eow.
        # So logit at qp = first_eow - 1 should predict EOW as argmax.
        # At earlier positions (still in target chunk), argmax should be target token.
        target_chunk_len_tokens = first_eow - first_eor - 1  # count of target tokens
        # First-token position within target chunk: 0 = first target token, ..., target_chunk_len_tokens - 1 = last, target_chunk_len_tokens = EOW pos
        eow_results.append({
            "index": row["index"],
            "target_chunk_len_tokens": target_chunk_len_tokens,
            "p_eow_at_first_target_token": p_eow_at_tgt_pos[0],  # predicting FIRST target token
            "p_eow_at_end_of_target": p_eow_at_tgt_pos[-1] if p_eow_at_tgt_pos else None,  # predicting EOW
            "argmax_at_first_target": argmax_at_tgt_pos[0],
            "argmax_at_end_of_target": argmax_at_tgt_pos[-1] if argmax_at_tgt_pos else None,
        })

    p_eow_first = [r["p_eow_at_first_target_token"] for r in eow_results]
    p_eow_end = [r["p_eow_at_end_of_target"] for r in eow_results if r["p_eow_at_end_of_target"] is not None]
    n_argmax_eow_at_first = sum(1 for r in eow_results if r["argmax_at_first_target"] == eow_id)
    n_argmax_eow_at_end = sum(1 for r in eow_results if r["argmax_at_end_of_target"] == eow_id)

    results["test4_eow"] = {
        "per_row": eow_results,
        "aggregate": {
            "p_eow_at_first_target_mean": st.mean(p_eow_first),
            "p_eow_at_end_of_target_mean": st.mean(p_eow_end) if p_eow_end else None,
            "n_argmax_eow_at_first": n_argmax_eow_at_first,
            "n_argmax_eow_at_end": n_argmax_eow_at_end,
            "n_rows": len(eow_results),
        },
    }
    ag = results["test4_eow"]["aggregate"]
    print(f"  p(EOW) at first target token: mean {ag['p_eow_at_first_target_mean']:.4f}")
    print(f"  p(EOW) at END of target chunk: mean {ag['p_eow_at_end_of_target_mean']:.4f}")
    print(f"  argmax == EOW at first token: {n_argmax_eow_at_first}/{len(eow_results)}")
    print(f"  argmax == EOW at end of chunk: {n_argmax_eow_at_end}/{len(eow_results)}", flush=True)

    # ========================================================
    # Summary + go/no-go decision
    # ========================================================
    print("\n" + "="*60)
    print("SUMMARY + DIAGNOSIS")
    print("="*60, flush=True)

    t1 = results["test1_bytecompare"]
    t2 = results["test2_positional"]["aggregate"]
    t3 = results["test3_latency_swap"]["aggregate"]
    t4 = results["test4_eow"]["aggregate"]

    diagnosis = []
    if t1["n_fail"] > 0:
        diagnosis.append(f"❌ TOKENIZATION MISMATCH: {t1['n_fail']}/{N_BYTECOMPARE} rows byte-differ. Fix inference before anything else.")
    else:
        diagnosis.append(f"✅ Tokenization byte-identical: {t1['n_pass']}/{N_BYTECOMPARE}")

    p_eor_ratio = t2["p_eor_at_annotator_commit_mean"] / max(t2["p_eor_at_noncommit_mean"], 1e-9) if t2["p_eor_at_noncommit_mean"] else None
    if p_eor_ratio and p_eor_ratio > 3.0:
        diagnosis.append(f"✅ Model DID learn commit positions: p(EOR)@commit/p(EOR)@noncommit = {p_eor_ratio:.2f}× — threshold too permissive is the issue → soft-commit sweep")
    elif t2["p_eor_at_noncommit_mean"] and t2["p_eor_at_noncommit_mean"] > 0.20:
        diagnosis.append(f"⚠️ p(EOR) UNIFORMLY HIGH at non-commit positions (mean {t2['p_eor_at_noncommit_mean']:.3f}) — Test B α=5 likely over-boosted → retrain v5 with α<5")
    else:
        diagnosis.append(f"? Ambiguous: p(EOR)@commit={t2['p_eor_at_annotator_commit_mean']:.3f} vs @noncommit={t2['p_eor_at_noncommit_mean']:.3f}")

    delta_w1 = abs(t3["delta_p_eor_word1_low_minus_high"])
    if delta_w1 < 0.05:
        diagnosis.append(f"❌ LATENCY CONDITIONING BROKEN: Δp(EOR)@w1 (low-high) = {t3['delta_p_eor_word1_low_minus_high']:.4f} < 0.05 → retrain with stronger init differentiation")
    else:
        diagnosis.append(f"✅ Latency conditioning works: Δp(EOR)@w1 (low-high) = {t3['delta_p_eor_word1_low_minus_high']:.4f}")

    if t4["p_eow_at_first_target_mean"] > 0.20:
        diagnosis.append(f"⚠️ p(EOW) HIGH at first target token (mean {t4['p_eow_at_first_target_mean']:.3f}) — EOW is over-eager symmetric to EOR")
    else:
        diagnosis.append(f"✅ p(EOW) is low at first target token (mean {t4['p_eow_at_first_target_mean']:.3f})")

    # Interpolation verdict (from test 2 v2)
    t2v2 = results["test2_multilatency"]
    ip_lm = t2v2["interpolation_within_range"]["low+medium"]
    ip_mh = t2v2["interpolation_within_range"]["medium+high"]
    lm_frac = t2v2.get("interpolation_fraction_low_medium_mean")
    mh_frac = t2v2.get("interpolation_fraction_medium_high_mean")
    if ip_lm and ip_mh:
        diagnosis.append(f"✅ Interpolation mechanism (feed-both-tokens) produces intermediate p(EOR) — low+medium frac={lm_frac:.2f}, medium+high frac={mh_frac:.2f}")
    else:
        diagnosis.append(f"❌ Interpolation mechanism BROKEN: low+medium in-range={ip_lm}, medium+high in-range={ip_mh}. Try averaging embeddings instead.")

    for line in diagnosis:
        print(f"  {line}", flush=True)
    results["diagnosis"] = diagnosis

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
