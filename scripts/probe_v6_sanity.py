"""End-to-end v6b byte-compare sanity: verify training tokenization ==
annotator tokenization == streaming inference tokenization.

For sample rows across 8 directions (ar-en, vi-en, de-en, ru-en + reverses),
verify:
  1. Direct-ids splice (sft_v6.build_row_ids) produces input_ids whose
     BODY segment (between prefix and suffix) matches per-chunk concat of
     source_chunk_ids + [EOR] + target_chunk_ids + [EOW].
  2. Streaming tokenize_source_by_words(src) produces byte-identical ids
     to tok(src) — i.e., matches the annotator's tokenization.
  3. If we walk streaming through the source word-by-word (concatenating
     per-word spans), we recover the same source_chunk_ids at chunk
     boundaries when we replay the annotator's commit points.

If ALL rows pass on all 3 checks, training and inference are aligned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from transformers import AutoTokenizer

from src.annotator.annotate import _chunks_from_commit, _enforce_monotone, _is_cjk_lang
from src.annotator.east_format import END_OF_READ, END_OF_WRITE
from src.eval.extrinsic import tokenize_source_by_words
from src.train.sft_v6 import build_row_ids, render_chat_open_close_ids

TOK_DIR = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended-v6"
POOL_DIR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/multilingual_source_pool_v5_per_direction")
MATRICES_ROOT = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/")

PROBE_DIRS = ["ar-en", "vi-en", "de-en", "ru-en", "en-ar", "en-vi", "en-de", "en-ru"]
N_PER_DIR = 3
TAU = 0.30


def commit_from_matrix(matrix, tau, n):
    import math
    if not matrix or not matrix[0]:
        return []
    m = len(matrix[0])
    commit = [n] * m
    for i_row, row in enumerate(matrix):
        i = i_row + 1
        for j in range(m):
            if commit[j] < n:
                continue
            d = row[j]
            if not math.isnan(d) and d < tau:
                commit[j] = i
    return _enforce_monotone(commit)


def build_row_dict(pool_row, rec, tok):
    src_clean = pool_row["source"].strip()
    tgt_clean = pool_row["target"].strip()
    src_ids = tok(src_clean, add_special_tokens=False)["input_ids"]
    tgt_ids = tok(tgt_clean, add_special_tokens=False)["input_ids"]
    n, m = rec["n_src_tok"], rec["n_tgt_tok"]
    if len(src_ids) != n or len(tgt_ids) != m:
        return None
    commit = commit_from_matrix(rec["matrix"], TAU, n)
    src_chunks, tgt_chunks, src_chunk_ids, tgt_chunk_ids = _chunks_from_commit(
        commit, src_ids, tgt_ids, tok, n, src_lang=pool_row["src_lang"]
    )
    if not src_chunks:
        return None
    return {
        "index": rec["index"],
        "source": src_clean,
        "target": tgt_clean,
        "src_lang": pool_row["src_lang"],
        "tgt_lang": pool_row["tgt_lang"],
        "latency": "medium",
        "source_chunks": src_chunks,
        "target_chunks": tgt_chunks,
        "source_chunk_ids": src_chunk_ids,
        "target_chunk_ids": tgt_chunk_ids,
    }


def probe(tok, row):
    eor_id = tok.convert_tokens_to_ids(END_OF_READ)
    eow_id = tok.convert_tokens_to_ids(END_OF_WRITE)

    # ── Check 1: direct-ids splice preserves chunk_ids byte-exact
    feat = build_row_ids(
        row, tok, max_length=4096,
        eor_id=eor_id, eow_id=eow_id,
        prefix_cache={}, suffix_cache={},
    )
    prefix_ids, suffix_ids = render_chat_open_close_ids(
        tok, row["src_lang"], row["tgt_lang"], row["latency"]
    )
    input_ids = feat["input_ids"]
    assert input_ids[:len(prefix_ids)] == prefix_ids
    assert input_ids[-len(suffix_ids):] == suffix_ids
    body_ids = input_ids[len(prefix_ids):-len(suffix_ids)]

    # Recompose expected body
    expected_body = []
    for s, t in zip(row["source_chunk_ids"], row["target_chunk_ids"]):
        expected_body.extend(s)
        expected_body.append(eor_id)
        expected_body.extend(t)
        expected_body.append(eow_id)

    check1_ok = body_ids == expected_body

    # Labels sanity: prefix all -100, body/suffix un-masked
    labels = feat["labels"]
    check1b_ok = all(x == -100 for x in labels[:len(prefix_ids)]) \
                  and labels[len(prefix_ids):] == input_ids[len(prefix_ids):]

    # ── Check 2: streaming tokenize matches annotator
    full_ids, word_spans = tokenize_source_by_words(tok, row["source"], row["src_lang"])
    annotator_ids = tok(row["source"], add_special_tokens=False)["input_ids"]
    check2_ok = full_ids == annotator_ids

    # ── Check 3: replay commit points on streaming spans recovers chunks
    # (per-word spans concatenated then sliced by chunk-end positions from
    # annotator's chunk_ids lengths should equal source_chunk_ids)
    concat_ids = [t for span in word_spans for t in span]
    if concat_ids != annotator_ids:
        check3_ok = False
    else:
        cursor = 0
        check3_ok = True
        for expected_chunk in row["source_chunk_ids"]:
            L = len(expected_chunk)
            actual_chunk = concat_ids[cursor:cursor + L]
            if actual_chunk != expected_chunk:
                check3_ok = False
                break
            cursor += L
        if cursor != len(annotator_ids) and check3_ok:
            # There might be leftover if _chunks_from_commit merged tail —
            # that's OK. Only the concatenated-till-cursor prefix matters.
            pass

    return {
        "index": row["index"],
        "dir": f"{row['src_lang']}-{row['tgt_lang']}",
        "n_chunks": len(row["source_chunk_ids"]),
        "check1_body_ok": check1_ok,
        "check1b_labels_ok": check1b_ok,
        "check2_stream_matches_annotator": check2_ok,
        "check3_stream_chunks_match": check3_ok,
        "all_ok": check1_ok and check1b_ok and check2_ok and check3_ok,
    }


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    total = 0
    ok = 0
    fail_details = []
    for d in PROBE_DIRS:
        pool_path = POOL_DIR / f"{d}.json"
        matrices_path = MATRICES_ROOT / f"annot_ot_multi_{d}" / "matrices.jsonl"
        if not pool_path.exists() or not matrices_path.exists():
            print(f"[skip] {d}: missing pool or matrices")
            continue
        with open(pool_path) as f:
            pool = json.load(f)
        pool_by_idx = {r["index"]: r for r in pool}

        print(f"\n=== {d} ===")
        n_seen = 0
        with open(matrices_path) as f:
            for line in f:
                if n_seen >= N_PER_DIR:
                    break
                rec = json.loads(line)
                if rec["index"] not in pool_by_idx:
                    continue
                if rec["n_src_tok"] < 8:
                    continue
                row = build_row_dict(pool_by_idx[rec["index"]], rec, tok)
                if row is None:
                    continue
                out = probe(tok, row)
                total += 1
                if out["all_ok"]:
                    ok += 1
                    print(f"  [OK] idx={out['index']} chunks={out['n_chunks']}")
                else:
                    print(f"  [FAIL] idx={out['index']} {out}")
                    fail_details.append(out)
                n_seen += 1

    print(f"\n=== summary === {ok}/{total} rows pass ALL 3 sanity checks")
    if fail_details:
        print("Failures:")
        for f in fail_details:
            print(" ", f)


if __name__ == "__main__":
    main()
