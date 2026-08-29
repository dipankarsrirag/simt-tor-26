"""Probe: does v6's build_assistant_body + apply_chat_template + retokenize
preserve the annotator's per-chunk BPE ids?

For 5 sample rows across (ar, vi, de, ru, en) sources, we:
  1. Take the annotator's original source/target token ids (from matrices.jsonl).
  2. Commit at tau=0.30 → chunks + chunk_ids (via _chunks_from_commit).
  3. Render the assistant body via build_assistant_body (strings).
  4. Apply Gemma-4-it chat template.
  5. Retokenize the full chat text.
  6. Locate assistant-body span; split on EOR/EOW ids; extract per-chunk ids.
  7. Compare per-chunk ids against annotator's original chunk_ids.

Reports: for each row, whether train-time tokens == annotator tokens.

If ANY row fails, the string-round-trip path is broken and we must build
input_ids directly from chunk_ids at train time (bypass strings).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from transformers import AutoTokenizer

from src.annotator.annotate import _chunks_from_commit, _enforce_monotone, _is_cjk_lang
from src.annotator.east_format import (
    END_OF_READ, END_OF_WRITE,
    build_assistant_body, build_user_instruction,
    DEFAULT_SYSTEM_PROMPT,
)

TOK_DIR = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended-v6"
POOL_DIR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/multilingual_source_pool_v5_per_direction")
MATRICES_ROOT = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/")

# Pick one directory per source-lang variety.
PROBE_DIRS = ["ar-en", "vi-en", "de-en", "ru-en", "en-ar", "en-vi", "en-de", "en-ru"]
N_PER_DIR = 2   # 2 rows per direction

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


def find_subsequence(hay, needle, start=0):
    """Find first index i >= start where hay[i:i+len(needle)] == needle."""
    if not needle:
        return start
    L = len(needle)
    for i in range(start, len(hay) - L + 1):
        if hay[i:i+L] == needle:
            return i
    return -1


def probe_row(tok, row_pool: dict, rec: dict, src_lang: str, tgt_lang: str):
    """One probe. Returns dict with `ok` bool and diagnostic fields."""
    src_clean = row_pool["source"].strip()
    tgt_clean = row_pool["target"].strip()

    # Annotator's original tokenization (no leading space, as annotator saw it)
    src_ids_orig = tok(src_clean, add_special_tokens=False)["input_ids"]
    tgt_ids_orig = tok(tgt_clean, add_special_tokens=False)["input_ids"]
    n, m = rec["n_src_tok"], rec["n_tgt_tok"]
    matches_shape = (len(src_ids_orig) == n and len(tgt_ids_orig) == m)
    if not matches_shape:
        return {
            "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
            "ok": False, "reason": f"annotator-shape mismatch: src {len(src_ids_orig)} vs {n}, tgt {len(tgt_ids_orig)} vs {m}",
        }

    commit = commit_from_matrix(rec["matrix"], TAU, n)
    if not commit or max(commit) == 0:
        return {
            "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
            "ok": False, "reason": "empty/degenerate commit",
        }

    source_chunks, target_chunks, source_chunk_ids, target_chunk_ids = _chunks_from_commit(
        commit, src_ids_orig, tgt_ids_orig, tok, n, src_lang=src_lang
    )
    if len(source_chunks) != len(target_chunks) or not source_chunks:
        return {
            "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
            "ok": False, "reason": "chunk-count mismatch or empty",
        }

    # Build assistant body via strings
    body = build_assistant_body(source_chunks, target_chunks, src_lang, tgt_lang)
    user_instr = build_user_instruction(src_lang, tgt_lang, "medium")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_instr},
        {"role": "assistant", "content": body},
    ]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        messages = [
            {"role": "user", "content": DEFAULT_SYSTEM_PROMPT + "\n\n" + user_instr},
            {"role": "assistant", "content": body},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    text_ids = tok(text, add_special_tokens=False)["input_ids"]

    eor_id = tok(END_OF_READ, add_special_tokens=False)["input_ids"][0]
    eow_id = tok(END_OF_WRITE, add_special_tokens=False)["input_ids"][0]

    # Locate the first EOR; then walk EOR<tgt>EOW<src>EOR... pattern.
    # Walk through text_ids extracting between EOW and EOR (source) and
    # between EOR and EOW (target).
    positions_eor = [i for i, t in enumerate(text_ids) if t == eor_id]
    positions_eow = [i for i, t in enumerate(text_ids) if t == eow_id]
    if len(positions_eor) != len(source_chunks) or len(positions_eow) != len(source_chunks):
        return {
            "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
            "ok": False,
            "reason": f"EOR/EOW counts wrong: got {len(positions_eor)} EOR, {len(positions_eow)} EOW, expected {len(source_chunks)}",
        }

    # Extract per-chunk source/target ids from the round-tripped ids.
    # Chunk k source spans between (positions_eow[k-1]+1 if k>0 else assistant_start, positions_eor[k])
    # We need assistant_start. Find the first EOR and walk back to the last non-source-content token.
    # Simpler: split by [EOR, EOW] sentinels — the alternating segments between them are chunks.
    # segments = [seg_before_first_EOR, EOR, seg_before_first_EOW, EOW, seg_before_second_EOR, EOR, ...]
    # After the ASSISTANT-turn opener but before first EOR is source_chunk[0].
    # After first EOR up to first EOW is target_chunk[0]. Etc.

    # Assistant start: everything before the *first* source_chunk starts. Since
    # source_chunk[0] is the first content after the chat opener + system + user turns,
    # we find the first EOR and the source_chunk[0] ends at positions_eor[0].
    # Source_chunk[0] STARTS at whatever position the assistant body begins.
    # Approach: look for a specific marker anchor — use the last occurrence of
    # the assistant-turn opener string tokenized. But easier: the chunks are
    # split by EOR/EOW; the SOURCE chunk k lies between EOW_{k-1} and EOR_{k}.
    # For k=0, we don't know its start from EOR/EOW alone.

    # Trick: use offset_mapping to find where the assistant body starts in the string.
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == text_ids
    body_char_start = text.rfind(body)  # last occurrence (defensive)
    # First body token: first token whose char start >= body_char_start
    body_tok_start = None
    for i, (a, b) in enumerate(enc.offset_mapping):
        if a >= body_char_start:
            body_tok_start = i
            break
    if body_tok_start is None:
        return {
            "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
            "ok": False, "reason": "cannot find assistant body start in tokenized text",
        }

    per_chunk_src_actual = []
    per_chunk_tgt_actual = []
    cursor = body_tok_start
    for k in range(len(source_chunks)):
        # source chunk k: cursor .. positions_eor[k]
        src_ids_k = text_ids[cursor:positions_eor[k]]
        per_chunk_src_actual.append(src_ids_k)
        cursor = positions_eor[k] + 1
        # target chunk k: cursor .. positions_eow[k]
        tgt_ids_k = text_ids[cursor:positions_eow[k]]
        per_chunk_tgt_actual.append(tgt_ids_k)
        cursor = positions_eow[k] + 1

    # Compare
    src_ok = per_chunk_src_actual == source_chunk_ids
    tgt_ok = per_chunk_tgt_actual == target_chunk_ids

    diag = {
        "index": rec["index"], "src_lang": src_lang, "tgt_lang": tgt_lang,
        "n_chunks": len(source_chunks),
        "src_ok": src_ok, "tgt_ok": tgt_ok,
        "ok": src_ok and tgt_ok,
    }
    if not src_ok:
        # Show first differing chunk
        for k in range(len(source_chunks)):
            if per_chunk_src_actual[k] != source_chunk_ids[k]:
                diag["src_first_diff"] = {
                    "chunk": k,
                    "expected": source_chunk_ids[k],
                    "expected_tokens": [tok.convert_ids_to_tokens(t) for t in source_chunk_ids[k]],
                    "actual": per_chunk_src_actual[k],
                    "actual_tokens": [tok.convert_ids_to_tokens(t) for t in per_chunk_src_actual[k]],
                    "chunk_str": source_chunks[k],
                }
                break
    if not tgt_ok:
        for k in range(len(target_chunks)):
            if per_chunk_tgt_actual[k] != target_chunk_ids[k]:
                diag["tgt_first_diff"] = {
                    "chunk": k,
                    "expected": target_chunk_ids[k],
                    "expected_tokens": [tok.convert_ids_to_tokens(t) for t in target_chunk_ids[k]],
                    "actual": per_chunk_tgt_actual[k],
                    "actual_tokens": [tok.convert_ids_to_tokens(t) for t in per_chunk_tgt_actual[k]],
                    "chunk_str": target_chunks[k],
                }
                break
    return diag


def main():
    print(f"Loading tokenizer {TOK_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)

    total, ok = 0, 0
    src_fail_dirs, tgt_fail_dirs = set(), set()
    for d in PROBE_DIRS:
        src_lang, tgt_lang = d.split("-")
        pool_path = POOL_DIR / f"{d}.json"
        matrices_path = MATRICES_ROOT / f"annot_ot_multi_{d}" / "matrices.jsonl"
        if not pool_path.exists() or not matrices_path.exists():
            print(f"[skip] {d}: missing pool or matrices", flush=True)
            continue
        with open(pool_path) as f:
            pool = json.load(f)
        pool_by_idx = {r["index"]: r for r in pool}

        print(f"\n=== {d} ===", flush=True)
        n_seen = 0
        with open(matrices_path) as f:
            for line in f:
                if n_seen >= N_PER_DIR:
                    break
                rec = json.loads(line)
                if rec["index"] not in pool_by_idx:
                    continue
                # Prefer rows with non-trivial chunk counts (skip tiny ones for interest)
                if rec["n_src_tok"] < 8:
                    continue
                out = probe_row(tok, pool_by_idx[rec["index"]], rec, src_lang, tgt_lang)
                total += 1
                if out.get("ok"):
                    ok += 1
                    print(f"  [OK]  idx={out['index']} chunks={out.get('n_chunks','?')}", flush=True)
                else:
                    print(f"  [FAIL] idx={out['index']} reason={out.get('reason','')}"
                          f" src_ok={out.get('src_ok')} tgt_ok={out.get('tgt_ok')}", flush=True)
                    if "src_first_diff" in out:
                        src_fail_dirs.add(d)
                        fd = out["src_first_diff"]
                        print(f"    src chunk {fd['chunk']} str={fd['chunk_str']!r}", flush=True)
                        print(f"      expected {fd['expected_tokens']}", flush=True)
                        print(f"      actual   {fd['actual_tokens']}", flush=True)
                    if "tgt_first_diff" in out:
                        tgt_fail_dirs.add(d)
                        fd = out["tgt_first_diff"]
                        print(f"    tgt chunk {fd['chunk']} str={fd['chunk_str']!r}", flush=True)
                        print(f"      expected {fd['expected_tokens']}", flush=True)
                        print(f"      actual   {fd['actual_tokens']}", flush=True)
                n_seen += 1

    print(f"\n=== summary ===\n{ok}/{total} rows preserved chunk-ids through the string round-trip", flush=True)
    print(f"src fails in dirs: {sorted(src_fail_dirs)}", flush=True)
    print(f"tgt fails in dirs: {sorted(tgt_fail_dirs)}", flush=True)


if __name__ == "__main__":
    main()
