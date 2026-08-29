# results/annotate/

Annotator output matrices. The single source of truth for OT chunk placements.

## Convention

```
results/annotate/
├── README.md                    ← you are here
├── {annotator-model}/           ← e.g. gemma-4-E2B-it, EAST-8B, gemma-4-E4B-it
│   ├── {lang-pair}/             ← e.g. de-en, en-de, ar-en
│   │   └── matrices.jsonl       ← one row per source sentence, gitignored (100-300MB)
│   └── ...
└── ...
```

**Keyed by (annotator model, language pair) — NOT by experiment tag.** Rationale: the same annotator on the same corpus is reusable across many downstream experiments. E.g., `east_8b_curated`, `east_8b_curated_waitk`, and `east_8b_curated_conv` (three different training recipes) can all consume the same `gemma-4-E2B-it/de-en/matrices.jsonl` when the experiment specifies cross-annotation.

## Currently landed (v6b baseline annotations, moved from _archive/ on 2026-08-29)

| Annotator | Pair | Rows | Corpus |
|---|---|---|---|
| gemma-4-E2B-it | de-en | 9,612 | curated v1 |
| gemma-4-E2B-it | en-de | 9,937 | curated v1 |
| gemma-4-E2B-it | ru-en | 9,804 | curated v1 |
| gemma-4-E2B-it | en-ru | 9,978 | curated v1 |
| gemma-4-E2B-it | ar-en | 10,000 | curated v1 |
| gemma-4-E2B-it | en-ar | 10,000 | curated v1 |
| gemma-4-E2B-it | vi-en | 10,000 | curated v1 |
| gemma-4-E2B-it | en-vi | 10,000 | curated v1 |

Total: 79,331 unique sentence-pair annotations. Each was expanded to 3 latency variants (low/medium/high) at SFT-dataset build time → ~238K SFT rows (see `_archive/results/v6b_gemma_2b/sft_dataset_multilingual_v6b_htgt_final.json`).

These matrices were produced by 16 PBS jobs across Aug 22-24 (see `_archive/jobs/v6b_gemma_2b/htgt_annot/`). For de/ru, the annotation ran in two shards (`_a`/`_b` covering disjoint 5K-row halves per direction); those are now concatenated into a single `matrices.jsonl` per pair here.

## Adding a new annotator

Say you want to annotate with EAST-8B:

1. Ensure `configs/{tag}.yaml` names `annotate.annotator: EAST-8B` (or its HF id / local path).
2. Fire the annotation stage: `qsub jobs/annotate/{tag}_de-en.pbs` etc. (or `bin/02_annotate` on a GPU box).
3. Output lands at `results/annotate/EAST-8B/{pair}/matrices.jsonl` automatically.
4. Downstream stages (SFT dataset build, training) reference the config's annotator; they know to look here.

## Cross-annotation experiments

For Fig 5 in `docs/followup-experiments.md`: same corpus, different backbone, but annotations from Gemma-4-E2B-it. The training PBS reads `configs/{tag}.yaml`'s `annotate.annotator: gemma-4-E2B-it` and pulls `matrices.jsonl` from `results/annotate/gemma-4-E2B-it/{pair}/` — no re-annotation needed, no duplication.

## File format

Each line is one JSON dict per source sentence pair:

```json
{
  "index": 100000,
  "source": "…",
  "target": "…",
  "src_lang": "de",
  "tgt_lang": "en",
  "commit_positions": [3, 7, 12, 18, 22],
  "chunks_src": ["…", "…"],
  "chunks_tgt": ["…", "…"],
  "chunks_per_sent": 5,
  "latency": "medium"  // OT-derived, used by build_sft_dataset for the 3-bucket split
}
```

Full schema in `src/annotator/annotate.py`; the write happens at the end of `annotate_pair()`.
