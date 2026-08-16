# docs/ — reading order

Written for whoever picks up the project between sessions (student / Dipankar / me next week). Read in this order.

| File | What it is |
|---|---|
| **`method_overview.md`** | How the annotator works, mechanically. The commit criterion, monotonicity, chunk grouping, EAST interleave. Read first — everything else assumes it. |
| **`hypotheses.md`** | The seven falsifiable hypotheses (H1–H7) we've tested or queued to test. Each has a rationale, a prediction, and — where done — the outcome. This is the backbone; experiments trace back to hypotheses. |
| **`experiments.md`** | The runs and results. Which configs we've tried, what each found, cross-linked to the hypothesis it addressed. |
| **`random_floor_and_ot.md`** | Intuition + worked examples for two concepts that recur in the results: (1) "random floor" — the matched-chunk-count null Pearson; (2) OT with embedding-grounded ground cost. Read when the tables in `experiments.md` mention "beats random" or when you want to understand why OT catches things JS misses. |
| **`data.md`** | Datasets: SiMT-De-En-660K, WMT15/WMT22 test sets, RWTH gold alignments (URL, format, extract). Also EAST Eq. 4 metric definition. |
| **`next_steps.md`** | What to do next, in order. Blockers, gates, and the sequencing of OT / cross-backbone / scale-up. |

The single source of authority is still:
- `../CLAUDE.md` — the project claim and non-negotiable invariants (kept lean; points to docs/).
- `../METHOD.md` — the annotation algorithm, precisely.
- `../EXPERIMENTS.md` — the ablation grid, baselines, metrics.
- `../TIMELINE.md` — phases and gates.
- `../LOG.md` — append-only run + decision log. Primary record; docs/ is a summary of it.
- `../HOUSEKEEPING.md` — compute, paths, accounts, ops rules.
