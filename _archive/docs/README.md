# docs/_archive/

Documents no longer maintained. Read only for historical context.

## `method-formal.md` (was top-level `METHOD.md`)
The formal spec of the annotation algorithm — notation, commit criterion, chunk emission, EAST-format wrapping. Written for the paper's method section; supersedes nothing in `docs/method.md` (which is the implementation walkthrough) but reads as pure math. If the paper's method section changes, edit this file to match; then port back to `docs/method.md`.

## `phase1-annotator-experiments.md` (was `docs/03-phase1_*.md`)
Phase 1 (Aug 14–16) annotator experiments — the tau-sweep, entropy-sweep, and random-floor runs that closed Gate 1. Cross-referenced from `LOG.md` for evidence but doesn't drive current work.

## `random-floor-and-ot.md` (was `docs/04-random_floor_and_ot.md`)
Random-floor baseline construction + the OT chunking derivation. Reference material for the paper's method section.

## `phase2-sft-and-streaming.md` (was `docs/05-phase2_sft_and_streaming.md`)
Phase 2 (Aug 16–22) SFT recipe experiments and streaming inference walkthrough. Documents the v5 → v6 → v6b progression up to the freeze. Everything current is in `docs/method.md` + `LOG.md`.

## `TIMELINE.md`
14-week project timeline with phase gates. Superseded by `LOG.md` — every decision-worthy event is now dated in the log.

## `OPTIONALS.md`
74 KB dumping ground of paper-strategy discussion, venue targeting, method-improvement backlog, and reviewer-anticipation. ~80% is stale (ACL window closed; IWSLT preferred). Grep for a specific topic if needed; do not read cover to cover.

## When to look here
- Method section wording for the paper → `method-formal.md`.
- "How did we decide τ = 0.30" → `phase1-annotator-experiments.md`.
- "Why v6 pivoted to instruct backbones" → `phase2-sft-and-streaming.md` + `LOG.md` 2026-08-21.
- Everything else → **do not**. Current docs are in the parent directory.
