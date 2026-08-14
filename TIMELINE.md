# Timeline

Fourteen weeks, five phases, four gates. **A gate that fails stops the next phase** — the point is to find out early, not to discover at week twelve that the criterion never had signal.

Weeks are indicative. Gates are not.

---

## Phase 0 — Ground truth (weeks 1–2)

Understand what exists before building anything.

- [ ] Read EAST (arXiv 2504.09570) properly. §3 (both training stages, loss recipe on source+target+special tokens) and Appendices A, C, E.4 are load-bearing.
- [ ] Confirm `SiMT-De-En-660K` (~220K rows per latency level, De→En, WMT15-derived) and `SiMT-Multi-90K` (8 directions) — both fetched by `scripts/download_data.sh`. `Off-Multi-120K` is a stretch-only build (see `HOUSEKEEPING.md` §3).
- [ ] Confirm the shipped `source_chunks`/`target_chunks` fields are the GPT-4 baseline. Reconstruct one training row by hand: interleave with `<|eor|>`/`<|eow|>`, prepend the `low`/`medium`/`high` latency indicator, verify against EAST's Figure 18 format.
- [ ] Get the RWTH De→En gold alignment data (App. E.4). URL is a TODO in `scripts/download_data.sh` — confirm from the paper and re-run the copyq job.
- [ ] **Fix Stage-I scope in `LOG.md`** as a decision entry — Stage II is stretch, see `EXPERIMENTS.md`.

**Gate 0:** you can state, in two sentences, what EAST does and what we are changing. If not, re-read.

---

## Phase 1 — The annotator (weeks 3–5)

This is the project. Everything else is plumbing.

- [ ] Implement `P_full[j]` and `P_pre[i][j]` extraction. Batch the prefixes.
- [ ] Implement the OT criterion (top-k support, embedding cost, Sinkhorn).
- [ ] Implement KL and entropy variants — same interface, swappable.
- [ ] Annotate 200 sentences. Run every sanity check in `METHOD.md` §8.
- [ ] Score our tags and GPT-4's tags on the RWTH intrinsic measure.

**Gate 1 (week 5) — the important one.** Do our tags beat GPT-4's on gold-alignment coverage, and is `i*[j]` non-degenerate?

- Pass → Phase 2.
- Degenerate (commit points track position, not content) → diagnose before proceeding. Do not train on tags you cannot defend.
- Tags fine but OT ≈ KL → proceed with KL, drop the OT framing, tell Dipankar. Still a paper.

---

## Phase 2 — Training and the primary result (weeks 6–10)

- [ ] Annotate 10K, then 50K.
- [ ] SFT pipeline (LLaMA-Factory, following EAST's setup).
- [ ] **Gate 2 (week 8):** one SFT run completes and produces sane streaming output with tags in sensible places. If the model emits no tags, or all tags, stop and debug the data.
- [ ] Train condition A (GPT-4 tags) and condition B (ours), matched.
- [ ] Sweep `tau`; evaluate on WMT15 De→En with the full metric set.

**Gate 3 (week 10):** the primary comparison exists, with AL-CA reported. Whatever it says.

---

## Phase 3 — Ablations (weeks 11–12)

Run in the order given in `EXPERIMENTS.md`. Divergence, then annotator model, then monotonicity, then the rest.

The cross-annotation ablation is not optional — it is the answer to the first question any reviewer asks about self-annotation.

---

## Phase 4 — Write-up (weeks 13–14)

- [ ] Draft. Lead with the intrinsic result, then extrinsic.
- [ ] State the exposure-bias limitation (`METHOD.md` §9) rather than waiting to be asked.
- [ ] Authorship was agreed at project start — see `HOUSEKEEPING.md`.

---

## Stretch A: multilingual (EAST Stage II)

**Only after Gate 3 passes.** Costs a second training stage on top of the Stage-I checkpoint plus multilingual eval — real time, real SU. Skip if Stage I hasn't landed a defensible result.

Rationale if it happens: EAST's Stage II is LoRA on `SiMT-Multi-90K` (8 directions, GPT-4-chunked same way as Stage I) plus `Off-Multi-120K` (OMT training on WMT17-21 test data à la ALMA) to keep full-sentence quality. The `SiMT-Multi-90K` chunks were produced by the **same GPT-4 pipeline as SiMT-De-En-660K**, so re-annotating with our method extends the primary claim across De/Zh/Ru/Cs — where Zh and Cs sit on the reordering side of the divergence axis (`CLAUDE.md` §Separable prefix verb / Fronted object argue this in German; those constructions are more extreme in Zh and Cs).

Two things to settle before running anything:

- **Off-Multi-120K assembly.** Not published on HF. Rebuild from WMT17-21 test data following ALMA (Xu et al. 2024a). `HOUSEKEEPING.md` §3 flags this as a TODO — the assembly script is not written.
- **Which directions actually get re-annotated.** All 8 is expensive. Pick a subset that supports the divergence-widens-with-reordering claim (De/Zh/Cs → En, maybe En→Zh). Argue the subset in `LOG.md` before running.

## Stretch B: document-level SiMT (EAST §4.3)

**Only if Stretch A lands.** Requires a Stage-II-trained model. Test set is WMT22 De/Ru→En grouped by `docid` — no extra data fetch (sacrebleu already ships it, see `scripts/download_data.sh`). EAST evaluates zero-shot; we do the same.

## Stretch C: conversational SiMT

**Only after Gate 3 passes.** Two half-finished halves are worse than one finished result.

Rationale if it happens: in dialogue, the information needed to commit often lives in prior turns — ellipsis, anaphora, elliptical replies. Our annotator reads the same conversational context the deployed model will see; GPT-4 segmenting sentence-by-sentence does not. That predicts the gap *widens* on dialogue relative to news.

Two things to settle before running anything:

- **Latency across turns is unresolved.** Does AL reset per turn or accumulate? Pick one, justify it, state it. There is no standard answer and a reviewer will ask.
- **Check data availability and licensing yourself.** WMT chat-task data and BMELD are the usual candidates; verify current status rather than trusting a citation.

Note: "Conversational SimulMT" (Wang, Vu, Shareghi & Haffari, arXiv 2402.10552) already exists and is a baseline in EAST. Read it first. It is about dialogue *format*, not dialogue *data* — the gap survives, but know exactly where it is.

---

## Standing rules

- Log every run in `LOG.md` before starting the next.
- Weekly checkpoint with Dipankar. Bring the log, not a summary.
- A negative result at any gate is a result. Report it the same week.