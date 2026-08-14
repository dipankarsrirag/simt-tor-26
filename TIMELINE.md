# Timeline

Fourteen weeks, five phases, four gates. **A gate that fails stops the next phase** — the point is to find out early, not to discover at week twelve that the criterion never had signal.

Weeks are indicative. Gates are not.

---

## Phase 0 — Ground truth (weeks 1–2)

Understand what exists before building anything.

- [ ] Read EAST (arXiv 2504.09570) properly. Sections 3 and Appendices A, C, E.4 are load-bearing.
- [ ] Pull `biaofu-xmu/SiMT-De-En-660K`. Inspect the format. Confirm it ships GPT-4 tags.
- [ ] Reconstruct one training example by hand from raw parallel text to interleaved sequence. Confirm you can reproduce EAST's Figure 18 format exactly.
- [ ] Confirm compute availability against `HOUSEKEEPING.md`. EAST used 8×A100 for full-weight tuning of Llama-3-8B. If that is not available, scope the backbone down now and drop the direct-comparison claim.
- [ ] Get the RWTH De→En gold alignment data.

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

## Stretch: conversational SiMT

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