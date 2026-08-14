# Related work

Enough to position the project and stop us re-deriving things. Not a literature review.

## The paper we extend

**EAST** — Fu, Liao, Fan, Li, Zhang, Chen, Shi. *LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline.* Findings of ACL 2025 (arXiv 2504.09570). Read Sections 3, Appendix A, Appendix C, Appendix E.4.

Three things they do that matter to us:

1. **GPT-4 generates the training data** (§3.1, Figure 19). It segments into semantic chunks *and* produces the chunk translations at three latency levels. The WMT references appear only as a quality filter (BLEURT < 80 dropped). GPT-4 does not place `<|eor|>`/`<|eow|>` — the authors insert those when interleaving.
2. **They filter out non-monotonic examples** (Appendix C): pairs with unequal source/target chunk counts are dropped, which they say "often result from non-monotonic translations." This is the weakness we target, stated in their own appendix.
3. **Their inference reuses the KV cache**, giving ~49 ms/word against ~977 ms/word for prompt-updating wait-k. We inherit this and must not break it.

Same group published **EASiST** (AAAI 2026), the speech version. Still prompt-based curation. They ship on this line roughly annually.

## Directly adjacent

- **Conversational SimulMT** — Wang, Vu, Shareghi & Haffari (arXiv 2402.10552). Monash. Builds SFT data by segmenting parallel sentences with `fast_align` (~30% error rate, per EAST's Appendix A) and wraps it in a chat template. Dialogue *format*, not dialogue *data*. Relevant if we attempt the conversational extension.
- **Simul-LLM** (ACL 2024), **TransLLaMa** (Findings EMNLP 2024), **SimulMask/SM²** (EMNLP 2024), **DST** (ACL 2024) — decoder-only LLM SiMT with fixed policies, learned wait-tokens, or bespoke architectures. Context, not competition.

## Signals we are not the first to consider

Read enough of these to cite them correctly and not claim their ideas:

- **ITST** (EMNLP 2022) — information-transport policy, OT-*inspired*, encoder-decoder.
- **DiG-SST** (AAAI 2024) — read/write from expected divergence in translation distributions under future input. Encoder-decoder speech.
- **REINA** (AAAI 2026) — trains a read/write policy from mutual-information gain, computed by comparing full vs truncated input. Whisper, speech. **Structurally the closest published idea to our criterion**, applied at training time to supervise a policy head rather than to annotate data. Cite it and distinguish clearly.
- **FAST** (EMNLP 2023) — shows streaming representations differ substantially from full-utterance ones, and distills the gap. Same group as EAST.
- **Local Agreement / hold-n** (Polák et al., IWSLT 2022) — commits on surface-string stability across re-decodes.
- **AlignAtt / EDAtt** (Papi et al.) — attention-derived policies for encoder-decoder.

## What OT is claimed to add

Embedding-grounded OT distinguishes uncertainty among *semantically near* candidates from uncertainty among *distant* ones; KL cannot. The general principle is old — Word Mover's Distance (ICML 2015), Wasserstein losses, Sinkhorn distillation — and has been applied to next-token distributions elsewhere. **We are not claiming the tool is new.** We are claiming it is the right criterion for a commit decision, and the OT-vs-KL ablation is what tests that. If KL matches, we say so and ship the cheaper criterion.

## Directions already considered and dropped

Recorded so nobody spends a month rediscovering why:

- **Wait-k-specific LoRA adapters with an information-theoretic gate.** Per-step adapter switching invalidates the KV cache and self-penalises under AL-CA; adjacent-k adapters are probably not distinct enough to mix; a single multipath LoRA with a latency token likely matches it.
- **Distilling full-context confidence into a prefix model.** Already done — Future-Guided Incremental Transformer (AAAI 2021), FAST (EMNLP 2023). Also risks training the model to be confidently wrong where information is genuinely absent from the prefix.
- **The commit criterion as an inference-time policy.** Requires the full source at test time. Fatal. This is precisely why the method lives in data construction instead — see `METHOD.md` §1.