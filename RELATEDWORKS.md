# Related work

Enough to position the project and stop us re-deriving things. Not a literature review.

## The paper we extend

**EAST** — Fu, Liao, Fan, Li, Zhang, Chen, Shi. *LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline.* Findings of ACL 2025 (arXiv 2504.09570). Read Sections 3, Appendix A, Appendix C, Appendix E.4.

Four things they do that matter to us:

1. **GPT-4 generates the training data** (§3.1, Figure 19). It segments into semantic chunks *and* produces the chunk translations at three latency levels (`low`/`medium`/`high` — ≈220K rows each in the 660K release). The WMT references appear only as a quality filter (BLEURT < 80 dropped). GPT-4 does not place `<|eor|>`/`<|eow|>` — the authors insert those when interleaving.
2. **They filter out non-monotonic examples** (Appendix C): pairs with unequal source/target chunk counts are dropped, which they say "often result from non-monotonic translations." This is the weakness we target, stated in their own appendix.
3. **Two training stages** (§3.2). *Stage I:* full-weight SFT for one epoch on `SiMT-De-En-660K` (WMT15 De→En training) to activate adaptive read/write. *Stage II:* LoRA on `SiMT-Multi-90K` (8 directions) plus `Off-Multi-120K` (WMT17-21 test data as OMT training, à la ALMA) to generalise multilingually while preserving offline quality. Loss is cross-entropy on **source + target + special tokens** — an intentional break from Wang et al. 2024's target-only masking, so the read/write decision itself is trained. We inherit both stages' *shape*; we scope our project to Stage I (see `EXPERIMENTS.md` and `LOG.md`).
4. **Their inference reuses the KV cache**, giving ~49 ms/word against ~977 ms/word for prompt-updating wait-k. We inherit this and must not break it.

Same group published **EASiST** (AAAI 2026), the speech version. Still prompt-based curation. They ship on this line roughly annually.

## Directly adjacent

- **Conversational SimulMT** — Wang, Vu, Shareghi & Haffari (arXiv 2402.10552). Monash. Builds SFT data by segmenting parallel sentences with `fast_align` (~30% error rate, per EAST's Appendix A) and wraps it in a chat template. Dialogue *format*, not dialogue *data*. Relevant if we attempt the conversational extension.
### LLM SiMT competitors — one-line framing and empirical positioning

All 4 papers below put the streaming intelligence at **inference time** or in **architecture**. We put it in **data construction**. This is the axis on which we differentiate.

- **Simul-LLM** (Agostinelli et al., ACL 2024). Fine-tunes LLaMA-2-7B on wait-k-formatted pairs (source truncated to k ahead of target). Their SFT commits to ONE wait-k value at training time — that's the paper's weakness reviewers raise. We train on variable-latency chunks (cond-B: 28% single-chunk-collapse + rest 2-100+ chunks); model can be deployed under any wait-k or check_argmax at inference. Empirical: their WMT De→En LLaMA-2-7B ≈ 24-26 BLEU at AL 4-6; our cond-B/Gemma-2B/n=10K = 26.94 @ AL 3.54 (comparable-to-better on 3.5× smaller model, matched training size).

- **TransLLaMa** (Koshkin et al., Findings EMNLP 2024). Adds ONE `<wait>` token; LLaMA-2 learns to emit `<wait>` from current context. Reactive policy at inference — inherits classic exposure-bias problem (train-vs-test mismatch on the policy prediction). Our EAST-format SFT trains on the full multi-chunk interleave `src <eor> tgt <eow> src <eor> tgt <eow>...`; the loss covers what the model does AFTER committing, not just when to commit. Empirical: TransLLaMa Table 3 LLaMA-2-7B ≈ 22-24 BLEU at AL 4-6; ours 27-28 at same AL on 2B model.

- **SimulMask / SM²** (EMNLP 2024). Custom attention-masking on standard encoder-decoder transformers — architectural surgery. Works on 300M-700M encoder-decoder MT models. We use unmodified decoder-only LLMs (Gemma-4-E2B/E4B, Qwen3.5-2B tested with zero code changes). Different ecosystem — cite for completeness, not head-to-head.

- **DST — Decoder-only Streaming Transformer** (Guo et al., ACL 2024). Bespoke architecture with streaming-aware self-attention, trained from scratch on the full parallel corpus. Cost story: DST is multi-day from-scratch training on millions of pairs. Ours is 40 min SFT on 1 H200 with 10K training rows on stock Gemma-2B. Same or better BLEU at same AL. This is exactly the paper's "no API cost, no bespoke architecture, no from-scratch training" story.

### The 2×2 that positions us

|  | External-oracle signal | Model-native signal |
|---|---|---|
| **Runtime policy** | Wang et al. 2024 (fast_align annots) | TransLLaMa (`<wait>`), Simul-LLM (wait-k SFT), AlignAtt (attn), DST (streaming attn arch), SimulMask (attn masks), DiG-SST, REINA |
| **Data-construction annotation** | EAST (GPT-4 chunks) | **← we go here (empty cell before us)** |

Every prior LLM-based SiMT paper is in the top-right (model-native runtime policy). EAST is the only one in bottom-left (external-oracle data construction). We occupy the bottom-right — model-native data-construction annotation — which combines the "no runtime policy overhead" of EAST with the "no external API" of the runtime-policy family.

### What we owe empirically to beat them convincingly

1. **Full BLEU-vs-AL curve** with 6-8 wait-k points on Gemma-2B, Gemma-4B, Qwen-2B (in flight).
2. **Compare against a wait-k-trained baseline** (Simul-LLM's actual method) — add cond-C: cond-A variant trained on wait-k-truncated data. ~1 day.
3. **Compare against `<wait>` policy** (TransLLaMa's actual method) — cond-D: add `<wait>` token, SFT with `<wait>` at truncation points. ~1 day.
4. **RWTH-A intrinsic on all 4 methods** — direct policy-quality comparison against human alignments. Blocked on RWTH baseline decision (see 07-next_steps §9).

### Simul-LLM (Agostinelli et al., ACL 2024) — additional context

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