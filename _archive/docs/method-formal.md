# Method

## 1. Setup

Parallel pair: source `S = s_1..s_n`, target `T = t_1..t_m`. Backbone LLM `M`.

Two families of next-token distributions, both teacher-forced on the reference target:

- **Full-source:** `P_full[j] = p_M(y_j | S, T_<j)` — what the model would predict having read everything.
- **Prefix:** `P_pre[i][j] = p_M(y_j | S_<=i, T_<j)` — what it predicts having read only `i` source tokens.

Both require the full source, which we have at data-construction time and never at inference. That asymmetry is the point: training amortises an offline oracle into an autoregressive prediction.

## 2. Commit criterion

Target token `j` is **committable at prefix length `i`** when its predictive distribution has converged:

```
D( P_full[j] , P_pre[i][j] ) < tau
```

Commit point:

```
i*[j] = min { i in 1..n : D(P_full[j], P_pre[i][j]) < tau }
        (fall back to n if the criterion never fires)
```

Then enforce monotonicity — a read/write policy cannot un-read source:

```
i*[j] = max( i*[j] , i*[j-1] )
```

## 3. The divergence D

**Primary: embedding-grounded optimal transport.** Compare the two distributions over the vocabulary with a ground cost given by distance in the model's input embedding space.

The motivation: KL treats "cat" vs "dog" as exactly as wrong as "cat" vs "the". OT with an embedding cost knows the first pair is close. The hypothesis this encodes — *uncertainty among semantically near candidates is committable; uncertainty among semantically distant candidates is not* — is what the OT-vs-KL ablation tests. If KL matches OT, we drop the OT framing and ship the same method with a cheaper criterion. That is a valid outcome, not a failure.

Implementation:

1. Support: `V_k = topk(P_full[j]) ∪ topk(P_pre[i][j])`, renormalise both over `V_k`. Start `k = 128`.
2. Cost matrix `C_ab = 1 - cos(E_a, E_b)` over `V_k`, from the input embedding matrix `E`.
3. Entropic-regularised Sinkhorn. Start `eps = 0.05`; check convergence, it is sensitive.

**Ablation criteria** (same pipeline, swap `D`): KL divergence; entropy of `P_pre[i][j]` alone (ignores `P_full`, tests whether the oracle is doing any work); random placement at matched latency (floor).

## 4. Tag placement

Consecutive target tokens sharing a commit point form one write chunk. Emit EAST's format:

```
[ s_1..s_{i*_1} , <|eor|> , chunk_1 , <|eow|> , s_{i*_1+1}..s_{i*_2} , <|eor|> , chunk_2 , <|eow|> , ... ]
```

Then SFT exactly as EAST: loss on source, target, and special tokens (not the prompt). At inference the model predicts token-by-token and switches between read and write on emitting a tag.

## 5. Annotator = backbone

The annotator is the same model we fine-tune. Cleaner story (no external teacher, no GPT-4 dependency, no distillation), and it means the tags are calibrated to the model that has to act on them.

**The risk this creates is error amplification, not contamination.** Wherever `M` is miscalibrated — German verb-final constructions are the obvious candidate — the tags inherit the miscalibration and training makes `M` confident about it. `docs/experiments.md` includes a cross-annotation ablation that settles this. Run it; do not assume.

## 6. Latency control

EAST gets three latency levels from GPT-4 generating three segmentation granularities, tokenised as `low`/`medium`/`high` in the prompt.

Ours is continuous: `tau` is the knob. Lower `tau` = stricter convergence = later commits = higher latency.

- Sweep `tau` to trace the quality-latency curve.
- Pick `tau` values landing near EAST's published AL points so the curves share an x-axis.
- Optionally bin into `low`/`medium`/`high` prompt tokens to match their interface exactly.

A continuous latency knob is a real advantage over three discrete prompts. Say so in the paper.

## 7. Compute

Per sentence: one full-source pass, plus `n` prefix passes. **Batch the `n` prefixes as a single batched call** — they are independent.

Cost control, in order of preference:

1. **Annotate a subset.** EAST's Figure 6 shows ~10K examples already gets most of the benefit, with refinement up to 100K. Annotate 10–50K, not 660K. This is justified by the source paper, so it costs nothing in review.
2. **Stride over `i`** (evaluate every 2nd or 3rd source token, optionally refine near the crossing).
3. Sentences under 20 words are already filtered out of the corpus by EAST's pipeline.

## 8. Sanity checks before trusting any output

- Plot `i*[j]` against `j` for a handful of sentences. Should be monotone, roughly diagonal, with visible jumps at reordering points. A flat line or a step at the end means the criterion is degenerate.
- **Positional degeneracy is the main failure mode.** If commit points cluster near sentence end, the criterion is measuring position, not committability. Check `i*[j]/n` against `j/m` — if it is essentially the identity, there is no signal.
- Hand-inspect 20 annotated examples against the GPT-4 tags in the released dataset. Disagreements should be interpretable.

## 9. Known limitation to state in the paper

`P_pre[i][j]` conditions on the *reference* prefix `T_<j`; at inference the model conditions on its own output. This exposure-bias gap is inherent to teacher-forced data construction and is shared with EAST. Measure it if time allows; state it regardless.