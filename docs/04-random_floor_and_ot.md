# What "random floor" and "OT" are actually doing

Read this alongside `method_overview.md` (mechanics) and `hypotheses.md` (why each experiment). This document gives the intuition behind two concepts that keep coming up in the results tables.

## Random floor

### The problem it addresses

Our commit criterion produces a per-target-token commit trace `i*[j]` and we score it with Pearson(i*/n, j/m). But a completely uninformed policy — "pick chunk boundaries at random" — will produce a monotone trace too, and any monotone trace has some Pearson (usually high — monotonic ≈ diagonal). A Pearson_med of 0.8 doesn't by itself mean the criterion has signal; a random policy with the same chunk count might score 0.8 too.

We need a **matched-chunk-count null**: what Pearson would we get by placing this many chunks *at random*?

### What it computes

For each sentence, at each τ:

1. **Get our criterion's chunk count `k` at this τ.** Ours produces some trace; let `k` = number of consecutive-run chunks in it.
2. **Sample a random monotone commit trace with exactly `k` chunks:**
   - Sample `k` distinct source-token boundaries uniformly from `1..n`, sort → `[b_1, ..., b_k]`.
   - Sample `k-1` split points in target-token space `1..m-1`, sort → `[s_1, ..., s_{k-1}]`.
   - Target tokens in `[0, s_1)` commit at `b_1`; those in `[s_1, s_2)` at `b_2`; etc.
   - This is monotone by construction.
3. Compute Pearson of this random trace.
4. Repeat (2)-(3) 100 times per sentence; average the Pearsons → per-sentence random_Pearson.
5. Aggregate across sentences: random_Pearson_med.
6. Compare: does our-criterion Pearson_med < random_Pearson_med (i.e. *less* diagonal than random with matched chunks = uses signal beyond monotonicity)?

Code: `scripts/phase1_random_floor.py`.

### Concrete example

Sentence with `n=40` source tokens, `m=30` target tokens. Our JS criterion at τ=0.15 produces 5 chunks with trace:

```
i*[j] = [8, 8, 8,          # target tokens 0-2 commit at source position 8
         15, 15, 15, 15,   # target tokens 3-6 commit at source position 15
         22, 22, 22, 22, 22, 22,   # target tokens 7-12 commit at source position 22
         30, 30, 30, 30, 30, 30, 30, 30,   # target tokens 13-20 commit at position 30
         40, 40, 40, 40, 40, 40, 40, 40, 40]   # target tokens 21-29 commit at position 40
```

- **Our Pearson(i*/n, j/m) = 0.94.**
- **Random-at-matched-latency (k=5):** on one trial, we might sample source boundaries `{5, 11, 19, 27, 35}` and target splits `{6, 14, 21, 27}`, giving trace `[5]×6 + [11]×8 + [19]×7 + [27]×6 + [35]×3`. Pearson = 0.96. Averaged over 100 trials: **random_Pearson ≈ 0.93.**

**Read:** our 0.94 doesn't beat random's 0.93 — we're doing about the same as chance. If our trace had been non-monotonic-looking, say `[35, 35, 35, 8, 8, 8, 12, 12, 22, ..., 30, 30]` (some early commits placed late in target, some late commits placed early), Pearson would drop meaningfully below 0.93 and we'd beat random.

### Where this shows up in our results

| Config | τ range where our_med < random_med |
|---|---|
| A (-it, raw, JS) | none |
| B (-it, chat, JS) | none |
| C (base, raw, JS) | τ=0.15 only |
| D (base, raw, OT ≤0.50) | τ=0.20 AND τ=0.30 |
| D-ext (base, raw, OT ≤1.00) | τ=0.30 (τ=1.00 is trivially yes because both collapse) |

OT's "beats random" range is wider than JS's — that's part of the H5-supported evidence.

---

## OT (embedding-grounded optimal transport)

### The problem it addresses

Jensen-Shannon and KL treat the vocabulary as a flat set of symbols. Consider two distributions on 4 vocabulary tokens `{cat, kitten, feline, tiger}`:

- `P` puts 0.5 on `cat`, 0.5 on `kitten`.
- `Q` puts 0.5 on `feline`, 0.5 on `tiger`.

JS(P, Q) sees zero overlap in support and returns near-maximal divergence (≈ ln 2). But semantically, both P and Q are saying "small-ish furry cat-like animal" — the mass just shifted between synonyms. JS ignores that these tokens are related in meaning.

For our commit criterion:
- If `P_full[j]` puts mass on `{mat, floor, couch, sofa}` and `P_pre[i][j]` puts mass on `{bed, floor, chair, couch}`, JS says "far apart, don't commit."
- But the model is expressing the same underlying concept ("furniture/surface"). The commit is essentially safe.

OT knows the difference — if you give it a ground metric where semantically-close tokens are cheap to transport between, then "P shifted mass from cat to feline" has low transport cost, while "P shifted mass from cat to airplane" has high transport cost.

This is METHOD §3's primary claim: **uncertainty among semantically-nearby candidates is committable; uncertainty among semantically-distant candidates is not.** JS can't distinguish the two; OT with an embedding-based ground cost can.

### What it computes

For each `(P_full[j], P_pre[i][j])` pair:

1. **Restrict to a shared support.** `V_k = topk(P_full[j]) ∪ topk(P_pre[i][j])`, size ≤ `2k`. We use `k=128`.
2. **Renormalise** both distributions over `V_k`.
3. **Build the ground cost matrix** `C ∈ R^{|V_k| × |V_k|}` where
   `C[a, b] = 1 - cos(E[a], E[b])`
   and `E` is the model's input-embedding matrix (`gemma-4-E2B` has shape `(262144, 1536)`). Cosine distance is in `[0, 2]`, so `C ∈ [0, 2]`.
4. **Solve the entropic-regularised OT problem:**
   `T* = argmin_T ⟨T, C⟩ - eps · H(T)`
   subject to marginals `T·1 = a` (from `P_full` restricted) and `T^⊤·1 = b` (from `P_pre` restricted). We use `eps=0.05`. `H(T)` is the entropy of the transport plan.
5. **Compute the transport cost** `⟨T*, C⟩` and return it as the divergence value.

The algorithm is **log-stabilised Sinkhorn** — an iterative algorithm that alternately updates the two dual variables in log-space (avoids numerical underflow at small `eps`). We use `pot`'s `ot.bregman.sinkhorn_log` with 200 iterations. Code: `src/annotator/criterion.py::ot_divergence_pair`.

### Concrete example (from an actual annotation position)

Target position `j` where the reference translation is `mat`, source prefix is `Die Katze schläft` (3 of 40 source tokens read):

**`P_full[j]`** (full source `Die Katze schläft auf der Matte und ...`):
```
mat:    0.40
floor:  0.20
couch:  0.15
sofa:   0.10
ground: 0.05
carpet: 0.03
rug:    0.02
...
```

**`P_pre[i=3][j]`** (only `Die Katze schläft` — hasn't seen "Matte" yet):
```
bed:    0.30
floor:  0.25
couch:  0.20
sofa:   0.15
chair:  0.10
...
```

**What JS says.** Top tokens differ (`mat` vs `bed`). Support overlap is small. JS ≈ 0.35 nats. At τ=0.10 (a moderate JS threshold), this position does not commit. Wait.

**What OT says.**
- Cost matrix `C` has `C[mat, bed] ≈ 0.15`, `C[mat, floor] ≈ 0.12`, `C[floor, floor] = 0`, `C[couch, couch] = 0`, `C[sofa, sofa] = 0` — many small entries.
- Optimal transport moves the 0.4 mass from `mat` (P_full) mostly to `bed` and `floor` (P_pre) at small cost; the shared support (`floor`, `couch`, `sofa`) transports at zero cost.
- Transport cost ≈ 0.08. At τ=0.30 (a moderate OT threshold), this position commits.

**JS says wait; OT says commit.** If the reference happens to be `mat`, JS was right and OT gambled wrong. If the reference is `bed` or `floor` or `couch`, OT was right and JS wasted latency.

For our data (WMT De→En), the actual reference is `Matte` → `mat`. So on this specific token JS is technically more faithful. But on many tokens the reference falls anywhere in the near-synonym cluster — and there OT saves latency without hurting the translation.

The empirical question the tau sweep answers: on which sentences (and which target-position types) does this trade-off favour OT vs JS? Answer from Phase 1: on the reordering candidates (idx=537446, 359904, 367208), OT commits where JS refuses, and the resulting late-commit pattern matches GPT-4's. That's the H5-supported catch.

### Where this shows up in our results

| Metric | JS (Config C) | OT-ext (Config D-ext) |
|---|---|---|
| Chunk-count near GPT-4's 4.06 | 3.46 at τ=0.10 | 3.98 at per-sent tau |
| Chunk-count delta mean_abs | 1.44 → 0.60 | 0.62 |
| Reordering catches (top-8) | 5/8 | **6/8** |
| Per-sentence r(GPT-4, ours) | 0.175 | 0.222 (n=47) / 0.306 (n=37 narrow) |
| Cost per sentence | 1.3 s on H200 | 31 s on H200 (~24×) |

### When OT and JS give the same answer

- Positions where the model is very confident (both P_full and P_pre concentrate mass on the same single token) — both criteria return near-zero, both commit.
- Positions where the model has genuinely uncertain support that includes unrelated tokens (e.g. verb-final: model doesn't know if the verb is `announced` or `terminated`) — both criteria return large values, both refuse to commit.

The divergence is at positions where P_full and P_pre have DIFFERENT support but the different tokens are NEARBY in embedding space. JS says far, OT says near. Our results say those are the positions where the improvement lives.

---

## Cheat sheet

| Concept | Question it answers | Implementation |
|---|---|---|
| **Random floor** | Is our Pearson better than chance given the chunk count? | `phase1_random_floor.py` — sample 100 monotone random traces with matched chunks, compute Pearson, average. |
| **JS divergence** | Do the two next-token distributions overlap token-by-token? | `js_divergence` in `criterion.py` — symmetric KL through the midpoint mixture, in nats. |
| **OT with embedding cost** | Do the two next-token distributions concentrate on *semantically similar* tokens? | `ot_divergence_pair` in `criterion.py` — top-k support union, cost `1 - cos(E)`, log-Sinkhorn via `pot`. |
