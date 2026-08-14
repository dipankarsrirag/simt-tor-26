# Log

Append-only. Newest at the top. Two kinds of entry: **decisions** (what we chose and why) and **runs** (what we executed and what happened).

Log the run *before* starting the next one. A run without an entry did not happen.

---

## Template — decision

```
### [DECISION] YYYY-MM-DD — one-line summary
**Context:** what prompted this
**Options:** what was considered
**Chose:** what and why
**Revisit if:** the condition that would change this
```

## Template — run

```
### [RUN] YYYY-MM-DD — run-id
**Config:** backbone / data size / criterion / tau / seed
**Command:** exact invocation
**Result:** numbers, with the metric named
**Read:** what this means for the next step
```

---

<!-- entries below -->

### [DECISION] YYYY-MM-DD — Annotator is the same model as the fine-tuning backbone
**Context:** EAST uses GPT-4 as an external annotator. We need to decide whether to self-annotate or use a larger teacher.
**Options:** (a) same model, (b) larger external annotator, (c) GPT-4 as in EAST.
**Chose:** (a). Cleaner claim — no external teacher, no distillation dependency, and tags are calibrated to the model that must act on them. A larger annotator would likely give better tags but reintroduces exactly the dependency we are criticising.
**Revisit if:** the cross-annotation ablation shows same-model annotation underperforms — that would mean error amplification dominates self-calibration.