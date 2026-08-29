"""
Verify: batched OT (ot_divergence_row_batched) matches per-pair OT
(ot_divergence_row) within numerical tolerance.

If this passes, the ~10× speedup is a pure engineering win with no
semantic drift in the tau sweep.

Runs on CPU with a small vocab so it's cheap; the Sinkhorn math is
identical to the H200 path.
"""

from __future__ import annotations

import sys, time
sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from src.annotator.criterion import ot_divergence_row, ot_divergence_row_batched

torch.manual_seed(0)
V, D = 500, 16    # small vocab so it runs on login-node CPU in seconds
m = 25
topk = 32
eps = 0.05
iters = 100

# Random embedding matrix.
E = torch.randn(V, D)

# Random logits → softmax → per-target-token distributions.
p_full = torch.randn(m, V).softmax(dim=-1)
p_pre = torch.randn(m, V).softmax(dim=-1)

t0 = time.time()
ref = ot_divergence_row(p_full, p_pre, embedding_matrix=E, topk=topk, eps=eps, sinkhorn_iters=iters)
t_ref = time.time() - t0

t0 = time.time()
out = ot_divergence_row_batched(p_full, p_pre, embedding_matrix=E, topk=topk, eps=eps, sinkhorn_iters=iters)
t_batched = time.time() - t0

diff = (ref - out).abs()
print(f"n_pairs (m) = {m}, V = {V}, topk = {topk}, eps = {eps}, iters = {iters}")
print(f"per-pair OT: {t_ref:.3f}s, batched OT: {t_batched:.3f}s  (speedup {t_ref/t_batched:.1f}x on CPU)")
print(f"L∞ diff:  {diff.max().item():.6f}")
print(f"L1 diff:  {diff.sum().item():.6f}")
print(f"L2 diff:  {diff.pow(2).sum().sqrt().item():.6f}")

if diff.max().item() < 5e-3:
    print("\nPASS: batched matches per-pair within 5e-3 L∞ (Sinkhorn numerical tolerance).")
else:
    print("\nFAIL: batched deviates too much from per-pair.")
    print(f"  per-pair values: {ref.tolist()[:10]}")
    print(f"  batched values:  {out.tolist()[:10]}")
