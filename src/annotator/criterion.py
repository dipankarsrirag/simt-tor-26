"""
Divergence functions for the commit criterion (METHOD §3).

docs/_archive/method-formal.md's primary criterion is embedding-grounded OT (Sinkhorn). JS/KL
are cheap baselines that answer the prior question — do i*[j] traces
have signal at all? OT tests whether uncertainty spread across
semantically-nearby vocabulary tokens is committable (JS calls it "far"
because it ignores the ground metric; OT knows it's near).

All divergence functions take softmaxed probability rows over the full
vocabulary and return non-negative divergence values. Shapes are (V,)
for a single distribution or (..., V) for batched.

OT requires the input-embedding matrix E for the ground cost, so it is
wrapped in a factory that binds `embedding_matrix`, `topk`, and `eps`
before use — call `make_ot(...)` and register the result in CRITERIA.
"""

from functools import partial

import ot  # POT — Python Optimal Transport (https://pythonot.github.io/)
import torch


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric Jensen–Shannon divergence in nats (natural log).

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5 * (P + Q).

    Bounded in [0, ln 2] ≈ [0, 0.693]; a proper metric under sqrt.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_forward(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(P || Q) in nats. Non-symmetric; use JS unless direction is needed."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


@torch.no_grad()
def ot_divergence_pair(
    p_full: torch.Tensor,
    p_pre: torch.Tensor,
    embedding_matrix: torch.Tensor,
    topk: int = 128,
    eps: float = 0.05,
    sinkhorn_iters: int = 200,
) -> torch.Tensor:
    """One-pair OT with embedding-grounded ground cost.

    Support V_k = topk(p_full) ∪ topk(p_pre), size ≤ 2·topk.
    Renormalise both onto V_k; C[a,b] = 1 - cos(E[a], E[b]); log-stabilised
    Sinkhorn via POT (`ot.bregman.sinkhorn_log`).

    p_full, p_pre: (V,) probability rows.
    embedding_matrix: (V, D).
    Returns: scalar tensor on the same device as inputs.
    """
    _, tk_f = p_full.topk(topk)
    _, tk_p = p_pre.topk(topk)
    support = torch.unique(torch.cat([tk_f, tk_p]))  # (n_supp,) with n_supp <= 2*topk
    a = p_full[support]
    b = p_pre[support]
    a = (a / a.sum().clamp_min(1e-30)).float()
    b = (b / b.sum().clamp_min(1e-30)).float()

    E = embedding_matrix[support].float()  # (n_supp, D)
    E_norm = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cost = (1.0 - (E_norm @ E_norm.T)).clamp_min(0.0)  # (n_supp, n_supp) in [0, 2]

    # POT log-stabilised Sinkhorn — returns the transport plan T; cost is (T * C).sum().
    # Keep everything on-device via POT's torch backend.
    T = ot.bregman.sinkhorn_log(a, b, cost, reg=eps, numItermax=sinkhorn_iters)
    return (T * cost).sum()


def ot_divergence_row(
    p_full_row: torch.Tensor,
    p_pre_row: torch.Tensor,
    *,
    embedding_matrix: torch.Tensor,
    topk: int = 128,
    eps: float = 0.05,
    sinkhorn_iters: int = 100,
) -> torch.Tensor:
    """Vectorise ot_divergence_pair over the target-position axis (naive
    Python loop; used as reference impl for the batched version below)."""
    m = p_full_row.shape[0]
    out = torch.zeros(m, device=p_full_row.device, dtype=torch.float32)
    for j in range(m):
        out[j] = ot_divergence_pair(
            p_full_row[j], p_pre_row[j],
            embedding_matrix=embedding_matrix,
            topk=topk, eps=eps, sinkhorn_iters=sinkhorn_iters,
        )
    return out


@torch.no_grad()
def ot_divergence_row_batched(
    p_full_row: torch.Tensor,
    p_pre_row: torch.Tensor,
    *,
    embedding_matrix: torch.Tensor,
    topk: int = 128,
    eps: float = 0.05,
    sinkhorn_iters: int = 100,
) -> torch.Tensor:
    """Batched-across-target-tokens OT: one GPU-saturating log-domain Sinkhorn
    call for all m target tokens at this source-prefix length.

    Same math as ot_divergence_pair but with an added leading (m) batch
    dimension. POT's `sinkhorn_log` batched mode requires a shared cost
    matrix, which we don't have (support varies per j). Custom torch impl.

    Support handling: each j has support = topk(p_full[j]) ∪ topk(p_pre[j]).
    Sizes vary per j once duplicates are collapsed. We pad to a fixed size
    S = 2*topk *including duplicates*: keep both topk lists as-is (possible
    duplicates), which is a valid support (mass is just split across the
    duplicates, and the Sinkhorn cost is the same regardless of how the
    unified support is partitioned). Verified against the per-pair impl in
    scripts/phase2_batched_ot_smoke.py.

    p_full_row: (m, V), p_pre_row: (m, V).
    Returns: (m,) OT distances.
    """
    m, V = p_full_row.shape
    device = p_full_row.device
    S = 2 * topk

    # Per-j topk support — keep duplicates for a fixed padding size.
    _, tk_f = p_full_row.topk(topk, dim=-1)   # (m, topk) int
    _, tk_p = p_pre_row.topk(topk, dim=-1)    # (m, topk) int
    support = torch.cat([tk_f, tk_p], dim=-1)  # (m, S) int — MAY contain duplicates

    # Dedup within each row: mark first-occurrence positions per row (True),
    # duplicates (False). Zeroing duplicate probability mass then re-
    # normalising gives Sinkhorn a sparse support with unique tokens only —
    # matches the per-pair impl's `torch.unique(cat(...))` semantics without
    # requiring variable-length supports.
    sorted_support, sort_idx = support.sort(dim=-1)          # (m, S)
    is_first_sorted = torch.ones_like(sorted_support, dtype=torch.bool)
    is_first_sorted[:, 1:] = sorted_support[:, 1:] != sorted_support[:, :-1]
    # Unsort back to the original support order.
    _, inv_idx = sort_idx.sort(dim=-1)
    is_first = is_first_sorted.gather(1, inv_idx)             # (m, S) bool

    # Gather probabilities on each row's support; zero out duplicate positions
    # so they have no mass and Sinkhorn ignores them; then renormalise.
    a = p_full_row.gather(1, support).float()      # (m, S)
    b = p_pre_row.gather(1, support).float()       # (m, S)
    a = a * is_first
    b = b * is_first
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    # Ground cost: 1 - cosine similarity between input embeddings.
    E = embedding_matrix[support].float()          # (m, S, D)
    E_norm = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # (m, S, S) — element (b, i, j) = 1 - cos(E[b,i], E[b,j])
    cost = (1.0 - torch.bmm(E_norm, E_norm.transpose(1, 2))).clamp_min(0.0)

    # Log-domain Sinkhorn. K = exp(-C/eps). Updates:
    #   log_v[b,j] = log_b[b,j] - logsumexp_i(log_K[b,i,j] + log_u[b,i])
    #   log_u[b,i] = log_a[b,i] - logsumexp_j(log_K[b,i,j] + log_v[b,j])
    log_K = -cost / eps                             # (m, S, S)
    log_a = a.clamp_min(1e-30).log()                # (m, S)
    log_b = b.clamp_min(1e-30).log()                # (m, S)
    log_u = torch.zeros(m, S, device=device, dtype=torch.float32)
    log_v = torch.zeros(m, S, device=device, dtype=torch.float32)
    for _ in range(sinkhorn_iters):
        # v update: sum over source axis (dim=1). Broadcast log_u (m, S) as (m, S, 1).
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=1)
        # u update: sum over target axis (dim=2). Broadcast log_v (m, S) as (m, 1, S).
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)

    # Transport plan T = exp(log_u + log_K + log_v). Reconstruct and inner-product with cost.
    log_T = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(1)  # (m, S, S)
    T = log_T.exp()
    return (T * cost).sum(dim=(1, 2))                          # (m,)


def make_ot(embedding_matrix: torch.Tensor, topk: int = 128, eps: float = 0.05,
            sinkhorn_iters: int = 100, batched: bool = True):
    """Return a criterion callable `(p_full_row, p_pre_row) -> (m,)`.

    `batched=True` (default) selects the ~10× faster batched-Sinkhorn path;
    `batched=False` selects the per-pair reference impl (used for
    correctness verification). Numerical outputs should agree within ~1e-3
    L∞ on the same (p_full, p_pre) input.
    """
    fn = ot_divergence_row_batched if batched else ot_divergence_row
    return partial(
        fn,
        embedding_matrix=embedding_matrix,
        topk=topk,
        eps=eps,
        sinkhorn_iters=sinkhorn_iters,
    )


CRITERIA = {
    "js": js_divergence,
    "kl": kl_forward,
    # "ot" is registered at annotate-time via make_ot(); it depends on the
    # backbone's input-embedding matrix.
}
