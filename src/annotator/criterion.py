"""
Divergence functions for the commit criterion (METHOD §3).

METHOD.md's primary criterion is embedding-grounded OT (Sinkhorn). JS/KL
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
    """Vectorise ot_divergence_pair over the target-position axis.

    p_full_row: (m, V), p_pre_row: (m, V).
    Returns: (m,) OT distances.
    """
    m = p_full_row.shape[0]
    out = torch.zeros(m, device=p_full_row.device, dtype=torch.float32)
    for j in range(m):
        out[j] = ot_divergence_pair(
            p_full_row[j], p_pre_row[j],
            embedding_matrix=embedding_matrix,
            topk=topk, eps=eps, sinkhorn_iters=sinkhorn_iters,
        )
    return out


def make_ot(embedding_matrix: torch.Tensor, topk: int = 128, eps: float = 0.05,
            sinkhorn_iters: int = 100):
    """Return a criterion callable `(p_full_row, p_pre_row) -> (m,)` with
    the embedding matrix and hyperparameters pre-bound. Register in CRITERIA
    at annotate-time when the model is loaded."""
    return partial(
        ot_divergence_row,
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
