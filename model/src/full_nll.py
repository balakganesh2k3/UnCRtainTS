"""
Full-covariance (Cholesky) Gaussian NLL for UnCRtainTS, C = 13 bands.

The network predicts K = C*(C+1)/2 = 91 RAW values per pixel. They fill the lower
triangle of L (row-major, torch.tril_indices order). Diagonal entries go through
softplus + eps so they are strictly positive; off-diagonal entries stay raw.
Sigma = L L^T is then symmetric positive definite by construction.

NLL per pixel = 0.5 * ||L^-1 (y - mu)||^2 + sum_i log L_ii  (+ 0.5*C*log(2*pi))
Optional beta-NLL (Seitzer et al. 2022): each pixel's NLL is weighted by
(geometric-mean variance, detached)^beta. beta = 0 is the plain NLL.
"""
import math
import torch
import torch.nn.functional as F


def n_tril(C):
    return C * (C + 1) // 2


def diag_positions(C):
    """Indices (within the K tril entries) that belong to the diagonal of L."""
    r, c = torch.tril_indices(C, C)
    return (r == c).nonzero().flatten()


def raw_to_cholesky(raw, C=13, eps=1e-4):
    """raw: [B, K, H, W] -> L: [B, H, W, C, C] (lower triangular, positive diagonal)."""
    B, K, H, W = raw.shape
    assert K == n_tril(C), f"expected {n_tril(C)} channels, got {K}"
    r, c = torch.tril_indices(C, C, device=raw.device)
    vals = raw.permute(0, 2, 3, 1)                      # [B, H, W, K]
    L = raw.new_zeros(B, H, W, C, C)
    is_diag = (r == c)
    v = torch.where(is_diag, F.softplus(vals) + eps, vals)  # softplus on diagonal only
    L[..., r, c] = v
    return L


def cholesky_to_covar(L):
    """L: [B, H, W, C, C] -> Sigma in the repo's layout [B, 1, C, C, H, W]."""
    S = L @ L.transpose(-1, -2)                         # [B, H, W, C, C]
    return S.permute(0, 3, 4, 1, 2).unsqueeze(1)


def full_nll(mean, raw, target, mask=None, C=13, eps=1e-4, const=True, reduction='mean', beta=0.0):
    """
    mean, target: [B, C, H, W]; raw: [B, K, H, W]; mask (optional): [B, H, W] or [B, 1, H, W],
    1 = pixel contributes. Returns NLL per pixel (scalar for 'mean', [B,H,W] for 'none').
    Computed in float32 for stability.
    """
    mean, raw, target = mean.float(), raw.float(), target.float()
    L = raw_to_cholesky(raw, C, eps)                    # [B, H, W, C, C]
    res = (target - mean).permute(0, 2, 3, 1).unsqueeze(-1)   # [B, H, W, C, 1]
    z = torch.linalg.solve_triangular(L, res, upper=False)    # L^-1 r
    maha = (z ** 2).sum(dim=(-1, -2))                   # [B, H, W]
    logdet_half = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # 0.5*log|Sigma|
    nll = 0.5 * maha + logdet_half
    if const: nll = nll + 0.5 * C * math.log(2 * math.pi)
    if beta > 0:
        w = torch.exp(2.0 * logdet_half / C).detach().pow(beta)   # (geo-mean variance)^beta, no gradient
        nll = nll * w
    if mask is not None:
        m = mask.float()
        if m.dim() == 4: m = m[:, 0]
        nll = nll * m
        if reduction == 'mean': return nll.sum() / m.sum().clamp_min(1.0)
    if reduction == 'mean': return nll.mean()
    return nll
