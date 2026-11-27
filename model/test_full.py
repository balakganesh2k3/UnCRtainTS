"""Run from UnCRtainTS/model:  python test_full_nll.py   (needs only torch). All checks must print OK."""
import math, torch
from src.full_nll import raw_to_cholesky, cholesky_to_covar, full_nll, n_tril

torch.manual_seed(0)
C, B, H, W = 13, 2, 5, 6
K = n_tril(C); assert K == 91

mean   = torch.randn(B, C, H, W)
target = torch.randn(B, C, H, W)
raw    = torch.randn(B, K, H, W) * 0.3

# 1) L is lower triangular with positive diagonal; Sigma symmetric positive definite
L = raw_to_cholesky(raw, C)
assert torch.allclose(L, torch.tril(L)), "L not lower triangular"
assert (torch.diagonal(L, dim1=-2, dim2=-1) > 0).all(), "diagonal not positive"
S = cholesky_to_covar(L)
assert S.shape == (B, 1, C, C, H, W), S.shape
Sp = S[:, 0].permute(0, 3, 4, 1, 2)                     # [B,H,W,C,C]
assert torch.allclose(Sp, Sp.transpose(-1, -2), atol=1e-5)
assert torch.linalg.eigvalsh(Sp.double()).min() > 0
print("OK 1: Cholesky structure and covariance layout [B,1,C,C,H,W]")

# 2) matches torch.distributions.MultivariateNormal
mine = full_nll(mean, raw, target, reduction='none')
ref  = torch.zeros(B, H, W)
for b in range(B):
    for i in range(H):
        for j in range(W):
            d = torch.distributions.MultivariateNormal(mean[b, :, i, j].double(), covariance_matrix=Sp[b, i, j].double())
            ref[b, i, j] = -d.log_prob(target[b, :, i, j].double()).float()
assert torch.allclose(mine, ref, atol=1e-3, rtol=1e-4), (mine - ref).abs().max()
print("OK 2: NLL equals MultivariateNormal log-prob")

# 3) with off-diagonals ~0 it equals the diagonal Gaussian NLL
raw0 = torch.zeros(B, K, H, W)
r_, c_ = torch.tril_indices(C, C)
diag_ch = (r_ == c_).nonzero().flatten()
raw0[:, diag_ch] = torch.randn(B, C, H, W)
var = torch.nn.functional.softplus(raw0[:, diag_ch]) ** 2 + 0  # sigma_i^2 = (softplus+eps)^2
var = (torch.nn.functional.softplus(raw0[:, diag_ch]) + 1e-4) ** 2
diag_nll = (0.5 * ((target - mean) ** 2 / var + torch.log(var)) + 0.5 * math.log(2 * math.pi)).sum(1)
assert torch.allclose(full_nll(mean, raw0, target, reduction='none'), diag_nll, atol=1e-3, rtol=1e-4)
print("OK 3: reduces to the diagonal NLL when correlations are zero")

# 4) gradients are finite, including at an extreme raw value (no clamp/nan_to_num used)
raw_g = (raw * 3).clone().requires_grad_(True); mean_g = mean.clone().requires_grad_(True)
full_nll(mean_g, raw_g, target).backward()
assert torch.isfinite(raw_g.grad).all() and torch.isfinite(mean_g.grad).all()
print("OK 4: finite gradients")

# 5) calibration: samples drawn from the predicted Gaussian give mean Mahalanobis^2 ~ C
n = 20000
Lr = raw_to_cholesky(torch.randn(1, K, 1, 1) * 0.3, C)[0, 0, 0]
y  = (Lr @ torch.randn(C, n)).T                         # y ~ N(0, L L^T)
z  = torch.linalg.solve_triangular(Lr, y.T, upper=False)
m2 = (z ** 2).sum(0).mean().item()
assert abs(m2 - C) < 0.3, m2
print(f"OK 5: mean squared Mahalanobis = {m2:.2f} (expected {C})")

# 6) mask: masked-out pixels do not change the loss
mask = torch.ones(B, H, W); mask[:, :2] = 0
a = full_nll(mean, raw, target, mask=mask)
t2 = target.clone(); t2[:, :, :2] += 100.0
b = full_nll(mean, raw, t2, mask=mask)
assert torch.allclose(a, b, atol=1e-5)
print("OK 6: mask respected")

# 7) beta-NLL: beta=0 is the plain NLL; beta>0 weights by the detached geometric-mean variance^beta
plain = full_nll(mean, raw, target, reduction='none')
assert torch.allclose(full_nll(mean, raw, target, reduction='none', beta=0.0), plain)
Lb = raw_to_cholesky(raw, C)
gm = torch.exp(2 * torch.log(torch.diagonal(Lb, dim1=-2, dim2=-1)).sum(-1) / C)
assert torch.allclose(full_nll(mean, raw, target, reduction='none', beta=0.5), plain * gm ** 0.5, rtol=1e-4, atol=1e-5)
print("OK 7: beta-NLL weighting")
