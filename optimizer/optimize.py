"""
Gradient-descent intrinsic decomposition.

Shared albedo A and per-image grayscale shading S_k optimised jointly in
log-space:  log I_k  =  log A  +  log S_k

Loss terms
----------
data          : reconstruction fidelity  ||log A + log S_k - log I_k||²
consistency   : ||log S_0 - log S_k - mean_c(log I_0 - log I_k)||²
                (albedo cancels in the difference → direct shading anchor)
shading_smooth: isotropic TV on each S_k  (allows shadows, removes noise)
albedo_sparse : isotropic TV on A         (piece-wise constant reflectance)
"""

import numpy as np
import torch


def _tv(x):
    """Isotropic total variation on a [1, C, H, W] tensor."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    return (dh ** 2 + 1e-8).sqrt().mean() + (dw ** 2 + 1e-8).sqrt().mean()


def decompose(images_np, n_iter=2000, lr=5e-3,
              lambda_shading_smooth=0.1,
              lambda_albedo_sparse=1.5,
              lambda_consistency=5.0):
    """
    Parameters
    ----------
    images_np : list of ndarray [H, W, 3] uint8
    n_iter    : optimisation steps
    lr        : Adam learning rate

    Returns
    -------
    albedo   : ndarray [H, W, 3] float in [0, 1]
    shadings : list of ndarray [H, W, 3] float  (grayscale broadcast to RGB)
    history  : list of scalar loss values (one per 200 iterations)
    """
    eps = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_log(arr):
        t = torch.from_numpy(arr.astype("float32") / 255.0).to(device)
        return torch.log(t.clamp(min=eps))  # [H, W, 3]

    log_Is = [to_log(img) for img in images_np]
    N = len(log_Is)

    log_A_init = sum(log_Is) / N
    log_Ss_init = [(lI - log_A_init).mean(dim=-1, keepdim=True) for lI in log_Is]

    log_A  = log_A_init.clone().requires_grad_(True)
    log_Ss = [s.clone().requires_grad_(True) for s in log_Ss_init]

    diff_targets = [
        (log_Is[0] - log_Is[k]).mean(dim=-1, keepdim=True).detach()
        for k in range(N)
    ]

    optimizer = torch.optim.Adam([log_A] + log_Ss, lr=lr)

    def chw(x):
        return x.permute(2, 0, 1).unsqueeze(0)

    history = []
    for i in range(n_iter):
        optimizer.zero_grad()

        loss_data = sum(
            ((log_A + log_Ss[k] - log_Is[k]) ** 2).mean() for k in range(N)
        )
        loss_consist = lambda_consistency * sum(
            ((log_Ss[0] - log_Ss[k] - diff_targets[k]) ** 2).mean()
            for k in range(1, N)
        )
        loss_smooth = lambda_shading_smooth * sum(
            _tv(chw(log_Ss[k])) for k in range(N)
        )
        loss_sparse = lambda_albedo_sparse * _tv(chw(log_A))

        loss = loss_data + loss_consist + loss_smooth + loss_sparse
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            history.append(loss.item())
            print(f"[{i:4d}] total={loss.item():.5f}  "
                  f"data={loss_data.item():.5f}  "
                  f"consist={loss_consist.item():.5f}  "
                  f"smooth={loss_smooth.item():.5f}  "
                  f"sparse={loss_sparse.item():.5f}")

    def to_np(t):
        return t.detach().cpu().numpy()

    albedo   = to_np(torch.exp(log_A))
    shadings = [to_np(torch.exp(log_Ss[k]).expand_as(log_Is[k])) for k in range(N)]

    return albedo, shadings, history
