from raw_renderer_gpu import shade_ct_sh
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

import wandb

from copy import deepcopy
from itertools import product

DEFAULT_CFG = dict(
    N_imgs=10,
    H=3,
    W=6,
    N_unique_normals=9,
    use_sh0=True,
    z_min=0.0,
    z_max=1.0,
    metallic=0.0,
    albedo_min=[1, 0, 0],
    albedo_max=[0, 0, 1],
    loss="L2",
    lr=5e-3,
    name="default",
    n_iter=100000,
    log_every=10000,
    seed=1,
)


OVERRIDES = [
    {"N_imgs": 2, "name": "2 images"},
    {"H": 6, "W": 12, "name": "6x12 image"},
    {"N_unique_normals": 8, "name": "8 unique normals"},
    {"use_sh0": False, "name": "exclude SH0"},
    {"z_min": 0.0, "z_max": 0.1, "name": "normals with z [0.0, 0.1]"},
    {"z_min": 0.9, "z_max": 1.0, "name": "normals with z [0.9, 1.0]"},
    {"albedo_min": [0.5, 0, 0], "albedo_max": [
        0, 0, 0.5], "name": "reduced albedo range"},
    {"loss": "L1", "name": "L1 loss"},
]

device = "cuda"


def _rand_sh_coeffs(rng, ambient=True):
    """Random order-2 SH coefficients (9, 3) for plausible diffuse lighting."""
    coeffs = np.zeros((9, 3), dtype=np.float32)
    color = _rand_color(rng)
    intensity = float(rng.uniform(0.5, 2.0))
    if ambient:
        coeffs[0] = color * intensity
    scale1 = 0.5 * intensity
    scale2 = 0.3 * intensity
    coeffs[1:4] = rng.uniform(-scale1, scale1, (3, 3)).astype(np.float32)
    coeffs[4:9] = rng.uniform(-scale2, scale2, (5, 3)).astype(np.float32)
    return torch.from_numpy(coeffs)


def _rand_surface_normal(rng) -> np.ndarray:
    d = rng.standard_normal(3).astype(np.float32)
    d[2] = abs(d[2])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_color(rng) -> np.ndarray:
    c = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    if rng.random() > 0.5:
        c[0] = min(1.0, c[0] * 1.15)
    else:
        c[2] = min(1.0, c[2] * 1.15)
    return c


def _make_rand_normal_img(n_unique: int) -> np.ndarray:
    unique = np.stack([_rand_surface_normal(rng) for _ in range(n_unique)])
    assign = rng.integers(0, n_unique, size=H*W)
    for j in range(n_unique):
        if j not in assign:
            assign[rng.integers(0, H*W)] = j
    normals = unique[assign]
    # encoded = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    # return encoded.reshape(3, 6, 3)
    return normals


def _make_structured_normal_img(n_unique: int, z_min=0.2, z_max=1.0, H=3, W=6):
    """
    Evenly sample n_unique normals on the upper hemisphere cap
    with z in [z_min, z_max], then repeat until N_px.
    """
    z_min = float(np.clip(z_min, 0.0, 1.0))
    z_max = float(np.clip(z_max, 0.0, 1.0))
    i = np.arange(n_unique, dtype=np.float32)
    z = np.linspace(z_max, z_min, n_unique, dtype=np.float32)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = i * golden_angle
    r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    unique = np.stack([x, y, z], axis=-1).astype(np.float32)
    unique /= np.linalg.norm(unique, axis=1, keepdims=True).clip(min=1e-8)
    reps = int(np.ceil(H*W / n_unique))
    normals_px = np.tile(unique, (reps, 1))[:H*W]
    return normals_px


def _albedo_rmse(est, gt):
    est = est.detach()
    gt = gt.detach()
    num = (gt * est).sum(0)
    den = (est * est).sum(0).clamp_min(1e-8)
    scale = num / den
    est *= scale
    return torch.sqrt(((gt - est)**2).mean())


def _to_img(tensor, H, W):
    return tensor.detach().reshape(H, W, 3).clamp(0, 1).cpu().numpy()


def _albedo_err_map(gt, pred, H, W):
    gt_img = _to_img(gt, H, W)
    pred_img = _to_img(pred, H, W)
    err_img = np.abs(gt_img - pred_img).mean(axis=-1)
    max_err = err_img.max()
    if max_err > 1e-8:
        err_img = err_img / max_err
    return (255 * err_img).clip(0, 255).astype(np.uint8)


def _grad_to_img(grad, H, W):
    g = grad.detach().reshape(H, W, 3).abs().cpu().numpy()
    return g / (g.max() + 1e-8)


def _mse(t1, t2):
    return np.mean((t1.cpu().numpy() - t2.cpu().numpy())**2)


def run_experiment(cfg):
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    albedo_min = torch.tensor(cfg["albedo_min"], dtype=torch.float32)
    albedo_max = torch.tensor(cfg["albedo_max"], dtype=torch.float32)

    # Create GT

    # normals_np = _make_rand_normal_img(N_unique_normals)
    normals_np = _make_structured_normal_img(
        cfg["N_unique_normals"], z_min=cfg["z_min"], z_max=cfg["z_max"], H=cfg["H"], W=cfg["W"])
    normals_np /= np.linalg.norm(normals_np, axis=1,
                                 keepdims=True).clip(min=1e-8)
    N_t = torch.tensor(normals_np, device=device)

    # albedo_gt = torch.rand(H*W, 3, dtype=torch.float32, device=device)
    t = torch.linspace(0, 1, cfg["H"] * cfg["W"]).unsqueeze(-1)
    albedo_gt = ((1 - t) * albedo_min +
                 t * albedo_max).to(device)

    coeffs_gt = torch.zeros(cfg["N_imgs"], 9, 3, device=device)
    images = torch.zeros(cfg["N_imgs"], cfg["H"]*cfg["W"], 3, device=device)

    for i in range(cfg["N_imgs"]):
        coeffs_gt[i] = _rand_sh_coeffs(rng, ambient=cfg["use_sh0"])
        images[i] = shade_ct_sh(
            N_t, albedo_gt, coeffs_gt[i], metallic=cfg["metallic"])

    # OPTIMIZATION

    albedo_init = images.mean(0).clamp(0.05, 0.95)
    albedo = albedo_init.clone().to(device).requires_grad_(True)

    def _forward():
        loss_data = albedo.new_zeros(())
        for k in range(cfg["N_imgs"]):
            if cfg["use_sh0"]:
                coeffs_k = sh_coeffs[k]
            else:
                coeffs_k = torch.zeros(9, 3, device=device)
                coeffs_k[1:] = sh_coeffs_rest[k]
            recon = shade_ct_sh(N_t, albedo, coeffs_k,
                                metallic=cfg["metallic"])
            if cfg["loss"] == "L1":
                diff = torch.abs(recon - images[k])
            else:
                diff = (recon - images[k]) ** 2
            loss_data = loss_data + diff.mean()

        # loss_sparse = lambda_sparse * _tv(log_albedo.permute(2, 0, 1))
        # loss_white = lambda_white * (torch.exp(log_albedo).mean() - 0.5) ** 2
        # return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white
        return loss_data

    sh_init = torch.zeros(cfg["N_imgs"], 9, 3, device=device)
    if cfg["use_sh0"]:
        sh_init[:, 0, :] = 1.5
        sh_coeffs = sh_init.clone().to(device).requires_grad_(True)
        opt = torch.optim.Adam([albedo, sh_coeffs], lr=cfg["lr"])
    else:
        sh_coeffs_rest = sh_init[:, 1:, :].clone(
        ).detach().requires_grad_(True)
        opt = torch.optim.Adam([albedo, sh_coeffs_rest], lr=cfg["lr"])

    run = wandb.init(
        entity="dlvc-image-intrinsics",
        project="simple_sh_decomp",
        config=cfg,
        name=(cfg['name']),
        reinit=True,
    )

    run.log({
        "gt_albedo": wandb.Image(_to_img(albedo_gt, cfg["H"], cfg["W"])),
        "gt_normals": wandb.Image(_to_img(N_t / 2 + 0.5, cfg["H"], cfg["W"])),
        "gt_sh": coeffs_gt,
    }, step=0)

    # ── Optimisation loop ────────────────────────────────────────────────────
    loss_history = []
    albedo_history = []
    for i in range(cfg["n_iter"]):
        opt.zero_grad()
        loss = _forward()
        loss.backward()

        if i % cfg["log_every"] == 0:
            with torch.no_grad():
                grad_logs = {}
                if albedo.grad is not None:
                    grad_logs["grad/albedo_norm"] = albedo.grad.norm().item()
                    grad_logs["grad/albedo_abs_mean"] = albedo.grad.abs().mean().item()
                if cfg["use_sh0"]:
                    if sh_coeffs.grad is not None:
                        grad_logs["grad/sh_norm"] = sh_coeffs.grad.norm().item()
                        grad_logs["grad/sh_abs_mean"] = sh_coeffs.grad.abs().mean().item()
                elif sh_coeffs_rest.grad is not None:
                    grad_logs["grad/sh_norm"] = sh_coeffs_rest.grad.norm().item()
                    grad_logs["grad/sh_abs_mean"] = sh_coeffs_rest.grad.abs().mean().item()

                albedo_error = _albedo_rmse(albedo, albedo_gt)
                loss_history.append(loss.item())
                albedo_history.append(albedo_error.item())

                print(
                    f"[{i:4d}] total={loss.item():.2e}   albedo_error: {albedo_error:.2e}")

                run.log({
                    "iter": i,
                    "loss": loss.item(),
                    "albedo_error": albedo_error.item(),
                    "sh_error": _mse(sh_coeffs.detach(), coeffs_gt) if cfg["use_sh0"] else _mse(sh_coeffs_rest.detach(), coeffs_gt[:, 1:, :]),
                    "pred_albedo": wandb.Image(_to_img(albedo, cfg["H"], cfg["W"])),
                    "albedo_err_map": wandb.Image(_albedo_err_map(albedo_gt, albedo, cfg["H"], cfg["W"])),
                    "albedo_grad_map": wandb.Image(_grad_to_img(albedo.grad, cfg["H"], cfg["W"])),
                    "pred_sh": (
                        sh_coeffs.detach().cpu().numpy()
                        if cfg["use_sh0"]
                        else sh_coeffs_rest.detach().cpu().numpy()
                    ),
                    **grad_logs,
                }, step=i)

        opt.step()

    run.finish()


for override in OVERRIDES:
    cfg = deepcopy(DEFAULT_CFG)
    cfg.update(override)
    run_experiment(cfg)
