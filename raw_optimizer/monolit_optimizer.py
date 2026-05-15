from raw_renderer_gpu import shade_phong_sh, shade_ct_sh
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

import wandb

from copy import deepcopy
from itertools import product

from helper import _albedo_err_map, _albedo_rmse, _grad_to_img, _make_structured_normal_img, _mse, _rand_color, _rand_sh_coeffs, _shininess_to_img, _to_img, _uint8_compression

DEFAULT_CFG = dict(
    name="default",
    device="cpu",
    optimizer="LBFGS",
    lr=1.0,
    n_iter=3000,
    log_every=500,
    seed=1,
    N_imgs=10,
    H=3,
    W=6,
    loss="L2",
    N_unique_normals=9,
    uint8_comp=False,
    z_min=0.0,
    z_max=1.0,
    albedo_min=[1, 0, 0],
    albedo_max=[0, 0, 1],
    lambda_white=0,
    use_sh0=True,
    lambda_SH0_0=0,
    lambda_SH0_energy=0,
    lambda_SH_energy=0,
    exclude_sh0_0=False,
    single_shininess=False,
    # shader
    shader="Phong",
    # Phong
    shininess_range=(16, 64),
    opt_ks=False,
    ka=0.00,
    kd=1.00,
    ks=0.00,
    # Cook Torrance
    metallic=0.0,
)


OVERRIDES = [
    # {"name": "default (LBFGS-sw)"},
    # {"optimizer": "ADAM", "lr": 5e-3, "n_iter": 100000,
    #     "log_every": 10000, "name": "ADAM"},
    # {"optimizer": "ADAM", "lr": 5e-3, "n_iter": 1000,
    #     "log_every": 500, "kd": 0.8, "ks": 0.3,
    #     "name": "opt specularity (ADAM)"},
    # {"N_imgs": 2, "name": "2 images"},
    # {"N_imgs": 20, "name": "20 images"},
    # {"use_sh0": False, "name": "exclude SH0"},
    # {"lambda_white": 0.1, "name": "regularize albedo"},
    # {"lambda_SH_energy": 0.1, "name": "regularize SH"},
    # {"loss": "L1", "name": "L1 loss"},
    # {"H": 6, "W": 12, "name": "6x12 image"},
    # {"N_unique_normals": 8, "name": "8 unique normals"},
    # {"exclude_sh0_0": True, "name": "fix img0 SH0 to 1"},
    # {"N_unique_normals": 4, "name": "4 unique normals"},
    # {"N_unique_normals": 18, "name": "18 unique normals"},
    # {"z_min": 0.0, "z_max": 0.1, "name": "normals with z [0.0, 0.1]"},
    # {"z_min": 0.9, "z_max": 1.0, "name": "normals with z [0.9, 1.0]"},
    # {"albedo_min": [0.5, 0, 0], "albedo_max": [
    #     0, 0, 0.5], "name": "reduced albedo range"},
    # {"uint8_comp": True, "name": "simulate uint8 compression"},
    # {"lambda_SH0_energy": 0.1, "name": "regularize SH0"},
    # {"lambda_SH0_0": 0.1, "name": "regularize 1st SH0 to 1"},
    {"kd": 0.8, "ks": 0.3, "opt_ks": False,
        "name": "opt specularity w/o ks"},
    {"kd": 0.8, "ks": 0.3, "opt_ks": True,
        "name": "opt specularity incl. ks"},
    # {"kd": 0.8, "ks": 0.3, "single_shininess": True,
    #     "name": "single-value specularity"},
    # {"shader": "CT", "name": "CT shader"},

]


def run_experiment(cfg):

    device = cfg["device"]

    # Create GT

    H = cfg["H"]
    W = cfg["W"]

    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    albedo_min = torch.tensor(cfg["albedo_min"], dtype=torch.float32)
    albedo_max = torch.tensor(cfg["albedo_max"], dtype=torch.float32)

    # normals_np = _make_rand_normal_img(N_unique_normals)
    normals_np = _make_structured_normal_img(
        cfg["N_unique_normals"], z_min=cfg["z_min"], z_max=cfg["z_max"], H=H, W=W)
    normals_np /= np.linalg.norm(normals_np, axis=1,
                                 keepdims=True).clip(min=1e-8)
    N_t = torch.tensor(normals_np, device=device)

    # albedo_gt = torch.rand(H*W, 3, dtype=torch.float32, device=device)
    t = torch.linspace(0, 1, H * W).unsqueeze(-1)
    albedo_gt = ((1 - t) * albedo_min +
                 t * albedo_max).to(device)

    coeffs_gt = torch.zeros(cfg["N_imgs"], 9, 3, device=device)
    images = torch.zeros(cfg["N_imgs"], H*W, 3, device=device)

    min_s, max_s = cfg["shininess_range"]
    if cfg["single_shininess"]:
        shininess_gt = torch.full(
            (albedo_gt.shape[0], 1), 48.0, dtype=albedo_gt.dtype, device=device)
    else:
        shininess_gt = torch.rand(H * W, 1,
                                  dtype=torch.float32, device=device)*(max_s - min_s) + min_s
    V_gt = _make_structured_normal_img(
        n_unique=W*H, z_min=0.7, z_max=1, H=H, W=W)
    V_gt /= np.linalg.norm(V_gt, axis=1,
                           keepdims=True).clip(min=1e-8)

    V_gt = torch.tensor(V_gt, device=device)

    ks_gt = torch.rand(H * W, 1, dtype=torch.float32,
                       device=device) * cfg["ks"]

    if cfg["uint8_comp"]:
        N_t = _uint8_compression(N_t*0.5 + 0.5) * 2.0 - 1.0
        albedo_gt = _uint8_compression(albedo_gt)

    for i in range(cfg["N_imgs"]):
        coeffs_gt[i] = _rand_sh_coeffs(rng, ambient=cfg["use_sh0"])
        images[i] = shade_phong_sh(V_gt, N_t, cfg["ka"], cfg["kd"], ks_gt, shininess_gt,
                                   albedo_gt, coeffs_gt[i])

    if cfg["uint8_comp"]:
        images = _uint8_compression(images)

    # OPTIMIZATION

    def _forward(sh_est, shininess):
        loss_data = albedo.new_zeros(())
        for k in range(cfg["N_imgs"]):
            coeffs_k = sh_est[k]
            # recon = shade_ct_sh(N_t, albedo, coeffs_k,
            #                     metallic=cfg["metallic"])
            recon = shade_phong_sh(V_gt, N_t, cfg["ka"], cfg["kd"], ks_est, shininess,
                                   albedo, coeffs_k)
            if cfg["uint8_comp"]:
                recon = _uint8_compression(recon)
            if cfg["loss"] == "L1":
                diff = torch.abs(recon - images[k])
            else:
                diff = (recon - images[k]) ** 2
            loss_data = loss_data + diff.mean()

        loss_data *= 1e6
        loss_data = loss_data / cfg["N_imgs"]
        loss_white = 0
        loss_sh_energy = 0
        loss_sh0_energy = 0
        loss_sh0_0 = 0

        if cfg["lambda_white"] > 0:
            loss_white = cfg["lambda_white"] * (albedo.mean() - 0.5) ** 2
        if cfg["lambda_SH_energy"] > 0:
            loss_sh_energy = cfg["lambda_SH_energy"] * (sh_est ** 2).mean()
        if cfg["lambda_SH0_energy"] > 0:
            loss_sh0_energy = cfg["lambda_SH0_energy"] * \
                (sh_est[:, 0, :] ** 2).mean()
        if cfg["lambda_SH0_0"] > 0:
            loss_sh0_0 = cfg["lambda_SH0_0"] * \
                ((1 - sh_est[0, 0, :]) ** 2).mean()

        summed_loss = loss_data + loss_white + \
            loss_sh_energy + loss_sh0_energy + loss_sh0_0

        return summed_loss, loss_data, loss_white, loss_sh_energy, loss_sh0_energy, loss_sh0_0

    albedo_init = images.mean(0).clamp(0.05, 0.95)
    albedo = albedo_init.clone().to(device).requires_grad_(True)

    sh_init = torch.zeros(cfg["N_imgs"], 9, 3, device=device)

    shininess = shininess_gt.clone()

    ks_est = ks_gt

    optimizer = torch.optim.Adam
    opt_params = {}
    opt_params["lr"] = cfg["lr"]
    if cfg["optimizer"] == "LBFGS":
        optimizer = torch.optim.LBFGS
        # opt_params["line_search_fn"] = "strong_wolfe"
        opt_params = {
            "lr": 1.0,
            "max_iter": 50,
            "tolerance_grad": 0,
            "tolerance_change": 0,
            "line_search_fn": "strong_wolfe",
        }

    if cfg["exclude_sh0_0"]:
        sh_coeffs_0_0 = torch.tensor(
            [[1, 1, 1]], dtype=torch.float32, device=device)
        sh_coeffs_0 = sh_init[0, 1:, :].clone(
        ).detach().requires_grad_(True)
        sh_coeffs_rest = sh_init[1:, 0:, :].clone(
        ).detach().requires_grad_(True)
        params = [albedo, sh_coeffs_rest, sh_coeffs_0]
    elif not cfg["use_sh0"]:
        sh_coeffs_rest = sh_init[:, 1:, :].clone(
        ).detach().requires_grad_(True)
        params = [albedo, sh_coeffs_rest]
    else:
        sh_init[:, 0, :] = 1.5
        sh_coeffs = sh_init.clone().to(device).requires_grad_(True)

        if cfg["ks"] > 0:
            if cfg["single_shininess"]:
                s_min, s_max = cfg["shininess_range"]
                shininess_val = torch.tensor(
                    (s_max + s_min) / 2, device=device).requires_grad_(True)
                shininess = shininess_val.expand_as(shininess_gt)
                params = [albedo, sh_coeffs, shininess_val]
            else:
                # shininess = torch.full(
                #     (albedo.shape[0], 1), 32.0, dtype=albedo.dtype, device=device).requires_grad_(True)

                shininess_raw = torch.zeros(
                    H * W, 1, device=device, requires_grad=True)
                # albedo = albedo_gt.clone()
                # sh_coeffs = coeffs_gt.clone()

                # params = [albedo, sh_coeffs, shininess]
                params = [albedo, sh_coeffs, shininess_raw]
                if cfg["opt_ks"]:
                    ks = torch.rand(H * W, 1, dtype=torch.float32,
                                    device=device).requires_grad_(True)
                    params.append(ks)
                    ks_est = ks

        else:
            params = [albedo, sh_coeffs]

    opt = optimizer(params, **opt_params)

    run = wandb.init(
        # entity="dlvc-image-intrinsics",
        entity="DLVC-intrinsics",
        project="simple_sh_decomp",
        config=cfg,
        name=(cfg['name']),
        reinit=True,
    )

    run.log({
        "gt_albedo": wandb.Image(_to_img(albedo_gt, H, W)),
        "gt_normals": wandb.Image(_to_img(N_t / 2 + 0.5, H, W)),
        "gt_V": wandb.Image(_to_img(V_gt / 2 + 0.5, H, W)),
        "gt_shininess": wandb.Image(_shininess_to_img(shininess_gt, H, W, cfg["shininess_range"])),
        "gt_sh": coeffs_gt,
        "gt_ks": wandb.Image(_to_img(ks_gt, H, W)),
    }, step=0)

    # ── Optimisation loop ────────────────────────────────────────────────────
    loss_history = []
    albedo_history = []

    def get_sh_coeffs():
        sh_est = torch.zeros(cfg["N_imgs"], 9, 3, device=device)
        if cfg["exclude_sh0_0"]:
            sh0 = torch.cat([sh_coeffs_0_0, sh_coeffs_0], dim=0)
            sh_est = torch.cat([sh0[None, ...], sh_coeffs_rest], dim=0)
        elif cfg["use_sh0"]:
            sh_est = sh_coeffs
        else:
            sh_est[:, 1:, :] = sh_coeffs_rest
        return sh_est

    for i in range(cfg["n_iter"]):

        first_closure_call = [True]

        def closure():
            opt.zero_grad()
            shininess = shininess_raw.sigmoid() * (max_s - min_s) + min_s

            sh_est = get_sh_coeffs()
            loss, *_ = _forward(sh_est, shininess)
            loss.backward()
            if first_closure_call[0] and i % cfg["log_every"] == 0:
                print(
                    f"iter {i}: first-call grad norm = {shininess_raw.grad.norm().item():.2e}")
                first_closure_call[0] = False
            return loss

        loss = opt.step(closure)

        if i % cfg["log_every"] == 0:
            with torch.no_grad():
                shininess_cur = shininess_raw.sigmoid() * (max_s - min_s) + min_s
                grad_logs = {}
                sh_est = get_sh_coeffs()
                loss, loss_data, loss_white, loss_sh, loss_sh0, loss_sh0_0 = _forward(
                    sh_est, shininess_cur)

                if albedo.grad is not None:
                    grad_logs["grad/albedo_norm"] = albedo.grad.norm().item()
                    grad_logs["grad/albedo_abs_mean"] = albedo.grad.abs().mean().item()
                if cfg["use_sh0"] and not cfg["exclude_sh0_0"]:
                    if sh_coeffs.grad is not None:
                        grad_logs["grad/sh_norm"] = sh_coeffs.grad.norm().item()
                        grad_logs["grad/sh_abs_mean"] = sh_coeffs.grad.abs().mean().item()
                elif sh_coeffs_rest.grad is not None:
                    grad_logs["grad/sh_rest_norm"] = sh_coeffs_rest.grad.norm().item()
                    grad_logs["grad/sh_rest_abs_mean"] = sh_coeffs_rest.grad.abs().mean().item()

                albedo_error, scale = _albedo_rmse(albedo, albedo_gt)
                loss_history.append(loss.item())
                albedo_history.append(albedo_error.item())

                print(
                    f"[{i:4d}] total={loss.item():.2e}   albedo_error: {albedo_error:.2e}")

                run.log(
                    {
                        "iter": i,
                        "loss": loss.item(),
                        "loss_data": loss_data,
                        "loss_white": loss_white,
                        "loss_sh": loss_sh,
                        "loss_sh0": loss_sh0,
                        "loss_sh0_0": loss_sh0_0,
                        "albedo_error": albedo_error.item(),
                        "albedo_scale": scale,
                        "sh_error": _mse(sh_est.detach(), coeffs_gt*scale.view(1, 1, 3)),
                        "pred_albedo": wandb.Image(_to_img(albedo, H, W)),
                        "pred_ks": wandb.Image(_to_img(ks_est, H, W)),
                        "pred_shininess": wandb.Image(_shininess_to_img(shininess_cur, H, W, cfg["shininess_range"])),
                        "shininess_error": _mse(shininess_cur, shininess_gt),
                        "albedo_err_map": wandb.Image(_albedo_err_map(albedo_gt, albedo, H, W)),
                        "albedo_grad_map": wandb.Image(_grad_to_img(albedo.grad, H, W)) if albedo.grad is not None else None,
                        "pred_sh": sh_est.detach().cpu().numpy(),
                        **grad_logs,
                        "shininess/min": shininess_cur.min().item(),
                        "shininess/max": shininess_cur.max().item(),
                        "shininess/mean": shininess_cur.mean().item(),
                        "shininess/grad_norm": shininess_raw.grad.norm().item() if shininess_raw.grad is not None else 0,
                        "ks/min": ks_est.min().item(),
                        "ks/max": ks_est.max().item(),
                        "ks/mean": ks_est.mean().item(),
                        "ks/grad_norm": ks_est.grad.norm().item() if ks_est.grad is not None else 0,
                    }, step=i
                )
    run.finish()


for override in OVERRIDES:
    cfg = deepcopy(DEFAULT_CFG)
    cfg.update(override)
    run_experiment(cfg)
