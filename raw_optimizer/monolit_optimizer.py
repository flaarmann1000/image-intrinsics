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
    lighting="Phong",
    # Phong
    shininess_range=(16, 64),
    opt_ks=False,
    ka=0.00,
    kd=1.00,
    ks=0.00,
    # Cook Torrance
    metallic_range=(0, 1),
    roughness_range=(0, 1),
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
    # {"kd": 0.8, "ks": 0.3, "opt_ks": False,
    #     "name": "opt specularity w/o ks"},
    # {"kd": 0.8, "ks": 0.3, "opt_ks": True,
    #     "name": "opt specularity incl. ks"},
    # {"kd": 0.8, "ks": 0.3, "single_shininess": True,
    #     "name": "single-value specularity"},
    {"lighting": "CT", "name": "CT lighting"},
    {"lighting": "CT", "metallic_range": (0.25, 0.75),
     "roughness_range": (0.25, 0.75), "name": "CT lighting - reduced ranges"},

]


def run_experiment(cfg):

    device = cfg["device"]

#      ____ _____
#    / ___|_   _|
#   | |  _  | |
#   | |_| | | |
#    \____| |_|

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

    V_gt = _make_structured_normal_img(
        n_unique=W*H, z_min=0.7, z_max=1, H=H, W=W)
    V_gt /= np.linalg.norm(V_gt, axis=1,
                           keepdims=True).clip(min=1e-8)

    V_gt = torch.tensor(V_gt, device=device)

    # phong
    min_s, max_s = cfg["shininess_range"]
    if cfg["single_shininess"]:
        shininess_gt = torch.full(
            (albedo_gt.shape[0], 1), 48.0, dtype=albedo_gt.dtype, device=device)
    else:
        shininess_gt = torch.rand(H * W, 1,
                                  dtype=torch.float32, device=device)*(max_s - min_s) + min_s
    ks_gt = torch.rand(H * W, 1, dtype=torch.float32,
                       device=device) * cfg["ks"]

    # cook torrance
    m_min, m_max = cfg["metallic_range"]
    metallic_gt = torch.rand(H * W, 1, dtype=torch.float32,
                             device=device) * (m_max - m_min) + m_min

    r_min, r_max = cfg["roughness_range"]
    roughness_gt = torch.rand(H * W, 1, dtype=torch.float32,
                              device=device) * (r_max - r_min) + r_min

    if cfg["uint8_comp"]:
        N_t = _uint8_compression(N_t*0.5 + 0.5) * 2.0 - 1.0
        albedo_gt = _uint8_compression(albedo_gt)

    images = torch.zeros(cfg["N_imgs"], H*W, 3, device=device)
    for i in range(cfg["N_imgs"]):
        coeffs_gt[i] = _rand_sh_coeffs(rng, ambient=cfg["use_sh0"])
        if cfg["lighting"] == "Phong":
            images[i] = shade_phong_sh(V_gt, N_t, cfg["ka"], cfg["kd"], ks_gt, shininess_gt,
                                       albedo_gt, coeffs_gt[i])
        else:
            images[i] = shade_ct_sh(
                V_gt, N_t, albedo_gt, coeffs_gt[i], metallic_gt, roughness_gt)

    if cfg["uint8_comp"]:
        images = _uint8_compression(images)

#      ___        _   _           _          _   _
#    / _ \ _ __ | |_(_)_ __ ___ (_)______ _| |_(_) ___  _ __
#   | | | | '_ \| __| | '_ ` _ \| |_  / _` | __| |/ _ \| '_ \
#   | |_| | |_) | |_| | | | | | | |/ / (_| | |_| | (_) | | | |
#    \___/| .__/ \__|_|_| |_| |_|_/___\__,_|\__|_|\___/|_| |_|
#         |_|

    def _forward():
        loss_data = albedo.new_zeros(())
        sh_est = get_sh_coeffs()
        for k in range(cfg["N_imgs"]):
            coeffs_k = sh_est[k]
            # recon = shade_ct_sh(N_t, albedo, coeffs_k,
            #                     metallic=cfg["metallic"])
            if cfg["lighting"] == "Phong":
                shininess = get_shininess()
                recon = shade_phong_sh(V_gt, N_t, cfg["ka"], cfg["kd"], ks_est, shininess,
                                       albedo, coeffs_k)
            else:
                recon = shade_ct_sh(
                    V_gt, N_t, albedo, coeffs_k, metallic_est, roughness_est)
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

    # --- albedo ---
    albedo_init = images.mean(0).clamp(0.05, 0.95)
    albedo = albedo_init.clone().to(device).requires_grad_(True)
    params = {"albedo": albedo}

    # --- SH ---
    sh_init = torch.zeros(cfg["N_imgs"], 9, 3, device=device)
    if cfg["exclude_sh0_0"]:
        sh_coeffs_0_0 = torch.tensor(
            [[1, 1, 1]], dtype=torch.float32, device=device)
        sh_coeffs_0 = sh_init[0, 1:, :].clone(
        ).detach().requires_grad_(True)
        sh_coeffs_rest = sh_init[1:, 0:, :].clone(
        ).detach().requires_grad_(True)
        params["sh_coeffs_0"] = sh_coeffs_0
        params["sh_coeffs_rest"] = sh_coeffs_rest
    elif not cfg["use_sh0"]:
        sh_coeffs_rest = sh_init[:, 1:, :].clone(
        ).detach().requires_grad_(True)
        params["sh_coeffs_rest"] = sh_coeffs_rest
    else:
        sh_init[:, 0, :] = 1.5
        sh_coeffs = sh_init.clone().to(device).requires_grad_(True)
        params["sh_coeffs"] = sh_coeffs

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

    # --- Specularity ---
    ks_est = ks_gt
    if cfg["lighting"] == "Phong" and cfg["ks"] > 0:
        if cfg["single_shininess"]:
            s_min, s_max = cfg["shininess_range"]
            shininess_val = torch.tensor(
                (s_max + s_min) / 2, device=device).requires_grad_(True)
            params["shininess_val"] = shininess_val
        else:
            shininess_raw = torch.zeros(
                H * W, 1, device=device, requires_grad=True)
            params["shininess_raw"] = shininess_raw
        if cfg["opt_ks"]:
            ks = torch.rand(H * W, 1, dtype=torch.float32,
                            device=device).requires_grad_(True)
            params["ks"] = ks
            ks_est = ks

    def get_shininess():
        shininess_est = shininess_gt
        if cfg["ks"] > 0:
            if cfg["single_shininess"]:
                shininess_est = shininess_val.expand_as(shininess_gt)
            else:
                shininess_est = shininess_raw.sigmoid() * (max_s - min_s) + min_s

        return shininess_est
    # --- Cook Torrance ---

    metallic_est = metallic_gt
    roughness_est = roughness_gt
    if cfg["lighting"] == "CT":
        metallic_est = torch.full((albedo_gt.shape[0], 1), (m_max - m_min) / 2, dtype=torch.float32,
                                  device=device).requires_grad_(True)
        params["metallic"] = metallic_est
        roughness_est = torch.full((albedo_gt.shape[0], 1), (m_max - m_min) / 2, dtype=torch.float32,
                                   device=device).requires_grad_(True)
        params["roughness"] = roughness_est

    opt = optimizer(list(params.values()), **opt_params)

    run = wandb.init(
        entity="DLVC-intrinsics",
        project="simple_sh_decomp",
        config=cfg,
        name=(cfg['name']),
        reinit=True,
    )

    lighting_gt = {}
    if cfg["lighting"] == "Phong":
        lighting_gt = {
            "gt_shininess": wandb.Image(_shininess_to_img(shininess_gt, H, W, cfg["shininess_range"])),
            "gt_ks": wandb.Image(_to_img(ks_gt, H, W)), }
    else:
        lighting_gt = {
            "gt_metallic": wandb.Image(_to_img(metallic_gt, H, W)),
            "gt_roughness": wandb.Image(_to_img(roughness_gt, H, W)), }

    run.log({
        "gt_albedo": wandb.Image(_to_img(albedo_gt, H, W)),
        "gt_normals": wandb.Image(_to_img(N_t / 2 + 0.5, H, W)),
        "gt_V": wandb.Image(_to_img(V_gt / 2 + 0.5, H, W)),
        "gt_sh": coeffs_gt,
        **lighting_gt
    }, step=0)

    # ── Optimisation loop ────────────────────────────────────────────────────
    loss_history = []
    albedo_history = []

    for i in range(cfg["n_iter"]):

        def closure():
            opt.zero_grad()
            # shininess_est = get_shininess()
            # sh_est = get_sh_coeffs()
            # loss, *_ = _forward(sh_est, shininess_est)
            loss, *_ = _forward()
            loss.backward()

            return loss

        loss = opt.step(closure)

        if i % cfg["log_every"] == 0:
            with torch.no_grad():

                sh_est = get_sh_coeffs()
                shininess_cur = get_shininess()
                loss, loss_data, loss_white, loss_sh, loss_sh0, loss_sh0_0 = _forward()
                # loss, loss_data, loss_white, loss_sh, loss_sh0, loss_sh0_0 = _forward(
                #     sh_est, shininess_cur)

                albedo_error, scale = _albedo_rmse(albedo, albedo_gt)
                loss_history.append(loss.item())
                albedo_history.append(albedo_error.item())

                grad_logs = {}
                for name, p in params.items():
                    if p.grad is not None:
                        grad_logs[f"grad/{name}_norm"] = p.grad.norm().item()
                        grad_logs[f"grad/{name}_abs_mean"] = p.grad.abs().mean().item()

                lighting_logs = {}
                if cfg["lighting"] == "Phong":
                    lighting_logs = {
                        "pred_shininess": wandb.Image(_shininess_to_img(shininess_cur, H, W, cfg["shininess_range"])),
                        "shininess_error": _mse(shininess_cur, shininess_gt),
                        "shininess/min": shininess_cur.min().item(),
                        "shininess/max": shininess_cur.max().item(),
                        "shininess/mean": shininess_cur.mean().item(),
                        "pred_ks": wandb.Image(_to_img(ks_est, H, W)),
                        "ks/min": ks_est.min().item(),
                        "ks/max": ks_est.max().item(),
                        "ks/mean": ks_est.mean().item(),
                        "ks/error": _mse(ks_est, ks_gt),
                    }
                else:
                    lighting_logs = {
                        "pred_metallic": wandb.Image(_to_img(metallic_est, H, W)),
                        "metallic/min": metallic_est.min().item(),
                        "metallic/max": metallic_est.max().item(),
                        "metallic/mean": metallic_est.mean().item(),
                        "metallic/error": _mse(metallic_est, metallic_gt),
                        "pred_roughness": wandb.Image(_to_img(roughness_est, H, W)),
                        "roughness/min": roughness_est.min().item(),
                        "roughness/max": roughness_est.max().item(),
                        "roughness/mean": roughness_est.mean().item(),
                        "roughness/error": _mse(roughness_est, roughness_gt),
                    }

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
                        "pred_albedo": wandb.Image(_to_img(albedo, H, W)),
                        "albedo_grad_map": wandb.Image(_grad_to_img(albedo.grad, H, W)) if albedo.grad is not None else None,
                        "albedo_err_map": wandb.Image(_albedo_err_map(albedo_gt, albedo, H, W)),
                        "sh_error": _mse(sh_est.detach(), coeffs_gt*scale.view(1, 1, 3)),
                        "pred_sh": sh_est.detach().cpu().numpy(),
                        **lighting_logs,
                        **grad_logs,
                    }, step=i
                )
    run.finish()


for override in OVERRIDES:
    cfg = deepcopy(DEFAULT_CFG)
    cfg.update(override)
    run_experiment(cfg)
