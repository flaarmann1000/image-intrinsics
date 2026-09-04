"""decompose_scene — fit intrinsics + lighting to one scene directory.

The end-to-end driver: load the scene, optionally downsample, split off validation
lights, run the optimizer for the requested shader, then compute metrics and write
artifacts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from PIL import Image

from idr.render import EnvMap, SHLighting, build_sh_basis, shade_ct_sh, shade_ct_env
from idr.render.brdf import _get_ggx_sh_lut
from idr.data.scene_io import load_scene, linear_to_srgb
from idr.data.geometry import make_proxy_geometry, _subsample_mask
from idr.data.build import render_scene, render_3dfront_dataset
from idr.optim.models.ct_sh import _optimize_ct_sh   # curriculum warm-start only
from idr.optim.registry import optimize
from idr.optim.result import EnvGrid
from idr.track.wandb_log import _sh_coeffs_to_env_img, _env_flat_to_img
from idr.data.synthetic_scene import _make_lights_random_sh
from idr.config import DEFAULT_CFG, NAMED_TRANSFORMS, LIGHT_COLOR, LIGHT_INTENSITY
from idr.eval.metrics import _albedo_rmse

_WANDB_ENTITY  = "DLVC-intrinsics"
_WANDB_PROJECT = "3dfront_ct_decomp"

def transforms_cfg(name: str) -> dict:
    """Convert a named transform preset to individual cfg_overrides keys.

    Usage::
        decompose_scene(..., cfg_overrides={**transforms_cfg("all"), "lambda_tv": 1e-3})

    Available presets: "none", "all", "only_softplus", "only_shininess"
    """
    tr = NAMED_TRANSFORMS[name]
    return {
        "tr_albedo":   tr["albedo"],
        "tr_metallic": tr["metallic"],
        "tr_roughness": tr["roughness"],
        "tr_env":      tr["env"],
    }


_RUN_NAME_SKIP = frozenset({
    "n_iter", "lbfgs_max_iter", "log_every", "sbatch",
    "lr", "lr_end", "lr_schedule", "lr_schedule_step", "lr_schedule_gamma",
    "loss", "optimizer",
    "shininess_min", "shininess_max",
    # meta-params handled specially by make_run_name / decompose_scene
    "shader", "no_shadow", "init_from_gt", "freeze_intrinsics",
    "use_npy", "gt_npy", "double", "opt_params", "n_images",
    "log_gt_recon_images",
    # LM tuning knobs: keep them out of run names (lm_batch_size stays in — it is
    # the semantically interesting one: full batch vs mini-batch).
    "lm_solver", "lm_damping_init", "lm_damping_factor", "lm_damping_min",
    "lm_damping_max", "lm_adaptive_damping", "lm_learning_rate",
    "lm_attempts_per_step", "lm_jacobian_max_num_rows", "lm_jacobian_mode",
    "lm_structured", "lm_dense_max_params", "lm_image_chunk", "lm_cg_tol",
    "lm_cg_maxiter", "lm_schur_max_gb",
    "curriculum", "init_spec_noise_std", "init_seed",
})


def make_run_name(
    scene_dir: Path,
    cfg_overrides: Optional[dict] = None,
) -> str:
    """Build a deterministic run / output-directory name from decompose_scene params.

    All options — including shader, no_shadow, freeze_intrinsics, init_from_gt,
    use_npy, double, n_images, opt_params — can be passed inside cfg_overrides.
    The name encodes only the non-default, semantically meaningful options.
    """
    cfg = dict(cfg_overrides or {})
    shader            = cfg.get("shader",            "ct_sh")
    no_shadow         = cfg.get("no_shadow",         False)
    freeze_intrinsics = cfg.get("freeze_intrinsics", False)
    init_from_gt      = cfg.get("init_from_gt",      False)
    use_npy           = cfg.get("use_npy",            False)
    gt_npy            = cfg.get("gt_npy",             False)
    double            = cfg.get("double",             False)
    n_images          = cfg.get("n_images",           None)
    opt_params_raw    = cfg.get("opt_params",         None)
    if opt_params_raw is not None and not isinstance(opt_params_raw, frozenset):
        opt_params = frozenset(opt_params_raw)
    else:
        opt_params = opt_params_raw

    scene_dir = Path(scene_dir)
    scene_name = scene_dir.parent.name + "/" + scene_dir.name

    def _fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    override_tags = "_".join(
        f"{k}={_fmt(v)}"
        for k, v in cfg.items()
        if k not in _RUN_NAME_SKIP and v != DEFAULT_CFG.get(k)
    )

    opt_tag = ""
    if opt_params is not None:
        opt_tag = "_only_" + "+".join(sorted(opt_params))

    return (
        f"{scene_name}_{shader}"
        + ("_noshadow"          if no_shadow          else "")
        + ("_freeze_intrinsics" if freeze_intrinsics   else "")
        + ("_init_from_gt"      if init_from_gt        else "")
        + ("_npy"               if use_npy             else "")
        + ("_gtnpy"             if gt_npy              else "")
        + ("_f64"               if double              else "")
        + (f"_N{n_images}"      if n_images is not None else "")
        + opt_tag
        + (f"_{override_tags}"  if override_tags       else "")
    )


def _load_seg_labels(scene_dir, hw):
    """segmentation.png (RGB, one colour per SAM mask, black=unlabeled) -> (H,W) int32
    label map: contiguous class ids >= 0, or -1 where unlabeled. None if absent."""
    p = Path(scene_dir) / "segmentation.png"
    if not p.exists():
        return None
    rgb = np.asarray(Image.open(p).convert("RGB"))
    if rgb.shape[:2] != tuple(hw):
        rgb = np.asarray(Image.fromarray(rgb).resize((hw[1], hw[0]), Image.NEAREST))
    flat = rgb.reshape(-1, 3).astype(np.int64)
    codes = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    labels = np.full(codes.shape, -1, np.int32)
    fg = codes != 0                                   # black (0,0,0) -> unlabeled
    if fg.any():
        _, inv = np.unique(codes[fg], return_inverse=True)
        labels[fg] = inv.astype(np.int32)
    return labels.reshape(rgb.shape[:2])


def decompose_scene(
    scene_dir: Path,
    out_dir: Path,
    shader: str = "ct_sh",
    cfg_overrides: Optional[dict] = None,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    no_shadow: bool = False,
    init_from_gt: bool = False,
    freeze_intrinsics: bool = False,
    opt_params: Optional[frozenset] = None,
    log_gradients: bool = False,
    use_npy: bool = False,
    gt_npy: bool = False,
    double: bool = False,
    n_images: Optional[int] = None,
    device: str = "cuda",
    wandb_entity: str = _WANDB_ENTITY,
    wandb_project: str = _WANDB_PROJECT,
) -> dict:
    """Run CT intrinsic decomposition on a 3D-Front scene.

    Uses the pre-rendered light_*.png images as input observations and
    estimates albedo + per-image SH or env-map lighting.  GT metallic and
    roughness are fixed at the provided values (not optimized).

    Saves results in the same format as run_decomposition() so that
    show_results() and plotting helpers work unchanged.

    Returns the metrics dict.
    """
    scene_dir = Path(scene_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow meta-params and opt_params to be specified inside cfg_overrides.
    # cfg_overrides wins if the same key is also passed as an explicit kwarg.
    cfg_overrides = dict(cfg_overrides or {})
    shader            = cfg_overrides.pop("shader",            shader)
    no_shadow         = cfg_overrides.pop("no_shadow",         no_shadow)
    init_from_gt      = cfg_overrides.pop("init_from_gt",      init_from_gt)
    freeze_intrinsics = cfg_overrides.pop("freeze_intrinsics", freeze_intrinsics)
    use_npy           = cfg_overrides.pop("use_npy",           use_npy)
    gt_npy            = cfg_overrides.pop("gt_npy",            gt_npy)
    double            = cfg_overrides.pop("double",            double)
    n_images          = cfg_overrides.pop("n_images",          n_images)

    # External warm-start (ct_sh only): natural-space maps
    # {"albedo","sh","metallic","roughness"} that seed the optimizer's init, exactly like a
    # curriculum phase hands off to the next — but supplied from OUTSIDE the call, e.g. from
    # an earlier LBFGS run's saved estimates. This lets a VarPro polish continue from a saved
    # LBFGS result instead of re-running LBFGS as a curriculum phase. Popped before the cfg
    # merge so it never lands in the wandb config or the run name. A curriculum, if also
    # given, overwrites this after its phases run.
    init_maps_ext     = cfg_overrides.pop("init_maps", None)

    if opt_params is None and "opt_params" in cfg_overrides:
        _op = cfg_overrides.pop("opt_params")
        opt_params = frozenset(_op) if not isinstance(_op, frozenset) else _op
    else:
        cfg_overrides.pop("opt_params", None)  # discard if kwarg already given

    cfg = {**DEFAULT_CFG, **cfg_overrides}

    torch_dtype = torch.float64 if double else torch.float32

    scene = load_scene(scene_dir, no_shadow=no_shadow, use_npy=use_npy, gt_npy=gt_npy)

    # ── optional strided downsampling (nearest: GT maps/mask stay crisp) ──────
    _ds = int(cfg.get("downsample", 1) or 1)
    # Keep the full-res normals/mask: the proxy geometry is built full-res and
    # strided (stride=_ds) so the view vectors match a full-res render of the
    # scene rather than the coarse grid — otherwise the recomputed rays drift
    # sub-pixel and inject a specular error that scales with the stride.
    _full_normals_np, _full_mask_np = scene["normals_np"], scene["mask_np"]
    if _ds > 1:
        for k in ("normals_np", "mask_np", "albedo_np", "metallic_np", "roughness_np"):
            scene[k] = np.ascontiguousarray(scene[k][::_ds, ::_ds])
        scene["images"] = [np.ascontiguousarray(im[::_ds, ::_ds]) for im in scene["images"]]
        scene["H"], scene["W"] = scene["normals_np"].shape[:2]
        print(f"  [downsample] x{_ds} -> {scene['H']}x{scene['W']}")

    H, W = scene["H"], scene["W"]
    # SAM segmentation labels for the cohesion prior (segmentation.png: one colour per
    # mask, black = unlabeled). Loaded at GT res, strided to match the (downsampled) grid.
    _seg_labels_hw = _load_seg_labels(scene_dir, _full_mask_np.shape[:2])
    if _seg_labels_hw is not None and _ds > 1:
        _seg_labels_hw = np.ascontiguousarray(_seg_labels_hw[::_ds, ::_ds])
    images     = scene["images"]
    light_keys = scene["light_keys"]
    mask_np    = scene["mask_np"]

    if n_images is not None:
        images     = images[:n_images]
        light_keys = light_keys[:n_images]

    normals_hw, frag_pos_hw, mask_hw, cam_pos = make_proxy_geometry(
        _full_normals_np, _full_mask_np,
        fov_deg=fov_deg, cam_dist=cam_dist, device=device, dtype=torch_dtype, stride=_ds,
    )

    gt_metallic  = scene["metallic_np"]   # (H, W, 1)
    gt_roughness = scene["roughness_np"]  # (H, W, 1)
    gt_albedo    = scene["albedo_np"]     # (H, W, 3)
    gt_sh_coeffs = scene.get("sh_coeffs")  # list of (9,3) arrays or None
    if n_images is not None and gt_sh_coeffs is not None:
        gt_sh_coeffs = gt_sh_coeffs[:n_images]

    # ── validation split: hold out the LAST val_images for the relighting metric ─
    n_val = int(cfg.get("val_images", 0) or 0)
    val_imgs = val_sh = None
    val_keys = []
    if n_val > 0:
        if gt_sh_coeffs is None:
            print("  [val] disabled: no sh_XXX.npy GT lighting found in the scene dir")
            n_val = 0
        elif n_val >= len(images):
            raise ValueError(f"val_images={n_val} >= available images ({len(images)})")
        else:
            val_imgs     = images[-n_val:]
            val_sh       = gt_sh_coeffs[-n_val:]
            val_keys     = light_keys[-n_val:]
            images       = images[:-n_val]
            light_keys   = light_keys[:-n_val]
            gt_sh_coeffs = gt_sh_coeffs[:-n_val]
            print(f"  [val] holding out last {n_val} images for relighting "
                  f"({len(images)} train images)")

    grad_log_dir = out_dir / "gradient_flow" if log_gradients else None

    # ── build env-map sampling grid (needed for ct_env) ───────────────────────
    _sh_ref  = SHLighting.directional(
        np.array([0, 0, 1], dtype=np.float32), LIGHT_COLOR, intensity=LIGHT_INTENSITY
    )
    _env_ref = EnvMap.from_sh(_sh_ref, resolution=32)
    env_dirs, env_dw = _env_ref._dirs, _env_ref._solid_angles
    env_H, env_W     = _env_ref.image.shape[:2]

    # ── wandb run ─────────────────────────────────────────────────────────────
    scene_name = Path(scene_dir).parent.name + "/" + Path(scene_dir).name

    # Build a merged dict that make_run_name reads (meta-params + optimizer overrides)
    _meta = dict(
        shader=shader, no_shadow=no_shadow, init_from_gt=init_from_gt,
        freeze_intrinsics=freeze_intrinsics, use_npy=use_npy, gt_npy=gt_npy, double=double,
    )
    if n_images is not None:
        _meta["n_images"] = n_images
    if opt_params is not None:
        _meta["opt_params"] = opt_params
    run_name = make_run_name(scene_dir, {**_meta, **cfg_overrides})

    def _fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    wandb_tags = [shader]
    if freeze_intrinsics:
        wandb_tags.append("freeze_intrinsics")
    if init_from_gt:
        wandb_tags.append("init_from_gt")
    if no_shadow:
        wandb_tags.append("noshadow")
    if use_npy:
        wandb_tags.append("npy")
    if gt_npy:
        wandb_tags.append("gtnpy")
    if double:
        wandb_tags.append("f64")
    if opt_params is not None:
        wandb_tags.append("only_" + "+".join(sorted(opt_params)))
    wandb_tags += [
        f"{k}={_fmt(v)}"
        for k, v in cfg_overrides.items()
        if k not in _RUN_NAME_SKIP
    ]

    run = wandb.init(
        entity  =wandb_entity,
        project =wandb_project,
        config  =dict(
            **cfg,
            shader=shader, scene=str(scene_dir),
            fov_deg=fov_deg, cam_dist=cam_dist,
            no_shadow=no_shadow, init_from_gt=init_from_gt,
            freeze_intrinsics=freeze_intrinsics, use_npy=use_npy, double=double,
            opt_params=sorted(opt_params) if opt_params is not None else None,
            n_images=len(images), H=H, W=W,
        ),
        name    =run_name,
        tags    =wandb_tags,
        reinit  =True,
    )
    _log_gt_recon = bool(cfg.get("log_gt_recon_images", False))
    if _log_gt_recon:
        run.log({"gt_images": [wandb.Image(img) for img in images]}, step=0)

    # ── optimize ──────────────────────────────────────────────────────────────
    t0 = time.time()

    # Resolve effective opt_params:
    # - explicit opt_params overrides everything
    # - freeze_intrinsics falls back to the legacy "only optimize lighting" behaviour
    # - None means optimize everything
    if opt_params is not None:
        _eff_op_sh  = opt_params
        _eff_op_env = opt_params
    elif freeze_intrinsics:
        _eff_op_sh  = frozenset({"sh"})
        _eff_op_env = frozenset({"env"})
    else:
        _eff_op_sh  = None
        _eff_op_env = None

    # ── curriculum warm-start (ct_sh only) ────────────────────────────────────
    # cfg["curriculum"] = list of phase dicts, each {n_iter, opt_params?, sh_order?,
    # pixel_frac?}. Each phase optimizes a sub-problem and hands its natural-space
    # maps to the next via init_maps; the main optimize below warm-starts from the
    # last phase. Lets us freeze groups / go SH1->SH2 / fit lighting on a pixel
    # subset first, all within the existing metrics + artifact pipeline.
    _curr_init_maps = init_maps_ext  # external warm-start seeds the first phase / the final run
    _wstep = 0                       # running wandb step offset across curriculum phases
    _curriculum = cfg.pop("curriculum", None)
    if shader == "ct_sh" and _curriculum:
        _seed0 = int(cfg.get("init_seed", 0) or 0)
        for _pi, _ph in enumerate(_curriculum):
            # A phase may override ANY cfg key, not just the four that used to be
            # special-cased. That is what lets a phase switch optimizer -- e.g.
            # [{"optimizer": "LBFGS", ...}, {"optimizer": "VARPRO", ...}] to warm-start
            # variable projection from a gradient-descent run -- without this loop
            # needing to know which keys exist.
            _pcfg = {**cfg, **{k: v for k, v in _ph.items()
                               if k not in ("pixel_frac", "opt_params")}}
            _pcfg["n_iter"]   = int(_ph.get("n_iter", 60))
            _pcfg["sh_order"] = int(_ph.get("sh_order", cfg.get("sh_order", 2)))
            _pmask = mask_hw
            if _ph.get("pixel_frac"):
                _pmask = _subsample_mask(mask_hw, float(_ph["pixel_frac"]), seed=_seed0 + _pi)
            _pop = frozenset(_ph["opt_params"]) if _ph.get("opt_params") else _eff_op_sh
            print(f"  [curriculum {_pi + 1}/{len(_curriculum)}] "
                  f"{_pcfg.get('optimizer', 'LBFGS')} "
                  f"opt={sorted(_pop) if _pop else 'all'} sh{_pcfg['sh_order']} "
                  f"n_iter={_pcfg['n_iter']}"
                  + (f" pixels={_ph['pixel_frac']:.0%}" if _ph.get("pixel_frac") else ""),
                  flush=True)
            # Curriculum phases now log to the SAME wandb run as the final stage, laid
            # end to end on one step timeline via _wstep, tagged with the phase index so
            # the GD->VarPro boundary is visible on every metric. (They used to pass
            # wandb_run=None and were invisible.)
            _a, _s, _m, _b, *_ = _optimize_ct_sh(
                images, normals_hw, frag_pos_hw, _pmask, cam_pos,
                gt_metallic, gt_roughness, _pcfg,
                wandb_run=run, gt_sh_coeffs=gt_sh_coeffs, gt_albedo=gt_albedo,
                opt_params=_pop, init_from_gt=init_from_gt,
                val_images=None, val_sh_coeffs=None, init_maps=_curr_init_maps,
                wandb_step_offset=_wstep, wandb_phase=_pi)
            _curr_init_maps = {"albedo": _a, "sh": _s, "metallic": _m, "roughness": _b}
            _wstep += _pcfg["n_iter"]

    if shader not in ("ct_sh", "ct_env"):
        raise ValueError(f"shader must be 'ct_sh' or 'ct_env', got {shader!r}")
    _res = optimize(
        shader, images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
        gt_metallic, gt_roughness, cfg,
        env=EnvGrid(env_dirs, env_dw, env_H, env_W) if shader == "ct_env" else None,
        wandb_run=run,
        gt_sh_coeffs=gt_sh_coeffs,
        gt_albedo=gt_albedo,
        opt_params=_eff_op_env if shader == "ct_env" else _eff_op_sh,
        **({"seg_labels_hw": _seg_labels_hw} if shader in ("ct_sh", "ct_env") else {}),
        init_from_gt=init_from_gt,
        log_gradients=log_gradients,
        grad_log_dir=grad_log_dir,
        val_images=val_imgs,
        val_sh_coeffs=val_sh,
        # Continue the curriculum's step timeline into the final stage and give it the
        # next phase index. ct_sh only: the offset kwargs are ct_sh's, and the
        # curriculum (hence a non-zero offset) is ct_sh-only anyway.
        **({} if shader == "ct_env" else {
            "init_maps": _curr_init_maps,
            "wandb_step_offset": _wstep,
            "wandb_phase": len(_curriculum) if _curriculum else None,
        }),
    )
    albedo, mat_a, mat_b = _res.albedo, _res.mat_a, _res.mat_b
    shadings, history, elapsed = _res.shadings, _res.history, _res.elapsed
    sh_out, env_maps_out = _res.sh, _res.env_maps

    # ── albedo RMSE/MAE + scale ────────────────────────────────────────────────
    mask_flat  = mask_np.reshape(-1)
    est_px  = torch.from_numpy(albedo[mask_np])
    gt_px   = torch.from_numpy(gt_albedo[mask_np])
    rmse_t, scale_t = _albedo_rmse(est_px, gt_px)
    rmse    = float(rmse_t)
    scale   = scale_t.numpy()
    albedo_mae = float((est_px * scale_t - gt_px).abs().mean())

    # ── final relighting error on the held-out val images ─────────────────────
    # Render the held-out lights with the ESTIMATED intrinsics + their GT
    # lighting; save target / relit / residual arrays per light under
    # out_dir/relight/ so the caller can plot them, and record per-light metrics.
    relight = {}
    _sh_ord_final = int(cfg.get("sh_order", 2))
    def _pad_sh_final(s):
        s = np.asarray(s, np.float32)
        n = (_sh_ord_final + 1) ** 2
        if s.shape[0] < n:
            s = np.concatenate([s, np.zeros((n - s.shape[0], 3), np.float32)])
        return s
    if val_imgs is not None:
        from idr.render import build_sh_basis
        relight_dir = out_dir / "relight"
        relight_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            _fm  = mask_hw.reshape(-1)
            _N   = normals_hw.reshape(-1, 3)[_fm]
            _V   = torch.nn.functional.normalize(
                cam_pos.unsqueeze(0) - frag_pos_hw.reshape(-1, 3)[_fm], dim=-1)
            _ab  = torch.from_numpy(albedo).to(device, torch_dtype).reshape(-1, 3)[_fm]
            _me  = torch.from_numpy(mat_a).to(device, torch_dtype).reshape(-1, 1)[_fm]
            _ro  = torch.from_numpy(mat_b).to(device, torch_dtype).reshape(-1, 1)[_fm]
            _fm_np = _fm.cpu().numpy()
            _lut = _get_ggx_sh_lut(device, n_bands=_sh_ord_final + 1).to(torch_dtype) \
                if shader == "ct_sh" else None
            rs, ms = [], []
            for vi, (v_img, v_sh) in enumerate(zip(val_imgs, val_sh)):
                _dfres = bool(cfg.get("diffuse_fresnel", True))
                if shader == "ct_sh":
                    sh_t = torch.from_numpy(_pad_sh_final(v_sh)).to(device, torch_dtype)
                    recon = shade_ct_sh(_V, _N, _ab, sh_t, _me, _ro, lut=_lut,
                                        diffuse_fresnel=_dfres,
                                        hl_mode=str(cfg.get("hl_mode", "analytic")))
                else:
                    env_pix = np.maximum(build_sh_basis(env_dirs) @ np.asarray(v_sh, np.float32), 0.0)
                    recon = shade_ct_env(
                        _V, _N, _ab,
                        torch.from_numpy(env_pix).to(device, torch_dtype),
                        torch.from_numpy(env_dirs).to(device, torch_dtype),
                        torch.from_numpy(env_dw).to(device, torch_dtype),
                        _me, _ro, sbatch=cfg.get("sbatch", 64),
                        spec_importance=cfg.get("spec_importance", False),
                        spec_samples=cfg.get("spec_samples", 64),
                        diffuse_fresnel=_dfres)
                tgt = torch.from_numpy(
                    np.asarray(v_img, np.float32).reshape(-1, 3)[_fm_np]).to(device, torch_dtype)
                d = recon - tgt
                r, m = float(d.pow(2).mean().sqrt()), float(d.abs().mean())
                rs.append(r); ms.append(m)
                relit_full = np.zeros((H * W, 3), np.float32)
                relit_full[_fm_np] = recon.float().cpu().numpy()
                vk = val_keys[vi] if vi < len(val_keys) else f"val_{vi:03d}"
                np.save(relight_dir / f"relit_{vk}.npy", relit_full.reshape(H, W, 3))
        relight = dict(relight_rmse=float(np.mean(rs)), relight_mae=float(np.mean(ms)),
                       relight_rmse_per_light=[float(x) for x in rs],
                       relight_mae_per_light=[float(x) for x in ms],
                       relight_keys=list(val_keys))

    inv_scale = 1.0 / np.maximum(scale[None, None, :], 1e-8)
    if shader == "ct_sh":
        sh_out_rescaled   = sh_out * inv_scale
        env_maps_rescaled = np.empty(0)
    else:
        sh_out_rescaled   = np.empty(0)
        env_maps_rescaled = env_maps_out * inv_scale

    # Use np.subtract (not the `-` operator) for the albedo error: `albedo_scaled - gt_albedo`
    # hit a numpy edge case that wrote the difference back INTO albedo_scaled in place,
    # corrupting the saved albedo_scaled.npy into the signed albedo error. np.subtract into a
    # fresh output leaves albedo_scaled intact.
    albedo_scaled = (albedo * scale).clip(0, 1)
    albedo_err    = np.abs(np.subtract(albedo_scaled, gt_albedo)) * mask_np[:, :, None]

    mat_a_err = np.abs(mat_a - gt_metallic)
    mat_b_err = np.abs(mat_b - gt_roughness)

    recon_err  = [np.abs(s - img) * mask_np[:, :, None]
                  for s, img in zip(shadings, images)]
    # recon_mae keeps the historical name's value (mean abs err); recon_rmse is
    # a true per-image RMSE averaged over the training lights.
    recon_mae  = float(np.mean([e[mask_np].mean() for e in recon_err]))
    recon_rmse = float(np.mean([
        np.sqrt(((s - img) ** 2)[mask_np].mean()) for s, img in zip(shadings, images)]))

    metrics = dict(
        albedo_rmse=rmse, albedo_mae=albedo_mae, final_loss=float(history[-1]),
        recon_rmse=recon_rmse, recon_mae=recon_mae,
        **relight,
        n_train_images=len(images), n_val_images=n_val,
        albedo_scale=scale.tolist(), loss_history=history,
        metallic_est_mean=float(mat_a[mask_np].mean()),
        metallic_gt=float(gt_metallic[mask_np].mean()),
        metallic_err_mean=float(mat_a_err[mask_np].mean()),
        roughness_est_mean=float(mat_b[mask_np].mean()),
        roughness_gt=float(gt_roughness[mask_np].mean()),
        roughness_err_mean=float(mat_b_err[mask_np].mean()),
        elapsed_s=elapsed,
        shader=shader,
        scene=str(scene_dir),
    )

    # ── save to disk ──────────────────────────────────────────────────────────
    def _save_gray(arr_hw1, path):
        Image.fromarray(
            (arr_hw1.squeeze(-1).clip(0, 1) * 255).astype(np.uint8)
        ).save(path)

    recon_dir = out_dir / "reconstructions"
    recon_dir.mkdir(exist_ok=True)

    # GT maps (so show_results can find them in the expected location)
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    np.save(gt_dir / "albedo.npy",    gt_albedo.astype(np.float32))
    np.save(gt_dir / "metallic.npy",  gt_metallic.astype(np.float32))
    np.save(gt_dir / "roughness.npy", gt_roughness.astype(np.float32))

    Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
        out_dir / "albedo_est.png")
    np.save(out_dir / "albedo_est.npy", albedo.astype(np.float32))
    Image.fromarray((albedo_scaled * 255).astype(np.uint8)).save(
        out_dir / "albedo_scaled.png")
    np.save(out_dir / "albedo_scaled.npy", albedo_scaled.astype(np.float32))
    np.save(out_dir / "albedo_err.npy", albedo_err.astype(np.float32))

    _save_gray(mat_a, out_dir / "metallic_est.png")
    np.save(out_dir / "metallic_est.npy", mat_a.astype(np.float32))
    _save_gray(mat_b, out_dir / "roughness_est.png")
    np.save(out_dir / "roughness_est.npy", mat_b.astype(np.float32))

    _save_gray(mat_a_err * mask_np[:, :, None], out_dir / "metallic_err.png")
    np.save(out_dir / "metallic_err.npy", (mat_a_err * mask_np[:, :, None]).astype(np.float32))
    _save_gray(mat_b_err * mask_np[:, :, None], out_dir / "roughness_err.png")
    np.save(out_dir / "roughness_err.npy", (mat_b_err * mask_np[:, :, None]).astype(np.float32))

    for k, (s, e, lk) in enumerate(zip(shadings, recon_err, light_keys)):
        # recon is LINEAR radiance; sRGB-encode the PNG preview (the .npy stays linear).
        Image.fromarray((linear_to_srgb(s.clip(0, 1)) * 255).astype(np.uint8)).save(
            recon_dir / f"recon_{lk}.png")
        np.save(recon_dir / f"recon_{lk}.npy", s.astype(np.float32))
        Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
            recon_dir / f"recon_err_{lk}.png")

    if shader == "ct_sh":
        np.save(out_dir / "sh_coeffs_est.npy", sh_out_rescaled)
        sh_env_imgs = []
        for k, sh_k in enumerate(sh_out_rescaled):
            sh_env_img = _sh_coeffs_to_env_img(sh_k)
            sh_env_imgs.append(sh_env_img)
            Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                out_dir / f"sh_env_map_{light_keys[k]}.png")
            np.save(out_dir / f"sh_env_map_{light_keys[k]}.npy", sh_env_img.astype(np.float32))
        light_img_key   = "est_sh_env_maps"
        light_imgs_wandb = [wandb.Image(e) for e in sh_env_imgs]
    else:
        np.save(out_dir / "env_maps_est.npy", env_maps_rescaled)
        env_imgs = []
        for k, env_k in enumerate(env_maps_rescaled):
            env_img = _env_flat_to_img(env_k, env_H, env_W)
            env_imgs.append(env_img)
            Image.fromarray((env_img * 255).astype(np.uint8)).save(
                out_dir / f"env_map_{light_keys[k]}.png")
            np.save(out_dir / f"env_map_{light_keys[k]}.npy", env_img.astype(np.float32))
        light_img_key   = "est_env_maps"
        light_imgs_wandb = [wandb.Image(e) for e in env_imgs]

    # ── wandb final summary ────────────────────────────────────────────────────
    # Intrinsics estimations (albedo shown scaled only) + GT, side by side — no
    # error maps. GT input images and recon error maps only when explicitly asked.
    _final = {
        "albedo_scaled":   wandb.Image(albedo_scaled),
        "metallic_est":    wandb.Image(mat_a.squeeze(-1)),
        "roughness_est":   wandb.Image(mat_b.squeeze(-1)),
        "gt_albedo":       wandb.Image(gt_albedo),
        "gt_metallic":     wandb.Image(gt_metallic[:, :, 0]),
        "gt_roughness":    wandb.Image(gt_roughness[:, :, 0]),
        light_img_key:     light_imgs_wandb,
        "albedo_rmse":     rmse,
        "albedo_mae":      albedo_mae,
        "recon_rmse":      recon_rmse,
        "final_loss":      history[-1],
        "elapsed_s":       elapsed,
        **relight,
    }
    if _log_gt_recon:
        _final["gt_images"]      = [wandb.Image(img) for img in images]
        _final["recon_err_maps"] = [wandb.Image(e.mean(-1)) for e in recon_err]
    # Land the final summary at the END of the (possibly multi-phase) timeline, not at
    # the main run's own n_iter -- a curriculum shifts every phase forward by _wstep, so
    # cfg["n_iter"] alone would drop this back on top of the last curriculum phase.
    run.log(_final, step=_wstep + cfg["n_iter"])
    run.finish()

    # metrics.json is the completion marker (written last); write it atomically so
    # an interrupt mid-write can't leave a truncated file that poisons a resume.
    import os as _os
    _tmp = out_dir / "metrics.json.tmp"
    with open(_tmp, "w") as fh:
        json.dump(metrics, fh, indent=2)
    _os.replace(_tmp, out_dir / "metrics.json")

    print(
        f"  {elapsed:.1f}s  albedo RMSE={rmse:.4f}"
        f"  metallic={metrics['metallic_est_mean']:.3f}(GT={metrics['metallic_gt']:.3f})"
        f"  roughness={metrics['roughness_est_mean']:.3f}(GT={metrics['roughness_gt']:.3f})"
        f"  -> {out_dir}"
    )
    return metrics


def _build_parser():
    p = argparse.ArgumentParser(description="CT decomp/render for 3D-Front scenes")
    sub = p.add_subparsers(dest="cmd", required=True)

    # render sub-command
    r = sub.add_parser("render", help="Render 3D-Front scene from GT maps + directional SH")
    r.add_argument("--scene",  required=True)
    r.add_argument("--out",    required=True)
    r.add_argument("--az",     type=float, default=45.0, help="azimuth degrees (0=frontal)")
    r.add_argument("--el",     type=float, default=0.0,  help="elevation degrees")
    r.add_argument("--intensity", type=float, default=LIGHT_INTENSITY)
    r.add_argument("--shader", choices=["ct_sh", "ct_env"], default="ct_sh")
    r.add_argument("--fov",    type=float, default=60.0)
    r.add_argument("--device", default="cuda")

    # dataset sub-command
    ds = sub.add_parser("dataset", help="Render 3D-Front GT with random SH lighting dataset")
    ds.add_argument("--scene",    required=True)
    ds.add_argument("--out",      required=True)
    ds.add_argument("--n_lights", type=int,   default=16)
    ds.add_argument("--seed",     type=int,   default=0)
    ds.add_argument("--fov",      type=float, default=60.0)
    ds.add_argument("--device",   default="cuda")

    # decompose sub-command
    d = sub.add_parser("decompose", help="Decompose 3D-Front scene: estimate albedo + lighting")
    d.add_argument("--scene",      required=True)
    d.add_argument("--out",        required=True)
    d.add_argument("--shader",     choices=["ct_sh", "ct_env"], default="ct_sh")
    d.add_argument("--n_iter",     type=int,   default=100)
    d.add_argument("--lambda_tv",  type=float, default=0.0)
    d.add_argument("--fov",        type=float, default=60.0)
    d.add_argument("--no_shadow",  action="store_true")
    d.add_argument("--log_grads",  action="store_true")
    d.add_argument("--device",     default="cuda")
    return p


def main():
    args = _build_parser().parse_args()
    if args.cmd == "render":
        render_scene(
            Path(args.scene), Path(args.out),
            az_deg=args.az, el_deg=args.el, intensity=args.intensity,
            shader=args.shader, fov_deg=args.fov, device=args.device,
        )
    elif args.cmd == "dataset":
        render_3dfront_dataset(
            Path(args.scene), Path(args.out),
            n_lights=args.n_lights, seed=args.seed,
            fov_deg=args.fov, device=args.device,
        )
    else:
        cfg = {"n_iter": args.n_iter, "lambda_tv": args.lambda_tv}
        decompose_scene(
            Path(args.scene), Path(args.out),
            shader=args.shader, cfg_overrides=cfg,
            fov_deg=args.fov, no_shadow=args.no_shadow,
            log_gradients=args.log_grads, device=args.device,
        )
