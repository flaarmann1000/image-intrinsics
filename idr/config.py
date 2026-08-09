"""Default configuration, parameter-transform presets, and scene material presets.

DEFAULT_CFG is the single source of truth for optimizer settings; callers pass
overrides and every entry point merges them on top of this.
"""
from __future__ import annotations

import numpy as np

from idr.render import Camera

MATERIAL_CONFIGS: dict[str, dict] = {
    "default":          dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.4),
    "albedo_0":         dict(albedo=[0.8, 0.3, 0.2], metallic=0.1, roughness=0.4),
    "albedo_1":         dict(albedo=[0.2, 0.5, 0.8], metallic=0.1, roughness=0.4),
    "metallic_0":       dict(albedo=[0.5, 0.5, 0.5], metallic=0.0, roughness=0.4),
    "metallic_1":       dict(albedo=[0.5, 0.5, 0.5], metallic=0.8, roughness=0.4),
    "roughness_0":      dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.1),
    "roughness_1":      dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.8),
    # checkerboard: two colors sampled from [0.1, 0.9] per channel
    "albedo_checker":   dict(albedo_checker=([0.8, 0.3, 0.2], [0.2, 0.5, 0.8]),
                             metallic=0.1, roughness=0.4, n_tiles=4),
    # random patch textures for all parameters
    "all_texture":      dict(albedo_range=([0.1, 0.1, 0.1], [0.9, 0.9, 0.9]),
                             metallic_range=(0.0, 1.0),
                             roughness_range=(0.1, 0.9),
                             n_tiles=16, seed=42),
}


PHONG_MATERIAL_CONFIGS: dict[str, dict] = {
    "default":           dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_0":          dict(albedo=[0.8, 0.3, 0.2], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_1":          dict(albedo=[0.2, 0.5, 0.8], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "shininess_0":       dict(albedo=[0.5, 0.5, 0.5], shininess=4.0,   ks=0.5, ka=0.0, kd=1.0),
    "shininess_1":       dict(albedo=[0.5, 0.5, 0.5], shininess=128.0, ks=0.5, ka=0.0, kd=1.0),
    "ks_0":              dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.1, ka=0.0, kd=1.0),
    "ks_1":              dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.9, ka=0.0, kd=1.0),
    # checkerboard: two values sampled from per-parameter allowed range
    "albedo_checker":    dict(albedo_checker=([0.8, 0.3, 0.2], [0.2, 0.5, 0.8]),
                              shininess=32.0, ks=0.5, ka=0.0, kd=1.0, n_tiles=4),
    "shininess_checker": dict(albedo=[0.5, 0.5, 0.5],
                              shininess_checker=(4.0, 63.0),
                              ks=0.5, ka=0.0, kd=1.0, n_tiles=4),
    "ks_checker":        dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,
                              ks_checker=(0.1, 0.9), ka=0.0, kd=1.0, n_tiles=4),
    # random patch textures for all parameters
    "all_texture":       dict(albedo_range=([0.1, 0.1, 0.1], [0.9, 0.9, 0.9]),
                              shininess_range=(4.0, 63.0),
                              ks_range=(0.1, 0.9),
                              ka=0.0, kd=1.0, n_tiles=16, seed=42),
}


SHININESS_RANGE = (1.0, 63.0)


LIGHT_ANGLES_DEG = [0, 18, 36, 54, 72, 90]


LIGHT_COLOR      = np.array([1.0, 0.9, 0.8], dtype=np.float32)


LIGHT_INTENSITY  = 2.0


DEFAULT_CAMERA = Camera(  # type: ignore[call-arg]
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target  =np.array([0.0, 0.0, 0.0], dtype=np.float32),
)


DEFAULT_CFG = dict(
    optimizer      = "LBFGS",
    n_iter         = 50,
    lbfgs_max_iter = 20,
    lr             = 1.0,
    lambda_sparse  = 0.0,
    lambda_white   = 0.0,
    lambda_tv      = 0.0,
    sbatch         = 64,
    log_every      = 20,
    loss           = "L2",
    shininess_min  = SHININESS_RANGE[0],
    shininess_max  = SHININESS_RANGE[1],
    spec_warmup_steps    = 0,
    min_metallic_steps   = 0,
    init_spec_zero       = False,
    init_roughness_zero  = False,
    lambda_metallic_l1        = 0.0,
    lambda_metallic_binarize  = 0.0,
    # Soft box constraint for natural-space (identity-transform) material optimisation:
    # penalises albedo/metallic/roughness values outside [0, 1] with a squared hinge, so
    # they stay physical without the gradient-saturation of a sigmoid reparameterisation.
    # No-op (0.0) by default and under sigmoid transforms (values already in range).
    lambda_box                = 0.0,
    # Lighting prior. Pulls the per-image SH toward a reference (pass `light_prior=<(K,n_sh,3)>`
    # to optimize()) with ||sh - ref||^2; if no reference is given, falls back to an SH
    # SMOOTHNESS prior that shrinks the non-DC (directional) coefficients toward 0. Targets the
    # albedo<->lighting ambiguity from the lighting side. No-op at 0.0.
    lambda_light_prior        = 0.0,
    # Monochrome-light prior (nvdiffrec/nvdiffrecmc-style): penalise the per-image SH's colour
    # imbalance ||sh - mean_RGB(sh)||^2, pushing lighting toward achromatic so *colour* is
    # explained by albedo instead of baked into the light. Breaks the colour axis of the
    # albedo<->lighting ambiguity. No-op at 0.0.
    lambda_light_mono         = 0.0,
    lr_end            = 0.0,
    lr_schedule       = "none",
    lr_schedule_step  = 50,
    lr_schedule_gamma = 0.5,
    # per-parameter transform ("none" | "sigmoid" | "log" | "softplus")
    tr_albedo    = "none",
    tr_metallic  = "none",
    tr_roughness = "none",
    tr_env       = "none",
    # rescale albedo+lighting toward GT after every N steps (0 = disabled)
    rescale_every = 0,
    # accumulate gradients over chunks of this many images per step (0 = all
    # images in one autograd graph). Bounds peak memory to ~img_batch images;
    # numerically identical to full-batch up to float summation order.
    img_batch = 0,
    # SH lighting order for the ct_sh shader: 2 (9 coeffs, default) or 3
    # (16 coeffs). Band 3 has zero Lambertian irradiance weight, so order 3
    # only sharpens the SPECULAR term. GT SH given as (9,3) is zero-padded.
    sh_order = 2,
    # Source of the GGX specular zonal-band weights h_l in shade_ct_sh:
    # "analytic" (closed form, correct at every roughness — default) or "lut"
    # (the shipped uniform table, bit-identical to the pre-2026-08 behaviour but
    # under-resolved below roughness ~0.08, where its low-r knots collapse toward 0
    # and its lerp gives a staircase dh/dr). Datasets rendered with one mode are an
    # exact inverse crime only when decomposed with the SAME mode; set "lut" to
    # reproduce / decompose datasets rendered before the analytic default landed.
    hl_mode = "analytic",
    # integer stride for downsampling images + GT maps before optimization
    # (nearest/strided, keeps GT crisp). 1 = full resolution.
    downsample = 1,
    # cap on the number of PER-IMAGE wandb previews (recons, env maps, err maps)
    # logged each step. Scalar metrics still use ALL images; this only limits the
    # image uploads, which otherwise dominate runtime for large N. None = all.
    wandb_max_images = None,
    # diffuse Fresnel: multiply the diffuse by (1-F) on top of (1-metallic).
    # MUST match the data generator + final shadings + relight (all default True,
    # i.e. shade_ct_sh/shade_ct_env default), or recon_rmse decouples from the
    # data loss. True = energy-conserving (specular takes energy from diffuse).
    diffuse_fresnel = True,
    # Huber transition point (linear radiance), used when loss == "huber"
    huber_delta = 0.05,
    # ct_env only: compute the specular term by GGX importance sampling
    # (deterministic, valid at all roughness) instead of the texel-grid
    # Riemann sum, which aliases below roughness ~0.3 on the 32x64 grid.
    spec_importance = False,
    spec_samples    = 64,
    # hold out the last N images as a validation set: they are excluded from
    # optimization and, at every log step, re-rendered with the CURRENT
    # intrinsics + their GT lighting -> "relight_rmse"/"relight_mae".
    # Requires GT SH (sh_XXX.npy) in the scene dir. 0 = off.
    val_images = 0,
)


NAMED_TRANSFORMS: dict[str, dict] = {
    "none": dict(albedo="none", metallic="none", roughness="none",
                 shininess="none", ks="none", env="none"),
    "all":  dict(albedo="log",  metallic="sigmoid", roughness="sigmoid",
                 shininess="sigmoid", ks="sigmoid", env="softplus"),
    "only_softplus":  dict(albedo="none",  metallic="none", roughness="none",
                 shininess="none", ks="none", env="softplus"),
    "only_shininess":  dict(albedo="none",  metallic="none", roughness="none",
                 shininess="log", ks="none", env="none"),
}


_CT_SH_PARAMS    = frozenset({"albedo", "sh",  "metallic",  "roughness"})


_CT_ENV_PARAMS   = frozenset({"albedo", "env", "metallic",  "roughness"})


_PHONG_SH_PARAMS = frozenset({"albedo", "sh",  "shininess", "ks"})


_PHONG_ENV_PARAMS= frozenset({"albedo", "env", "shininess", "ks"})
