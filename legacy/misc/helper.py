import numpy as np
import torch


def _uint8_compression(tensor):
    q = (tensor.clamp(0, 1) * 255).round() / 255.0
    return tensor + (q - tensor).detach()


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


def _rand_color(rng) -> np.ndarray:
    c = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    if rng.random() > 0.5:
        c[0] = min(1.0, c[0] * 1.15)
    else:
        c[2] = min(1.0, c[2] * 1.15)
    return c


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
    est = est.detach().clone()
    gt = gt.detach()
    num = (gt * est).sum(0)
    den = (est * est).sum(0).clamp_min(1e-8)
    scale = num / den
    est *= scale
    return torch.sqrt(((gt - est)**2).mean()), scale


def _to_img(tensor, H, W):
    return tensor.detach().reshape(H, W, -1).clamp(0, 1).cpu().numpy()


def _shininess_to_img(tensor, H, W, range):
    min_s, max_s = range
    return ((tensor.detach().reshape(H, W, -1).cpu().numpy() - min_s) / (max_s - min_s)).clip(0, 1)


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
