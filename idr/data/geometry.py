"""Proxy camera geometry for scenes that have no depth buffer.

3D-Front renders give normals but no depth, so fragment positions are approximated by
spreading pixels over the z=0 plane under a perspective camera. That reproduces the
view-direction variation across the image, which is what the specular term needs.
"""
from __future__ import annotations

import numpy as np
import torch

def make_proxy_geometry(
    normals_np: np.ndarray,
    mask_np: np.ndarray,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """Return (normals_hw, frag_pos_hw, mask_hw, cam_pos) tensors on device.

    Since the 3D-Front scene has no depth map, fragment positions are
    approximated by placing the camera at (0, 0, cam_dist) and distributing
    pixel positions on the z=0 plane according to fov_deg (perspective).
    This gives correct view-direction variation across the image without
    actual depth.
    """
    H, W = normals_np.shape[:2]

    # Pixel grid in NDC [-1, 1] (y flipped: top row = +1)
    ys = np.linspace(1.0, -1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0,  1.0, W, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)  # (H, W)

    tan_half = float(np.tan(np.radians(fov_deg / 2)))
    frag_pos = np.stack([
        xg * tan_half,
        yg * tan_half,
        np.zeros((H, W), dtype=np.float32),
    ], axis=-1)  # (H, W, 3)

    np_fdtype = np.float64 if dtype == torch.float64 else np.float32
    cam_pos = np.array([0.0, 0.0, cam_dist], dtype=np_fdtype)

    def _tf(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np_fdtype)).to(device, dtype)

    return (
        _tf(normals_np),
        _tf(frag_pos),
        torch.from_numpy(mask_np.astype(np.bool_)).to(device),
        _tf(cam_pos),
    )


def _subsample_mask(mask_hw: torch.Tensor, frac: float, seed: int = 0) -> torch.Tensor:
    """A boolean mask keeping a random `frac` of the foreground pixels (for the
    'fit lighting on a pixel subset first' curriculum phase)."""
    flat = mask_hw.reshape(-1)
    idx = torch.nonzero(flat, as_tuple=False).squeeze(-1)
    n_keep = max(1, int(round(frac * idx.numel())))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    keep = idx[torch.randperm(idx.numel(), generator=g)[:n_keep].to(idx.device)]
    out = torch.zeros_like(flat)
    out[keep] = True
    return out.reshape(mask_hw.shape)
