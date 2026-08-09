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
    stride: int = 1,
) -> tuple:
    """Return (normals_hw, frag_pos_hw, mask_hw, cam_pos) tensors on device.

    Since the 3D-Front scene has no depth map, fragment positions are
    approximated by placing the camera at (0, 0, cam_dist) and distributing
    pixel positions on the z=0 plane according to fov_deg (perspective).
    This gives correct view-direction variation across the image without
    actual depth.

    ``stride`` > 1 analyses a strided sub-grid **at the full-resolution
    coordinates**: the NDC grid (hence every ray direction) is built on the full
    ``(H, W)`` and only then subsampled ``[::stride, ::stride]``. Pass the
    FULL-resolution ``normals_np`` / ``mask_np`` together with ``stride`` — this
    keeps the view vectors identical to a full-res render of the same scene, which
    a naive "downsample the maps first, then rebuild geometry on the coarse grid"
    does NOT (the coarse ``linspace`` places rays at different NDC than the kept
    pixels, injecting a specular error that scales with the stride). ``stride=1``
    is the identity and reproduces the previous behaviour bit-for-bit.
    """
    H, W = normals_np.shape[:2]

    # Pixel grid in NDC [-1, 1] (y flipped: top row = +1), built at FULL resolution
    # so a strided subset keeps the same coordinates a full-res render used.
    ys = np.linspace(1.0, -1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0,  1.0, W, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)  # (H, W)

    tan_half = float(np.tan(np.radians(fov_deg / 2)))
    frag_pos = np.stack([
        xg * tan_half,
        yg * tan_half,
        np.zeros((H, W), dtype=np.float32),
    ], axis=-1)  # (H, W, 3)

    if stride > 1:
        sl = (slice(None, None, stride), slice(None, None, stride))
        normals_np = normals_np[sl]
        mask_np = mask_np[sl]
        frag_pos = np.ascontiguousarray(frag_pos[sl])

    np_fdtype = np.float64 if dtype == torch.float64 else np.float32
    cam_pos = np.array([0.0, 0.0, cam_dist], dtype=np_fdtype)

    def _tf(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x).astype(np_fdtype)).to(device, dtype)

    return (
        _tf(normals_np),
        _tf(frag_pos),
        torch.from_numpy(np.ascontiguousarray(mask_np).astype(np.bool_)).to(device),
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
