import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, TexturesVertex,
)

import optimize_sh

OUT_DIR = "pt3d"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup — same camera/raster as create_sh_dataset.py ───────────────────────
device = "cuda"
torch.manual_seed(42)

R, T = look_at_view_transform(dist=2.7, elev=10, azim=20)
cameras_pt3d = FoVPerspectiveCameras(device=device, R=R, T=T)
raster_pt3d = RasterizationSettings(
    image_size=512, blur_radius=0.0, faces_per_pixel=1)

# ── Ground truth: two spheres with distinct albedo colors ─────────────────────
# A single uniform sphere can't separate SH0 from albedo (a*SH0 is degenerate).
# Two different colors fix the ratio a1/a2 from the data, breaking that ambiguity.
SCALE = 0.55
POSITIONS = [
    torch.tensor([[[-0.75, 0.0, 0.0]]], device=device),
    torch.tensor([[[0.75, 0.0, 0.0]]], device=device),
]
GT_COLORS = [
    torch.tensor([0.8, 0.2, 0.3], device=device),  # warm red
    torch.tensor([0.2, 0.6, 0.8], device=device),  # cool blue
]


def _sphere(pos, color=None, scale=SCALE):
    s = ico_sphere(level=3, device=device)
    s = s.update_padded(s.verts_padded() * scale + pos)
    if color is not None:
        V = s.verts_padded().shape[1]
        s.textures = TexturesVertex(
            verts_features=color.expand(V, -1).unsqueeze(0))
    return s


mesh_gt = join_meshes_as_scene([_sphere(p, c)
                               for p, c in zip(POSITIONS, GT_COLORS)])

# ── Render N images with random SH coefficients (same as create_sh_dataset.py)
N = 10
sh_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras_pt3d,
                              raster_settings=raster_pt3d),
    shader=optimize_sh.Pt3dSHShader(device=device),
)
albedo_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras_pt3d,
                              raster_settings=raster_pt3d),
    shader=optimize_sh.Pt3dAlbedoShader(),
)

test_images, gt_sh_list = [], []
for _ in range(N):
    sh = torch.zeros(9, 3, device=device)
    # sh[0] = torch.rand(3, device=device) * 1.5 + 1.5
    sh[0] = 3
    sh[1:4] = torch.randn(3, 3, device=device) * 0.5
    sh[4:] = torch.randn(5, 3, device=device) * 0.2
    gt_sh_list.append(sh)
    with torch.no_grad():
        img = sh_renderer(mesh_gt, sh_coeffs=sh)
        test_images.append(img[0, :, :, :3].cpu().numpy())

print(f"Generated {N} images  shape={test_images[0].shape}  min: {np.asarray(test_images).min()}    max: {np.asarray(test_images).max()}")

for i, img in enumerate(test_images):
    plt.imsave(os.path.join(OUT_DIR, f"input_{i:03d}.png"), img.clip(0, 1))

# ── Recover using the pytorch3d-based decomposition ──────────────────────────
mesh_opt = join_meshes_as_scene([_sphere(p) for p in POSITIONS])

print("── decompose_pytorch3d ──────────────────────────────────────────────")
albedo_pt3d, shadings_pt3d, history_pt3d = optimize_sh.decompose_pytorch3d(
    test_images, mesh_opt, cameras_pt3d, raster_pt3d,
    n_iter=2000, lr=5e-3, lambda_white=0,
)

# ── Ground-truth albedo image (via albedo renderer) ───────────────────────────
with torch.no_grad():
    gt_albedo_img = albedo_renderer(mesh_gt)[0, :, :, :3].cpu().numpy()

plt.imsave(os.path.join(OUT_DIR, "albedo_gt.png"),    gt_albedo_img.clip(0, 1))
plt.imsave(os.path.join(OUT_DIR, "albedo_recovered.png"),
           albedo_pt3d.clip(0, 1))

for i, shading in enumerate(shadings_pt3d):
    plt.imsave(os.path.join(
        OUT_DIR, f"shading_{i:03d}.png"), shading.clip(0, 1))

# ── Visualise ─────────────────────────────────────────────────────────────────
n_show = 3
fig, axes = plt.subplots(n_show + 1, 4, figsize=(10, 4 * (n_show + 1)))

axes[0, 0].imshow(gt_albedo_img)
axes[0, 0].set_title("GT albedo")
axes[0, 0].axis("off")
axes[0, 1].imshow(albedo_pt3d)
axes[0, 1].set_title("Recovered albedo (pt3d)")
axes[0, 1].axis("off")
diff = np.abs(gt_albedo_img - albedo_pt3d)
axes[0, 2].imshow(diff.clip(0, 0.3) / 0.3)
axes[0, 2].set_title(f"|diff|  MAE={diff.mean():.4f}")
axes[0, 2].axis("off")

print(f"gt_albedo max: {gt_albedo_img.max()} - albedo_pt3d max: {albedo_pt3d.max()}")
for i in range(n_show):
    gt_shading = (test_images[i] / gt_albedo_img.clip(1e-9,2))
    print(f"test img {i} min: {test_images[i].min()}, max: {test_images[i].max()}")
    print(f"recon img {i} min: {(albedo_pt3d * shadings_pt3d[i]).min()}, max: {(albedo_pt3d * shadings_pt3d[i]).max()}")
    print(f"gt_shading {i} max: {gt_shading.max()},shadings_pt3d max: {shadings_pt3d[i].max()}")
    axes[i+1, 0].imshow(test_images[i])
    axes[i+1, 0].set_title(f"Input {i}")
    axes[i+1, 0].axis("off")
    axes[i+1, 1].imshow(gt_shading)
    axes[i+1, 1].set_title(f"GT shading {i} (input/albedo)")
    axes[i+1, 1].axis("off")
    axes[i+1, 2].imshow(shadings_pt3d[i])
    axes[i+1, 2].set_title(f"Recovered shading {i}")
    axes[i+1, 2].axis("off")
    axes[i+1, 3].imshow(albedo_pt3d * shadings_pt3d[i])
    axes[i+1, 3].set_title(f"Recovered Image {i}")
    axes[i+1, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison.png"),
            dpi=100, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(np.log(history_pt3d))
plt.xlabel("iteration (×200)")
plt.ylabel("log loss")
plt.title("decompose_pytorch3d — loss curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"),
            dpi=100, bbox_inches="tight")
plt.show()

print(f"Saved results to {OUT_DIR}/")

# ── Alternating optimisation ──────────────────────────────────────────────────
ALT_DIR = os.path.join(OUT_DIR, "alternating")
os.makedirs(ALT_DIR, exist_ok=True)

mesh_alt = join_meshes_as_scene([_sphere(p) for p in POSITIONS])

print("\n── decompose_pytorch3d (alternating) ───────────────────────────────")
albedo_alt, shadings_alt, history_alt = optimize_sh.decompose_pytorch3d(
    test_images, mesh_alt, cameras_pt3d, raster_pt3d,
    n_iter=10000, lr=5e-3, lambda_white=0,
    alternating=True,
)

plt.imsave(os.path.join(ALT_DIR, "albedo_recovered.png"),
           albedo_alt.clip(0, 1))
for i, shading in enumerate(shadings_alt):
    plt.imsave(os.path.join(
        ALT_DIR, f"shading_{i:03d}.png"), shading.clip(0, 1))

# ── Compare joint vs alternating ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0, 0].imshow(gt_albedo_img.clip(0, 1))
axes[0, 0].set_title("GT albedo")
axes[0, 0].axis("off")
axes[0, 1].imshow(albedo_pt3d.clip(0, 1))
axes[0, 1].set_title("Joint")
axes[0, 1].axis("off")
axes[0, 2].imshow(albedo_alt.clip(0, 1))
axes[0, 2].set_title("Alternating")
axes[0, 2].axis("off")
axes[1, 0].imshow(shadings_pt3d[0].clip(0, 1))
axes[1, 0].set_title("Joint shading 0")
axes[1, 0].axis("off")
axes[1, 1].imshow(shadings_alt[0].clip(0, 1))
axes[1, 1].set_title("Alt shading 0")
axes[1, 1].axis("off")
diff_joint = np.abs(gt_albedo_img - albedo_pt3d)
diff_alt = np.abs(gt_albedo_img - albedo_alt)
axes[1, 2].bar(["joint", "alternating"], [diff_joint.mean(), diff_alt.mean()])
axes[1, 2].set_ylabel("MAE vs GT albedo")
axes[1, 2].set_title("Albedo error")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "joint_vs_alternating.png"),
            dpi=100, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(np.log(history_pt3d), label="joint")
plt.plot(np.log(history_alt),  label="alternating")
plt.xlabel("iteration (×200)")
plt.ylabel("log loss")
plt.title("joint vs alternating — loss curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_comparison.png"),
            dpi=100, bbox_inches="tight")
plt.show()

print(f"Saved alternating results to {ALT_DIR}/")
