import os, json, math, random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere, torus
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    TexturesVertex, BlendParams, hard_rgb_blend
)
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.structures import join_meshes_as_scene


def main():

    device = "cuda"
    out_dir = "synthetic_pytorch3d_dataset"
    os.makedirs(out_dir, exist_ok=True)

    def make_cube(device="cuda"):
        verts = torch.tensor([[
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
        ]], dtype=torch.float32, device=device)

        faces = torch.tensor([[
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ]], dtype=torch.int64, device=device)

        return Meshes(verts=verts, faces=faces)

    def random_mesh(device="cuda"):
        shape = random.choice(["sphere", "torus", "cube"])

        if shape == "sphere":
            mesh = ico_sphere(level=3, device=device)

        elif shape == "torus":
            mesh = torus(
                r=random.uniform(0.25, 0.45),
                R=random.uniform(0.75, 1.1),
                sides=32,
                rings=64,
                device=device,
            )

        else:
            mesh = make_cube(device=device)

        return mesh, shape
    
    def random_scene_mesh(device="cuda", min_objects=2, max_objects=5):
        objects = []
        object_meta = []

        num_objects = random.randint(min_objects, max_objects)

        for obj_idx in range(num_objects):
            mesh, shape = random_mesh(device=device)
            mesh, rot, scale = apply_random_transform(mesh)
            mesh, albedo = apply_random_albedo(mesh)

            # random xy translation so objects don't fully overlap
            verts = mesh.verts_padded()
            translation = torch.tensor(
                [[
                    random.uniform(-0.9, 0.9),
                    random.uniform(-0.7, 0.7),
                    random.uniform(-0.2, 0.2),
                ]],
                device=device,
                dtype=verts.dtype,
            )

            verts = verts + translation[:, None, :]
            mesh = mesh.update_padded(verts)

            objects.append(mesh)

            object_meta.append({
                "object_id": obj_idx,
                "shape": shape,
                "albedo_color": albedo.detach().cpu().tolist(),
                "scale": scale,
                "translation": translation[0].detach().cpu().tolist(),
                "rotation_matrix": rot.detach().cpu().tolist(),
            })

        scene_mesh = join_meshes_as_scene(objects)
        return scene_mesh, object_meta

    def random_rotation_matrix(device="cuda"):
        a, b, c = [random.uniform(0, 2 * math.pi) for _ in range(3)]

        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a),  math.cos(a)],
        ], dtype=torch.float32, device=device)

        Ry = torch.tensor([
            [ math.cos(b), 0, math.sin(b)],
            [0, 1, 0],
            [-math.sin(b), 0, math.cos(b)],
        ], dtype=torch.float32, device=device)

        Rz = torch.tensor([
            [math.cos(c), -math.sin(c), 0],
            [math.sin(c),  math.cos(c), 0],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        return Rz @ Ry @ Rx

    def apply_random_transform(mesh):
        verts = mesh.verts_padded()
        R = random_rotation_matrix(device=verts.device)

        scale = random.uniform(0.75, 1.15)
        verts = torch.matmul(verts, R.T) * scale

        return mesh.update_padded(verts), R, scale

    def apply_random_albedo(mesh):
        color = torch.rand(1, 1, 3, device=mesh.device) * 0.8 + 0.2
        verts_rgb = color.expand_as(mesh.verts_padded())
        mesh.textures = TexturesVertex(verts_features=verts_rgb)
        return mesh, color[0, 0]

    class HardSHShader(nn.Module):
        def __init__(self, device="cuda", blend_params=None):
            super().__init__()
            self.device = device
            self.blend_params = blend_params or BlendParams(background_color=(0, 0, 0))

        def sh_basis(self, normals):
            x = normals[..., 0]
            y = normals[..., 1]
            z = normals[..., 2]

            return torch.stack([
                0.282095 * torch.ones_like(x),
                0.488603 * y,
                0.488603 * z,
                0.488603 * x,
                1.092548 * x * y,
                1.092548 * y * z,
                0.315392 * (3.0 * z * z - 1.0),
                1.092548 * x * z,
                0.546274 * (x * x - y * y),
            ], dim=-1)

        def forward(self, fragments, meshes, **kwargs):
            sh_coeffs = kwargs["sh_coeffs"]

            faces = meshes.faces_packed()
            verts = meshes.verts_packed()

            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]

            face_normals = torch.nn.functional.normalize(
                torch.cross(v1 - v0, v2 - v0, dim=1),
                dim=1,
            )

            pix_to_face = fragments.pix_to_face
            mask = pix_to_face >= 0

            pixel_normals = torch.zeros(
                *pix_to_face.shape,
                3,
                device=verts.device,
                dtype=verts.dtype,
            )

            pixel_normals[mask] = face_normals[pix_to_face[mask]]

            basis = self.sh_basis(pixel_normals)
            lighting = torch.einsum("...b,bc->...c", basis, sh_coeffs)
            lighting = torch.clamp(lighting, min=0.0)

            albedo = meshes.sample_textures(fragments)
            colors = albedo * lighting
            colors[~mask] = 0.0

            return hard_rgb_blend(colors, fragments, self.blend_params)
        
    class HardAlbedoShader(nn.Module):
        def __init__(self):
            super().__init__()
            self.blend_params = BlendParams(background_color=(0, 0, 0))

        def forward(self, fragments, meshes, **kwargs):
            albedo = meshes.sample_textures(fragments)
            return hard_rgb_blend(albedo, fragments, self.blend_params)
        
    class HardFlatNormalShader(nn.Module):
        def __init__(self):
            super().__init__()
            self.blend_params = BlendParams(background_color=(0, 0, 0))

        def forward(self, fragments, meshes, **kwargs):
            cameras = kwargs.get("cameras", None)

            faces = meshes.faces_packed()
            verts = meshes.verts_packed()

            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]

            face_normals = torch.nn.functional.normalize(
                torch.cross(v1 - v0, v2 - v0, dim=1),
                dim=1,
            )

            pix_to_face = fragments.pix_to_face
            mask = pix_to_face >= 0

            normals = torch.zeros(
                *pix_to_face.shape, 3,
                device=verts.device, dtype=verts.dtype,
            )
            normals[mask] = face_normals[pix_to_face[mask]]

            if cameras is not None:
                # Use only the rotation matrix — no built-in axis flips
                R = cameras.R  # (1, 3, 3)  world-to-cam rotation
                normals_flat = normals.reshape(-1, 3)
                # R in PyTorch3D is row-major: v_cam = v_world @ R
                normals_flat = normals_flat @ R[0]
                # PyTorch3D camera looks down +Z, but standard camera space
                # (OpenCV / your likely expectation) looks down -Z, so flip Z
                normals_flat[..., 2] = -normals_flat[..., 2]
                normals = normals_flat.reshape_as(normals)
                normals = torch.nn.functional.normalize(normals, dim=-1)

            # Remap [-1, 1] -> [0, 1] for visualization
            colors = 0.5 * normals + 0.5
            colors[~mask] = 0.0

            return hard_rgb_blend(colors, fragments, self.blend_params)
        
        
    R, T = look_at_view_transform(2.7, 10, 20)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    rgb_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardSHShader(device=device),
    )

    albedo_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardAlbedoShader(),
    )

    normal_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardFlatNormalShader(),
    )

    def random_sh_coeffs(device="cuda", white_light=False):
        if white_light:
            sh = torch.zeros(9, 1, device=device)

            sh[0] = torch.rand(1, device=device) * 1.5 + 1.5
            sh[1:4] = torch.randn(3, 1, device=device) * 0.7
            sh[4:] = torch.randn(5, 1, device=device) * 0.25

        else:
            sh = torch.zeros(9, 3, device=device)

            sh[0] = torch.rand(3, device=device) * 1.5 + 1.5
            sh[1:4] = torch.randn(3, 3, device=device) * 0.7
            sh[4:] = torch.randn(5, 3, device=device) * 0.25

        return sh

    def save_image_tensor(path, image):
        img = image[0, ..., :3].detach().cpu()
        img = torch.clamp(img, 0.0, 1.0)
        plt.imsave(path, img.numpy())
    
    num_shapes = 10
    num_lights_per_shape = 20

    metadata = []

    for shape_idx in tqdm(range(num_shapes)):
        mesh, objects_meta = random_scene_mesh(
            device=device,
            min_objects=2,
            max_objects=5,
        )

        stem = f"{shape_idx:05d}"
        sample_dir = os.path.join(out_dir, stem)
        os.makedirs(sample_dir, exist_ok=True)

        alb_path = os.path.join(sample_dir, "albedo.png")
        nrm_path = os.path.join(sample_dir, "normal.png")

        alb = albedo_renderer(mesh)
        nrm = normal_renderer(mesh, cameras=cameras)

        save_image_tensor(alb_path, alb)
        save_image_tensor(nrm_path, nrm)

        sample_meta = {
            "id": stem,
            "folder": sample_dir,
            "albedo": alb_path,
            "normal": nrm_path,
            "objects": objects_meta,
            "illuminations": [],
        }

        for j in range(num_lights_per_shape):
            sh_coeffs = random_sh_coeffs(device=device)
            rgb = rgb_renderer(mesh, sh_coeffs=sh_coeffs)

            rgb_path = os.path.join(sample_dir, f"rgb_{j:03d}.png")
            save_image_tensor(rgb_path, rgb)

            sample_meta["illuminations"].append({
                "light_id": f"{j:03d}",
                "rgb": rgb_path,
                "sh_coeffs": sh_coeffs.detach().cpu().tolist(),
            })

        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(sample_meta, f, indent=2)

        metadata.append(sample_meta)
        
if __name__ == "__main__":
    main()