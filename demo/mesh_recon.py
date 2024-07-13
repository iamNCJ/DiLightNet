import tempfile
from typing import Optional

import numpy as np
import cv2
import torch
import trimesh
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device).eval()


import torchvision.transforms as tvf
import PIL.Image
import numpy as np
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_single_image(img_array):
    imgs = []
    for i in range(2):
        img = PIL.Image.fromarray(img_array)
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=i, instance=str(len(imgs))))

    return imgs


def get_intrinsics(H, W, fov=55.):
    """
    Intrinsics for a pinhole camera model.
    Assume central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def depth_to_points(depth, R=None, t=None, fov=55.):
    K = get_intrinsics(depth.shape[1], depth.shape[2], fov=fov)
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]


def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def mesh_reconstruction(
        masked_image: np.ndarray,
        mask: np.ndarray,
        remove_edges: bool = True,
        fov: Optional[float] = None,
        mask_threshold: float = 25.,
):
    masked_image = cv2.resize(masked_image, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    images = load_single_image(masked_image)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    if fov is not None:
        # do not optimize focal length if fov is provided
        focal = scene.imshapes[0][1] / (2 * np.tan(0.5 * fov * np.pi / 180.))
        scene.preset_focal([focal, focal])
    _loss = scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    if fov is None:
        # get the focal length from the optimized parameters
        focals = scene.get_focals()
        fov = 2 * (np.arctan((scene.imshapes[0][1] / (focals[0] + focals[1])).detach().cpu().numpy()) * 180 / np.pi)[0]
    depth = scene.get_depthmaps()[0].detach().cpu().numpy()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    rgb = masked_image[..., :3].transpose(2, 0, 1) / 255.

    pts3d = depth_to_points(depth[None], fov=fov)
    pts3d = pts3d.reshape(-1, 3)
    pts3d = pts3d.reshape(-1, 3)
    verts = pts3d.reshape(-1, 3)
    rgb = rgb.transpose(1, 2, 0)
    mask = mask[..., 0] > mask_threshold
    edge_mask = depth_edges_mask(depth)
    if remove_edges:
        mask = np.logical_and(mask, ~edge_mask)
    triangles = create_triangles(rgb.shape[0], rgb.shape[1], mask=mask)
    colors = rgb.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    # Save as glb tmp file (obj will look inverted in ui)
    mesh_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    mesh_file_path = mesh_file.name
    mesh.export(mesh_file_path)
    return mesh_file_path, fov
