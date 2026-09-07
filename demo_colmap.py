#!/usr/bin/env python3
"""Export DarkVGGT's RGB-Thermal feedforward predictions for a scene directly to COLMAP format."""

import argparse
import glob
import os
import tempfile

import numpy as np
import torch
import trimesh
from PIL import Image

from darkvggt.utils.colmap import batch_np_matrix_to_pycolmap_wo_track, rescale_colmap_recons_for_crop
from darkvggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from darkvggt.utils.load_fn import load_and_preprocess_images
from darkvggt.utils.pose_enc import pose_encoding_to_extri_intri
from darkvggt.utils.thermal import preprocess_thermal
from inference import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Directory to save the COLMAP reconstruction (defaults to SCENE_DIR/colmap)")
    parser.add_argument("--vggt-checkpoint", default="checkpoints/vggt_1b.pt")
    parser.add_argument("--darkvggt-checkpoint", default="checkpoints/darkvggt.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--conf_thres_value", type=float, default=1.5, help="Confidence threshold value for point filtering"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def demo_fn(args):
    output_dir = args.output_dir or os.path.join(args.scene_dir, "colmap")
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    print(f"Using device: {device}")

    model = build_model(args.vggt_checkpoint, args.darkvggt_checkpoint, device)
    print("Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "rgb")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    thermal_dir = os.path.join(args.scene_dir, "thermal")
    thermal_path_list = sorted(glob.glob(os.path.join(thermal_dir, "*")))
    if len(thermal_path_list) == 0:
        raise ValueError(f"No thermal frames found in {thermal_dir}")

    images = load_and_preprocess_images(image_path_list).to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    with tempfile.TemporaryDirectory(prefix="darkvggt-thermal-") as staging:
        processed_thermal_paths = preprocess_thermal(thermal_path_list, staging, image_path_list)
        thermal_images = load_and_preprocess_images(processed_thermal_paths).to(device)
    print(f"Loaded {len(thermal_images)} thermal frames from {thermal_dir}")

    with torch.no_grad():
        predictions = model(images, thermal_images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    points_3d = predictions["world_points"].squeeze(0).cpu().numpy()
    point_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()

    # in the feedforward manner, we do not support shared camera
    shared_camera = False
    # in the feedforward manner, we only support PINHOLE camera
    camera_type = "PINHOLE"

    conf_thres_value = args.conf_thres_value
    max_points_for_colmap = 100000  # randomly sample 3D points

    image_size = np.array([images.shape[-1], images.shape[-2]])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = (images.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = point_conf >= conf_thres_value
    # at most writing 100000 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    original_sizes = np.array([Image.open(path).size for path in image_path_list], dtype=np.float32)
    reconstruction = rescale_colmap_recons_for_crop(
        reconstruction,
        base_image_path_list,
        original_sizes,
        canvas_width=image_size[0],
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {output_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
