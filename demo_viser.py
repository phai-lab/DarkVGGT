#!/usr/bin/env python3
"""View DarkVGGT's predicted point cloud and camera poses for a paired RGB-T sequence."""

import argparse
import os
import time

import numpy as np
import torch
import viser
import viser.transforms as viser_tf

from darkvggt.utils.geometry import closed_form_inverse_se3
from darkvggt.utils.load_fn import load_and_preprocess_images
from darkvggt.utils.pose_enc import pose_encoding_to_extri_intri
from inference import build_model, list_images, load_thermal_images

PRED_COLOR = (220, 60, 60)


def umeyama_sim3(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean
    spread = np.sqrt((source_centred ** 2).sum())
    if spread < 1e-8:
        return None
    scale = np.sqrt((target_centred ** 2).sum()) / spread
    u, _, vt = np.linalg.svd(source_centred.T @ target_centred)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
    return scale, rotation, target_mean - scale * (rotation @ source_mean)


def sim3_apply(points, poses, transform):
    scale, rotation, translation = transform
    new_points = (scale * (rotation @ points.T).T + translation).astype(np.float32)
    new_poses = poses.copy()
    new_poses[:, :3, :3] = rotation @ poses[:, :3, :3]
    new_poses[:, :3, 3] = scale * (rotation @ poses[:, :3, 3].T).T + translation
    return new_points, new_poses


@torch.no_grad()
def build_entry(model, rgb, thermal, sources):
    predictions = model(rgb, thermal_images=thermal)
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], rgb.shape[-2:])
    world_points = predictions["world_points"].squeeze(0).cpu().numpy()

    points = world_points[..., :3].reshape(-1, 3)
    colors = {name: images.reshape(-1, 3) for name, images in sources.items()}
    keep = np.flatnonzero(np.isfinite(points).all(axis=1))
    return {
        "frames": rgb.shape[0],
        "points": points[keep],
        "colors": {name: value[keep] for name, value in colors.items()},
        "poses": closed_form_inverse_se3(extrinsic.squeeze(0).cpu().numpy()),
    }


def scene_bounds(points, poses):
    finite = np.concatenate([points, poses[:, :3, 3]])
    finite = finite[np.isfinite(finite).all(axis=1)]
    if len(finite) < 2:
        return None, None
    lower, upper = finite.min(axis=0), finite.max(axis=0)
    return (lower + upper) / 2.0, max(float(np.linalg.norm(upper - lower) * 0.5), 1e-3)


def add_cameras(server, poses, size, node, color, aspect):
    for index, cam_to_world in enumerate(poses):
        server.scene.add_camera_frustum(
            f"{node}/{index:04d}", fov=1.0, aspect=aspect, scale=size,
            line_width=2.0, color=color,
            wxyz=viser_tf.SO3.from_matrix(cam_to_world[:3, :3]).wxyz,
            position=cam_to_world[:3, 3],
        )


def create_viewer(entries, aspect, args):
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    if args.share:
        server.request_share_url()
    sources = list(entries["DarkVGGT"]["colors"])

    with server.gui.add_folder("Point Cloud"):
        gui_model = (server.gui.add_dropdown("Model", options=list(entries),
                                             initial_value="DarkVGGT")
                     if len(entries) > 1 else None)
        gui_color = server.gui.add_dropdown("Color Mode", options=sources,
                                           initial_value=sources[0])
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.05,
                                               step=0.001, initial_value=args.point_size)

    with server.gui.add_folder("Cameras"):
        gui_show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
        gui_camera_size = server.gui.add_slider("Camera Size", min=0.005, max=0.2,
                                                step=0.005, initial_value=args.camera_size)

    def current_model():
        return gui_model.value if gui_model is not None else "DarkVGGT"

    def current_entry():
        return entries[current_model()]

    def sync_color_options():
        options = ["RGB"] if current_model() == "VGGT-1B" else sources
        gui_color.options = options
        if gui_color.value not in options:
            gui_color.value = options[0]

    def update_points():
        server.scene.remove_by_name("/points")
        entry = current_entry()
        server.scene.add_point_cloud(
            "/points", points=entry["points"], colors=entry["colors"][gui_color.value],
            point_size=gui_point_size.value, point_shape="rounded")

    def update_cameras():
        server.scene.remove_by_name("/cameras")
        if not gui_show_cameras.value:
            return
        add_cameras(server, current_entry()["poses"], gui_camera_size.value,
                    "/cameras/pred", PRED_COLOR, aspect)

    def refit():
        entry = current_entry()
        center, radius = scene_bounds(entry["points"], entry["poses"])
        if center is None:
            return
        direction = np.array([1.0, 1.0, 0.5])
        direction /= np.linalg.norm(direction)
        for client in server.get_clients().values():
            client.camera.position = tuple((center + direction * radius * 2.5).astype(np.float32))
            client.camera.look_at = tuple(center.astype(np.float32))
            client.camera.up = (0.0, 1.0, 0.0)

    def on_switch(_):
        sync_color_options()
        update_points()
        update_cameras()

    if gui_model is not None:
        gui_model.on_update(on_switch)
    for control in (gui_color, gui_point_size):
        control.on_update(lambda _: update_points())
    for control in (gui_show_cameras, gui_camera_size):
        control.on_update(lambda _: update_cameras())

    @server.on_client_connect
    def _(client):
        refit()

    on_switch(None)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", default="samples",
                         help="folder holding an rgb/ subfolder and, optionally, a thermal/ one")
    parser.add_argument("--vggt-checkpoint", default="checkpoints/vggt_1b.pt")
    parser.add_argument("--darkvggt-checkpoint", default="checkpoints/darkvggt.pt")
    parser.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                         help="also run the RGB-only VGGT-1B baseline, selectable from the "
                              "viewer's Model dropdown")
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--camera-size", type=float, default=0.025)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true",
                         help="expose a public viser.studio share URL")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rgb_dir = os.path.join(args.scene_dir, "rgb")
    thermal_dir = os.path.join(args.scene_dir, "thermal")

    rgb_paths = list_images(rgb_dir)
    rgb = load_and_preprocess_images(rgb_paths).to(args.device)
    sources = {"RGB": (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)}

    thermal = None
    if os.path.isdir(thermal_dir):
        thermal = load_thermal_images(thermal_dir, rgb_paths).to(args.device)
        sources["Thermal"] = (thermal.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    def run(darkvggt_checkpoint, thermal_images):
        model = build_model(args.vggt_checkpoint, darkvggt_checkpoint, args.device)
        entry = build_entry(model, rgb, thermal_images, sources)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return entry

    entries = {"DarkVGGT": run(args.darkvggt_checkpoint, thermal)}
    if args.compare:
        entries["VGGT-1B"] = run(None, None)
        reference = entries["DarkVGGT"]["poses"][:, :3, 3]
        for name, entry in entries.items():
            if name == "DarkVGGT":
                continue
            transform = umeyama_sim3(entry["poses"][:, :3, 3], reference)
            if transform is None:
                print(f"{name} trajectory has no spread, skipping alignment to DarkVGGT")
                continue
            entry["points"], entry["poses"] = sim3_apply(
                entry["points"], entry["poses"], transform)

    create_viewer(entries, rgb.shape[-1] / rgb.shape[-2], args)
    print(f"open http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
