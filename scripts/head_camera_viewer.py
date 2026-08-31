#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_SERIAL = "347522072040"
WINDOW_NAME = "Head Camera Viewer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the head RealSense camera color and depth.")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="RealSense serial number; empty uses first camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-depth-mm", type=float, default=3000.0, help="Depth colormap range in millimeters")
    parser.add_argument("--save-dir", default=str(Path("/home/test/sup_robot/head_camera_viewer_outputs")))
    return parser


def start_realsense(serial: str, width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align, float, str]:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    return pipeline, align, depth_scale, profile.get_device().get_info(rs.camera_info.serial_number)


def warmup_frames(pipeline: rs.pipeline, count: int = 10) -> None:
    for _ in range(count):
        pipeline.wait_for_frames(timeout_ms=2000)


def grab_frames(pipeline: rs.pipeline, align: rs.align) -> tuple[np.ndarray, np.ndarray]:
    frames = pipeline.wait_for_frames(timeout_ms=3000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("failed to get RealSense color/depth frame")
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


def depth_to_colormap(depth_image: np.ndarray, max_depth_mm: float) -> np.ndarray:
    depth_clipped = np.clip(depth_image, 0, max_depth_mm).astype(np.float32)
    depth_norm = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / max(max_depth_mm, 1.0))
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def save_frame(save_dir: Path, color: np.ndarray, depth: np.ndarray, depth_vis: np.ndarray) -> tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    color_path = save_dir / f"color_{stamp}.png"
    depth_path = save_dir / f"depth_{stamp}.png"
    depth_vis_path = save_dir / f"depth_vis_{stamp}.png"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(depth_path), depth)
    cv2.imwrite(str(depth_vis_path), depth_vis)
    return color_path, depth_path, depth_vis_path


def draw_status(frame: np.ndarray, serial: str, depth_scale: float, fps: float, depth_value_mm: float | None) -> np.ndarray:
    view = frame.copy()
    lines = [
        f"serial={serial or 'auto'}",
        f"depth_scale={depth_scale:.6f} m/unit",
        f"fps={fps:.1f}",
    ]
    if depth_value_mm is not None:
        lines.append(f"center_depth={depth_value_mm:.1f} mm")
    y = 24
    for line in lines:
        cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return view


def run_gui(args: argparse.Namespace) -> int:
    pipeline = None
    try:
        pipeline, align, depth_scale, serial = start_realsense(args.serial, args.width, args.height, args.fps)
        warmup_frames(pipeline)
        print(f"RealSense started: serial={serial}, {args.width}x{args.height}@{args.fps}")
        print("按 q 退出，按 s 保存当前帧。")

        last_t = time.time()
        fps = 0.0
        while True:
            color_image, depth_image = grab_frames(pipeline, align)
            now = time.time()
            dt = max(now - last_t, 1e-6)
            fps = 1.0 / dt
            last_t = now

            depth_vis = depth_to_colormap(depth_image, args.max_depth_mm)
            center_depth_mm = float(depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2])

            color_panel = draw_status(color_image, serial, depth_scale, fps, center_depth_mm)
            depth_panel = draw_status(depth_vis, serial, depth_scale, fps, center_depth_mm)
            combined = np.hstack((color_panel, depth_panel))
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                color_path, depth_path, depth_vis_path = save_frame(Path(args.save_dir), color_image, depth_image, depth_vis)
                print(f"saved color:    {color_path}")
                print(f"saved depth:    {depth_path}")
                print(f"saved depth vis: {depth_vis_path}")
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
