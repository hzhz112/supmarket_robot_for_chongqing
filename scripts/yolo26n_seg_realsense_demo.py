#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


DEFAULT_MODEL = "yolo26n-seg.pt"
DEFAULT_SERIAL = "347522072040"
WINDOW_NAME = "YOLO26n-Seg RealSense Demo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal yolo26n-seg demo on a RealSense camera.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO segmentation model path or name")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="RealSense serial number; empty uses first camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="Ultralytics device string, e.g. cpu")
    parser.add_argument("--save-dir", default=str(Path("/home/test/sup_robot/yolo_demo_outputs")))
    parser.add_argument("--headless", action="store_true", help="Run one frame and save output without opening a window")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class ids or names to keep, e.g. '39' or 'bottle' or 'bottle,cell phone'",
    )
    return parser


def resolve_class_filter(model: YOLO, classes_text: str) -> list[int] | None:
    classes_text = classes_text.strip()
    if not classes_text:
        return None
    name_map = model.names
    normalized_name_to_id = {str(name).strip().lower(): int(idx) for idx, name in name_map.items()}
    resolved: list[int] = []
    for part in classes_text.replace("，", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            resolved.append(int(token))
            continue
        key = token.lower()
        if key not in normalized_name_to_id:
            available = ", ".join(str(name) for _, name in sorted(name_map.items())[:20])
            raise ValueError(f"unknown class name: {token}. examples: {available}")
        resolved.append(normalized_name_to_id[key])
    return sorted(set(resolved))


def start_realsense(serial: str, width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


def warmup_frames(pipeline: rs.pipeline, count: int = 10) -> None:
    for _ in range(count):
        pipeline.wait_for_frames(timeout_ms=2000)


def grab_frame(pipeline: rs.pipeline, align: rs.align) -> tuple[np.ndarray, np.ndarray]:
    frames = pipeline.wait_for_frames(timeout_ms=3000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("failed to get RealSense color/depth frame")
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


def depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_clipped = np.clip(depth_image, 0, 3000).astype(np.float32)
    depth_norm = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / 3000.0)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def overlay_results(result, color_image: np.ndarray) -> np.ndarray:
    rendered = result.plot()
    if rendered is None:
        return color_image
    return rendered


def save_outputs(save_dir: Path, color_image: np.ndarray, depth_colormap: np.ndarray, overlay_image: np.ndarray) -> tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    color_path = save_dir / f"color_{stamp}.png"
    depth_path = save_dir / f"depth_{stamp}.png"
    overlay_path = save_dir / f"overlay_{stamp}.png"
    cv2.imwrite(str(color_path), color_image)
    cv2.imwrite(str(depth_path), depth_colormap)
    cv2.imwrite(str(overlay_path), overlay_image)
    return color_path, depth_path, overlay_path


def run_headless(args: argparse.Namespace) -> int:
    model = YOLO(args.model)
    classes = resolve_class_filter(model, args.classes)
    pipeline, align = start_realsense(args.serial, args.width, args.height, args.fps)
    try:
        warmup_frames(pipeline)
        color_image, depth_image = grab_frame(pipeline, align)
        depth_colormap = depth_to_colormap(depth_image)
        results = model.predict(
            color_image,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            classes=classes,
        )
        overlay_image = overlay_results(results[0], color_image)
        color_path, depth_path, overlay_path = save_outputs(Path(args.save_dir), color_image, depth_colormap, overlay_image)
        print(f"model: {args.model}")
        print(f"classes: {args.classes or 'all'}")
        print(f"saved color:   {color_path}")
        print(f"saved depth:   {depth_path}")
        print(f"saved overlay: {overlay_path}")
        boxes = 0 if results[0].boxes is None else len(results[0].boxes)
        masks = 0 if results[0].masks is None or results[0].masks.data is None else int(results[0].masks.data.shape[0])
        print(f"detections: boxes={boxes} masks={masks}")
        return 0
    finally:
        pipeline.stop()


def run_gui(args: argparse.Namespace) -> int:
    model = YOLO(args.model)
    classes = resolve_class_filter(model, args.classes)
    pipeline, align = start_realsense(args.serial, args.width, args.height, args.fps)
    try:
        warmup_frames(pipeline)
        print(f"classes={args.classes or 'all'}")
        print("按 q 退出，按 s 保存当前三张图。")
        while True:
            color_image, depth_image = grab_frame(pipeline, align)
            depth_colormap = depth_to_colormap(depth_image)
            results = model.predict(
                color_image,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
                classes=classes,
            )
            overlay_image = overlay_results(results[0], color_image)
            combined = np.hstack((overlay_image, depth_colormap))
            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                color_path, depth_path, overlay_path = save_outputs(Path(args.save_dir), color_image, depth_colormap, overlay_image)
                print(f"saved color={color_path}")
                print(f"saved depth={depth_path}")
                print(f"saved overlay={overlay_path}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    return run_headless(args) if args.headless else run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
