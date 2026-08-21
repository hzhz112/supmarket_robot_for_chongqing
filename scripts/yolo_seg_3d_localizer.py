#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from ultralytics import YOLO


DEFAULT_INTRINSICS_JSON = "/home/test/linkhand/realsense_camera_intrinsics_visual.json"
DEFAULT_CAMERA_SERIAL = "347522072040"
WINDOW_NAME = "YOLO-Seg 3D Localizer"


def resolve_class_filter(model: YOLO, class_text: str) -> list[int] | None:
    text = str(class_text).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    mapping = {str(name).strip().lower(): int(index) for index, name in model.names.items()}
    resolved: list[int] = []
    for part in text.replace("，", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            resolved.append(int(token))
            continue
        key = token.lower()
        if key not in mapping:
            raise ValueError(f"unknown class name: {token}")
        resolved.append(mapping[key])
    return sorted(set(resolved))


def ensure_mask_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    # Keep nearest-neighbor semantics so the segmentation mask maps back to the original image grid.
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def depth_to_colormap(depth_m: np.ndarray, max_depth_m: float = 2.0) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, max_depth_m)
    gray = cv2.convertScaleAbs(clipped, alpha=255.0 / max(max_depth_m, 1e-6))
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def mask_to_pointcloud(mask: np.ndarray, depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_idx, u_idx = np.nonzero(mask)
    z = depth_m[v_idx, u_idx]
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z)), u_idx, v_idx


def robust_extent(points: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.percentile(points, low, axis=0)
    upper = np.percentile(points, high, axis=0)
    return lower, upper, upper - lower


def save_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


class YoloSeg3DLocalizer(Node):
    def __init__(self) -> None:
        super().__init__("yolo_seg_3d_localizer")
        self.declare_parameter("model", "yolo26n-seg.pt")
        self.declare_parameter("target_class", "bottle")
        self.declare_parameter("serial", DEFAULT_CAMERA_SERIAL)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 15)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("depth_band_m", 0.05)
        self.declare_parameter("mask_erode_kernel", 5)
        self.declare_parameter("headless", False)
        self.declare_parameter("save_dir", "/home/test/sup_robot/yolo_3d_outputs")
        self.declare_parameter("intrinsics_json", DEFAULT_INTRINSICS_JSON)
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")

        self.model_path = self.get_parameter("model").value
        self.target_class = self.get_parameter("target_class").value
        self.serial = self.get_parameter("serial").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_band_m = float(self.get_parameter("depth_band_m").value)
        self.mask_erode_kernel = max(1, int(self.get_parameter("mask_erode_kernel").value))
        self.headless = bool(self.get_parameter("headless").value)
        self.save_dir = Path(str(self.get_parameter("save_dir").value))
        self.intrinsics_json = Path(str(self.get_parameter("intrinsics_json").value))
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value)

        self.model = YOLO(self.model_path)
        self.class_filter = resolve_class_filter(self.model, self.target_class)
        self.get_logger().info(
            f"model={self.model_path} target_class={self.target_class or 'all'} "
            f"class_filter={self.class_filter}"
        )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.serial:
            self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.fx, self.fy, self.cx, self.cy = self._load_intrinsics()
        self.get_logger().info(
            f"intrinsics fx={self.fx:.6f} fy={self.fy:.6f} cx={self.cx:.6f} cy={self.cy:.6f} "
            f"frame_id={self.camera_frame_id}"
        )

    def _load_intrinsics(self) -> tuple[float, float, float, float]:
        if self.intrinsics_json.exists():
            try:
                payload = json.loads(self.intrinsics_json.read_text(encoding="utf-8"))
                json_width = int(payload.get("width", -1))
                json_height = int(payload.get("height", -1))
                if json_width == self.width and json_height == self.height:
                    self.get_logger().info(f"use intrinsics json: {self.intrinsics_json}")
                    return (
                        float(payload["fx"]),
                        float(payload["fy"]),
                        float(payload["ppx"]),
                        float(payload["ppy"]),
                    )
                self.get_logger().warn(
                    f"intrinsics json size {json_width}x{json_height} != current {self.width}x{self.height}, fallback to live intrinsics"
                )
            except Exception as exc:
                self.get_logger().warn(f"read intrinsics json failed: {exc}, fallback to live intrinsics")
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        return float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    def warmup(self, count: int = 10) -> None:
        for _ in range(count):
            self.pipeline.wait_for_frames(timeout_ms=2000)

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("failed to get aligned color/depth frames")
        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
        return color_image, depth_m

    def select_target(self, result) -> tuple[int, float, int]:
        if result.boxes is None or result.masks is None or result.masks.data is None:
            raise RuntimeError("no segmentation result")
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        if len(confs) == 0:
            raise RuntimeError("no detections")
        best_index = int(np.argmax(confs))
        return best_index, float(confs[best_index]), int(classes[best_index])

    def localize_once(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        color_image, depth_m = self.capture()
        results = self.model.predict(
            color_image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.class_filter,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no results")
        result = results[0]
        target_index, confidence, class_id = self.select_target(result)
        class_name = str(self.model.names[class_id])

        raw_mask = result.masks.data[target_index].cpu().numpy() > 0.5
        mask = ensure_mask_shape(raw_mask, color_image.shape[1], color_image.shape[0])

        kernel_size = self.mask_erode_kernel
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        else:
            inner_mask = mask.copy()

        valid_mask = (
            inner_mask
            & np.isfinite(depth_m)
            & (depth_m > self.min_depth_m)
            & (depth_m < self.max_depth_m)
        )
        valid_depths = depth_m[valid_mask]
        if valid_depths.size == 0:
            raise RuntimeError("no valid depth points inside eroded mask")

        median_depth = float(np.median(valid_depths))
        band_mask = valid_mask & (np.abs(depth_m - median_depth) < self.depth_band_m)
        if int(np.count_nonzero(band_mask)) < 20:
            self.get_logger().warn("depth band removed too many pixels, fallback to valid_mask")
            band_mask = valid_mask

        points, u_idx, v_idx = mask_to_pointcloud(band_mask, depth_m, self.fx, self.fy, self.cx, self.cy)
        if points.size == 0:
            raise RuntimeError("point cloud is empty after projection")

        center = np.median(points, axis=0)
        lower, upper, size = robust_extent(points, 2.0, 98.0)

        overlay = color_image.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 255, 180], dtype=np.float32)).astype(np.uint8)
        overlay[band_mask] = (0.25 * overlay[band_mask] + 0.75 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
        cv2.circle(overlay, (int(np.median(u_idx)), int(np.median(v_idx))), 6, (255, 255, 255), -1)
        cv2.putText(
            overlay,
            f"{class_name} {confidence:.2f} center=({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f})m",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        stats = {
            "class": class_name,
            "confidence": confidence,
            "frame_id": self.camera_frame_id,
            "center_camera": {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            },
            "size": {
                "width": float(size[0]),
                "height": float(size[1]),
                "depth": float(size[2]),
            },
            "bounds_camera": {
                "xmin": float(lower[0]),
                "xmax": float(upper[0]),
                "ymin": float(lower[1]),
                "ymax": float(upper[1]),
                "zmin": float(lower[2]),
                "zmax": float(upper[2]),
            },
            "valid_depth_pixels": int(np.count_nonzero(band_mask)),
            "median_depth_m": median_depth,
        }
        return stats, points, overlay, color_image, depth_m

    def save_outputs(self, stats: dict, points: np.ndarray, overlay: np.ndarray, color_image: np.ndarray, depth_m: np.ndarray) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stats_path = self.save_dir / f"target_stats_{stamp}.json"
        points_path = self.save_dir / f"target_points_{stamp}.npy"
        ply_path = self.save_dir / f"target_points_{stamp}.ply"
        overlay_path = self.save_dir / f"overlay_{stamp}.png"
        color_path = self.save_dir / f"color_{stamp}.png"
        depth_path = self.save_dir / f"depth_{stamp}.png"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(points_path, points)
        save_ply(ply_path, points)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(color_path), color_image)
        cv2.imwrite(str(depth_path), depth_to_colormap(depth_m, self.max_depth_m))
        self.get_logger().info(
            f"saved stats={stats_path} points={points_path} ply={ply_path} overlay={overlay_path}"
        )

    def log_stats(self, stats: dict) -> None:
        self.get_logger().info(
            "\n"
            f"class: {stats['class']}\n"
            f"confidence: {stats['confidence']:.2f}\n"
            f"frame_id: {stats['frame_id']}\n"
            f"center_camera:\n"
            f"  x: {stats['center_camera']['x']:+.4f} m\n"
            f"  y: {stats['center_camera']['y']:+.4f} m\n"
            f"  z: {stats['center_camera']['z']:+.4f} m\n"
            f"size:\n"
            f"  width: {stats['size']['width']:+.4f} m\n"
            f"  height: {stats['size']['height']:+.4f} m\n"
            f"  depth: {stats['size']['depth']:+.4f} m\n"
            f"valid_depth_pixels: {stats['valid_depth_pixels']}\n"
            f"median_depth_m: {stats['median_depth_m']:+.4f}"
        )

    def run_headless(self) -> int:
        self.warmup()
        stats, points, overlay, color_image, depth_m = self.localize_once()
        self.log_stats(stats)
        self.save_outputs(stats, points, overlay, color_image, depth_m)
        return 0

    def run_gui(self) -> int:
        self.warmup()
        self.get_logger().info("press q to quit, s to save current frame result")
        while rclpy.ok():
            try:
                stats, points, overlay, color_image, depth_m = self.localize_once()
                depth_vis = depth_to_colormap(depth_m, self.max_depth_m)
                combined = np.hstack((overlay, depth_vis))
                cv2.imshow(WINDOW_NAME, combined)
                self.log_stats(stats)
            except Exception as exc:
                blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(blank, f"{exc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(WINDOW_NAME, blank)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                try:
                    self.save_outputs(stats, points, overlay, color_image, depth_m)
                except Exception as exc:
                    self.get_logger().warn(f"save failed: {exc}")
        return 0

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    node = YoloSeg3DLocalizer()
    try:
        return node.run_headless() if node.headless else node.run_gui()
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
