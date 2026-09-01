from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_sup_robot")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets


BOX_LENGTH_M = 0.48
BOX_WIDTH_M = 0.35
BOX_HEIGHT_M = 0.17
BOX_CLASS_ALIASES = {"box", "xiangzi", "carton", "crate", "boxy"}
BOX_READY_LEFT_ARM_DEG = (15.8, 22.5, -19.3, -100.4, -22.7, -12.4, 6.9)
BOX_READY_RIGHT_ARM_DEG = (-15.7, -14.6, 16.0, 103.1, 16.0, 15.0, -1.7)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("zero-length vector")
    return vec / norm


def _vec_text(vec: np.ndarray, digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):+.{digits}f}" for v in vec) + "]"


def _matrix_text(mat: np.ndarray, digits: int = 4) -> str:
    rows = []
    for row in np.asarray(mat, dtype=np.float64):
        rows.append("[" + ", ".join(f"{float(v):+.{digits}f}" for v in row) + "]")
    return "\n".join(rows)


def _ensure_points_array(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("point cloud must be N x 3")
    return arr


def transform_points(points_camera: np.ndarray, T_base_camera: np.ndarray) -> np.ndarray:
    points_camera = _ensure_points_array(points_camera)
    T_base_camera = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    points_h = np.column_stack((points_camera, np.ones((len(points_camera), 1), dtype=np.float64)))
    return (T_base_camera @ points_h.T).T[:, :3]


def mask_depth_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
    *,
    erode_kernel: int = 5,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.00,
    depth_band_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    height, width = depth_m.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

    kernel = np.ones((max(1, int(erode_kernel)), max(1, int(erode_kernel))), dtype=np.uint8)
    inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    fx, fy, cx, cy = camera_intrinsics
    valid = (
        inner_mask
        & np.isfinite(depth_m)
        & (depth_m > float(min_depth_m))
        & (depth_m < float(max_depth_m))
    )
    if int(np.count_nonzero(valid)) < 20:
        raise ValueError("mask depth points are too sparse")
    median_depth = float(np.median(depth_m[valid]))
    valid &= np.abs(depth_m - median_depth) < float(depth_band_m)
    if int(np.count_nonzero(valid)) < 20:
        valid = inner_mask & np.isfinite(depth_m) & (depth_m > float(min_depth_m)) & (depth_m < float(max_depth_m))

    v_idx, u_idx = np.nonzero(valid)
    z = depth_m[v_idx, u_idx].astype(np.float64)
    points = np.column_stack(
        (
            (u_idx.astype(np.float64) - float(cx)) * z / float(fx),
            (v_idx.astype(np.float64) - float(cy)) * z / float(fy),
            z,
        )
    )
    if len(points) == 0:
        raise ValueError("point cloud is empty after projection")
    return points, np.column_stack((u_idx, v_idx))


@dataclass
class BoxBimanualGraspSolution:
    T_base_box: np.ndarray
    front_points: np.ndarray
    front_plane_normal: np.ndarray
    debug_info: dict[str, Any]
    candidates: dict[str, dict[str, np.ndarray]]
    assignment: dict[str, str]


def estimate_front_plane(
    points_base: np.ndarray,
    z_base: np.ndarray | None = None,
    *,
    preferred_face_span_m: float = BOX_LENGTH_M,
    preferred_height_m: float = BOX_HEIGHT_M,
    distance_threshold: float = 0.012,
    iterations: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = _ensure_points_array(points_base)
    if len(points) < 30:
        raise ValueError("box point cloud is too small")
    z_base = _normalize(np.asarray(z_base if z_base is not None else [0.0, 0.0, 1.0], dtype=np.float64))
    rng = np.random.default_rng(42)
    best: Optional[tuple[float, np.ndarray, float, np.ndarray]] = None
    sample_count = min(int(iterations), max(64, len(points) // 4))

    def plane_spans(inlier_points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
        normal = _normalize(np.asarray(normal, dtype=np.float64).reshape(3))
        horizontal = np.cross(z_base, normal)
        if float(np.linalg.norm(horizontal)) < 1e-9:
            return 0.0, 0.0
        horizontal = _normalize(horizontal)
        vertical = z_base
        h_span = float(np.percentile(inlier_points @ horizontal, 95.0) - np.percentile(inlier_points @ horizontal, 5.0))
        v_span = float(np.percentile(inlier_points @ vertical, 95.0) - np.percentile(inlier_points @ vertical, 5.0))
        return abs(h_span), abs(v_span)

    for _ in range(sample_count):
        idx = rng.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        verticality = abs(float(np.dot(normal, z_base)))
        if verticality > 0.35:
            continue
        d = -float(np.dot(normal, p1))
        distances = np.abs(points @ normal + d)
        inliers = distances < float(distance_threshold)
        count = int(np.count_nonzero(inliers))
        if count < 25:
            continue
        h_span, v_span = plane_spans(points[inliers], normal)
        if h_span <= 0.0 or v_span <= 0.0:
            continue
        # The requested 48 cm front face is the primary constraint. A side
        # plane can contain more pixels, so point count must not dominate the
        # known-size match.
        face_error = abs(h_span - float(preferred_face_span_m)) / max(float(preferred_face_span_m), 1e-6)
        height_error = abs(v_span - float(preferred_height_m)) / max(float(preferred_height_m), 1e-6)
        dimension_error = face_error + height_error
        score = -100.0 * dimension_error - 0.05 * verticality + 0.0001 * float(count)
        if best is None or score > best[0]:
            best = (score, normal, d, inliers)

    if best is None:
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if abs(float(np.dot(normal, z_base))) > 0.35:
            normal = vh[1]
        d = -float(np.dot(normal, np.median(points, axis=0)))
        distances = np.abs(points @ normal + d)
        inliers = distances < float(distance_threshold)
    else:
        _, normal, d, inliers = best

    front_points = points[inliers]
    if len(front_points) < 20:
        raise ValueError("failed to isolate the front plane")
    front_h_span, front_v_span = plane_spans(front_points, normal)
    debug = {
        "plane_d": float(d),
        "inlier_count": int(len(front_points)),
        "inlier_ratio": float(len(front_points) / len(points)),
        "normal_norm": float(np.linalg.norm(normal)),
        "front_h_span_m": float(front_h_span),
        "front_v_span_m": float(front_v_span),
        "preferred_face_span_m": float(preferred_face_span_m),
    }
    return front_points, _normalize(np.asarray(normal, dtype=np.float64)), debug


def estimate_box_frame(
    front_points: np.ndarray,
    front_normal: np.ndarray,
    camera_pos_base: np.ndarray,
    *,
    box_length_m: float = BOX_LENGTH_M,
    box_width_m: float = BOX_WIDTH_M,
    box_height_m: float = BOX_HEIGHT_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = _ensure_points_array(front_points)
    camera_pos_base = np.asarray(camera_pos_base, dtype=np.float64).reshape(3)
    z_box = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    front_center = np.median(points, axis=0)
    front_normal = _normalize(np.asarray(front_normal, dtype=np.float64).reshape(3))
    to_camera = _normalize(camera_pos_base - front_center)

    # front_normal is outward, so +X_box must point inward.
    if float(np.dot(front_normal, to_camera)) >= 0.0:
        x_box = -front_normal
    else:
        x_box = front_normal
    y_box = _normalize(np.cross(z_box, x_box))
    x_box = _normalize(np.cross(y_box, z_box))
    R_base_box = np.column_stack((x_box, y_box, z_box))
    if float(np.linalg.det(R_base_box)) < 0.0:
        y_box = -y_box
        R_base_box = np.column_stack((x_box, y_box, z_box))

    s_x = points @ x_box
    s_y = points @ y_box
    s_z = points @ z_box
    x_front = float(np.median(s_x))
    y_mid = float(0.5 * (np.percentile(s_y, 5.0) + np.percentile(s_y, 95.0)))
    z_top = float(np.percentile(s_z, 95.0))
    p_front_top = x_front * x_box + y_mid * y_box + z_top * z_box
    # The robot-facing side is the long 48 cm face. Keep that as the visible
    # face in visualization and grasp generation instead of swapping by span.
    box_face_span_m = float(max(box_length_m, box_width_m))
    box_depth_m = float(min(box_length_m, box_width_m))
    face_is_long_side = True
    box_center = p_front_top + 0.5 * box_depth_m * x_box - 0.5 * float(box_height_m) * z_box

    T_base_box = np.eye(4, dtype=np.float64)
    T_base_box[:3, :3] = R_base_box
    T_base_box[:3, 3] = box_center

    debug = {
        "front_center_base": front_center,
        "camera_pos_base": camera_pos_base,
        "x_box_base": x_box,
        "y_box_base": y_box,
        "z_box_base": z_box,
        "front_normal_base": front_normal,
        "to_camera_base": to_camera,
        "x_front": x_front,
        "y_mid": y_mid,
        "z_top": z_top,
        "face_span_m": box_face_span_m,
        "box_depth_m": box_depth_m,
        "face_is_long_side": face_is_long_side,
        "measured_face_span_m": float(np.percentile(s_y, 95.0) - np.percentile(s_y, 5.0)),
        "box_center_base": box_center,
        "estimated_spans": {
            "x": float(np.percentile(s_x, 95.0) - np.percentile(s_x, 5.0)),
            "y": float(np.percentile(s_y, 95.0) - np.percentile(s_y, 5.0)),
            "z": float(np.percentile(s_z, 95.0) - np.percentile(s_z, 5.0)),
        },
        "determinant": float(np.linalg.det(R_base_box)),
        "camera_alignment": float(np.dot(x_box, to_camera)),
        "front_center_distance": float(np.linalg.norm(front_center - camera_pos_base)),
        "p_front_top_base": p_front_top,
    }
    return T_base_box, debug


def generate_side_grasp_candidates(
    T_base_box: np.ndarray,
    *,
    face_span_m: float = max(BOX_LENGTH_M, BOX_WIDTH_M),
    box_depth_m: float = min(BOX_LENGTH_M, BOX_WIDTH_M),
    pregrasp_dist_m: float = 0.08,
    grasp_z_offset_m: float = 0.0,
    grasp_x_offset_m: float = 0.0,
    front_center_base: np.ndarray | None = None,
    box_center_base: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    T_base_box = np.asarray(T_base_box, dtype=np.float64).reshape(4, 4)
    R = T_base_box[:3, :3]
    center = T_base_box[:3, 3]
    x_box, y_box, z_box = R[:, 0], R[:, 1], R[:, 2]
    face_span_m = float(face_span_m)
    if front_center_base is not None and box_center_base is not None:
        front_center_base = np.asarray(front_center_base, dtype=np.float64).reshape(3)
        box_center_base = np.asarray(box_center_base, dtype=np.float64).reshape(3)
        grasp_x = float(front_center_base[0])
        grasp_z = float(box_center_base[2] + grasp_z_offset_m)
        grasp_y_offset = 0.5 * face_span_m
        left_grasp_base = np.array([grasp_x, box_center_base[1] + grasp_y_offset, grasp_z], dtype=np.float64)
        right_grasp_base = np.array([grasp_x, box_center_base[1] - grasp_y_offset, grasp_z], dtype=np.float64)
        left_pre_base = left_grasp_base + np.array([0.0, +float(pregrasp_dist_m), 0.0], dtype=np.float64)
        right_pre_base = right_grasp_base + np.array([0.0, -float(pregrasp_dist_m), 0.0], dtype=np.float64)
        use_direct_base = True
    else:
        grasp_x = float(grasp_x_offset_m)
        grasp_z = float(grasp_z_offset_m)
        box_depth_m = float(box_depth_m)
        left_grasp_box = np.array([grasp_x, +0.5 * face_span_m, grasp_z], dtype=np.float64)
        right_grasp_box = np.array([grasp_x, -0.5 * face_span_m, grasp_z], dtype=np.float64)
        left_pre_box = left_grasp_box + np.array([0.0, +float(pregrasp_dist_m), 0.0], dtype=np.float64)
        right_pre_box = right_grasp_box + np.array([0.0, -float(pregrasp_dist_m), 0.0], dtype=np.float64)
        use_direct_base = False

    def to_base(point_box: np.ndarray) -> np.ndarray:
        return center + point_box[0] * x_box + point_box[1] * y_box + point_box[2] * z_box

    candidates = {
        "left": {
            "grasp_box": left_grasp_base if use_direct_base else left_grasp_box,
            "pregrasp_box": left_pre_base if use_direct_base else left_pre_box,
            "approach_box": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        },
        "right": {
            "grasp_box": right_grasp_base if use_direct_base else right_grasp_box,
            "pregrasp_box": right_pre_base if use_direct_base else right_pre_box,
            "approach_box": np.array([0.0, +1.0, 0.0], dtype=np.float64),
        },
    }
    for side, item in candidates.items():
        grasp_base = item["grasp_box"] if use_direct_base else to_base(item["grasp_box"])
        pregrasp_base = item["pregrasp_box"] if use_direct_base else to_base(item["pregrasp_box"])
        approach_base = _normalize(R @ item["approach_box"])
        rotation = R.copy()
        item.update(
            {
                "grasp_base": grasp_base,
                "pregrasp_base": pregrasp_base,
                "approach_base": approach_base,
                "rotation_base": rotation,
                "pose_grasp_base": np.array(
                    [
                        float(grasp_base[0]),
                        float(grasp_base[1]),
                        float(grasp_base[2]),
                        *rotation_matrix_to_rpy(rotation),
                    ],
                    dtype=np.float64,
                ),
                "pose_pregrasp_base": np.array(
                    [
                        float(pregrasp_base[0]),
                        float(pregrasp_base[1]),
                        float(pregrasp_base[2]),
                        *rotation_matrix_to_rpy(rotation),
                    ],
                    dtype=np.float64,
                ),
            }
        )
    return candidates


def assign_candidates_to_arms(
    candidates: dict[str, dict[str, np.ndarray]],
    left_tcp: Optional[np.ndarray],
    right_tcp: Optional[np.ndarray],
) -> dict[str, str]:
    def dist(a: Optional[np.ndarray], b: np.ndarray) -> float:
        if a is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)))

    left_pre = candidates["left"]["pregrasp_base"]
    right_pre = candidates["right"]["pregrasp_base"]
    if left_tcp is not None and right_tcp is not None:
        keep = dist(left_tcp, left_pre) + dist(right_tcp, right_pre)
        swap = dist(left_tcp, right_pre) + dist(right_tcp, left_pre)
        if keep <= swap:
            return {"left": "left", "right": "right", "policy": "distance_keep"}
        return {"left": "right", "right": "left", "policy": "distance_swap"}
    if float(left_pre[1]) <= float(right_pre[1]):
        return {"left": "left", "right": "right", "policy": "fallback_y_order"}
    return {"left": "right", "right": "left", "policy": "fallback_y_order"}


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    if sy > 1e-6:
        return (
            math.atan2(matrix[2, 1], matrix[2, 2]),
            math.atan2(-matrix[2, 0], sy),
            math.atan2(matrix[1, 0], matrix[0, 0]),
        )
    return (
        math.atan2(-matrix[1, 2], matrix[1, 1]),
        math.atan2(-matrix[2, 0], sy),
        0.0,
    )


class BoxBimanualGraspPanel(QtWidgets.QWidget):
    def __init__(self, host: Any) -> None:
        super().__init__()
        self._host = host
        self._solution: Optional[BoxBimanualGraspSolution] = None
        self._last_points_base: Optional[np.ndarray] = None
        self._last_target_name = "--"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info = QtWidgets.QLabel(
            "先在“视觉 YOLO”页完成箱体分割，再回到这里生成 box frame、左右预抓取位姿和 3D 可视化。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        control_box = QtWidgets.QGroupBox("grasp box")
        controls = QtWidgets.QGridLayout(control_box)
        self.pregrasp_spin = QtWidgets.QDoubleSpinBox()
        self.pregrasp_spin.setRange(0.03, 0.20)
        self.pregrasp_spin.setDecimals(3)
        self.pregrasp_spin.setSingleStep(0.005)
        self.pregrasp_spin.setValue(0.08)
        self.pregrasp_spin.setSuffix(" m")

        self.grasp_z_spin = QtWidgets.QDoubleSpinBox()
        self.grasp_z_spin.setRange(-0.12, 0.12)
        self.grasp_z_spin.setDecimals(3)
        self.grasp_z_spin.setSingleStep(0.005)
        self.grasp_z_spin.setValue(0.0)
        self.grasp_z_spin.setSuffix(" m")

        self.front_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.front_threshold_spin.setRange(0.004, 0.030)
        self.front_threshold_spin.setDecimals(3)
        self.front_threshold_spin.setSingleStep(0.001)
        self.front_threshold_spin.setValue(0.012)
        self.front_threshold_spin.setSuffix(" m")

        self.max_plot_points_spin = QtWidgets.QSpinBox()
        self.max_plot_points_spin.setRange(500, 20000)
        self.max_plot_points_spin.setSingleStep(500)
        self.max_plot_points_spin.setValue(3500)

        self.generate_btn = QtWidgets.QPushButton("生成预抓取点")
        self.refresh_btn = QtWidgets.QPushButton("刷新视图")
        self.ready_pose_btn = QtWidgets.QPushButton("箱子初始姿态")
        self.left_pregrasp_btn = QtWidgets.QPushButton("左手预抓取")
        self.left_forward_btn = QtWidgets.QPushButton("左手前进")
        self.right_pregrasp_btn = QtWidgets.QPushButton("右手预抓取")
        self.right_forward_btn = QtWidgets.QPushButton("右手前进")
        self.both_pregrasp_btn = QtWidgets.QPushButton("双手预抓取")
        self.both_forward_btn = QtWidgets.QPushButton("双手前进")
        self.generate_btn.clicked.connect(self.generate)
        self.refresh_btn.clicked.connect(self.refresh_view)
        self.ready_pose_btn.clicked.connect(self.send_box_ready_pose)
        self.left_pregrasp_btn.clicked.connect(lambda: self.send_box_arm_target("left", "pregrasp"))
        self.left_forward_btn.clicked.connect(lambda: self.send_box_arm_target("left", "grasp"))
        self.right_pregrasp_btn.clicked.connect(lambda: self.send_box_arm_target("right", "pregrasp"))
        self.right_forward_btn.clicked.connect(lambda: self.send_box_arm_target("right", "grasp"))
        self.both_pregrasp_btn.clicked.connect(lambda: self.send_box_both_targets("pregrasp"))
        self.both_forward_btn.clicked.connect(lambda: self.send_box_both_targets("grasp"))

        controls.addWidget(QtWidgets.QLabel("预抓取距离"), 0, 0)
        controls.addWidget(self.pregrasp_spin, 0, 1)
        controls.addWidget(QtWidgets.QLabel("抓取高度偏移"), 0, 2)
        controls.addWidget(self.grasp_z_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("平面阈值"), 1, 0)
        controls.addWidget(self.front_threshold_spin, 1, 1)
        controls.addWidget(QtWidgets.QLabel("点云采样上限"), 1, 2)
        controls.addWidget(self.max_plot_points_spin, 1, 3)
        controls.addWidget(self.generate_btn, 2, 0)
        controls.addWidget(self.refresh_btn, 2, 1)
        controls.addWidget(self.ready_pose_btn, 2, 2)
        controls.addWidget(self.left_pregrasp_btn, 3, 0)
        controls.addWidget(self.left_forward_btn, 3, 1)
        controls.addWidget(self.right_pregrasp_btn, 3, 2)
        controls.addWidget(self.right_forward_btn, 3, 3)
        controls.addWidget(self.both_pregrasp_btn, 4, 0, 1, 2)
        controls.addWidget(self.both_forward_btn, 4, 2, 1, 2)
        layout.addWidget(control_box)

        self.result_label = QtWidgets.QLabel("请先在 YOLO 页识别 box / xiangzi")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.result_label)

        self.figure = Figure(figsize=(10.6, 8.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(420)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

    def reset_from_vision(self, message: str = "已切换模型，请重新识别箱体") -> None:
        self._solution = None
        self._last_points_base = None
        self.result_label.setText(message)
        self.figure.clear()
        self.canvas.draw_idle()
        self._append_log("[BOX] " + message)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        if hasattr(self._host, "append_log"):
            self._host.append_log(text)

    def _current_target(self) -> dict[str, Any]:
        target = getattr(self._host, "_vision_target", None)
        if not target:
            raise ValueError("请先在“视觉 YOLO”页识别目标")
        class_name = str(target.get("class_name", "")).strip().lower()
        if class_name and class_name not in BOX_CLASS_ALIASES:
            self._append_log(f"[BOX] 当前标签不是 box 类: {class_name}")
        return target

    def _extract_points_base(self, target: dict[str, Any]) -> np.ndarray:
        points_base = target.get("points_base")
        if points_base is not None:
            return _ensure_points_array(points_base)
        points_camera = target.get("points_camera")
        if points_camera is None:
            raise ValueError("YOLO 结果里没有可用的点云")
        T_base_camera = getattr(self._host, "_camera_to_base", None)
        if T_base_camera is None:
            raise ValueError("没有相机外参，无法把点云转到 base_link")
        return transform_points(np.asarray(points_camera, dtype=np.float64), np.asarray(T_base_camera, dtype=np.float64))

    def _extract_car_link_geometry(self, target: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        snap = self._host._backend.snapshot()
        if snap.car_from_body is None:
            raise ValueError("car_link <- body_link TF 未就绪，无法生成 car_link 下的箱体位姿")
        T_car_body = np.asarray(snap.car_from_body, dtype=np.float64).reshape(4, 4)

        points_body = target.get("points_car")
        if points_body is not None:
            points_car = _ensure_points_array(points_body)
        else:
            points_base = target.get("points_base")
            if points_base is not None:
                points_body = _ensure_points_array(points_base)
            else:
                points_camera = target.get("points_camera")
                if points_camera is None:
                    raise ValueError("YOLO 结果里没有可用的点云")
                T_body_camera = getattr(self._host, "_camera_to_base", None)
                if T_body_camera is None:
                    raise ValueError("没有相机外参，无法把点云转到 body_link")
                points_body = transform_points(
                    np.asarray(points_camera, dtype=np.float64),
                    np.asarray(T_body_camera, dtype=np.float64),
                )
            points_body_h = np.column_stack((points_body, np.ones((len(points_body), 1), dtype=np.float64)))
            points_car = (T_car_body @ points_body_h.T).T[:, :3]

        T_body_camera = getattr(self._host, "_camera_to_base", None)
        if T_body_camera is None:
            raise ValueError("没有相机外参，无法确定 car_link 下相机位置")
        T_body_camera = np.asarray(T_body_camera, dtype=np.float64).reshape(4, 4)
        camera_pos_body = T_body_camera[:3, 3]
        camera_pos_car = (T_car_body @ np.array([camera_pos_body[0], camera_pos_body[1], camera_pos_body[2], 1.0], dtype=np.float64))[:3]
        return points_car, camera_pos_car

    def _plot_solution(self, points_base: np.ndarray, solution: BoxBimanualGraspSolution) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        points = _ensure_points_array(points_base)
        max_points = int(self.max_plot_points_spin.value())
        if len(points) > max_points:
            rng = np.random.default_rng(7)
            sample_idx = rng.choice(len(points), max_points, replace=False)
            sample = points[sample_idx]
        else:
            sample = points

        front_points = solution.front_points
        front_sample = front_points
        if len(front_sample) > max_points // 2:
            rng = np.random.default_rng(11)
            sample_idx = rng.choice(len(front_sample), max_points // 2, replace=False)
            front_sample = front_sample[sample_idx]

        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.5, c="#94a3b8", alpha=0.13)
        ax.scatter(front_sample[:, 0], front_sample[:, 1], front_sample[:, 2], s=2.5, c="#f59e0b", alpha=0.28)

        T = solution.T_base_box
        center = T[:3, 3]
        x_axis, y_axis, z_axis = T[:3, 0], T[:3, 1], T[:3, 2]
        axis_len = 0.12
        for vec, color in (
            (x_axis, "#ef4444"),
            (y_axis, "#22c55e"),
            (z_axis, "#3b82f6"),
        ):
            ax.quiver(
                center[0], center[1], center[2],
                vec[0], vec[1], vec[2],
                length=axis_len,
                color=color,
                arrow_length_ratio=0.18,
                linewidth=2.0,
            )

        self._draw_box_wireframe(ax, T)

        for side, color in (("left", "#8b5cf6"), ("right", "#14b8a6")):
            cand = solution.candidates[side]
            grasp = cand["grasp_base"]
            pregrasp = cand["pregrasp_base"]
            ax.scatter([grasp[0]], [grasp[1]], [grasp[2]], s=42, c=color, marker="o")
            ax.scatter([pregrasp[0]], [pregrasp[1]], [pregrasp[2]], s=42, c=color, marker="^")
            ax.plot(
                [pregrasp[0], grasp[0]],
                [pregrasp[1], grasp[1]],
                [pregrasp[2], grasp[2]],
                color=color,
                linewidth=2.0,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((1.0, 1.0, 0.8))
        self._set_axes_equal(ax)
        self.canvas.draw_idle()

    def _draw_box_wireframe(self, ax: Any, T_base_box: np.ndarray) -> None:
        box_debug = getattr(self._solution, "debug_info", {}).get("box", {}) if self._solution is not None else {}
        depth_m = float(box_debug.get("box_depth_m", min(BOX_LENGTH_M, BOX_WIDTH_M)))
        face_span_m = float(box_debug.get("face_span_m", max(BOX_LENGTH_M, BOX_WIDTH_M)))
        R = np.asarray(T_base_box[:3, :3], dtype=np.float64)
        c = np.asarray(T_base_box[:3, 3], dtype=np.float64)
        hx = 0.5 * depth_m
        hy = 0.5 * face_span_m
        hz = 0.5 * BOX_HEIGHT_M
        corners = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    corners.append(c + R @ np.array([sx, sy, sz], dtype=np.float64))
        corners = np.asarray(corners, dtype=np.float64)
        edges = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )
        for i, j in edges:
            xs = [corners[i, 0], corners[j, 0]]
            ys = [corners[i, 1], corners[j, 1]]
            zs = [corners[i, 2], corners[j, 2]]
            ax.plot(xs, ys, zs, color="#475569", linewidth=1.2, alpha=0.8)

    def _set_axes_equal(self, ax: Any) -> None:
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ],
            dtype=np.float64,
        )
        centers = limits.mean(axis=1)
        spans = limits[:, 1] - limits[:, 0]
        radius = 0.5 * float(np.max(spans))
        ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
        ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
        ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

    def _format_pose(self, pose: np.ndarray) -> str:
        return (
            f"{pose[0]:+.4f} {pose[1]:+.4f} {pose[2]:+.4f} "
            f"{math.degrees(pose[3]):+.2f} {math.degrees(pose[4]):+.2f} {math.degrees(pose[5]):+.2f}"
        )

    def _require_solution(self) -> BoxBimanualGraspSolution:
        if self._solution is None:
            raise ValueError("请先生成箱体位姿")
        return self._solution

    def _current_arm_targets(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        snap = self._host._backend.snapshot()
        left = tuple(self._host._pose_xyzrpy(snap.left_ee, "左臂 TCP"))
        right = tuple(self._host._pose_xyzrpy(snap.right_ee, "右臂 TCP"))
        return left, right

    def _solution_pose_for_arm(self, arm: str, stage: str) -> tuple[float, ...]:
        solution = self._require_solution()
        if arm not in solution.candidates:
            raise ValueError(f"{arm} 臂没有可用的预抓取点")
        key = "pose_pregrasp_base" if stage == "pregrasp" else "pose_grasp_base"
        pose = solution.candidates[arm][key]
        return tuple(float(value) for value in pose.tolist())

    def send_box_arm_target(self, arm: str, stage: str) -> None:
        try:
            left, right = self._current_arm_targets()
            target = self._solution_pose_for_arm(arm, stage)
            if arm == "left":
                left = target
                label = "左手"
            elif arm == "right":
                right = target
                label = "右手"
            else:
                raise ValueError(f"unknown arm: {arm}")
            self._host._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
            action = "预抓取" if stage == "pregrasp" else "前进"
            text = f"{label}{action}目标已发送: {self._format_pose(np.asarray(target, dtype=np.float64))}"
            self.result_label.setText(text)
            self._append_log("[BOX] " + text)
        except Exception as exc:
            self.result_label.setText(f"发送失败: {exc}")
            self._append_log(f"[BOX] 发送失败: {exc}")

    def send_box_both_targets(self, stage: str) -> None:
        try:
            left = self._solution_pose_for_arm("left", stage)
            right = self._solution_pose_for_arm("right", stage)
            self._host._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
            action = "预抓取" if stage == "pregrasp" else "前进"
            text = (
                f"双手{action}目标已发送 | "
                f"左 {self._format_pose(np.asarray(left, dtype=np.float64))} | "
                f"右 {self._format_pose(np.asarray(right, dtype=np.float64))}"
            )
            self.result_label.setText(text)
            self._append_log("[BOX] " + text)
        except Exception as exc:
            self.result_label.setText(f"双手发送失败: {exc}")
            self._append_log(f"[BOX] 双手发送失败: {exc}")

    def send_box_ready_pose(self) -> None:
        try:
            left = tuple(math.radians(value) for value in BOX_READY_LEFT_ARM_DEG)
            right = tuple(math.radians(value) for value in BOX_READY_RIGHT_ARM_DEG)
            self._host._backend.send("arm_joint", left=left, right=right, vel=0.30, acc=0.50)
            text = (
                "已发送箱子抓取初始姿态 | "
                "左 " + " ".join(f"{value:+.1f}" for value in BOX_READY_LEFT_ARM_DEG) + " | "
                "右 " + " ".join(f"{value:+.1f}" for value in BOX_READY_RIGHT_ARM_DEG)
            )
            self.result_label.setText(text)
            self._append_log("[BOX] " + text)
        except Exception as exc:
            self.result_label.setText(f"箱子初始姿态发送失败: {exc}")
            self._append_log(f"[BOX] 箱子初始姿态发送失败: {exc}")

    def generate(self) -> None:
        try:
            target = self._current_target()
            points_car, camera_pos_car = self._extract_car_link_geometry(target)
            self._last_points_base = points_car
            frame_id = "car_link"
            class_name = str(target.get("class_name", "")).strip().lower()
            confidence = float(target.get("confidence", 0.0))
            self._append_log(f"[BOX] 读取 YOLO 结果: class={class_name or '--'} conf={confidence:.2f} frame={frame_id}")

            front_points, front_normal, plane_debug = estimate_front_plane(
                points_car,
                preferred_face_span_m=BOX_LENGTH_M,
                preferred_height_m=BOX_HEIGHT_M,
                distance_threshold=float(self.front_threshold_spin.value()),
            )
            T_base_box, box_debug = estimate_box_frame(
                front_points,
                front_normal,
                camera_pos_car,
                box_length_m=BOX_LENGTH_M,
                box_width_m=BOX_WIDTH_M,
                box_height_m=BOX_HEIGHT_M,
            )
            candidates = generate_side_grasp_candidates(
                T_base_box,
                face_span_m=float(box_debug["face_span_m"]),
                box_depth_m=float(box_debug["box_depth_m"]),
                pregrasp_dist_m=float(self.pregrasp_spin.value()),
                grasp_z_offset_m=float(self.grasp_z_spin.value()),
                grasp_x_offset_m=-0.5 * float(box_debug["box_depth_m"]),
                front_center_base=box_debug["front_center_base"],
                box_center_base=box_debug["box_center_base"],
            )
            snap = self._host._backend.snapshot() if hasattr(self._host, "_backend") else None
            left_tcp = None
            right_tcp = None
            if snap is not None:
                if snap.left_ee is not None:
                    left_tcp = np.array(self._host._pose_xyzrpy(snap.left_ee, "左臂 TCP")[:3], dtype=np.float64)
                if snap.right_ee is not None:
                    right_tcp = np.array(self._host._pose_xyzrpy(snap.right_ee, "右臂 TCP")[:3], dtype=np.float64)
            assignment = assign_candidates_to_arms(candidates, left_tcp, right_tcp)

            self._solution = BoxBimanualGraspSolution(
                T_base_box=T_base_box,
                front_points=front_points,
                front_plane_normal=front_normal,
                debug_info={
                    "plane": plane_debug,
                    "box": box_debug,
                    "input_point_count": int(len(points_car)),
                    "assigned_policy": assignment["policy"],
                    "task_frame": frame_id,
                },
                candidates=candidates,
                assignment=assignment,
            )

            self._append_log(
                "[BOX] "
                f"front={_vec_text(box_debug['front_center_base'])} "
                f"center={_vec_text(box_debug['box_center_base'])} "
                f"left_pre={_vec_text(candidates['left']['pregrasp_base'])} "
                f"right_pre={_vec_text(candidates['right']['pregrasp_base'])}"
            )

            self.result_label.setText(
                "box frame ready | "
                f"front {_vec_text(box_debug['front_center_base'])} | "
                f"center {_vec_text(box_debug['box_center_base'])} | "
                f"left pre {_vec_text(candidates['left']['pregrasp_base'])} | "
                f"right pre {_vec_text(candidates['right']['pregrasp_base'])}"
            )
            self._plot_solution(points_car, self._solution)
        except Exception as exc:
            self.result_label.setText(f"生成失败: {exc}")
            self._append_log(f"[BOX] 生成失败: {exc}")

    def refresh_view(self) -> None:
        if self._solution is None or self._last_points_base is None:
            self.generate()
            return
        self._plot_solution(self._last_points_base, self._solution)


def build_box_bimanual_grasp_page(host: Any) -> BoxBimanualGraspPanel:
    return BoxBimanualGraspPanel(host)
