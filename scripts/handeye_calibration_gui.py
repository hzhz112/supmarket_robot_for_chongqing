#!/usr/bin/env python3
"""通用 ROS 2 手眼标定工具。

这个文件的整体流程是：
1. 从相机图像中检测标定板，得到 target_T_camera。
2. 从 TF 或 ROS 位姿话题中得到 base_T_tool。
3. 点击网页中的“采样”，把同一时刻的两组位姿保存下来。
4. 使用 OpenCV calibrateHandEye 计算手眼外参。
5. 将结果和全部原始样本保存为 YAML，方便后续检查或离线重算。

OpenCV 的坐标约定：
  eye-in-hand: 输入 tool_T_base 和 target_T_camera，输出 camera_T_tool。
  eye-to-hand: 输入 tool_T_base 和 target_T_camera，输出 camera_T_base。
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import queue
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener
except ImportError as exc:  # Keep --help and offline inspection usable.
    rclpy = None
    Node = object
    _ROS_IMPORT_ERROR = exc


# 当前文件位于 sup_robot/scripts，下一级父目录就是 sup_robot。
ROOT = Path(__file__).resolve().parents[1]
# 将机器人控制工作区加入 Python 搜索路径，以便导入 DualArmControlClient。
RUNTIME_SRC = ROOT / "ros2_robot_controller_runtime" / "src"
if RUNTIME_SRC.exists():
    import sys
    sys.path.insert(0, str(RUNTIME_SRC))

try:
    from sailor_r1_pro_description.arm_topics import ArmCartesianTarget, DualArmControlClient
except ImportError:
    ArmCartesianTarget = None
    DualArmControlClient = None


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数转 3x3 旋转矩阵。

    ROS geometry_msgs 中的四元数顺序是 x、y、z、w。
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n  #进行归一化
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def pose_to_matrix(pose: Any) -> np.ndarray:
    """把 geometry_msgs/Pose 转成 4x4 齐次变换矩阵。"""
    p, q = pose.position, pose.orientation
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    out[:3, 3] = [p.x, p.y, p.z]
    return out


def matrix_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """用旋转矩阵 R 和平移向量 t 组成齐次矩阵 T。"""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def matrix_to_quat(T: np.ndarray) -> tuple[float, float, float, float]:
    """4x4 齐次矩阵转四元数，当前主要用于结果显示或扩展。"""
    r = T[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


def rpy_deg(T: np.ndarray) -> list[float]:
    """从旋转矩阵提取 RPY，并转换为角度。"""
    r = T[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        vals = (math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], sy), math.atan2(r[1, 0], r[0, 0]))
    else:
        vals = (math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0)
    return [round(math.degrees(v), 4) for v in vals]


def invert(T: np.ndarray) -> np.ndarray:
    """求齐次变换的逆矩阵。

    例如 TF 查询得到 base_T_tool，OpenCV 需要 tool_T_base，
    所以在送入 calibrateHandEye 前需要调用本函数。
    """
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def solver_method(name: str) -> int:
    """把 YAML 中的算法名称转换为 OpenCV 常量。"""
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if name.lower() not in methods:
        raise ValueError(f"unknown method {name}, choose one of {sorted(methods)}")
    return methods[name.lower()]


def image_from_ros(msg: Any) -> np.ndarray:
    """将 ROS sensor_msgs/Image 转换为 OpenCV 图像。

    这里不依赖 cv_bridge，避免额外安装 Python 包。
    当前主要处理 mono8、8UC1、rgb8 和常见的 BGR 三通道图像。
    """
    channels = 1 if msg.encoding in ("mono8", "8UC1") else 3
    dtype = np.uint16 if "16" in msg.encoding else np.uint8
    image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // np.dtype(dtype).itemsize)[:, :msg.width * channels]
    if channels == 3:
        image = image.reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image)


class TargetDetector:
    """标定板检测器。

    detect() 返回：
      found: 是否成功得到位姿
      view: 叠加检测结果和坐标轴后的图像
      target_T_camera: 标定板坐标系到相机坐标系的变换
      visible: 检测到的角点或 marker 数量
    """

    def __init__(self, cfg: dict[str, Any], K: np.ndarray, dist: np.ndarray):
        self.cfg, self.K, self.dist = cfg, K, dist
        self.kind = str(cfg.get("type", "charuco")).lower()
        self.axis = float(cfg.get("axis_length_m", 0.05))
        # 棋盘格只需要构造平面角点的三维坐标。
        if self.kind == "chessboard":
            self.pattern = tuple(int(v) for v in cfg.get("pattern_size", [11, 8]))
            self.object_points = np.zeros((self.pattern[0] * self.pattern[1], 3), np.float64)
            self.object_points[:, :2] = np.mgrid[0:self.pattern[0], 0:self.pattern[1]].T.reshape(-1, 2) * float(cfg.get("square_size_m", 0.02))
        else:
            # ArUco、ChArUco、GridBoard 共用同一个字典。
            dict_id = getattr(cv2.aruco, str(cfg.get("aruco_dictionary", "DICT_4X4_250")))
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
            if self.kind == "charuco":
                sx, sy = int(cfg.get("squares_x", 14)), int(cfg.get("squares_y", 9))
                self.board = cv2.aruco.CharucoBoard((sx, sy), float(cfg.get("square_size_m", 0.02)), float(cfg.get("marker_size_m", 0.015)), self.dictionary)
            elif self.kind == "single":
                half = float(cfg.get("marker_size_m", 0.04)) / 2
                self.object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)
            else:
                self.board = cv2.aruco.GridBoard((int(cfg["cols"]), int(cfg["rows"])), float(cfg["marker_size_m"]), float(cfg.get("separation_m", 0.01)), self.dictionary)

    def detect(self, image: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray | None, int]:
        """检测当前帧，并通过 solvePnP 求标定板相对相机的位姿。"""
        view = image.copy()
        if self.kind == "chessboard":
            # findChessboardCornersSB 比旧版 findChessboardCorners 对光照更稳。
            found, corners = cv2.findChessboardCornersSB(view, self.pattern)
            if not found:
                return False, view, None, 0
            ok, rvec, tvec = cv2.solvePnP(self.object_points, corners, self.K, self.dist)
            if not ok:
                return False, view, None, 0
            cv2.drawChessboardCorners(view, self.pattern, corners, found)
            count = len(corners)
        else:
            corners, ids, _ = self.detector.detectMarkers(view)
            count = 0 if ids is None else len(ids)
            if ids is None:
                return False, view, None, count
            cv2.aruco.drawDetectedMarkers(view, corners, ids)
            if self.kind == "charuco":
                # ChArUco 先检测 marker，再由棋盘模型恢复亚像素角点。
                cc, ci, _, _ = cv2.aruco.CharucoDetector(self.board).detectBoard(view)
                if ci is None or len(ci) < int(self.cfg.get("min_corners", 6)):
                    return False, view, None, len(ci) if ci is not None else 0
                obj, img = self.board.matchImagePoints(cc, ci)
                ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist)
                count = len(ci)
            elif self.kind == "grid":
                # GridBoard 直接通过 marker ID 与棋盘模型匹配三维点。
                obj, img = self.board.matchImagePoints(corners, ids)
                if obj is None or len(obj) < 4:
                    return False, view, None, count
                ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist)
            else:
                # 单 ArUco 只使用配置中指定的 marker_id。
                marker_id = int(self.cfg.get("marker_id", 0))
                matches = np.where(ids.reshape(-1) == marker_id)[0]
                if len(matches) == 0:
                    return False, view, None, count
                ok, rvec, tvec = cv2.solvePnP(self.object_points, corners[int(matches[0])].reshape(4, 2), self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return False, view, None, count
        cv2.drawFrameAxes(view, self.K, self.dist, rvec, tvec, self.axis)
        return True, view, matrix_from_rt(cv2.Rodrigues(rvec)[0], tvec), count


class CalibrationSession:
    """一个标定任务的状态。

    一个任务由 YAML 中的 calibration 节点描述，例如：
      left_eye_in_hand = 左臂 + 左相机 + eye-in-hand。
    页面切换任务时，实际切换的就是本类的实例。
    """

    def __init__(self, name: str, cfg: dict[str, Any], camera_cfg: dict[str, Any], arm_cfg: dict[str, Any], out_dir: Path):
        self.name, self.cfg, self.camera_cfg, self.arm_cfg = name, cfg, camera_cfg, arm_cfg
        self.out_dir = out_dir
        self.samples: list[dict[str, Any]] = []
        self.result: np.ndarray | None = None
        self.last_target: np.ndarray | None = None
        self.last_robot: np.ndarray | None = None
        self.last_found, self.last_visible, self.last_frame = False, 0, None
        self.last_error = ""
        self.lock = threading.Lock()
        K = np.asarray(camera_cfg.get("camera_matrix"), dtype=np.float64) if camera_cfg.get("camera_matrix") else None
        dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
        self.K, self.dist = K, dist.reshape(-1, 1)
        self.detector = TargetDetector(cfg["target"], self.K, self.dist) if K is not None else None

    @property
    def mode(self) -> str:
        return str(self.cfg.get("mode", "eye_in_hand")).lower()

    def add_sample(self) -> bool:
        """保存当前帧的机械臂位姿和标定板位姿。"""
        with self.lock:
            if self.last_target is None or self.last_robot is None or not self.last_found:
                self.last_error = "当前没有同时获得有效的标定板位姿和机械臂位姿"
                return False
            self.samples.append({"robot": self.last_robot.copy(), "target": self.last_target.copy(), "time": time.time()})
            self.last_error = f"采样成功 {len(self.samples)}"
            return True

    def undo(self) -> None:
        """删除最后一组样本。"""
        with self.lock:
            if self.samples:
                self.samples.pop()
            self.last_error = f"撤销后剩余 {len(self.samples)}"

    def compute(self) -> np.ndarray:
        """调用 OpenCV 计算手眼矩阵。

        TF 查询得到的是 base_T_tool，而 OpenCV 的输入要求是
        tool_T_base，所以两种模式都要先求逆。
        """
        with self.lock:
            if len(self.samples) < int(self.cfg.get("min_samples", 10)):
                raise ValueError(f"样本不足：{len(self.samples)}，至少需要 {self.cfg.get('min_samples', 10)}")
            robot = [x["robot"] for x in self.samples]
            target = [x["target"] for x in self.samples]
        robot = [invert(x) for x in robot]  # base_T_tool -> tool_T_base
        R1, t1 = zip(*[(x[:3, :3], x[:3, 3]) for x in robot])
        R2, t2 = zip(*[(x[:3, :3], x[:3, 3]) for x in target])
        R, t = cv2.calibrateHandEye(list(R1), list(t1), list(R2), list(t2), method=solver_method(self.cfg.get("method", "park")))
        self.result = matrix_from_rt(R, t)
        return self.result

    def save(self) -> Path:
        """保存计算结果、相机内参以及所有原始采样数据。"""
        if self.result is None:
            self.compute()
        path = self.out_dir / f"{self.name}.yaml"
        payload = {
            "calibration_name": self.name,
            "mode": self.mode,
            "transform_direction": "camera_T_tool" if self.mode == "eye_in_hand" else "camera_T_base",
            "arm": self.cfg.get("arm"),
            "camera": self.cfg.get("camera"),
            "base_frame": self.arm_cfg.get("base_frame"),
            "tool_frame": self.arm_cfg.get("tool_frame"),
            "method": self.cfg.get("method", "park"),
            "samples": len(self.samples),
            "sample_data": [
                {
                    "time_unix": float(sample["time"]),
                    "robot_pose": np.asarray(sample["robot"], dtype=np.float64).tolist(),
                    "target_pose": np.asarray(sample["target"], dtype=np.float64).tolist(),
                }
                for sample in self.samples
            ],
            "camera_matrix": self.K.tolist(),
            "distortion": self.dist.reshape(-1).tolist(),
            "extrinsic_matrix": self.result.tolist(),
            "translation_m": self.result[:3, 3].tolist(),
            "rpy_deg": rpy_deg(self.result),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def status(self) -> dict[str, Any]:
        """生成网页需要的当前状态。"""
        with self.lock:
            return {
                "name": self.name, "mode": self.mode, "samples": len(self.samples),
                "min_samples": int(self.cfg.get("min_samples", 10)), "found": self.last_found,
                "visible": self.last_visible, "error": self.last_error,
                "result": self.result.tolist() if self.result is not None else None,
                "rpy_deg": rpy_deg(self.result) if self.result is not None else None,
            }


class HandEyeNode(Node):
    """ROS 2 节点：订阅图像/位姿，并提供 TF 查询。"""

    def __init__(self, app: "HandEyeApp"):
        super().__init__("handeye_calibration_gui")
        self.app = app
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_cache: dict[str, np.ndarray] = {}
        self.image_cache: dict[str, np.ndarray] = {}
        self.image_subs = []
        # 相机图像可以来自 ROS Image 话题。
        for name, cfg in app.config.get("cameras", {}).items():
            if cfg.get("source", "ros") == "ros" and cfg.get("image_topic"):
                self.image_subs.append(self.create_subscription(Image, cfg["image_topic"], lambda msg, n=name: self.image_cache.__setitem__(n, image_from_ros(msg)), 10))
            if cfg.get("source", "ros") == "ros" and cfg.get("camera_info_topic"):
                self.image_subs.append(self.create_subscription(CameraInfo, cfg["camera_info_topic"], lambda msg, n=name: self._camera_info(n, msg), 10))
        # 机械臂位姿可以来自 EndEffectorPose 话题；如果 pose_source=tf，
        # 则不会使用这里缓存的位姿，而是每一帧直接查询 TF。
        self.pose_subs = []
        for name, arm in app.config.get("arms", {}).items():
            topic = arm.get("pose_topic")
            if topic:
                from robot_control_msg.msg import EndEffectorPose
                self.pose_subs.append(self.create_subscription(EndEffectorPose, topic, lambda msg, n=name: self._pose_msg(n, msg), 10))

    def _pose_msg(self, arm_name: str, msg: Any) -> None:
        """缓存 EndEffectorPose 中对应手臂的末端位姿。"""
        field = "left_ee_pose" if arm_name == "left" else "right_ee_pose"
        self.pose_cache[arm_name] = pose_to_matrix(getattr(msg, field))

    def _camera_info(self, camera_name: str, msg: Any) -> None:
        """从 CameraInfo 自动更新相机内参。"""
        camera = self.app.config["cameras"][camera_name]
        camera["camera_matrix"] = [list(msg.k[0:3]), list(msg.k[3:6]), list(msg.k[6:9])]
        camera["distortion"] = list(msg.d)
        for session in self.app.sessions.values():
            if session.cfg.get("camera") != camera_name:
                continue
            session.K = np.asarray(camera["camera_matrix"], dtype=np.float64)
            session.dist = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1, 1)
            session.detector = TargetDetector(session.cfg["target"], session.K, session.dist)

    def robot_pose(self, arm: str, cfg: dict[str, Any]) -> np.ndarray | None:
        """获得当前机械臂的 base_T_tool。

        注意：这里返回的是 base_T_tool，CalibrationSession.compute()
        中会统一转换为 OpenCV 所需的 tool_T_base。
        """
        if cfg.get("pose_source", "tf") == "topic":
            return self.pose_cache.get(arm)
        try:
            tr = self.tf_buffer.lookup_transform(cfg["base_frame"], cfg["tool_frame"], rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05))
            out = np.eye(4)
            t, q = tr.transform.translation, tr.transform.rotation
            out[:3, :3], out[:3, 3] = quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z]
            return out
        except Exception:
            return None


class HandEyeApp:
    """应用层：管理标定任务、网页服务器、相机和机械臂运动。"""

    # 这是一个极简的内置网页，不需要额外安装 Flask、Qt 等界面框架。
    PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>通用手眼标定</title><style>
body{margin:0;background:#102027;color:#e8f1f2;font-family:system-ui,sans-serif}header{padding:14px 20px;background:#17343b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:14px}.card{background:#17343b;border:1px solid #2d5860;border-radius:10px;padding:14px}img{width:100%;max-height:78vh;object-fit:contain;background:#071316;border-radius:8px}button,select{box-sizing:border-box;width:100%;padding:10px;margin:5px 0;border:0;border-radius:6px;font-size:15px}button{background:#2b9c8c;color:white;cursor:pointer}button.alt{background:#31515a}button.warn{background:#a85b45}button:disabled{opacity:.4}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#bfe1df}@media(max-width:850px){main{grid-template-columns:1fr}}</style>
<header><b>通用手眼标定</b><span id="summary">连接中...</span></header><main><section class="card"><img id="frame" src="/frame.jpg"><pre id="result"></pre></section><aside class="card"><label>标定配置</label><select id="session"></select><button onclick="cmd('sample')">采样当前位姿</button><button class="alt" onclick="cmd('undo')">撤销上次采样</button><button class="alt" onclick="cmd('compute')">计算结果</button><button class="alt" onclick="cmd('save')">保存 YAML</button><button class="alt" onclick="cmd('move')">移动到下一预设位姿</button><button class="warn" onclick="cmd('clear')">清空当前样本</button><pre id="status"></pre></aside></main>
<script>
async function cmd(c){let s=document.getElementById('session').value;let r=await fetch('/command?session='+encodeURIComponent(s)+'&cmd='+c);let j=await r.json();document.getElementById('result').textContent=j.message||'';update()}
async function update(){let r=await fetch('/status.json');let j=await r.json();let s=j.active;document.getElementById('summary').textContent=s.name+' | '+s.mode+' | '+s.samples+'/'+s.min_samples+(s.found?' | 目标已找到':' | 未找到');document.getElementById('status').textContent=JSON.stringify(s,null,2);document.getElementById('frame').src='/frame.jpg?t='+Date.now()}
async function load(){let r=await fetch('/sessions.json');let a=await r.json();let e=document.getElementById('session');e.innerHTML=a.map(x=>'<option>'+x+'</option>').join('');e.onchange=()=>fetch('/select?session='+encodeURIComponent(e.value));update()}
setInterval(update,500);load();
</script></html>"""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config, self.config_path = config, config_path
        self.out_dir = Path(config.get("output_dir", str(config_path.parent / "handeye_results"))).expanduser()
        self.sessions: dict[str, CalibrationSession] = {}
        # 每个 calibration 配置都会创建一个独立的标定任务。
        for name, cfg in config.get("calibrations", {}).items():
            camera = config["cameras"][cfg["camera"]]
            arm = config["arms"][cfg["arm"]]
            self.sessions[name] = CalibrationSession(name, cfg, camera, arm, self.out_dir)
        self.active = next(iter(self.sessions))
        self.node: HandEyeNode | None = None
        self.httpd = None
        self.server_thread = None
        self.commands: queue.Queue[tuple[str, str]] = queue.Queue()
        self.camera_pipelines: dict[str, Any] = {}
        self.motion_index = 0
        self.motion_node = None

    def start_http(self, host: str, port: int) -> None:
        """启动网页服务器。浏览器通过 HTTP 调用采样和计算命令。"""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        app = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def send_json(self, data):
                body = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                p = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(p.query)
                if p.path == "/":
                    body = app.PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif p.path == "/sessions.json": self.send_json(list(app.sessions))
                elif p.path == "/select": app.active = q.get("session", [app.active])[0]; self.send_json({"ok": True})
                elif p.path == "/status.json": self.send_json({"active": app.status()})
                elif p.path == "/command":
                    app.commands.put((q.get("session", [app.active])[0], q.get("cmd", [""])[0])); self.send_json({"ok": True, "message": "命令已发送"})
                elif p.path == "/frame.jpg":
                    frame = app.sessions[app.active].last_frame
                    if frame is None: self.send_error(404); return
                    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok: self.send_error(500); return
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data.tobytes())
                else: self.send_error(404)
        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True); self.server_thread.start()

    def status(self) -> dict[str, Any]:
        """返回当前选中任务的状态。"""
        return self.sessions[self.active].status()

    def camera_frame(self, name: str, cfg: dict[str, Any]) -> np.ndarray | None:
        """取得当前相机的一帧图像。

        source=ros      从 HandEyeNode 的 ROS 图像缓存读取。
        source=realsense 直接按序列号打开 pyrealsense2。
        """
        if cfg.get("source", "ros") == "ros":
            return self.node.image_cache.get(name) if self.node else None
        try:
            import pyrealsense2 as rs
            if name not in self.camera_pipelines:
                # RealSense 只在第一次使用该相机时启动，后续复用 pipeline。
                pipe, c = rs.pipeline(), rs.config()
                if cfg.get("serial"): c.enable_device(str(cfg["serial"]))
                w, h, fps = cfg.get("width", 1280), cfg.get("height", 720), cfg.get("fps", 30)
                c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                profile = pipe.start(c); self.camera_pipelines[name] = (pipe, profile)
                intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                s = self.sessions[self.active]
                s.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float64)
                s.dist = np.asarray(intr.coeffs, np.float64).reshape(-1, 1)
                s.detector = TargetDetector(s.cfg["target"], s.K, s.dist)
            frames = self.camera_pipelines[name][0].wait_for_frames(1000)
            f = frames.get_color_frame()
            return np.asanyarray(f.get_data()).copy() if f else None
        except Exception as exc:
            self.sessions[self.active].last_error = f"相机失败: {exc}"
            return None

    def move(self, session: CalibrationSession) -> str:
        """调用现有双臂笛卡尔绝对位姿接口移动机器人。

        motion_poses 中的 RPY 单位是度，ArmCartesianTarget 内部使用弧度。
        每点击一次按钮，就发送下一组预设位姿。
        """
        poses = self.config.get("motion_poses", [])
        if not poses or DualArmControlClient is None:
            return "没有配置 motion_poses，或机器人控制库不可用"
        if self.motion_node is None:
            self.motion_node = DualArmControlClient(node_name="handeye_motion_client")
        pose = poses[self.motion_index % len(poses)]
        self.motion_index += 1
        def target(data):
            return ArmCartesianTarget(*[float(x) for x in data[:3]], *[math.radians(float(x)) for x in data[3:6]])
        left = target(pose.get("left", [0.3, 0.2, 0.8, 0, 0, 0]))
        right = target(pose.get("right", [0.3, -0.2, 0.8, 0, 0, 0]))
        name, response = self.motion_node.move_l(left, right, vel=float(self.config.get("motion", {}).get("vel", .08)), acc=float(self.config.get("motion", {}).get("acc", .15)))
        return f"已发送 {name}: {getattr(response, 'message', '')}"

    def handle_commands(self) -> None:
        """处理网页发来的命令，避免 HTTP 线程直接操作 ROS。"""
        while True:
            try: name, command = self.commands.get_nowait()
            except queue.Empty: return
            if name not in self.sessions: continue
            s = self.sessions[name]
            try:
                if command == "sample": s.add_sample()
                elif command == "undo": s.undo()
                elif command == "clear": s.samples.clear(); s.result = None
                elif command == "compute": s.compute(); s.last_error = "计算成功"
                elif command == "save": s.save(); s.last_error = "保存成功"
                elif command == "move": s.last_error = self.move(s)
            except Exception as exc:
                s.last_error = str(exc)

    def loop(self) -> None:
        """主循环：处理 ROS 回调、网页命令、图像检测和状态更新。"""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            self.handle_commands()
            s = self.sessions[self.active]
            camera = self.config["cameras"][s.cfg["camera"]]
            frame = self.camera_frame(s.cfg["camera"], camera)
            if frame is None or s.detector is None:
                time.sleep(.01); continue
            found, view, target, visible = s.detector.detect(frame)
            robot = self.node.robot_pose(s.cfg["arm"], self.config["arms"][s.cfg["arm"]]) if self.node else None
            with s.lock:
                s.last_found, s.last_visible, s.last_target, s.last_robot, s.last_frame = found, visible, target, robot, view
            time.sleep(.01)


def main() -> int:
    """程序入口：读取 YAML、初始化 ROS、启动网页和主循环。"""
    parser = argparse.ArgumentParser(description="通用 ROS 2 双臂双相机手眼标定网页工具")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "handeye_calibration.yaml"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if rclpy is None:
        raise RuntimeError(f"ROS 2 不可用，请先 source ROS 环境: {_ROS_IMPORT_ERROR}")
    config_path = Path(args.config).expanduser()
    app = HandEyeApp(yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path)
    rclpy.init()
    app.node = HandEyeNode(app)
    app.start_http(args.host, args.port)
    print(f"Hand-eye UI: http://127.0.0.1:{args.port}")
    try:
        app.loop()
    finally:
        for pipe, _ in app.camera_pipelines.values():
            pipe.stop()
        if app.httpd: app.httpd.shutdown()
        if app.motion_node: app.motion_node.destroy_node()
        app.node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
