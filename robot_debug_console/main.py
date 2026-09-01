#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import queue
import signal
import sys
import threading
import time
import csv
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    ROOT / "ros2_robot_controller_runtime" / "src",
    ROOT / "ros2_control_source_partial",
    ROOT / "ros2_robot_controller_runtime" / "install" / "local" / "lib" / "python3.10" / "dist-packages",
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from PyQt5 import QtCore, QtGui, QtWidgets

import rclpy
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt8
from tf2_ros import Buffer, TransformListener

from robot_control_msg.msg import ArmMotionStatus, ArmPowerStatus, EndEffectorPose, LegMotionStatus, LegPowerStatus, Robotarmjoint, Robotarmservomsg, Robotlegjoint, Robotservomsg
from robot_control_msg.srv import CartesianAbsoluteControl, JointAbsoluteControl, LegAbsoluteControl, LegCartesianControl, SetRobotPower

try:
    from .box_bimanual_grasp_agent import build_box_bimanual_grasp_page
except ImportError:
    from box_bimanual_grasp_agent import build_box_bimanual_grasp_page


ARM_JOINTS = ("ljoint1", "ljoint2", "ljoint3", "ljoint4", "ljoint5", "ljoint6", "ljoint7", "rjoint1", "rjoint2", "rjoint3", "rjoint4", "rjoint5", "rjoint6", "rjoint7")
LEFT_ARM_JOINTS = ARM_JOINTS[:7]
RIGHT_ARM_JOINTS = ARM_JOINTS[7:]
LEG_JOINTS = ("ankle_joint", "knee_joint", "hip_pitch_joint", "hip_yaw_joint")
LEG_POWER_READY = 39
RIGHT_HAND_GRASP_LABELS = {"guazi", "xiangzi", "pai", "cui"}
RIGHT_HAND_TCP_Y_OFFSET_M = -0.10
RIGHT_HAND_PREGRASP_DISTANCE_M = 0.10
RECORD_IDLE_SECONDS = 2.0
JOINT_IDLE_EPS_RAD = math.radians(0.2)
CART_IDLE_POS_EPS_M = 0.002
CART_IDLE_ROT_EPS_RAD = math.radians(1.0)
RESET_LEFT_ARM_DEG = (17.7, 22.5, -6.6, -106.8, -18.6, -17.1, 0.0)
RESET_RIGHT_ARM_DEG = (-18.8, -29.5, 9.6, 103.0, 23.9, 19.3, -4.0)
L1 = 0.375
L2 = 0.365
L3 = 0.0
CARTESIAN_Y_OFFSET = L1 + L2 - 0.000001
DEFAULT_CAMERA_SERIAL = "347522072040"
DEFAULT_EXTRINSIC_YAML = ROOT / "config" / "realsense_347522072040_extrin.yaml"
O7_RS485_SCRIPT = ROOT.parent / "linkhand" / "linker_hand_python_sdk" / "example" / "o7_rs485_control.py"
LEFT_GRIPPER_SCRIPT = ROOT / "omni_picker_rs485_test.py"
O7_RS485_PORT = "/dev/ttyS3"
LEFT_GRIPPER_PORT = "/dev/ttyS2"


def clamp_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def fmt(value: Optional[float], digits: int = 4) -> str:
    return "--" if value is None else f"{value:+.{digits}f}"


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rotation_matrix_to_rpy(rotation: "np.ndarray") -> tuple[float, float, float]:
    """将 3x3 旋转矩阵转换成机器人接口使用的 RPY 弧度。"""
    import numpy as np

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


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> "np.ndarray":
    import numpy as np

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def forward_kinematics(ankle: float, knee: float, hip_pitch: float) -> tuple[float, float, float]:
    theta12 = ankle + knee
    theta123 = theta12 + hip_pitch
    x = L1 * math.sin(ankle) + L2 * math.sin(theta12) + L3 * math.sin(theta123)
    y = L1 * math.cos(ankle) + L2 * math.cos(theta12) + L3 * math.cos(theta123)
    return x, y, theta123


def transform_point(mat4: "np.ndarray", point: tuple[float, float, float]) -> tuple[float, float, float]:
    import numpy as np

    vec = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    out = mat4 @ vec
    return float(out[0]), float(out[1]), float(out[2])


def transform_matrix_from_pose(
    tx: float,
    ty: float,
    tz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> "np.ndarray":
    import numpy as np

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = [tx, ty, tz]
    return out


@dataclass
class RobotSnapshot:
    joint_state: dict[str, float] = field(default_factory=dict)
    arm_power: Optional[tuple[bool, tuple[float, ...]]] = None
    leg_power: Optional[tuple[bool, tuple[float, ...]]] = None
    arm_motion: Optional[tuple[bool, bool]] = None
    leg_motion: Optional[tuple[bool, bool]] = None
    left_ee: Optional[Pose] = None
    right_ee: Optional[Pose] = None
    car_from_body: Optional[object] = None
    joint_update: float = 0.0
    tcp_update: float = 0.0
    last_update: float = 0.0

    def arm_joint_values(self, names: tuple[str, ...]) -> tuple[Optional[float], ...]:
        return tuple(self.joint_state.get(name) for name in names)

    def leg_pose(self) -> dict[str, float]:
        if all(name in self.joint_state for name in LEG_JOINTS):
            x, y, phi = forward_kinematics(
                self.joint_state["ankle_joint"],
                self.joint_state["knee_joint"],
                self.joint_state["hip_pitch_joint"],
            )
            return {"x": x, "y": y, "phi": phi, "waist": self.joint_state["hip_yaw_joint"]}
        return {}


class RobotBackend(QtCore.QObject):
    log_line = QtCore.pyqtSignal(str)
    status_ready = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._queue: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()
        self._snapshot = RobotSnapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        
    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def snapshot(self) -> RobotSnapshot:
        with self._lock:
            return RobotSnapshot(
                joint_state=dict(self._snapshot.joint_state),
                arm_power=self._snapshot.arm_power,
                leg_power=self._snapshot.leg_power,
                arm_motion=self._snapshot.arm_motion,
                leg_motion=self._snapshot.leg_motion,
                left_ee=self._snapshot.left_ee,
                right_ee=self._snapshot.right_ee,
                car_from_body=self._snapshot.car_from_body,
                joint_update=self._snapshot.joint_update,
                tcp_update=self._snapshot.tcp_update,
                last_update=self._snapshot.last_update,
            )

    def send(self, action: str, **payload: Any) -> None:
        self._queue.put((action, payload))

    def _set_snapshot(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._snapshot, key, value)
            self._snapshot.last_update = time.time()

    def _run(self) -> None:
        try:
            rclpy.init()
            node = RobotDebugNode(self)
        except Exception as exc:
            self.log_line.emit(f"[ROS2] init failed: {exc}")
            return

        self.log_line.emit("[ROS2] backend ready")
        self.status_ready.emit()
        try:
            while rclpy.ok() and not self._stop.is_set():
                self._drain_queue(node)
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
            self.log_line.emit("[ROS2] backend stopped")

    def _drain_queue(self, node: "RobotDebugNode") -> None:
        while True:
            try:
                action, payload = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_action(node, action, payload)
            except Exception as exc:
                self.log_line.emit(f"[ERR] {action}: {exc}")

    def _handle_action(self, node: "RobotDebugNode", action: str, payload: dict[str, Any]) -> None:
        if action == "power":
            enable = bool(payload["enable"])
            self.log_line.emit(f"[CMD] power {'on' if enable else 'off'}")
            node.set_robot_power(enable)
            return
        if action == "arm_mode":
            mode = int(payload["mode"])
            self.log_line.emit(f"[CMD] arm mode -> {mode}")
            node.set_arm_mode(mode)
            return
        if action == "arm_joint":
            self.log_line.emit("[CMD] arm joint target sent")
            node.publish_arm_joint(
                payload["left"],
                payload["right"],
                vel=float(payload["vel"]),
                acc=float(payload["acc"]),
            )
            return
        if action == "arm_cartesian":
            self.log_line.emit("[CMD] arm cartesian target sent")
            node.publish_arm_cartesian(payload["left"], payload["right"], vel=float(payload["vel"]), acc=float(payload["acc"]))
            return
        if action == "leg_joint":
            self.log_line.emit("[CMD] leg joint target sent")
            node.publish_leg_joint(payload["values"], vel=float(payload["vel"]), acc=float(payload["acc"]))
            return
        if action == "leg_lift":
            self.log_line.emit("[CMD] leg lift target sent")
            node.publish_leg_lift(float(payload["delta"]), float(payload["waist_delta"]), float(payload["vel"]))
            return
        if action == "leg_cartesian":
            self.log_line.emit("[CMD] leg cartesian target sent")
            node.publish_leg_cartesian(
                float(payload["x"]),
                float(payload["y"]),
                float(payload["phi"]),
                float(payload["waist"]),
                float(payload["vel"]),
            )
            return
        if action == "base_cmd":
            self.log_line.emit("[CMD] base cmd sent")
            node.publish_base_cmd(float(payload["linear_x"]), float(payload["angular_z"]))
            return
        raise ValueError(f"unknown action: {action}")


class RobotDebugNode(Node):
    def __init__(self, backend: RobotBackend) -> None:
        super().__init__("robot_debug_console")
        self._backend = backend
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self.create_subscription(JointState, "/whole/joint_states", self._on_joint_state, 10)
        self.create_subscription(ArmMotionStatus, "/whole/arm_controller/motion_status", self._on_arm_motion, 10)
        self.create_subscription(LegMotionStatus, "/whole/leg_controller/motion_status", self._on_leg_motion, 10)
        self.create_subscription(ArmPowerStatus, "/whole/robot/status/arm_power", self._on_arm_power, 10)
        self.create_subscription(ArmPowerStatus, "/robot/status/arm_power", self._on_arm_power, 10)
        self.create_subscription(LegPowerStatus, "/whole/robot/status/leg_power", self._on_leg_power, 10)
        self.create_subscription(EndEffectorPose, "/arm_tcp_pose", self._on_end_effector_pose, 10)

        self._power_pub = self.create_publisher(Bool, "/robot_poweron", 10)
        self._arm_mode_pub = self.create_publisher(UInt8, "/whole/control_mode_cmd", 10)
        self._arm_joint_pub = self.create_publisher(Robotarmjoint, "/arm_joint_absolute_cmd", 10)
        self._arm_axis_pub = self.create_publisher(Robotarmservomsg, "/arm_axis_position_cmd", 10)
        self._leg_joint_pub = self.create_publisher(Robotlegjoint, "/leg_joint_position_cmd", 10)
        self._leg_axis_pub = self.create_publisher(Robotservomsg, "/axis_position_cmd", 10)
        self._base_pub = self.create_publisher(Twist, "/car_cmd_vel", 10)

        self._power_client = self.create_client(SetRobotPower, "/set_robot_power")
        self._arm_joint_client = self.create_client(JointAbsoluteControl, "/arm_absolute_control")
        self._arm_cart_clients = [
            ("/cartesian_absolute_control", self.create_client(CartesianAbsoluteControl, "/cartesian_absolute_control")),
            ("/whole/cartesian_absolute_control", self.create_client(CartesianAbsoluteControl, "/whole/cartesian_absolute_control")),
        ]
        self._leg_abs_client = self.create_client(LegAbsoluteControl, "/leg_absolute_control")
        self._leg_cart_client = self.create_client(LegCartesianControl, "/leg_cartesian_control")
        self.create_timer(0.5, self._update_car_from_body)

    def _on_joint_state(self, msg: JointState) -> None:
        data = dict(self._backend.snapshot().joint_state)
        for name, value in zip(msg.name, msg.position):
            data[name] = float(value)
        self._backend._set_snapshot(joint_state=data, joint_update=time.monotonic())

    def _on_arm_motion(self, msg: ArmMotionStatus) -> None:
        self._backend._set_snapshot(arm_motion=(bool(msg.is_moving), bool(msg.goal_reached)))

    def _on_leg_motion(self, msg: LegMotionStatus) -> None:
        self._backend._set_snapshot(leg_motion=(bool(msg.is_moving), bool(msg.goal_reached)))

    def _on_arm_power(self, msg: ArmPowerStatus) -> None:
        self._backend._set_snapshot(arm_power=(bool(msg.is_enabled), tuple(float(v) for v in msg.motor_status)))

    def _on_leg_power(self, msg: LegPowerStatus) -> None:
        self._backend._set_snapshot(leg_power=(bool(msg.is_enabled), tuple(float(v) for v in msg.motor_status)))

    def _on_end_effector_pose(self, msg: EndEffectorPose) -> None:
        self._backend._set_snapshot(left_ee=msg.left_ee_pose, right_ee=msg.right_ee_pose, tcp_update=time.monotonic())

    def _update_car_from_body(self) -> None:
        try:
            transform = self._tf_buffer.lookup_transform("car_link", "body_link", rclpy.time.Time())
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self._backend._set_snapshot(
                car_from_body=transform_matrix_from_pose(
                    float(translation.x),
                    float(translation.y),
                    float(translation.z),
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                )
            )
        except Exception:
            return

    def set_robot_power(self, enable: bool) -> None:
        req = SetRobotPower.Request()
        req.enable = bool(enable)
        if self._power_client.wait_for_service(timeout_sec=2.0):
            future = self._power_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None:
                result = future.result()
                self._backend.log_line.emit(f"[SRV] /set_robot_power -> {result.success} {result.message}")
                return
        msg = Bool()
        msg.data = bool(enable)
        self._power_pub.publish(msg)

    def set_arm_mode(self, mode: int) -> None:
        msg = UInt8()
        msg.data = int(mode)
        self._arm_mode_pub.publish(msg)

    def publish_arm_joint(self, left: tuple[float, ...], right: tuple[float, ...], vel: float, acc: float) -> None:
        if self._arm_joint_client.wait_for_service(timeout_sec=0.2):
            req = JointAbsoluteControl.Request()
            for i, value in enumerate(left, start=1):
                setattr(req, f"ljoint{i}", float(value))
            for i, value in enumerate(right, start=1):
                setattr(req, f"rjoint{i}", float(value))
            req.vel = float(vel)
            req.acc = float(acc)
            future = self._arm_joint_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            if future.done() and future.result() is not None:
                result = future.result()
                self._backend.log_line.emit(f"[SRV] arm_absolute_control -> {result.success} {result.message}")
                return
        msg = Robotarmjoint()
        for i, value in enumerate(left, start=1):
            setattr(msg, f"ljoint{i}", float(value))
        for i, value in enumerate(right, start=1):
            setattr(msg, f"rjoint{i}", float(value))
        msg.vel = float(vel)
        msg.acc = float(acc)
        self._arm_joint_pub.publish(msg)

    def publish_arm_cartesian(self, left: tuple[float, ...], right: tuple[float, ...], vel: float, acc: float) -> None:
        req = CartesianAbsoluteControl.Request()
        left_xyzrpy = tuple(left)
        right_xyzrpy = tuple(right)
        for prefix, values in (("l", left_xyzrpy), ("r", right_xyzrpy)):
            x, y, z, roll, pitch, yaw = values
            setattr(req, f"{prefix}x", float(x))
            setattr(req, f"{prefix}y", float(y))
            setattr(req, f"{prefix}z", float(z))
            setattr(req, f"{prefix}roll", float(roll))
            setattr(req, f"{prefix}pitch", float(pitch))
            setattr(req, f"{prefix}yaw", float(yaw))
            setattr(req, f"{prefix}qx", 0.0)
            setattr(req, f"{prefix}qy", 0.0)
            setattr(req, f"{prefix}qz", 0.0)
            # Zero quaternion fields selects the service's RPY input mode.
            setattr(req, f"{prefix}qw", 0.0)
        req.vel = float(vel)
        req.acc = float(acc)

        for service_name, client in self._arm_cart_clients:
            if not client.wait_for_service(timeout_sec=0.2):
                continue
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            if future.done() and future.result() is not None:
                result = future.result()
                self._backend.log_line.emit(f"[SRV] {service_name} -> {result.success} {result.message}")
                return
            raise RuntimeError(f"{service_name} returned no result")

        raise RuntimeError("cartesian absolute service not available: /cartesian_absolute_control or /whole/cartesian_absolute_control")

    def publish_leg_joint(self, values: tuple[float, ...], vel: float, acc: float) -> None:
        if self._leg_abs_client.wait_for_service(timeout_sec=0.2):
            req = LegAbsoluteControl.Request()
            req.ankle_joint, req.knee_joint, req.hip_pitch_joint, req.hip_yaw_joint = map(float, values)
            req.vel = float(vel)
            req.acc = float(acc)
            future = self._leg_abs_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            if future.done() and future.result() is not None:
                result = future.result()
                self._backend.log_line.emit(f"[SRV] leg_absolute_control -> {result.success} {result.message}")
                return
        msg = Robotlegjoint()
        msg.ankle_joint, msg.knee_joint, msg.hip_pitch_joint, msg.hip_yaw_joint = map(float, values)
        msg.vel = float(vel)
        msg.acc = float(acc)
        self._leg_joint_pub.publish(msg)

    def publish_leg_lift(self, delta: float, waist_delta: float, vel: float) -> None:
        snap = self._backend.snapshot()
        if not all(name in snap.joint_state for name in LEG_JOINTS):
            raise RuntimeError("missing leg joint states")
        current = snap.leg_pose()
        target_y = current["y"] + delta
        target_x = current["x"]
        target_phi = current["phi"]
        target_waist = snap.joint_state["hip_yaw_joint"] + waist_delta
        self.publish_leg_cartesian(target_x, target_y - CARTESIAN_Y_OFFSET, target_phi, target_waist, vel)

    def publish_leg_cartesian(self, x: float, y: float, phi: float, waist: float, vel: float) -> None:
        if self._leg_cart_client.wait_for_service(timeout_sec=0.2):
            req = LegCartesianControl.Request()
            req.x = float(x)
            req.y = float(y)
            req.phi = float(phi)
            req.hip_yaw_position = float(waist)
            req.vel = float(vel)
            req.mode_leg_select = 0.0
            future = self._leg_cart_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            if future.done() and future.result() is not None:
                result = future.result()
                self._backend.log_line.emit(f"[SRV] leg_cartesian_control -> {result.success} {result.message}")
                return
        msg = Robotservomsg()
        msg.run_mode = 1
        msg.x = float(x)
        msg.y = float(y)
        msg.phi = float(phi)
        msg.hip_yaw_position = float(waist)
        msg.vel = float(vel)
        self._leg_axis_pub.publish(msg)

    def publish_base_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._base_pub.publish(msg)


class JointPanel(QtWidgets.QGroupBox):
    def __init__(self, title: str, joint_names: tuple[str, ...]) -> None:
        super().__init__(title)
        self._current: dict[str, QtWidgets.QLabel] = {}
        self._target: dict[str, QtWidgets.QDoubleSpinBox] = {}
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        layout.addWidget(QtWidgets.QLabel("关节"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("当前"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("目标"), 0, 2)
        for row, name in enumerate(joint_names, start=1):
            layout.addWidget(QtWidgets.QLabel(name), row, 0)
            current = QtWidgets.QLabel("--")
            current.setMinimumWidth(100)
            layout.addWidget(current, row, 1)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.01)
            spin.setSuffix(" rad")
            layout.addWidget(spin, row, 2)
            self._current[name] = current
            self._target[name] = spin

    def set_current(self, values: tuple[Optional[float], ...]) -> None:
        for name, value in zip(self._current, values):
            self._current[name].setText(fmt(value))

    def target_values(self) -> tuple[float, ...]:
        return tuple(widget.value() for widget in self._target.values())

    def set_targets(self, values: tuple[float, ...]) -> None:
        for widget, value in zip(self._target.values(), values):
            widget.setValue(float(value))


class CartesianPanel(QtWidgets.QGroupBox):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._fields: dict[str, QtWidgets.QDoubleSpinBox] = {}
        grid = QtWidgets.QGridLayout(self)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        labels = ("x", "y", "z", "roll", "pitch", "yaw")
        for row, name in enumerate(labels):
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.01)
            if name in {"roll", "pitch", "yaw"}:
                spin.setSuffix(" rad")
            else:
                spin.setSuffix(" m")
            grid.addWidget(spin, row, 1)
            self._fields[name] = spin

    def values(self) -> tuple[float, float, float, float, float, float]:
        return (
            self._fields["x"].value(),
            self._fields["y"].value(),
            self._fields["z"].value(),
            self._fields["roll"].value(),
            self._fields["pitch"].value(),
            self._fields["yaw"].value(),
        )


class BaseTeleopPanel(QtWidgets.QGroupBox):
    def __init__(self, backend: RobotBackend) -> None:
        super().__init__("底盘键盘控制")
        self._backend = backend
        self._pressed: set[int] = set()
        self._last_active = False

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_InputMethodEnabled, False)

        layout = QtWidgets.QVBoxLayout(self)
        self.tip_label = QtWidgets.QLabel("点击本页后，按住 W/A/S/D 控制底盘，松开自动停止。")
        self.tip_label.setWordWrap(True)
        speed_row = QtWidgets.QHBoxLayout()
        self.linear_speed_spin = QtWidgets.QDoubleSpinBox()
        self.linear_speed_spin.setRange(0.01, 1.00)
        self.linear_speed_spin.setDecimals(2)
        self.linear_speed_spin.setSingleStep(0.05)
        self.linear_speed_spin.setValue(0.20)
        self.linear_speed_spin.setSuffix(" m/s")
        self.linear_speed_spin.setToolTip("W/S 前进后退速度")
        self.angular_speed_spin = QtWidgets.QDoubleSpinBox()
        self.angular_speed_spin.setRange(0.05, 3.00)
        self.angular_speed_spin.setDecimals(2)
        self.angular_speed_spin.setSingleStep(0.05)
        self.angular_speed_spin.setValue(0.70)
        self.angular_speed_spin.setSuffix(" rad/s")
        self.angular_speed_spin.setToolTip("A/D 原地转向速度")
        speed_row.addWidget(QtWidgets.QLabel("前进速度"))
        speed_row.addWidget(self.linear_speed_spin)
        speed_row.addWidget(QtWidgets.QLabel("转向速度"))
        speed_row.addWidget(self.angular_speed_spin)
        speed_row.addStretch(1)
        self.state_label = QtWidgets.QLabel("当前：停止")
        self.state_label.setWordWrap(True)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.log_view.setMaximumHeight(120)
        layout.addWidget(self.tip_label)
        layout.addLayout(speed_row)
        layout.addWidget(self.state_label)
        layout.addWidget(self.log_view)
        layout.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _pressed_names(self) -> str:
        mapping = {
            QtCore.Qt.Key_W: "W",
            QtCore.Qt.Key_A: "A",
            QtCore.Qt.Key_S: "S",
            QtCore.Qt.Key_D: "D",
        }
        names = [mapping[key] for key in sorted(self._pressed) if key in mapping]
        return "+".join(names) if names else "--"

    def _publish(self, linear_x: float, angular_z: float) -> None:
        self._backend.send("base_cmd", linear_x=linear_x, angular_z=angular_z)

    def _tick(self) -> None:
        forward = (QtCore.Qt.Key_W in self._pressed) - (QtCore.Qt.Key_S in self._pressed)
        turn = (QtCore.Qt.Key_A in self._pressed) - (QtCore.Qt.Key_D in self._pressed)
        linear_x = float(self.linear_speed_spin.value()) * float(forward)
        angular_z = float(self.angular_speed_spin.value()) * float(turn)
        active = bool(forward or turn)
        if active:
            self._publish(linear_x, angular_z)
        elif self._last_active:
            self._publish(0.0, 0.0)
        self._last_active = active
        self.state_label.setText(
            f"当前：{self._pressed_names()} | linear.x={linear_x:+.2f} m/s | angular.z={angular_z:+.2f} rad/s"
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() in {QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D}:
            self._pressed.add(int(event.key()))
            self._tick()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() in {QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D}:
            self._pressed.discard(int(event.key()))
            self._tick()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # type: ignore[override]
        self._pressed.clear()
        self._last_active = False
        self._publish(0.0, 0.0)
        self.state_label.setText("当前：停止")
        super().focusOutEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus()
        super().mousePressEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, backend: RobotBackend) -> None:
        super().__init__()
        self._backend = backend
        self.setWindowTitle("Robot Debug Console")
        self.resize(1280, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setMaximumWidth(560)
        left_outer = QtWidgets.QVBoxLayout(self.left_panel)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("获取状态")
        self.power_on_btn = QtWidgets.QPushButton("上使能")
        self.power_off_btn = QtWidgets.QPushButton("下使能")
        self.teach_btn = QtWidgets.QPushButton("示教")
        self.teach_btn.setCheckable(True)
        self.refresh_btn.clicked.connect(self.refresh_state)
        self.power_on_btn.clicked.connect(lambda: self._backend.send("power", enable=True))
        self.power_off_btn.clicked.connect(lambda: self._backend.send("power", enable=False))
        self.teach_btn.toggled.connect(self._on_teach_toggled)
        for widget in (self.refresh_btn, self.power_on_btn, self.power_off_btn, self.teach_btn):
            top_row.addWidget(widget)
        top_row.addStretch(1)
        left_outer.addLayout(top_row)

        self.left_arm_input = QtWidgets.QLineEdit()
        self.right_arm_input = QtWidgets.QLineEdit()
        self.leg_input = QtWidgets.QLineEdit()
        self.left_cart_input = QtWidgets.QLineEdit()
        self.right_cart_input = QtWidgets.QLineEdit()
        self.lift_step = QtWidgets.QDoubleSpinBox()
        self.lift_step.setRange(0.0, 0.3)
        self.lift_step.setDecimals(3)
        self.lift_step.setSingleStep(0.01)
        self.lift_step.setValue(0.03)
        self.lift_step.setSuffix(" m")
        self.waist_delta = QtWidgets.QDoubleSpinBox()
        self.waist_delta.setRange(-90.0, 90.0)
        self.waist_delta.setDecimals(2)
        self.waist_delta.setSingleStep(1.0)
        self.waist_delta.setValue(0.0)
        self.waist_delta.setSuffix(" deg")
        self.left_arm_input.setPlaceholderText("7个角度, 空格或逗号, deg")
        self.right_arm_input.setPlaceholderText("7个角度, 空格或逗号, deg")
        self.leg_input.setPlaceholderText("4个角度, 空格或逗号, deg")
        self.left_cart_input.setPlaceholderText("x y z roll pitch yaw")
        self.right_cart_input.setPlaceholderText("x y z roll pitch yaw")

        self.status_box = QtWidgets.QGroupBox("状态")
        status_layout = QtWidgets.QFormLayout(self.status_box)
        status_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        status_layout.setFormAlignment(QtCore.Qt.AlignTop)
        status_layout.setHorizontalSpacing(8)
        status_layout.setVerticalSpacing(4)
        self.status_system = QtWidgets.QLabel("--")
        self.status_power = QtWidgets.QLabel("--")
        self.status_motion = QtWidgets.QLabel("--")
        self.status_body = QtWidgets.QLabel("--")
        self.status_arms = QtWidgets.QLabel("--")
        self.status_legs = QtWidgets.QLabel("--")
        self.status_ee = QtWidgets.QLabel("--")
        for label in (self.status_system, self.status_power, self.status_motion, self.status_body, self.status_arms, self.status_legs, self.status_ee):
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setWordWrap(True)
        status_layout.addRow("系统", self.status_system)
        status_layout.addRow("使能", self.status_power)
        status_layout.addRow("运动", self.status_motion)
        status_layout.addRow("身体", self.status_body)
        status_layout.addRow("手臂", self.status_arms)
        status_layout.addRow("腿腰", self.status_legs)
        status_layout.addRow("末端", self.status_ee)
        left_outer.addWidget(self.status_box)

        self._build_compact_rows(left_outer)

        self.right_placeholder = QtWidgets.QFrame()
        self.right_placeholder.setObjectName("rightPlaceholder")
        self.right_placeholder.setMinimumWidth(260)
        self.right_placeholder.setMaximumWidth(320)
        right_layout = QtWidgets.QVBoxLayout(self.right_placeholder)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.addWidget(QtWidgets.QLabel("右侧预留：相机 / 画面 / 其他信息"))
        right_layout.addStretch(1)

        right_column = QtWidgets.QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(8)
        right_column.addWidget(self.right_placeholder, 0, QtCore.Qt.AlignTop)
        right_column.addStretch(1)

        root.addWidget(self.left_panel, 1)
        root.addLayout(right_column, 0)
        root.setStretch(0, 1)
        root.setStretch(1, 0)

        self._backend.log_line.connect(self.append_log)
        self._backend.status_ready.connect(self.refresh_state)

        self.refresh_state()

        self.setStyleSheet(
            """
            QWidget { font-size: 12px; background: #f4f6fa; color: #1d2433; }
            QGroupBox {
                border: 1px solid #cfd8e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: white;
                font-weight: 600;
            }
            QGroupBox::title {
                color: #2a4f8f;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel { color: #1f2937; }
            #rightPlaceholder {
                background: white;
                border: 1px dashed #c7d2e4;
                border-radius: 8px;
            }
            QLineEdit, QDoubleSpinBox {
                background: white;
                color: #1f2937;
                border: 1px solid #c7d2e4;
                border-radius: 6px;
                padding: 5px;
                min-height: 24px;
            }
            QPushButton {
                background: #2f6fed;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: 600;
                min-width: 78px;
            }
            QPushButton:hover { background: #4a82f0; }
            QPushButton:checked { background: #d05a5a; }
            QTextEdit {
                background: white;
                color: #1f2937;
                border: 1px solid #c7d2e4;
                border-radius: 6px;
            }
            """
        )

    def _build_compact_rows(self, outer: QtWidgets.QVBoxLayout) -> None:
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        lift_row = QtWidgets.QHBoxLayout()
        self.lift_up_btn = QtWidgets.QPushButton("上升")
        self.lift_down_btn = QtWidgets.QPushButton("下降")
        self.leg_joint_btn = QtWidgets.QPushButton("发腿关节")
        self.first_layer_btn = QtWidgets.QPushButton("第一层高度")
        self.second_layer_btn = QtWidgets.QPushButton("第二层高度")
        self.arm_joint_btn = QtWidgets.QPushButton("发臂关节")
        self.arm_cart_btn = QtWidgets.QPushButton("发臂XYZ")
        self.lift_up_btn.clicked.connect(lambda: self._backend.send("leg_lift", delta=self.lift_step.value(), waist_delta=math.radians(self.waist_delta.value()), vel=0.08))
        self.lift_down_btn.clicked.connect(lambda: self._backend.send("leg_lift", delta=-self.lift_step.value(), waist_delta=math.radians(self.waist_delta.value()), vel=0.08))
        self.leg_joint_btn.clicked.connect(self._send_leg_joint)
        self.first_layer_btn.clicked.connect(self._send_first_layer_height)
        self.second_layer_btn.clicked.connect(self._send_second_layer_height)
        self.arm_joint_btn.clicked.connect(self._send_arm_joint)
        self.arm_cart_btn.clicked.connect(self._send_arm_cartesian)
        for widget in (QtWidgets.QLabel("步长"), self.lift_step, QtWidgets.QLabel("腰部偏移(deg)"), self.waist_delta, self.lift_up_btn, self.lift_down_btn, self.leg_joint_btn, self.first_layer_btn, self.second_layer_btn, self.arm_joint_btn, self.arm_cart_btn):
            lift_row.addWidget(widget)
        lift_row.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("左臂关节"), 0, 0)
        grid.addWidget(self.left_arm_input, 0, 1)
        grid.addWidget(QtWidgets.QLabel("右臂关节"), 1, 0)
        grid.addWidget(self.right_arm_input, 1, 1)
        grid.addWidget(QtWidgets.QLabel("腿/腰关节"), 2, 0)
        grid.addWidget(self.leg_input, 2, 1)
        grid.addWidget(QtWidgets.QLabel("左臂XYZRPY"), 3, 0)
        grid.addWidget(self.left_cart_input, 3, 1)
        grid.addWidget(QtWidgets.QLabel("右臂XYZRPY"), 4, 0)
        grid.addWidget(self.right_cart_input, 4, 1)
        grid.addWidget(QtWidgets.QLabel("腰部直上直下"), 5, 0)
        grid.addLayout(lift_row, 5, 1)
        outer.addLayout(grid)

        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(64)
        outer.addWidget(self.log_view)

    def _on_teach_toggled(self, checked: bool) -> None:
        self._backend.send("arm_mode", mode=1 if checked else 0)
        self.append_log(f"[UI] teach {'ON' if checked else 'OFF'}")

    def _send_arm_joint(self) -> None:
        snap = self._backend.snapshot()
        left = self._parse_deg_list_or_current(self.left_arm_input.text(), 7, snap.arm_joint_values(LEFT_ARM_JOINTS), "左臂")
        right = self._parse_deg_list_or_current(self.right_arm_input.text(), 7, snap.arm_joint_values(RIGHT_ARM_JOINTS), "右臂")
        self._backend.send("arm_joint", left=left, right=right, vel=0.30, acc=0.50)

    def _send_arm_cartesian(self) -> None:
        self._backend.send("arm_cartesian", left=self._parse_cartesian(self.left_cart_input.text()), right=self._parse_cartesian(self.right_cart_input.text()), vel=0.10, acc=0.20)

    def _send_leg_joint(self) -> None:
        snap = self._backend.snapshot()
        values = self._parse_deg_list_or_current(self.leg_input.text(), 4, snap.arm_joint_values(LEG_JOINTS), "腿/腰")
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)

    def _send_first_layer_height(self) -> None:
        values = tuple(math.radians(v) for v in (50.6, -103.1, 52.6, 0.0))
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)
        self.append_log("[CMD] first layer height -> +50.6 -103.1 +52.6 +0.0")

    def _send_second_layer_height(self) -> None:
        values = (0.0, 0.0, 0.0, 0.0)
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)
        self.append_log("[CMD] second layer height -> 0 0 0 0")

    def append_log(self, text: str) -> None:
        self.log_view.append(text)

    def _parse_deg_list(self, text: str, count: int) -> tuple[float, ...]:
        normalized = text.replace("，", " ").replace(",", " ")
        parts = [part.strip() for part in normalized.split() if part.strip()]
        if len(parts) != count:
            raise ValueError(f"需要 {count} 个角度")
        return tuple(math.radians(float(part)) for part in parts)

    def _parse_deg_list_or_current(
        self,
        text: str,
        count: int,
        current: tuple[Optional[float], ...],
        label: str,
    ) -> tuple[float, ...]:
        if text.strip():
            return self._parse_deg_list(text, count)
        if len(current) != count or any(value is None for value in current):
            raise ValueError(f"{label} 为空，且当前状态还没读全，无法自动补齐")
        return tuple(float(value) for value in current if value is not None)

    def _parse_cartesian(self, text: str) -> tuple[float, ...]:
        normalized = text.replace("，", " ").replace(",", " ")
        parts = [part.strip() for part in normalized.split() if part.strip()]
        if len(parts) != 6:
            raise ValueError("需要 6 个值: x,y,z,roll,pitch,yaw")
        return (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            math.radians(float(parts[3])),
            math.radians(float(parts[4])),
            math.radians(float(parts[5])),
        )

    def refresh_state(self) -> None:
        snap = self._backend.snapshot()
        arm_power = self._format_power(snap.arm_power)
        leg_power = self._format_power(snap.leg_power)
        arm_motion = self._format_motion(snap.arm_motion)
        leg_motion = self._format_motion(snap.leg_motion)
        pose = snap.leg_pose()
        body_text = f"x={fmt(pose.get('x'), 3)} y={fmt(pose.get('y'), 3)} phi={fmt(math.degrees(pose['phi']) if 'phi' in pose else None, 2)}deg" if pose else "--"
        self.status_system.setText("在线" if snap.last_update else "等待状态")
        self.status_power.setText(f"手臂 {arm_power} | 腿部 {leg_power}")
        self.status_motion.setText(f"手臂 {arm_motion} | 腿部 {leg_motion}")
        self.status_body.setText(body_text)
        self.status_arms.setText("左 " + self._plain_joint_text(snap.arm_joint_values(LEFT_ARM_JOINTS)) + " | 右 " + self._plain_joint_text(snap.arm_joint_values(RIGHT_ARM_JOINTS)))
        self.status_legs.setText(self._plain_joint_text(snap.arm_joint_values(LEG_JOINTS)))
        self.status_ee.setText(f"L {self._pose_text(snap.left_ee)} | R {self._pose_text(snap.right_ee)}")

    def _pose_text(self, pose: Optional[Pose]) -> str:
        if pose is None:
            return "--"
        return f"{pose.position.x:+.2f},{pose.position.y:+.2f},{pose.position.z:+.2f}"

    def _format_power(self, power: Optional[tuple[bool, tuple[float, ...]]]) -> str:
        if power is None:
            return "--"
        enabled, motors = power
        ready = enabled and motors and all(int(round(v)) == LEG_POWER_READY for v in motors[: len(LEG_JOINTS)])
        if not enabled:
            return "未使能"
        return "已使能(就绪)" if ready else "已使能"

    def _format_motion(self, motion: Optional[tuple[bool, bool]]) -> str:
        if motion is None:
            return "--"
        return "移动" if motion[0] else "静止"

    def _compact_joint_text(self, snap: RobotSnapshot) -> str:
        values = snap.arm_joint_values(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEG_JOINTS)
        degs = [fmt(math.degrees(value), 1) if value is not None else "--" for value in values]
        return ",".join(degs)

    def _plain_joint_text(self, values: tuple[Optional[float], ...]) -> str:
        return " ".join(fmt(math.degrees(value), 1) if value is not None else "--" for value in values)

    def closeEvent(self, event: QtCore.QCloseEvent) -> None:  # type: ignore[override]
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        self._backend.stop()
        QtWidgets.QApplication.quit()
        QtCore.QTimer.singleShot(300, lambda: os._exit(0))


class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object, object)
    data_ready = QtCore.pyqtSignal(object, object, object)
    camera_info_ready = QtCore.pyqtSignal(float, float, float, float, float, str)
    status_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 15) -> None:
        super().__init__()
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._stop_event = threading.Event()
        self._pipeline = None

    def stop(self) -> None:
        self._stop_event.set()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            import numpy as np
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = pipeline.start(config)
            self._pipeline = pipeline
            align = rs.align(rs.stream.color)
            colorizer = rs.colorizer()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_profile.get_intrinsics()
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self.camera_info_ready.emit(
                float(intrinsics.fx),
                float(intrinsics.fy),
                float(intrinsics.ppx),
                float(intrinsics.ppy),
                depth_scale,
                "camera_color_optical_frame",
            )
            self.status_line.emit(
                f"相机已连接 RealSense serial={self.serial or 'auto'} "
                f"{self.width}x{self.height}@{self.fps}"
            )

            for _ in range(10):
                if self._stop_event.is_set():
                    return
                pipeline.wait_for_frames(timeout_ms=2000)

            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    aligned = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    depth_m = depth_raw.astype(np.float32) * depth_scale
                    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    if color_image.size == 0 or depth_image.size == 0 or depth_m.size == 0:
                        continue
                    color_copy = color_image.copy()
                    self.frame_ready.emit(color_copy, depth_image.copy())
                    self.data_ready.emit(color_copy, depth_image.copy(), depth_m.copy())
                except RuntimeError as exc:
                    if not self._stop_event.is_set():
                        self.status_line.emit(f"相机取帧失败: {exc}")
        except Exception as exc:
            self.status_line.emit(f"相机启动失败: {exc}")
        finally:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
            self.finished.emit()


class DmpMainWindow(QtWidgets.QMainWindow):
    """Restored debug console with separate joint and Cartesian DMP pages."""

    def __init__(self, backend: RobotBackend) -> None:
        super().__init__()
        self._backend = backend
        self._record_dir = ROOT / "dmp_joint_data"
        self._model_dir = ROOT / "dmp_joint_models"
        self._cart_record_dir = ROOT / "dmp_cartesian_data"
        self._cart_model_dir = ROOT / "dmp_cartesian_models"
        for directory in (self._record_dir, self._model_dir, self._cart_record_dir, self._cart_model_dir):
            directory.mkdir(exist_ok=True)
        self._dmp_process: Optional[QtCore.QProcess] = None
        self._grasp_process: Optional[QtCore.QProcess] = None
        self._recording_active = False
        self._record_rows: list[list[float]] = []
        self._record_start = 0.0
        self._record_request_time = 0.0
        self._record_last_values: Optional[tuple[float, ...]] = None
        self._record_idle_since: Optional[float] = None
        self._cart_recording_active = False
        self._cart_record_rows: list[list[float]] = []
        self._cart_record_start = 0.0
        self._cart_record_request_time = 0.0
        self._cart_record_last_values: Optional[tuple[float, ...]] = None
        self._cart_record_idle_since: Optional[float] = None
        self._right_dplus_joint_timer: Optional[QtCore.QTimer] = None
        self._right_dplus_joint_pending = False
        self._right_dplus_joint_started_at = 0.0
        self._right_dplus_original_joints: Optional[tuple[float, ...]] = None
        self._camera_color_image = None
        self._camera_depth_m = None
        self._camera_intrinsics: Optional[tuple[float, float, float, float, float, str]] = None
        self._camera_to_base: Optional[object] = None
        self._yolo_model = None
        self._yolo_model_path: Optional[Path] = None
        self._last_vision_capture = None
        # 自定义 YOLO 数据集的原始图片目录，和识别结果图分开保存。
        self._yolo_training_image_dir = ROOT / "yolo_training_dataset" / "images"
        self._yolo_training_image_dir.mkdir(parents=True, exist_ok=True)
        # YOLO 最近一次识别出的目标，坐标保存于 car_link。
        self._vision_target: Optional[dict[str, Any]] = None
        # 抓取页最近一次计算出的预抓取 TCP 目标。
        self._grasp_target: Optional[tuple[float, ...]] = None
        # 抓取页最近一次计算出的实际接触 TCP 目标。
        self._grasp_contact_target: Optional[tuple[float, ...]] = None
        self._grasp_target_arm: Optional[str] = None
        self._grasp_position = None
        self._grasp_approach_axis = None
        self._grasp_contact_axis = None
        self._grasp_rotation = None
        self._grasp_pregrasp_distance_m: Optional[float] = None
        self._grasp_right_side_mode = False
        self._post_grasp_reset_timer: Optional[QtCore.QTimer] = None
        self._post_grasp_leg_lift_timer: Optional[QtCore.QTimer] = None
        self._post_grasp_reset_stage = 0
        self._post_grasp_reset_arm: Optional[str] = None
        self._post_grasp_reset_lift_target: Optional[tuple[float, ...]] = None
        self._post_grasp_reset_back_target: Optional[tuple[float, ...]] = None
        self._box_grasp_page = None
        self._pending_logs: list[str] = []
        self._load_extrinsic_transform()

        self.setWindowTitle("Robot Debug Console")
        self.resize(1280, 760)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setMinimumWidth(860)
        self.left_panel.setMaximumWidth(1040)
        outer = QtWidgets.QVBoxLayout(self.left_panel)
        outer.setContentsMargins(0, 0, 0, 0)

        top = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("获取状态")
        self.power_on_btn = QtWidgets.QPushButton("上使能")
        self.power_off_btn = QtWidgets.QPushButton("下使能")
        self.reset_btn = QtWidgets.QPushButton("复位机械臂")
        self.reset_full_btn = QtWidgets.QPushButton("复位机械臂+腰部")
        self.teach_btn = QtWidgets.QPushButton("示教")
        self.dmp_stop_btn = QtWidgets.QPushButton("停止DMP")
        self.dmp_stop_btn.setEnabled(False)
        self.teach_btn.setCheckable(True)
        self.refresh_btn.clicked.connect(self.refresh_state)
        self.power_on_btn.clicked.connect(lambda: self._backend.send("power", enable=True))
        self.power_off_btn.clicked.connect(lambda: self._backend.send("power", enable=False))
        self.reset_btn.clicked.connect(self._send_reset_arms)
        self.reset_full_btn.clicked.connect(self._send_reset)
        self.dmp_stop_btn.clicked.connect(self._stop_dmp)
        self.teach_btn.toggled.connect(lambda checked: self._backend.send("arm_mode", mode=1 if checked else 0))
        for widget in (self.refresh_btn, self.power_on_btn, self.power_off_btn, self.reset_btn, self.reset_full_btn, self.teach_btn, self.dmp_stop_btn):
            top.addWidget(widget)
        top.addStretch(1)
        outer.addLayout(top)

        self.status_box = QtWidgets.QGroupBox("状态")
        status = QtWidgets.QFormLayout(self.status_box)
        self.status_system = QtWidgets.QLabel("--")
        self.status_power = QtWidgets.QLabel("--")
        self.status_motion = QtWidgets.QLabel("--")
        self.status_body = QtWidgets.QLabel("--")
        self.status_arms = QtWidgets.QLabel("--")
        self.status_legs = QtWidgets.QLabel("--")
        self.status_ee = QtWidgets.QLabel("--")
        for label in (self.status_system, self.status_power, self.status_motion, self.status_body, self.status_arms, self.status_legs, self.status_ee):
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setWordWrap(True)
        status.addRow("系统", self.status_system)
        status.addRow("使能", self.status_power)
        status.addRow("运动", self.status_motion)
        status.addRow("身体", self.status_body)
        status.addRow("手臂", self.status_arms)
        status.addRow("腿腰", self.status_legs)
        status.addRow("TCP", self.status_ee)
        outer.addWidget(self.status_box)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_joint_page(), "关节 / 笛卡尔控制")
        self.tabs.addTab(self._build_base_teleop_page(), "底盘 WSAD")
        self.tabs.addTab(self._build_cartesian_page(), "DMP（关节 / 笛卡尔）")
        self.tabs.addTab(self._build_vision_page(), "视觉 YOLO")
        self.tabs.addTab(self._build_box_grasp_page(), "grasp box")
        self.tabs.addTab(self._build_grasp_page(), "瓶子抓取")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self.tabs, 1)
        root.addWidget(self.left_panel, 1)

        camera_box = QtWidgets.QGroupBox("RealSense 相机")
        camera_box.setMinimumWidth(360)
        camera_box.setMaximumWidth(520)
        camera_layout = QtWidgets.QVBoxLayout(camera_box)
        self.camera_status = QtWidgets.QLabel("相机启动中...")
        self.camera_status.setWordWrap(True)
        self.camera_color_label = self._make_camera_label("原图")
        self.camera_depth_label = self._make_camera_label("深度图")
        camera_layout.addWidget(self.camera_status)
        camera_layout.addWidget(QtWidgets.QLabel("原图"))
        camera_layout.addWidget(self.camera_color_label, 1)
        camera_layout.addWidget(QtWidgets.QLabel("深度图（伪彩，已对齐到原图）"))
        camera_layout.addWidget(self.camera_depth_label, 1)
        root.addWidget(camera_box, 0)

        self._backend.log_line.connect(self.append_log)
        self._backend.status_ready.connect(self.refresh_state)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._capture_samples)
        self._timer.start(50)
        self._start_camera()
        self._refresh_files()
        self.refresh_state()
        self.setStyleSheet(
            """
            QWidget { font-size: 12px; background: #f4f6fa; color: #1d2433; }
            QGroupBox { border: 1px solid #cfd8e6; border-radius: 8px; margin-top: 10px; padding-top: 10px; background: white; font-weight: 600; }
            QGroupBox::title { color: #2a4f8f; subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel { color: #1f2937; }
            #rightPlaceholder { background: white; border: 1px dashed #c7d2e4; border-radius: 8px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: white; color: #1f2937; border: 1px solid #c7d2e4; border-radius: 6px; padding: 5px; min-height: 24px; }
            QPushButton { background: #2f6fed; color: white; border: none; border-radius: 6px; padding: 6px 10px; font-weight: 600; min-width: 84px; }
            QPushButton:hover { background: #4a82f0; }
            QPushButton:checked { background: #d05a5a; }
            QTextEdit { background: white; color: #1f2937; border: 1px solid #c7d2e4; border-radius: 6px; }
            """
        )

    def _build_joint_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        grid = QtWidgets.QGridLayout()
        self.left_joint_input = QtWidgets.QLineEdit()
        self.right_joint_input = QtWidgets.QLineEdit()
        self.leg_input = QtWidgets.QLineEdit()
        self.left_cart_input = QtWidgets.QLineEdit()
        self.right_cart_input = QtWidgets.QLineEdit()
        self.left_cart_rel_input = QtWidgets.QLineEdit()
        self.right_cart_rel_input = QtWidgets.QLineEdit()
        self.left_joint_input.setPlaceholderText("7个角度，单位 deg；留空保持当前")
        self.right_joint_input.setPlaceholderText("7个角度，单位 deg；留空保持当前")
        self.leg_input.setPlaceholderText("4个角度，单位 deg")
        self.left_cart_input.setPlaceholderText("x y z roll pitch yaw，单位 m/rad")
        self.right_cart_input.setPlaceholderText("x y z roll pitch yaw，单位 m/rad")
        self.left_cart_rel_input.setPlaceholderText("x y z，单位 cm；姿态保持不变")
        self.right_cart_rel_input.setPlaceholderText("x y z，单位 cm；姿态保持不变")
        for row, label, widget in (
            (0, "左臂关节", self.left_joint_input),
            (1, "右臂关节", self.right_joint_input),
            (2, "腿/腰关节", self.leg_input),
            (3, "左臂 TCP", self.left_cart_input),
            (4, "右臂 TCP", self.right_cart_input),
            (5, "左臂相对XYZ", self.left_cart_rel_input),
            (6, "右臂相对XYZ", self.right_cart_rel_input),
        ):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        self.arm_joint_btn = QtWidgets.QPushButton("发臂关节")
        self.arm_cart_btn = QtWidgets.QPushButton("发臂TCP")
        self.arm_cart_rel_btn = QtWidgets.QPushButton("发相对XYZ")
        self.leg_joint_btn = QtWidgets.QPushButton("发腿关节")
        self.first_layer_btn = QtWidgets.QPushButton("第一层高度")
        self.second_layer_btn = QtWidgets.QPushButton("第二层高度")
        self.arm_joint_btn.clicked.connect(self._send_arm_joint)
        self.arm_cart_btn.clicked.connect(self._send_arm_cartesian)
        self.arm_cart_rel_btn.clicked.connect(self._send_arm_cartesian_relative)
        self.leg_joint_btn.clicked.connect(self._send_leg_joint)
        self.first_layer_btn.clicked.connect(self._send_first_layer_height)
        self.second_layer_btn.clicked.connect(self._send_second_layer_height)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.arm_joint_btn)
        buttons.addWidget(self.arm_cart_btn)
        buttons.addWidget(self.arm_cart_rel_btn)
        buttons.addWidget(self.leg_joint_btn)
        buttons.addWidget(self.first_layer_btn)
        buttons.addWidget(self.second_layer_btn)
        buttons.addStretch(1)
        grid.addLayout(buttons, 7, 1)
        layout.addLayout(grid)

        lift_box = QtWidgets.QGroupBox("腿部升降 / 全局高度")
        lift_layout = QtWidgets.QGridLayout(lift_box)
        self.lift_step = QtWidgets.QDoubleSpinBox()
        self.lift_step.setRange(0.0, 0.30)
        self.lift_step.setDecimals(3)
        self.lift_step.setSingleStep(0.01)
        self.lift_step.setValue(0.03)
        self.lift_step.setSuffix(" m")
        self.waist_delta = QtWidgets.QDoubleSpinBox()
        self.waist_delta.setRange(-90.0, 90.0)
        self.waist_delta.setDecimals(1)
        self.waist_delta.setSingleStep(1.0)
        self.waist_delta.setValue(0.0)
        self.waist_delta.setSuffix(" deg")
        self.lift_up_btn = QtWidgets.QPushButton("上升")
        self.lift_down_btn = QtWidgets.QPushButton("下降")
        self.lift_preview_label = QtWidgets.QLabel("当前高度 -- | 上升后 -- | 下降后 --")
        self.lift_preview_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lift_preview_label.setWordWrap(True)
        self.lift_up_btn.clicked.connect(lambda: self._send_leg_lift(+1.0))
        self.lift_down_btn.clicked.connect(lambda: self._send_leg_lift(-1.0))
        self.lift_step.valueChanged.connect(lambda _value: self._update_lift_preview())
        self.waist_delta.valueChanged.connect(lambda _value: self._update_lift_preview())
        lift_layout.addWidget(QtWidgets.QLabel("升降步长"), 0, 0)
        lift_layout.addWidget(self.lift_step, 0, 1)
        lift_layout.addWidget(QtWidgets.QLabel("腰部偏移"), 0, 2)
        lift_layout.addWidget(self.waist_delta, 0, 3)
        lift_layout.addWidget(self.lift_up_btn, 0, 4)
        lift_layout.addWidget(self.lift_down_btn, 0, 5)
        lift_layout.addWidget(self.lift_preview_label, 1, 0, 1, 6)
        layout.addWidget(lift_box)

        layout.addStretch(1)
        return page

    def _build_base_teleop_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.base_teleop_panel = BaseTeleopPanel(self._backend)
        layout.addWidget(self.base_teleop_panel)
        layout.addStretch(1)
        return page

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is not None and hasattr(self, "base_teleop_panel") and widget is self.base_teleop_panel.parentWidget():
            self.base_teleop_panel.setFocus()

    def _build_cartesian_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        info = QtWidgets.QLabel("记录当前手臂 TCP 坐标点：x y z roll pitch yaw，位置单位 m，姿态单位 rad。回放使用 /cartesian_path_absolute_control。")
        info.setWordWrap(True)
        layout.addWidget(info)

        joint_record_box = QtWidgets.QGroupBox("关节示教采集")
        joint_record = QtWidgets.QHBoxLayout(joint_record_box)
        self.record_start_btn = QtWidgets.QPushButton("开始采集关节")
        self.record_stop_btn = QtWidgets.QPushButton("停止并保存")
        self.record_stop_btn.setEnabled(False)
        self.record_status = QtWidgets.QLabel("未采集")
        self.record_start_btn.clicked.connect(self._start_joint_recording)
        self.record_stop_btn.clicked.connect(self._stop_joint_recording)
        joint_record.addWidget(self.record_start_btn)
        joint_record.addWidget(self.record_stop_btn)
        joint_record.addWidget(self.record_status)
        joint_record.addStretch(1)
        layout.addWidget(joint_record_box)

        joint_box = QtWidgets.QGroupBox("关节DMP")
        joint_dmp = QtWidgets.QGridLayout(joint_box)
        self.demo_combo = QtWidgets.QComboBox()
        self.model_combo = QtWidgets.QComboBox()
        self.arm_combo = QtWidgets.QComboBox()
        self.arm_combo.addItems(["left", "right"])
        self.weight_spin = QtWidgets.QSpinBox()
        self.weight_spin.setRange(10, 200)
        self.weight_spin.setValue(30)
        self.train_btn = QtWidgets.QPushButton("训练DMP")
        self.play_btn = QtWidgets.QPushButton("回放DMP")
        self.generalize_goal_input = QtWidgets.QLineEdit()
        self.generalize_goal_input.setPlaceholderText("7个终点角度，单位 deg")
        self.generalize_btn = QtWidgets.QPushButton("泛化到终点")
        self.refresh_files_btn = QtWidgets.QPushButton("刷新文件")
        self.train_btn.clicked.connect(self._train_joint)
        self.play_btn.clicked.connect(self._play_joint)
        self.generalize_btn.clicked.connect(self._generalize_joint)
        self.refresh_files_btn.clicked.connect(self._refresh_files)
        joint_dmp.addWidget(QtWidgets.QLabel("示教文件"), 0, 0)
        joint_dmp.addWidget(self.demo_combo, 0, 1, 1, 3)
        joint_dmp.addWidget(QtWidgets.QLabel("手臂"), 0, 4)
        joint_dmp.addWidget(self.arm_combo, 0, 5)
        joint_dmp.addWidget(QtWidgets.QLabel("权重"), 0, 6)
        joint_dmp.addWidget(self.weight_spin, 0, 7)
        joint_dmp.addWidget(self.train_btn, 0, 8)
        joint_dmp.addWidget(QtWidgets.QLabel("模型文件"), 1, 0)
        joint_dmp.addWidget(self.model_combo, 1, 1, 1, 7)
        joint_dmp.addWidget(self.play_btn, 1, 8)
        joint_dmp.addWidget(QtWidgets.QLabel("泛化终点"), 2, 0)
        joint_dmp.addWidget(self.generalize_goal_input, 2, 1, 1, 7)
        joint_dmp.addWidget(self.generalize_btn, 2, 8)
        joint_dmp.addWidget(self.refresh_files_btn, 3, 8)
        layout.addWidget(joint_box)
        record_box = QtWidgets.QGroupBox("TCP 示教采集")
        record = QtWidgets.QHBoxLayout(record_box)
        self.cart_record_arm_combo = QtWidgets.QComboBox()
        self.cart_record_arm_combo.addItems(["left", "right"])
        self.cart_record_start_btn = QtWidgets.QPushButton("开始采集 TCP")
        self.cart_record_stop_btn = QtWidgets.QPushButton("停止并保存")
        self.cart_record_stop_btn.setEnabled(False)
        self.cart_record_status = QtWidgets.QLabel("未采集")
        self.cart_record_start_btn.clicked.connect(self._start_cart_record)
        self.cart_record_stop_btn.clicked.connect(self._stop_cart_record)
        record.addWidget(QtWidgets.QLabel("手臂"))
        record.addWidget(self.cart_record_arm_combo)
        record.addWidget(self.cart_record_start_btn)
        record.addWidget(self.cart_record_stop_btn)
        record.addWidget(self.cart_record_status)
        record.addStretch(1)
        layout.addWidget(record_box)

        box = QtWidgets.QGroupBox("笛卡尔 TCP DMP")
        dmp = QtWidgets.QGridLayout(box)
        self.cart_demo_combo = QtWidgets.QComboBox()
        self.cart_model_combo = QtWidgets.QComboBox()
        self.cart_weight_spin = QtWidgets.QSpinBox()
        self.cart_weight_spin.setRange(10, 200)
        self.cart_weight_spin.setValue(30)
        self.cart_train_btn = QtWidgets.QPushButton("训练笛卡尔DMP")
        self.cart_play_btn = QtWidgets.QPushButton("回放到原终点")
        self.cart_goal_input = QtWidgets.QLineEdit()
        self.cart_goal_input.setPlaceholderText("可选目标：x y z roll pitch yaw，单位 m/rad")
        self.cart_generalize_btn = QtWidgets.QPushButton("泛化到目标")
        self.cart_refresh_btn = QtWidgets.QPushButton("刷新文件")
        self.cart_train_btn.clicked.connect(self._train_cart)
        self.cart_play_btn.clicked.connect(lambda: self._play_cart(None))
        self.cart_generalize_btn.clicked.connect(lambda: self._play_cart(self.cart_goal_input.text().strip()))
        self.cart_refresh_btn.clicked.connect(self._refresh_files)
        dmp.addWidget(QtWidgets.QLabel("示教文件"), 0, 0)
        dmp.addWidget(self.cart_demo_combo, 0, 1, 1, 5)
        dmp.addWidget(QtWidgets.QLabel("权重"), 0, 6)
        dmp.addWidget(self.cart_weight_spin, 0, 7)
        dmp.addWidget(self.cart_train_btn, 0, 8)
        dmp.addWidget(QtWidgets.QLabel("模型文件"), 1, 0)
        dmp.addWidget(self.cart_model_combo, 1, 1, 1, 7)
        dmp.addWidget(self.cart_play_btn, 1, 8)
        dmp.addWidget(QtWidgets.QLabel("目标"), 2, 0)
        dmp.addWidget(self.cart_goal_input, 2, 1, 1, 7)
        dmp.addWidget(self.cart_generalize_btn, 2, 8)
        dmp.addWidget(self.cart_refresh_btn, 3, 8)
        layout.addWidget(box)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(90)
        layout.addWidget(self.log_view)
        layout.addStretch(1)
        return page

    def _build_vision_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        info = QtWidgets.QLabel(
            "使用当前 RealSense 帧进行 YOLO-Seg 识别。坐标为 camera_color_optical_frame："
            "+X 向右、+Y 向下、+Z 向前，单位 m。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        control_box = QtWidgets.QGroupBox("目标识别")
        controls = QtWidgets.QGridLayout(control_box)
        self.vision_capture_btn = QtWidgets.QPushButton("截取当前图像")
        self.vision_save_training_btn = QtWidgets.QPushButton("保存训练图片")
        self.vision_model_combo = QtWidgets.QComboBox()
        WEIGHT_DIR = ROOT / "weight"
        model_choices = (
            ("best.pt（自定义 guazi）", WEIGHT_DIR / "best.pt"),
            ("best_xiangzi.pt（自定义 xiangzi）", WEIGHT_DIR / "best_xiangzi.pt"),
            ("yolo26n-seg.pt（COCO，含 bottle）", WEIGHT_DIR / "yolo26n-seg.pt"),
            ("ear.pt（ear）", WEIGHT_DIR / "ear.pt"),
            ("guazi）.pt）", WEIGHT_DIR / "guazi）.pt"),
        )
        for label, model_path in model_choices:
            if model_path.exists():
                self.vision_model_combo.addItem(label, str(model_path))
        self.vision_model_combo.setToolTip("切换模型后需要重新点击“识别并输出坐标”")
        self.vision_model_combo.currentIndexChanged.connect(self._on_vision_model_changed)
        self.vision_class_combo = QtWidgets.QComboBox()
        self.vision_class_combo.setEditable(True)
        self.vision_class_combo.addItems(["guazi", "xiangzi", "box", "bottle", "ear"])
        self.vision_class_combo.setCurrentText("guazi")
        self.vision_class_combo.setToolTip("可输入具体标签或 all；右手标签会自动强制右臂，其余标签按当前手臂抓取")
        self.vision_detect_btn = QtWidgets.QPushButton("识别并输出坐标")
        self.vision_capture_status = QtWidgets.QLabel("等待相机图像...")
        self.vision_capture_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.vision_capture_btn.clicked.connect(self._capture_vision_image)
        self.vision_save_training_btn.clicked.connect(self._save_training_image)
        self.vision_detect_btn.clicked.connect(self._detect_vision_target)
        controls.addWidget(QtWidgets.QLabel("识别模型"), 0, 0)
        controls.addWidget(self.vision_model_combo, 0, 1, 1, 4)
        controls.addWidget(self.vision_capture_btn, 1, 0)
        controls.addWidget(self.vision_save_training_btn, 1, 1)
        controls.addWidget(QtWidgets.QLabel("识别标签"), 1, 2)
        controls.addWidget(self.vision_class_combo, 1, 3)
        controls.addWidget(self.vision_detect_btn, 1, 4)
        controls.addWidget(self.vision_capture_status, 2, 0, 1, 5)
        training_info = QtWidgets.QLabel(
            f"训练图片目录: {self._yolo_training_image_dir}\n"
            "保存的是原始彩色图像，后续还需要用 Labelme、CVAT 或 Roboflow 标注分割轮廓。"
        )
        training_info.setWordWrap(True)
        controls.addWidget(training_info, 3, 0, 1, 5)
        layout.addWidget(control_box)

        result_box = QtWidgets.QGroupBox("识别结果")
        result_layout = QtWidgets.QVBoxLayout(result_box)
        self.vision_result_image = self._make_camera_label("识别结果图")
        self.vision_result_image.setMinimumSize(480, 300)
        self.vision_result_label = QtWidgets.QLabel("尚未识别")
        self.vision_result_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.vision_result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.vision_result_label.setWordWrap(True)
        result_layout.addWidget(self.vision_result_image, 1)
        result_layout.addWidget(self.vision_result_label)
        layout.addWidget(result_box)
        layout.addStretch(1)
        return page

    def _on_vision_model_changed(self, _index: int) -> None:
        """切换模型后丢弃旧模型和旧识别结果，避免混用类别表。"""
        self._yolo_model = None
        self._yolo_model_path = None
        self._vision_target = None
        self._grasp_target = None
        self._grasp_contact_target = None
        self._grasp_target_arm = None
        self._grasp_position = None
        self._grasp_approach_axis = None
        self._grasp_rotation = None
        self._grasp_pregrasp_distance_m = None
        model_path = str(self.vision_model_combo.currentData() or "")
        if hasattr(self, "vision_class_combo"):
            model_name = Path(model_path).name
            if model_name == "best.pt":
                default_class = "guazi"
            elif model_name == "best_xiangzi.pt":
                default_class = "xiangzi"
            elif "box" in model_name:
                default_class = "box"
            elif model_name == "ear.pt":
                default_class = "ear"
            else:
                default_class = "bottle"
            self.vision_class_combo.setCurrentText(default_class)
        if hasattr(self, "vision_result_label"):
            self.vision_result_label.setText("模型已切换，请重新识别")
        if hasattr(self, "grasp_result_label"):
            self.grasp_result_label.setText("请重新识别目标后生成抓取姿态")
        if hasattr(self, "_box_grasp_page") and self._box_grasp_page is not None:
            self._box_grasp_page.reset_from_vision("模型已切换，请重新识别箱体")
        self.append_log("[VISION] 模型已切换，等待重新加载")

    def _build_box_grasp_page(self) -> QtWidgets.QWidget:
        self._box_grasp_page = build_box_bimanual_grasp_page(self)
        return self._box_grasp_page

    def _build_grasp_page(self) -> QtWidgets.QWidget:
        """构造通用物体抓取页。

        根据视觉标签自动分流：
        1. bottle：沿用瓶身中上部、正面接近的抓取方式。
        2. guazi / xiangzi：取物体最右侧中部，从右侧水平接近。

        生成姿态不会移动机械臂；运动仍由两个独立按钮控制。
        """
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        info = QtWidgets.QLabel(
            "使用视觉 YOLO 页最近一次识别的目标。左手保持原来的通用抓取，"
            "只有右手从物体右侧沿 +Y 方向推进。TCP 坐标约定：X 为接近方向，Z 向上。"
            "生成姿态不会移动机械臂；运动按钮仍然分开控制。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        control_box = QtWidgets.QGroupBox("通用物体抓取接口")
        controls = QtWidgets.QGridLayout(control_box)
        self.grasp_arm_combo = QtWidgets.QComboBox()
        self.grasp_arm_combo.addItems(["left", "right"])
        self.grasp_arm_combo.setCurrentText("right")
        self.grasp_height_spin = QtWidgets.QDoubleSpinBox()
        self.grasp_height_spin.setRange(0.35, 0.75)
        self.grasp_height_spin.setSingleStep(0.05)
        self.grasp_height_spin.setValue(0.55)
        self.grasp_height_spin.setSuffix("  目标高度比例（通用）")
        self.grasp_forward_spin = QtWidgets.QDoubleSpinBox()
        self.grasp_forward_spin.setRange(0.03, 0.30)
        self.grasp_forward_spin.setSingleStep(0.01)
        self.grasp_forward_spin.setValue(0.20)
        self.grasp_forward_spin.setSuffix(" m")
        self.grasp_contact_gap_spin = QtWidgets.QDoubleSpinBox()
        self.grasp_contact_gap_spin.setRange(1.0, 20.0)
        self.grasp_contact_gap_spin.setSingleStep(1.0)
        self.grasp_contact_gap_spin.setValue(5.0)
        self.grasp_contact_gap_spin.setSuffix(" cm")
        self.grasp_contact_gap_spin.setToolTip("相对步长：向前/向左移动多少厘米")
        self.grasp_generate_btn = QtWidgets.QPushButton("生成预抓取姿态")
        self.grasp_dmp_pre_btn = QtWidgets.QPushButton("D+预")
        self.grasp_align_btn = QtWidgets.QPushButton("调整到抓取位置")
        self.grasp_forward_btn = QtWidgets.QPushButton("沿当前 TCP +X 前进")
        self.grasp_forward_btn.setToolTip("按当前末端姿态的局部 +X 方向推进")
        self.grasp_backward_btn = QtWidgets.QPushButton("向后移动")
        self.grasp_generate_btn.clicked.connect(self._generate_grasp_pose)
        self.grasp_dmp_pre_btn.clicked.connect(self._play_guazi_dmp_then_pregrasp)
        self.grasp_align_btn.clicked.connect(self._move_to_grasp_align)
        self.grasp_forward_btn.clicked.connect(self._move_to_grasp_front)
        self.grasp_backward_btn.clicked.connect(self._move_to_grasp_back)

        controls.addWidget(QtWidgets.QLabel("使用手臂"), 0, 0)
        controls.addWidget(self.grasp_arm_combo, 0, 1)
        controls.addWidget(QtWidgets.QLabel("抓取高度"), 0, 2)
        controls.addWidget(self.grasp_height_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("预抓取距离"), 1, 0)
        controls.addWidget(self.grasp_forward_spin, 1, 1)
        controls.addWidget(self.grasp_generate_btn, 1, 2)
        controls.addWidget(self.grasp_dmp_pre_btn, 1, 3)
        controls.addWidget(self.grasp_align_btn, 1, 4)
        controls.addWidget(QtWidgets.QLabel("相对前进距离"), 2, 0)
        controls.addWidget(self.grasp_contact_gap_spin, 2, 1)
        controls.addWidget(self.grasp_forward_btn, 2, 2)
        controls.addWidget(self.grasp_backward_btn, 2, 3, 1, 2)
        layout.addWidget(control_box)

        final_box = QtWidgets.QGroupBox("灵巧手独立开合")
        final_layout = QtWidgets.QGridLayout(final_box)
        self.final_o7_port_input = QtWidgets.QLineEdit(O7_RS485_PORT)
        self.final_gripper_port_input = QtWidgets.QLineEdit(LEFT_GRIPPER_PORT)
        self.final_o7_close_btn = QtWidgets.QPushButton("右手灵巧手抓取")
        self.final_o7_open_btn = QtWidgets.QPushButton("右手灵巧手张开")
        self.final_gripper_close_btn = QtWidgets.QPushButton("左手夹爪闭合")
        self.final_gripper_open_btn = QtWidgets.QPushButton("左手夹爪张开")
        self.final_right_lift_reset_btn = QtWidgets.QPushButton("右手抬起+复位")
        self.final_left_lift_reset_btn = QtWidgets.QPushButton("左手抬起+复位")
        self.final_right_place_btn = QtWidgets.QPushButton("右手放置")
        self.final_left_place_btn = QtWidgets.QPushButton("左手放置")
        self.final_action_status = QtWidgets.QLabel("等待你操作灵巧手")
        self.final_action_status.setWordWrap(True)
        self.final_o7_close_btn.clicked.connect(self._run_right_hand_grasp)
        self.final_o7_open_btn.clicked.connect(self._run_right_hand_open)
        self.final_gripper_close_btn.clicked.connect(self._run_left_gripper_close)
        self.final_gripper_open_btn.clicked.connect(self._run_left_gripper_open)
        self.final_right_lift_reset_btn.clicked.connect(lambda: self._lift_arm_then_reset("right"))
        self.final_left_lift_reset_btn.clicked.connect(lambda: self._lift_arm_then_reset("left"))
        self.final_right_place_btn.clicked.connect(self._run_right_place)
        self.final_left_place_btn.clicked.connect(self._run_left_place)
        final_layout.addWidget(QtWidgets.QLabel("右手灵巧手串口"), 0, 0)
        final_layout.addWidget(self.final_o7_port_input, 0, 1)
        final_layout.addWidget(self.final_o7_close_btn, 0, 2)
        final_layout.addWidget(self.final_o7_open_btn, 0, 3)
        final_layout.addWidget(self.final_right_lift_reset_btn, 0, 4)
        final_layout.addWidget(QtWidgets.QLabel("左手夹爪串口"), 1, 0)
        final_layout.addWidget(self.final_gripper_port_input, 1, 1)
        final_layout.addWidget(self.final_gripper_close_btn, 1, 2)
        final_layout.addWidget(self.final_gripper_open_btn, 1, 3)
        final_layout.addWidget(self.final_left_lift_reset_btn, 1, 4)
        final_layout.addWidget(self.final_right_place_btn, 2, 2)
        final_layout.addWidget(self.final_left_place_btn, 2, 3)
        final_layout.addWidget(self.final_action_status, 3, 0, 1, 5)
        layout.addWidget(final_box)

        result_box = QtWidgets.QGroupBox("抓取位姿结果")
        result_layout = QtWidgets.QVBoxLayout(result_box)
        self.grasp_result_label = QtWidgets.QLabel("请先在“视觉 YOLO”页识别目标")
        self.grasp_result_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.grasp_result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.grasp_result_label.setWordWrap(True)
        result_layout.addWidget(self.grasp_result_label)
        layout.addWidget(result_box)
        layout.addStretch(1)
        return page

    def _start_final_grasp_process(self, command: list[str], message: str) -> None:
        if self._grasp_process is not None and self._grasp_process.state() != QtCore.QProcess.NotRunning:
            raise ValueError("已有最终抓取任务在运行")
        self.append_log(message)
        process = QtCore.QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self.append_log(
                bytes(process.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()
            )
        )
        process.finished.connect(self._final_grasp_process_finished)
        process.errorOccurred.connect(lambda error: self._final_grasp_process_error(process, error))
        process.start()
        if not process.waitForStarted(2000):
            process.deleteLater()
            raise RuntimeError("最终抓取进程启动失败")
        self._grasp_process = process
        self._set_final_grasp_busy(True)

    def _set_final_grasp_busy(self, busy: bool) -> None:
        if hasattr(self, "final_o7_close_btn"):
            self.final_o7_close_btn.setEnabled(not busy)
        if hasattr(self, "final_o7_open_btn"):
            self.final_o7_open_btn.setEnabled(not busy)
        if hasattr(self, "final_gripper_close_btn"):
            self.final_gripper_close_btn.setEnabled(not busy)
        if hasattr(self, "final_gripper_open_btn"):
            self.final_gripper_open_btn.setEnabled(not busy)
        if hasattr(self, "final_right_place_btn"):
            self.final_right_place_btn.setEnabled(not busy)
        if hasattr(self, "final_left_place_btn"):
            self.final_left_place_btn.setEnabled(not busy)
        if hasattr(self, "final_action_status"):
            self.final_action_status.setText("运行中..." if busy else "等待你操作灵巧手")

    def _final_grasp_process_finished(self, code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        process = self._grasp_process
        self._grasp_process = None
        self._set_final_grasp_busy(False)
        self.append_log(f"[GRASP] finished exit={code}")
        if process is not None:
            process.deleteLater()

    def _final_grasp_process_error(self, process: QtCore.QProcess, error: QtCore.QProcess.ProcessError) -> None:
        self.append_log(f"[GRASP] process error={int(error)}: {process.errorString()}")
        if process.state() == QtCore.QProcess.NotRunning and self._grasp_process is process:
            self._grasp_process = None
            self._set_final_grasp_busy(False)

    def _stop_final_grasp_process(self) -> None:
        process = self._grasp_process
        if process is None:
            self._set_final_grasp_busy(False)
            return
        if process.state() != QtCore.QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1500)
        self._grasp_process = None
        self._set_final_grasp_busy(False)
        self.append_log("[GRASP] stopped by user")
        process.deleteLater()

    def _run_right_hand_grasp(self) -> None:
        port = self.final_o7_port_input.text().strip() or O7_RS485_PORT
        command = [
            sys.executable,
            "-u",
            str(O7_RS485_SCRIPT),
            "--port",
            port,
            "--hand_type",
            "right",
            "--once",
            "--preset",
            "fist",
            "--no-open-on-exit",
        ]
        self._start_final_grasp_process(command, f"[GRASP] 右手灵巧手抓取: port={port}")

    def _run_right_hand_open(self) -> None:
        port = self.final_o7_port_input.text().strip() or O7_RS485_PORT
        command = [
            sys.executable,
            "-u",
            str(O7_RS485_SCRIPT),
            "--port",
            port,
            "--hand_type",
            "right",
            "--once",
            "--preset",
            "open",
            "--no-open-on-exit",
        ]
        self._start_final_grasp_process(command, f"[GRASP] 右手灵巧手张开: port={port}")

    def _run_left_gripper_close(self) -> None:
        port = self.final_gripper_port_input.text().strip() or LEFT_GRIPPER_PORT
        command = [
            sys.executable,
            "-u",
            str(LEFT_GRIPPER_SCRIPT),
            "--port",
            port,
            "--close-once",
        ]
        self._start_final_grasp_process(command, f"[GRASP] 左手夹爪闭合: port={port}")

    def _run_left_gripper_open(self) -> None:
        port = self.final_gripper_port_input.text().strip() or LEFT_GRIPPER_PORT
        command = [
            sys.executable,
            "-u",
            str(LEFT_GRIPPER_SCRIPT),
            "--port",
            port,
            "--open-once",
        ]
        self._start_final_grasp_process(command, f"[GRASP] 左手夹爪张开: port={port}")

    def _lift_arm_then_reset(self, arm: str) -> None:
        """抓取完成后先抬高 5 cm，再沿 TCP -X 后退 10 cm，然后复位，最后延时腿部上升。"""
        snap = self._backend.snapshot()
        left = self._pose_xyzrpy(snap.left_ee, "左臂 TCP")
        right = self._pose_xyzrpy(snap.right_ee, "右臂 TCP")
        lift_delta = 0.05
        if arm == "left":
            lift_pose = (left[0], left[1], left[2] + lift_delta, left[3], left[4], left[5])
            back_pose = self._offset_pose_local(lift_pose, -0.10, 0.0, 0.0)
            left = lift_pose
        else:
            lift_pose = (right[0], right[1], right[2] + lift_delta, right[3], right[4], right[5])
            back_pose = self._offset_pose_local(lift_pose, -0.10, 0.0, 0.0)
            right = lift_pose
        self._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
        self._post_grasp_reset_stage = 0
        self._post_grasp_reset_arm = arm
        self._post_grasp_reset_lift_target = lift_pose
        self._post_grasp_reset_back_target = back_pose
        self.append_log(f"[GRASP] {arm} arm lift +5 cm, then back 10 cm, then reset")
        self.final_action_status.setText(f"{arm} 臂已发送抬高 5 cm，2.5 秒后后退 10 cm")
        if self._post_grasp_reset_timer is not None:
            self._post_grasp_reset_timer.stop()
            self._post_grasp_reset_timer.deleteLater()
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._complete_post_grasp_reset)
        timer.start(2500)
        self._post_grasp_reset_timer = timer

    def _start_post_grasp_leg_lift(self) -> None:
        if self._post_grasp_leg_lift_timer is not None:
            self._post_grasp_leg_lift_timer.stop()
            self._post_grasp_leg_lift_timer.deleteLater()
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._complete_post_grasp_leg_lift)
        timer.start(3000)
        self._post_grasp_leg_lift_timer = timer
        if hasattr(self, "final_action_status"):
            self.final_action_status.setText("复位完成，3 秒后腿部到达指定高度")

    def _complete_post_grasp_leg_lift(self) -> None:
        timer = self._post_grasp_leg_lift_timer
        self._post_grasp_leg_lift_timer = None
        if timer is not None:
            timer.deleteLater()
        target = (
            math.radians(29.7),
            math.radians(-60.2),
            math.radians(30.6),
            math.radians(-0.0),
        )
        self._backend.send("leg_joint", values=target, vel=0.08, acc=0.20)
        self.append_log("[GRASP] leg joint target -> +29.7 -60.2 +30.6 -0.0")
        if hasattr(self, "final_action_status"):
            self.final_action_status.setText("腿部已到达指定高度")

    def _complete_post_grasp_reset(self) -> None:
        timer = self._post_grasp_reset_timer
        self._post_grasp_reset_timer = None
        if timer is not None:
            timer.deleteLater()
        arm = self._post_grasp_reset_arm
        if self._post_grasp_reset_stage == 0 and arm in {"left", "right"} and self._post_grasp_reset_back_target is not None:
            snap = self._backend.snapshot()
            left = self._pose_xyzrpy(snap.left_ee, "左臂 TCP")
            right = self._pose_xyzrpy(snap.right_ee, "右臂 TCP")
            if arm == "left":
                left = self._post_grasp_reset_back_target
            else:
                right = self._post_grasp_reset_back_target
            self._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
            self._post_grasp_reset_stage = 1
            if hasattr(self, "final_action_status"):
                self.final_action_status.setText(f"{arm} 臂已后退 10 cm，准备复位")
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._complete_post_grasp_reset)
            timer.start(2500)
            self._post_grasp_reset_timer = timer
            return

        self._send_reset_arms()
        self._send_waist_reset_only()
        self._post_grasp_reset_stage = 0
        self._post_grasp_reset_arm = None
        self._post_grasp_reset_lift_target = None
        self._post_grasp_reset_back_target = None
        if hasattr(self, "final_action_status"):
            self.final_action_status.setText("已发送后退后复位命令（腰部已回正）")
        self._start_post_grasp_leg_lift()

    def _run_left_place(self) -> None:
        self._run_place("left", 30.0, -10.0, 0.0)

    def _run_right_place(self) -> None:
        self._run_place("right", 30.0, 10.0, 0.0)

    def _run_place(self, arm: str, dx_cm: float, dy_cm: float, dz_cm: float) -> None:
        snap = self._backend.snapshot()
        if arm == "left":
            pose = snap.left_ee
            label = "左手"
        else:
            pose = snap.right_ee
            label = "右手"
        if pose is None:
            raise ValueError(f"{label} TCP 状态不可用")
        current = self._pose_xyzrpy(pose, f"{label} TCP")
        target = self._offset_pose_local(current, dx_cm / 100.0, dy_cm / 100.0, dz_cm / 100.0)
        left = self._pose_xyzrpy(snap.left_ee, "左臂 TCP") if snap.left_ee is not None else None
        right = self._pose_xyzrpy(snap.right_ee, "右臂 TCP") if snap.right_ee is not None else None
        if arm == "left":
            left = target
        else:
            right = target
        if left is None or right is None:
            raise ValueError("左右臂 TCP 状态不可用")
        self._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
        self.append_log(f"[GRASP] {label}放置 relative=({dx_cm:+.0f},{dy_cm:+.0f},{dz_cm:+.0f})cm")
        if hasattr(self, "final_action_status"):
            self.final_action_status.setText(
                f"{label}放置已发送：相对位移 x={dx_cm:+.0f} cm, y={dy_cm:+.0f} cm, z={dz_cm:+.0f} cm"
            )

    def _send_waist_reset_only(self) -> None:
        """保持腿前三轴不变，仅将腰部 hip_yaw_joint 复位到 0。"""
        snap = self._backend.snapshot()
        current = snap.arm_joint_values(LEG_JOINTS)
        if len(current) != 4 or any(value is None for value in current):
            raise ValueError("当前腿/腰状态还没读全，无法只复位腰部")
        values = (
            float(current[0]),
            float(current[1]),
            float(current[2]),
            0.0,
        )
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)
        self.append_log("[CMD] waist reset only; ankle/knee/hip_pitch unchanged")

    def _offset_pose_local(self, pose: tuple[float, ...], dx: float, dy: float, dz: float) -> tuple[float, ...]:
        """按 TCP 局部坐标偏移一个绝对位姿。"""
        import numpy as np

        x, y, z, roll, pitch, yaw = pose
        rotation = rpy_to_rotation_matrix(roll, pitch, yaw)
        delta = rotation @ np.array([dx, dy, dz], dtype=np.float64)
        return (
            float(x + delta[0]),
            float(y + delta[1]),
            float(z + delta[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )

    def _offset_pose_car_link(self, pose: tuple[float, ...], dx: float, dy: float, dz: float) -> tuple[float, ...]:
        """按 car_link 坐标直接偏移一个绝对位姿。"""
        x, y, z, roll, pitch, yaw = pose
        return (
            float(x + dx),
            float(y + dy),
            float(z + dz),
            float(roll),
            float(pitch),
            float(yaw),
        )

    def _generate_grasp_pose(self) -> None:
        """根据目标标签生成预抓取 TCP 位姿，但不发送运动命令。"""
        import numpy as np

        target = self._vision_target
        if target is None:
            self.grasp_result_label.setText("请先在“视觉 YOLO”页点击“识别并输出坐标”")
            return
        try:
            snap = self._backend.snapshot()
            arm = self.grasp_arm_combo.currentText()
            current_pose = snap.left_ee if arm == "left" else snap.right_ee
            # /arm_tcp_pose 已经是 car_link 下的 TCP 位姿，不能再套 car_from_body。
            current_tcp = np.array(self._pose_xyzrpy(current_pose, f"{arm} TCP")[:3], dtype=np.float64)
            points_car_value = target.get("points_car")
            # 识别时 TF 可能尚未更新；生成抓取时用最新 TF 从已缓存的 body 点云重算。
            if points_car_value is None and target.get("points_base") is not None:
                latest_snap = self._backend.snapshot()
                if latest_snap.car_from_body is not None:
                    points_base = np.asarray(target["points_base"], dtype=np.float64)
                    points_base_h = np.column_stack(
                        (points_base, np.ones((len(points_base), 1), dtype=np.float64))
                    )
                    points_car_value = (
                        np.asarray(latest_snap.car_from_body, dtype=np.float64)
                        @ points_base_h.T
                    ).T[:, :3]
                    target["points_car"] = points_car_value
                    self.append_log("[GRASP] 识别时 car_link TF 未就绪，已用最新 TF 重算目标点云")
            if points_car_value is None:
                raise ValueError(
                    "没有 car_link 点云：请确认相机外参已加载，并等待 "
                    "car_link <- body_link TF 就绪后重新识别"
                )
            points_car = np.asarray(points_car_value, dtype=np.float64)
            if points_car.shape[0] < 20:
                raise ValueError("目标有效三维点太少")

            self._grasp_target_arm = arm
            # 左右手使用完全相同的抓取计算：目标点、水平接近向量和 TCP +X 都一致。
            self._generate_bottle_grasp_pose(
                target,
                points_car,
                current_tcp,
                side_offset_m=RIGHT_HAND_TCP_Y_OFFSET_M if arm == "right" else 0.0,
            )
        except Exception as exc:
            self._grasp_target = None
            self._grasp_contact_target = None
            self._grasp_target_arm = None
            self._grasp_right_side_mode = False
            self.grasp_result_label.setText(f"生成预抓取姿态失败: {exc}")
            self.append_log(f"[GRASP] 生成失败: {exc}")

    def _make_grasp_targets(
        self,
        grasp_position: "np.ndarray",
        x_axis: "np.ndarray",
        rotation: "np.ndarray",
        final_gap_m: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], "np.ndarray", "np.ndarray"]:
        """按统一距离生成预抓取和接近抓取目标。"""
        import numpy as np

        pregrasp_distance = float(self.grasp_forward_spin.value())
        final_gap_m = float(final_gap_m)
        if pregrasp_distance <= 0.10:
            raise ValueError("预抓取距离必须大于 0.10 m")
        if final_gap_m <= 0.0:
            raise ValueError("最终离目标距离必须大于 0")
        if final_gap_m >= pregrasp_distance:
            raise ValueError("最终离目标距离必须小于预抓取距离")
        x_axis = np.asarray(x_axis, dtype=np.float64)
        x_axis /= np.linalg.norm(x_axis)
        pregrasp_position = grasp_position - x_axis * pregrasp_distance
        # 第二阶段停在距离目标 final_gap_m 的位置，不直接闭合夹爪。
        forward_target_position = grasp_position - x_axis * final_gap_m
        roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
        pregrasp_target = (
            float(pregrasp_position[0]),
            float(pregrasp_position[1]),
            float(pregrasp_position[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )
        forward_target = (
            float(forward_target_position[0]),
            float(forward_target_position[1]),
            float(forward_target_position[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )
        return pregrasp_target, forward_target, pregrasp_position, forward_target_position

    def _refresh_grasp_contact_target(self) -> float:
        """按当前输入值，沿目标接近矢量重新计算最终目标点。"""
        import numpy as np

        if (
            self._grasp_position is None
            or self._grasp_rotation is None
            or self._grasp_pregrasp_distance_m is None
        ):
            raise ValueError("请先点击“生成预抓取姿态”")

        final_gap_m = float(self.grasp_contact_gap_spin.value()) / 100.0
        pregrasp_distance = float(self._grasp_pregrasp_distance_m)
        if final_gap_m <= 0.0:
            raise ValueError("最终离目标距离必须大于 0")
        if final_gap_m >= pregrasp_distance:
            raise ValueError("最终离目标距离必须小于预抓取距离")

        grasp_position = np.asarray(self._grasp_position, dtype=np.float64)
        rotation = np.asarray(self._grasp_rotation, dtype=np.float64)
        if self._grasp_right_side_mode:
            forward_target_position = grasp_position.copy()
            forward_target_position[1] -= final_gap_m
        else:
            x_axis = np.asarray(self._grasp_contact_axis, dtype=np.float64)
            forward_target_position = grasp_position - x_axis * final_gap_m
        roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
        self._grasp_contact_target = (
            float(forward_target_position[0]),
            float(forward_target_position[1]),
            float(forward_target_position[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )
        return final_gap_m

    def _generate_bottle_grasp_pose(
        self,
        target: dict[str, Any],
        points_car: "np.ndarray",
        current_tcp: "np.ndarray",
        side_offset_m: float = 0.0,
        right_side_mode: bool = False,
    ) -> None:
        """通用逻辑：抓目标中上部，沿当前水平接近方向运动。

        side_offset_m 是通用模式下 TCP 局部坐标系的 Y 偏移；右手侧向模式不使用它。
        """
        import numpy as np

        # 取瓶子 car_link 坐标系 Z 方向的 2%/98% 范围，降低深度离群点影响。
        z_min = float(np.percentile(points_car[:, 2], 2))
        z_max = float(np.percentile(points_car[:, 2], 98))
        bottle_height = z_max - z_min
        if bottle_height < 0.05:
            raise ValueError(f"估计瓶高异常: {bottle_height:.3f} m")

        # 瓶身中上部抓取：默认瓶底到瓶顶的 55% 高度。
        grasp_z = z_min + bottle_height * float(self.grasp_height_spin.value())
        grasp_position = np.array(
            [
                float(np.median(points_car[:, 0])),
                float(np.median(points_car[:, 1])),
                grasp_z,
            ],
            dtype=np.float64,
        )

        # TCP +X 指向瓶子；抓取瓶子时只沿水平面接近。
        x_axis = grasp_position - current_tcp
        x_axis[2] = 0.0
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            raise ValueError("机械臂 TCP 与瓶子水平距离太小，无法确定 +X 方向")
        x_axis /= x_norm

        # 你的 TCP 坐标定义：Z 向上，Y 由右手系计算。
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        roll, pitch, yaw = rotation_matrix_to_rpy(rotation)

        final_gap_m = float(self.grasp_contact_gap_spin.value()) / 100.0
        self._grasp_position = grasp_position.copy()
        self._grasp_approach_axis = x_axis.copy()
        self._grasp_rotation = rotation.copy()
        pregrasp_distance_m = float(self.grasp_forward_spin.value())
        self._grasp_right_side_mode = bool(right_side_mode)
        if right_side_mode:
            if pregrasp_distance_m <= 0.10:
                raise ValueError("预抓取距离必须大于 0.10 m")
            if final_gap_m <= 0.0:
                raise ValueError("最终离目标距离必须大于 0")
            if final_gap_m >= pregrasp_distance_m:
                raise ValueError("最终离目标距离必须小于预抓取距离")
            self._grasp_contact_axis = None
            self._grasp_pregrasp_distance_m = pregrasp_distance_m
            grasp_pose = (
                float(grasp_position[0]),
                float(grasp_position[1]),
                float(grasp_position[2]),
                float(roll),
                float(pitch),
                float(yaw),
            )
            pregrasp_position = self._offset_pose_car_link(grasp_pose, 0.0, -pregrasp_distance_m, 0.0)
            forward_target_position = self._offset_pose_car_link(grasp_pose, 0.0, -final_gap_m, 0.0)
            self._grasp_target = (
                float(pregrasp_position[0]),
                float(pregrasp_position[1]),
                float(pregrasp_position[2]),
                float(roll),
                float(pitch),
                float(yaw),
            )
            self._grasp_contact_target = (
                float(forward_target_position[0]),
                float(forward_target_position[1]),
                float(forward_target_position[2]),
                float(roll),
                float(pitch),
                float(yaw),
            )
        else:
            self._grasp_right_side_mode = False
            self._grasp_contact_axis = x_axis.copy()
            self._grasp_pregrasp_distance_m = pregrasp_distance_m
            self._grasp_target, self._grasp_contact_target, pregrasp_position, forward_target_position = (
                self._make_grasp_targets(grasp_position, x_axis, rotation, final_gap_m)
            )
        # 左右手主抓取计算相同；右手额外沿末端局部 -Y 向右平移，
        # 用于让夹爪中心而不是手掌中心对准物体。
        if abs(float(side_offset_m)) > 1e-9 and not right_side_mode:
            side_offset_m = float(side_offset_m)
            grasp_pose = (
                float(grasp_position[0]),
                float(grasp_position[1]),
                float(grasp_position[2]),
                float(roll),
                float(pitch),
                float(yaw),
            )
            grasp_pose = self._offset_pose_local(grasp_pose, 0.0, side_offset_m, 0.0)
            self._grasp_target = self._offset_pose_local(self._grasp_target, 0.0, side_offset_m, 0.0)
            self._grasp_contact_target = self._offset_pose_local(self._grasp_contact_target, 0.0, side_offset_m, 0.0)
            grasp_position = np.array(grasp_pose[:3], dtype=np.float64)
            pregrasp_position = np.array(self._grasp_target[:3], dtype=np.float64)
            forward_target_position = np.array(self._grasp_contact_target[:3], dtype=np.float64)
            self._grasp_position = grasp_position.copy()
        side_offset_text = (
            f"右手 TCP 本地 -Y 偏移: {side_offset_m * 100:+.1f} cm\n"
            if abs(float(side_offset_m)) > 1e-9 and not right_side_mode
            else ""
        )
        mode_text = (
            f"抓取模式: {self._grasp_target_arm or 'unknown'} 右手侧向逻辑\n"
            if right_side_mode
            else f"抓取模式: {self._grasp_target_arm or 'unknown'} 通用逻辑\n"
        )
        axis_text = "TCP +Y: 右手侧向前进方向\n" if right_side_mode else ""
        sequence_text = (
            f"操作顺序: 先到物体右侧外 {pregrasp_distance_m * 100:.0f} cm -> 再沿 TCP +Y 前进到距目标约 {final_gap_m * 100:.0f} cm"
            if right_side_mode
            else f"操作顺序: 先到目标前 {pregrasp_distance_m * 100:.0f} cm -> "
            f"再向前 {pregrasp_distance_m * 100 - final_gap_m * 100:.0f} cm，"
            f"最终距目标约 {final_gap_m * 100:.0f} cm"
        )
        self.grasp_result_label.setText(
            f"目标: {target['class_name']}  置信度: {target['confidence']:.2f}\n"
            f"目标高度: {bottle_height:.4f} m\n"
            f"抓取点 car_link XYZ: "
            f"{grasp_position[0]:+.4f} {grasp_position[1]:+.4f} {grasp_position[2]:+.4f} m\n"
            f"预抓取点 car_link XYZ: "
            f"{pregrasp_position[0]:+.4f} {pregrasp_position[1]:+.4f} {pregrasp_position[2]:+.4f} m\n"
            f"向前 {final_gap_m * 100:.0f} cm 后的位置 car_link XYZ: "
            f"{forward_target_position[0]:+.4f} {forward_target_position[1]:+.4f} {forward_target_position[2]:+.4f} m\n"
            f"识别位置 car_link XYZ: {grasp_position[0]:+.4f} {grasp_position[1]:+.4f} {grasp_position[2]:+.4f} m\n"
            f"预抓取距离: {float(self.grasp_forward_spin.value()) * 100:.1f} cm | 最终距离: {final_gap_m * 100:.1f} cm\n"
            f"{side_offset_text}"
            f"预抓取姿态 RPY: "
            f"{math.degrees(roll):+.2f} {math.degrees(pitch):+.2f} {math.degrees(yaw):+.2f} deg\n"
            f"TCP +X: {x_axis[0]:+.3f} {x_axis[1]:+.3f} {x_axis[2]:+.3f}\n"
            f"{axis_text}"
            f"{mode_text}"
            f"状态: 已生成，尚未移动\n"
            f"{sequence_text}"
        )
        self.append_log(
            f"[GRASP] {target['class_name']} 识别位置={grasp_position[0]:+.4f},{grasp_position[1]:+.4f},{grasp_position[2]:+.4f} "
            f"预抓取位置={pregrasp_position[0]:+.4f},{pregrasp_position[1]:+.4f},{pregrasp_position[2]:+.4f} "
            f"最终位置={forward_target_position[0]:+.4f},{forward_target_position[1]:+.4f},{forward_target_position[2]:+.4f} "
            f"预抓取距离={float(self.grasp_forward_spin.value()) * 100:.1f}cm 最终距离={final_gap_m * 100:.1f}cm"
        )
        self.append_log(
            f"[GRASP] 已生成 {target['class_name']} {self._grasp_target_arm or 'unknown'} 通用预抓取姿态"
        )

    def _generate_side_grasp_pose(
        self,
        target: dict[str, Any],
        points_car: "np.ndarray",
        arm: str,
    ) -> None:
        """右手沿 car_link +Y 从物体右侧推进，数据处理方式与左手共用。"""
        import numpy as np

        if arm != "right":
            raise ValueError("侧向抓取当前只支持右手")

        # 右手预抓取点固定在识别目标点的 car_link -Y 方向 10 cm。
        approach_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        side_label = "右侧"
        grasp_position = np.median(points_car, axis=0).astype(np.float64)

        # 固定为 car_link 轴向：+Y 是右手向物体推进方向，+Z 保持竖直。
        rotation = np.eye(3, dtype=np.float64)
        x_axis = rotation[:, 0]

        roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
        final_gap_m = float(self.grasp_contact_gap_spin.value()) / 100.0
        pregrasp_distance_m = RIGHT_HAND_PREGRASP_DISTANCE_M
        pregrasp_position = grasp_position - approach_axis * pregrasp_distance_m
        forward_target_position = grasp_position - approach_axis * final_gap_m
        if final_gap_m <= 0.0 or final_gap_m >= pregrasp_distance_m:
            raise ValueError(
                "右手最终离目标距离必须大于 0 且严格小于右侧预抓取距离 "
                f"(当前最终距离={final_gap_m * 100:.1f} cm, 预抓取距离={pregrasp_distance_m * 100:.1f} cm, "
                f"识别位置={grasp_position[0]:+.4f},{grasp_position[1]:+.4f},{grasp_position[2]:+.4f} "
                f"预抓取位置={pregrasp_position[0]:+.4f},{pregrasp_position[1]:+.4f},{pregrasp_position[2]:+.4f} "
                f"最终位置={forward_target_position[0]:+.4f},{forward_target_position[1]:+.4f},{forward_target_position[2]:+.4f})"
            )
        self._grasp_position = grasp_position.copy()
        self._grasp_right_side_mode = True
        self._grasp_contact_axis = approach_axis.copy()
        self._grasp_approach_axis = approach_axis.copy()
        self._grasp_rotation = rotation.copy()
        self._grasp_pregrasp_distance_m = pregrasp_distance_m
        self._grasp_target = (
            float(pregrasp_position[0]),
            float(pregrasp_position[1]),
            float(pregrasp_position[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )
        self._grasp_contact_target = (
            float(forward_target_position[0]),
            float(forward_target_position[1]),
            float(forward_target_position[2]),
            float(roll),
            float(pitch),
            float(yaw),
        )
        self.grasp_result_label.setText(
            f"目标: {target['class_name']}  置信度: {target['confidence']:.2f}\n"
            f"抓取模式: {arm} 臂{side_label}侧抓\n"
            f"侧面抓取点 car_link XYZ: "
            f"{grasp_position[0]:+.4f} {grasp_position[1]:+.4f} {grasp_position[2]:+.4f} m\n"
            f"预抓取点 car_link XYZ: "
            f"{pregrasp_position[0]:+.4f} {pregrasp_position[1]:+.4f} {pregrasp_position[2]:+.4f} m\n"
            f"向前 {final_gap_m * 100:.0f} cm 后的位置 car_link XYZ: "
            f"{forward_target_position[0]:+.4f} {forward_target_position[1]:+.4f} {forward_target_position[2]:+.4f} m\n"
            f"识别位置 car_link XYZ: {grasp_position[0]:+.4f} {grasp_position[1]:+.4f} {grasp_position[2]:+.4f} m\n"
            f"预抓取距离: {pregrasp_distance_m * 100:.1f} cm | 最终距离: {final_gap_m * 100:.1f} cm\n"
            f"末端姿态: 单位矩阵 | RPY: {math.degrees(roll):+.2f} {math.degrees(pitch):+.2f} {math.degrees(yaw):+.2f} deg\n"
            f"TCP +X: 向前，+Y: 向左，+Z: 向上\n"
            f"状态: 已生成，尚未移动\n"
            f"操作顺序: 先到物体右侧外 {pregrasp_distance_m * 100:.0f} cm -> 再沿 car_link +Y 前进到距目标约 {final_gap_m * 100:.0f} cm"
        )
        self.append_log(
            f"[GRASP] {arm} 识别位置={grasp_position[0]:+.4f},{grasp_position[1]:+.4f},{grasp_position[2]:+.4f} "
            f"预抓取位置={pregrasp_position[0]:+.4f},{pregrasp_position[1]:+.4f},{pregrasp_position[2]:+.4f} "
            f"最终位置={forward_target_position[0]:+.4f},{forward_target_position[1]:+.4f},{forward_target_position[2]:+.4f} "
            f"预抓取距离={pregrasp_distance_m * 100:.1f}cm 最终距离={final_gap_m * 100:.1f}cm"
        )
        self.append_log(f"[GRASP] 已生成 {target['class_name']} {arm} 侧抓姿态")

    def _send_grasp_target(self, target: tuple[float, ...], status: str) -> None:
        """只发送一个抓取阶段目标，不自动发送下一阶段。"""
        snap = self._backend.snapshot()
        left = self._pose_xyzrpy(snap.left_ee, "左臂 TCP")
        right = self._pose_xyzrpy(snap.right_ee, "右臂 TCP")
        target_arm = self._grasp_target_arm or self.grasp_arm_combo.currentText()
        if target_arm == "left":
            left = target
        else:
            right = target
        self._backend.send("arm_cartesian", left=left, right=right, vel=0.05, acc=0.10)
        self.grasp_result_label.setText(self.grasp_result_label.text() + f"\n状态: {status}")
        self.append_log(f"[GRASP] {status}")

    def _move_to_grasp_align(self) -> None:
        """移动到预抓取位姿：对准目标，但仍保持安全距离。"""
        if self._grasp_target is None:
            self.grasp_result_label.setText("请先点击“生成预抓取姿态”")
            return
        try:
            class_name = str((self._vision_target or {}).get("class_name", "")).strip().lower()
            arm = self._grasp_target_arm or self.grasp_arm_combo.currentText()
            pregrasp_cm = float(self._grasp_pregrasp_distance_m or 0.0) * 100.0
            self._send_grasp_target(
                self._grasp_target,
                f"已发送{arm}臂预备抓取点，当前位于 {class_name or '目标'} 外侧约 {pregrasp_cm:.0f} cm"
                if class_name in RIGHT_HAND_GRASP_LABELS
                else f"已发送调整姿态目标，当前位于目标外侧约 {pregrasp_cm:.0f} cm",
            )
            if arm == "right" and self._right_dplus_joint_pending:
                self._right_dplus_joint_pending = False
                self._apply_right_dplus_joint_pose()
        except Exception as exc:
            self.grasp_result_label.setText(f"调整到抓取位置失败: {exc}")
            self.append_log(f"[GRASP] 调整失败: {exc}")

    def _play_guazi_dmp_then_pregrasp(self) -> None:
        """播放 guazi 关节 DMP，成功结束后发送当前预抓取 TCP 位姿。"""
        model = self._model_dir / "guazi.npz"
        if not model.exists():
            self.grasp_result_label.setText(f"guazi DMP 模型不存在: {model}")
            return
        if self._grasp_target is None:
            self.grasp_result_label.setText("请先点击“生成预抓取姿态”")
            return
        self._right_dplus_joint_pending = self._grasp_target_arm == "right"
        self._right_dplus_joint_started_at = time.monotonic()
        self._right_dplus_original_joints = None
        if self._right_dplus_joint_pending:
            right_current = self._backend.snapshot().arm_joint_values(RIGHT_ARM_JOINTS)
            if len(right_current) != 7 or any(value is None for value in right_current):
                self.grasp_result_label.setText("右手D+预失败：执行前无法读取完整的右臂关节状态")
                self.append_log("[GRASP] right D+预 failed: original right joint state unavailable")
                self._right_dplus_joint_pending = False
                return
            self._right_dplus_original_joints = tuple(float(value) for value in right_current if value is not None)
            self.append_log(
                "[GRASP] right D+预 保存原始关节: "
                + " ".join(f"{math.degrees(value):+.1f}" for value in self._right_dplus_original_joints)
            )
        self._start_process(
            [sys.executable, str(ROOT / "scripts/joint_dmp_pipeline.py"), "play", "--model", str(model)],
            "[GRASP] D+预: play guazi DMP",
            preferred_model=model,
            on_success=self._move_to_grasp_align,
        )

    def _apply_right_dplus_joint_pose(self, current: Optional[tuple[float, ...]] = None, target: Optional[tuple[float, ...]] = None) -> None:
        snap = self._backend.snapshot()
        right_current = self._right_dplus_original_joints or snap.arm_joint_values(RIGHT_ARM_JOINTS)
        left_current = snap.arm_joint_values(LEFT_ARM_JOINTS)
        if len(right_current) != 7 or len(left_current) != 7:
            raise ValueError("当前关节状态不完整，无法执行右手 D+预后的关节变化")
        if any(value is None for value in right_current) or any(value is None for value in left_current):
            raise ValueError("当前关节状态不完整，无法执行右手 D+预后的关节变化")
        right_values = list(float(value) for value in right_current if value is not None)
        right_values[3] = math.radians(110.0)
        right_values[6] = math.radians(-4.7)
        try:
            self._backend.send(
                "arm_joint",
                left=tuple(float(value) for value in left_current if value is not None),
                right=tuple(right_values),
                vel=0.30,
                acc=0.50,
            )
            msg = (
                "[GRASP] right D+预 joint pose -> rjoint4=+110.0deg rjoint7=-4.7deg 成功"
                + " final_right=" + " ".join(f"{math.degrees(value):+.1f}" for value in right_values)
                + (f" target={target[0]:+.4f},{target[1]:+.4f},{target[2]:+.4f}" if target is not None else "")
                + (f" current={current[0]:+.4f},{current[1]:+.4f},{current[2]:+.4f}" if current is not None else "")
            )
            self.append_log(msg)
            if hasattr(self, "grasp_result_label"):
                self.grasp_result_label.setText(self.grasp_result_label.text() + "\n状态: 右手已切换到最终关节姿态")
        except Exception as exc:
            msg = (
                f"[GRASP] right D+预 joint pose -> rjoint4=+110.0deg rjoint7=-4.7deg 失败: {exc}"
                + (f" target={target[0]:+.4f},{target[1]:+.4f},{target[2]:+.4f}" if target is not None else "")
                + (f" current={current[0]:+.4f},{current[1]:+.4f},{current[2]:+.4f}" if current is not None else "")
            )
            self.append_log(msg)
            if hasattr(self, "grasp_result_label"):
                self.grasp_result_label.setText(self.grasp_result_label.text() + f"\n状态: 右手关节切换失败: {exc}")

    def _move_to_grasp_front(self) -> None:
        """按当前末端局部 +X 方向前进，不自动执行夹爪闭合。"""
        try:
            step_cm = float(self.grasp_contact_gap_spin.value())
            step_m = step_cm / 100.0
            class_name = str((self._vision_target or {}).get("class_name", "")).strip().lower()
            target = self._build_relative_forward_target(step_m)
            self._send_grasp_target(
                target,
                f"已沿当前 TCP +X 前进 {step_cm:.0f} cm，未闭合夹爪",
            )
        except Exception as exc:
            self.grasp_result_label.setText(f"发送预抓取位姿失败: {exc}")
            self.append_log(f"[GRASP] 发送失败: {exc}")

    def _build_relative_forward_target(self, step_m: float) -> tuple[float, ...]:
        """根据当前 TCP 状态生成沿局部 +X 的前进目标。"""
        snap = self._backend.snapshot()
        arm = self._grasp_target_arm or self.grasp_arm_combo.currentText()
        if arm == "left":
            pose = snap.left_ee
            if pose is None:
                raise ValueError("左臂当前没有 TCP 状态")
            current = self._pose_xyzrpy(pose, "左臂 TCP")
            target = self._offset_pose_local(current, step_m, 0.0, 0.0)
        else:
            pose = snap.right_ee
            if pose is None:
                raise ValueError("右臂当前没有 TCP 状态")
            current = self._pose_xyzrpy(pose, "右臂 TCP")
            target = self._offset_pose_local(current, step_m, 0.0, 0.0)
        self._grasp_contact_target = target
        return target

    def _move_to_grasp_back(self) -> None:
        """退回到当前生成的预抓取位姿，不自动执行张开动作。"""
        if self._grasp_target is None:
            self.grasp_result_label.setText("请先点击“生成预抓取姿态”")
            return
        try:
            class_name = str((self._vision_target or {}).get("class_name", "")).strip().lower()
            arm = self._grasp_target_arm or self.grasp_arm_combo.currentText()
            pregrasp_cm = float(self._grasp_pregrasp_distance_m or 0.0) * 100.0
            self._send_grasp_target(
                self._grasp_target,
                f"已发送{arm}臂后退目标，回到 {class_name or '目标'} 外侧约 {pregrasp_cm:.0f} cm 的预抓取位姿"
                if class_name in RIGHT_HAND_GRASP_LABELS
                else f"已发送{arm}臂后退目标，回到目标外侧预抓取位姿",
            )
        except Exception as exc:
            self.grasp_result_label.setText(f"发送后退位姿失败: {exc}")
            self.append_log(f"[GRASP] 后退失败: {exc}")

    def _load_extrinsic_transform(self) -> None:
        try:
            import numpy as np

            path = DEFAULT_EXTRINSIC_YAML
            if not path.exists():
                self.append_log(f"[VISION] 外参文件不存在: {path}")
                return
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            matrix = np.array(payload.get("extrinsic_matrix", []), dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"bad matrix shape: {matrix.shape}")
            if str(payload.get("type", "")).strip().lower() != "camera_to_base":
                self.append_log(f"[VISION] 外参类型不是 camera_to_base: {payload.get('type')}")
            self._camera_to_base = matrix
            self.append_log(f"[VISION] 已加载相机外参: {path}")
        except Exception as exc:
            self._camera_to_base = None
            self.append_log(f"[VISION] 读取相机外参失败: {exc}")

    def _make_camera_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(320, 210)
        label.setStyleSheet(
            "QLabel { background: #101318; color: #b7c0cf; border: 1px solid #536078; }"
        )
        return label

    def _start_camera(self) -> None:
        self._camera_thread = QtCore.QThread(self)
        self._camera_worker = CameraWorker(DEFAULT_CAMERA_SERIAL, width=640, height=480, fps=15)
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_thread.started.connect(self._camera_worker.run)
        self._camera_worker.frame_ready.connect(self._on_camera_frame)
        self._camera_worker.data_ready.connect(self._on_camera_data)
        self._camera_worker.camera_info_ready.connect(self._on_camera_info)
        self._camera_worker.status_line.connect(self._on_camera_status)
        self._camera_worker.finished.connect(self._camera_thread.quit)
        self._camera_worker.finished.connect(self._camera_worker.deleteLater)
        self._camera_thread.finished.connect(self._camera_thread.deleteLater)
        self._camera_thread.start()

    def _on_camera_status(self, text: str) -> None:
        try:
            self.camera_status.setText(text)
            self.append_log("[CAMERA] " + text)
        except RuntimeError:
            # The worker can finish while the Qt window is closing.
            pass

    def _on_camera_frame(self, color_image: object, depth_image: object) -> None:
        try:
            self._camera_color_image = color_image
            self._camera_depth_image = depth_image
            self._set_camera_pixmap(self.camera_color_label, color_image, QtGui.QImage.Format_BGR888)
            self._set_camera_pixmap(self.camera_depth_label, depth_image, QtGui.QImage.Format_RGB888)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._on_camera_status(f"相机画面显示失败: {exc}")

    def _on_camera_data(self, color_image: object, _depth_image: object, depth_m: object) -> None:
        self._camera_color_image = color_image
        self._camera_depth_m = depth_m
        if hasattr(self, "vision_capture_status"):
            self.vision_capture_status.setText("相机图像已更新，可以截取或识别")

    def _on_camera_info(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
        frame_id: str,
    ) -> None:
        self._camera_intrinsics = (fx, fy, cx, cy, depth_scale, frame_id)

    def _capture_vision_image(self) -> None:
        image = self._camera_color_image
        if image is None:
            self.vision_capture_status.setText("还没有收到相机图像")
            return
        try:
            import cv2

            output_dir = ROOT / "yolo_vision_outputs"
            output_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"capture_{stamp}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("cv2.imwrite 失败")
            self._last_vision_capture = path
            self.vision_capture_status.setText(f"已截取: {path}")
            self.append_log(f"[VISION] 已截取当前图像: {path}")
        except Exception as exc:
            self.vision_capture_status.setText(f"截取失败: {exc}")

    def _save_training_image(self) -> None:
        """保存当前原始彩色帧，作为自定义 YOLO 分割数据集图片。"""
        image = self._camera_color_image
        if image is None:
            self.vision_capture_status.setText("还没有收到相机图像，暂时不能保存")
            return
        try:
            import cv2

            self._yolo_training_image_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            # 同一秒内连续点击时加入毫秒，避免覆盖上一张图片。
            filename = f"{stamp}_{time.time_ns() % 1_000_000_000:09d}.jpg"
            path = self._yolo_training_image_dir / filename
            if not cv2.imwrite(
                str(path),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            ):
                raise RuntimeError("cv2.imwrite 失败")

            count = len(list(self._yolo_training_image_dir.glob("*.jpg")))
            self._last_vision_capture = path
            self.vision_capture_status.setText(
                f"训练图片已保存（共 {count} 张）: {path}"
            )
            self.append_log(f"[VISION] 已保存训练图片: {path}")
        except Exception as exc:
            self.vision_capture_status.setText(f"保存训练图片失败: {exc}")
            self.append_log(f"[VISION] 保存训练图片失败: {exc}")

    def _detect_vision_target(self) -> None:
        color_image = self._camera_color_image
        depth_m = self._camera_depth_m
        intrinsics = self._camera_intrinsics
        if color_image is None or depth_m is None or intrinsics is None:
            self.vision_result_label.setText("相机还没有准备好，请稍等后再识别")
            return
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO

            selected_model = self.vision_model_combo.currentData()
            if not selected_model:
                raise RuntimeError("没有可用的 YOLO 模型")
            model_path = Path(str(selected_model))
            if self._yolo_model is None or self._yolo_model_path != model_path:
                if not model_path.exists():
                    raise RuntimeError(f"分割模型不存在: {model_path}")
                self._yolo_model = YOLO(str(model_path))
                self._yolo_model_path = model_path
                self.append_log(f"[VISION] 已加载 YOLO 分割模型: {model_path}")
                # 保留两个抓取流程的固定选项，同时补充模型中的其他类别。
                model_names = [
                    str(name) for _, name in sorted(self._yolo_model.names.items())
                ]
                existing_names = {
                    self.vision_class_combo.itemText(index).strip().lower()
                    for index in range(self.vision_class_combo.count())
                }
                for name in model_names:
                    if name.strip().lower() not in existing_names:
                        self.vision_class_combo.addItem(name)

            target_text = self.vision_class_combo.currentText().strip().lower()
            names = {str(name).lower(): int(index) for index, name in self._yolo_model.names.items()}
            class_ids = None
            if target_text not in {"", "all", "*"}:
                class_ids = [int(target_text)] if target_text.isdigit() else [names[target_text]]

            results = self._yolo_model.predict(
                color_image,
                conf=0.25,
                iou=0.45,
                device="cpu",
                classes=class_ids,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or result.masks is None or len(result.boxes) == 0:
                raise RuntimeError(f"没有检测到标签: {target_text or 'all'}")

            model_name = model_path.name.lower()
            if model_name == "ear.pt" or target_text == "ear":
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                masks = result.masks.data.cpu().numpy() > 0.5
                height, width = color_image.shape[:2]
                if masks.shape[1:] != (height, width):
                    masks = np.stack(
                        [
                            cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                            for mask in masks
                        ],
                        axis=0,
                    )

                detections: list[dict[str, Any]] = []
                for index in range(len(result.boxes)):
                    mask = masks[index]
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                    valid = (
                        inner_mask
                        & np.isfinite(depth_m)
                        & (depth_m > 0.10)
                        & (depth_m < 2.00)
                    )
                    if int(np.count_nonzero(valid)) < 20:
                        continue

                    median_depth = float(np.median(depth_m[valid]))
                    valid &= np.abs(depth_m - median_depth) < 0.05
                    v_idx, u_idx = np.nonzero(valid)
                    z = depth_m[v_idx, u_idx].astype(np.float64)
                    points_camera = np.column_stack(
                        (
                            (u_idx.astype(np.float64) - intrinsics[2]) * z / intrinsics[0],
                            (v_idx.astype(np.float64) - intrinsics[3]) * z / intrinsics[1],
                            z,
                        )
                    )
                    if len(points_camera) == 0:
                        continue

                    points_base = None
                    points_car = None
                    center_camera = np.median(points_camera, axis=0)
                    grasp_point = center_camera
                    if self._camera_to_base is not None:
                        camera_points_h = np.column_stack(
                            (points_camera, np.ones((len(points_camera), 1), dtype=np.float64))
                        )
                        points_base = (
                            np.asarray(self._camera_to_base, dtype=np.float64)
                            @ camera_points_h.T
                        ).T[:, :3]
                        if points_base.size:
                            grasp_point = np.median(points_base, axis=0)
                        snap = self._backend.snapshot()
                        if snap.car_from_body is not None:
                            points_base_h = np.column_stack(
                                (points_base, np.ones((len(points_base), 1), dtype=np.float64))
                            )
                            points_car = (
                                np.asarray(snap.car_from_body, dtype=np.float64)
                                @ points_base_h.T
                            ).T[:, :3]
                            if points_car.size:
                                grasp_point = np.median(points_car, axis=0)

                    detections.append(
                        {
                            "index": index,
                            "confidence": float(confs[index]),
                            "class_name": str(self._yolo_model.names[int(classes[index])]),
                            "pixel": (int(np.median(u_idx)), int(np.median(v_idx))),
                            "grasp_point": tuple(float(v) for v in grasp_point.tolist()),
                            "center_camera": tuple(float(v) for v in center_camera.tolist()),
                            "points_camera": points_camera,
                            "points_base": points_base,
                            "points_car": points_car,
                        }
                    )

                if not detections:
                    raise RuntimeError("ear 检测到了目标，但没有得到足够的点云")

                detections.sort(key=lambda item: item["pixel"][0])
                if len(detections) >= 2:
                    selected = [detections[0], detections[-1]]
                    labels = ["左", "右"]
                else:
                    selected = [detections[0]]
                    labels = ["左"]

                overlay = color_image.copy()
                for det in detections:
                    px, py = det["pixel"]
                    mask = masks[det["index"]]
                    valid = (
                        mask
                        & np.isfinite(depth_m)
                        & (depth_m > 0.10)
                        & (depth_m < 2.00)
                    )
                    if int(np.count_nonzero(valid)) == 0:
                        continue
                    overlay[mask] = (
                        0.35 * overlay[mask] + 0.65 * np.array([0, 255, 180], dtype=np.float32)
                    ).astype(np.uint8)
                    overlay[valid] = (
                        0.25 * overlay[valid] + 0.75 * np.array([0, 0, 255], dtype=np.float32)
                    ).astype(np.uint8)
                    cv2.circle(overlay, (px, py), 6, (255, 255, 255), -1)

                colors = [(0, 255, 0), (0, 165, 255)]
                lines = [f"标签: ear", f"检测到 {len(detections)} 个目标", "已生成抓取点:"]
                ear_targets = []
                for i, det in enumerate(selected):
                    px, py = det["pixel"]
                    color = colors[i % len(colors)]
                    cv2.circle(overlay, (px, py), 10, color, 2)
                    cv2.putText(
                        overlay,
                        f"{labels[i]}#{i+1}",
                        (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    gp = det["grasp_point"]
                    lines.append(
                        f"{labels[i]}抓取点: {gp[0]:+.4f} {gp[1]:+.4f} {gp[2]:+.4f} m"
                    )
                    ear_targets.append(det)

                self._vision_target = {
                    "class_name": "ear",
                    "confidence": float(max(det["confidence"] for det in detections)),
                    "center_camera": np.median(
                        np.vstack([det["center_camera"] for det in detections]), axis=0
                    ),
                    "points_camera": np.vstack([det["points_camera"] for det in detections]),
                    "points_base": None
                    if any(det["points_base"] is None for det in detections)
                    else np.vstack([det["points_base"] for det in detections]),
                    "points_car": None
                    if any(det["points_car"] is None for det in detections)
                    else np.vstack([det["points_car"] for det in detections]),
                    "frame_id": intrinsics[5],
                    "ear_targets": ear_targets,
                }
                self._set_camera_pixmap(self.camera_color_label, overlay, QtGui.QImage.Format_BGR888)
                self._set_camera_pixmap(self.vision_result_image, overlay, QtGui.QImage.Format_BGR888)
                text = "\n".join(lines)
                self.vision_result_label.setText(text)
                self.append_log("[VISION]\n" + text)
                return

            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy() > 0.5
            height, width = color_image.shape[:2]
            if masks.shape[1:] != (height, width):
                masks = np.stack(
                    [
                        cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for mask in masks
                    ],
                    axis=0,
                )

            detections = []
            for index in range(len(result.boxes)):
                det = self._build_segmentation_detection(
                    index,
                    int(classes[index]),
                    str(self._yolo_model.names[int(classes[index])]),
                    float(confs[index]),
                    masks[index],
                    color_image,
                    depth_m,
                    intrinsics,
                    cv2,
                    np,
                )
                if det is not None:
                    detections.append(det)

            if not detections:
                raise RuntimeError("目标 mask 内有效深度点太少")

            detections.sort(key=lambda item: (-item["pixel"][1], item["median_depth"], -item["confidence"]))
            selected = detections[0]
            mask = masks[int(selected["index"])]
            valid = (
                mask
                & np.isfinite(depth_m)
                & (depth_m > 0.10)
                & (depth_m < 2.00)
            )

            # 缓存最近一次视觉结果，供“瓶子抓取”页使用。
            # 不直接在视觉页执行运动，避免采集数据时误触发机械臂。
            self._vision_target = {
                "class_name": selected["class_name"],
                "confidence": float(selected["confidence"]),
                "center_camera": np.asarray(selected["center_camera"], dtype=np.float64),
                "points_camera": np.asarray(selected["points_camera"], dtype=np.float64),
                "points_base": selected["points_base"],
                "points_car": selected["points_car"],
                "frame_id": intrinsics[5],
            }

            overlay = color_image.copy()
            overlay[mask] = (
                0.35 * overlay[mask] + 0.65 * np.array([0, 255, 180], dtype=np.float32)
            ).astype(np.uint8)
            overlay[valid] = (
                0.25 * overlay[valid] + 0.75 * np.array([0, 0, 255], dtype=np.float32)
            ).astype(np.uint8)
            px, py = selected["pixel"]
            cv2.circle(overlay, (px, py), 6, (255, 255, 255), -1)
            self._set_camera_pixmap(self.camera_color_label, overlay, QtGui.QImage.Format_BGR888)
            self._set_camera_pixmap(self.vision_result_image, overlay, QtGui.QImage.Format_BGR888)

            base_text = (
                f"base/body_link XYZ: {np.mean(selected['points_base'], axis=0)[0]:+.4f} {np.mean(selected['points_base'], axis=0)[1]:+.4f} {np.mean(selected['points_base'], axis=0)[2]:+.4f} m\n"
                if selected["points_base"] is not None
                else ""
            )
            car_link_text = (
                f"car_link XYZ: {np.mean(selected['points_car'], axis=0)[0]:+.4f} {np.mean(selected['points_car'], axis=0)[1]:+.4f} {np.mean(selected['points_car'], axis=0)[2]:+.4f} m\n"
                if selected["points_car"] is not None
                else ""
            )
            text = (
                f"标签: {selected['class_name']}\n"
                f"置信度: {selected['confidence']:.2f}\n"
                f"坐标系: {intrinsics[5]}\n"
                f"相机坐标 XYZ: {selected['center_camera'][0]:+.4f} {selected['center_camera'][1]:+.4f} {selected['center_camera'][2]:+.4f} m\n"
                f"{base_text}"
                f"{car_link_text}"
                f"有效深度点: {len(selected['points_camera'])}\n"
                f"主深度中位数: {selected['median_depth']:+.4f} m\n"
                f"抓取结果: 已选前排目标"
            )
            self.vision_result_label.setText(text)
            self.append_log("[VISION]\n" + text)
        except KeyError:
            self.vision_result_label.setText(f"模型中没有这个标签: {target_text}")
        except Exception as exc:
            self.vision_result_label.setText(f"识别失败: {exc}")
            self.append_log(f"[VISION] 识别失败: {exc}")

    def _set_camera_pixmap(self, label: QtWidgets.QLabel, image: object, image_format: QtGui.QImage.Format) -> None:
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            return
        height, width = int(image.shape[0]), int(image.shape[1])
        channels = int(image.shape[2]) if len(image.shape) >= 3 else 1
        if channels != 3:
            return
        bytes_per_line = width * channels
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            image_format,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _pose_xyzrpy(self, pose: Optional[Pose], label: str = "TCP") -> tuple[float, ...]:
        if pose is None:
            raise ValueError(f"{label} 当前没有状态")
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            *quaternion_to_rpy(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
        )

    def _capture_samples(self) -> None:
        snap = self._backend.snapshot()
        now = time.monotonic()
        if self._recording_active:
            values = snap.arm_joint_values(ARM_JOINTS)
            if snap.joint_update >= self._record_request_time and not any(value is None for value in values):
                sample = tuple(float(value) for value in values if value is not None)
                if not self._record_rows:
                    self._record_start = now
                    self.record_status.setText("采集中 0")
                self._record_rows.append([now - self._record_start] + list(sample))
                self.record_status.setText(f"采集中 {len(self._record_rows)}")
                self._update_joint_record_idle(sample, now)
        if self._cart_recording_active:
            pose = snap.left_ee if self.cart_record_arm_combo.currentText() == "left" else snap.right_ee
            if snap.tcp_update >= self._cart_record_request_time and pose is not None:
                sample = self._pose_xyzrpy(pose)
                if not self._cart_record_rows:
                    self._cart_record_start = now
                    self.cart_record_status.setText("采集中 0")
                self._cart_record_rows.append([now - self._cart_record_start] + list(sample))
                self.cart_record_status.setText(f"采集中 {len(self._cart_record_rows)}")
                self._update_cart_record_idle(sample, now)

    def _update_joint_record_idle(self, sample: tuple[float, ...], now: float) -> None:
        if self._record_last_values is None:
            self._record_last_values = sample
            self._record_idle_since = now
            return
        max_delta = max(abs(value - previous) for value, previous in zip(sample, self._record_last_values))
        self._record_last_values = sample
        if max_delta > JOINT_IDLE_EPS_RAD:
            self._record_idle_since = now
            return
        if self._record_idle_since is not None and now - self._record_idle_since >= RECORD_IDLE_SECONDS:
            self._stop_joint_recording(auto=True, reason="机械臂关节 2 秒未变化，自动停止并保存")

    def _update_cart_record_idle(self, sample: tuple[float, ...], now: float) -> None:
        if self._cart_record_last_values is None:
            self._cart_record_last_values = sample
            self._cart_record_idle_since = now
            return
        pos_delta = max(abs(value - previous) for value, previous in zip(sample[:3], self._cart_record_last_values[:3]))
        rot_delta = max(abs(clamp_angle(value - previous)) for value, previous in zip(sample[3:], self._cart_record_last_values[3:]))
        self._cart_record_last_values = sample
        if pos_delta > CART_IDLE_POS_EPS_M or rot_delta > CART_IDLE_ROT_EPS_RAD:
            self._cart_record_idle_since = now
            return
        if self._cart_record_idle_since is not None and now - self._cart_record_idle_since >= RECORD_IDLE_SECONDS:
            self._stop_cart_record(auto=True, reason="机械臂 TCP 2 秒未变化，自动停止并保存")

    def _start_joint_recording(self) -> None:
        if not self.teach_btn.isChecked():
            raise ValueError("请先点示教")
        self._record_rows = []
        self._record_start = 0.0
        self._record_request_time = time.monotonic()
        self._record_last_values = None
        self._record_idle_since = None
        self._recording_active = True
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.record_status.setText("等待关节数据...")

    def _stop_joint_recording(self, auto: bool = False, reason: str = "") -> None:
        if not self._recording_active:
            return
        self._recording_active = False
        self._record_last_values = None
        self._record_idle_since = None
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        if not self._record_rows:
            self.record_status.setText("没有采集到数据")
            return
        path = self._record_dir / f"joint_demo_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t"] + list(ARM_JOINTS))
            writer.writerows(self._record_rows)
        prefix = "自动停止，" if auto else ""
        self.append_log(f"[REC] {reason or 'manual stop'}; saved {len(self._record_rows)} samples -> {path}")
        self.record_status.setText(f"{prefix}已保存 {len(self._record_rows)} 点")
        self._refresh_files(preferred_demo=path)

    def _start_cart_record(self) -> None:
        if not self.teach_btn.isChecked():
            raise ValueError("请先点示教")
        self._cart_record_rows = []
        self._cart_record_start = 0.0
        self._cart_record_request_time = time.monotonic()
        self._cart_record_last_values = None
        self._cart_record_idle_since = None
        self._cart_recording_active = True
        self.cart_record_arm_combo.setEnabled(False)
        self.cart_record_start_btn.setEnabled(False)
        self.cart_record_stop_btn.setEnabled(True)
        self.cart_record_status.setText("等待 TCP 数据...")

    def _stop_cart_record(self, auto: bool = False, reason: str = "") -> None:
        if not self._cart_recording_active:
            return
        self._cart_recording_active = False
        self._cart_record_last_values = None
        self._cart_record_idle_since = None
        self.cart_record_arm_combo.setEnabled(True)
        self.cart_record_start_btn.setEnabled(True)
        self.cart_record_stop_btn.setEnabled(False)
        if len(self._cart_record_rows) < 2:
            self.cart_record_status.setText("采样点不足")
            return
        arm = self.cart_record_arm_combo.currentText()
        path = self._cart_record_dir / f"cartesian_demo_{arm}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "x", "y", "z", "roll", "pitch", "yaw"])
            writer.writerows(self._cart_record_rows)
        prefix = "自动停止，" if auto else ""
        self.cart_record_status.setText(f"{prefix}已保存 {len(self._cart_record_rows)} 点")
        self.append_log(f"[CART-REC] {reason or 'manual stop'}; saved {len(self._cart_record_rows)} samples -> {path}")
        self._refresh_files(preferred_cart_demo=path)

    def _current_path(self, combo: QtWidgets.QComboBox) -> Optional[Path]:
        value = combo.currentData()
        return Path(str(value)) if value else None

    def _fill_combo(self, combo: QtWidgets.QComboBox, paths: list[Path], preferred: Optional[Path] = None) -> None:
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for path in paths:
            combo.addItem(path.name, str(path))
        target = str(preferred) if preferred else previous
        if target:
            index = combo.findData(target)
            if index >= 0:
                combo.setCurrentIndex(index)
        if combo.count() and combo.currentIndex() < 0:
            combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _refresh_files(self, preferred_demo: Optional[Path] = None, preferred_model: Optional[Path] = None, preferred_cart_demo: Optional[Path] = None, preferred_cart_model: Optional[Path] = None) -> None:
        self._fill_combo(self.demo_combo, sorted(self._record_dir.glob("joint_demo_*.csv")), preferred_demo)
        self._fill_combo(self.model_combo, sorted(self._model_dir.glob("*.npz")), preferred_model)
        self._fill_combo(self.cart_demo_combo, sorted(self._cart_record_dir.glob("cartesian_demo_*.csv")), preferred_cart_demo)
        self._fill_combo(self.cart_model_combo, sorted(self._cart_model_dir.glob("*.npz")), preferred_cart_model)

    def _start_process(
        self,
        command: list[str],
        message: str,
        preferred_model: Optional[Path] = None,
        preferred_cart_model: Optional[Path] = None,
        on_success: Optional[object] = None,
    ) -> None:
        if self._dmp_process is not None:
            if not self._dmp_process_alive():
                self._dmp_process.deleteLater()
                self._dmp_process = None
            else:
                raise ValueError("已有 DMP 任务在运行；如果界面没有输出，请先点“停止DMP”")
        self.append_log(message)
        process = QtCore.QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self.append_log(bytes(process.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()))
        process.finished.connect(lambda code, status: self._process_finished(code, preferred_model, preferred_cart_model, on_success))
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        process.start()
        if not process.waitForStarted(2000):
            process.deleteLater()
            raise RuntimeError("DMP 进程启动失败")
        self._dmp_process = process
        self.dmp_stop_btn.setEnabled(True)
        if hasattr(self, "grasp_dmp_pre_btn"):
            self.grasp_dmp_pre_btn.setEnabled(False)

    def _dmp_process_alive(self) -> bool:
        process = self._dmp_process
        if process is None or process.state() == QtCore.QProcess.NotRunning:
            return False
        pid = int(process.processId())
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _process_finished(
        self,
        code: int,
        preferred_model: Optional[Path],
        preferred_cart_model: Optional[Path],
        on_success: Optional[object] = None,
    ) -> None:
        process = self._dmp_process
        self._dmp_process = None
        self.dmp_stop_btn.setEnabled(False)
        if hasattr(self, "grasp_dmp_pre_btn"):
            self.grasp_dmp_pre_btn.setEnabled(True)
        self._refresh_files(preferred_model=preferred_model, preferred_cart_model=preferred_cart_model)
        self.append_log(f"[DMP] finished exit={code}")
        if process is not None:
            process.deleteLater()
        if code == 0 and on_success is not None:
            on_success()

    def _process_error(self, process: QtCore.QProcess, error: QtCore.QProcess.ProcessError) -> None:
        self.append_log(f"[DMP] process error={int(error)}: {process.errorString()}")
        if process.state() == QtCore.QProcess.NotRunning and self._dmp_process is process:
            self._dmp_process = None
            self.dmp_stop_btn.setEnabled(False)
            if hasattr(self, "grasp_dmp_pre_btn"):
                self.grasp_dmp_pre_btn.setEnabled(True)

    def _stop_dmp(self) -> None:
        process = self._dmp_process
        if process is None:
            self.dmp_stop_btn.setEnabled(False)
            return
        if process.state() != QtCore.QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1500)
        self._dmp_process = None
        self.dmp_stop_btn.setEnabled(False)
        if hasattr(self, "grasp_dmp_pre_btn"):
            self.grasp_dmp_pre_btn.setEnabled(True)
        self.append_log("[DMP] stopped by user")
        process.deleteLater()

    def _train_joint(self) -> None:
        demo = self._current_path(self.demo_combo)
        if demo is None:
            raise ValueError("没有关节示教文件")
        arm = self.arm_combo.currentText()
        model = self._model_dir / f"{arm}_{demo.stem}.npz"
        self._start_process([sys.executable, str(ROOT / "scripts/joint_dmp_pipeline.py"), "train", "--demo", str(demo), "--arm", arm, "--output", str(model), "--weights", str(self.weight_spin.value())], "[DMP] train joint", preferred_model=model)

    def _play_joint(self) -> None:
        model = self._current_path(self.model_combo)
        if model is None:
            raise ValueError("没有关节 DMP 模型")
        self._start_process([sys.executable, str(ROOT / "scripts/joint_dmp_pipeline.py"), "play", "--model", str(model)], "[DMP] play joint")

    def _generalize_joint(self) -> None:
        model = self._current_path(self.model_combo)
        goal = self.generalize_goal_input.text().strip()
        if model is None or not goal:
            raise ValueError("请选择模型并输入 7 个终点角度")
        self._start_process([sys.executable, str(ROOT / "scripts/joint_dmp_pipeline.py"), "play", "--model", str(model), "--goal-deg", goal], "[DMP] generalize joint")

    def _train_cart(self) -> None:
        demo = self._current_path(self.cart_demo_combo)
        if demo is None:
            raise ValueError("没有 TCP 示教文件")
        arm = self.cart_record_arm_combo.currentText()
        model = self._cart_model_dir / f"{arm}_{demo.stem}.npz"
        self._start_process([sys.executable, str(ROOT / "scripts/cartesian_dmp_pipeline.py"), "train", "--demo", str(demo), "--arm", arm, "--output", str(model), "--weights", str(self.cart_weight_spin.value())], "[CART-DMP] train TCP", preferred_cart_model=model)

    def _play_cart(self, goal: Optional[str]) -> None:
        model = self._current_path(self.cart_model_combo)
        if model is None:
            raise ValueError("没有笛卡尔 DMP 模型")
        command = [sys.executable, str(ROOT / "scripts/cartesian_dmp_pipeline.py"), "play", "--model", str(model)]
        if goal:
            command.extend(["--goal", goal])
        self._start_process(command, "[CART-DMP] play TCP")

    def _parse_deg(self, text: str, count: int) -> tuple[float, ...]:
        parts = text.replace("，", " ").replace(",", " ").split()
        if len(parts) != count:
            raise ValueError(f"需要 {count} 个角度")
        return tuple(math.radians(float(value)) for value in parts)

    def _parse_deg_list_or_current(
        self,
        text: str,
        count: int,
        current: tuple[Optional[float], ...],
        label: str,
    ) -> tuple[float, ...]:
        """允许只填写一侧关节，空侧沿用当前状态。"""
        if text.strip():
            return self._parse_deg(text, count)
        if len(current) != count or any(value is None for value in current):
            raise ValueError(f"{label} 为空，且当前状态还没读全，无法自动补齐")
        return tuple(float(value) for value in current if value is not None)

    def _parse_cart(self, text: str) -> tuple[float, ...]:
        parts = text.replace("，", " ").replace(",", " ").split()
        if len(parts) != 6:
            raise ValueError("需要 x y z roll pitch yaw 六个值，姿态单位 rad")
        return tuple(float(value) for value in parts)

    def _parse_cart_delta_cm(self, text: str) -> tuple[float, float, float]:
        parts = text.replace("，", " ").replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("需要 x y z 三个值，单位 cm")
        return tuple(float(value) / 100.0 for value in parts)

    def _build_segmentation_detection(
        self,
        index: int,
        class_id: int,
        class_name: str,
        conf: float,
        mask: "np.ndarray",
        color_image: "np.ndarray",
        depth_m: "np.ndarray",
        intrinsics: tuple[float, float, float, float, float, str],
        cv2: Any,
        np: Any,
    ) -> Optional[dict[str, Any]]:
        height, width = color_image.shape[:2]
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        kernel = np.ones((5, 5), dtype=np.uint8)
        inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        fx, fy, cx, cy, _depth_scale, frame_id = intrinsics
        valid = (
            inner_mask
            & np.isfinite(depth_m)
            & (depth_m > 0.10)
            & (depth_m < 2.00)
        )
        if int(np.count_nonzero(valid)) < 20:
            return None
        median_depth = float(np.median(depth_m[valid]))
        valid &= np.abs(depth_m - median_depth) < 0.05
        v_idx, u_idx = np.nonzero(valid)
        z = depth_m[v_idx, u_idx].astype(np.float64)
        points = np.column_stack(
            (
                (u_idx.astype(np.float64) - cx) * z / fx,
                (v_idx.astype(np.float64) - cy) * z / fy,
                z,
            )
        )
        if len(points) == 0:
            return None
        center = np.median(points, axis=0)
        base_center = None
        car_link_center = None
        points_base = None
        points_car = None
        if self._camera_to_base is not None:
            camera_points_h = np.column_stack(
                (points, np.ones((len(points), 1), dtype=np.float64))
            )
            points_base = (
                np.asarray(self._camera_to_base, dtype=np.float64)
                @ camera_points_h.T
            ).T[:, :3]
            base_center = tuple(np.median(points_base, axis=0).tolist())
            snap = self._backend.snapshot()
            if snap.car_from_body is not None:
                points_base_h = np.column_stack(
                    (points_base, np.ones((len(points_base), 1), dtype=np.float64))
                )
                points_car = (
                    np.asarray(snap.car_from_body, dtype=np.float64)
                    @ points_base_h.T
                ).T[:, :3]
                car_link_center = tuple(np.median(points_car, axis=0).tolist())
        grasp_source = car_link_center or base_center or tuple(center.tolist())
        return {
            "index": index,
            "confidence": float(conf),
            "class_name": class_name,
            "pixel": (int(np.median(u_idx)), int(np.median(v_idx))),
            "grasp_point": tuple(float(v) for v in grasp_source),
            "center_camera": tuple(float(v) for v in center.tolist()),
            "points_camera": points,
            "points_base": points_base,
            "points_car": points_car,
            "median_depth": median_depth,
        }

    def _update_lift_preview(self, snap: Optional[RobotSnapshot] = None) -> None:
        current_snap = self._backend.snapshot() if snap is None else snap
        pose = current_snap.leg_pose()
        if not pose:
            self.lift_preview_label.setText("当前高度 -- | 上升后 -- | 下降后 --")
            return
        current_height = float(pose["y"])
        step = float(self.lift_step.value())
        waist_deg = float(self.waist_delta.value())
        self.lift_preview_label.setText(
            f"当前高度 {current_height:+.3f} m | "
            f"上升后 {current_height + step:+.3f} m | "
            f"下降后 {current_height - step:+.3f} m | "
            f"腰部偏移 {waist_deg:+.1f} deg"
        )

    def _send_leg_lift(self, direction: float) -> None:
        snap = self._backend.snapshot()
        pose = snap.leg_pose()
        if not pose:
            raise ValueError("当前腿部状态还没读全，无法做升降")
        delta = float(self.lift_step.value()) * float(direction)
        waist_delta_rad = math.radians(float(self.waist_delta.value()))
        target_height = float(pose["y"]) + delta
        self._backend.send("leg_lift", delta=delta, waist_delta=waist_delta_rad, vel=0.08)
        self.append_log(
            f"[CMD] leg lift delta={delta:+.3f}m target_height={target_height:+.3f}m "
            f"waist_delta={self.waist_delta.value():+.1f}deg"
        )

    def _send_arm_joint(self) -> None:
        snap = self._backend.snapshot()
        left = self._parse_deg_list_or_current(
            self.left_joint_input.text(),
            7,
            snap.arm_joint_values(LEFT_ARM_JOINTS),
            "左臂",
        )
        right = self._parse_deg_list_or_current(
            self.right_joint_input.text(),
            7,
            snap.arm_joint_values(RIGHT_ARM_JOINTS),
            "右臂",
        )
        self._backend.send("arm_joint", left=left, right=right, vel=0.30, acc=0.50)

    def _send_arm_cartesian(self) -> None:
        snap = self._backend.snapshot()
        left_text = self.left_cart_input.text().strip()
        right_text = self.right_cart_input.text().strip()
        if not left_text and not right_text:
            raise ValueError("左臂和右臂 TCP 至少填写一侧")
        left = self._pose_xyzrpy(snap.left_ee, "左臂 TCP") if not left_text else self._parse_cart(left_text)
        right = self._pose_xyzrpy(snap.right_ee, "右臂 TCP") if not right_text else self._parse_cart(right_text)
        self._backend.send("arm_cartesian", left=left, right=right, vel=0.10, acc=0.20)

    def _send_arm_cartesian_relative(self) -> None:
        snap = self._backend.snapshot()
        left_text = self.left_cart_rel_input.text().strip()
        right_text = self.right_cart_rel_input.text().strip()
        if not left_text and not right_text:
            raise ValueError("左臂和右臂相对XYZ至少填写一侧")
        left_pose = self._pose_xyzrpy(snap.left_ee, "左臂 TCP")
        right_pose = self._pose_xyzrpy(snap.right_ee, "右臂 TCP")
        if left_text:
            dx, dy, dz = self._parse_cart_delta_cm(left_text)
            left = self._offset_pose_local(left_pose, dx, dy, dz)
        else:
            left = left_pose
        if right_text:
            dx, dy, dz = self._parse_cart_delta_cm(right_text)
            right = self._offset_pose_local(right_pose, dx, dy, dz)
        else:
            right = right_pose
        self._backend.send("arm_cartesian", left=left, right=right, vel=0.10, acc=0.20)

    def _send_leg_joint(self) -> None:
        self._backend.send("leg_joint", values=self._parse_deg(self.leg_input.text(), 4), vel=0.08, acc=0.20)

    def _send_first_layer_height(self) -> None:
        values = tuple(math.radians(v) for v in (50.6, -103.1, 52.6, 0.0))
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)
        self.append_log("[CMD] first layer height -> +50.6 -103.1 +52.6 +0.0")

    def _send_second_layer_height(self) -> None:
        values = (0.0, 0.0, 0.0, 0.0)
        self._backend.send("leg_joint", values=values, vel=0.08, acc=0.20)
        self.append_log("[CMD] second layer height -> 0 0 0 0")

    def _send_reset_arms(self) -> None:
        snap = self._backend.snapshot()
        leg_current = snap.arm_joint_values(LEG_JOINTS)
        if len(leg_current) != 4 or any(value is None for value in leg_current):
            raise ValueError("当前腰腿状态还没读全，无法复位机械臂")
        self._backend.send(
            "arm_joint",
            left=tuple(math.radians(v) for v in RESET_LEFT_ARM_DEG),
            right=tuple(math.radians(v) for v in RESET_RIGHT_ARM_DEG),
            vel=0.30,
            acc=0.50,
        )
        self._backend.send(
            "leg_joint",
            values=(
                float(leg_current[0]),
                float(leg_current[1]),
                float(leg_current[2]),
                0.0,
            ),
            vel=0.08,
            acc=0.20,
        )
        self.append_log("[CMD] reset arms; keep leg joints and zero hip_yaw_joint")

    def _send_reset(self) -> None:
        self._send_reset_arms()
        self._backend.send(
            "leg_joint",
            values=(0.0, 0.0, 0.0, 0.0),
            vel=0.08,
            acc=0.20,
        )
        self.append_log("[CMD] reset arms + leg/waist joints to 0 deg")

    def append_log(self, text: str) -> None:
        if not text:
            return
        if not hasattr(self, "log_view"):
            self._pending_logs.append(text)
            return
        while self._pending_logs:
            self.log_view.append(self._pending_logs.pop(0))
        self.log_view.append(text)

    def _plain_joints(self, values: tuple[Optional[float], ...]) -> str:
        return " ".join("--" if value is None else f"{math.degrees(value):+.1f}" for value in values)

    def _pose_text(self, pose: Optional[Pose]) -> str:
        if pose is None:
            return "--"
        return " ".join(f"{value:+.3f}" for value in self._pose_xyzrpy(pose)) + " rad"

    def refresh_state(self) -> None:
        snap = self._backend.snapshot()
        pose = snap.leg_pose()
        self.status_system.setText("在线" if snap.last_update else "等待状态")
        self.status_power.setText(f"手臂 {self._format_power(snap.arm_power)} | 腿部 {self._format_power(snap.leg_power)}")
        self.status_motion.setText(f"手臂 {self._format_motion(snap.arm_motion)} | 腿部 {self._format_motion(snap.leg_motion)}")
        if pose:
            self.status_body.setText(
                f"x={pose['x']:+.3f}m 高度={pose['y']:+.3f}m "
                f"phi={math.degrees(pose['phi']):+.1f}deg "
                f"waist={math.degrees(pose['waist']):+.1f}deg"
            )
        else:
            self.status_body.setText("--")
        self.status_arms.setText("左 " + self._plain_joints(snap.arm_joint_values(LEFT_ARM_JOINTS)) + " | 右 " + self._plain_joints(snap.arm_joint_values(RIGHT_ARM_JOINTS)))
        self.status_legs.setText(self._plain_joints(snap.arm_joint_values(LEG_JOINTS)))
        self.status_ee.setText("左 TCP " + self._pose_text(snap.left_ee) + "\n右 TCP " + self._pose_text(snap.right_ee))
        self._update_lift_preview(snap)

    def _format_power(self, power: Optional[tuple[bool, tuple[float, ...]]]) -> str:
        if power is None:
            return "--"
        return "已使能" if power[0] else "未使能"

    def _format_motion(self, motion: Optional[tuple[bool, bool]]) -> str:
        return "--" if motion is None else ("移动" if motion[0] else "静止")

    def closeEvent(self, event: QtCore.QCloseEvent) -> None:
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        self._backend.send("base_cmd", linear_x=0.0, angular_z=0.0)
        self._recording_active = False
        self._cart_recording_active = False
        self._stop_final_grasp_process()
        if hasattr(self, "_camera_worker"):
            self._camera_worker.frame_ready.disconnect(self._on_camera_frame)
            self._camera_worker.data_ready.disconnect(self._on_camera_data)
            self._camera_worker.camera_info_ready.disconnect(self._on_camera_info)
            self._camera_worker.status_line.disconnect(self._on_camera_status)
            self._camera_worker.stop()
        if hasattr(self, "_camera_thread") and self._camera_thread.isRunning():
            if not self._camera_thread.wait(2500):
                self._camera_thread.terminate()
                self._camera_thread.wait(1000)
        self._stop_dmp()
        self._backend.stop()
        QtWidgets.QApplication.quit()
        QtCore.QTimer.singleShot(300, lambda: os._exit(0))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    backend = RobotBackend()
    backend.start()
    window = DmpMainWindow(backend)

    signal_timer = QtCore.QTimer()
    signal_timer.start(100)
    signal_timer.timeout.connect(lambda: None)

    def _quit(*_: object) -> None:
        QtCore.QTimer.singleShot(0, window.shutdown)

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)

    window.showMaximized()
    code = app.exec_()
    backend.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
