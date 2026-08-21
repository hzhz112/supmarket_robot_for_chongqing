#!/usr/bin/env python3
"""Sailor R1 Pro 升降机构和腰部关节控制辅助脚本。

在 /home/test/LJM 当前这套 ROS 2 接口里，没有单独暴露名为 elevator/lift 的电机。
身体升降由腿部/升降四关节联动完成：
ankle_joint, knee_joint, hip_pitch_joint, hip_yaw_joint。
腰部旋转关节对应 hip_yaw_joint。

脚本默认只做 dry-run 打印目标，不会真正下发控制命令。
确认机器人周围安全后，加 --execute 才会发送命令。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import rclpy
from rclpy.node import Node
from robot_control_msg.msg import LegPowerStatus, Robotlegjoint, Robotservomsg
from robot_control_msg.srv import LegAbsoluteControl, LegCartesianControl, SetRobotPower
from sensor_msgs.msg import JointState


# 腿部/升降机构的四个 ROS2 关节名。hip_yaw_joint 同时也是腰部旋转关节。
LEG_JOINTS = ("ankle_joint", "knee_joint", "hip_pitch_joint", "hip_yaw_joint")

# /whole/robot/status/leg_power 的 motor_status 顺序，来自机器人开发说明。
MOTOR_STATUS_NAMES = (
    "ankle_joint",
    "knee_joint",
    "hip_pitch_joint",
    "hip_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)

# 文档/URDF 中的关节限位，单位 rad。脚本会用它做基础安全检查。
JOINT_LIMITS = {
    "ankle_joint": (-1.57, 1.57),
    "knee_joint": (-2.10, 2.10),
    "hip_pitch_joint": (-1.57, 1.57),
    "hip_yaw_joint": (-3.14, 3.14),
}

# 控制器 cartesian_target_to_joint_target 里使用的逆解限位。
IK_JOINT_LIMITS = (
    (-math.pi / 2.0, math.pi / 2.0),
    (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0),
    (-math.pi / 2.0, math.pi / 2.0),
)
IK_LIMIT_MARGIN = 0.02
IK_START_TOLERANCE = 0.003
IK_CONTROLLED_JOINTS = ("ankle_joint", "knee_joint", "hip_pitch_joint")

# 腿部连杆参数来自控制器源码 CommandProcessor_leg.cpp。
L1 = 0.375
L2 = 0.365
L3 = 0.0

# 控制器内部把命令 y 加上 CARTESIAN_Y_OFFSET 后再进入逆运动学。
# 所以命令层 command_y=0 约等于站直最高位，负值表示降低身体。
CARTESIAN_Y_OFFSET = L1 + L2 - 0.000001
CARTESIAN_Y_MAX = L1 + L2 + L3 - 0.000001
CARTESIAN_Y_MIN = 0.38990183791231324
COMMAND_Y_MIN = CARTESIAN_Y_MIN - CARTESIAN_Y_OFFSET
COMMAND_Y_MAX = CARTESIAN_Y_MAX - CARTESIAN_Y_OFFSET
EPS = 1e-9


@dataclass(frozen=True)
class CartesianPose:
    x: float
    y_cartesian: float
    command_y: float
    phi: float
    mode_leg_select: int


def clamp_angle(angle: float) -> float:
    """把角度规整到 [-pi, pi]，和控制器里的 normalize_angle 行为一致。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angular_difference(lhs: float, rhs: float) -> float:
    return abs(clamp_angle(lhs - rhs))


def parse_angle(value: str) -> float:
    """解析角度输入：默认按 rad，支持 90deg / 90° / 1.57rad。"""
    text = value.strip().lower()
    if text.endswith("deg"):
        return math.radians(float(text[:-3]))
    if text.endswith("rad"):
        return float(text[:-3])
    if text.endswith("°"):
        return math.radians(float(text[:-1]))
    return float(text)


ANGLE_OPTION_NAMES = {
    "--ankle",
    "--knee",
    "--hip-pitch",
    "--waist",
    "--hip-yaw",
    "--ankle-delta",
    "--knee-delta",
    "--hip-pitch-delta",
    "--waist-delta",
    "--hip-yaw-delta",
    "--phi",
    "--phi-delta",
    "--max-delta",
    "--max-phi-delta",
    "--max-waist-delta",
}


def normalize_angle_option_argv(argv: Iterable[str]) -> list[str]:
    """把 `--opt -20deg` 规范成 `--opt=-20deg`，避免 argparse 把负号当成新选项。"""
    argv_list = list(argv)
    normalized: list[str] = []
    index = 0
    while index < len(argv_list):
        token = argv_list[index]
        if token in ANGLE_OPTION_NAMES and index + 1 < len(argv_list):
            value = argv_list[index + 1]
            try:
                parse_angle(value)
            except Exception:
                pass
            else:
                normalized.append(f"{token}={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def forward_kinematics(ankle: float, knee: float, hip_pitch: float) -> Tuple[float, float, float]:
    """用当前三段腿部角度估算腿端笛卡尔位置，用来构造安全小量升降目标。"""
    theta12 = ankle + knee
    theta123 = theta12 + hip_pitch
    x = L1 * math.sin(ankle) + L2 * math.sin(theta12) + L3 * math.sin(theta123)
    y = L1 * math.cos(ankle) + L2 * math.cos(theta12) + L3 * math.cos(theta123)
    return x, y, theta123


def inverse_kinematics(x: float, y: float, phi: float) -> Optional[Tuple[float, ...]]:
    """腿部逆解检查。这里主要用于提前拦截明显不可达的笛卡尔目标。"""
    if y > L1 + L2 + L3 or y < 0.0:
        return None

    radius_sq = x * x + y * y
    cos_theta2 = (radius_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if cos_theta2 < -1.0 - 1e-12 or cos_theta2 > 1.0 + 1e-12:
        return None
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    solutions = []
    for sign in (1.0, -1.0):
        sin_theta2 = sign * math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
        theta2 = math.atan2(sin_theta2, cos_theta2)
        a = L1 + L2 * cos_theta2
        b = L2 * sin_theta2
        denominator = a * a + b * b
        if denominator < 1e-12:
            solutions.extend((0.0, 0.0, 0.0))
            continue
        sin_theta1 = (a * x - b * y) / denominator
        cos_theta1 = (a * y + b * x) / denominator
        theta1 = math.atan2(sin_theta1, cos_theta1)
        theta3 = clamp_angle(phi - (theta1 + theta2))
        solutions.extend((theta1, theta2, theta3))

    return tuple(solutions)


def ik_branch_solution(x: float, command_y: float, phi: float, mode_leg_select: int) -> Tuple[float, float, float]:
    """按控制器 mode_leg_select 选择 IK 分支并返回三关节目标。"""
    solutions = inverse_kinematics(x, CARTESIAN_Y_OFFSET + command_y, phi)
    if solutions is None:
        raise ValueError("笛卡尔目标逆解不可达")
    base_index = 0 if mode_leg_select > 0 else 3
    return solutions[base_index], solutions[base_index + 1], solutions[base_index + 2]


def ik_solution_limit_errors(solution: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """计算 IK 三关节目标相对控制器限位的越界量。0 表示在限位内。"""
    errors = []
    for value, (low, high) in zip(solution, IK_JOINT_LIMITS):
        if value < low:
            errors.append(low - value)
        elif value > high:
            errors.append(value - high)
        else:
            errors.append(0.0)
    return tuple(errors)


def ensure_cartesian_start_is_valid(current: Dict[str, float]) -> None:
    """控制器要求笛卡尔规划起点本身也在 IK 限位内。"""
    start_solution = (
        current["ankle_joint"],
        current["knee_joint"],
        current["hip_pitch_joint"],
    )
    errors = ik_solution_limit_errors(start_solution)
    if not any(error > IK_START_TOLERANCE + EPS for error in errors):
        return

    details = []
    suggested_targets = {}
    for joint_name, value, error, (low, high) in zip(
        IK_CONTROLLED_JOINTS,
        start_solution,
        errors,
        IK_JOINT_LIMITS,
    ):
        if error <= IK_START_TOLERANCE + EPS:
            continue
        if value < low:
            suggested_targets[joint_name] = low + IK_LIMIT_MARGIN
        else:
            suggested_targets[joint_name] = high - IK_LIMIT_MARGIN
        details.append(
            f"{joint_name}={value:+.4f} 超出 [{low:.4f}, {high:.4f}]，越界 {error:.4f} rad"
        )

    suggestion_parts = []
    if "ankle_joint" in suggested_targets:
        suggestion_parts.append(f"--ankle {suggested_targets['ankle_joint']:.4f}")
    if "knee_joint" in suggested_targets:
        suggestion_parts.append(f"--knee {suggested_targets['knee_joint']:.4f}")
    if "hip_pitch_joint" in suggested_targets:
        suggestion_parts.append(f"--hip-pitch {suggested_targets['hip_pitch_joint']:.4f}")
    suggestion = " ".join(suggestion_parts)
    raise ValueError(
        "当前姿态已经超出控制器笛卡尔 IK 启动限位，不能直接执行 lift/cartesian。"
        f"{'；'.join(details)}。"
        "请先用 joint 子命令把越界关节收回限位内，例如："
        f"/home/test/LJM/run_lift_waist_control.sh joint {suggestion} --vel 0.03 --execute"
    )


def pick_reachable_lift_phi(
    current: Dict[str, float],
    target_x: float,
    target_command_y: float,
    mode_leg_select: int,
) -> Tuple[float, Tuple[float, float, float]]:
    """升降时自动选择一个控制器 IK 限位内的 phi。

    之前保持当前 phi 会让 hip_pitch 目标落在 -1.60rad 左右，超过控制器
    cartesian_target_to_joint_target 的 -pi/2 限位，控制器会保持不动。
    这里固定 x/y 和 IK 分支，把 hip_pitch 收回到安全限位内，再反推 phi。
    """
    solutions = inverse_kinematics(target_x, CARTESIAN_Y_OFFSET + target_command_y, 0.0)
    if solutions is None:
        raise ValueError("升降目标逆解不可达")

    base_index = 0 if mode_leg_select > 0 else 3
    theta1 = solutions[base_index]
    theta2 = solutions[base_index + 1]

    hip_low, hip_high = IK_JOINT_LIMITS[2]
    safe_low = hip_low + IK_LIMIT_MARGIN
    safe_high = hip_high - IK_LIMIT_MARGIN
    target_hip_pitch = max(safe_low, min(safe_high, current["hip_pitch_joint"]))
    target_phi = clamp_angle(theta1 + theta2 + target_hip_pitch)
    return target_phi, (theta1, theta2, target_hip_pitch)


def infer_mode_leg_select(joints: Dict[str, float]) -> int:
    """根据当前关节姿态推断控制器需要的 mode_leg_select 分支。"""
    x, y, phi = forward_kinematics(
        joints["ankle_joint"],
        joints["knee_joint"],
        joints["hip_pitch_joint"],
    )
    solutions = inverse_kinematics(x, y, phi)
    if solutions is None:
        return 0

    branch0_error = (
        angular_difference(solutions[0], joints["ankle_joint"])
        + angular_difference(solutions[1], joints["knee_joint"])
        + angular_difference(solutions[2], joints["hip_pitch_joint"])
    )
    branch1_error = (
        angular_difference(solutions[3], joints["ankle_joint"])
        + angular_difference(solutions[4], joints["knee_joint"])
        + angular_difference(solutions[5], joints["hip_pitch_joint"])
    )
    return 1 if branch0_error <= branch1_error else 0


def current_cartesian_pose(joints: Dict[str, float]) -> CartesianPose:
    """由 /whole/joint_states 的四关节状态推导当前升降姿态。"""
    x, y_cartesian, phi = forward_kinematics(
        joints["ankle_joint"],
        joints["knee_joint"],
        joints["hip_pitch_joint"],
    )
    return CartesianPose(
        x=x,
        y_cartesian=y_cartesian,
        command_y=y_cartesian - CARTESIAN_Y_OFFSET,
        phi=phi,
        mode_leg_select=infer_mode_leg_select(joints),
    )


def fmt_rad(value: float) -> str:
    return f"{value:+.4f} rad ({math.degrees(value):+.2f} deg)"


def print_joint_table(title: str, joints: Dict[str, float]) -> None:
    print(title)
    for name in LEG_JOINTS:
        label = "腰部/hip_yaw" if name == "hip_yaw_joint" else name
        print(f"  {label:15s} {fmt_rad(joints[name])}")


def print_cartesian_pose(title: str, pose: CartesianPose) -> None:
    print(title)
    print(f"  x                 {pose.x:+.4f} m")
    print(f"  command_y          {pose.command_y:+.4f} m  (0 接近最高位，负值降低身体)")
    print(f"  cartesian_y        {pose.y_cartesian:+.4f} m")
    print(f"  phi               {fmt_rad(pose.phi)}")
    print(f"  mode_leg_select   {pose.mode_leg_select}")


def print_leg_power_status(is_enabled: bool, motor_status: Tuple[float, ...]) -> None:
    print("腿部电机使能状态：")
    print(f"  is_enabled        {str(is_enabled).lower()}")
    if not motor_status:
        print("  motor_status      无状态")
        return
    for index, value in enumerate(motor_status):
        label = MOTOR_STATUS_NAMES[index] if index < len(MOTOR_STATUS_NAMES) else f"axis_{index}"
        ready_text = "ready" if int(value) == 39 else "not_ready"
        print(f"  [{index}] {label:16s} {value:5.0f}  {ready_text}")


def validate_joint_targets(
    current: Dict[str, float],
    target: Dict[str, float],
    max_delta: float,
    allow_large_step: bool,
) -> Iterable[str]:
    """检查关节空间目标，避免单次步长过大或继续冲出限位。"""
    warnings = []
    for joint_name in LEG_JOINTS:
        low, high = JOINT_LIMITS[joint_name]
        value = target[joint_name]
        current_value = current[joint_name]
        if not math.isfinite(value):
            raise ValueError(f"{joint_name} 目标值不是有限数字")

        current_limit_error = limit_error(joint_name, current_value)
        target_limit_error = limit_error(joint_name, value)
        if target_limit_error > 0.0 and target_limit_error > current_limit_error + 1e-6:
            raise ValueError(
                f"{joint_name} 目标 {value:.4f} 超出限位 [{low:.2f}, {high:.2f}]"
            )
        if target_limit_error > 0.0:
            warnings.append(
                f"{joint_name} 当前仍在文档限位 [{low:.2f}, {high:.2f}] 外，"
                f"目标为 {value:.4f} rad，但没有继续向外运动"
            )

        delta = abs(value - current_value)
        if delta > max_delta + EPS and not allow_large_step:
            raise ValueError(
                f"{joint_name} 单次变化 {delta:.4f} rad 超过 --max-delta {max_delta:.4f}；"
                "请减小指令，或确认风险后使用 --allow-large-step"
            )
    return warnings


def limit_error(joint_name: str, value: float) -> float:
    low, high = JOINT_LIMITS[joint_name]
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def validate_cartesian_target(
    current_pose: CartesianPose,
    target_x: float,
    target_command_y: float,
    target_phi: float,
    target_waist: float,
    mode_leg_select: int,
    max_cartesian_delta: float,
    max_phi_delta: float,
    max_waist_delta: float,
    allow_large_step: bool,
) -> None:
    """检查笛卡尔升降目标，避免 y 范围、姿态变化或腰部变化过大。"""
    if target_command_y < COMMAND_Y_MIN or target_command_y > COMMAND_Y_MAX:
        raise ValueError(
            f"command_y {target_command_y:.4f} m 超出脚本安全范围 "
            f"[{COMMAND_Y_MIN:.4f}, {COMMAND_Y_MAX:.4f}]"
        )
    if not allow_large_step:
        if abs(target_x - current_pose.x) > max_cartesian_delta + EPS:
            raise ValueError(
                f"x 单次变化 {abs(target_x - current_pose.x):.4f} m 超过 "
                f"--max-cartesian-delta {max_cartesian_delta:.4f}"
            )
        if abs(target_command_y - current_pose.command_y) > max_cartesian_delta + EPS:
            raise ValueError(
                f"command_y 单次变化 {abs(target_command_y - current_pose.command_y):.4f} m 超过 "
                f"--max-cartesian-delta {max_cartesian_delta:.4f}"
            )
        if angular_difference(target_phi, current_pose.phi) > max_phi_delta + EPS:
            raise ValueError(
                f"phi 单次变化 {angular_difference(target_phi, current_pose.phi):.4f} rad 超过 "
                f"--max-phi-delta {max_phi_delta:.4f}"
            )
        if abs(target_waist) > JOINT_LIMITS["hip_yaw_joint"][1]:
            raise ValueError("腰部目标超过 hip_yaw_joint 限位")

    low, high = JOINT_LIMITS["hip_yaw_joint"]
    if target_waist < low or target_waist > high:
        raise ValueError(f"腰部目标 {target_waist:.4f} rad 超出 [{low:.2f}, {high:.2f}]")

    cartesian_y = CARTESIAN_Y_OFFSET + target_command_y
    if inverse_kinematics(target_x, cartesian_y, target_phi) is None:
        raise ValueError(
            f"笛卡尔目标逆解不可达：x={target_x:.4f}, "
            f"command_y={target_command_y:.4f}, phi={target_phi:.4f}"
        )

    solution = ik_branch_solution(target_x, target_command_y, target_phi, mode_leg_select)
    errors = ik_solution_limit_errors(solution)
    if any(error > 0.003 + EPS for error in errors):
        formatted_solution = ", ".join(f"{value:+.4f}" for value in solution)
        formatted_errors = ", ".join(f"{value:.4f}" for value in errors)
        raise ValueError(
            "笛卡尔目标在当前 mode_leg_select 分支下会超过控制器 IK 关节限位；"
            f"IK=({formatted_solution}), 越界量=({formatted_errors})"
        )


class LiftWaistNode(Node):
    def __init__(self) -> None:
        super().__init__("lift_waist_control_helper")
        self._joint_positions: Dict[str, float] = {}
        self._leg_power_enabled: Optional[bool] = None
        self._leg_motor_status: Tuple[float, ...] = ()

        # 状态订阅：读取当前腿部/升降/腰部四个关节的位置。
        self.create_subscription(JointState, "/whole/joint_states", self._on_joint_state, 10)
        self.create_subscription(LegPowerStatus, "/whole/robot/status/leg_power", self._on_leg_power, 10)

        # 话题备用通道：当服务节点不可用时，可以直接发布控制器订阅的话题。
        self._leg_joint_pub = self.create_publisher(Robotlegjoint, "/leg_joint_position_cmd", 10)
        self._axis_pub = self.create_publisher(Robotservomsg, "/axis_position_cmd", 10)

        # 首选服务通道：服务会等待运动完成并返回 success/message。
        self._leg_abs_client = self.create_client(LegAbsoluteControl, "/leg_absolute_control")
        self._leg_cart_client = self.create_client(LegCartesianControl, "/leg_cartesian_control")
        self._power_client = self.create_client(SetRobotPower, "/set_robot_power")

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in LEG_JOINTS:
                self._joint_positions[name] = float(position)

    def _on_leg_power(self, msg: LegPowerStatus) -> None:
        self._leg_power_enabled = bool(msg.is_enabled)
        self._leg_motor_status = tuple(float(value) for value in msg.motor_status)

    def wait_for_leg_state(self, timeout_sec: float) -> Dict[str, float]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if all(name in self._joint_positions for name in LEG_JOINTS):
                return dict(self._joint_positions)
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = [name for name in LEG_JOINTS if name not in self._joint_positions]
        raise TimeoutError(f"等待 /whole/joint_states 超时，缺少关节：{missing}")

    def wait_for_leg_power(self, timeout_sec: float) -> Tuple[bool, Tuple[float, ...]]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if self._leg_power_enabled is not None:
                return self._leg_power_enabled, self._leg_motor_status
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError("等待 /whole/robot/status/leg_power 超时")

    def wait_until_leg_power(self, expected: bool, timeout_sec: float) -> Tuple[bool, Tuple[float, ...]]:
        deadline = time.monotonic() + timeout_sec
        last_enabled: Optional[bool] = None
        last_status: Tuple[float, ...] = ()
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._leg_power_enabled is None:
                continue
            last_enabled = self._leg_power_enabled
            last_status = self._leg_motor_status
            if last_enabled == expected:
                return last_enabled, last_status
        if last_enabled is None:
            raise TimeoutError("等待 /whole/robot/status/leg_power 超时")
        return last_enabled, last_status

    def require_leg_power(self, timeout_sec: float) -> None:
        is_enabled, motor_status = self.wait_for_leg_power(timeout_sec)
        if is_enabled:
            return
        status_text = ", ".join(f"{value:.0f}" for value in motor_status) or "无状态"
        raise RuntimeError(
            "腿部电机没有真正上使能，拒绝下发运动命令。"
            f"当前 is_enabled=false, motor_status=[{status_text}]。"
            "请先执行：/home/test/LJM/run_lift_waist_control.sh power on --execute"
        )

    def call_leg_absolute(
        self,
        target: Dict[str, float],
        vel: float,
        acc: float,
        service_wait: float,
        motion_timeout: float,
    ) -> Optional[object]:
        if not self._leg_abs_client.wait_for_service(timeout_sec=service_wait):
            return None
        request = LegAbsoluteControl.Request()
        request.ankle_joint = target["ankle_joint"]
        request.knee_joint = target["knee_joint"]
        request.hip_pitch_joint = target["hip_pitch_joint"]
        request.hip_yaw_joint = target["hip_yaw_joint"]
        request.vel = vel
        request.acc = acc
        future = self._leg_abs_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=motion_timeout)
        if not future.done():
            raise TimeoutError(f"/leg_absolute_control 在 {motion_timeout:.1f}s 内未返回")
        return future.result()

    def publish_leg_joint(self, target: Dict[str, float], vel: float, acc: float) -> None:
        msg = Robotlegjoint()
        msg.ankle_joint = target["ankle_joint"]
        msg.knee_joint = target["knee_joint"]
        msg.hip_pitch_joint = target["hip_pitch_joint"]
        msg.hip_yaw_joint = target["hip_yaw_joint"]
        msg.vel = vel
        msg.acc = acc
        # 连发几次是为了避免一次性话题发布被现场 DDS/订阅端初始化时机吞掉。
        self._publish_reliably(self._leg_joint_pub, msg)

    def call_leg_cartesian(
        self,
        x: float,
        command_y: float,
        phi: float,
        waist: float,
        vel: float,
        mode_leg_select: int,
        service_wait: float,
        motion_timeout: float,
    ) -> Optional[object]:
        if not self._leg_cart_client.wait_for_service(timeout_sec=service_wait):
            return None
        request = LegCartesianControl.Request()
        request.x = x
        request.y = command_y
        request.phi = phi
        request.hip_yaw_position = waist
        request.vel = vel
        request.mode_leg_select = float(mode_leg_select)
        future = self._leg_cart_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=motion_timeout)
        if not future.done():
            raise TimeoutError(f"/leg_cartesian_control 在 {motion_timeout:.1f}s 内未返回")
        return future.result()

    def publish_axis_cartesian(
        self,
        x: float,
        command_y: float,
        phi: float,
        waist: float,
        vel: float,
        mode_leg_select: int,
    ) -> None:
        msg = Robotservomsg()
        msg.run_mode = 1
        msg.x = x
        msg.y = command_y
        msg.phi = phi
        msg.hip_yaw_position = waist
        msg.vel = vel
        msg.mode_leg_select = int(mode_leg_select)
        self._publish_reliably(self._axis_pub, msg)

    def set_power(self, enable: bool, service_wait: float, timeout_sec: float) -> object:
        if not self._power_client.wait_for_service(timeout_sec=service_wait):
            raise TimeoutError("/set_robot_power 服务不可用")
        request = SetRobotPower.Request()
        request.enable = enable
        future = self._power_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(f"/set_robot_power 在 {timeout_sec:.1f}s 内未返回")
        return future.result()

    def _publish_reliably(self, publisher, msg: object) -> None:
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
            publisher.publish(msg)
            time.sleep(0.05)


def add_common_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true", help="真正下发命令；不加时只打印 dry-run")
    parser.add_argument("--transport", choices=("auto", "service", "topic"), default="auto", help="发送方式")
    parser.add_argument("--unsafe", action="store_true", help="跳过脚本侧步长、限位和上使能检查")
    parser.add_argument("--vel", type=float, default=0.08, help="最大速度")
    parser.add_argument("--acc", type=float, default=0.20, help="关节控制最大加速度")
    parser.add_argument("--state-timeout", type=float, default=3.0, help="等待 /whole/joint_states 的超时时间")
    parser.add_argument("--service-wait", type=float, default=2.0, help="等待 ROS2 服务出现的超时时间")
    parser.add_argument("--motion-timeout", type=float, default=30.0, help="等待服务返回/运动完成的超时时间")
    parser.add_argument("--power-timeout", type=float, default=3.0, help="等待腿部电机使能状态的超时时间")
    parser.add_argument("--skip-power-check", action="store_true", help="跳过腿部上使能检查，不建议现场调试使用")
    parser.add_argument("--allow-large-step", action="store_true", help="允许超过默认小步长保护")


def add_joint_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ankle", type=parse_angle, help="ankle_joint 绝对目标，默认 rad，支持 90deg")
    parser.add_argument("--knee", type=parse_angle, help="knee_joint 绝对目标，默认 rad，支持 90deg")
    parser.add_argument("--hip-pitch", type=parse_angle, help="hip_pitch_joint 绝对目标，默认 rad，支持 90deg")
    parser.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, help="腰部 hip_yaw_joint 绝对目标，默认 rad，支持 90deg")
    parser.add_argument("--ankle-delta", type=parse_angle, default=0.0, help="ankle_joint 增量，默认 rad，支持 90deg")
    parser.add_argument("--knee-delta", type=parse_angle, default=0.0, help="knee_joint 增量，默认 rad，支持 90deg")
    parser.add_argument("--hip-pitch-delta", type=parse_angle, default=0.0, help="hip_pitch_joint 增量，默认 rad，支持 90deg")
    parser.add_argument("--waist-delta", "--hip-yaw-delta", dest="waist_delta", type=parse_angle, default=0.0, help="腰部增量，默认 rad，支持 90deg")
    parser.add_argument("--max-delta", type=parse_angle, default=0.25, help="单次关节变化保护阈值，默认 rad，支持 90deg")


def build_joint_target(args: argparse.Namespace, current: Dict[str, float]) -> Dict[str, float]:
    """根据当前关节状态、绝对目标和增量目标合成最终关节目标。"""
    target = dict(current)
    absolute_fields = {
        "ankle_joint": args.ankle,
        "knee_joint": args.knee,
        "hip_pitch_joint": args.hip_pitch,
        "hip_yaw_joint": args.waist,
    }
    delta_fields = {
        "ankle_joint": args.ankle_delta,
        "knee_joint": args.knee_delta,
        "hip_pitch_joint": args.hip_pitch_delta,
        "hip_yaw_joint": args.waist_delta,
    }
    for joint_name, value in absolute_fields.items():
        if value is not None:
            target[joint_name] = value
    for joint_name, delta in delta_fields.items():
        target[joint_name] += delta
    return target


def command_has_joint_change(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (args.ankle, args.knee, args.hip_pitch, args.waist)
    ) or any(
        abs(value) > 0.0
        for value in (args.ankle_delta, args.knee_delta, args.hip_pitch_delta, args.waist_delta)
    )


def run_status(node: LiftWaistNode, args: argparse.Namespace) -> int:
    current = node.wait_for_leg_state(args.state_timeout)
    print_joint_table("当前腿部/升降/腰部关节：", current)
    print_cartesian_pose("当前推导出的升降笛卡尔姿态：", current_cartesian_pose(current))
    print(f"脚本允许的 command_y 安全范围：[{COMMAND_Y_MIN:.4f}, {COMMAND_Y_MAX:.4f}] m")
    try:
        is_enabled, motor_status = node.wait_for_leg_power(args.power_timeout)
        print_leg_power_status(is_enabled, motor_status)
    except TimeoutError as exc:
        print(f"腿部电机使能状态：读取失败，{exc}")
    return 0


def run_joint(node: LiftWaistNode, args: argparse.Namespace) -> int:
    current = node.wait_for_leg_state(args.state_timeout)
    target = build_joint_target(args, current)

    if not command_has_joint_change(args):
        print_joint_table("当前关节状态：未请求目标变化", current)
        return 0

    if args.unsafe:
        print("警告：已启用 --unsafe，脚本侧将跳过步长、限位和上使能检查。")
        warnings = ()
    else:
        warnings = validate_joint_targets(current, target, args.max_delta, args.allow_large_step)
    print_joint_table("当前关节：", current)
    print_joint_table("目标关节：", target)
    for warning in warnings:
        print(f"警告：{warning}")
    print(f"速度={args.vel:.3f} rad/s 加速度={args.acc:.3f} rad/s^2 发送方式={args.transport}")

    if not args.execute:
        print("当前是 dry-run，只打印目标。加 --execute 才会真正下发。")
        return 0

    if not args.skip_power_check and not args.unsafe:
        node.require_leg_power(args.power_timeout)

    if args.transport in ("auto", "service"):
        response = node.call_leg_absolute(
            target,
            args.vel,
            args.acc,
            args.service_wait,
            args.motion_timeout,
        )
        if response is not None:
            print(f"/leg_absolute_control: success={response.success} message={response.message}")
            return 0 if response.success else 2
        if args.transport == "service":
            raise TimeoutError("/leg_absolute_control 服务不可用")

    node.publish_leg_joint(target, args.vel, args.acc)
    print("已发布 /leg_joint_position_cmd")
    return 0


def target_from_optional(base: float, absolute: Optional[float], delta: float) -> float:
    """如果给了绝对值就以绝对值为基准，否则从当前值叠加增量。"""
    value = base if absolute is None else absolute
    return value + delta


def run_lift(node: LiftWaistNode, args: argparse.Namespace) -> int:
    current = node.wait_for_leg_state(args.state_timeout)
    pose = current_cartesian_pose(current)
    if not args.unsafe:
        ensure_cartesian_start_is_valid(current)

    requested_delta = args.y_delta + args.up - args.down
    if requested_delta == 0.0 and args.waist is None and args.waist_delta == 0.0:
        print_joint_table("当前关节状态：未请求目标变化", current)
        print_cartesian_pose("当前推导出的升降笛卡尔姿态：", pose)
        return 0

    target_x = pose.x
    target_command_y = pose.command_y + requested_delta
    target_phi, joint_preview = pick_reachable_lift_phi(
        current,
        target_x,
        target_command_y,
        pose.mode_leg_select,
    )
    target_waist = target_from_optional(current["hip_yaw_joint"], args.waist, args.waist_delta)
    target_cart_y = CARTESIAN_Y_OFFSET + target_command_y

    if not args.unsafe:
        validate_cartesian_target(
            pose,
            target_x,
            target_command_y,
            target_phi,
            target_waist,
            pose.mode_leg_select,
            args.max_cartesian_delta,
            args.max_phi_delta,
            args.max_waist_delta,
            args.allow_large_step,
        )
        if not args.allow_large_step and abs(target_waist - current["hip_yaw_joint"]) > args.max_waist_delta + EPS:
            raise ValueError(
                f"腰部单次变化 {abs(target_waist - current['hip_yaw_joint']):.4f} rad 超过 "
                f"--max-waist-delta {args.max_waist_delta:.4f}"
            )

    print_joint_table("当前关节：", current)
    print_cartesian_pose("当前推导出的升降笛卡尔姿态：", pose)
    print("目标升降命令：")
    print(f"  x                 {target_x:+.4f} m")
    print(f"  command_y          {target_command_y:+.4f} m")
    print(f"  cartesian_y        {target_cart_y:+.4f} m")
    print(f"  phi               {fmt_rad(target_phi)}")
    print(f"  腰部/hip_yaw      {fmt_rad(target_waist)}")
    print(f"  mode_leg_select   {pose.mode_leg_select}")
    print(f"  速度              {args.vel:.3f}")
    print("预估腿部 IK 关节目标：")
    print(f"  ankle_joint       {fmt_rad(joint_preview[0])}")
    print(f"  knee_joint        {fmt_rad(joint_preview[1])}")
    print(f"  hip_pitch_joint   {fmt_rad(joint_preview[2])}")

    if not args.execute:
        print("当前是 dry-run，只打印目标。加 --execute 才会真正下发。")
        return 0

    return send_cartesian(
        node,
        args,
        target_x,
        target_command_y,
        target_phi,
        target_waist,
        pose.mode_leg_select,
    )


def run_cartesian(node: LiftWaistNode, args: argparse.Namespace) -> int:
    current = node.wait_for_leg_state(args.state_timeout)
    pose = current_cartesian_pose(current)
    if not args.unsafe:
        ensure_cartesian_start_is_valid(current)

    target_x = target_from_optional(pose.x, args.x, args.x_delta)
    target_command_y = target_from_optional(pose.command_y, args.y, args.y_delta)
    target_phi = target_from_optional(pose.phi, args.phi, args.phi_delta)
    target_waist = target_from_optional(current["hip_yaw_joint"], args.waist, args.waist_delta)
    mode_leg_select = pose.mode_leg_select if args.mode_leg_select is None else args.mode_leg_select

    if not args.unsafe:
        validate_cartesian_target(
            pose,
            target_x,
            target_command_y,
            target_phi,
            target_waist,
            mode_leg_select,
            args.max_cartesian_delta,
            args.max_phi_delta,
            args.max_waist_delta,
            args.allow_large_step,
        )
        if not args.allow_large_step and abs(target_waist - current["hip_yaw_joint"]) > args.max_waist_delta + EPS:
            raise ValueError(
                f"腰部单次变化 {abs(target_waist - current['hip_yaw_joint']):.4f} rad 超过 "
                f"--max-waist-delta {args.max_waist_delta:.4f}"
            )

    print_joint_table("当前关节：", current)
    print_cartesian_pose("当前推导出的升降笛卡尔姿态：", pose)
    print("目标笛卡尔命令：")
    print(f"  x                 {target_x:+.4f} m")
    print(f"  command_y          {target_command_y:+.4f} m")
    print(f"  cartesian_y        {CARTESIAN_Y_OFFSET + target_command_y:+.4f} m")
    print(f"  phi               {fmt_rad(target_phi)}")
    print(f"  腰部/hip_yaw      {fmt_rad(target_waist)}")
    print(f"  mode_leg_select   {mode_leg_select}")
    print(f"  速度              {args.vel:.3f}")

    if not args.execute:
        print("当前是 dry-run，只打印目标。加 --execute 才会真正下发。")
        return 0

    return send_cartesian(
        node,
        args,
        target_x,
        target_command_y,
        target_phi,
        target_waist,
        mode_leg_select,
    )


def send_cartesian(
    node: LiftWaistNode,
    args: argparse.Namespace,
    x: float,
    command_y: float,
    phi: float,
    waist: float,
    mode_leg_select: int,
) -> int:
    """优先调用 /leg_cartesian_control，服务不可用且 transport=auto 时退回话题发布。"""
    if not args.skip_power_check and not args.unsafe:
        node.require_leg_power(args.power_timeout)

    if args.transport in ("auto", "service"):
        response = node.call_leg_cartesian(
            x,
            command_y,
            phi,
            waist,
            args.vel,
            mode_leg_select,
            args.service_wait,
            args.motion_timeout,
        )
        if response is not None:
            print(f"/leg_cartesian_control: success={response.success} message={response.message}")
            return 0 if response.success else 2
        if args.transport == "service":
            raise TimeoutError("/leg_cartesian_control 服务不可用")

    node.publish_axis_cartesian(x, command_y, phi, waist, args.vel, mode_leg_select)
    print("已发布 /axis_position_cmd")
    return 0


def run_power(node: LiftWaistNode, args: argparse.Namespace) -> int:
    enable = args.state == "on"
    print(f"电机使能请求：{'上使能' if enable else '下使能'}")
    if not args.execute:
        print("当前是 dry-run。加 --execute 才会调用 /set_robot_power。")
        return 0
    response = node.set_power(enable, args.service_wait, args.timeout)
    print(f"/set_robot_power: success={response.success} message={response.message}")
    if not response.success:
        return 2

    actual_enabled, motor_status = node.wait_until_leg_power(enable, args.verify_timeout)
    status_text = ", ".join(f"{value:.0f}" for value in motor_status) or "无状态"
    print(f"/whole/robot/status/leg_power: is_enabled={actual_enabled} motor_status=[{status_text}]")
    if actual_enabled != enable:
        print("服务已返回，但硬件 power 状态没有达到目标。需要排查急停、电机驱动或 EtherCAT 状态。")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sailor R1 Pro 升降机构和腰部关节控制辅助脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="读取 /whole/joint_states 当前状态")
    status.add_argument("--state-timeout", type=float, default=3.0, help="等待关节状态超时时间")
    status.add_argument("--power-timeout", type=float, default=3.0, help="等待腿部电机使能状态超时时间")
    status.set_defaults(func=run_status)

    joint = subparsers.add_parser("joint", help="关节空间控制：绝对目标或增量目标")
    add_common_motion_args(joint)
    add_joint_target_args(joint)
    joint.set_defaults(func=run_joint)

    lift = subparsers.add_parser("lift", help="升降控制：基于当前姿态修改 command_y，可同时带腰部目标")
    add_common_motion_args(lift)
    lift.add_argument("--down", type=float, default=0.0, help="降低身体，单位 m")
    lift.add_argument("--up", type=float, default=0.0, help="升高身体，单位 m")
    lift.add_argument("--y-delta", type=float, default=0.0, help="直接指定 command_y 增量，单位 m")
    lift.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, help="腰部绝对目标，默认 rad，支持 90deg")
    lift.add_argument("--waist-delta", "--hip-yaw-delta", dest="waist_delta", type=parse_angle, default=0.0, help="腰部增量，默认 rad，支持 90deg")
    lift.add_argument("--max-cartesian-delta", type=float, default=0.03, help="单次笛卡尔位置变化保护阈值，单位 m")
    lift.add_argument("--max-phi-delta", type=parse_angle, default=0.12, help="单次 phi 姿态变化保护阈值，默认 rad，支持 90deg")
    lift.add_argument("--max-waist-delta", type=parse_angle, default=0.20, help="单次腰部变化保护阈值，默认 rad，支持 90deg")
    lift.set_defaults(func=run_lift)

    cartesian = subparsers.add_parser("cartesian", help="原始腿部笛卡尔控制")
    add_common_motion_args(cartesian)
    cartesian.add_argument("--x", type=float, help="x 绝对目标，单位 m")
    cartesian.add_argument("--y", type=float, help="command_y 绝对目标，单位 m")
    cartesian.add_argument("--phi", type=parse_angle, help="phi 绝对目标，默认 rad，支持 90deg")
    cartesian.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, help="腰部绝对目标，默认 rad，支持 90deg")
    cartesian.add_argument("--x-delta", type=float, default=0.0, help="x 增量，单位 m")
    cartesian.add_argument("--y-delta", type=float, default=0.0, help="command_y 增量，单位 m")
    cartesian.add_argument("--phi-delta", type=parse_angle, default=0.0, help="phi 增量，默认 rad，支持 90deg")
    cartesian.add_argument("--waist-delta", "--hip-yaw-delta", dest="waist_delta", type=parse_angle, default=0.0, help="腰部增量，默认 rad，支持 90deg")
    cartesian.add_argument("--mode-leg-select", type=int, choices=(0, 1), help="逆解分支；默认根据当前姿态自动推断")
    cartesian.add_argument("--max-cartesian-delta", type=float, default=0.03, help="单次笛卡尔位置变化保护阈值，单位 m")
    cartesian.add_argument("--max-phi-delta", type=parse_angle, default=0.12, help="单次 phi 姿态变化保护阈值，默认 rad，支持 90deg")
    cartesian.add_argument("--max-waist-delta", type=parse_angle, default=0.20, help="单次腰部变化保护阈值，默认 rad，支持 90deg")
    cartesian.set_defaults(func=run_cartesian)

    power = subparsers.add_parser("power", help="调用 /set_robot_power 上使能或下使能")
    power.add_argument("state", choices=("on", "off"))
    power.add_argument("--execute", action="store_true", help="真正调用 /set_robot_power")
    power.add_argument("--service-wait", type=float, default=2.0, help="等待服务出现的超时时间")
    power.add_argument("--timeout", type=float, default=5.0, help="等待服务返回的超时时间")
    power.add_argument("--verify-timeout", type=float, default=5.0, help="等待真实 leg_power 状态到位的超时时间")
    power.set_defaults(func=run_power)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    normalized_argv = normalize_angle_option_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    rclpy.init()
    node = LiftWaistNode()
    try:
        return int(args.func(node, args))
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
