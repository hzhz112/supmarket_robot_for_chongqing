#!/usr/bin/env python3
"""Interactive Cartesian absolute-control console."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    ROOT / "ros2_robot_controller_runtime" / "src",
    ROOT / "ros2_robot_controller_runtime" / "src" / "sailor_r1_pro_description",
    ROOT / "ros2_control_source_partial",
    ROOT / "ros2_robot_controller_runtime" / "install" / "local" / "lib" / "python3.10" / "dist-packages",
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from robot_control_msg.msg import ArmPowerStatus, EndEffectorPose
from robot_control_msg.srv import CartesianPathAbsoluteControl, SetRobotPower
from sailor_r1_pro_description.arm_topics import ArmCartesianTarget, DualArmControlClient


def parse_angle(text: str) -> float:
    value = text.strip().lower()
    if value.endswith("deg"):
        return math.radians(float(value[:-3]))
    if value.endswith("rad"):
        return float(value[:-3])
    if value.endswith("\u00b0"):
        return math.radians(float(value[:-1]))
    # Status output is printed in degrees, so bare RPY values are degrees.
    # Append "rad" when radians are intended.
    return math.radians(float(value))


def split_values(text: str) -> list[str]:
    return [part for part in text.replace("，", " ").replace(",", " ").split() if part]


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


def pose_to_target(pose: Pose) -> ArmCartesianTarget:
    roll, pitch, yaw = quaternion_to_rpy(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return ArmCartesianTarget(
        x=float(pose.position.x),
        y=float(pose.position.y),
        z=float(pose.position.z),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_conjugate(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / norm for component in quaternion)


def target_quaternion(target: ArmCartesianTarget) -> tuple[float, float, float, float]:
    if any(abs(value) > 1e-9 for value in (target.qx, target.qy, target.qz, target.qw)):
        return normalize_quaternion((target.qx, target.qy, target.qz, target.qw))
    return normalize_quaternion(rpy_to_quaternion(target.roll, target.pitch, target.yaw))


def rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    rotated = quaternion_multiply(
        quaternion_multiply((qx, qy, qz, qw), (vx, vy, vz, 0.0)),
        quaternion_conjugate((qx, qy, qz, qw)),
    )
    return rotated[0], rotated[1], rotated[2]


def compose_target(
    base: ArmCartesianTarget,
    offset_translation: tuple[float, float, float],
    offset_rotation: tuple[float, float, float, float],
) -> ArmCartesianTarget:
    base_quaternion = target_quaternion(base)
    rotated_offset = rotate_vector(base_quaternion, offset_translation)
    tcp_quaternion = normalize_quaternion(quaternion_multiply(base_quaternion, offset_rotation))
    roll, pitch, yaw = quaternion_to_rpy(*tcp_quaternion)
    return ArmCartesianTarget(
        x=base.x + rotated_offset[0],
        y=base.y + rotated_offset[1],
        z=base.z + rotated_offset[2],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        qx=tcp_quaternion[0],
        qy=tcp_quaternion[1],
        qz=tcp_quaternion[2],
        qw=tcp_quaternion[3],
    )


def build_tcp_from_ee_target(
    current_ee: ArmCartesianTarget,
    current_tcp: ArmCartesianTarget,
    target_ee: ArmCartesianTarget,
) -> ArmCartesianTarget:
    current_ee_quaternion = target_quaternion(current_ee)
    current_tcp_quaternion = target_quaternion(current_tcp)
    offset_rotation = normalize_quaternion(
        quaternion_multiply(quaternion_conjugate(current_ee_quaternion), current_tcp_quaternion)
    )
    position_delta_world = (
        current_tcp.x - current_ee.x,
        current_tcp.y - current_ee.y,
        current_tcp.z - current_ee.z,
    )
    offset_translation = rotate_vector(quaternion_conjugate(current_ee_quaternion), position_delta_world)
    return compose_target(target_ee, offset_translation, offset_rotation)


def fmt_target(target: ArmCartesianTarget) -> str:
    return (
        f"x={target.x:+.3f} y={target.y:+.3f} z={target.z:+.3f} "
        f"rpy=({math.degrees(target.roll):+.1f},{math.degrees(target.pitch):+.1f},{math.degrees(target.yaw):+.1f})deg"
    )


def fmt_copyable_target(target: ArmCartesianTarget) -> str:
    return (
        f"{target.x:+.3f} {target.y:+.3f} {target.z:+.3f} "
        f"{math.degrees(target.roll):+.1f} {math.degrees(target.pitch):+.1f} {math.degrees(target.yaw):+.1f}"
    )


def target_to_pose_msg(target: ArmCartesianTarget) -> Pose:
    qx, qy, qz, qw = target_quaternion(target)
    pose = Pose()
    pose.position.x = float(target.x)
    pose.position.y = float(target.y)
    pose.position.z = float(target.z)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


def merge_pose_input(text: str, current: ArmCartesianTarget) -> ArmCartesianTarget:
    parts = split_values(text)
    if len(parts) == 3:
        return ArmCartesianTarget(
            x=float(parts[0]),
            y=float(parts[1]),
            z=float(parts[2]),
            roll=current.roll,
            pitch=current.pitch,
            yaw=current.yaw,
        )
    if len(parts) == 6:
        return ArmCartesianTarget(
            x=float(parts[0]),
            y=float(parts[1]),
            z=float(parts[2]),
            roll=parse_angle(parts[3]),
            pitch=parse_angle(parts[4]),
            yaw=parse_angle(parts[5]),
        )
    raise ValueError("请输入 3 个值(x y z) 或 6 个值(x y z roll pitch yaw)")


def apply_pose_input(text: str, current: ArmCartesianTarget) -> ArmCartesianTarget:
    return merge_pose_input(text, current)


def prompt_menu(title: str, options: list[tuple[str, str]]) -> str | None:
    print(title)
    for key, label in options:
        print(f"{key}: {label}")
    choice = input("请输入编号: ").strip()
    valid = {key for key, _ in options}
    if choice not in valid:
        print(f"无效输入，只能是: {' / '.join(sorted(valid))}")
        return None
    return choice


def interpolate_target(start: ArmCartesianTarget, goal: ArmCartesianTarget, ratio: float) -> ArmCartesianTarget:
    return ArmCartesianTarget(
        x=start.x + (goal.x - start.x) * ratio,
        y=start.y + (goal.y - start.y) * ratio,
        z=start.z + (goal.z - start.z) * ratio,
        roll=start.roll + (goal.roll - start.roll) * ratio,
        pitch=start.pitch + (goal.pitch - start.pitch) * ratio,
        yaw=start.yaw + (goal.yaw - start.yaw) * ratio,
    )


def cartesian_distance(start: ArmCartesianTarget, goal: ArmCartesianTarget) -> float:
    return math.sqrt(
        (goal.x - start.x) ** 2 +
        (goal.y - start.y) ** 2 +
        (goal.z - start.z) ** 2
    )


def angular_distance_deg(start_deg: float, goal_deg: float) -> float:
    delta = goal_deg - start_deg
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return abs(delta)


def pose_error(current: ArmCartesianTarget, goal: ArmCartesianTarget) -> tuple[float, float]:
    pos_err = cartesian_distance(current, goal)
    rot_err_deg = max(
        angular_distance_deg(math.degrees(current.roll), math.degrees(goal.roll)),
        angular_distance_deg(math.degrees(current.pitch), math.degrees(goal.pitch)),
        angular_distance_deg(math.degrees(current.yaw), math.degrees(goal.yaw)),
    )
    return pos_err, rot_err_deg


def joint_target_text(arm: str, left_target: ArmCartesianTarget, right_target: ArmCartesianTarget) -> str:
    if arm == "left":
        return f"左臂TCP  {fmt_target(left_target)}"
    if arm == "right":
        return f"右臂TCP  {fmt_target(right_target)}"
    return f"左臂TCP  {fmt_target(left_target)}\n右臂TCP  {fmt_target(right_target)}"


def build_waypoints(start: ArmCartesianTarget, goal: ArmCartesianTarget, step_m: float) -> list[ArmCartesianTarget]:
    segment_count = max(1, int(math.ceil(cartesian_distance(start, goal) / max(step_m, 1e-4))))
    return [interpolate_target(start, goal, index / segment_count) for index in range(1, segment_count + 1)]


def build_targets_from_source(
    source: str,
    arm: str,
    left_text: str | None,
    right_text: str | None,
    left_ee_current: ArmCartesianTarget,
    right_ee_current: ArmCartesianTarget,
    left_tcp_current: ArmCartesianTarget,
    right_tcp_current: ArmCartesianTarget,
) -> tuple[ArmCartesianTarget, ArmCartesianTarget, ArmCartesianTarget, ArmCartesianTarget]:
    if arm == "left":
        left_source_target = apply_pose_input(left_text or "", left_tcp_current if source == "tcp" else left_ee_current)
        right_source_target = right_tcp_current if source == "tcp" else right_ee_current
    elif arm == "right":
        left_source_target = left_tcp_current if source == "tcp" else left_ee_current
        right_source_target = apply_pose_input(right_text or "", right_tcp_current if source == "tcp" else right_ee_current)
    else:
        left_source_target = apply_pose_input(left_text or "", left_tcp_current if source == "tcp" else left_ee_current)
        right_source_target = apply_pose_input(right_text or "", right_tcp_current if source == "tcp" else right_ee_current)

    if source == "tcp":
        left_target = left_source_target
        right_target = right_source_target
    else:
        left_target = build_tcp_from_ee_target(left_ee_current, left_tcp_current, left_source_target)
        right_target = build_tcp_from_ee_target(right_ee_current, right_tcp_current, right_source_target)

    return left_source_target, right_source_target, left_target, right_target


class InteractiveState(Node):
    def __init__(self) -> None:
        super().__init__("cartesian_interactive_console")
        self.left_ee_pose: Pose | None = None
        self.right_ee_pose: Pose | None = None
        self.left_tcp_pose: Pose | None = None
        self.right_tcp_pose: Pose | None = None
        self.arm_power: ArmPowerStatus | None = None
        self.create_subscription(EndEffectorPose, "/end_effector_pose", self._on_ee_pose, 10)
        self.create_subscription(EndEffectorPose, "/arm_tcp_pose", self._on_tcp_pose, 10)
        self.create_subscription(ArmPowerStatus, "/whole/robot/status/arm_power", self._on_power, 10)
        self.create_subscription(ArmPowerStatus, "/robot/status/arm_power", self._on_power, 10)
        self._power_client = self.create_client(SetRobotPower, "/set_robot_power")
        self._cartesian_path_client = self.create_client(
            CartesianPathAbsoluteControl,
            "/cartesian_path_absolute_control",
        )

    def _on_ee_pose(self, msg: EndEffectorPose) -> None:
        self.left_ee_pose = msg.left_ee_pose
        self.right_ee_pose = msg.right_ee_pose

    def _on_tcp_pose(self, msg: EndEffectorPose) -> None:
        self.left_tcp_pose = msg.left_ee_pose
        self.right_tcp_pose = msg.right_ee_pose

    def _on_power(self, msg: ArmPowerStatus) -> None:
        self.arm_power = msg

    def wait_for_state(self, timeout_sec: float = 3.0) -> tuple[Pose, Pose, Pose, Pose]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if all(
                pose is not None
                for pose in (
                    self.left_ee_pose,
                    self.right_ee_pose,
                    self.left_tcp_pose,
                    self.right_tcp_pose,
                )
            ):
                return self.left_ee_pose, self.right_ee_pose, self.left_tcp_pose, self.right_tcp_pose
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError("等待末端状态超时")

    def power_on(self) -> None:
        req = SetRobotPower.Request()
        req.enable = True
        if self._power_client.wait_for_service(timeout_sec=2.0):
            future = self._power_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

    def call_cartesian_path_absolute(
        self,
        left_waypoints: list[ArmCartesianTarget],
        right_waypoints: list[ArmCartesianTarget],
        vel: float,
        acc: float,
        service_wait: float = 2.0,
        motion_timeout: float = 30.0,
    ) -> tuple[str, object]:
        if not self._cartesian_path_client.wait_for_service(timeout_sec=service_wait):
            raise TimeoutError("等待 /cartesian_path_absolute_control 服务超时")
        req = CartesianPathAbsoluteControl.Request()
        req.left_waypoints = [target_to_pose_msg(target) for target in left_waypoints]
        req.right_waypoints = [target_to_pose_msg(target) for target in right_waypoints]
        req.left_blend_radii = []
        req.right_blend_radii = []
        req.vel = float(vel)
        req.acc = float(acc)
        future = self._cartesian_path_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=motion_timeout)
        if not future.done():
            raise TimeoutError(f"/cartesian_path_absolute_control did not return within {motion_timeout:.1f}s")
        return "/cartesian_path_absolute_control", future.result()


def print_status(state: InteractiveState) -> tuple[ArmCartesianTarget, ArmCartesianTarget, ArmCartesianTarget, ArmCartesianTarget]:
    left_ee_pose, right_ee_pose, left_tcp_pose, right_tcp_pose = state.wait_for_state()
    left_ee = pose_to_target(left_ee_pose)
    right_ee = pose_to_target(right_ee_pose)
    left_tcp = pose_to_target(left_tcp_pose)
    right_tcp = pose_to_target(right_tcp_pose)
    print("\n当前状态")
    print(f"左臂 EE  {fmt_copyable_target(left_ee)}")
    print(f"左臂 TCP {fmt_copyable_target(left_tcp)}")
    print(f"右臂 EE  {fmt_copyable_target(right_ee)}")
    print(f"右臂 TCP {fmt_copyable_target(right_tcp)}")
    if state.arm_power is not None:
        status = ",".join(str(int(round(v))) for v in state.arm_power.motor_status)
        print(f"手臂使能  enabled={state.arm_power.is_enabled} motor_status=[{status}]")
    return left_ee, right_ee, left_tcp, right_tcp


def wait_until_segment_reached(
    state: InteractiveState,
    arm: str,
    left_target: ArmCartesianTarget,
    right_target: ArmCartesianTarget,
    timeout_sec: float = 8.0,
    pos_tol: float = 0.008,
    rot_tol_deg: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and rclpy.ok():
        try:
            _, _, left_tcp, right_tcp = print_status(state)
        except Exception:
            time.sleep(0.1)
            continue

        if arm == "left":
            pos_err, rot_err_deg = pose_error(left_tcp, left_target)
            if pos_err <= pos_tol and rot_err_deg <= rot_tol_deg:
                return True
        elif arm == "right":
            pos_err, rot_err_deg = pose_error(right_tcp, right_target)
            if pos_err <= pos_tol and rot_err_deg <= rot_tol_deg:
                return True
        else:
            left_pos_err, left_rot_err_deg = pose_error(left_tcp, left_target)
            right_pos_err, right_rot_err_deg = pose_error(right_tcp, right_target)
            if (
                left_pos_err <= pos_tol and left_rot_err_deg <= rot_tol_deg and
                right_pos_err <= pos_tol and right_rot_err_deg <= rot_tol_deg
            ):
                return True
        time.sleep(0.1)
    return False


def execute_segmented_move(
    state: InteractiveState,
    client: DualArmControlClient,
    arm: str,
    left_goal: ArmCartesianTarget,
    right_goal: ArmCartesianTarget,
    step_m: float,
    vel: float,
    acc: float,
) -> None:
    left_ee_current, right_ee_current, left_tcp_current, right_tcp_current = print_status(state)
    if arm == "left":
        segment_count = max(1, int(math.ceil(cartesian_distance(left_tcp_current, left_goal) / max(step_m, 1e-4))))
    elif arm == "right":
        segment_count = max(1, int(math.ceil(cartesian_distance(right_tcp_current, right_goal) / max(step_m, 1e-4))))
    else:
        left_count = int(math.ceil(cartesian_distance(left_tcp_current, left_goal) / max(step_m, 1e-4)))
        right_count = int(math.ceil(cartesian_distance(right_tcp_current, right_goal) / max(step_m, 1e-4)))
        segment_count = max(1, left_count, right_count)

    print(f"\n分段执行  segments={segment_count} step_m<={step_m:.3f}")
    state.power_on()

    for index in range(1, segment_count + 1):
        try:
            _, _, fresh_left_tcp, fresh_right_tcp = print_status(state)
        except Exception as exc:
            print(f"第 {index}/{segment_count} 段刷新状态失败: {exc}")
            return

        ratio = index / segment_count
        if arm == "left":
            left_target = interpolate_target(fresh_left_tcp, left_goal, 1.0 / (segment_count - index + 1))
            right_target = fresh_right_tcp
        elif arm == "right":
            left_target = fresh_left_tcp
            right_target = interpolate_target(fresh_right_tcp, right_goal, 1.0 / (segment_count - index + 1))
        else:
            left_target = interpolate_target(fresh_left_tcp, left_goal, 1.0 / (segment_count - index + 1))
            right_target = interpolate_target(fresh_right_tcp, right_goal, 1.0 / (segment_count - index + 1))

        print(f"\n第 {index}/{segment_count} 段 ratio={ratio:.3f}")
        print(joint_target_text(arm, left_target, right_target))
        try:
            service_name, response = client.call_cartesian_absolute(
                left=left_target,
                right=right_target,
                vel=vel,
                acc=acc,
                service_wait=2.0,
                motion_timeout=30.0,
            )
        except Exception as exc:
            print(f"第 {index}/{segment_count} 段调用失败: {exc}")
            return

        success = bool(getattr(response, "success", False))
        print(f"service: {service_name}")
        print(f"success: {success}")
        print(f"message: {getattr(response, 'message', '')}")
        if not success:
            print("分段执行中止")
            return
        if not wait_until_segment_reached(state, arm, left_target, right_target):
            print("当前分段等待到位超时，分段执行中止")
            return


def execute_path_move(
    state: InteractiveState,
    arm: str,
    left_start: ArmCartesianTarget,
    right_start: ArmCartesianTarget,
    left_goal: ArmCartesianTarget,
    right_goal: ArmCartesianTarget,
    step_m: float,
    vel: float,
    acc: float,
    multi_waypoint: bool,
) -> None:
    if arm == "left":
        left_waypoints = build_waypoints(left_start, left_goal, step_m) if multi_waypoint else [left_goal]
        right_waypoints: list[ArmCartesianTarget] = []
    elif arm == "right":
        left_waypoints = []
        right_waypoints = build_waypoints(right_start, right_goal, step_m) if multi_waypoint else [right_goal]
    else:
        left_waypoints = build_waypoints(left_start, left_goal, step_m) if multi_waypoint else [left_goal]
        right_waypoints = build_waypoints(right_start, right_goal, step_m) if multi_waypoint else [right_goal]

    print("\n路径目标")
    print(f"left_waypoints={len(left_waypoints)} right_waypoints={len(right_waypoints)} step_m={step_m:.3f}")
    if left_waypoints:
        print(f"左臂末点  {fmt_target(left_waypoints[-1])}")
    if right_waypoints:
        print(f"右臂末点  {fmt_target(right_waypoints[-1])}")

    state.power_on()
    try:
        service_name, response = state.call_cartesian_path_absolute(
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            vel=vel,
            acc=acc,
        )
    except Exception as exc:
        print(f"路径服务调用失败: {exc}")
        return

    print(f"service: {service_name}")
    print(f"success: {getattr(response, 'success', None)}")
    print(f"message: {getattr(response, 'message', '')}")


def prompt_cartesian_move(state: InteractiveState, client: DualArmControlClient) -> None:
    left_ee_current, right_ee_current, left_tcp_current, right_tcp_current = print_status(state)
    source_choice = prompt_menu("输入类型", [("1", "TCP"), ("2", "EE")])
    if source_choice is None:
        return
    source = "tcp" if source_choice == "1" else "ee"
    arm_choice = prompt_menu("选择手臂", [("1", "左臂"), ("2", "右臂"), ("3", "双臂")])
    if arm_choice is None:
        return
    arm = {"1": "left", "2": "right", "3": "both"}[arm_choice]

    try:
        if arm == "left":
            left_text = input("输入左臂目标 x y z [roll pitch yaw，角度默认deg]: ").strip()
            left_text = left_text
            right_text = None
        elif arm == "right":
            right_text = input("输入右臂目标 x y z [roll pitch yaw，角度默认deg]: ").strip()
            left_text = None
            right_text = right_text
        else:
            left_text = input("输入左臂目标 x y z [roll pitch yaw，角度默认deg]: ").strip()
            right_text = input("输入右臂目标 x y z [roll pitch yaw，角度默认deg]: ").strip()
    except ValueError as exc:
        print(f"输入错误: {exc}")
        return

    # 先用当前快照预览，真正发送前会再刷新一次当前状态，
    # 避免“未操作那只手”因为漂移而被服务端当成超差目标。
    left_source_target, right_source_target, left_target, right_target = build_targets_from_source(
        source,
        arm,
        left_text,
        right_text,
        left_ee_current,
        right_ee_current,
        left_tcp_current,
        right_tcp_current,
    )

    print("\n目标姿态")
    print(f"输入源  {source}")
    print(f"左臂输入  {fmt_target(left_source_target)}")
    print(f"右臂输入  {fmt_target(right_source_target)}")
    if source == "ee":
        print(f"左臂TCP  {fmt_target(left_target)}")
        print(f"右臂TCP  {fmt_target(right_target)}")
    mode_choice = prompt_menu("执行模式", [("1", "单次发送"), ("2", "自动分段"), ("3", "路径服务")])
    if mode_choice is None:
        return
    mode = {"1": "single", "2": "segment", "3": "path"}[mode_choice]
    step_m = 0.015
    vel = 0.10
    acc = 0.20
    path_multi_waypoint = False
    if mode in {"segment", "path"}:
        step_text = input("分段步长米数 [0.015]: ").strip()
        vel_text = input("vel [0.10]: ").strip()
        acc_text = input("acc [0.20]: ").strip()
        try:
            if step_text:
                step_m = float(step_text)
            if vel_text:
                vel = float(vel_text)
            if acc_text:
                acc = float(acc_text)
        except ValueError:
            print("步长/vel/acc 输入无效")
            return
    if mode == "path":
        path_choice = prompt_menu("路径类型", [("1", "单点waypoint"), ("2", "自动多waypoint")])
        if path_choice is None:
            return
        path_multi_waypoint = path_choice == "2"
    confirm_choice = prompt_menu("确认发送", [("1", "发送"), ("2", "取消")])
    if confirm_choice != "1":
        print("已取消")
        return

    try:
        fresh_left_ee, fresh_right_ee, fresh_left_tcp, fresh_right_tcp = print_status(state)
    except Exception as exc:
        print(f"发送前刷新状态失败: {exc}")
        return

    left_source_target, right_source_target, left_target, right_target = build_targets_from_source(
        source,
        arm,
        left_text,
        right_text,
        fresh_left_ee,
        fresh_right_ee,
        fresh_left_tcp,
        fresh_right_tcp,
    )

    print("\n发送前最终目标")
    print(f"左臂TCP  {fmt_target(left_target)}")
    print(f"右臂TCP  {fmt_target(right_target)}")

    if mode == "segment":
        execute_segmented_move(state, client, arm, left_target, right_target, step_m, vel, acc)
        return
    if mode == "path":
        execute_path_move(
            state,
            arm,
            fresh_left_tcp,
            fresh_right_tcp,
            left_target,
            right_target,
            step_m,
            vel,
            acc,
            path_multi_waypoint,
        )
        return

    state.power_on()
    try:
        service_name, response = client.call_cartesian_absolute(
            left=left_target,
            right=right_target,
            vel=vel,
            acc=acc,
            service_wait=2.0,
            motion_timeout=30.0,
        )
    except Exception as exc:
        print(f"调用失败: {exc}")
        return

    print(f"service: {service_name}")
    print(f"success: {getattr(response, 'success', None)}")
    print(f"message: {getattr(response, 'message', '')}")


def main() -> int:
    rclpy.init()
    state = InteractiveState()
    client = DualArmControlClient(node_name="cartesian_interactive_client")
    try:
        print("笛卡尔绝对位置交互台")
        print("1: 获取状态")
        print("2: 控制左右手TCP姿态")
        print("q: 退出")
        while True:
            cmd = input("\n请输入指令: ").strip().lower()
            if cmd == "q":
                return 0
            if cmd == "1":
                try:
                    print_status(state)
                except Exception as exc:
                    print(f"读取状态失败: {exc}")
                continue
            if cmd == "2":
                prompt_cartesian_move(state, client)
                continue
            print("无效指令，可用: 1 / 2 / q")
    except KeyboardInterrupt:
        return 130
    finally:
        state.destroy_node()
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
