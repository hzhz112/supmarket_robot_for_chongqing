#!/usr/bin/env python3
"""Absolute Cartesian arm control helper.

Examples:
  python3 scripts/cartesian_absolute_control.py --arm right --target "0.26 -0.18 0.87"
  python3 scripts/cartesian_absolute_control.py --arm left --target "0.06 0.22 0.94 0 0 0"
  python3 scripts/cartesian_absolute_control.py --left "0.06 0.22 0.94 0 0 0" --right "0.26 -0.18 0.87 0 0 0"
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONTROLLER_ROOT = Path("/home/test/robot_controller")
for candidate in (
    ROBOT_CONTROLLER_ROOT / "install" / "local" / "lib" / "python3.10" / "dist-packages",
    ROBOT_CONTROLLER_ROOT / "install" / "lib" / "python3.10" / "site-packages",
    ROBOT_CONTROLLER_ROOT / "src",
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
from robot_control_msg.msg import EndEffectorPose
from sailor_r1_pro_description.arm_topics import ArmCartesianTarget, DualArmControlClient


def parse_angle(value: str) -> float:
    text = value.strip().lower()
    if text.endswith("deg"):
        return math.radians(float(text[:-3]))
    if text.endswith("rad"):
        return float(text[:-3])
    if text.endswith("\u00b0"):
        return math.radians(float(text[:-1]))
    # Bare RPY values match the interactive status output and are degrees.
    # Append "rad" when radians are intended.
    return math.radians(float(text))


def split_values(value: str) -> list[str]:
    return [part for part in value.replace("，", " ").replace(",", " ").split() if part]


def parse_pose_text(value: str) -> tuple[float, ...]:
    parts = split_values(value)
    if len(parts) not in (3, 6):
        raise argparse.ArgumentTypeError("target needs 3 values(x y z) or 6 values(x y z roll pitch yaw)")
    xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
    if len(parts) == 3:
        return tuple(xyz)
    return (
        xyz[0],
        xyz[1],
        xyz[2],
        parse_angle(parts[3]),
        parse_angle(parts[4]),
        parse_angle(parts[5]),
    )


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


def pose_msg_to_target(pose: Pose) -> ArmCartesianTarget:
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


def rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    v_quaternion = (vx, vy, vz, 0.0)
    rotated = quaternion_multiply(
        quaternion_multiply((qx, qy, qz, qw), v_quaternion),
        quaternion_conjugate((qx, qy, qz, qw)),
    )
    return rotated[0], rotated[1], rotated[2]


def normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / norm for component in quaternion)


def target_quaternion(target: ArmCartesianTarget) -> tuple[float, float, float, float]:
    if any(abs(value) > 1e-9 for value in (target.qx, target.qy, target.qz, target.qw)):
        return normalize_quaternion((target.qx, target.qy, target.qz, target.qw))
    return normalize_quaternion(rpy_to_quaternion(target.roll, target.pitch, target.yaw))


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


class PoseReader(Node):
    def __init__(self) -> None:
        super().__init__("cartesian_absolute_control_reader")
        self.left_tcp_pose: Pose | None = None
        self.right_tcp_pose: Pose | None = None
        self.left_ee_pose: Pose | None = None
        self.right_ee_pose: Pose | None = None
        self.create_subscription(EndEffectorPose, "/arm_tcp_pose", self._on_tcp_pose, 10)
        self.create_subscription(EndEffectorPose, "/end_effector_pose", self._on_ee_pose, 10)

    def _on_tcp_pose(self, msg: EndEffectorPose) -> None:
        self.left_tcp_pose = msg.left_ee_pose
        self.right_tcp_pose = msg.right_ee_pose

    def _on_ee_pose(self, msg: EndEffectorPose) -> None:
        self.left_ee_pose = msg.left_ee_pose
        self.right_ee_pose = msg.right_ee_pose

    def wait_for_pose_set(self, timeout_sec: float) -> tuple[Pose, Pose, Pose, Pose]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if all(
                pose is not None
                for pose in (
                    self.left_tcp_pose,
                    self.right_tcp_pose,
                    self.left_ee_pose,
                    self.right_ee_pose,
                )
            ):
                return (
                    self.left_tcp_pose,
                    self.right_tcp_pose,
                    self.left_ee_pose,
                    self.right_ee_pose,
                )
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError("waiting for TCP and end-effector poses timed out")


def format_target(label: str, target: ArmCartesianTarget) -> str:
    return (
        f"{label}: xyz=({target.x:+.3f},{target.y:+.3f},{target.z:+.3f}) "
        f"rpy=({math.degrees(target.roll):+.1f},{math.degrees(target.pitch):+.1f},{math.degrees(target.yaw):+.1f})deg"
    )


def merge_target(raw: tuple[float, ...], current: ArmCartesianTarget) -> ArmCartesianTarget:
    if len(raw) == 3:
        return replace(current, x=raw[0], y=raw[1], z=raw[2])
    return ArmCartesianTarget(
        x=raw[0],
        y=raw[1],
        z=raw[2],
        roll=raw[3],
        pitch=raw[4],
        yaw=raw[5],
    )


def build_tcp_from_ee_target(current_ee: ArmCartesianTarget, current_tcp: ArmCartesianTarget, target_ee: ArmCartesianTarget) -> ArmCartesianTarget:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call /cartesian_moveP with explicit TCP or raw end-effector targets."
    )
    parser.add_argument("--arm", choices=("left", "right"), help="single-arm mode; the other arm keeps current pose")
    parser.add_argument("--target", type=parse_pose_text, help="single-arm target: x y z [roll pitch yaw]")
    parser.add_argument("--left", type=parse_pose_text, help="left target: x y z [roll pitch yaw]")
    parser.add_argument("--right", type=parse_pose_text, help="right target: x y z [roll pitch yaw]")
    parser.add_argument(
        "--source",
        choices=("tcp", "ee"),
        default="tcp",
        help="interpret input targets as TCP poses or raw end-effector poses; service always receives TCP",
    )
    parser.add_argument("--vel", type=float, default=0.10)
    parser.add_argument("--acc", type=float, default=0.20)
    parser.add_argument("--state-timeout", type=float, default=3.0)
    parser.add_argument("--service-wait", type=float, default=2.0)
    parser.add_argument("--motion-timeout", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true", help="actually call the service")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.arm:
        if args.target is None:
            parser.error("--arm requires --target")
        if args.left is not None or args.right is not None:
            parser.error("--arm/--target cannot be mixed with --left/--right")
    else:
        if args.left is None and args.right is None:
            parser.error("provide --arm/--target or at least one of --left/--right")

    rclpy.init()
    reader = PoseReader()
    client = DualArmControlClient(node_name="cartesian_absolute_control_client")
    try:
        left_tcp_pose, right_tcp_pose, left_ee_pose, right_ee_pose = reader.wait_for_pose_set(args.state_timeout)
        left_tcp_current = pose_msg_to_target(left_tcp_pose)
        right_tcp_current = pose_msg_to_target(right_tcp_pose)
        left_ee_current = pose_msg_to_target(left_ee_pose)
        right_ee_current = pose_msg_to_target(right_ee_pose)

        left_source_current = left_tcp_current if args.source == "tcp" else left_ee_current
        right_source_current = right_tcp_current if args.source == "tcp" else right_ee_current

        left_source_target = left_source_current
        right_source_target = right_source_current

        if args.arm == "left":
            left_source_target = merge_target(args.target, left_source_current)
        elif args.arm == "right":
            right_source_target = merge_target(args.target, right_source_current)
        else:
            if args.left is not None:
                left_source_target = merge_target(args.left, left_source_current)
            if args.right is not None:
                right_source_target = merge_target(args.right, right_source_current)

        if args.source == "tcp":
            left_target = left_source_target
            right_target = right_source_target
        else:
            left_target = build_tcp_from_ee_target(left_ee_current, left_tcp_current, left_source_target)
            right_target = build_tcp_from_ee_target(right_ee_current, right_tcp_current, right_source_target)

        print(f"input source={args.source} service_target=tcp")
        print(format_target("left input ", left_source_target))
        print(format_target("right input", right_source_target))
        if args.source == "ee":
            print(format_target("left tcp  ", left_target))
            print(format_target("right tcp ", right_target))
        print(f"vel={args.vel:.3f} acc={args.acc:.3f}")

        if not args.execute:
            print("dry-run only; add --execute to call /cartesian_moveP")
            return 0

        client.require_arm_power(args.state_timeout)
        service_name, response = client.call_cartesian_absolute(
            left=left_target,
            right=right_target,
            use_left=args.arm == "left" or args.left is not None,
            use_right=args.arm == "right" or args.right is not None,
            vel=args.vel,
            acc=args.acc,
            service_wait=args.service_wait,
            motion_timeout=args.motion_timeout,
        )
        print(f"service: {service_name}")
        print(f"success: {getattr(response, 'success', None)}")
        print(f"message: {getattr(response, 'message', '')}")
        return 0 if bool(getattr(response, "success", False)) else 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        reader.destroy_node()
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
