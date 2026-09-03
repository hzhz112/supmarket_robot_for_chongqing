#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONTROLLER_ROOT = Path("/home/test/robot_controller")
for candidate in (
    ROBOT_CONTROLLER_ROOT / "install" / "local" / "lib" / "python3.10" / "dist-packages",
    ROBOT_CONTROLLER_ROOT / "install" / "lib" / "python3.10" / "site-packages",
    ROBOT_CONTROLLER_ROOT / "src",
    ROOT / "Code" / "movement_primitives",
    ROOT / "ros2_robot_controller_runtime" / "src",
    ROOT / "ros2_control_source_partial",
    ROOT / "ros2_robot_controller_runtime" / "install" / "local" / "lib" / "python3.10" / "dist-packages",
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import pytransform3d.rotations  # type: ignore  # noqa: F401
except Exception:
    # DMP uses only xyzrpy here. Keep the optional Cartesian quaternion
    # helpers from blocking the generic six-dimensional DMP import.
    pytransform3d_module = types.ModuleType("pytransform3d")
    rotations_module = types.ModuleType("pytransform3d.rotations")
    batch_rotations_module = types.ModuleType("pytransform3d.batch_rotations")
    transformations_module = types.ModuleType("pytransform3d.transformations")
    pytransform3d_module.rotations = rotations_module  # type: ignore[attr-defined]
    pytransform3d_module.batch_rotations = batch_rotations_module  # type: ignore[attr-defined]
    pytransform3d_module.transformations = transformations_module  # type: ignore[attr-defined]
    sys.modules["pytransform3d"] = pytransform3d_module
    sys.modules["pytransform3d.rotations"] = rotations_module
    sys.modules["pytransform3d.batch_rotations"] = batch_rotations_module
    sys.modules["pytransform3d.transformations"] = transformations_module

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

from movement_primitives.dmp import DMP
from robot_control_msg.msg import EndEffectorPose
from robot_control_msg.srv import CartesianPathAbsoluteControl, SetRobotPower


FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=float)


def read_demo(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("笛卡尔示教至少需要 2 个采样点")
    t = np.array([float(row["t"]) for row in rows], dtype=float)
    xyzrpy = np.array([[float(row[field]) for field in FIELDS] for row in rows], dtype=float)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("示教时间必须严格递增")
    return t - t[0], xyzrpy


def train_model(demo_path: Path, arm: str, output_path: Path, n_weights: int) -> None:
    t, xyzrpy = read_demo(demo_path)
    execution_time = float(t[-1])
    dt = float(np.median(np.diff(t)))
    dmp = DMP(
        n_dims=6,
        execution_time=execution_time,
        dt=max(dt, 0.005),
        n_weights_per_dim=n_weights,
    )
    dmp.imitate(t, xyzrpy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        arm=arm,
        dt=dmp.dt_,
        execution_time=execution_time,
        n_weights=n_weights,
        start_y=xyzrpy[0],
        goal_y=xyzrpy[-1],
        weights=dmp.get_weights(),
        demo_path=str(demo_path),
    )
    print(f"saved Cartesian DMP model: {output_path}")


def parse_xyzrpy(text: str) -> np.ndarray:
    parts = text.replace("，", " ").replace(",", " ").split()
    if len(parts) != 6:
        raise ValueError("目标需要 6 个值：x y z roll pitch yaw，单位 m/rad")
    return np.array([float(part) for part in parts], dtype=float)


class CartesianPathClient(Node):
    def __init__(self) -> None:
        super().__init__("cartesian_dmp_player")
        self.left_tcp: Pose | None = None
        self.right_tcp: Pose | None = None
        self.create_subscription(EndEffectorPose, "/arm_tcp_pose", self._on_tcp, 10)
        self.power_client = self.create_client(SetRobotPower, "/set_robot_power")
        self.path_client = self.create_client(
            CartesianPathAbsoluteControl,
            "/cartesian_path_absolute_control",
        )

    def _on_tcp(self, msg: EndEffectorPose) -> None:
        self.left_tcp = msg.left_ee_pose
        self.right_tcp = msg.right_ee_pose

    def wait_for_tcp(self, arm: str, timeout_sec: float = 3.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            pose = self.left_tcp if arm == "left" else self.right_tcp
            if pose is not None:
                quaternion = np.array(
                    [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
                    dtype=float,
                )
                return np.array(
                    [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        *quaternion_to_rpy(quaternion),
                    ],
                    dtype=float,
                )
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError(f"等待{arm}臂 TCP 状态超时")

    def power_on(self) -> None:
        if not self.power_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("等待 /set_robot_power 服务超时")
        request = SetRobotPower.Request()
        request.enable = True
        future = self.power_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise TimeoutError("/set_robot_power 未返回")

    def call_path(self, arm: str, xyzrpy: np.ndarray, vel: float, acc: float) -> object:
        if not self.path_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("等待 /cartesian_path_absolute_control 服务超时")
        poses: list[Pose] = []
        for row in xyzrpy:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(value) for value in row[:3]]
            quaternion = rpy_to_quaternion(*row[3:])
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = [
                float(value) for value in quaternion
            ]
            poses.append(pose)
        request = CartesianPathAbsoluteControl.Request()
        if arm == "left":
            request.left_waypoints = poses
            request.right_waypoints = []
            request.left_blend_radii = []
            request.right_blend_radii = []
        else:
            request.left_waypoints = []
            request.right_waypoints = poses
            request.left_blend_radii = []
            request.right_blend_radii = []
        request.vel = float(vel)
        request.acc = float(acc)
        future = self.path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=max(30.0, len(poses) * 0.2))
        if not future.done() or future.result() is None:
            raise TimeoutError("/cartesian_path_absolute_control 未返回")
        return future.result()


def play_model(
    model_path: Path,
    goal_text: str | None,
    vel: float,
    acc: float,
    dump_csv: Path | None,
) -> None:
    model = np.load(model_path, allow_pickle=True)
    arm = str(model["arm"])
    n_weights = int(model["n_weights"])
    dmp = DMP(
        n_dims=6,
        execution_time=float(model["execution_time"]),
        dt=float(model["dt"]),
        n_weights_per_dim=n_weights,
    )
    dmp.set_weights(model["weights"])
    recorded_goal = model["goal_y"].copy()
    rclpy.init()
    node = CartesianPathClient()
    try:
        current = node.wait_for_tcp(arm)
        goal = parse_xyzrpy(goal_text) if goal_text else recorded_goal
        start = current
        dmp.configure(start_y=start, goal_y=goal)
        _, generated_xyzrpy = dmp.open_loop()
        if dump_csv is not None:
            dump_csv.parent.mkdir(parents=True, exist_ok=True)
            with dump_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t"] + list(FIELDS))
                for index, row in enumerate(generated_xyzrpy):
                    writer.writerow([index * dmp.dt_] + [float(value) for value in row])
        node.power_on()
        response = node.call_path(arm, generated_xyzrpy, vel, acc)
        print(f"service: /cartesian_path_absolute_control")
        print(f"success: {getattr(response, 'success', None)}")
        print(f"message: {getattr(response, 'message', '')}")
        print(f"played {len(generated_xyzrpy)} TCP waypoints, arm={arm}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and play a TCP Cartesian DMP.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--demo", required=True, type=Path)
    train.add_argument("--arm", required=True, choices=("left", "right"))
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--weights", type=int, default=30)
    play = subparsers.add_parser("play")
    play.add_argument("--model", required=True, type=Path)
    play.add_argument("--goal", default=None, help="optional x y z roll pitch yaw in m/rad")
    play.add_argument("--vel", type=float, default=0.10)
    play.add_argument("--acc", type=float, default=0.20)
    play.add_argument("--dump-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train_model(args.demo, args.arm, args.output, args.weights)
    else:
        play_model(args.model, args.goal, args.vel, args.acc, args.dump_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
