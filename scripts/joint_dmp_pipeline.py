#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    # Joint-space DMP does not use quaternion helpers, but movement_primitives
    # imports Cartesian modules from __init__. Provide a tiny stub so that we
    # can use the joint-space DMP without pulling extra dependencies first.
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
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8

from movement_primitives.dmp import DMP
from robot_control_msg.msg import Robotarmservomsg
try:
    from robot_control_msg.srv import SetSystemControlMode
except ImportError:
    SetSystemControlMode = None  # type: ignore[assignment]


LEFT_ARM_JOINTS = ("ljoint1", "ljoint2", "ljoint3", "ljoint4", "ljoint5", "ljoint6", "ljoint7")
RIGHT_ARM_JOINTS = ("rjoint1", "rjoint2", "rjoint3", "rjoint4", "rjoint5", "rjoint6", "rjoint7")
ROBOT_CONTROL_OWNER = 1
ROBOT_CONTROL_MASK_JOINT = 4


def read_demo_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty demo file: {path}")
    t = np.array([float(row["t"]) for row in rows], dtype=float)
    data: dict[str, np.ndarray] = {}
    for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        data[name] = np.array([float(row[name]) for row in rows], dtype=float)
    return t, data


def build_demo_matrix(data: dict[str, np.ndarray], arm: str) -> np.ndarray:
    names = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
    return np.column_stack([data[name] for name in names])


def train_model(demo_path: Path, arm: str, output_path: Path, n_weights: int, dt: float | None) -> None:
    t, data = read_demo_csv(demo_path)
    y_demo = build_demo_matrix(data, arm)
    execution_time = float(t[-1] - t[0])
    if execution_time <= 0.0:
        raise ValueError("demo duration must be > 0")
    train_dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    model_dt = float(dt) if dt is not None else train_dt
    dmp = DMP(n_dims=7, execution_time=execution_time, dt=model_dt, n_weights_per_dim=n_weights)
    dmp.imitate(t - t[0], y_demo)
    dmp.configure(start_y=y_demo[0], goal_y=y_demo[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        arm=arm,
        dt=model_dt,
        execution_time=execution_time,
        train_dt=train_dt,
        n_weights=n_weights,
        start_y=y_demo[0],
        goal_y=y_demo[-1],
        weights=dmp.get_weights(),
        demo_path=str(demo_path),
    )
    print(f"saved model: {output_path}")


def load_model(path: Path) -> tuple[str, DMP]:
    model = np.load(path, allow_pickle=True)
    arm = str(model["arm"])
    dmp = DMP(
        n_dims=7,
        execution_time=float(model["execution_time"]),
        dt=float(model["dt"]),
        n_weights_per_dim=int(model["n_weights"]),
    )
    dmp.configure(start_y=model["start_y"], goal_y=model["goal_y"])
    dmp.set_weights(model["weights"])
    return arm, dmp


def parse_goal_deg(text: str) -> np.ndarray:
    normalized = text.replace("，", " ").replace(",", " ")
    parts = [part.strip() for part in normalized.split() if part.strip()]
    if len(parts) != 7:
        raise ValueError("goal must contain 7 joint angles in deg")
    return np.radians(np.array([float(part) for part in parts], dtype=float))


def dump_trajectory_csv(path: Path, arm: str, t: np.ndarray, y: np.ndarray) -> None:
    names = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + list(names))
        for ti, row in zip(t, y):
            writer.writerow([float(ti)] + [float(value) for value in row])
    print(f"saved trajectory: {path}")


class JointStateHold(Node):
    def __init__(self) -> None:
        super().__init__("joint_dmp_player")
        self._joints: dict[str, float] = {}
        self.create_subscription(JointState, "/whole/joint_states", self._on_joint_state, 10)
        self._arm_mode_pub = self.create_publisher(UInt8, "/whole/control_mode_cmd", 10)
        self._arm_axis_pub = self.create_publisher(Robotarmservomsg, "/arm_axis_position_cmd", 10)
        self._control_mode_client = self.create_client(SetSystemControlMode, "/robot_system_manager/set_mode") if SetSystemControlMode is not None else None

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
                self._joints[name] = float(position)

    def wait_for_current(self, timeout_sec: float = 3.0) -> dict[str, float]:
        deadline = time.monotonic() + timeout_sec
        required = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        while time.monotonic() < deadline and rclpy.ok():
            if all(name in self._joints for name in required):
                return dict(self._joints)
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = [name for name in required if name not in self._joints]
        raise TimeoutError(f"missing joint states: {missing}")

    def set_position_mode(self) -> None:
        msg = UInt8()
        msg.data = 0
        self._arm_mode_pub.publish(msg)

    def set_joint_control_mode(self) -> None:
        if SetSystemControlMode is None or self._control_mode_client is None:
            msg = UInt8()
            msg.data = 0
            self._arm_mode_pub.publish(msg)
            return
        req = SetSystemControlMode.Request()
        req.mode = ROBOT_CONTROL_OWNER
        req.robot_control_mask = ROBOT_CONTROL_MASK_JOINT
        req.enable_leg_control = False
        req.enable_waist_control = False
        if not self._control_mode_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("service unavailable: /robot_system_manager/set_mode")
        future = self._control_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("/robot_system_manager/set_mode returned no result")
        result = future.result()
        if not bool(getattr(result, "success", False)):
            raise RuntimeError(f"failed to set control mode mask={ROBOT_CONTROL_MASK_JOINT}: {getattr(result, 'message', '')}")

    def publish_axis_target(
        self,
        left_values: np.ndarray,
        right_values: np.ndarray,
        run_mode: int,
        run_time: float,
    ) -> None:
        msg = Robotarmservomsg()
        for index, value in enumerate(left_values, start=1):
            setattr(msg, f"ljoint{index}_position", float(value))
        for index, value in enumerate(right_values, start=1):
            setattr(msg, f"rjoint{index}_position", float(value))
        msg.run_mode = int(run_mode)
        msg.robot_power = True
        msg.run_time = float(run_time)
        msg.stamp = self.get_clock().now().to_msg()
        self._arm_axis_pub.publish(msg)


def play_model(
    model_path: Path,
    rate_hz: float,
    run_mode: int,
    hold_scale: float,
    goal_deg: str | None = None,
    dump_csv: Path | None = None,
) -> None:
    arm, dmp = load_model(model_path)
    sample_dt = dmp.dt_
    rclpy.init()
    node = JointStateHold()
    try:
        current = node.wait_for_current()
        node.set_joint_control_mode()
        left_hold = np.array([current[name] for name in LEFT_ARM_JOINTS], dtype=float)
        right_hold = np.array([current[name] for name in RIGHT_ARM_JOINTS], dtype=float)
        active_start = left_hold if arm == "left" else right_hold
        active_goal = parse_goal_deg(goal_deg) if goal_deg else dmp.goal_y
        dmp.configure(start_y=active_start, goal_y=active_goal)
        t, y = dmp.open_loop(run_t=dmp.execution_time_)
        if dump_csv is not None:
            dump_trajectory_csv(dump_csv, arm, t, y)
        sleep_dt = 1.0 / rate_hz if rate_hz > 0.0 else sample_dt
        run_time = max(sample_dt * hold_scale, sleep_dt, 0.05)
        for row in y:
            if arm == "left":
                left_values = row
                right_values = right_hold
            else:
                left_values = left_hold
                right_values = row
            node.publish_axis_target(left_values, right_values, run_mode=run_mode, run_time=run_time)
            rclpy.spin_once(node, timeout_sec=0.001)
            time.sleep(sleep_dt)
        goal_deg_text = " ".join(f"{value:+.1f}" for value in np.degrees(active_goal))
        print(f"played {len(y)} waypoints from {model_path}")
        print(f"active arm: {arm}, goal(deg): {goal_deg_text}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and play a joint-space DMP from Qt-recorded demos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a 7-DoF joint DMP from a recorded CSV")
    train.add_argument("--demo", required=True, type=Path, help="recorded CSV from dmp_joint_data")
    train.add_argument("--arm", choices=("left", "right"), required=True)
    train.add_argument("--output", required=True, type=Path, help="output .npz model path")
    train.add_argument("--weights", type=int, default=30)
    train.add_argument("--dt", type=float, default=None)

    play = subparsers.add_parser("play", help="play a trained joint DMP through /arm_axis_position_cmd")
    play.add_argument("--model", required=True, type=Path)
    play.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz")
    play.add_argument("--run-mode", type=int, default=1, choices=(0, 1))
    play.add_argument("--hold-scale", type=float, default=1.5, help="run_time = dt * hold_scale")
    play.add_argument("--goal-deg", type=str, default=None, help="7 joint target angles in deg, split by space or comma")
    play.add_argument("--dump-csv", type=Path, default=None, help="optional path to save generated generalized trajectory")
    return parser




def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_model(args.demo, args.arm, args.output, args.weights, args.dt)
        return 0
    if args.command == "play":
        play_model(args.model, args.rate, args.run_mode, args.hold_scale, args.goal_deg, args.dump_csv)
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
