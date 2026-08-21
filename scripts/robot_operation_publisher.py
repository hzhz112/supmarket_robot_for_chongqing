#!/usr/bin/env python3
"""Topic-based arm and leg operation helper for Sailor R1 Pro.

Run this file after sourcing ROS 2 and the robot controller workspace:

    source /opt/ros/humble/setup.bash
    source /home/test/robot_controller/install/setup.bash
    export ROS_DOMAIN_ID=11

This script is intentionally conservative: running it with no motion arguments
prints help and does not publish motion commands.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, Sequence

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, UInt8
from robot_control_msg.msg import (
    ArmMotionStatus,
    LegMotionStatus,
    Robotarmjoint,
    Robotarmservomsg,
    Robotlegjoint,
    Robotservomsg,
)


DEFAULT_REPEAT = 5
DEFAULT_SETTLE_SEC = 0.05

EXAMPLE_LEFT_ARM = [0.0, 0.20, 0.0, -1.00, 0.0, 0.0, 0.0]
EXAMPLE_RIGHT_ARM = [0.0, -0.20, 0.0, 1.00, 0.0, 0.0, 0.0]
EXAMPLE_LEG = [0.05, -0.10, 0.05, 0.0]


class RobotOperationPublisher(Node):
    """Publish robot operation commands and subscribe to motion status."""

    def __init__(self) -> None:
        super().__init__("robot_operation_publisher")

        self._robot_power_pub = self.create_publisher(Bool, "/robot_poweron", 10)
        self._arm_mode_pub = self.create_publisher(UInt8, "/whole/control_mode_cmd", 10)

        self._arm_joint_pub = self.create_publisher(
            Robotarmjoint,
            "/arm_joint_absolute_cmd",
            10,
        )
        self._arm_axis_pub = self.create_publisher(
            Robotarmservomsg,
            "/arm_axis_position_cmd",
            10,
        )
        self._leg_joint_pub = self.create_publisher(
            Robotlegjoint,
            "/leg_joint_position_cmd",
            10,
        )
        self._leg_axis_pub = self.create_publisher(
            Robotservomsg,
            "/axis_position_cmd",
            10,
        )

        self._arm_status: ArmMotionStatus | None = None
        self._leg_status: LegMotionStatus | None = None
        self.create_subscription(
            ArmMotionStatus,
            "/whole/arm_controller/motion_status",
            self._on_arm_status,
            10,
        )
        self.create_subscription(
            LegMotionStatus,
            "/whole/leg_controller/motion_status",
            self._on_leg_status,
            10,
        )

    def _on_arm_status(self, msg: ArmMotionStatus) -> None:
        self._arm_status = msg

    def _on_leg_status(self, msg: LegMotionStatus) -> None:
        self._leg_status = msg

    def set_robot_power(
        self,
        enabled: bool,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self._publish_reliably(self._robot_power_pub, msg, repeat, settle_sec)
        self.get_logger().info(f"published /robot_poweron: {msg.data}")

    def set_arm_position_mode(
        self,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        msg = UInt8()
        msg.data = 0
        self._publish_reliably(self._arm_mode_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /whole/control_mode_cmd: 0 POSITION")

    def set_arm_effort_mode(
        self,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        msg = UInt8()
        msg.data = 1
        self._publish_reliably(self._arm_mode_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /whole/control_mode_cmd: 1 EFFORT")

    def publish_arm_joint(
        self,
        left_joints: Sequence[float],
        right_joints: Sequence[float],
        vel: float = 0.30,
        acc: float = 0.50,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        """Publish absolute 7-DOF left arm + 7-DOF right arm joint target."""
        left = _require_count(left_joints, 7, "left_joints")
        right = _require_count(right_joints, 7, "right_joints")

        msg = Robotarmjoint()
        for index, value in enumerate(left, start=1):
            setattr(msg, f"ljoint{index}", float(value))
        for index, value in enumerate(right, start=1):
            setattr(msg, f"rjoint{index}", float(value))
        msg.vel = float(vel)
        msg.acc = float(acc)

        self._arm_status = None
        self._publish_reliably(self._arm_joint_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /arm_joint_absolute_cmd")

    def publish_arm_axis_position(
        self,
        left_joints: Sequence[float],
        right_joints: Sequence[float],
        run_mode: int = 1,
        run_time: float = 3.0,
        robot_power: bool = True,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        """Publish low-level 14-joint arm position target."""
        left = _require_count(left_joints, 7, "left_joints")
        right = _require_count(right_joints, 7, "right_joints")

        msg = Robotarmservomsg()
        for index, value in enumerate(left, start=1):
            setattr(msg, f"ljoint{index}_position", float(value))
        for index, value in enumerate(right, start=1):
            setattr(msg, f"rjoint{index}_position", float(value))
        msg.run_mode = int(run_mode)
        msg.robot_power = bool(robot_power)
        msg.run_time = float(run_time)
        msg.stamp = self.get_clock().now().to_msg()

        self._arm_status = None
        self._publish_reliably(self._arm_axis_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /arm_axis_position_cmd")

    def publish_leg_joint(
        self,
        ankle_joint: float,
        knee_joint: float,
        hip_pitch_joint: float,
        hip_yaw_joint: float,
        vel: float = 0.08,
        acc: float = 0.20,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        msg = Robotlegjoint()
        msg.ankle_joint = float(ankle_joint)
        msg.knee_joint = float(knee_joint)
        msg.hip_pitch_joint = float(hip_pitch_joint)
        msg.hip_yaw_joint = float(hip_yaw_joint)
        msg.vel = float(vel)
        msg.acc = float(acc)

        self._leg_status = None
        self._publish_reliably(self._leg_joint_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /leg_joint_position_cmd")

    def publish_leg_axis_joint(
        self,
        ankle_position: float,
        knee_position: float,
        hip_pitch_position: float,
        hip_yaw_position: float,
        vel: float = 0.08,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        """Publish leg joint target through /axis_position_cmd run_mode=0."""
        msg = Robotservomsg()
        msg.run_mode = 0
        msg.ankle_position = float(ankle_position)
        msg.knee_position = float(knee_position)
        msg.hip_pitch_position = float(hip_pitch_position)
        msg.hip_yaw_position = float(hip_yaw_position)
        msg.x = 0.0
        msg.y = 0.0
        msg.phi = 0.0
        msg.mode_leg_select = 0
        msg.vel = float(vel)

        self._leg_status = None
        self._publish_reliably(self._leg_axis_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /axis_position_cmd run_mode=0")

    def publish_leg_cartesian(
        self,
        x: float,
        y: float,
        phi: float,
        hip_yaw_position: float = 0.0,
        mode_leg_select: int = 0,
        vel: float = 0.08,
        repeat: int = DEFAULT_REPEAT,
        settle_sec: float = DEFAULT_SETTLE_SEC,
    ) -> None:
        """Publish leg Cartesian target through /axis_position_cmd run_mode=1."""
        msg = Robotservomsg()
        msg.run_mode = 1
        msg.ankle_position = 0.0
        msg.knee_position = 0.0
        msg.hip_pitch_position = 0.0
        msg.hip_yaw_position = float(hip_yaw_position)
        msg.x = float(x)
        msg.y = float(y)
        msg.phi = float(phi)
        msg.mode_leg_select = int(mode_leg_select)
        msg.vel = float(vel)

        self._leg_status = None
        self._publish_reliably(self._leg_axis_pub, msg, repeat, settle_sec)
        self.get_logger().info("published /axis_position_cmd run_mode=1")

    def wait_for_arm_goal(self, timeout_sec: float = 30.0) -> ArmMotionStatus:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._arm_status is None:
                continue
            if self._arm_status.goal_reached and not self._arm_status.is_moving:
                self.get_logger().info("arm goal reached")
                return self._arm_status
        raise TimeoutError(f"arm goal was not reached within {timeout_sec:.1f}s")

    def wait_for_leg_goal(self, timeout_sec: float = 30.0) -> LegMotionStatus:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._leg_status is None:
                continue
            if self._leg_status.goal_reached and not self._leg_status.is_moving:
                self.get_logger().info("leg goal reached")
                return self._leg_status
        raise TimeoutError(f"leg goal was not reached within {timeout_sec:.1f}s")

    def _publish_reliably(self, publisher, msg: object, repeat: int, settle_sec: float) -> None:
        for _ in range(max(1, int(repeat))):
            rclpy.spin_once(self, timeout_sec=0.02)
            publisher.publish(msg)
            time.sleep(max(0.0, float(settle_sec)))


def _require_count(values: Sequence[float] | Iterable[float], count: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != count:
        raise ValueError(f"{name} requires {count} values, got {len(result)}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Sailor R1 Pro arm and leg operation topics.",
    )
    parser.add_argument("--power-on", action="store_true", help="Publish /robot_poweron true first.")
    parser.add_argument("--power-off", action="store_true", help="Publish /robot_poweron false and exit.")
    parser.add_argument(
        "--skip-arm-position-mode",
        action="store_true",
        help="Do not publish /whole/control_mode_cmd=0 before arm commands.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Publish the built-in small example arm and leg joint targets.",
    )
    parser.add_argument(
        "--arm-joints",
        nargs=14,
        type=float,
        metavar="RAD",
        help="Publish /arm_joint_absolute_cmd with 14 values: left7 then right7.",
    )
    parser.add_argument("--arm-vel", type=float, default=0.30)
    parser.add_argument("--arm-acc", type=float, default=0.50)
    parser.add_argument(
        "--arm-axis",
        action="store_true",
        help="Use /arm_axis_position_cmd instead of /arm_joint_absolute_cmd for --arm-joints.",
    )
    parser.add_argument("--arm-run-mode", type=int, default=1)
    parser.add_argument("--arm-run-time", type=float, default=3.0)
    parser.add_argument(
        "--leg-joints",
        nargs=4,
        type=float,
        metavar="RAD",
        help="Publish /leg_joint_position_cmd: ankle knee hip_pitch hip_yaw.",
    )
    parser.add_argument("--leg-vel", type=float, default=0.08)
    parser.add_argument("--leg-acc", type=float, default=0.20)
    parser.add_argument(
        "--leg-axis-joints",
        nargs=4,
        type=float,
        metavar="RAD",
        help="Publish /axis_position_cmd run_mode=0: ankle knee hip_pitch hip_yaw.",
    )
    parser.add_argument(
        "--leg-cartesian",
        nargs=3,
        type=float,
        metavar=("X", "Y", "PHI"),
        help="Publish /axis_position_cmd run_mode=1 with x y phi.",
    )
    parser.add_argument("--leg-hip-yaw", type=float, default=0.0)
    parser.add_argument("--leg-mode-select", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--wait", action="store_true", help="Wait for motion_status goal_reached after commands.")
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    has_motion = any(
        [
            args.demo,
            args.arm_joints is not None,
            args.leg_joints is not None,
            args.leg_axis_joints is not None,
            args.leg_cartesian is not None,
            args.power_on,
            args.power_off,
        ]
    )
    if not has_motion:
        parser.print_help()
        return 0

    rclpy.init()
    node = RobotOperationPublisher()
    arm_commanded = False
    leg_commanded = False

    try:
        if args.power_off:
            node.set_robot_power(False, repeat=args.repeat, settle_sec=args.settle_sec)
            return 0

        if args.power_on:
            node.set_robot_power(True, repeat=args.repeat, settle_sec=args.settle_sec)

        if not args.skip_arm_position_mode and (args.demo or args.arm_joints is not None):
            node.set_arm_position_mode(repeat=args.repeat, settle_sec=args.settle_sec)

        if args.demo:
            node.publish_arm_joint(
                EXAMPLE_LEFT_ARM,
                EXAMPLE_RIGHT_ARM,
                vel=args.arm_vel,
                acc=args.arm_acc,
                repeat=args.repeat,
                settle_sec=args.settle_sec,
            )
            arm_commanded = True
            node.publish_leg_joint(
                *EXAMPLE_LEG,
                vel=args.leg_vel,
                acc=args.leg_acc,
                repeat=args.repeat,
                settle_sec=args.settle_sec,
            )
            leg_commanded = True

        if args.arm_joints is not None:
            left = args.arm_joints[:7]
            right = args.arm_joints[7:]
            if args.arm_axis:
                node.publish_arm_axis_position(
                    left,
                    right,
                    run_mode=args.arm_run_mode,
                    run_time=args.arm_run_time,
                    repeat=args.repeat,
                    settle_sec=args.settle_sec,
                )
            else:
                node.publish_arm_joint(
                    left,
                    right,
                    vel=args.arm_vel,
                    acc=args.arm_acc,
                    repeat=args.repeat,
                    settle_sec=args.settle_sec,
                )
            arm_commanded = True

        if args.leg_joints is not None:
            node.publish_leg_joint(
                *args.leg_joints,
                vel=args.leg_vel,
                acc=args.leg_acc,
                repeat=args.repeat,
                settle_sec=args.settle_sec,
            )
            leg_commanded = True

        if args.leg_axis_joints is not None:
            node.publish_leg_axis_joint(
                *args.leg_axis_joints,
                vel=args.leg_vel,
                repeat=args.repeat,
                settle_sec=args.settle_sec,
            )
            leg_commanded = True

        if args.leg_cartesian is not None:
            node.publish_leg_cartesian(
                x=args.leg_cartesian[0],
                y=args.leg_cartesian[1],
                phi=args.leg_cartesian[2],
                hip_yaw_position=args.leg_hip_yaw,
                mode_leg_select=args.leg_mode_select,
                vel=args.leg_vel,
                repeat=args.repeat,
                settle_sec=args.settle_sec,
            )
            leg_commanded = True

        if args.wait:
            if arm_commanded:
                node.wait_for_arm_goal(args.wait_timeout)
            if leg_commanded:
                node.wait_for_leg_goal(args.wait_timeout)

        return 0
    except Exception as exc:
        node.get_logger().error(str(exc))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
