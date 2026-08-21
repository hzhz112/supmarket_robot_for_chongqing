#!/usr/bin/env python3
"""Direct topic test helper for leg and waist commands."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, Iterable

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from sailor_r1_pro_description import (
    publish_axis_cartesian_command,
    publish_axis_joint_command,
    publish_leg_joint_command,
    publish_waist_from_current,
)

LEG_JOINTS = ("ankle_joint", "knee_joint", "hip_pitch_joint", "hip_yaw_joint")


def parse_angle(value: str) -> float:
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
    "--phi",
}


def normalize_angle_option_argv(argv: Iterable[str]) -> list[str]:
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


def fmt_angle(value: float) -> str:
    return f"{value:+.4f} rad ({math.degrees(value):+.2f} deg)"


def print_joint_table(title: str, joints: Dict[str, float]) -> None:
    print(title)
    for name in LEG_JOINTS:
        label = "waist/hip_yaw" if name == "hip_yaw_joint" else name
        print(f"  {label:15s} {fmt_angle(joints[name])}")


class JointStateReader(Node):
    def __init__(self) -> None:
        super().__init__("leg_waist_topic_test_reader")
        self._joint_positions: Dict[str, float] = {}
        self.create_subscription(JointState, "/whole/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in LEG_JOINTS:
                self._joint_positions[name] = float(position)

    def wait_for_leg_state(self, timeout_sec: float) -> Dict[str, float]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if all(name in self._joint_positions for name in LEG_JOINTS):
                return dict(self._joint_positions)
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = [name for name in LEG_JOINTS if name not in self._joint_positions]
        raise TimeoutError(f"waiting for /whole/joint_states timed out, missing: {missing}")


def add_joint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ankle", type=parse_angle, required=True, help="ankle_joint target, rad or deg")
    parser.add_argument("--knee", type=parse_angle, required=True, help="knee_joint target, rad or deg")
    parser.add_argument("--hip-pitch", type=parse_angle, required=True, help="hip_pitch_joint target, rad or deg")
    parser.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, required=True, help="hip_yaw_joint target, rad or deg")
    parser.add_argument("--vel", type=float, default=0.08, help="velocity")
    parser.add_argument("--acc", type=float, default=0.20, help="acceleration")
    parser.add_argument("--repeat", type=int, default=5, help="publish repeats")
    parser.add_argument("--settle-sec", type=float, default=0.05, help="pause between repeats")


def add_axis_joint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ankle", type=parse_angle, required=True, help="ankle_position, rad or deg")
    parser.add_argument("--knee", type=parse_angle, required=True, help="knee_position, rad or deg")
    parser.add_argument("--hip-pitch", type=parse_angle, required=True, help="hip_pitch_position, rad or deg")
    parser.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, required=True, help="hip_yaw_position, rad or deg")
    parser.add_argument("--vel", type=float, default=0.08, help="velocity")
    parser.add_argument("--repeat", type=int, default=5, help="publish repeats")
    parser.add_argument("--settle-sec", type=float, default=0.05, help="pause between repeats")


def add_axis_cartesian_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x", type=float, required=True, help="x target, m")
    parser.add_argument("--y", type=float, required=True, help="y target, m")
    parser.add_argument("--phi", type=parse_angle, required=True, help="phi target, rad or deg")
    parser.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, required=True, help="hip_yaw_position, rad or deg")
    parser.add_argument("--mode-leg-select", type=int, choices=(0, 1), default=0, help="IK branch")
    parser.add_argument("--vel", type=float, default=0.08, help="velocity")
    parser.add_argument("--repeat", type=int, default=5, help="publish repeats")
    parser.add_argument("--settle-sec", type=float, default=0.05, help="pause between repeats")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct publish test helper for leg and waist topics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="print /whole/joint_states for the four joints")
    status.add_argument("--state-timeout", type=float, default=3.0, help="wait timeout")
    status.set_defaults(func=run_status)

    joint = subparsers.add_parser("joint", help="publish /leg_joint_position_cmd")
    add_joint_args(joint)
    joint.set_defaults(func=run_joint)

    waist = subparsers.add_parser("waist", help="read current joints and only change hip_yaw_joint")
    waist.add_argument("--waist", "--hip-yaw", dest="waist", type=parse_angle, required=True, help="new hip_yaw_joint, rad or deg")
    waist.add_argument("--vel", type=float, default=0.08, help="velocity")
    waist.add_argument("--acc", type=float, default=0.20, help="acceleration")
    waist.add_argument("--state-timeout", type=float, default=3.0, help="wait timeout")
    waist.add_argument("--repeat", type=int, default=5, help="publish repeats")
    waist.add_argument("--settle-sec", type=float, default=0.05, help="pause between repeats")
    waist.set_defaults(func=run_waist)

    axis_joint = subparsers.add_parser("axis-joint", help="publish run_mode=0 on /axis_position_cmd")
    add_axis_joint_args(axis_joint)
    axis_joint.set_defaults(func=run_axis_joint)

    cartesian = subparsers.add_parser("cartesian", help="publish run_mode=1 on /axis_position_cmd")
    add_axis_cartesian_args(cartesian)
    cartesian.set_defaults(func=run_axis_cartesian)

    return parser


def run_status(node: JointStateReader, args: argparse.Namespace) -> int:
    joints = node.wait_for_leg_state(args.state_timeout)
    print_joint_table("current joints:", joints)
    return 0


def run_joint(node: JointStateReader, args: argparse.Namespace) -> int:
    print("publishing /leg_joint_position_cmd")
    print(f"  ankle       {fmt_angle(args.ankle)}")
    print(f"  knee        {fmt_angle(args.knee)}")
    print(f"  hip_pitch   {fmt_angle(args.hip_pitch)}")
    print(f"  hip_yaw     {fmt_angle(args.waist)}")
    print(f"  vel         {args.vel:.3f}")
    print(f"  acc         {args.acc:.3f}")
    publish_leg_joint_command(
        args.ankle,
        args.knee,
        args.hip_pitch,
        args.waist,
        vel=args.vel,
        acc=args.acc,
        repeat=args.repeat,
        settle_sec=args.settle_sec,
    )
    return 0


def run_waist(node: JointStateReader, args: argparse.Namespace) -> int:
    current = node.wait_for_leg_state(args.state_timeout)
    target = dict(current)
    target["hip_yaw_joint"] = args.waist
    print_joint_table("current joints:", current)
    print_joint_table("target joints:", target)
    publish_waist_from_current(
        current,
        args.waist,
        vel=args.vel,
        acc=args.acc,
        repeat=args.repeat,
        settle_sec=args.settle_sec,
    )
    return 0


def run_axis_joint(node: JointStateReader, args: argparse.Namespace) -> int:
    print("publishing /axis_position_cmd in joint mode")
    print(f"  ankle       {fmt_angle(args.ankle)}")
    print(f"  knee        {fmt_angle(args.knee)}")
    print(f"  hip_pitch   {fmt_angle(args.hip_pitch)}")
    print(f"  hip_yaw     {fmt_angle(args.waist)}")
    print(f"  vel         {args.vel:.3f}")
    publish_axis_joint_command(
        args.ankle,
        args.knee,
        args.hip_pitch,
        args.waist,
        vel=args.vel,
        repeat=args.repeat,
        settle_sec=args.settle_sec,
    )
    return 0


def run_axis_cartesian(node: JointStateReader, args: argparse.Namespace) -> int:
    print("publishing /axis_position_cmd in cartesian mode")
    print(f"  x                {args.x:+.4f} m")
    print(f"  y                {args.y:+.4f} m")
    print(f"  phi              {fmt_angle(args.phi)}")
    print(f"  hip_yaw          {fmt_angle(args.waist)}")
    print(f"  mode_leg_select  {args.mode_leg_select}")
    print(f"  vel              {args.vel:.3f}")
    publish_axis_cartesian_command(
        args.x,
        args.y,
        args.phi,
        args.waist,
        mode_leg_select=args.mode_leg_select,
        vel=args.vel,
        repeat=args.repeat,
        settle_sec=args.settle_sec,
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = normalize_angle_option_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    rclpy.init()
    node = JointStateReader()
    try:
        return int(args.func(node, args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
