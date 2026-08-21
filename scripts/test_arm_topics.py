#!/usr/bin/env python3
"""Dual-arm service/topic test helper."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from typing import Iterable, Sequence

import rclpy

from sailor_r1_pro_description.arm_topics import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    ArmCartesianTarget,
    ArmPowerSnapshot,
    DualArmControlClient,
)


def parse_angle(value: str) -> float:
    text = value.strip().lower()
    if text.endswith("deg"):
        return math.radians(float(text[:-3]))
    if text.endswith("rad"):
        return float(text[:-3])
    if text.endswith("\u00b0"):
        return math.radians(float(text[:-1]))
    return float(text)


def _split_values(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def parse_joint_csv(value: str) -> tuple[float, ...]:
    parts = _split_values(value)
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(f"expected 7 joint values, got {len(parts)}")
    try:
        return tuple(parse_angle(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_pose_csv(value: str) -> ArmCartesianTarget:
    parts = _split_values(value)
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected 6 values: x,y,z,roll,pitch,yaw")
    try:
        return ArmCartesianTarget(
            x=float(parts[0]),
            y=float(parts[1]),
            z=float(parts[2]),
            roll=parse_angle(parts[3]),
            pitch=parse_angle(parts[4]),
            yaw=parse_angle(parts[5]),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_quat_csv(value: str) -> tuple[float, float, float, float]:
    parts = _split_values(value)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected 4 quaternion values: qx,qy,qz,qw")
    try:
        quat = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    norm = math.sqrt(sum(part * part for part in quat))
    if norm <= 0.0:
        raise argparse.ArgumentTypeError("quaternion norm must be non-zero")
    return quat


SINGLE_ANGLE_OPTIONS = {
    f"--l{index}" for index in range(1, 8)
} | {
    f"--r{index}" for index in range(1, 8)
} | {
    f"--ljoint{index}" for index in range(1, 8)
} | {
    f"--rjoint{index}" for index in range(1, 8)
}

CSV_OPTIONS = {
    "--left",
    "--right",
    "--left-quat",
    "--right-quat",
}


def normalize_option_argv(argv: Iterable[str]) -> list[str]:
    argv_list = list(argv)
    normalized: list[str] = []
    index = 0
    while index < len(argv_list):
        token = argv_list[index]
        if token in SINGLE_ANGLE_OPTIONS and index + 1 < len(argv_list):
            value = argv_list[index + 1]
            try:
                parse_angle(value)
            except Exception:
                pass
            else:
                normalized.append(f"{token}={value}")
                index += 2
                continue
        if token in CSV_OPTIONS and index + 1 < len(argv_list):
            value = argv_list[index + 1]
            if _looks_like_csv_option_value(token, value):
                normalized.append(f"{token}={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _looks_like_csv_option_value(option: str, value: str) -> bool:
    parsers = (parse_joint_csv, parse_pose_csv) if option in {"--left", "--right"} else (parse_quat_csv,)
    for parser in parsers:
        try:
            parser(value)
        except Exception:
            continue
        return True
    return False


def fmt_angle(value: float) -> str:
    return f"{value:+.4f} rad ({math.degrees(value):+.2f} deg)"


def print_joint_table(title: str, left: Sequence[float], right: Sequence[float]) -> None:
    print(title)
    print("  left:")
    for name, value in zip(LEFT_ARM_JOINTS, left):
        print(f"    {name:7s} {fmt_angle(value)}")
    print("  right:")
    for name, value in zip(RIGHT_ARM_JOINTS, right):
        print(f"    {name:7s} {fmt_angle(value)}")


def print_pose_table(title: str, left: ArmCartesianTarget, right: ArmCartesianTarget) -> None:
    print(title)
    for label, target in (("left", left), ("right", right)):
        print(f"  {label}:")
        print(f"    xyz  {target.x:+.4f}, {target.y:+.4f}, {target.z:+.4f} m")
        print(
            "    rpy  "
            f"{fmt_angle(target.roll)}, {fmt_angle(target.pitch)}, {fmt_angle(target.yaw)}"
        )
        if any(abs(value) > 0.0 for value in (target.qx, target.qy, target.qz, target.qw)):
            print(
                "    quat "
                f"{target.qx:+.5f}, {target.qy:+.5f}, {target.qz:+.5f}, {target.qw:+.5f}"
            )
        else:
            print("    quat 0,0,0,0 (service uses rpy)")


def print_power_status(snapshot: ArmPowerSnapshot) -> None:
    status_text = ", ".join(f"{value:.0f}" for value in snapshot.motor_status) or "none"
    ready = snapshot.is_enabled and len(snapshot.motor_status) == 14 and all(
        int(round(value)) == 39 for value in snapshot.motor_status
    )
    print(f"arm power topic: {snapshot.topic}")
    print(f"  is_enabled  {snapshot.is_enabled}")
    print(f"  all_ready   {ready}")
    print(f"  status      [{status_text}]")


def side_from_current(current: dict[str, float], names: Sequence[str]) -> tuple[float, ...]:
    return tuple(current[name] for name in names)


def any_joint_target_arg(args: argparse.Namespace) -> bool:
    if args.left is not None or args.right is not None:
        return True
    return any(getattr(args, f"l{index}") is not None for index in range(1, 8)) or any(
        getattr(args, f"r{index}") is not None for index in range(1, 8)
    )


def build_joint_target(
    node: DualArmControlClient,
    args: argparse.Namespace,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, float] | None]:
    if not any_joint_target_arg(args):
        raise ValueError("provide --left/--right, or one of --l1..--l7/--r1..--r7")

    current: dict[str, float] | None = None
    if args.left is None or args.right is None:
        current = node.wait_for_arm_joint_state(args.state_timeout)

    if args.left is not None:
        left = list(args.left)
    elif current is not None:
        left = list(side_from_current(current, LEFT_ARM_JOINTS))
    else:
        raise ValueError("left arm target is missing")

    if args.right is not None:
        right = list(args.right)
    elif current is not None:
        right = list(side_from_current(current, RIGHT_ARM_JOINTS))
    else:
        raise ValueError("right arm target is missing")

    for index in range(1, 8):
        left_value = getattr(args, f"l{index}")
        right_value = getattr(args, f"r{index}")
        if left_value is not None:
            left[index - 1] = left_value
        if right_value is not None:
            right[index - 1] = right_value

    return tuple(left), tuple(right), current


def maybe_require_power(node: DualArmControlClient, args: argparse.Namespace) -> None:
    if not args.execute or args.skip_power_check:
        return
    snapshot = node.require_arm_power(args.power_timeout)
    print_power_status(snapshot)


def response_success(response: object) -> bool:
    return bool(getattr(response, "success", False))


def print_response(service_name: str, response: object) -> None:
    print(f"service: {service_name}")
    print(f"  success  {getattr(response, 'success', None)}")
    print(f"  message  {getattr(response, 'message', '')}")


def add_joint_override_args(parser: argparse.ArgumentParser) -> None:
    for index in range(1, 8):
        parser.add_argument(
            f"--l{index}",
            f"--ljoint{index}",
            dest=f"l{index}",
            type=parse_angle,
            help=f"left arm joint {index} target, rad or deg",
        )
        parser.add_argument(
            f"--r{index}",
            f"--rjoint{index}",
            dest=f"r{index}",
            type=parse_angle,
            help=f"right arm joint {index} target, rad or deg",
        )


def add_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true", help="send command; without this it is dry-run only")
    parser.add_argument("--skip-power-check", action="store_true", help="do not require arm power ready before send")
    parser.add_argument("--power-timeout", type=float, default=3.0, help="arm power status wait timeout")
    parser.add_argument("--service-wait", type=float, default=2.0, help="service wait timeout")
    parser.add_argument("--motion-timeout", type=float, default=30.0, help="service response/motion wait timeout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dual-arm control test helper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="print arm power and current arm joints")
    status.add_argument("--state-timeout", type=float, default=3.0, help="joint state wait timeout")
    status.add_argument("--power-timeout", type=float, default=3.0, help="arm power status wait timeout")
    status.set_defaults(func=run_status)

    joint = subparsers.add_parser(
        "joint",
        aliases=("move-j", "movej"),
        help="control both arms in joint space",
    )
    joint.add_argument(
        "--left",
        type=parse_joint_csv,
        help="left arm 7 joints, comma list; e.g. 0deg,0deg,0deg,-60deg,0deg,0deg,0deg",
    )
    joint.add_argument(
        "--right",
        type=parse_joint_csv,
        help="right arm 7 joints, comma list; e.g. 0deg,0deg,0deg,60deg,0deg,0deg,0deg",
    )
    add_joint_override_args(joint)
    joint.add_argument("--transport", choices=("service", "topic"), default="service", help="send path")
    joint.add_argument("--vel", type=float, default=0.40, help="joint velocity")
    joint.add_argument("--acc", type=float, default=0.80, help="joint acceleration")
    joint.add_argument("--state-timeout", type=float, default=3.0, help="joint state wait timeout")
    joint.add_argument("--repeat", type=int, default=5, help="topic publish repeats")
    joint.add_argument("--settle-sec", type=float, default=0.05, help="topic publish pause")
    add_motion_args(joint)
    joint.set_defaults(func=run_joint)

    cartesian = subparsers.add_parser(
        "cartesian",
        aliases=("move-l", "movel"),
        help="control both arm TCP targets by service",
    )
    cartesian.add_argument(
        "--left",
        type=parse_pose_csv,
        required=True,
        help="left TCP target: x,y,z,roll,pitch,yaw; meters and rad/deg",
    )
    cartesian.add_argument(
        "--right",
        type=parse_pose_csv,
        required=True,
        help="right TCP target: x,y,z,roll,pitch,yaw; meters and rad/deg",
    )
    cartesian.add_argument("--left-quat", type=parse_quat_csv, help="optional left qx,qy,qz,qw")
    cartesian.add_argument("--right-quat", type=parse_quat_csv, help="optional right qx,qy,qz,qw")
    cartesian.add_argument("--vel", type=float, default=0.10, help="cartesian velocity")
    cartesian.add_argument("--acc", type=float, default=0.20, help="cartesian acceleration")
    add_motion_args(cartesian)
    cartesian.set_defaults(func=run_cartesian)

    return parser


def run_status(node: DualArmControlClient, args: argparse.Namespace) -> int:
    status_code = 0
    try:
        current = node.wait_for_arm_joint_state(args.state_timeout)
    except TimeoutError as exc:
        print(f"current joints: {exc}")
        status_code = 1
    else:
        left = side_from_current(current, LEFT_ARM_JOINTS)
        right = side_from_current(current, RIGHT_ARM_JOINTS)
        print_joint_table("current arm joints:", left, right)

    try:
        snapshot = node.wait_for_arm_power(args.power_timeout)
    except TimeoutError as exc:
        print(f"arm power: {exc}")
        status_code = 1
    else:
        print_power_status(snapshot)

    return status_code


def run_joint(node: DualArmControlClient, args: argparse.Namespace) -> int:
    left, right, current = build_joint_target(node, args)
    if current is not None:
        print_joint_table(
            "current arm joints:",
            side_from_current(current, LEFT_ARM_JOINTS),
            side_from_current(current, RIGHT_ARM_JOINTS),
        )
    print_joint_table("target arm joints:", left, right)
    print(f"vel: {args.vel:.3f}")
    print(f"acc: {args.acc:.3f}")
    print(f"transport: {args.transport}")

    if not args.execute:
        print("dry-run only; add --execute to send")
        return 0

    maybe_require_power(node, args)
    if args.transport == "topic":
        node.publish_joint_absolute(
            left_joints=left,
            right_joints=right,
            vel=args.vel,
            acc=args.acc,
            repeat=args.repeat,
            settle_sec=args.settle_sec,
        )
        print("published /arm_joint_absolute_cmd")
        return 0

    service_name, response = node.call_joint_absolute(
        left_joints=left,
        right_joints=right,
        vel=args.vel,
        acc=args.acc,
        service_wait=args.service_wait,
        motion_timeout=args.motion_timeout,
    )
    print_response(service_name, response)
    return 0 if response_success(response) else 2


def run_cartesian(node: DualArmControlClient, args: argparse.Namespace) -> int:
    left = args.left
    right = args.right
    if args.left_quat is not None:
        left = replace(
            left,
            qx=args.left_quat[0],
            qy=args.left_quat[1],
            qz=args.left_quat[2],
            qw=args.left_quat[3],
        )
    if args.right_quat is not None:
        right = replace(
            right,
            qx=args.right_quat[0],
            qy=args.right_quat[1],
            qz=args.right_quat[2],
            qw=args.right_quat[3],
        )

    print_pose_table("target arm TCP:", left, right)
    print(f"vel: {args.vel:.3f}")
    print(f"acc: {args.acc:.3f}")

    if not args.execute:
        print("dry-run only; add --execute to send")
        return 0

    maybe_require_power(node, args)
    service_name, response = node.call_cartesian_absolute(
        left=left,
        right=right,
        vel=args.vel,
        acc=args.acc,
        service_wait=args.service_wait,
        motion_timeout=args.motion_timeout,
    )
    print_response(service_name, response)
    return 0 if response_success(response) else 2


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = normalize_option_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    rclpy.init()
    node = DualArmControlClient()
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
