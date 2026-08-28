#!/usr/bin/env python3
"""Interactive tuner for LinkerHand O7 over RS485."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox


SDK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "linkhand", "linker_hand_python_sdk")
)
sys.path.insert(0, SDK_ROOT)


JOINT_NAMES = [
    "thumb_pitch",
    "thumb_yaw",
    "index_pitch",
    "middle_pitch",
    "ring_pitch",
    "little_pitch",
    "thumb_roll",
]

PRESET_POSES = {
    "open": [250, 250, 250, 250, 250, 250, 250],
    "fist": [40, 60, 40, 40, 40, 40, 100],
    "pinch": [55, 45, 60, 250, 250, 250, 100],
    "relax": [180, 180, 180, 180, 180, 180, 180],
}

DEFAULT_SPEED = [120, 120, 120, 120, 120, 120, 120]
DEFAULT_TORQUE = [180, 180, 180, 180, 180, 180, 180]


def parse_u8_list(values: list[str] | None, name: str) -> list[int] | None:
    if values is None:
        return None
    parts = values[0].split(",") if len(values) == 1 and "," in values[0] else values
    parsed = [int(value) for value in parts]
    if len(parsed) != len(JOINT_NAMES):
        raise argparse.ArgumentTypeError(f"{name} needs 7 values, got {len(parsed)}")
    if any(value < 0 or value > 255 for value in parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be 0..255")
    return parsed


def print_named_values(title: str, values: list[int]) -> None:
    print(title)
    for name, value in zip(JOINT_NAMES, values):
        print(f"  {name:13s}: {value}")


def detect_serial_port() -> str | None:
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def choose_port(port: str | None) -> str:
    if port:
        return port
    detected = detect_serial_port()
    if detected:
        print(f"auto port: {detected}")
        return detected
    raise RuntimeError("No RS485 port found. Pass --port /dev/ttyS3.")


def parse_pose_text(text: str, current: list[int] | None = None) -> list[int]:
    text = text.strip()
    if not text:
        raise ValueError("empty input")
    if text in PRESET_POSES:
        return list(PRESET_POSES[text])
    if text.startswith("thumb "):
        if current is None:
            raise ValueError("thumb command needs a current pose first")
        parts = text.split()
        if len(parts) != 4:
            raise ValueError("usage: thumb <pitch> <yaw> <roll>")
        out = list(current)
        out[0] = int(parts[1])
        out[1] = int(parts[2])
        out[6] = int(parts[3])
        return out
    values = [int(value) for value in text.replace(",", " ").split()]
    if len(values) != len(JOINT_NAMES):
        raise ValueError("need 7 values: thumb_pitch thumb_yaw index middle ring little thumb_roll")
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("all values must be 0..255")
    return values


def read_status(hand: object) -> list[int]:
    current = hand.get_current_status()
    print_named_values("current position:", current)
    print_named_values("speed:", hand.get_speed())
    print_named_values("torque:", hand.get_torque())
    print_named_values("temperature:", hand.get_temperature())
    print_named_values("fault:", hand.get_fault())
    return current


def run_interactive(hand: object) -> None:
    current = read_status(hand)
    print("\nCommands:")
    print("  open / fist / pinch / relax")
    print("  7 values, e.g. 40 60 40 40 40 40 100")
    print("  thumb <pitch> <yaw> <roll>, e.g. thumb 40 60 100")
    print("  read")
    print("  q")
    while True:
        raw = input("\nhand> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return
        if raw == "read":
            current = read_status(hand)
            continue
        try:
            pose = parse_pose_text(raw, current)
        except ValueError as exc:
            print(f"input error: {exc}")
            continue
        print_named_values("target:", pose)
        hand.set_joint_positions(pose)
        current = pose


class HandTunerApp:
    def __init__(self, root: tk.Tk, hand: object) -> None:
        self.root = root
        self.hand = hand
        self.values = [tk.IntVar(value=value) for value in PRESET_POSES["open"]]
        self.status_var = tk.StringVar(value="ready")
        self.pose_var = tk.StringVar(value="")

        root.title("LinkerHand O7 Tuner")
        root.geometry("720x460")

        header = tk.Frame(root, padx=12, pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text="LinkerHand O7 滑动调节", font=("Sans", 16, "bold")).pack(side=tk.LEFT)
        tk.Label(header, textvariable=self.status_var).pack(side=tk.RIGHT)

        body = tk.Frame(root, padx=12)
        body.pack(fill=tk.BOTH, expand=True)
        for row, name in enumerate(JOINT_NAMES):
            tk.Label(body, text=name, width=14, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
            slider = tk.Scale(
                body,
                from_=0,
                to=255,
                orient=tk.HORIZONTAL,
                variable=self.values[row],
                length=430,
                command=lambda _value: self._refresh_pose_text(),
            )
            slider.grid(row=row, column=1, sticky="ew", pady=4)
            tk.Spinbox(
                body,
                from_=0,
                to=255,
                textvariable=self.values[row],
                width=5,
                command=self._refresh_pose_text,
            ).grid(row=row, column=2, padx=(8, 0))
        body.columnconfigure(1, weight=1)

        presets = tk.Frame(root, padx=12, pady=8)
        presets.pack(fill=tk.X)
        for name in ("open", "fist", "pinch", "relax"):
            tk.Button(presets, text=name, command=lambda preset=name: self.set_pose(PRESET_POSES[preset])).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(presets, text="读取状态", command=self.read_status).pack(side=tk.LEFT, padx=(12, 8))
        tk.Button(presets, text="发送当前值", command=self.send_current).pack(side=tk.LEFT)

        footer = tk.Frame(root, padx=12, pady=8)
        footer.pack(fill=tk.X)
        tk.Label(footer, text="当前 7 值:").pack(side=tk.LEFT)
        tk.Entry(footer, textvariable=self.pose_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        tk.Button(footer, text="复制到终端", command=self.copy_pose).pack(side=tk.LEFT)
        self._refresh_pose_text()

    def current_pose(self) -> list[int]:
        return [int(value.get()) for value in self.values]

    def set_pose(self, pose: list[int]) -> None:
        for var, value in zip(self.values, pose):
            var.set(int(value))
        self._refresh_pose_text()

    def _refresh_pose_text(self) -> None:
        self.pose_var.set(" ".join(str(value) for value in self.current_pose()))

    def send_current(self) -> None:
        pose = self.current_pose()
        try:
            self.hand.set_joint_positions(pose)
        except Exception as exc:
            self.status_var.set("send failed")
            messagebox.showerror("发送失败", str(exc))
            return
        self.status_var.set("sent: " + " ".join(str(value) for value in pose))

    def read_status(self) -> None:
        try:
            pose = self.hand.get_current_status()
        except Exception as exc:
            self.status_var.set("read failed")
            messagebox.showerror("读取失败", str(exc))
            return
        self.set_pose([int(value) for value in pose])
        self.status_var.set("read current position")

    def copy_pose(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.pose_var.get())
        self.status_var.set("copied")


def run_gui(hand: object) -> None:
    root = tk.Tk()
    HandTunerApp(root, hand)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune LinkerHand O7 finger positions.")
    parser.add_argument("--port", default="/dev/ttyS3", help="RS485 serial port")
    parser.add_argument("--hand-type", choices=("left", "right"), default="right")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--speed", nargs="+", help="7 speed values, 0..255")
    parser.add_argument("--torque", nargs="+", help="7 max torque values, 0..255")
    parser.add_argument("--position", nargs="+", help="send one 7-value target and exit")
    parser.add_argument("--preset", choices=sorted(PRESET_POSES), help="send one preset and exit")
    parser.add_argument("--read", action="store_true", help="read status before exit")
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--cli", action="store_true", help="use terminal interactive mode instead of slider GUI")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from LinkerHand.core.rs485.linker_hand_l7_rs485 import LinkerHandL7RS485

    port = choose_port(args.port)
    hand_id = 0x28 if args.hand_type == "left" else 0x27
    speed = parse_u8_list(args.speed, "speed") or DEFAULT_SPEED
    torque = parse_u8_list(args.torque, "torque") or DEFAULT_TORQUE
    target = parse_u8_list(args.position, "position")
    if args.preset:
        target = list(PRESET_POSES[args.preset])

    print(f"connect LinkerHand O7: port={port}, baudrate={args.baudrate}, id={hex(hand_id)}")
    with LinkerHandL7RS485(hand_id=hand_id, modbus_port=port, baudrate=args.baudrate) as hand:
        hand.set_speed(speed)
        hand.set_torque(torque)
        if target is not None:
            print_named_values("target:", target)
            hand.set_joint_positions(target)
            time.sleep(args.hold)
            if args.read:
                read_status(hand)
            return 0
        if args.cli:
            run_interactive(hand)
        else:
            run_gui(hand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
