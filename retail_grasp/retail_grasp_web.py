"""独立零售抓取网页平台。

启动：``python3 retail_grasp_web.py``，局域网访问 ``http://机器人IP:8080``。
网页不嵌入旧调试界面；抓取流程由 retail_grasp_controller 独立执行。
"""
from __future__ import annotations
import asyncio, os, threading, time, subprocess, sys, json
from pathlib import Path
from typing import Any
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
from retail_grasp_controller import (
    DirectLeftGrasp, DirectRightGrasp, BoxGrasp, RetailVision,
    RESET_L, RESET_R, LAYER_FL, LAYER_SL,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RIGHT_HAND_SCRIPT = PROJECT_ROOT.parent / "linkhand" / "linker_hand_python_sdk" / "example" / "o7_rs485_control.py"
LEFT_GRIPPER_SCRIPT = PROJECT_ROOT / "omni_picker_rs485_test.py"
# 零售抓取链路左夹爪闭合力度（OmniPicker 协议范围 0..255）。
LEFT_GRIPPER_FORCE = 255
CATALOG = PROJECT_ROOT / "config" / "grasp_catalog.yaml"
app = FastAPI(title="Retail Grasp Console")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}
products = [x for x in catalog.get("objects", []) if not str(x.get("name", "")).startswith("箱子")]
boxes = [
    {"name": "第一层箱子", "label": "xiangzi", "box_layer": "FL"},
    {"name": "第二层箱子", "label": "xiangzi", "box_layer": "SL"},
]

class CameraCapture:
    """不依赖 Qt 事件循环的 RealSense 采集线程，供网页 JPEG 和视觉识别共用。"""
    def __init__(self, host: Any, log: Any) -> None:
        self.host, self.log, self.stop_event = host, log, threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
    def _run(self) -> None:
        pipe = None
        try:
            import numpy as np, pyrealsense2 as rs
            pipe, cfg = rs.pipeline(), rs.config()
            serial = os.environ.get("REALSENSE_SERIAL", "347522072040")
            if serial: cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            profile = pipe.start(cfg); align = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self.host._camera_intrinsics = (float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy), scale, "camera_color_optical_frame")
            self.log(f"网页相机已连接 RealSense serial={serial}")
            while not self.stop_event.is_set():
                frames = align.process(pipe.wait_for_frames(timeout_ms=1000)); color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth: continue
                self.host._camera_color_image = np.asanyarray(color.get_data()).copy()
                self.host._camera_depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
        except Exception as exc:
            self.error = str(exc)
            self.log(f"网页相机启动失败：{exc}")
        finally:
            try: pipe.stop()
            except Exception: pass
    def stop(self) -> None: self.stop_event.set()

class Controller:
    def __init__(self) -> None:
        self.running = False; self.logs: list[str] = []; self._thread: threading.Thread | None = None
        self.backend: Any = None; self.host: Any = None
        self._end_effector_lock = threading.Lock()
        self._startup_done = threading.Event()
        self._startup_error: str | None = None
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt5 import QtWidgets
            from robot_debug_console.main import RobotBackend
            self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            self.backend = RobotBackend(); self.backend.log_line.connect(self.log); self.backend.start()
            self.host = type("CameraState", (), {})()
            self.camera = CameraCapture(self.host, self.log)
            self.vision = RetailVision(self.host, self.backend, self.log)
            self.grasp = DirectLeftGrasp(self.backend, self.log, lambda action: self.gripper("left", action))
            self.right_grasp = DirectRightGrasp(self.backend, self.log, lambda action: self.gripper("right", action))
            self.box_grasp = BoxGrasp(self.backend, self.log, lambda action: self.gripper("left", action), lambda action: self.gripper("right", action), self._right_hand_pose)
            threading.Thread(target=self._startup_grippers, daemon=True).start()
        except Exception as exc:
            self.log(f"控制器初始化失败（网页仍可打开）: {exc}")

    def log(self, text: str) -> None:
        self.logs.append(str(text)); self.logs[:] = self.logs[-300:]

    def _startup_grippers(self) -> None:
        """启动自检：左右末端各闭合一次再张开，最终保持张开。"""
        time.sleep(1.0)
        try:
            self.gripper("left", "close"); time.sleep(2); self.gripper("left", "open")
            self.gripper("right", "close"); time.sleep(2); self.gripper("right", "open")
            self.log("启动末端自检完成：左右末端均为张开状态")
        except Exception as exc:
            self._startup_error = str(exc)
            self.log(f"启动末端自检失败：{exc}")
        finally:
            self._startup_done.set()

    def power(self, enable: bool) -> None:
        if not self.backend: raise RuntimeError("ROS 控制器未初始化")
        self.backend.send("power", enable=enable); self.log("上使能已发送" if enable else "下使能已发送")

    def manual_arm_command(self, action: str) -> None:
        """执行网页上的独立复位/高度命令，不启动商品抓取流程。"""
        if self.running: raise RuntimeError("自动抓取运行中，暂时不能执行复位或高度命令")
        if not self.backend: raise RuntimeError("ROS 控制器未初始化")
        if action == "reset":
            self.backend.send("arm_joint", left=RESET_L, right=RESET_R, vel=.70, acc=.50)
            self.log("已发送机械臂复位命令（第一层抓取复位姿态）")
        elif action == "layer-fl":
            self.backend.send("leg_joint", values=LAYER_FL, vel=.12, acc=.20)
            self.log("已发送第一层高度命令")
        elif action == "layer-sl":
            self.backend.send("leg_joint", values=LAYER_SL, vel=.12, acc=.20)
            self.log("已发送第二层高度命令")
        else: raise ValueError("未知机械臂命令")

    def gripper(self, hand: str, action: str) -> None:
        if action not in {"close", "open"}: raise RuntimeError("动作必须是 close 或 open")
        if hand == "left":
            port = os.environ.get("LEFT_GRIPPER_PORT", "/dev/ttyS2")
            command = [sys.executable, "-u", str(LEFT_GRIPPER_SCRIPT), "--port", port, f"--{'close' if action == 'close' else 'open'}-once", "--force", str(LEFT_GRIPPER_FORCE)]
        elif hand == "right":
            port = os.environ.get("RIGHT_HAND_PORT", "/dev/ttyS3")
            command = [sys.executable, "-u", str(RIGHT_HAND_SCRIPT), "--port", port, "--hand_type", "right", "--once", "--preset", "fist" if action == "close" else "open", "--no-open-on-exit"]
        else: raise RuntimeError("未知末端执行器")
        if hand == "left":
            self.log(f"左手夹爪{'闭合' if action == 'close' else '张开'}：开始执行（{port}）")
        try:
            with self._end_effector_lock:
                result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=15)
        except Exception as exc:
            raise RuntimeError(f"末端程序启动失败：{exc}") from exc
        output = (result.stdout + result.stderr).strip()
        if output:
            self.log(output)
        if result.returncode != 0:
            detail = output[-1200:] if output else "无脚本输出"
            raise RuntimeError(f"末端程序退出码 {result.returncode}: {detail}")
        self.log(f"{'左手夹爪' if hand == 'left' else '右手灵巧手'}{'闭合' if action == 'close' else '张开'}完成")

    def _right_hand_pose(self, values: tuple[int, ...]) -> None:
        if len(values) != 7 or any(int(v) < 0 or int(v) > 255 for v in values):
            raise ValueError(f"右手姿态必须是 7 个 0-255 整数，当前={values}")
        port = os.environ.get("RIGHT_HAND_PORT", "/dev/ttyS3")
        command = [sys.executable, "-u", str(RIGHT_HAND_SCRIPT), "--port", port, "--hand_type", "right", "--once", "--position", *[str(int(v)) for v in values], "--no-open-on-exit"]
        with self._end_effector_lock:
            result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=15)
        output = (result.stdout + result.stderr).strip()
        if output:
            self.log(output)
        if result.returncode != 0:
            detail = output[-1200:] if output else "无脚本输出"
            raise RuntimeError(f"右手姿态程序退出码 {result.returncode}: {detail}")

    def _wait_motion(self, arm: bool = True, leg: bool = False, timeout: float = 45.0) -> None:
        """等待 ROS motion_status：先观察到 moving，再等待停止/到达。"""
        deadline, saw_arm, saw_leg = time.monotonic() + timeout, False, False
        while self.running and time.monotonic() < deadline:
            snap = self.backend.snapshot()
            am, lm = snap.arm_motion, snap.leg_motion
            if arm and am is not None:
                saw_arm |= bool(am[0])
            if leg and lm is not None:
                saw_leg |= bool(lm[0])
            elapsed = timeout - max(0.0, deadline - time.monotonic())
            # 有些控制器动作很快，不一定能采到 moving 状态；至少等待 1 秒后接受 goal_reached。
            arm_done = not arm or ((saw_arm and am is not None and not am[0]) or (elapsed > 1.0 and am is not None and am[1]))
            leg_done = not leg or ((saw_leg and lm is not None and not lm[0]) or (elapsed > 1.0 and lm is not None and lm[1]))
            if arm_done and leg_done: return
            time.sleep(.1)
        if self.running: raise TimeoutError("等待机械臂/升降运动完成超时")
        raise RuntimeError("任务已停止")

    def start(self, item: dict[str, Any], hand: str = "left", max_step: int = 11) -> None:
        if self.running: raise RuntimeError("已有任务运行中")
        hand = str(hand).lower()
        if hand not in {"left", "right"}: raise ValueError("抓取手臂必须是 left 或 right")
        # 仅覆盖本次任务选择的手臂，不改动目录数据及其余抓取参数。
        task_item = dict(item)
        task_item["hand"] = hand
        # stop() 会将流程对象标记为 cancelled；新任务开始前必须清除旧状态。
        self.grasp.cancelled = False
        self.right_grasp.cancelled = False
        self.box_grasp.cancelled = False
        self.running = True; self._thread = threading.Thread(target=self._workflow, args=(task_item, max_step), daemon=True); self._thread.start()

    def start_box_place_hand(self, hand: str) -> None:
        if self.running: raise RuntimeError("已有任务运行中")
        hand = str(hand).lower()
        if hand not in {"left", "right"}: raise ValueError("放置手臂必须是 left 或 right")
        self.box_grasp.cancelled = False
        self.running = True
        self._thread = threading.Thread(target=self._box_place_hand_workflow, args=(hand,), daemon=True)
        self._thread.start()

    def _box_place_hand_workflow(self, hand: str) -> None:
        try:
            if not self.backend: raise RuntimeError("ROS 控制器未初始化")
            if not self._startup_done.wait(timeout=30.0):
                raise TimeoutError("等待启动末端自检结束超时")
            if self._startup_error:
                raise RuntimeError(f"启动末端自检失败，未执行任务: {self._startup_error}")
            self.box_grasp.place_hand(hand)
        except Exception as exc:
            self.log(f"{hand}手放置流程失败：{exc}")
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False; self.log("自动抓取已停止")
        if hasattr(self, "grasp"): self.grasp.stop()
        if hasattr(self, "right_grasp"): self.right_grasp.stop()
        if hasattr(self, "box_grasp"): self.box_grasp.stop()

    def _workflow(self, item: dict[str, Any], max_step: int) -> None:
        try:
            if not self.backend: raise RuntimeError("ROS 控制器未初始化")
            if not self._startup_done.wait(timeout=30.0):
                raise TimeoutError("等待启动末端自检结束超时")
            if self._startup_error:
                raise RuntimeError(f"启动末端自检失败，未执行任务: {self._startup_error}")
            if item.get("box_layer"):
                self.box_grasp.run(str(item["box_layer"]), self.vision.detect, max_step=max_step)
            else:
                runner = self.right_grasp if str(item.get("hand", "left")).lower() == "right" else self.grasp
                runner.run(item, self.vision.detect, max_step=max_step)
        except Exception as exc: self.log(f"流程失败：{exc}")
        finally: self.running = False

controller = Controller()

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # 禁止浏览器复用旧页面，确保前端选择层修改立即生效。
    return HTMLResponse(
        (ROOT / "retail_grasp_web.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
@app.get("/api/products")
def get_products(): return Response(content=json.dumps(products, ensure_ascii=False), media_type="application/json", headers={"Cache-Control": "no-store"})
@app.get("/api/boxes")
def get_boxes(): return boxes
@app.get("/api/status")
def status(): return {"running": controller.running, "logs": controller.logs}
@app.post("/api/logs/clear")
def clear_logs():
    controller.logs.clear()
    return {"ok": True}
@app.get("/api/camera.jpg")
def camera_jpg():
    try:
        import cv2
        image = getattr(controller.host, "_camera_color_image", None)
        if image is None: return Response(status_code=204)
        ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return Response(content=data.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}) if ok else Response(status_code=204)
    except Exception:
        return Response(status_code=204)
@app.get("/api/camera/status")
def camera_status():
    image = getattr(controller.host, "_camera_color_image", None)
    return {"ready": image is not None, "shape": list(image.shape) if image is not None else None,
            "error": getattr(getattr(controller, "camera", None), "error", None),
            "logs": [x for x in controller.logs if "相机" in x or "camera" in x.lower()][-5:]}
@app.post("/api/power/{enable}")
def set_power(enable: bool): controller.power(enable); return {"ok": True}
@app.post("/api/arm/{action}")
def arm_command(action: str):
    if action not in {"reset", "layer-fl", "layer-sl"}:
        return {"ok": False, "error": "命令必须是 reset、layer-fl 或 layer-sl"}
    try:
        controller.manual_arm_command(action)
        return {"ok": True, "action": action}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
@app.post("/api/gripper/{hand}/{action}")
def gripper(hand: str, action: str):
    if controller.running:
        return {"ok": False, "error": "自动抓取运行中，暂时不能手动控制末端"}
    if not controller._startup_done.is_set():
        return {"ok": False, "error": "启动末端自检进行中，请稍后再控制末端"}
    if hand not in {"left", "right"} or action not in {"open", "close"}:
        return {"ok": False, "error": "参数必须是 hand=left/right、action=open/close"}
    controller.log(f"手动末端操作：{hand} {action}")
    controller.gripper(hand, action)
    return {"ok": True, "hand": hand, "action": action}
@app.post("/api/start/{name}")
def start(name: str, hand: str = "left", max_step: int = 11):
    item = next((x for x in products + boxes if x.get("name") == name), None)
    if not item: return {"ok": False, "error": "商品不存在"}
    hand = str(hand).lower()
    if hand not in {"left", "right"}:
        return {"ok": False, "error": "抓取手臂必须是 left 或 right"}
    # 瓜子、脆升升、派为右手特殊商品，始终锁定右手。
    if str(item.get("label", "")).lower() in {"guazi", "cui", "pai"} and hand != "right":
        return {"ok": False, "error": "该特殊商品只能使用右手抓取"}
    if not 1 <= max_step <= 10:
        return {"ok": False, "error": "max_step 必须是 1 到 10"}
    controller.log(f"收到调试任务：{name}，{('左' if hand == 'left' else '右')}手，最大执行步骤={max_step}")
    controller.start(item, hand, max_step); return {"ok": True, "max_step": max_step}
@app.post("/api/box/{layer}/{action}")
def box_action(layer: str, action: str, max_step: int = 8):
    if layer not in {"FL", "SL"} or action not in {"grasp", "place"}:
        return {"ok": False, "error": "参数必须是 layer=FL/SL、action=grasp/place"}
    item = next(x for x in boxes if x["box_layer"] == layer)
    if controller.running: return {"ok": False, "error": "已有任务运行中"}
    if not 1 <= max_step <= 8: return {"ok": False, "error": "max_step 必须是 1 到 8"}
    controller.log(f"收到箱子{action}任务：{item['name']}，最大执行步骤={max_step}")
    try:
        if action == "grasp":
            controller.start(item, "left", max_step)
        else:
            controller.box_grasp.place(layer, max_step)
        return {"ok": True, "max_step": max_step, "action": action}
    except Exception as exc:
        controller.log(f"箱子{action}失败：{exc}")
        return {"ok": False, "error": str(exc)}

@app.post("/api/box-hand-place/{hand}")
def box_place_hand(hand: str):
    """新增的左右手固定坐标放置入口；具体动作在线程中执行。"""
    hand = str(hand).lower()
    if hand not in {"left", "right"}:
        return {"ok": False, "error": "hand 必须是 left 或 right"}
    if controller.running:
        return {"ok": False, "error": "已有任务运行中"}
    controller.log(f"收到{hand}手放置任务")
    try:
        controller.start_box_place_hand(hand)
        return {"ok": True, "hand": hand}
    except Exception as exc:
        controller.log(f"{hand}手放置任务启动失败：{exc}")
        return {"ok": False, "error": str(exc)}

@app.post("/api/stop")
def stop(): controller.stop(); return {"ok": True}
@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept(); seen = 0
    while True:
        await asyncio.sleep(.3)
        logs = controller.logs
        if len(logs) < seen:
            seen = 0
        if len(logs) != seen: await socket.send_json({"running": controller.running, "logs": logs[seen:]}); seen = len(logs)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
