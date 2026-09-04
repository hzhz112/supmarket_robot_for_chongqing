"""独立零售抓取网页平台。

启动：``python3 retail_grasp_web.py``，局域网访问 ``http://机器人IP:8080``。
网页不嵌入旧调试界面；抓取流程由 retail_grasp_controller 独立执行。
"""
from __future__ import annotations
import asyncio, os, threading, time, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, Response
import uvicorn
from retail_grasp_controller import DirectLeftGrasp, RetailVision

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RIGHT_HAND_SCRIPT = ROOT.parent / "linkhand" / "linker_hand_python_sdk" / "example" / "o7_rs485_control.py"
LEFT_GRIPPER_SCRIPT = PROJECT_ROOT / "omni_picker_rs485_test.py"
CATALOG = PROJECT_ROOT / "config" / "grasp_catalog.yaml"
app = FastAPI(title="Retail Grasp Console")
catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}
products = [x for x in catalog.get("objects", []) if not str(x.get("name", "")).startswith("箱子")]

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
            self.log(f"启动末端自检失败：{exc}")

    def power(self, enable: bool) -> None:
        if not self.backend: raise RuntimeError("ROS 控制器未初始化")
        self.backend.send("power", enable=enable); self.log("上使能已发送" if enable else "下使能已发送")

    def gripper(self, hand: str, action: str) -> None:
        if action not in {"close", "open"}: raise RuntimeError("动作必须是 close 或 open")
        if hand == "left":
            port = os.environ.get("LEFT_GRIPPER_PORT", "/dev/ttyS2")
            command = ["python3", "-u", str(LEFT_GRIPPER_SCRIPT), "--port", port, f"--{'close' if action == 'close' else 'open'}-once"]
        elif hand == "right":
            port = os.environ.get("RIGHT_HAND_PORT", "/dev/ttyS3")
            command = ["python3", "-u", str(RIGHT_HAND_SCRIPT), "--port", port, "--hand_type", "right", "--once", "--preset", "fist" if action == "close" else "open", "--no-open-on-exit"]
        else: raise RuntimeError("未知末端执行器")
        self.log(f"{'左手夹爪' if hand == 'left' else '右手灵巧手'}{'闭合' if action == 'close' else '张开'}：开始执行（{port}）")
        try:
            result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=15)
        except Exception as exc:
            raise RuntimeError(f"末端程序启动失败：{exc}") from exc
        output = (result.stdout + result.stderr).strip()
        if output: self.log(output)
        if result.returncode != 0: raise RuntimeError(f"末端程序退出码 {result.returncode}")
        self.log(f"{'左手夹爪' if hand == 'left' else '右手灵巧手'}{'闭合' if action == 'close' else '张开'}完成")

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
        if hand != "left": raise RuntimeError("右手流程尚未实现")
        self.running = True; self._thread = threading.Thread(target=self._workflow, args=(item, max_step), daemon=True); self._thread.start()

    def stop(self) -> None:
        self.running = False; self.log("自动抓取已停止")
        if hasattr(self, "grasp"): self.grasp.stop()

    def _workflow(self, item: dict[str, Any], max_step: int) -> None:
        try:
            if not self.backend: raise RuntimeError("ROS 控制器未初始化")
            self.grasp.run(item, self.vision.detect, max_step=max_step)
        except Exception as exc: self.log(f"流程失败：{exc}")
        finally: self.running = False

controller = Controller()

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "retail_grasp_web.html").read_text(encoding="utf-8")
@app.get("/api/products")
def get_products(): return products
@app.get("/api/status")
def status(): return {"running": controller.running, "logs": controller.logs}
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
@app.post("/api/gripper/{hand}/{action}")
def gripper(hand: str, action: str): controller.gripper(hand, action); return {"ok": True}
@app.post("/api/start/{name}")
def start(name: str, hand: str = "left", max_step: int = 11):
    item = next((x for x in products if x.get("name") == name), None)
    if not item: return {"ok": False, "error": "商品不存在"}
    controller.start(item, hand, max_step); return {"ok": True, "max_step": max_step}
@app.post("/api/stop")
def stop(): controller.stop(); return {"ok": True}
@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept(); seen = 0
    while True:
        await asyncio.sleep(.3)
        logs = controller.logs
        if len(logs) != seen: await socket.send_json({"running": controller.running, "logs": logs[seen:]}); seen = len(logs)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
