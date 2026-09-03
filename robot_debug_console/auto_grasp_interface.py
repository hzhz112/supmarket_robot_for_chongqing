"""商品选择与左手自动抓取流程。底层运动仍由 :mod:`main` 提供。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from PyQt5 import QtCore, QtGui, QtWidgets


class AutoGraspPage(QtWidgets.QWidget):
    """将已有按钮动作串成一个可停止的左手抓取流程。"""

    def __init__(self, host: Any, catalog_path: Path) -> None:
        super().__init__()
        self.host = host
        self.root = Path(__file__).resolve().parents[1]
        self.running = False
        self.item: dict[str, Any] | None = None
        self._stage = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._advance)
        self.items = self._load(catalog_path)
        self._build()
        self.setStyleSheet("""
            AutoGraspPage { background: #f7f9fc; }
            QLabel#title { color: #172554; font-size: 20px; font-weight: 700; }
            QPushButton { background: white; color: #172033; border: 1px solid #d7deea; border-radius: 12px; padding: 8px; font-size: 13px; }
            QPushButton:hover { border: 2px solid #4f7cff; background: #f1f5ff; }
            QPushButton[selected="true"] { background: #e7efff; border: 2px solid #3567e8; color: #1d4ed8; }
            QPushButton#confirm { background: #3567e8; color: white; font-weight: 700; padding: 10px 18px; }
            QPushButton#stop { background: #fff1f2; color: #be123c; border-color: #fecdd3; padding: 10px 18px; }
            QTextEdit { background: #101827; color: #dbeafe; border-radius: 10px; padding: 8px; }
        """)

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [x for x in data.get("objects", []) if not str(x.get("name", "")).startswith("箱子")]

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("自动抓取运行台"); title.setObjectName("title")
        root.addWidget(title)
        root.addWidget(QtWidgets.QLabel("选择商品后点击确认，系统将自动完成左手抓取。图片放在 config/product_images/ 下即可自动显示。"))
        grid = QtWidgets.QGridLayout()
        self.buttons: list[QtWidgets.QPushButton] = []
        for i, obj in enumerate(self.items):
            button = QtWidgets.QPushButton()
            button.setMinimumSize(150, 115)
            button.setToolTip(f"{obj['name']} / {obj.get('label', '')}")
            image = obj.get("image")
            image_path = (self.root / str(image)) if image else self.root / "config" / "product_images" / f"{obj.get('label', '')}.png"
            if image_path.exists():
                button.setIcon(QtGui.QIcon(str(image_path)))
                button.setIconSize(QtCore.QSize(82, 72))
            button.setText(str(obj["name"]))
            button.clicked.connect(lambda _=False, value=obj: self._select(value))
            grid.addWidget(button, i // 4, i % 4)
            self.buttons.append(button)
        root.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        self.selected_label = QtWidgets.QLabel("尚未选择商品")
        self.confirm_btn = QtWidgets.QPushButton("确认并开始左手抓取")
        self.confirm_btn.setObjectName("confirm")
        self.stop_btn = QtWidgets.QPushButton("停止自动抓取")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._confirm)
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.selected_label, 1); row.addWidget(self.confirm_btn); row.addWidget(self.stop_btn)
        root.addLayout(row)
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(180)
        root.addWidget(self.log)

    def _select(self, obj: dict[str, Any]) -> None:
        self.item = obj
        for button, value in zip(self.buttons, self.items):
            button.setProperty("selected", value is obj)
            button.style().unpolish(button); button.style().polish(button)
        self.selected_label.setText(f"已选择：{obj['name']}（{obj.get('shelf_layer', '--')}，默认{obj.get('hand', '未设置')}手）")

    def _confirm(self) -> None:
        if self.running or not self.item:
            return
        hand = str(self.item.get("hand", "left")).lower()
        if hand == "right":
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("选择抓取手")
            box.setText(f"{self.item['name']} 默认使用右手，请选择本次抓取手臂")
            left_btn = box.addButton("左手", QtWidgets.QMessageBox.AcceptRole)
            right_btn = box.addButton("右手", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is right_btn:
                self._write("用户选择右手；右手自动流程尚未启用")
                return
            if box.clickedButton() is not left_btn:
                return
        self.running = True; self._stage = 0
        self.confirm_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._write(f"开始自动抓取：{self.item['name']}")
        self._timer.start()

    def _write(self, message: str) -> None:
        self.log.append(message); self.host.append_log("[AUTO] " + message)

    def _advance(self) -> None:
        if not self.running or not self.item:
            return
        if getattr(self, "_wait", 0.0) > 0.0:
            self._wait -= 0.2
            return
        try:
            # 每阶段只发送一次，等待运动状态稳定后再进入下一阶段。
            if self._stage == 0:
                self.host._send_reset_arms(); self._write("机械臂复位已发送"); self._stage = 1; self._wait = 1.5; return
            if self._stage == 1:
                layer = str(self.item.get("shelf_layer", "FL")).upper()
                if layer == "SL":
                    self.host._send_second_layer_height()
                    self.host._send_second_layer_grasp_pose()
                    self._write("第二层高度和第二层抓取姿态已发送")
                else:
                    self.host._send_first_layer_height(); self._write("第一层高度已发送")
                self._stage = 2; self._wait = 1.5; return
            if self._stage == 2:
                label = str(self.item["label"]).lower()
                label_index = self.host.vision_class_combo.findData(label)
                if label_index < 0:
                    raise ValueError(f"视觉标签不存在：{label}")
                self.host.vision_class_combo.setCurrentIndex(label_index)
                for i in range(self.host.vision_model_combo.count()):
                    if Path(str(self.host.vision_model_combo.itemData(i))).name.lower() == "best.pt":
                        self.host.vision_model_combo.setCurrentIndex(i)
                        break
                self.host.grasp_arm_combo.setCurrentText("left")
                self.host.grasp_forward_spin.setValue(0.08); self.host.grasp_contact_gap_spin.setValue(10.0); self.host.grasp_height_spin.setValue(0.45)
                self.host._detect_vision_target(); self._write("视觉识别完成")
                self.host._generate_grasp_pose(); self._write("预抓取姿态计算完成"); self._stage = 3; self._wait = 0.3; return
            if self._stage == 3:
                self.host._move_to_grasp_align(); self._write("已到达预抓取姿态"); self._stage = 4; self._wait = 2.0; return
            if self._stage == 4:
                self.host._move_to_grasp_front(); self._write("TCP 前进 10 cm 完成"); self._stage = 5; self._wait = 2.0; return
            if self._stage == 5:
                self.host._run_left_gripper_close(); self._write("左夹爪闭合命令已发送"); self._stage = 6; self._wait = 3.0; return
            if self._stage == 6:
                if getattr(self.host, "_grasp_process", None) is not None:
                    return
                self.host._lift_arm_then_reset("left"); self._write("左手抬起并复位命令已发送"); self._finish(); return
        except Exception as exc:
            self._write(f"流程失败：{exc}"); self.stop()

    def _finish(self) -> None:
        self._timer.stop(); self.running = False; self.confirm_btn.setEnabled(True); self.stop_btn.setEnabled(False); self._write("自动抓取流程完成")

    def stop(self) -> None:
        self._timer.stop(); self.running = False; self.confirm_btn.setEnabled(True); self.stop_btn.setEnabled(False); self._write("自动抓取已停止")


def run() -> int:
    """独立启动入口：不修改 main.py，在原控制台上挂载自动抓取页。"""
    import signal
    import sys
    try:
        from .main import RobotBackend, DmpMainWindow
    except ImportError:
        from main import RobotBackend, DmpMainWindow
    app = QtWidgets.QApplication(sys.argv)
    backend = RobotBackend(); backend.start()
    window = DmpMainWindow(backend)
    page = AutoGraspPage(window, Path(__file__).resolve().parents[1] / "config" / "grasp_catalog.yaml")
    window.tabs.addTab(page, "自动抓取运行台")
    window.tabs.setCurrentWidget(page)
    window.showMaximized()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(run())
