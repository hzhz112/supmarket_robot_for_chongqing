"""网页抓取的独立控制接口。

这里不创建 Qt 窗口，也不调用 DmpMainWindow；只使用 ROS backend 的快照和
发布接口。一个任务的每一步都在同一个函数中显式发送并等待完成。
"""
from __future__ import annotations
import math, time, sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from robot_debug_console.main import RobotBackend, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEG_JOINTS
from robot_debug_console.box_bimanual_grasp_agent import (
    BOX_LEFT_GRASP_Z_OFFSET_M,
    BOX_LEFT_PREGRASP_DISTANCE_M,
    BOX_LEFT_PREGRASP_X_OFFSET_M,
    BOX_LIFT_DISTANCE_M,
    BOX_ARM_Z_STEP_M,
    BOX_RIGHT_GRASP_Z_OFFSET_M,
    BOX_RIGHT_PREGRASP_DISTANCE_M,
    BOX_RIGHT_PREGRASP_X_OFFSET_M,
    BOX_HAND_POSE,
    BOX_FRONT_PLANE_THRESHOLD_M,
    BOX_HEIGHT_M,
    BOX_LENGTH_M,
    BOX_WIDTH_M,
    generate_side_grasp_candidates,
    estimate_box_frame,
    estimate_front_plane,
)

RESET_L = tuple(math.radians(x) for x in (17.7, 22.5, -6.6, -106.8, -18.6, -17.1, 0.0))
RESET_R = tuple(math.radians(x) for x in (-18.8, -29.5, 9.6, 103.0, 23.9, 19.3, -4.0))
SECOND_L = tuple(math.radians(x) for x in (13.0, 22.2, -27.0, -110.0, -35.2, -19.2, 18.2))
SECOND_R = tuple(math.radians(x) for x in (-16.7, -29.5, 23.3, 109.8, 34.4, 24.9, -3.2))
PLACE_LEG_065 = tuple(math.radians(x) for x in (28.1, -57.1, 29.0, 0.0))
PLACE_XYZ = (0.40, 0.00, 1.10)
PLACE_Z_DROP_M = 0.10
LAYER_FL = tuple(math.radians(x) for x in (50.6, -103.1, 52.6, 0.0))
LAYER_SL = (0.0, 0.0, 0.0, 0.0)
RIGHT_HAND_PREGRASP_Y_OFFSET_M = -0.03
RIGHT_HAND_TCP_Y_OFFSETS_M = {"guazi": -0.06, "cui": -0.06, "pai": -0.04}
# 普通右手商品默认参数；特殊商品使用下方 SPECIAL_* 参数。
RIGHT_HAND_NORMAL_PREGRASP_DISTANCE_M = 0.05
RIGHT_HAND_NORMAL_FORWARD_DISTANCE_M = 0.07
RIGHT_HAND_NORMAL_PREGRASP_Y_OFFSET_M = -0.05
# 特殊商品（guazi/cui/pai）专用距离；普通右手商品保持原有参数。
SPECIAL_RIGHT_PREGRASP_DISTANCE_M = 0.05
SPECIAL_RIGHT_FORWARD_DISTANCE_M = 0.07
BOX_PLACE_DROP_FIRST_LAYER_M = 0.060
BOX_PLACE_DROP_SECOND_LAYER_M = 0.062

class RetailVision:
    """迁移自旧控制台的 YOLO-seg + 深度/外参目标点计算。"""
    def __init__(self, state: Any, backend: RobotBackend, log: Callable[[str], None]):
        self.state, self.backend, self.log = state, backend, log
        self.model = None
        self._models: dict[str, Any] = {}
        self.last_points_car = None
        self.last_camera_pos_car = None
        self.extrinsic = None
        path = PROJECT_ROOT / "config" / "realsense_347522072040_extrin.yaml"
        if path.exists():
            import yaml, numpy as np
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            matrix = np.asarray(data.get("extrinsic_matrix", []), dtype=float)
            if matrix.shape == (4, 4): self.extrinsic = matrix

    def detect(self, label: str, model_path: Path | None = None) -> tuple[float, float, float]:
        import cv2, numpy as np
        image, depth, intr = self.state._camera_color_image, self.state._camera_depth_m, self.state._camera_intrinsics
        if image is None or depth is None or intr is None: raise RuntimeError("相机尚未就绪")
        from ultralytics import YOLO
        model_path = model_path or (PROJECT_ROOT / "weight" / "best.pt")
        if not model_path.exists(): raise RuntimeError(f"视觉模型不存在: {model_path}")
        model_key = str(model_path)
        if model_key not in self._models:
            self._models[model_key] = YOLO(model_key); self.log(f"[VISION] 已加载模型: {model_path}")
        self.model = self._models[model_key]
        names = {str(v).lower(): int(k) for k,v in self.model.names.items()}
        if label not in names: raise RuntimeError(f"模型中没有标签: {label}")
        result = self.model.predict(image, conf=.25, iou=.45, device="cpu", classes=[names[label]], verbose=False)[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0: raise RuntimeError(f"没有检测到标签: {label}")
        best = int(result.boxes.conf.argmax().item()); mask = result.masks.data[best].cpu().numpy() > .5
        h,w=image.shape[:2]
        if mask.shape != (h,w): mask=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        inner=cv2.erode(mask.astype('uint8'),np.ones((5,5),dtype='uint8'),iterations=1).astype(bool)
        valid=inner & np.isfinite(depth) & (depth>.10) & (depth<2.0)
        if np.count_nonzero(valid)<20: raise RuntimeError("目标 mask 内有效深度点太少")
        med=float(np.median(depth[valid])); valid &= np.abs(depth-med)<.05
        v,u=np.nonzero(valid); z=depth[v,u].astype(float)
        pts=np.column_stack(((u-intr[2])*z/intr[0],(v-intr[3])*z/intr[1],z))
        if self.extrinsic is not None:
            pts=(self.extrinsic @ np.column_stack((pts,np.ones(len(pts)))).T).T[:,:3]
            tf=self.backend.snapshot().car_from_body
            if tf is not None:
                pts=(np.asarray(tf) @ np.column_stack((pts,np.ones(len(pts)))).T).T[:,:3]
                camera_pos_car = (np.asarray(tf) @ np.array([self.extrinsic[0,3], self.extrinsic[1,3], self.extrinsic[2,3], 1.0])).reshape(4)[:3]
            else:
                camera_pos_car = None
        else:
            camera_pos_car = None
        # 与 main.py 通用按钮链路一致：Z 取 2%~98% 范围的中上部(45%)，XY 取中位数。
        z_min, z_max = float(np.percentile(pts[:,2], 2)), float(np.percentile(pts[:,2], 98))
        if z_max - z_min < .05: raise RuntimeError(f"估计目标高度异常: {z_max-z_min:.3f} m")
        p=np.array([float(np.median(pts[:,0])), float(np.median(pts[:,1])), z_min + .45*(z_max-z_min)])
        self.last_points_car = pts.copy()
        self.last_camera_pos_car = None if camera_pos_car is None else camera_pos_car.copy()
        self.log(f"[VISION] {label} car_link XYZ={p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}，有效点={len(pts)}")
        return tuple(float(x) for x in p)

class DirectLeftGrasp:
    """单一、可追踪的左臂抓取流程（11 个阶段）。"""
    def __init__(self, backend: RobotBackend, log: Callable[[str], None], gripper: Callable[[str], None]):
        self.backend, self.log, self.gripper = backend, log, gripper
        self.cancelled = False

    def stop(self) -> None:
        self.cancelled = True

    def wait(self, arm: bool = True, leg: bool = False, timeout: float = 45.0) -> None:
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        saw_arm_moving = False
        saw_leg_moving = False
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            snap = self.backend.snapshot(); am, lm = snap.arm_motion, snap.leg_motion
            if arm and am is not None: saw_arm_moving |= bool(am[0])
            if leg and lm is not None: saw_leg_moving |= bool(lm[0])
            elapsed = time.monotonic() - started
            # 控制器动作很快时可能采不到 moving；至少观察 1 秒后再接受 goal_reached。
            arm_ok = not arm or (
                am is not None and not am[0] and
                (saw_arm_moving or (elapsed >= 1.0 and bool(am[1])))
            )
            leg_ok = not leg or (
                lm is not None and not lm[0] and
                (saw_leg_moving or (elapsed >= 1.0 and bool(lm[1])))
            )
            if arm_ok and leg_ok: return
            time.sleep(.1)
        snap = self.backend.snapshot()
        raise TimeoutError(
            f"等待机器人运动完成超时：arm_motion={snap.arm_motion}, leg_motion={snap.leg_motion}"
        )

    def wait_left_tcp(self, target: tuple[float, ...], timeout: float = 45.0) -> None:
        """按左 TCP 实际位姿确认笛卡尔命令到位，避免旧 goal_reached 秒过。"""
        deadline = time.monotonic() + timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            current = self.pose(self.backend.snapshot().left_ee)
            pos_err = math.sqrt(sum((current[i] - target[i]) ** 2 for i in range(3)))
            rot_err = max(abs(current[i] - target[i]) for i in range(3, 6))
            if pos_err <= 0.01 and rot_err <= math.radians(1.0): return
            now = time.monotonic()
            if now - last_log >= 1.0:
                self.log(f"等待左 TCP 到位：位置误差={pos_err * 1000:.1f}mm，姿态误差={math.degrees(rot_err):.2f}deg")
                last_log = now
            time.sleep(1.0)
        current = self.pose(self.backend.snapshot().left_ee)
        raise TimeoutError(f"左 TCP 到位超时：目标={target[:3]}，当前={current[:3]}")

    def wait_joints(self, targets: dict[str, float], timeout: float = 45.0) -> None:
        """逐秒读取关节状态，所有目标关节误差在 ±1 度内才算到位。"""
        deadline = time.monotonic() + timeout
        last_log = 0.0
        tolerance = math.radians(1.0)
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            state = self.backend.snapshot().joint_state
            missing = [name for name in targets if name not in state]
            if not missing:
                errors = {name: abs(float(state[name]) - target) for name, target in targets.items()}
                max_error = max(errors.values(), default=0.0)
                if max_error <= tolerance: return
                now = time.monotonic()
                if now - last_log >= 1.0:
                    self.log(f"等待关节到位：最大角度误差={math.degrees(max_error):.2f}度")
                    last_log = now
            else:
                self.log(f"等待关节状态：缺少 {','.join(missing)}")
            time.sleep(1.0)
        state = self.backend.snapshot().joint_state
        max_error = max((abs(float(state.get(n, 0.0)) - t) for n, t in targets.items() if n in state), default=float('inf'))
        raise TimeoutError(f"关节到位超时：最大角度误差={math.degrees(max_error):.2f}度")

    @staticmethod
    def pose(p: Any) -> tuple[float, ...]:
        if p is None: raise RuntimeError("没有 TCP 状态")
        # 抓取姿态使用 RPY；当前网页流程保持末端姿态不变。
        q = p.orientation; x,y,z,w = q.x,q.y,q.z,q.w
        roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = math.asin(max(-1.0,min(1.0,2*(w*y-z*x))))
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return (float(p.position.x),float(p.position.y),float(p.position.z),roll,pitch,yaw)

    def run(self, item: dict[str, Any], target_provider: Callable[[str], tuple[float,float,float]], max_step: int = 11) -> None:
        """执行：复位→层高→识别→预抓取→前进→闭合→抬高→后退→腰部复位→机械臂复位。"""
        max_step = max(1, min(10, int(max_step)))
        self.log(f"[1/11] 开始任务：{item['name']}")
        snap = self.backend.snapshot()
        if len(snap.joint_state) == 0: raise RuntimeError("尚未收到关节状态")
        # 根据货架层选择对应的机械臂“复位/抓取准备”姿态。
        # 第一层使用 RESET；第二层直接使用 main.py 中的 SECOND_LAYER 姿态。
        layer = str(item.get("shelf_layer", "FL")).upper()
        arm_left, arm_right = (SECOND_L, SECOND_R) if layer == "SL" else (RESET_L, RESET_R)
        self.backend.send("arm_joint", left=arm_left, right=arm_right, vel=.70, acc=.50)
        self.backend.send("leg_joint", values=tuple(snap.joint_state.get(n,0.0) if n != 'hip_yaw_joint' else 0.0 for n in LEG_JOINTS), vel=.08, acc=.20)
        reset_targets = {f"ljoint{i}": arm_left[i - 1] for i in range(1, 8)}
        reset_targets.update({f"rjoint{i}": arm_right[i - 1] for i in range(1, 8)})
        reset_targets.update({n: (float(snap.joint_state.get(n, 0.0)) if n != "hip_yaw_joint" else 0.0) for n in LEG_JOINTS})
        self.wait_joints(reset_targets); self.log("[2/11] 复位完成（关节误差≤1度）")
        if max_step == 2:
            self.log("测试模式：已到第 2 步，流程暂停；机器人保持当前姿态")
            return
        #读取层数
        self.backend.send("leg_joint", values=LAYER_SL if layer == "SL" else LAYER_FL, vel=.12, acc=.20)
        layer_targets = {n: (LAYER_SL if layer == "SL" else LAYER_FL)[i] for i, n in enumerate(LEG_JOINTS)}
        self.wait_joints(layer_targets)
        if layer == "SL":
            self.log("[3/11] 第二层高度完成，双臂保持第二层抓取姿态")
        else:
            self.log("[3/11] 第一层高度完成，双臂保持第一层复位姿态")
        if max_step == 3:
            self.log("测试模式：已到第 3 步，流程暂停；机器人保持当前姿态")
            return
        #识别物品的坐标 
        gx,gy,gz = target_provider(str(item["label"]).lower()); self.log(f"[4/11] 视觉目标 {gx:.3f},{gy:.3f},{gz:.3f}")
        if max_step == 4:
            self.log("测试模式：已到第 4 步，流程暂停；机器人保持当前姿态")
            return
        snap = self.backend.snapshot(); cur = self.pose(snap.left_ee)
        dx,dy = gx-cur[0], gy-cur[1]; norm = math.hypot(dx,dy)
        if norm < 1e-4: raise RuntimeError("TCP 与目标水平距离过小")
        ux,uy=dx/norm,dy/norm
        # 与 main.py 的 _make_grasp_targets 一致：预抓取点距目标 8cm。
        # 同时复现 main.py 的姿态生成：TCP 局部 +X 指向目标，+Z 保持竖直。
        x_axis = (ux, uy, 0.0)
        y_axis = (-uy, ux, 0.0)
        z_axis = (0.0, 0.0, 1.0)
        roll = math.atan2(0.0, 1.0)
        pitch = math.atan2(-x_axis[2], math.hypot(x_axis[0], x_axis[1]))
        yaw = math.atan2(x_axis[1], x_axis[0])
        # 与 main.py 同步：预抓取点沿 car_link 全局 Y 轴增加 2 cm。
        pre=(gx-.08*ux,gy-.08*uy+0.02,gz,roll,pitch,yaw)
        right = self.pose(snap.right_ee)
        self.backend.send("arm_cartesian", left=pre, right=right, vel=.70, acc=.10); self.wait_left_tcp(pre); self.log("[5/10] 预抓取姿态到达（TCP误差≤1cm，姿态≤1度）")
        if max_step == 5:
            self.log("测试模式：已到第 5 步，流程暂停；机器人保持当前姿态")
            return
        # 与 main.py 的 _build_relative_forward_target 一致：从当前 TCP 沿局部 +X 相对前进10cm。
        current = self.pose(self.backend.snapshot().left_ee)
        roll, pitch, yaw = current[3:]
        x_axis = (
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        )
        contact = tuple(current[i] + (0.10 * x_axis[i] if i < 3 else 0.0) for i in range(6))
        self.backend.send("arm_cartesian", left=contact, right=right, vel=.70, acc=.10); self.wait_left_tcp(contact); self.log("[6/10] 沿当前 TCP 局部 +X 前进 10 cm 完成（TCP误差≤1cm，姿态≤1度）")
        if max_step == 6:
            self.log("测试模式：已到第 6 步，流程暂停；机器人保持当前姿态")
            return
        self.gripper("close"); self.log("[7/11] 左夹爪闭合"); time.sleep(3)
        if max_step == 7:
            self.log("测试模式：已到第 7 步，流程暂停；机器人保持当前姿态")
            return
        lift=(contact[0],contact[1],contact[2]+.05,*contact[3:]); self.backend.send("arm_cartesian", left=lift, right=right, vel=.70, acc=.10); self.wait_left_tcp(lift); self.log("[8/10] 抬高 5 cm 完成")
        if max_step == 8:
            self.log("测试模式：已到第 8 步，流程暂停；机器人保持当前姿态")
            return
        back=(lift[0]-.15*ux,lift[1]-.15*uy,lift[2],*lift[3:]); self.backend.send("arm_cartesian", left=back, right=right, vel=.70, acc=.10); self.wait_left_tcp(back); self.log("[9/10] 后退 15 cm 完成")
        if max_step == 9:
            self.log("测试模式：已到第 9 步，流程暂停；机器人保持当前姿态")
            return
        waist_state = self.backend.snapshot().joint_state
        waist_targets = {n: float(waist_state[n]) for n in LEG_JOINTS if n in waist_state}
        if len(waist_targets) != 4: raise RuntimeError("腿部四个关节状态不完整，无法复位腰部")
        waist_targets["hip_yaw_joint"] = 0.0
        self.backend.send("leg_joint", values=tuple(waist_targets[n] for n in LEG_JOINTS), vel=1.2, acc=.20)
        self.wait_joints(waist_targets)
        self.log("[10/10] 腰部复位完成（仅 hip_yaw_joint 归零，速度=1.0）")
        self.backend.send("arm_joint", left=arm_left, right=arm_right, vel=.70, acc=.50)
        final_targets = {f"ljoint{i}": arm_left[i - 1] for i in range(1, 8)}
        final_targets.update({f"rjoint{i}": arm_right[i - 1] for i in range(1, 8)})
        self.wait_joints(final_targets)
        self.log("[10/10] 机械臂复位完成，抓取流程结束")
        if max_step == 10:
            self.log("测试模式：已到第 10 步，流程暂停；机器人保持当前姿态")
            return

class DirectRightGrasp(DirectLeftGrasp):
    """右手流程：普通标签沿 TCP +X，guazi/pai/cui 按 car_link +Y 侧向抓取。"""
    SPECIAL = {"guazi", "pai", "cui"}

    def _wait_right_tcp(self, target: tuple[float, ...], timeout: float = 45.0) -> None:
        deadline = time.monotonic() + timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            cur = self.pose(self.backend.snapshot().right_ee)
            pos = math.sqrt(sum((cur[i] - target[i]) ** 2 for i in range(3)))
            rot = max(abs(cur[i] - target[i]) for i in range(3, 6))
            if pos <= .01 and rot <= math.radians(1.0): return
            now = time.monotonic()
            if now - last_log >= 1.0:
                self.log(f"等待右 TCP 到位：位置误差={pos * 1000:.1f}mm，姿态误差={math.degrees(rot):.2f}deg")
                last_log = now
            time.sleep(1.0)
        cur = self.pose(self.backend.snapshot().right_ee)
        raise TimeoutError(
            f"右 TCP 到位超时：目标={target[:3]}，当前={cur[:3]}，"
            f"位置误差={math.sqrt(sum((cur[i] - target[i]) ** 2 for i in range(3))) * 1000:.1f}mm"
        )

    def run(self, item: dict[str, Any], target_provider: Callable[[str], tuple[float,float,float]], max_step: int = 10) -> None:
        max_step = max(1, min(10, int(max_step))); label = str(item["label"]).lower(); special = label in self.SPECIAL
        self.log(f"[1/10] 开始右手任务：{item['name']}（{'侧向' if special else '普通'}抓取）")
        snap = self.backend.snapshot()
        if not snap.joint_state: raise RuntimeError("尚未收到关节状态")
        layer = str(item.get("shelf_layer", "FL")).upper(); arm_left, arm_right = (SECOND_L, SECOND_R) if layer == "SL" else (RESET_L, RESET_R)
        self.backend.send("arm_joint", left=arm_left, right=arm_right, vel=.70, acc=.50)
        self.backend.send("leg_joint", values=tuple(snap.joint_state.get(n, 0.0) if n != "hip_yaw_joint" else 0.0 for n in LEG_JOINTS), vel=.08, acc=.20)
        targets = {f"ljoint{i}": arm_left[i-1] for i in range(1,8)} | {f"rjoint{i}": arm_right[i-1] for i in range(1,8)}
        targets.update({n: float(snap.joint_state.get(n, 0.0)) if n != "hip_yaw_joint" else 0.0 for n in LEG_JOINTS}); self.wait_joints(targets); self.log("[2/10] 复位完成（关节误差≤1度）")
        if max_step == 2: self.log("测试模式：已到第 2 步，流程暂停"); return
        values = LAYER_SL if layer == "SL" else LAYER_FL; self.backend.send("leg_joint", values=values, vel=.12, acc=.20); self.wait_joints({n: values[i] for i,n in enumerate(LEG_JOINTS)}); self.log(f"[3/10] {'第二层' if layer == 'SL' else '第一层'}高度完成")
        if max_step == 3: self.log("测试模式：已到第 3 步，流程暂停"); return
        gx,gy,gz = target_provider(label); self.log(f"[4/10] 视觉目标 {gx:.3f},{gy:.3f},{gz:.3f}")
        if max_step == 4: self.log("测试模式：已到第 4 步，流程暂停"); return
        snap = self.backend.snapshot(); cur = self.pose(snap.right_ee)
        if special:
            # 与 main.py 的实际“生成预抓取姿态”按钮一致：
            # 先按当前 TCP 到目标的水平方向确定局部 +X，再叠加右手
            # 全局 Y 偏移和标签对应的 TCP 局部 -Y 偏移。
            side_offset = RIGHT_HAND_TCP_Y_OFFSETS_M[label]
            dx, dy = gx - cur[0], gy - cur[1]
            norm = math.hypot(dx, dy)
            if norm < 1e-4: raise RuntimeError("右 TCP 与目标水平距离过小")
            ux, uy = dx / norm, dy / norm
            # 特殊商品预抓取距离按当前标定要求为 5 cm，右手通用全局 Y 偏移为 -3 cm。
            yaw = math.atan2(uy, ux)
            pre_distance = SPECIAL_RIGHT_PREGRASP_DISTANCE_M
            pre_x = gx - pre_distance * ux + side_offset * (-uy)
            pre_y = gy - pre_distance * uy + RIGHT_HAND_PREGRASP_Y_OFFSET_M + side_offset * ux
            pre = (pre_x, pre_y, gz, 0.0, 0.0, yaw)
        else:
            dx,dy=gx-cur[0],gy-cur[1]; norm=math.hypot(dx,dy)
            if norm < 1e-4: raise RuntimeError("右 TCP 与目标水平距离过小")
            ux,uy=dx/norm,dy/norm; pre=(gx-RIGHT_HAND_NORMAL_PREGRASP_DISTANCE_M*ux, gy-RIGHT_HAND_NORMAL_PREGRASP_DISTANCE_M*uy+RIGHT_HAND_NORMAL_PREGRASP_Y_OFFSET_M, gz, 0.0, 0.0, math.atan2(uy,ux))
        left=self.pose(snap.left_ee); self.backend.send("arm_cartesian", left=left, right=pre, vel=.70, acc=.10); self._wait_right_tcp(pre); self.log("[5/10] 右手预抓取姿态到达")
        if max_step == 5: self.log("测试模式：已到第 5 步，流程暂停"); return
        cur=self.pose(self.backend.snapshot().right_ee)
        if special:
            # 与 main.py 的“沿当前 TCP +X 前进”按钮一致。
            roll, pitch, yaw = cur[3:]
            xa = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
            contact = tuple(cur[i] + (SPECIAL_RIGHT_FORWARD_DISTANCE_M * xa[i] if i < 3 else 0.0) for i in range(6))
        else:
            roll,pitch,yaw=cur[3:]; xa=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),-math.sin(pitch)); contact=tuple(cur[i]+(RIGHT_HAND_NORMAL_FORWARD_DISTANCE_M*xa[i] if i<3 else 0.0) for i in range(6))
        self.backend.send("arm_cartesian", left=left, right=contact, vel=.70, acc=.10); self._wait_right_tcp(contact); self.log(f"[6/10] 右手前进 {(SPECIAL_RIGHT_FORWARD_DISTANCE_M if special else RIGHT_HAND_NORMAL_FORWARD_DISTANCE_M) * 100:.0f} cm 完成")
        if max_step == 6: self.log("测试模式：已到第 6 步，流程暂停"); return
        self.gripper("close"); time.sleep(3); self.log("[7/10] 右灵巧手闭合完成")
        if max_step == 7: self.log("测试模式：已到第 7 步，流程暂停"); return
        lift=(contact[0],contact[1],contact[2]+.05,*contact[3:]); self.backend.send("arm_cartesian", left=left, right=lift, vel=.70, acc=.10); self._wait_right_tcp(lift); self.log("[8/10] 右手抬高 5 cm 完成")
        if max_step == 8: self.log("测试模式：已到第 8 步，流程暂停"); return
        back=(lift[0]-.15*math.cos(cur[5]),lift[1]-.15*math.sin(cur[5]),lift[2],*lift[3:]); self.backend.send("arm_cartesian", left=left, right=back, vel=.70, acc=.10); self._wait_right_tcp(back); self.log("[9/10] 右手后退 15 cm 完成")
        if max_step == 9: self.log("测试模式：已到第 9 步，流程暂停"); return
        waist=self.backend.snapshot().joint_state; wt={n:float(waist[n]) for n in LEG_JOINTS if n in waist};
        if len(wt)!=4: raise RuntimeError("腿部四个关节状态不完整")
        wt["hip_yaw_joint"]=0.0; self.backend.send("leg_joint", values=tuple(wt[n] for n in LEG_JOINTS), vel=1.0, acc=.20); self.wait_joints(wt)
        self.backend.send("arm_joint", left=arm_left, right=arm_right, vel=.70, acc=.50); self.wait_joints({f"ljoint{i}":arm_left[i-1] for i in range(1,8)} | {f"rjoint{i}":arm_right[i-1] for i in range(1,8)}); self.log("[10/10] 腰部复位并完成右臂复位，任务结束")


class BoxGrasp(DirectLeftGrasp):
    """网页箱子双手抓取流程；放置动作暂留在网页接口中。"""
    BOX_MODEL = PROJECT_ROOT / "weight" / "best_xiangzi.pt"
    READY_L = tuple(math.radians(x) for x in (15.8, 22.5, -24.7, -106.5, -27.0, -13.5, 5.6))
    READY_R = tuple(math.radians(x) for x in (-15.6, -18.7, 19.1, 110.0, 25.0, 14.0, -14.7))
    READY2_L = tuple(math.radians(x) for x in (8.6, 28.6, -29.2, -110.0, -35.6, -14.2, 22.2))
    # 第二层右臂第 4 关节的机械/控制实际可达位置约为 110°。
    # 原来的 117.2° 会导致该关节长期保留约 7.2° 误差，BoxGrasp
    # 在 wait_joints() 中一直无法通过；第一层对应值本来就是 110°。
    READY2_R = tuple(math.radians(x) for x in (-14.2, -20.6, 29.6, 110.2, 33.9, 14.2, -16.7))
    def __init__(self, backend, log, left_gripper, right_gripper, right_pose):
        super().__init__(backend, log, left_gripper); self.right_gripper = right_gripper; self.right_pose = right_pose
    def run(self, layer: str, target_provider: Callable, max_step: int = 8) -> None:
        max_step = max(1, min(8, int(max_step))); layer = str(layer).upper()
        self.cancelled = False; self.log(f"[BOX] 开始{('第二' if layer == 'SL' else '第一')}层箱子抓取")
        self.gripper("open"); self.right_pose(BOX_HAND_POSE); self.log("[BOX 1/8] 左手打开，右手设置灵巧手姿态");
        if max_step == 1: return
        snap = self.backend.snapshot(); left_j, right_j = (self.READY2_L, self.READY2_R) if layer == "SL" else (self.READY_L, self.READY_R)
        self.backend.send("arm_joint", left=left_j, right=right_j, vel=.40, acc=.10); self.wait_joints({f"ljoint{i}": left_j[i-1] for i in range(1,8)} | {f"rjoint{i}": right_j[i-1] for i in range(1,8)}); self.log("[BOX 2/8] 箱子初始姿态完成")
        if max_step == 2: return
        if layer == "SL":
            values = LAYER_SL
            self.backend.send("leg_joint", values=values, vel=.12, acc=.20)
            self.wait_joints({n: values[i] for i, n in enumerate(LEG_JOINTS)})
            self.log("[BOX 3/8] 第二层升降机高度完成")
        else:
            # 第一层箱子抓取要求实际腿部末端高度为 0.400 m。
            # 使用腿部笛卡尔升降保持当前姿态，只补偿高度，不再直接使用
            # LAYER_FL（其正运动学高度约为 0.459 m）。
            pose = self.backend.snapshot().leg_pose()
            if not pose:
                raise RuntimeError("当前腿部状态还没读全，无法调整第一层高度")
            delta = 0.400 - float(pose["y"])
            if abs(delta) > 1e-3:
                self.backend.send("leg_lift", delta=delta, waist_delta=0.0, vel=.12)
                self.wait(arm=False, leg=True, timeout=45.0)
            final_pose = self.backend.snapshot().leg_pose()
            actual_y = None if not final_pose else float(final_pose["y"])
            self.log(f"[BOX 3/8] 第一层高度完成：目标 0.400m，实际 {actual_y:.3f}m" if actual_y is not None else "[BOX 3/8] 第一层高度完成")
        if max_step == 3: return
        gx, gy, gz = target_provider("xiangzi", self.BOX_MODEL); self.log(f"[BOX 4/8] 箱体识别位置 {gx:.3f},{gy:.3f},{gz:.3f}")
        if max_step == 4: return
        left = self.pose(self.backend.snapshot().left_ee); right = self.pose(self.backend.snapshot().right_ee)
        vision = getattr(target_provider, "__self__", None)
        points_car = getattr(vision, "last_points_car", None)
        camera_pos_car = getattr(vision, "last_camera_pos_car", None)
        if points_car is None or camera_pos_car is None:
            raise RuntimeError("未获取箱体点云，无法按 grasp box 算法生成预抓取姿态")
        front, normal, _ = estimate_front_plane(points_car, preferred_face_span_m=BOX_LENGTH_M, preferred_height_m=BOX_HEIGHT_M, distance_threshold=BOX_FRONT_PLANE_THRESHOLD_M)
        T, debug = estimate_box_frame(front, normal, camera_pos_car, box_length_m=BOX_LENGTH_M, box_width_m=BOX_WIDTH_M, box_height_m=BOX_HEIGHT_M)
        candidates = generate_side_grasp_candidates(T, face_span_m=float(debug["face_span_m"]), box_depth_m=float(debug["box_depth_m"]), pregrasp_dist_m=BOX_LEFT_PREGRASP_DISTANCE_M, grasp_z_offset_m=BOX_LEFT_GRASP_Z_OFFSET_M, grasp_x_offset_m=BOX_LEFT_PREGRASP_X_OFFSET_M, front_center_base=debug["front_center_base"], box_center_base=debug["box_center_base"])
        right_candidate = generate_side_grasp_candidates(T, face_span_m=float(debug["face_span_m"]), box_depth_m=float(debug["box_depth_m"]), pregrasp_dist_m=BOX_RIGHT_PREGRASP_DISTANCE_M, grasp_z_offset_m=BOX_RIGHT_GRASP_Z_OFFSET_M, grasp_x_offset_m=BOX_RIGHT_PREGRASP_X_OFFSET_M, front_center_base=debug["front_center_base"], box_center_base=debug["box_center_base"])["right"]
        left_pre = tuple(float(v) for v in candidates["left"]["pose_pregrasp_base"])
        right_pre = tuple(float(v) for v in right_candidate["pose_pregrasp_base"])
        # 双手前往预抓取位置：降低速度和加速度，避免接近箱体时动作过快。
        self.backend.send("arm_cartesian", left=left_pre, right=right_pre, vel=.30, acc=.08); self.wait_left_tcp(left_pre); self._wait_right_tcp_box(right_pre); self.log("[BOX 5/8] 双手预抓取完成")
        if max_step == 5: return
        # 与主界面“右手前进”一致：从右手预抓取位姿只沿基座 X 轴前进，
        # 保持 Y/Z 和末端姿态不变，避免误沿箱体侧向 Y 轴移动。
        right_final = (right_pre[0] + .10, right_pre[1], right_pre[2], *right_pre[3:])
        self.backend.send("arm_cartesian", left=left_pre, right=right_final, vel=.70, acc=.10); self._wait_right_tcp_box(right_final); self.log("[BOX 6/8] 右手沿 X 轴单独前进 10cm 完成")
        if max_step == 6: return
        self.gripper("close"); self.right_gripper("close"); self.log("[BOX 7/8] 左右手同步闭合完成")
        if max_step == 7: return
        if layer == "SL":
            left_now = self.pose(self.backend.snapshot().left_ee)
            right_now = self.pose(self.backend.snapshot().right_ee)
            left_up = (left_now[0], left_now[1], left_now[2] + BOX_ARM_Z_STEP_M, *left_now[3:])
            right_up = (right_now[0], right_now[1], right_now[2] + BOX_ARM_Z_STEP_M, *right_now[3:])
            self.backend.send("arm_cartesian", left=left_up, right=right_up, vel=.70, acc=.10)
            self.wait_left_tcp(left_up); self._wait_right_tcp_box(right_up)
            self.log("[BOX 8/8] 第二层双手基座 Z 轴上移 5cm 完成")
        else:
            if not self.backend.snapshot().leg_pose():
                raise RuntimeError("当前腰部状态还没读全，无法执行第一层腰部上升")
            self.backend.send("leg_lift", delta=BOX_LIFT_DISTANCE_M, waist_delta=0.0, vel=.08)
            self.wait(arm=False, leg=True)
            self.log("[BOX 8/8] 第一层腰部上升 10cm 完成")
    def place_hand(self, hand: str) -> None:
        """按指定末端执行器执行固定坐标箱子放置流程。"""
        hand = str(hand).lower()
        if hand not in {"left", "right"}:
            raise ValueError("放置手臂必须是 left 或 right")
        self.cancelled = False
        self.log(f"[BOX PLACE {hand}] 开始固定坐标放置流程")
        snap = self.backend.snapshot()
        if not snap.joint_state:
            raise RuntimeError("尚未收到关节状态")

        # 1. 双臂复位到 SECOND 姿态。
        self.backend.send("arm_joint", left=SECOND_L, right=SECOND_R, vel=.70, acc=.50)
        self.wait_joints(
            {f"ljoint{i}": SECOND_L[i - 1] for i in range(1, 8)}
            | {f"rjoint{i}": SECOND_R[i - 1] for i in range(1, 8)}
        )
        self.log(f"[BOX PLACE {hand}] 双臂 SECOND 姿态完成")

        # 2. 腰部到 0.65 m：28.1, -57.1, 29.0, 0 度。
        self.backend.send("leg_joint", values=PLACE_LEG_065, vel=.12, acc=.20)
        self.wait_joints({n: PLACE_LEG_065[i] for i, n in enumerate(LEG_JOINTS)})
        self.log(f"[BOX PLACE {hand}] 腰部 0.65m 高度完成")

        # 3. 指定 TCP 到 car_link 固定坐标，保留复位后的末端姿态。
        current = self.backend.snapshot()
        left_now = self.pose(current.left_ee)
        right_now = self.pose(current.right_ee)
        target = (*PLACE_XYZ, *(left_now[3:] if hand == "left" else right_now[3:]))
        if hand == "left":
            self.backend.send("arm_cartesian", left=target, right=right_now, vel=.30, acc=.08)
            self.wait_left_tcp(target)
        else:
            self.backend.send("arm_cartesian", left=left_now, right=target, vel=.30, acc=.08)
            self._wait_right_tcp_box(target)
        self.log(f"[BOX PLACE {hand}] TCP 已到 car_link (0.40, 0.00, 1.10)")

        # 4. 指定 TCP 沿基座 Z 轴下降 10 cm。
        drop = (target[0], target[1], target[2] - PLACE_Z_DROP_M, *target[3:])
        if hand == "left":
            self.backend.send("arm_cartesian", left=drop, right=right_now, vel=.30, acc=.08)
            self.wait_left_tcp(drop)
        else:
            self.backend.send("arm_cartesian", left=left_now, right=drop, vel=.30, acc=.08)
            self._wait_right_tcp_box(drop)
        self.log(f"[BOX PLACE {hand}] 指定末端沿基座 Z 轴下降 10cm")

        # 5. 只打开被点击按钮对应的末端执行器。
        if hand == "left":
            self.gripper("open")
        else:
            self.right_gripper("open")
        self.log(f"[BOX PLACE {hand}] {'左夹爪' if hand == 'left' else '右灵巧手'} 已打开")

        # 6. 腰部保持 0.65 m，第四个关节 hip_yaw_joint 为 0。
        self.backend.send("leg_joint", values=PLACE_LEG_065, vel=.12, acc=.20)
        self.wait_joints({n: PLACE_LEG_065[i] for i, n in enumerate(LEG_JOINTS)})
        self.log(f"[BOX PLACE {hand}] 腰部保持 0.65m，第四关节归零")

        # 7. 双臂复位到 SECOND 姿态。
        self.backend.send("arm_joint", left=SECOND_L, right=SECOND_R, vel=.70, acc=.50)
        self.wait_joints(
            {f"ljoint{i}": SECOND_L[i - 1] for i in range(1, 8)}
            | {f"rjoint{i}": SECOND_R[i - 1] for i in range(1, 8)}
        )
        self.log(f"[BOX PLACE {hand}] 双臂复位完成")

    def _wait_right_tcp_box(self, target, timeout=45.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            cur = self.pose(self.backend.snapshot().right_ee)
            if math.sqrt(sum((cur[i]-target[i])**2 for i in range(3))) <= .01: return
            time.sleep(.2)
        raise TimeoutError(f"右 TCP 到位超时：目标={target[:3]}")

    def place(self, layer: str, max_step: int = 8) -> None:
        """放置箱子。

        第一层：双手直接下降 6 cm。
        第二层：双手先下降 5 cm，腰部调整到 0.400 m，再双手下降 6.2 cm。
        ``max_step`` 用于网页调试时在各阶段后暂停。
        """
        max_step = max(1, min(8, int(max_step)))
        layer = str(layer).upper()
        self.cancelled = False
        if layer not in {"FL", "SL"}:
            raise ValueError(f"不支持的箱子层：{layer}")
        snap = self.backend.snapshot()
        if snap.left_ee is None or snap.right_ee is None:
            raise RuntimeError("当前末端姿态未就绪，无法执行箱子放置")

        def drop_both(distance_m: float, label: str) -> None:
            current = self.backend.snapshot()
            left_now = self.pose(current.left_ee)
            right_now = self.pose(current.right_ee)
            left_drop = (left_now[0], left_now[1], left_now[2] - distance_m, *left_now[3:])
            right_drop = (right_now[0], right_now[1], right_now[2] - distance_m, *right_now[3:])
            self.backend.send("arm_cartesian", left=left_drop, right=right_drop, vel=0.05, acc=0.10)
            self.wait_left_tcp(left_drop, timeout=45.0)
            self._wait_right_tcp_box(right_drop, timeout=45.0)
            self.log(f"[BOX PLACE] {label}：双手同步下降 {distance_m:.3f}m 完成")

        def open_both() -> None:
            """下降确认完成后释放箱子：左夹爪打开，右手回到箱子张开姿态。"""
            self.gripper("open")
            # grasp box 页“灵巧手姿势”对应的张开姿态，不使用通用 preset，
            # 以确保右手发送 [100, 0, 255, 255, 255, 255, 255]。
            self.right_pose(BOX_HAND_POSE)
            self.log("[BOX PLACE] 左右手已打开，右手姿态=100 0 255 255 255 255 255")

        def lower_right_after_open() -> None:
            """确认双手释放后，右臂 TCP 再沿基座 Z 轴下降 1 cm。"""
            current = self.backend.snapshot()
            left_now = self.pose(current.left_ee)
            right_now = self.pose(current.right_ee)
            right_lower = (right_now[0], right_now[1], right_now[2] - 0.010, *right_now[3:])
            # 左臂保持当前位置，右臂单独下降；等待右 TCP 到位后才结束放置流程。
            self.backend.send("arm_cartesian", left=left_now, right=right_lower, vel=0.05, acc=0.08)
            self._wait_right_tcp_box(right_lower, timeout=45.0)
            self.log("[BOX PLACE] 双手打开完成，右臂末端沿 Z 轴下降 1cm 完成")

        if layer == "FL":
            drop_both(BOX_PLACE_DROP_FIRST_LAYER_M, "第一层")
            open_both()
            lower_right_after_open()
            return

        # 第二层第一阶段：先把箱子整体向下放 5 cm。
        drop_both(0.050, "第二层第 1 步")
        if max_step == 1:
            return

        # 第二层第二阶段：再将腰部/升降机构调整到 0.400 m。
        pose = self.backend.snapshot().leg_pose()
        if not pose:
            raise RuntimeError("当前腿部状态还没读全，无法执行第二层放箱子")
        current_y = float(pose["y"])
        delta = 0.400 - current_y
        if abs(delta) > 1e-3:
            self.backend.send("leg_lift", delta=delta, waist_delta=0.0, vel=0.12)
            self.wait(arm=False, leg=True, timeout=45.0)
        self.log("[BOX PLACE] 第二层第 2 步：腰部已到位 0.400m")
        if max_step == 2:
            return

        # 第二层第三阶段：腰部到位后，双手再下降 6.2 cm。
        drop_both(BOX_PLACE_DROP_SECOND_LAYER_M, "第二层第 3 步")
        open_both()
        lower_right_after_open()
