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

RESET_L = tuple(math.radians(x) for x in (17.7, 22.5, -6.6, -106.8, -18.6, -17.1, 0.0))
RESET_R = tuple(math.radians(x) for x in (-18.8, -29.5, 9.6, 103.0, 23.9, 19.3, -4.0))
SECOND_L = tuple(math.radians(x) for x in (13.0, 22.2, -27.0, -110.0, -35.2, -19.2, 18.2))
SECOND_R = tuple(math.radians(x) for x in (-16.7, -29.5, 23.3, 109.8, 34.4, 24.9, -3.2))
LAYER_FL = tuple(math.radians(x) for x in (50.6, -103.1, 52.6, 0.0))
LAYER_SL = (0.0, 0.0, 0.0, 0.0)

class RetailVision:
    """迁移自旧控制台的 YOLO-seg + 深度/外参目标点计算。"""
    def __init__(self, state: Any, backend: RobotBackend, log: Callable[[str], None]):
        self.state, self.backend, self.log = state, backend, log
        self.model = None
        self.extrinsic = None
        path = PROJECT_ROOT / "config" / "realsense_347522072040_extrin.yaml"
        if path.exists():
            import yaml, numpy as np
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            matrix = np.asarray(data.get("extrinsic_matrix", []), dtype=float)
            if matrix.shape == (4, 4): self.extrinsic = matrix

    def detect(self, label: str) -> tuple[float, float, float]:
        import cv2, numpy as np
        image, depth, intr = self.state._camera_color_image, self.state._camera_depth_m, self.state._camera_intrinsics
        if image is None or depth is None or intr is None: raise RuntimeError("相机尚未就绪")
        from ultralytics import YOLO
        model_path = PROJECT_ROOT / "weight" / "best.pt"
        if not model_path.exists(): raise RuntimeError(f"视觉模型不存在: {model_path}")
        if self.model is None: self.model = YOLO(str(model_path)); self.log(f"[VISION] 已加载模型: {model_path}")
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
            if tf is not None: pts=(np.asarray(tf) @ np.column_stack((pts,np.ones(len(pts)))).T).T[:,:3]
        # 与旧 _generate_bottle_grasp_pose 一致：Z 取 2%~98% 范围的中上部(45%)，XY 取中位数。
        z_min, z_max = float(np.percentile(pts[:,2], 2)), float(np.percentile(pts[:,2], 98))
        if z_max - z_min < .05: raise RuntimeError(f"估计目标高度异常: {z_max-z_min:.3f} m")
        p=np.array([float(np.median(pts[:,0])), float(np.median(pts[:,1])), z_min + .45*(z_max-z_min)])
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
        while time.monotonic() < deadline:
            if self.cancelled: raise RuntimeError("任务已停止")
            cur = self.pose(self.backend.snapshot().right_ee)
            pos = math.sqrt(sum((cur[i] - target[i]) ** 2 for i in range(3)))
            rot = max(abs(cur[i] - target[i]) for i in range(3, 6))
            if pos <= .01 and rot <= math.radians(1.0): return
            time.sleep(1.0)
        raise TimeoutError(f"右 TCP 到位超时：目标={target[:3]}")

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
        if special: pre = (gx, gy - .10, gz, 0.0, 0.0, 0.0)
        else:
            dx,dy=gx-cur[0],gy-cur[1]; norm=math.hypot(dx,dy)
            if norm < 1e-4: raise RuntimeError("右 TCP 与目标水平距离过小")
            ux,uy=dx/norm,dy/norm; pre=(gx-.08*ux, gy-.08*uy-.02, gz, 0.0, 0.0, math.atan2(uy,ux))
        left=self.pose(snap.left_ee); self.backend.send("arm_cartesian", left=left, right=pre, vel=.70, acc=.10); self._wait_right_tcp(pre); self.log("[5/10] 右手预抓取姿态到达")
        if max_step == 5: self.log("测试模式：已到第 5 步，流程暂停"); return
        cur=self.pose(self.backend.snapshot().right_ee)
        if special: contact=(cur[0],cur[1]+.10,cur[2],*cur[3:])
        else:
            roll,pitch,yaw=cur[3:]; xa=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),-math.sin(pitch)); contact=tuple(cur[i]+(.10*xa[i] if i<3 else 0.0) for i in range(6))
        self.backend.send("arm_cartesian", left=left, right=contact, vel=.70, acc=.10); self._wait_right_tcp(contact); self.log("[6/10] 右手前进 10 cm 完成")
        if max_step == 6: self.log("测试模式：已到第 6 步，流程暂停"); return
        self.gripper("close"); time.sleep(3); self.log("[7/10] 右灵巧手闭合完成")
        if max_step == 7: self.log("测试模式：已到第 7 步，流程暂停"); return
        lift=(contact[0],contact[1],contact[2]+.05,*contact[3:]); self.backend.send("arm_cartesian", left=left, right=lift, vel=.70, acc=.10); self._wait_right_tcp(lift); self.log("[8/10] 右手抬高 5 cm 完成")
        if max_step == 8: self.log("测试模式：已到第 8 步，流程暂停"); return
        back=(lift[0],lift[1]-.15,lift[2],*lift[3:]) if special else (lift[0]-.15*math.cos(cur[5]),lift[1]-.15*math.sin(cur[5]),lift[2],*lift[3:]); self.backend.send("arm_cartesian", left=left, right=back, vel=.70, acc=.10); self._wait_right_tcp(back); self.log("[9/10] 右手后退 15 cm 完成")
        if max_step == 9: self.log("测试模式：已到第 9 步，流程暂停"); return
        waist=self.backend.snapshot().joint_state; wt={n:float(waist[n]) for n in LEG_JOINTS if n in waist};
        if len(wt)!=4: raise RuntimeError("腿部四个关节状态不完整")
        wt["hip_yaw_joint"]=0.0; self.backend.send("leg_joint", values=tuple(wt[n] for n in LEG_JOINTS), vel=1.0, acc=.20); self.wait_joints(wt)
        self.backend.send("arm_joint", left=arm_left, right=arm_right, vel=.70, acc=.50); self.wait_joints({f"ljoint{i}":arm_left[i-1] for i in range(1,8)} | {f"rjoint{i}":arm_right[i-1] for i in range(1,8)}); self.log("[10/10] 腰部复位并完成右臂复位，任务结束")
