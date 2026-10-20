"""机械臂逆解、路线规划、手指联动和物理执行。位置用米，关节角用弧度。"""

import json
import math
import mujoco
import numpy as np
from scene import ROOT, assets, matrix, pose

CHANNELS = ('thumb_cmc_yaw', 'thumb_cmc_pitch', 'index_mcp_pitch', 'middle_mcp_pitch', 'ring_mcp_pitch', 'pinky_mcp_pitch')

def rotation_error(target, current):
    """计算当前朝向与目标朝向还差多少。"""
    error = np.empty(4)
    mujoco.mju_mulQuat(error, target, np.r_[current[0], -current[1:]])
    if error[0] < 0:
        error *= -1
    length = np.linalg.norm(error[1:])
    return error[1:] * 2 * np.arctan2(length, error[0]) / length if length > 1e-12 else np.zeros(3)

def smooth(t):
    """让动作慢慢起步、慢慢停下；输入和输出都在 0 到 1 之间。"""
    s = float(np.clip(t, 0, 1))
    return s * s * s * (10 + s * (-15 + 6 * s))

class ArmIK:
    """机械臂逆解工具：根据目标手腕姿态调整七个关节角。"""

    def __init__(self, model):
        """准备关节编号、角度范围和计算时要用到的数组。"""
        self.model = model
        self.data = mujoco.MjData(model)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rh_hand_base_link')
        self.joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'lsrobot_arm_joint{i}') for i in range(1, 8)])
        self.qp = model.jnt_qposadr[self.joints]
        self.dof = model.jnt_dofadr[self.joints]
        self.limits = model.jnt_range[self.joints]
        self.jp, self.jr = (np.zeros((3, model.nv)), np.zeros((3, model.nv)))

    def solve(self, target, seed, base=None, iterations=180, tolerance=2e-06):
        """从给定关节角开始反复微调，返回最接近目标的一组角度和剩余误差。"""
        if base is not None:
            self.data.qpos[:] = base
        q = np.clip(seed, self.limits[:, 0] + 0.005, self.limits[:, 1] - 0.005)
        best = (np.inf, q.copy(), np.inf, np.inf)
        for _ in range(iterations):
            self.data.qpos[self.qp] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            # 先算位置和朝向还差多少。
            ep = target[:3] - self.data.xpos[self.body]
            er = rotation_error(target[3:], self.data.xquat[self.body])
            error = np.r_[ep, 0.2 * er]
            score = np.linalg.norm(error)
            if score < best[0]:
                best = (score, q.copy(), float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
            if np.linalg.norm(ep) < tolerance and np.linalg.norm(er) < tolerance:
                break
            # 计算各关节小幅转动后，手腕会怎样移动。
            mujoco.mj_jacBody(self.model, self.data, self.jp, self.jr, self.body)
            jac = np.vstack((self.jp[:, self.dof], 0.2 * self.jr[:, self.dof]))
            damping = 0.001 if score < 0.02 else 0.01
            # 根据当前误差算出这一轮的关节调整量。
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(6), error)
            step *= min(1.0, 0.18 / max(np.max(np.abs(step)), 1e-12))
            # 每轮只走一小步，并把关节角限制在模型范围内。
            q = np.clip(q + step, self.limits[:, 0] + 0.005, self.limits[:, 1] - 0.005)
        self.data.qpos[self.qp] = best[1]
        mujoco.mj_forward(self.model, self.data)
        return (best[1], best[2], best[3])

class MimicSnap:
    """按模型里给出的联动关系，同步两只手的十个从动关节。"""

    def __init__(self, model):
        """读取每组主动关节、从动关节和它们之间的比例。"""
        self.model = model
        self.couplings = []
        for i in range(model.neq):
            if int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            passive, active = (int(model.eq_obj1id[i]), int(model.eq_obj2id[i]))
            if active < 0:
                continue
            qp, qa = (int(model.jnt_qposadr[passive]), int(model.jnt_qposadr[active]))
            vp = int(model.jnt_dofadr[passive])
            self.couplings.append((i, qp, qa, vp, model.eq_data[i, :5].copy(), model.jnt_range[passive].copy()))
            model.eq_active0[i] = 0
        self.eq_ids = np.array([c[0] for c in self.couplings])
        self.passive_qpos = np.array([c[1] for c in self.couplings])
        self.passive_dof = np.array([c[3] for c in self.couplings])

    def targets(self, data):
        """根据主动关节角计算从动关节角，并限制在允许范围内。"""
        values = []
        for _, _, qa, _, p, limits in self.couplings:
            q = data.qpos[qa]
            value = p[0] + q * (p[1] + q * (p[2] + q * (p[3] + q * p[4])))
            values.append(np.clip(value, *limits))
        return np.array(values)

    def apply(self, data):
        """写入从动关节角，并将从动关节速度清零；这是仿真近似处理。"""
        data.eq_active[self.eq_ids] = 0
        data.qpos[self.passive_qpos] = self.targets(data)
        data.qvel[self.passive_dof] = 0
        mujoco.mj_kinematics(self.model, data)




class ForceMimicSnap(MimicSnap):
    """手指转动后同步从动关节，同时保留原有的关节约束作用力。"""

    def __init__(self, model):
        """准备需要联动的手指关节。"""
        super().__init__(model)
        model.eq_active0[self.eq_ids] = 1

    def apply(self, data):
        """同步从动关节角度，并恢复关节约束。"""
        super().apply(data)
        data.eq_active[self.eq_ids] = 1

def interpolate_pose(a, b, f):
    """在两个手腕姿态中间取一个姿态；位置和朝向一起过渡。"""
    out = (1 - f) * a + f * b
    qa = a[3:].copy()
    qb = b[3:].copy()
    if qa @ qb < 0:
        qb = -qb
    out[3:] = (1 - f) * qa + f * qb
    out[3:] /= np.linalg.norm(out[3:])
    return out

class SideIK(ArmIK):
    """给左臂或右臂选择对应的七个关节和手腕。"""

    def __init__(self, m, side):
        """lh 是左手，rh 是右手；名称带 _m 的机械臂关节属于左臂。"""
        super().__init__(m)
        self.body = m.body(side + '_hand_base_link').id
        suffix = '_m' if side == 'lh' else ''
        self.joints = np.array([m.joint(f'lsrobot_arm_joint{i}' + suffix).id for i in range(1, 8)])
        self.qp = m.jnt_qposadr[self.joints]
        self.dof = m.jnt_dofadr[self.joints]
        self.limits = m.jnt_range[self.joints].copy()
        self.limits[5] = [-math.radians(118), math.radians(118)]

class Robot:
    """集中管理场景、机械臂、手指、运动路线和结果保存。"""

    def __init__(self):
        """读取场景和设置，记录各关节、物体以及料框在仿真中的编号。"""
        self.cfg = json.loads((ROOT / 'config.json').read_text())
        self.aa = assets()
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
        self.snap = ForceMimicSnap(self.m)
        self.d = mujoco.MjData(self.m)
        self.kd = mujoco.MjData(self.m)
        m = self.m
        self.iks = {s: SideIK(m, s) for s in ('lh', 'rh')}
        self.hq = {}
        self.hv = {}
        self.ha = {}
        for s in self.iks:
            jj = np.array([m.joint(s + '_' + n).id for n in CHANNELS])
            self.hq[s] = m.jnt_qposadr[jj]
            self.hv[s] = m.jnt_dofadr[jj]
            self.ha[s] = np.array([m.actuator(s + '_' + n + '_motor').id for n in CHANNELS])
            self.d.qpos[self.hq[s]] = [0.08, 0, 0, 0, 0, 0]
        self.ra = np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('lsrobot_')])
        rj = m.actuator_trnid[self.ra, 0]
        self.rq = m.jnt_qposadr[rj]
        self.rv = m.jnt_dofadr[rj]
        self.rows = {s: np.array([list(self.rq).index(q) for q in ik.qp]) for s, ik in self.iks.items()}
        self.oq = [int(m.jnt_qposadr[m.joint(f'item{i}_object_free').id]) for i in range(5)]
        self.ob = [m.body(f'item{i}_object').id for i in range(5)]
        self.cq = int(m.jnt_qposadr[m.joint('crate_free').id])
        self.cb = m.body('crate').id
        self.snap.apply(self.d)
        mujoco.mj_forward(m, self.d)
        self.fixed = self.d.qpos[self.rq].copy()
        self.rng = np.random.default_rng(self.cfg['seed'])
        self.frames = []
        self.velocities = []
        self.controls = []
        self.log = []
        self.events = []
        self.phase_names = []
        self.peak_collision = 0.0
        self.total_steps = 0
        self.arm_integral = np.zeros(len(self.ra))
        self.held = {}
        self.allowed = set()
        self.hand_targets = self.hand_q()

    def body_pose(self, body, data=None):
        """读取一个物体或部件当前的位置与朝向。"""
        d = self.d if data is None else data
        return np.r_[d.xpos[body], d.xquat[body]]

    def arm_q(self):
        """读取左右机械臂当前的关节角。"""
        return {s: self.d.qpos[ik.qp].copy() for s, ik in self.iks.items()}

    def hand_q(self):
        """读取两只手当前的六个主动关节角。"""
        return {s: self.d.qpos[q].copy() for s, q in self.hq.items()}

    def wrists(self, data=None):
        """读取两个手腕当前的位置和朝向。"""
        return {s: self.body_pose(ik.body, data) for s, ik in self.iks.items()}

    def collision(self, d, allowed=None, crate_grasp=False):
        """检查不允许的碰撞；手指抓物体等正常接触单独放行。"""
        allowed = self.allowed if allowed is None else allowed
        bad = []
        m = self.m
        carried = {idx for idx, _ in self.held.values()}
        for c in d.contact[:d.ncon]:
            if c.dist >= -0.0007:
                continue
            ns = [m.geom(int(g)).name or '' for g in (c.geom1, c.geom2)]
            a, b = ns
            if any((a.startswith(s + '_') and b.startswith(s + '_') for s in self.iks)):
                continue
            if 'floor' in ns and any((n.startswith('lsrobot_base') for n in ns)):
                continue
            ca = [n.startswith('crate_') for n in ns]
            item = [int(n[4:n.index('_')]) if n.startswith('item') else None for n in ns]
            if any(ca):
                other = ns[1] if ca[0] else ns[0]
                if other == 'table_top':
                    continue
                if other.startswith('item') and int(other[4:other.index('_')]) not in carried:
                    continue
                if crate_grasp and other.startswith(('lh_', 'rh_')):
                    continue
            if any((i is not None for i in item)):
                j = 0 if item[0] is not None else 1
                other = ns[1 - j]
                if other == 'table_top' and item[j] not in carried:
                    continue
                if other.startswith('item') and (not any((idx in carried for idx in item if idx is not None))):
                    continue
                side = other[:2]
                if (side, item[j]) in allowed:
                    continue
            bad.append((float(-c.dist * 1000), a, b))
        return bad

    def kinematic(self, arms, hands, base=None, held=None):
        """只试摆姿态来检查路线，不推进真实的物理运动。"""
        d = self.kd
        m = self.m
        d.qpos[:] = self.d.qpos if base is None else base
        d.qvel[:] = 0
        for s, q in arms.items():
            d.qpos[self.iks[s].qp] = q
        for s, q in hands.items():
            d.qpos[self.hq[s]] = q
        self.snap.apply(d)
        mujoco.mj_forward(m, d)
        for s, (idx, t_hand_obj) in (self.held if held is None else held).items():
            op = pose(matrix(self.body_pose(self.iks[s].body, d)) @ t_hand_obj)
            d.qpos[self.oq[idx]:self.oq[idx] + 7] = op
        mujoco.mj_forward(m, d)
        return d

    def ik(self, goals, hands=None, seed=None, tries=24, allowed=None, held=None):
        """计算目标关节角，优先沿用当前姿态，并排除超限或发生碰撞的结果。"""
        hands = self.hand_q() if hands is None else hands
        seed = self.arm_q() if seed is None else seed
        solutions = {s: q.copy() for s, q in seed.items()}
        base = self.d.qpos.copy()
        for s, goal in goals.items():
            ik = self.iks[s]
            found = []
            for attempt in range(tries):
                q0 = seed[s] if attempt == 0 else self.rng.uniform(ik.limits[:, 0] * 0.82, ik.limits[:, 1] * 0.82)
                q, pe, re = ik.solve(goal, q0, base, iterations=230)
                if pe > 4e-05 or re > 0.0004:
                    continue
                trial = {**solutions, s: q}
                kd = self.kinematic(trial, hands, base, held)
                if self.collision(kd, allowed):
                    continue
                elbow = kd.xpos[self.m.body('lsrobot_arm_link4' + ('_m' if s == 'lh' else '')).id]
                cost = np.linalg.norm(q - seed[s]) + 0.5 * max(0, elbow[2] + 0.03) ** 2
                found.append((cost, q.copy()))
                # 当前姿态出发的解可用时直接采用；否则再比较找到的几个候选。
                if attempt == 0 or len(found) >= 3:
                    break
            if not found:
                raise RuntimeError(f'{s} IK不可达或碰撞: {np.round(goal[:3], 3).tolist()}')
            solutions[s] = min(found, key=lambda x: x[0])[1]
            base[ik.qp] = solutions[s]
        kd = self.kinematic(solutions, hands, base, held)
        if self.collision(kd, allowed):
            raise RuntimeError('双臂目标组合碰撞')
        return solutions

    def edge(self, a, b, hands, allowed=None, held=None, n=31):
        """检查两个关节姿态之间，沿途是否会碰到东西。"""
        for f in np.linspace(0, 1, n):
            qs = {s: (1 - f) * a[s] + f * b[s] for s in self.iks}
            if self.collision(self.kinematic(qs, hands, held=held), allowed):
                return False
        return True

    def shortcut_path(self, path, hands, allowed=None, held=None):
        """尝试把绕行路线拉直，去掉没有必要的中间点。"""
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                distance = max((np.max(np.abs(path[j][s] - path[i][s])) for s in self.iks))
                if self.edge(path[i], path[j], hands, allowed, held, n=max(21, int(distance / 0.025) + 1)):
                    result.append(path[j])
                    i = j
                    break
            else:
                raise RuntimeError('避障路径连接失效')
        return result

    def joint_path(self, target, hands, allowed=None, held=None, start=None):
        """先试直接走过去；有障碍时，再从起点和终点两边找绕行路线。"""
        start = self.arm_q() if start is None else start
        # 能直接过去就直连；否则才搜索绕行路线。
        if self.edge(start, target, hands, allowed, held, n=81):
            return [start, target]
        sides = tuple(self.iks)
        pack = lambda q: np.concatenate([q[s] for s in sides])
        unpack = lambda v: {s: v[j * 7:(j + 1) * 7] for j, s in enumerate(sides)}
        limits = np.vstack([self.iks[s].limits for s in sides])
        # 分别从起点和终点往外找，直到两边的路线能够连上。
        trees = [[(pack(start), -1)], [(pack(target), -1)]]
        for iteration in range(1400):
            which = iteration % 2
            tree = trees[which]
            other = trees[1 - which]
            sample = other[0][0] if iteration % 5 == 0 else self.rng.uniform(limits[:, 0], limits[:, 1])
            k = int(np.argmin([np.linalg.norm(v - sample) for v, _ in tree]))
            delta = sample - tree[k][0]
            v = tree[k][0] + delta * min(1.0, 0.3 / max(np.linalg.norm(delta), 1e-09))
            if not self.edge(unpack(tree[k][0]), unpack(v), hands, allowed, held, n=6):
                continue
            tree.append((v, k))
            new = len(tree) - 1
            near = int(np.argmin([np.linalg.norm(w - v) for w, _ in other]))
            w = other[near][0]
            if np.linalg.norm(w - v) > 0.7 or not self.edge(unpack(v), unpack(w), hands, allowed, held, n=15):
                continue

            def trace(t, j):
                path = []
                while j >= 0:
                    path.append(unpack(t[j][0]))
                    j = t[j][1]
                return path[::-1]
            p1 = trace(tree, new)
            p2 = trace(other, near)
            path = p1 + p2[::-1] if which == 0 else p2 + p1[::-1]
            return self.shortcut_path(path, hands, allowed, held)
        raise RuntimeError('未找到双臂无碰撞接近路径')

    def contacts(self, idx, side=None):
        """统计手与物体的接触力、接触部位，以及物体是否仍被桌子托住。"""
        normals = 0.0
        links = set()
        table = 0.0
        depth = 0.0
        m, d = (self.m, self.d)
        for j, c in enumerate(d.contact[:d.ncon]):
            bids = m.geom_bodyid[[c.geom1, c.geom2]]
            if self.ob[idx] not in bids:
                continue
            other = int(bids[1] if bids[0] == self.ob[idx] else bids[0])
            name = m.body(other).name
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, j, force)
            if side and name.startswith(side + '_'):
                normals += abs(float(force[0]))
                links.add(name)
                depth = max(depth, -float(c.dist) * 1000)
            if 'table_top' in [m.geom(int(g)).name for g in (c.geom1, c.geom2)]:
                table += abs(float(force[0]))
        return (normals, len(links), table, depth)

    def execute(self, arms, hands, label, cap=3.0, crate_grasp=False):
        """按顺序执行关节目标；每个物理小步更新控制，并记录运动结果。"""
        arm = {s: np.asarray(v) for s, v in arms.items()}
        hand = {s: np.asarray(v) for s, v in hands.items()}
        count = len(next(iter(arm.values())))
        vel = {s: np.gradient(v, 0.02, axis=0) for s, v in arm.items()}
        if label is not None:
            self.phase_names.append(label)
            print(f'{self.d.time:.2f}s {label}', flush=True)
        phase = len(self.phase_names) - 1
        # 轨迹点间隔 20 毫秒；每两个点之间再走 20 个 1 毫秒的物理小步。
        steps = (count - 1) * 20
        m, d = (self.m, self.d)
        for s in self.iks:
            ids = self.ha[s]
            m.actuator_gainprm[ids, :] = 0
            m.actuator_biasprm[ids, :] = 0
            m.actuator_forcerange[ids] = [-cap, cap]
            if cap >= 1:
                m.actuator_biastype[ids] = int(mujoco.mjtBias.mjBIAS_AFFINE)
                m.actuator_gainprm[ids, 0] = 5.0
                m.actuator_biasprm[ids, 1] = -5.0
                m.actuator_biasprm[ids, 2] = -0.15
            else:
                m.actuator_biastype[ids] = int(mujoco.mjtBias.mjBIAS_NONE)
                m.actuator_gainprm[ids, 0] = 1.0
        for k in range(steps):
            i = k // 20
            a = (k % 20 + 1) / 20
            j = i + 1
            target = self.fixed.copy()
            velocity = np.zeros_like(target)
            for s in self.iks:
                target[self.rows[s]] = (1 - a) * arm[s][i] + a * arm[s][j]
                velocity[self.rows[s]] = (1 - a) * vel[s][i] + a * vel[s][j]
            kp = m.actuator_gainprm[self.ra, 0]
            kv = -m.actuator_biasprm[self.ra, 2]
            self.arm_integral = np.clip(self.arm_integral + 0.004 * (target - d.qpos[self.rq]), -0.01, 0.01)
            d.ctrl[self.ra] = target + kv / kp * velocity + self.arm_integral
            d.qfrc_applied[:] = 0
            d.qfrc_applied[self.rv] = d.qfrc_bias[self.rv]
            for s in self.iks:
                bias = d.qfrc_bias[self.hv[s]].copy()
                for _, _, qa, vp, poly, _ in self.snap.couplings:
                    match = np.flatnonzero(self.hq[s] == qa)
                    if len(match):
                        bias[match[0]] += poly[1] * d.qfrc_bias[vp]
                q = (1 - a) * hand[s][i] + a * hand[s][j]
                if cap >= 1:
                    d.ctrl[self.ha[s]] = q
                else:
                    gains = np.array([0.55, 1.2, 0.6, 0.6, 0.6, 0.6])
                    d.ctrl[self.ha[s]] = np.clip(gains * (q - d.qpos[self.hq[s]]) - 0.03 * d.qvel[self.hv[s]] + bias, -cap, cap)
            # 推进物理仿真，再同步手指的从动关节。
            mujoco.mj_step(m, d)
            self.snap.apply(d)
            mujoco.mj_forward(m, d)
            bad = self.collision(d, crate_grasp=crate_grasp)
            if bad:
                self.peak_collision = max(self.peak_collision, max((b[0] for b in bad)))
            if self.total_steps % 20 == 0:
                self.frames.append(d.qpos.copy())
                self.velocities.append(d.qvel.copy())
                self.controls.append(d.ctrl.copy())
                self.log.append([float(d.time), phase, *d.xpos[self.cb], *sum((d.xpos[b].tolist() for b in self.ob), [])])
            self.total_steps += 1
        self.hand_targets = {s: v[-1].copy() for s, v in hand.items()}

    def wait(self, duration, label, cap=3.0, crate_grasp=False):
        """保持当前姿态一段时间，让物体和接触稳定下来。"""
        n = round(duration / 0.02) + 1
        self.execute({s: np.tile(q, (n, 1)) for s, q in self.arm_q().items()}, {s: np.tile(q, (n, 1)) for s, q in self.hand_targets.items()}, label, cap, crate_grasp)

    @staticmethod
    def cubic_path(points, progress):
        """把中间关节姿态连成平滑曲线，避免走到每个点都停一下。"""
        q = np.asarray(points)
        lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
        keep = np.r_[True, lengths > 1e-08]
        q = q[keep]
        if len(q) == 1:
            return np.tile(q[0], (len(progress), 1))
        knots = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        knots /= knots[-1]
        h = np.diff(knots)
        n = len(q)
        a = np.zeros((n, n))
        rhs = np.zeros_like(q)
        slopes = np.diff(q, axis=0) / h[:, None]
        a[0, 0] = 2 * h[0]
        a[0, 1] = h[0]
        rhs[0] = 6 * slopes[0]
        a[-1, -2] = h[-1]
        a[-1, -1] = 2 * h[-1]
        rhs[-1] = -6 * slopes[-1]
        for i in range(1, n - 1):
            a[i, i - 1] = h[i - 1]
            a[i, i] = 2 * (h[i - 1] + h[i])
            a[i, i + 1] = h[i]
            rhs[i] = 6 * (slopes[i] - slopes[i - 1])
        second = np.linalg.solve(a, rhs)
        x = np.asarray(progress)
        i = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, n - 2)
        gap = h[i, None]
        left = (knots[i + 1] - x)[:, None]
        right = (x - knots[i])[:, None]
        return (second[i] * left ** 3 + second[i + 1] * right ** 3) / (6 * gap) + (q[i] - second[i] * gap ** 2 / 6) * left / gap + (q[i + 1] - second[i + 1] * gap ** 2 / 6) * right / gap

    def move_route(self, path, hands, duration, label, allowed=None):
        """把整段路线变平滑；动作太快就延长时间，再检查沿途碰撞和限位。"""
        self.allowed = self.allowed if allowed is None else allowed
        sides = tuple(self.iks)
        points = np.array([np.concatenate([q[s] for s in sides]) for q in path])
        current = {s: q.copy() for s, q in self.hand_targets.items()}
        limits = np.vstack([self.iks[s].limits for s in sides])
        for density in (1, 2, 4, 8, 16):
            refined = np.vstack([np.array([(1 - f) * a + f * b for f in np.linspace(0, 1, density + 1)[:-1]]) for a, b in zip(points[:-1], points[1:])] + [points[-1:]])
            count = max(3, round(duration / 0.02) + 1)
            fractions = np.linspace(0, 1, count)
            progress = np.array([smooth(t) for t in fractions])
            samples = self.cubic_path(refined, progress)
            speed = np.max(np.abs(np.gradient(samples, 0.02, axis=0)))
            acceleration = np.max(np.abs(np.gradient(np.gradient(samples, 0.02, axis=0), 0.02, axis=0)))
            scale = max(1.0, speed / 1.4, np.sqrt(acceleration / 3.0))
            actual = np.ceil(duration * scale / 0.02) * 0.02
            fractions = np.linspace(0, 1, round(actual / 0.02) + 1)
            progress = np.array([smooth(t) for t in fractions])
            samples = self.cubic_path(refined, progress)
            hh = {s: np.array([current[s] + f * (hands[s] - current[s]) for f in progress]) for s in sides}
            valid = bool(np.all(samples >= limits[:, 0] - 1e-08) and np.all(samples <= limits[:, 1] + 1e-08))
            if valid:
                for j, row in enumerate(samples):
                    qs = {s: row[i * 7:(i + 1) * 7] for i, s in enumerate(sides)}
                    hs = {s: hh[s][j] for s in sides}
                    if self.collision(self.kinematic(qs, hs), self.allowed):
                        valid = False
                        break
            if valid:
                break
        else:
            raise RuntimeError(f'{label} 的连续样条未通过碰撞/限位检查')
        self.execute({s: samples[:, i * 7:(i + 1) * 7] for i, s in enumerate(sides)}, hh, label)

    def move_joints(self, target, hands, duration, label, allowed=None):
        """先安排关节路线，再执行整段运动。"""
        path = self.joint_path(target, hands, allowed)
        self.move_route(path, hands, duration, label, allowed)

    def cartesian_path(self, goals, hands, allowed=None, points=65, start=None, seed=None):
        """先在手腕起点与终点之间取点，再逐点计算机械臂关节角。"""
        start = self.wrists() if start is None else start
        seed = self.arm_q() if seed is None else seed
        path = []
        for t in np.linspace(0, 1, points):
            gg = {s: interpolate_pose(start[s], goals.get(s, start[s]), t) for s in self.iks}
            # 用上一个点的关节角作为本点的起点，尽量让动作连续。
            seed = self.ik(gg, hands, seed=seed, tries=1, allowed=allowed)
            path.append({s: q.copy() for s, q in seed.items()})
        return path

    def move_cartesian(self, goals, hands, duration, label, allowed=None):
        """让手腕沿参考直线运动，通常用于下降抓取和向上退出。"""
        start = self.wrists()
        seed = self.arm_q()
        aa = {s: [] for s in self.iks}
        hh = {s: [] for s in self.iks}
        hq = self.hand_targets
        for t in np.linspace(0, 1, round(duration / 0.02) + 1):
            f = smooth(t)
            gg = {s: interpolate_pose(start[s], goals.get(s, start[s]), f) for s in self.iks}
            hs = {s: (1 - f) * hq[s] + f * hands[s] for s in self.iks}
            seed = self.ik(gg, hs, seed=seed, tries=1, allowed=allowed)
            for s in self.iks:
                aa[s].append(seed[s])
                hh[s].append(hs[s])
        if allowed is not None:
            self.allowed = allowed
        self.execute(aa, hh, label)

    def save(self, result):
        """保存全部关节状态、控制量、各阶段名称和任务结果。"""
        arrays = {'qpos': np.asarray(self.frames), 'qvel': np.asarray(self.velocities), 'ctrl': np.asarray(self.controls), 'log': np.asarray(self.log)}
        np.savez_compressed(ROOT / 'results/trajectory.npz', **arrays)
        result.update(phase_names=self.phase_names, events=self.events, maximum_forbidden_penetration_mm=self.peak_collision)
        (ROOT / 'results/result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return (result, arrays)
