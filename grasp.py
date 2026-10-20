"""抓取主流程：看物体位置，抓起来，放进料框，最后抬起料框。"""
import json
import time
import numpy as np
from robot import Robot, smooth
from scene import SOURCE, REFLECT, matrix, pose, yaw
from vision import Vision

# robot 管理机械臂和场景，robot.aa 保存五个物体及其抓取记录。
# lh 表示左手，rh 表示右手。位置单位为米，关节角单位为弧度。

SLOTS_LOCAL = [(0.18, -0.09), (-0.17, 0.09), (0.17, 0.09), (-0.18, -0.09), (0.0, 0.0)]

def drop_object_target(crate_pose, object_pose, table_z, slot, rim_height, angle):
    """计算物体在料框上方的松手位置；留一点高度，让物体松手后落入框里。"""
    local = np.eye(4)
    local[:3, :3] = yaw(angle)[:3, :3] @ crate_pose[:3, :3].T @ object_pose[:3, :3]
    object_support_height = max(0.005, float(object_pose[2, 3] - table_z))
    local[:3, 3] = [*slot, rim_height + 0.035 + object_support_height]
    return crate_pose @ local

def grasp_relative(a, side, frame=219):
    """从保存的抓取记录中，取出手腕相对于物体的位置和朝向。"""
    relative = np.linalg.inv(matrix(a.record['qpos'][frame, a.object_q:a.object_q + 7])) @ a.hand[frame]
    return REFLECT @ relative @ REFLECT if side == 'lh' else relative

def vertical_hover_seed(robot, grasp_poses, open_fingers, allowed):
    """先试算整段上下运动，找出能连续下降到物体、又能抬起来的关节姿态。"""
    carry = {side: g.copy() for side, g in grasp_poses.items()}
    for g in carry.values():
        g[2] += 0.32
    for attempt in range(16):
        seed = robot.arm_q()
        if attempt:
            for side in grasp_poses:
                seed[side] = robot.rng.uniform(robot.iks[side].limits[:, 0] * 0.8, robot.iks[side].limits[:, 1] * 0.8)
        try:
            seed = robot.ik(carry, open_fingers, seed, tries=8, allowed=allowed)
            hover = None
            for height in np.linspace(0.32, 0, 65):
                goals = {side: g.copy() for side, g in grasp_poses.items()}
                for g in goals.values():
                    g[2] += height
                seed = robot.ik(goals, open_fingers, seed, tries=1, allowed=allowed)
                if abs(height - 0.22) < 1e-06:
                    hover = {side: q.copy() for side, q in seed.items()}
            return hover
        except RuntimeError:
            continue
    raise RuntimeError('没有同时覆盖抓取与搬运高度的连续机械臂构型')

def close_grip(robot, active):
    """手腕保持不动，五根手指一起闭合到保存的抓取角度。"""
    previous = {side: q.copy() for side, q in robot.hand_targets.items()}
    final = {side: q.copy() for side, q in previous.items()}
    for side, object_index in active.items():
        final[side] = robot.aa[object_index].fingers(side, 219)
        if side == 'lh':
            final[side][2:] += 0.08
    n = 111
    current = robot.arm_q()
    robot.execute(
        {side: np.tile(q, (n, 1)) for side, q in current.items()},
        {
            side: np.array([
                previous[side] + smooth(t) * (final[side] - previous[side])
                for t in np.linspace(0, 1, n)
            ])
            for side in robot.iks
        },
        '双臂同步闭合（复用稳定抓取姿态）',
    )

def collect_round(robot, object_indices, round_number):
    """完成一轮：识别物体、接近、抓住、抬起、放进料框，然后退出。"""
    # 1. 确定本轮左右手各抓哪个物体。没有分配物体的手保持不动。
    active = {side: object_index for side, object_index in zip(('lh', 'rh'), object_indices) if object_index is not None}
    allowed = {(side, object_index) for side, object_index in active.items()}
    robot.allowed = set()
    robot.held = {}
    #读取初始位置
    initial = {object_index: robot.d.xpos[robot.ob[object_index]].copy() for object_index in active.values()}
    if round_number > 1:
        robot.vision.observe(robot, list(active.values()), f'round_{round_number}')
    # 2. 从视觉结果里取位置和朝向，计算这只手应该在哪里抓住物体。
    object_estimates = {object_index: robot.vision.pose(object_index) for object_index in active.values()}
    grasp_poses = {side: pose(object_estimates[object_index] @ grasp_relative(robot.aa[object_index], side)) for side, object_index in active.items()}
    open_fingers = robot.hand_q()
    for side, object_index in active.items():
        open_fingers[side] = robot.aa[object_index].fingers(side)
        if object_index == 2:
            open_fingers[side][1:] = 0
    # 3. 先找到能上下连续运动的关节姿态，再移动到物体上方。
    target = vertical_hover_seed(robot, grasp_poses, open_fingers, allowed)
    robot.move_joints(
        target,
        open_fingers,
        3.0,
        f'第{round_number}轮：双臂接近物体上方' if round_number == 1
        else f'第{round_number}轮：直接接近下一组物体',
        allowed=set(),
    )
    # 4. 下降到抓取位置，然后让手指闭合。
    robot.move_cartesian(
        grasp_poses, open_fingers, 1.4,
        f'第{round_number}轮：同步下降到抓取姿态', allowed=allowed,
    )
    close_grip(robot, active)
    for side, object_index in active.items():
        robot.held[side] = (object_index, np.linalg.inv(matrix(robot.body_pose(robot.iks[side].body))) @ object_estimates[object_index])
    # 5. 再看一次料框，确定放置位置，避免一直使用旧观测。
    robot.vision.observe(robot, [5], f'round_{round_number}_placement_crate')
    crate_estimate = robot.vision.pose(5)
    table_z = robot.vision.observations[-1]['table_z_estimated_m']
    rim = robot.vision.estimator.catalog[5].rim_height
    goals = robot.wrists()
    for side in active:
        goals[side][2] += 0.32
    # 6. 抬起物体，准备搬到料框上方。
    rise = robot.cartesian_path(goals, robot.hand_targets, allowed=allowed)
    target = {side: q.copy() for side, q in rise[-1].items()}
    for side, object_index in active.items():
        # 试几个松手朝向，优先选机械臂需要转动较小的一种。
        choices = []
        for angle in np.linspace(-np.pi, np.pi, 17)[:-1]:
            ot = drop_object_target(crate_estimate, object_estimates[object_index], table_z, SLOTS_LOCAL[object_index], rim, angle)
            goal = pose(ot @ np.linalg.inv(robot.held[side][1]))
            if goal[0] < 0.34:
                continue
            try:
                q = robot.ik({side: goal}, robot.hand_targets, seed=target, tries=8, allowed=allowed)
            except RuntimeError:
                continue
            choices.append((np.linalg.norm(q[side] - target[side]), q))
        if not choices:
            raise RuntimeError(f'{robot.aa[object_index].name} 料框上方没有可达放置朝向')
        _, target = min(choices, key=lambda c: c[0])
    # 7. 把抬升与搬运连起来，途中不再回到水平张开姿态。
    transfer = robot.joint_path(target, robot.hand_targets, allowed=allowed, start=rise[-1])
    robot.move_route(
        rise + transfer[1:], robot.hand_targets, 5.6,
        f'第{round_number}轮：双臂连续抬升并搬运至料框', allowed=allowed,
    )
    # 8. 根据抬升高度、接触和桌面支撑情况判断是否抓住了。
    event = {'round': round_number, 'objects': []}
    for side, object_index in active.items():
        force, links, support, depth = robot.contacts(object_index, side)
        lift = float((robot.d.xpos[robot.ob[object_index], 2] - initial[object_index][2]) * 1000)
        row = {
            'index': object_index,
            'name': robot.aa[object_index].name,
            'side': side,
            'lift_mm': lift,
            'contact_n': force,
            'contact_links': links,
            'table_support_n': support,
            'penetration_mm': depth,
            'success': bool(lift > 60 and links >= 2 and support < 0.1),
        }
        event['objects'].append(row)
        print('抓取验证', json.dumps(row, ensure_ascii=False), flush=True)
        if not row['success']:
            robot.events.append(event)
            raise RuntimeError(f'{robot.aa[object_index].name} 抓取未通过')
    robot.events.append(event)
    # 9. 到达料框上方后张开手指，让物体落入框内。
    opened = robot.hand_q()
    for side, object_index in active.items():
        opened[side] = np.array([robot.aa[object_index].fingers(side)[0], 0, 0, 0, 0, 0])
    current = robot.arm_q()
    n = 41
    hq = robot.hand_targets
    robot.held = {}
    robot.execute(
        {side: np.tile(q, (n, 1)) for side, q in current.items()},
        {
            side: np.array([
                hq[side] + smooth(t) * (opened[side] - hq[side])
                for t in np.linspace(0, 1, n)
            ])
            for side in robot.iks
        },
        f'第{round_number}轮：同时松手放入料框',
    )
    goals = robot.wrists()
    # 10. 手腕上移 10 厘米退出料框，接着去抓下一组物体。
    for side in active:
        goals[side][2] += 0.1
    robot.move_cartesian(goals, opened, 1.0, f'第{round_number}轮：退出料框', allowed=allowed)
    for side, object_index in active.items():
        local = np.linalg.inv(matrix(robot.body_pose(robot.cb))) @ np.r_[robot.d.xipos[robot.ob[object_index]], 1.0]
        inside = bool(abs(local[0]) < 0.285 and abs(local[1]) < 0.185 and (0.005 < local[2] < 0.255))
        event['objects'][list(active.values()).index(object_index)]['inside_crate_after_drop'] = inside
        if not inside:
            raise RuntimeError(f'{robot.aa[object_index].name} 未落入料框: {local[:3]}')
    robot.allowed = set()

def lift_loaded_crate(robot):
    """根据当前看到的料框位置，调整保存的抓框动作，再用双手抬起来。"""
    saved_path = {k: v.copy() for k, v in np.load(SOURCE / 'results/plan.npz').items()}
    saved_config = json.loads((SOURCE / 'config.json').read_text())
    before = robot.body_pose(robot.cb)
    robot.vision.observe(robot, [5], 'loaded_crate_grasp')
    # 把旧抓框位置换到相机这次看到的新料框位置。
    pose_offset = robot.vision.pose(5) @ np.linalg.inv(matrix(np.asarray(saved_config['crate_pose_xyz_wxyz'])))
    item_before = np.array([robot.d.xpos[b].copy() for b in robot.ob])
    robot.allowed = set()
    robot.held = {}
    start = int(np.searchsorted(saved_path['time'], 4.6))
    hover = {side: pose(pose_offset @ matrix(saved_path[side + '_wrist'][start])) for side in robot.iks}
    open_fingers = {side: saved_path[side + '_hand'][start] for side in robot.iks}
    source_seed = {side: saved_path[side + '_arm'][start] for side in robot.iks}
    target = robot.ik(hover, open_fingers, source_seed, tries=24, allowed=set())
    robot.move_joints(target, open_fingers, 3.0, '全部物体入框：直接接近抓框位置', allowed=set())
    arm_path = {side: [] for side in robot.iks}
    finger_path = {side: [] for side in robot.iks}
    seed = robot.arm_q()
    # 按保存的手腕路线逐点求逆解，手指角度仍沿用原抓框动作。
    for j in range(start, len(saved_path['time'])):
        t = saved_path['time'][j]
        goals = {side: pose(pose_offset @ matrix(saved_path[side + '_wrist'][j])) for side in robot.iks}
        finger_targets = {side: saved_path[side + '_hand'][j] for side in robot.iks}
        base = robot.d.qpos.copy()
        for side, ik in robot.iks.items():
            q, pe, re = ik.solve(goals[side], seed[side], base, iterations=240)
            if pe > 4e-05 or re > 0.0004:
                raise RuntimeError(f'载物抓框 IK失败 {side} t={t:.2f}')
            seed[side] = q
            base[ik.qp] = q
        kd = robot.kinematic(seed, finger_targets, base)
        bad = robot.collision(kd, crate_grasp=True)
        if bad:
            raise RuntimeError(f'载物抓框路径碰撞 t={t:.2f}: {bad[:2]}')
        for side in robot.iks:
            arm_path[side].append(seed[side].copy())
            finger_path[side].append(finger_targets[side])
    robot.execute(arm_path, finger_path, '全部物体入框：双臂同步抓框、抬升和保持', cap=0.24, crate_grasp=True)
    after = robot.body_pose(robot.cb)
    lift = float((after[2] - before[2]) * 1000)
    inv = np.linalg.inv(matrix(after))
    contained = []
    item_lift = []
    for object_index, b in enumerate(robot.ob):
        local = inv @ np.r_[robot.d.xipos[b], 1.0]
        contained.append(bool(abs(local[0]) < 0.285 and abs(local[1]) < 0.185 and (0.005 < local[2] < 0.255)))
        item_lift.append(float((robot.d.xpos[b, 2] - item_before[object_index, 2]) * 1000))
    tilt = float(np.degrees(np.arccos(np.clip(robot.d.xmat[robot.cb].reshape(3, 3)[2, 2], -1, 1))))
    return {
        'success': bool(
            lift > 80 and all(contained) and min(item_lift) > 70
            and tilt < 10 and robot.peak_collision < 2
        ),
        'crate_lift_mm': lift,
        'crate_final_tilt_deg': tilt,
        'objects_inside': contained,
        'object_lift_mm': item_lift,
        'loaded_mass_kg': 2.5,
        'seed': robot.cfg['seed'],
    }

def run():
    """固定主流程：三轮抓取五个物体，全部入框后双臂抬起料框。"""
    robot = Robot()
    robot.vision = Vision()
    started = time.perf_counter()

    # 水平张开开始，先拍一帧识别所有物体和料框。
    robot.wait(0.5, '双臂水平张开：物体稳定')
    robot.vision.observe(robot, list(range(6)), 'initial')

    # 前两轮双臂同时抓，第三轮抓剩下的一个；轮间直接去下一个物体。
    for number, object_indices in enumerate(robot.cfg['rounds'], start=1):
        collect_round(robot, object_indices, number)

    result = lift_loaded_crate(robot)
    result['compute_seconds'] = time.perf_counter() - started
    result['vision'] = robot.vision.report()
    result, trajectory = robot.save(result)
    print('任务结果：' + ('成功' if result['success'] else '未通过'))
    print(f"料框抬升 {result['crate_lift_mm']:.1f} mm，倾斜 {result['crate_final_tilt_deg']:.2f}°")
    return result, trajectory
