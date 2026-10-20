"""窗口里的文字和线条：显示物体名称、识别评分和红绿蓝坐标轴。"""
import numpy as np
import mujoco
NAMES = ['Box', 'Can', 'Pot', 'Ball', 'Mini fridge', 'Crate']
COLORS = [(1.0, 0.18, 0.15), (0.15, 1.0, 0.3), (0.2, 0.55, 1.0)]

def projector(scene, rect):
    """把三维位置换成当前窗口上的二维位置。"""
    cameras = scene.camera
    pos = 0.5 * (np.array(cameras[0].pos) + np.array(cameras[1].pos))
    forward = 0.5 * (np.array(cameras[0].forward) + np.array(cameras[1].forward))
    forward /= np.linalg.norm(forward)
    up = 0.5 * (np.array(cameras[0].up) + np.array(cameras[1].up))
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c = cameras[0]
    near = float(c.frustum_near)
    bottom = float(c.frustum_bottom)
    top = float(c.frustum_top)
    width = float(c.frustum_width) or (top - bottom) * rect.width / rect.height
    center = 0.5 * (float(cameras[0].frustum_center) + float(cameras[1].frustum_center))

    def project(point):
        delta = np.asarray(point) - pos
        depth = float(delta @ forward)
        if depth < near:
            return None
        scale = 1.0 if c.orthographic else near / depth
        x = rect.left + (delta @ right * scale - center + width / 2) / width * rect.width
        y = rect.bottom + (delta @ up * scale - bottom) / (top - bottom) * rect.height
        if not np.isfinite(x + y):
            return None
        return np.array([x, y])
    return project

def label(ctx, text, x, y, width=None, color=(1.0, 1.0, 1.0), background=(0.035, 0.045, 0.06, 0.9), bounds=None):
    """在窗口上画一个带背景的文字标签。"""
    actual_width = sum((int(ctx.charWidth[ord(c)]) for c in text))
    width = max(int(width or 0), actual_width + 12)
    height = int(ctx.charHeight) + 8
    if bounds is not None:
        x = np.clip(x, bounds.left + 2, bounds.left + bounds.width - width - 2)
        y = np.clip(y, bounds.bottom + 2, bounds.bottom + bounds.height - height - 2)
    r = mujoco.MjrRect(int(x), int(y), width, height)
    mujoco.mjr_label(r, mujoco.mjtFont.mjFONT_NORMAL, text, *background, *color, ctx)

def draw(ctx, scene, rect, canvas, estimates, camera_transform, head_view=False):
    """根据识别结果画物体名称、评分和红绿蓝坐标轴。"""
    from OpenGL import GL
    project = projector(scene, rect)
    frames = []
    labels = []
    for idx, row in estimates.items():
        transform = np.asarray(row['T_world_object'])
        origin = project(transform[:3, 3])
        if origin is None:
            continue
        if not (rect.left - 30 < origin[0] < rect.left + rect.width + 30 and rect.bottom - 30 < origin[1] < rect.bottom + rect.height + 30):
            continue
        frames.append((transform, 0.11 if idx == 5 else 0.065))
        text = f"{NAMES[idx]}  {row['yolo_confidence']:.3f}"
        if head_view and 'bbox_xyxy' in row:
            x1, y1, x2, y2 = row['bbox_xyxy']
            point = [rect.left + x1 * rect.width / 640, rect.bottom + rect.height - y1 * rect.height / 480 + 3]
        else:
            point = origin + [10, 14]
        labels.append((text, point, (0.1, 0.9, 1.0) if idx == 5 else (1.0, 1.0, 1.0)))
    if not head_view:
        frames.extend([(np.eye(4), 0.21), (np.asarray(camera_transform), 0.13)])
        for text, transform in [('BASE / world', np.eye(4)), ('CAMERA optical', np.asarray(camera_transform))]:
            point = project(transform[:3, 3])
            if point is not None:
                labels.append((text, point + [8, -28], (1.0, 0.85, 0.3)))
    axis_labels = []
    lines = []
    points = []
    for transform, length in frames:
        origin = project(transform[:3, 3])
        if origin is None:
            continue
        points.append(origin)
        for axis, color in enumerate(COLORS):
            endpoint = project(transform[:3, 3] + transform[:3, axis] * length)
            if endpoint is None:
                continue
            direction = endpoint - origin
            norm = np.linalg.norm(direction)
            if norm < 3:
                continue
            unit = direction / norm
            perp = np.array([-unit[1], unit[0]])
            lines.append((origin, endpoint, color))
            lines.extend([(endpoint, endpoint - 7 * unit + 3 * perp, color), (endpoint, endpoint - 7 * unit - 3 * perp, color)])
            axis_labels.append(('XYZ'[axis], endpoint + 3, color))
    GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
    GL.glViewport(0, 0, *canvas)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glScissor(rect.left, rect.bottom, rect.width, rect.height)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_LIGHTING)
    GL.glDisable(GL.GL_TEXTURE_2D)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    GL.glOrtho(0, canvas[0], 0, canvas[1], -1, 1)
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    GL.glLineWidth(2.5)
    GL.glBegin(GL.GL_LINES)
    for a, b, color in lines:
        GL.glColor3f(*color)
        GL.glVertex2f(*a)
        GL.glVertex2f(*b)
    GL.glEnd()
    GL.glPointSize(6)
    GL.glBegin(GL.GL_POINTS)
    GL.glColor3f(1.0, 1.0, 1.0)
    for point in points:
        GL.glVertex2f(*point)
    GL.glEnd()
    GL.glPopMatrix()
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPopMatrix()
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPopAttrib()
    for text, point, color in labels:
        label(ctx, text, *point, color=color, bounds=rect)
    for text, point, color in axis_labels:
        label(ctx, text, *point, width=20, color=color, background=(0.02, 0.03, 0.04, 0.7), bounds=rect)

def draw_observation_panel(ctx, rect, rows, observed_times, at, closure_times):
    """显示最近一次识别的坐标和时间，物体开始运动后标出 moving。"""
    text = ['YOLOv8 confidence | last RGB-D observation', 'BASE / world: lsrobot_base_link | units: m', 'Object       score    X       Y       Z    time']
    for idx, row in sorted(rows.items()):
        xyz = np.asarray(row['T_world_object'])[:3, 3]
        suffix = ' moving' if at >= closure_times.get(idx, float('inf')) else ''
        text.append(f"{NAMES[idx]:<11} {row['yolo_confidence']:.3f} {xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f} {observed_times[idx]:.1f}s{suffix}")
    text.append('Axes: red X | green Y | blue Z')
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, rect, '\n'.join(text), '', ctx)
    camera_text = 'Camera: head_camera_optical_frame\nX right | Y down | Z forward'
    line = int(ctx.charHeight) + 5
    legend_height = 2 * line + 10
    legend = mujoco.MjrRect(rect.left, rect.bottom + rect.height - 7 * line - 18 - legend_height, rect.width, legend_height)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, legend, camera_text, '', ctx)


def show(robot, trajectory, result):
    """显示完整轨迹、识别置信度和坐标轴。"""
    import time
    import glfw
    from vision import set_pitch, metadata

    glfw.init()
    window = glfw.create_window(1500, 1000, '双臂抓取与料框搬运', None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    model, data = robot.m, robot.d
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    scene = mujoco.MjvScene(model, maxgeom=16000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.27, 0, -0.59)
    camera.distance, camera.azimuth, camera.elevation = 3.0, 180, -20
    head_camera = mujoco.MjvCamera()
    head_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    head_camera.fixedcamid = set_pitch(model, 55.0)
    times = trajectory['log'][:, 0]
    observations = result['vision']['observations']
    end = float(times[-1])
    state = {'time': observations[0]['time_s'], 'paused': True, 'head': False}
    cursor = [0.0, 0.0]

    # 识别坐标来自拍图时刻；物体被抓动后，不再把旧坐标轴贴在物体上。
    closure_times = {}
    for event in result['events']:
        phase = next(i for i, name in enumerate(result['phase_names'])
                     if name.startswith(f"第{event['round']}轮：同步下降"))
        indices = np.flatnonzero(trajectory['log'][:, 1] == phase)
        for row in event['objects']:
            closure_times[row['index']] = float(times[indices[-1]])
    phase = result['phase_names'].index('全部物体入框：双臂同步抓框、抬升和保持')
    closure_times[5] = float(times[np.flatnonzero(trajectory['log'][:, 1] == phase)[0]])

    def keys(win, key, scan, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            state['paused'] = not state['paused']
        elif key == glfw.KEY_R:
            state.update(time=observations[0]['time_s'], paused=False)
        elif key == glfw.KEY_C:
            state['head'] = not state['head']

    def mouse(win, x, y):
        dx, dy = x - cursor[0], y - cursor[1]
        cursor[:] = x, y
        _, height = glfw.get_window_size(win)
        if state['head']:
            return
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / height, dy / height, scene, camera)
        elif glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / height, dy / height, scene, camera)

    glfw.set_key_callback(window, keys)
    glfw.set_cursor_pos_callback(window, mouse)
    glfw.set_scroll_callback(window, lambda win, x, y: None if state['head'] else
                             mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y, scene, camera))

    def render_camera(rect, selected, estimates, transform, canvas, head=False):
        mujoco.mjv_updateScene(model, data, option, None, selected, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(rect, scene, context)
        draw(context, scene, rect, canvas, estimates, transform, head)

    print('空格：播放/暂停；R：重播；C：相机视角；Esc：退出。', flush=True)
    previous = time.monotonic()
    try:
        while not glfw.window_should_close(window):
            now = time.monotonic()
            if not state['paused']:
                state['time'] = min(end, state['time'] + min(0.1, now - previous) * 0.7)
                state['paused'] = state['time'] >= end
            previous = now
            at = state['time']
            index = int(np.clip(np.searchsorted(times, at, side='right') - 1, 0, len(times) - 1))
            data.qpos[:] = trajectory['qpos'][index]
            data.qvel[:] = trajectory['qvel'][index]
            data.ctrl[:] = trajectory['ctrl'][index]
            data.time = float(times[index])
            robot.snap.apply(data)
            mujoco.mj_forward(model, data)

            rows, seen = {}, {}
            for observation in observations:
                if observation['time_s'] <= at + 0.001:
                    for row in observation['detections']:
                        rows[row['index']] = row
                        seen[row['index']] = observation['time_s']
            active = {idx: row for idx, row in rows.items() if at < closure_times[idx]}
            transform = metadata(model, data, 55.0)['T_world_optical']
            width, height = glfw.get_framebuffer_size(window)
            canvas = (width, height)
            full = mujoco.MjrRect(0, 0, width, height)
            if state['head']:
                view_width = min(width, int(height * 4 / 3))
                view_height = min(height, int(width * 3 / 4))
                rect = mujoco.MjrRect((width - view_width) // 2, (height - view_height) // 2, view_width, view_height)
                mujoco.mjr_rectangle(full, 0.04, 0.04, 0.04, 1.0)
                render_camera(rect, head_camera, active, transform, canvas, True)
            else:
                render_camera(full, camera, active, transform, canvas)
                view_width = int(width * 0.32)
                rect = mujoco.MjrRect(width - view_width - 16, 16, view_width, int(view_width * 3 / 4))
                render_camera(rect, head_camera, active, transform, canvas, True)
            draw_observation_panel(context, full, rows, seen, at, closure_times)
            text = f'Time: {data.time:.2f} / {end:.2f} s\nSpace: play/pause | R: replay | C: head camera | Esc: close'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, full, text, '', context)
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
