"""头部相机拍图 → YOLO 分割 → 深度点云与物体模型对齐 → 输出物体位姿。"""
import json
import os
import time
import cv2
import mujoco
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scene import ROOT, PACK

MIRROR = np.diag([1.0, -1.0, 1.0])
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.yolo'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def config():
    """读取相机安装位置、俯角和画面尺寸。"""
    return json.loads((ROOT / 'head_camera.json').read_text())

def rotations(pitch_deg):
    """计算相机在头部的朝向，并换成 MuJoCo 使用的方向约定。"""
    a = np.deg2rad(pitch_deg)
    c, s = (np.cos(a), np.sin(a))
    link = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    graphics = link @ np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return (link, graphics)

def set_pitch(model, pitch_deg):
    """修改仿真相机的下俯角度。"""
    cfg = config()
    cid = model.camera(cfg['name']).id
    _, r = rotations(pitch_deg)
    mujoco.mju_mat2Quat(model.cam_quat[cid], np.ascontiguousarray(r).ravel())
    return cid

def metadata(model, data, pitch_deg):
    """计算相机内参，以及相机坐标和机器人坐标之间的关系。"""
    cfg = config()
    cid = model.camera(cfg['name']).id
    width, height = cfg['resolution']
    focal = height / (2.0 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2.0))
    k = [[focal, 0.0, (width - 1.0) / 2.0], [0.0, focal, (height - 1.0) / 2.0], [0.0, 0.0, 1.0]]
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = data.cam_xpos[cid]
    bid = model.body(cfg['parent_link']).id
    head = np.eye(4)
    head[:3, :3] = data.xmat[bid].reshape(3, 3)
    head[:3, 3] = data.xpos[bid]
    return dict(cfg, downward_pitch_deg=float(pitch_deg), time_s=float(data.time), K=k, distortion_model='ideal_pinhole', T_world_optical=transform.tolist(), T_head_optical=(np.linalg.inv(head) @ transform).tolist(), depth_units='m', depth_definition='optical_axis_z')

def rz(angle):
    """计算绕竖直方向转动后的朝向。"""
    c, s = (np.cos(angle), np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

class CAD:
    """读取一个物体的几何形状，用它与相机看到的点对齐。"""

    def __init__(self, item):
        """读取模型并取点，保存抓取记录中物体原来的摆放朝向。"""
        self.index = item['index'] - 1
        self.name = item['label']
        self.item = item
        m = mujoco.MjModel.from_xml_path(str(PACK / item['object_model']))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        triangles = []
        for gid in range(m.ngeom):
            if m.geom_group[gid] != 2 or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            v = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            triangles.append(v[f])
        triangles = np.concatenate(triangles)
        self.set_geometry(triangles, 1729 + self.index, 100000)
        ref = mujoco.MjModel.from_xml_path(str(PACK / item['model']))
        qa = ref.jnt_qposadr[ref.joint('object_free').id]
        with np.load(PACK / item['record'], allow_pickle=False) as a:
            q = a['qpos'][0, qa + 3:qa + 7]
        r = np.empty(9)
        mujoco.mju_quat2Mat(r, q)
        self.rest = r.reshape(3, 3)

    def set_geometry(self, triangles, seed, count):
        """按表面面积取点，准备后面查找距离最近的模型点。"""
        self.vertices = np.unique(triangles.reshape(-1, 3), axis=0)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(area[:, None], 1e-20)
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(triangles), count, p=area / area.sum())
        uv = rng.random((len(ids), 2))
        uv[uv.sum(1) > 1.0] = 1.0 - uv[uv.sum(1) > 1.0]
        tri = triangles[ids]
        self.points = tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
        self.normals = normals[ids]
        self.tree = cKDTree(self.points)

    def fit(self, observed, table_z, side):
        """先试不同转角，再细调位置和朝向；球和罐子的对称方向按约定处理。"""
        start = time.perf_counter()
        observed = observed[::max(1, len(observed) // 1800)]
        coarse = observed[::max(1, len(observed) // 350)]
        r0 = MIRROR @ self.rest @ MIRROR if side == 'lh' else self.rest.copy()
        if self.index == 3:
            centre0 = (observed.min(0) + observed.max(0)) / 2.0
            fit = least_squares(lambda x: np.linalg.norm(observed - x[:3], axis=1) - x[3], np.r_[centre0, 0.055], loss='soft_l1', f_scale=0.001, bounds=(np.r_[centre0 - 0.08, 0.03], np.r_[centre0 + 0.08, 0.09]))
            r = rz(np.pi if side == 'lh' else 0.0) @ r0
            local_centre = (self.vertices.min(0) + self.vertices.max(0)) / 2.0
            t = fit.x[:3] - r @ local_centre
            r, t, err = self.icp(observed, r, t, table_z, planar=False, iterations=55)
            return self.result(r, t, err, start, 'sphere_initialization_then_CAD; semantic_rotation_ambiguous', radius=float(fit.x[3]))
        candidates = []
        for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False):
            r = rz(angle) @ r0
            bounds = self.vertices @ r.T
            t = np.array([*(0.5 * (coarse.min(0) + coarse.max(0) - bounds.min(0) - bounds.max(0)))[:2], table_z - float(bounds[:, 2].min()) + 0.0002])
            r, t, err = self.icp(coarse, r, t, table_z, planar=True, iterations=20)
            score = float(np.mean(np.minimum(err, 0.012) ** 2))
            candidates.append((score, r, t))
        candidates.sort(key=lambda x: x[0])
        solutions = []
        for _, r, t in candidates[:8]:
            r, t, err = self.icp(observed, r.copy(), t.copy(), table_z, planar=False, iterations=55)
            score = float(np.mean(np.minimum(err, 0.012) ** 2))
            solutions.append((score, r, t, err))
        _, r, t, err = min(solutions, key=lambda x: x[0])
        if self.index == 1:
            ref = rz(np.pi if side == 'lh' else 0.0) @ r0
            candidates = []
            axis = np.array([0.0, 1.0, 0.0])
            for a in np.linspace(-np.pi, np.pi, 361):
                rr = r @ Rotation.from_rotvec(axis * a).as_matrix()
                candidates.append((np.linalg.norm(rr - ref), rr))
            r = min(candidates, key=lambda x: x[0])[1]
            flag = 'axis_and_position; canonical_axial_yaw'
        else:
            flag = 'CAD_registration_6d_with_table_rest_initialization'
        return self.result(r, t, err, start, flag)

    def icp(self, observed, r, t, table_z, planar, iterations):
        """反复寻找相邻点并小幅调整，让模型表面贴近观测点。"""
        for _ in range(iterations):
            local = (observed - t) @ r
            dist, ids = self.tree.query(local, workers=1)
            keep = dist < max(0.004, min(0.04, float(np.quantile(dist, 0.9)) * 1.4))
            if keep.sum() < 20:
                break
            source = self.points[ids[keep]] @ r.T
            normal = self.normals[ids[keep]] @ r.T
            residual = np.einsum('ij,ij->i', source + t - observed[keep], normal)
            weight = np.sqrt(np.minimum(1.0, 0.0025 / np.maximum(abs(residual), 1e-12)))
            cross = np.cross(source, normal)
            jac = np.c_[normal[:, :2], cross[:, 2]] if planar else np.c_[normal, cross]
            step = np.linalg.lstsq(jac * weight[:, None], -residual * weight, rcond=1e-06)[0]
            if planar:
                step[:2] = np.clip(step[:2], -0.008, 0.008)
                step[2] = np.clip(step[2], -0.12, 0.12)
                t[:2] += step[:2]
                r = rz(step[2]) @ r
                t[2] = table_z - float((self.vertices @ r.T)[:, 2].min()) + 0.0002
            else:
                step[:3] = np.clip(step[:3], -0.003, 0.003)
                step[3:] = np.clip(step[3:], -0.035, 0.035)
                t += step[:3]
                r = Rotation.from_rotvec(step[3:]).as_matrix() @ r
            if np.linalg.norm(step) < 2e-06:
                break
        err, _ = self.tree.query((observed - t) @ r, workers=1)
        return (r, t, err)

    def result(self, r, t, err, start, method, **extra):
        """整理对齐后的姿态、点之间的距离和计算时间。"""
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = (r, t)
        return dict(index=self.index, name=self.name, T_world_object=transform.tolist(), registration_rmse_mm=float(np.sqrt(np.mean(err ** 2)) * 1000), registration_median_mm=float(np.median(err) * 1000), compute_seconds=time.perf_counter() - start, method=method, **extra)

class CrateCAD(CAD):
    """料框的几何模型与对齐方法。"""

    def __init__(self):
        """读取料框表面，并记录框口高度。"""
        self.index = 5
        self.name = '料框'
        path = ROOT / 'assets/crate/crate_visual.obj'
        vertices = []
        faces = []
        for line in path.open():
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(s.split('/')[0]) - 1 for s in line.split()[1:4]])
        self.set_geometry(np.asarray(vertices)[np.asarray(faces)], 1734, 240000)
        self.rest = np.eye(3)
        self.rim_height = float(self.vertices[:, 2].max())

    def fit(self, observed, table_z, side=None):
        """先根据料框长边猜方向，再对齐；用统一规则区分相差 180 度的方向。"""
        started = time.perf_counter()
        observed = observed[::max(1, len(observed) // 3500)]
        coarse = observed[::max(1, len(observed) // 650)]
        covariance = np.cov(coarse[:, :2].T)
        values, vectors = np.linalg.eigh(covariance)
        main = vectors[:, np.argmax(values)]
        heading = np.arctan2(main[1], main[0])
        solutions = []
        for a in [heading + k * np.pi / 2 + offset for k in range(4) for offset in (-0.18, 0.0, 0.18)]:
            r = rz(a)
            bounds = self.vertices @ r.T
            t = np.array([*(0.5 * (coarse.min(0) + coarse.max(0) - bounds.min(0) - bounds.max(0)))[:2], table_z - bounds[:, 2].min() + 0.0002])
            r, t, err = self.icp(coarse, r, t, table_z, planar=True, iterations=35)
            if r[1, 0] >= 0:
                continue
            solutions.append((float(np.mean(np.minimum(err, 0.015) ** 2)), r, t))
        solutions.sort(key=lambda x: x[0])
        refined = []
        for _, r, t in solutions[:4]:
            r, t, err = self.icp(observed, r.copy(), t.copy(), table_z, planar=False, iterations=65)
            refined.append((float(np.mean(np.minimum(err, 0.015) ** 2)), r, t, err))
        _, r, t, err = min(refined, key=lambda x: x[0])
        return self.result(r, t, err, started, 'crate_CAD_registration', frame_symmetry='180_deg_yaw; long_X_toward_robot_negative_Y', rim_height_m=self.rim_height)

class Perception:
    """处理一帧彩色图和深度图，输出六类物体的位置与朝向。"""

    def __init__(self):
        """加载六类分割权重和物体模型；可用 CPU 运行。"""
        import torch
        from ultralytics import YOLO
        weights = ROOT / 'weights/six_classes.pt'
        torch.set_num_threads(4)
        self.device = os.environ.get('YOLO_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = YOLO(str(weights))
        self.confidence = 0.25
        manifest = json.loads((PACK / 'grasps.json').read_text())
        self.catalog = [CAD(x) for x in manifest['grasps']] + [CrateCAD()]
        self.global_models = {}

    def segment(self, rgb):
        """用 YOLO 找物体区域；每一类保留评分最高的检测。"""
        result = self.model.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=640, conf=self.confidence, iou=0.5, device=self.device, retina_masks=True, verbose=False)[0]
        detected = {}
        if result.masks is None:
            return detected
        for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes):
            idx = int(box.cls.item())
            score = float(box.conf.item())
            if idx not in range(6):
                continue
            mask = cv2.resize(mask.astype('float32'), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            if mask.sum() < 35:
                continue
            if idx not in detected or score > detected[idx]['confidence']:
                detected[idx] = dict(mask=mask, confidence=score, bbox=box.xyxy[0].cpu().numpy().tolist())
        return detected

    def global_candidate(self, idx, points):
        """比较模型和实测点的局部形状，先猜一组可能的对齐位置。"""
        import open3d as o3d
        o3d.utility.random.seed(611 + idx)
        voxel = 0.004 if idx == 5 else 0.003
        cad = self.catalog[idx]

        def preprocess(points, normals=None):
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            if normals is not None:
                p.normals = o3d.utility.Vector3dVector(normals)
            p = p.voxel_down_sample(voxel)
            if normals is None:
                p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
            f = o3d.pipelines.registration.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=80))
            return (p, f)
        if idx not in self.global_models:
            self.global_models[idx] = preprocess(cad.points, cad.normals)
        source, f = self.global_models[idx]
        target, tf = preprocess(points)
        if len(target.points) < 30:
            return None
        reg = o3d.pipelines.registration
        return reg.registration_ransac_based_on_feature_matching(source, target, f, tf, True, voxel * 2, reg.TransformationEstimationPointToPoint(False), 3, [reg.CorrespondenceCheckerBasedOnEdgeLength(0.85), reg.CorrespondenceCheckerBasedOnDistance(voxel * 2)], reg.RANSACConvergenceCriteria(12000, 0.995))

    def fit(self, idx, points, table_z, side):
        """比较几种对齐方法，选出与观测点更接近的一组结果。"""
        cad = self.catalog[idx]
        result = cad.fit(points, table_z, side)
        if idx == 3:
            result.update(method='YOLOv8_mask + sphere/CAD fit; canonical_spherical_frame', global_candidate_selected=False)
            return result
        started = time.perf_counter()
        gf = self.global_candidate(idx, points)
        result.update(global_registration='Open3D_CPU_FPFH_RANSAC', global_candidate_selected=False, registration_prior='table_support_and_reference_rest_orientation', method='YOLOv8_mask + FPFH_RANSAC/table_multistart + robust_CAD_ICP')
        if gf is not None:
            result['global_ransac_fitness'] = float(gf.fitness)
            rr = gf.transformation[:3, :3].copy()
            tt = gf.transformation[:3, 3].copy()
            sampled = points[::max(1, len(points) // 2500)]
            rr, tt, err = cad.icp(sampled, rr, tt, table_z, False, 65)
            if idx == 5 and rr[1, 0] > 0:
                rr = rr @ np.diag([-1.0, -1.0, 1.0])
            rmse = float(np.sqrt(np.mean(err ** 2)) * 1000)
            result['global_candidate_rmse_mm'] = rmse
            if idx != 1 and rmse < result['registration_rmse_mm'] * 0.98 and (abs((cad.vertices @ rr.T + tt)[:, 2].min() - table_z) < 0.008):
                transform = np.eye(4)
                transform[:3, :3] = rr
                transform[:3, 3] = tt
                result.update(T_world_object=transform.tolist(), registration_rmse_mm=rmse, registration_median_mm=float(np.median(err) * 1000), global_candidate_selected=True)
        result['compute_seconds'] += time.perf_counter() - started
        return result

    def estimate(self, rgb, depth, calibration, wanted):
        """找区域、读取深度、计算桌面高度，再逐个对齐物体模型。"""
        started = time.perf_counter()
        # 先从彩色图里找出各个物体区域。
        detections = self.segment(rgb)
        requested = set(wanted)
        requested.add(5)
        k = np.asarray(calibration['K'])
        t = np.asarray(calibration['T_world_optical'])
        yy, xx = np.indices(depth.shape)
        # 把每个像素的深度换成三维点，再换到机器人基座坐标。
        optical = np.stack([(xx - k[0, 2]) / k[0, 0] * depth, (yy - k[1, 2]) / k[1, 1] * depth, depth], axis=-1)
        world = optical @ t[:3, :3].T + t[:3, 3]
        region = (world[:, :, 0] > 0.16) & (world[:, :, 0] < 0.88) & (abs(world[:, :, 1]) < 0.79) & (world[:, :, 2] > -0.61) & (world[:, :, 2] < -0.28) & np.isfinite(depth) & (depth > 0)
        heights = world[:, :, 2][region]
        # 从点云中找出桌面的高度。
        counts, edges = np.histogram(heights, np.arange(-0.61, -0.28, 0.001))
        z0 = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
        table_z = float(np.median(heights[abs(heights - z0) < 0.002]))
        foreground = region & (world[:, :, 2] > table_z + 0.003)
        output = {}
        for idx in sorted(requested, reverse=True):
            row = detections[idx]
            mask = row['mask'] & foreground
            interior = binary_erosion(mask)
            pts = world[interior if interior.sum() > 50 else mask]
            side = 'both' if idx == 5 else 'lh' if np.median(pts[:, 1]) > 0 else 'rh'
            # 把物体模型和这片点云对齐，得到位置和朝向。
            result = self.fit(idx, pts, table_z, side)
            ct = np.asarray(result['T_world_object'])
            cad = self.catalog[idx]
            distance, _ = cad.tree.query((pts - ct[:3, 3]) @ ct[:3, :3])
            clean = pts[distance < 0.004]
            if len(clean) > 80:
                rr, tt, err = cad.icp(clean, ct[:3, :3].copy(), ct[:3, 3].copy(), table_z, False, 30)
                if idx not in (1, 3):
                    ct[:3, :3] = rr
                    ct[:3, 3] = tt
                    result['T_world_object'] = ct.tolist()
                result['inlier_rmse_mm'] = float(np.sqrt(np.mean(err ** 2)) * 1000)
            fraction = float(np.mean(distance < 0.004))
            result['cad_inlier_fraction'] = fraction
            result.update(side=side, pixel_count=int(mask.sum()), yolo_confidence=row['confidence'], bbox_xyxy=[int(x) for x in row['bbox']], segmentation='YOLOv8n-seg_six_classes', inference_device=self.device)
            output[idx] = result
            print(f"YOLOv8 {result['name']}: score={row['confidence']:.3f}, CAD inliers={fraction:.3f}, residual={result.get('inlier_rmse_mm', result['registration_rmse_mm']):.3f} mm", flush=True)
        report = dict(source='RGB + depth + calibration + known CAD', recognition='trained_YOLOv8n_instance_segmentation', no_simulator_object_pose_or_instance_id_input=True, no_color_rule_fallback=True, camera_pitch_deg=calibration['downward_pitch_deg'], table_z_estimated_m=table_z, detections=list(output.values()), missing_indices=[], compute_seconds=time.perf_counter() - started, score_note='YOLO scores are not calibrated pose-accuracy probabilities')
        return (output, report)

class Vision:
    """保存相机观测和物体识别结果，供抓取流程读取。"""

    def __init__(self):
        """加载视觉模型，准备保存每次识别结果。"""
        self.estimator = Perception()
        self.observations = []
        self.last = {}

    def observe(self, engine, wanted, label):
        """拍摄当前场景，识别物体，并保存本次估计姿态。"""
        m, d = (engine.m, engine.d)
        set_pitch(m, 55.0)
        mujoco.mj_forward(m, d)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        with mujoco.Renderer(m, width=640, height=480, max_geom=16000) as renderer:
            renderer.update_scene(d, camera='head_camera', scene_option=option)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
        calibration = metadata(m, d, 55.0)
        # 识别函数只接收图片、深度和相机参数。
        estimates, report = self.estimator.estimate(rgb, depth, calibration, wanted)
        self.last.update(estimates)
        self.observations.append(dict(label=label, time_s=float(d.time), **report))
        return estimates


    def pose(self, index):
        """取出某个物体最近一次识别到的位置和朝向。"""
        return np.asarray(self.last[index]['T_world_object']).copy()

    def report(self):
        """汇总本次运行使用的视觉结果。"""
        return {'observations': self.observations}
