"""读取场景、物体模型和已有抓取记录。"""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'assets/grasp_library'
SOURCE = ROOT / 'assets/crate_reference'
REFLECT = np.diag([1.0, -1.0, 1.0, 1.0])

def matrix(p):
    """把位置、朝向换成 4×4 矩阵。"""
    t = np.eye(4)
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(p[3:]))
    t[:3, :3] = r.reshape(3, 3)
    t[:3, 3] = p[:3]
    return t

def pose(t):
    """把矩阵换回位置与四元数。"""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, t[:3, :3].ravel())
    return np.r_[t[:3, 3], q]

def yaw(a):
    """生成绕竖直方向转动的矩阵。"""
    t = np.eye(4)
    c, s = (np.cos(a), np.sin(a))
    t[:2, :2] = [[c, -s], [s, c]]
    return t

class Asset:
    """一个物体及其成功抓取时的手腕、手指记录。"""

    def __init__(self, item):
        self.name = item['label']
        with np.load(PACK / item['record'], allow_pickle=False) as record:
            self.record = {key: record[key].copy() for key in record.files}
        model = mujoco.MjModel.from_xml_path(str(PACK / item['model']))
        self.object_q = int(model.jnt_qposadr[model.joint('object_free').id])
        hand_q = int(model.jnt_qposadr[model.joint('hand_free').id])
        self.hand = np.asarray([matrix(row[hand_q:hand_q + 7]) for row in self.record['qpos']])

    def fingers(self, side, frame=0):
        # 抓取记录使用右手；左手拇指按两只手的角度范围换算。
        angles = self.record['ctrl'][frame].copy()
        if side == 'lh':
            angles[1] *= 1.86 / 2.29
        return angles


def assets():
    """读取五个物体的抓取记录。"""
    items = json.loads((PACK / 'grasps.json').read_text())['grasps']
    return [Asset(item) for item in items]
