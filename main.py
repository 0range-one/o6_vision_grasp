"""识别物体，双臂抓取入框，最后抬起料框。"""
import argparse
import json
import numpy as np
from scene import ROOT
from grasp import run
from robot import Robot
from viewer import show


def main():
    parser = argparse.ArgumentParser(description='双臂抓取与料框搬运')
    parser.add_argument('--replay', action='store_true', help='只查看保存的完整结果')
    args = parser.parse_args()

    if args.replay:
        result = json.loads((ROOT / 'results/result.json').read_text())
        with np.load(ROOT / 'results/trajectory.npz', allow_pickle=False) as data:
            trajectory = {key: data[key].copy() for key in data.files}
    else:
        # 运行识别、抓取、放置与抬框。
        result, trajectory = run()

    show(Robot(), trajectory, result)


if __name__ == '__main__':
    main()
