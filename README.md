# 双臂抓取与料框搬运

在本目录打开终端，安装依赖后运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

程序先识别桌面的五个物体和料框，再让双臂分三轮抓取入框，最后共同抬起料框。计算结束后打开显示窗口，按空格播放。

只查看已有结果：`python main.py --replay`。
窗口中空格播放或暂停，R 重播，C 切换头部相机视角，Esc 退出；鼠标左键旋转、右键平移、滚轮缩放。

从 `main.py` 开始看，主流程在 `grasp.py`。`robot.py` 负责机械臂和手指运动，`vision.py` 负责识别，`viewer.py` 负责显示，`scene.py` 读取模型和抓取记录。代码内有中文注释。

`assets` 是模型和抓取记录，`weights` 是分割权重，`results` 保存完整结果。场景在 `scene.xml`，抓取顺序在 `config.json`。Windows 激活环境使用 `.venv\Scripts\activate`。


