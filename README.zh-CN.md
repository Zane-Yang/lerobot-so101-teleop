# 🤖 LeRobot × SO-101：遥操作 → 模仿学习 → 真机部署

<div align="center">

[**English**](README.md) · 🌐 **中文**

</div>

> 一套**端到端真实机器人**项目：以 Hugging Face [LeRobot](https://github.com/huggingface/lerobot) 为框架，基于低成本的 **SO-101 六自由度从臂**（Feetech STS3215 × 6），自研**无主臂遥操作**（键盘/手柄），使用 **ACT** 训练，并在**真实机械臂**上完成“抓取糖果放入盖子”任务。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)
[![HF LeRobot](https://img.shields.io/badge/based_on-HuggingFace%20LeRobot-yellow)](https://github.com/huggingface/lerobot)

---

## 🎬 演示

<p align="center">
  <img src="media/demo1.gif" alt="真机糖果抓取-放置演示" width="480"><br/>
  <i>ACT 策略在真实机械臂上自主执行：抓取并放入（GIF 预览，2倍速）</i>
</p>

<details open>
<summary>▶️ 在线播放完整 MP4（无需下载）</summary>

<video src="media/demo.mp4" controls width="480" autoplay loop muted playsinline></video>

</details>

> ✔️ `media/demo.mp4` 可在本页面直接内嵌播放，无需下载。
> ✔️ `media/demo1.gif` 为轻量动态封面图，便于快速预览。

---

## ✨ 项目亮点

在不依赖主臂的条件下，打通从“数据”到“策略”再到“真机行为”的完整闭环：

```
┌────────────┐  键盘 / 手柄遥操作      ┌────────────────┐
│   操作者    │ ────────────────────▶ │  无主臂遥操作   │
└────────────┘                        └────────────────┘
                                              │ 动作
                                              ▼
┌────────────┐ 状态/图像               ┌──────────────────┐
│  SO-101    │ ◀───────────────────  │ LeRobot 管线      │
│   从臂     │  ───────────────────▶  │ (采集/训练/评估)  │
└────────────┘ observation.images     └──────────────────┘
               observation.state(6 关节)
```

| 环节 | 实现 |
|------|------|
| 🖱️ 遥操作 | 手柄模拟量速度控制（`gamepad_joints`）与键盘增量步进控制（`keyboard_joints`），均无主臂 |
| 🗃️ 数据 | 55+ 条真实“取糖-放入盖子”演示，含数据清洗/单条剔除与帧精确视频索引（清洗后 48 条 / 25942 帧） |
| 🧠 策略 | **ACT**（Action Chunking with Transformers，ResNet18 backbone），本地训练 |
| 🎯 部署 | 通过 LeRobot rollout 在真实机械臂上完成抓取-放置评估，并有录像佐证 |

> **亮点在于硬件受限仍能跑通全流程：仅需一台 RTX 3060 6GB 显存的笔记本**，即可完成：遥操作采集 → 训练 → 真机部署。

---

## 🧱 相对 LeRobot 的自研增量

本仓库**不是整份代码的 fork**，只收纳在 LeRobot 之上**自研/手调**的核心增量：

1. **无主臂遥操作（从臂独立可控）**
   - `gamepad_joints`：**模拟摇杆速度映射**——两个摇杆控制 4 个关节，肩/扳机键控制腕部与夹爪；支持速度/方向/死区/关节限位配置。
   - `keyboard_joints`：无手柄时用键盘实现关节级步进；整合终端/pynput 两种按键捕获，SSH 下也可用。
   - 二者都作为 LeRobot **teleoperator 子类** 注册，直接接入 record / teleoperate，不改动训练链路。

2. **数据工程（保证模仿学习数据质量）**
   - 单条 episode 质量筛除与保留（对挑选后的干净集执行 `delete_episodes`）。
   - 修复录制中断续录导致的时间戳/索引不一致问题。
   - `examples/dataset_episode_viewer.py`：用 pyAV 将单条 episode 导出为“**相机 + 关节 action/state 曲线同帧**”的 mp4 预览；在 torchcodec 环境损坏时也能用。
   - `examples/gamepad_probe.py`：打印任意 USB 手柄实时的轴/按键编号，便于按机型重映射。

3. **本地端到端训练 + 真机评估**
   - 提供采/训/评三合一脚本，默认参数已在 6GB 显卡环境验证通过，并把 `pyAV` 解码参数一并固化（避开 torchcodec 兼容问题）。

> 📦 上游：[Hugging Face LeRobot](https://github.com/huggingface/lerobot)。本仓库刻意不整包 vendoring 上游源码——请按 [`docs/PIPELINE.md`](docs/PIPELINE.md) 中的上游版本与补丁方式使用。

---

## 🚀 快速开始

### 1. 硬件与运行环境

- **SO-101** 从臂：Feetech STS3215 串行舵机总线（SO-100 亦可），已完成标定
- USB 摄像头暴露为 `/dev/video2`（640×480 @ 30）
- USB **手柄**（双摇杆 + 按键；已在多款手柄上验证）

```bash
conda activate lerobot   # Python 3.12 + PyTorch/CUDA + feetech SDK + lerobot(源码安装)
```

> 注意：本环境统一使用 **pyAV** 编解码，不依赖默认 `torchcodec`，减少环境安装失败概率。

### 2. 用手柄采集 pick-and-place 数据集

```bash
bash scripts/record_pick_place.sh
```

手柄负责动作；键盘键位负责录制控制：
- `→` / `n`：提前结束当前 episode；`←` / `r`：重录；`Esc` / `q`：停止

### 3.（可选）逐条预览 / 剔除异常数据

```bash
python examples/dataset_episode_viewer.py \
  --repo-id zane/pick_place_block_clean \
  --root ./datasets/pick_place_block_clean --episode 0
```

### 4. 本地训练 ACT（RTX 3060 6GB）

```bash
bash scripts/train_act.sh
```

### 5. 真机评估

```bash
bash scripts/rollout_eval.sh
```

---

## 🗂 目录结构

```text
.
├── README.md / README.zh-CN.md   # 中英双语入口
├── media/
│   ├── demo.mp4                    # 真机推理视频（页面内可播）
│   └── demo1.gif                   # 轻量动态封面
├── docs/
│   ├── ARCHITECTURE.md             # 系统设计与遥操作细节
│   ├── PIPELINE.md                 # 上游版本 + 复现步骤
│   └── TROUBLESHOOTING.md          # 实际遇到的问题与修复
├── lerobot_mods/
│   ├── gamepad/                    # gamepad_joints 遥操作（配置/实现）
│   └── keyboard/                   # keyboard_joints 遥操作（配置/实现）
├── examples/
│   ├── dataset_episode_viewer.py   # mp4 预览，含关节曲线（pyAV）
│   └── gamepad_probe.py            # 手柄轴/键探测
└── scripts/
    ├── record_pick_place.sh
    ├── train_act.sh
    └── rollout_eval.sh
```

---

## 📐 主要设计取舍

- **策略 = ACT，输入 = 6 自由度状态 + 一路前置相机**：是低显存/快速迭代的甜点方案。
- **无主臂 ≠ 笛卡尔空间**：关节由摇杆做“速度式”指令，避免在 5-DOF 腕上做粗糙 IK 造成的不稳定。
- **数据质量 > 数据数量**：干净 48 条优于含噪 55 条；用 episode 级筛查 + 工具保证一致性。
- **统一 pyAV 解码**：规避 torchcodec/FFmpeg 环境坑，使训练与工具在同样数据上稳定一致。

---

## 📚 Roadmap（后续计划）

- [ ] 对 N≥10 次真机运行给出正式成功率统计
- [ ] 增加数据多样性（糖果朝向 / 位置 / 光照）采集，以缓解初测中观察到的泛化差距
- [ ] 将数据集与 checkpoint 上传 Hugging Face Hub，方便完全复现

---

## 🙏 致谢

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [SO-101 开源机械臂社区](https://github.com/TheRobotStudio/SO-ARM100)
- ACT: *Embodied AI via Action Chunking with Transformers*（Zhao et al., 2023）

## License

上游 LeRobot 为 Apache-2.0；本仓库原创代码与文档同样以 [Apache-2.0](LICENSE) 发布。
