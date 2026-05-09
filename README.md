<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0f,40:0d1f3c,80:0e3a5c,100:112244&height=220&section=header&text=aadi_revobots_ai&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Revobots%20×%20🤗%20LeRobot%20—%20AI%20for%20Task-Adaptive%20Robotics&descAlignY=60&descSize=17&animation=fadeIn" alt="banner"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LeRobot](https://img.shields.io/badge/🤗%20LeRobot-v0.5%2B-FF6B35?style=for-the-badge)](https://github.com/huggingface/lerobot)
[![ffmpeg](https://img.shields.io/badge/ffmpeg-7.1.1-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org)

[![Owner](https://img.shields.io/badge/Owner-Aditya%20Raj%20%40Aadi0032007-9B59B6?style=for-the-badge&logo=github)](https://github.com/Aadi0032007)
[![Revobots](https://img.shields.io/badge/Built%20for-REVOBOTS-00C9FF?style=for-the-badge)](https://revobots.ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

<br/>

> ### *"REVOlutionizing the Future of Work — one robot at a time."*
>
> **[REVOBOTS](https://revobots.ai)** · Phoenix, Arizona 🇺🇸  
> AI-driven, 3D-printed, task-adaptive robotics built for the real world.

<br/>

</div>

---

## 📖 What Is This Repository?

**`aadi_revobots_ai`** is the official AI training and robot-control stack for **[Revobots](https://revobots.ai)**, owned and maintained by **Aditya Raj ([@Aadi0032007](https://github.com/Aadi0032007))**.

It is built on top of [🤗 HuggingFace LeRobot](https://github.com/huggingface/lerobot) — the state-of-the-art open-source library for end-to-end robot learning in PyTorch — extended with custom robot interfaces, hardware drivers, and training configurations tuned specifically for the **Revobots hardware ecosystem**.

LeRobot provides:
- 🔌 A hardware-agnostic `Robot` base class — standardising control across any platform
- 🗄️ The **LeRobotDataset v3** format (Parquet + MP4) for scalable, HuggingFace Hub–hosted datasets
- 🧠 State-of-the-art policies for imitation learning, reinforcement learning, and Vision-Language-Action models
- 🛠️ A complete CLI pipeline for calibration, teleoperation, recording, training, and evaluation

Revobots builds on all of the above, adding the robot-specific glue that makes it work on **TASKBOT, TASKBOT AVA, TASKBOT SCOUT**, and the **Nero Arm**.

---

## 🤖 The Revobots Fleet

---

### 🦾 TASKBOT

> *The flagship — an AI-powered, 3D-printed humanoid built for adaptability, precision, and industrial-grade task execution.*

TASKBOT is Revobots' core platform: modular, 3D-printed, and designed for real-world tasks in demanding industrial environments. It is the primary training and deployment target for policies in this repo.

---

### 🤝 TASKBOT AVA

> *The collaborative humanoid — designed for human-adjacent environments.*

AVA is optimised for working alongside humans in service, light-industrial, and cooperative settings. Its human-in-the-loop capabilities make it a natural fit for HIL-SERL training and interactive data collection.

---

### 🔍 TASKBOT SCOUT

> *Autonomous inspection, surveillance, and field operations.*

SCOUT is the mobile form-factor platform built for reconnaissance, inspection, and surveying. Its camera-rich observation setup is well-suited for VLA (Vision-Language-Action) policy training and on-robot edge inference.

---

### 💪 Nero Arm

> *Revobots' precision robotic arm for dexterous manipulation.*

The Nero Arm is the dedicated manipulation platform for pick-and-place, assembly, and fine-motor tasks. Its well-defined joint-state observation and action spaces make it the ideal testbed for ACT and Diffusion policy training.

---

> 💡 All four platforms implement the **LeRobot `Robot` base interface**, enabling a single unified pipeline for teleoperation, dataset recording, policy training, and on-robot deployment across the entire fleet.

---

## 🧬 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       REVOBOTS AI STACK                                 │
│                       (aadi_revobots_ai)                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Revobots Hardware Layer                       │    │
│  │                                                                 │    │
│  │  ┌──────────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐   │    │
│  │  │  TASKBOT     │ │ TASKBOT    │ │ TASKBOT   │ │ Nero Arm  │   │    │
│  │  │  (Humanoid)  │ │ AVA        │ │ SCOUT     │ │ (Arm)     │   │    │
│  │  └──────┬───────┘ └─────┬──────┘ └─────┬─────┘ └─────┬─────┘   │    │
│  │         └───────────────┴──────────────┴─────────────┘          │    │
│  │                 All implement LeRobot BaseRobot                  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                       │
│  ┌──────────────────────────────▼──────────────────────────────────┐    │
│  │                🤗 LeRobot v0.5+ (HuggingFace)                   │    │
│  │                                                                 │    │
│  │  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐  │    │
│  │  │   Policies    │  │    Datasets     │  │  Robot Control   │  │    │
│  │  │               │  │                 │  │                  │  │    │
│  │  │ • ACT         │  │ • DS v3         │  │ • BaseRobot API  │  │    │
│  │  │ • Diffusion   │  │ • Parquet+MP4   │  │ • Teleop layer   │  │    │
│  │  │ • π₀ / FAST   │  │ • HF Hub sync   │  │ • Camera I/O     │  │    │
│  │  │ • SmolVLA     │  │ • Dataset tools │  │ • Sensor layers  │  │    │
│  │  │ • GR00T N1.5  │  │                 │  │                  │  │    │
│  │  │ • X-VLA       │  │                 │  │                  │  │    │
│  │  └───────────────┘  └─────────────────┘  └──────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Revobots adds on top of LeRobot

| Layer | What Revobots Contributes |
|---|---|
| **Robot Classes** | `Robot` implementations for TASKBOT, AVA, SCOUT, and Nero Arm |
| **Hardware Drivers** | Custom actuator, motor, and sensor drivers for Revobots hardware |
| **Configs** | Per-robot YAML hardware configurations |
| **Dataset Namespace** | `revobots/` on HuggingFace Hub |
| **Training Recipes** | Hyperparameters and policy configs tuned for the TASKBOT embodiment |
| **Edge Deployment** | On-robot inference pipeline for real-time task execution |

---

## 📦 Repository Structure

```
aditya/                                        ← Workspace root (always 'aditya')
│
└── aadi_revobots_ai/                          ← git clone https://github.com/Aadi0032007/aadi_revobots_ai.git
    ├── src/
    │   └── revobots/
    │       ├── robots/
    │       │   ├── taskbot.py                 ← TASKBOT Robot class
    │       │   ├── taskbot_ava.py             ← AVA Robot class
    │       │   ├── taskbot_scout.py           ← SCOUT Robot class
    │       │   └── nero_arm.py                ← Nero Arm Robot class
    │       ├── configs/
    │       │   ├── taskbot.yaml
    │       │   ├── taskbot_ava.yaml
    │       │   ├── taskbot_scout.yaml
    │       │   └── nero_arm.yaml
    │       ├── teleop/                        ← Teleoperation scripts
    │       └── utils/
    ├── examples/
    ├── pyproject.toml
    └── README.md
```

---

## ⚡ Setup & Installation

[![Installation Guide](https://img.shields.io/badge/📖%20Full%20Installation%20Guide-INSTALLATION.md-112244?style=for-the-badge&logo=github&logoColor=white)](./INSTALLATION.md)

---

## 🗂 Full Workflow: Hardware → Dataset → Policy → Deployment

```
╔══════════════════════════════════════════════════════════╗
║  STEP 1 — FIND & CALIBRATE                              ║
╚══════════════════════════════════════════════════════════╝

  lerobot-find-motors

  lerobot-calibrate \
    --robot.type=taskbot \
    --robot.port=/dev/ttyUSB0


╔══════════════════════════════════════════════════════════╗
║  STEP 2 — TELEOPERATE (verify hardware)                 ║
╚══════════════════════════════════════════════════════════╝

  lerobot-teleoperate \
    --robot.type=taskbot \
    --teleop.type=keyboard


╔══════════════════════════════════════════════════════════╗
║  STEP 3 — RECORD DEMONSTRATIONS                         ║
╚══════════════════════════════════════════════════════════╝

  lerobot-record \
    --robot.type=taskbot \
    --dataset.repo_id=revobots/taskbot_pick_place \
    --dataset.num_episodes=50 \
    --dataset.push_to_hub=true


╔══════════════════════════════════════════════════════════╗
║  STEP 4 — VISUALIZE & VALIDATE                          ║
╚══════════════════════════════════════════════════════════╝

  lerobot-visualize-dataset \
    --repo-id=revobots/taskbot_pick_place \
    --episode-index=0


╔══════════════════════════════════════════════════════════╗
║  STEP 5 — TRAIN POLICY                                  ║
╚══════════════════════════════════════════════════════════╝

  lerobot-train \
    --policy=act \
    --dataset.repo_id=revobots/taskbot_pick_place \
    --output_dir=outputs/train/taskbot_act


╔══════════════════════════════════════════════════════════╗
║  STEP 6 — EVALUATE ON ROBOT                             ║
╚══════════════════════════════════════════════════════════╝

  lerobot-record \
    --robot.type=taskbot \
    --policy.path=outputs/train/taskbot_act/last/pretrained_model \
    --dataset.num_episodes=10
```

---

## 🛠 Supported Policies

| Policy | Type | Revobots Use Case |
|---|---|---|
| **[ACT](https://huggingface.co/docs/lerobot/act)** | Action Chunking Transformer | Pick-and-place & assembly on Nero Arm |
| **Diffusion Policy** | Diffusion BC | High-precision multi-modal tasks on AVA |
| **[π₀ (Pi0)](https://huggingface.co/docs/lerobot/pi0)** | Vision-Language-Action | General instruction-following on TASKBOT |
| **[π₀-FAST](https://huggingface.co/docs/lerobot/pi0fast)** | Fast VLA | Low-latency inference on SCOUT |
| **[SmolVLA](https://huggingface.co/docs/lerobot/smolvla)** | Compact VLA | On-robot edge deployment |
| **[GR00T N1.5](https://huggingface.co/docs/lerobot/groot)** | NVIDIA Foundation Model | Humanoid generalisation on TASKBOT |
| **HIL-SERL** | Human-in-the-loop RL | Continuous improvement with AVA |

---

## 📡 Dataset Format

All demonstrations are stored in **LeRobotDataset v3** and hosted under the `revobots/` namespace on HuggingFace Hub.

```
revobots/taskbot_pick_place/
├── meta/info.json                          ← fps, shapes, feature types
├── data/chunk-000/episode_*.parquet        ← state, action, timestamps
└── videos/
    ├── observation.images.cam_high/        ← overhead RGB (MP4/episode)
    └── observation.images.cam_wrist/       ← wrist RGB (MP4/episode)
```

| Feature | Description |
|---|---|
| `observation.state` | Joint positions & velocities |
| `observation.images.cam_high` | Overhead RGB camera frame |
| `observation.images.cam_wrist` | Wrist-mounted camera frame |
| `action` | Target joint commands |
| `timestamp` | Synchronisation timestamp |
| `episode_index` | Episode identifier |
| `frame_index` | Frame index within episode |

---

## 🚀 Roadmap
- [x] Koch Leader Remote Teleoperator for TASKBOT
- [x] Nero Arm `Robot` class implementation
- [x] ACT training pipeline on Nero Arm pick-and-place
- [ ] TASKBOT `Robot` class implementation
- [ ] TASKBOT AVA `Robot` class implementation
- [ ] TASKBOT SCOUT `Robot` class implementation
- [ ] π₀ general instruction-following on TASKBOT
- [ ] SmolVLA edge inference pipeline for SCOUT
- [ ] Push `revobots/` datasets & checkpoints to HuggingFace Hub
- [ ] HIL-SERL continuous improvement loop with AVA

---

## 📚 References

| Resource | Link |
|---|---|
| 🤗 LeRobot | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 🤗 LeRobot Docs | [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot) |
| 🤗 Installation | [huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation) |
| 🤗 Bring Your Own Hardware | [huggingface.co/docs/lerobot/integrate_hardware](https://huggingface.co/docs/lerobot/integrate_hardware) |
| Revobots | [revobots.ai](https://revobots.ai) |
| TASKBOT | [revobots.ai/taskbot](https://revobots.ai/taskbot) |

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:112244,50:0e3a5c,100:0a0a0f&height=120&section=footer&animation=fadeIn" alt="footer"/>

**© 2026 Revobots · Aditya Raj ([@Aadi0032007](https://github.com/Aadi0032007)) · Built on 🤗 LeRobot**

*Transforming workforce automation with AI-driven, 3D-printed, task-adaptive robotics.*

</div>