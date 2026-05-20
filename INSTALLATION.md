<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0f,40:0d1f3c,80:0e3a5c,100:112244&height=180&section=header&text=Installation%20Guide&fontSize=44&fontColor=ffffff&fontAlignY=40&desc=aadi_revobots_ai%20×%20🤗%20LeRobot&descAlignY=62&descSize=18&animation=fadeIn" alt="banner"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LeRobot](https://img.shields.io/badge/🤗%20LeRobot-v0.5%2B-FF6B35?style=for-the-badge)](https://github.com/huggingface/lerobot)
[![ffmpeg](https://img.shields.io/badge/ffmpeg-conda--forge-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://anaconda.org/conda-forge/ffmpeg)
[![conda-forge](https://img.shields.io/badge/conda--forge-✓-44A833?style=for-the-badge&logo=condaforge&logoColor=white)](https://conda-forge.org)

<br/>

> 📖 **New here?** Read the [**→ Project README**](./README.md) first for an overview of the Revobots AI stack, the robot fleet, and the full workflow.

<br/>

</div>

---

## Overview

This guide sets up the complete **Revobots AI environment** — [`aadi_revobots_ai`](https://github.com/Aadi0032007/aadi_revobots_ai) on top of [🤗 LeRobot](https://github.com/huggingface/lerobot).

Two environment managers are supported. Choose one track and follow it end-to-end:

| | **conda** *(recommended)* | **uv** |
|---|---|---|
| Python | managed by conda | system Python ≥ 3.12 required |
| ffmpeg | installed inside the env | installed system-wide |
| Best for | most users, hardware development | fast installs, CI pipelines |

> **Workspace convention:** everything lives inside an `aditya/` parent folder.

```bash
mkdir aditya && cd aditya
```

---

## 🅰 conda Track

### Step 1 — Install Miniforge

> Skip if you already have `conda` or `mamba`.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Restart your shell after installation, or source your profile:

```bash
source ~/.bashrc    # Linux / WSL
source ~/.zshrc     # macOS zsh
```

---

### Step 2 — Create the Environment

Create the `aditya` environment with **Python 3.12** and **ffmpeg 7.1.1** in a single command:

```bash
conda create -y -n aditya -c conda-forge python=3.12 ffmpeg
```

> conda-forge will resolve the latest ffmpeg compatible with your platform. Verify `libsvtav1` is present after activation (see Step 2 verification below).

Activate the environment — repeat this every time you open a new shell:

```bash
conda activate aditya
```

Verify the install:

```bash
ffmpeg -version                 # confirm version
ffmpeg -encoders | grep svt    # libsvtav1 must appear
```

> **WSL (Windows Subsystem for Linux) users:** additionally install `evdev`:
> ```bash
> conda install evdev -c conda-forge
> ```

---

### Step 3 — Clone & Install `aadi_revobots_ai`

```bash
git clone https://github.com/Aadi0032007/aadi_revobots_ai.git
cd aadi_revobots_ai
pip install -e .
cd ..
```

---

### Step 4 — Optional Extras

Multiple extras can be combined (e.g. `.[feetech,aloha]`). Run from inside the `lerobot/` directory, or use `pip install 'lerobot[...]'` if you installed from PyPI.

```bash
pip install -e ".[feetech]"      # Feetech servo motors (SO-100, SO-101)
pip install -e ".[dynamixel]"    # Dynamixel motors (Koch v1.1)
pip install -e ".[aloha]"        # Aloha simulation environment
pip install -e ".[pusht]"        # PushT simulation environment
pip install -e ".[all]"          # All optional features
```

For a full list of available tags see: [pypi.org/project/lerobot](https://pypi.org/project/lerobot/)

---

### Step 5 — Experiment Tracking *(optional)*

```bash
wandb login
```

---

### Step 6 — Verify

```bash
python -c "import lerobot; print('✅ LeRobot:', lerobot.__version__)"
python -c "from revobots.robots.taskbot import TASKBOT; print('✅ Revobots SDK: OK')"
lerobot-info
```

---

## 🅱 uv Track

With `uv`, ffmpeg must be installed **system-wide** — `uv` and `torchcodec` dynamically link to the system binary rather than a conda-managed one.

---

### Step 1 — Install System ffmpeg

**Linux / WSL:**

```bash
sudo apt-get update
sudo apt-get install ffmpeg
ffmpeg -version                 # verify 7.x — see note below if 8.x appears
ffmpeg -encoders | grep svt    # libsvtav1 must appear
```

**macOS:**

```bash
brew install ffmpeg
ffmpeg -version
ffmpeg -encoders | grep svt
```

> ⚠️ **If your package manager installs ffmpeg 8.x:** LeRobot does not yet support it. On Linux, compile ffmpeg 7.x from source with `libsvtav1` following the [ffmpeg Ubuntu compilation guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), then point to the binary with `which ffmpeg`.

---

### Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc    # or restart your shell
```

---

### Step 3 — Create the Virtual Environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

---

### Step 4 — Install LeRobot 🤗

**From source** *(recommended)*:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
cd ..
```

**From PyPI:**

```bash
uv pip install lerobot
```

---

### Step 5 — Clone & Install `aadi_revobots_ai`

```bash
git clone https://github.com/Aadi0032007/aadi_revobots_ai.git
cd aadi_revobots_ai
uv pip install -e .
cd ..
```

---

### Step 6 — Optional Extras

```bash
uv pip install -e ".[feetech]"      # Feetech servo motors
uv pip install -e ".[dynamixel]"    # Dynamixel motors
uv pip install -e ".[aloha]"        # Aloha simulation environment
uv pip install -e ".[pusht]"        # PushT simulation environment
uv pip install -e ".[all]"          # All optional features
```

---

### Step 7 — Experiment Tracking *(optional)*

```bash
uv pip install wandb
wandb login
```

---

### Step 8 — Verify

```bash
python -c "import lerobot; print('✅ LeRobot:', lerobot.__version__)"
python -c "from revobots.robots.taskbot import TASKBOT; print('✅ Revobots SDK: OK')"
lerobot-info
```

---

## 🗂 Final Workspace Layout

After completing either track your `aditya/` workspace should look like this:

```
aditya/
├── lerobot/                          ← 🤗 LeRobot (HuggingFace upstream)
│   ├── src/lerobot/
│   │   ├── common/policies/          ← ACT, Diffusion, π₀, SmolVLA…
│   │   ├── common/datasets/          ← LeRobotDataset v3
│   │   └── common/robot_devices/     ← BaseRobot, cameras, motors
│   └── pyproject.toml
│
└── aadi_revobots_ai/                 ← Revobots AI stack (this repo)
    ├── src/revobots/
    │   ├── robots/                   ← TASKBOT, AVA, SCOUT, Nero Arm
    │   ├── configs/                  ← Per-robot YAML hardware configs
    │   └── teleop/                   ← Teleoperation scripts
    └── pyproject.toml
```

---

## 🔧 Troubleshooting

### Build errors on `pip install`

Install the required system libraries first:

```bash
# Linux / WSL
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

For other platforms see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

---

### `libsvtav1` not found

```bash
ffmpeg -encoders | grep svt
```

If nothing appears, your ffmpeg build is missing the encoder:

- **conda:** reinstall without a version pin — `conda install ffmpeg -c conda-forge`
- **uv / system:** compile ffmpeg from source — [libsvtav1 guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1)

---

### `lerobot-info` not found

Your environment is likely not activated, or LeRobot was not installed in editable mode:

```bash
conda activate aditya                 # conda
# or
source aditya/.venv/bin/activate     # uv

cd lerobot && pip install -e .
```

---

## 📚 References

| Resource | Link |
|---|---|
| 🤗 Official LeRobot Installation | [huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation) |
| 🤗 LeRobot GitHub | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 🤗 Bring Your Own Hardware | [huggingface.co/docs/lerobot/integrate_hardware](https://huggingface.co/docs/lerobot/integrate_hardware) |
| Miniforge | [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge) |
| uv | [astral.sh/uv](https://astral.sh/uv) |
| PyAV / ffmpeg | [pyav.org/docs](https://pyav.org/docs/develop/overview/installation.html) |
| `aadi_revobots_ai` | [github.com/Aadi0032007/aadi_revobots_ai](https://github.com/Aadi0032007/aadi_revobots_ai) |
| Revobots | [revobots.ai](https://revobots.ai) |

---

<div align="center">

<br/>

[![Back to README](https://img.shields.io/badge/←%20Back%20to-Project%20README-112244?style=for-the-badge&logo=github&logoColor=white)](./README.md)

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:112244,50:0e3a5c,100:0a0a0f&height=120&section=footer&animation=fadeIn" alt="footer"/>

**© 2025 Revobots · Aditya Raj ([@Aadi0032007](https://github.com/Aadi0032007)) · Built on 🤗 LeRobot**

</div>
