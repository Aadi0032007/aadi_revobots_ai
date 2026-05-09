<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0f,40:0d1f3c,80:0e3a5c,100:112244&height=160&section=header&text=Contributing&fontSize=40&fontColor=ffffff&fontAlignY=42&desc=aadi_revobots_ai%20×%20REVOBOTS&descAlignY=62&descSize=16&animation=fadeIn" alt="banner"/>

</div>

---

Thank you for your interest in contributing to **aadi_revobots_ai**. This project powers the AI stack for [Revobots](https://revobots.ai) robotic systems deployed in real customer environments. Contributions are welcome — and held to a high standard for that reason.

Please read this document fully before opening an issue or pull request.

> Also read: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) · [SECURITY.md](./SECURITY.md)

---

## 📋 Table of Contents

- [Who Can Contribute](#-who-can-contribute)
- [What You Can Contribute](#-what-you-can-contribute)
- [Before You Start](#-before-you-start)
- [Development Setup](#-development-setup)
- [Branching Strategy](#-branching-strategy)
- [Making Changes](#-making-changes)
- [Commit Messages](#-commit-messages)
- [Pull Request Process](#-pull-request-process)
- [Code Standards](#-code-standards)
- [Adding a New Robot](#-adding-a-new-robot)
- [Dataset Contributions](#-dataset-contributions)
- [What Not to Contribute](#-what-not-to-contribute)

---

## 👥 Who Can Contribute

This is an open-source project with a business mandate. Contributions are welcome from:

- **Revobots team members** — primary contributors with full write access
- **Community contributors** — external contributors via fork + pull request
- **Research partners** — academic or industry collaborators working with Revobots hardware

All contributors must agree to the [Code of Conduct](./CODE_OF_CONDUCT.md) and [License](./LICENSE) terms before contributing.

---

## 🔧 What You Can Contribute

| Type | Examples |
|---|---|
| **Bug fixes** | Incorrect joint mappings, broken teleoperation, dataset schema errors |
| **New robot support** | Implement the `Robot` interface for a new Revobots platform |
| **Policy improvements** | Tuning, new training configs, evaluation scripts |
| **Dataset tooling** | Scripts for recording, validating, or converting datasets |
| **Documentation** | Guides, docstrings, examples, tutorials |
| **Tests** | Unit tests, integration tests, hardware-in-the-loop test stubs |
| **Performance** | Inference speed, memory efficiency, data loading improvements |

---

## 🔍 Before You Start

1. **Search existing issues** — check that your bug or feature isn't already tracked
2. **Open an issue first** for non-trivial changes — get feedback before investing time in a PR
3. **For breaking changes** — always open an issue and get maintainer sign-off before writing code
4. **For security issues** — do **not** open a public issue; follow [SECURITY.md](./SECURITY.md) instead

---

## 💻 Development Setup

Follow [INSTALLATION.md](./INSTALLATION.md) to set up your environment, then:

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/aadi_revobots_ai.git
cd aadi_revobots_ai

# Add upstream remote
git remote add upstream https://github.com/Aadi0032007/aadi_revobots_ai.git

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## 🌿 Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code — protected |
| `dev` | Integration branch for active development |
| `feature/<name>` | New features or robot integrations |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation-only changes |
| `release/<version>` | Release preparation |

Always branch off `dev`, not `main`:

```bash
git checkout dev
git pull upstream dev
git checkout -b feature/your-feature-name
```

---

## ✏️ Making Changes

- Keep changes focused — one feature or fix per PR
- Write or update tests for any code you change
- Update relevant documentation and docstrings
- For robot interface changes, update the corresponding config YAML
- For breaking changes to dataset schemas or robot APIs, clearly document the migration path in the PR description

---

## 💬 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Types:**

| Type | Use for |
|---|---|
| `feat` | New feature or robot support |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure, no behaviour change |
| `test` | Adding or updating tests |
| `perf` | Performance improvement |
| `chore` | Build, config, dependency updates |
| `ci` | CI/CD pipeline changes |

**Examples:**

```
feat(robots): add TASKBOT SCOUT Robot class implementation
fix(nero_arm): correct joint velocity limit in config
docs(installation): update uv track for ffmpeg system install
```

---

## 🔁 Pull Request Process

1. **Ensure your branch is up to date** with `dev` before opening a PR
2. **Fill out the PR template** completely — do not delete sections
3. **Link the related issue** using `Closes #<issue-number>`
4. **All checks must pass** — linting, formatting, and tests
5. **At least one maintainer review** is required before merge
6. **Breaking changes** require sign-off from [@Aadi0032007](https://github.com/Aadi0032007)
7. **Do not merge your own PRs** without review — no exceptions for production-affecting code

### PR Title Format

Follow the same Conventional Commits format as commit messages:

```
feat(robots): add TASKBOT AVA calibration support
```

---

## 🧹 Code Standards

### Style

- **Python 3.12+**
- [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting (replaces black + flake8)
- Type hints on all public functions and class methods
- Docstrings on all public classes and functions (Google style)

```bash
ruff check .          # lint
ruff format .         # format
```

### Tests

```bash
pytest tests/         # run all tests
pytest tests/robots/  # run robot-specific tests
```

### Pre-commit

All hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

---

## 🤖 Adding a New Robot

To add support for a new Revobots hardware platform, implement the LeRobot `Robot` base interface:

```python
# src/revobots/robots/my_robot.py

from lerobot.robots import Robot
from dataclasses import dataclass

@dataclass
class MyRobotConfig:
    port: str = "/dev/ttyUSB0"
    # ... hardware-specific config fields

class MyRobot(Robot):
    def __init__(self, config: MyRobotConfig):
        self.config = config

    def connect(self) -> None:
        """Establish connection to hardware."""
        ...

    def disconnect(self) -> None:
        """Safely disconnect from hardware."""
        ...

    def get_observation(self) -> dict:
        """Return current sensor observations."""
        ...

    def send_action(self, action) -> None:
        """Send joint commands to actuators."""
        ...
```

Also provide:
- A config YAML at `src/revobots/configs/my_robot.yaml`
- Unit tests at `tests/robots/test_my_robot.py`
- Documentation in the PR description

---

## 📡 Dataset Contributions

If contributing demonstration datasets:

- Use the **LeRobotDataset v3** format (Parquet + MP4)
- Host datasets under the `revobots/` namespace on HuggingFace Hub
- Include a `meta/info.json` with correct fps, shapes, and feature descriptions
- Do **not** include personally identifiable information or proprietary customer data in any dataset
- Verify your dataset with `lerobot-visualize-dataset` before submitting

---

## 🚫 What Not to Contribute

- **Customer-specific code or configs** that contain proprietary information
- **Hardcoded credentials**, API keys, or access tokens — use environment variables
- **Untested changes** to safety-critical paths (joint limits, emergency stop, velocity caps)
- **Large binary files** — use Git LFS or HuggingFace Hub for datasets and model weights
- **Code with unclear licensing** — all contributions must be compatible with the project license

---

## 📬 Contact

For questions not covered here, reach out to the maintainer:

**Aditya Raj** · [@Aadi0032007](https://github.com/Aadi0032007)  
📧 [ms.adityaraj@gmail.com](mailto:ms.adityaraj@gmail.com)

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:112244,50:0e3a5c,100:0a0a0f&height=100&section=footer&animation=fadeIn" alt="footer"/>

**© 2025 Revobots · [revobots.ai](https://revobots.ai)**

</div>