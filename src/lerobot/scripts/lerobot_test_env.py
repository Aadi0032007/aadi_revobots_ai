#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Smoke-test the local environment: checks that the LeRobot install, PyTorch, CUDA and a real
policy train/inference step all work. Unlike `lerobot-info` (which only prints versions), every
check here actually executes something, and the command exits non-zero if a required check fails.

Examples:

```shell
lerobot-test-env                                  # full check on the auto-selected device
lerobot-test-env --device cuda:1                  # pin a specific GPU
lerobot-test-env --policy cortex_agv              # smoke-test another policy
lerobot-test-env --skip-policy                    # imports + CUDA only (fast)
lerobot-test-env --dataset lerobot/pusht          # also load a dataset and decode one frame
```
"""

import argparse
import importlib
import logging
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

# Policy types the generic (state + image + action) smoke-test batch is valid for.
SMOKE_TEST_POLICIES = ("act", "cortex_agv")

# Overrides needed to build a policy config that does not depend on external files.
POLICY_CONFIG_OVERRIDES = {
    # Discretized actions need a bins file produced by `compute_action_bins.py`.
    "cortex_agv": {"discretize_actions": False},
}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    required: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", required: bool = True) -> Check:
        check = Check(name, status, detail, required)
        self.checks.append(check)
        symbol = {PASS: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]", SKIP: "[SKIP]"}[status]
        print(f"{symbol} {name}" + (f": {detail}" if detail else ""), flush=True)
        return check

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL and c.required]


def run(report: Report, name: str, fn, required: bool = True):
    """Run a check function, turning any exception into a FAIL (or WARN if optional) entry.

    The check function returns either a detail string or a `(status, detail)` tuple, and may raise
    to signal failure. Returns the function's value on success, `None` otherwise.
    """
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001 - a diagnostic tool reports errors, it doesn't raise them
        report.add(name, FAIL if required else WARN, f"{type(e).__name__}: {e}", required)
        logging.debug(traceback.format_exc())
        return None

    if isinstance(result, tuple) and len(result) == 2 and result[0] in (PASS, FAIL, WARN, SKIP):
        status, detail = result
        report.add(name, status, detail, required)
        return None if status in (FAIL, SKIP) else result
    report.add(name, PASS, str(result) if result is not None else "", required)
    return result


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def check_python() -> str:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"LeRobot requires Python >= 3.12, found {platform.python_version()}")
    return f"{platform.python_version()} on {platform.platform()}"


def check_imports(report: Report) -> None:
    """Import the packages the rest of the checks (and training) rely on."""

    def _import(module_name: str, required: bool):
        def _check():
            module = importlib.import_module(module_name)
            return getattr(module, "__version__", "installed")

        run(report, f"import {module_name}", _check, required=required)

    for module_name in ("torch", "torchvision", "numpy", "huggingface_hub", "datasets", "draccus"):
        _import(module_name, required=True)
    for module_name in ("torchcodec", "transformers"):
        _import(module_name, required=False)


def check_lerobot_import() -> str:
    import lerobot
    from lerobot.policies.factory import get_policy_class  # noqa: F401
    from lerobot.utils.device_utils import auto_select_torch_device  # noqa: F401

    return f"{getattr(lerobot, '__version__', 'unknown')} from {lerobot.__file__}"


def check_torch_build() -> str:
    import torch

    built_cuda = torch.version.cuda or "CPU-only build"
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    return f"torch {torch.__version__}, built for CUDA {built_cuda}, cuDNN {cudnn}"


def check_cuda_available() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        hint = (
            "torch is a CPU-only build, reinstall with a CUDA wheel"
            if torch.version.cuda is None
            else "torch has CUDA support but no usable GPU/driver was found"
        )
        return FAIL, f"torch.cuda.is_available() is False ({hint})"
    return PASS, f"{torch.cuda.device_count()} device(s) visible"


def check_cuda_devices() -> str:
    import torch

    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        lines.append(
            f"cuda:{i} {props.name} | sm_{props.major}{props.minor} | "
            f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free"
        )
    return "; ".join(lines)


def resolve_device(requested: str | None) -> str:
    from lerobot.utils.device_utils import auto_select_torch_device

    return requested if requested else auto_select_torch_device().type


def check_device_resolution(device: str) -> str:
    from lerobot.utils.device_utils import get_safe_torch_device, is_amp_available

    resolved = get_safe_torch_device(device, log=False)
    return f"lerobot resolves '{device}' to {resolved} (AMP available: {is_amp_available(resolved.type)})"


def synchronize(device: str) -> None:
    import torch

    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def check_compute(device: str) -> str:
    """Multiply two matrices on the device and verify the result against the CPU."""
    import torch

    torch.manual_seed(0)
    size = 2048
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    expected = a @ b

    a_dev, b_dev = a.to(device), b.to(device)
    actual = (a_dev @ b_dev).cpu()
    if not torch.allclose(actual, expected, atol=1e-2, rtol=1e-3):
        max_err = (actual - expected).abs().max().item()
        raise RuntimeError(f"device matmul disagrees with CPU (max abs error {max_err:.3e})")

    # Rough throughput, after a warm-up so kernel autotuning is not part of the measurement.
    iterations = 20
    for _ in range(3):
        a_dev @ b_dev
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        a_dev @ b_dev
    synchronize(device)
    tflops = (2 * size**3 * iterations) / (time.perf_counter() - start) / 1e12
    return f"{size}x{size} matmul matches CPU, ~{tflops:.1f} TFLOP/s fp32"


def check_conv_backward(device: str) -> str:
    """Exercise the conv + autograd path (cuDNN on CUDA) that vision backbones depend on."""
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    ).to(device)
    loss = model(torch.randn(4, 3, 64, 64, device=device)).square().mean()
    loss.backward()
    synchronize(device)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("no gradients were produced by the backward pass")
    if any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("backward pass produced non-finite gradients")
    return f"conv+batchnorm forward/backward ok (loss {loss.item():.4f})"


def check_amp(device: str) -> tuple[str, str]:
    """Check autocast and the fp16 GradScaler used by `lerobot-train --policy.use_amp=true`."""
    import torch
    from torch import nn

    from lerobot.utils.device_utils import is_amp_available

    device_type = torch.device(device).type
    if not is_amp_available(device_type):
        return SKIP, f"AMP not available on '{device_type}'"

    dtypes = []
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    dtypes.append(torch.float16)

    model = nn.Linear(64, 64).to(device)
    x = torch.randn(8, 64, device=device)
    tested = []
    for dtype in dtypes:
        with torch.autocast(device_type=device_type, dtype=dtype):
            out = model(x)
        if not torch.isfinite(out).all():
            raise RuntimeError(f"autocast {dtype} produced non-finite values")
        tested.append(str(dtype).replace("torch.", ""))

    scaler = torch.amp.GradScaler(device_type)
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        loss = model(x).square().mean()
    scaler.scale(loss).backward()
    synchronize(device)
    return PASS, f"autocast ok for {', '.join(tested)}; GradScaler backward ok"


def check_video_backend() -> tuple[str, str]:
    from lerobot.utils.import_utils import get_safe_default_video_backend

    backend = get_safe_default_video_backend()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return WARN, f"decode backend '{backend}', but ffmpeg is not on PATH (needed to record/encode)"
    return PASS, f"decode backend '{backend}', ffmpeg at {ffmpeg}"


def build_smoke_policy(policy_type: str, device: str):
    """Build a small policy from scratch with synthetic features (no dataset, no Hub download)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    overrides = {
        "device": device,
        # Keep the smoke test offline and small: no pretrained backbone download.
        "pretrained_backbone_weights": None,
        "chunk_size": 4,
        "n_action_steps": 4,
        "dim_model": 64,
        "dim_feedforward": 128,
        "n_heads": 4,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            f"{OBS_IMAGE}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    }
    overrides.update(POLICY_CONFIG_OVERRIDES.get(policy_type, {}))

    # Drop any override the policy type does not define rather than failing the whole check.
    defaults = make_policy_config(policy_type)
    cfg = make_policy_config(policy_type, **{k: v for k, v in overrides.items() if hasattr(defaults, k)})

    return cfg, get_policy_class(policy_type)(cfg).to(device)


def build_smoke_batch(cfg, device: str) -> dict:
    import torch

    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    batch_size = 2
    chunk = getattr(cfg, "chunk_size", 1)
    return {
        OBS_STATE: torch.randn(batch_size, cfg.input_features[OBS_STATE].shape[0], device=device),
        f"{OBS_IMAGE}.top": torch.rand(
            batch_size, *cfg.input_features[f"{OBS_IMAGE}.top"].shape, device=device
        ),
        ACTION: torch.randn(batch_size, chunk, cfg.output_features[ACTION].shape[0], device=device),
        "action_is_pad": torch.zeros(batch_size, chunk, dtype=torch.bool, device=device),
    }


def check_policy_train_step(policy_type: str, device: str) -> str:
    """Build a policy, run forward + backward + an optimizer step, then an inference call."""
    import torch

    from lerobot.utils.constants import OBS_STR

    cfg, policy = build_smoke_policy(policy_type, device)
    batch = build_smoke_batch(cfg, device)
    n_params = sum(p.numel() for p in policy.parameters())

    policy.train()
    loss, loss_dict = policy.forward(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"policy loss is not finite: {loss}")
    loss.backward()

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    optimizer.step()
    optimizer.zero_grad()

    policy.reset()
    action = policy.select_action({k: v for k, v in batch.items() if k.startswith(OBS_STR)})
    if not torch.isfinite(action).all():
        raise RuntimeError("policy.select_action produced non-finite actions")
    synchronize(device)

    losses = ", ".join(f"{k}={v:.4f}" for k, v in loss_dict.items() if isinstance(v, float))
    return (
        f"{policy_type}: {n_params / 1e6:.1f}M params, train step ok (loss {loss.item():.4f}"
        + (f"; {losses}" if losses else "")
        + f"), select_action -> {tuple(action.shape)}"
    )


def check_memory_headroom(device: str) -> tuple[str, str]:
    import torch

    if torch.device(device).type != "cuda":
        return SKIP, "not a CUDA device"

    index = torch.device(device).index or 0
    peak = torch.cuda.max_memory_allocated(index)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(index)
    detail = (
        f"peak allocated {peak / 1024**2:.0f} MiB during checks, "
        f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free"
    )
    if free / total < 0.1:
        return WARN, detail + " - little headroom left for training"
    return PASS, detail


def check_dataset(repo_id: str) -> str:
    """Load a dataset and decode one frame end to end (needs network on first run)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id)
    item = dataset[0]
    shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in item.items() if hasattr(v, "shape") and v.ndim)
    return f"{repo_id}: {dataset.num_episodes} episodes, {dataset.num_frames} frames; frame 0 -> {shapes}"


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot-test-env",
        description="Check that the LeRobot environment, CUDA and a real policy step all work.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to test (e.g. 'cuda', 'cuda:1', 'cpu', 'mps'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--policy",
        default="act",
        help=f"Policy type for the train-step check. Known-good: {', '.join(SMOKE_TEST_POLICIES)}.",
    )
    parser.add_argument("--skip-policy", action="store_true", help="Skip the policy train-step check.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset repo_id to load and decode one frame from (requires network).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail (instead of warn) when no CUDA device is available.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tracebacks for failures.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    report = Report()

    print("== Environment ==", flush=True)
    run(report, "python version", check_python)
    check_imports(report)
    if run(report, "lerobot import", check_lerobot_import) is None:
        print("\nlerobot itself could not be imported, skipping the remaining checks.", flush=True)
        return 1
    run(report, "torch build", check_torch_build)

    print("\n== Accelerator ==", flush=True)
    wants_cuda = args.require_cuda or (args.device or "").startswith("cuda")
    if run(report, "cuda available", check_cuda_available, required=wants_cuda) is not None:
        run(report, "cuda devices", check_cuda_devices)

    device = resolve_device(args.device)
    print(f"-> testing on device: {device}", flush=True)
    if run(report, "device resolution", lambda: check_device_resolution(device)) is None:
        print("\nThe requested device is unusable, skipping the remaining checks.", flush=True)
        return 1
    run(report, "device compute", lambda: check_compute(device))
    run(report, "conv + autograd", lambda: check_conv_backward(device))
    run(report, "mixed precision", lambda: check_amp(device))

    print("\n== LeRobot stack ==", flush=True)
    run(report, "video backend", check_video_backend, required=False)

    if args.skip_policy:
        report.add("policy train step", SKIP, "--skip-policy", required=False)
    else:
        if args.policy not in SMOKE_TEST_POLICIES:
            print(
                f"note: '{args.policy}' is not in the known-good list {SMOKE_TEST_POLICIES}, "
                "the synthetic batch may not match its expected inputs.",
                flush=True,
            )
        run(report, "policy train step", lambda: check_policy_train_step(args.policy, device))

    if args.dataset:
        run(report, "dataset load", lambda: check_dataset(args.dataset), required=False)
    else:
        report.add("dataset load", SKIP, "pass --dataset <repo_id> to test", required=False)

    run(report, "memory headroom", lambda: check_memory_headroom(device), required=False)

    print("\n== Summary ==", flush=True)
    counts = {status: sum(c.status == status for c in report.checks) for status in (PASS, FAIL, WARN, SKIP)}
    print(
        f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[WARN]} warnings, {counts[SKIP]} skipped",
        flush=True,
    )

    failures = report.failed()
    if failures:
        for check in failures:
            print(f"  FAILED: {check.name}: {check.detail}", flush=True)
        if not args.verbose:
            print("  (re-run with --verbose for tracebacks)", flush=True)
        return 1

    print(f"Environment looks good on '{device}'.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
