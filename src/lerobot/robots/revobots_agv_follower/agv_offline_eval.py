# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:46:32 2026

@author: Aadi
"""
from __future__ import annotations


"""
agv_offline_eval.py — offline policy evaluation.

Two input modes (mutually exclusive):

  1. --parent-dir <path>
     Walks every session_<i>/ folder (build_lerobot_dataset.py layout),
     reconstructs each per-frame observation, runs the policy, and compares
     predicted (lin_x, ang_z) against the recorded action in the JSONL.

  2. --dataset-repo-id-eval <repo>
     Downloads a LeRobot dataset from HF and iterates it episode-by-episode,
     frame-by-frame. Ground-truth actions come from the dataset's `action`
     field; observations are read from the dataset (image + state) and fed
     to the policy. No session_<i>/ folders required.

At the end you get a per-episode/session breakdown, an overall summary
(MAE, RMSE, Pearson r, R², sign agreement, error ranges), a
turning/direction diagnostic block, and a top-K worst-frames table.

Defaults:
  --value-tol  = 0.05   (magnitude accuracy tolerance)
  --zero-eps   = 0.025  (±0.025 counts as straight / stopped)

Usage
-----
    # From session folders on disk:
    python3 agv_offline_eval.py \\
        --parent-dir /path/to/sessions \\
        --policy-path /home/elephant/policies/agv_v1 \\
        --dataset-repo-id user/revobots_agv_v1 \\
        --device cuda

    # From a LeRobot dataset on HF:
    python3 agv_offline_eval.py \\
        --dataset-repo-id-eval user/revobots_agv_v1_eval \\
        --policy-path /home/elephant/policies/agv_v1 \\
        --dataset-repo-id user/revobots_agv_v1 \\
        --device cuda

    # Save per-frame CSV:
    python3 agv_offline_eval.py ... --csv-out eval.csv

    # Limit episodes:
    python3 agv_offline_eval.py ... --max-sessions 5
    python3 agv_offline_eval.py ... --sessions session_3 session_7
    python3 agv_offline_eval.py ... --episodes 0 4 9
"""


import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Make `import LAB.xxx` work no matter where this is launched from
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# LeRobot pieces — import paths for the current lerobot version
from lerobot.common.control_utils       import predict_action
from lerobot.configs.policies           import PreTrainedConfig
from lerobot.datasets                   import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory           import make_policy, make_pre_post_processors
from lerobot.policies.utils             import make_robot_action
from lerobot.processor                  import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants            import ACTION, OBS_STR
from lerobot.utils.device_utils         import get_safe_torch_device
from lerobot.utils.feature_utils        import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.utils                import init_logging


# ── Policy pipeline (inlined from the reference callable-inference API) ──

def _resolve_policy_dir(root: str, subfolder: Optional[str],
                        checkpoint_hint: Optional[str] = None) -> str:
    """Return the directory that actually contains config.json.

    Handles three layouts:
      A) config.json at snapshot root (flat push)
      B) pretrained_model/config.json at root (single-checkpoint export)
      C) checkpoints/<N>/pretrained_model/config.json (multi-checkpoint repo)

    Resolution order:
      1. --policy-subfolder if given (explicit).
      2. checkpoints/<checkpoint_hint>/pretrained_model/  (when a revision
         was requested — HF may have ignored it as a branch and just given
         us the whole tree; the hint is the checkpoint number to pick).
      3. config.json at root.
      4. pretrained_model/ (and a couple of siblings) at root.
      5. Recursive search for config.json, up to a shallow depth.
    """
    root_p = Path(root)

    if subfolder:
        candidate = root_p / subfolder
        if not (candidate / "config.json").is_file():
            raise FileNotFoundError(
                f"--policy-subfolder='{subfolder}' → {candidate} has no config.json"
            )
        return str(candidate)

    # Requested checkpoint number, if any — pick that checkpoint from the tree.
    if checkpoint_hint:
        for cand in (
            root_p / "checkpoints" / checkpoint_hint / "pretrained_model",
            root_p / "checkpoints" / checkpoint_hint,
            root_p / checkpoint_hint / "pretrained_model",
            root_p / checkpoint_hint,
        ):
            if (cand / "config.json").is_file():
                print(f"[*] policy config found: {cand.relative_to(root_p)}/")
                return str(cand)

    if (root_p / "config.json").is_file():
        return str(root_p)

    for name in ("pretrained_model", "pretrained", "model"):
        cand = root_p / name
        if (cand / "config.json").is_file():
            print(f"[*] policy config found in subfolder: {name}/")
            return str(cand)

    # Shallow recursive search — up to 3 levels deep, avoids scanning weights.
    hits = []
    for p in root_p.rglob("config.json"):
        try:
            depth = len(p.relative_to(root_p).parts)
        except ValueError:
            continue
        if depth <= 4 and p.is_file():
            hits.append(p.parent)
    if len(hits) == 1:
        rel = hits[0].relative_to(root_p)
        print(f"[*] policy config found: {rel}/")
        return str(hits[0])
    if len(hits) > 1:
        listing = "\n    ".join(str(h.relative_to(root_p)) for h in sorted(hits))
        raise FileNotFoundError(
            f"Multiple config.json candidates under {root_p}:\n    {listing}\n"
            f"Pass --policy-subfolder=<relative-path> to pick one."
        )

    listing = "\n    ".join(sorted(p.name for p in root_p.iterdir()))
    raise FileNotFoundError(
        f"No config.json found under {root_p}. Contents:\n    {listing}\n"
        f"Pass --policy-subfolder=<name> to point at the checkpoint dir."
    )


def build_policy_pipeline(
    policy_path:      str,
    dataset_repo_id:  str,
    device:           str = "cuda",
    rename_map:       Optional[dict] = None,
):
    """Builds policy + processors for offline eval. Mirrors the reference
    build_policy_pipeline from the current lerobot version."""
    if rename_map is None:
        rename_map = {}

    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device          = device
    policy_cfg.pretrained_path = Path(policy_path)

    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg      = policy_cfg,
        pretrained_path = policy_path,
        dataset_stats   = rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides = {
            "device_processor":              {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    return (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        ds_meta.features,
    )


# ── Dataset feature schema (mirrors build_lerobot_dataset.py exactly) ────────

CAMERA_KEY    = "front"
CAMERA_HEIGHT = 480
CAMERA_WIDTH  = 640

ROBOT_ACTION_FEATURES = {
    "lin_x": float,
    "ang_z": float,
}

ROBOT_OBSERVATION_FEATURES = {
    "lin_x": float,
    "ang_z": float,
    "lat":   float,
    "long":  float,
    "orientation": float,
    CAMERA_KEY: (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
}


def build_dataset_features() -> dict:
    action_features = hw_to_dataset_features(ROBOT_ACTION_FEATURES, ACTION)
    obs_features    = hw_to_dataset_features(ROBOT_OBSERVATION_FEATURES, OBS_STR)
    return {**action_features, **obs_features}


def row_to_raw_obs_and_action(row: dict, image_rgb: np.ndarray) -> tuple[dict, dict]:
    """Same function as build_lerobot_dataset.row_to_raw_obs_and_action.
    Inlined here so this script has zero coupling to the converter file."""
    lin_x = float(row.get("linear_velocity",  0.0))
    ang_z = float(row.get("angular_velocity", 0.0))
    lat   = float(row.get("gps_latitude",     0.0))
    lon   = float(row.get("gps_longitude",    0.0))
    ori   = float(row.get("orientation",      0.0))

    raw_action = {"lin_x": lin_x, "ang_z": ang_z}
    raw_observation = {
        "lin_x":       lin_x,
        "ang_z":       ang_z,
        "lat":         lat,
        "long":        lon,
        "orientation": ori,
        CAMERA_KEY:    image_rgb,
    }
    return raw_observation, raw_action


# ── Session discovery ───────────────────────────────────────────────────────

def find_session_folders(input_root: Path) -> list[Path]:
    sessions = []
    for sub in sorted(input_root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if (sub / f"{name}.mp4").is_file() and (sub / f"{name}.jsonl").is_file():
            sessions.append(sub)
    return sessions


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    session:  str
    frame:    int
    gt_lin:   float
    gt_ang:   float
    pred_lin: float
    pred_ang: float

    @property
    def err_lin(self) -> float: return self.pred_lin - self.gt_lin
    @property
    def err_ang(self) -> float: return self.pred_ang - self.gt_ang


@dataclass
class ComponentMetrics:
    """Per-component metrics for either lin_x or ang_z."""
    name:     str
    n:        int   = 0
    mae:      float = 0.0
    rmse:     float = 0.0
    max_err:  float = 0.0
    pearson:  float = 0.0
    r2:       float = 0.0
    sign_agreement: float = 0.0
    accuracy_within_tol: float = 0.0
    value_tol: float = 0.0
    gt_min:   float = 0.0
    gt_max:   float = 0.0
    pred_min: float = 0.0
    pred_max: float = 0.0

    @classmethod
    def compute(
        cls,
        name: str,
        gt: np.ndarray,
        pred: np.ndarray,
        value_tol: float = 0.05,
        zero_eps: float = 1e-3,
    ) -> "ComponentMetrics":
        if gt.size == 0:
            return cls(name=name)
        err   = pred - gt
        absE  = np.abs(err)
        mae   = float(absE.mean())
        rmse  = float(np.sqrt((err ** 2).mean()))
        maxE  = float(absE.max())

        # Pearson correlation — undefined when either side has no variance.
        gt_std   = float(gt.std())
        pred_std = float(pred.std())
        if gt_std > 1e-9 and pred_std > 1e-9:
            pearson = float(np.corrcoef(gt, pred)[0, 1])
        else:
            pearson = float("nan")

        # R² = 1 - SS_res / SS_tot. If gt variance is ~0, R² is undefined.
        ss_tot = float(((gt - gt.mean()) ** 2).sum())
        ss_res = float((err ** 2).sum())
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        # Sign agreement — consider only frames where at least one side is
        # non-trivial. Avoids both being ~0 inflating the score.
        mask = (np.abs(gt) > zero_eps) | (np.abs(pred) > zero_eps)
        if mask.any():
            sign_agree = float(
                (np.sign(gt[mask]) == np.sign(pred[mask])).mean()
            )
        else:
            sign_agree = float("nan")

        # Percentage-style regression accuracy: prediction is counted correct
        # when it is within +/- value_tol of the ground truth.
        accuracy_within_tol = float((absE <= value_tol).mean())

        return cls(
            name=name, n=int(gt.size),
            mae=mae, rmse=rmse, max_err=maxE,
            pearson=pearson, r2=r2,
            sign_agreement=sign_agree,
            accuracy_within_tol=accuracy_within_tol,
            value_tol=float(value_tol),
            gt_min=float(gt.min()),   gt_max=float(gt.max()),
            pred_min=float(pred.min()), pred_max=float(pred.max()),
        )


@dataclass
class TurningMetrics:
    """Classification-style metrics for angular turning vs straight driving.

    A frame is "turning" when abs(ang_z) > zero_eps.
    A frame is "straight" when abs(ang_z) <= zero_eps.
    """
    n: int = 0
    zero_eps: float = 1e-3
    gt_turning_n: int = 0
    gt_straight_n: int = 0
    pred_turning_n: int = 0
    pred_straight_n: int = 0
    turning_match_n: int = 0
    straight_match_n: int = 0
    missed_turn_n: int = 0       # gt turning, pred straight
    false_turn_n: int = 0        # gt straight, pred turning
    turning_match_accuracy: float = 0.0
    straight_match_accuracy: float = 0.0
    overall_state_accuracy: float = 0.0

    @classmethod
    def compute(
        cls,
        gt_ang: np.ndarray,
        pred_ang: np.ndarray,
        zero_eps: float = 1e-3,
    ) -> "TurningMetrics":
        if gt_ang.size == 0:
            return cls(zero_eps=float(zero_eps))

        gt_turning = np.abs(gt_ang) > zero_eps
        pred_turning = np.abs(pred_ang) > zero_eps
        gt_straight = ~gt_turning
        pred_straight = ~pred_turning

        turning_match_n = int((gt_turning & pred_turning).sum())
        straight_match_n = int((gt_straight & pred_straight).sum())
        gt_turning_n = int(gt_turning.sum())
        gt_straight_n = int(gt_straight.sum())

        turning_match_accuracy = (
            turning_match_n / gt_turning_n if gt_turning_n else float("nan")
        )
        straight_match_accuracy = (
            straight_match_n / gt_straight_n if gt_straight_n else float("nan")
        )
        overall_state_accuracy = float((gt_turning == pred_turning).mean())

        return cls(
            n=int(gt_ang.size),
            zero_eps=float(zero_eps),
            gt_turning_n=gt_turning_n,
            gt_straight_n=gt_straight_n,
            pred_turning_n=int(pred_turning.sum()),
            pred_straight_n=int(pred_straight.sum()),
            turning_match_n=turning_match_n,
            straight_match_n=straight_match_n,
            missed_turn_n=int((gt_turning & pred_straight).sum()),
            false_turn_n=int((gt_straight & pred_turning).sum()),
            turning_match_accuracy=float(turning_match_accuracy),
            straight_match_accuracy=float(straight_match_accuracy),
            overall_state_accuracy=overall_state_accuracy,
        )


@dataclass
class AngularDirectionMetrics:
    """Direction-specific metrics for ang_z.

    This avoids the old misleading sign score. It separates:
      1) Did the model detect turning vs straight?
      2) When GT and prediction are both turning, did it choose the same sign?
      3) How many left/right confusions happened?
      4) Is the magnitude close when the direction is correct?
    """
    n: int = 0
    zero_eps: float = 1e-3
    value_tol: float = 0.05

    gt_left_n: int = 0
    gt_right_n: int = 0
    gt_straight_n: int = 0
    pred_left_n: int = 0
    pred_right_n: int = 0
    pred_straight_n: int = 0

    both_turning_n: int = 0
    correct_direction_n: int = 0
    opposite_direction_n: int = 0
    gt_left_pred_right_n: int = 0
    gt_right_pred_left_n: int = 0
    gt_left_pred_straight_n: int = 0
    gt_right_pred_straight_n: int = 0
    gt_straight_pred_left_n: int = 0
    gt_straight_pred_right_n: int = 0

    direction_accuracy_when_both_turn: float = 0.0
    direction_recall_on_gt_turns: float = 0.0
    turn_precision: float = 0.0
    left_recall: float = 0.0
    right_recall: float = 0.0
    left_precision: float = 0.0
    right_precision: float = 0.0
    magnitude_acc_when_correct_dir: float = 0.0
    mae_when_correct_dir: float = 0.0
    mae_when_wrong_dir: float = 0.0

    @classmethod
    def compute(
        cls,
        gt_ang: np.ndarray,
        pred_ang: np.ndarray,
        zero_eps: float = 1e-3,
        value_tol: float = 0.05,
    ) -> "AngularDirectionMetrics":
        if gt_ang.size == 0:
            return cls(zero_eps=float(zero_eps), value_tol=float(value_tol))

        gt_left = gt_ang > zero_eps
        gt_right = gt_ang < -zero_eps
        gt_straight = ~(gt_left | gt_right)

        pred_left = pred_ang > zero_eps
        pred_right = pred_ang < -zero_eps
        pred_straight = ~(pred_left | pred_right)

        gt_turn = gt_left | gt_right
        pred_turn = pred_left | pred_right
        both_turn = gt_turn & pred_turn
        correct_dir = (gt_left & pred_left) | (gt_right & pred_right)
        opposite_dir = (gt_left & pred_right) | (gt_right & pred_left)

        gt_turn_n = int(gt_turn.sum())
        pred_turn_n = int(pred_turn.sum())
        both_turning_n = int(both_turn.sum())
        correct_direction_n = int(correct_dir.sum())
        opposite_direction_n = int(opposite_dir.sum())

        gt_left_n = int(gt_left.sum())
        gt_right_n = int(gt_right.sum())
        pred_left_n = int(pred_left.sum())
        pred_right_n = int(pred_right.sum())

        # Direction accuracy only among frames where both GT and prediction are non-zero.
        direction_accuracy_when_both_turn = (
            correct_direction_n / both_turning_n if both_turning_n else float("nan")
        )

        # Recall on all GT turns: count correct left/right direction, not merely non-zero prediction.
        direction_recall_on_gt_turns = (
            correct_direction_n / gt_turn_n if gt_turn_n else float("nan")
        )

        # Of all predicted turns, how many were real turns with the correct sign?
        turn_precision = (
            correct_direction_n / pred_turn_n if pred_turn_n else float("nan")
        )

        left_recall = int((gt_left & pred_left).sum()) / gt_left_n if gt_left_n else float("nan")
        right_recall = int((gt_right & pred_right).sum()) / gt_right_n if gt_right_n else float("nan")
        left_precision = int((gt_left & pred_left).sum()) / pred_left_n if pred_left_n else float("nan")
        right_precision = int((gt_right & pred_right).sum()) / pred_right_n if pred_right_n else float("nan")

        abs_err = np.abs(pred_ang - gt_ang)
        magnitude_acc_when_correct_dir = (
            float((abs_err[correct_dir] <= value_tol).mean()) if correct_direction_n else float("nan")
        )
        mae_when_correct_dir = float(abs_err[correct_dir].mean()) if correct_direction_n else float("nan")
        mae_when_wrong_dir = float(abs_err[opposite_dir].mean()) if opposite_direction_n else float("nan")

        return cls(
            n=int(gt_ang.size),
            zero_eps=float(zero_eps),
            value_tol=float(value_tol),
            gt_left_n=gt_left_n,
            gt_right_n=gt_right_n,
            gt_straight_n=int(gt_straight.sum()),
            pred_left_n=pred_left_n,
            pred_right_n=pred_right_n,
            pred_straight_n=int(pred_straight.sum()),
            both_turning_n=both_turning_n,
            correct_direction_n=correct_direction_n,
            opposite_direction_n=opposite_direction_n,
            gt_left_pred_right_n=int((gt_left & pred_right).sum()),
            gt_right_pred_left_n=int((gt_right & pred_left).sum()),
            gt_left_pred_straight_n=int((gt_left & pred_straight).sum()),
            gt_right_pred_straight_n=int((gt_right & pred_straight).sum()),
            gt_straight_pred_left_n=int((gt_straight & pred_left).sum()),
            gt_straight_pred_right_n=int((gt_straight & pred_right).sum()),
            direction_accuracy_when_both_turn=float(direction_accuracy_when_both_turn),
            direction_recall_on_gt_turns=float(direction_recall_on_gt_turns),
            turn_precision=float(turn_precision),
            left_recall=float(left_recall),
            right_recall=float(right_recall),
            left_precision=float(left_precision),
            right_precision=float(right_precision),
            magnitude_acc_when_correct_dir=float(magnitude_acc_when_correct_dir),
            mae_when_correct_dir=float(mae_when_correct_dir),
            mae_when_wrong_dir=float(mae_when_wrong_dir),
        )


@dataclass
class LinZeroMetrics:
    """Classification-style metrics for lin_x being zero vs non-zero.

    A frame is "lin_x zero" when abs(lin_x) <= zero_eps.
    This answers: when gt lin_x is 0, is predicted lin_x also 0?
    """
    n: int = 0
    zero_eps: float = 1e-3
    gt_zero_n: int = 0
    gt_nonzero_n: int = 0
    pred_zero_n: int = 0
    pred_nonzero_n: int = 0
    zero_match_n: int = 0
    nonzero_match_n: int = 0
    missed_zero_n: int = 0      # gt zero, pred non-zero
    false_zero_n: int = 0       # gt non-zero, pred zero
    zero_match_accuracy: float = 0.0
    nonzero_match_accuracy: float = 0.0
    overall_state_accuracy: float = 0.0

    @classmethod
    def compute(
        cls,
        gt_lin: np.ndarray,
        pred_lin: np.ndarray,
        zero_eps: float = 1e-3,
    ) -> "LinZeroMetrics":
        if gt_lin.size == 0:
            return cls(zero_eps=float(zero_eps))

        gt_zero = np.abs(gt_lin) <= zero_eps
        pred_zero = np.abs(pred_lin) <= zero_eps
        gt_nonzero = ~gt_zero
        pred_nonzero = ~pred_zero

        zero_match_n = int((gt_zero & pred_zero).sum())
        nonzero_match_n = int((gt_nonzero & pred_nonzero).sum())
        gt_zero_n = int(gt_zero.sum())
        gt_nonzero_n = int(gt_nonzero.sum())

        zero_match_accuracy = (
            zero_match_n / gt_zero_n if gt_zero_n else float("nan")
        )
        nonzero_match_accuracy = (
            nonzero_match_n / gt_nonzero_n if gt_nonzero_n else float("nan")
        )
        overall_state_accuracy = float((gt_zero == pred_zero).mean())

        return cls(
            n=int(gt_lin.size),
            zero_eps=float(zero_eps),
            gt_zero_n=gt_zero_n,
            gt_nonzero_n=gt_nonzero_n,
            pred_zero_n=int(pred_zero.sum()),
            pred_nonzero_n=int(pred_nonzero.sum()),
            zero_match_n=zero_match_n,
            nonzero_match_n=nonzero_match_n,
            missed_zero_n=int((gt_zero & pred_nonzero).sum()),
            false_zero_n=int((gt_nonzero & pred_zero).sum()),
            zero_match_accuracy=float(zero_match_accuracy),
            nonzero_match_accuracy=float(nonzero_match_accuracy),
            overall_state_accuracy=overall_state_accuracy,
        )


@dataclass
class SessionResult:
    name:       str
    n_frames:   int
    lin_metric: ComponentMetrics
    ang_metric: ComponentMetrics
    turning_metric: TurningMetrics
    angular_direction_metric: AngularDirectionMetrics
    lin_zero_metric: LinZeroMetrics
    records:    list[FrameRecord] = field(default_factory=list)

    @property
    def combined_rmse(self) -> float:
        return float(np.sqrt((self.lin_metric.rmse ** 2 + self.ang_metric.rmse ** 2) / 2.0))


# ── Common eval helpers ─────────────────────────────────────────────────────

def _tensor_to_hwc_rgb_uint8(x) -> np.ndarray:
    """LeRobotDataset returns images as torch tensors, usually float CxHxW in [0,1].
    Convert to HxWx3 uint8 RGB so it matches the shape build_lerobot_dataset
    stored (what the policy was trained on)."""
    import torch
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        # CxHxW -> HxWxC
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _finalize_session_result(
    name: str,
    gt_lin_buf: np.ndarray,
    gt_ang_buf: np.ndarray,
    pred_lin_buf: np.ndarray,
    pred_ang_buf: np.ndarray,
    records: list[FrameRecord],
    n_done: int,
    value_tol: float,
    zero_eps: float,
) -> SessionResult:
    gt_lin_buf   = gt_lin_buf[:n_done]
    gt_ang_buf   = gt_ang_buf[:n_done]
    pred_lin_buf = pred_lin_buf[:n_done]
    pred_ang_buf = pred_ang_buf[:n_done]

    lin_m = ComponentMetrics.compute("lin_x", gt_lin_buf, pred_lin_buf,
                                     value_tol=value_tol, zero_eps=zero_eps)
    ang_m = ComponentMetrics.compute("ang_z", gt_ang_buf, pred_ang_buf,
                                     value_tol=value_tol, zero_eps=zero_eps)
    turning_m = TurningMetrics.compute(gt_ang_buf, pred_ang_buf, zero_eps=zero_eps)
    angular_direction_m = AngularDirectionMetrics.compute(
        gt_ang_buf, pred_ang_buf, zero_eps=zero_eps, value_tol=value_tol
    )
    lin_zero_m = LinZeroMetrics.compute(gt_lin_buf, pred_lin_buf, zero_eps=zero_eps)
    return SessionResult(name=name, n_frames=n_done,
                         lin_metric=lin_m, ang_metric=ang_m,
                         turning_metric=turning_m,
                         angular_direction_metric=angular_direction_m,
                         lin_zero_metric=lin_zero_m,
                         records=records)


# ── Per-session evaluation (session_<i>/ folder mode) ───────────────────────

def eval_one_session(
    session_dir:                Path,
    policy,
    preprocessor,
    postprocessor,
    robot_observation_processor,
    features:                   dict,
    device,
    use_amp:                    bool,
    robot_type:                 str,
    single_task:                Optional[str],
    reset_policy_each_session:  bool,
    value_tol:                  float,
    zero_eps:                   float,
) -> SessionResult:

    name       = session_dir.name
    mp4_path   = session_dir / f"{name}.mp4"
    jsonl_path = session_dir / f"{name}.jsonl"
    rows       = load_jsonl(jsonl_path)

    if reset_policy_each_session:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[!] could not open {mp4_path} — skipping")
        return SessionResult(name=name, n_frames=0,
                             lin_metric=ComponentMetrics(name="lin_x"),
                             ang_metric=ComponentMetrics(name="ang_z"),
                             turning_metric=TurningMetrics(zero_eps=zero_eps),
                             angular_direction_metric=AngularDirectionMetrics(zero_eps=zero_eps, value_tol=value_tol),
                             lin_zero_metric=LinZeroMetrics(zero_eps=zero_eps))

    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames     = min(total_frames_vid, len(rows))

    gt_lin_buf   = np.zeros(total_frames, dtype=np.float64)
    gt_ang_buf   = np.zeros(total_frames, dtype=np.float64)
    pred_lin_buf = np.zeros(total_frames, dtype=np.float64)
    pred_ang_buf = np.zeros(total_frames, dtype=np.float64)
    records:      list[FrameRecord] = []
    n_done = 0

    try:
        pbar = tqdm(total=total_frames, desc=f"  {name}",
                    unit="f", leave=False, dynamic_ncols=True)
        for i in range(total_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 1) Same observation construction as build_lerobot_dataset.py
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_obs, raw_action = row_to_raw_obs_and_action(rows[i], frame_rgb)

            # 2) Same processor symmetry as the converter
            obs_processed = robot_observation_processor(raw_obs)

            # 3) Same packing the dataset uses
            observation_frame = build_dataset_frame(
                features, obs_processed, prefix=OBS_STR
            )

            # 4) Policy inference (mirrors inference_loop in inference_api.py)
            action_values = predict_action(
                observation = observation_frame,
                policy      = policy,
                device      = device,
                preprocessor  = preprocessor,
                postprocessor = postprocessor,
                use_amp     = use_amp,
                task        = single_task,
                robot_type  = robot_type,
            )
            act_pred = make_robot_action(action_values, features)
            pred_lin = float(act_pred.get("lin_x", 0.0))
            pred_ang = float(act_pred.get("ang_z", 0.0))

            gt_lin   = raw_action["lin_x"]
            gt_ang   = raw_action["ang_z"]

            gt_lin_buf[i]   = gt_lin
            gt_ang_buf[i]   = gt_ang
            pred_lin_buf[i] = pred_lin
            pred_ang_buf[i] = pred_ang

            records.append(FrameRecord(
                session=name, frame=i,
                gt_lin=gt_lin, gt_ang=gt_ang,
                pred_lin=pred_lin, pred_ang=pred_ang,
            ))
            n_done += 1

            if (i & 0x1F) == 0:
                pbar.set_postfix(
                    lin=f"gt{gt_lin:+.2f}/pr{pred_lin:+.2f}",
                    ang=f"gt{gt_ang:+.2f}/pr{pred_ang:+.2f}",
                )
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()

    return _finalize_session_result(
        name, gt_lin_buf, gt_ang_buf, pred_lin_buf, pred_ang_buf,
        records, n_done, value_tol, zero_eps,
    )


# ── Per-episode evaluation (HF LeRobot dataset mode) ────────────────────────

def _episode_bounds(ds: LeRobotDataset) -> tuple[list[int], list[int]]:
    """Return (from_indices, to_indices) for every episode. Handles several
    lerobot versions:
      - older: ds.episode_data_index["from" / "to"] as tensors
      - newer: ds.meta.episodes is a dict {ep_idx: {"length": n, ...}} or
               a list [{"length": n, ...}, ...]; frame indices are contiguous
    """
    idx = getattr(ds, "episode_data_index", None)
    if idx is not None and "from" in idx and "to" in idx:
        return idx["from"].tolist(), idx["to"].tolist()

    meta = getattr(ds, "meta", None)
    episodes = getattr(meta, "episodes", None) if meta is not None else None
    if episodes is None:
        raise RuntimeError(
            "Cannot determine episode boundaries: dataset has neither "
            "`episode_data_index` nor `meta.episodes`. Available attrs: "
            f"{sorted(a for a in dir(ds) if not a.startswith('_'))[:40]}"
        )

    # Normalize to an ordered list of per-episode records.
    if isinstance(episodes, dict):
        try:
            keys = sorted(episodes.keys(), key=lambda k: int(k))
        except (TypeError, ValueError):
            keys = list(episodes.keys())
        records = [episodes[k] for k in keys]
    else:
        records = list(episodes)

    def _len(rec) -> int:
        for key in ("length", "num_frames", "n_frames", "frames"):
            if isinstance(rec, dict) and key in rec:
                return int(rec[key])
            if hasattr(rec, key):
                return int(getattr(rec, key))
        raise KeyError(f"episode record has no length field: {rec!r}")

    from_indices: list[int] = []
    to_indices:   list[int] = []
    cur = 0
    for rec in records:
        n = _len(rec)
        from_indices.append(cur)
        to_indices.append(cur + n)
        cur += n
    return from_indices, to_indices


def _pick_ds_action_keys(ds_features: dict) -> tuple[str, str]:
    """Locate the lin_x / ang_z keys the dataset actually stores actions under."""
    # Common layouts: single "action" vector [lin_x, ang_z], or per-key
    # "action.lin_x" / "action.ang_z" columns.
    keys = list(ds_features.keys())
    if f"{ACTION}.lin_x" in ds_features and f"{ACTION}.ang_z" in ds_features:
        return f"{ACTION}.lin_x", f"{ACTION}.ang_z"
    if ACTION in ds_features:
        return ACTION, ACTION  # both come from the same vector
    raise KeyError(
        f"Could not find action fields in dataset. Have keys: {keys}. "
        f"Expected either '{ACTION}' or '{ACTION}.lin_x'/'{ACTION}.ang_z'."
    )


def _extract_gt_action(sample: dict, key_lin: str, key_ang: str) -> tuple[float, float]:
    if key_lin == key_ang:
        vec = sample[key_lin]
        import torch
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        vec = np.asarray(vec).reshape(-1)
        # Convention across this repo: action = [lin_x, ang_z]
        return float(vec[0]), float(vec[1])
    return float(np.asarray(sample[key_lin]).item()), float(np.asarray(sample[key_ang]).item())


def _extract_state_field(sample: dict, name: str) -> float:
    """Read a scalar from either observation.state (vector, ordered
    [lin_x, ang_z, lat, long, orientation]) or observation.state.<name>."""
    per_key = f"{OBS_STR}.state.{name}"
    if per_key in sample:
        return float(np.asarray(sample[per_key]).item())
    vec_key = f"{OBS_STR}.state"
    if vec_key in sample:
        vec = sample[vec_key]
        import torch
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        vec = np.asarray(vec).reshape(-1)
        order = ["lin_x", "ang_z", "lat", "long", "orientation"]
        if name in order and vec.size > order.index(name):
            return float(vec[order.index(name)])
    return 0.0


def eval_one_episode(
    ds:                         LeRobotDataset,
    episode_index:              int,
    from_idx:                   int,
    to_idx:                     int,
    key_lin:                    str,
    key_ang:                    str,
    image_key:                  str,
    policy,
    preprocessor,
    postprocessor,
    robot_observation_processor,
    features:                   dict,
    device,
    use_amp:                    bool,
    robot_type:                 str,
    single_task:                Optional[str],
    reset_policy_each_session:  bool,
    value_tol:                  float,
    zero_eps:                   float,
) -> SessionResult:

    name = f"episode_{episode_index}"

    if reset_policy_each_session:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    total_frames = to_idx - from_idx

    gt_lin_buf   = np.zeros(total_frames, dtype=np.float64)
    gt_ang_buf   = np.zeros(total_frames, dtype=np.float64)
    pred_lin_buf = np.zeros(total_frames, dtype=np.float64)
    pred_ang_buf = np.zeros(total_frames, dtype=np.float64)
    records:      list[FrameRecord] = []
    n_done = 0

    pbar = tqdm(total=total_frames, desc=f"  {name}",
                unit="f", leave=False, dynamic_ncols=True)
    for i in range(total_frames):
        sample = ds[from_idx + i]

        gt_lin, gt_ang = _extract_gt_action(sample, key_lin, key_ang)

        # Reconstruct the same raw observation the converter used.
        img_tensor = sample[image_key]
        frame_rgb  = _tensor_to_hwc_rgb_uint8(img_tensor)

        raw_obs = {
            "lin_x":       _extract_state_field(sample, "lin_x"),
            "ang_z":       _extract_state_field(sample, "ang_z"),
            "lat":         _extract_state_field(sample, "lat"),
            "long":        _extract_state_field(sample, "long"),
            "orientation": _extract_state_field(sample, "orientation"),
            CAMERA_KEY:    frame_rgb,
        }

        obs_processed = robot_observation_processor(raw_obs)
        observation_frame = build_dataset_frame(
            features, obs_processed, prefix=OBS_STR
        )

        action_values = predict_action(
            observation = observation_frame,
            policy      = policy,
            device      = device,
            preprocessor  = preprocessor,
            postprocessor = postprocessor,
            use_amp     = use_amp,
            task        = single_task,
            robot_type  = robot_type,
        )
        act_pred = make_robot_action(action_values, features)
        pred_lin = float(act_pred.get("lin_x", 0.0))
        pred_ang = float(act_pred.get("ang_z", 0.0))

        gt_lin_buf[i]   = gt_lin
        gt_ang_buf[i]   = gt_ang
        pred_lin_buf[i] = pred_lin
        pred_ang_buf[i] = pred_ang

        records.append(FrameRecord(
            session=name, frame=i,
            gt_lin=gt_lin, gt_ang=gt_ang,
            pred_lin=pred_lin, pred_ang=pred_ang,
        ))
        n_done += 1

        if (i & 0x1F) == 0:
            pbar.set_postfix(
                lin=f"gt{gt_lin:+.2f}/pr{pred_lin:+.2f}",
                ang=f"gt{gt_ang:+.2f}/pr{pred_ang:+.2f}",
            )
        pbar.update(1)
    pbar.close()

    return _finalize_session_result(
        name, gt_lin_buf, gt_ang_buf, pred_lin_buf, pred_ang_buf,
        records, n_done, value_tol, zero_eps,
    )


# ── Reporting ───────────────────────────────────────────────────────────────

_WIDTH = 100  # total console width for banners and separators


def _fmt(x: float, w: int = 8, p: int = 4) -> str:
    """Format a float, showing 'n/a' for None/NaN."""
    if x is None or (isinstance(x, float) and (x != x)):
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.{p}f}"


def _pct(x: float, w: int = 6) -> str:
    """Format a fraction in [0, 1] as a percentage string."""
    if x is None or (isinstance(x, float) and (x != x)):
        return f"{'n/a':>{w}}"
    return f"{x * 100:>{w-1}.1f}%"


def _banner(title: str, char: str = "=") -> None:
    print(char * _WIDTH)
    pad = max(0, (_WIDTH - len(title) - 2) // 2)
    print(f"{char * pad} {title} {char * (_WIDTH - pad - len(title) - 2)}")
    print(char * _WIDTH)


def _section(title: str) -> None:
    """Section header with a rule underneath."""
    print()
    print(f"  {title}")
    print("  " + "-" * (_WIDTH - 4))


def _note(text: str) -> None:
    """Inline explanation printed above a metric block."""
    for line in text.strip().splitlines():
        print(f"  # {line.strip()}")


def _draw_table(headers: list[str], rows: list[list[str]], align: str = None) -> None:
    """Draw a simple ASCII table with padded columns.

    align is a string of 'l' / 'r' per column ('r' = right-align). Defaults
    to right-align for numeric-looking columns, left for the first."""
    cols = list(zip(headers, *rows))
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    if align is None:
        align = "l" + "r" * (len(headers) - 1)

    def _pad(cell: str, w: int, a: str) -> str:
        s = str(cell)
        return s.ljust(w) if a == "l" else s.rjust(w)

    sep = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _fmt_row(row) -> str:
        return "  | " + " | ".join(_pad(c, w, a) for c, w, a in zip(row, widths, align)) + " |"

    print(sep)
    print(_fmt_row(headers))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
    print(sep)


# ── Glossary ────────────────────────────────────────────────────────────────

_GLOSSARY = """\
Symbols and metric definitions (values in the model's action space — raw ang_z is
±3.5 rad/s pre-scale; the MotionController applies × ang_z_scale (typically 0.20)
before /cmd_vel, so wheel-space error ≈ raw error × 0.20):

  Regression metrics (per action component)
    MAE           mean absolute error, |pred - gt| averaged across all frames.
                    Lower is better. Same units as the action.
    RMSE          root-mean-square error. Penalises large errors more than MAE.
    Max           worst single-frame absolute error.
    r (Pearson)   linear correlation of pred vs gt. 1.0 = perfect, 0 = none.
    R²            fraction of gt variance the prediction explains. 1.0 = perfect.
                    Negative means the prediction is worse than always guessing
                    the gt mean.
    sign          fraction of frames where sign(pred) matches sign(gt), among
                    frames where at least one side is non-trivial. Legacy signal;
                    the direction diagnostics below are more informative.
    acc±V         fraction of frames with |pred - gt| ≤ V. V = value-tol.

  Turning / straight-state classification (ang_z)
    A frame is "turning"  when |ang_z| >  zero-eps.
    A frame is "straight" when |ang_z| <= zero-eps.
    turn acc      recall on gt-turning frames (did we detect the turn?).
    straight acc  recall on gt-straight frames (did we stay straight?).
    overall acc   fraction of frames where turn/straight state matches.
    missed turns  gt turning, pred straight — turns the policy silently ignored.
    false turns   gt straight, pred turning — the policy wobbled off-centre.

  Angular direction diagnostics (ang_z, left = positive)
    dir acc | both turn      of frames where GT & pred both turn, fraction with
                              the SAME sign. This is the "did it choose the right
                              side?" number, the one that matters for driving.
    dir recall | gt turn     of all gt turns, fraction the policy caught in the
                              correct direction. Sensitive to missed turns.
    turn precision           of all predicted turns, fraction that were real
                              turns AND correctly directed. Low = jittering.
    left/right recall        per-side recall on real left/right turns.
    left/right precision     per-side precision on predicted left/right turns.
    magnitude acc±V          on correct-direction turns: fraction with the
                              magnitude also within ±V.
    MAE correct-dir / wrong-dir  magnitude MAE conditional on direction correct
                              vs opposite.

  lin_x zero / non-zero classification
    A frame is "zero"     when |lin_x| <= zero-eps (robot idle).
    zero acc          recall on gt-zero frames (did we stay at rest?).
    non-zero acc      recall on gt-moving frames (did we move?).
    gt-zero/pred-nz   gt idle but pred wants to move (false starts).
    gt-nz/pred-zero   gt moving but pred says stop (dropped commands).

  Per-episode "combined" column is sqrt((lin_RMSE^2 + ang_RMSE^2) / 2) — a rough
  single-number ranking used to sort worst episodes first.
"""


def _print_glossary() -> None:
    print()
    for line in _GLOSSARY.rstrip().splitlines():
        print("  " + line if line else "")
    print()


# ── Metric block printers ───────────────────────────────────────────────────

def print_component(m: ComponentMetrics, label: str) -> None:
    """Print one row of continuous-value regression metrics."""
    line1 = (
        f"  {label:8s}  n={m.n:>6d}   "
        f"MAE={_fmt(m.mae)}   "
        f"RMSE={_fmt(m.rmse)}   "
        f"Max={_fmt(m.max_err)}"
    )
    line2 = (
        f"  {'':8s}  "
        f"r={_fmt(m.pearson, w=7, p=3)}   "
        f"R²={_fmt(m.r2, w=7, p=3)}   "
        f"sign={_pct(m.sign_agreement, w=6)}   "
        f"acc±{m.value_tol:g}={_pct(m.accuracy_within_tol, w=6)}"
    )
    line3 = (
        f"  {'range':8s}  "
        f"gt=[{_fmt(m.gt_min, 7, 3)}, {_fmt(m.gt_max, 7, 3)}]   "
        f"pred=[{_fmt(m.pred_min, 7, 3)}, {_fmt(m.pred_max, 7, 3)}]"
    )
    print(line1)
    print(line2)
    print(line3)


def print_turning_metrics(m: TurningMetrics) -> None:
    _section(f"Turning vs straight state — ang_z  (|ang_z| ≤ {m.zero_eps:g} counts as straight)")
    _note("""
        turn acc      : recall on gt-turning frames — did the policy detect the turn?
        straight acc  : recall on gt-straight frames — did it stay centred?
        missed turns  : gt turning + pred straight (turns silently dropped)
        false turns   : gt straight + pred turning (wobble off centre)
    """)
    _draw_table(
        ["state", "gt frames", "pred match", "recall", "notes"],
        [
            ["gt turning",  f"{m.gt_turning_n:>6d}",  f"{m.turning_match_n:>6d}",
             _pct(m.turning_match_accuracy), f"missed turns = {m.missed_turn_n}"],
            ["gt straight", f"{m.gt_straight_n:>6d}", f"{m.straight_match_n:>6d}",
             _pct(m.straight_match_accuracy), f"false turns  = {m.false_turn_n}"],
            ["overall",     f"{m.n:>6d}", "", _pct(m.overall_state_accuracy), ""],
        ],
    )


def print_angular_direction_metrics(m: AngularDirectionMetrics) -> None:
    _section(f"Angular direction diagnostics — ang_z  (|ang_z| ≤ {m.zero_eps:g} counts as straight)")
    _note("""
        This section separates "did the policy detect a turn?" from
        "did it pick the right side (left vs right)?" and "was the magnitude close?".
        Left = positive ang_z, right = negative.
    """)

    # Frame counts by side
    print()
    print("  Frame counts by side")
    _draw_table(
        ["side", "GT frames", "Pred frames"],
        [
            ["left  (ang_z > +eps)", f"{m.gt_left_n:>6d}",  f"{m.pred_left_n:>6d}"],
            ["right (ang_z < -eps)", f"{m.gt_right_n:>6d}", f"{m.pred_right_n:>6d}"],
            ["straight",             f"{m.gt_straight_n:>6d}", f"{m.pred_straight_n:>6d}"],
        ],
    )

    # Headline numbers
    print()
    print("  Headline direction accuracy")
    _note("""
        dir acc | both turn      : when GT & pred both turn, fraction with same side.
                                   The "picks the right way" number.
        dir recall on gt turns   : of all gt turns, fraction the policy caught w/ correct side.
        turn precision           : of all predicted turns, fraction that were real correct-side turns.
                                   Low = policy jitters (many false turns).
    """)
    _draw_table(
        ["metric", "value", "n (denominator)"],
        [
            ["dir acc | both turn",    _pct(m.direction_accuracy_when_both_turn), f"{m.both_turning_n:>6d}"],
            ["dir recall | gt turn",   _pct(m.direction_recall_on_gt_turns),      f"{m.gt_left_n + m.gt_right_n:>6d}"],
            ["turn precision",         _pct(m.turn_precision),                     f"{m.pred_left_n + m.pred_right_n:>6d}"],
        ],
    )

    # Per-side recall / precision
    print()
    print("  Per-side recall and precision")
    _note("""
        recall     : how many actual left/right turns were caught correctly.
        precision  : how many predicted left/right turns were real left/right turns.
    """)
    _draw_table(
        ["side",  "recall", "precision"],
        [
            ["left",  _pct(m.left_recall),  _pct(m.left_precision)],
            ["right", _pct(m.right_recall), _pct(m.right_precision)],
        ],
    )

    # Confusion breakdown
    print()
    print("  Confusion breakdown")
    _note("""
        opposite-side confusions are the dangerous ones — the policy actively steered
        the wrong way. straight→turn are wobble; turn→straight are missed turns.
    """)
    _draw_table(
        ["gt \\ pred",  "→ left", "→ right", "→ straight"],
        [
            ["gt left",     "-",                                     f"{m.gt_left_pred_right_n:>6d}",  f"{m.gt_left_pred_straight_n:>6d}"],
            ["gt right",    f"{m.gt_right_pred_left_n:>6d}",         "-",                              f"{m.gt_right_pred_straight_n:>6d}"],
            ["gt straight", f"{m.gt_straight_pred_left_n:>6d}",      f"{m.gt_straight_pred_right_n:>6d}", "-"],
        ],
    )

    # Magnitude conditional on direction
    print()
    print("  Magnitude accuracy conditional on direction")
    _note("""
        Given the direction is correct, how close is the magnitude?
        wrong-dir MAE is included for reference — it shows how far off the value is
        when the policy did steer the wrong way.
    """)
    _draw_table(
        ["condition",              f"acc±{m.value_tol:g}", "MAE",                             "n"],
        [
            ["correct-direction turns", _pct(m.magnitude_acc_when_correct_dir),
             _fmt(m.mae_when_correct_dir, w=7, p=3),
             f"{m.correct_direction_n:>6d}"],
            ["opposite-direction turns", "",
             _fmt(m.mae_when_wrong_dir, w=7, p=3),
             f"{m.opposite_direction_n:>6d}"],
        ],
    )


def print_lin_zero_metrics(m: LinZeroMetrics) -> None:
    _section(f"lin_x zero vs non-zero state  (|lin_x| ≤ {m.zero_eps:g} counts as zero / idle)")
    _note("""
        zero acc         : did the policy stay at rest when it should have?
        non-zero acc     : did the policy start moving when it should have?
        gt-zero/pred-nz  : false starts (policy tried to move on an idle frame)
        gt-nz/pred-zero  : dropped commands (policy stopped on an active frame)
    """)
    _draw_table(
        ["state",         "gt frames",                    "pred match",                    "recall",                             "notes"],
        [
            ["gt zero",     f"{m.gt_zero_n:>6d}",          f"{m.zero_match_n:>6d}",         _pct(m.zero_match_accuracy),          f"gt-zero/pred-nz = {m.missed_zero_n}"],
            ["gt non-zero", f"{m.gt_nonzero_n:>6d}",       f"{m.nonzero_match_n:>6d}",      _pct(m.nonzero_match_accuracy),       f"gt-nz/pred-zero = {m.false_zero_n}"],
            ["overall",     f"{m.n:>6d}",                   "",                             _pct(m.overall_state_accuracy),       ""],
        ],
    )


def print_summary(
    sessions: list[SessionResult],
    wall_time_s: float,
    worst_k: int,
    value_tol: float,
    zero_eps: float,
) -> None:

    total_frames = sum(s.n_frames for s in sessions)
    n_sessions   = len([s for s in sessions if s.n_frames > 0])
    fps = total_frames / wall_time_s if wall_time_s > 0 else 0.0

    print()
    _banner("OFFLINE POLICY EVALUATION SUMMARY", "=")
    print()
    print(f"  Sessions/episodes evaluated : {n_sessions}")
    print(f"  Total frames                : {total_frames}")
    print(f"  Eval wall time              : {wall_time_s:.1f}s   ({fps:.1f} frames/s)")
    print(f"  value_tol (acc window)      : ±{value_tol:g}")
    print(f"  zero_eps  (straight/idle)   : ±{zero_eps:g}")

    _print_glossary()

    all_records: list[FrameRecord] = []
    for s in sessions:
        all_records.extend(s.records)

    if not all_records:
        print("  (no frames evaluated)")
        return

    gt_lin   = np.array([r.gt_lin   for r in all_records])
    gt_ang   = np.array([r.gt_ang   for r in all_records])
    pred_lin = np.array([r.pred_lin for r in all_records])
    pred_ang = np.array([r.pred_ang for r in all_records])

    overall_lin = ComponentMetrics.compute("lin_x", gt_lin, pred_lin,
                                           value_tol=value_tol, zero_eps=zero_eps)
    overall_ang = ComponentMetrics.compute("ang_z", gt_ang, pred_ang,
                                           value_tol=value_tol, zero_eps=zero_eps)
    overall_turning = TurningMetrics.compute(gt_ang, pred_ang, zero_eps=zero_eps)
    overall_angular_direction = AngularDirectionMetrics.compute(
        gt_ang, pred_ang, zero_eps=zero_eps, value_tol=value_tol
    )
    overall_lin_zero = LinZeroMetrics.compute(gt_lin, pred_lin, zero_eps=zero_eps)

    # ── Overall regression metrics ────────────────────────────────────────
    _section("Overall regression metrics")
    _note("""
        Per-action-component error against the ground-truth teleop command.
        MAE/RMSE in the action's units (raw ang_z ≈ 5 × wheel ang_z under ang_z_scale=0.20).
    """)
    print_component(overall_lin, "lin_x")
    print()
    print_component(overall_ang, "ang_z")

    # ── Classification blocks ─────────────────────────────────────────────
    print_turning_metrics(overall_turning)
    print_angular_direction_metrics(overall_angular_direction)
    print_lin_zero_metrics(overall_lin_zero)

    # ── Per-session table ─────────────────────────────────────────────────
    _section("Per-episode breakdown  (sorted by combined RMSE, worst first)")
    _note("""
        combined RMSE      : sqrt((lin_RMSE² + ang_RMSE²) / 2) — quick single-number rank.
        turn acc           : overall turn/straight state accuracy for that episode.
        dir acc            : direction accuracy given both GT & pred turn (100% ideal).
        turn precision     : predicted turns that were real correct-side turns.
        lin0 acc           : recall on gt-zero lin_x frames (was the policy idle when it should be?).
    """)
    session_rows = []
    for s in sorted(sessions, key=lambda x: -x.combined_rmse):
        if s.n_frames == 0:
            continue
        session_rows.append([
            s.name,
            f"{s.n_frames}",
            f"{s.lin_metric.mae:.4f}",
            f"{s.lin_metric.rmse:.4f}",
            f"{s.ang_metric.mae:.4f}",
            f"{s.ang_metric.rmse:.4f}",
            f"{s.combined_rmse:.4f}",
            _pct(s.turning_metric.overall_state_accuracy),
            _pct(s.angular_direction_metric.direction_accuracy_when_both_turn),
            _pct(s.angular_direction_metric.turn_precision),
            _pct(s.lin_zero_metric.zero_match_accuracy),
        ])
    _draw_table(
        ["episode", "frames", "lin MAE", "lin RMSE", "ang MAE", "ang RMSE",
         "combined", "turn acc", "dir acc", "turn prec", "lin0 acc"],
        session_rows,
    )

    # ── Worst-K frames ────────────────────────────────────────────────────
    if worst_k > 0:
        _section(f"Top {worst_k} worst frames by |lin_x error|")
        _note("Frames where the linear-velocity prediction was furthest from the recorded command.")
        worst_lin = sorted(all_records, key=lambda r: -abs(r.err_lin))[:worst_k]
        _draw_table(
            ["episode", "frame", "gt lin_x", "pred lin_x", "error"],
            [[r.session, f"{r.frame}", f"{r.gt_lin:+.4f}", f"{r.pred_lin:+.4f}", f"{r.err_lin:+.4f}"]
             for r in worst_lin],
        )

        _section(f"Top {worst_k} worst frames by |ang_z error|")
        _note("Frames where the angular-velocity prediction was furthest off. "
              "Multiply by ang_z_scale (0.20) for wheel-space error.")
        worst_ang = sorted(all_records, key=lambda r: -abs(r.err_ang))[:worst_k]
        _draw_table(
            ["episode", "frame", "gt ang_z", "pred ang_z", "error"],
            [[r.session, f"{r.frame}", f"{r.gt_ang:+.4f}", f"{r.pred_ang:+.4f}", f"{r.err_ang:+.4f}"]
             for r in worst_ang],
        )

    print()
    _banner("END OF REPORT", "=")
    print()


def write_csv(out_path: Path, sessions: list[SessionResult], zero_eps: float) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "frame",
                    "gt_lin",   "pred_lin",   "err_lin",
                    "gt_ang",   "pred_ang",   "err_ang",
                    "gt_turning", "pred_turning", "turning_state_match",
                    "gt_ang_dir", "pred_ang_dir", "ang_dir_match_when_both_turn",
                    "gt_lin_zero", "pred_lin_zero", "lin_zero_state_match"])
        for s in sessions:
            for r in s.records:
                gt_turning = abs(r.gt_ang) > zero_eps
                pred_turning = abs(r.pred_ang) > zero_eps
                gt_lin_zero = abs(r.gt_lin) <= zero_eps
                pred_lin_zero = abs(r.pred_lin) <= zero_eps
                gt_ang_dir = 1 if r.gt_ang > zero_eps else (-1 if r.gt_ang < -zero_eps else 0)
                pred_ang_dir = 1 if r.pred_ang > zero_eps else (-1 if r.pred_ang < -zero_eps else 0)
                both_turning = gt_ang_dir != 0 and pred_ang_dir != 0
                dir_match_when_both_turn = int(gt_ang_dir == pred_ang_dir) if both_turning else ""
                w.writerow([
                    r.session, r.frame,
                    f"{r.gt_lin:.6f}",   f"{r.pred_lin:.6f}",   f"{r.err_lin:.6f}",
                    f"{r.gt_ang:.6f}",   f"{r.pred_ang:.6f}",   f"{r.err_ang:.6f}",
                    int(gt_turning), int(pred_turning), int(gt_turning == pred_turning),
                    gt_ang_dir, pred_ang_dir, dir_match_when_both_turn,
                    int(gt_lin_zero), int(pred_lin_zero), int(gt_lin_zero == pred_lin_zero),
                ])
    print(f"  CSV written: {out_path}  ({sum(s.n_frames for s in sessions)} rows)")


def write_json_summary(
    out_path: Path,
    sessions: list[SessionResult],
    wall_time_s: float,
    value_tol: float,
    zero_eps: float,
) -> None:
    all_records = [r for s in sessions for r in s.records]
    gt_lin   = np.array([r.gt_lin   for r in all_records])
    gt_ang   = np.array([r.gt_ang   for r in all_records])
    pred_lin = np.array([r.pred_lin for r in all_records])
    pred_ang = np.array([r.pred_ang for r in all_records])
    overall_lin = ComponentMetrics.compute("lin_x", gt_lin, pred_lin,
                                           value_tol=value_tol, zero_eps=zero_eps)
    overall_ang = ComponentMetrics.compute("ang_z", gt_ang, pred_ang,
                                           value_tol=value_tol, zero_eps=zero_eps)
    overall_turning = TurningMetrics.compute(gt_ang, pred_ang, zero_eps=zero_eps)
    overall_angular_direction = AngularDirectionMetrics.compute(
        gt_ang, pred_ang, zero_eps=zero_eps, value_tol=value_tol
    )
    overall_lin_zero = LinZeroMetrics.compute(gt_lin, pred_lin, zero_eps=zero_eps)

    def _m(m: ComponentMetrics) -> dict:
        return {k: getattr(m, k) for k in
                ("name","n","mae","rmse","max_err","pearson","r2",
                 "sign_agreement","accuracy_within_tol","value_tol",
                 "gt_min","gt_max","pred_min","pred_max")}

    def _t(m: TurningMetrics) -> dict:
        return {k: getattr(m, k) for k in
                ("n","zero_eps","gt_turning_n","gt_straight_n",
                 "pred_turning_n","pred_straight_n","turning_match_n",
                 "straight_match_n","missed_turn_n","false_turn_n",
                 "turning_match_accuracy","straight_match_accuracy",
                 "overall_state_accuracy")}

    def _d(m: AngularDirectionMetrics) -> dict:
        return {k: getattr(m, k) for k in
                ("n","zero_eps","value_tol","gt_left_n","gt_right_n",
                 "gt_straight_n","pred_left_n","pred_right_n",
                 "pred_straight_n","both_turning_n","correct_direction_n",
                 "opposite_direction_n","gt_left_pred_right_n",
                 "gt_right_pred_left_n","gt_left_pred_straight_n",
                 "gt_right_pred_straight_n","gt_straight_pred_left_n",
                 "gt_straight_pred_right_n","direction_accuracy_when_both_turn",
                 "direction_recall_on_gt_turns","turn_precision",
                 "left_recall","right_recall","left_precision","right_precision",
                 "magnitude_acc_when_correct_dir","mae_when_correct_dir",
                 "mae_when_wrong_dir")}

    def _z(m: LinZeroMetrics) -> dict:
        return {k: getattr(m, k) for k in
                ("n","zero_eps","gt_zero_n","gt_nonzero_n",
                 "pred_zero_n","pred_nonzero_n","zero_match_n",
                 "nonzero_match_n","missed_zero_n","false_zero_n",
                 "zero_match_accuracy","nonzero_match_accuracy",
                 "overall_state_accuracy")}

    payload = {
        "wall_time_s": wall_time_s,
        "total_frames": int(sum(s.n_frames for s in sessions)),
        "n_sessions":   int(len([s for s in sessions if s.n_frames > 0])),
        "overall": {
            "lin_x": _m(overall_lin),
            "ang_z": _m(overall_ang),
            "turning": _t(overall_turning),
            "angular_direction": _d(overall_angular_direction),
            "lin_x_zero": _z(overall_lin_zero),
        },
        "sessions": [{
            "name":     s.name,
            "n_frames": s.n_frames,
            "lin_x":    _m(s.lin_metric),
            "ang_z":    _m(s.ang_metric),
            "turning":  _t(s.turning_metric),
            "angular_direction": _d(s.angular_direction_metric),
            "lin_x_zero": _z(s.lin_zero_metric),
        } for s in sessions],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON written: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parent-dir", default=None,
                     help="Parent folder containing session_<i>/ subfolders.")
    src.add_argument("--dataset-repo-id-eval", default=None,
                     help="HF repo id of a LeRobot dataset to evaluate on. "
                          "Iterates it episode-by-episode. No session folders "
                          "required.")
    pol = ap.add_mutually_exclusive_group(required=True)
    pol.add_argument("--policy-path", default=None,
                     help="Local pretrained policy checkpoint directory.")
    pol.add_argument("--policy-id", default=None,
                     help="HuggingFace repo id of the policy checkpoint "
                          "(e.g. 'Aadi/act_scout_dataset_03'). Downloaded via "
                          "snapshot_download and used as the checkpoint path.")
    ap.add_argument("--policy.pretrained_revision", dest="policy_pretrained_revision",
                    default=None,
                    help="Optional git revision/branch/tag for --policy-id. "
                         "Use this to select a specific checkpoint from the "
                         "HF repo (e.g. --policy.pretrained_revision=040000). "
                         "Ignored with --policy-path.")
    ap.add_argument("--policy-subfolder", default=None,
                    help="Subfolder inside the checkpoint dir where config.json "
                         "lives (typical lerobot layout: 'pretrained_model'). "
                         "Leave unset for auto-detect.")
    ap.add_argument("--dataset-repo-id", required=True,
                    help="HF dataset repo id used for the policy's feature "
                         "schema + normalization stats. (Same repo as the "
                         "training dataset.)")
    ap.add_argument("--robot-type",      default="revobots_agv_follower",
                    help="robot_type string passed to predict_action.")
    ap.add_argument("--task",            default=None)
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--no-reset-per-session", action="store_true",
                    help="Don't reset policy state between sessions/episodes.")
    ap.add_argument("--max-sessions",    type=int, default=0,
                    help="If > 0, only evaluate the first N sessions/episodes.")
    ap.add_argument("--sessions",        nargs="*", default=None,
                    help="[session-folder mode] Evaluate only these session "
                         "names (e.g. session_3 session_7).")
    ap.add_argument("--episodes",        nargs="*", type=int, default=None,
                    help="[repo mode] Evaluate only these episode indices "
                         "(e.g. 0 4 9).")
    ap.add_argument("--worst-k",         type=int, default=10,
                    help="Show top-K worst frames per component (default 10).")
    ap.add_argument("--value-tol",       type=float, default=0.05,
                    help="Tolerance for magnitude accuracy on continuous "
                         "values. A prediction counts as correct when "
                         "abs(pred - gt) <= this (default 0.05).")
    ap.add_argument("--zero-eps",        type=float, default=0.025,
                    help="Threshold for treating a value as zero. abs(x) <= "
                         "this counts as straight (ang_z) or stopped (lin_x). "
                         "Default 0.025 — the AGV control deadband.")
    ap.add_argument("--csv-out",         type=str, default=None,
                    help="If set, write per-frame predictions to this CSV.")
    ap.add_argument("--json-out",        type=str, default=None,
                    help="If set, write metrics summary to this JSON.")
    return ap.parse_args()


def _run_session_folder_mode(args, pipeline_bits) -> tuple[list[SessionResult], float]:
    (policy, preprocessor, postprocessor, _rap,
     robot_observation_processor, _features_meta) = pipeline_bits

    input_root = Path(args.parent_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"[!] not a directory: {input_root}")

    sessions_all = find_session_folders(input_root)
    if not sessions_all:
        raise SystemExit(f"[!] no session_<i>/ found in {input_root}")

    if args.sessions:
        wanted = set(args.sessions)
        sessions = [s for s in sessions_all if s.name in wanted]
        missing  = wanted - {s.name for s in sessions}
        if missing:
            print(f"[!] requested sessions not found: {sorted(missing)}")
    elif args.max_sessions > 0:
        sessions = sessions_all[:args.max_sessions]
    else:
        sessions = sessions_all

    print(f"[*] mode       : session folders")
    print(f"[*] input root : {input_root}")
    print(f"[*] sessions   : {len(sessions)} / {len(sessions_all)} available")
    print(f"[*] reset/sess : {not args.no_reset_per_session}")
    print()

    features = build_dataset_features()
    device   = get_safe_torch_device(policy.config.device)
    use_amp  = policy.config.use_amp

    t_start = time.perf_counter()
    results: list[SessionResult] = []
    outer = tqdm(total=len(sessions), desc="Sessions", unit="ep", dynamic_ncols=True)
    for s_dir in sessions:
        outer.set_description(f"Sessions ({s_dir.name})")
        res = eval_one_session(
            session_dir                 = s_dir,
            policy                      = policy,
            preprocessor                = preprocessor,
            postprocessor               = postprocessor,
            robot_observation_processor = robot_observation_processor,
            features                    = features,
            device                      = device,
            use_amp                     = use_amp,
            robot_type                  = args.robot_type,
            single_task                 = args.task,
            reset_policy_each_session   = not args.no_reset_per_session,
            value_tol                   = args.value_tol,
            zero_eps                    = args.zero_eps,
        )
        results.append(res)
        outer.update(1)
    outer.close()
    return results, time.perf_counter() - t_start


def _run_repo_mode(args, pipeline_bits) -> tuple[list[SessionResult], float]:
    (policy, preprocessor, postprocessor, _rap,
     robot_observation_processor, _features_meta) = pipeline_bits

    print(f"[*] mode       : LeRobot dataset repo")
    print(f"[*] eval repo  : {args.dataset_repo_id_eval}")
    print(f"[*] loading dataset (this may download)...")
    ds = LeRobotDataset(args.dataset_repo_id_eval)

    n_episodes = int(ds.num_episodes)
    print(f"[*] episodes   : {n_episodes} total")

    # Figure out where actions and images live in the dataset.
    key_lin, key_ang = _pick_ds_action_keys(ds.features)
    image_keys = [k for k in ds.features if k.startswith(f"{OBS_STR}.images.")]
    if not image_keys:
        raise SystemExit(f"[!] no {OBS_STR}.images.* keys in dataset. "
                         f"Have: {list(ds.features)}")
    preferred = f"{OBS_STR}.images.{CAMERA_KEY}"
    image_key = preferred if preferred in image_keys else image_keys[0]
    if image_key != preferred:
        print(f"[!] '{preferred}' not found; using '{image_key}' instead.")

    # Pick which episodes to evaluate.
    if args.episodes:
        wanted = sorted(set(args.episodes))
        ep_indices = [e for e in wanted if 0 <= e < n_episodes]
        missing = [e for e in wanted if not (0 <= e < n_episodes)]
        if missing:
            print(f"[!] episode indices out of range (0..{n_episodes-1}): {missing}")
    elif args.max_sessions > 0:
        ep_indices = list(range(min(args.max_sessions, n_episodes)))
    else:
        ep_indices = list(range(n_episodes))

    print(f"[*] evaluating : {len(ep_indices)} episodes")
    print(f"[*] action keys: lin='{key_lin}'  ang='{key_ang}'")
    print(f"[*] image key  : {image_key}")
    print(f"[*] reset/ep   : {not args.no_reset_per_session}")
    print()

    features = build_dataset_features()
    device   = get_safe_torch_device(policy.config.device)
    use_amp  = policy.config.use_amp

    # Episode-frame index bounds. Support several lerobot layouts:
    #   - older: ds.episode_data_index["from"/"to"] (tensors)
    #   - newer: ds.meta.episodes is a dict/list with "length"/"num_frames"
    from_indices, to_indices = _episode_bounds(ds)

    t_start = time.perf_counter()
    results: list[SessionResult] = []
    outer = tqdm(total=len(ep_indices), desc="Episodes", unit="ep", dynamic_ncols=True)
    for ep in ep_indices:
        outer.set_description(f"Episodes (episode_{ep})")
        res = eval_one_episode(
            ds                          = ds,
            episode_index               = ep,
            from_idx                    = int(from_indices[ep]),
            to_idx                      = int(to_indices[ep]),
            key_lin                     = key_lin,
            key_ang                     = key_ang,
            image_key                   = image_key,
            policy                      = policy,
            preprocessor                = preprocessor,
            postprocessor               = postprocessor,
            robot_observation_processor = robot_observation_processor,
            features                    = features,
            device                      = device,
            use_amp                     = use_amp,
            robot_type                  = args.robot_type,
            single_task                 = args.task,
            reset_policy_each_session   = not args.no_reset_per_session,
            value_tol                   = args.value_tol,
            zero_eps                    = args.zero_eps,
        )
        results.append(res)
        outer.update(1)
    outer.close()
    return results, time.perf_counter() - t_start


def main() -> int:
    args = parse_args()
    init_logging()

    # Resolve policy source (download from HF if --policy-id was given).
    if args.policy_path:
        policy_path = args.policy_path
        policy_source = f"local: {policy_path}"
    else:
        print(f"[*] downloading policy from HF: {args.policy_id}"
              + (f" @ {args.policy_pretrained_revision}" if args.policy_pretrained_revision else ""))
        policy_path = snapshot_download(
            repo_id  = args.policy_id,
            revision = args.policy_pretrained_revision,
        )
        policy_source = f"HF: {args.policy_id} → {policy_path}"

    # Resolve subfolder: honour --policy-subfolder if given, otherwise
    # auto-detect config.json. Handles flat, pretrained_model/, and
    # checkpoints/<N>/pretrained_model/ layouts. If a checkpoint number
    # was requested via --policy.pretrained_revision, use it as a hint to
    # pick the right folder when HF fell back to main and gave us the tree.
    policy_path = _resolve_policy_dir(
        policy_path,
        args.policy_subfolder,
        checkpoint_hint=args.policy_pretrained_revision,
    )

    print(f"[*] policy     : {policy_source}")
    print(f"[*] dataset    : {args.dataset_repo_id}   (schema + stats)")
    print(f"[*] device     : {args.device}")
    print(f"[*] value_tol  : ±{args.value_tol}   zero_eps: ±{args.zero_eps}")
    print()

    print("[*] building policy pipeline...")
    pipeline_bits = build_policy_pipeline(
        policy_path     = policy_path,
        dataset_repo_id = args.dataset_repo_id,
        device          = args.device,
        rename_map      = None,
    )

    if args.parent_dir:
        results, t_wall = _run_session_folder_mode(args, pipeline_bits)
    else:
        results, t_wall = _run_repo_mode(args, pipeline_bits)

    print_summary(results, wall_time_s=t_wall, worst_k=args.worst_k,
                  value_tol=args.value_tol, zero_eps=args.zero_eps)

    if args.csv_out:
        write_csv(Path(args.csv_out).expanduser(), results, zero_eps=args.zero_eps)
    if args.json_out:
        write_json_summary(Path(args.json_out).expanduser(), results, t_wall,
                           value_tol=args.value_tol, zero_eps=args.zero_eps)

    return 0


if __name__ == "__main__":
    sys.exit(main())