# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:32:52 2026

@author: Aadi
"""
from __future__ import annotations

"""
compute_action_bins.py — precompute action bin centers and class weights for
ACT policy with action discretization (Method 1 — mode-collapse fix).

What this does
--------------
Reads your LeRobot dataset, computes the same MEAN_STD normalization the policy
applies to actions, then bins the normalized action distribution into K
categorical buckets per action dimension. For each bucket it computes a class
weight inversely proportional to the bucket's frequency, so cross-entropy at
training time pays disproportionate attention to rare actions (e.g. sharp
turns, full-throttle).

Output is a single .pt file the modified ACT policy loads at init.

Usage
-----
    python compute_action_bins.py \\
        --dataset-repo-id Aadi/scout_dataset_03 \\
        --out class_weights.pt \\
        --n-bins 31 \\
        --strategy uniform \\
        --alpha 0.5

Tuning hints
------------
    --n-bins        31 is a good default. More bins = finer control, less data
                    per bin. For ang_z range ±3.5 with 31 bins you get ~0.23
                    rad/s resolution. For lin_x range [-0.4, 1.0] with 31 bins
                    you get ~0.047 m/s resolution. Both reasonable.

    --strategy      uniform: evenly spaced bins across the range. Simple,
                    interpretable, good first try.
                    quantile: bin edges at percentiles so each bin holds equal
                    mass. Better for severely skewed data (your case). Try
                    second if uniform isn't enough.

    --alpha         Class weight exponent:
                    0.0 = uniform (no rebalancing — pointless for our purpose)
                    0.5 = sqrt-inverse-frequency (RECOMMENDED START)
                    1.0 = full inverse-frequency (aggressive)
                    Higher = more weight on rare bins. Above 1.0 rarely helps.
"""



import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


# ── Binning utilities ───────────────────────────────────────────────────────

def compute_bin_centers_uniform(min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    """Linearly spaced bin centers. Outermost bins land exactly at min_val and max_val,
    so any in-distribution action maps cleanly to some bin."""
    return np.linspace(min_val, max_val, n_bins)


def compute_bin_centers_quantile(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Place bin edges at percentiles so each bin contains ~equal frame mass.
    Bin centers are midpoints between consecutive edges. For long-tail data
    this concentrates resolution where signal lives."""
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    # Edges can collapse if data has heavy spikes — fix by epsilon nudges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return 0.5 * (edges[:-1] + edges[1:])


def actions_to_bin_indices(actions: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """For each action, return the closest bin index. actions: (N,). centers: (n_bins,)."""
    distances = np.abs(actions[:, None] - bin_centers[None, :])
    return distances.argmin(axis=1)


def compute_class_weights(bin_idx: np.ndarray, n_bins: int, alpha: float = 0.5) -> np.ndarray:
    """Class weights inversely proportional to bin count, raised to the `alpha` power.
    Normalized so the mean weight is 1 (keeps overall loss scale unchanged)."""
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    weights = 1.0 / np.power(counts + 1.0, alpha)
    weights = weights * n_bins / weights.sum()
    return weights


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-repo-id", required=True,
                    help="HF dataset repo id (must match what you'll train ACT on).")
    ap.add_argument("--out", required=True,
                    help="Output .pt path. The ACT config's `action_bins_path` points here.")
    ap.add_argument("--n-bins", type=int, default=31)
    ap.add_argument("--strategy", choices=("uniform", "quantile"), default="uniform")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Class-weight exponent (0=uniform, 0.5=sqrt-inv-freq, 1=inv-freq).")
    ap.add_argument("--action-keys", nargs="+", default=["lin_x", "ang_z"],
                    help="Names of action components for logging only.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[*] loading dataset metadata: {args.dataset_repo_id}")
    meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    # ── Action stats — same source the preprocessor uses ─────────────────────
    action_stats = meta.stats["action"]
    action_mean = np.array(action_stats["mean"], dtype=np.float64)
    action_std  = np.array(action_stats["std"],  dtype=np.float64)
    action_dim  = action_mean.shape[0]

    print(f"    action_dim   = {action_dim}")
    print(f"    action_mean  = {action_mean}")
    print(f"    action_std   = {action_std}")
    print(f"    action_min   = {np.array(action_stats['min'])}")
    print(f"    action_max   = {np.array(action_stats['max'])}")

    if len(args.action_keys) != action_dim:
        print(f"[!] warning: {len(args.action_keys)} action_keys provided but "
              f"dataset has action_dim={action_dim}. Using generic names where needed.")

    # ── Pull all actions out of the dataset ──────────────────────────────────
    print(f"\n[*] loading dataset frames...")
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=None)

    # Read actions directly from the underlying HF dataset for speed —
    # avoids per-frame video decode that LeRobotDataset.__getitem__ does.
    raw = dataset.hf_dataset.with_format("numpy")["action"]
    actions = np.stack([np.asarray(a, dtype=np.float64) for a in raw], axis=0)
    print(f"    loaded {actions.shape[0]} frames, shape={actions.shape}")

    # Normalize using the exact same stats the policy preprocessor will use
    actions_norm = (actions - action_mean) / action_std

    # ── Compute bin centers + class weights per action dim ───────────────────
    print(f"\n[*] computing bins (strategy={args.strategy}, n_bins={args.n_bins}, alpha={args.alpha})")

    bin_centers_norm = np.zeros((action_dim, args.n_bins), dtype=np.float64)
    bin_centers_raw  = np.zeros((action_dim, args.n_bins), dtype=np.float64)
    class_weights    = np.zeros((action_dim, args.n_bins), dtype=np.float64)

    for d in range(action_dim):
        name = args.action_keys[d] if d < len(args.action_keys) else f"action_{d}"

        # Bin in normalized space — the policy will receive normalized actions during training
        if args.strategy == "uniform":
            norm_min = float(actions_norm[:, d].min())
            norm_max = float(actions_norm[:, d].max())
            centers_norm = compute_bin_centers_uniform(norm_min, norm_max, args.n_bins)
        else:
            centers_norm = compute_bin_centers_quantile(actions_norm[:, d], args.n_bins)

        # Map back to raw for human-readable logging only
        centers_raw = centers_norm * action_std[d] + action_mean[d]

        # Assign every training-set frame to a bin, then compute weights from counts
        bin_idx = actions_to_bin_indices(actions_norm[:, d], centers_norm)
        weights = compute_class_weights(bin_idx, args.n_bins, alpha=args.alpha)

        bin_centers_norm[d] = centers_norm
        bin_centers_raw[d]  = centers_raw
        class_weights[d]    = weights

        # Diagnostics — show the user what they're getting
        counts = np.bincount(bin_idx, minlength=args.n_bins)
        top3 = np.argsort(counts)[-3:][::-1]
        bot3 = np.argsort(counts)[:3]

        print(f"\n    [{name}]")
        print(f"      raw range    : [{centers_raw.min():+.3f}, {centers_raw.max():+.3f}]")
        print(f"      weight range : [{weights.min():.3f}, {weights.max():.3f}]   "
              f"(rare bins get {weights.max() / weights.min():.0f}x more attention)")
        print(f"      most populated bins:")
        for b in top3:
            print(f"        bin {b:>3d}  raw={centers_raw[b]:+7.3f}  "
                  f"count={counts[b]:>6d}  weight={weights[b]:6.3f}")
        print(f"      least populated bins:")
        for b in bot3:
            print(f"        bin {b:>3d}  raw={centers_raw[b]:+7.3f}  "
                  f"count={counts[b]:>6d}  weight={weights[b]:6.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "bin_centers_normalized": torch.from_numpy(bin_centers_norm).float(),
        "bin_centers_raw":        torch.from_numpy(bin_centers_raw).float(),
        "class_weights":          torch.from_numpy(class_weights).float(),
        "action_mean":            torch.from_numpy(action_mean).float(),
        "action_std":             torch.from_numpy(action_std).float(),
        "n_bins":                 args.n_bins,
        "action_dim":             action_dim,
        "action_keys":            args.action_keys,
        "strategy":               args.strategy,
        "alpha":                  args.alpha,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\n[✓] saved → {out_path}")
    print(f"    point ACT's `action_bins_path` config at this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())