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


def compute_bin_centers_signed_log(min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    """Symmetric signed-logarithmic bin centers.

    Motivation (the steering problem):
      - uniform wastes bins: with ±3.5 range and a >90%% zero-spike, most of the
        bins land in the sparsely-populated tail and the few bins near zero are
        too coarse to resolve small corrective turns.
      - quantile compresses the range: the zero-spike eats most percentile mass,
        so edges cluster near zero and the extremes (±3.5) collapse into 1-2 bins,
        losing the ability to command a hard turn.

    Signed-log fixes both: fine resolution near zero (small corrections AND the
    bulk of the data), progressively coarser toward the tails (rare large turns,
    where exact magnitude matters less — a 3.2 vs 3.4 rad/s turn is
    operationally the same). The full range is preserved end-to-end.

    Centers are symmetric about zero and include an exact zero center when
    n_bins is odd (so straight-driving frames map to a dedicated zero bin).

    Works in whatever space you pass in (we pass normalized space, consistent
    with the other strategies).
    """
    # Symmetric half-range so positive and negative turns are binned identically.
    half = max(abs(min_val), abs(max_val))

    if n_bins % 2 == 1:
        # Odd: one center at exactly 0, (n_bins-1)/2 on each side.
        per_side = (n_bins - 1) // 2
        # log-spaced magnitudes from a small epsilon up to `half`.
        # epsilon controls how fine the smallest non-zero bin is.
        eps = half / 1000.0 if half > 0 else 1e-6
        mags = np.geomspace(eps, half, per_side)
        centers = np.concatenate([-mags[::-1], [0.0], mags])
    else:
        # Even: no exact zero; smallest bins straddle zero symmetrically.
        per_side = n_bins // 2
        eps = half / 1000.0 if half > 0 else 1e-6
        mags = np.geomspace(eps, half, per_side)
        centers = np.concatenate([-mags[::-1], mags])

    return centers.astype(np.float64)


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
    ap.add_argument("--strategy", choices=("uniform", "quantile", "signed_log"), default="uniform")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Class-weight exponent (0=uniform, 0.5=sqrt-inv-freq, 1=inv-freq).")
    ap.add_argument("--action-keys", nargs="+", default=["lin_x", "ang_z"],
                    help="Names of action components for logging only.")
    return ap.parse_args()


def print_bin_diagnostics(
    name,
    actions_norm_d,
    actions_raw_d,
    centers_norm,
    centers_raw,
    bin_idx,
    weights,
    n_bins,
    hist_width=40,
):
    """Full analysis of the binning for one action dimension.

    Sections:
      1. Distribution stats of the raw values (mean/std/min/max/quantiles/zero mass)
      2. Full per-bin table: center, count, freq %, cumulative %, class weight
      3. Text histogram of frequency across bins
      4. Coverage / quantization analysis (bin width, half-bin error, empty bins)
      5. Weight distribution summary
    """
    N = actions_raw_d.shape[0]
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    freq = counts / max(N, 1)
    cum = np.cumsum(freq)

    sep = "    " + "-" * 72
    print("")
    print(sep)
    print(f"    [{name}]  ({N} frames)")
    print(sep)

    # 1. distribution stats (raw units)
    q = np.percentile(actions_raw_d, [1, 5, 25, 50, 75, 95, 99])
    zero_frac = np.mean(np.abs(actions_raw_d) < 1e-3)
    print("    distribution (raw units):")
    print(f"      mean={actions_raw_d.mean():+.4f}  std={actions_raw_d.std():.4f}  "
          f"min={actions_raw_d.min():+.4f}  max={actions_raw_d.max():+.4f}")
    print(f"      q01={q[0]:+.4f}  q05={q[1]:+.4f}  q25={q[2]:+.4f}  "
          f"q50={q[3]:+.4f}  q75={q[4]:+.4f}  q95={q[5]:+.4f}  q99={q[6]:+.4f}")
    print(f"      zero-spike (fraction |value|<1e-3): {zero_frac*100:.1f}%")

    # 2. full per-bin table
    print(f"\n    per-bin table (all {n_bins} bins):")
    print(f"      {'bin':>3}  {'center_raw':>11}  {'center_norm':>11}  "
          f"{'count':>7}  {'freq%':>7}  {'cum%':>7}  {'weight':>7}")
    mode_count = counts.max()
    for b in range(n_bins):
        flag = ""
        if counts[b] == 0:
            flag = "  <EMPTY>"
        elif counts[b] == mode_count:
            flag = "  <- mode"
        print(f"      {b:>3d}  {centers_raw[b]:>+11.4f}  {centers_norm[b]:>+11.4f}  "
              f"{counts[b]:>7d}  {freq[b]*100:>6.2f}%  {cum[b]*100:>6.2f}%  "
              f"{weights[b]:>7.3f}{flag}")

    # 3. text histogram
    print(f"\n    frequency histogram (bar ~ share of frames in bin):")
    fmax = freq.max() if freq.max() > 0 else 1.0
    for b in range(n_bins):
        bar = "#" * int(round(freq[b] / fmax * hist_width))
        print(f"      {b:>3d} raw={centers_raw[b]:>+8.3f} |{bar:<{hist_width}}| {freq[b]*100:5.2f}%")

    # 4. coverage / quantization
    n_empty = int((counts == 0).sum())
    n_nonempty = n_bins - n_empty
    raw_widths = np.diff(centers_raw)
    quant_err = np.abs(actions_raw_d - centers_raw[bin_idx])
    print(f"\n    coverage / quantization:")
    print(f"      bins used (non-empty): {n_nonempty}/{n_bins} "
          f"({n_nonempty/n_bins*100:.0f}%)   empty bins: {n_empty}")
    print(f"      raw bin width: mean={raw_widths.mean():.4f}  "
          f"min={raw_widths.min():.4f}  max={raw_widths.max():.4f}")
    print(f"      half-bin (theoretical min quantization error): ~{raw_widths.mean()/2:.4f}")
    print(f"      actual quantization error |value-center|: "
          f"mean={quant_err.mean():.4f}  max={quant_err.max():.4f}")

    # 5. weight distribution
    per_frame_w = weights[bin_idx]
    print(f"\n    class weights:")
    print(f"      range: [{weights.min():.3f}, {weights.max():.3f}]   "
          f"spread: {weights.max()/max(weights.min(),1e-9):.1f}x")
    print(f"      mean weight (should be ~1.0): {weights.mean():.3f}")
    print(f"      data-weighted mean (per-frame): {per_frame_w.mean():.3f}  "
          f"(>1 => rare bins emphasized on average)")
    print(sep)


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
        elif args.strategy == "signed_log":
            norm_min = float(actions_norm[:, d].min())
            norm_max = float(actions_norm[:, d].max())
            centers_norm = compute_bin_centers_signed_log(norm_min, norm_max, args.n_bins)
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

        # Diagnostics — full per-bin frequency + weight analysis
        print_bin_diagnostics(
            name           = name,
            actions_norm_d = actions_norm[:, d],
            actions_raw_d  = actions[:, d],
            centers_norm   = centers_norm,
            centers_raw    = centers_raw,
            bin_idx        = bin_idx,
            weights        = weights,
            n_bins         = args.n_bins,
        )

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