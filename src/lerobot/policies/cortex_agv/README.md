# REVO Scout AGV — ACT Policy: Discretization, Bins, Training & Inference

This document explains the *why* and the *what* behind the changes made to the
ACT (Action Chunking Transformer) policy for the REVO Scout AGV, how the
precomputed bin file (`.pt`) is used during training and inference, and how to
run inference on the robot.

---

## 1. The problem we were solving

### 1.1 The symptom

The first ACT policy, trained with the stock L1 regression loss, learned to
drive forward well but **refused to turn**. The offline eval showed:

```
lin_x   MAE=0.0152  RMSE=0.0423  r=0.978  R²=0.954   ← excellent
ang_z   MAE=0.1503  RMSE=0.6545  r=0.073  R²=-0.002  ← total collapse
ang_z pred range = [-0.050, +0.006]   GT range = [-3.5, +3.5]
```

The policy predicted essentially zero angular velocity no matter what the road
was doing. On a hard-right turn where the operator held −3.5 rad/s for 45
frames, the policy output +0.002 the whole time.

### 1.2 The root cause

The dataset is dominated by straight driving. Over 90% of `ang_z` values are
essentially zero:

```
ang_z   q50 = +0.0005   q90 = +0.004   q01 = -2.13   q99 = +2.46
```

**L1 loss is minimized by the median of the target distribution.** When 90% of
the targets are zero, the median is zero, so the model collapses to predicting
zero everywhere. This is a mode collapse driven by class imbalance in a
regression setting — the rare turning frames contribute too little to the loss
to move the model off the zero solution.

### 1.3 The deployment constraints

We could not fix this the easy way (collect more turning data), because:

- The operational route is mostly straight — we can't manufacture turns.
- We can't downsample straight frames (chunking makes naive frame removal wrong).
- We can't drop GPS from observations (the policy needs the route signal).
- We can't use horizontal-flip augmentation.

So the fix had to make the model **learn harder from the turns that already
exist**, not generate new ones.

---

## 2. What we did

Two coordinated changes, plus an inference deployment path.

### 2.1 Action discretization with class-weighted cross-entropy (Method 1)

Instead of regressing a continuous action with L1, we:

1. Split each action dimension's range into **K discrete bins** (default 31).
2. Change the action head to output **logits over bins** instead of a scalar.
3. Train with **class-weighted cross-entropy** — rare bins (sharp turns,
   full throttle) get higher weight, so the loss can no longer ignore them.
4. At inference, convert the bin distribution back to a continuous value via
   the **expected value** over bin centers (softmax · centers).

This reframes "predict a number" as "predict which bucket," which is robust to
the zero-spike: the rare turning buckets are explicitly weighted up so the
gradient pays attention to them.

### 2.2 CE + auxiliary L1 hybrid loss

Cross-entropy alone answers *"which bin?"* but does not directly penalize the
continuous expected-value output. If the softmax is split across two neighboring
bins, the bin *index* can be right while the expected value lands between bins —
correct direction, wrong magnitude.

The hybrid loss adds a small L1 term on the expected-value action:

```
total = ce_loss + aux_l1_weight * L1(expected_value, target) + kl_weight * kld
```

The L1 gradient flows back through the softmax and tightens the distribution
around the correct bin, improving magnitude accuracy without reintroducing the
zero-collapse (as long as `aux_l1_weight` stays small — start at 0.1).

### 2.3 Better bin placement (signed-log strategy)

Uniform bins waste resolution: with a ±3.5 range and a huge zero-spike, most
bins land in the sparse tail while the few bins near zero are too coarse to
resolve small corrective turns. Quantile bins have the opposite failure — the
zero-spike eats the percentile mass and the extremes collapse into 1–2 bins,
losing the ability to command a hard turn.

**Signed-log** binning fixes both: fine resolution near zero (small corrections
*and* the bulk of the data), progressively coarser toward the tails (rare large
turns, where exact magnitude matters less), full ±range preserved end to end,
with a dedicated exact-zero bin.

---

## 3. The `.pt` bin file — why it exists and what's in it

### 3.1 Why a separate file

The bin centers and class weights are computed **once, offline**, from the
dataset's action distribution — before training starts. They are not learned
parameters; they are fixed reference values the model needs at initialization.
Precomputing them in a standalone file (`compute_action_bins.py` → `.pt`) means:

- The exact same bins are used for training and inference (no drift).
- The bins are computed from the same normalization stats the policy uses.
- You can regenerate bins for a new dataset without touching model code.

### 3.2 What's inside

`compute_action_bins.py` produces a `.pt` file containing:

| Key                       | Shape              | Meaning                                            |
|---------------------------|--------------------|----------------------------------------------------|
| `bin_centers_normalized`  | (action_dim, K)    | Bin centers in **normalized** (MEAN_STD) space     |
| `bin_centers_raw`         | (action_dim, K)    | Same centers in raw units (human-readable only)    |
| `class_weights`           | (action_dim, K)    | Per-bin CE weights (rare bins weighted up)         |
| `action_mean`, `action_std` | (action_dim,)    | The normalization stats used                       |
| `n_bins`, `action_dim`, `strategy`, `alpha` | scalars | Metadata for reproducibility          |

### 3.3 How the bins are computed

For each action dimension (`lin_x`, `ang_z`):

1. **Load all actions** from the dataset.
2. **Normalize** with MEAN_STD using the dataset's own stats (identical to what
   the policy's preprocessor applies during training). Binning happens in
   normalized space so it lines up with the values the model actually sees.
3. **Place bin centers** according to the chosen strategy:
   - `uniform`   — evenly spaced across the observed range.
   - `quantile`  — edges at percentiles (equal mass per bin).
   - `signed_log`— symmetric, log-spaced magnitudes, dedicated zero bin.
4. **Assign every frame to its nearest bin**, count per-bin frequency.
5. **Compute class weights** inversely proportional to bin frequency:
   ```
   weight[b] = 1 / (count[b] + 1) ** alpha      (then normalized to mean 1)
   ```
   - `alpha = 0.0` → uniform (no rebalancing)
   - `alpha = 0.5` → sqrt-inverse-frequency
   - `alpha = 1.0` → full inverse-frequency (aggressive)
   Higher alpha = more attention to rare bins. Too high overcorrects and
   produces false turns; for this dataset `alpha = 0.1` was the sweet spot.

### 3.4 Generating the file

```bash
python3 compute_action_bins.py \
    --dataset-repo-id revolabs/scout_dataset_03 \
    --out action_bins_uniform_a01.pt \
    --n-bins 31 \
    --strategy uniform \
    --alpha 0.1
```

For the signed-log strategy:

```bash
python3 compute_action_bins.py \
    --dataset-repo-id revolabs/scout_dataset_03 \
    --out action_bins_signedlog_a01.pt \
    --n-bins 31 \
    --strategy signed_log \
    --alpha 0.1
```

---

## 4. How the bins are used **during training**

The config points at the bin file and enables discretization:

```python
discretize_actions: bool = True
n_action_bins: int = 31
action_bins_path: str | None = ".../action_bins_uniform_a01.pt"
aux_l1_weight: float = 0.1
```

At policy init (`ACT.__init__`):

1. The `.pt` file is loaded.
2. `bin_centers_normalized` and `class_weights` are registered as **buffers**
   on the model — they travel with the checkpoint and are not trained.
3. The action head becomes `nn.Linear(dim_model, action_dim * n_bins)` — it
   emits one logit per bin per action dimension.

In the forward/loss pass (`ACTPolicy.forward`):

1. The model produces per-bin **logits**, shape `(B, chunk, action_dim, n_bins)`.
2. The continuous target action (already normalized by the preprocessor) is
   mapped to its **nearest bin index** using the stored bin centers.
3. **Class-weighted cross-entropy** is computed per action dimension, using
   `class_weights` so rare bins dominate the gradient, then masked over
   non-padded frames.
4. If `aux_l1_weight > 0`, the **expected-value action** (softmax · centers) is
   also compared to the target with L1, and added in:
   ```
   main_loss = ce_loss + aux_l1_weight * aux_l1
   ```
5. The VAE KL term is added as usual: `loss = main_loss + kl_weight * kld`.

So training optimizes bin classification (CE) plus, optionally, magnitude
accuracy of the expected value (aux L1).

---

## 5. How the bins are used **during inference**

At inference the model runs the same forward pass, but only the head output
matters (`ACT.forward`, discretized branch):

1. The action head emits logits, reshaped to `(B, chunk, action_dim, n_bins)`.
2. **Softmax** over the bin dimension → a probability distribution per action.
3. **Expected value** = `Σ (probability · bin_center)` over bins → a single
   continuous action per dimension, in normalized space.
4. The policy's postprocessor **un-normalizes** it back to raw action units
   using `action_mean` / `action_std`.

The result is a normal continuous `(lin_x, ang_z)` action — the discretization
is entirely internal. Downstream code never sees bins; it sees a velocity
command, exactly as it would from a continuous-regression policy.

The bins matter at inference only because they are baked into the checkpoint as
buffers. **This is why a checkpoint is locked to the bins it was trained with —
you cannot swap the `.pt` file for an existing checkpoint.** Changing bins
requires a fresh training run.

---

## 6. The full data → deployment pipeline

```
  RECORDING (teleop.py + record.py)
    operator drives, camera + motion + GPS logged per frame
    JSONL: linear_velocity, angular_velocity, gps fields
    MP4:   camera frames
        │
        ▼
  SPLIT (split_session_agv.py)
    long recordings → 2-minute episode chunks
        │
        ▼
  CONVERT (data_convert_agv.py)
    per frame: MP4 frame (BGR→RGB) + JSONL row
    → obs {lin_x, ang_z, lat, long, orientation, "front"} + action {lin_x, ang_z}
    → LeRobot HuggingFace dataset
        │
        ▼
  BIN PRECOMPUTE (compute_action_bins.py)      ← produces the .pt file
    dataset actions → normalized → bins + class weights → .pt
        │
        ▼
  TRAIN (lerobot-train, ACTConfig discretize_actions=True)
    loads .pt at init, CE (+ aux L1) loss, saves checkpoints
        │
        ▼
  EVAL (agv_offline_eval.py)
    per-checkpoint metrics → pick best step
        │
        ▼
  INFERENCE (lab_inference.py / lab_inference_udp.py)
    live camera + GPS → policy → velocity command → robot
```

---

## 7. Running inference

Two inference scripts exist. **They differ only in how the command reaches the
robot.**

### 7.1 `lab_inference.py` — via rclpy / MotionController

Uses the in-process `MotionController` (rclpy) to publish `/cmd_vel` directly.
Use this when running inference as the sole controller of the robot.

```bash
python3 lab_inference.py \
    --policy-path .../checkpoints/080000/pretrained_model \
    --dataset-repo-id revolabs/scout_dataset_03 \
    --device cuda \
    --duration 60 \
    --send \
    --temporal-ensemble-coeff 0.01 \
    --ang-deadband 0.15
```

### 7.2 `lab_inference_udp.py` — via UDP to teleop.py

No rclpy. Sends JSON motion packets to teleop.py's motion port
(`127.0.0.1:55999`), the same port and format the gamepad uses. teleop.py
receives them and publishes `/cmd_vel` on its side. **teleop.py must be
running.**

```bash
python3 lab_inference_udp.py \
    --policy-path .../checkpoints/080000/pretrained_model \
    --dataset-repo-id revolabs/scout_dataset_03 \
    --device cuda \
    --duration 60 \
    --send \
    --temporal-ensemble-coeff 0.01 \
    --ang-deadband 0.15
```

### 7.3 Flags that matter

| Flag                          | Purpose                                                              |
|-------------------------------|---------------------------------------------------------------------|
| `--send`                      | Actually move the robot. Without it: print predictions only (dry run). |
| `--duration N`                | Stop after N seconds. Default: run until Ctrl+C.                     |
| `--temporal-ensemble-coeff C` | Re-predict every frame, average overlapping chunks. Reduces false turns. |
| `--ang-deadband D`            | Zero `ang_z` predictions below magnitude D. Cleans up noise floor.   |
| `--device`                    | `cuda` or `cpu`. **cuda strongly recommended** (see §8).            |

### 7.4 Always dry-run first

Run without `--send` and inspect the printed table:

- `frame_mean` — confirms the camera is delivering real imagery (not blank).
- `obs_lin` / `obs_ang` — the state fed **into** the policy (its own last
  command, closed-loop).
- `ang_z_pred` — raw policy output, with `DB` marking deadbanded-to-zero frames.
- `ang_z_cmd` / `ang_z_sent` — what actually goes to the robot.
- `timestamp` — check frame-to-frame gaps stay near the 15 Hz budget (~67 ms).

Only `--send` once the dry-run looks sane.

---

## 8. Why the scale handling matters (ang_z)

The `MotionController` applies an internal `ang_z_scale` (0.20) — turning is
attenuated so it feels proportional to forward speed. Whether inference needs to
account for this depends on **which dataset the model was trained on**:

- **Old dataset (raw ang_z):** the recorder stored the pre-scale gamepad value.
  The policy predicts raw ang_z → send it directly; the ×0.20 is applied
  downstream (by `MotionController` or by teleop.py on the UDP path).
- **New dataset (scaled ang_z):** the recorder stored the post-scale value
  (`published_state`). The policy predicts scaled ang_z → divide by `ang_z_scale`
  before `motion.command()` so the downstream ×0.20 lands you back on the
  intended value.

The observation state fed back into the policy must be in the **same space** the
dataset stored (raw for the old model, scaled for the new). The inference
scripts handle this consistently on both the input (obs) and output (command)
sides.

---

## 9. Deployment-time false-turn mitigation

The discretized model gets turn **direction** right (100% when turning) but has
two residual issues on straight roads: small-magnitude noise ("false turns")
and imprecise magnitude on real turns. Two inference-time mitigations, no
retrain:

1. **Temporal ensembling** (`--temporal-ensemble-coeff 0.01`): the biggest win.
   The policy re-predicts every frame and averages overlapping chunk predictions
   with exponential weighting, so isolated false-turn spikes get diluted.
2. **Deadband** (`--ang-deadband 0.15`): zeroes predictions below a threshold,
   cleaning up the residual noise floor on straight frames.

The deeper magnitude fix is the CE + L1 hybrid retrain (§2.2), which sharpens
the distribution so small predictions become meaningful again and the deadband
can be lowered.

---

## 10. Latency note (CPU vs GPU)

Temporal ensembling runs a full ACT forward pass **every frame** instead of once
per chunk. On CPU this measured ~1 second per forward pass, collapsing the
control loop to ~1 Hz — far below the 15 Hz the policy was trained at, and
**unsafe to drive on**. Run inference on CUDA (e.g. Jetson Orin NX), where the
forward pass fits inside the 67 ms frame budget. Watch the timestamp column: if
frame-to-frame gaps exceed ~70 ms, do not `--send`.

---

## 11. File reference

| File                        | Role                                                        |
|-----------------------------|-------------------------------------------------------------|
| `compute_action_bins.py`    | Precomputes the `.pt` bin file (centers + class weights).   |
| `configuration_act.py`      | ACT config: `discretize_actions`, `n_action_bins`, `action_bins_path`, `aux_l1_weight`. |
| `modeling_act.py`           | Discretized action head, CE + aux L1 loss, expected-value inference. |
| `agv_offline_eval.py`       | Offline eval against a checkpoint.                          |
| `lab_inference.py`          | Live inference via rclpy / MotionController.                |
| `lab_inference_udp.py`      | Live inference via UDP to teleop.py (no rclpy).             |

---

## 12. Quick-start checklist for a new dataset

1. Build the LeRobot dataset (record → split → convert) and push to HuggingFace.
2. `compute_action_bins.py` → generate `action_bins_*.pt`.
3. Point `action_bins_path` in `configuration_act.py` at the new `.pt`; set
   `discretize_actions=True`, `aux_l1_weight=0.1`.
4. `lerobot-train` → produce checkpoints.
5. `agv_offline_eval.py` on each checkpoint → pick the best step.
6. Dry-run `lab_inference.py` (or `_udp`) → verify camera, GPS, timing.
7. Add `--send` with `--temporal-ensemble-coeff 0.01 --ang-deadband 0.15`.

**Change one thing at a time.** When comparing bin strategies or the hybrid
loss, vary a single knob per training run and keep the eval fixed, or you won't
know which change moved the numbers.