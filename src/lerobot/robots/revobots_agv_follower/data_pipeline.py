# -*- coding: utf-8 -*-
"""
End-to-end AGV data pipeline: split raw sessions, optionally semantic-mask
the video frames, then build a LeRobot dataset.

Lives at: src/lerobot/robots/revobots_agv_follower/data_pipeline.py

Steps:
    1.   split_session_agv   — chop raw session(s) into session_<i>/ chunks.
    1.5  mask_sessions_agv   — (optional) replace each session_<i>.mp4 with a
                               semantic-segmentation-masked version. .jsonl
                               files pass through untouched so frame indices
                               stay aligned with the masked frames.
    2.   data_convert_agv    — turn the chunks into a LeRobot HF dataset.

Examples:
    # Full run with masking
    python data_pipeline.py \
        --input-path ~/recordings/run_01 \
        --repo-id aadi/revobots_agv_v1 \
        --apply-mask \
        --push-to-hub

    # Reuse an already-split dir, mask in place, no push
    python data_pipeline.py \
        --split-dir ~/.cache/scout/lab/split_session/split_session_XXXX \
        --repo-id aadi/revobots_agv_v1 \
        --apply-mask --mask-in-place
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../robots/revobots_agv_follower
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # .../src
_MASK_DIR = os.path.join(_REPO_SRC, "lerobot", "cameras", "image_detection_tracking")

# The two step modules live inside the robot package dir; importing them via
# the package (lerobot.robots.revobots_agv_follower.*) would run its
# __init__.py, which pulls in ROS. Load them as flat top-level modules from
# their sibling directories.
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _MASK_DIR)

from split_session_agv import run_split      # noqa: E402
from data_convert_agv import run_convert     # noqa: E402
from mask_sessions_agv import run_mask       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Step 1: split ------------------------------------------------------
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-path", dest="input_path",
                     help="Raw session folder (or parent of many) to split.")
    src.add_argument("--split-dir", dest="split_dir",
                     help="Skip split step; use an existing split_session_<...>/ dir.")
    p.add_argument("--split-out", dest="split_out", default=None,
                   help="Where the split step writes its chunks.")
    p.add_argument("--segment-minutes", type=int, default=2)

    # --- Step 1.5: mask -----------------------------------------------------
    p.add_argument("--apply-mask", action="store_true",
                   help="Run semantic segmentation on every split mp4 before "
                        "feeding it to the LeRobot converter.")
    p.add_argument("--mask-in-place", action="store_true",
                   help="Overwrite mp4s inside the split dir (originals kept "
                        "as session_<i>.raw.mp4). Default writes a mirror tree.")
    p.add_argument("--mask-out", default=None,
                   help="Output dir for masked mirror tree (default: "
                        "<split_dir>_masked/). Ignored if --mask-in-place.")
    p.add_argument("--seg-mode", choices=["semantic", "segmentation"],
                   default="semantic")
    p.add_argument("--seg-weights", default="yolo26n-sem.pt")
    p.add_argument("--det-weights", default="yolo26n.pt")
    p.add_argument("--seg-conf", type=float, default=0.25)
    p.add_argument("--overlay-alpha", type=float, default=0.0,
                   help="0 = pure black bg (matches video_mask_batch.py). "
                        ">0 fades the raw frame in behind the mask.")
    p.add_argument("--det-classes", type=int, nargs="*",
                   default=[0, 1, 9, 10, 11, 12])

    # --- Step 2: convert ----------------------------------------------------
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", default="Drive the AGV.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--robot-type", default="revobots_agv_follower")
    p.add_argument("--image-writer-threads", type=int, default=4)
    p.add_argument("--video-encoding-batch-size", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--no-obs-processor", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Step 1: split ------------------------------------------------------
    if args.split_dir:
        parent_dir = args.split_dir
        print(f"[pipeline] Skipping split; using: {parent_dir}\n")
    else:
        print("[pipeline] === Step 1/3: splitting sessions ===")
        parent_dir = run_split(
            args.input_path,
            output_root=args.split_out,
            fps=args.fps,
            segment_minutes=args.segment_minutes,
        )
        if not parent_dir:
            raise SystemExit("[pipeline] Split step produced no output; aborting.")

    # --- Step 1.5: mask (optional) -----------------------------------------
    if args.apply_mask:
        print("\n[pipeline] === Step 2/3: masking session videos ===")
        parent_dir = run_mask(
            parent_dir=parent_dir,
            output_dir=args.mask_out,
            seg_mode=args.seg_mode,
            seg_weights=args.seg_weights,
            det_weights=args.det_weights,
            det_classes_to_keep=tuple(args.det_classes),
            seg_conf=args.seg_conf,
            overlay_alpha=args.overlay_alpha,
            in_place=args.mask_in_place,
        )
    else:
        print("\n[pipeline] Skipping mask step (--apply-mask not set).")

    # --- Step 2: convert ---------------------------------------------------
    print("\n[pipeline] === Step 3/3: building LeRobot dataset ===")
    dataset_root = run_convert(
        parent_dir=parent_dir,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        robot_type=args.robot_type,
        image_writer_threads=args.image_writer_threads,
        video_encoding_batch_size=args.video_encoding_batch_size,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
        no_videos=args.no_videos,
        no_obs_processor=args.no_obs_processor,
    )

    print(f"\n[pipeline] Done. Dataset at: {dataset_root}")


if __name__ == "__main__":
    main()