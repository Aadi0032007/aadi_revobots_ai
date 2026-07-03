# -*- coding: utf-8 -*-
"""
End-to-end AGV data pipeline: split raw sessions, then build a LeRobot dataset.

Runs, in order:
    1. split_session_agv  — chop raw session(s) (video.mp4 + data.jsonl) into
                            fixed-length session_<i>/ chunks.
    2. data_convert_agv   — turn those chunks into a LeRobot HuggingFace dataset.

The split step's output folder (a timestamped dir under
~/.cache/scout/lab/split_session/) is captured and fed straight into the
convert step, so you only run one command.

Example:
    python data-pipeline.py \
        --input-path ~/recordings/run_01 \
        --repo-id user/revobots_agv_v1 \
        --task "Drive the AGV." \
        --push-to-hub

To reuse an already-split folder and skip step 1:
    python data-pipeline.py --split-dir ~/.cache/scout/lab/split_session/split_session_XXXX \
        --repo-id user/revobots_agv_v1
"""

import argparse
import os
import sys

# The two step modules live inside the robot package dir, but importing them
# via the package (lerobot.robots.revobot_agv_follower.*) would run that
# package's __init__.py, which pulls in ROS (geometry_msgs, cameras, ...).
# We only need the pure data helpers, so load the files directly by putting
# their directory on sys.path and importing them as top-level modules.
_ROBOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "lerobot", "robots", "revobot_agv_follower",
)
sys.path.insert(0, _ROBOT_DIR)

from split_session_agv import run_split      # noqa: E402
from data_convert_agv import run_convert     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Step 1: split ------------------------------------------------------
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-path", dest="input_path",
                     help="Raw session folder (or parent of many) to split. "
                          "Runs the split step, then converts its output.")
    src.add_argument("--split-dir", dest="split_dir",
                     help="Skip the split step and convert an existing "
                          "split_session_<...>/ folder directly.")
    p.add_argument("--split-out", dest="split_out", default=None,
                   help="Where the split step writes its chunks. Default: a "
                        "timestamped dir under ~/.cache/scout/lab/split_session/.")
    p.add_argument("--segment-minutes", type=int, default=2,
                   help="Length of each split chunk in minutes (default 2).")

    # --- Step 2: convert ----------------------------------------------------
    p.add_argument("--repo-id", required=True,
                   help="HuggingFace dataset repo id, e.g. user/revobots_agv_v1")
    p.add_argument("--task", default="Drive the AGV.",
                   help="Task description string stored with every frame.")
    p.add_argument("--fps", type=int, default=15,
                   help="Recording fps; shared by both steps (default 15).")
    p.add_argument("--robot-type", default="revobots_agv_follower")
    p.add_argument("--image-writer-threads", type=int, default=4)
    p.add_argument("--video-encoding-batch-size", type=int, default=1)
    p.add_argument("--resume", action="store_true",
                   help="Resume into an existing dataset (convert step).")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push to HuggingFace Hub after writing all episodes.")
    p.add_argument("--no-videos", action="store_true",
                   help="Store images as PNG instead of encoded video.")
    p.add_argument("--no-obs-processor", action="store_true",
                   help="Skip the observation processor in the convert step.")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Step 1: split (unless a pre-split dir was supplied) ----------------
    if args.split_dir:
        parent_dir = args.split_dir
        print(f"[pipeline] Skipping split; using existing chunks: {parent_dir}\n")
    else:
        print("[pipeline] === Step 1/2: splitting sessions ===")
        parent_dir = run_split(
            args.input_path,
            output_root=args.split_out,
            fps=args.fps,
            segment_minutes=args.segment_minutes,
        )
        if not parent_dir:
            raise SystemExit("[pipeline] Split step produced no output; aborting.")

    # --- Step 2: convert ---------------------------------------------------
    print("\n[pipeline] === Step 2/2: building LeRobot dataset ===")
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