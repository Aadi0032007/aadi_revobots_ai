# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:55:11 2026

@author: Aadi
"""
"""
Split scout/lab sessions into 5-minute chunks.

Output:
  ~/.cache/scout/lab/split_session/split_session_<datetime>/
      ├── session_0/
      │   ├── session_0.mp4
      │   ├── session_0.jsonl
      │   ├── session_0.json            segment metadata (fps, size, depth shape)
      │   └── session_0_depth.u16       only when the recording has depth.u16
      ├── session_1/...

Depth: a RealSense recording (LAB/record.py, use_depth=True) carries depth.u16
next to video.mp4 — raw uint16 millimetre maps, frame i == video frame i. It is
sliced on the same [start:end] frame range as the video and JSONL, so every
segment stays aligned across all three files.
"""
import cv2
import os
import json
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

DEPTH_FILE = "depth.u16"     # written by LAB/record.py next to video.mp4


def get_args():
    parser = argparse.ArgumentParser(description="Split scout/lab sessions into 5-minute chunks.")
    parser.add_argument("input_path", help="Path to a single session folder OR a parent folder containing multiple sessions")
    return parser.parse_args()


def is_session_folder(path):
    return (os.path.isfile(os.path.join(path, "video.mp4"))
            and os.path.isfile(os.path.join(path, "data.jsonl")))


def find_sessions(input_path):
    if is_session_folder(input_path):
        return [(os.path.basename(os.path.normpath(input_path)), input_path)]

    sessions = []
    for name in sorted(os.listdir(input_path)):
        sub = os.path.join(input_path, name)
        if os.path.isdir(sub) and is_session_folder(sub):
            sessions.append((name, sub))
    return sessions


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_session_meta(session_path):
    """session.json written by LAB/record.py, or {} when absent / unreadable."""
    meta_path = os.path.join(session_path, "session.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f) or {}
    except Exception as exc:
        print(f"[!] could not read {meta_path}: {exc}")
        return {}


def open_depth(session_path, meta, fallback_hw):
    """Memory-map the recorder's depth.u16 as (n, H, W) uint16 mm, or None.

    Frame i of the file pairs with frame i of video.mp4 / row i of data.jsonl.
    The shape comes from session.json (depth_height / depth_width); sessions
    without those keys fall back to the video size.
    """
    name = meta.get("depth") or DEPTH_FILE
    path = os.path.join(session_path, name)
    if not os.path.isfile(path):
        return None
    h = int(meta.get("depth_height") or fallback_hw[0])
    w = int(meta.get("depth_width") or fallback_hw[1])
    n = os.path.getsize(path) // (h * w * 2)
    if n == 0:
        print(f"[!] {path} is empty — ignoring depth")
        return None
    return np.memmap(path, dtype=np.uint16, mode="r", shape=(n, h, w))


def split_one_session(session_name, session_path, output_root, start_session_idx, fps=15, segment_minutes=2):
    """Returns the next session_idx to use (so multi-session runs keep numbering continuous)."""
    video_path = os.path.join(session_path, "video.mp4")
    jsonl_path = os.path.join(session_path, "data.jsonl")

    frames_per_segment = segment_minutes * 60 * fps  # 4500

    print(f"\n[*] === Splitting input: {session_name} ===")
    json_data = load_jsonl(jsonl_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Could not open video {video_path}, skipping.")
        return start_session_idx

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_json = len(json_data)
    total_frames = min(total_frames_json, total_frames_vid)

    meta = load_session_meta(session_path)
    depth = open_depth(session_path, meta, (height, width))
    if depth is not None:
        total_frames = min(total_frames, depth.shape[0])

    print(f"[*] Frames — video: {total_frames_vid}, json: {total_frames_json}"
          f"{f', depth: {depth.shape[0]}' if depth is not None else ''}, using: {total_frames}")
    if depth is not None:
        print(f"[*] Depth — {depth.shape[2]}x{depth.shape[1]} uint16 mm → session_<i>_depth.u16")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pbar = tqdm(total=total_frames, desc=f"{session_name}", unit="frame")

    current_frame_idx = 0
    session_idx = start_session_idx

    while current_frame_idx < total_frames:
        seg_name = f"session_{session_idx}"
        seg_dir = os.path.join(output_root, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        seg_video_path = os.path.join(seg_dir, f"{seg_name}.mp4")
        seg_jsonl_path = os.path.join(seg_dir, f"{seg_name}.jsonl")

        out = cv2.VideoWriter(seg_video_path, fourcc, fps, (width, height))

        start_idx = current_frame_idx
        end_idx = min(current_frame_idx + frames_per_segment, total_frames)

        # Save JSONL (one object per line)
        json_slice = json_data[start_idx:end_idx]
        with open(seg_jsonl_path, 'w') as f:
            for row in json_slice:
                f.write(json.dumps(row) + "\n")

        # Depth slice over the same frame range, plus a sidecar describing the
        # segment so data_convert_agv.py never has to guess shapes.
        seg_meta = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": end_idx - start_idx,
            "source_session": session_name,
            "source_from": start_idx,
            "source_to": end_idx,
            "depth": None,
        }
        if depth is not None:
            seg_depth_name = f"{seg_name}_depth.u16"
            with open(os.path.join(seg_dir, seg_depth_name), "wb") as f:
                for a in range(start_idx, end_idx, 256):          # chunked: 256 frames ≈ 157 MB
                    f.write(np.ascontiguousarray(depth[a:min(a + 256, end_idx)]).tobytes())
            seg_meta.update(
                depth=seg_depth_name,
                depth_width=int(depth.shape[2]),
                depth_height=int(depth.shape[1]),
                depth_dtype="uint16",
                depth_unit="mm",
            )
        with open(os.path.join(seg_dir, f"{seg_name}.json"), "w") as f:
            json.dump(seg_meta, f, indent=2)

        # Write video frames
        for _ in range(start_idx, end_idx):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame_idx += 1
            pbar.update(1)

        out.release()
        session_idx += 1

    cap.release()
    pbar.close()
    return session_idx


def run_split(input_path, output_root=None, fps=15, segment_minutes=2):
    """Split all sessions under `input_path` into fixed-length chunks.

    Returns the output_root path containing session_0/, session_1/, ...
    (the exact layout data_convert_agv expects), or None if nothing was
    written. If `output_root` is not given, a timestamped folder under
    ~/.cache/scout/lab/split_session/ is created.
    """
    input_path = os.path.expanduser(input_path)

    if not os.path.isdir(input_path):
        print(f"[!] Not a directory: {input_path}")
        return None

    sessions = find_sessions(input_path)
    if not sessions:
        print(f"[!] No valid sessions (with video.mp4 + data.jsonl) found in {input_path}")
        return None

    if output_root is None:
        timestamp = datetime.now().strftime("%d%m%Y%H%M")
        output_root = os.path.expanduser(
            f"~/.cache/scout/lab/split_session/split_session_{timestamp}"
        )
    else:
        output_root = os.path.expanduser(output_root)
    os.makedirs(output_root, exist_ok=True)

    print(f"[*] Input:        {input_path}")
    print(f"[*] Output root:  {output_root}")
    print(f"[*] Sessions found: {len(sessions)}")

    next_idx = 0
    for session_name, session_path in sessions:
        next_idx = split_one_session(
            session_name, session_path, output_root, next_idx,
            fps=fps, segment_minutes=segment_minutes,
        )

    print(f"\n[+] All done. {next_idx} total segments written.")
    print(f"[+] Output: {output_root}")
    return output_root


def main():
    args = get_args()
    run_split(args.input_path)


if __name__ == "__main__":
    main()