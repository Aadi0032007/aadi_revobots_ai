#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Sending half of a network-split Koch leader arm.

Run this on the machine the physical Koch leader is plugged into. It reads the arm at `--fps` and streams
each pose as a UDP datagram to a `koch_leader_remote` teleoperator running on the robot machine:

```shell
uv run python -m lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client \
    --ip=192.168.1.42 --port=5001 --com=COM3 --id=my_koch_leader
```

Being connectionless, there is nothing to connect or reconnect to: this keeps streaming whether or not the
far end is listening, and the server picks the stream up mid-flight whenever it starts.
"""

import argparse
import json
import logging
import socket
import time
import uuid

from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

logger = logging.getLogger(__name__)


def display_action(action: dict[str, float], seq: int, loop_s: float) -> None:
    """Print the live joint table in place, leaving the cursor where it started."""
    width = max(len(motor) for motor in action)
    lines = [
        "-" * (width + 12),
        f"{'MOTOR':<{width}} | {'POS':>8}",
        *(f"{motor:<{width}} | {val:>8.2f}" for motor, val in action.items()),
        "-" * (width + 12),
        f"packet {seq}  |  {loop_s * 1e3:6.2f}ms  ({1 / loop_s if loop_s else 0:.0f} Hz)",
    ]
    print("\n".join(lines))
    move_cursor_up(len(lines))


def run_remote_client(
    ip: str,
    port: int,
    com_port: str,
    robot_id: str,
    fps: int,
    gripper_open_pos: float,
) -> None:
    leader = KochLeader(KochLeaderConfig(port=com_port, id=robot_id, gripper_open_pos=gripper_open_pos))

    logger.info(f"Calibration file: {leader.calibration_fpath}")
    if not leader.calibration_fpath.is_file():
        logger.warning("No calibration file found for this id, you will be asked to calibrate.")

    leader.connect(calibrate=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # A new id per run tells the server its sequence numbers just restarted from zero.
    session = uuid.uuid4().hex[:8]
    seq = 0
    loop_s = 1 / fps

    logger.info(f"Streaming leader actions to {ip}:{port} at {fps} fps (session {session}). Ctrl-C to stop.")
    try:
        while True:
            loop_start = time.perf_counter()

            action = leader.get_action()
            packet = json.dumps({"session": session, "seq": seq, "action": action})
            try:
                sock.sendto(packet.encode("utf-8"), (ip, port))
            except ConnectionResetError:
                # Windows surfaces an ICMP "port unreachable" from a previous send here: nothing is
                # listening on the far end yet. Keep streaming, it will catch up when it starts.
                logger.debug("No listener on the far end yet.")
            except OSError as e:
                logger.debug(f"Failed to send action packet {seq}: {e}")

            seq += 1
            display_action(action, seq, loop_s)

            precise_sleep(1 / fps - (time.perf_counter() - loop_start))
            loop_s = time.perf_counter() - loop_start
    except KeyboardInterrupt:
        pass
    finally:
        # Step past the in-place table so the shutdown logs don't overwrite it.
        print("\n" * (len(leader.action_features) + 4))
        sock.close()
        leader.disconnect()
        logger.info(f"Stopped after {seq} packets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP of the machine running the robot.")
    parser.add_argument("--port", type=int, default=5001, help="UDP port the teleoperator listens on.")
    parser.add_argument("--com", type=str, default="COM3", help="Serial port of the Koch leader arm.")
    parser.add_argument("--id", type=str, default="aadi", help="Id of the arm, selects its calibration file.")
    parser.add_argument("--fps", type=int, default=60, help="Rate at which the arm is read and streamed.")
    parser.add_argument("--gripper-open-pos", type=float, default=45.0, help="Gripper spring-back position.")
    args = parser.parse_args()

    init_logging()
    run_remote_client(args.ip, args.port, args.com, args.id, args.fps, args.gripper_open_pos)


if __name__ == "__main__":
    main()
