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
Sending half of a network-split SO-100 / SO-101 leader arm.

Run this on the machine the physical SO leader is plugged into. It reads the arm at `--fps` and streams each
pose as a UDP datagram to a `so101_leader_remote` teleoperator running on the robot machine:

```shell
uv run python -m lerobot.teleoperators.so_leader_remote.so_leader_remote_client \
    --ip=192.168.1.42 --port=5002 --com=/dev/ttyACM1 --id=my_so101_leader
```

Being connectionless, there is nothing to connect or reconnect to: this keeps streaming whether or not the
far end is listening, and the server picks the stream up mid-flight whenever it starts.

By default values are streamed exactly as the leader reports them (`--use-degrees`, default on, must match the
follower's `use_degrees`): an SO leader drives an SO follower one to one. When the follower is an AgileX Nero,
pass `--is-follower-nero` and every pose is translated into Nero joint space here (offsets in
`so_remote_nero_config.py`) before it is sent.
"""

import argparse
import json
import logging
import socket
import time
import uuid

from lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client import display_action
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.teleoperators.so_leader_remote.so_remote_nero_config import SORemoteNeroConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def run_remote_client(
    ip: str,
    port: int,
    com_port: str,
    robot_id: str,
    fps: int,
    use_degrees: bool = True,
    is_follower_nero: bool | None = None,
) -> None:
    nero_cfg = SORemoteNeroConfig()
    if is_follower_nero is not None:
        nero_cfg.is_follower_nero = is_follower_nero
    if nero_cfg.is_follower_nero and not use_degrees:
        logger.warning("Nero mapping is tuned for degrees; with --no-use-degrees the offsets will be wrong.")

    leader = SO101Leader(SO101LeaderConfig(port=com_port, id=robot_id, use_degrees=use_degrees))

    logger.info(f"Calibration file: {leader.calibration_fpath}")
    if not leader.calibration_fpath.is_file():
        logger.warning("No calibration file found for this id, you will be asked to calibrate.")

    leader.connect(calibrate=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # A new id per run tells the server its sequence numbers just restarted from zero.
    session = uuid.uuid4().hex[:8]
    seq = 0
    loop_s = 1 / fps

    units = "degrees" if use_degrees else "-100..100 range"
    target = "Nero joint space" if nero_cfg.is_follower_nero else f"raw SO values, {units}"
    logger.info(
        f"Streaming leader actions ({target}) to {ip}:{port} at {fps} fps (session {session}). Ctrl-C to stop."
    )
    try:
        while True:
            loop_start = time.perf_counter()

            action = nero_cfg.apply(leader.get_action())
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
    parser.add_argument("--port", type=int, default=5002, help="UDP port the teleoperator listens on.")
    parser.add_argument("--com", type=str, default="/dev/ttyACM1", help="Serial port of the SO leader arm.")
    parser.add_argument("--id", type=str, default="so101_leader", help="Id of the arm, selects its calibration file.")
    parser.add_argument("--fps", type=int, default=60, help="Rate at which the arm is read and streamed.")
    parser.add_argument(
        "--use-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read joints in degrees (default) or in the -100..100 range. Must match the follower.",
    )
    parser.add_argument(
        "--is-follower-nero",
        action="store_true",
        default=None,  # absent -> use is_follower_nero from so_remote_nero_config.py (False)
        help="Map poses into Nero joint space using so_remote_nero_config.py offsets.",
    )
    args = parser.parse_args()

    init_logging()
    run_remote_client(
        args.ip, args.port, args.com, args.id, args.fps, args.use_degrees, args.is_follower_nero
    )


if __name__ == "__main__":
    main()
