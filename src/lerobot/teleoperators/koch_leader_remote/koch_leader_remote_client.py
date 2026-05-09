# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 07:45:37 2026

@author: Aadi
"""

#!/usr/bin/env python

import logging
import socket
import json
import time
import os
import sys
import argparse
from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def move_cursor_up(n):
    sys.stdout.write(f"\033[{n}A")

def precise_sleep(duration):
    if duration > 0:
        time.sleep(duration)

def run_remote_client(server_ip, port, robot_id, fps):
    config = KochLeaderConfig(port="COM3", id=robot_id, gripper_open_pos=45.0)
    leader = KochLeader(config)

    print("-" * 50)
    print(f"EXPECTED CALIBRATION PATH: {leader.calibration_fpath}")
    if os.path.exists(leader.calibration_fpath):
        print("STATUS: [SUCCESS] Calibration file found!")
    else:
        print("STATUS: [WARNING] Calibration file NOT FOUND.")
    print("-" * 50)

    leader.connect(calibrate=True)

    while True:
        client_sock = None
        try:
            print(f"[Client] Connecting to {server_ip}:{port}...")
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.connect((server_ip, port))
            print("[Client] Connected.\n")

            while True:
                loop_start = time.perf_counter()
                data = client_sock.recv(1024).decode('utf-8').strip()
                if not data: break
                
                if "GET_ACTION" in data:
                    robot_action_to_send = leader.get_action()
                    
                    # --- DISPLAY LOGIC ---
                    display_len = max(len(m) for m in robot_action_to_send.keys()) if robot_action_to_send else 20
                    print("")
                    print("-" * (display_len + 10))
                    print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                    
                    for motor, value in robot_action_to_send.items():
                        print(f"{motor:<{display_len}} | {value:>7.2f}")
                    
                    # Move cursor back up for the next refresh (Motors + Header + Footer)
                    move_cursor_up(len(robot_action_to_send) + 2)
                    client_sock.sendall((json.dumps(robot_action_to_send) + "\n").encode('utf-8'))

                    dt_s = time.perf_counter() - loop_start
                    precise_sleep(1 / fps - dt_s)
                    loop_s = time.perf_counter() - loop_start
                    print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                    move_cursor_up(2)

        except (ConnectionRefusedError, socket.timeout, ConnectionResetError):
            print("[Client] Server lost. Retrying in 2 seconds...")
            time.sleep(0.05)
        finally:
            if client_sock: client_sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--id", type=str, default="aadi")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    run_remote_client(args.ip, args.port, args.id, args.fps)