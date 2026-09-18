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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("koch_leader_remote")
@dataclass
class KochLeaderRemoteConfig(TeleoperatorConfig):
    # Interface to bind the UDP socket to. "0.0.0.0" listens on all interfaces.
    host: str = "0.0.0.0"

    # UDP port the leader arm client streams its actions to.
    port: int = 5001

    # Actions older than this (measured on arrival, not with the client's clock) are considered stale:
    # `get_action` waits for a fresh one instead of handing a frozen pose to the robot.
    max_action_age_s: float = 0.5

    # How often to log that we're still waiting for the leader arm, while starving for actions.
    waiting_log_period_s: float = 2.0
