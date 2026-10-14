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

import logging

from ..koch_leader_remote.koch_leader_remote import KochLeaderRemote
from .config_so_leader_remote import SOLeaderRemoteConfig

logger = logging.getLogger(__name__)

# Same six joints as `SOLeader`, in the same order.
MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class SOLeaderRemote(KochLeaderRemote):
    """
    Receiving half of a network-split SO-100 / SO-101 leader arm.

    The physical arm lives on another machine running
    :pymod:`~lerobot.teleoperators.so_leader_remote.so_leader_remote_client`, which streams its joint
    positions here over UDP. This class owns no hardware: it binds a socket and hands the freshest action it
    has received to the teleoperation loop.

    The transport, sequence handling and freshness rules are exactly those of
    :class:`~lerobot.teleoperators.koch_leader_remote.KochLeaderRemote`; only the registered name, the
    default port and the joint list differ. Values arrive in whatever space the leader was configured with
    on the client (`use_degrees`, default True), which must match the follower.
    """

    config_class = SOLeaderRemoteConfig
    name = "so_leader_remote"

    def __init__(self, config: SOLeaderRemoteConfig):
        super().__init__(config)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}


SO100LeaderRemote = SOLeaderRemote
SO101LeaderRemote = SOLeaderRemote
