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
SO-100 / SO-101 leader -> AgileX Nero joint mapping, applied on the leader PC by `so_leader_remote_client`.

The Nero follower takes actions in its own joint space (degrees, gripper in its native units) and applies no
leader-specific math. Each leader owns the translation into that space, so any leader can drive the Nero
without touching the follower. Edit the numbers here, not in `agx_nero_follower.py`.

For every SO joint: `nero = sign * so + offset`. The gripper is `min(so * gripper_scale, gripper_max)`.
SO joints not listed in `joints` are passed through unchanged.

The numbers assume the leader streams degrees (`--use-degrees`, the default). They are NOT the Koch numbers:
the Koch leader reports joints in -100..100, the SO leader in real degrees around its own calibration zero.
Tune them by posing the SO leader and the Nero alike and reading off each joint's difference.
"""

from dataclasses import dataclass, field


@dataclass
class JointMap:
    sign: float = 1.0
    offset: float = 0.0


@dataclass
class SORemoteNeroConfig:
    # False (default) -> raw SO leader values are sent (e.g. when the follower is an SO arm).
    # True -> the client translates every pose into Nero joint space before sending it.
    # The client's `--is-follower-nero` flag turns this on.
    is_follower_nero: bool = False

    # Identity until tuned on the real arms.
    joints: dict[str, JointMap] = field(
        default_factory=lambda: {
            "shoulder_pan": JointMap(sign=1.0, offset=0.0),
            "shoulder_lift": JointMap(sign=1.0, offset=0.0),
            "elbow_flex": JointMap(sign=1.0, offset=0.0),
            "wrist_flex": JointMap(sign=1.0, offset=0.0),
            "wrist_roll": JointMap(sign=1.0, offset=0.0),
        }
    )

    # The SO gripper is 0..100 like the Koch one; these are the Koch values as a starting point.
    gripper_scale: float = 2.0
    gripper_max: float = 95.0

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        """Map an SO leader action (`<joint>.pos` keys) into Nero joint space. No-op if not `is_follower_nero`."""
        if not self.is_follower_nero:
            return dict(action)

        out: dict[str, float] = {}
        for key, val in action.items():
            joint = key.removesuffix(".pos")
            if joint == "gripper":
                out[key] = min(val * self.gripper_scale, self.gripper_max)
            elif joint in self.joints:
                m = self.joints[joint]
                out[key] = m.sign * val + m.offset
            else:
                out[key] = val
        return out
