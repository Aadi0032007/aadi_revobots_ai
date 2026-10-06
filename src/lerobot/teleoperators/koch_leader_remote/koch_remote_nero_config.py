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
Koch leader -> AgileX Nero joint mapping, applied on the leader PC by `koch_leader_remote_client`.

The Nero follower takes actions in its own joint space (degrees, gripper in its native units) and applies no
leader-specific math. Each leader owns the translation into that space, so any leader can drive the Nero
without touching the follower. Edit the numbers here, not in `agx_nero_follower.py`.

For every Koch joint: `nero = sign * koch + offset`. The gripper is `min(koch * gripper_scale, gripper_max)`.
Koch joints not listed in `joints` are passed through unchanged.
"""

from dataclasses import dataclass, field


@dataclass
class JointMap:
    sign: float = 1.0
    offset: float = 0.0


@dataclass
class KochRemoteNeroConfig:
    # False (default) -> raw Koch leader values are sent (e.g. when the follower is a Koch arm).
    # True -> the client translates every pose into Nero joint space before sending it.
    # The client's `--is-follower-nero` flag turns this on.
    is_follower_nero: bool = False

    joints: dict[str, JointMap] = field(
        default_factory=lambda: {
            "shoulder_pan": JointMap(sign=1.0, offset=-5.0),
            "shoulder_lift": JointMap(sign=1.0, offset=43.7466),
            "elbow_flex": JointMap(sign=-1.0, offset=90.0),
            "wrist_flex": JointMap(sign=1.0, offset=0.0),
            "wrist_roll": JointMap(sign=1.0, offset=0.0),
        }
    )

    gripper_scale: float = 2.0
    gripper_max: float = 95.0

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        """Map a Koch leader action (`<joint>.pos` keys) into Nero joint space. No-op if not `is_follower_nero`."""
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
