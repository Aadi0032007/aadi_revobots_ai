# Nero Arm — Complete Reference Guide

---

## Table of Contents

1. [Important Files](#1-important-files)
2. [Connect, Read, Write](#2-connect-read-write)
3. [All Features and Functions — Detailed Reference](#3-all-features-and-functions--detailed-reference)
   - [Motion Modes](#31-motion-modes)
   - [Safety and Control](#32-safety-and-control)
   - [Leader-Follower Dual-Arm Teaching](#33-leader-follower-dual-arm-teaching)
   - [TCP / Tool Offset](#34-tcp--tool-offset)
   - [MIT Impedance Control](#35-mit-impedance-control)
   - [CPV Parameter Tuning](#36-cpv-parameter-tuning)
   - [End-Effector Attachment](#37-end-effector-attachment)
4. [State Reading Reference](#4-state-reading-reference)
5. [Firmware Versions](#5-firmware-versions)
6. [Joint Limits](#6-joint-limits)
7. [Radians and Degrees Conversion](#7-radians-and-degrees-conversion)
8. [Finding CAN Ports](#8-finding-can-ports)

---

## 1. Important Files

| File | Purpose |
|------|---------|
| `pyAgxArm/protocols/can_protocol/drivers/nero/default/driver.py` | Main Nero driver — firmware ≤ 1.10 |
| `pyAgxArm/protocols/can_protocol/drivers/nero/versions/v111/driver.py` | Nero driver — firmware ≥ 1.11 |
| `pyAgxArm/protocols/can_protocol/drivers/core/arm_driver_interface.py` | Abstract interface — all method signatures defined here |
| `pyAgxArm/protocols/can_protocol/drivers/core/arm_driver_abstract.py` | Shared logic inherited by all drivers |
| `pyAgxArm/api/arm_options.py` | Constants: `ArmModel.NERO`, `NeroFW.DEFAULT`, `NeroFW.V111` |
| `pyAgxArm/api/constants.py` | Joint limits and MDH kinematic parameters for each model |
| `pyAgxArm/demos/nero/test1.py` | Complete working demo covering all modes |
| `tests/test_nero_driver_virtual_can.py` | Unit tests — good reference for usage patterns |

---

## 2. Connect, Read, Write

### Step 1 — Create Config and Connect

```python
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,   # or NeroFW.V111 for firmware >= 1.11
    interface="socketcan",              # Linux: "socketcan" | Windows: "agx_cando" | macOS: "slcan"
    channel="can0",                     # Windows uses "0", "1", etc.
)

robot = AgxArmFactory.create_arm(cfg)
robot.connect()     # Opens CAN socket and starts background reader thread
```

### Step 2 — Enable Motors (required before any motion)

```python
import time

while not robot.enable(255):    # 255 = all 7 joints at once
    time.sleep(0.01)
```

### Step 3 — Read State

```python
# Joint angles (7 values, radians)
ja = robot.get_joint_angles()
if ja:
    print(ja.msg)       # [j1, j2, j3, j4, j5, j6, j7]
    print(ja.hz)        # update rate in Hz
    print(ja.timestamp) # Unix timestamp of last update

# End-effector pose [x, y, z, roll, pitch, yaw]  (meters and radians)
fp = robot.get_flange_pose()
if fp:
    print(fp.msg)

# Arm status
st = robot.get_arm_status()
if st:
    print(st.msg.motion_status)   # 0 = at target, 1 = moving
    print(st.msg.arm_status)      # error/state code
    print(st.msg.ctrl_mode)       # active control mode

# Per-joint diagnostics (joint index 1–7)
ds = robot.get_driver_states(1)
if ds:
    print(ds.msg.vol)             # voltage (V)
    print(ds.msg.motor_temp)      # motor temperature (°C)
    print(ds.msg.bus_current)     # current (A)

ms = robot.get_motor_states(1)
if ms:
    print(ms.msg.position)        # rad
    print(ms.msg.velocity)        # rad/s
    print(ms.msg.torque)          # N·m
```

### Step 4 — Write / Command Motion

```python
# Joint space motion (smooth trajectory)
robot.move_j([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Cartesian space (smooth trajectory)
robot.move_p([-0.45, 0.0, 0.45, -1.5708, 0.0, -3.14159])

# Wait until motion completes
while True:
    st = robot.get_arm_status()
    if st and st.msg.motion_status == 0:
        break
    time.sleep(0.05)

# Speed control (0–100%)
robot.set_speed_percent(50)
```

### Step 5 — Disconnect

```python
robot.disable(255)
robot.disconnect()
```

---

## 3. All Features and Functions — Detailed Reference

---

### 3.1 Motion Modes

The Nero arm supports 7 distinct motion modes (6 on firmware V111). Each mode uses a different control strategy and is suited to different tasks.

---

#### `move_j(joints: list[float])` — Joint Space Motion

**What it does:** Commands the arm to move all 7 joints simultaneously to the specified angles. The controller generates a smooth velocity-profiled trajectory (trapezoidal or S-curve) so the arm accelerates, travels, and decelerates smoothly.

**When to use:** General-purpose positioning. The most common command for moving the arm from one configuration to another. Joint paths are predictable and stay within joint limits.

**What to know:**
- Takes a list of 7 float values, one per joint, in **radians**
- The arm moves all joints simultaneously and arrives together
- Respects the speed percentage set by `set_speed_percent()`
- Does not guarantee a straight Cartesian path — the end-effector follows an arc through space

```python
robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # Home position
robot.move_j([0.5, -0.3, 0.2, 1.0, 0.0, 0.5, 0.0])   # Arbitrary configuration
```

---

#### `move_p(pose: list[float])` — Cartesian Point-to-Point Motion

**What it does:** Commands the end-effector to move to a target pose `[x, y, z, roll, pitch, yaw]` in the base frame. Uses smooth trajectory planning in Cartesian space. The controller internally runs inverse kinematics to determine the joint path.

**When to use:** When you need the end-effector to arrive at a specific position and orientation in 3D space, and you don't care about the path it takes to get there (only the destination matters).

**What to know:**
- Units: `x, y, z` in **meters**; `roll, pitch, yaw` in **radians** (Euler XYZ extrinsic)
- Trajectory is smooth but the Cartesian path is not guaranteed to be a straight line
- Can fail if the target pose is outside the workspace or causes a singular configuration

```python
pose = [-0.45, 0.0, 0.45, -1.5708, 0.0, -3.14159]
robot.move_p(pose)
```

---

#### `move_l(pose: list[float])` — Linear Cartesian Motion

**What it does:** Moves the end-effector in a **straight line** from the current pose to the target pose. Orientation also interpolates linearly (SLERP for rotation). The controller segments the straight path into many small joint-space steps.

**When to use:** Assembly tasks, welding, painting, or any task where the tool must travel a straight line — for example, inserting a pin or running a bead. Also useful when obstacles require the tool to avoid curved paths.

**What to know:**
- Guarantees a straight Cartesian path (unlike `move_p`)
- More computationally demanding and more likely to fail mid-path if the arm passes near a singularity
- Units same as `move_p`

```python
start = [-0.45, -0.2, 0.45, -1.5708, 0.0, -3.14159]
end   = [-0.45,  0.2, 0.45, -1.5708, 0.0, -3.14159]
robot.move_l(end)   # Draws a straight line from current pose to end
```

---

#### `move_c(start_pose, mid_pose, end_pose)` — Circular Arc Motion

**What it does:** Moves the end-effector along a **circular arc** defined by three Cartesian poses: start, a midpoint on the arc, and the end. The three poses uniquely define a circle in space; the arm travels along the arc from start to end passing through mid.

**When to use:** Circular machining or grinding, tracing a circular weld seam, polishing curved surfaces, or any application requiring a repeatable arc path.

**What to know:**
- All three arguments are `[x, y, z, roll, pitch, yaw]` poses in the base frame
- The arm must already be at `start_pose` before the command (or close to it)
- The arc radius is implicitly defined by the geometry of the three points

```python
start = [-0.45, -0.2, 0.45, -1.5708, 0.0, -3.14159]
mid   = [-0.45,  0.0, 0.5,  -1.5708, 0.0, -3.14159]
end   = [-0.45,  0.2, 0.45, -1.5708, 0.0, -3.14159]
robot.move_c(start, mid, end)
```

---

#### `move_js(joints: list[float])` — Fast Joint Motion (No Smoothing)

**What it does:** Sends joint targets directly to the low-level controller with **no trajectory planning or velocity smoothing**. The arm responds as fast as the hardware allows.

**When to use:** High-frequency control loops (e.g., teleoperation, replaying a pre-recorded trajectory at the controller level) where you are managing the trajectory yourself and want minimal latency between command and execution.

**What NOT to use it for:** Normal positioning. Without smoothing, sudden large commands cause mechanical shock, can trigger motor faults, and are dangerous to nearby objects and people.

**What to know:**
- You are responsible for generating smooth, safe position sequences
- Command it at a high rate (e.g., 100 Hz) with small increments between each command
- Respects joint limits if `set_joint_limits_enabled(True)` is set

```python
# Safe use: sending pre-computed smooth trajectory at high frequency
for angles in pre_computed_trajectory:
    robot.move_js(angles)
    time.sleep(0.01)   # 100 Hz
```

---

#### `move_mit(joint_index, p_des, v_des, kp, kd, t_ff)` — MIT Impedance Control

**What it does:** Implements **per-joint impedance control** using the MIT Mini-Cheetah torque control law. The torque sent to the joint motor is:

```
T = kp * (p_des - p_actual) + kd * (v_des - v_actual) + t_ff
```

By tuning `kp`, `kd`, and `t_ff`, you can implement position control, velocity control, torque control, or any blend of the three — all within a single command.

**When to use:**
- **Compliant contact tasks** — Set low `kp`/`kd` so the joint yields to external forces (e.g., soft pushing, surface following)
- **Drag teaching** — Set `kp=0`, `kd=small`, `t_ff=0` for gravity-compensated free motion
- **Torque control** — Set `kp=0`, `kd=0`, command `t_ff` directly
- **Velocity control** — Set `kp=0`, `kd=moderate`, command `v_des`
- **Stiff position control** — Set high `kp` and `kd`, command `p_des`

**Parameter ranges:**

| Parameter | Range |
|-----------|-------|
| `p_des` (position) | −12.5 to +12.5 rad |
| `v_des` (velocity) | −45.0 to +45.0 rad/s |
| `kp` | 0.0 to 500.0 |
| `kd` | −5.0 to +5.0 |
| `t_ff` (joints 1–2) | ±24 N·m |
| `t_ff` (joints 3–4) | ±18 N·m (DEFAULT fw), ±16 N·m (V111) |
| `t_ff` (joints 5–7) | ±8 N·m |

```python
# Stiff position control on joint 1
robot.move_mit(1, p_des=0.5, v_des=0.0, kp=100.0, kd=2.0, t_ff=0.0)

# Velocity control on joint 2
robot.move_mit(2, p_des=0.0, v_des=1.0, kp=0.0, kd=0.5, t_ff=0.0)

# Pure torque on joint 3
robot.move_mit(3, p_des=0.0, v_des=0.0, kp=0.0, kd=0.0, t_ff=2.0)

# Compliant position (yields to external force)
robot.move_mit(1, p_des=0.5, v_des=0.0, kp=10.0, kd=0.3, t_ff=0.0)
```

---

#### `move_cpv_pos(joint_index, pos)` and `move_cpv_vel(joint_index, vel)` — CPV Mode (DEFAULT firmware only)

**What it does:** CPV (Custom Profile Velocity) is a per-joint command mode with its own configurable motion profile parameters (acceleration, deceleration, contour velocity, PID gains). `move_cpv_pos` sets a **position reference** for a joint; `move_cpv_vel` sets a **velocity reference**.

**When to use:** When you need fine-grained per-joint motion tuning that the standard joint-space commands do not expose — for example, running joint 1 with a very slow acceleration while joint 3 uses aggressive gains.

**What to know:**
- Only available on **DEFAULT firmware (≤ 1.10)**; not present in V111
- Must call `robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.CPV)` first
- Each joint's profile (accel, decel, velocity, gains) is independently configurable via `set_cpv_*` methods

```python
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.CPV)

robot.set_cpv_acc(1, 2.0)       # Joint 1 acceleration limit
robot.set_cpv_dcc(1, 2.0)       # Joint 1 deceleration limit
robot.set_cpv_cv(1, 1.5)        # Joint 1 max contour velocity

robot.move_cpv_pos(1, 0.5)      # Move joint 1 to 0.5 rad
robot.move_cpv_vel(1, 1.0)      # Drive joint 1 at 1.0 rad/s
```

---

### 3.2 Safety and Control

---

#### `electronic_emergency_stop()` — Damped Emergency Stop

**What it does:** Sends an emergency stop command to the arm controller. Unlike a hard cut of power, this is a **damped stop** — the controller applies high damping (braking) to decelerate the arm as quickly as possible while avoiding joint shock. Motors remain energized but bring the arm to rest.

**When to use:** Any time you detect an unsafe condition in your application — collision detection, workspace violation, unexpected force reading, or operator intervention. Should be the first line of response to any safety event.

**What to know:**
- Does not cut motor power; joints remain stiff and in place after stopping
- The arm stays in an error state after E-stop; call `reset()` to resume normal operation
- Faster and safer than calling `disable()` during motion, which would cause the arm to drop

```python
robot.electronic_emergency_stop()
time.sleep(0.5)   # Wait for arm to settle
robot.reset()     # Clear error state before next command
```

---

#### `reset()` — Reset Motion Controller

**What it does:** Clears any error or fault state in the motion controller and returns the arm to a ready state. Does not move the arm.

**When to use:** After an emergency stop, after a joint limit violation, or any time `get_arm_status()` reports an error code. Must be called before new motion commands will be accepted.

```python
robot.reset()
```

---

#### `disable(joint_index)` — Disable Motor(s)

**What it does:** De-energizes the specified joint motor(s). The joints become limp and will drop under gravity if not mechanically supported.

**When to use:** After completing a task before disconnecting; during drag teaching (use `set_leader_mode()` instead for zero-force mode); safe shutdown sequence.

**What to know:**
- `disable(255)` disables all 7 joints simultaneously
- `disable(1)` through `disable(7)` disables individual joints
- Returns `True` when confirmed, `False` if not yet confirmed — poll in a loop

```python
while not robot.disable(255):
    time.sleep(0.01)
```

---

#### `set_joint_limits_enabled(enabled: bool)` — Software Joint Limit Guard

**What it does:** Enables or disables the software layer that clamps motion commands to the joint limit ranges defined in `constants.py`. When enabled, any command that would exceed a joint's range is silently clipped to the boundary.

**When to use:** Always enable during normal operation to prevent collisions or motor damage from out-of-range commands. Disable only when deliberately testing limit behavior or sending raw commands for calibration.

```python
robot.set_joint_limits_enabled(True)   # Recommended for normal use
```

---

#### `set_auto_set_motion_mode_enabled(enabled: bool)` — Automatic Mode Switching

**What it does:** When enabled, the driver automatically switches the arm's internal motion mode to match each command you send. For example, calling `move_j()` will first switch to joint mode, then send the joint command. When disabled, you must manually call `set_motion_mode()` before each type of command.

**When to use:** Enable for convenience in applications that mix different motion commands. Disable when you want explicit control over mode transitions — useful if mode switching has a latency cost in time-critical loops.

```python
robot.set_auto_set_motion_mode_enabled(True)   # Convenient for mixed-mode use

# Manual mode management (when disabled):
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
robot.move_j([...])
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
robot.move_p([...])
```

---

#### `set_speed_percent(percent: int)` — Global Speed Scaling

**What it does:** Scales the velocity of all motion commands by the given percentage. At 100%, the arm moves at full programmed speed. At 50%, all motions run at half speed. This applies to `move_j`, `move_p`, `move_l`, `move_c`.

**When to use:** Reduce speed during testing, commissioning, or when operating near obstacles or people. Set to 100% for production throughput.

```python
robot.set_speed_percent(20)   # Slow for testing
robot.set_speed_percent(100)  # Full speed for production
```

---

### 3.3 Leader-Follower Dual-Arm Teaching

This feature supports a **two-arm teleoperation or teaching setup** where one physical arm is held by a human (leader) and a second arm (follower) mirrors its motion in real time.

---

#### `set_leader_mode()` — Zero-Force Drag Teaching Mode

**What it does:** Switches the arm into a **gravity-compensated zero-force mode**. Motor torques are reduced to exactly counteract gravity, so the arm feels weightless when you push it. A human operator can physically grab any link and move the arm freely through space to record a path.

**When to use:** Programming by demonstration — move the arm through a desired task path by hand, recording the joint angles at each waypoint for later playback.

**What to know:**
- The arm is still powered; motors actively compensate gravity
- Do not apply heavy external loads — the torque compensation is calibrated for the arm's own weight
- Joint encoder data is continuously updated and can be read with `get_joint_angles()`

```python
robot.set_leader_mode()
# Human operator physically moves the arm
ja = robot.get_joint_angles()   # Read current dragged position
```

---

#### `set_follower_mode()` — Synchronized Follower Mode

**What it does:** Sets the arm to continuously track and replicate the joint angles of a paired leader arm. The follower reads the leader's joint angles from a shared CAN bus and commands its own joints to match, creating real-time mirroring.

**When to use:** Teleoperation setups where a human moves one arm (leader) and a remote or separate arm (follower) replicates the motion — for example, hazardous environment manipulation or collaborative assembly.

```python
robot.set_follower_mode()
# Follower now automatically tracks the leader arm
leader_ja = robot.get_leader_joint_angles()   # Read what the leader is doing
```

---

#### `set_normal_mode()` — Single-Arm Control

**What it does:** Returns the arm to standard single-arm operation, cancelling leader or follower mode.

```python
robot.set_normal_mode()
```

---

#### `get_leader_joint_angles()` — Read Leader Arm State

**What it does:** Returns the current joint angles of the paired leader arm as reported on the shared CAN bus. Useful for logging leader positions, implementing custom follower logic, or monitoring bilateral teleoperation.

```python
leader_ja = robot.get_leader_joint_angles()
if leader_ja:
    print(leader_ja.msg)   # [j1..j7] in radians
```

---

### 3.4 TCP / Tool Offset

TCP (Tool Center Point) allows you to define a rigid offset from the arm's flange (wrist output) to the actual working point of a tool (e.g., tip of a gripper finger, welding torch tip). All Cartesian poses can then be expressed relative to the tool tip rather than the flange.

---

#### `set_tcp_offset(pose: list[float])` — Define Tool Geometry

**What it does:** Registers the transform from the flange frame to the tool tip frame as `[x, y, z, roll, pitch, yaw]`. Once set, `get_tcp_pose()` returns poses at the tool tip, and Cartesian commands target the tool tip.

**When to use:** Any time a tool is mounted on the flange. Avoids having to manually add tool offsets to every pose command in your application.

```python
# Gripper with 10 cm finger length along flange Z-axis
robot.set_tcp_offset([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])

# Tool with both offset and rotation
robot.set_tcp_offset([0.05, 0.0, 0.12, 0.0, 0.1745, 0.0])   # 10° tilt
```

---

#### `get_tcp_pose()` — End-Effector Pose at Tool Tip

**What it does:** Returns the current pose of the **tool tip** in the base frame, after applying the TCP offset transform to the raw flange pose.

```python
tcp = robot.get_tcp_pose()
if tcp:
    x, y, z, roll, pitch, yaw = tcp.msg
```

---

#### `get_flange2tcp_pose(flange_pose)` and `get_tcp2flange_pose(tcp_pose)` — Manual Transforms

**What they do:** Utility functions to manually convert between flange-frame and TCP-frame poses using the registered TCP offset. Useful when building trajectory planners that need to work in TCP space but command in flange space.

```python
flange = robot.get_flange_pose().msg
tcp    = robot.get_flange2tcp_pose(flange)   # Convert to tool-tip frame
```

---

### 3.5 MIT Impedance Control

See the full description under **Section 3.1 — `move_mit`** above for the control law and parameter ranges.

**Torque limits by joint:**

| Joints | DEFAULT firmware | V111 firmware |
|--------|-----------------|---------------|
| 1–2    | ±24 N·m         | ±24 N·m       |
| 3–4    | ±18 N·m         | ±16 N·m       |
| 5–7    | ±8 N·m          | ±8 N·m        |

**Common patterns:**

```python
# Gravity compensation (compliant mode)
for i in range(1, 8):
    robot.move_mit(i, p_des=0, v_des=0, kp=0, kd=0.5, t_ff=gravity_torque[i])

# Stiff position hold
robot.move_mit(1, p_des=0.5, v_des=0.0, kp=200.0, kd=3.0, t_ff=0.0)

# Soft contact (compliant position)
robot.move_mit(1, p_des=target, v_des=0.0, kp=15.0, kd=1.0, t_ff=0.0)
```

---

### 3.6 CPV Parameter Tuning (DEFAULT firmware only)

CPV (Custom Profile Velocity) exposes the underlying motion profile and PID parameters for each joint individually. Useful for fine-tuning motion quality — smoothness, settling time, and stiffness — on a per-joint basis.

**Read parameters:**
```python
acc = robot.get_cpv_acc(1)   # Acceleration (rad/s²)
dcc = robot.get_cpv_dcc(1)   # Deceleration (rad/s²)
cv  = robot.get_cpv_cv(1)    # Contour (max) velocity
pp  = robot.get_cpv_pp(1)    # Position-loop proportional gain
kp  = robot.get_cpv_kp(1)    # Velocity-loop Kp
ki  = robot.get_cpv_ki(1)    # Velocity-loop Ki
```

**Write parameters (with read-back verification, returns `bool`):**
```python
ok = robot.set_cpv_acc(1, 2.0)    # Acceleration
ok = robot.set_cpv_dcc(1, 2.0)    # Deceleration
ok = robot.set_cpv_cv(1, 1.5)     # Contour velocity
ok = robot.set_cpv_pp(1, 100.0)   # Position-loop Kp
ok = robot.set_cpv_kp(1, 50.0)    # Velocity-loop Kp
ok = robot.set_cpv_ki(1, 10.0)    # Velocity-loop Ki
```

**What each parameter does:**

| Parameter | Effect |
|-----------|--------|
| `acc` | How fast the joint ramps up from zero velocity. Higher = snappier start |
| `dcc` | How fast the joint ramps down to zero. Higher = sharper stop |
| `cv`  | Maximum velocity the joint will travel at during the motion |
| `pp`  | Position-loop Kp: how aggressively the loop corrects position error |
| `kp`  | Velocity-loop Kp: how aggressively the loop corrects velocity error |
| `ki`  | Velocity-loop Ki: eliminates steady-state velocity error over time |

---

### 3.7 End-Effector Attachment

**What it does:** `init_effector()` initializes a gripper or dexterous hand driver on the same CAN bus as the arm. Returns a separate driver object for controlling the effector.

**What to know:**
- Can only be called **once per session** — subsequent calls raise an error
- The effector shares the arm's CAN channel
- Returns an `EffectorDriver` with its own `connect()`, `open()`, `close()`, `get_state()` methods

```python
# AgxGripper (parallel jaw)
gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
gripper.connect()
gripper.open()
gripper.close()

# Revo2 (dexterous hand)
hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
hand.connect()
```

---

## 4. State Reading Reference

All getter methods return a `MessageAbstract` wrapper (or `None` if data is not yet available).

```python
result = robot.get_joint_angles()
result.msg        # Actual data payload
result.hz         # Update rate in Hz
result.timestamp  # Unix timestamp of last CAN frame
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_joint_angles()` | `list[float]` (7 values, rad) | Current joint positions |
| `get_flange_pose()` | `list[float]` [x,y,z,r,p,y] | End-effector pose at flange |
| `get_tcp_pose()` | `list[float]` [x,y,z,r,p,y] | Pose at tool tip (needs TCP offset set) |
| `get_leader_joint_angles()` | `list[float]` (7 values, rad) | Leader arm joint angles |
| `get_arm_status()` | `ArmMsgFeedbackStatus` | Control mode, error flags, motion status |
| `get_driver_states(joint)` | `ArmMsgFeedbackLowSpd` | Voltage, temperatures, current, collision flags |
| `get_motor_states(joint)` | `ArmMsgFeedbackHighSpd` | Position, velocity, current, torque |
| `get_joint_enable_status(joint)` | `bool` | Whether the joint motor is enabled |
| `get_joints_enable_status_list()` | `list[bool]` (7 values) | Enable status of all joints |
| `get_firmware()` | `dict` | Firmware version information |
| `get_fps()` | `float` | CAN data refresh rate (Hz) |
| `fk(joint_angles)` | `list[float]` [x,y,z,r,p,y] | Forward kinematics result |

**Polling for motion completion:**

```python
def wait_motion_done(robot, timeout=10.0):
    start = time.time()
    while True:
        st = robot.get_arm_status()
        if st and st.msg.motion_status == 0:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(0.05)
```

---

## 5. Firmware Versions

| Feature | `NeroFW.DEFAULT` (≤ 1.10) | `NeroFW.V111` (≥ 1.11) |
|---------|--------------------------|------------------------|
| Motion modes | 7 (p, j, l, c, mit, js, **cpv**) | 6 (p, j, l, c, mit, js) |
| CPV mode | Yes | No |
| MIT torque — joints 3–4 | ±18 N·m | ±16 N·m |
| MIT torque encoding | 8-bit | 12-bit |
| Driver class | `NeroDriverDefault` | `NeroDriverV111` |

Select firmware when creating config:

```python
cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.V111,   # or NeroFW.DEFAULT
    interface="socketcan",
    channel="can0",
)
```

---

## 6. Joint Limits (radians)

| Joint | Minimum | Maximum | Range (degrees) |
|-------|---------|---------|----------------|
| 1 | −2.705 | +2.705 | ±155° |
| 2 | −1.745 | +1.745 | ±100° |
| 3 | −2.758 | +2.758 | ±158° |
| 4 | −1.012 | +2.147 | −58° to +123° |
| 5 | −2.758 | +2.758 | ±158° |
| 6 | −0.733 | +0.960 | −42° to +55° |
| 7 | −1.571 | +1.571 | ±90° |

Joint 4 is asymmetric — take care when planning configurations that approach its negative limit.

---

---

## 7. Radians and Degrees Conversion

**Why this matters:** The Nero arm API works entirely in radians — every joint angle you read from `get_joint_angles()` and every angle you send via `move_j()` or `move_mit()` is in radians. If your application logic works in degrees, you must convert at the boundary.

The relationship is:

```
degrees = radians × (180 / π)
radians = degrees × (π / 180)
```

A full rotation (360°) equals 2π ≈ 6.2832 radians. Some common reference points:

| Degrees | Radians |
|---------|---------|
| 0°      | 0.0     |
| 45°     | 0.7854  |
| 90°     | 1.5708  |
| 180°    | 3.1416  |
| −90°    | −1.5708 |
| −180°   | −3.1416 |

### Utility Functions

Use these in your code whenever you read from or write to the arm:

```python
import math

def deg_to_rad(degrees):
    """Convert a single angle or list of angles from degrees to radians."""
    if isinstance(degrees, (list, tuple)):
        return [math.radians(d) for d in degrees]
    return math.radians(degrees)


def rad_to_deg(radians):
    """Convert a single angle or list of angles from radians to degrees."""
    if isinstance(radians, (list, tuple)):
        return [math.degrees(r) for r in radians]
    return math.degrees(radians)
```

### Usage Examples

**Reading joint angles (arm gives radians → your code wants degrees):**

```python
ja = robot.get_joint_angles()
if ja:
    angles_deg = rad_to_deg(ja.msg)
    print(angles_deg)   # [j1, j2, j3, j4, j5, j6, j7] in degrees
```

**Commanding a move (your code has degrees → arm expects radians):**

```python
target_deg = [90.0, -45.0, 0.0, 30.0, 0.0, 20.0, 0.0]
robot.move_j(deg_to_rad(target_deg))
```

**Reading a single joint:**

```python
ja = robot.get_joint_angles()
if ja:
    joint1_deg = rad_to_deg(ja.msg[0])
    print(f"Joint 1: {joint1_deg:.2f}°")
```

**MIT impedance — position target in degrees:**

```python
target_deg = 45.0
robot.move_mit(1, p_des=deg_to_rad(target_deg), v_des=0.0, kp=100.0, kd=2.0, t_ff=0.0)
```

**Checking against joint limits in degrees:**

```python
# Joint limits in degrees for reference
JOINT_LIMITS_DEG = {
    1: (-155.0,  155.0),
    2: (-100.0,  100.0),
    3: (-158.0,  158.0),
    4: ( -58.0,  123.0),
    5: (-158.0,  158.0),
    6: ( -42.0,   55.0),
    7: ( -90.0,   90.0),
}

def clamp_to_limits_deg(joint_index, angle_deg):
    lo, hi = JOINT_LIMITS_DEG[joint_index]
    return max(lo, min(hi, angle_deg))
```

---

---

## 8. Finding CAN Ports

Use the bundled shell script to list all available CAN ports on Linux/Ubuntu:

```bash
bash pyAgxArm/scripts/linux/find_all_can_port.sh
```

An identical script is also available at `pyAgxArm/scripts/ubuntu/find_all_can_port.sh`. Use whichever matches your distro. The output lists all detected CAN interfaces (e.g. `can0`, `can1`) that you can then pass as the `channel` argument to `create_agx_arm_config()`.

---

*Generated from pyAgxArm source — Nero arm (7-DOF, CAN bus). Firmware: DEFAULT (≤1.10) and V111 (≥1.11).*
