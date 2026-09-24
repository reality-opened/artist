# Next-agent handoff: SO101 bring-up

Updated 2026-09-14, 21:39 UTC. Workspace:
`/Users/zhangbocheng/code/projects/research/arm-harness`.

## Start here

The arm has moved under Codex control, with visible wrist rotation in a concurrent
camera recording. Both camera streams were live at handoff. **Precise positioning
and a complete out-and-return test have not succeeded.** All five motion attempts
ended with a timeout; do not describe them as successful round trips.

The last test turned wrist ID 5 from 1774 to 1674 encoder ticks, about 8.8 degrees.
It stopped 14 ticks short of the 1660 target and skipped the return. Torque was
released and independently read back as off. The latest read-only snapshot of
all six motors and both camera statuses is [handoff-state-20260914.json](handoff-state-20260914.json).
This is a historical snapshot, not live state.

The user's immediate goal, physical movement visible through a camera stream, is
complete. The README's longer-term ambitions are object pickup and painting;
neither capability is implemented. Continue with reliable positioning before those.

## Fast resume: read first, then inspect

Run from the workspace. The existing `.venv` uses Python 3.12; rebuilding it is
normally unnecessary. If needed, `uv sync --locked` recreates the pinned environment.
`pyproject.toml` / `uv.lock` pin pyserial 3.5, feetech-servo-sdk 1.0.0,
numpy 2.5.3, opencv-python 5.0.0.93, and pyorbbecsdk2 2.1.2.
The Servo SDK import is `scservo_sdk`; the Orbbec import is `pyorbbecsdk`.
FFmpeg is installed at `/opt/homebrew/bin/ffmpeg`.

Use absolute interpreter/script paths for hardware commands; installed Codex rules
match those prefixes. Run commands individually to avoid shell wrappers, wildcards,
or redirection preventing a rule match.

```sh
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/arm.py devices
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/arm.py motors --port /dev/cu.usbmodem5AE60846541 --ids 1,2,3,4,5,6
curl --max-time 5 --fail --silent --show-error http://127.0.0.1:8766/status
curl --max-time 5 --fail --silent --show-error http://127.0.0.1:8765/status
curl --max-time 5 --fail --silent --show-error http://127.0.0.1:8766/frame.jpg -o /tmp/arm-gemini-current.jpg
curl --max-time 5 --fail --silent --show-error http://127.0.0.1:8765/frame.jpg -o /tmp/arm-wrist-current.jpg
```

Inspect the downloaded images with Codex's `view_image` tool. A successful status
request alone does not show clearance. `/status` must say `live: true` with increasing
sequence numbers; `/frame.jpg` rejects frames older than two seconds. The user
accepts a room background because of space constraints. Inspect actual hands,
gripper and nearby objects before motion. The most recent operator clearance was
visible after both hands were moved away from the desk. No new motion was run
while preparing this handoff.

## Hardware and state left behind

The user confirmed **six motors**, including the gripper. Port last observed:
`/dev/cu.usbmodem5AE60846541`; USB VID 6790, PID 21971, serial `5AE6084654`.
Bus: 1,000,000 baud, `scservo_sdk.PacketHandler(0)`. All six report model 777 (STS3215).
Joint names below come from EXP-23's calibration; `arm.py motors` deliberately
reports IDs without assuming a mapping for arbitrary devices.

| ID | Joint | Last position, raw ticks | SRAM torque limit |
| --- | --- | ---: | ---: |
| 1 | shoulder_pan | 1786 | 1000 |
| 2 | shoulder_lift | 2060 | 1000 |
| 3 | elbow_flex | 2063 | 1000 |
| 4 | wrist_flex | 3644 | 1000 |
| 5 | wrist_roll | 1674 | 500 |
| 6 | gripper | 1676 | 500 |

At handoff, all six had torque off, mode 0, no reported faults, zero speed/load/current,
25–29 C temperature and 5.3–5.4 V. Only ID 5 was commanded to move in this harness.
ID 5 retains goal 1660, speed setting 60, acceleration 254 and SRAM torque limit 500.
Its EEPROM maximum torque limit is 1000. Never enable torque against a stale goal:
`nudge.py` stages the freshly read position before enabling. It does not restore
the original tuning after a test. These values can change after power cycling or
manual repositioning; read them again.

The supply question is settled: EXP-23 `BUILD-LOG.md` identifies a **5 V 6 A PSU**
and Waveshare Bus Servo Adapter (A) v1.1, jumper B for USB-to-servo. The user
confirmed this is the intended configuration. Do not ask for the voltage label again
or treat 5.3 V alone as a new blocker. Some earlier motion logs contain transient
4.4 V readings, but the final reverse test stayed at 5.4 V. No PSU adjustment was made.

## Camera operation and recovery

| Camera | Exact AVFoundation name | Local view |
| --- | --- | --- |
| Wrist webcam | `USB Camera` | http://127.0.0.1:8765 |
| Gemini RGB | `Orbbec Gemini 2 RGB Camera` | http://127.0.0.1:8766 |

Both use FFmpeg/AVFoundation by exact name. OpenCV numeric indices changed order
and sometimes selected FaceTime or OBS, so do not use numeric indices to verify
robot operation. Native Orbbec SDK discovery succeeded, but capture encountered
`uvc_open` error -3. **Depth, intrinsics and aligned RGB-D remain unverified.**
The HTTP UI was not verified through the in-app browser; snapshots and actual
MJPEG video were verified directly.

If no server is running, start each in a separate long-running process:

```sh
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/camera_stream.py --device-name 'Orbbec Gemini 2 RGB Camera' --port 8766
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/camera_stream.py --device-name 'USB Camera' --port 8765
```

Last Codex process handles: Gemini `36027`, wrist `34990`. These are session
handles, not OS PIDs, and may not survive a new agent/session. Check HTTP and
process state rather than assuming they are valid or starting duplicate servers.

USB disconnects happened repeatedly. Sometimes only the serial controller vanished;
once the entire USB hub and both cameras disappeared. Inspect current USB state:

```sh
/usr/sbin/ioreg -p IOUSB -w 0
ps -axo pid,ppid,command
```

`system_profiler` sometimes returned an empty USB inventory despite devices being
present in `ioreg`. `arm.py devices` still correctly listed serial ports. If the
controller is absent from both inventories, reconnect hardware; changing Python
packages cannot fix the missing device.

`camera_stream.py` currently blocks in FFmpeg stdout reads when hardware disappears.
It correctly reports stale frames but does not automatically reconnect. Ctrl-C
stopped the Python servers in this session, but two FFmpeg children were orphaned.
We identified them with `ps`, terminated those exact PIDs with `kill -TERM`, then
restarted the servers. **Do not reuse historical PIDs or kill unrelated FFmpeg jobs.**
If a process is merely taking time to respond, inspect/poll it before restarting.
Adding timeout/reconnection and reliable child cleanup is a useful follow-up.

## Physical operations actually executed

All tests used ID 5, raw relative tick goals and speed register 60. Paths below
are relative to the workspace; each directory contains `motion.jsonl`.
Read the saved plan for the settings actually used: the script evolved between runs.

| Capture directory | Delta | Torque limit | Encoder start → end | Outcome |
| --- | ---: | ---: | --- | --- |
| `captures/20260914T033738-0820a1dc` | +24 | 150 | 1745 → 1745 | No movement; acceleration 5 |
| `captures/20260914T162711-8b23ada5` | +24 | 300 | 1748 → 1748 | No movement; acceleration 254 and lock sequence |
| `captures/20260914T170235-0520bc57` | +114 | 300 | 1748 → 1786 | About 3.3°; load reached configured cap |
| `captures/20260914T170959-2ac66a72` | +114 | 500 | 1779 → 1791 | About 1.1°; controller disappeared after release |
| `captures/20260914T213620-ddbf40a9` | −114 | 500 | 1774 → 1674 | About 8.8°; rotation visible in recording |

Every run timed out before the outbound target; **no return phase completed**.
Torque release is recorded in the last four logs. The first script version did
not log release, but a subsequent read in the session confirmed torque off.
The last test also had an independent post-test read confirming position 1674,
torque off and status 0. Degrees here are approximate encoder displacement
(`ticks * 360 / 4096`), not calibrated absolute joint angles.

Recordings:

- `captures/first-motion/gemini.avi`: first attempt, 10 seconds; not proof of motion.
- `captures/wrist-test-500.avi`: positive 500-limit attempt, 12 seconds, 300 frames.
- `captures/wrist-reverse-500.avi`: final reverse attempt, 12 seconds, 300 frames at
  25 fps. Frames at 0, 3 and 6 seconds were inspected and show wrist rotation.

For immediate visual review with `view_image`, extracted frames are saved as
[0 seconds](../captures/wrist-reverse-500-frames/00s.jpg),
[3 seconds](../captures/wrist-reverse-500-frames/03s.jpg), and
[6 seconds](../captures/wrist-reverse-500-frames/06s.jpg).

Exact final motion command, preserved for reference. **Executing it again uses
the new current position and moves another −114 ticks; it does not replay the old
absolute target or return the wrist to 1774.** First run without `--execute` for
read-only preflight, inspect live clearance, and decide whether the next motion
is useful for the current task.

```sh
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/nudge.py --port /dev/cu.usbmodem5AE60846541 --id 5 --delta -114 --torque-limit 500 --execute
```

Concurrent recording command used, launched just before the motion command:

```sh
ffmpeg -hide_banner -loglevel error -n -i http://127.0.0.1:8766/stream.mjpg -t 12 -an -c:v copy captures/wrist-reverse-500.avi
```

Choose a new output filename for another recording; `-n` preserves existing files.
Poll both process handles to completion. A motor timeout exits 1, even when some
physical motion occurred. Inspect the JSONL and read torque independently afterward.

## Current control semantics and changes made

`arm.py` performs discovery, RGB snapshots, SDK enumeration and read-only motor
telemetry. `camera_stream.py` provides named RGB streaming, freshness checks and
HTTP snapshot/MJPEG endpoints. `nudge.py` is a bounded smoke-test driver, not a
general six-joint controller.

`nudge.py` requires torque initially off, STS3215/model 777, position mode, valid
single-turn register limits with an 8-tick margin, temperature below 50 C and
voltage within the motor's stored limits. Wrist displacement is 8–114 ticks;
gripper displacement is 8–24. Requested torque is 1–500, capped by the EEPROM
maximum. A bug that silently capped new requests by a previous test's lowered
SRAM torque limit was fixed; regression coverage was added.

Write sequence: SRAM torque limit (48), acceleration 254 (41), goal time 0 (44),
speed 60 (46), present position as goal (42), torque on (40), lock 1 (55), then
outbound goal (42). Writes are read back. No EEPROM calibration or gain writes
were performed by this harness. The lock write follows EXP-23's enable sequence.

Each phase has a four-second deadline and needs three observations within 4 ticks
of its goal. Polls log position, voltage, load, current, torque, goal readback,
speed, temperature and status. Faults or excess travel abort. Return runs only
after outbound success. Finally, torque release is attempted up to three times
and verified by readback. Connection loss or forced termination can prevent release.
The current script does not implement visual feedback or collision detection;
camera inspection is performed by the agent outside the control loop.

## EXP-23: reuse the source, do not rediscover it

Previous working experiment, supplied by the user:
`/Users/zhangbocheng/code/projects/startups/platform/experiments/exp23_arm_refinement_loop`.

- `BUILD-LOG.md`: hardware, PSU and assembly decisions.
- `BOOTH-ARM-READY.md`: prior August hardware validation and motion descriptions.
- `scripts/smoke_move.py`, `scripts/executor.py`: previous LeRobot control paths.
- `.venv/lib/python3.11/site-packages/lerobot/robots/so_follower/so_follower.py`:
  position mode, P=16 / I=0 / D=32; gripper max torque 500,
  protection current 250, overload torque 25.
- `.venv/lib/python3.11/site-packages/lerobot/motors/feetech/feetech.py`:
  return delay 0, acceleration/maximum acceleration 254, torque enable then lock 1.
- Old environments `.venv` (LeRobot 0.4.4) and `.venv-lerobot06` (LeRobot 0.6)
  were found. Current harness does not require importing either.

Calibration file:
`/Users/zhangbocheng/.cache/huggingface/lerobot/calibration/robots/so_follower/exp23_follower.json`.
It contains the six-joint mapping, offsets and ranges. It is historical; consistency
with current EEPROM and mechanics has not been validated here. The gripper's
observed 1676 ticks is near/below that file's range minimum 1679. Current nudge
tests use live raw register limits, not this calibration. Do not blindly apply it.
Old scripts have stale-calibration handling; don't bypass it just to get motion.
Calling old `robot.connect()` can configure hardware and enable multiple motors;
do not use it as a read-only probe or launch a full booth routine for a small jog.
No EXP-23 source files were edited.

## Permissions, validation and next useful work

The user authorized the uv environment, camera operation and bounded physical
motion, and asked to minimize repeated permission prompts. Local rules were copied
to `~/.codex/rules/arm-harness.rules`; the reviewable project copy is
[arm-harness.rules](arm-harness.rules). At handoff the runtime uses workspace-write
with automatic approval review. Use current tool policy as authoritative; don't
assume a future session has identical permissions. No global config was modified
by this documentation task. Physical clearance is independent of command approval.

Last code validation: **12 tests passed** after the torque-cap fix:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The camera HTTP test requires local socket access. Tests cover telemetry rejection,
freshness, HTTP delivery, preflight, stale-goal staging, torque limits and release
on faults; they are not proof of mechanical performance. PyObjC AVFoundation
packages were installed experimentally in `.venv` during discovery but are not
declared runtime dependencies; production camera code uses FFmpeg.

Next motion work should explain the directional asymmetry: positive commands
stopped far short, while the negative command traveled 100 of 114 ticks with
much lower reported load. Cable tension, contact, static friction and control
tracking are hypotheses, not diagnosed causes. Inspect the mechanics and compare
telemetry/source before increasing force. Do not loosen completion criteria merely
to turn these failed round trips into passing tests. Add camera reconnect/cleanup
if unattended stream recovery is needed. Full-arm calibrated control, camera-to-arm
transforms, object localization, pickup and painting remain unimplemented.
