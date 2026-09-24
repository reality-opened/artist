# Bring-up

For the session's executed commands, motion log, recovery procedure and latest
state snapshot, start with [the next-agent handoff](HANDOFF.md).

The Python 3.12 uv environment is installed. `pyproject.toml` and `uv.lock`
record its dependencies. Run commands from the project root:

```sh
uv sync --locked
.venv/bin/python arm.py devices
.venv/bin/python arm.py orbbec-list
.venv/bin/python arm.py webcam --index 0
.venv/bin/python arm.py motors --port /dev/cu.usbmodemREPLACE --ids 1,2,3,4,5,6
```

Choose the webcam index after identifying the connected cameras; 0 is only an
example and may select FaceTime. System inventory order may differ from OpenCV
indices. Snapshots go in unique `captures/` directories for inspection by Codex.

For a live browser view on this Mac, select the exact camera name. OpenCV numeric
indices proved inconsistent with AVFoundation enumeration and changed across
launches, so do not use numeric indices for robot-operation verification.

```sh
.venv/bin/python camera_stream.py --device-name 'USB Camera'
# In another terminal, for the Orbbec color stream:
.venv/bin/python camera_stream.py --device-name 'Orbbec Gemini 2 RGB Camera' --port 8766
```

Open `http://127.0.0.1:8765` (or port 8766). `/status` reports frame sequence,
age and capture errors; `/frame.jpg` serves a fresh frame for Codex inspection;
`/stream.mjpg` is the continuous video feed. Images older than two seconds are
marked unavailable. Use Ctrl-C to stop. Name selection requires `ffmpeg` on PATH
(installed here at `/opt/homebrew/bin/ffmpeg`). This serves camera color only,
without depth alignment or motor control. Both named streams have delivered
visually inspected physical frames. The rules proposal includes this script
using absolute paths.

The motor command only pings and reads registers. Supply the actual port and IDs.
It does not assign IDs, change torque, calibrate, or command motion. Positions are
uncalibrated encoder ticks. Unknown models, device faults and failed reads produce
errors. The user confirmed six motors, consistent with the standard SO101.

## Codex permissions

Review [arm-harness.rules](arm-harness.rules), copy it to
`~/.codex/rules/arm-harness.rules`, and restart Codex. This project copy is an
inactive proposal. No `config.toml` change is needed. The existing `on-request`
approval policy, `workspace-write` sandbox and disabled sandbox networking can
remain. Matching allow rules permit execution outside the sandbox.

The hardware rule matches absolute interpreter and script paths, for example:

```sh
/Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/arm.py devices
```

That prefix also supports `orbbec-list`, `webcam`, and `motors`. Rules match command
arguments, not file contents, so they trust this script as it changes. The proposal
does not allow arbitrary Python commands or a motion subcommand. macOS Camera
permission is a separate OS permission.

Test a rule without executing the hardware command:

```sh
codex execpolicy check --rules docs/arm-harness.rules -- /Users/zhangbocheng/code/projects/research/arm-harness/.venv/bin/python /Users/zhangbocheng/code/projects/research/arm-harness/arm.py devices
```

References: [official OpenAI rules documentation](https://developers.openai.com/codex/rules)
and [configuration reference](https://developers.openai.com/codex/config-reference).

## Hardware findings

Initially no robot hardware was visible. After connection, both cameras and
`/dev/cu.usbmodem5AE60846541` appeared. All six motor IDs 1–6 respond as STS3215
(model 777), with no reported faults and torque disabled. Read voltages were
approximately 5.3 V, temperatures 27–30 C. Named USB/Gemini RGB streams now work
simultaneously. The Gemini native SDK returned `uvc_open` error -3, so depth
access remains unverified. RGB streaming uses FFmpeg/AVFoundation.

The [standard SO101](https://huggingface.co/docs/lerobot/main/en/so101) uses six
STS3215 motors including the gripper. Confirm actual IDs, models, directions and
mechanical ranges before calibration. The [LeRobot register table](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py)
is the reference for the telemetry decoder.

Both verified named RGB streams use FFmpeg/AVFoundation. Investigating Gemini 2
depth, intrinsics and color/depth alignment requires the
[official Orbbec SDK](https://github.com/orbbec/pyorbbecsdk/tree/v2-main).
The installed package is `pyorbbecsdk2`; its import is `pyorbbecsdk`.
It includes `examples/quick_start.py`, `examples/beginner/03_color_and_depth_aligned.py`
and `examples/beginner/04_camera_calibration.py` under
`.venv/lib/python3.12/site-packages/pyorbbecsdk/`.
The harness supports Orbbec SDK profile enumeration and color streaming by serial
when SDK access succeeds. Verified RGB streaming currently uses exact camera names.
RGB-D capture remains pending native SDK troubleshooting.

## Bounded movement bring-up

`nudge.py` prepares a single wrist-roll/gripper test for confirmed standard IDs
5 or 6. It requires torque initially off, position mode, valid single-turn bounds,
voltage within the motor's configured limits and temperature below 50 C. It caps
the displacement at 114 ticks for wrist ID 5 (about 10 degrees) and 24 ticks for
gripper ID 6, stages the current goal before enabling torque,
uses a reduced torque limit, monitors the outbound and return motion, and releases
torque at the end. This is for a supported rest pose with hands clear, not general
arm control. It changes SRAM tuning and leaves reduced torque/speed settings in
place; it does not write EEPROM. Torque release cannot be guaranteed after a lost
physical connection or forced process termination.

Read-only preflight (verified on the connected motor):

```sh
.venv/bin/python nudge.py --port /dev/cu.usbmodem5AE60846541 --id 5 --delta 24
```

Adding `--execute` attempts physical motion and writes telemetry to a unique
`captures/*/motion.jsonl`. The first test on ID 5 (1745 to 1769 ticks at torque
limit 150) timed out after four seconds with the encoder unchanged at 1745.
Torque release was confirmed by a subsequent register read. Evidence is in
`captures/20260914T033738-0820a1dc/motion.jsonl`; the recorded ten-second camera
clip is `captures/first-motion/gemini.avi`. This was a failed motion test.

Tests now log voltage, current, load, torque, goal readback, speed, temperature
and status during movement, and stop if voltage or command readback is invalid.
EXP-23's `BUILD-LOG.md` confirms this arm's 5 V 6 A supply; its previous control
stack used acceleration 254 and enabled the SRAM lock after enabling torque.
The current harness uses that acceleration and enable sequence.

The 114-tick wrist test at torque limit 300 in
`captures/20260914T170235-0520bc57/motion.jsonl` measured motion from 1748 to
1786 ticks (about 3.3 degrees), then stopped short of the 1862 target at the
configured torque cap. The test timed out and verified torque release; a full
out-and-return test remains incomplete. The test accepts `--torque-limit 1..500`
(default 150), capped by the motor's EEPROM maximum. A previous test's reduced
SRAM limit no longer silently overrides the requested limit.

The subsequent 500-limit test in
`captures/20260914T170959-2ac66a72/motion.jsonl` moved from 1779 to 1791 ticks
(about 1 degree) and timed out before reaching 1893. Its torque-off write and
readback succeeded. `captures/wrist-test-500.avi` records 12 seconds of the Gemini
stream during this test (300 frames, 25 fps); the small motion is difficult to
resolve visually in the close camera view. The controller disappeared from the
serial inventory after the test, preventing an independent follow-up motor read.
Do not infer a completed return or reliable motion from these partial tests.

### First visually verified wrist movement

After reconnection and visual confirmation of hands clear, the reverse-direction
test moved ID 5 from 1774 to 1674 ticks (100 ticks, about 8.8 degrees).
`captures/20260914T213620-ddbf40a9/motion.jsonl` records the motion; the concurrent
`captures/wrist-reverse-500.avi` contains 300 frames at 25 fps. Inspected frames at
0, 3 and 6 seconds show the wrist/gripper rotating. Both named live streams were
also confirmed fresh afterward.

This establishes the bring-up objective of commanding physical movement and
observing it through the camera stream. It does not establish precise tracking:
the wrist stopped 14 ticks short of the requested 1660 target, so the four-second
timeout skipped the return phase. Torque release was verified both by the test
and an independent subsequent motor read (1674 ticks, torque off, status 0,
27 C, 5.4 V). Reliable bidirectional positioning remains future work; do not
increase force to compensate without diagnosing the directional resistance.

After detection: verify both streams, map and calibrate motors, establish joint
limits, then test bounded motion with an operator present. Autonomous pickup
requires camera-to-arm transforms, a matching kinematic model, target localization,
collision checks and visual feedback. Painting additionally needs a calibrated
tool tip and contact handling. These capabilities are not yet implemented.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Twelve tests cover invalid and broadcast IDs, explicit ID lists, refusing telemetry
from unknown or faulted devices, read-only telemetry with signed encoder decoding,
color conversion, stale frames, and actual HTTP snapshot/MJPEG delivery using a
generated test image, movement preflight rejection, stale-goal prevention, and
torque release on a detected motion fault. The HTTP test requires localhost socket access outside the
Codex sandbox; generated test frames do not constitute physical camera validation.
OpenCV, Feetech and Orbbec SDK imports succeeded. CLI help, device enumeration,
physical servo reads, physical RGB frames and proposed rule matching were checked.
Physical wrist movement is now verified by encoder telemetry and the concurrent
camera recording. A completed movement-and-return test remains pending.
