# Pencil task, 2026-09-14

The requested objective remains **pick up the pencil and write HI on the sticky
note**. Neither pickup nor writing has happened. This supersedes the older
HANDOFF.md for physical state; always recheck live.

## Current physical issue

Live images show the gripper tips staying at the tabletop while the base shifts
and lifts during commands. In particular, a shoulder change from 2426 to 2369
ticks moved the base upward while the tips stayed in place. The last step was
reversed, then all six motors were released. Independent `arm.py motors` reads
confirmed torque off on every motor. The user has been asked to clamp/bolt the
base flat to the desk before continuing. Do not assume the URDF base frame is
fixed to the desk or its predicted z equals actual tip clearance.

Final raw positions: IDs 1..6 = 2115, 2406, 1632, 3630, 1674, 1793.
All temperatures 27–30 C, supply 5.3–5.4 V, status 0. Arm SRAM torque limits
are now 500, gripper 150. These are historical observations, not live state.
The controller repeatedly disconnected earlier, but was connected at the final
read. Both camera HTTP streams were delivering fresh frames.

## Implemented and exercised

- `open_gripper.py`: empty gripper opening with limited recovery of a resting
  position up to four ticks below its EEPROM minimum. Two opening attempts moved
  ID 6 from 1676 to 1794, with later readings near 1797. Both timed out short of
  their targets and released torque.
- `kinematics.py`: NumPy URDF FK, matches the installed LeRobot/Placo matrix at
  the initial pose to 7.8e-16. This establishes software agreement, **not physical
  registration**. Uses the old EXP-23 URDF and calibration file paths.
- `joint_step.py`: bounded arm steps, all-motor observation, calibration and
  model path gates. Releases selected motors after every step. Elbow drift after
  release showed that this cannot retain all approach poses.
- `arm_session.py`: JSON-line session with all six motors held between commands,
  current-goal staging, calibrated limits, model clearance checks, telemetry,
  fault monitoring, release, and return to session start. Maximum command delta
  is 114 ticks per joint. Deltas apply to measured positions. Four-second
  settling timeouts report `targets_reached: false` and continue holding, never
  claim success. Commands use a PTY; send one JSON line at a time.

Examples (recheck clearance and physical base fixation first):

```sh
.venv/bin/python arm.py motors --port /dev/cu.usbmodem5AE60846541 --ids 1,2,3,4,5,6
.venv/bin/python arm_session.py --port /dev/cu.usbmodem5AE60846541 --torque-limit 500
.venv/bin/python arm_session.py --port /dev/cu.usbmodem5AE60846541 --torque-limit 500 --execute
```

Session commands: `{"status":true}`, `{"move":{"2":24}}`,
`{"park":true}`, `{"release":true}`. EOF/fault releases torque. At 120 seconds
without a command it attempts return to its session start, then release.
`park` uses bounded joint-space steps toward the session start; it is not a
collision-certified path reversal. Do not leave a live motion session unattended.
No control session remains active: PTY handles 72338, 39482, 63126 all exited
after release/park. Revalidate before starting a new one.

## Evidence and limitations

Live EEPROM limits and offsets matched the old calibration file for all six
motors; no calibration or gain EEPROM writes were made. Motor 3 often stopped
short under load; at torque limit 300 it saturated. At 500 the arm moved further,
but actual vs commanded positions still differed. No full precision trajectory
has passed. Eight new tests across open_gripper, joint_step and arm_session
passed when added; those tests are software checks, not proof of grasp safety
or mechanical control. The original nudge tests were not changed.

Most recent session telemetry:
`captures/20260914T220607-3935acdc/session.jsonl`.
Previous holding sessions:
`captures/20260914T220211-7c94ce6a/session.jsonl` and
`captures/20260914T220306-0203c37b/session.jsonl`.
The pencil remains on the blank note. User authorized motion generally and
specifically approved a ten-degree elbow move after an automatic review block.
Runtime permissions later changed to unrestricted with no approval prompts.
This does not establish physical base fixation.
