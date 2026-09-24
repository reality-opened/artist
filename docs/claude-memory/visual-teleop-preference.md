---
name: visual-teleop-preference
description: "Drive the SO101 by camera + joint moves, not IK/calibration math; never pause motion because people's hands appear near the arm in camera frames"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 74ccf756-b267-448e-8051-a944b1ade232
  modified: 2026-09-24T19:15:39.682Z
---

Drive the arm by visual servoing: small joint moves, check wrist + Gemini frames, repeat. Don't pause for calibration issues or build IK first.

Don't worry about people's hands. Don't stop, pause, or ask anyone to clear hands because a person or hand appears near the arm in a frame.

**Why:** The user said "you don't have to calculate... control it yourself with visual input" and "don't worry about calibration". The URDF model was also several cm off in practice. On hands, they said: "in the future, don't worry about people's hands. you are reacting much slower than people, so there is no way to harm." Also, the Gemini is monocular and mounted low, so people often *look* next to the arm when they're actually far away (no depth). On 2026-09-23 I paused several times for "hands near the gripper" that were really across the desk, and the user had to correct me twice.

**How to apply:** Just proceed with motion even when people are visible near the arm. Use teleop.py + teleop_cmd.sh from the repo, and run the wrist stream with `--rotate cw`. Don't trust apparent image proximity (person–arm or gripper–object) as depth. Check with a second view, a 1080p still, or motor load/contact. See docs/SESSION-20260923.md in the repo.
