---
name: verify-grip-orientation
description: "Before asking the user to re-place a held object (marker), confirm its orientation with an enlarged Gemini crop — the wrist cam can't tell along-jaws from across-jaws"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 51d769f3-b373-40c7-8cd6-be87b40f782e
  modified: 2026-09-24T21:34:45.446Z
---

Judge how an object sits in the SO101 gripper from an **enlarged crop of the Gemini frame**
(cv2 crop + 2-2.5x resize), never from the wrist cam alone. Say which end is the tip and
whether it is along or across the jaws, then act.

**Why:** On 2026-09-24 I misread the marker orientation twice. The wrist cam looks along the
jaws, so along-jaws and across-jaws look the same. A 640px Gemini frame at a low angle
also fooled me once. The user had to re-place the marker 4 times. Their correct placement
(tip out past the jaw tips) got "fixed" by me. They eventually interrupted.

**How to apply:** Crop and enlarge the gripper region before any "please flip/re-place"
request. If still ambiguous, move the wrist to show a side profile to the Gemini first. Asking the user once
with a clear description beats multiple wrong corrections. Related: [[visual-teleop-preference]],
[[so101-hardware-quirks]], [[whiteboard-writing-task]].
