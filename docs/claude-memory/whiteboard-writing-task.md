---
name: whiteboard-writing-task
description: "Paused task (2026-09-24) — write \"hello\", sign \"claude\" in black, draw Anthropic A\\ logo in red on flat whiteboard; strategy + where it stopped"
metadata:
  node_type: memory
  type: project
  originSessionId: 51d769f3-b373-40c7-8cd6-be87b40f782e
  modified: 2026-09-24T21:34:56.553Z
---

Goal from the user (2026-09-24): write "hello" on the whiteboard and sign off "claude" in black,
then draw the Anthropic logo (A\) in red. **Not started drawing yet**; paused while the user
switched laptops. The full plan is in the repo's docs/SESSION-20260924.md.

Strategy that works so far:
- The user hands markers over, uncapped. The arm can't uncap them.
- Grip **along the jaws near the tips**, chisel tip 4-5 cm past the jaw tips, gripper torque 500.
- Write with the jaws tilted 60-70 deg down. The pen-tip reach at the board is then ~0.28-0.41 m
  (URDF base frame). Text runs tangentially via pan.
- `pen.py` does relative Cartesian moves (fixed jaw pitch, captured sag offsets) on top of
  teleop. Find the board by descending with `pen_probe.py` until measured stops following commanded.

**Why:** Saves the next session from re-deriving grip, workspace and tooling.
**How to apply:** Resume at step 2 of the plan in docs/SESSION-20260924.md. Recheck live state first.
Related: [[verify-grip-orientation]], [[so101-hardware-quirks]], [[visual-teleop-preference]].
