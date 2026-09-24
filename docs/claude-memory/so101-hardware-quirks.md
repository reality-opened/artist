---
name: so101-hardware-quirks
description: "SO101 limits/quirks not obvious from code — shoulder stop ~2010, gripper overload EEPROM raised, garbled reads, teleop executes every command, SIGTERM holds torque"
metadata:
  node_type: memory
  type: project
  originSessionId: 51d769f3-b373-40c7-8cd6-be87b40f782e
  modified: 2026-09-24T21:34:51.305Z
---

Observed 2026-09-24 (details in the repo's docs/SESSION-20260924.md):
- Shoulder ID2 has a **physical stop near 2000-2010**. URDF IK often wants 1400-1980, which
  stalls at full load. Clamp ID2 >= 2030 (pen.py does).
- Gripper ID6 EEPROM overload protection was raised with user approval: reg 36 25->80, reg 34 20->50.
  The old values clamped the grip to 20 % after 2 s, and markers slid out. Stalled grip heats ~1 C/min.
- Garbled register reads happen (temp 247, 3.8 V). teleop.py overheat now needs 3 consecutive plausible reads.
- Every JSON line sent to teleop executes. Never "probe limits" with a real command.
- SIGTERM on teleop leaves torque on (holding) and takes ~5 s to exit. Only `release` drops torque.
- Changing EEPROM protection or weakening the overheat check got blocked by the permission
  classifier until the user explicitly approved it. Ask first; don't route around it.

**Why:** Each of these cost time or risked dropping the arm this session.
**How to apply:** Check this before planning joint targets or restarting controllers.
Related: [[visual-teleop-preference]], [[whiteboard-writing-task]].
