# Robot arm controller

Current pencil task state and controls: [PENCIL-TASK.md](docs/PENCIL-TASK.md).

Latest session (2026-09-24, whiteboard writing, paused): [SESSION-20260924.md](docs/SESSION-20260924.md).
Previous (2026-09-23, scissors pickup, camera/teleop setup): [SESSION-20260923.md](docs/SESSION-20260923.md).

Claude Code memory for this project is mirrored in [docs/claude-memory/](docs/claude-memory/);
copy it into `~/.claude/projects/<this-repo-path-with-dashes>/memory/` on a new machine.

Next agent: start with the [operations handoff](docs/HANDOFF.md) for exact commands,
verified motion evidence, current state and known limitations.

Environment, commands, proposed Codex permissions, and current hardware status:
[Setup guide](docs/SETUP.md).

you are going to build a harness that controls the SO101 robot, specifically here is my set up: I have six motors, each with a shiftable angle, and I have two cameras, one is a wrist camera that is a cheap WebCam and the other is an orbecc gemini 2. I want you to familiarize the existing cameras and ways to operate the robot completely from Codex, and then be able to use it to pick up mundane items and eventually do crazier stuff like painting.
