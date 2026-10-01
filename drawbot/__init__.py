"""SO101 marker drawing with a depth camera: RGB-D sensing, paper frame, closed-loop joint control.

State/config JSON files live in $DRAWBOT_STATE (default ./drawbot_state): frame.json, site.json,
tilt_ref.json, draw2_state.json (Jacobian + contact_h), config.json (optional overrides), park.json,
kin_model.json (depth-calibrated URDF model, kinmodel.py) and calib_samples.jsonl (calib.py).
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULTS = dict(
    ids=[1, 3, 4],                                   # controlled joints: pan, elbow, wrist (ID2 held)
    lim={"1": [700, 3400], "3": [700, 3300], "4": [1735, 3775]},
    site_bounds=[[-200, 200], [-350, 350], [-220, 210]],  # around site.json hover_u while drawing
    relocate_bounds=[[1450, 2150], [900, 2000], [1750, 2450]],
    track="orange",
    pen_L=28.0,        # barrel center -> pen tip, mm
    pen_up=12.0,       # pen-up height above contact (barrel h), mm
    press=2.5,         # extra open-loop "down" beyond contact, mm
    step=2.0,          # waypoint spacing, mm
    speed=4,
    gripper_open=1000,   # field open command on a marker pick
    gripper_stall=794,   # where the jaws stall on a marker barrel; close = stall - margin
    gripper_margin=38,   # command only ~35-40 ticks past the stall (a stalled servo heats ~1 C/min)
    grasp_tool_mm=None,  # grasp center in the gripper frame (mm); None = from kinmodel.GRASP_OBS
)


def state_dir():
    return Path(os.environ.get("DRAWBOT_STATE", Path.cwd() / "drawbot_state"))


def path(name):
    return state_dir() / name


def load(name, default=None):
    p = path(name)
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"{p} missing (set DRAWBOT_STATE or create it; see docs/DRAWBOT.md)")
    return json.loads(p.read_text())


def save(name, obj):
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def config():
    return {**DEFAULTS, **load("config.json", {})}
