"""Read-only SO101 URDF forward kinematics for planning; no hardware access."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


URDF = Path(os.environ.get("ARM_URDF", Path(__file__).parent / "calibration" / "so101_new_calib.urdf"))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class Kinematics:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        self.by_child = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            t = np.eye(4)
            t[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
            t[:3, 3] = xyz
            axis = j.find("axis")
            limit = j.find("limit")
            record = dict(name=j.get("name"), kind=j.get("type"), origin=t,
                          parent=j.find("parent").get("link"),
                          axis=np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else None,
                          limits=(float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else None)
            self.joints[record["name"]] = record
            self.by_child[j.find("child").get("link")] = record

    def fk(self, radians, link="gripper_frame_link"):
        q = dict(zip(JOINTS, radians))
        chain = []
        while link in self.by_child:
            joint = self.by_child[link]
            chain.append(joint)
            link = joint["parent"]
        result = np.eye(4)
        for joint in reversed(chain):
            result = result @ joint["origin"]
            if joint["kind"] != "fixed":
                r = np.eye(4)
                r[:3, :3] = rotation(joint["axis"], q[joint["name"]])
                result = result @ r
        return result

    def jacobian(self, radians):
        q = np.asarray(radians, dtype=float)
        eps = 1e-5
        return np.column_stack([(self.fk(q + np.eye(5)[i] * eps)[:3, 3] -
                                 self.fk(q - np.eye(5)[i] * eps)[:3, 3]) / (2 * eps)
                                for i in range(5)])


def ticks_to_radians(ticks, calibration):
    # Mirrors the installed LeRobot DEGREES conversion; offsets are already
    # applied by the servo to its Present_Position register.
    return np.array([(ticks[i] - (calibration[name]["range_min"] +
                                 calibration[name]["range_max"]) / 2) * 2 * np.pi / 4095
                     for i, name in enumerate(JOINTS)])
