"""Descend the pen in 5 mm steps, printing commanded vs measured model z (contact finder).

Usage: pen_probe.py STEPS [--reset] [--out]   (--out first moves to r=0.40 m)
"""
from pen import *
import numpy as np, sys
pen = Pen(reset="--reset" in sys.argv)
print("start", pen.p.round(3), "axis_z", round(pen.axis_z, 3))
if "--out" in sys.argv:
    ang = np.arctan2(pen.p[1], pen.p[0])
    pen.line([0.40*np.cos(ang), 0.40*np.sin(ang), pen.p[2]], step=0.005, speed=4)
steps = int(sys.argv[1])
for k in range(steps):
    pen.goto(pen.p + [0, 0, -0.005], speed=3)
    time.sleep(0.4)
    m = status()["motors"]
    meas = [m[str(i)]["pos"] for i in range(1, 6)]
    pm, am = fk(meas)
    print("cmd z %.3f  meas z %.3f  dz %+.1fmm  loads %s" % (pen.p[2], pm[2], (pm[2]-pen.p[2])*1000,
          [m[str(i)]["load"] for i in range(1, 5)]))
pen.save()
