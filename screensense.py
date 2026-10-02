"""Fast change sensor on one window (Quartz window capture): the contact sensor when typing into a real app.

ScreenSensor("hello.py").changed() is True once the window differs from its baseline by more than the cursor
blink does (threshold measured by calibrate_noise()). Captures the window itself, so overlapping toasts and
other windows do not count.
"""
import time

import numpy as np
import Quartz


def find_window(title_part, owner=None):
    infos = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    for w in infos:
        name = w.get("kCGWindowName") or ""
        if title_part in name and (owner is None or w.get("kCGWindowOwnerName") == owner):
            return int(w["kCGWindowNumber"]), w["kCGWindowBounds"]
    raise RuntimeError(f"no window with {title_part!r} in its title")


def grab(wid):
    img = Quartz.CGWindowListCreateImage(Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, wid,
                                         Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution)
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    bpr = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    a = np.frombuffer(data, np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3]
    return a.mean(2)


class ScreenSensor:
    def __init__(self, title_part, owner="Code", band=(0.0, 0.6)):
        self.wid, self.bounds = find_window(title_part, owner)
        self.band = band  # vertical fraction of the window to watch (editor text, not the status bar)
        self.threshold = 60
        self.rebase()

    def region(self):
        g = grab(self.wid)
        h = g.shape[0]
        return g[int(self.band[0] * h):int(self.band[1] * h)]

    def rebase(self):
        self.bases = [self.region()]

    @staticmethod
    def _d(a, b):
        return int((np.abs(a - b) > 40).sum())

    def diff(self):
        g = self.region()
        return min(self._d(g, b) for b in self.bases)

    def calibrate_noise(self, seconds=1.6):
        """Collect the cursor-blink states as extra baselines, then set the threshold above what remains."""
        self.rebase()
        t_end = time.time() + seconds
        while time.time() < t_end:
            g = self.region()
            if min(self._d(g, b) for b in self.bases) > 15 and len(self.bases) < 4:
                self.bases.append(g)
            time.sleep(0.03)
        t_end, worst = time.time() + seconds, 0
        while time.time() < t_end:
            worst = max(worst, self.diff())
            time.sleep(0.03)
        self.threshold = max(25, 3 * worst)
        return worst, self.threshold, len(self.bases)

    def changed(self):
        return self.diff() > self.threshold
