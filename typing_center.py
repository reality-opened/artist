"""Verify saved keys on the browser logger; correct nearby key misses before editor typing."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import keytap
import keytype as kt
import typing_trial
from drawbot import arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        return print("dry run: add --yes")
    assert kt.frontmost() == "Safari"
    Path(f"captures/keymap-before-centering-{time.time_ns()}.json").write_text(kt.STATE.read_text())
    for key in dict.fromkeys(a.keys):
        for attempt in range(3):
            sys.argv = ["typing_trial.py", key, "--depth", "30", "--timeout", "2", "--yes"]
            typing_trial.main()
            p = max(Path("captures").glob("typing-trial-*.json"), key=lambda x: x.stat().st_mtime_ns)
            report = json.loads(p.read_text())
            ev = report["events"]
            downs = [e for e in ev if e["type"] == "down"]
            if not downs and attempt < 2:
                m = report["samples"][-1]["motors"]["2"]
                if abs(m["pos"] - m["goal"]) <= 20 and (m["load"] & 1023) < 120:
                    found = kt.cal2(key, start_above=40, max_ticks=100)
                    if found is None:
                        raise RuntimeError(f"contact search did not find {key!r}")
                    if found == key:
                        print(f"CONTACT VERIFIED {key!r}", flush=True)
                        continue
                    st = kt.load()
                    k = st["keys"][key]
                    contact = arm.status()["motors"]["2"]["target"] + kt.RELEASE
                    k["q_contact"][1] += contact - k["c2"]
                    k["c2"] = contact
                    kt.save(st)
                    downs = [{"keysym": found}]
            if len(downs) != 1:
                raise RuntimeError(f"{key!r}: expected one key-down, got {len(downs)}")
            hit = kt.key_of(downs[0])
            if hit == key:
                print(f"CENTER VERIFIED {key!r}", flush=True)
                break
            if hit not in kt.LAYOUT or key not in kt.LAYOUT:
                raise RuntimeError(f"unknown key miss {hit!r}")
            dg = np.array(kt.LAYOUT[key]) - np.array(kt.LAYOUT[hit])
            if np.linalg.norm(dg) > 1.5:
                raise RuntimeError(f"key miss too far away: {key!r} -> {hit!r}")
            st = kt.load()
            A, _ = kt.fit(st["obs"])
            dxy = .65 * (A @ dg)
            k = st["keys"][key]
            q = np.array(k["q_contact"])
            q[1] = k["c2"]
            dq = keytap.solve(np.r_[q, 3593], np.r_[dxy, 0])
            k["q_contact"] = (np.array(k["q_contact"]) + dq).round(1).tolist()
            k["c2"] = round(k["c2"] + dq[1])
            k["v"] = kt.vertical(np.r_[k["q_contact"], 3593]).round(3).tolist()
            kt.save(st)
            print(f"CENTER ADJUST {key!r} hit {hit!r}: xy {dxy.round(1).tolist()}", flush=True)
        else:
            raise RuntimeError(f"could not center {key!r}")


if __name__ == "__main__":
    main()
