# drawbot: marker drawing with a depth camera

`drawbot/` packages the field-tested scratch scripts from 2026-09-26 (scene3d, servo, draw2,
run_layer, relocate, marker_pose, align/fingers, wristcam, smiley, and later calib_collect, kin_fit,
kinmodel, manip, heightmap, case_state) into one library plus
`drawbot_cli.py`. The arm is driven through `teleop.py`'s command file (`$ARM_CTL`, default `./ctl`).

## Pipeline

1. **RGB-D** (`rgbd.py`). Gemini color `:8766/frame.jpg`, color-aligned uint16 mm depth `:8767/depth.png`,
   intrinsics `:8767/intrinsics`. `grab(avg)` takes the per-pixel median of `avg` depth frames
   (0 and >= 5 m ignored). `backproject`/`project` convert between pixels+mm and camera XYZ
   (x right, y down, z forward).
2. **Paper frame** (`frame.py`). The paper is the largest bright low-saturation blob. RANSAC plane
   (3 mm inliers) on eroded paper pixels, then SVD refinement (`full_matrices=False`: a full SVD of
   >100k points needs an N x N matrix and runs out of memory). Corners come from the convex hull of
   the mask (arm shadows notch it) with the corner rays intersected with the plane. Plane coordinates:
   `x, y` mm on the paper (origin at the paper center), `h` = height above the paper. `e1` lies along
   the paper edge closest to image-left, `e2 = n x e1` points toward the camera (image bottom).
   `BASE` is the robot pan axis in plane xy. It is the center of the circle the marker traces when only
   the pan moves (`base_from_pan`, >= 3 pans). The same fit gives ticks per degree for `pan_for_angle`.
3. **Tracking** (`track.py`). The marker is a colored blob. Its median 3D point, mapped into the
   plane, is the barrel position `(x, y, h)`. Candidates must have area > 150 px, `|x| < 250`,
   `|y| < 120`, and `min_h < h < 250`. When a prior is given, the nearest blob within `gate` mm (40)
   wins, so distractors far from the pen are ignored. `tip_estimate` gets the pen axis from the 3D
   principal direction and radial/tangential tilt relative to BASE. The tip is taken as the lowest
   colored points. `list_blobs` reports every colored blob near the paper: color, xyh, area, pixel,
   2D axis, length and width, and the tip direction. The tip end is found by looking for white pixels
   just past each end of the barrel. It also gives r and angle from BASE, the tip-vs-radial angle,
   and flags `on_table` (h < 40) and `marker_like`.
4. **Closed loop** (`control.Ctl2`). Control `u = (pan ID1, elbow ID3, wrist ID4)`. Shoulder ID2 is held.
   Output `y` = barrel xyh, modeled as `y ~= y0 + J du`. `J` (3x3, mm/tick) comes from central differences
   (`estimate`, +-35/40/40 ticks). `servo` takes damped steps `du = lstsq(J, 0.6 err)`, clipped to 60
   ticks. Each measurement is checked against the prediction `y_prev + J du`: readings more than 20 mm
   off in xy are re-measured, and after 3 tries the prediction is used. More than 5 rejects raises.
   `freeze_h` ignores h error while the pen is in contact. `blend` mixes the prediction into the reading
   (0.5 while drawing). Broyden refinement is available (`learn=True`) but off by default.
5. **Skills** (`skills.py`).
   - `Pen` holds the pen geometry. Tilt = `(ID3 - ref3 + ID4 - ref4) * 2pi/4096`, positive when the
     pen bottom points toward the base. `tilt_ref.json` is where the pen is vertical. The barrel target
     for a tip target is `tip + (L sin(tilt), 0)` with L = 28 mm. The correction is applied along plane
     x only, because the base sits at -x from the site.
   - `pen_down`: servo to `contact_h + 2.5`, then an open-loop `-(2.5 + press)` mm step through J.
   - `pen_up`: `+(pen_up + press)`.
   - `draw_layer`: smiley strokes mapped to the plane with face +x -> plane -x and face +y -> plane -y
     (upright from the Gemini), scaled by `r_face / 35` and resampled every 2 mm. Each waypoint gets
     one `freeze_h` servo iteration. On a tracking failure it lifts, re-acquires (then again with a
     100 mm gate), returns to the waypoint, puts the pen down and resumes. `--start K` resumes stroke 0
     at waypoint K. Every point is logged as JSON to `layer_<marker>.log`, and the final image is saved
     to `after_<marker>.jpg`.
   - `relocate`: moves in the air in legs of at most 25 mm (lifting with the wrist first if near
     contact) and re-estimates J when a leg gains less than 5 mm.
   - `probe_contact`: lowers the pen 1.5 mm per step through J. A step is *free* when the measured
     dh is at least 35 % of the predicted dh. Contact is declared after 2 consecutive non-free steps,
     but only once 2 free steps have been verified and only while teleop's targets equal the
     commanded ones. This way a stopped controller, a rejected command or lost tracking cannot be
     mistaken for contact. Afterwards the pen is lifted, and `contact_h` plus `site.json` `tip_contact`
     are saved.
   - `park`: lifts through J, then moves to `park.json`.
   - Also: `track.marker_pose`, `jaws`, `fingertips` and `grasp_alignment` (from depth, noisy), and
     `rgbd.wrist_blob` (wrist cam orange centroid and angle).

### Why depth tracking

The URDF/FK is only good for *relative* motion: gravity sag, backlash, and an uncalibrated hand-eye
transform give centimeter-scale absolute errors. The depth camera measures the pen barrel directly,
in metric units, in the paper's own frame, so closing the loop on it absorbs all of that. The
Jacobian is re-estimated at the drawing site because it changes with pose.

## Kinematic model, calibration and manipulation

The drawing loop above never trusts the URDF. For moves in free space (hovering, poking, picking markers out of
the case) a URDF model *calibrated against the depth camera* is good enough: ~4 mm rms in the paper frame.

### Model (`kinmodel.py`, `$DRAWBOT_STATE/kin_model.json`)

`p_plane = R_PB @ (fk_urdf(q)[tool] * 1000) + T_PB` with `q_j = (ticks_j - MID_j) * 2pi/4095` (MID = middle of
the LeRobot calibration range) and `q_j = s_j q_j + o_j` for ID2..ID4. Twelve parameters
`x = [rotvec R_PB, T_PB mm, o2, o3, o4 rad, tool point m (gripper frame)]`, plus `signs` (always `[1, 1, 1]` so
far). Pan and roll have no offsets: the pan offset is absorbed by `R_PB`.

- `fk(ticks5, tool)` gives `(xyh mm, jaw axis, rotation)` in the plane frame. The jaw axis is the gripper-frame z
  axis, pointing out of the jaws, so `(0, 0, -1)` means jaws straight down. `tool=None` is the TCP =
  `gripper_frame_link` = the **fixed-jaw tip**. `m.TOOL` is the fitted orange-marker point.
  `m.grasp_tool` is the grasp center (see below). `close_dir(ticks5)` is the jaw closing direction. From the
  URDF this is the gripper-frame x axis.
- `ik(target, axis=None, roll=3746, seed=None, tool=None, w_axis=60)` runs multi-start damped least squares on
  (pan, ID2, ID3, ID4). It starts from `seed` plus 6 fixed seeds, with the pan aimed at the target from BASE
  (`1977 - (ang - 0.7) * 250/21.45`), using 2-tick numeric Jacobians and steps clipped to 150 ticks. Joint window
  `WIN`: ID1 800..3300, ID2 2030..3950 (shoulder stop), ID3 900..2080 (elbow fold stop), ID4 1730..3770. The
  result is `IKResult(q, res, axis, axis_err)`. `res` is the position residual; treat it as unreachable when
  `res > 4 mm` (or the axis error is > 8 deg): `result.ok()`.
- `goto(target, axis, roll, seed, tool, speed=5, rounds=3, tol=6, w_axis=25, max_res=4)` runs IK and raises
  when the rounded-joint residual is > `max_res`, *before* sending anything. It then commands the joints, reads
  the **measured** positions, and adds `0.8 * (q - pos)` to the command until every joint is within `tol` ticks
  (joint sag under gravity, mostly ID2/ID3). It does not use the camera. Roll (ID5) is not commanded; `roll`
  only tells IK where the wrist is.

### Calibration workflow (`calib.py`)

1. Hold the orange marker firmly in the jaws. Check that `track` sees it and that the table under the arm is
   clear.
2. `calib-collect --yes`: 6 anchor poses x 10 perturbations (pan/shoulder/elbow, wrist relative to `level4`,
   ID4 clipped to 1760..3760). At each pose it records the measured ticks and the tracked marker (plane xyh).
   When the marker is below 60 mm the arm returns to the safe home (`2=3250 3=1550 4=level4`) before the next
   pose. It also goes home between anchors. Samples are **appended** to `calib_samples.jsonl` (the `cam` key
   holds plane `x, y, h`; `cam_xyz` holds camera mm). `--start-anchor K` resumes.
3. `kin-fit`: Kabsch initial guess, then Levenberg-Marquardt (numeric Jacobian, vectorized batched FK, < 1 s),
   with 4 starts (zero offsets plus 3 random +-1.2 rad). Samples with error > 15 mm are dropped (`--thresh`) and
   the fit is repeated from the previous solution. `--signs` also searches the 8 sign patterns of ID2..ID4
   (8x slower; the field answer was `[1, 1, 1]`). `--no-save` only prints. Field result: **rms 4.2 mm on
   52/53 samples** (median 3.4, max 9.0), offsets -101.6 / 87.8 / -10.0 deg, marker tool (26, -11, -19) mm.
4. Check: `fk` (current pose), `ik X Y H --down`, `goto ... ` (dry run) against what the camera shows.

**Identifiability.** All field samples used roll 3746. With one roll, the wrist-flex offset `o4` and the tool
point trade off exactly: a rotation about the wrist-flex axis can be absorbed by moving the tool point. So the
data do not calibrate the **jaw axis direction**, and `kin-fit` prints a note about it. The fitted `o4`
(-10 deg) is simply where LM stopped starting from 0. Jaws-down moves looked right in the field, but treat the
jaw axis as good to a few degrees at best. To pin it, collect a second set at the flipped roll:
`calib-collect --roll 1698 --yes` (a guarded roll turn at the safe home comes first), then refit on both sets.
The shoulder offset `o2` is only weakly separated from the base rotation, because the pose set spans about
+-25 deg of pan. Compare fits by prediction error, not by their parameters.

### Reach map (table height, from BASE = pan axis in `frame.json`)

| r from BASE | what works |
| --- | --- |
| <= ~245 mm | jaws straight down |
| ~245-300 mm | jaws tilted 35-50 deg outward (tips away from the base) |
| > ~300-310 mm | nothing; `pick` refuses beyond 310 even if the model claims a solution |

With jaws down, the wrist limit (ID4 <= 3770) caps the jaws-down height at about 60 mm above the table near the
middle of the range, so approach and lift legs are 40 mm (shortened automatically if needed). `markers` flags
each blob `down` / `tilted` / `out` and `reachable` (r <= 300).

### Grasp strategy (`manip.py`, `scene.py`)

- `scene.scan()` returns markers per color (green, pink, blue, yellow, orange) in the case region
  (x -200..150, y -200..60, h 3..60). Each has the median plane center, a 3D and a 2D axis, its length, mean V
  brightness, r/angle from BASE and a reach class. It also reports the image brightness, with `dark` when the
  mean V is < 50: the hue masks need V > 80, so turn the light on.
- `pick_marker(pose)` only plans. The grasp center is the marker center, reached with the grasp offset as the
  tool point. It tries jaws down first, then tilted 35/40/45/50 deg outward. The wrist roll is chosen so that the
  closing direction is perpendicular to the marker axis in the plane. Both 180-deg roll branches are tried, and
  the one nearest the reference roll whose fixed-jaw tip stays above the table wins. Returns approach
  (along -axis), grasp and lift joint targets (ID1..ID5), the gripper open/close ticks, and the tried variants.
  `pick COLOR` prints it. `pick COLOR --yes` runs `execute_pick`: open, guarded roll, approach, grasp
  (goto with sag correction), close, lift.
- **The tip direction does not matter at pickup.** A 180-deg roll (3746 <-> 1698) flips a marker held across
  the jaws. `flip --yes` turns ID5 in 100-tick steps and backs off 150 ticks when |load| > 300 (the raw Feetech
  register is sign-magnitude; bit 10 is the direction) or lag > 40. Roll can travel about 190 deg,
  3746 -> ~1600, with the cable load rising near the end (the field run tripped at 1546 and backed off to 1696).
  `ROLL_WIN` = 1600..4090.
- **Grasp offset.** The TCP is the fixed-jaw tip, not the grasp center. One observation (pan 2164, roll 4080,
  jaws down, joints 2164/2870/1550/3734, gripper 780): model TCP (-108, -51.6, 22) vs grasped marker center
  (-109.4, -39.9, 29). Rotated into the gripper frame with the model rotation this gives
  **(-5.2, -10.4, -7.2) mm**, computed by `KinModel.grasp_tool_from_obs`. The z part (7 mm back from the tip) and
  the x part (toward the moving jaw) are gripper properties. The y part lies along the hinge axis, which is also
  the marker's own axis, so it probably only says where along its length that marker was held. Override it with
  `config.json` `grasp_tool_mm: [x, y, z]`, or `grasp_tool` in `kin_model.json`.
- Gripper: `gripper_open` 1000; close = `gripper_stall` (794) - `gripper_margin` (38) = 756. Never command
  further past the stall point: the stalled servo heats. teleop now relaxes **only the gripper** on a gripper
  overheat (the arm keeps holding); the next `gripper` command re-enables it below 50 C.
- `poke(p0, p1, h)`: closed fingertip, jaws down. Hover at 55 mm, descend to `h`, slide to p1 in 3 goto steps
  (loads logged), then lift. `heightmap X0 X1 Y0 Y1` gives the per-cell 90th-percentile height from depth (the
  median of 7 frames), which is useful before poking or descending into the case.

## Picking markers (what works)

Field result 2026-09-26: **3 successful picks** (pink from the table, orange from the case edge, blue on paper) once
the fixed-jaw offset below was applied; several misses before it. Code: `drawbot/pick.py`, `drawbot/slide.py`
(ported from the scratch `bump.py`, `find_color.py`/`dark_scan.py`, `pick3.py`/`pick2.py`, `drag_marker.py`).
The older `manip.pick_marker` / `pick` (URDF closing direction, grasp offset from one observation) is kept, but
`pick2` is the one that worked.

1. **Find by shape, not colour.** In a dark room the hue masks fail or leak, while depth does not.
   `find COLOR` normalises the image brightness to a mean of 90 before HSV (bands in `pick.BANDS`, V > 60, depth
   250..900 mm, plane region x -200..150, y -220..200). It uses the colour blob only as a rough spot.
   `locate X Y` (`locate_points`) takes plane points 6..28 mm above the table within `radius` (60) of the guess,
   keeps the largest 8-connected cluster on a 3 mm grid, and fits PCA to get the axis. The center is the middle of
   the 2..98 % extent along and across the axis, because the camera sees more of the near end and of one side.
   `find` runs a 35 mm disc first, then a +-8 mm strip along that axis so a clipped end does not bias the
   center. It also reports length, width, top height, r from BASE, the end nearest BASE, and `tall_nearby`
   (points > 40 mm, e.g. the case wall). Checked in the field by projecting the result into the image: it was exact.
2. **The TCP is the fixed-jaw fingertip.** The model TCP (`gripper_frame_link`) is not the grasp center.
   `C_G` = the gripper-frame direction from the fixed jaw toward the moving jaw. It is **anchored by a wrist-camera
   roll sweep**, not taken from the URDF (the model's roll zero is uncalibrated): at roll 2330, tilt 20, over
   (-210, 143), the jaws closed across a marker with axis (-0.48, 0.88). This is stored as config `pick_anchor`
   (override it in `config.json` after a new sweep). `close_dir_gripper()` turns it into `C_G` with the model
   rotation at that pose.
3. **Roll.** `best_roll(p, tilt, axis, C_G)` searches ID5 1660..4080 in 20-tick steps (IK per roll, seeded with the
   previous one) for the roll whose in-plane `R(q, roll) C_G` is perpendicular to the marker axis.
4. **Offset.** TCP target = marker center - 25 mm * u (u = in-plane `R C_G`, config `pick_offset_mm`). This puts the
   marker between the open fingers. The roll is re-solved at the offset point.
5. **Motion.** Jaws tilted 20 deg outward (`pick_tilt`). Open 1000, hover at TCP h 60, descend **vertically**
   through 30 and 18 to 9 mm (`pick_grasp_h`), close to **730** (`pick_close`), then lift to 60. Each waypoint is
   IK-checked (residual <= 2.5 mm, lift <= 10), must lie inside `kinmodel.WIN` with the roll in 1660..4080, and is
   reached with joint-position feedback (`go_joints`: 3 rounds, +0.9 x error, 5-tick tolerance). Roll is commanded
   together with the other joints (as in the field).
6. **Grip: close hard, never ease.** The barrel stalls the jaws at ~822-827, so 730 is ~90 ticks past the stall.
   Easing to 60 past the stall afterwards let two markers slip out while lifting or levelling. This overrides the
   35-40 tick margin used by `manip` (and the stalled servo does warm up; teleop relaxes only the gripper on an
   overheat).
7. **Verify.** `verify_grasp` reads the gripper after the close and again after the lift. It counts as held when
   the position stopped at least 30 ticks short of the command (`grasp_min_gap`) and is <= 950: empty jaws reach
   the command. The load (sign-magnitude register) is reported but not used.

Wrist-camera lessons (not ported; see the scratch `wrist_servo.py`/`wrist_pick.py` if you need it): the image
angle vs roll is non-linear (oblique camera, about -0.104 deg/tick measured). Only centre from a safe height,
with the fingertips above the marker tops (~27-33 mm in the case), and use a parallax-corrected target: about
-6.4 px per mm of height, grasp target (218, 385) at TCP h 8. Probing at a low height pushes markers. At the roll
limits, fall back to roll +-2048 within 1650..4080. Wrist hue bands include pink (168..4, sat 80..205).

**Out of reach (r > ~300): slide it closer.** `slide COLOR --yes` presses the closed fingertip (gripper 700) on the
marker top (`top - 7` mm), starting 8 mm in from the end nearest BASE, and drags it **along its own axis** toward
the robot in 10 mm steps (tilts 0..65 deg allowed, smallest joint change wins, roll 3746). It stops once the
center would be within 240 mm (`stop_r`). Field: r ~339 -> ~262 mm. Pushing sideways instead shoves the whole case.
`--max-load N` lifts and stops when a joint load exceeds N while dragging (loads are logged either way).

The grasp height is fixed at 9 mm (fixed-jaw tip). For a marker sitting high (top > 25 mm, e.g. in the case) the
plan warns; pass `--grasp-h`. Config keys (`config.json`): `pick_anchor`, `pick_offset_mm`, `pick_tilt`,
`pick_hover`, `pick_descent`, `pick_grasp_h`, `pick_lift`, `pick_open`, `pick_close`, `grasp_min_gap`,
`grasp_max_pos`.

```sh
$P drawbot_cli.py find pink                       # colour spot + shape pose (read-only)
$P drawbot_cli.py locate -210 143 [--radius 60]   # shape pose near a rough spot (read-only)
$P drawbot_cli.py pick2 pink                      # plan (roll, offset TCP, joint waypoints); --yes executes
$P drawbot_cli.py pick2 -210 143 -0.48 0.88 --dry # explicit center + axis
$P drawbot_cli.py slide orange [--dist 80] --yes  # drag an out-of-reach marker toward the robot
```
`pick2 --yes` exits 0 when holding, 3 when the grasp check fails, and 2 when the plan is not executable.

## Safety

- `drawbot.arm.send` refuses until `drawbot.arm.enable()` is called. The CLI enables it only with
  `--yes`, so every moving subcommand is a dry run by default. Library scripts must call `enable()`
  themselves.
- `arm.move` refuses ID2 < 2030 (physical stop ~2010) and ID4 outside 1718..3782 (EEPROM). It
  raises if teleop reports it rejected the command.
- Hard limits: `Ctl2.lim` (config `lim`: ID1 700..3400, ID3 700..3300, ID4 1735..3775).
- While drawing, per-site bounds apply around `site.json` `hover_u`: pan +-200, ID3 -350/+350,
  ID4 -220/+210 (config `site_bounds`). Relocation uses absolute `relocate_bounds`.
- Every JSON line sent to teleop executes. Never "probe limits" with a real command.

## Known pitfalls

- **Backlash / deadband (P = 16).** Low servo P gain leaves several ticks of steady-state error, and
  reversals lose ticks. Very small steps may not move at all. The loop measures instead of trusting
  ticks. Broyden updates only happen for |du| > 6. Expect about 1-2 mm of residual jitter.
- **Wrist ID4 limit 1718..3782** (EEPROM). `level4(a2, a3) = 2934 - ((a2 - 2149) + (a3 - 2047))`
  keeps the jaws level, and `down4 = level4 + 1024` points them straight down. Check that the result
  is within range before sending.
- **Elbow fold stop at ID3 ~2086.** `lim` allows up to 3300, so in the drawing configuration rely on
  the site/relocate bounds (ID3 <= ~2000), or set `lim["3"]` in `config.json`.
- **Blue distractors.** Blue tape on the table, the scissors, the tape roll, and the marker case lid
  all match the blue hue. Tape mode limits depth to 550 mm and needs h > 15 mm. Use a prior and gate
  when tracking, and treat `blobs` output for blue with suspicion (`marker_like` helps).
- **Pink vs orange.** Pink caps (hue ~169-176, sat ~110-175) fall inside the orange band. The orange
  tracker's sat > 150 mostly excludes them. `list_blobs` uses sat > 185 for orange and sat <= 185 for
  pink, and removes overlapping detections.
- **Gripper stall heats the servo.** Command the gripper only about 35-40 ticks past the point where it
  stalls on the barrel (for example, a stall at ~794 means commanding ~755-760, not 700). A stalled
  grip heats about 1 C/min. teleop relaxes an overheated gripper, but an overheat on the other joints
  releases the arm.
- **Tape tracking.** Blue tape wrapped around the gripper and marker gives a larger, steadier blob
  than the orange cap when the pen is tilted or partly occluded (`TRACK=tape` / `--mode tape`,
  `blue` is an alias). Contact heights and J measured with one tracking mode do not transfer to the
  other, so re-probe after switching.
- **The marker case is light.** Pushing packed markers sideways moves the whole plastic case. Dragging on a
  marker top rolls the markers apart instead of pulling the case. To separate markers, press down on one and
  drag it; to move the case, push the case wall.
- **Model vs reality.** The model is ~4 mm rms, and worse at the edge of the calibrated volume (poses far from
  the anchors, other rolls). `goto` corrects joint sag but not model error. Near contact, come down slowly and
  check with the camera or `heightmap`.
- The Gemini gets bumped. `frame --fit-paper` re-fits the plane from the paper. If the paper also
  moved, BASE must be re-fit (pan circle).

## State files (`$DRAWBOT_STATE`, default `./drawbot_state/`, gitignored)

| file | format |
| --- | --- |
| `frame.json` | `{"ctr":[3],"e1":[3],"e2":[3],"n":[3],"d":f,"base":[2],"pan_ang":{"1717":deg,"1967":deg,...}}` + optional `pan_ref`, `ticks_per_deg`, `pan_offset` (defaults 1967, 250/21.45, 10) |
| `site.json` | `{"tip_contact":[x,y],"contact_h":f,"center":[x,y],"r_face":mm,"hover_u":[pan,id3,id4]}` |
| `tilt_ref.json` | `{"id3":ticks,"id4":ticks}` (pen vertical) |
| `draw2_state.json` | `{"J":[[3x3]] mm/tick, "contact_h":f}` |
| `park.json` | `{"1":ticks,"2":ticks,"3":ticks,"4":ticks}` |
| `config.json` | optional overrides of `drawbot.DEFAULTS` (ids, lim, site_bounds, relocate_bounds, track, pen_L, pen_up, press, step, speed, gripper_open, gripper_stall, gripper_margin, grasp_tool_mm) |
| `kin_model.json` | `{"signs":[3],"x":[12],"rms":f,"n":int}` + optional `mid`, `dropped`, `median`, `max`, `base`, `grasp_tool` (m, gripper frame) |
| `calib_samples.jsonl` | one sample per line: `{"q":[ID1..ID5 pos],"target":[4],"cam":[x,y,h] plane mm,"area","px","t","cam_xyz":[3] camera mm}` |

The file names match the scratch session's, so `DRAWBOT_STATE=<scratchpad>` works on those files
directly. Outputs also go here: `layer_<marker>.log`, `after_<marker>.jpg`, `relocate_out.json`.

## CLI

```sh
P=.venv/bin/python
$P drawbot_cli.py status                          # joints, loads, temps, last teleop result
$P drawbot_cli.py snap --out captures --crop 400 250 800 500 --scale 3   # Gemini + wrist + enlarged crop
$P drawbot_cli.py blobs [--colors orange green pink yellow blue]
$P drawbot_cli.py track [--mode tape] [--n 5 --follow]
$P drawbot_cli.py frame [--fit-paper [--save]]
$P drawbot_cli.py move 1=1900 4=2100 [--speed 8] --yes
$P drawbot_cli.py level [--down] --yes            # ID4 = level4 / down4 of current ID2, ID3
$P drawbot_cli.py pan 10 --yes                    # point at plane angle 10 deg about BASE
$P drawbot_cli.py jacobian [--steps 35 40 40] --yes
$P drawbot_cli.py probe-contact [--step 1.5] --yes
$P drawbot_cli.py relocate 20 60 45 --yes         # barrel to plane x y h in the air
$P drawbot_cli.py draw-layer orange [--start K] [--dry] --yes
$P drawbot_cli.py park --yes
# depth-calibrated kinematics / manipulation
$P drawbot_cli.py calib-collect [--start-anchor K] [--roll 1698] --yes   # append calib_samples.jsonl
$P drawbot_cli.py kin-fit [--signs] [--thresh 15] [--no-save]           # -> kin_model.json
$P drawbot_cli.py fk [ID1 ID2 ID3 ID4 ID5]        # default: measured joints; TCP, marker point, grasp center
$P drawbot_cli.py ik X Y H [--down | --axis AX AY AZ] [--tool [grasp|tcp|X Y Z]] [--roll R]  # exit 2 = unreachable
$P drawbot_cli.py goto X Y H [--down | --axis ...] [--tool ...] --yes   # IK + sag correction
$P drawbot_cli.py heightmap X0 X1 Y0 Y1 [--cell 6]
$P drawbot_cli.py markers [--all] [--region X0 X1 Y0 Y1]
$P drawbot_cli.py poke X0 Y0 X1 Y1 H [--clear 55] --yes
$P drawbot_cli.py flip [--to ROLL] --yes          # 180-deg wrist roll with load/lag guard
$P drawbot_cli.py pick green [--dry]              # plan; --yes executes (older planner)
$P drawbot_cli.py find pink | locate X Y          # marker pose by shape (read-only)
$P drawbot_cli.py pick2 pink | pick2 X Y AX AY [--dry] --yes   # field-proven pick (see "Picking markers")
$P drawbot_cli.py slide COLOR [--dist 80] --yes  # drag along its axis toward the robot
```

`--tool` with no value selects the fitted marker point. `grasp` selects the grasp center, `tcp` the fixed-jaw
tip, and `X Y Z` a point in mm in the gripper frame.

Without `--yes`, each moving command prints the current state and the plan. Typical layer:
put the marker in the gripper, run `track` to check it is seen, then `relocate` to above the site,
`jacobian --yes`, `probe-contact --yes`, and `draw-layer <marker> --yes`. Swap the marker and repeat.
