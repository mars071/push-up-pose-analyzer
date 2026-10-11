# Push-Up Form Checker — Rule-Based Scoring Design

**Status:** Design only. No implementation yet.
**Input:** Side-view video, person facing with **head to the left, feet to the right** in the frame, moving up and down (plank orientation).
**Source data:** MediaPipe Pose Landmarker per-frame output (33 landmarks, normalized `x`, `y`, `z`, `visibility`).

---

## 1. Purpose and scope

We score push-up form with a **deterministic, rule-based system** (no neural network). Every index is a closed-form function of landmark coordinates, so each number in the final report is traceable to a specific joint angle or distance.

This document defines:

1. Which landmarks feed each index.
2. The exact formula for each index.
3. The "ideal" value and the scoring function that maps a measurement to a 0–1 score.
4. How per-rep scores aggregate into an overall score.
5. The setup-validation gate that rejects non-side-view input.

**Hard scope limit:** elbow *flare* (shoulder abduction in the frontal plane) is **not measurable from a pure side view** — see §9. It is deliberately excluded from scoring and flagged as a limitation.

---

## 2. Landmark conventions

MediaPipe Pose emits 33 landmarks. Indices used here:

| Joint | Left index | Right index | Averaged symbol |
|---|---|---|---|
| ear (head) | 7 (`left_ear`) | 8 (`right_ear`) | `Ear` |
| shoulder | 11 | 12 | `S` |
| elbow | 13 | 14 | `El` |
| wrist | 15 | 16 | `W` |
| hip | 23 | 24 | `H` |
| knee | 25 | 26 | `K` |
| ankle | 27 | 28 | `A` |

### Side-view averaging

In a true side view the left and right joints project to nearly the same `(x, y)`. We collapse each left/right pair into a single 2D point using a **visibility-weighted mean** (robust to one arm/leg being occluded):

```
J = ( P_left * v_left + P_right * v_right ) / ( v_left + v_right )
```

where `P` is the `(x, y)` position and `v` is `visibility`. If `v_left + v_right` is below a floor (e.g. `< 0.2`), that joint is treated as *missing* for that frame.

### Ear (head position)

Per the earlier design decision, the head is represented by **only the ear**, never the nose/eyes/mouth. In profile both ears project to roughly the same `(x, y)`, so:

```
Ear = P7 if v7 >= v8 else P8
```

### Coordinate frame

- `x` ∈ [0,1], increasing **left → right** (head at small `x`, feet at large `x`).
- `y` ∈ [0,1], increasing **top → bottom** (the floor is toward large `y`).
- `z` is **ignored everywhere** — it is rough relative depth and too noisy for joint angles.

Only `x` and `y` are used.

---

## 3. Primitives

All formulas operate on 2D points in normalized coordinates. Angles are therefore **camera-tilt and scale invariant**; distances are normalized by a body segment (see `L_torso`) to be scale invariant.

### 3.1 Distance

```
dist(A, B) = sqrt( (x_A - x_B)^2 + (y_A - y_B)^2 )
```

### 3.2 Angle at vertex B (between rays B→A and B→C), degrees ∈ [0, 180]

```
u = (x_A - x_B, y_A - y_B)
w = (x_C - x_B, y_C - y_B)
angle(A, B, C) = acos( dot(u, w) / (|u| * |w|) ) * 180 / pi
```

Guard: if `|u|` or `|w|` ≈ 0, the frame is skipped (degenerate joint).

### 3.3 Signed perpendicular distance of point P from the directed line A→B

```
d = B - A
signed_dist(P, A, B) = cross(d, P - A) / |d|
cross(d, q) = d.x * q.y - d.y * q.x
```

With the y-down frame this gives a useful sign convention:

- For the body line `S → A` (shoulder → ankle): **positive = the point is below the line (toward the floor)**.
- For a horizontal plank this means **positive = hip sag / head droop**, **negative = pike / craning**.

### 3.4 Normalization references (scale)

```
L_torso = median over all frames of dist(S, H)     // torso length, ~constant
L_arm   = median over all frames of dist(S, El)    // upper-arm length, ~constant
```

`L_torso` normalizes all position offsets; `L_arm` normalizes depth of the shoulder drop. Using the median (not a single frame) makes them robust to one bad frame.

---

## 4. Rep segmentation

Range of motion, stability, and tempo are per-rep quantities, so we first split the video into reps using the **elbow angle** as the phase signal.

### 4.1 Elbow angle time series

```
theta_elbow(t) = angle(S(t), El(t), W(t))
```

- Top of a rep: arms extended → `theta_elbow` ≈ 170–180°.
- Bottom of a rep: arms flexed → `theta_elbow` ≈ 90° (or less).

Smooth `theta_elbow(t)` with a small moving average (window ≈ 3–5 frames) before peak detection.

### 4.2 Peak / valley detection

On the smoothed series, find:

- **Top** frames: local maxima with `theta_elbow > 155°`.
- **Bottom** frames: local minima with `theta_elbow < 115°`.

### 4.3 Rep definition

A full rep is **top → bottom → top**:

```
rep_i:  top_start_i  ->  bottom_i  ->  top_end_i
        down phase          up phase
```

- `d_down_i = (bottom_i - top_start_i) * INTERVAL`  (seconds)
- `d_up_i   = (top_end_i - bottom_i)   * INTERVAL`
- `d_rep_i  = d_down_i + d_up_i`

**Edge handling:**

- Drop incomplete leading/trailing half-reps (video starts mid-rep, ends mid-rep).
- If fewer than one full rep is found, fall back to a single "static hold" analysis using the global elbow-angle minimum as the pseudo-bottom (report it as such).
- Require a minimum fraction of frames per rep to have valid landmarks; otherwise mark the rep `unrated`.

---

## 5. Indices and scoring functions

Each index produces a raw measurement and a **0–1 sub-score**. Scoring uses piecewise-linear interpolation; define

```
clamp(v, lo, hi) = max(lo, min(hi, v))
```

A score of `1.0` = ideal, `0.0` = clearly bad. **Every numeric threshold below is a default to be calibrated** against labeled video (§11).

---

### Group A — Depth / Range of motion

**A1. Bottom elbow angle** (how deep the push-up goes)

```
theta_bottom_i = elbow angle at frame bottom_i
score_A1 = clamp( (160 - theta_bottom_i) / (160 - 100), 0, 1 )
```

- `theta_bottom ≤ 100°` → 1.0 (full depth)
- `theta_bottom ≥ 160°` → 0.0 (essentially no bend)
- Target ≈ 90°.

**A2. Elbow range of motion** (total excursion of the joint)

```
theta_top_i   = elbow angle at frame top_start_i
ROM_i         = theta_top_i - theta_bottom_i
score_A2 = clamp( (ROM_i - 30) / (75 - 30), 0, 1 )
```

- `ROM ≥ 75°` → 1.0
- `ROM ≤ 30°` → 0.0

**A3. Normalized shoulder drop** (secondary depth check, catches cheating that A1/A2 miss)

```
delta_S_i       = y_S(bottom_i) - y_S(top_start_i)   // positive = shoulder went down
depth_ratio_i   = delta_S_i / L_arm
score_A3 = clamp( (depth_ratio_i - 0.2) / (0.6 - 0.2), 0, 1 )
```

- `depth_ratio ≥ 0.6` → 1.0
- `depth_ratio ≤ 0.2` → 0.0

`A1` and `A2` are the authoritative depth metrics (tilt/scale invariant); `A3` is a consistency check that is only valid when the camera is roughly level with the body.

---

### Group B — Body alignment (the "straight line")

**B1. Hip angle** (sag / pike)

```
theta_hip_i = angle(S(bottom_i), H(bottom_i), A(bottom_i))
dev_B1_i    = |180 - theta_hip_i|
score_B1 = clamp( (30 - dev_B1_i) / (30 - 8), 0, 1 )
```

- `dev ≤ 8°` → 1.0 (straight)
- `dev ≥ 30°` → 0.0

Measured at the bottom frame (where sag/pike is worst). To distinguish *sag* from *pike* for the report, use the signed distance:

```
sag_sign_i = signed_dist(H(bottom_i), S(bottom_i), A(bottom_i))
```

- `sag_sign > 0` → **hip sag** (hip dropped toward floor)
- `sag_sign < 0` → **pike** (hip raised)

Both reduce `score_B1`; the sign only labels the report.

**B2. Knee angle** (legs straight)

```
theta_knee_i = angle(H(bottom_i), K(bottom_i), A(bottom_i))
dev_B2_i     = |180 - theta_knee_i|
score_B2 = clamp( (40 - dev_B2_i) / (40 - 10), 0, 1 )
```

- `dev ≤ 10°` → 1.0
- `dev ≥ 40°` → 0.0 (bent knees = cheating the lever)

**B3. Head / neck neutrality** (ear only — no nose/eyes/mouth)

```
head_offset_i = signed_dist(Ear(bottom_i), S(bottom_i), H(bottom_i)) / L_torso
score_B3 = clamp( (0.20 - |head_offset_i|) / (0.20 - 0.05), 0, 1 )
```

- `|head_offset| ≤ 0.05` → 1.0 (ear on the shoulder–hip line)
- `|head_offset| ≥ 0.20` → 0.0

`head_offset > 0` → head drooping toward floor (most common error); `< 0` → craning up.

---

### Group C — Arm & hand mechanics

**C1. Hand placement** (wrist under the shoulder)

```
hand_offset_i = (x_W(bottom_i) - x_S(bottom_i)) / L_torso
score_C1 = clamp( (0.30 - |hand_offset_i|) / (0.30 - 0.10), 0, 1 )
```

- `|hand_offset| ≤ 0.10` → 1.0 (hands under shoulders)
- `|hand_offset| ≥ 0.30` → 0.0

Sign (head-left convention, `-x` is "toward the head"):

- `hand_offset < 0` → hands too far **forward** (under the face)
- `hand_offset > 0` → hands too far **back** (under the chest/stomach)

**C2. Elbow flare — NOT scored (side-view limitation).** See §9.

---

### Group D — Stability

**D1. Body-line stability** (hip wobble through the rep)

```
hip_dev_i(t) = signed_dist(H(t), S(t), A(t)) / L_torso,  for t in rep_i
stab_i       = stddev over t of hip_dev_i(t)
score_D1 = clamp( (0.08 - stab_i) / (0.08 - 0.02), 0, 1 )
```

- `stab ≤ 0.02` → 1.0 (rock steady)
- `stab ≥ 0.08` → 0.0

**D2. Longitudinal drift** (body sliding forward/back during the set)

```
drift_i = | x_A(top_end_i) - x_A(top_start_i) | / L_torso
score_D2 = clamp( (0.15 - drift_i) / (0.15 - 0.03), 0, 1 )
```

- `drift ≤ 0.03` → 1.0
- `drift ≥ 0.15` → 0.0

**D3. Tempo consistency** (rhythm across the whole set, not per-rep)

```
CV_rep = stddev(d_rep) / mean(d_rep)   over all reps
score_D3 = clamp( (0.4 - CV_rep) / (0.4 - 0.1), 0, 1 )
```

- `CV ≤ 0.1` → 1.0
- `CV ≥ 0.4` → 0.0

---

### Group E — Tempo / rhythm

**E1. Rep duration** (controlled pace)

```
score_E1 = clamp( (d_rep_i - 0.8) / (1.5 - 0.8), 0, 1 ), capped at 1.0
```

- `d_rep ≤ 0.8 s` → 0.0 (too fast = using momentum)
- `d_rep ≥ 1.5 s` → 1.0

**E2. Down/up ratio** (descent not faster than ascent)

```
ratio_i = d_down_i / d_up_i
score_E2 = clamp( (ratio_i - 0.6) / (1.5 - 0.6), 0, 1 )
```

- `ratio ≥ 1.5` → 1.0 (controlled 2:1-ish descent)
- `ratio ≤ 0.6` → 0.0 (free-fall on the way down)

---

## 6. Aggregation

### Per-rep score

Each group is the mean of its sub-scores; the group scores are weighted:

| Group | Weight | Sub-indices |
|---|---|---|
| Depth / ROM | 0.30 | A1, A2, A3 |
| Alignment | 0.35 | B1, B2, B3 |
| Hand placement | 0.10 | C1 |
| Stability | 0.10 | D1, D2 (D3 is set-level) |
| Tempo | 0.15 | E1, E2 |

```
rep_score_i = 100 * ( 0.30 * mean(A1,A2,A3)
                    + 0.35 * mean(B1,B2,B3)
                    + 0.10 * C1
                    + 0.10 * mean(D1,D2)
                    + 0.15 * mean(E1,E2) )
```

### Overall video score

```
overall_score = mean over reps of rep_score_i
```

Weights are defaults; they should be tunable.

### Qualitative flags

For any sub-index with score `< 0.5`, emit a human-readable flag, e.g.:

- "Shallow depth" (A1/A2 low)
- "Hip sag in rep 3" / "Pike in reps 1–2" (B1 low, sign from `sag_sign`)
- "Knees bending" (B2 low)
- "Head drooping" (B3 low, `head_offset > 0`)
- "Hands under face" / "Hands under stomach" (C1 low, sign)
- "Unstable hips" (D1 low)
- "Body sliding" (D2 low)
- "Inconsistent rhythm" (D3 low)
- "Too fast" (E1 low)
- "Dropping on the way down" (E2 low)

---

## 7. Report structure

Proposed output (per video):

1. **Overall score** (0–100) + a one-line summary.
2. **Per-rep table**: rep number, rep duration, down/up durations, bottom elbow angle, ROM, hip angle, knee angle, head offset, per-rep score.
3. **Set-level stats**: rep count, mean/CV of rep duration, mean alignment, mean stability.
4. **Flags**: the qualitative list above, ordered by severity.
5. **Raw measurements**: every index value for every rep (traceability).

---

## 8. Setup-validation gate (run before scoring)

Reject or warn when the video is not a usable side-view push-up:

1. **Profile check (robust):** shoulder width must be small relative to torso length.

```
profile_ratio = dist(P11, P12) / dist(S, H)
```

- `profile_ratio < 0.30` → true side view (shoulders overlap). **Accept.**
- `0.30 ≤ profile_ratio < 0.60` → 3/4 view. **Warn** (angles degrade).
- `≥ 0.60` → front view. **Reject.**

2. **Horizontal plank check:** the torso should be mostly horizontal.

```
| x_S - x_H | > | y_S - y_H |
```

(i.e. shoulder–hip spans more x than y). If the body is upright (vertical), reject.

3. **Orientation note:** the sign conventions in §3.3 and §5 assume **head-left**. If the person is filmed mirrored (head-right), all *magnitudes* are unchanged; only the *directional labels* ("forward/back", "sag/pike", "droop/crane") flip. Detect with `x_nose < x_hip` (head-left) vs `x_nose > x_hip` (head-right) and flip labels accordingly. This is a one-time setup check — the head is **not** used for any per-frame scoring.

4. **Frame validity:** a frame is usable only if `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle` all have `visibility > 0.5`. A rep is `unrated` if fewer than ~70% of its frames are usable.

---

## 9. Limitations and caveats

1. **Elbow flare (frontal-plane shoulder abduction) is not measurable from side view.** Flare moves the elbow mostly in the `z` (depth) direction, and MediaPipe `z` is too noisy. Measuring it requires a front or 3/4 view. It is excluded from scoring; if it matters, add a separate front-view pass.
2. **`z` is ignored globally** — angles are computed from `(x, y)` only.
3. **Camera level assumption:** angle-based indices (A1, A2, B1–B3, C1, E2) are tilt-invariant; the two `y`-based depth/position indices (A3, parts of D) assume the camera is roughly level with the body. Strong tilt biases A3.
4. **Left/right averaging assumes true side view.** At 3/4 view the averaging still works but accuracy degrades.
5. **No ground-truth "floor"** — depth is inferred from the body's own geometry (elbow angle, shoulder drop), never from a detected floor line.
6. **One person, one pose** (`numPoses = 1`). Multi-person frames are not handled.
7. **Occlusion at the bottom** — wrists/hands may drop in `visibility` at full depth; the visibility floor in §8.4 handles this.
8. **All thresholds are defaults.** They reflect common coaching standards but MUST be calibrated (see §11).

---

## 10. Open design questions

- Should `overall_score` use the **mean** or the **minimum** rep score (min = stricter, punishes one bad rep)?
- Is a 2:1 down/up ratio the right target, or should we just penalize `ratio < 1`?
- Should a "static hold" (no reps detected) get its own scoring path, or be reported as "no reps found" only?
- For A3/D depth metrics, do we need a camera-level self-calibration step (e.g. infer the body-plane angle from the shoulder–ankle line)?

---

## 11. Calibration plan

Before trusting any number:

1. Collect ~10–20 side-view push-up videos spanning good and bad form (already have `pushup_video.mp4` as a start).
2. Run the landmark export (already built) and compute every raw index above.
3. Plot each index's distribution vs. manual good/bad labels; adjust the `score = clamp((hi - x)/(hi - lo))` breakpoints so "good" frames land ≥ 0.8 and "bad" frames land ≤ 0.3.
4. Lock the weights only after the sub-score distributions look sensible.

---

## Appendix — Landmark index reference (MediaPipe Pose)

| Index | Name | | Index | Name |
|---|---|---|---|---|
| 0 | nose | | 17 | left_pinky |
| 1–3 | left eye (inner/center/outer) | | 18 | right_pinky |
| 4–6 | right eye (inner/center/outer) | | 19 | left_index |
| 7 | left_ear | | 20 | right_index |
| 8 | right_ear | | 21 | left_thumb |
| 9 | mouth_left | | 22 | right_thumb |
| 10 | mouth_right | | 23 | left_hip |
| 11 | left_shoulder | | 24 | right_hip |
| 12 | right_shoulder | | 25 | left_knee |
| 13 | left_elbow | | 26 | right_knee |
| 14 | right_elbow | | 27 | left_ankle |
| 15 | left_wrist | | 28 | right_ankle |
| 16 | right_wrist | | 29/30 | heels |
| | | | 31/32 | foot index |
