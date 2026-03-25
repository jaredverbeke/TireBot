# TreadLab Whitepaper

TreadLab (formerly TireBot) recommends the fastest tire + pressure setup for a specific event route, with a practical “tire issue risk” call.

## 1) What TireBot does

TreadLab recommends the fastest tire and pressure setup for a specific gravel event route.

The app combines:

- route surface segmentation (`road`, `cat1`, `cat2`, `cat3`, plus `above`)
- rolling resistance (CRR) test data by tire and surface
- rider inputs (weight, average speed)
- race dynamics weighting (extra emphasis on early race sectors)
- aerodynamic width effects (optionally scaled by GPX grade so steep downhills and steep climbs contribute less to the aero term)
- optional tire-mass effects over route climbing
- rough-surface impedance penalty (pressure/width/load dependent rolling penalty proxy)
- tire issue risk (Low / Medium / High), optionally learned from labeled examples

The output is a practical race-day recommendation:

- best tire
- front/rear pressure
- top ranked alternatives
- estimated power breakdown

---

## 2) Data inputs

### 2.1 Tire rolling resistance data

Source file in this project:

- `Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv`

For each tire, TireBot uses CRR values (or interpolation) on:

- Smooth Pavement
- Cat 1 Gravel
- Cat 2 Gravel
- Cat 3 Gravel

### 2.2 Route segment data

Each event has a segment CSV in `Routes/<Event>/`.

Supported fields:

- `segment_start`, `segment_end` (**miles** along course), `surface_type`, optional `selection_risk`, optional `technicality`
- or `distance_mi` (or legacy `distance_km` column storing **miles**) + `race_position`

TireBot converts miles to kilometers internally for rolling-resistance distance weighting.

GPX files use latitude/longitude (degrees). The app computes **horizontal track length in miles** from the trackpoint sequence for reference. GPX `<ele>` values are **meters** per the GPX spec; the UI shows elevation gain in feet and meters. Tire scoring still uses **segment CSV** mileage as the course model.

When a GPX file is present **and** trackpoints include elevation, TireBot also uses GPX to classify **which horizontal distance counts toward the aero width penalty** (see section 3.3): steep downhills and steep climbs are excluded from that distance fraction. Elevation is lightly smoothed along the track before grade is computed, to reduce GPS noise.

Each segment contributes differently depending on:

- distance
- surface type
- early-race weighting
- optional technicality / selection risk multipliers

#### Surface types (including “Above Category”)

Segment surface types are:

- `road`, `cat1`, `cat2`, `cat3`
- `above` = “Above Category”: very technical / unusually rough sections that don’t fit the standard categories.

At present:

- `above` is treated like `cat3` for CRR lookup (rolling-resistance scoring), but is tracked separately in route stats.
- if a route has **more than 2.0 miles** tagged `above`, the app restricts recommendations to **MTB tires only** (≥ 2.2" width).

### 2.3 Rider and race inputs

From app inputs:

- rider weight (kg)
- average speed (mph), on a single slider (used for rolling power everywhere, and as the speed in the aero formula where GPX-based distance weighting applies)
- early-race weighting multiplier

For pressure recommendations, TireBot uses a heuristic model that depends on tire width, rider weight, route roughness, and the selected average speed. A coarse internal “speed tier” is inferred from average speed (≥23 mph → “pro”, &lt;18 mph → “ride”, otherwise “amateur”) only as an input to that heuristic—not for rolling or aero math.

### 2.4 Tire mass data (optional)

File:

- `data/tire_mass_overrides.csv`

If available, exact tire mass is used for mass-penalty calculations.
If not, TireBot uses a width-based mass estimate.

---

## 3) Core calculations

## 3.1 Route-weighted rolling score

For each segment:

1. Determine segment CRR for that tire and surface.
2. Apply weighting:
   - early-race phase weighting
   - technicality multiplier
   - selection-risk multiplier
3. Multiply by segment distance.

Tire route score:

- `route_score = sum(segment_crr * distance * weight)`

Lower is better.

## 3.2 Rough-surface impedance penalty (CRR-equivalent)

Rough surfaces often reward *lower* pressures and *more support* (wider casings) beyond what a pure CRR table can capture.
TreadLab adds a tunable, pressure/width/load dependent penalty on rougher surfaces as an extra CRR-equivalent contribution:

- It estimates a segment-specific heuristic pressure from rider weight, tire width, inferred speed tier, and surface roughness.
- It computes a “stiffness proxy” from pressure × width ÷ rider weight.
- If stiffness is above a surface-specific target, it adds a penalty that grows nonlinearly with stiffness.

This term is meant to be **directionally correct** and **tunable**, not a definitive physics model.

## 3.3 Rolling resistance power (watts)

TireBot converts route score into estimated rolling power at the selected speed and weight:

- derive effective route CRR
- apply `P_rr = Crr * m * g * v`

Where:

- `m` is rider + bike system mass estimate
- `v` is average speed

## 3.4 Aero width penalty (watts)

TireBot adds an **aero width penalty** based on tire width relative to a baseline width, with magnitude that scales approximately with **speed cubed** (same functional form as before).

**Instantaneous-style penalty at the selected average speed**

For a given tire width `w` (mm), baseline width 40 mm, and rider-chosen average speed `v` (mph), TireBot computes a width penalty in watts using the app’s existing proxy (coefficient and reference speed are implementation details in `app.py`). Conceptually:

- narrower tires → lower penalty at the same speed
- wider tires → more aero penalty where this model applies

**Where on the course the penalty applies (GPX grade filter)**

Rolling resistance still uses the segment CSV and the same average speed over the **whole** course model.

For **aero only**, TireBot optionally scales that penalty by a **distance fraction** derived from the event GPX (when the file exists next to the route and trackpoints include `<ele>`):

1. Parse the GPX track as a sequence of points with latitude, longitude, and elevation (meters).
2. Apply a short **moving average** to the elevation series (e.g. five points) to damp GPS jitter before computing grade.
3. For each consecutive pair of points, compute **horizontal distance** (great-circle, km) and **grade** = Δelevation (m) / horizontal distance (m).
4. Count horizontal distance as **aero-eligible** only if grade is in a moderate band:
   - **Exclude** segments with grade **steeper downhill** than about **−1.2%** (coasting / aero not representative of pedaling).
   - **Exclude** segments with grade **steeper uphill** than about **+7%** (“large climb” where speed is low and this simple aero term is de-emphasized).
5. Let `f = (aero_eligible_horizontal_km) / (total_GPX_horizontal_km)`. If GPX or elevation is missing, **`f = 1`** (legacy behavior: aero applies as if the whole course were in the eligible band).

**Race-average aero contribution**

The **reported aero penalty** (watts) is:

`aero_penalty_watts = P_aero(width, v) * f`

So the rider’s **average speed slider** still sets how strong the aero proxy is at a given width; **`f`** reduces that term when much of the GPX track is steep descent or steep climbing. For context, the UI may express **`f`** as an approximate **effective aero mileage** = segment-course miles × **`f`** (segment CSV remains the distance authority for rolling resistance and overall route stats).

**Fallbacks**

- No GPX, unparseable track, or no per-point elevation → **`f = 1`**, same aero model as pre-GPX-filter behavior.

Result:

- narrower tires generally get less aero penalty
- wider tires can gain rolling advantages on rough surfaces but may lose more aero watts on the **eligible** portions of the course at the chosen average speed

## 3.5 Tire mass penalty (watts)

TireBot estimates energy required to lift tire mass over total route elevation gain, then converts that energy to average watts over estimated race time.

This term is usually small, but can matter for close calls on hillier routes.

## 3.6 Total resistance ranking

Final ranking target:

- `total_watts = rolling_watts + aero_penalty_watts + mass_penalty_watts`

Fastest tire = lowest total watts.

---

## 4) Pressure recommendation logic

TireBot pressure path:

1. Use a heuristic pressure model with tire width, rider weight, average speed, and route roughness.

Pressures are presented as race-day starting points, not absolute final values.

---

## 5) Tire issue risk and custom model training

TreadLab reports a **tire issue risk** label (**Low / Medium / High**) intended to capture puncture/handling risk for a given tire on a given route.

### 5.1 Two modes: heuristic and learned

- **Heuristic fallback**: if no training data exists, the app uses a simple width + rough-miles heuristic (including special handling for “Above Category” segments).
- **Learned model (recommended)**: if training labels exist, the app predicts risk from prior labeled examples.

### 5.2 What gets labeled

You can label risk for any route by assigning each tire a label:

- `Low`, `Medium`, or `High`

These labels are stored in:

- `data/risk_labels.csv`

Each row includes:

- `route` (path to the route segments CSV)
- surface-mile breakdown: `road_mi`, `cat1_mi`, `cat2_mi`, `cat3_mi`, `above_mi`, `total_mi`
- tire features: `tire_name`, `width_mm`, `tire_class`
- target: `label`

### 5.3 How predictions are made (kNN)

TreadLab uses a lightweight **k-nearest-neighbors** approach:

1. Build a feature vector for the current route (surface miles/mix) and the candidate tire (width + class).
2. Find the \(k\) most similar labeled examples.
3. Predict the label by a distance-weighted vote.

This makes the risk model **universal across routes** and improves automatically as you add more labeled events.

---

## 5) Why early-race weighting matters

In many gravel races, the first quarter is decisive.
If a setup causes you to lose the front group early, later gains may not matter.

TireBot allows higher weighting for early segments to reflect real race dynamics.

---

## 6) Practical interpretation

Use the recommendation as a decision aid, then validate on-bike:

- compare top 3 tires, not only #1
- check sensitivity to average speed and early weighting
- verify pressure on real terrain and weather

Small differences (< 1-2 W total) are often within modeling noise and should be treated as effectively tied.

---

## 7) Current limitations

- CRR coverage can be incomplete for some tires/surfaces (interpolation required)
- Surface categories are simplified into a few classes; `above` is a manual “super rough” flag
- Aero model is a practical proxy, not full CFD; GPX grade thresholds for excluding downhills/climbs are heuristics, not physics
- Tire mass may be estimated if measured data is missing
- Pressure model is heuristic and should be validated on real terrain and equipment
- Impedance penalty is a tunable proxy (helps capture “too stiff on rough” effects), not a calibrated suspension/terrain model
- Tire issue risk can be either heuristic or learned from labeled examples; more labels across more routes will improve generalization

---

## 8) Recommended next improvements

- add measured tire mass for all key tires
- include wind assumptions in aero model
- include temperature/wet-condition modifiers
- add confidence intervals around rankings
- expand the labeled risk dataset across more routes and weather conditions

---

## 9) Safety and responsibility

Always stay within tire and rim manufacturer pressure limits.
TireBot recommendations are informational and should be validated against equipment specs, rider handling preferences, and race-day conditions.

