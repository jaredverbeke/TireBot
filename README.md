# TireBot MVP

Initial CLI recommender for gravel race tire selection using:

- your tire rolling resistance CSV
- custom route segment CSV
- heavier weighting in the first quarter of race (front-group survival)
- no risk proxy (rolling-resistance-only scoring)

## Run

From the project root:

```bash
python3 recommend_tires.py
```

## Web app

Install dependency:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app lets you:

- pick an event from `Routes/`
- choose speed tier (`Pro`, `Amateur race`, `Ride`)
- enter rider weight in kg
- get tire recommendation and estimated front/rear pressure

## Wolf Tooth pressure baseline file

The app supports a data-backed pressure lookup from:

- `data/wolf_tooth_baseline.csv`

Expected columns:

- `bike_type,tire_setup,casing_type,terrain_class,weight_kg,width_mm,rear_psi,front_psi`

Notes:

- If the file has rows, pressure uses baseline lookup/interpolation.
- If the file is empty, the app uses the heuristic fallback model.
- Terrain is mapped from route roughness into classes `2`, `3`, `4` for lookup.

## Inputs

### Tire data

Default file:

- `Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv`

### Route segments

Default file:

- `Routes/example_route_segments.csv`

Columns:

- `segment_name` - free text
- `distance_km` - segment length
- `surface_type` - one of `road`, `cat1`, `cat2`, `cat3`
- `technicality` - usually 0.8 to 1.8
- `selection_risk` - race split potential, usually 0.8 to 1.8
- `race_position` - 0.0 start to 1.0 finish

Alternate route format also supported:

- `segment_name`
- `segment_start` (km from start)
- `segment_end` (km from start)
- `surface_type` (`road`, `cat1`, `cat2`, `cat3`)
- optional `technicality` (defaults to 1.0)
- optional `selection_risk` (defaults to 1.0)

When using `segment_start` and `segment_end`, the script automatically computes:

- `distance_km = segment_end - segment_start`
- `race_position` from segment midpoint across route distance

## Tune model behavior

```bash
python3 recommend_tires.py --early-boost 2.1 --top-n 10
```

- `--early-boost`: increases first-quarter weighting

## Notes

- Missing CRR values are interpolated when possible.
- Width is parsed from tire names for display only.
- This is a v1 ranking model (relative score), not a literal finish-time predictor.
- Pressure is an estimated starting point and should be validated in real ride conditions.
