# Spray-angle vertex calibration

## Summary

The hit-coordinate origin this repo uses for spray angle — `HOME_PLATE_X = 125.42`,
`HOME_PLATE_Y = 199.02` in `Model/feature_engineering.py` — was fitted so that
*distance* from the origin reconciles with Statcast's reported batted-ball
distance. It was never fitted as an **angular** vertex, and used as one it is
measurably wrong: about 5% of home runs come out beyond 45° of raw spray, i.e.
landing in foul territory, which cannot happen in a real game.

Two changes follow from that, and only the first has been made:

* **Done (Option A):** the spray chart and the displayed Pull/Center/Oppo table
  labels now measure angles from a calibrated rendering vertex,
  `(127.4, 215.0)`, defined in `Simulator/visualizations.py` as
  `calculate_spray_angle_calibrated`. Nothing under `Model/` imports it.
* **Proposed (Option B, this document):** moving the *model feature* pipeline to
  the calibrated vertex. That cannot be done in this repo alone and requires a
  retrain plus the normal bake-off gate. It may also correctly end in "no".

## Evidence

A home run cannot land foul, so `|raw spray| > 45°` is impossible by rule and
makes a clean, assumption-free error metric. Measured on this repo's own batted
ball data (`Data/batted_balls/batted_balls_{2024,2025}.parquet`):

| Season | Home runs | Foul-HR rate, current vertex | Max \|angle\| | Foul-HR rate, calibrated vertex | Max \|angle\| |
|---|---|---|---|---|---|
| 2024 | 5,545 | 4.905% | 52.1° | 0.234% | 48.2° |
| 2025 | 5,738 | 4.723% | 100.8° | 0.122% | 47.5° |

Distance is *not* able to discriminate between the two vertices. Reconciling
radial coordinate distance against Statcast `totalDistance` on 2024 air balls
(reported distance > 200 ft, n = 54,415), each vertex paired with the scale that
best fits it, lands in the same place: 16.6 ft mean absolute error for the
current vertex (best-fit 2.43 hc units/ft, close to the `COORD_TO_FT = 2.495`
this repo ships) versus 16.8 ft for the calibrated vertex (best-fit 2.18, close
to the 2.29 ft/unit documented by the GeomMLBStadiums project). A 0.2 ft gap on
a ~17 ft error floor decides nothing; the ~20× gap in impossible-foul home runs
does.

Impact on the feature itself, if the calibrated vertex were adopted: the
`spray_direction` label flips on 7.7–7.9% of batted balls, and the
pull/center/oppo shares move from 43.7 / 28.8 / 27.4 to 39.4 / 36.5 / 24.0
(2024; 2025 is nearly identical).

The residual ~0.2% of home runs still outside the foul lines under the
calibrated vertex is coordinate noise in the source data, not a sign that a
third vertex would be better.

## Every consumer of the vertex

### `Model/feature_engineering.py` — the frozen model path

| Location | Use |
|---|---|
| `HOME_PLATE_X` / `HOME_PLATE_Y` (~line 19) | The constants themselves; now carry a freeze comment. |
| `calculate_spray_angle` | Raw arctan2 angle from the vertex. |
| `_ray_polygon_distance` / `wall_distance_at_spray` | Ray-cast origin **and** direction; converts hc units to feet with `COORD_TO_FT = 2.495`. |
| `create_features_for_prediction` | Produces `spray_angle_adj`, `spray_angle_abs`, `spray_direction`, and the pull/oppo indicator + interaction features derived from them. |
| Import-time sanity check (bottom of file) | Uses the vertex to synthesize a center-field coordinate. |

### `Model/bbe_physics.py`

`nathan_spin` and `spin_aware_carry` take `spray_angle_adj` as an input, so the
vertex feeds six additional F6 numeric features: `total_spin_rpm`,
`sidespin_abs_rpm`, `carry_ft_spin`, `over_fence_margin_spin`,
`carry_ft_spin_temp`, `over_fence_margin_spin_temp`.

### `Simulator/visualizations.py` — display only, now calibrated

| Location | Use |
|---|---|
| `calculate_spray_angle_calibrated` | The calibrated vertex and angle function. |
| `get_spray_direction` | Pull/Center/Oppo labels shown in tables. |
| `spray_chart` | Plot angle. Radius comes from Statcast `totalDistance`, and the marker's expected-bases color comes from the model features, not from the plotted position — so the plot angle is pure rendering. |

### `Documentation/readme_image_generator.ipynb` — still on the frozen vertex

The notebook keeps its own inline copy of `HOME_PLATE_X/Y` and
`calculate_spray_angle` (it is a standalone illustration, not an import of the
library). It feeds the README's spray-angle validation figure, which plots raw
hit coordinates colored by adjusted spray angle. That figure demonstrates the
handedness normalization rather than foul-line placement, so it was left on the
frozen vertex; regenerating it is a separate, cosmetic decision.

### Wall polygons

`Model/data/mlb_park_walls.csv` is in hc units and was fitted consistently with
the current vertex and the 2.495 scale. Its generator lives outside this repo,
and the polygons are deliberately closed behind home plate. They are not
independent of the vertex.

### Tests

`tests/test_spray_vertex.py` pins both vertices: the model path must keep
producing angles measured from `(125.42, 199.02)`, and the rendering path must
keep using `(127.4, 215.0)`. Before that file existed, nothing guarded either.

## Option B — what a migration would require

**(a) The vertex, `COORD_TO_FT`, and the wall polygons move together or not at
all.** The wall geometry is stored in hc units and consumed by a ray cast that
starts at the vertex and converts to feet with `COORD_TO_FT`. Changing the
vertex alone silently re-aims every ray, shifting `wall_distance_ft`,
`over_fence_margin`, and both spin/temp over-fence features for every batted
ball. A migration means: regenerate the polygons against the calibrated vertex,
switch `COORD_TO_FT` to the matching 2.29 ft/unit scale, and validate the wall
distances against known park dimensions before touching anything downstream.

**(b) A retrain, evaluated as an F7 candidate under the normal bake-off gate.**
`spray_direction` is a trained categorical input, and the six spin/carry
features are trained numeric inputs. Shipping the calibrated vertex into the
feature pipeline changes those inputs for every row, so the shipped model
becomes invalid the moment the vertex moves — this is not a hot-swappable
constant. The retrain has to clear the usual bake-off criteria (log loss,
calibration error, and home-run tail behavior) against the shipped F6 model on
a season-forward split, and only wins if it is better on the tail as well as in
aggregate. Training happens outside this repo; nothing here can complete Option
B on its own.

**(c) "No material improvement" is a valid, complete outcome.** The mis-set
vertex is a systematic rotation, not random noise. Gradient-boosted trees fit on
the rotated angle can absorb a systematic rotation into their split thresholds,
so the model may already be compensating for it, and correcting the input may
buy nothing measurable. If the F7 bake-off comes back flat or worse, the correct
end state is exactly what ships today: the calibrated vertex stays
rendering-only, the model keeps the frozen vertex, and this document explains
why the two intentionally disagree. Nothing further is required.

## What is safe to change today

Safe: anything under `Simulator/visualizations.py` that consumes
`calculate_spray_angle_calibrated`, since it affects only what is drawn and
labeled.

Not safe without the full Option B cycle above: `HOME_PLATE_X`, `HOME_PLATE_Y`,
`COORD_TO_FT`, `calculate_spray_angle`, `adjust_spray_for_handedness`,
`categorize_spray_direction`, the produced `spray_direction` values, and
`Model/data/mlb_park_walls.csv`.
