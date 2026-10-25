# PG-T1: lifted, held and placed

Status: scorer `pg-t1-v1`, for datasets in the format `physgraspgen-public` 2.0
([format](../dataset/format.md)).

PG-T1 asks whether a grasp lifts the object, holds it and puts it down. Each episode gets three
stages, `lifted`, `held` and `placed`, and each stage is true, false or not reached. They are
judged under two criteria read from the same recording:

| Criterion | Rise | For | Reading |
|---|---|---|---|
| C | 5 cm, with the lowest point 5 mm off the bench | 2 s | the criterion the project agreed to report |
| B | 12 cm | 4 s | the benchmark criterion of the public table |

Every verdict reads the recorded trajectory of the object and the robot. The configured schedule
(`frames_per_waypoint`) and each row's waypoint and phase are read only where the generator's own
judge reads them: whether a phase recorded every row it was scheduled for (which separates C's
`never_crossed` from `incomplete` and decides whether the put-down is judged), the attempt a retry
ended in, the length of the put-back's hold and the decidable windows. The same rows can therefore
score differently under two schedules. The schedule is part of each episode's metadata, so the
published table can be recomputed from the dataset alone, on a CPU. The scorer reads
`metadata.yaml` and each episode's trajectory file with its footer metadata; it never reads the
frames file or the videos.

## Running it

```console
pip install "openphysgraspgen[dataset]"
openphysgraspgen benchmark pg-t1 path/to/dataset --json pg-t1.json
```

The command prints the episodes it refused, each with its rule and why, and the trajectory files
the dataset does not list (the first 10 of each), then one table per task. It writes the report to
the `--json` path (default `pg-t1.json`) and one record per episode beside it as JSON lines
(`pg-t1.episodes.jsonl`).
`python -m openphysgraspgen benchmark pg-t1 ...` does the same. It exits with 0 when every listed
episode was scored and no check failed, 1 when an episode was refused or a trajectory file is not
listed or a check failed (the report is written in every case), and 2 when the dataset cannot be
read or the report cannot be written. The table is printed before the report is written, and the
directory of the `--json` path is created when it does not exist. The reference pipeline still runs
as `openphysgraspgen` without a subcommand.

From Python:

```python
from openphysgraspgen.benchmark import report_text, score_dataset

report, episodes = score_dataset("path/to/dataset")
print(report_text(report))
```

`score_episode(columns, metadata)` scores one episode from its trajectory columns (as
`read_trajectory` returns them) and its footer metadata, and raises `Refused` with the rule it
breaks. `task_block_problems(block, recorded_fps)` lists the problems of a task block, as a
`task_block` refusal names them, so an exporter can check its blocks before writing them; without
`recorded_fps` it does not compare `criterion.fps`.

## Inputs

### Trajectory columns

| Column | Read as |
|---|---|
| `sim_step` | the row order: row i is step i + 1 |
| `finite` | a row whose `finite` is false has no object values |
| `object.centroid`, `object.bbox_min`, `object.bbox_max` | the object's centroid, lowest corner and highest corner [m] |
| `robot.ee_pose` | the reached tool pose: position [m] and quaternion (x, y, z, w) |
| `robot.gripper_width_cmd_m` | the jaw width the waypoint commands [m] |
| `object.n_robot_contacts` | contacts between the object and the robot |
| `robot.waypoint`, `robot.phase` | the waypoint being flown and its phase |
| `force.ground_force_mag` | the force on the bench [N], one recorded fact only |

A value that is NaN reads as missing, and so do the object's values on a row whose `finite` is
false. The commanded pose (`robot.ee_pose_cmd`) is not read: every rule that follows the tool reads
where it was, not where it was sent.

### Episode metadata

From the footer: `format` and `format_version`, `episode_uid` and `object_type` (which must match
the episode's entry in `metadata.yaml`), `recorded_fps`, `n_steps`, `source_n_frames`, `label`
(`success`, `fracture`, `fail` or `unknown`; reported, never judged) and `task`.

### The task block

`task` holds what the scorer needs besides the trajectory. Its keys are these; "always" means the
key must be present, with a value that may be null where the type allows it. The scorer names every
key that is missing or malformed.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `layout` | `"pg-t1-task-1"` | always | the layout of this block |
| `source` | `"generation"` | always | a demonstration written by the data generator, the only kind PG-T1 scores |
| `name` | string | always | the task, one of the eight below |
| `phases` | list of strings | always | the planned phase of each waypoint, equal to the task's |
| `frames_per_waypoint` | list of ints >= 1 | always | the configured schedule in rows at `recorded_fps`: waypoint w runs entry min(w, length - 1); one entry, or at least one per waypoint. It is the schedule as configured, never the rows a waypoint actually ran |
| `stop_reason` | null, `"dropped"` or `"never_lifted"` | always | why the generator stopped the episode early. `"never_lifted"` does not imply a complete lift: a blow-up inside the lift waypoint is recorded with it |
| `criterion_id` | string or null | always | `agreement-v1`, `carry-place-v1` or `carry-place-10cm-v1` for the three tasks that have one, null for the other five |
| `floor_arm` | null, `"no_vision_pick"` or `"reflection"` | always | null for a demonstration; the scripted baseline otherwise (`no_vision_pick` only under `search_lift_hold_2s`, `reflection` only under a task that carries) |
| `judge` | mapping | always | the numbers the generator judged the episode with, below |
| `place_target` | mapping or null | a mapping under a task that carries, else null or absent | the mat: `x_m`, `y_m` and `half_extent_m` (an axis-aligned square in the world frame), optionally `centre_x_m` and `centre_y_m` |
| `keep_labels` | list of labels, or null | always | the labels the generation run kept; null when it kept every label |
| `keep_requires_lift` | bool | always | true when the run kept only the episodes whose own `lifted` verdict was true, that it stopped on a drop or, under `search_lift_hold_2s`, that had a row up under C; false when `keep_labels` is null, since the run applies this condition only together with a label list |
| `stages` | mapping of stage to true, false or null, or null | always | the generator's own stage verdicts, for the check |
| `agreement` | mapping or null | always | the generator's record of the C verdict, for the check |
| `place` | mapping or null | always | the generator's put-down record, for the check |
| `criterion` | mapping or null | optional | the generator's criterion record; where it gives `id`, `task`, `fps` (with `recorded_fps`), `internal.hold_s`, `internal.carry_m` or an `internal` number named like a `judge` key, each must agree with the episode, and a `placement` must name the same `floor_arm`, which it leaves out for a demonstration |
| `n_retries` | int >= 0 or null | optional | recoveries flown (reported) |
| `timing_scale` | mapping of phase to factor > 0, or null | optional | the per-episode timing factors of phases before the close (`search`, `approach`, `descend`); a block that scales a later phase is refused |

The keys of `judge`. A key that a task does not require may be absent or null; when it is given
anyway, it must still have its type.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `lift_success_m` | number > 0 | always | the generator's own lift threshold [m]; it anchors `placed`, while B keeps 12 cm |
| `approach_height_m` | number above `lift_success_m` | always | the commanded lift [m], checked only |
| `place_tol_m` | number > 0 | tasks that place | put-back: the return radius; put-down: the drag bound, and half of it the push bound [m] |
| `place_rest_radius_m` | number > 0 | tasks that place | how far the object may still wander at rest [m] |
| `place_max_descent_factor` | number > 0 | tasks that place | how much faster than the tool the object may come down |
| `place_release_drop_tol_m` | number > 0 | tasks that place | how far it may fall when the jaws open [m]; under a put-down also the height under which it counts as touching the bench |
| `carry_slip_tol_m` | number > 0 | tasks that carry | how far it may move in the jaws [m] |
| `set_down_speed_max_m_s` | number > 0 | tasks that carry | the fastest set-down [m/s] |
| `grip_force_open_deadband_m` | number > 0 | tasks that carry | how far past the held width the commanded width must go to open [m] |
| `open_ramp_frames` | int >= 1 | tasks that carry | for how many rows it must stay there |
| `opening_to_end_frames` | int >= 1 | tasks that carry | the rows judged after the opening |
| `rest_window_frames` | int >= 1 | tasks that carry | the rows at the end of that span the object must rest in |
| `place_error_tol_m` | number > 0, or null | a number under `search_lift_hold_carry_place_10cm`, else null or absent | how far the final centroid may lie from the mat centre [m] |

`judge` must hold the numbers the generator judged with, from the same sources, so that the
scorer's put-down is the generator's. JSON has no NaN, so a fact the generator could not measure is
written as null.

### Tasks

| Task | Phases, one per waypoint | Hold | Carries | `criterion_id` | `placed` |
|---|---|---|---|---|---|
| `pick` | approach, descend, close, lift | none | no | null | not planned |
| `lift_hold` | approach, descend, close, lift, hold | 4.0 s | no | null | not planned |
| `lift_hold_place` | as `lift_hold`, then place, place | 4.0 s | no | null | put back |
| `search_lift_hold` | search, search, then as `lift_hold` | 4.0 s | no | null | not planned |
| `lift_hold_retry` | as `lift_hold`, release, recover, then as `lift_hold` again | 4.0 s | no | null | not planned |
| `search_lift_hold_2s` | as `search_lift_hold` | 2.0 s | no | `agreement-v1` | not planned |
| `search_lift_hold_carry_place` | as `search_lift_hold`, then carry, place, place, retreat, retreat | 4.0 s | 0.30 m | `carry-place-v1` | put down |
| `search_lift_hold_carry_place_10cm` | as above | 3.0 s | 0.10 m | `carry-place-10cm-v1` | put down, with the centroid bound |

A stage a task does not plan is absent from its records and from every rate. It is never counted
as a failure.

### Acceptance

An episode that breaks one of these rules is refused: it is not scored, its rule is counted under
`refusals` and listed with why under `refused_episodes`, and the run is `incomplete`.

| Word | Rule |
|---|---|
| `listing` | `metadata.yaml` lists the episode's uid and object type as strings and its path under the dataset's root, no earlier entry lists that uid and object type, and the file holds that episode of that object type |
| `unreadable` | the trajectory file and its footer can be read |
| `format` | the footer says `physgraspgen-public` 2.0 |
| `label` | the footer's `label` is `success`, `fracture`, `fail` or `unknown` |
| `recorded_fps` | `recorded_fps` is an int >= 1 |
| `columns` | the trajectory has every column above |
| `rows` | the rows are steps 1, 2, ..., n: the first row is the first recorded frame |
| `n_steps` | the footer's and the listing's `n_steps` are n |
| `trajectory_cut` | `source_n_frames` is n: the trajectory holds every frame the generator recorded |
| `task_block` | the task block follows the tables above |
| `phase_row` | every row's waypoint is a waypoint of the plan and its phase is that waypoint's phase |

An episode that `metadata.yaml` lists more than once is scored at most once, from its first
entry. A trajectory file under `trajectories/` that `metadata.yaml` does not list is not scored and
is counted under `unlisted`.

## The rules

Rows are counted at `recorded_fps`. Arithmetic is float64 in the order written, and no tolerance
applies where none is written. With fps the recorded rate, B's hold is `round(4.0 * fps)` rows (240
at 60 fps), C's window `round(2.0 * fps)` rows (120) and the task's own hold `round(hold * fps)`
rows; `round` rounds half to even.

### Criterion B: 12 cm for 4 s

The rest height is the centroid height of the first row that has one. The crossing c is the first
row whose centroid has risen at least 0.12 m above it, with no tolerance.

- `lifted` is true when there is a crossing, false when there is none, and not reached when no row
  has a centroid.
- `held` is decided on rows c to c + 240: true when every row of that window that has a centroid
  stays at least 0.12 m up, false when one does not; a row without a centroid is skipped. When the
  last row that has a centroid comes before row c + 240, the window is cut short and `held` is not
  reached, or false when the generator stopped the episode on a drop. The truncation is decided
  first, so a window cut short is not reached even when a row inside it already dipped. There is
  no `held` without a crossing.

The window spans 241 rows, 4.0 s from the crossing to its last row.

### Criterion C: 5 cm for 2 s

C is the agreement criterion as it is implemented and as the generator records it. The rest heights
are the centroid height and the lowest-point height (`object.bbox_min` z) of the first row that has
both. A row is up when its centroid has risen at least 0.05 m and its lowest point at least 0.005 m,
each within 1e-9 m. The tie only absorbs float64 rounding: the heights must be the generator's
float64 values, since a float32 round trip can move a rise by a few 1e-9 m either way. k is the
first row that is up, and the window is rows k to k + 120 - 1; a row without both heights is
skipped.

- No row is up: false when the generator stopped the episode as never lifted or as dropped, false
  when the lift phase recorded every row it was scheduled for (`never_crossed`), and not reached
  otherwise (`incomplete`).
- A row of the window is not up: false, `dipped_in_window` when the centroid of some row of the
  window is under 5 cm, else `still_supported` (only lowest points under 5 mm). A recorded dip
  decides the window even when the window is cut short.
- Otherwise, when the last row that has both heights is at or after row k + 120 - 1: true.
- Otherwise the window is cut short: false when the generator stopped the episode on a drop
  (`stopped_dropped_in_window`), not reached otherwise (`window_truncated`).

`lifted` under C is true when some row is up, and otherwise the verdict above (false or not
reached); `held` is the verdict once a row is up. The two stages together restate the generator's
agreement verdict without loss.

This criterion supersedes the wording of the earlier benchmark specification, which judged C on the
centroid alone and excluded the lowest point. The underside term is there because an object gripped
near one end can pivot about the pads: its centroid rises 5 cm while its far end still rests on the
bench, which nobody watching would call lifted. Two consequences follow. C's window spans 120 rows
from the first row up (1.983 s at 60 fps) while B's spans 241 rows from the crossing. And C is not
looser than B in every episode: an object whose lowest point never leaves the bench is held under B
and not lifted under C.

### The attempt the episode ended in

Under `lift_hold_retry` an episode may fly two attempts. A phase is complete when the attempt the
episode ended in recorded, for that phase, at least as many rows with a centroid as the schedule
plans for its waypoints in that attempt; a waypoint the generator skipped still counts in the plan.
The attempt the episode ended in is the last one whose first waypoint index is at most the highest
waypoint index recorded; that first waypoint may itself have been skipped. The rules of B and C read
every row of the recording, so under a retry the first crossing decides, wherever it falls.

### Placed

`placed` is the task's own placement verdict. It is judged against the generator's own threshold
(`judge.lift_success_m`) and its own hold, and it is reported under a criterion only when that
criterion's `held` is true; when its `held` is not true, `placed` is not reached.

**The put-back** (`lift_hold_place`). Over the rows that have a centroid: the first row whose
centroid has risen `lift_success_m` over the first, and the hold's entry of the schedule of rows
after it, must all stay that high; otherwise `placed` is not reached. From the last row of that hold
on: the jaws must open (a commanded width more than 1 um wider than at that row), and over the rows
up to the opening the object must come down with the tool (over every 3-row span its descent speed
is at most `place_max_descent_factor` times the tool's, and the tool does descend somewhere). It
must then fall no more than `place_release_drop_tol_m` from the opening to the last row, rest (its
last 10 rows within `place_rest_radius_m` of the last) and end within `place_tol_m` of where it
rested at the start. Each failed condition gives its word: `never_released`,
`descent_not_with_tool`, `fell_on_release`, `not_at_rest`, `not_back_at_start`.

**The put-down** (the two tasks that carry). The internal hold is the task's own (4.0 s or 3.0 s)
from the crossing of `lift_success_m`, read as B reads its window; when it is not true, `placed` is
not reached. The jaws open on the first row of the first run of `open_ramp_frames` rows whose
commanded width exceeds the widest width of that hold by more than `grip_force_open_deadband_m`,
and they are fully open on the row after that run, `open_ramp_frames` rows after the opening.

The carried rows are the rows with a centroid from the crossing to the row before the opening, or to
the last row when the jaws never open. The carry is judged when the episode stopped on a drop, or
when the carry phase is complete and the jaws opened or the place phase is complete. It fails when
the drop stop came before any opening (`dropped_in_carry`), when the object slipped more than
`carry_slip_tol_m` in the jaws (`slipped_in_jaws`), or when a carried row whose lowest point is less
than `place_release_drop_tol_m` over its rest height lies more than `place_tol_m` in plan from where
the object is at the opening, or at the last carried row when the jaws never open (`dragged`). A
failed carry is a failed placement. The slip is read from the object's offset from the tool point in
the reached tool frame: a is its position along the jaw axis, the tool's y axis, and r its distance
from that axis. With a0 and r0 those of the first carried row with a tool pose, the slip is the
largest `hypot(a - a0, r - r0)` over the carried rows with one.

After a good carry, and when both place waypoints and both retreat waypoints are complete, the
judged span runs from the opening for `opening_to_end_frames` rows, clamped to the recording. With
the final row the last row up to the span's end that has the object's centroid and bounds, the
object is placed when all seven checks pass:

| Word | Fails when |
|---|---|
| `off_target` | the object's bounds at the final row leave the mat, or, under the 10 cm task, its centroid lies more than `place_error_tol_m` from the mat centre |
| `came_down_fast` | over some 3-row span from the crossing to the row the jaws are fully open, the object came down more than `place_release_drop_tol_m` beyond `place_max_descent_factor` times what the tool came down over the same span |
| `set_down_too_fast` | over the 3 rows before it first came within 1 mm of its rest height underneath, it came down faster than `set_down_speed_max_m_s` |
| `fell_on_release` | it fell more than `place_release_drop_tol_m` from the opening to the final row |
| `pushed_after_release` | it moved more than half of `place_tol_m` in plan from the opening to the final row |
| `still_touching` | a robot contact remains on a row of the span after the row the jaws are fully open, or the span has no row after it |
| `not_at_rest` | over the last `rest_window_frames` rows of the span it wanders more than `place_rest_radius_m` from its last position |

A jaw that never opens fails with `never_opened`. When the placement cannot be judged, `placed`
is not reached and the record says why: `internal_lift_not_met`, `internal_hold_not_met` or
`internal_hold_truncated` when the internal lift or hold did not happen in full; for the put-down
`carry_not_judgeable` or `put_down_incomplete`, and `no_final_frame`, `no_centroid_at_opening` or
`empty_rest_window`, which no generator recording reaches; for the put-back `no_usable_rows`,
`fewer_than_10_rows_after_hold`, `no_grip_width`, `ee_pose_missing` or `too_few_carried_rows`.
The record of a put-down also holds the facts the generator records for it, under the generator's
names, among them `set_down_bench_force_peak_n`, the highest bench force from the crossing to the
row the jaws are fully open.

### Decidable windows

For a task with a hold, let S be the scheduled rows before the first hold waypoint and L all
scheduled rows. B's `held` is decidable when S + 240 <= L and C's when S + 120 - 1 <= L (at 60
fps): the window, opened at the last row of the lift, ends inside the planned recording. Where it is
not decidable, the criterion's `held` and `placed` are left out of every rate and counted as
undecidable. Under `search_lift_hold_2s` B is never decidable: its recording ends 2 s after the
lift.

### Failure classes

Each episode falls in one class per criterion, tested in this order:

| Class | When | Words |
|---|---|---|
| `not_decidable` | the task holds, the window is not decidable and `lifted` is true | `window_longer_than_recording` |
| `success` | every planned stage is true | |
| `not_lifted` | `lifted` is false | C: `never_crossed`, `stopped_never_lifted` or `stopped_dropped`; B: `no_crossing` |
| `dropped` | lifted, then `held` is false, or the generator stopped the episode on a drop before the last planned stage was true | `stopped_dropped` after a drop stop; else C: `dipped_in_window` or `still_supported`; B: `below_threshold_in_window` |
| `not_placed` | held, and `placed` is false | every word of the placement verdict |
| `undecided` | anything else: a stage is not reached | C: `incomplete` or `window_truncated`; B: `no_usable_rows` or `window_truncated`; or why `placed` is not reached |

An episode whose window is not decidable is `not_lifted` when `lifted` is false and `undecided`
when it is not reached.

## The report

A scope is one task, over all its episodes, or one task and one object type. Rates are never pooled
over tasks, since the hold, the placement and the criterion differ by task; within a scope every
episode counts once, whatever its object type.

For each criterion and planned stage, over the scope's episodes (for `held` and `placed`, those
whose window is decidable):

- `of_attempted` is true over true plus false: when the episode got this far, how often the stage
  worked. It is null when no episode got there.
- `of_all` is true over all of them: the unconditional rate.
- Each comes with its Wilson 95 % interval, z = 1.959963984540054.
- The task score is the mean of the three `of_all` rates over the episodes with a decidable window.
  `task_score_n` counts those episodes, and the table prints it after the score ("0.694 over 12
  decidable"), since the `lifted` row counts every episode. Rates over all are used because their
  mean moves with every stage: with conditional rates, a grasp that lifts 10 of 100 objects and
  holds and places all 10 would score 0.70 instead of 0.10. It is null with its reason for a task
  that places nothing or has no episode with a decidable window, and it carries no interval: it is
  not a proportion. For a task that places nothing, `task_score_n` is null too.
- The headline is the last planned stage over all with its interval: `placed`, the fraction of
  episodes lifted, held and placed, for a task that places, else `held`, else `lifted`.
- The failure breakdown counts each class and each class's words; an episode with several words
  adds each.
- The population states which episodes the dataset holds: the labels the generation run kept and
  whether it kept only lifted episodes. A table over such a corpus is conditional on the lift: on
  every task but `search_lift_hold_2s`, with a generator threshold of at least 12 cm, its B
  `lifted` rate is 1.0 by construction. `keep_labels` is compared as a set and listed in the order
  `success`, `fracture`, `fail`, `unknown`; it and `keep_requires_lift` are `mixed` when the task's
  episodes disagree. The labels of the scored episodes are counted; damage itself is not part of
  PG-T1.

A cell no episode can decide is null and prints as "not planned", "not decidable" or "not
reached", never as 0. Floor-arm baselines are scored the same way and reported apart, under
`floor_arms`, never with the demonstrations.

The JSON report holds `benchmark`, `scorer` (`id`, `version`), `dataset`, `criteria`, `episodes`
(`listed`, `scored`, `refused`, `refusals`, `refused_episodes` with each refused episode's
`episode_uid` and `object_type` as `metadata.yaml` lists them (as text where they are not strings),
rule (`refused`) and why (`detail`), `unlisted`, `unlisted_paths`), `tasks`, `floor_arms`, `check`
and `status`. Under `tasks`, each task has its `criterion_id`, `placed_rule` (`put-back`, `put-down`
or null), `n_object_types`, `population`, `overall` and `by_object_type`; each table holds `n` and,
for C and B, the three stages (`planned`, `undecidable`, `true`, `false`, `not_reached`,
`of_attempted`, `of_all` and the two intervals), `task_score`, `task_score_n`,
`task_score_why_null`, `headline` and `failures` (`counts`, `of_all`, `reasons`). Each line of the
episodes file is one episode, ordered by object type, then episode uid: its verdicts, class and
words per criterion, C's agreement record, B's crossing and window end, the placement verdict with
its facts, and its check, or for a refused episode its rule and why.

The output is deterministic: no randomness, episodes ordered by object type, then episode uid, and
numbers written as the shortest float64 strings that read back exactly.

## The check against the verdicts each episode carries

Where the task block carries the generator's own verdicts, the scorer compares its verdicts with
them. Some must agree exactly; a difference means a scorer bug, an export bug or edited data, and
the run status becomes `check_failed`.

| Check | Applies when | Must hold |
|---|---|---|
| E1 | `agreement` is given, or `stages` has `agreement` | the agreement record, entry by entry, and the verdict |
| E2 | `stages.lifted` is true and `lift_success_m` is at least 0.12 | B's `lifted` is true |
| E3 | `search_lift_hold_carry_place`, `stages.lifted` true and `lift_success_m` 0.12 | `stages.held` equals B's `held` |
| E4 | `place` is given | every entry of the put-down record but `carried`, `placed` and the two particle facts (`final_yaw_change_rad`, `final_tilt_change_rad`) |
| E5 | a task that carries, `stages.held` true | `stages.carried` and `stages.placed` equal the put-down's |

The generator judges some stages by other rules, so other stored verdicts may differ. Each
difference must fall under exactly one of these explanations, tested in this order, or it fails the
check too:

| Code | Why |
|---|---|
| `other_lift_threshold` | `lift_success_m` is not 0.12, so the generator's `lifted` and `held` use another height than B |
| `other_hold_length` | the task holds 2.0 s or 3.0 s, so the generator's `held` is not B's 4.0 s |
| `retry_whole_recording` | the episode flew a recovery; the generator judged the final attempt, the scorer every row |
| `lift_not_reached` | the generator judges `lifted` only on a complete lift phase or a drop stop |
| `crossing_after_lift_phase` | the generator reads the last row of the lift phase, and the crossing came after it |
| `below_threshold_at_lift_end` | the object was below 12 cm at that row although it crossed before |
| `held_not_judged_by_generator` | the generator judges `held` only once its own `lifted` is true and the hold ran |
| `window_vs_hold_phase` | under the four older tasks with a hold, the generator's `held` covers the hold phase, not the window from the crossing |
| `generator_gate` | the generator judges the put-down only after its own `lifted` and `held` |
| `put_back_phase_rule` | the generator's `placed` for `lift_hold_place` follows its phase rule, not the put-back above |

A check key that is null is reported as unavailable, never as passed. The report's `check` gives
the `result` (`passed`, `failed`, or `unavailable` when the episodes carry nothing to compare), each
exact check's `applied` and `failed` counts, the stage verdicts `matched`, the `explained` codes,
the `unavailable` keys and the `failures`. `status` is `ok` when every listed episode was scored
and no check failed, `incomplete` when an episode was refused or a file is not listed, and
`check_failed`, which wins, when an exact check failed or a difference has no explanation.

## Limits

- A corpus of `search_lift_hold_2s`, the task most generation runs fly, decides C's `lifted` and
  `held` and B's `lifted` only. The full table needs a task that places.
- Under `search_lift_hold_carry_place_10cm` the plan keeps the object airborne for the task's 3.0 s,
  and B's 241-row window fits it by only a few rows, so B's `held` and `placed` there partly measure
  the schedule.
- `frames_per_waypoint` is the configured schedule. Under a timing perturbation the waypoints before
  the close fly more or fewer rows; the rows before the hold then move with the whole plan, so the
  decidable windows stay as computed, and the scorer refuses a block that scales a later phase.
- The two pose columns cannot be told apart from the data: an exporter must write the reached pose
  as `robot.ee_pose`.
