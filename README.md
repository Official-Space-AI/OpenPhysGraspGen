# OpenPhysGraspGen

[![CI](https://github.com/Official-Space-AI/OpenPhysGraspGen/actions/workflows/ci.yml/badge.svg)](https://github.com/Official-Space-AI/OpenPhysGraspGen/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-public%20beta-orange)

[한국어 README](README.ko.md)

OpenPhysGraspGen generates physics-labelled grasp data for **deformable objects**.
It calibrates a material model from measurements, transfers the calibrated
parameters to the simulator you generate data with, and records contact force and
internal stress for every grasp so that a policy can learn how hard it is
squeezing, not only where it is grasping.

It is the open part of SpaceAI's PhysGraspGen project. The library is pure Python
and NumPy, with sensor layers that read contact buffers from
[NVIDIA Newton](https://github.com/newton-physics/newton) (VBD solver).

> **Status: public beta (`0.1.0b0`).** APIs can still change between minor
> versions. See the [roadmap](#roadmap) for what is coming in the release
> candidate.

## What is included

| Module | What it does |
|---|---|
| `constitutive` | Hyperelastic laws (Yeoh, Mooney–Rivlin, three-term Ogden): strain energy, Cauchy stress, von Mises stress |
| `formulations`, `generation` | Calibrate material parameters against measured response, transfer them from a calibration model to a production model, build training samples with fracture labels |
| `estimation`, `confidence` | Joint estimator of material and internal stress from robot observations, plus a confidence gate that limits action scale when the estimate is unreliable |
| `sensors` | `DeformableContactSensor` and `SensorRecorder`: per-contact and net gripper/ground force labels from a Newton simulation frame |
| `records` | Typed records shared by the stages above |

The high-precision solver core, customer data and internal tooling are not part of
this repository. See [Scope of the open release](#scope-of-the-open-release).

## Install

Python 3.10 or newer. The package is `openphysgraspgen` and is published on PyPI from
`1.0.0rc1`:

```bash
pip install openphysgraspgen
```

Until that release, install from GitHub:

```bash
pip install "git+https://github.com/Official-Space-AI/OpenPhysGraspGen.git"
```

For development:

```bash
git clone https://github.com/Official-Space-AI/OpenPhysGraspGen.git
cd OpenPhysGraspGen
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

The same checks run in CI on every pull request, with tests on Python 3.10 to 3.14.

## Quick start

Run the end-to-end reference pipeline (calibration → transfer → data generation →
estimation → confidence gate) on a small synthetic problem:

```bash
openphysgraspgen --fracture-threshold 0.25 --confidence-threshold 0.5
```

It prints a JSON summary with the calibrated and transferred parameters, the number
of training and fracture samples, the estimate, and whether the confidence gate
allowed it.

Score the PG-T1 benchmark from a dataset in the [public format](docs/dataset/format.md),
with the `dataset` extra installed (`pip install "openphysgraspgen[dataset]"`):

```bash
openphysgraspgen benchmark pg-t1 path/to/dataset --json pg-t1.json
```

It prints one table per task and writes the report to `pg-t1.json`, with one record
per episode in `pg-t1.episodes.jsonl` beside it. The scoring rules are in
[docs/benchmark/pg-t1.md](docs/benchmark/pg-t1.md).

Check a dataset against the four items of the dataset test (scale, metadata,
consistency, diversity) with the script it ships, from the dataset's root, the folder
that holds `metadata.yaml`:

```bash
python -I tools/check_dataset.py . --seed <the tester's seed>
```

It needs only Python 3.10 or newer with NumPy, PyArrow and PyYAML, and nothing from
this package. The options, the acceptance values and the report are under
[The check](docs/dataset/format.md#the-check).

Evaluate a constitutive law directly:

```python
import numpy as np
from openphysgraspgen import YeohParams, cauchy_stress, von_mises_stress

silicone = YeohParams(c10=0.12e6, c20=0.01e6, c30=0.0, nu=0.45)
F = np.diag([1.2, 0.92, 0.92])  # uniaxial stretch
sigma = cauchy_stress(F, silicone)  # 3x3 Cauchy stress [Pa]
print(von_mises_stress(sigma))  # about 1.4e5 Pa
```

Record contact force labels from a Newton simulation:

```python
from openphysgraspgen import DeformableContactSensor, SensorRecorder, SimFrame

recorder = SensorRecorder([DeformableContactSensor(contact_provenance="newton-vbd")])
frame = SimFrame(
    model=model,
    particle_q=state.particle_q,
    contacts=contacts,
    body_q=state.body_q,
    n_robot_shapes=n_gripper_shapes,
)
labels = recorder.record_frame(frame)  # {"force": ForceLabel(...)}
```

## Roadmap

Releases follow the plan of the funded project.

| Release | Target | Contents |
|---|---|---|
| Beta `0.1.0b0` | Aug 2026 | Constitutive laws, contact sensor, calibration/transfer pipeline (published 2026-08-25) |
| `1.0.0rc1` | Oct 2026 | Grasp data recorder, PyPI package, public dataset on Hugging Face (1,000,000 frames) with its check script, PG-T1 benchmark reproduction from the dataset, Docker image, CI and SBOM on every release |
| `1.0.0` | Nov 2026 | Benchmarks, tutorials, license verification |

## Scope of the open release

PhysGraspGen is developed as open core.

| Part | Where | License |
|---|---|---|
| Library (this repository) | GitHub | Apache-2.0 |
| Grasp dataset | Hugging Face Hub (with `1.0.0rc1`) | CC-BY-NC-4.0 |
| Benchmarks and reproduction scripts | This repository (PG-T1 with `1.0.0rc1`, the rest with `1.0.0`) | Apache-2.0 |
| High-precision solver core | Not published | Commercial |

## How development works

Day-to-day development of PhysGraspGen happens in a private repository that also
holds customer-specific code. This repository receives the general-purpose parts as
reviewed pull requests and tagged releases, so its commit history is shorter than
the project's. From now on, work on the open library is planned and reviewed here
in public: every change starts as an issue and lands through a reviewed pull
request, and releases ship with release notes and an SBOM.

## Community and support

- Questions and ideas: [GitHub Discussions](https://github.com/Official-Space-AI/OpenPhysGraspGen/discussions)
- Bugs: [GitHub Issues](https://github.com/Official-Space-AI/OpenPhysGraspGen/issues)
- Where to ask what, and response times: [COMMUNITY.md](COMMUNITY.md)
- Security vulnerabilities: [SECURITY.md](SECURITY.md) (please do not open a public issue)

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/Official-Space-AI/OpenPhysGraspGen/labels/good%20first%20issue)
label. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Who
decides what, and how releases are approved, is in [GOVERNANCE.md](GOVERNANCE.md).

## License

Code is released under the [Apache License 2.0](LICENSE). The dataset is released
under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
