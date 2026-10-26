# Open source inventory

The third-party components this repository uses, how it uses them and the evaluation
that covers each. Versions are as accepted on `main`; "resolves to" is what a fresh
install chose on 2026-10-26 on Python 3.12. The exact list for a release is the SBOM
attached to that release ([RELEASING.md](../../RELEASING.md)).

A component is added here when its [evaluation](evaluation.md) is approved; until then
the status says so.

## Runtime

Installed with the package. The `dataset` extra is optional
(`pip install "openphysgraspgen[dataset]"`).

| Component | Accepted | Resolves to | License | Used for | Evaluation | Status |
|---|---|---|---|---|---|---|
| NumPy | `>=1.24`; `>=1.26` with the `dataset` extra | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Array math throughout the package | [#52][52] (re-evaluation) | Approval pending |
| PyArrow | `>=17` (`dataset` extra) | 26.0.0 | Apache-2.0 | Reading and writing the dataset's Parquet files | [#45][45] | Approval pending; condition on the floor, see the [vulnerability log](vulnerability-log.md) |
| PyYAML | `>=6` (`dataset` extra) | 6.0.3 | MIT | `metadata.yaml` and the check criteria, loaded with `SafeLoader` only | [#46][46] | Approval pending |

## External programs

Installed by the user, run as a separate process, and not distributed by us.

| Component | Accepted | License | Used for | Evaluation | Status |
|---|---|---|---|---|---|
| FFmpeg | an `ffmpeg` with the `libsvtav1` encoder and an AV1 decoder | LGPL-2.1-or-later, or GPL-2.0-or-later as most builds are (review required) | Encoding and decoding the dataset's RGB videos | [#47][47] | Approval pending |

## Build, test and CI

Not installed with the package and not distributed.

| Component | Accepted | Resolves to | License | Used for | Evaluation | Status |
|---|---|---|---|---|---|---|
| setuptools | `>=77` (`[build-system]`) | 84.2.1 | MIT | Build backend | [#66][66] | Approval pending; condition on the floor |
| build | `==1.6.1` | 1.6.1 | MIT | Building the sdist and wheel in CI and releases | [#66][66] | Approval pending |
| twine | `==7.0.0` | 7.0.0 | Apache-2.0 | `twine check --strict` before the PyPI upload | [#66][66] | Approval pending |
| pytest | `>=8` (`dev` extra) | 9.1.1 | MIT | Tests | [#50][50] | Approval pending; condition on the floor |
| pytest-cov, coverage | `>=5` (`dev` extra) | 7.1.0, 7.17.0 | MIT, Apache-2.0 | Test coverage | [#50][50] | Approval pending |
| Ruff | `==0.17.1` (`dev` extra) | 0.17.1 | MIT | Lint and format | [#49][49] | Approval pending |
| REUSE | `==6.2.0` (CI only) | 6.2.0 | GPL-3.0-or-later (review required) | `reuse lint` | [#48][48] | Approval pending; CI only |

## GitHub Actions

Workflows in `.github/workflows/`; the release and dependency review workflows pin
each action to a commit SHA. All are covered by [#51][51] (approval pending).

| Action | Version on `main` | License | Used for |
|---|---|---|---|
| actions/checkout | v7.0.1 (SHA), v7 | MIT | Checking out the repository |
| actions/setup-python | v7.1.0 (SHA), v6 | MIT | Installing Python |
| actions/upload-artifact, actions/download-artifact | v7.0.2, v8.0.2 | MIT | Passing the build between release jobs |
| actions/dependency-review-action | v5.0.0 | MIT | License and vulnerability check on pull requests |
| anchore/sbom-action (Syft) | v0.24.3 (Syft v1.54.0) | Apache-2.0 | SPDX and CycloneDX SBOMs for each release |
| anchore/scan-action (Grype) | v7.4.2 (Grype v0.118.0) | MIT | Vulnerability scan of the release SBOM |
| actions/attest | v4.2.2 | MIT | Build provenance attestations |
| pypa/gh-action-pypi-publish | v1.14.3 | BSD-3-Clause | Trusted publishing to PyPI |
| CodeQL (default setup) | managed by GitHub | MIT (action), GitHub CodeQL terms (CLI) | Code scanning |

[45]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/45
[46]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/46
[47]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/47
[48]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/48
[49]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/49
[50]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/50
[51]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/51
[52]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/52
[66]: https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/66
