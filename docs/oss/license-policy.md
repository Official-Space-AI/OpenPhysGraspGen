# License policy

This policy sorts licenses into the classes that decide what this repository may
depend on: runtime dependencies, build, test and CI tools, GitHub Actions, external
programs, data and models. The dependency review check enforces its allowed class on
every pull request ([`.github/dependency-review-config.yml`](../../.github/dependency-review-config.yml)),
and each [OSS evaluation](evaluation.md) applies the rest. The classes are about what
we take in. The licenses of what we publish ourselves are under
[What we publish](#what-we-publish).

The policy takes effect when it is merged. It changes only through a pull request
that the project lead approves, so the history of this file is the history of the
policy.

## License classes

| Class | Licenses | Rule |
|---|---|---|
| Allowed | 0BSD, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0, ISC, MIT, MIT-CMU, PSF-2.0, Python-2.0, Unlicense, Zlib | The dependency review check passes them (`allow-licenses`). Keep their license and copyright notices |
| Conditional | LGPL-2.1, LGPL-3.0, MPL-2.0, EPL-2.0 | LGPL code is used unmodified and dynamically linked; MPL and EPL files keep their license when modified. Needs an approved evaluation, after which the package's purl is listed under `allow-dependencies-licenses` with the approval date |
| Review required | GPL-2.0, GPL-3.0, AGPL-3.0 and other strong copyleft licenses | Never a runtime dependency of the package, and never bundled into its sdist or wheel. A tool used only in CI or development, or an external program the user installs, can be approved by an evaluation that explains why nothing we distribute is affected |
| Prohibited | No license; non-commercial or no-derivatives terms (such as CC-BY-NC, CC-BY-ND); source-available terms (such as SSPL, BUSL, the Elastic License) | Not used |
| Case by case | Model licenses (such as OpenRAIL and the Llama licenses) and dataset licenses | Evaluated for their use, output and redistribution terms before use |

A component under several licenses at once (`AND`) must meet the rule of each. With a
choice (`OR`), we use an allowed one and record which in its evaluation.

A container image we publish also redistributes its base image's packages and the
wheels installed in it, some of them under the GPL or the LGPL (such as the GCC
runtime libraries inside NumPy's wheels). The base image is evaluated as one
component, is used unmodified and pinned by digest, and every image ships an SBOM
listing these components and a notice that says where their source is.

Components outside the allowed class that have been approved are listed in the
[inventory](inventory.md) with their evaluation.

## When an evaluation is needed

Every new runtime dependency, build, test or CI tool, GitHub Action, external program,
dataset or model is evaluated in an issue with the **OSS evaluation** form before it is
added, whatever its license, as [CONTRIBUTING.md](../../CONTRIBUTING.md) asks. The
steps are in [evaluation.md](evaluation.md).

## What we publish

Our code is under Apache-2.0, and every file carries SPDX license information that
`reuse lint` checks in CI. The grasp dataset is under CC BY-NC 4.0
([#79](https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/79)), which each
release also states in its dataset card. Before code, data or a model is published, a maintainer fills in the **OSS
evaluation** form with the gate *Publish our OSS*, as [GOVERNANCE.md](../../GOVERNANCE.md)
describes.
