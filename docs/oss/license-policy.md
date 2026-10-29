# License policy

This policy sorts licenses into the classes that decide what this repository may
depend on: runtime dependencies, build, test and CI tools, GitHub Actions, external
programs, data and models. The dependency review check enforces its allowed class on
every pull request ([`.github/dependency-review-config.yml`](../../.github/dependency-review-config.yml)),
and each [OSS evaluation](evaluation.md) applies the rest. The classes are about what
we take in. The licenses of what we publish ourselves are under
[What we publish](#what-we-publish).

These are the license rules of SpaceAI's open source policy, in effect since
23 October 2026, as they apply to this repository, its releases and the public
dataset; the rule for container base images applies from 25 October 2026. The
classes below and the allow list in the dependency review config change through a
pull request that the project lead approves, or that SpaceAI's CEO approves when the
project lead proposed it. Any other rule here changes only with the company policy,
which SpaceAI's CEO approves.

## License classes

| Class | Licenses (SPDX) | Rule |
|---|---|---|
| Allowed | MIT, MIT-CMU, Apache-2.0, BSD-2-Clause, BSD-3-Clause, 0BSD, ISC, Zlib, PSF-2.0, Python-2.0, CC0-1.0, Unlicense, and CC-BY-4.0 for data and assets | Used once its evaluation is confirmed, as [evaluation.md](evaluation.md) describes. Keep their notices |
| Conditional | MPL-2.0, EPL-2.0, LGPL-2.1, LGPL-3.0, CC-BY-SA-4.0, model licenses (such as OpenRAIL and the Llama licenses), vendor terms (such as the NVIDIA software license and LicenseRef-TOST-1.0) | The evaluation states how the component is combined and distributed, and the project lead approves it |
| Review required | GPL-2.0, GPL-3.0, AGPL-3.0 | Not included in or linked from the repository or anything we distribute. A development tool that stays out of everything we distribute may be used with the project lead's approval. Container base images follow the rule below |
| Prohibited | SSPL-1.0, non-commercial terms (such as CC-BY-NC), no-derivatives terms when we would modify the work (such as CC-BY-ND), no license, terms that forbid modification or redistribution | Not used |

The dependency review check passes the allowed class except CC-BY-4.0, which is for
data and assets (`allow-licenses`). A package under a conditional license passes once
its evaluation is approved and its purl is listed under `allow-dependencies-licenses`
with the approval date. LGPL code is used unmodified and dynamically linked, and MPL
and EPL files keep their license when modified.

A component under several licenses at once (`AND`) must meet the rule of each. With a
choice (`OR`), we use an allowed one and record which in its evaluation.

The OS packages and system libraries in a container image's base image (an official
image of an operating system and a language runtime) are programs separate from our
work. They may be shipped in our image even under a review-required or conditional
license when all of the following hold; a package under a prohibited license is never
shipped.

1. The official image is used unmodified and pinned by digest.
2. Our code only runs those packages as separate programs or uses them as the
   system's libraries, and never modifies them.
3. The release's image SBOM lists every package in the image with its license.
4. The image's notice names the base image and its digest and says where the sources
   of the packages are, and we provide the source of that version to anyone who asks,
   for at least three years after the image was last distributed.
5. These conditions are checked again, and recorded in an evaluation, whenever the
   base image changes or is updated.

Conditions 2 to 5 also apply to runtime libraries that a dependency installed in the
image bundles in its binaries, such as the GCC runtime libraries inside NumPy's
wheels.

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
