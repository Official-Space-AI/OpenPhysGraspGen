# Releasing

Releases are cut by pushing a version tag. The `Release` workflow
(`.github/workflows/release.yml`) then:

1. checks that the tag matches `version` in `pyproject.toml`,
2. builds the sdist and wheel and checks their metadata,
3. installs the wheel into a clean environment and generates SPDX and CycloneDX SBOMs of
   that environment, so the SBOMs list the package as a user installs it together with the
   dependencies it resolves,
4. fails if the SBOMs miss the package or a dependency, or if a known vulnerability of
   severity high or above is found,
5. runs the `Image` workflow (`.github/workflows/image.yml`) on the same wheel: it builds the
   container image on the official Python image pinned in the `Dockerfile`, checks it with
   `.github/scripts/check_image.sh`, generates its SBOM and completes it with
   `.github/scripts/complete_image_sbom.py` (the licenses Syft cannot read and the libraries
   bundled in the NumPy wheel), records every known vulnerability in a Grype report, and
   fails if a package has no license or if a known vulnerability of severity high or above
   has a fix and is not one of the assessed exceptions in `docker/grype.yaml`,
6. only then publishes the sdist and wheel to PyPI as `openphysgraspgen`,
7. attests the build provenance and the SBOM, and creates a GitHub release with the sdist,
   wheel, the three SBOMs, `SHA256SUMS` and the image's Grype report attached. Tags ending in
   `aN`, `bN` or `rcN` are marked as pre-releases,
8. pushes the image built in step 5 to `ghcr.io/official-space-ai/openphysgraspgen` under the
   version (for example `1.0.0rc1`), and attests its provenance and its SBOM.

The PyPI project is `openphysgraspgen`, the repository's name. It is not `physgraspgen`,
because that is the import name of the project's internal solver package (#28).

## Cutting a release

1. Open a pull request that sets `version` in `pyproject.toml` (for example `1.0.0rc1`) and
   updates the roadmap in the README if needed. Merge it after review.
2. Check the Grype report of the latest `Image` run on `main` (its `image-scan` artifact):
   every finding without a fix that scores CVSS 7.0 or above needs an assessment, as
   described under "Findings without a fix". Then tag the merge commit and push the tag:

   ```sh
   git switch main && git pull
   git tag v1.0.0rc1
   git push origin v1.0.0rc1
   ```

3. Before approving the `pypi` environment deployment, open this run's `image-scan` artifact
   and check that every finding without a fix that scores CVSS 7.0 or above is assessed, as
   in step 2: the run scans with a newer vulnerability database than the `main` run did. The run
   asks for the approval only after the build and the image have passed every check. Then
   check that the release page has the seven assets and that the image is in the registry.
4. Attach the image's source to the release as `openphysgraspgen-image-<version>-sources.tar`,
   built as described under "Answering a source request", with a `SHA256SUMS` inside it.
   `NOTICE.container` in the image points there. Until #73 adds this to the release
   workflow, a maintainer builds and attaches it by hand.

## When a release run fails

- **Before the `pypi` job** (`build` or `image` failed, or the deployment was rejected),
  nothing has been published. If the cause was passing, such as a network error, re-run the
  failed jobs. Otherwise fix it in a pull request; for a fixable vulnerability in the base
  image, update the digest as described under "Keeping the workflow current". Then delete
  the tag and push it again on the new merge commit:

  ```sh
  git push origin --delete v1.0.0rc1
  git tag --delete v1.0.0rc1
  git switch main && git pull
  git tag v1.0.0rc1
  git push origin v1.0.0rc1
  ```

- **After the `pypi` job** the version is on PyPI, which never accepts it again, so the tag
  stays. Re-run the failed job from the run's page. `github-release` attaches the same files
  again. `docker` loads the image that the `image` job built and checked and pushes that
  image (the same image ID) without building it again; the saved image is kept for 30 days,
  as long as a run can be re-run. If a job still cannot succeed, release the next version
  (for example `1.0.0rc2`).

## Verifying a release

Download the assets from the release page into one folder, then check the checksums and
the build provenance:

```sh
sha256sum -c SHA256SUMS --ignore-missing
gh attestation verify openphysgraspgen-1.0.0rc1-py3-none-any.whl --repo Official-Space-AI/OpenPhysGraspGen
```

The image is published for linux/amd64 only, under its version, with no `latest` tag. Look
up the digest of the version, check that this repository's release workflow built it from
the tag and attested its SBOM, and pull it by that digest:

```sh
image=ghcr.io/official-space-ai/openphysgraspgen
digest=$(docker buildx imagetools inspect "$image:1.0.0rc1" --format '{{.Manifest.Digest}}')
for type in https://slsa.dev/provenance/v1 https://spdx.dev/Document/v2.3; do
  gh attestation verify "oci://$image@$digest" --repo Official-Space-AI/OpenPhysGraspGen \
    --signer-workflow Official-Space-AI/OpenPhysGraspGen/.github/workflows/release.yml \
    --source-ref refs/tags/v1.0.0rc1 --predicate-type "$type"
done
docker pull "$image@$digest"
```

## Keeping the workflow current

Third-party actions are pinned to full commit SHAs, with the release tag in a comment.
Dependabot (`.github/dependabot.yml`) opens a pull request each week when an action, a
Python dependency, a package the image installs (`docker/requirements.txt`) or the base
image has a new release; review it like any other change.
`pypa/gh-action-pypi-publish` must stay at v1.14.3 or later (GHSA-5xjx-3w67-w9x9).

The `Image` workflow runs the release's image steps without pushing on every pull request
that changes the image, on every push to `main` and every Monday, so a failing check, a new
fixable vulnerability or a new finding to assess shows up before a release does.

The container's base image is pinned by digest in the `Dockerfile`'s `FROM` line and used
unmodified. `BASE_IMAGE` repeats the reference for the image's labels and
`/usr/share/doc/openphysgraspgen/NOTICE.container`, which also says where to get the source
of its copyleft components, and `BASE_DOCKERFILE` links the docker-library Dockerfile that
built the base image. `check_image.sh` fails when either does not match. When Dependabot
proposes a new digest, when the scan finds a fixable vulnerability, or before a release when
the digest is old, update the digest:

1. In a pull request, set `FROM` and `BASE_IMAGE` to the current digest of
   `python:3.12-slim-trixie`, and `BASE_DOCKERFILE` to the commit and directory in the
   source annotation of its linux/amd64 manifest:

   ```sh
   docker buildx imagetools inspect docker.io/library/python:3.12-slim-trixie --format '{{json .Manifest}}' |
     jq -r '.manifests[] | select(.platform.architecture == "amd64") | .annotations["org.opencontainers.image.source"]'
   ```

2. The `Image` workflow checks the image built from that branch. Re-check the policy for
   container base images on it as well:
   - the base image is the official image, used unmodified and pinned by digest;
   - the image SBOM completion finds no package without a license. It prints each package
     it filled in with its version: copy that list into the issue below, and read the
     license of each package whose version changed;
   - `NOTICE.container` still covers every component under the GPL, the LGPL, the MPL-2.0
     or another license that requires source availability;
   - `check_image.sh` passes, so among other things there is no FFmpeg in the image.

   The same checks can be run locally:

   ```sh
   python -m build --wheel
   docker build --platform linux/amd64 --provenance=false --tag openphysgraspgen:check .
   sh .github/scripts/check_image.sh openphysgraspgen:check
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock anchore/syft:v1.54.0 \
     scan docker:openphysgraspgen:check --platform linux/amd64 -o spdx-json > image.spdx.json
   docker run --rm --platform linux/amd64 --entrypoint sh openphysgraspgen:check \
     -c 'ls -d /usr/local/lib/python3*/site-packages/numpy.libs/* 2>/dev/null || true' > numpy-libs.txt
   python3 .github/scripts/complete_image_sbom.py image.spdx.json numpy-libs.txt
   docker run --rm --platform linux/amd64 --entrypoint cat openphysgraspgen:check \
     /usr/share/doc/openphysgraspgen/NOTICE.container
   ```

3. Record the check in an issue labelled `re-evaluation`, with the old and new digests and
   the result of each condition, and link it from the pull request before it is merged.

## Findings without a fix

The image's vulnerability gate fails only on a vulnerability of severity high or above that
has a fix. Findings without a fix are mostly in the base image's Debian packages. Decide
the action on each within the time that its CVSS score sets in the
[response targets](docs/oss/vulnerability-log.md#response-targets): a day from 9.0, a week
from 7.0, a month from 4.0 and three months below. The report's severity is not that
score, since for a Debian package Grype gives Debian's own rating, so read the score in
the report's CVSS entries. Assess each one that scores 7.0 or above also before the first
release that ships it, in the base image's evaluation issue or in an issue labelled
`re-evaluation`:

- When the image does not contain or never runs the vulnerable code, record it as not
  affected, with the reason: for example a command that the image never runs, a function
  that no file in the image calls, or an installed version outside the affected range.
- Otherwise accept it only until a stated date, no later than the next quarterly
  re-evaluation, and assess it again by then. The project lead approves the acceptance
  in the issue, or SpaceAI's CEO when the project lead proposed it.

Record the decisions on the lower findings in an issue labelled `re-evaluation`; the
quarterly one reviews all findings without a fix again. Each release attaches the full
Grype report.

Some findings count as fixed although no digest of `python:3.12-slim-trixie` clears them:
Grype matches CPython by CPE against NVD ranges that often have no 3.12 bound, and the pip
that comes with the base image is fixed only in a later pip. When one of severity high or
above stops the gate, assess it as above in an issue labelled `re-evaluation`, and then add
an entry to `docker/grype.yaml` in a reviewed pull request: the vulnerability, the package
name and its exact version, and a reason that links the issue. Only the gate reads that
file, so the report still lists the finding, and an entry stops matching when a new digest
changes the version. A release whose gate stopped is then cut again as described under
"When a release run fails".

Two kinds of bundled libraries are out of reach of the automated matching, and their
advisories are checked by hand in the quarterly issue and in the review of every pull
request that changes NumPy or PyArrow in `docker/requirements.txt`:

- The libraries in `numpy.libs` (OpenBLAS, libgfortran, libquadmath) are listed in the image
  SBOM under their file names and without a version on purpose. Their files carry no
  version, and Grype would match a package without a version to every advisory for its
  name, so it matches none to them.
- The zlib that the PyArrow wheel links into its own libraries is in no SBOM. PyArrow
  26.0.0 carries zlib 1.3.2, which #72 assessed as not affected by CVE-2026-85091.

## Answering a source request

Every image carries a written offer in `NOTICE.container`: for at least three years after
we last distribute an image, SpaceAI gives anyone who asks a complete machine-readable copy
of the corresponding source of its components under the GPL, the LGPL, the MPL-2.0 or
another license that requires source availability. Requests come to oss.spaceai@gmail.com.
The maintainers answer within 30 days and charge nothing beyond the cost of physically
performing the distribution; a download link costs nothing. From 1.0.0rc1 on, each release
also attaches the image's source as `openphysgraspgen-image-<version>-sources.tar`, so a
link to it answers most requests.

To produce the source for an image, by tag or digest:

1. List the Debian source packages and their versions in the image, with the sources that
   the `Built-Using` and `Static-Built-Using` fields name: those are compiled into a binary
   package, such as rustc and the Rust crates in `sqv`. The image SBOM on the release lists
   the binary packages; this gives the sources, which differ between architectures:

   ```sh
   image=ghcr.io/official-space-ai/openphysgraspgen:1.0.0rc1
   query() { docker run --rm --platform linux/amd64 --entrypoint dpkg-query "$image" -W -f "$1"; }
   {
     query '${source:Package} ${source:Version}\n'
     query '${Built-Using},${Static-Built-Using},\n' |
       tr ',' '\n' | sed -n 's/^ *\([^ ]*\) (= \(.*\))$/\1 \2/p'
   } | sort -u
   ```

2. Download each source package at that version, with `apt-get source <package>=<version>`
   from deb.debian.org (`deb-src` entries for trixie, trixie-updates and trixie-security)
   while the version is current there, and otherwise from snapshot.debian.org
   (`https://snapshot.debian.org/package/<package>/<version>/`). Check each one's files
   against its `.dsc` with `dscverify --no-sig-check`.
3. Add `Python-<version>.tar.xz` from python.org, with the version `NOTICE.container` names,
   checked against `PYTHON_SHA256` in the docker-library Dockerfile at the commit it links,
   and that Dockerfile.
4. For libgfortran and libquadmath in `numpy.libs`, add the source of the GCC build they
   come from. NumPy's wheels take them from their manylinux build image, and each file
   names its build in its debug link: `strings` shows
   `libgfortran.so.5.0.0-8.5.0-28.el8_10.alma.1.x86_64.debug` for NumPy 2.5.4, that is
   AlmaLinux 8.10's gcc 8.5.0-28, whose source RPM is in AlmaLinux's vault
   (https://vault.almalinux.org/).
5. Send the files or a link to them, and keep the request and the answer.

For the components under GPL-3.0 or LGPL-3.0, such as bash and coreutils, a download is
conveyed under GPLv3 section 6(d): the source may be on another server when clear
directions next to the image say where it is, and we stay responsible for it being
available. `NOTICE.container` points to the sources archive on the release and to Debian's
snapshot archive, and the written offer covers the rest. #73 builds the archive in the
release workflow and checks on pull requests that every source in the image can be found.

## One-time setup

- On PyPI, add a trusted publisher for the project `openphysgraspgen`: owner
  `Official-Space-AI`, repository `OpenPhysGraspGen`, workflow `release.yml`, environment
  `pypi`. Before the first upload this is a "pending publisher" under the account's
  publishing settings.
- The repository has an environment `pypi` that only `v*` tags can deploy to, with a required
  reviewer, so every upload waits for a maintainer's approval.
- After the first image is pushed and its sources archive is on the release, make the
  `openphysgraspgen` container package public in the organization's package settings and
  link it to this repository. A public package cannot be made private again.
