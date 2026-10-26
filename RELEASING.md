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
   has a fix,
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
   every finding of severity high or above without a fix needs an assessment, as described
   under "Findings without a fix". Then tag the merge commit and push the tag:

   ```sh
   git switch main && git pull
   git tag v1.0.0rc1
   git push origin v1.0.0rc1
   ```

3. Approve the `pypi` environment deployment when the run asks for it, which it does only
   after the build and the image have passed every check. Then check that the release page
   has the seven assets and that the image is in the registry.

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

```sh
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
has a fix. Findings without a fix are mostly in the base image's Debian packages. Assess
each one of severity high or above before the first release that ships it, in the base
image's evaluation issue or in an issue labelled `re-evaluation`:

- When the image does not contain or never runs the vulnerable code, record it as not
  affected, with the reason: for example a command that the image never runs, a function
  that no file in the image calls, or an installed version outside the affected range.
- Otherwise accept it only until a stated date, no later than the next quarterly
  re-evaluation, and assess it again by then.

The quarterly `re-evaluation` issue reviews the other findings without a fix. Each release
attaches the full Grype report. The libraries in `numpy.libs` are listed in the image SBOM
under their file names and without a version on purpose. Their files carry no version, and
Grype would match a package without a version to every advisory for its name, so it matches
none to them; their advisories (OpenBLAS, libgfortran, libquadmath) are checked by hand in
the quarterly issue, for the NumPy version in `docker/requirements.txt`.

## Answering a source request

Every image carries a written offer in `NOTICE.container`: for at least three years after
we last distribute an image, SpaceAI gives anyone who asks a complete machine-readable copy
of the corresponding source of its components under the GPL, the LGPL, the MPL-2.0 or
another license that requires source availability. Requests come to oss.spaceai@gmail.com.
The maintainers answer within 30 days and charge nothing beyond the cost of physically
performing the distribution; a download link costs nothing.

To produce the source for an image, by tag or digest:

1. List the Debian source packages and their versions in the image. The image SBOM on the
   release lists the binary packages; this gives the sources they were built from:

   ```sh
   docker run --rm --entrypoint dpkg-query ghcr.io/official-space-ai/openphysgraspgen:1.0.0rc1 \
     -W -f '${source:Package} ${source:Version}\n' | sort -u
   ```

2. Download each source package at that version, with `apt-get source <package>=<version>`
   from deb.debian.org (`deb-src` entries for trixie, trixie-updates and trixie-security)
   while the version is current there, and otherwise from snapshot.debian.org
   (`https://snapshot.debian.org/package/<package>/<version>/`).
3. Add `Python-<version>.tar.xz` from python.org, with the version `NOTICE.container` names,
   and the docker-library Dockerfile at the commit it links.
4. For libgfortran and libquadmath in `numpy.libs`, add the source of the GCC release they
   were built from (https://gcc.gnu.org). NumPy's wheels take them from their manylinux
   build image; that NumPy version's wheel build names the image and its GCC.
5. Send the files or a link to them, and keep the request and the answer.

Publishing these sources next to each image, so that the written offer is only a backup,
is planned.

## One-time setup

- On PyPI, add a trusted publisher for the project `openphysgraspgen`: owner
  `Official-Space-AI`, repository `OpenPhysGraspGen`, workflow `release.yml`, environment
  `pypi`. Before the first upload this is a "pending publisher" under the account's
  publishing settings.
- The repository has an environment `pypi` that only `v*` tags can deploy to, with a required
  reviewer, so every upload waits for a maintainer's approval.
- After the first image is pushed, make the `openphysgraspgen` container package public in the
  organization's package settings and link it to this repository.
