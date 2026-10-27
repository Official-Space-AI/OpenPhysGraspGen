# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| Latest release on `main` | Yes |
| Older pre-releases | No, please upgrade |

## Reporting a vulnerability

Please **do not** open a public issue, discussion or pull request for a security
problem.

Report it privately through
[GitHub private vulnerability reporting](https://github.com/Official-Space-AI/OpenPhysGraspGen/security/advisories/new)
or by email to **oss.spaceai@gmail.com**.

Include:

- the affected version or commit,
- a description of the problem and its impact,
- steps or a minimal script to reproduce it.

## What happens next

| Step | Target |
|---|---|
| Acknowledge the report | 48 hours |
| Initial assessment and severity | 7 business days |
| Fix or mitigation plan shared with the reporter | 30 days |

We will credit reporters in the release notes unless they prefer to stay
anonymous. Fixes are released as a new version, and a GitHub security advisory is
published once a fix is available.

## Supply chain

Every release publishes an SBOM in SPDX and CycloneDX formats together with
SHA-256 checksums as release assets. Dependency changes in pull requests are
checked for known vulnerabilities and license policy. Vulnerabilities found in our
dependencies, and what was done about each, are recorded in
[docs/oss/vulnerability-log.md](docs/oss/vulnerability-log.md).

The container image is built from the released wheel on a pinned official Python
image, and its build provenance and SBOM are attested. A release stops on a known
vulnerability of severity high or above that has a fix. Findings without a fix are
mostly in the base image's Debian packages. Those of severity high or above are
assessed one by one in a public issue before the first release that ships them: a
finding in code that the image does not contain or never runs is recorded as not
affected, with the reason, and any other is accepted only until a stated date, when
it is assessed again. The others are reviewed in the quarterly `re-evaluation`
issue. The full Grype report is attached to each release. The libraries that NumPy
bundles in `numpy.libs` carry no version in the image SBOM on purpose, so automated
matching skips them; their advisories are checked by hand in the quarterly issue.
