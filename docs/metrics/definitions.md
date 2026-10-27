# Project metrics

We track a few numbers that show whether people get answers and whether contributions
land, not only how popular the repository is. Stars and commit counts are context, not
targets.

[`scripts/collect_metrics.py`](../../scripts/collect_metrics.py) computes them from the
GitHub REST API, so the definitions and the queries are versioned together. The
**Monthly metrics** workflow runs it on the 1st of every month for the month before,
and a maintainer adds the row to [`kpi.csv`](kpi.csv) in the monthly review pull
request. To collect a month by hand:

```sh
GITHUB_TOKEN=... python scripts/collect_metrics.py 2026-10 --csv docs/metrics/kpi.csv
```

A second run for the same month replaces its row.

## Columns

Bots (Dependabot and other app accounts) are left out of every count. "Outside" means
an author who is not an owner, member or collaborator of the repository.

| Column | Definition | Why |
|---|---|---|
| `month` | The calendar month counted, in UTC | |
| `collected_at` | The UTC date the row was computed. A month that was still running is counted up to that moment | Tells a partial month from a full one |
| `issues_opened`, `issues_closed` | Issues, not pull requests, created or closed in the month | Inflow and throughput |
| `prs_opened`, `prs_merged` | Pull requests created or merged in the month | Contribution flow |
| `outside_items_opened` | Issues and pull requests created in the month by outside authors | How much reaches us from outside |
| `median_first_response_hours` | For those outside items: the median time from creation to the first comment or review by someone other than the author. Empty when there were none | Responsiveness; [COMMUNITY.md](../../COMMUNITY.md) promises first triage within 48 hours |
| `items_without_response_7d` | Outside items created in the month that waited more than 7 days for a first response from someone other than the author, whether one came later or not | People left waiting; target 0 |
| `outside_pr_authors` | Distinct outside authors who opened a pull request in the month, merged or not | New contributors trying |
| `first_time_contributors` | Authors whose first merged pull request is in the month | Contributions that land |
| `repeat_contributors` | Authors with a merged pull request in the month who also had one before | Contributors who come back |
| `open_good_first_issues` | Issues labelled `good first issue` that were open at the end of the month (or when collected), by the labels they carry now | COMMUNITY.md keeps at least five open |
| `releases`, `releases_with_sbom` | Releases published in the month, and those with SPDX or CycloneDX assets | SBOM on every release; target 100% |

Not collected by the script yet: Discussions questions and how many were answered,
issues filed from other channels (`community` label), and maintainer hours.

## Baseline and targets

| Metric | Baseline (2026-09) | October 2026 (to the 27th) | Target (2026-12) |
|---|---|---|---|
| Median first response to outside items | none opened | 7.8 h (2 items) | ≤ 48 h |
| Outside items without a response after 7 days | 0 | 0 | 0 |
| Outside pull request authors | 0 | 2 | set at the November review |
| Open `good first issue` | 0 | 5 | ≥ 5 |
| Releases with SBOM | none released | none released | every release |

September had no public activity: the repository was published on 2026-08-25 with
three pull requests by one maintainer, and work resumed in the open on 2026-10-22.
The rows in `kpi.csv` for August to October were collected on 2026-10-27, and the
1 November run replaces the October row with the full month.

## Review

Each month the maintainers look at the new row, post a short summary in
[Discussions → Announcements](https://github.com/Official-Space-AI/OpenPhysGraspGen/discussions/categories/announcements),
and open at most one or two improvement issues, each with an owner, a due date and
what counts as done. The targets themselves are reviewed every quarter.
