# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""scripts/collect_metrics.py on a fake GitHub API."""

import csv
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_metrics.py"
spec = importlib.util.spec_from_file_location("collect_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

REPO = "org/repo"
HUMAN = {"type": "User"}
BOT = {"login": "dependabot[bot]", "type": "Bot"}


def user(login):
    return {**HUMAN, "login": login}


def item(number, login, created, association="NONE", pr=False, closed=None, labels=()):
    entry = {
        "number": number,
        "user": user(login),
        "author_association": association,
        "created_at": created,
        "closed_at": closed,
        "labels": [{"name": name} for name in labels],
    }
    if pr:
        entry["pull_request"] = {}
    return entry


def pull(number, login, created, merged=None):
    return {
        "number": number,
        "user": user(login),
        "created_at": created,
        "merged_at": merged,
    }


def fake(responses):
    def fetch(path):
        base, _, query = path.partition("?")
        query = "&".join(
            part
            for part in query.split("&")
            if not part.startswith(("per_page", "page"))
        )
        key = f"{base}?{query}" if query else base
        return responses.get(key, []) if "page=1" in path else []

    return fetch


@pytest.fixture
def api():
    return {
        f"/repos/{REPO}/issues?state=all": [
            # A maintainer's own issue with no reply is not waiting for anyone.
            item(
                1, "lead", "2026-10-02T00:00:00Z", "MEMBER", labels=["good first issue"]
            ),
            # An outside issue answered after 5 hours.
            item(2, "visitor", "2026-10-03T00:00:00Z", closed="2026-10-10T00:00:00Z"),
            # An outside pull request no one answered for more than 7 days.
            item(
                3, "newcomer", "2026-10-05T00:00:00Z", "FIRST_TIME_CONTRIBUTOR", pr=True
            ),
            # Opened the month before, so outside October.
            item(4, "visitor", "2026-09-20T00:00:00Z"),
            {**item(5, "x", "2026-10-06T00:00:00Z", pr=True), "user": BOT},
        ],
        f"/repos/{REPO}/pulls?state=all": [
            pull(10, "lead", "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z"),
            pull(11, "lead", "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z"),
            pull(12, "helper", "2026-10-07T00:00:00Z", "2026-10-08T00:00:00Z"),
            pull(3, "newcomer", "2026-10-05T00:00:00Z"),
            {
                **pull(5, "x", "2026-10-06T00:00:00Z", "2026-10-06T01:00:00Z"),
                "user": BOT,
            },
        ],
        f"/repos/{REPO}/issues/2/comments": [
            {"user": user("visitor"), "created_at": "2026-10-03T01:00:00Z"},
            {"user": user("lead"), "created_at": "2026-10-03T05:00:00Z"},
        ],
        f"/repos/{REPO}/releases": [
            {
                "published_at": "2026-10-20T00:00:00Z",
                "assets": [{"name": "openphysgraspgen-v1.0.0rc1.spdx.json"}],
            },
            {
                "published_at": "2026-10-21T00:00:00Z",
                "assets": [{"name": "SHA256SUMS"}],
            },
        ],
    }


def test_a_month_is_counted(api):
    end = datetime(2026, 11, 1, tzinfo=timezone.utc)
    row = metrics.collect(fake(api), REPO, "2026-10", now=end)
    assert row == {
        "month": "2026-10",
        "collected_at": "2026-11-01",
        "issues_opened": 2,
        "issues_closed": 1,
        "prs_opened": 3,
        "prs_merged": 2,
        "outside_items_opened": 2,
        "median_first_response_hours": 5.0,
        "items_without_response_7d": 1,
        "outside_pr_authors": 1,
        "first_time_contributors": 1,
        "repeat_contributors": 1,
        "open_good_first_issues": 1,
        "releases": 2,
        "releases_with_sbom": 1,
    }


def test_a_running_month_stops_at_now(api):
    # On 10 October the unanswered pull request has waited 5 days, not more than 7.
    now = datetime(2026, 10, 10, tzinfo=timezone.utc)
    row = metrics.collect(fake(api), REPO, "2026-10", now=now)
    assert row["items_without_response_7d"] == 0
    assert row["prs_merged"] == 2
    assert row["releases"] == 0


def test_an_answer_after_7_days_still_counts_as_waiting(api):
    # Answered 9 days after it was opened: in the median, and also left waiting.
    api[f"/repos/{REPO}/issues?state=all"].append(
        item(6, "visitor", "2026-10-10T00:00:00Z")
    )
    api[f"/repos/{REPO}/issues/6/comments"] = [
        {"user": user("lead"), "created_at": "2026-10-19T00:00:00Z"}
    ]
    end = datetime(2026, 11, 1, tzinfo=timezone.utc)
    row = metrics.collect(fake(api), REPO, "2026-10", now=end)
    assert row["median_first_response_hours"] == 110.5
    assert row["items_without_response_7d"] == 2


def test_a_listed_maintainer_seen_as_contributor_is_not_outside(api):
    # The workflow's token sees a maintainer with a private membership as CONTRIBUTOR.
    api[f"/repos/{REPO}/issues?state=all"].append(
        item(7, "teammate", "2026-10-11T00:00:00Z", "CONTRIBUTOR", pr=True)
    )
    end = datetime(2026, 11, 1, tzinfo=timezone.utc)
    unlisted = metrics.collect(fake(api), REPO, "2026-10", now=end)
    assert unlisted["outside_items_opened"] == 3
    assert unlisted["outside_pr_authors"] == 2
    listed = metrics.collect(
        fake(api), REPO, "2026-10", now=end, maintainers=frozenset({"teammate"})
    )
    assert listed["outside_items_opened"] == 2
    assert listed["outside_pr_authors"] == 1
    assert listed["items_without_response_7d"] == 1


def test_the_maintainers_are_read_from_the_governance_table(tmp_path):
    path = tmp_path / "GOVERNANCE.md"
    path.write_text(
        "# Governance\n\n"
        "Write to [@someone](https://github.com/someone) for anything else.\n\n"
        "### Current maintainers\n\n"
        "| Name | GitHub | Area |\n"
        "|---|---|---|\n"
        "| Lead | [@lead](https://github.com/lead) | Releases |\n"
        "| Helper | [@help-er2](https://github.com/help-er2) | Review |\n\n"
        "The project lead is [@lead](https://github.com/lead).\n\n"
        "## Decisions\n\n"
        "| Who | [@other](https://github.com/other) |\n",
        encoding="utf-8",
    )
    assert metrics.maintainer_logins(path) == {"lead", "help-er2"}


def test_the_repository_governance_names_its_maintainers():
    logins = metrics.maintainer_logins(metrics.GOVERNANCE)
    assert {"oollccddss", "kms8720"} <= logins


def test_a_row_replaces_the_same_month(tmp_path):
    path = tmp_path / "kpi.csv"
    empty = dict.fromkeys(metrics.FIELDS, 0)
    metrics.write_row(path, {**empty, "month": "2026-10", "prs_merged": 1})
    metrics.write_row(path, {**empty, "month": "2026-09"})
    metrics.write_row(path, {**empty, "month": "2026-10", "prs_merged": 4})
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["month"] for r in rows] == ["2026-09", "2026-10"]
    assert rows[1]["prs_merged"] == "4"
