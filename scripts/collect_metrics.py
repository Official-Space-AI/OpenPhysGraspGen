# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Collect one month of project health metrics from the GitHub REST API.

Usage: python scripts/collect_metrics.py 2026-10 [--repo owner/name] [--csv PATH]

docs/metrics/definitions.md defines every column. Set GITHUB_TOKEN to raise the API
rate limit. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

API = "https://api.github.com"
FIELDS = [
    "month",
    "collected_at",
    "issues_opened",
    "issues_closed",
    "prs_opened",
    "prs_merged",
    "outside_items_opened",
    "median_first_response_hours",
    "items_without_response_7d",
    "outside_pr_authors",
    "first_time_contributors",
    "repeat_contributors",
    "open_good_first_issues",
    "releases",
    "releases_with_sbom",
]
# Authors whose issues and pull requests are not waiting for a maintainer's answer.
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}
WAIT = timedelta(days=7)

Fetch = Callable[[str], Any]


def get(path: str) -> Any:
    request = urllib.request.Request(API + path)
    request.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def get_all(fetch: Fetch, path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = fetch(f"{path}{separator}per_page=100&page={page}")
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def parse(timestamp: str | None) -> datetime | None:
    if timestamp is None:
        return None
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def is_bot(user: dict[str, Any] | None) -> bool:
    return user is None or user.get("type") == "Bot"


def first_response(fetch: Fetch, repo: str, item: dict[str, Any]) -> datetime | None:
    """The earliest comment or review by a person other than the author."""
    author = item["user"]["login"]
    times = [
        parse(comment["created_at"])
        for comment in get_all(fetch, f"/repos/{repo}/issues/{item['number']}/comments")
        if not is_bot(comment["user"]) and comment["user"]["login"] != author
    ]
    if "pull_request" in item:
        times += [
            parse(review["submitted_at"])
            for review in get_all(
                fetch, f"/repos/{repo}/pulls/{item['number']}/reviews"
            )
            if review.get("submitted_at")
            and not is_bot(review["user"])
            and review["user"]["login"] != author
        ]
    return min((t for t in times if t is not None), default=None)


def collect(
    fetch: Fetch, repo: str, month: str, now: datetime | None = None
) -> dict[str, object]:
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    end = (start + timedelta(days=32)).replace(day=1)
    now = now or datetime.now(timezone.utc)
    # A month still running is counted up to now; the run after it ends replaces it.
    cutoff = min(end, now)

    def within(timestamp: str | None) -> bool:
        moment = parse(timestamp)
        return moment is not None and start <= moment < cutoff

    items = [
        i
        for i in get_all(fetch, f"/repos/{repo}/issues?state=all")
        if not is_bot(i["user"])
    ]
    issues = [i for i in items if "pull_request" not in i]
    pulls = [
        p
        for p in get_all(fetch, f"/repos/{repo}/pulls?state=all")
        if not is_bot(p["user"])
    ]

    outside = [
        i
        for i in items
        if within(i["created_at"]) and i["author_association"] not in MAINTAINERS
    ]
    response_hours, waiting = [], 0
    for item in outside:
        created = parse(item["created_at"])
        responded = first_response(fetch, repo, item)
        if responded is not None and responded < cutoff:
            response_hours.append((responded - created).total_seconds() / 3600)
        elif cutoff - created > WAIT:
            waiting += 1

    merged = [p for p in pulls if p.get("merged_at")]
    authors_before = {
        p["user"]["login"] for p in merged if parse(p["merged_at"]) < start
    }
    authors_now = {p["user"]["login"] for p in merged if within(p["merged_at"])}

    good_first = [
        i
        for i in issues
        if any(label["name"] == "good first issue" for label in i["labels"])
        and parse(i["created_at"]) < cutoff
        and (i["closed_at"] is None or parse(i["closed_at"]) >= cutoff)
    ]

    releases = [
        r
        for r in get_all(fetch, f"/repos/{repo}/releases")
        if within(r["published_at"])
    ]
    with_sbom = [
        r
        for r in releases
        if any(a["name"].endswith((".spdx.json", ".cdx.json")) for a in r["assets"])
    ]

    return {
        "month": month,
        "collected_at": now.strftime("%Y-%m-%d"),
        "issues_opened": sum(within(i["created_at"]) for i in issues),
        "issues_closed": sum(within(i["closed_at"]) for i in issues),
        "prs_opened": sum(within(p["created_at"]) for p in pulls),
        "prs_merged": sum(within(p["merged_at"]) for p in merged),
        "outside_items_opened": len(outside),
        "median_first_response_hours": (
            round(statistics.median(response_hours), 1) if response_hours else ""
        ),
        "items_without_response_7d": waiting,
        "outside_pr_authors": len(
            {i["user"]["login"] for i in outside if "pull_request" in i}
        ),
        "first_time_contributors": len(authors_now - authors_before),
        "repeat_contributors": len(authors_now & authors_before),
        "open_good_first_issues": len(good_first),
        "releases": len(releases),
        "releases_with_sbom": len(with_sbom),
    }


def write_row(path: Path, row: dict[str, object]) -> None:
    """Add the month's row to the CSV, replacing an earlier row for the same month."""
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["month"] != row["month"]]
    rows.append(row)
    rows.sort(key=lambda r: str(r["month"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("month", help="YYYY-MM")
    parser.add_argument("--repo", default="Official-Space-AI/OpenPhysGraspGen")
    parser.add_argument("--csv", type=Path, help="add the row to this CSV")
    args = parser.parse_args(argv)

    row = collect(get, args.repo, args.month)
    json.dump(row, sys.stdout, indent=2)
    print()
    if args.csv:
        write_row(args.csv, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
