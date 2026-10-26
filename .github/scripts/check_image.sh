#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
#
# Check a built image before anything is published. Run it from the repository root:
#
#   sh .github/scripts/check_image.sh IMAGE
#
# The Image workflow runs it on every image it builds, the release's included. It fails
# unless the Dockerfile's FROM line and BASE_IMAGE name the same base image and the
# image's label names it too, the notice is in the image, the CLI and the dataset extra
# work, the image runs as a numeric user other than root, and no FFmpeg file is in the
# image: the FFmpeg evaluation covers running a user's own FFmpeg, not shipping one.
set -eu

image=$1

fail() {
  echo "check_image.sh: $*" >&2
  exit 1
}

from=$(sed -n 's/^FROM //p' Dockerfile)
arg=$(sed -n 's/^ARG BASE_IMAGE=//p' Dockerfile)
{ [ -n "$from" ] && [ "$from" = "$arg" ]; } ||
  fail "FROM ($from) and ARG BASE_IMAGE ($arg) in the Dockerfile must name the same image"
label=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}' "$image")
[ "$label" = "$from" ] || fail "the image's base.name label is '$label', not '$from'"
docker run --rm --entrypoint test "$image" -s /usr/share/doc/openphysgraspgen/NOTICE.container ||
  fail "the image has no NOTICE.container"

# The CLI runs the pipeline, which needs only NumPy; the dataset extra adds PyArrow and
# PyYAML.
docker run --rm "$image" > /dev/null || fail "the CLI does not run"
docker run --rm --entrypoint python "$image" -c 'import openphysgraspgen.dataset, pyarrow, yaml' ||
  fail "the dataset extra does not import"

user=$(docker image inspect --format '{{.Config.User}}' "$image")
case "$user" in
  "" | 0 | 0:* | *[!0-9:]*) fail "the image's user is '$user', not a numeric user other than root" ;;
esac
[ "$(docker run --rm --entrypoint id "$image" -u)" != 0 ] || fail "the image runs as root"

found=$(docker run --rm --user root --entrypoint find "$image" / -xdev \
  \( -iname '*ffmpeg*' -o -name 'libav*' \) -print)
[ -z "$found" ] || fail "the image holds FFmpeg files, which its evaluation does not cover: $found"

echo "check_image.sh: $image passed"
