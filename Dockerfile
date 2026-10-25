# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
#
# The release image: the published wheel, with its dataset extra, on the official Python
# image. The base image is used unmodified and pinned by digest; its Debian packages keep
# their own licenses. The image SBOM attached to each GitHub release lists every package.
# Build it from a built wheel: python -m build, then docker build .
ARG BASE_IMAGE=docker.io/library/python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

FROM ${BASE_IMAGE}
ARG BASE_IMAGE

COPY dist/*.whl /tmp/dist/
RUN wheel="$(ls /tmp/dist/*.whl)" \
    && python -m pip install --no-cache-dir --root-user-action=ignore "${wheel}[dataset]" \
    && rm -rf /tmp/dist \
    && useradd --create-home --uid 1000 openpgg

COPY LICENSE NOTICE /usr/share/doc/openphysgraspgen/
RUN printf '%s\n' \
    "OpenPhysGraspGen container image" \
    "" \
    "This image holds OpenPhysGraspGen (Apache-2.0, see LICENSE and NOTICE in this folder)" \
    "and its Python dependencies from PyPI, on top of this official Docker image, used" \
    "unmodified:" \
    "" \
    "  ${BASE_IMAGE}" \
    "" \
    "The SBOM attached to each GitHub release lists every package in the image with its" \
    "version and license. Its components under the GPL, the LGPL, the MPL-2.0 or another" \
    "license that requires their source to be available are:" \
    "" \
    "- The Debian packages of the base image, many of them GPL or LGPL. ca-certificates" \
    "  holds the Mozilla CA list under the MPL-2.0, and libdb5.3t64 is Berkeley DB under" \
    "  the Sleepycat license. Their source is available from Debian at those versions" \
    "  (https://snapshot.debian.org, https://sources.debian.org)." \
    "- certifi, which pip vendors: the same Mozilla CA list under the MPL-2.0, in source" \
    "  form in the image." \
    "- Python ${PYTHON_VERSION} (CPython), whose modules link Berkeley DB, GNU dbm and GNU" \
    "  Readline. The official image compiles it, without changes, from the source release" \
    "  https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
    "  with the Dockerfile at" \
    "  https://github.com/docker-library/python/blob/master/3.12/slim-trixie/Dockerfile" \
    "  (its history has the revision that built this image)." \
    "- The GCC runtime libraries that the NumPy wheel bundles in numpy.libs: libgfortran," \
    "  GPL-3.0-or-later with the GCC Runtime Library Exception 3.1, and libquadmath," \
    "  LGPL-2.1-or-later. Their source is part of GCC (https://gcc.gnu.org)." \
    "" \
    "For three years after an image is released we also provide, to anyone who asks, the" \
    "corresponding source of any component in it under the GPL, the LGPL, the MPL-2.0 or" \
    "another license that requires source availability, including the CPython build and" \
    "the GCC runtime libraries above: write to oss.spaceai@gmail.com with the image tag" \
    "and the component names." \
    > /usr/share/doc/openphysgraspgen/NOTICE.container

LABEL org.opencontainers.image.title="OpenPhysGraspGen" \
      org.opencontainers.image.source="https://github.com/Official-Space-AI/OpenPhysGraspGen" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

USER openpgg
WORKDIR /home/openpgg
ENTRYPOINT ["openphysgraspgen"]
