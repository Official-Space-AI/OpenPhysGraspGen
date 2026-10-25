# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Complete the release image's SBOM, and fail if a package in it has no license.

Usage: complete_image_sbom.py SBOM NUMPY_LIBS

SBOM is the SPDX JSON that Syft writes for the image; it is completed in place.
NUMPY_LIBS is a file with the paths in the image's numpy.libs folder, one per line.

Syft leaves the license as NOASSERTION where it cannot read one, and it does not list
the libraries that the NumPy wheel bundles in numpy.libs. This fills in the licenses
below and adds a package for each of those libraries. A package still without a
license, such as a new one after a base image update, stops the release: read its
license and add it here.
"""

import json
import re
import sys

# Licenses Syft cannot read, by purl type or, for a package without a purl, the folder
# Syft found it in, and then by package name. Each comment says where it was read.
LICENSES = {
    "pkg:deb": {
        # /usr/share/doc/debian-archive-keyring/copyright: GNU GPL 2 or later
        "debian-archive-keyring": "GPL-2.0-or-later",
        # /usr/share/doc/gcc-14-base/copyright, its only file: GCC is GPL 3 or later
        "gcc-14-base": "GPL-3.0-or-later",
        # The same file: libgcc is GPL 3 or later with the GCC Runtime Library Exception
        "libgcc-s1": "GPL-3.0-or-later WITH GCC-exception-3.1",
        # The same for libstdc++-v3; the headers of its gdb printers say GPL 3 or later
        "libstdc++6": "GPL-3.0-or-later WITH GCC-exception-3.1 AND GPL-3.0-or-later",
        # /usr/share/doc/libcrypt1/copyright: LGPL 2.1 or later, some hashes BSD, 0BSD
        "libcrypt1": "LGPL-2.1-or-later AND BSD-2-Clause AND BSD-3-Clause AND 0BSD",
    },
    "pkg:generic": {
        # /usr/local/lib/python3.12/LICENSE.txt: PSF License Version 2
        "python": "PSF-2.0",
    },
    "/usr/local/lib/python3.12/site-packages/pip/_vendor/distlib": {
        # PC/launcher.c in the distlib sdist; pip vendors distlib without its license
        "Simple Launcher": "BSD-2-Clause",
    },
}

# The libraries in numpy.libs, by the name their file names start with, with the
# licenses that NumPy's numpy-*.dist-info/licenses/LICENSE.txt gives for them. Their
# files carry no version, and Grype matches a package without a version to every
# advisory for its name, so they are listed under these names and not as "OpenBLAS".
NUMPY_LIBS = {
    # "OpenBLAS", and "LAPACK", which is bundled in OpenBLAS
    "libscipy_openblas64_": "BSD-3-Clause AND BSD-3-Clause-Open-MPI",
    # "GCC runtime library"
    "libgfortran": "GPL-3.0-or-later WITH GCC-exception-3.1",
    # "libquadmath"
    "libquadmath": "LGPL-2.1-or-later",
}

NOTE = "Set by .github/scripts/complete_image_sbom.py, which says where it was read."


def origin(package):
    """Return the purl type, or else the folder of the first file it was found in."""
    for ref in package.get("externalRefs", []):
        if ref["referenceType"] == "purl":
            return ref["referenceLocator"].partition("/")[0]
    # Syft writes "acquired package info from the following paths: /a/b, /c/d".
    first = package.get("sourceInfo", "").rpartition(": ")[2].split(", ")[0]
    return first.rpartition("/")[0]


def unlicensed(package):
    return package.get("licenseDeclared", "NOASSERTION") in ("NOASSERTION", "NONE")


def main(sbom_path, numpy_libs_path):
    with open(sbom_path, encoding="utf-8") as f:
        sbom = json.load(f)
    packages, relationships = sbom["packages"], sbom["relationships"]

    filled = 0
    for package in packages:
        license_ = LICENSES.get(origin(package), {}).get(package["name"])
        if license_ and unlicensed(package):
            package.update(licenseDeclared=license_, licenseComments=NOTE)
            filled += 1

    with open(numpy_libs_path, encoding="utf-8") as f:
        paths = f.read().split()
    numpy = [p for p in packages if p["name"] == "numpy" and origin(p) == "pkg:pypi"]
    if paths and not numpy:
        sys.exit("The SBOM has no numpy package to hold the libraries in numpy.libs.")
    for path in paths:
        file_name = path.rpartition("/")[2]
        name = next((lib for lib in NUMPY_LIBS if file_name.startswith(lib)), file_name)
        # SPDX ids allow only letters, digits, "." and "-".
        spdx_id = "SPDXRef-numpy.libs-" + re.sub(r"[^A-Za-z0-9.-]", "-", file_name)
        packages.append(
            {
                "name": name,
                "SPDXID": spdx_id,
                "packageFileName": path,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "sourceInfo": f"bundled in the NumPy wheel: {path}",
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": NUMPY_LIBS.get(name, "NOASSERTION"),
                "licenseComments": NOTE,
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": numpy[0]["SPDXID"],
                "relatedSpdxElement": spdx_id,
                "relationshipType": "CONTAINS",
            }
        )

    image = [
        r["relatedSpdxElement"]
        for r in relationships
        if r["relationshipType"] == "DESCRIBES"
    ]
    missing = [
        f"  {p['name']} ({origin(p)}) {p.get('versionInfo', '')}"
        for p in packages
        if p["SPDXID"] not in image and unlicensed(p)
    ]
    if missing:
        sys.exit(
            f"{len(missing)} package(s) in the image SBOM have no license. Read the"
            " license of each and add it to LICENSES or NUMPY_LIBS in"
            " .github/scripts/complete_image_sbom.py:\n" + "\n".join(missing)
        )

    sbom["creationInfo"]["creators"].append("Tool: complete_image_sbom.py")
    with open(sbom_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, ensure_ascii=False)
    print(f"Filled in {filled} licenses, added {len(paths)} libraries from numpy.libs.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
