# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
#
# PyInstaller spec for Proteia's Windows bundle (ADR 0003): one folder, with a
# console window (v0.1), holding Proteia.exe and its _internal folder. build.py
# runs it in a build environment made from uv.lock:
#
#   python -m PyInstaller --noconfirm --clean --distpath DIST --workpath WORK proteia.spec
#
# The lists come from bundle.py, next to this file, which the tests check too.
# What the spec does beyond PyInstaller's defaults:
# - collects the web client (proteia/web/static) as package data;
# - limits matplotlib to the Agg, SVG and PDF backends (its hook collects
#   mpl-data, the fonts the charts are drawn with among it);
# - leaves out napari, Qt, tkinter and the build tools (bundle.EXCLUDES), and the
#   Universal CRT DLLs, which Windows 10 and later provide;
# - copies the metadata of every distribution the bundle takes code or data from:
#   the record reads versions from it, and notices.py reads the license texts;
#   the build stops if a bundled top-level name belongs to no distribution;
# - no UPX, and a Windows version resource.
# It writes bundle-manifest.json into the work folder: the distributions and
# their versions, and what was left out.

import importlib.metadata
import json
import os
import sys

sys.path.insert(0, SPECPATH)  # noqa: F821 (PyInstaller defines it)

import bundle  # noqa: E402
from PyInstaller.utils.hooks import collect_data_files, copy_metadata  # noqa: E402
from PyInstaller.utils.win32.versioninfo import (  # noqa: E402
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

import proteia  # noqa: E402

version = proteia.__version__
numbers = bundle.numeric_version(version)

a = Analysis(  # noqa: F821
    [os.path.join(SPECPATH, "proteia_entry.py")],  # noqa: F821
    pathex=[],
    binaries=[],
    datas=collect_data_files("proteia", includes=list(bundle.PACKAGE_DATA)),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": list(bundle.MATPLOTLIB_BACKENDS)}},
    runtime_hooks=[],
    excludes=list(bundle.EXCLUDES),
    noarchive=False,
    optimize=0,
)

left_out = sorted(dest for dest, _, _ in a.binaries if bundle.is_universal_crt(dest))
a.binaries = [entry for entry in a.binaries if not bundle.is_universal_crt(entry[0])]

found, unknown = bundle.bundled_distributions(
    (name for name, _, _ in a.pure),
    (dest for dest, _, _ in list(a.binaries) + list(a.datas)),
    importlib.metadata.packages_distributions(),
    sys.stdlib_module_names,
)
if unknown:
    raise SystemExit(
        "proteia.spec: bundled top-level names that belong to no distribution, so the"
        f" notices could not cover them: {sorted(unknown)}"
    )
dists = sorted(
    name for name in found if bundle.normalize(name) != bundle.normalize(bundle.OWN_DISTRIBUTION)
)
missing = bundle.missing_record_distributions(dists)
if missing:
    raise SystemExit(f"proteia.spec: the record reads versions of {missing}, not bundled")
for dist in dists:
    for source, dest in copy_metadata(dist):
        # direct_url.json would name where the package was installed from.
        a.datas += Tree(source, prefix=dest, excludes=["direct_url.json"])  # noqa: F821

with open(os.path.join(workpath, "bundle-manifest.json"), "w", encoding="utf-8") as f:  # noqa: F821
    manifest = {
        "proteia": version,
        "python": sys.version.split()[0],
        "distributions": {name: importlib.metadata.version(name) for name in dists},
        "top_level": {name: sorted(names) for name, names in sorted(found.items())},
        "universal_crt_left_out": left_out,
    }
    json.dump(manifest, f, indent=1)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=bundle.APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed launchers draw more antivirus false positives
    console=True,  # v0.1 keeps the console window (ADR 0003)
    disable_windowed_traceback=False,
    version=VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=numbers,
            prodvers=numbers,
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", bundle.PUBLISHER),
                            StringStruct("FileDescription", bundle.APP_NAME),
                            StringStruct("FileVersion", version),
                            StringStruct("InternalName", bundle.APP_NAME),
                            StringStruct("LegalCopyright", "Licensed under the Apache License 2.0"),
                            StringStruct("OriginalFilename", bundle.EXE_NAME),
                            StringStruct("ProductName", bundle.APP_NAME),
                            StringStruct("ProductVersion", version),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [0x0409, 0x04B0])]),
        ],
    ),
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=bundle.APP_NAME,
)
