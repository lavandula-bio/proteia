# SPDX-License-Identifier: Apache-2.0
"""The third-party notices of the Windows bundle (ADR 0003).

``python notices.py BUNDLE --version VERSION [--out FILE] [--inno-license FILE]``
writes ``BUNDLE/THIRD_PARTY_NOTICES.txt`` (or ``FILE``): what the bundle holds
besides Proteia ``VERSION``, and every license text it needs:

1. the Python runtime: the interpreter's own ``LICENSE.txt`` (the PSF license,
   the conditions for the Microsoft Distributable Code in a Windows build, and
   the libraries it names), CPython's list of incorporated software
   (``licenses/cpython-X.Y-incorporated.txt``, with OpenSSL, expat, libffi, zlib
   and libmpdec) and liblzma (``licenses/xz-liblzma.txt``);
2. the Visual C++ runtime DLLs, wherever they are in the bundle, each with where
   it came from (the Python build, the build machine's System32, a package's
   wheel), under a pointer to Microsoft's terms for redistributing them;
3. every Python package, from the metadata the spec copied into the bundle:
   name, version, license and every license file it carries (NumPy's, SciPy's
   and Pillow's name the libraries their wheels bundle);
4. matplotlib's fonts, with the license files in its font folders;
5. PyInstaller's bootloader and runtime hooks (its ``COPYING.txt``);
6. with ``--inno-license``, Inno Setup, which made the setup program.

It runs with the build environment's Python (``build.py`` does so), since the
runtime and the packages it describes are that environment's. It audits the
bundle first and exits 1, writing nothing, when an entry of ``_internal`` is
accounted for by no package's ``RECORD``, the standard library or a runtime, or
when a package carries no license text: every file shipped needs a notice.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import platform
import re
import sys
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath

import bundle

HERE = Path(__file__).resolve().parent
SUPPLEMENTS = HERE / "licenses"
NOTICES_FILE = "THIRD_PARTY_NOTICES.txt"
CONTENTS = "_internal"  # PyInstaller's folder next to the executable
RULE = "=" * 78
THIN = "-" * 78
# Files in a .dist-info folder that are metadata, not license texts.
_METADATA_FILES = frozenset(
    {"METADATA", "RECORD", "INSTALLER", "REQUESTED", "WHEEL", "DELVEWHEEL", "zip-safe"}
)
_LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|notice|authors?)", re.IGNORECASE)
FONT_FOLDER = PurePosixPath("matplotlib/mpl-data/fonts")
# Microsoft's page on redistributing the Visual C++ runtime: the license terms
# that govern it and the list of files they allow.
VC_REDIST_TERMS = "https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files"


class NoticesError(Exception):
    """The bundle holds something the notices cannot account for."""


@dataclass(frozen=True)
class Package:
    """A distribution in the bundle, from its ``.dist-info`` folder."""

    name: str
    version: str
    license: str  # a short name: the SPDX expression, the classifiers, or the License field
    urls: tuple[str, ...]
    texts: tuple[tuple[str, str], ...]  # (file name in the .dist-info folder, text)
    top_level: frozenset[str] = field(default_factory=frozenset)


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _record_top_level(record: str) -> frozenset[str]:
    """The top-level entries a ``RECORD`` lists: first parts of the paths inside
    the installation folder, the ``.dist-info`` folder itself left out."""
    names = set()
    for row in csv.reader(io.StringIO(record)):
        if not row or not row[0]:
            continue
        parts = PurePosixPath(row[0].replace("\\", "/")).parts
        if parts[0] == ".." or parts[0].endswith(".dist-info"):
            continue
        names.add(parts[0])
    return frozenset(names)


def _short_license(meta: object) -> str:
    expression = meta.get("License-Expression")  # type: ignore[attr-defined]
    if expression:
        return " ".join(expression.split())
    classifiers = [
        value.split("::")[-1].strip()
        for value in meta.get_all("Classifier") or []  # type: ignore[attr-defined]
        if value.startswith("License ::") and value.count("::") > 1
    ]
    if classifiers:
        return ", ".join(classifiers)
    text = (meta.get("License") or "").strip()  # type: ignore[attr-defined]
    if text and "\n" not in text and len(text) <= 80:
        return text
    return "see the license text below"


def read_package(folder: Path) -> Package:
    """The package whose ``.dist-info`` folder is ``folder``."""
    meta = BytesParser(policy=compat32).parsebytes((folder / "METADATA").read_bytes())
    texts: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if not path.is_file() or relative in _METADATA_FILES:
            continue
        if relative.startswith("licenses/") or _LICENSE_NAME.match(path.name):
            texts[relative] = _read_text(path)
    if not texts:
        body = (meta.get("License") or "").strip()
        if "\n" in body:  # the whole text in the metadata, as older packages give it
            texts["METADATA (License)"] = body
    urls = [value.split(",", 1)[-1].strip() for value in meta.get_all("Project-URL") or []]
    if meta.get("Home-page"):
        urls.insert(0, meta["Home-page"].strip())
    record = folder / "RECORD"
    top = _record_top_level(_read_text(record)) if record.is_file() else frozenset()
    return Package(
        name=meta["Name"],
        version=meta["Version"],
        license=_short_license(meta),
        urls=tuple(dict.fromkeys(urls))[:2],
        texts=tuple(texts.items()),
        top_level=top,
    )


def read_packages(contents: Path) -> list[Package]:
    """Every package whose metadata the bundle's ``contents`` folder holds."""
    folders = sorted(contents.glob("*.dist-info"), key=lambda path: path.name.lower())
    return [read_package(folder) for folder in folders]


def audit(contents: Path, packages: Sequence[Package], stdlib: Iterable[str]) -> list[str]:
    """The entries of ``contents`` nothing accounts for: neither a package's
    ``RECORD``, Proteia's own package, a standard-library extension module, nor
    the Python or Visual C++ runtime; and the packages without a license text."""
    stdlib = set(stdlib)
    claimed = set().union(*(package.top_level for package in packages))
    problems = [f"{p.name} {p.version} carries no license text" for p in packages if not p.texts]
    for entry in sorted(contents.iterdir(), key=lambda path: path.name.lower()):
        name = entry.name
        if name.endswith(".dist-info") or name in claimed or name == bundle.OWN_DISTRIBUTION:
            continue
        if entry.is_file() and (
            bundle.PYTHON_RUNTIME_FILES.match(name) or bundle.VC_RUNTIME_FILES.match(name)
        ):
            continue
        if entry.is_file() and name.endswith(".pyd") and name.split(".")[0] in stdlib:
            continue
        problems.append(f"{CONTENTS}/{name} is accounted for by no package or runtime")
    return problems


def vc_runtime_files(root: Path) -> list[str]:
    """The Visual C++ runtime DLLs anywhere in the bundle, as paths from ``root``;
    NumPy and SciPy carry copies under names of their own (``msvcp140-<hash>.dll``)."""
    pattern = re.compile(r"^(vcruntime140(_1)?|msvcp140(_\w+)?)([-.][^\\/]*)?\.dll$", re.IGNORECASE)
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.dll")
        if pattern.match(path.name)
    )


def _same_file_in(folder: Path, file: Path) -> bool:
    """Whether ``folder`` holds a file of ``file``'s name (in any case) with the
    same bytes."""
    try:
        twin = next((p for p in folder.iterdir() if p.name.lower() == file.name.lower()), None)
    except OSError:
        return False
    return twin is not None and twin.is_file() and twin.read_bytes() == file.read_bytes()


def vc_runtime_sources(
    root: Path,
    files: Sequence[str],
    packages: Sequence[Package],
    python_dir: Path,
    system_dir: Path,
) -> list[tuple[str, str]]:
    """Each of the Visual C++ runtime ``files`` (paths from ``root``) with where
    it came from: a package's wheel (a folder of the package in ``_internal``),
    the Python build (the same bytes in ``python_dir``, the interpreter's
    folder), or the build machine's ``system_dir`` (System32, which the Visual
    C++ Redistributable installs into and PyInstaller copies from)."""
    owners = {name: p for p in packages for name in p.top_level}
    sources = []
    for name in files:
        parts = PurePosixPath(name).parts
        owner = owners.get(parts[1]) if len(parts) > 2 and parts[0] == CONTENTS else None
        if owner is not None:
            source = f"from the {owner.name} {owner.version} wheel (section 3)"
        elif _same_file_in(python_dir, root / name):
            source = "from the Python runtime's Windows build (section 1)"
        elif _same_file_in(system_dir, root / name):
            source = (
                "from the build machine's System32 folder, where the Visual C++"
                " Redistributable installs it; the Python build does not include it"
            )
        else:
            source = "source not identified"
        sources.append((name, source))
    return sources


def font_notes(contents: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """matplotlib's font files, counted per folder, and the license and read-me
    files in its font folders."""
    fonts = contents / FONT_FOLDER
    counts = []
    texts = []
    if fonts.is_dir():
        for folder in sorted(path for path in fonts.iterdir() if path.is_dir()):
            files = [path for path in folder.iterdir() if path.suffix.lower() in (".ttf", ".afm")]
            counts.append(f"{FONT_FOLDER.as_posix()}/{folder.name}: {len(files)} font files")
        for path in sorted(fonts.rglob("*")):
            if path.is_file() and (_LICENSE_NAME.match(path.name) or path.stem.lower() == "readme"):
                texts.append((path.relative_to(contents).as_posix(), _read_text(path)))
    return counts, texts


@dataclass(frozen=True)
class Runtime:
    """The Python runtime the bundle holds: the build environment's."""

    version: str
    libraries: tuple[tuple[str, str | None], ...]  # (name, version if known) built into it
    license_text: str
    supplements: tuple[tuple[str, str], ...]  # (file name, text)


def build_runtime() -> Runtime:
    """This interpreter as a :class:`Runtime`: its version, the versions of the
    libraries built into it, its ``LICENSE.txt`` and the supplements for its
    minor version. ``NoticesError`` when one is missing."""
    import decimal
    import pyexpat
    import ssl
    import zlib

    license_file = Path(sys.base_prefix) / "LICENSE.txt"
    if not license_file.is_file():
        raise NoticesError(f"the Python runtime has no license file at {license_file}")
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    names = (f"cpython-{minor}-incorporated.txt", "xz-liblzma.txt")
    missing = [name for name in names if not (SUPPLEMENTS / name).is_file()]
    if missing:
        raise NoticesError(
            f"no {missing} in {SUPPLEMENTS}: add the notices of Python {minor}'s incorporated"
            " software (see cpython-3.13-incorporated.txt for how it was made)"
        )
    libraries = (
        ("OpenSSL", ssl.OPENSSL_VERSION.split()[1]),
        ("zlib", zlib.ZLIB_RUNTIME_VERSION),
        ("expat", pyexpat.EXPAT_VERSION.removeprefix("expat_")),
        ("libmpdec", decimal.__libmpdec_version__),
        ("libffi", None),
        ("bzip2", None),
        ("liblzma", None),
    )
    return Runtime(
        version=platform.python_version(),
        libraries=libraries,
        license_text=_read_text(license_file),
        supplements=tuple((name, _read_text(SUPPLEMENTS / name)) for name in names),
    )


def freetype_versions() -> list[str]:
    """Where FreeType is built in, with its version, as far as the packages say."""
    found = []
    try:
        from PIL import features

        found.append(f"Pillow (FreeType {features.version('freetype2')})")
    except Exception:  # a missing or older Pillow: named without a version
        found.append("Pillow")
    try:
        from matplotlib import ft2font

        found.append(f"matplotlib (FreeType {ft2font.__freetype_version__})")
    except Exception:
        found.append("matplotlib")
    return found


def pyinstaller_license() -> tuple[str, str]:
    """PyInstaller's version and license text (``COPYING.txt``)."""
    import importlib.metadata

    dist = importlib.metadata.distribution("pyinstaller")
    texts = [
        file.read_text(encoding="utf-8")
        for file in dist.files or []
        if PurePosixPath(str(file)).name.upper().startswith("COPYING")
    ]
    if not texts:
        raise NoticesError("PyInstaller's COPYING.txt was not found")
    return dist.version, "\n".join(texts)


def _block(title: str, text: str) -> list[str]:
    return [THIN, title, THIN, "", text.strip("\n"), ""]


def _para(*sentences: str) -> list[str]:
    """``sentences`` as one paragraph wrapped at 78 columns, then a blank line."""
    return [*textwrap.wrap(" ".join(sentences), 78), ""]


def _item(text: str) -> list[str]:
    """A list item wrapped at 78 columns, its later lines indented."""
    return textwrap.wrap(text, 78, subsequent_indent="   ")


def _libraries(runtime: Runtime) -> str:
    return ", ".join(
        name if version is None else f"{name} {version}" for name, version in runtime.libraries
    )


def render(
    *,
    proteia_version: str,
    packages: Sequence[Package],
    runtime: Runtime,
    vc_files: Sequence[tuple[str, str]],
    fonts: tuple[list[str], list[tuple[str, str]]],
    pyinstaller: tuple[str, str],
    freetype: Sequence[str],
    inno_license: str | None = None,
) -> str:
    """The notices file's text."""
    title = f"Third-party notices for Proteia {proteia_version} for Windows"
    counts, font_texts = fonts
    out = [title, "=" * len(title), ""]
    out += _para(
        "Proteia is licensed under the Apache License, Version 2.0: see LICENSE.txt",
        "next to this file. This installation also holds the third-party software",
        "listed below, each under its own terms; the license texts follow the list.",
        "This file was generated from the files in the bundle when it was built.",
    )
    out += _para(
        f"FreeType is built into {' and '.join(freetype)}, under the FreeType License:",
        "portions of this software are copyright (c) The FreeType Project",
        "(www.freetype.org). All rights reserved.",
    )
    out += ["Contents", "--------", ""]
    out += _item(
        f"1. Python runtime: CPython {runtime.version} (PSF-2.0), with {_libraries(runtime)}."
    )
    out += _item("2. Microsoft Visual C++ runtime: " + ", ".join(n for n, _ in vc_files) + ".")
    out.append(f"3. Python packages ({len(packages)}):")
    width = max((len(f"{p.name} {p.version}") for p in packages), default=0)
    out += [f"     {f'{p.name} {p.version}':<{width}}  {p.license}" for p in packages]
    out += _item("4. Fonts, in matplotlib: " + "; ".join(counts) + ".")
    out += _item(
        f"5. PyInstaller {pyinstaller[0]}: the launcher in Proteia.exe and the runtime hooks."
    )
    if inno_license is not None:
        out.append("6. Inno Setup: the setup program that installed Proteia.")
    out.append("")

    out += [RULE, f"1. Python runtime: CPython {runtime.version}", RULE, ""]
    out += _para(
        "The Python runtime is python3*.dll, base_library.zip, the standard library's",
        "extension modules (*.pyd in _internal) and the libraries they load",
        f"(libcrypto, libssl, libffi). Built into it: {_libraries(runtime)}.",
    )
    out += _block("LICENSE.txt of the Python runtime", runtime.license_text)
    for name, text in runtime.supplements:
        out += _block(name, text)

    out += [RULE, "2. Microsoft Visual C++ runtime", RULE, ""]
    out += _para(
        "These files are the Microsoft Visual C++ runtime, copyright Microsoft",
        "Corporation. Their redistribution is subject to Microsoft's license terms",
        'for the Visual C++ runtime: the "Distributable Code" terms of the',
        "Microsoft Visual Studio license terms (see",
        f"{VC_REDIST_TERMS}).",
        "The conditions in the Python runtime's LICENSE.txt (section 1, \"Additional",
        'Conditions for this Windows binary build") concern the Microsoft code in',
        "the Python build only. The files, and where each came from:",
    )
    for name, source in vc_files:
        out.append(f"  {name}")
        out += textwrap.wrap(source, 78, initial_indent="     ", subsequent_indent="     ")
    out.append("")

    out += [RULE, f"3. Python packages ({len(packages)})", RULE, ""]
    for package in packages:
        heading = f"{package.name} {package.version}"
        out += [heading, "~" * len(heading), f"License: {package.license}"]
        out += [f"Home: {url}" for url in package.urls]
        out.append("")
        for name, text in package.texts:
            out += _block(f"{package.name}: {name}", text)

    out += [RULE, "4. Fonts, in matplotlib", RULE, ""]
    out += _para("matplotlib's license (section 3) names its fonts and their terms too.")
    out += [*(f"  {line}" for line in counts), ""]
    for name, text in font_texts:
        out += _block(name, text)

    out += [RULE, f"5. PyInstaller {pyinstaller[0]}", RULE, ""]
    out += _para(
        "Proteia.exe is PyInstaller's bootloader with Proteia's archive appended, and",
        "the bundle holds PyInstaller's runtime hooks (pyi_rth_*). PyInstaller's",
        "license text, which gives the terms for both (the bootloader's exception",
        "and the runtime hooks' license):",
    )
    out += _block("PyInstaller: COPYING.txt", pyinstaller[1])

    if inno_license is not None:
        out += [RULE, "6. Inno Setup", RULE, ""]
        out += _para(
            "The setup program that installed Proteia, and its uninstaller, were made",
            "with Inno Setup (https://jrsoftware.org/isinfo.php). Its license:",
        )
        out += _block("Inno Setup: license.txt", inno_license)
    return "\n".join(out).rstrip("\n") + "\n"


def generate(root: Path, proteia_version: str, inno_license: Path | None = None) -> str:
    """The notices of the bundle folder ``root`` (``Proteia.exe`` and
    ``_internal``), after its audit; ``NoticesError`` lists what failed it."""
    if inno_license is not None and not inno_license.is_file():
        raise NoticesError(f"Inno Setup's license was not found at {inno_license}")
    contents = root / CONTENTS
    packages = read_packages(contents)
    problems = audit(contents, packages, sys.stdlib_module_names)
    if problems:
        raise NoticesError("the bundle cannot be covered:\n  " + "\n  ".join(problems))
    system_dir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return render(
        proteia_version=proteia_version,
        packages=packages,
        runtime=build_runtime(),
        vc_files=vc_runtime_sources(
            root, vc_runtime_files(root), packages, Path(sys.base_prefix), system_dir
        ),
        fonts=font_notes(contents),
        pyinstaller=pyinstaller_license(),
        freetype=freetype_versions(),
        inno_license=None if inno_license is None else _read_text(inno_license),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path, help="the bundle folder (Proteia.exe, _internal)")
    parser.add_argument("--version", required=True, help="Proteia's version")
    parser.add_argument("--out", type=Path, help=f"where to write (default: BUNDLE/{NOTICES_FILE})")
    parser.add_argument("--inno-license", type=Path, help="Inno Setup's license.txt")
    args = parser.parse_args(argv)
    try:
        text = generate(args.bundle, args.version, args.inno_license)
    except NoticesError as exc:
        print(f"notices: {exc}", file=sys.stderr)
        return 1
    out = args.out or args.bundle / NOTICES_FILE
    out.write_bytes(text.encode("utf-8"))
    print(f"notices: wrote {out} ({len(text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
