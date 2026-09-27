# SPDX-License-Identifier: Apache-2.0
"""Molecular-weight ladder presets (#58, D3).

A preset is one ladder product in one gel and buffer system: the apparent MWs
of its bands, top to bottom, as the vendor's migration chart gives them for that
system, and its reference bands. A prestained ladder's bands run at apparent
MWs that depend on the buffer system, so a product has one preset per system
its vendor charts.

Keys are ``<product>/<system>`` and stable: ``MwCalibration.ladder`` stores
them. A custom ladder's name holds no ``/``. Choosing a preset copies its MWs
into ``MwCalibration.ladder_kda``, so a project keeps the values it was
calibrated with when a later build corrects a preset; :data:`PRESETS_VERSION`
counts such corrections, and every export record reports it.

Pure data: no numpy and no file access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ReferenceBand:
    """A band the vendor marks out, by colour or intensity, to orient the ladder."""

    kda: float
    colour: str  # as the vendor names it: "orange", "green", "pink", "blue (more intense)"


@dataclass(frozen=True)
class LadderPreset:
    """One ladder product's apparent MWs in one gel and buffer system."""

    key: str  # stable; stored in MwCalibration.ladder
    product: str
    catalog_numbers: tuple[str, ...]
    system: str  # the gel and buffer system the values hold for (the page's label)
    kda: tuple[float, ...]  # apparent MWs, top to bottom
    reference: tuple[ReferenceBand, ...]  # top to bottom
    source: str  # document, revision and URL, verbatim


# Bumped when a preset's value changes; reported in every export record.
PRESETS_VERSION: Final = 1


def _bands(colour: str, *kda: float) -> tuple[ReferenceBand, ...]:
    return tuple(ReferenceBand(k, colour) for k in kda)


# PageRuler Plus Prestained Protein Ladder, 10 to 250 kDa (#26619, #26620, #26621):
# Thermo Scientific MAN0011773 Rev. C.01, "Migration Patterns",
# https://documents.thermofisher.com/TFS-Assets/LSG/manuals/MAN0011773_PgRuler_Plus_Prestain_Protein_Lad_UG.pdf
# Within one buffer system the values are the same at every gel percentage;
# the Tris-acetate list ends where the vendor's lane ends.
_PAGERULER_PLUS = "PageRuler Plus Prestained Protein Ladder, 10 to 250 kDa"
_PAGERULER_PLUS_NUMBERS = ("26619", "26620", "26621")
_PAGERULER_PLUS_SOURCE = (
    'Thermo Scientific MAN0011773 Rev. C.01, "Migration Patterns",'
    " https://documents.thermofisher.com/TFS-Assets/LSG/manuals/"
    "MAN0011773_PgRuler_Plus_Prestain_Protein_Lad_UG.pdf"
)
# PageRuler Prestained Protein Ladder, 10 to 180 kDa (#26616, #26617, #26618):
# Thermo Scientific MAN0011772 Rev. C.00,
# https://documents.thermofisher.com/TFS-Assets/LSG/manuals/MAN0011772_PgRuler_Prestain_Protein_Lad_UG.pdf
_PAGERULER = "PageRuler Prestained Protein Ladder, 10 to 180 kDa"
_PAGERULER_NUMBERS = ("26616", "26617", "26618")
_PAGERULER_SOURCE = (
    "Thermo Scientific MAN0011772 Rev. C.00,"
    " https://documents.thermofisher.com/TFS-Assets/LSG/manuals/"
    "MAN0011772_PgRuler_Prestain_Protein_Lad_UG.pdf"
)
_TRIS_GLYCINE_TGX = "Tris-glycine (Laemmli, incl. TGX)"
_TRIS_GLYCINE = "Tris-glycine (Laemmli)"
_MORE_INTENSE_50 = (ReferenceBand(50, "blue (more intense)"),)

PRESETS: Final[tuple[LadderPreset, ...]] = (
    LadderPreset(
        key="pageruler_plus/tris_glycine",
        product=_PAGERULER_PLUS,
        catalog_numbers=_PAGERULER_PLUS_NUMBERS,
        system=_TRIS_GLYCINE_TGX,
        kda=(250, 130, 100, 70, 55, 35, 25, 15, 10),
        reference=(*_bands("orange", 70, 25), *_bands("green", 10)),
        source=_PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/bis_tris_mops",
        product=_PAGERULER_PLUS,
        catalog_numbers=_PAGERULER_PLUS_NUMBERS,
        system="Bis-Tris + MOPS",
        kda=(185, 115, 80, 65, 50, 30, 25, 15, 10),
        reference=(*_bands("orange", 65, 25), *_bands("green", 10)),
        source=_PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/bis_tris_mes",
        product=_PAGERULER_PLUS,
        catalog_numbers=_PAGERULER_PLUS_NUMBERS,
        system="Bis-Tris + MES",
        kda=(190, 115, 80, 70, 50, 30, 25, 15, 10),
        reference=(*_bands("orange", 70, 25), *_bands("green", 10)),
        source=_PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/tris_acetate",
        product=_PAGERULER_PLUS,
        catalog_numbers=_PAGERULER_PLUS_NUMBERS,
        system="Tris-acetate",
        kda=(205, 120, 85, 65, 50, 30, 25),
        reference=_bands("orange", 65, 25),
        source=_PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler/tris_glycine",
        product=_PAGERULER,
        catalog_numbers=_PAGERULER_NUMBERS,
        system=_TRIS_GLYCINE_TGX,
        kda=(180, 130, 100, 70, 55, 40, 35, 25, 15, 10),
        reference=(*_bands("orange", 70), *_bands("green", 10)),
        source=_PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/bis_tris_mops",
        product=_PAGERULER,
        catalog_numbers=_PAGERULER_NUMBERS,
        system="Bis-Tris + MOPS",
        kda=(140, 115, 80, 65, 50, 40, 30, 25, 15, 10),
        reference=(*_bands("orange", 65), *_bands("green", 10)),
        source=_PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/bis_tris_mes",
        product=_PAGERULER,
        catalog_numbers=_PAGERULER_NUMBERS,
        system="Bis-Tris + MES",
        kda=(140, 115, 80, 70, 50, 40, 30, 25, 15, 10),
        reference=(*_bands("orange", 70), *_bands("green", 10)),
        source=_PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/tris_acetate",
        product=_PAGERULER,
        catalog_numbers=_PAGERULER_NUMBERS,
        system="Tris-acetate",
        kda=(150, 120, 85, 65, 50, 40, 30, 25),
        reference=_bands("orange", 65),
        source=_PAGERULER_SOURCE,
    ),
    # Precision Plus Protein Dual Color Standards (#1610374): Bio-Rad 4110025 Rev B,
    # https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110025B.pdf
    LadderPreset(
        key="precision_plus_dual_color/tris_glycine",
        product="Precision Plus Protein Dual Color Standards",
        catalog_numbers=("1610374",),
        system=_TRIS_GLYCINE,
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10),
        reference=(*_bands("pink", 75), *_MORE_INTENSE_50, *_bands("pink", 25)),
        source=(
            "Bio-Rad 4110025 Rev B,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110025B.pdf"
        ),
    ),
    # Two more Bio-Rad products, as the D3 note asks. Bio-Rad gives one set of
    # values for every gel chemistry, so each is one preset.
    # Precision Plus Protein Dual Xtra Standards (#1610377): Bio-Rad 10018393 Rev D,
    # https://www.bio-rad.com/webroot/web/pdf/lsr/literature/10018393.pdf
    # (linear on Criterion Tris-HCl, Tris-Tricine and Criterion XT Bis-Tris MES gels;
    # 2, 25 and 75 kD pink, a more intense 50 kD band, the rest blue).
    LadderPreset(
        key="precision_plus_dual_xtra/any_gel",
        product="Precision Plus Protein Dual Xtra Standards",
        catalog_numbers=("1610377",),
        system="Any gel (Bio-Rad gives one set: Tris-HCl, Tris-Tricine, Bis-Tris MES)",
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10, 5, 2),
        reference=(*_bands("pink", 75), *_MORE_INTENSE_50, *_bands("pink", 25, 2)),
        source=(
            "Bio-Rad 10018393 Rev D,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/10018393.pdf"
        ),
    ),
    # Precision Plus Protein All Blue Standards (#1610373): Bio-Rad 4110024 Rev D,
    # https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110024.pdf
    # (MWs "confirmed by migration in a Laemmli SDS-PAGE system"; reference bands
    # 25, 50 and 75 kD, all blue).
    LadderPreset(
        key="precision_plus_all_blue/tris_glycine",
        product="Precision Plus Protein All Blue Standards",
        catalog_numbers=("1610373",),
        system=_TRIS_GLYCINE,
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10),
        reference=_bands("blue", 75, 50, 25),
        source=(
            "Bio-Rad 4110024 Rev D,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110024.pdf"
        ),
    ),
)

_BY_KEY: Final = {preset.key: preset for preset in PRESETS}


def preset(key: str) -> LadderPreset | None:
    """The preset stored under ``key``; None for a custom ladder or an unknown key."""
    return _BY_KEY.get(key)
