# SPDX-License-Identifier: Apache-2.0
"""Tests for the ladder presets (#58, D3): every value as the vendor tables give
it, the keys, and the sample blot's ladder."""

import itertools
import re

from proteia import samples
from proteia.core import ladders
from proteia.core.ladders import PRESETS, PRESETS_VERSION, LadderPreset, ReferenceBand, preset

PAGERULER_PLUS = "PageRuler Plus Prestained Protein Ladder, 10 to 250 kDa"
PAGERULER_PLUS_NUMBERS = ("26619", "26620", "26621")
PAGERULER_PLUS_SOURCE = (
    'Thermo Scientific MAN0011773 Rev. C.01, "Migration Patterns",'
    " https://documents.thermofisher.com/TFS-Assets/LSG/manuals/"
    "MAN0011773_PgRuler_Plus_Prestain_Protein_Lad_UG.pdf"
)
PAGERULER = "PageRuler Prestained Protein Ladder, 10 to 180 kDa"
PAGERULER_NUMBERS = ("26616", "26617", "26618")
PAGERULER_SOURCE = (
    "Thermo Scientific MAN0011772 Rev. C.00,"
    " https://documents.thermofisher.com/TFS-Assets/LSG/manuals/"
    "MAN0011772_PgRuler_Prestain_Protein_Lad_UG.pdf"
)
TRIS_GLYCINE = "Tris-glycine (Laemmli, incl. TGX)"


def _orange(*kda: float) -> tuple[ReferenceBand, ...]:
    return tuple(ReferenceBand(k, "orange") for k in kda)


GREEN_10 = (ReferenceBand(10, "green"),)
EXPECTED = (
    LadderPreset(
        key="pageruler_plus/tris_glycine",
        product=PAGERULER_PLUS,
        catalog_numbers=PAGERULER_PLUS_NUMBERS,
        system=TRIS_GLYCINE,
        kda=(250, 130, 100, 70, 55, 35, 25, 15, 10),
        reference=(*_orange(70, 25), *GREEN_10),
        source=PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/bis_tris_mops",
        product=PAGERULER_PLUS,
        catalog_numbers=PAGERULER_PLUS_NUMBERS,
        system="Bis-Tris + MOPS",
        kda=(185, 115, 80, 65, 50, 30, 25, 15, 10),
        reference=(*_orange(65, 25), *GREEN_10),
        source=PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/bis_tris_mes",
        product=PAGERULER_PLUS,
        catalog_numbers=PAGERULER_PLUS_NUMBERS,
        system="Bis-Tris + MES",
        kda=(190, 115, 80, 70, 50, 30, 25, 15, 10),
        reference=(*_orange(70, 25), *GREEN_10),
        source=PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler_plus/tris_acetate",
        product=PAGERULER_PLUS,
        catalog_numbers=PAGERULER_PLUS_NUMBERS,
        system="Tris-acetate",
        kda=(205, 120, 85, 65, 50, 30, 25),
        reference=_orange(65, 25),
        source=PAGERULER_PLUS_SOURCE,
    ),
    LadderPreset(
        key="pageruler/tris_glycine",
        product=PAGERULER,
        catalog_numbers=PAGERULER_NUMBERS,
        system=TRIS_GLYCINE,
        kda=(180, 130, 100, 70, 55, 40, 35, 25, 15, 10),
        reference=(*_orange(70), *GREEN_10),
        source=PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/bis_tris_mops",
        product=PAGERULER,
        catalog_numbers=PAGERULER_NUMBERS,
        system="Bis-Tris + MOPS",
        kda=(140, 115, 80, 65, 50, 40, 30, 25, 15, 10),
        reference=(*_orange(65), *GREEN_10),
        source=PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/bis_tris_mes",
        product=PAGERULER,
        catalog_numbers=PAGERULER_NUMBERS,
        system="Bis-Tris + MES",
        kda=(140, 115, 80, 70, 50, 40, 30, 25, 15, 10),
        reference=(*_orange(70), *GREEN_10),
        source=PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="pageruler/tris_acetate",
        product=PAGERULER,
        catalog_numbers=PAGERULER_NUMBERS,
        system="Tris-acetate",
        kda=(150, 120, 85, 65, 50, 40, 30, 25),
        reference=_orange(65),
        source=PAGERULER_SOURCE,
    ),
    LadderPreset(
        key="precision_plus_dual_color/tris_glycine",
        product="Precision Plus Protein Dual Color Standards",
        catalog_numbers=("1610374",),
        system="Tris-glycine (Laemmli)",
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10),
        reference=(
            ReferenceBand(75, "pink"),
            ReferenceBand(50, "blue (more intense)"),
            ReferenceBand(25, "pink"),
        ),
        source=(
            "Bio-Rad 4110025 Rev B,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110025B.pdf"
        ),
    ),
    # The two Bio-Rad products the D3 note asks for; Bio-Rad gives one set of
    # values for every gel chemistry.
    LadderPreset(
        key="precision_plus_dual_xtra/any_gel",
        product="Precision Plus Protein Dual Xtra Standards",
        catalog_numbers=("1610377",),
        system="Any gel (Bio-Rad gives one set: Tris-HCl, Tris-Tricine, Bis-Tris MES)",
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10, 5, 2),
        reference=(
            ReferenceBand(75, "pink"),
            ReferenceBand(50, "blue (more intense)"),
            ReferenceBand(25, "pink"),
            ReferenceBand(2, "pink"),
        ),
        source=(
            "Bio-Rad 10018393 Rev D,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/10018393.pdf"
        ),
    ),
    LadderPreset(
        key="precision_plus_all_blue/tris_glycine",
        product="Precision Plus Protein All Blue Standards",
        catalog_numbers=("1610373",),
        system="Tris-glycine (Laemmli)",
        kda=(250, 150, 100, 75, 50, 37, 25, 20, 15, 10),
        reference=(
            ReferenceBand(75, "blue"),
            ReferenceBand(50, "blue"),
            ReferenceBand(25, "blue"),
        ),
        source=(
            "Bio-Rad 4110024 Rev D,"
            " https://www.bio-rad.com/webroot/web/pdf/lsr/literature/4110024.pdf"
        ),
    ),
)


def test_presets_match_the_vendor_tables():
    assert [p.key for p in PRESETS] == [p.key for p in EXPECTED]
    for found, expected in zip(PRESETS, EXPECTED, strict=True):
        # Field by field, so a failure names the preset and the field.
        for field in ("product", "catalog_numbers", "system", "kda", "reference", "source"):
            assert getattr(found, field) == getattr(expected, field), (found.key, field)
    assert PRESETS == EXPECTED


def test_preset_keys_are_unique_and_slashed():
    keys = [p.key for p in PRESETS]
    assert len(keys) == len(set(keys)) == 11
    for key in keys:
        assert re.fullmatch(r"[a-z0-9_]+/[a-z0-9_]+", key), key
        assert preset(key) is next(p for p in PRESETS if p.key == key)
    assert preset("nope") is None
    assert preset("PageRuler Plus Prestained") is None  # a custom name, as the fixture holds


def test_preset_kda_strictly_decrease_and_hold_their_references():
    for p in PRESETS:
        assert all(a > b > 0 for a, b in itertools.pairwise(p.kda)), p.key
        references = [band.kda for band in p.reference]
        assert references, p.key
        assert set(references) <= set(p.kda), p.key
        assert references == sorted(references, reverse=True), p.key  # top to bottom
        assert all(band.colour for band in p.reference), p.key


def test_sample_ladder_is_the_pageruler_plus_tris_glycine_preset():
    ladder = preset("pageruler_plus/tris_glycine")
    assert ladder is not None
    assert samples.LADDER_KDA == ladder.kda
    assert set(samples.LADDER_REFERENCE_KDA) <= {band.kda for band in ladder.reference}


def test_presets_version():
    assert PRESETS_VERSION == ladders.PRESETS_VERSION == 1
