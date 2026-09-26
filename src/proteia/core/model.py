# SPDX-License-Identifier: Apache-2.0
"""Core data model: the project as it is saved in ``project.json``.

GUI-independent (see ``docs/adr/0001``), with no filesystem access and no numpy;
saving, loading and the content hash live in :mod:`proteia.core.storage`.

Hierarchy. A :class:`Project` holds one :class:`Batch` (one run): its lane table,
the reference condition, its membranes and its proteins. A :class:`Membrane` is one
physical blot: its images (chemiluminescence exposures, reprobes, the visible-light
marker, merged overlays) and one molecular-weight calibration. A :class:`Protein`
is one protein quantified on one image: per lane and expected band, it has a
:class:`Band` (a box), an :class:`UndetectedBand` (a detector measured the lane
and the band stayed below its detection limit), or neither (not measured). A
reprobe is simply another image of the same membrane. Objects refer to each
other by stable, project-unique ids (``mem-N``, ``img-N``, ``prot-N``,
``band-N``) and to lanes by index, never by list position or name; a
not-detected record has no id of its own, only its protein, lane and band index.

The stored numbers (each band's net, each image's background) are the raw data.
Normalization and statistics combine several proteins downstream in
:mod:`proteia.core.analyze`, whose ``Batch`` is built from this module's ``Batch``.

Models are mutable. Edits go through :func:`apply_change`, which works on a copy
and re-validates the whole tree, so a failed edit leaves the project untouched.

History. ``Project.log`` records every committed change, oldest first: one
immutable :class:`LogEntry` each, with its time, action, parameters, the Proteia
version that made it and the content hash it left. The log is not content: it
says how the data got there, not what it is, so the content hash leaves it out.
A change never edits the log; the session appends the entry when it commits.

Geometry convention: boxes are axis-aligned and anchored at their top-left
corner in image pixel coordinates (numpy ``image[y, x]``). All boxes of one
protein share its box size, so they have equal area; only their positions vary.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

# Bumped, with a registered migration, by every change to the saved form after
# the v0.1 tag (see proteia.core.storage). Stays 1 until then.
SCHEMA_VERSION: Final = 1
# Stored image suffixes: what the import dialog accepts today (#45 may change it).
IMAGE_SUFFIXES: Final = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class _Model(BaseModel):
    """Base of every saved class: unknown keys and NaN/Infinity are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _positive_zero(v: float) -> float:
    # -0.0 -> 0.0, so equal models give equal bytes and hashes.
    return v + 0.0


def _non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("must not be blank")
    return v


def _plain_file_name(v: str) -> str:
    # A base name only: it is metadata and never becomes a path.
    if not v.strip() or any(c in v for c in "/\\\x00") or v in {".", ".."}:
        raise ValueError("original name must be a plain file name")
    return v


def _real_time(v: str) -> str:
    datetime.fromisoformat(v)  # the pattern allows 2026-02-30; this rejects it
    return v


Finite = Annotated[float, AfterValidator(_positive_zero)]  # finiteness comes from the config
NonNegative = Annotated[Finite, Field(ge=0)]
Kda = Annotated[Finite, Field(gt=0)]  # molecular weight in kDa
# Ids are lowercase (Windows file names ignore case) and hold no path characters.
# The default (Rust) regex engine does not let ``$`` match before a trailing newline.
_N = r"[1-9][0-9]{0,8}"
MembraneId = Annotated[str, StringConstraints(pattern=rf"^mem-{_N}$")]
ImageId = Annotated[str, StringConstraints(pattern=rf"^img-{_N}$")]
ProteinId = Annotated[str, StringConstraints(pattern=rf"^prot-{_N}$")]
BandId = Annotated[str, StringConstraints(pattern=rf"^band-{_N}$")]
IdPrefix = Literal["mem", "img", "prot", "band"]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
# A stable snake_case code: warning codes and action names.
_CODE = r"^[a-z][a-z0-9_]{0,63}$"
WarningCode = Annotated[str, StringConstraints(pattern=_CODE)]
# Text is stored exactly as given: no Unicode normalization or trimming here.
ProteinName = Annotated[str, AfterValidator(_non_blank)]
OriginalName = Annotated[str, StringConstraints(max_length=255), AfterValidator(_plain_file_name)]
# UTC with milliseconds and a Z, as format_timestamp writes it: one spelling per
# instant, so times sort as text. [0-9], not \d: pydantic's Rust regex lets \d
# match fullwidth and Arabic-Indic digits.
_TIME = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
Timestamp = Annotated[str, StringConstraints(pattern=_TIME), AfterValidator(_real_time)]
ActionName = Annotated[str, StringConstraints(pattern=_CODE)]
SoftwareVersion = Annotated[str, StringConstraints(pattern=r"^[0-9A-Za-z][0-9A-Za-z.+!_-]{0,63}$")]


def format_timestamp(moment: datetime) -> str:
    """UTC with milliseconds and a Z, e.g. ``"2026-09-26T08:15:30.123Z"``
    (truncated, not rounded). Raises ``ValueError`` for a naive datetime."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("the clock must return an aware datetime")
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class UnknownIdError(LookupError):
    """No object with this id in the batch."""


# --- Enums: their values are the file format ---


class Role(StrEnum):
    TARGET = "target"
    LOADING_CONTROL = "loading control"


class ImageKind(StrEnum):
    CHEMILUMINESCENCE = "chemiluminescence"
    VISIBLE_MARKER = "visible_marker"  # visible-light photo of the ladder and membrane
    MERGED = "merged"  # imager overlay of chemiluminescence and marker


class Polarity(StrEnum):
    DARK_ON_LIGHT = "dark_on_light"
    LIGHT_ON_DARK = "light_on_dark"

    @property
    def dark_on_light(self) -> bool:
        """The flag ``quantify.net_signal`` and ``grow.grow_box`` take."""
        return self is Polarity.DARK_ON_LIGHT


class ProposalSource(StrEnum):
    """How a band's box was placed, or which detector wrote a not-detected record."""

    CLICK = "click"  # seed click grown by grow.grow_box
    ROW_BOX = "row_box"  # detection inside a dragged row box (#51)
    MW_GUIDED = "mw_guided"  # detection inside the row the calibration predicts (#58)
    MANUAL = "manual"  # placed or drawn by the user with no detection


# The sources that run a detector over a lane slot, so they alone can find nothing there.
DETECTING_SOURCES: Final = frozenset({ProposalSource.ROW_BOX, ProposalSource.MW_GUIDED})


class UndetectedReason(StrEnum):
    """Why a lane holds a not-detected record."""

    BELOW_DETECTION_LIMIT = "below_detection_limit"  # a detector measured the slot: snr < threshold


class CalibrationPointSource(StrEnum):
    VISIBLE_MARKER = "visible_marker"  # ladder band on a visible-light marker (or merged) image
    CHEMILUMINESCENCE_MARKER = "chemiluminescence_marker"  # faint marker on a signal image
    STRIP_EDGE = "strip_edge"  # known-MW edge of a cut membrane strip


class FitMethod(StrEnum):
    LOG_LINEAR = "log_linear"  # log(MW) fitted linearly against vertical position (#58)


# --- Geometry and the lane table ---

# (x0, y0, x1, y1), half-open on the high edge.
Rect = tuple[int, int, int, int]


def overlaps(a: Rect, b: Rect) -> bool:
    """True if two axis-aligned rectangles share any area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


class BoxSize(_Model):
    """The uniform ROI box size shared by every box of one protein (equal area)."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def area(self) -> int:
        return self.width * self.height


class Lane(_Model):
    """One gel lane.

    ``index`` is the stable key used downstream to join analyses (e.g. a target
    against its loading control) by sample. ``label`` is a human-facing name
    (a lane number or a condition name).

    ``sample`` names the *biological* sample this lane belongs to: lanes sharing
    the same ``(label, sample)`` are *technical* repeats of one sample and are
    averaged before statistics, so they do not inflate n. Lanes with the same
    ``label`` but different ``sample`` are biological replicates (the real n).
    ``included`` excludes presentation-only lanes from quantification. ``metadata``
    carries arbitrary ``item: content`` annotations (any may become a grouping
    axis later).
    """

    index: int = Field(ge=0)
    label: str
    sample: str | None = None
    included: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)


class Box(_Model):
    """An axis-aligned box anchored at its top-left corner (image coords)."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)

    def rect(self, size: BoxSize) -> Rect:
        return self.x, self.y, self.x + size.width, self.y + size.height


class Region(_Model):
    """An axis-aligned rectangle in image pixels, half-open on the high edge."""

    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int  # beyond x0 (checked below), so positive
    y1: int  # beyond y0, likewise

    @model_validator(mode="after")
    def _check_not_empty(self) -> Region:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("a region must have a positive width and height")
        return self

    def rect(self) -> Rect:
        return self.x0, self.y0, self.x1, self.y1


# --- Images and the molecular-weight calibration ---


class ImageWarning(_Model):
    """A problem found when the image was imported (codes are defined by #45)."""

    code: WarningCode  # e.g. "lossy_format", "color_channels_differ"
    message: Annotated[str, StringConstraints(min_length=1)]


class ImageRef(_Model):
    """One image of a membrane; its pixels live in ``images/<file>`` in the project folder."""

    id: ImageId
    file: str  # stored name: the id plus a lowercase suffix from IMAGE_SUFFIXES
    original_name: OriginalName  # the imported file's name, e.g. "β-actin 10 µM.tif"; metadata
    kind: ImageKind
    marker_image_id: ImageId | None = None  # chemiluminescence image -> the marker taken with it
    sha256: Sha256  # of the stored file's bytes
    width: int = Field(gt=0)  # of the analysis array, in pixels
    height: int = Field(gt=0)
    bit_depth: int | None = Field(default=None, ge=8, le=16)  # None = not recorded yet (#45)
    polarity: Polarity  # required: the import chooses it; there is no silent default
    background: Finite  # quantify.estimate_background of the analysis array
    import_warnings: list[ImageWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_file_and_marker(self) -> ImageRef:
        if self.file not in {self.id + suffix for suffix in IMAGE_SUFFIXES}:
            raise ValueError(
                f"image {self.id}: file name must be the image id plus an image suffix"
            )
        if self.marker_image_id is not None:
            if self.kind is not ImageKind.CHEMILUMINESCENCE:
                raise ValueError(
                    f"image {self.id}: only a chemiluminescence image has a marker image"
                )
            if self.marker_image_id == self.id:
                raise ValueError(f"image {self.id}: an image cannot be its own marker image")
        return self


class CalibrationPoint(_Model):
    """One known molecular weight at a vertical position."""

    image_id: ImageId  # the image whose pixel rows y is measured in (same membrane)
    y: NonNegative  # pixels from the top of that image; sub-pixel allowed
    mw: Kda
    source: CalibrationPointSource


class MwCalibration(_Model):
    """A membrane's molecular-weight calibration: ladder points and the fit (#58)."""

    ladder: str | None = None  # ladder product: a preset key or a custom name
    points: list[CalibrationPoint] = Field(default_factory=list)
    fit_method: FitMethod = FitMethod.LOG_LINEAR
    fit_quality: Finite | None = None  # set by #58; None = no curve fitted

    @model_validator(mode="after")
    def _canonical_points(self) -> MwCalibration:
        # Point order carries no meaning: sort so equal calibrations give equal bytes.
        self.points.sort(key=lambda p: (p.y, p.mw, p.source.value, p.image_id))
        if self.fit_quality is not None and len(self.points) < 2:
            raise ValueError("a calibration fit needs at least two points")
        return self


class Membrane(_Model):
    """One physical blot: its images (exposures, reprobes, marker, merged) and one
    molecular-weight calibration, which applies to every image of the membrane."""

    id: MembraneId
    images: list[ImageRef] = Field(default_factory=list)
    calibration: MwCalibration = Field(default_factory=MwCalibration)

    @model_validator(mode="after")
    def _check_references(self) -> Membrane:
        images = {image.id: image for image in self.images}
        for image in self.images:
            if image.marker_image_id is None:
                continue
            marker = images.get(image.marker_image_id)
            if marker is None or marker.kind is not ImageKind.VISIBLE_MARKER:
                raise ValueError(
                    f"image {image.id}: marker image {image.marker_image_id!r} is not"
                    f" a visible-light marker image of membrane {self.id}"
                )
        for point in self.calibration.points:
            image = images.get(point.image_id)
            if image is None:
                raise ValueError(
                    f"membrane {self.id}: calibration point on {point.image_id!r},"
                    " which is not an image of this membrane"
                )
            if point.y > image.height:
                raise ValueError(
                    f"membrane {self.id}: calibration point at y={point.y}"
                    f" is below the bottom of {image.id}"
                )
        return self


# --- Proteins and their bands ---


class Band(_Model):
    """One box of one protein in one lane."""

    id: BandId
    lane_index: int = Field(ge=0)  # index into Batch.lanes: the band's stored identity
    band_index: int = Field(default=0, ge=0)  # which expected band; 0 for single-band proteins
    box: Box  # top-left in the protein's image; its size is Protein.box_size
    net: NonNegative  # quantify.net_signal at placement; load never recomputes it
    apparent_mw: Kda | None = None  # from the calibration (#58); None = not computed
    clipped: bool | None = None  # #44; None = not checked (not "passed")
    source: ProposalSource
    manually_edited: bool = False  # moved or edited by the user after it was proposed


class UndetectedBand(_Model):
    """An expected band a detector looked for in one lane and did not find: the
    lane was measured, and the band is below the detection limit.

    It carries no value, so statistics leave the lane out as they do a lane with
    no box; results and exports can still tell "not measured" from "below the
    detection limit". ``threshold`` is the limit in force when the record was
    written, so each record backs its own claim. The detector's window comes from
    code constants, not from the protein's box size, so a size change never makes
    ``snr`` stale.
    """

    lane_index: int = Field(ge=0)  # index into Batch.lanes, as for a band
    band_index: int = Field(default=0, ge=0)  # which expected band
    reason: UndetectedReason
    snr: Finite  # the detector's statistic for this slot; any sign
    threshold: Annotated[Finite, Field(gt=0)]  # the detection limit it was compared with
    region: Region  # the slot the detector measured, in the protein's image
    source: ProposalSource  # one of DETECTING_SOURCES

    @model_validator(mode="after")
    def _check_measurement(self) -> UndetectedBand:
        if self.source not in DETECTING_SOURCES:
            detectors = " or ".join(sorted(source.value for source in DETECTING_SOURCES))
            raise ValueError(
                f"a not-detected record comes from a detector ({detectors}),"
                f" not {self.source.value!r}"
            )
        if not self.snr < self.threshold:
            raise ValueError(
                f"a not-detected record's snr {self.snr} is not below its threshold"
                f" {self.threshold}"
            )
        return self


class Protein(_Model):
    """One protein quantified on one image (the successor of ``Analysis``).

    Every band shares ``box_size``, the effective size that was quantified. A
    target normalizes against ``loading_control_ids``; an empty list means the
    batch's single loading control. Each (lane, band index) holds a band, a
    not-detected record, or neither; a record only ever concerns an expected band.
    """

    id: ProteinId
    name: ProteinName  # unique in the batch (exact match)
    role: Role
    image_id: ImageId  # an image of any membrane of the batch
    loading_control_ids: list[ProteinId] = Field(default_factory=list)  # targets only
    expected_mw: Kda | None = None
    expected_band_count: int = Field(default=1, ge=1)
    mw_tolerance: Annotated[Finite, Field(gt=0, lt=1)] = 0.10  # relative: 0.10 = ±10%
    box_size: BoxSize
    bands: list[Band] = Field(default_factory=list)
    # Left out of the saved form when empty, so a project without records keeps its
    # bytes and hash. Do not "normalize" this: see "Canonical form" in storage.
    undetected: list[UndetectedBand] = Field(default_factory=list, exclude_if=lambda v: not v)

    @model_validator(mode="after")
    def _check_bands_and_loading_controls(self) -> Protein:
        # Band order carries no meaning: sort so equal proteins give equal bytes.
        # No band_index < expected_band_count check: #58 stores extra bands to flag them.
        self.bands.sort(key=lambda band: (band.lane_index, band.band_index))
        for a, b in itertools.pairwise(self.bands):
            if (a.lane_index, a.band_index) == (b.lane_index, b.band_index):
                raise ValueError(
                    f"protein {self.id}: two bands in lane {a.lane_index}"
                    f" with band index {a.band_index}"
                )
        # Boxes of one protein must not overlap; other proteins' boxes may.
        rects = [(band.id, band.box.rect(self.box_size)) for band in self.bands]
        for i, (a_id, a_rect) in enumerate(rects):
            for b_id, b_rect in rects[i + 1 :]:
                if overlaps(a_rect, b_rect):
                    raise ValueError(f"protein {self.id}: boxes of {a_id} and {b_id} overlap")

        ids = self.loading_control_ids
        if self.role is Role.LOADING_CONTROL and ids:
            raise ValueError(f"protein {self.id}: a loading control cannot list loading controls")
        if len(set(ids)) != len(ids):
            raise ValueError(f"protein {self.id}: a loading control is listed twice")
        if self.id in ids:
            raise ValueError(f"protein {self.id}: a protein cannot be its own loading control")
        return self

    @model_validator(mode="after")
    def _check_undetected(self) -> Protein:
        # Record order carries no meaning either: sorted like the bands.
        self.undetected.sort(key=lambda record: (record.lane_index, record.band_index))
        for a, b in itertools.pairwise(self.undetected):
            if (a.lane_index, a.band_index) == (b.lane_index, b.band_index):
                raise ValueError(
                    f"protein {self.id}: two not-detected records in lane {a.lane_index}"
                    f" with band index {a.band_index}"
                )
        held = {(band.lane_index, band.band_index) for band in self.bands}
        for record in self.undetected:
            if (record.lane_index, record.band_index) in held:
                raise ValueError(
                    f"protein {self.id}: lane {record.lane_index} has both a band and a"
                    f" not-detected record for band index {record.band_index}"
                )
            if record.band_index >= self.expected_band_count:
                raise ValueError(
                    f"protein {self.id}: a not-detected record for band index"
                    f" {record.band_index}, but {self.expected_band_count} band(s) are expected"
                )
        return self


# --- History ---


class LogEntry(_Model):
    """One committed change: when, what, with which inputs, by which version, and
    the content hash it left. Immutable; entries are shared between snapshots."""

    model_config = ConfigDict(frozen=True)  # merged with extra="forbid", allow_inf_nan=False

    seq: int = Field(ge=1)  # 1, 2, 3, ... in list order: the entry's id
    time: Timestamp  # commit time from the session clock; informative, seq is the order
    action: ActionName  # the operation's function name, e.g. "place_box"
    version: SoftwareVersion  # proteia.__version__ that committed it
    params: dict[str, JsonValue] = Field(default_factory=dict)
    content_hash: Sha256  # storage.content_hash of the project this change left


# --- Batch and project ---


class Batch(_Model):
    """One run as stored: its lane table, membranes and proteins.

    Its statistics input is :class:`proteia.core.analyze.Batch`, which the compute
    step builds from it (loading-control ids become protein names).
    """

    lanes: list[Lane] = Field(default_factory=list)
    reference_condition: str | None = None  # a lane label: the fold-change reference
    membranes: list[Membrane] = Field(default_factory=list)
    proteins: list[Protein] = Field(default_factory=list)

    def iter_images(self) -> Iterator[ImageRef]:
        """Every image of every membrane, in membrane then image order."""
        for membrane in self.membranes:
            yield from membrane.images

    def find_image(self, image_id: str) -> ImageRef:
        for image in self.iter_images():
            if image.id == image_id:
                return image
        raise UnknownIdError(f"unknown image {image_id!r}")

    def membrane_of(self, image_id: str) -> Membrane:
        for membrane in self.membranes:
            if any(image.id == image_id for image in membrane.images):
                return membrane
        raise UnknownIdError(f"unknown image {image_id!r}")

    def find_protein(self, protein_id: str) -> Protein:
        for protein in self.proteins:
            if protein.id == protein_id:
                return protein
        raise UnknownIdError(f"unknown protein {protein_id!r}")

    def find_band(self, band_id: str) -> tuple[Protein, Band]:
        for protein in self.proteins:
            for band in protein.bands:
                if band.id == band_id:
                    return protein, band
        raise UnknownIdError(f"unknown band {band_id!r}")

    @model_validator(mode="after")
    def _check_references(self) -> Batch:
        # join_to_spine needs lane indices 0..n-1, in list order.
        if any(lane.index != i for i, lane in enumerate(self.lanes)):
            raise ValueError("lane indices must be 0, 1, 2, ... in list order")
        if self.reference_condition is not None and self.reference_condition not in {
            lane.label for lane in self.lanes
        }:
            raise ValueError(
                f"reference condition {self.reference_condition!r} is not a lane condition"
            )

        images = {image.id: image for image in self.iter_images()}
        roles = {protein.id: protein.role for protein in self.proteins}
        names: set[str] = set()
        for protein in self.proteins:
            if protein.name in names:
                raise ValueError(f"duplicate protein name {protein.name!r}")
            names.add(protein.name)
            image = images.get(protein.image_id)
            if image is None:
                raise ValueError(f"protein {protein.id}: unknown image {protein.image_id!r}")
            if image.kind is ImageKind.VISIBLE_MARKER:
                raise ValueError(
                    f"protein {protein.id}: {image.id} is a visible-light marker image,"
                    " not a signal image"
                )
            size = protein.box_size
            if size.width > image.width or size.height > image.height:
                raise ValueError(
                    f"protein {protein.id}: box size {size.width}x{size.height}"
                    f" exceeds the bounds of image {image.id}"
                )
            for lc_id in protein.loading_control_ids:
                if lc_id not in roles:
                    raise ValueError(f"protein {protein.id}: unknown loading control {lc_id!r}")
                if roles[lc_id] is not Role.LOADING_CONTROL:
                    raise ValueError(f"protein {protein.id}: {lc_id} is not a loading control")
            for band in protein.bands:
                if band.lane_index >= len(self.lanes):
                    raise ValueError(
                        f"protein {protein.id}: band {band.id} references"
                        f" unknown lane index {band.lane_index}"
                    )
                _, _, x1, y1 = band.box.rect(size)
                if x1 > image.width or y1 > image.height:
                    raise ValueError(
                        f"protein {protein.id}: box of {band.id}"
                        f" extends beyond the bounds of image {image.id}"
                    )
            for record in protein.undetected:
                if record.lane_index >= len(self.lanes):
                    raise ValueError(
                        f"protein {protein.id}: a not-detected record references"
                        f" unknown lane index {record.lane_index}"
                    )
                region = record.region
                if region.x1 > image.width or region.y1 > image.height:
                    raise ValueError(
                        f"protein {protein.id}: the not-detected region in lane"
                        f" {record.lane_index} extends beyond the bounds of image {image.id}"
                    )
        return self


class Project(_Model):
    """The whole saved project: ``project.json`` is its JSON form."""

    schema_version: Literal[1] = SCHEMA_VERSION
    # Bookkeeping, excluded from the content hash: the number of the next new id.
    next_id: int = Field(default=1, ge=1, le=10**9)
    batch: Batch = Field(default_factory=Batch)
    log: tuple[LogEntry, ...] = ()  # history, oldest first; excluded from the content hash

    def iter_ids(self) -> Iterator[str]:
        """Every object id: membranes, images, proteins, bands (not-detected records
        have none)."""
        batch = self.batch
        yield from (membrane.id for membrane in batch.membranes)
        yield from (image.id for image in batch.iter_images())
        yield from (protein.id for protein in batch.proteins)
        yield from (band.id for protein in batch.proteins for band in protein.bands)

    def new_id(self, prefix: IdPrefix) -> str:
        """A fresh id such as ``img-7``.

        Numbers come from one counter, so they are unique across kinds, never
        reused after a deletion, and the same under replay. Call it inside
        :func:`apply_change`, so a failed change consumes no number.
        """
        new = f"{prefix}-{self.next_id}"
        self.next_id += 1
        return new

    @model_validator(mode="after")
    def _check_ids(self) -> Project:
        seen: dict[int, str] = {}
        for obj_id in self.iter_ids():
            number = int(obj_id.rsplit("-", 1)[1])
            if number in seen:
                raise ValueError(f"duplicate id number {number}: {seen[number]} and {obj_id}")
            if number >= self.next_id:
                raise ValueError(f"id {obj_id} is not below next_id {self.next_id}")
            seen[number] = obj_id
        return self

    @model_validator(mode="after")
    def _check_log(self) -> Project:
        # An entry deleted by hand in the middle breaks the count. Time order is
        # not checked: a wall clock can step back.
        for expected, entry in enumerate(self.log, start=1):
            if entry.seq != expected:
                raise ValueError(f"log entry {entry.seq} is out of sequence: expected {expected}")
        return self


def revalidate(project: Project, *, log: bool = True) -> Project:
    """A freshly validated copy: reruns every nested validator on the current values.

    With ``log=False`` only the content is re-validated, and the copy shares the
    project's log (a tuple of frozen, already validated entries).
    """
    if log:
        return Project.model_validate(project.model_dump())
    fresh = Project.model_validate(project.model_dump(exclude={"log"}))
    return fresh.model_copy(update={"log": project.log})


def apply_change[T](project: Project, change: Callable[[Project], T]) -> tuple[Project, T]:
    """Run ``change`` on a copy, then re-validate the content.

    The copy is deep for the batch and shallow for the rest, so ``next_id`` and
    the batch are the draft's own and the log is shared: a change never copies
    or re-validates the history, and must not replace it (``RuntimeError``).
    A ``ValidationError`` leaves ``project`` untouched (``next_id`` included).
    ``change`` should return ids or plain values, not model objects: the returned
    project is a fresh copy.
    """
    log = project.log
    draft = project.model_copy(update={"batch": project.batch.model_copy(deep=True)})
    result = change(draft)
    if draft.log is not log:
        raise RuntimeError(
            "a change must not edit the log: the session appends its entry on commit"
        )
    return revalidate(draft, log=False), result
