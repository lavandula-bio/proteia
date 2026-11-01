# SPDX-License-Identifier: Apache-2.0
"""Core data model: the project as it is saved in ``project.json``.

GUI-independent (see ``docs/adr/0001``), with no filesystem access and no numpy;
saving, loading and the content hash live in :mod:`proteia.core.storage`.

Hierarchy. A :class:`Project` holds one :class:`Batch` (one run): its images,
kept once each, its lane tables, the reference condition, its membranes and
its proteins. A :class:`Membrane` is one physical blot: where it lies on each
of its images (chemiluminescence exposures, reprobes, the visible-light marker,
merged overlays), one :class:`Arena` (a membrane area) per image, the
:class:`LaneTable` that numbers its lanes, and one molecular-weight
calibration, fitted per register group (the images linked to one marker image)
from one or two ladders (:mod:`proteia.core.mwcal`). An area has an outline,
or none for the whole image, and an angle, which is recorded only: no pixel is
turned. A :class:`Protein` is one protein: its name, role and expectations, and
its :class:`Row` on one membrane area, which holds its boxes there: per lane
and expected band, a :class:`Band` (a box), an :class:`UndetectedBand` (a
detector measured the lane and the band stayed below its detection limit), or
neither (not measured). A reprobe is simply another image of the same
membrane. An image may hold several membranes' areas (strips cut from one
blot, or blots side by side), each then with its outline; a box, a
not-detected record and a calibration point belong to the one area that holds
its centre. Until lane tables can be added and a protein put on several
membranes, a project holds one lane table and one row per protein
(:meth:`Batch._check_supported`). Objects refer to each other by
stable, project-unique ids (``mem-N``, ``img-N``, ``lt-N``, ``arena-N``,
``prot-N``, ``row-N``, ``band-N``) and to lanes by index, never by list
position or name; a not-detected record has no id of its own, only its
protein, lane and band index.

The stored numbers (each band's net and the background it was measured above,
each image's median) are the raw data. Normalization and statistics combine
several proteins downstream in :mod:`proteia.core.analyze`, whose ``Batch`` is
built from this module's ``Batch``.

Background method. ``Project.background_method`` names how every stored net's
background was measured, project-wide, so a batch never mixes methods:
``ring_median_v1`` (:func:`proteia.core.quantify.band_backgrounds`, #83), each
band's own level from the membrane around its box, or ``global_median``, the
one image-wide median of a project quantified before #83 (a schema-1 project,
migrated: every band's level is its image's ``background``, its mode
``global_median`` and its spread 0, and nets floor each pixel at 0). The
model checks that each band's mode agrees with the method; the operations keep
every value equal to what the pixels give (:mod:`proteia.core.operations`).

Models are mutable. Edits go through :func:`apply_change`, which works on a copy
and re-validates the whole tree, so a failed edit leaves the project untouched.

History. ``Project.log`` records every committed change, oldest first: one
immutable :class:`LogEntry` each, with its time, action, parameters, the Proteia
version that made it and the content hash it left. The log is not content: it
says how the data got there, not what it is, so the content hash leaves it out.
A change never edits the log; the session appends the entry when it commits.

Geometry convention: boxes are axis-aligned and anchored at their top-left
corner in image pixel coordinates (numpy ``image[y, x]``), as are membrane
areas' outlines and calibration points. All boxes of one row share its box
size, so they have equal area; only their positions vary.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Protocol, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticKnownError

# Before v0.1, an additive change (a new optional field left out while empty, or
# a new enum value) does not bump; from v0.1 on, every saved-form change bumps and
# registers a migration (see proteia.core.storage). 2 (#83): the band background
# fields and the project's background method. 3 (#223): the images kept once by
# the batch, each membrane's area on each of its images, lane tables with ids,
# and each protein's boxes in its row on one area.
SCHEMA_VERSION: Final = 3
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
# An id's number has at most nine digits: below NEXT_ID_LIMIT, the largest next_id.
_N = r"[1-9][0-9]{0,8}"
NEXT_ID_LIMIT: Final = 10**9
MembraneId = Annotated[str, StringConstraints(pattern=rf"^mem-{_N}$")]
ImageId = Annotated[str, StringConstraints(pattern=rf"^img-{_N}$")]
ProteinId = Annotated[str, StringConstraints(pattern=rf"^prot-{_N}$")]
BandId = Annotated[str, StringConstraints(pattern=rf"^band-{_N}$")]
ArenaId = Annotated[str, StringConstraints(pattern=rf"^arena-{_N}$")]
RowId = Annotated[str, StringConstraints(pattern=rf"^row-{_N}$")]
LaneTableId = Annotated[str, StringConstraints(pattern=rf"^lt-{_N}$")]
IdPrefix = Literal["mem", "img", "prot", "band", "arena", "row", "lt"]
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
    """No object with this id in the batch. ``detail``, when given, is
    JSON-plain data naming what was asked for (an id the message leaves out)."""

    def __init__(self, message: str = "", *, detail: dict[str, JsonValue] | None = None) -> None:
        super().__init__(message)
        self.detail = detail


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


class LadderSide(StrEnum):
    """Which ladder of its register group a calibration point belongs to (#58)."""

    LEFT = "left"  # the only ladder, or the left one of two; strip edges
    RIGHT = "right"  # the optional second ladder, right of the first


class FitMethod(StrEnum):
    # The default: no #58 fit stored yet. (The issue first asked for one
    # least-squares line; no build ever wrote a calibration point under it.)
    LOG_LINEAR = "log_linear"
    # Per ladder, log10(MW) linear in y between neighbouring points (#58, D1);
    # with a right ladder, each MW's y linear in x between the two ladders.
    LOG_LINEAR_PIECEWISE = "log_linear_piecewise"


# How a band's background level was measured: a ring_median mode
# (proteia.core.quantify.BandBackground), or the image-wide median of a band
# quantified before #83.
BackgroundMode = Literal["symmetric", "asymmetric", "image", "global_median"]
# How every stored net's background was measured, project-wide (see the module
# docstring). A new ring_median version is a new value, reached by requantifying.
ProjectBackgroundMethod = Literal["global_median", "ring_median_v1"]
LEGACY_BACKGROUND_METHOD: Final = "global_median"
LOCAL_BACKGROUND_METHOD: Final = "ring_median_v1"  # what a new project starts with


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


class BoxPadding(_Model):
    """Whole pixels every box of one protein extends beyond its fitted size (the
    size its clicks, rows or typing asked for), on each side (#57)."""

    across: int = Field(default=0, ge=0)  # left and right
    along: int = Field(default=0, ge=0)  # above and below


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


def lane_number(index: int) -> int:
    """The number a user knows a lane by: its stored index plus one.

    Every text a user can read (a refusal, a validation message, a results
    notice, a detector's note) numbers lanes from 1, as the web UI does;
    stored indices, log params and API fields keep the 0-based index."""
    return index + 1


def lanes_phrase(indices: Iterable[int]) -> str:
    """Stored lane indices in words, numbered by :func:`lane_number` and
    ascending: ``lane 8``, ``lanes 3, 7``."""
    numbers = sorted(lane_number(index) for index in indices)
    return ("lane " if len(numbers) == 1 else "lanes ") + ", ".join(map(str, numbers))


def in_words(items: Sequence[str]) -> str:
    """Items of a message in words: ``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


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


# How far a membrane area's outline may reach from the image's corner, in pixels.
OUTLINE_LIMIT: Final = 10**6
_Coordinate = Annotated[int, Field(ge=-OUTLINE_LIMIT, le=OUTLINE_LIMIT)]


class Outline(_Model):
    """A rectangle drawn on an image as a membrane area, before its area's turn:
    corners (x0, y0) and (x1, y1) in image pixels. It may reach past the image;
    its centre may not (the batch checks it)."""

    x0: _Coordinate
    y0: _Coordinate
    x1: _Coordinate
    y1: _Coordinate

    @model_validator(mode="after")
    def _check_size(self) -> Outline:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                "a membrane area's outline needs a width and a height of at least 1 pixel"
            )
        return self

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


# --- Images and the molecular-weight calibration ---


class ImageWarning(_Model):
    """A problem found when the image was imported (codes are defined by #45)."""

    code: WarningCode  # e.g. "lossy_format", "color_channels_differ"
    message: Annotated[str, StringConstraints(min_length=1)]


# The import warnings that make the over-exposure check distrust an image's
# known bit depth (proteia.core.imaging.clipping_depth), and so let the
# near-limit check assess it instead. results names each in its
# clipping_not_checked and possibly_clipped notices (_UNCHECKED_WARNINGS, kept
# in step by a test), so a code added here needs a reason there. They live in
# the model, as limit_distrusted_by does, so that results reads them without
# loading the image readers.
# looks_processed is not one of them (#127). Levelling a figure's background to
# white moves the background onto a limit, not the bands, and on a lossless file
# the exact check still counts the band pixels the file holds at the limit (a
# lossy, colour or CMYK file is distrusted by its own warning already). The
# warning is a heuristic that also fires on some raw images (a dark background
# the imager cut at zero, a marker photo's white membrane), whose exact check it
# would weaken. A known limit: a palette reduced to few colours can move a
# saturated core a level or two off the limit, which the exact check then
# misses; the warning says that over-exposure may be hidden.
UNTRUSTED_WARNINGS: Final = frozenset({"lossy_format", "color_channels_differ", "cmyk_converted"})


class Channel(StrEnum):
    """What an image is measured on (#206): one channel of its file as read, or
    the unweighted mean of its red, green and blue, as ImageJ converts colour by
    default. A gray image, and a colour image no channel was chosen for, are
    measured on the mean (a gray image's own values)."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    MEAN = "mean"


class ChannelSource(StrEnum):
    """Who chose the channel an image is measured on."""

    CHOSEN = "chosen"  # the user (operations.set_channel)


class ChannelChoice(_Model):
    """The channel a colour image is measured on, and who chose it (#206)."""

    channel: Channel
    source: ChannelSource


def limit_distrusted_by(
    warnings: Iterable[ImageWarning], channel: Channel | None = None
) -> list[str]:
    """The codes of the import ``warnings`` that make the over-exposure check
    distrust an image's detector limit when it is measured on ``channel``
    (None: the mean), in their order: those of :data:`UNTRUSTED_WARNINGS`,
    except ``color_channels_differ`` for one colour channel. A channel
    saturated alone never brings the mean of differing channels to the limit,
    but it reaches the limit itself; compression and a CMYK conversion move the
    values of every channel."""
    one_channel = channel is not None and channel is not Channel.MEAN
    return [
        warning.code
        for warning in warnings
        if warning.code in UNTRUSTED_WARNINGS
        and not (one_channel and warning.code == "color_channels_differ")
    ]


def clipping_depth(
    bit_depth: int | None, warnings: Iterable[ImageWarning], channel: Channel | None = None
) -> int | None:
    """The bit depth whose detector limit the over-exposure check may trust, or
    None, for a reading with the import ``warnings`` measured on ``channel``
    (None: the mean, as for every image before #206). An image record reads
    its own (:attr:`ImageRef.exact_check_depth`); this takes the raw values for
    a reading with no record yet (an import, the warnings a polarity or channel
    change assesses) and for tools.

    None when the depth is unknown (float data), when lossy compression moved
    saturated pixels off the limit (``lossy_format``), when color was averaged
    into gray (``color_channels_differ`` on the mean): a channel saturated alone
    never brings the mean to the limit, or when the colours were converted from
    CMYK (``cmyk_converted``): its values are inks turned into approximate
    colours, not detector counts, and a profile may keep black off 0
    (:func:`limit_distrusted_by`). One colour channel of a lossless file is
    checked at the file's limit. A 12- or 14-bit camera writing a 16-bit file is
    checked against the container limit, so its saturation goes unseen (a known
    limit).
    """
    if bit_depth is None or limit_distrusted_by(warnings, channel):
        return None
    return bit_depth


def possible_clipping_depth(
    bit_depth: int | None, warnings: Iterable[ImageWarning], channel: Channel | None = None
) -> int | None:
    """The bit depth the "possibly over-exposed" check (#112,
    :func:`~proteia.core.quantify.is_possibly_clipped`) measures against, or
    None; an image record reads its own (:attr:`ImageRef.near_check_depth`).

    It runs where the exact check cannot (:func:`clipping_depth` is None) but the
    range is known: a lossy or CMYK-converted image, or a colour image measured
    on the mean, of 8- or 16-bit pixels; one colour channel of a lossy file is
    assessed on that channel. Its limit is that of the pixels as read: 0, or 255
    or 65535 for a light-on-dark image; for a CMYK file, that of the red, green
    and blue it was converted to, which have the inks' bit depth. A file
    converted through its ICC profile may bring black back well above 0 (a gray
    ramp made with the relative intent and black point compensation through
    SWOP press CMYK comes back at about 19), which hides its saturation from
    this check too (a known limit). An unknown bit depth has no limit to
    measure against, so neither check runs.
    """
    if bit_depth is None or clipping_depth(bit_depth, warnings, channel) is not None:
        return None
    return bit_depth


# Which over-exposure check an image's bands get (ImageRef.over_exposure_check):
# the exact one at the detector limit, the one near it, or none (unknown depth).
OverExposureCheck = Literal["exact", "near_limit", "none"]
# The color_channels_differ message as shown while an image is measured on one
# of its channels (ImageRef.shown_warnings); the stored one stays as imported,
# which is true of the mean.
_MEASURED_ON_CHANNEL: Final = (
    "The red, green and blue channels differ; the image is measured on its {channel} channel."
)


# Why no channel can be chosen for an image (ImageRef.channel_blocker): its
# red, green and blue are equal (or it is gray), it is a visible-light marker,
# whose ladder is of several colours, or its colours were converted from CMYK.
ChannelBlocker = Literal["gray", "marker", "cmyk"]
_CHANNEL_BLOCKED: Final[dict[ChannelBlocker, str]] = {
    "gray": "a channel is chosen only for an image whose red, green and blue differ",
    "marker": (
        "a visible-light marker image is measured on the mean of its colours, never on one channel"
    ),
    "cmyk": (
        "an image converted from CMYK is measured on the mean of its colours, never on one channel"
    ),
}


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
    # quantify.estimate_background, the global median of the analysis array: the
    # grow seed threshold, the image-mode fallback and the legacy background.
    background: Finite
    import_warnings: list[ImageWarning] = Field(default_factory=list)
    # #206: the channel it is measured on; None = the mean of its red, green and
    # blue (Channel.MEAN), as for a gray image and every image imported before.
    # Left out of the saved form while None, so a project without a choice keeps
    # its bytes and hash (see "Canonical form" in storage).
    channel: ChannelChoice | None = Field(default=None, exclude_if=lambda v: v is None)

    @property
    def measured_channel(self) -> Channel:
        """The channel the image is measured on: its choice's, else the mean."""
        return Channel.MEAN if self.channel is None else self.channel.channel

    # The image's over-exposure checks, from its bit depth, import warnings and
    # measured channel: the one place an image record's trust is read.

    @property
    def limit_distrusted(self) -> list[str]:
        """The codes of its import warnings that distrust its detector limit,
        on the channel it is measured on (:func:`limit_distrusted_by`)."""
        return limit_distrusted_by(self.import_warnings, self.measured_channel)

    @property
    def exact_check_depth(self) -> int | None:
        """The bit depth its exact over-exposure check runs at, or None
        (:func:`clipping_depth`)."""
        return clipping_depth(self.bit_depth, self.import_warnings, self.measured_channel)

    @property
    def near_check_depth(self) -> int | None:
        """The bit depth its near-limit check runs at, or None where the exact
        one runs or no limit is known (:func:`possible_clipping_depth`)."""
        return possible_clipping_depth(self.bit_depth, self.import_warnings, self.measured_channel)

    @property
    def over_exposure_check(self) -> OverExposureCheck:
        """Which check its bands get: ``exact``, ``near_limit`` or ``none``."""
        if self.exact_check_depth is not None:
            return "exact"
        return "none" if self.near_check_depth is None else "near_limit"

    @property
    def shown_warnings(self) -> list[ImageWarning]:
        """Its import warnings as they are shown: as stored, except that while it
        is measured on one colour channel its ``color_channels_differ`` says
        which, as the stored message, written for the mean, would not."""
        channel = self.measured_channel
        if channel is Channel.MEAN:
            return list(self.import_warnings)
        return [
            ImageWarning(code=w.code, message=_MEASURED_ON_CHANNEL.format(channel=channel))
            if w.code == "color_channels_differ"
            else w
            for w in self.import_warnings
        ]

    @property
    def channel_blocker(self) -> ChannelBlocker | None:
        """Why no channel can be chosen for the image, or None if one can, the
        first that holds: ``"gray"``, its import recorded no
        ``color_channels_differ``; ``"marker"``, it is a visible-light marker;
        ``"cmyk"``, its colours were converted from CMYK (``cmyk_converted``)."""
        codes = {warning.code for warning in self.import_warnings}
        if "color_channels_differ" not in codes:
            return "gray"
        if self.kind is ImageKind.VISIBLE_MARKER:
            return "marker"
        if "cmyk_converted" in codes:
            return "cmyk"
        return None

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

    @model_validator(mode="after")
    def _check_channel(self) -> ImageRef:
        # A choice is made only where operations.set_channel takes one.
        if self.channel is not None:
            blocker = self.channel_blocker
            if blocker is not None:
                raise ValueError(f"image {self.id}: {_CHANNEL_BLOCKED[blocker]}")
        return self


class CalibrationPoint(_Model):
    """One known molecular weight at a position on an image of the membrane.

    x and y are continuous coordinates of the image's analysis array (pixel
    column c covers [c, c+1), row r covers [r, r+1)), as box centres are,
    whatever produced that array."""

    image_id: ImageId  # the image whose analysis array x and y are measured in (same membrane)
    y: NonNegative  # from the top of that image; sub-pixel allowed
    mw: Kda
    source: CalibrationPointSource
    # Where the point was marked across the image: the ladder lane's x, or where
    # a strip edge was clicked. Every point an operation adds records it (#58);
    # files written before #58 hold none, so it is optional here. Left out of the
    # saved form while None, and ``side`` while left (see "Canonical form" in storage).
    x: NonNegative | None = Field(default=None, exclude_if=lambda v: v is None)
    side: LadderSide = Field(default=LadderSide.LEFT, exclude_if=lambda v: v == LadderSide.LEFT)

    @model_validator(mode="after")
    def _check_side(self) -> CalibrationPoint:
        if self.side == LadderSide.RIGHT:
            if self.source is CalibrationPointSource.STRIP_EDGE:
                raise ValueError("calibration point: a strip edge belongs to the left ladder")
            if self.x is None:
                raise ValueError("calibration point: a point of the right ladder needs its x")
        return self


class MwCalibration(_Model):
    """A membrane's molecular-weight calibration (#58): its ladder product and
    points, fitted per register group and ladder side (:mod:`proteia.core.mwcal`)."""

    ladder: str | None = None  # a preset key (proteia.core.ladders) or a custom name
    # The ladder's MWs, top to bottom: the preset's when it was chosen, or the
    # custom list. Left out of the saved form while empty.
    ladder_kda: list[Kda] = Field(default_factory=list, exclude_if=lambda v: not v)
    points: list[CalibrationPoint] = Field(default_factory=list)
    # How the stored fit_quality and apparent MWs were computed. log_linear, the
    # default, stays on a calibration no #58 build has fitted, so such membranes
    # keep their bytes.
    fit_method: FitMethod = FitMethod.LOG_LINEAR
    fit_quality: Finite | None = None  # the worst ladder's quality; None: none measurable

    @model_validator(mode="after")
    def _canonical_points(self) -> MwCalibration:
        # Point order carries no meaning: sort so equal calibrations give equal bytes.
        # Left points sort first, so a file with no right ladder keeps its order.
        self.points.sort(key=lambda p: (p.side.value, p.y, p.mw, p.source.value, p.image_id))
        if self.fit_quality is not None and len(self.points) < 2:
            raise ValueError("a calibration fit needs at least two points")
        # On log10(MW), as the points are (Membrane._check_ladders).
        for above, below in itertools.pairwise(self.ladder_kda):
            if not math.log10(above) > math.log10(below):
                raise ValueError(
                    "ladder MWs must decrease strictly from top to bottom"
                    f" ({above:g}, {below:g} kDa)"
                )
        if self.ladder_kda and self.ladder is None:
            raise ValueError("ladder MWs need a ladder name")
        return self


# The image kinds a chemiluminescence image may be linked to as its marker (D4).
_MARKER_KINDS: Final = frozenset({ImageKind.VISIBLE_MARKER, ImageKind.MERGED})


class ArenaSource(StrEnum):
    """How a membrane area came to be."""

    WHOLE_IMAGE = "whole_image"  # made with its image (an import or a conversion), never edited
    MANUAL = "manual"  # drawn or edited by the user


# The largest turn of a membrane area either way, in degrees.
MAX_ARENA_ANGLE: Final = 45.0
# How far outside its outline a point may lie and still count as inside, in
# pixels: what turning the outline's edges may round away.
ON_EDGE: Final = 1e-9


class Arena(_Model):
    """Where one membrane lies on one image (a membrane area): its outline, or
    none for the whole image, and its tilt. The angle is recorded, not applied:
    no pixel is turned.

    A drawn outline is its rectangle turned by ``angle_deg`` about its centre,
    a positive angle running its right side lower (image y runs down). An area
    without an outline holds the whole image whatever its angle, which then
    only records the membrane's tilt."""

    id: ArenaId
    image_id: ImageId
    # Left out of the saved form while None (the whole image), as every area a
    # conversion or an import makes has none (see "Canonical form" in storage).
    outline: Outline | None = Field(default=None, exclude_if=lambda v: v is None)
    angle_deg: Annotated[Finite, Field(ge=-MAX_ARENA_ANGLE, le=MAX_ARENA_ANGLE)] = 0.0
    source: ArenaSource

    def frame(self) -> AreaFrame | None:
        """The drawn outline's turned frame (:class:`AreaFrame`), the one
        geometry every reading of the area uses; None for the whole image."""
        outline = self.outline
        if outline is None:
            return None
        cx, cy = outline.centre
        turn = math.radians(self.angle_deg)
        return AreaFrame(
            cx=cx,
            cy=cy,
            cos=math.cos(turn),
            sin=math.sin(turn),
            half_width=(outline.x1 - outline.x0) / 2,
            half_height=(outline.y1 - outline.y0) / 2,
        )

    def contains(self, image: ImageRef, x: float, y: float) -> bool:
        """Whether the point (x, y), in pixels of ``image`` (the area's image),
        lies in the area. Closed: a point on its edge does."""
        frame = self.frame()
        if frame is None:
            return 0 <= x <= image.width and 0 <= y <= image.height
        return frame.holds(x, y)

    def inside(self, image: ImageRef, x: float, y: float) -> bool:
        """Whether the point (x, y) lies in the area past its edge
        (:meth:`contains` less the edge)."""
        frame = self.frame()
        if frame is None:
            return 0 < x < image.width and 0 < y < image.height
        return frame.inside(x, y)

    def corners(self, image: ImageRef) -> list[tuple[float, float]]:
        """The area's corners in pixels of ``image``: top left, top right,
        bottom right, bottom left, before the turn (the image's own for the
        whole image)."""
        frame = self.frame()
        if frame is None:
            w, h = float(image.width), float(image.height)
            return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        a, b = frame.half_width, frame.half_height
        return [frame.point(u, v) for u, v in ((-a, -b), (a, -b), (a, b), (-a, b))]

    def bounds(self, image: ImageRef) -> tuple[int, int, int, int]:
        """The whole pixels the area's corners span, ``(x0, y0, x1, y1)``:
        floor of the least, ceiling of the greatest; it may reach past the image."""
        corners = self.corners(image)
        xs, ys = [x for x, _ in corners], [y for _, y in corners]
        return math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))


@dataclass(frozen=True)
class AreaFrame:
    """A drawn membrane area's outline as turned: its centre, the cosine and
    sine of its angle (a positive angle runs its right side lower; image y runs
    down) and its half sizes before the turn."""

    cx: float
    cy: float
    cos: float
    sin: float
    half_width: float
    half_height: float

    def point(self, u: float, v: float) -> tuple[float, float]:
        """The image point at ``(u, v)`` from the centre in the outline's own
        axes: ``(u, v)`` turned by the angle."""
        return self.cx + u * self.cos - v * self.sin, self.cy + u * self.sin + v * self.cos

    def along(self, x: float, y: float) -> tuple[float, float]:
        """The image point ``(x, y)`` in the outline's own axes: turned back
        by the angle about the centre."""
        dx, dy = x - self.cx, y - self.cy
        return dx * self.cos + dy * self.sin, dy * self.cos - dx * self.sin

    def holds(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies in the turned outline, its edge included
        (to a billionth of a pixel)."""
        u, v = self.along(x, y)
        return abs(u) <= self.half_width + ON_EDGE and abs(v) <= self.half_height + ON_EDGE

    def inside(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies in the turned outline past its edge (by
        more than a billionth of a pixel)."""
        u, v = self.along(x, y)
        return abs(u) < self.half_width - ON_EDGE and abs(v) < self.half_height - ON_EDGE


class LaneTable(_Model):
    """The lanes of the membranes that use the table: their conditions,
    samples and inclusion, by stored index."""

    id: LaneTableId
    lanes: list[Lane] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_indices(self) -> LaneTable:
        # join_to_spine needs lane indices 0..n-1, in list order.
        if any(lane.index != i for i, lane in enumerate(self.lanes)):
            raise ValueError("lane indices must be 0, 1, 2, ... in list order")
        return self


class Membrane(_Model):
    """One physical blot: its area on each of its images (exposures, reprobes,
    marker, merged), in order, the lane table that numbers its lanes, and one
    molecular-weight calibration, fitted per register group (the images linked
    to one marker image) from one or two ladders.

    A register group is a set of images whose pixel rows line up: a
    chemiluminescence image and the marker image it was taken with. Linked
    images match pixel for pixel (D5) and hold the same membrane areas, so a
    point marked on one of them holds for all of them. The images themselves
    are the batch's (:attr:`Batch.images`); :meth:`Batch.images_of` gives a
    membrane's.
    """

    id: MembraneId
    lane_table_id: LaneTableId
    arenas: list[Arena] = Field(default_factory=list)
    calibration: MwCalibration = Field(default_factory=MwCalibration)


# --- Proteins and their bands ---


class Band(_Model):
    """One box of one protein in one lane.

    ``net`` and the ``background_*`` fields are what quantifying the band's image
    gave (:mod:`proteia.core.operations` recomputes them whenever a box on the
    image changes); loading never recomputes them. ``background_level`` is what
    :func:`~proteia.core.quantify.net_signal` subtracted, in pixel units;
    ``background_mode`` says how it was measured and ``background_spread`` is its
    QC, in level units (:class:`~proteia.core.quantify.BandBackground`).

    ``clipped`` is the exact over-exposure check (#44), and ``possibly_clipped``
    the heuristic that stands in for it where it cannot run (#112,
    :func:`~proteia.core.quantify.is_possibly_clipped`): a band has at most one
    of them, the other None.

    ``bands_found`` is how many bands the detector that placed the box found in
    its lane within the count window around it (#58, D10): the band itself,
    and each other peak there that reads as a band. Only a detector's band
    that nobody edited has one; an edit by hand, or a change to what the window
    came from, clears it.
    """

    id: BandId
    lane_index: int = Field(ge=0)  # into its membrane's lane table: the band's stored identity
    band_index: int = Field(default=0, ge=0)  # which expected band; 0 for single-band proteins
    box: Box  # top-left in its row's image; its size is Row.box_size
    net: NonNegative  # quantify.net_signal above background_level
    background_level: Finite
    background_mode: BackgroundMode
    background_spread: NonNegative
    # Its image's calibration at the box's centre (#58); None = no curve there, or the centre
    # lies outside its range.
    apparent_mw: Kda | None = None
    # #58: the bands a detector found in the count window; None = not counted (placed
    # or edited by hand, or cleared). Left out of the saved form while None, so a
    # project without a count keeps its bytes and hash (see "Canonical form" in storage).
    bands_found: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    clipped: bool | None = None  # #44; None = not checked (not "passed")
    # #112; None = not assessed: the exact check ran, the image has no known
    # range, or the project was saved before #112 (requantify assesses it).
    # False = no sign of it in the gray values, not "passed". Left out of the
    # saved form while None, so a project without a lossy, colour or CMYK image
    # keeps its bytes and hash (see "Canonical form" in storage).
    possibly_clipped: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    source: ProposalSource
    manually_edited: bool = False  # moved or edited by the user after it was proposed

    @model_validator(mode="after")
    def _one_clipping_check(self) -> Band:
        if self.clipped is not None and self.possibly_clipped is not None:
            raise ValueError(
                f"band {self.id}: checked for over-exposure (clipped) and assessed"
                " for it (possibly_clipped); only one of the checks runs on an image"
            )
        return self

    @model_validator(mode="after")
    def _counted_by_a_detector(self) -> Band:
        # A count belongs to the detector's box: a box edited by hand, or placed
        # without a detector, holds none, so a count cannot outlive its box.
        if self.bands_found is not None and (
            self.source not in DETECTING_SOURCES or self.manually_edited
        ):
            how = "edited by hand" if self.manually_edited else f"placed by {self.source.value}"
            raise ValueError(
                f"band {self.id}: a band count (bands_found) comes from a detector's box"
                f" that nobody edited, and this one was {how}"
            )
        return self


class UndetectedBand(_Model):
    """An expected band a detector looked for in one lane and did not find: the
    lane was measured, and the band is below the detection limit.

    It carries no value, so statistics leave the lane out as they do a lane with
    no box; results and exports can still tell "not measured" from "below the
    detection limit". ``threshold`` is the limit in force when the record was
    written, so each record backs its own claim. The detector's window comes from
    code constants, not from the protein's box size, so a size change never makes
    ``snr`` stale. A row placed by its expected MW also records a lane whose
    band lay off the row's line and off the expected row (#58,
    ``off_expected_row``): its ``snr`` is read on the row's line, the band's
    own tail there less its reflection about the band's centre, and its
    ``region`` is the rows that reading read.
    """

    lane_index: int = Field(ge=0)  # into its membrane's lane table, as for a band
    band_index: int = Field(default=0, ge=0)  # which expected band
    reason: UndetectedReason
    snr: Finite  # the detector's statistic for this slot; any sign
    threshold: Annotated[Finite, Field(gt=0)]  # the detection limit it was compared with
    region: Region  # the slot the detector measured, in its row's image
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


# The validation context key under which storage passes the ids of the rows
# whose area reference holds the image reference of their schema-2 protein:
# an image the converted file does not hold. Such a reference is read as the
# image reference it was (Row) and refused as schema 2 refused it (Batch).
UNKNOWN_IMAGE_ROWS: Final = "unknown_image_rows"
_IMAGE_ID: Final = TypeAdapter(ImageId)


def _unknown_image_rows(info: ValidationInfo) -> Collection[str]:
    rows: Collection[str] = (info.context or {}).get(UNKNOWN_IMAGE_ROWS, ())
    return rows


class Row(_Model):
    """One protein quantified on one membrane area: its boxes there.

    Every band shares ``box_size``, the effective size that was quantified: the
    fitted size (what clicks, rows or typing asked for) plus ``box_padding`` on
    each side (:attr:`fitted_size`). Each (lane, band index) holds a band, a
    not-detected record, or neither; a record only ever concerns an expected
    band. Its protein checks its boxes and records (:class:`Protein`)."""

    id: RowId
    arena_id: ArenaId  # the membrane area: its image and membrane
    box_size: BoxSize  # quantified: the fitted size plus box_padding on each side
    # Left out of the saved form while (0, 0), so a project without padding keeps its
    # bytes and hash (see "Canonical form" in storage).
    box_padding: BoxPadding = Field(
        default_factory=BoxPadding, exclude_if=lambda v: v == BoxPadding()
    )
    bands: list[Band] = Field(default_factory=list)
    # Left out of the saved form when empty, so a project without records keeps its
    # bytes and hash. Do not "normalize" this: see "Canonical form" in storage.
    undetected: list[UndetectedBand] = Field(default_factory=list, exclude_if=lambda v: not v)

    @field_validator("arena_id", mode="wrap")
    @classmethod
    def _schema_2_image(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> str:
        """An area reference, or, in a row of :data:`UNKNOWN_IMAGE_ROWS`, the
        image reference of its schema-2 protein, checked as schema 2 checked
        it (the batch then refuses the image it names)."""
        if info.data.get("id") not in _unknown_image_rows(info):
            return cast(str, handler(value))
        try:
            return _IMAGE_ID.validate_python(value, strict=True)
        except ValidationError as exc:
            [error] = exc.errors()
            raise PydanticKnownError(error["type"], error.get("ctx")) from None

    @property
    def fitted_size(self) -> BoxSize:
        """The size the bands asked for: ``box_size`` less ``box_padding`` on each side."""
        padding = self.box_padding
        return BoxSize(
            width=self.box_size.width - 2 * padding.across,
            height=self.box_size.height - 2 * padding.along,
        )


class Protein(_Model):
    """One protein (the successor of ``Analysis``): its name, role and
    expectations, and its row on the membrane area it is quantified on.

    A target normalizes against ``loading_control_ids``; an empty list means
    the batch's single loading control. Its rows' boxes and records are
    checked here, so a message names the protein.
    """

    id: ProteinId
    name: ProteinName  # unique in the batch (exact match)
    role: Role
    loading_control_ids: list[ProteinId] = Field(default_factory=list)  # targets only
    expected_mw: Kda | None = None
    expected_band_count: int = Field(default=1, ge=1)
    mw_tolerance: Annotated[Finite, Field(gt=0, lt=1)] = 0.10  # relative: 0.10 = ±10%
    rows: list[Row] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_box_padding(self) -> Protein:
        # The range (at most half the fitted size) is a rule of the operations
        # that set a padding, not of the model: a later fit keeps the padding
        # whatever size it fits, and must never make a file unloadable.
        for row in self.rows:
            size, padding = row.box_size, row.box_padding
            under = []
            if size.width <= 2 * padding.across:
                under.append(f"{padding.across} px of padding left and right")
            if size.height <= 2 * padding.along:
                under.append(f"{padding.along} px of padding above and below")
            if under:
                raise ValueError(
                    f"protein {self.id}: box size {size.width}x{size.height} leaves no fitted"
                    f" size under {' and '.join(under)}"
                )
        return self

    @model_validator(mode="after")
    def _check_bands_and_loading_controls(self) -> Protein:
        for row in self.rows:
            # Band order carries no meaning: sort so equal rows give equal bytes.
            # #58 counts extra bands (Band.bands_found) instead of boxing them; a box at
            # band index > 0 stays loadable for files that hold one.
            row.bands.sort(key=lambda band: (band.lane_index, band.band_index))
            for a, b in itertools.pairwise(row.bands):
                if (a.lane_index, a.band_index) == (b.lane_index, b.band_index):
                    raise ValueError(
                        f"protein {self.id}: two bands in lane {lane_number(a.lane_index)}"
                        f" with band index {a.band_index}"
                    )
            # Boxes of one row must not overlap; other rows' boxes may.
            rects = [(band.id, band.box.rect(row.box_size)) for band in row.bands]
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
        for row in self.rows:
            # Record order carries no meaning either: sorted like the bands.
            row.undetected.sort(key=lambda record: (record.lane_index, record.band_index))
            for a, b in itertools.pairwise(row.undetected):
                if (a.lane_index, a.band_index) == (b.lane_index, b.band_index):
                    raise ValueError(
                        f"protein {self.id}: two not-detected records in lane"
                        f" {lane_number(a.lane_index)} with band index {a.band_index}"
                    )
            held = {(band.lane_index, band.band_index) for band in row.bands}
            for record in row.undetected:
                if (record.lane_index, record.band_index) in held:
                    raise ValueError(
                        f"protein {self.id}: lane {lane_number(record.lane_index)} has both a"
                        f" band and a not-detected record for band index {record.band_index}"
                    )
                if record.band_index >= self.expected_band_count:
                    raise ValueError(
                        f"protein {self.id}: a not-detected record for band index"
                        f" {record.band_index}, but {self.expected_band_count} band(s) are"
                        " expected"
                    )
        return self


class Boxes(Protocol):
    """A protein's boxes as :meth:`Batch.boxes_of` gives them, to read and
    write: their quantified size and padding, the bands, and the not-detected
    records (its row's); nothing else of the protein."""

    box_size: BoxSize
    box_padding: BoxPadding
    bands: list[Band]
    undetected: list[UndetectedBand]

    @property
    def fitted_size(self) -> BoxSize: ...


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

# How a refusal of what a newer build writes ends.
_NEWER: Final = "which this Proteia build cannot read; open the project with a newer build"


def takes_point(
    own: Arena, other: Arena, other_first: bool, image: ImageRef, x: float, y: float
) -> bool:
    """Whether ``other``, another membrane area of ``own``'s image, takes the
    point ``(x, y)`` that ``own`` holds: ``other`` holds it too, and the point
    lies past the edge of one of them, or on the edge of both (the line two
    areas share, as when one is cut from the other) with ``other``'s membrane
    the earlier (``other_first``). A box, a not-detected record and a
    calibration point belong to one area, so a point on a shared edge belongs
    to the earlier membrane's."""
    if not other.contains(image, x, y):
        return False
    return other_first or own.inside(image, x, y) or other.inside(image, x, y)


def _sharing(
    own: Arena, others: Iterable[tuple[Arena, bool]], image: ImageRef, x: float, y: float
) -> Arena | None:
    """The first of ``others`` (other areas of the image, each with whether
    its membrane comes before ``own``'s) that takes the point ``(x, y)`` from
    ``own`` (:func:`takes_point`), or None."""
    return next(
        (other for other, first in others if takes_point(own, other, first, image, x, y)), None
    )


class Batch(_Model):
    """One run as stored: its images, lane tables, membranes and proteins.

    Its statistics input is :class:`proteia.core.analyze.Batch`, which the compute
    step builds from it (loading-control ids become protein names).
    """

    # Every image of every membrane, once, in membrane then image order (the
    # import keeps it so): a membrane reaches its images through its areas.
    images: list[ImageRef] = Field(default_factory=list)
    lane_tables: list[LaneTable] = Field(default_factory=list)
    reference_condition: str | None = None  # a lane label: the fold-change reference
    membranes: list[Membrane] = Field(default_factory=list)
    proteins: list[Protein] = Field(default_factory=list)

    # --- Access points (#222, #223) ---
    # Code outside this module reads a membrane's images and register groups,
    # the lanes, and a protein's image, membrane and boxes through these, one
    # per concept, so that a change of the stored shape changes them and not
    # their readers. The batch holds the images; a membrane holds its area on
    # each of its images and names its lane table; a protein holds its boxes
    # in its one row, on one area. tests/test_accessors.py lists any access
    # that goes around them.

    def iter_images(self) -> Iterator[ImageRef]:
        """Every image, in membrane then image order."""
        yield from self.images

    def images_of(self, membrane: Membrane) -> list[ImageRef]:
        """A membrane's images, in the order of its areas."""
        images = {image.id: image for image in self.images}
        return [images[arena.image_id] for arena in membrane.arenas]

    def find_image(self, image_id: str) -> ImageRef:
        for image in self.images:
            if image.id == image_id:
                return image
        raise UnknownIdError(f"unknown image {image_id!r}")

    def arenas_on(self, image_id: str) -> list[tuple[Membrane, Arena]]:
        """The membrane areas on an image, each with its membrane, in membrane
        order; [] for an unknown image."""
        return [
            (membrane, arena)
            for membrane in self.membranes
            for arena in membrane.arenas
            if arena.image_id == image_id
        ]

    def membranes_on(self, image_id: str) -> list[Membrane]:
        """The membranes with an area on an image, in membrane order; [] for
        an unknown image."""
        return [membrane for membrane, _ in self.arenas_on(image_id)]

    def membrane_of(self, image_id: str) -> Membrane:
        """The membrane on an image that holds one membrane's area, as the
        golden fixtures' generator (``tests/data/schema2``, frozen) reads it;
        ``UnknownIdError`` for an unknown image, ``ValueError`` for one holding
        several. The package reads :meth:`membranes_on` or an area's id."""
        found = self.membranes_on(image_id)
        if not found:
            raise UnknownIdError(f"unknown image {image_id!r}")
        if len(found) > 1:
            raise ValueError(f"image {image_id} holds {len(found)} membranes' areas")
        return found[0]

    def membrane_number(self, membrane: Membrane) -> int:
        """A membrane's number as the user reads it: its place in the batch,
        from 1 (a removal renumbers the later ones)."""
        for number, held in enumerate(self.membranes, start=1):
            if held is membrane or held.id == membrane.id:
                return number
        raise UnknownIdError(f"unknown membrane {membrane.id!r}")

    def register_groups(self, membrane: Membrane) -> list[frozenset[str]]:
        """A membrane's images whose pixel rows line up: the connected
        components of the links from each chemiluminescence image to its
        marker image (a visible-light marker or merged image). An unlinked
        image is its own group. Groups come in the order of their first image."""
        images = self.images_of(membrane)
        parent = {image.id: image.id for image in images}

        def root(image_id: str) -> str:
            while parent[image_id] != image_id:
                image_id = parent[image_id]
            return image_id

        for image in images:
            marker = image.marker_image_id
            if marker in parent:
                a, b = root(image.id), root(marker)
                if a != b:
                    parent[a] = b
        groups: dict[str, set[str]] = {}
        for image in images:
            groups.setdefault(root(image.id), set()).add(image.id)
        return [frozenset(members) for members in groups.values()]

    def group_of(self, membrane: Membrane, image_id: str) -> frozenset[str]:
        """The register group of ``membrane`` holding ``image_id``;
        ``UnknownIdError`` if it is not on the membrane."""
        for group in self.register_groups(membrane):
            if image_id in group:
                return group
        raise UnknownIdError(f"unknown image {image_id!r} on membrane {membrane.id}")

    def only_lanes(self) -> list[Lane]:
        """The lanes of the project's lane table, [] before it has one: a
        project holds one lane table at most (:meth:`_check_supported`), which
        every membrane uses."""
        return self.lane_tables[0].lanes if self.lane_tables else []

    def lane_table_of(self, membrane: Membrane) -> LaneTable:
        """The lane table a membrane names."""
        for table in self.lane_tables:
            if table.id == membrane.lane_table_id:
                return table
        raise UnknownIdError(f"unknown lane table {membrane.lane_table_id!r}")

    def find_arena(self, arena_id: str) -> tuple[Membrane, Arena]:
        """A membrane area and the membrane that holds it."""
        for membrane in self.membranes:
            for arena in membrane.arenas:
                if arena.id == arena_id:
                    return membrane, arena
        raise UnknownIdError(f"unknown membrane area {arena_id!r}")

    def only_row(self, protein: Protein) -> Row:
        """A protein's row: a protein holds one (:meth:`_check_supported`)."""
        [row] = protein.rows
        return row

    def lanes_of_protein(self, protein: Protein) -> list[Lane]:
        """The lanes a protein's bands and records index: its membrane's lane table's."""
        return self.lane_table_of(self.membrane_of_protein(protein)).lanes

    def image_id_of(self, protein: Protein) -> str:
        """The id of the image a protein is quantified on: its row's area's,
        found by a scan of the membrane areas (:meth:`image_of` also looks
        the image up)."""
        return self.find_arena(self.only_row(protein).arena_id)[1].image_id

    def image_of(self, protein: Protein) -> ImageRef:
        """The image a protein is quantified on (a lookup: where only its id
        is needed, :meth:`image_id_of`)."""
        return self.find_image(self.image_id_of(protein))

    def membrane_of_protein(self, protein: Protein) -> Membrane:
        """The membrane of the area a protein is quantified on."""
        return self.find_arena(self.only_row(protein).arena_id)[0]

    def arena_of(self, protein: Protein) -> Arena:
        """The membrane area a protein is quantified on (its row's; a lookup:
        where only its id is needed, :meth:`arena_id_of`)."""
        return self.find_arena(self.arena_id_of(protein))[1]

    def arena_id_of(self, protein: Protein) -> str:
        """The id of the membrane area a protein is quantified on: its row's."""
        return self.only_row(protein).arena_id

    def arenas_of(self, membrane: Membrane) -> list[Arena]:
        """A membrane's areas, one per image it is on, in its order."""
        return membrane.arenas

    def area_on(self, membrane: Membrane, image_id: str) -> Arena:
        """A membrane's area on one of its images (``UnknownIdError`` if it
        has none there)."""
        for arena in membrane.arenas:
            if arena.image_id == image_id:
                return arena
        raise UnknownIdError(f"membrane {membrane.id} has no area on {image_id!r}")

    def proteins_on(self, image_id: str) -> list[Protein]:
        """The proteins quantified on an image, in batch order; [] for none."""
        on_image = {arena.id for _, arena in self.arenas_on(image_id)}
        return [p for p in self.proteins if self.only_row(p).arena_id in on_image]

    def rows_on(self, image_id: str) -> list[tuple[Protein, Row]]:
        """The rows on an image, of every membrane area there, each with its
        protein: protein order, then row order; [] for none."""
        on_image = {arena.id for _, arena in self.arenas_on(image_id)}
        return [(p, row) for p in self.proteins for row in p.rows if row.arena_id in on_image]

    def rows_on_arena(self, arena_id: str) -> list[tuple[Protein, Row]]:
        """The rows on one membrane area, each with its protein: protein
        order, then row order; [] for none."""
        return [(p, row) for p in self.proteins for row in p.rows if row.arena_id == arena_id]

    def boxes_of(self, protein: Protein) -> Boxes:
        """A protein's boxes, to read and write (:class:`Boxes`): its row."""
        return self.only_row(protein)

    def find_protein(self, protein_id: str) -> Protein:
        for protein in self.proteins:
            if protein.id == protein_id:
                return protein
        raise UnknownIdError(f"unknown protein {protein_id!r}")

    def find_band(self, band_id: str) -> tuple[Protein, Band]:
        for protein in self.proteins:
            for band in self.only_row(protein).bands:
                if band.id == band_id:
                    return protein, band
        raise UnknownIdError(f"unknown band {band_id!r}")

    # --- Validation ---

    def _check_supported(self) -> None:
        """What this build reads: one lane table and one row per protein. A
        file with more was written by a build that adds lane tables or puts a
        protein on several membranes; the message names the file's part, never
        an id."""
        if len(self.lane_tables) > 1:
            raise ValueError(f"the project holds {len(self.lane_tables)} lane tables, {_NEWER}")
        for protein in self.proteins:
            if len(protein.rows) > 1:
                raise ValueError(
                    f"{protein.name!r} has rows on {len(protein.rows)} membrane areas, {_NEWER}"
                )

    def _check_shared_images(
        self, images: dict[str, ImageRef], twice: set[str]
    ) -> dict[str, list[tuple[Arena, bool]]]:
        """An image holding several membranes' areas holds none over the
        whole image: each has its outline (a whole-image area would hold every
        point of the others). Gives, per area, the other areas of its image
        whose bounds meet its own, the only ones that can share a point with it
        (:meth:`Arena.bounds`: whole pixels past its corners, so past the
        edge's tolerance too), each with whether its membrane comes first
        (:func:`takes_point`).
        The areas of an image listed ``twice`` are left to the id check."""
        on: dict[str, list[Arena]] = {}
        for membrane in self.membranes:
            for arena in membrane.arenas:
                on.setdefault(arena.image_id, []).append(arena)
        others: dict[str, list[tuple[Arena, bool]]] = {}
        for image_id, arenas in on.items():
            image = images.get(image_id)
            if image is None or image_id in twice or len(arenas) < 2:
                continue
            for arena in arenas:
                if arena.outline is None:
                    raise ValueError(
                        f"image {image_id} holds {len(arenas)} membrane areas and area"
                        f" {arena.id} has no outline: an image with several areas holds none"
                        " over the whole image"
                    )
            bounds = {arena.id: arena.bounds(image) for arena in arenas}
            rank = {arena.id: number for number, arena in enumerate(arenas)}  # membrane order
            for arena in arenas:
                x0, y0, x1, y1 = bounds[arena.id]
                others[arena.id] = [
                    (other, rank[other.id] < rank[arena.id])
                    for other in arenas
                    if other is not arena
                    and bounds[other.id][0] <= x1
                    and x0 <= bounds[other.id][2]
                    and bounds[other.id][1] <= y1
                    and y0 <= bounds[other.id][3]
                ]
        return others

    def _check_image_order(self, images: dict[str, ImageRef]) -> None:
        """The images are listed membrane by membrane, in the order of each
        membrane's areas (:attr:`images`); every image has an area and every
        area an image by now."""
        listed = dict.fromkeys(image.id for image in self.images)
        placed = dict.fromkeys(a.image_id for membrane in self.membranes for a in membrane.arenas)
        for here, due in zip(listed, placed, strict=True):
            if here != due:
                raise ValueError(
                    f"the project lists {images[here].original_name!r} out of order: images are"
                    " listed membrane by membrane, in the order of each membrane's areas"
                )

    @model_validator(mode="after")
    def _check_references(self, info: ValidationInfo) -> Batch:
        images = {image.id: image for image in self.images}
        listed = Counter(image.id for image in self.images)
        twice = {image_id for image_id, count in listed.items() if count > 1}
        self._check_supported()
        tables = {table.id: table for table in self.lane_tables}
        arenas: dict[str, tuple[Membrane, Arena]] = {}
        for membrane in self.membranes:
            if membrane.lane_table_id not in tables:
                raise ValueError(
                    f"membrane {membrane.id}: lane table {membrane.lane_table_id!r} is not one"
                    " of the project's"
                )
            for arena in membrane.arenas:
                image = images.get(arena.image_id)
                if image is None:
                    raise ValueError(
                        f"membrane {membrane.id}: area {arena.id} is on {arena.image_id!r},"
                        " which is not an image of the project"
                    )
                if arena.outline is not None:
                    cx, cy = arena.outline.centre
                    if not (0 <= cx <= image.width and 0 <= cy <= image.height):
                        raise ValueError(
                            f"membrane {membrane.id}: the centre ({cx:g}, {cy:g}) of area"
                            f" {arena.id} lies outside the {image.width}x{image.height}"
                            f" image {image.id}"
                        )
                arenas[arena.id] = (membrane, arena)
        on_some = {arena.image_id for _, arena in arenas.values()}
        for image in self.images:
            if image.id not in on_some:
                raise ValueError(f"image {image.id} is on no membrane: no membrane area is on it")
        self._check_image_order(images)
        others = self._check_shared_images(images, twice)
        for membrane in self.membranes:
            self._check_membrane(membrane, images, others)
        self._check_linked_areas(images, twice)
        # After the membranes, before the proteins, as schema 2 checked them.
        if self.reference_condition is not None and self.reference_condition not in {
            lane.label for table in self.lane_tables for lane in table.lanes
        }:
            raise ValueError(
                f"reference condition {self.reference_condition!r} is not a lane condition"
            )
        self._check_proteins(images, tables, arenas, others, _unknown_image_rows(info))
        return self

    def _check_membrane(
        self,
        membrane: Membrane,
        images: dict[str, ImageRef],
        others: dict[str, list[tuple[Arena, bool]]],
    ) -> None:
        """A membrane's marker links, calibration points and ladders. A point
        lies in its membrane's area on its image and in no other area there
        (``others``: :meth:`_check_shared_images`)."""
        own = {arena.image_id: arena for arena in membrane.arenas}
        for image_id in own:
            image = images[image_id]
            if image.marker_image_id is None:
                continue
            marker = images.get(image.marker_image_id)
            if marker is None or marker.id not in own or marker.kind not in _MARKER_KINDS:
                raise ValueError(
                    f"image {image.id}: marker image {image.marker_image_id!r} is not"
                    f" a visible-light marker or merged image of membrane {membrane.id}"
                )
            if (marker.width, marker.height) != (image.width, image.height):
                raise ValueError(
                    f"image {image.id} ({image.width}x{image.height}) and its marker image"
                    f" {marker.id} ({marker.width}x{marker.height}) differ in size;"
                    " linked images must match pixel for pixel"
                )
        for point in membrane.calibration.points:
            arena = own.get(point.image_id)
            if arena is None:
                raise ValueError(
                    f"membrane {membrane.id}: calibration point on {point.image_id!r},"
                    " which is not an image of this membrane"
                )
            image = images[point.image_id]
            if point.y > image.height:
                raise ValueError(
                    f"membrane {membrane.id}: calibration point at y={point.y}"
                    f" is below the bottom of {image.id}"
                )
            if point.x is not None and point.x > image.width:
                raise ValueError(
                    f"membrane {membrane.id}: calibration point at x={point.x}"
                    f" is right of the edge of {image.id}"
                )
            # A whole-image area's bounds are the image's, checked above; a drawn
            # one's are its turned outline, where a mark across the whole width (no
            # x) is read on the outline's middle.
            outline = arena.outline
            if outline is None:
                continue
            x = outline.centre[0] if point.x is None else point.x
            if not arena.contains(image, x, point.y):
                raise ValueError(
                    f"membrane {membrane.id}: calibration point at y={point.y} on {image.id}"
                    f" lies outside its membrane area {arena.id}"
                )
            shared = _sharing(arena, others.get(arena.id, ()), image, x, point.y)
            if shared is not None:
                raise ValueError(
                    f"membrane {membrane.id}: calibration point at y={point.y} on {image.id}"
                    f" lies in membrane areas {arena.id} and {shared.id}"
                )
        for group in self.register_groups(membrane):
            self._check_ladders(membrane, group)

    def _check_linked_areas(self, images: dict[str, ImageRef], twice: set[str]) -> None:
        """Linked images hold the same membrane areas: each membrane's area on
        the one equals its area on the other, in outline and angle. A link
        to or from an image listed ``twice`` is left to the id check."""
        areas: dict[str, dict[str, tuple[Outline | None, float]]] = {}
        for membrane in self.membranes:
            for arena in membrane.arenas:
                areas.setdefault(arena.image_id, {})[membrane.id] = (arena.outline, arena.angle_deg)
        for image in images.values():
            marker = image.marker_image_id
            if marker is None or not twice.isdisjoint((image.id, marker)):
                continue
            if areas.get(image.id) != areas.get(marker):
                raise ValueError(
                    f"image {image.id} and its marker image {marker} hold different membrane"
                    " areas; linked images hold the same areas"
                )

    def _check_ladders(self, membrane: Membrane, group: frozenset[str]) -> None:
        """The points of one register group: per ladder side, one point per MW and
        per position, MWs decreasing down the image; a right ladder only beside
        ladder bands with x, and right of the left one.

        MWs are compared on log10(MW), which :mod:`proteia.core.mwcal` fits: two
        MWs a float step apart (100 and 100.00000000000001) share one log10, and
        would give its curve a flat segment."""
        names = ", ".join(arena.image_id for arena in membrane.arenas if arena.image_id in group)
        points = [point for point in membrane.calibration.points if point.image_id in group]
        for side in LadderSide:
            ladder = sorted((p for p in points if p.side == side), key=lambda p: (p.y, p.mw))
            where = f"on the {side.value} ladder of {names}"
            zs: set[float] = set()
            for point in ladder:
                z = math.log10(point.mw)
                if z in zs:
                    raise ValueError(
                        f"membrane {membrane.id}: two calibration points at {point.mw:g} kDa"
                        f" {where}"
                    )
                zs.add(z)
            for above, below in itertools.pairwise(ladder):
                if above.y == below.y:
                    raise ValueError(
                        f"membrane {membrane.id}: two calibration points at y={below.y} {where}"
                    )
            for above, below in itertools.pairwise(ladder):
                if not math.log10(below.mw) < math.log10(above.mw):
                    raise ValueError(
                        f"membrane {membrane.id}: calibration points out of order {where}:"
                        f" {below.mw:g} kDa at y={below.y} lies below"
                        f" {above.mw:g} kDa at y={above.y}"
                    )
        right = [p for p in points if p.side == LadderSide.RIGHT]
        if not right:
            return
        for point in points:
            strip_edge = point.source is CalibrationPointSource.STRIP_EDGE
            if strip_edge or point.x is None:
                what = "is a strip edge" if strip_edge else "has none"
                raise ValueError(
                    f"membrane {membrane.id}: {names} {'has' if len(group) == 1 else 'have'}"
                    " a right ladder, so every calibration point there needs a ladder"
                    f" band's x ({point.mw:g} kDa {what})"
                )
        left_x = [p.x for p in points if p.side == LadderSide.LEFT and p.x is not None]
        right_x = min(p.x for p in right if p.x is not None)
        if left_x and not max(left_x) < right_x:
            raise ValueError(
                f"membrane {membrane.id}: the right ladder of {names} (x={right_x}) is not"
                f" right of its left ladder (x={max(left_x)})"
            )

    def _check_proteins(
        self,
        images: dict[str, ImageRef],
        tables: dict[str, LaneTable],
        arenas: dict[str, tuple[Membrane, Arena]],
        others: dict[str, list[tuple[Arena, bool]]],
        unknown_image_rows: Collection[str],
    ) -> None:
        """Each protein's name, loading controls and rows: each row on a known
        area of a signal image, its box size within the image, and each box and
        record inside the image, in a lane of its membrane's table, and centred
        in its area and in no other area of the image (``others``:
        :meth:`_check_shared_images`). A row of ``unknown_image_rows``
        (:data:`UNKNOWN_IMAGE_ROWS`) is refused for the image its schema-2
        protein named."""
        roles = {protein.id: protein.role for protein in self.proteins}
        names: set[str] = set()
        for protein in self.proteins:
            if protein.name in names:
                raise ValueError(f"duplicate protein name {protein.name!r}")
            names.add(protein.name)
            placed: list[tuple[Row, Arena, ImageRef, list[Lane]]] = []
            for row in protein.rows:
                found = arenas.get(row.arena_id)
                if found is None and row.id in unknown_image_rows:
                    raise ValueError(f"protein {protein.id}: unknown image {row.arena_id!r}")
                if found is None:
                    raise ValueError(
                        f"protein {protein.id}: unknown membrane area {row.arena_id!r}"
                    )
                membrane, arena = found
                image = images[arena.image_id]
                if image.kind is ImageKind.VISIBLE_MARKER:
                    raise ValueError(
                        f"protein {protein.id}: {image.id} is a visible-light marker image,"
                        " not a signal image"
                    )
                size = row.box_size
                if size.width > image.width or size.height > image.height:
                    raise ValueError(
                        f"protein {protein.id}: box size {size.width}x{size.height}"
                        f" exceeds the bounds of image {image.id}"
                    )
                placed.append((row, arena, image, tables[membrane.lane_table_id].lanes))
            for lc_id in protein.loading_control_ids:
                if lc_id not in roles:
                    raise ValueError(f"protein {protein.id}: unknown loading control {lc_id!r}")
                if roles[lc_id] is not Role.LOADING_CONTROL:
                    raise ValueError(f"protein {protein.id}: {lc_id} is not a loading control")
            for row, arena, image, lanes in placed:
                self._check_row(protein, row, arena, others.get(arena.id, ()), image, len(lanes))

    @staticmethod
    def _check_row(
        protein: Protein,
        row: Row,
        arena: Arena,
        others: Sequence[tuple[Arena, bool]],
        image: ImageRef,
        lanes: int,
    ) -> None:
        for band in row.bands:
            if band.lane_index >= lanes:
                raise ValueError(
                    f"protein {protein.id}: band {band.id} references"
                    f" unknown lane {lane_number(band.lane_index)}"
                )
            x0, y0, x1, y1 = band.box.rect(row.box_size)
            if x1 > image.width or y1 > image.height:
                raise ValueError(
                    f"protein {protein.id}: box of {band.id}"
                    f" extends beyond the bounds of image {image.id}"
                )
            if not arena.contains(image, (x0 + x1) / 2, (y0 + y1) / 2):
                raise ValueError(
                    f"protein {protein.id}: the centre of {band.id}'s box lies outside its"
                    f" membrane area {arena.id}"
                )
            shared = _sharing(arena, others, image, (x0 + x1) / 2, (y0 + y1) / 2)
            if shared is not None:
                raise ValueError(
                    f"protein {protein.id}: the centre of {band.id}'s box lies in membrane"
                    f" areas {arena.id} and {shared.id}"
                )
        for record in row.undetected:
            if record.lane_index >= lanes:
                raise ValueError(
                    f"protein {protein.id}: a not-detected record references"
                    f" unknown lane {lane_number(record.lane_index)}"
                )
            region = record.region
            if region.x1 > image.width or region.y1 > image.height:
                raise ValueError(
                    f"protein {protein.id}: the not-detected region in lane"
                    f" {lane_number(record.lane_index)} extends beyond the bounds of"
                    f" image {image.id}"
                )
            cx, cy = (region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2
            if not arena.contains(image, cx, cy):
                raise ValueError(
                    f"protein {protein.id}: the not-detected region in lane"
                    f" {lane_number(record.lane_index)} lies outside its membrane area"
                    f" {arena.id}"
                )
            shared = _sharing(arena, others, image, cx, cy)
            if shared is not None:
                raise ValueError(
                    f"protein {protein.id}: the not-detected region in lane"
                    f" {lane_number(record.lane_index)} lies in membrane areas {arena.id} and"
                    f" {shared.id}"
                )


class Project(_Model):
    """The whole saved project: ``project.json`` is its JSON form."""

    schema_version: Literal[3] = SCHEMA_VERSION
    background_method: ProjectBackgroundMethod = LOCAL_BACKGROUND_METHOD
    # Bookkeeping, excluded from the content hash: the number of the next new id.
    next_id: int = Field(default=1, ge=1, le=NEXT_ID_LIMIT)
    batch: Batch = Field(default_factory=Batch)
    log: tuple[LogEntry, ...] = ()  # history, oldest first; excluded from the content hash

    def iter_ids(self) -> Iterator[str]:
        """Every object id: the kinds schema 2 held, membranes, images,
        proteins and bands (by protein), then the kinds schema 3 added, lane
        tables, membrane areas (by membrane) and rows (by protein);
        not-detected records have none. :meth:`_check_ids` meets them in this
        order, so a schema-2 file whose ids are wrong is refused for the id it
        was refused for before, never one its conversion drew."""
        batch = self.batch
        yield from (membrane.id for membrane in batch.membranes)
        yield from (image.id for image in batch.images)
        yield from (protein.id for protein in batch.proteins)
        yield from (
            band.id for protein in batch.proteins for row in protein.rows for band in row.bands
        )
        yield from (table.id for table in batch.lane_tables)
        yield from (arena.id for membrane in batch.membranes for arena in membrane.arenas)
        yield from (row.id for protein in batch.proteins for row in protein.rows)

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
    def _check_background_method(self) -> Project:
        # One method for the whole batch; under the legacy one, the legacy values.
        legacy = self.background_method == LEGACY_BACKGROUND_METHOD
        batch = self.batch
        images = {image.id: image for image in batch.images}
        medians = {
            arena.id: images[arena.image_id].background
            for membrane in batch.membranes
            for arena in membrane.arenas
        }
        for protein in batch.proteins:
            for row in protein.rows:
                for band in row.bands:
                    self._check_legacy_band(band, legacy, medians[row.arena_id])
        return self

    def _check_legacy_band(self, band: Band, legacy: bool, median: float) -> None:
        if (band.background_mode == LEGACY_BACKGROUND_METHOD) != legacy:
            raise ValueError(
                f"band {band.id}: background mode {band.background_mode!r} does not"
                f" belong to the project's background method {self.background_method!r}"
            )
        if legacy and (band.background_level != median or band.background_spread != 0):
            raise ValueError(
                f"band {band.id}: a {LEGACY_BACKGROUND_METHOD} background is its"
                " image's median, with no spread"
            )

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
