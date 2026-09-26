# SPDX-License-Identifier: Apache-2.0
"""The project operations: one function per user action, with no GUI.

Every function takes an open :class:`~proteia.core.session.ProjectSession` (from
:func:`new_project` or :func:`open_project`) and holds its lock. A state change
runs as one :func:`~proteia.core.model.apply_change` on a copy of the project,
which re-validates the whole tree, and is committed through the session, which
then autosaves. So an edit is all or nothing:

* Checks run on the committed project first and raise :class:`OperationError`
  with a stable :class:`ErrorCode`; an unknown id raises
  :class:`~proteia.core.model.UnknownIdError`. A refusal changes nothing: the
  project (the same object), ``next_id``, the pixel cache and the ``images/``
  listing are as they were, and the autosave hook does not run.
* Pixels are fetched before the change, so an image-file problem changes nothing.
  New ids come only from ``new_id`` inside the change, so a refusal uses up no
  number.
* An edit that leaves the project equal to the committed one is a no-op: nothing
  is committed and the hook does not run.
* Every committed change appends one log entry (see
  :class:`~proteia.core.model.LogEntry`); a refusal or a no-op appends none, and
  so do :func:`compute`, :func:`export_lane_table` and :func:`save`, which change
  no state. The params record the inputs as they took effect (cleaned text, the
  stored spelling, the proposed lane, the snapped rect, the size used) and the
  ids created or removed; objects are named by id, never by path or typed text.
  Clients commit a drag or a cell edit once, when it ends, not on every pointer
  move or keystroke.

Stored values that depend on pixels or geometry are recomputed by the operation
that invalidates them: :func:`_quantify` is the one place a band's net is
computed, with its ``clipped`` flag, and :func:`_set_box` the
one place a box moves (it clears the position-derived ``apparent_mw``). A size
change or a polarity change recomputes every affected net, so every stored net
always equals ``net_signal`` of the stored pixels, box, size, background and
polarity, and every clipping flag these operations store is ``is_clipped`` of
the same.

A not-detected record (:class:`~proteia.core.model.UndetectedBand`) is a
detector's measurement that cannot be redone from the model alone, so an edit
that invalidates one drops it, in the same change, and logs it in full: a box
placed or moved into its lane replaces it (``replaced_undetected``), and a
polarity change or a lane table that cuts its lane drops it
(``dropped_undetected``). Removing a box never creates a record: the lane
becomes "not measured".

Functions return ids or small frozen dataclasses, never model objects. Typed text
follows :mod:`proteia.core.names`; box placement follows :mod:`proteia.core.boxes`.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Concatenate, Final

import numpy as np
from pydantic import JsonValue, ValidationError

from proteia.core import boxes, record, results, storage
from proteia.core.analyze import ReduceMethod
from proteia.core.export import LANE_COLUMNS, lane_table_bytes
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.imaging import clipping_depth, load_image
from proteia.core.model import (
    IMAGE_SUFFIXES,
    Band,
    Batch,
    Box,
    BoxSize,
    ImageKind,
    ImageRef,
    Lane,
    Membrane,
    Polarity,
    Project,
    ProposalSource,
    Protein,
    Rect,
    Role,
    UndetectedBand,
    UnknownIdError,
    apply_change,
    overlaps,
)
from proteia.core.names import (
    TextError,
    clean_optional,
    clean_text,
    name_key,
    resolve_label,
    unify_spellings,
)
from proteia.core.plotspec import ErrorType
from proteia.core.project import lane_anchors, propose_lane, spine_axes
from proteia.core.quantify import estimate_background, is_clipped, net_signal
from proteia.core.results import Results
from proteia.core.session import (
    ErrorCode,
    OperationError,
    ProjectSession,
    new_project,
    open_project,
)

__all__ = [
    "KEEP",
    "LANE_TABLE_FILE",
    "LANE_TABLE_RECORD_FILE",
    "Cascade",
    "ErrorCode",
    "Keep",
    "LaneInput",
    "LanesUpdate",
    "OperationError",
    "ProjectSession",
    "add_protein",
    "compute",
    "edit_protein",
    "export_lane_table",
    "import_image",
    "move_box",
    "new_project",
    "open_project",
    "place_box",
    "remove_box",
    "remove_image",
    "remove_protein",
    "remove_undetected",
    "save",
    "set_box_lane",
    "set_box_size",
    "set_lanes",
    "set_polarity",
    "set_reference_condition",
]

# The lane table export and its record: fixed names chosen here, never from client text.
LANE_TABLE_FILE: Final = "lane-table.csv"
LANE_TABLE_RECORD_FILE: Final = "lane-table.record.json"
# A protein name must not read as one of the lane table's own columns.
_RESERVED_KEYS: Final = frozenset(name_key(column) for column in LANE_COLUMNS)


class Keep(Enum):
    """The type of :data:`KEEP`."""

    KEEP = "keep"


# "Leave unchanged", where None is a real value (no expected MW, no reference).
KEEP: Final = Keep.KEEP


@dataclass(frozen=True)
class Cascade:
    """What a removal took with it."""

    removed: tuple[str, ...]  # every removed id: the object, then proteins, bands, membrane
    detached_targets: tuple[str, ...]  # targets that lost a loading control
    unpaired_images: tuple[str, ...]  # images whose marker_image_id was cleared
    unfitted_membranes: tuple[str, ...]  # lost calibration points: fit and apparent MWs cleared


@dataclass(frozen=True)
class LaneInput:
    """One row of the lane table as the user typed it."""

    condition: str
    sample: str | None = None
    included: bool = True


@dataclass(frozen=True)
class LanesUpdate:
    """What :func:`set_lanes` did beyond storing the table."""

    respelled: tuple[int, ...]  # lanes whose condition or sample took an existing spelling
    reference_cleared: bool  # the reference no longer named a lane and was cleared
    # (protein id, lane index, band index) of each not-detected record in a dropped lane
    dropped_undetected: tuple[tuple[str, int, int], ...] = ()


# --- Common machinery ---


def _locked[**P, R](
    fn: Callable[Concatenate[ProjectSession, P], R],
) -> Callable[Concatenate[ProjectSession, P], R]:
    """Run an operation under the session lock. A refusal leaves the pixel cache
    as it was, even if the operation read pixels before refusing."""

    @functools.wraps(fn)
    def locked(session: ProjectSession, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with session.transaction():
            return fn(session, *args, **kwargs)

    return locked


def _invalid(message: str, *, ids: Sequence[str] = ()) -> OperationError:
    return OperationError(ErrorCode.INVALID_INPUT, message, ids=ids)


def _prepare[T](session: ProjectSession, change: Callable[[Project], T]) -> tuple[Project, T]:
    """Run ``change`` on a copy of the committed project and re-validate it; a
    ``ValidationError`` becomes ``INVALID_INPUT`` with its first message."""
    try:
        return apply_change(session.project, change)
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"]).removeprefix("Value error, ")
        raise _invalid(message) from exc


_Params = Mapping[str, JsonValue]  # a log entry's params


def _apply[T](
    session: ProjectSession,
    action: str,
    change: Callable[[Project], T],
    params: Callable[[T], _Params],
    **commit_kw,
) -> T:
    """:func:`_prepare`, then commit with the log entry ``params(result)`` unless
    the project did not change."""
    new, result = _prepare(session, change)
    if new == session.project:  # a no-op: nothing committed, no entry, no hook
        return result
    session._commit(new, action=action, params=params(result), **commit_kw)
    return result


def _size(size: BoxSize) -> dict[str, JsonValue]:
    return {"width": size.width, "height": size.height}


def _cascade(cascade: Cascade) -> dict[str, JsonValue]:
    return {
        "removed": list(cascade.removed),
        "detached_targets": list(cascade.detached_targets),
        "unpaired_images": list(cascade.unpaired_images),
        "unfitted_membranes": list(cascade.unfitted_membranes),
    }


def _undetected_json(protein_id: str, record: UndetectedBand) -> dict[str, JsonValue]:
    """A not-detected record as a log entry names it: whole, since it has no id."""
    return {
        "protein_id": protein_id,
        "lane_index": record.lane_index,
        "band_index": record.band_index,
        "reason": record.reason.value,
        "snr": record.snr,
        "threshold": record.threshold,
        "region": list(record.region.rect()),
        "source": record.source.value,
    }


def _drop_undetected_where(
    protein: Protein, drop: Callable[[UndetectedBand], bool]
) -> list[dict[str, JsonValue]]:
    """Remove a draft protein's records for which ``drop`` is true; return their
    log forms in the stored order (plain values, as a change should return)."""
    dropped = [
        _undetected_json(protein.id, record) for record in protein.undetected if drop(record)
    ]
    if dropped:
        protein.undetected = [record for record in protein.undetected if not drop(record)]
    return dropped


def _drop_undetected(
    protein: Protein, lane_index: int, band_index: int
) -> dict[str, JsonValue] | None:
    """Remove a draft protein's record at (lane, band index): its log form, or None."""
    dropped = _drop_undetected_where(
        protein, lambda record: (record.lane_index, record.band_index) == (lane_index, band_index)
    )
    return dropped[0] if dropped else None


def _quantify(band: Band, protein: Protein, image: ImageRef, array: np.ndarray) -> None:
    """The one place a stored net and its clipping flag are computed. An image
    without a limit the check can trust leaves the band unchecked (None)."""
    dark_on_light = image.polarity.dark_on_light
    band.net = net_signal(
        array, band.box, protein.box_size, image.background, dark_on_light=dark_on_light
    )
    band.clipped = is_clipped(
        array,
        band.box,
        protein.box_size,
        bit_depth=clipping_depth(image.bit_depth, image.import_warnings),
        dark_on_light=dark_on_light,
    )


def _set_box(band: Band, rect: Rect) -> None:
    band.box = Box(x=rect[0], y=rect[1])
    band.apparent_mw = None  # position-derived (#58); stale once the box changes


def _requantify_image(draft: Project, image_id: str, array: np.ndarray) -> None:
    """Recompute every stored net on one image (its polarity or background changed)."""
    image = draft.batch.find_image(image_id)
    for protein in draft.batch.proteins:
        if protein.image_id == image_id:
            for band in protein.bands:
                _quantify(band, protein, image, array)


# --- Input checks ---


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{what} must be an integer, not {value!r}")
    return value


def _member[E: Enum](cls: type[E], value: object, what: str) -> E:
    try:
        return cls(value)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in cls)
        raise _invalid(f"{what} must be one of {choices}, not {value!r}") from exc


def _clean(text: object, what: str) -> str:
    if not isinstance(text, str):
        raise _invalid(f"{what} must be text, not {text!r}")
    try:
        return clean_text(text)
    except TextError as exc:
        raise OperationError(ErrorCode(exc.code), f"{what} {exc}") from exc


def _clean_optional(text: object, what: str) -> str | None:
    if text is not None and not isinstance(text, str):
        raise _invalid(f"{what} must be text or None, not {text!r}")
    try:
        return clean_optional(text)
    except TextError as exc:
        raise OperationError(ErrorCode(exc.code), f"{what} {exc}") from exc


def _condition(text: object, labels: Sequence[str]) -> str:
    """The lane label a typed condition names (its stored spelling)."""
    cleaned = _clean(text, "condition")
    label = resolve_label(cleaned, labels)
    if label is None:
        raise OperationError(ErrorCode.UNKNOWN_CONDITION, f"no lane has the condition {cleaned!r}")
    return label


def _protein_name(batch: Batch, name: object, *, protein_id: str | None = None) -> str:
    """The stored form of a protein name: unique among the other proteins by
    :func:`~proteia.core.names.name_key` and not a lane-table column."""
    cleaned = _clean(name, "protein name")
    key = name_key(cleaned)
    clash = [p.id for p in batch.proteins if p.id != protein_id and name_key(p.name) == key]
    if clash:
        raise OperationError(
            ErrorCode.DUPLICATE_NAME,
            f"protein {clash[0]} already has the name {cleaned!r}"
            " (names ignore case and look-alike characters)",
            ids=clash,
        )
    if key in _RESERVED_KEYS:
        raise OperationError(
            ErrorCode.RESERVED_NAME, f"{cleaned!r} is the name of a lane-table column"
        )
    # The lane table follows each protein's column with "<name> clipped".
    for other in batch.proteins:
        if other.id != protein_id and (
            key == name_key(f"{other.name} clipped")
            or name_key(f"{cleaned} clipped") == name_key(other.name)
        ):
            raise OperationError(
                ErrorCode.RESERVED_NAME,
                f"{cleaned!r} and {other.name!r} would collide in the lane table, which"
                " names each protein's clipping column '<name> clipped'",
                ids=[other.id],
            )
    return cleaned


def _expected_mw(value: object) -> float | None:
    if value is None:
        return None
    number: float | None = None
    if not isinstance(value, bool) and isinstance(value, int | float):
        try:
            number = float(value)  # an int too large for a float raises OverflowError
        except OverflowError:
            number = None
    if number is None or not math.isfinite(number) or number <= 0:
        raise _invalid(f"expected molecular weight must be a positive number of kDa, not {value!r}")
    return number


def _loading_controls(batch: Batch, protein_id: str | None, role: Role, ids: object) -> list[str]:
    """A checked list of loading-control ids (its order is the series order)."""
    if isinstance(ids, str) or not isinstance(ids, Sequence):
        raise _invalid("loading control ids must be a sequence of protein ids")
    chosen = list(ids)
    if not chosen:
        return []
    if role is not Role.TARGET:
        raise _invalid("only a target has loading controls")
    for lc_id in chosen:
        if batch.find_protein(lc_id).role is not Role.LOADING_CONTROL:
            raise _invalid(f"{lc_id} is not a loading control", ids=(lc_id,))
    if len(set(chosen)) != len(chosen):
        raise _invalid("a loading control is listed twice")
    if protein_id in chosen:
        raise _invalid(f"{protein_id} cannot be its own loading control", ids=(protein_id,))
    return chosen


def _fitting_size(size: object, image: ImageRef) -> BoxSize:
    if not isinstance(size, BoxSize):
        raise _invalid(f"box size must be a BoxSize, not {size!r}")
    if size.width > image.width or size.height > image.height:
        raise OperationError(
            ErrorCode.SIZE_OUT_OF_BOUNDS,
            f"box size {size.width}x{size.height} exceeds the"
            f" {image.width}x{image.height} image {image.id}",
        )
    return size


def _loading_control_ids(batch: Batch) -> list[str]:
    return [p.id for p in batch.proteins if p.role is Role.LOADING_CONTROL]


def _implicit_users(batch: Batch, protein_id: str) -> list[str]:
    """Targets that normalize against ``protein_id`` without naming it: it is the
    batch's only loading control, and they chose none."""
    if _loading_control_ids(batch) != [protein_id]:
        return []
    return [p.id for p in batch.proteins if p.role is Role.TARGET and not p.loading_control_ids]


def _users(batch: Batch, protein_id: str) -> list[str]:
    """Every target normalized against ``protein_id``, named or implicit."""
    named = [p.id for p in batch.proteins if protein_id in p.loading_control_ids]
    return named + _implicit_users(batch, protein_id)


def _pin_single_loading_control(batch: Batch) -> list[str]:
    """Before a second loading control appears, write the single one into the
    targets that use it implicitly, so their normalization does not change (with
    two loading controls and none chosen, a target is not normalized at all).
    Returns the ids of the targets it pinned."""
    only = _loading_control_ids(batch)
    if len(only) != 1:
        return []
    pinned = []
    for protein in batch.proteins:
        if protein.role is Role.TARGET and not protein.loading_control_ids:
            protein.loading_control_ids = list(only)
            pinned.append(protein.id)
    return pinned


def _check_lane(
    protein: Protein, lane: int | None, n: int, taken: dict[int, str], *, proposed: bool
) -> int:
    """A lane the box may take: one of the ``n`` declared lanes, where the protein
    has no box yet. ``proposed`` words the refusal for a lane read from position."""
    if lane is None:  # position could not propose one after all
        raise OperationError(ErrorCode.LANE_REQUIRED, "choose the lane")
    where = f" (the box lies at lane {lane}); choose the lane" if proposed else ""
    if not 0 <= lane < n:
        raise OperationError(
            ErrorCode.LANE_OUT_OF_RANGE, f"lane {lane} is not one of the {n} lanes{where}"
        )
    if lane in taken:
        raise OperationError(
            ErrorCode.LANE_OCCUPIED,
            f"{protein.name!r} already has a box in lane {lane}{where}",
            ids=[taken[lane]],
        )
    return lane


def _overlapped(rect: Rect, protein: Protein, *, skip: str | None = None) -> list[str]:
    """The protein's bands (other than ``skip``) whose box overlaps ``rect``."""
    return [
        band.id
        for band in protein.bands
        if band.id != skip and overlaps(rect, band.box.rect(protein.box_size))
    ]


# --- Images ---


@_locked
def import_image(
    session: ProjectSession,
    source: BinaryIO,
    original_name: str,
    *,
    kind: ImageKind,
    polarity: Polarity,
    membrane_id: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Store an image stream in the project and record it; return its id.

    The file is copied byte for byte into ``images/``, then read back: the
    recorded size, bit depth, background and warnings come from the stored copy.
    Without ``membrane_id`` the image starts a new membrane. ``kind`` and
    ``polarity`` are required (the model has no silent default).
    """
    kind = _member(ImageKind, kind, "image kind")
    polarity = _member(Polarity, polarity, "polarity")
    if membrane_id is not None and all(
        m.id != membrane_id for m in session.project.batch.membranes
    ):
        raise UnknownIdError(f"unknown membrane {membrane_id!r}")
    if not isinstance(original_name, str):
        raise _invalid(f"original name must be text, not {original_name!r}")
    suffix = PurePosixPath(original_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise OperationError(
            ErrorCode.UNSUPPORTED_IMAGE_TYPE,
            f"{original_name!r} is not a supported image type ({', '.join(IMAGE_SUFFIXES)})",
        )

    # An orphan may hold the id this import gets (an import that was never saved).
    session._remove_orphans()
    image_id = f"img-{session.project.next_id}"  # the id the change below will take
    try:
        stored = storage.store_image(
            session.folder, image_id, original_name, source, max_bytes=max_bytes
        )
    except storage.ImageTooLargeError as exc:
        raise OperationError(ErrorCode.IMAGE_TOO_LARGE, str(exc)) from exc
    except FileExistsError as exc:
        raise OperationError(
            ErrorCode.LEFTOVER_FILE,
            f"a leftover file for {image_id} in images/ could not be deleted;"
            " close the program holding it and try again",
            ids=(image_id,),
        ) from exc
    except ValueError as exc:
        raise OperationError(ErrorCode.INVALID_IMAGE, str(exc)) from exc

    path = session.folder / storage.IMAGES_DIR / stored.file
    try:
        try:
            loaded = load_image(path)
        except (ValueError, OSError) as exc:
            raise OperationError(
                ErrorCode.UNREADABLE_IMAGE, f"cannot read {original_name!r}: {exc}"
            ) from exc
        background = estimate_background(loaded.array)

        def change(draft: Project) -> str:
            if draft.new_id("img") != image_id:  # the lock makes this impossible
                raise RuntimeError(f"image id {image_id} changed while the file was stored")
            if membrane_id is None:
                membrane = Membrane(id=draft.new_id("mem"))
                draft.batch.membranes.append(membrane)
            else:
                membrane = next(m for m in draft.batch.membranes if m.id == membrane_id)
            membrane.images.append(
                ImageRef(
                    id=image_id,
                    file=stored.file,
                    original_name=original_name,
                    kind=kind,
                    sha256=stored.sha256,
                    width=loaded.width,
                    height=loaded.height,
                    bit_depth=loaded.bit_depth,
                    polarity=polarity,
                    background=background,
                    import_warnings=loaded.warnings,
                )
            )
            return membrane.id

        new, membrane_used = _prepare(session, change)
        params = {
            "image_id": image_id,
            "membrane_id": membrane_used,
            "new_membrane": membrane_id is None,
            "original_name": original_name,
            "kind": kind.value,
            "polarity": polarity.value,
            "sha256": stored.sha256,  # keeps a removed image's provenance
        }
        session._commit(
            new, action="import_image", params=params, add_pixels={image_id: loaded.array}
        )
    except BaseException:
        # _commit can raise before committing (a bad clock or params) or after
        # (a hook bug): keep the file only if the committed project uses it.
        if all(image.id != image_id for image in session.project.batch.iter_images()):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    return image_id


@_locked
def remove_image(session: ProjectSession, image_id: str) -> Cascade:
    """Remove an image with the proteins and bands on it.

    Targets using a removed loading control, by name or as the batch's only
    one, are detached and reported, marker pairings to the image are cleared, and calibration
    points on it are dropped, which clears the membrane's fit and its bands'
    apparent MWs. A membrane left with no image is removed. The file is deleted
    after the next successful save.
    """
    session.project.batch.find_image(image_id)

    def change(draft: Project) -> Cascade:
        batch = draft.batch
        gone = [p for p in batch.proteins if p.image_id == image_id]
        gone_ids = {p.id for p in gone}
        removed = [image_id, *(p.id for p in gone), *(b.id for p in gone for b in p.bands)]
        # Targets using a removed loading control without naming it lose it too.
        implicit = {t for p in gone for t in _implicit_users(batch, p.id)} - gone_ids
        batch.proteins = [p for p in batch.proteins if p.id not in gone_ids]

        detached = []
        for protein in batch.proteins:
            kept = [i for i in protein.loading_control_ids if i not in gone_ids]
            if len(kept) != len(protein.loading_control_ids):
                protein.loading_control_ids = kept
                detached.append(protein.id)
            elif protein.id in implicit:
                detached.append(protein.id)

        unpaired = []
        for image in batch.iter_images():
            if image.marker_image_id == image_id:
                image.marker_image_id = None
                unpaired.append(image.id)

        membrane = batch.membrane_of(image_id)
        membrane.images = [image for image in membrane.images if image.id != image_id]
        unfitted = []
        calibration = membrane.calibration
        points = [point for point in calibration.points if point.image_id != image_id]
        if len(points) != len(calibration.points):
            calibration.points = points
            calibration.fit_quality = None  # the fit no longer matches its points (#58 refits)
            on_membrane = {image.id for image in membrane.images}
            for protein in batch.proteins:
                if protein.image_id in on_membrane:
                    for band in protein.bands:
                        band.apparent_mw = None
            if membrane.images:
                unfitted.append(membrane.id)
        if not membrane.images:
            batch.membranes = [m for m in batch.membranes if m.id != membrane.id]
            removed.append(membrane.id)
        return Cascade(
            removed=tuple(removed),
            detached_targets=tuple(detached),
            unpaired_images=tuple(unpaired),
            unfitted_membranes=tuple(unfitted),
        )

    return _apply(
        session,
        "remove_image",
        change,
        lambda cascade: {"image_id": image_id, **_cascade(cascade)},
        evict=(image_id,),
    )


@_locked
def set_polarity(session: ProjectSession, image_id: str, polarity: Polarity) -> None:
    """Set an image's polarity and recompute the nets of every band on it (the
    background, a median, does not depend on it).

    The not-detected records of every protein on the image are dropped and
    logged: their SNR was measured with the other signal direction, against the
    row's fitted background, so it cannot be recomputed here. A later detection
    run writes them again.
    """
    polarity = _member(Polarity, polarity, "polarity")
    batch = session.project.batch
    if batch.find_image(image_id).polarity is polarity:
        return
    has_bands = any(p.bands for p in batch.proteins if p.image_id == image_id)
    array = session.pixels(image_id) if has_bands else None

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        draft.batch.find_image(image_id).polarity = polarity
        if array is not None:
            _requantify_image(draft, image_id, array)
        return [
            dropped
            for protein in draft.batch.proteins
            if protein.image_id == image_id
            for dropped in _drop_undetected_where(protein, lambda _: True)
        ]

    _apply(
        session,
        "set_polarity",
        change,
        lambda dropped: {
            "image_id": image_id,
            "polarity": polarity.value,
            "dropped_undetected": dropped,
        },
    )


# --- Lanes and the reference ---


@_locked
def set_lanes(
    session: ProjectSession,
    lanes: Sequence[LaneInput],
    *,
    reference_condition: str | None | Keep = KEEP,
) -> LanesUpdate:
    """Declare or replace the whole lane table; row i is lane i.

    Text is cleaned (:func:`~proteia.core.names.clean_text`). Look-alike spellings
    (equal NFKC key, e.g. the micro sign and Greek mu) are unified: where the new
    table mixes them, the spelling already in the table wins; a look-alike typed
    for every lane of a condition renames it. Case differences are kept.
    Lanes holding a box cannot be dropped (``LANES_IN_USE``): boxes are the user's
    measurements. Not-detected records in dropped lanes are dropped instead, and
    reported: one detection run makes them again, and a lane cut from the table
    means nothing any more. Lane indices are never remapped. Each kept lane keeps
    its metadata and its records.

    ``reference_condition``: ``KEEP`` re-resolves the current reference against
    the new conditions and clears it if none matches (reported in the result); a
    string sets it in the same change (``UNKNOWN_CONDITION`` if no lane has it);
    None clears it.
    """
    if isinstance(lanes, str) or not isinstance(lanes, Sequence):
        raise _invalid("lanes must be a sequence of LaneInput")
    typed_conditions: list[str] = []
    typed_samples: list[str | None] = []
    included: list[bool] = []
    for i, lane in enumerate(lanes):
        if not isinstance(lane, LaneInput):
            raise _invalid(f"lane {i} must be a LaneInput, not {lane!r}")
        typed_conditions.append(_clean(lane.condition, f"lane {i} condition"))
        typed_samples.append(_clean_optional(lane.sample, f"lane {i} sample"))
        if not isinstance(lane.included, bool):
            raise _invalid(f"lane {i} included must be True or False, not {lane.included!r}")
        included.append(lane.included)

    batch = session.project.batch
    conditions = unify_spellings(typed_conditions, existing=[lane.label for lane in batch.lanes])
    samples = unify_spellings(typed_samples, existing=[lane.sample for lane in batch.lanes])
    n = len(conditions)
    respelled = tuple(
        i
        for i in range(n)
        if conditions[i] != typed_conditions[i] or samples[i] != typed_samples[i]
    )

    cut = [band.id for p in batch.proteins for band in p.bands if band.lane_index >= n]
    if cut:
        raise OperationError(
            ErrorCode.LANES_IN_USE,
            f"{len(cut)} box(es) are in lanes the new table drops; remove them first",
            ids=cut,
        )

    labels = [c for c in conditions if c is not None]
    cleared = False
    if reference_condition is KEEP:
        current = batch.reference_condition
        reference = None if current is None else resolve_label(current, labels)
        cleared = current is not None and reference is None
    elif reference_condition is None:
        reference = None
    else:
        reference = _condition(reference_condition, labels)

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        old = draft.batch.lanes
        draft.batch.lanes = [
            Lane(
                index=i,
                label=conditions[i],
                sample=samples[i],
                included=included[i],
                metadata=dict(old[i].metadata) if i < len(old) else {},
            )
            for i in range(n)
        ]
        draft.batch.reference_condition = reference
        return [
            dropped
            for protein in draft.batch.proteins
            for dropped in _drop_undetected_where(protein, lambda record: record.lane_index >= n)
        ]

    # Only the rows that are new or differ from the committed table, as stored.
    stored = [(lane.label, lane.sample, lane.included) for lane in batch.lanes]
    rows = list(zip(conditions, samples, included, strict=True))
    changed: list[JsonValue] = [
        {"index": i, "condition": condition, "sample": sample, "included": include}
        for i, (condition, sample, include) in enumerate(rows)
        if i >= len(stored) or rows[i] != stored[i]
    ]

    def params(dropped: list[dict[str, JsonValue]]) -> _Params:
        return {
            "lane_count": n,
            "changed": changed,
            "reference_condition": reference,
            "dropped_undetected": dropped,
        }

    dropped = _apply(session, "set_lanes", change, params)
    keys = tuple((d["protein_id"], d["lane_index"], d["band_index"]) for d in dropped)
    return LanesUpdate(respelled=respelled, reference_cleared=cleared, dropped_undetected=keys)


@_locked
def set_reference_condition(session: ProjectSession, condition: str | None) -> None:
    """Choose the fold-change reference by its condition (the stored spelling is
    the lane's), or clear it with None."""
    batch = session.project.batch
    reference = None if condition is None else _condition(condition, [x.label for x in batch.lanes])

    def change(draft: Project) -> None:
        draft.batch.reference_condition = reference

    _apply(session, "set_reference_condition", change, lambda _: {"reference_condition": reference})


# --- Proteins ---


@_locked
def add_protein(
    session: ProjectSession,
    name: str,
    role: Role,
    image_id: str,
    *,
    expected_mw: float | None = None,
    loading_control_ids: Sequence[str] = (),
    box_size: BoxSize | None = None,
) -> str:
    """Add a protein on a signal image, last in the protein order; return its id.

    The name is stored cleaned and must be unique ignoring case and look-alike
    characters. ``loading_control_ids`` (targets only) keeps its order, the series
    order. ``box_size`` defaults to :func:`~proteia.core.boxes.initial_box_size`;
    the first grown box replaces it. The image cannot be changed later. Adding a
    second loading control writes the first into the targets that used it
    without naming it, so their results do not change.
    """
    batch = session.project.batch
    role = _member(Role, role, "role")
    name = _protein_name(batch, name)
    image = batch.find_image(image_id)
    if image.kind is ImageKind.VISIBLE_MARKER:
        raise OperationError(
            ErrorCode.MARKER_IMAGE,
            f"{image_id} is a visible-light marker image, not a signal image",
            ids=(image_id,),
        )
    mw = _expected_mw(expected_mw)
    controls = _loading_controls(batch, None, role, loading_control_ids)
    if box_size is None:
        size = boxes.initial_box_size(image.width, image.height)
    else:
        size = _fitting_size(box_size, image)

    def change(draft: Project) -> tuple[str, list[str]]:
        pinned = _pin_single_loading_control(draft.batch) if role is Role.LOADING_CONTROL else []
        protein_id = draft.new_id("prot")
        draft.batch.proteins.append(
            Protein(
                id=protein_id,
                name=name,
                role=role,
                image_id=image_id,
                loading_control_ids=controls,
                expected_mw=mw,
                box_size=size,
            )
        )
        return protein_id, pinned

    def params(result: tuple[str, list[str]]) -> _Params:
        protein_id, pinned = result
        return {
            "protein_id": protein_id,
            "name": name,
            "role": role.value,
            "image_id": image_id,
            "expected_mw": mw,
            "loading_control_ids": list(controls),
            "box_size": _size(size),
            "pinned_targets": pinned,
        }

    protein_id, _ = _apply(session, "add_protein", change, params)
    return protein_id


@_locked
def edit_protein(
    session: ProjectSession,
    protein_id: str,
    *,
    name: str | Keep = KEEP,
    role: Role | Keep = KEEP,
    expected_mw: float | None | Keep = KEEP,
    loading_control_ids: Sequence[str] | Keep = KEEP,
) -> None:
    """Edit a protein's fields; ``KEEP`` leaves one unchanged. No net changes.

    A target turned into a loading control loses its own loading controls; if it
    becomes the second one, the first is written into the targets that used it
    without naming it. A loading control that targets use, by name or as the
    batch's only one, cannot become a target (``LOADING_CONTROL_IN_USE``, with
    those targets): choose other loading controls for them first.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    new_name = protein.name if name is KEEP else _protein_name(batch, name, protein_id=protein_id)
    new_role = protein.role if role is KEEP else _member(Role, role, "role")
    mw = protein.expected_mw if expected_mw is KEEP else _expected_mw(expected_mw)
    if protein.role is Role.LOADING_CONTROL and new_role is Role.TARGET:
        users = _users(batch, protein_id)
        if users:
            raise OperationError(
                ErrorCode.LOADING_CONTROL_IN_USE,
                f"{protein.name!r} is the loading control of {len(users)} target(s)",
                ids=users,
            )
    if loading_control_ids is KEEP:
        # A loading control has none; a target keeps its choice.
        controls = [] if new_role is Role.LOADING_CONTROL else list(protein.loading_control_ids)
    else:
        controls = _loading_controls(batch, protein_id, new_role, loading_control_ids)

    def change(draft: Project) -> list[str]:
        pinned = []
        if protein.role is Role.TARGET and new_role is Role.LOADING_CONTROL:
            pinned = _pin_single_loading_control(draft.batch)
        edited = draft.batch.find_protein(protein_id)
        edited.name = new_name
        edited.role = new_role
        edited.expected_mw = mw
        edited.loading_control_ids = controls
        return [target for target in pinned if target != protein_id]  # its own is cleared

    def params(pinned: list[str]) -> _Params:
        # Only the fields whose stored value changed, a cleared list included.
        edits: dict[str, JsonValue] = {"protein_id": protein_id, "pinned_targets": pinned}
        if new_name != protein.name:
            edits["name"] = new_name
        if new_role is not protein.role:
            edits["role"] = new_role.value
        if mw != protein.expected_mw:
            edits["expected_mw"] = mw
        if controls != protein.loading_control_ids:
            edits["loading_control_ids"] = list(controls)
        return edits

    _apply(session, "edit_protein", change, params)


@_locked
def remove_protein(session: ProjectSession, protein_id: str) -> Cascade:
    """Remove a protein and its bands; targets using it as their loading control,
    by name or as the batch's only one, are detached and reported."""
    session.project.batch.find_protein(protein_id)

    def change(draft: Project) -> Cascade:
        batch = draft.batch
        removed = (protein_id, *(band.id for band in batch.find_protein(protein_id).bands))
        implicit = set(_implicit_users(batch, protein_id))
        batch.proteins = [p for p in batch.proteins if p.id != protein_id]
        detached = []
        for protein in batch.proteins:
            if protein_id in protein.loading_control_ids:
                protein.loading_control_ids.remove(protein_id)
                detached.append(protein.id)
            elif protein.id in implicit:
                detached.append(protein.id)
        return Cascade(
            removed=removed,
            detached_targets=tuple(detached),
            unpaired_images=(),
            unfitted_membranes=(),
        )

    return _apply(
        session,
        "remove_protein",
        change,
        lambda cascade: {"protein_id": protein_id, **_cascade(cascade)},
    )


# --- Boxes ---


@_locked
def place_box(
    session: ProjectSession,
    protein_id: str,
    x: int,
    y: int,
    *,
    lane_index: int | None = None,
    grow: bool,
) -> str:
    """Place a box of a protein in a lane at the image point ``(x, y)``; return
    the band id.

    Without ``lane_index``, the lane is proposed from the position
    (:func:`~proteia.core.project.propose_lane`): the clicked x for a fixed box,
    the grown band's centre for a seed click. It needs boxes in at least two
    lanes on the image to know the lane pitch (``LANE_REQUIRED`` otherwise), and
    a proposed lane outside the table (``LANE_OUT_OF_RANGE``) or already holding
    a box of the protein (``LANE_OCCUPIED``) is refused, never moved elsewhere.
    Either way the lane is stored with the band, and moving or removing other
    boxes never changes it.

    ``grow=True`` is a seed click: the band is grown from the point and the
    protein's shared size fitted to it (the first box sets the size, later ones
    only grow it, and the other boxes are re-centred and re-quantified).
    ``grow=False`` drops a box of the current size centred on the point, shifted
    inside the image. A protein's own boxes never overlap.

    A not-detected record in the lane is replaced by the box, with no
    confirmation: the log entry names it (``replaced_undetected``, else null).
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    x, y = _int(x, "x"), _int(y, "y")
    lane_index = None if lane_index is None else _int(lane_index, "lane index")
    proposed = lane_index is None
    if not isinstance(grow, bool):
        raise _invalid(f"grow must be True or False, not {grow!r}")
    n = len(batch.lanes)
    if n == 0:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before placing boxes")
    taken = {b.lane_index: b.id for b in protein.bands if b.band_index == 0}
    image = batch.find_image(protein.image_id)
    width, height = image.width, image.height
    if not (0 <= x < width and 0 <= y < height):
        raise OperationError(
            ErrorCode.OUT_OF_IMAGE, f"({x}, {y}) is outside the {width}x{height} image"
        )
    anchors: list[tuple[float, int]] = []
    if lane_index is not None:
        _check_lane(protein, lane_index, n, taken, proposed=False)
    else:
        # Before any pixel work: can a lane be proposed at all? A fixed box's lane
        # is where the user clicked; a seed click's waits for the grown band.
        anchors = lane_anchors(batch, image)
        if propose_lane(x, anchors) is None:
            raise OperationError(
                ErrorCode.LANE_REQUIRED,
                "choose the lane: position proposes one only once boxes in two lanes of"
                " this image show the lane spacing",
            )
        if not grow:
            lane_index = _check_lane(protein, propose_lane(x, anchors), n, taken, proposed=True)

    array = session.pixels(image.id)
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    if grow:
        # The settings the export record reports (record.settings), passed explicitly.
        grown = grow_box(
            array,
            (x, y),
            image.background,
            rel_threshold=REL_THRESHOLD,
            noise_k=NOISE_K,
            dark_on_light=image.polarity.dark_on_light,
        )
        if grown is None:
            raise OperationError(ErrorCode.NO_BAND_FOUND, f"no band found at ({x}, {y})")
        try:
            size, resized, rect = boxes.grow_to_fit(
                rects, protein.box_size, grown, width=width, height=height
            )
        except boxes.BoxRuleError as exc:  # overlap, or size_would_overlap
            raise OperationError(ErrorCode(exc.code), str(exc)) from exc
        source = ProposalSource.CLICK
        if lane_index is None:
            # The grown band's own centre (the box is shifted inside the image at the
            # edges); where the user clicked if the image edge cuts the band.
            gx0, _, gx1, _ = grown
            at = x if gx0 <= 0 or gx1 >= width else (gx0 + gx1) / 2
            lane_index = _check_lane(protein, propose_lane(at, anchors), n, taken, proposed=True)
    else:
        size, resized = protein.box_size, rects
        rect = boxes.centered_rect(x, y, size, width, height)
        hits = _overlapped(rect, protein)
        if hits:
            raise OperationError(
                ErrorCode.OVERLAP, "the box would overlap another box of this protein", ids=hits
            )
        source = ProposalSource.MANUAL
    if lane_index is None:  # unreachable: every branch above checks or proposes it
        raise RuntimeError("place_box left the lane unresolved")

    def change(draft: Project) -> tuple[str, dict[str, JsonValue] | None]:
        edited = draft.batch.find_protein(protein_id)
        edited_image = draft.batch.find_image(edited.image_id)
        size_changed = edited.box_size != size
        edited.box_size = size
        for band, old, new in zip(edited.bands, rects, resized, strict=True):
            if new != old:
                _set_box(band, new)
            if size_changed or new != old:
                _quantify(band, edited, edited_image, array)
        band = Band(
            id=draft.new_id("band"),
            lane_index=lane_index,
            band_index=0,
            box=Box(x=rect[0], y=rect[1]),
            net=0.0,  # set by _quantify
            source=source,
        )
        _quantify(band, edited, edited_image, array)
        edited.bands.append(band)
        return band.id, _drop_undetected(edited, lane_index, 0)

    def params(result: tuple[str, dict[str, JsonValue] | None]) -> _Params:
        band_id, replaced = result
        return {
            "band_id": band_id,
            "protein_id": protein_id,
            "x": x,
            "y": y,
            "grow": grow,
            "lane_index": lane_index,  # the stored lane, chosen or proposed
            "lane_proposed": proposed,
            "rect": list(rect),
            "box_size": _size(size),  # after the change: a seed click may grow it
            "replaced_undetected": replaced,
        }

    band_id, _ = _apply(session, "place_box", change, params)
    return band_id


@_locked
def move_box(session: ProjectSession, band_id: str, rect: Rect) -> None:
    """Move a box to where the user dragged or resized it.

    ``rect`` is read by its centre: the box keeps the protein's size, centred
    there and shifted inside the image. Only that band's net is recomputed; its
    lane never changes. Marks the band as manually edited.
    """
    batch = session.project.batch
    protein, band = batch.find_band(band_id)
    if isinstance(rect, str) or not isinstance(rect, Sequence) or len(rect) != 4:
        raise _invalid(f"rect must be (x0, y0, x1, y1), not {rect!r}")
    x0, y0, x1, y1 = (_int(v, "rect coordinate") for v in rect)
    rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    image = batch.find_image(protein.image_id)
    new = boxes.center_snap(rect, protein.box_size, image.width, image.height)
    if new == band.box.rect(protein.box_size):
        return
    hits = _overlapped(new, protein, skip=band_id)
    if hits:
        raise OperationError(
            ErrorCode.OVERLAP, "the box would overlap another box of this protein", ids=hits
        )
    array = session.pixels(image.id)

    def change(draft: Project) -> None:
        edited, moved = draft.batch.find_band(band_id)
        _set_box(moved, new)
        moved.manually_edited = True
        _quantify(moved, edited, draft.batch.find_image(edited.image_id), array)

    _apply(session, "move_box", change, lambda _: {"band_id": band_id, "rect": list(new)})


@_locked
def remove_box(session: ProjectSession, band_id: str) -> None:
    """Remove one box. The protein's size and its other boxes are unchanged."""
    protein, band = session.project.batch.find_band(band_id)
    params = {"band_id": band_id, "protein_id": protein.id, "lane_index": band.lane_index}

    def change(draft: Project) -> None:
        protein, _ = draft.batch.find_band(band_id)
        protein.bands = [band for band in protein.bands if band.id != band_id]

    _apply(session, "remove_box", change, lambda _: params)


@_locked
def set_box_lane(session: ProjectSession, band_id: str, lane_index: int) -> None:
    """Store another lane as the box's lane; the box stays where it is.

    The lane must be one of the declared lanes (``LANE_OUT_OF_RANGE``) where the
    protein has no box for the same band yet (``LANE_OCCUPIED``). The box counts
    as edited by the user. The same lane is a no-op. A not-detected record for
    the same band in the new lane is replaced (``replaced_undetected``); the lane
    the box leaves gets no record.
    """
    batch = session.project.batch
    protein, band = batch.find_band(band_id)
    lane = _int(lane_index, "lane index")
    if lane == band.lane_index:
        return
    taken = {
        b.lane_index: b.id
        for b in protein.bands
        if b.band_index == band.band_index and b.id != band_id
    }
    _check_lane(protein, lane, len(batch.lanes), taken, proposed=False)
    params = {
        "band_id": band_id,
        "protein_id": protein.id,
        "from_lane": band.lane_index,
        "lane_index": lane,
    }

    def change(draft: Project) -> dict[str, JsonValue] | None:
        edited_protein, edited = draft.batch.find_band(band_id)
        edited.lane_index = lane
        edited.manually_edited = True
        return _drop_undetected(edited_protein, lane, edited.band_index)

    _apply(
        session,
        "set_box_lane",
        change,
        lambda replaced: {**params, "replaced_undetected": replaced},
    )


@_locked
def set_box_size(session: ProjectSession, protein_id: str, size: BoxSize) -> None:
    """Change a protein's shared box size: every box is re-sized around its centre
    (shifted inside the image) and re-quantified. A size that would make boxes
    overlap is refused."""
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    image = batch.find_image(protein.image_id)
    size = _fitting_size(size, image)
    if size == protein.box_size:
        return
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    resized = boxes.resize_all(rects, size, width=image.width, height=image.height)
    if resized is None:
        raise OperationError(
            ErrorCode.SIZE_WOULD_OVERLAP,
            f"box size {size.width}x{size.height} would make boxes of {protein.name!r} overlap",
        )
    array = session.pixels(image.id) if protein.bands else None

    def change(draft: Project) -> None:
        edited = draft.batch.find_protein(protein_id)
        edited_image = draft.batch.find_image(edited.image_id)
        edited.box_size = size
        for band, old, new in zip(edited.bands, rects, resized, strict=True):
            if new != old:
                _set_box(band, new)
            _quantify(band, edited, edited_image, array)

    _apply(
        session,
        "set_box_size",
        change,
        lambda _: {"protein_id": protein_id, "box_size": _size(size)},
    )


# --- Not-detected records ---


@_locked
def remove_undetected(
    session: ProjectSession, protein_id: str, lane_index: int, *, band_index: int = 0
) -> None:
    """Remove a protein's not-detected record in a lane: the lane becomes "not
    measured", as if the detector had never looked there.

    The lane must be one of the declared lanes (``LANE_OUT_OF_RANGE``). No record
    at that lane and band index is a no-op. The log entry holds the whole record.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    lane = _int(lane_index, "lane index")
    band = _int(band_index, "band index")
    n = len(batch.lanes)
    if not 0 <= lane < n:
        raise OperationError(
            ErrorCode.LANE_OUT_OF_RANGE, f"lane {lane} is not one of the {n} lanes"
        )
    if band < 0:
        raise _invalid(f"band index must be 0 or more, not {band}")
    if all((u.lane_index, u.band_index) != (lane, band) for u in protein.undetected):
        return

    def change(draft: Project) -> dict[str, JsonValue] | None:
        return _drop_undetected(draft.batch.find_protein(protein_id), lane, band)

    _apply(
        session,
        "remove_undetected",
        change,
        lambda removed: {
            "protein_id": protein_id,
            "lane_index": lane,
            "band_index": band,
            "removed": removed,
        },
    )


# --- Results, export, save ---


def compute(
    session: ProjectSession,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
) -> Results:
    """Every result of the committed project (:func:`~proteia.core.results.compute_results`).

    Reads the committed project once: no lock, no pixels, no autosave.
    ``error_type`` and ``method`` may be their raw values (``"SEM"``, ``"mean"``);
    an unknown value is refused (``INVALID_INPUT``).
    """
    error_type = _member(ErrorType, error_type, "error type")
    method = _member(ReduceMethod, method, "method")
    project = session.project
    return results.compute_results(
        project.batch, plot_conditions=plot_conditions, error_type=error_type, method=method
    )


@_locked
def export_lane_table(session: ProjectSession) -> Path:
    """Write the raw per-lane table to ``exports/lane-table.csv`` with its
    reproducibility record, ``exports/lane-table.record.json``
    (:func:`~proteia.core.record.build_record`); return the table's path.

    The same stored-index nets the results table shows, each followed by its
    clipping flags, in UTF-8 with a BOM. An image file that is missing or changed
    since import is refused (``IMAGE_FILE_CHANGED``, with those images): a record
    never vouches for pixels that are no longer on disk. Both files are built
    before either is written, and each is replaced atomically, the table first;
    ``OSError`` propagates (with Excel holding the table, both old files stay; a
    first export whose record fails removes its table).
    Not a state change: no log entry, no autosave.
    """
    project = session.project  # one snapshot for both files
    batch = project.batch
    if not batch.lanes:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before exporting them")
    bad = storage.verify_images(project, session.folder)
    if bad:
        raise OperationError(
            ErrorCode.IMAGE_FILE_CHANGED,
            f"image files changed or missing since import: {', '.join(bad)};"
            " restore them before exporting",
            ids=bad,
        )
    conditions, samples, included = spine_axes(batch.lanes)
    nets, clipped = results.lane_nets(batch), results.lane_clipped(batch)
    table = lane_table_bytes(
        conditions,
        samples,
        included,
        [(p.name, nets[p.id]) for p in batch.proteins],
        clipped={p.name: clipped[p.id] for p in batch.proteins},
    )
    doc = record.build_record(
        project, exported_at=session.timestamp(), files={LANE_TABLE_FILE: table}
    )
    data = record.record_bytes(doc)

    exports = session.folder / storage.EXPORTS_DIR
    exports.mkdir(parents=True, exist_ok=True)
    path, record_path = exports / LANE_TABLE_FILE, exports / LANE_TABLE_RECORD_FILE
    had_record = record_path.is_file()
    storage.write_atomic(path, table)
    try:
        storage.write_atomic(record_path, data)
    except OSError:
        # An old record names other table bytes (detectable); with none, remove
        # the table rather than leave numbers that no record describes.
        if not had_record:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    return path


@_locked
def save(session: ProjectSession) -> Path:
    """Save now (:meth:`ProjectSession.save`); raises on failure, unlike autosave."""
    return session.save()
