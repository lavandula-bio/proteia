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

Stored values that depend on pixels or geometry are recomputed by the operation
that invalidates them: :func:`_quantify` is the one place a band's net is
computed (it resets the pixel-derived ``clipped`` flag), and :func:`_set_box` the
one place a box moves (it clears the position-derived ``apparent_mw``). A size
change or a polarity change recomputes every affected net, so every stored net
always equals ``net_signal`` of the stored pixels, box, size, background and
polarity.

Functions return ids or small frozen dataclasses, never model objects. Typed text
follows :mod:`proteia.core.names`; box placement follows :mod:`proteia.core.boxes`.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Concatenate, Final

import numpy as np
from pydantic import ValidationError

from proteia.core import boxes, results, storage
from proteia.core.analyze import ReduceMethod
from proteia.core.export import LANE_COLUMNS, write_lane_table
from proteia.core.grow import grow_box
from proteia.core.imaging import load_image
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
from proteia.core.project import spine_axes
from proteia.core.quantify import estimate_background, net_signal
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
    "save",
    "set_box_size",
    "set_lanes",
    "set_polarity",
    "set_reference_condition",
]

# The lane table export: a fixed name chosen here, never from client text.
LANE_TABLE_FILE: Final = "lane-table.csv"
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


def _apply[T](
    session: ProjectSession, action: str, change: Callable[[Project], T], **commit_kw
) -> T:
    """:func:`_prepare`, then commit unless the project did not change."""
    new, result = _prepare(session, change)
    if new == session.project:  # a no-op: nothing committed, no hook
        return result
    session._commit(new, action=action, **commit_kw)
    return result


def _quantify(band: Band, protein: Protein, image: ImageRef, array: np.ndarray) -> None:
    """The one place a stored net is computed (#44 adds the clipping flag here)."""
    band.net = net_signal(
        array,
        band.box,
        protein.box_size,
        image.background,
        dark_on_light=image.polarity.dark_on_light,
    )
    band.clipped = None  # pixel-derived; stale once the net is recomputed


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


def _pin_single_loading_control(batch: Batch) -> None:
    """Before a second loading control appears, write the single one into the
    targets that use it implicitly, so their normalization does not change (with
    two loading controls and none chosen, a target is not normalized at all)."""
    only = _loading_control_ids(batch)
    if len(only) != 1:
        return
    for protein in batch.proteins:
        if protein.role is Role.TARGET and not protein.loading_control_ids:
            protein.loading_control_ids = list(only)


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

        def change(draft: Project) -> None:
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

        new, _ = _prepare(session, change)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    session._commit(new, action="import_image", add_pixels={image_id: loaded.array})
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

    return _apply(session, "remove_image", change, evict=(image_id,))


@_locked
def set_polarity(session: ProjectSession, image_id: str, polarity: Polarity) -> None:
    """Set an image's polarity and recompute the nets of every band on it (the
    background, a median, does not depend on it)."""
    polarity = _member(Polarity, polarity, "polarity")
    batch = session.project.batch
    if batch.find_image(image_id).polarity is polarity:
        return
    has_bands = any(p.bands for p in batch.proteins if p.image_id == image_id)
    array = session.pixels(image_id) if has_bands else None

    def change(draft: Project) -> None:
        draft.batch.find_image(image_id).polarity = polarity
        if array is not None:
            _requantify_image(draft, image_id, array)

    _apply(session, "set_polarity", change)


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
    Lanes holding a box cannot be dropped (``LANES_IN_USE``); band lane indices
    are never remapped. Each kept lane keeps its metadata.

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

    def change(draft: Project) -> None:
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

    _apply(session, "set_lanes", change)
    return LanesUpdate(respelled=respelled, reference_cleared=cleared)


@_locked
def set_reference_condition(session: ProjectSession, condition: str | None) -> None:
    """Choose the fold-change reference by its condition (the stored spelling is
    the lane's), or clear it with None."""
    batch = session.project.batch
    reference = None if condition is None else _condition(condition, [x.label for x in batch.lanes])

    def change(draft: Project) -> None:
        draft.batch.reference_condition = reference

    _apply(session, "set_reference_condition", change)


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

    def change(draft: Project) -> str:
        if role is Role.LOADING_CONTROL:
            _pin_single_loading_control(draft.batch)
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
        return protein_id

    return _apply(session, "add_protein", change)


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

    def change(draft: Project) -> None:
        if protein.role is Role.TARGET and new_role is Role.LOADING_CONTROL:
            _pin_single_loading_control(draft.batch)
        edited = draft.batch.find_protein(protein_id)
        edited.name = new_name
        edited.role = new_role
        edited.expected_mw = mw
        edited.loading_control_ids = controls

    _apply(session, "edit_protein", change)


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

    return _apply(session, "remove_protein", change)


# --- Boxes ---


@_locked
def place_box(
    session: ProjectSession, protein_id: str, x: int, y: int, *, lane_index: int, grow: bool
) -> str:
    """Place a box of a protein in a lane at the image point ``(x, y)``; return
    the band id.

    ``grow=True`` is a seed click: the band is grown from the point and the
    protein's shared size fitted to it (the first box sets the size, later ones
    only grow it, and the other boxes are re-centred and re-quantified).
    ``grow=False`` drops a box of the current size centred on the point, shifted
    inside the image. A protein's own boxes never overlap.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    x, y = _int(x, "x"), _int(y, "y")
    lane_index = _int(lane_index, "lane index")
    if not isinstance(grow, bool):
        raise _invalid(f"grow must be True or False, not {grow!r}")
    n = len(batch.lanes)
    if n == 0:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before placing boxes")
    if not 0 <= lane_index < n:
        raise OperationError(
            ErrorCode.LANE_OUT_OF_RANGE, f"lane {lane_index} is not one of the {n} lanes"
        )
    occupied = [b.id for b in protein.bands if b.lane_index == lane_index and b.band_index == 0]
    if occupied:
        raise OperationError(
            ErrorCode.LANE_OCCUPIED,
            f"{protein.name!r} already has a box in lane {lane_index}",
            ids=occupied,
        )
    image = batch.find_image(protein.image_id)
    width, height = image.width, image.height
    if not (0 <= x < width and 0 <= y < height):
        raise OperationError(
            ErrorCode.OUT_OF_IMAGE, f"({x}, {y}) is outside the {width}x{height} image"
        )

    array = session.pixels(image.id)
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    if grow:
        grown = grow_box(
            array, (x, y), image.background, dark_on_light=image.polarity.dark_on_light
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
    else:
        size, resized = protein.box_size, rects
        rect = boxes.centered_rect(x, y, size, width, height)
        hits = _overlapped(rect, protein)
        if hits:
            raise OperationError(
                ErrorCode.OVERLAP, "the box would overlap another box of this protein", ids=hits
            )
        source = ProposalSource.MANUAL

    def change(draft: Project) -> str:
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
        return band.id

    return _apply(session, "place_box", change)


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

    _apply(session, "move_box", change)


@_locked
def remove_box(session: ProjectSession, band_id: str) -> None:
    """Remove one box. The protein's size and its other boxes are unchanged."""
    session.project.batch.find_band(band_id)

    def change(draft: Project) -> None:
        protein, _ = draft.batch.find_band(band_id)
        protein.bands = [band for band in protein.bands if band.id != band_id]

    _apply(session, "remove_box", change)


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

    _apply(session, "set_box_size", change)


# --- Results, export, save ---


def compute(
    session: ProjectSession,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType = ErrorType.SD,
    method: ReduceMethod = ReduceMethod.MEAN,
) -> Results:
    """Every result of the committed project (:func:`~proteia.core.results.compute_results`).

    Reads the committed project once: no lock, no pixels, no autosave.
    """
    project = session.project
    return results.compute_results(
        project.batch, plot_conditions=plot_conditions, error_type=error_type, method=method
    )


@_locked
def export_lane_table(session: ProjectSession) -> Path:
    """Write the raw per-lane table to ``exports/lane-table.csv`` and return its path.

    The same stored-index nets the results table shows, in UTF-8 with a BOM.
    ``OSError`` propagates (with Excel holding the file, nothing is truncated).
    Not a state change: no autosave.
    """
    batch = session.project.batch
    if not batch.lanes:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before exporting them")
    exports = session.folder / storage.EXPORTS_DIR
    exports.mkdir(parents=True, exist_ok=True)
    path = exports / LANE_TABLE_FILE
    conditions, samples, included = spine_axes(batch.lanes)
    nets = results.lane_nets(batch)
    write_lane_table(
        path, conditions, samples, included, [(p.name, nets[p.id]) for p in batch.proteins]
    )
    return path


@_locked
def save(session: ProjectSession) -> Path:
    """Save now (:meth:`ProjectSession.save`); raises on failure, unlike autosave."""
    return session.save()
