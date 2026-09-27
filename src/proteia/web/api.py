# SPDX-License-Identifier: Apache-2.0
"""The HTTP routes: thin adapters over the project operations (ADR 0002).

Every edit goes through :mod:`proteia.core.operations` and answers with the
whole project state (:func:`~proteia.web.state.project_state`) and its results
(:func:`~proteia.web.results_view.results_payload`), both from one snapshot
(:func:`~proteia.core.operations.compute_view`), so the browser redraws the
image, the table and the charts from what the server stored. Creating, opening
and reading the project answer the same way. Each chart in the results is
answered as a URL named by its content (:mod:`proteia.web.charts`), and
``GET /api/charts/{key}.svg`` serves it as SVG, drawn when first fetched. One
project is open at a time. A removal of a protein or an image also answers what
it took with it: every field of :class:`~proteia.core.operations.Cascade`, as
lists of ids. Undo and redo answer the change they took back or made again
(``action``, ``seq``) and the ids and not-detected record keys that went or came
back (:class:`~proteia.core.operations.Restored`); clearing a protein's boxes
answers the band ids removed and the (lane index, band index) of each record
dropped; a row box answers what it did in each lane: every field of
:class:`~proteia.core.operations.RowPlacement`, each empty lane as
``{lane_index, reason, snr, expected_x}``, each other protein's band it
re-measured as ``{band_id, net_before, net_after}`` and the largest change as
``{band_id, change}`` (or null); requantifying answers the images
re-quantified. Lane indices in requests and answers are 0-based, as stored. Any
box edit may change every net on its image (each band's background ring leaves
out every box there), and every answer carries every protein's numbers, so the
browser redraws them all.

``GET /api/images/{image_id}/preview`` serves an image as the view draws it: its
gray analysis array, which the nets are measured on, or, with
``?colour=original``, its stored file in its own colours, for display only.

``POST /api/projects/sample`` creates the sample project, set up on the
synthetic sample blot up to the row boxes (:mod:`proteia.web.sample_project`),
and answers as a create does, with ``sample``: the truth table's name in the
project folder and each protein's row, as a drag over it.

``POST /api/export`` writes the results into a new export folder
(:func:`~proteia.core.operations.export_bundle`), computed with the settings the
results are shown with, and answers the folder, relative to the project folder
(``exports/<name>``), and the names of the files in it. ``POST
/api/project/reveal`` shows the project folder in the system file manager, or,
with ``{"folder": "exports/<name>"}`` as the export answered it, that export
folder.

Errors answer JSON ``{"code", "message", "ids"}``: an operation's refusal is 422
with its :class:`~proteia.core.session.ErrorCode` value, and ``detail`` when the
refusal carries one (a row box's: what the detector saw); an unknown id 404, and
an export folder to reveal that does not exist 404 ``folder_not_found``;
``no_project`` 409 before a project is open; ``invalid_input`` 422 for a request
the routes cannot read.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
)

from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.analyze import ReduceMethod
from proteia.core.export import DEFAULT_CHART_FORMATS
from proteia.core.model import BoxSize, UnknownIdError
from proteia.core.plotspec import ErrorType, PlotSpec
from proteia.core.results import Results
from proteia.core.session import Clock, ErrorCode, OperationError, ProjectSession, utc_now
from proteia.core.storage import ProjectError
from proteia.web import projects, sample_project
from proteia.web.charts import ChartStore
from proteia.web.results_view import results_payload
from proteia.web.state import has_colour, original_png, preview_png, project_state, revision

MAX_UPLOAD_BYTES: Final = 512 * 1024 * 1024
_WRITE_BYTES: Final = 1024 * 1024  # an upload is written to disk in pieces this large
_PREVIEWS_KEPT: Final = 8
# How long a reopen waits for an operation running on the open session before
# answering it without reading its project.json again.
REOPEN_WAIT_S: Final = 5.0


class NoProjectError(RuntimeError):
    """No project is open."""


class UnsavedChangesError(RuntimeError):
    """The open project has changes that could not be saved; it stays open."""


class UploadTooLargeError(ValueError):
    """An upload longer than :data:`MAX_UPLOAD_BYTES`."""


class FolderNotFoundError(LookupError):
    """No export folder of the open project has this name."""


@dataclass(frozen=True)
class ResultSettings:
    """How the results are computed: the keyword arguments of
    :func:`~proteia.core.operations.compute_view`. The defaults until the web UI
    can choose them."""

    plot_conditions: tuple[str, ...] | None = None
    error_type: ErrorType = ErrorType.SD
    method: ReduceMethod = ReduceMethod.MEAN


_ResultsKey = tuple[int, int, ResultSettings]  # open id, revision, settings


def _same_folder(a: Path, b: Path, *, unknown: bool) -> bool:
    """Whether ``a`` and ``b`` are one folder, however the paths are spelled
    (case, Unicode normalization, links); ``unknown`` when that cannot be told
    (one of them is gone or cannot be read)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return unknown


class Workspace:
    """The server's state: the projects root and the one open project.

    A request keeps the session it started with: a project switch while it runs
    does not redirect it (one user, one tab, so this only matters in a race).
    Every create, and every open of another project, gives the new session the
    next open id, so answers about different openings never compare equal, even
    at the same revision. Opening the project already open answers its session
    (:meth:`open`), so no two sessions write one folder (#93); that session
    takes the next open id too if it reads a ``project.json`` changed outside
    Proteia again.
    The results of the open project's latest revision computed so far are kept
    (with the open id, the revision and the settings they belong to), since
    reading them again is common and computing them is not cheap. So are the
    charts of the answers about the latest opening (:class:`ChartStore`).
    """

    def __init__(
        self, root: Path, *, reveal: Callable[[Path], None], clock: Clock = utc_now
    ) -> None:
        self.root = root
        self.reveal = reveal
        self.clock = clock
        # Guards the open session, the open ids, the settings, the previews and the
        # results; never held while computing. Taken before the chart store's own
        # lock, never while holding it.
        self._lock = threading.Lock()
        self._switching = threading.Lock()  # one switch at a time; never held by readers
        self._session: ProjectSession | None = None
        self._open_id = 0  # the open id of the latest create or open
        # Each session's open id, for as long as a request still holds that session.
        self._open_ids: weakref.WeakKeyDictionary[ProjectSession, int] = weakref.WeakKeyDictionary()
        self._settings = ResultSettings()
        self._results: tuple[_ResultsKey, Results] | None = None
        # (image id, SHA-256, original colours) -> preview PNG, least recently shown first.
        self._previews: OrderedDict[tuple[str, str, bool], bytes] = OrderedDict()
        self._charts = ChartStore()

    def current(self) -> ProjectSession:
        with self._lock:
            if self._session is None:
                raise NoProjectError("create or open a project first")
            return self._session

    @property
    def settings(self) -> ResultSettings:
        """How the results are computed: for what the browser shows, and for exports."""
        with self._lock:
            return self._settings

    @property
    def open_name(self) -> str | None:
        with self._lock:
            return None if self._session is None else self._session.folder.name

    def _peek(self) -> ProjectSession | None:
        with self._lock:
            return self._session

    @staticmethod
    def _save(session: ProjectSession) -> None:
        try:
            session.save()
        except (OSError, ProjectError) as exc:
            raise UnsavedChangesError(f"the open project could not be saved: {exc}") from exc

    def flush(self) -> None:
        """Save the open project if it has unsaved changes (an autosave failed);
        :class:`UnsavedChangesError` if that fails again."""
        session = self._peek()
        if session is not None and session.dirty:
            self._save(session)

    def _switch(self, make: Callable[[], ProjectSession]) -> ProjectSession:
        """Replace the open project with ``make()``; the old one is saved first,
        and stays open, with its undo history, if ``make`` fails or its unsaved
        changes cannot be saved. Once replaced, the old one is closed
        (:meth:`~proteia.core.session.ProjectSession.close`): its history is
        gone, and so are the image files only that history kept, unless the new
        session's folder is, or may be, the old one's. Then the close deletes
        nothing, since the new session may already be storing files the old one
        does not know (the close waits for any request still running on the old
        one), and the new session's first save or import deletes those files.
        Opening the open project never switches (:meth:`open`), but a project
        created where the open one's folder was removed outside Proteia does.
        Saving, opening and closing run outside the lock readers take. Called
        with the switch lock held."""
        old = self._peek()
        if old is not None and old.dirty:
            self._save(old)
        session = make()
        with self._lock:
            self._session = session
            self._open_id += 1
            self._open_ids[session] = self._open_id
            self._previews.clear()
            self._results = None
            self._charts.reset(self._open_id)
        if old is not None:
            old.close(remove_files=not _same_folder(old.folder, session.folder, unknown=True))
        return session

    def close(self) -> None:
        """Close the open project: its undo history is gone, and so are the image
        files only that history kept. Called when the server stops, after
        :meth:`flush`."""
        with self._switching:
            session = self._peek()
            if session is not None:
                session.close()

    def create(self, name: object) -> ProjectSession:
        with self._switching:
            return self._switch(lambda: projects.create_project(self.root, name, clock=self.clock))

    def create_sample(self) -> ProjectSession:
        """Create the sample project
        (:func:`~proteia.web.sample_project.create_sample_project`) and open it,
        as :meth:`create` does."""
        with self._switching:
            return self._switch(
                lambda: sample_project.create_sample_project(self.root, clock=self.clock)
            )

    def open(self, name: object) -> ProjectSession:
        """Open the project ``name`` (:func:`~proteia.web.projects.project_folder`)
        in place of the open one; or, if its folder is the open project's,
        however the name is spelled, answer the open session. A second session
        on that folder would let an edit still running on the first save over
        the second's ``project.json``, and the cleanup after that save delete
        image files only the second knows (#93).

        The open session reads its ``project.json`` again if that was changed
        outside Proteia (:meth:`~proteia.core.session.ProjectSession.reload`), as
        opening a project reads it: then it takes the next open id (its revision
        may go back) and has no undo history or kept results. Otherwise it is
        answered as it is, with its open id: the same opening, whose revisions
        still order its answers (this one is of its latest revision, so a client
        that waited for its edits' answers finds it no older than what it
        shows); and with its undo history and kept results. Its unsaved changes
        (a failed autosave) are neither saved first, as a switch saves them, nor
        replaced by the older file: the next change or quitting saves them. An
        operation running on the session (an edit, or a read such as a preview)
        is waited for, up to :data:`REOPEN_WAIT_S`, before the file is checked:
        after an edit, which saved over it, there is nothing to read; after a
        read, a changed file is read. One that runs longer leaves the file
        unread, and the session is answered as it is; opening it again reads
        it. When it cannot be told whether the two folders are one (the open
        one is gone), the open is a switch, whose close deletes nothing."""
        with self._switching:
            folder = projects.project_folder(self.root, name)
            session = self._peek()
            if session is None or not _same_folder(session.folder, folder, unknown=False):
                return self._switch(lambda: ops.open_project(folder, clock=self.clock))
            if session.lock.acquire(timeout=REOPEN_WAIT_S):
                try:
                    # The new open id under the session's lock: no commit falls
                    # between the reload and it.
                    if session.reload():
                        with self._lock:
                            self._open_id += 1
                            self._open_ids[session] = self._open_id
                            self._previews.clear()
                            self._results = None
                            self._charts.reset(self._open_id)
                finally:
                    session.lock.release()
            return session

    def view(self, session: ProjectSession) -> tuple[int, ops.ComputedView]:
        """``session``'s open id, and its committed project with the results of
        it (:func:`~proteia.core.operations.compute_view`). The kept results are
        reused while the open id, the revision and the settings match; the
        computation itself runs outside the lock, and its results are kept only
        while ``session`` is still the open one and no later revision's results
        are kept."""
        with self._lock:
            open_id, settings = self._open_ids[session], self._settings
            kept = self._results
        project = session.project
        if kept is not None and kept[0] == (open_id, revision(project), settings):
            return open_id, ops.ComputedView(project, kept[1])
        view = ops.compute_view(session, **dataclasses.asdict(settings))
        key = (open_id, revision(view.project), settings)
        with self._lock:
            if open_id == self._open_id and not self._keeps_later(key):
                self._results = (key, view.results)
        return open_id, view

    def _keeps_later(self, key: _ResultsKey) -> bool:
        """Whether the kept results are of a later revision than ``key``, with the
        same open id and settings: results that finished late never replace them.
        Called with the lock held."""
        if self._results is None:
            return False
        open_id, kept_revision, settings = self._results[0]
        return (open_id, settings) == (key[0], key[2]) and kept_revision > key[1]

    def register_chart(self, spec: PlotSpec, *, open_id: int) -> str:
        """The URL of the chart drawn from ``spec``, in an answer about the
        opening ``open_id``; kept to be served only while that is the open one."""
        return self._charts.register(spec, open_id=open_id)

    def chart(self, key: str) -> bytes:
        """The chart ``key`` names, as SVG; :class:`UnknownIdError` for a key not
        given in the open project's answers, or no longer kept."""
        return self._charts.svg(key)

    def preview(self, session: ProjectSession, image_id: str, *, original: bool = False) -> bytes:
        """The image's preview PNG: of its gray analysis array, or, with
        ``original``, of its stored file in its own colours
        (:func:`~proteia.web.state.original_png`) if the file has colour to
        show (:func:`~proteia.web.state.has_colour`); otherwise it answers the
        gray one: a gray file's original colours are its gray levels, and the
        colours of a CMYK file, say, are not converted. The last few previews
        shown are kept, gray and colour alike."""
        image = session.project.batch.find_image(image_id)
        original = original and has_colour(session, image)
        key = (image_id, image.sha256, original)
        with self._lock:
            if key in self._previews:
                self._previews.move_to_end(key)
                return self._previews[key]
        if original:
            data = original_png(session.colour_pixels(image_id))
        else:
            data = preview_png(session.pixels(image_id, keep=False))
        with self._lock:
            self._previews[key] = data
            while len(self._previews) > _PREVIEWS_KEPT:
                self._previews.popitem(last=False)
        return data


# --- Request bodies ---


PositiveInt = Annotated[StrictInt, Field(gt=0)]


def _url_index(value: object) -> object:
    """An index in a URL as an int, only if it is written as ``str`` writes one:
    lax int parsing would also take "+1", " 1", "1.0" and "01", and read "1_2"
    as 12. Its range is the operation's to check."""
    if not isinstance(value, str):
        return value
    try:
        number = int(value)
    except ValueError:
        number = None
    if number is None or str(number) != value:
        raise ValueError(f"must be a whole number in plain digits, such as 0 or 3, not {value!r}")
    return number


UrlIndex = Annotated[int, BeforeValidator(_url_index)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NameBody(_Body):
    name: str


class PolarityBody(_Body):
    polarity: str


class LaneBody(_Body):
    condition: str
    sample: str | None = None
    included: StrictBool = True


class LanesBody(_Body):
    lanes: list[LaneBody]
    reference_condition: str | None = None  # absent: keep the current reference


class ReferenceBody(_Body):
    condition: str | None  # required: null clears the reference


# An expected MW in kDa: a JSON number (a whole one too), never true or "42".
ExpectedMw = StrictFloat | None


class ProteinBody(_Body):
    name: str
    role: str
    image_id: str
    expected_mw: ExpectedMw = None
    loading_control_ids: list[str] = []
    box_size: tuple[PositiveInt, PositiveInt] | None = None  # width, height


class ProteinEditBody(_Body):
    """The fields a protein edit changes. A field left out keeps its value: only
    the fields the request set are passed on, so these defaults are never used,
    and null is a value only for ``expected_mw`` (no expected MW)."""

    name: str = ""
    role: str = ""
    expected_mw: ExpectedMw = None
    loading_control_ids: list[str] = []


class BoxSizeBody(_Body):
    width: PositiveInt
    height: PositiveInt


class PlaceBody(_Body):
    protein_id: str
    x: StrictInt
    y: StrictInt
    lane_index: StrictInt | None = None  # None: proposed from the position
    grow: StrictBool = False


class MoveBody(_Body):
    rect: tuple[StrictInt, StrictInt, StrictInt, StrictInt]


# A row box corner in image pixels, held to 32 bits: the row is clipped to the
# image, but logged as given, and the project file's reader refuses a number of
# more than 4300 characters, which JSON requests may carry.
RowCoordinate = Annotated[StrictInt, Field(ge=-(2**31), lt=2**31)]


class RowBody(_Body):
    protein_id: str
    # x0, y0, x1, y1, end-exclusive
    rect: tuple[RowCoordinate, RowCoordinate, RowCoordinate, RowCoordinate]


class LaneIndexBody(_Body):
    lane_index: StrictInt


class ExportBody(_Body):
    # Chart formats ("svg", "png", "pdf"); left out or null: the default formats.
    formats: list[str] | None = None


class RevealBody(_Body):
    folder: str  # an export folder as POST /api/export answered it: "exports/<name>"


# --- Routes ---


def _workspace(request: Request) -> Workspace:
    return request.app.state.workspace


WorkspaceDep = Annotated[Workspace, Depends(_workspace)]


def _answer(workspace: Workspace, session: ProjectSession, **extra: Any) -> dict[str, Any]:
    """A route's answer: ``extra``, then the project state and its results, both
    from one snapshot of ``session``. Every answer registers its charts, kept
    results too, so a read of the project brings back a chart the store forgot."""
    open_id, view = workspace.view(session)

    def register(spec: PlotSpec) -> str:
        return workspace.register_chart(spec, open_id=open_id)

    return {
        **extra,
        "project": project_state(session.folder.name, session, view.project, open_id=open_id),
        "results": results_payload(
            view.results, open_id=open_id, revision=revision(view.project), charts=register
        ),
    }


def _cascade(cascade: ops.Cascade) -> dict[str, list[str]]:
    """Every field of what a removal took with it, as lists of ids."""
    return {field.name: list(getattr(cascade, field.name)) for field in dataclasses.fields(cascade)}


def _row_placement(placement: ops.RowPlacement) -> dict[str, Any]:
    """Every field of what a row box did
    (:class:`~proteia.core.operations.RowPlacement`), with lists for tuples, the
    box size as ``{width, height}``, each empty lane as ``{lane_index,
    reason, snr, expected_x}``, each band re-measured as ``{band_id, net_before,
    net_after}`` and the largest change as ``{band_id, change}`` (or None)."""
    size, largest = placement.box_size, placement.largest_change
    return {
        "band_ids": list(placement.band_ids),
        "box_size": {"width": size.width, "height": size.height},
        "kept_lanes": list(placement.kept_lanes),
        "replaced_band_ids": list(placement.replaced_band_ids),
        "removed_band_ids": list(placement.removed_band_ids),
        "undetected_lanes": list(placement.undetected_lanes),
        "unmeasured_lanes": list(placement.unmeasured_lanes),
        "empty": [
            {"lane_index": lane, "reason": reason, "snr": snr, "expected_x": expected_x}
            for lane, reason, snr, expected_x in placement.empty
        ],
        "flags": list(placement.flags),
        "notes": list(placement.notes),
        "right_to_left": placement.right_to_left,
        "remeasured": [
            {"band_id": band_id, "net_before": before, "net_after": after}
            for band_id, before, after in placement.remeasured
        ],
        "largest_change": None
        if largest is None
        else {"band_id": largest[0], "change": largest[1]},
    }


router = APIRouter(prefix="/api")


@router.get("/projects")
def list_projects(workspace: WorkspaceDep) -> dict[str, Any]:
    return {
        "root": str(workspace.root),
        "open": workspace.open_name,
        "projects": [
            {"name": entry.name, "modified": entry.modified}
            for entry in projects.list_projects(workspace.root)
        ],
    }


@router.post("/projects", status_code=201)
def create_project(body: NameBody, workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, workspace.create(body.name))


@router.post("/projects/sample", status_code=201)
def create_sample_project(workspace: WorkspaceDep) -> dict[str, Any]:
    """Create the sample project (:mod:`proteia.web.sample_project`), named
    ``Sample blot`` or the next free ``Sample blot (n)``, and open it. Answers as
    a create does, with ``sample`` (:func:`~proteia.web.sample_project.sample_payload`):
    the truth table's name in the project folder and each protein's row."""
    session = workspace.create_sample()
    return _answer(workspace, session, sample=sample_project.sample_payload(session.project))


@router.post("/projects/open")
def open_project(body: NameBody, workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, workspace.open(body.name))


@router.get("/project")
def get_project(workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, workspace.current())


def _export_folder(project: Path, folder: str) -> Path:
    """The export folder ``folder`` names in the project folder ``project``:
    ``exports/<name>``, with one plain name that no file system reads as more
    than a name (no separator, drive or control character, and no dot or space
    at either end, which Windows drops, so ``..`` and ``...`` are refused too).
    ``invalid_input`` for any other text; :class:`FolderNotFoundError` when no
    such folder exists."""
    parent, _, name = folder.partition("/")
    if (
        parent != storage.EXPORTS_DIR
        or not name
        or name != name.strip(" .")
        or any(c in "/\\:" or not c.isprintable() for c in name)
    ):
        raise OperationError(
            ErrorCode.INVALID_INPUT,
            f"{folder!r} is not an export folder: give one as the export answered it,"
            f" {storage.EXPORTS_DIR}/<name>",
        )
    path = project / storage.EXPORTS_DIR / name
    if not path.is_dir():
        raise FolderNotFoundError(f"the project has no export folder {folder!r}")
    return path


@router.post("/project/reveal", status_code=204)
def reveal_project(workspace: WorkspaceDep, body: RevealBody | None = None) -> Response:
    """Show the project folder in the system file manager or, with ``folder``,
    one of its export folders, as ``POST /api/export`` answered it."""
    session = workspace.current()
    folder = session.folder if body is None else _export_folder(session.folder, body.folder)
    workspace.reveal(folder)
    return Response(status_code=204)


@router.post("/images", status_code=201)
async def import_image(
    request: Request,
    workspace: WorkspaceDep,
    name: Annotated[str, Query(min_length=1)],
    kind: str,
    polarity: str,
    membrane_id: str | None = None,
) -> dict[str, Any]:
    """The request body is the file's bytes; ``name`` is its original name
    (percent-encoded in the URL), kept only as metadata."""
    session = await run_in_threadpool(workspace.current)  # never block the event loop
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(f"an image may have at most {MAX_UPLOAD_BYTES} bytes")
    # Spooled to a temporary file on the project's drive (not the system drive),
    # never held in memory whole; import_image then copies it into images/.
    with tempfile.TemporaryFile(dir=session.folder) as spool:
        size = 0
        pending = bytearray()
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise UploadTooLargeError(f"an image may have at most {MAX_UPLOAD_BYTES} bytes")
            pending += chunk
            if len(pending) >= _WRITE_BYTES:
                await run_in_threadpool(spool.write, bytes(pending))
                pending.clear()
        if pending:
            await run_in_threadpool(spool.write, bytes(pending))
        spool.seek(0)
        image_id = await run_in_threadpool(
            lambda: ops.import_image(
                session,
                spool,
                name,
                kind=kind,
                polarity=polarity,
                membrane_id=membrane_id,
                max_bytes=MAX_UPLOAD_BYTES,
            )
        )
    # Computing the results takes a while: off the event loop, like the import.
    return await run_in_threadpool(lambda: _answer(workspace, session, image_id=image_id))


@router.delete("/images/{image_id}")
def remove_image(image_id: str, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    cascade = ops.remove_image(session, image_id)
    return _answer(workspace, session, **_cascade(cascade))


@router.put("/images/{image_id}/polarity")
def set_polarity(image_id: str, body: PolarityBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.set_polarity(session, image_id, body.polarity)
    return _answer(workspace, session)


@router.get("/images/{image_id}/preview")
def image_preview(
    image_id: str, workspace: WorkspaceDep, colour: Literal["original"] | None = None
) -> Response:
    """The image as the view draws it, a PNG of the image's own size: its gray
    analysis array, which the nets are measured on, or with ``colour=original``
    its stored file in the file's own colours, for display only (the gray one
    for a file without colour to show: ``colour`` in the project state says
    which have it). Either is read after the stored file's SHA-256 is checked:
    ``image_file_changed`` or ``unreadable_image`` (422) otherwise."""
    session = workspace.current()
    data = workspace.preview(session, image_id, original=colour == "original")
    return Response(data, media_type="image/png")


@router.get("/charts/{key}.svg")
def chart(key: str, workspace: WorkspaceDep) -> Response:
    """A chart of an answer, at its ``chart_url``; 404 ``unknown_id`` for a key
    not given in the open project's answers, or no longer kept."""
    return Response(workspace.chart(key), media_type="image/svg+xml")


@router.put("/lanes")
def set_lanes(body: LanesBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    lanes = [ops.LaneInput(lane.condition, lane.sample, lane.included) for lane in body.lanes]
    if "reference_condition" in body.model_fields_set:
        update = ops.set_lanes(session, lanes, reference_condition=body.reference_condition)
    else:
        update = ops.set_lanes(session, lanes)
    return _answer(
        workspace,
        session,
        respelled=list(update.respelled),
        reference_cleared=update.reference_cleared,
    )


@router.put("/reference")
def set_reference_condition(body: ReferenceBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.set_reference_condition(session, body.condition)
    return _answer(workspace, session)


@router.post("/proteins", status_code=201)
def add_protein(body: ProteinBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    size = (
        None if body.box_size is None else BoxSize(width=body.box_size[0], height=body.box_size[1])
    )
    protein_id = ops.add_protein(
        session,
        body.name,
        body.role,
        body.image_id,
        expected_mw=body.expected_mw,
        loading_control_ids=body.loading_control_ids,
        box_size=size,
    )
    return _answer(workspace, session, protein_id=protein_id)


@router.patch("/proteins/{protein_id}")
def edit_protein(protein_id: str, body: ProteinEditBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    # Only the fields the request set: the operation keeps the others (KEEP).
    ops.edit_protein(session, protein_id, **body.model_dump(exclude_unset=True))
    return _answer(workspace, session)


@router.delete("/proteins/{protein_id}")
def remove_protein(protein_id: str, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    cascade = ops.remove_protein(session, protein_id)
    return _answer(workspace, session, **_cascade(cascade))


@router.put("/proteins/{protein_id}/box-size")
def set_box_size(protein_id: str, body: BoxSizeBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.set_box_size(session, protein_id, BoxSize(width=body.width, height=body.height))
    return _answer(workspace, session)


@router.delete("/proteins/{protein_id}/undetected/{lane_index}")
def remove_undetected(
    protein_id: str, lane_index: UrlIndex, workspace: WorkspaceDep, band_index: UrlIndex = 0
) -> dict[str, Any]:
    """Remove the protein's not-detected record in the lane, for its first band
    unless ``band_index`` names a later one; no record there is a no-op."""
    session = workspace.current()
    ops.remove_undetected(session, protein_id, lane_index, band_index=band_index)
    return _answer(workspace, session)


@router.post("/boxes", status_code=201)
def place_box(body: PlaceBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    band_id = ops.place_box(
        session, body.protein_id, body.x, body.y, lane_index=body.lane_index, grow=body.grow
    )
    return _answer(workspace, session, band_id=band_id)


@router.post("/boxes/row", status_code=201)
def detect_row_boxes(body: RowBody, workspace: WorkspaceDep) -> dict[str, Any]:
    """Box the protein's first band in every declared lane from the row box
    dragged over its row (:func:`~proteia.core.operations.detect_row_boxes`),
    in image pixels. Answers what the row did in each lane
    (:func:`_row_placement`); the same drag again changes nothing."""
    session = workspace.current()
    placement = ops.detect_row_boxes(session, body.protein_id, body.rect)
    return _answer(workspace, session, **_row_placement(placement))


@router.put("/boxes/{band_id}")
def move_box(band_id: str, body: MoveBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.move_box(session, band_id, body.rect)
    return _answer(workspace, session)


@router.delete("/boxes/{band_id}")
def remove_box(band_id: str, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.remove_box(session, band_id)
    return _answer(workspace, session)


@router.put("/boxes/{band_id}/lane")
def set_box_lane(band_id: str, body: LaneIndexBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    ops.set_box_lane(session, band_id, body.lane_index)
    return _answer(workspace, session)


@router.delete("/proteins/{protein_id}/boxes")
def clear_boxes(protein_id: str, workspace: WorkspaceDep) -> dict[str, Any]:
    """Remove every box and not-detected record of the protein; it keeps its box
    size. A protein with neither is a no-op."""
    session = workspace.current()
    cleared = ops.clear_boxes(session, protein_id)
    return _answer(
        workspace,
        session,
        removed=list(cleared.band_ids),
        dropped_undetected=[list(key) for key in cleared.undetected],
    )


def _restored(restored: ops.Restored) -> dict[str, Any]:
    """What an undo or redo took back or made again
    (:class:`~proteia.core.operations.Restored`)."""
    return {
        "action": restored.action,
        "seq": restored.seq,
        "removed": list(restored.removed),
        "restored": list(restored.restored),
        "undetected_removed": [list(key) for key in restored.undetected_removed],
        "undetected_restored": [list(key) for key in restored.undetected_restored],
    }


@router.post("/requantify")
def requantify(workspace: WorkspaceDep) -> dict[str, Any]:
    """Switch the project to the local background and re-quantify every band;
    answers the images re-quantified (``images``; empty for a project already
    on the local background, a no-op)."""
    session = workspace.current()
    images = ops.requantify(session)
    return _answer(workspace, session, images=list(images))


@router.post("/export", status_code=201)
def export_bundle(workspace: WorkspaceDep, body: ExportBody | None = None) -> dict[str, Any]:
    """Write the results into a new export folder
    (:func:`~proteia.core.operations.export_bundle`), computed with the settings
    they are shown with, in the chart ``formats`` asked for (by default
    :data:`~proteia.core.export.DEFAULT_CHART_FORMATS`). Answers the folder,
    relative to the project folder (``exports/<name>``, for ``POST
    /api/project/reveal``), and the files in it, by name, in the order written.
    Refusals are 422 with the operation's codes: ``no_lanes``,
    ``image_file_changed``, ``path_too_long`` and ``invalid_input``."""
    session = workspace.current()
    formats = DEFAULT_CHART_FORMATS if body is None or body.formats is None else body.formats
    bundle = ops.export_bundle(session, formats=formats, **dataclasses.asdict(workspace.settings))
    return _answer(
        workspace,
        session,
        folder=bundle.folder.relative_to(session.folder).as_posix(),
        files=list(bundle.files),
    )


@router.post("/undo")
def undo(workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    return _answer(workspace, session, **_restored(ops.undo(session)))


@router.post("/redo")
def redo(workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.current()
    return _answer(workspace, session, **_restored(ops.redo(session)))


# --- Errors ---


def _error(
    status: int,
    code: str,
    message: str,
    ids: tuple[str, ...] = (),
    detail: dict[str, Any] | None = None,
) -> JSONResponse:
    """An error answer: ``{code, message, ids}``, and ``detail`` when given."""
    body: dict[str, Any] = {"code": code, "message": message, "ids": list(ids)}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(body, status_code=status)


def install(app: FastAPI, workspace: Workspace) -> None:
    """Add the routes and their error answers to ``app``."""
    app.state.workspace = workspace
    app.include_router(router)
    answers: dict[type[Exception], Callable[[Exception], JSONResponse]] = {
        OperationError: lambda e: _error(422, e.code.value, str(e), e.ids, e.detail),
        UnknownIdError: lambda e: _error(404, "unknown_id", str(e)),
        FolderNotFoundError: lambda e: _error(404, "folder_not_found", str(e)),
        NoProjectError: lambda e: _error(409, "no_project", str(e)),
        UnsavedChangesError: lambda e: _error(409, "unsaved_changes", str(e)),
        UploadTooLargeError: lambda e: _error(413, "image_too_large", str(e)),
        projects.ProjectNameError: lambda e: _error(422, "invalid_project_name", str(e)),
        projects.ProjectExistsError: lambda e: _error(409, "project_exists", str(e)),
        projects.ProjectNotFoundError: lambda e: _error(404, "project_not_found", str(e)),
        ProjectError: lambda e: _error(422, "unreadable_project", str(e)),
        OSError: lambda e: _error(
            500, "file_error", str(e)
        ),  # e.g. a folder that cannot be written
        RequestValidationError: lambda e: _error(
            422, "invalid_input", "; ".join(_describe(error) for error in e.errors())
        ),
        # A model a route builds itself; operations turn theirs into OperationError.
        ValidationError: lambda e: _error(
            422, "invalid_input", "; ".join(_describe(error) for error in e.errors())
        ),
    }
    for kind, answer in answers.items():
        app.add_exception_handler(kind, _handler(answer))


def _handler(answer: Callable[[Exception], JSONResponse]):
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return answer(exc)

    return handle


def _describe(error: dict[str, Any]) -> str:
    where = ".".join(str(part) for part in error.get("loc", ()))
    return f"{where}: {error.get('msg', 'invalid')}"
