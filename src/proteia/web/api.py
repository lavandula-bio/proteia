# SPDX-License-Identifier: Apache-2.0
"""The HTTP routes: thin adapters over the project operations (ADR 0002).

Every edit goes through :mod:`proteia.core.operations` and answers with the
whole project state (:func:`~proteia.web.state.project_state`) and its results
(:func:`~proteia.web.results_view.results_payload`), both from one snapshot
(:func:`~proteia.core.operations.compute_view`), so the browser redraws the
image, the table and the charts from what the server stored. Creating, opening
and reading the project answer the same way. One project is open at a time. A
removal of a protein or an image also answers what it took with it: every field
of :class:`~proteia.core.operations.Cascade`, as lists of ids.

Errors answer JSON ``{"code", "message", "ids"}``: an operation's refusal is 422
with its :class:`~proteia.core.session.ErrorCode` value; an unknown id 404;
``no_project`` 409 before a project is open; ``invalid_input`` 422 for a request
the routes cannot read.
"""

from __future__ import annotations

import dataclasses
import tempfile
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final

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
from proteia.core.analyze import ReduceMethod
from proteia.core.model import BoxSize, UnknownIdError
from proteia.core.plotspec import ErrorType
from proteia.core.results import Results
from proteia.core.session import Clock, OperationError, ProjectSession, utc_now
from proteia.core.storage import ProjectError
from proteia.web import projects
from proteia.web.results_view import results_payload
from proteia.web.state import preview_png, project_state, revision

MAX_UPLOAD_BYTES: Final = 512 * 1024 * 1024
_WRITE_BYTES: Final = 1024 * 1024  # an upload is written to disk in pieces this large
_PREVIEWS_KEPT: Final = 8


class NoProjectError(RuntimeError):
    """No project is open."""


class UnsavedChangesError(RuntimeError):
    """The open project has changes that could not be saved; it stays open."""


class UploadTooLargeError(ValueError):
    """An upload longer than :data:`MAX_UPLOAD_BYTES`."""


@dataclass(frozen=True)
class ResultSettings:
    """How the results are computed: the keyword arguments of
    :func:`~proteia.core.operations.compute_view`. The defaults until the web UI
    can choose them."""

    plot_conditions: tuple[str, ...] | None = None
    error_type: ErrorType = ErrorType.SD
    method: ReduceMethod = ReduceMethod.MEAN


_ResultsKey = tuple[int, int, ResultSettings]  # open id, revision, settings


class Workspace:
    """The server's state: the projects root and the one open project.

    A request keeps the session it started with: a project switch while it runs
    does not redirect it (one user, one tab, so this only matters in a race).
    Every create or open gives the new session the next open id, so answers
    about different openings never compare equal, even at the same revision.
    The results of the open project's latest revision computed so far are kept
    (with the open id, the revision and the settings they belong to), since
    reading them again is common and computing them is not cheap.
    """

    def __init__(
        self, root: Path, *, reveal: Callable[[Path], None], clock: Clock = utc_now
    ) -> None:
        self.root = root
        self.reveal = reveal
        self.clock = clock
        # Guards the open session, the open ids, the settings, the previews and the
        # results; never held while computing.
        self._lock = threading.Lock()
        self._switching = threading.Lock()  # one switch at a time; never held by readers
        self._session: ProjectSession | None = None
        self._open_id = 0  # the open id of the latest create or open
        # Each session's open id, for as long as a request still holds that session.
        self._open_ids: weakref.WeakKeyDictionary[ProjectSession, int] = weakref.WeakKeyDictionary()
        self._settings = ResultSettings()
        self._results: tuple[_ResultsKey, Results] | None = None
        self._previews: OrderedDict[tuple[str, str], bytes] = OrderedDict()

    def current(self) -> ProjectSession:
        with self._lock:
            if self._session is None:
                raise NoProjectError("create or open a project first")
            return self._session

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
        and stays open if ``make`` fails or its unsaved changes cannot be saved.
        Saving and opening run outside the lock readers take."""
        with self._switching:
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
            return session

    def create(self, name: object) -> ProjectSession:
        return self._switch(lambda: projects.create_project(self.root, name, clock=self.clock))

    def open(self, name: object) -> ProjectSession:
        return self._switch(lambda: projects.open_named(self.root, name, clock=self.clock))

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

    def preview(self, session: ProjectSession, image_id: str) -> bytes:
        """The image's preview PNG, kept for the last few images shown."""
        image = session.project.batch.find_image(image_id)
        key = (image_id, image.sha256)
        with self._lock:
            if key in self._previews:
                self._previews.move_to_end(key)
                return self._previews[key]
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


class LaneIndexBody(_Body):
    lane_index: StrictInt


# --- Routes ---


def _workspace(request: Request) -> Workspace:
    return request.app.state.workspace


WorkspaceDep = Annotated[Workspace, Depends(_workspace)]


def _answer(workspace: Workspace, session: ProjectSession, **extra: Any) -> dict[str, Any]:
    """A route's answer: ``extra``, then the project state and its results, both
    from one snapshot of ``session``."""
    open_id, view = workspace.view(session)
    return {
        **extra,
        "project": project_state(session.folder.name, session, view.project, open_id=open_id),
        "results": results_payload(view.results, open_id=open_id, revision=revision(view.project)),
    }


def _cascade(cascade: ops.Cascade) -> dict[str, list[str]]:
    """Every field of what a removal took with it, as lists of ids."""
    return {field.name: list(getattr(cascade, field.name)) for field in dataclasses.fields(cascade)}


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


@router.post("/projects/open")
def open_project(body: NameBody, workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, workspace.open(body.name))


@router.get("/project")
def get_project(workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, workspace.current())


@router.post("/project/reveal", status_code=204)
def reveal_project(workspace: WorkspaceDep) -> Response:
    workspace.reveal(workspace.current().folder)
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
def image_preview(image_id: str, workspace: WorkspaceDep) -> Response:
    session = workspace.current()
    return Response(workspace.preview(session, image_id), media_type="image/png")


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


# --- Errors ---


def _error(status: int, code: str, message: str, ids: tuple[str, ...] = ()) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "ids": list(ids)}, status_code=status)


def install(app: FastAPI, workspace: Workspace) -> None:
    """Add the routes and their error answers to ``app``."""
    app.state.workspace = workspace
    app.include_router(router)
    answers: dict[type[Exception], Callable[[Exception], JSONResponse]] = {
        OperationError: lambda e: _error(422, e.code.value, str(e), e.ids),
        UnknownIdError: lambda e: _error(404, "unknown_id", str(e)),
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
