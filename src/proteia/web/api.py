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

``PUT /api/proteins/{protein_id}/box-size`` takes a protein's fitted size, as the
state shows it (``fitted_size``), not its box size: every box becomes that size
plus the protein's padding on each side
(:func:`~proteia.core.operations.set_box_size`), so the size shown, sent back
as it is, changes nothing. ``PUT /api/proteins/{protein_id}/box-padding`` sets
that padding, ``{across, along}`` in whole pixels on each side; a direction left
out keeps its value, so a page that sends only the one it changed never resets
the other (:func:`~proteia.core.operations.set_box_padding`). It answers every
field of :class:`~proteia.core.operations.PaddingChange`: ``box_size`` and
``fitted_size`` as ``{width, height}``, the padding as ``box_padding {across,
along}``, ``net_change`` as ``[smallest, largest]`` (or null), the band ids in
``edge_shifted`` and ``overlapping``, and the other proteins' bands it
re-measured and the largest change as a row box answers them.

Molecular-weight calibration (#58). ``GET /api/ladders`` lists the ladder
presets (:mod:`proteia.core.ladders`), each as ``{key, product,
catalog_numbers, system, kda, reference: [{kda, colour}], source}``; it reads no
project. ``PUT /api/images/{image_id}/marker`` links a chemiluminescence image
to its marker image, ``{marker_image_id}``, or unlinks it with null
(:func:`~proteia.core.operations.set_marker_image`); ``PUT
/api/membranes/{membrane_id}/calibration/ladder`` chooses the membrane's ladder,
``{ladder, kda?}`` (:func:`~proteia.core.operations.set_ladder`). A calibration
point is named by the register group of the image in the path, its ladder
``side`` (``left`` or ``right``; any other word answers 404 ``unknown_id``) and
its MW, written in the path as Python writes the number (``100``, ``61.5``):
``POST /api/images/{image_id}/calibration/{side}/points`` marks one, ``{y, mw,
source, x, snap?}`` (:func:`~proteia.core.operations.add_calibration_point`),
``PATCH .../points/{mw}`` moves or relabels it, ``{y?, mw?, snap?}``, a field
left out kept (:func:`~proteia.core.operations.edit_calibration_point`), and
``DELETE .../points/{mw}`` removes it; ``DELETE
/api/images/{image_id}/calibration`` removes every point of the image's group,
or those of one side with ``?side=``. These answer, besides the state, what the
change did (:class:`~proteia.core.operations.CalibrationUpdate`): the ``point``
as stored with ``snapped``, the group's ``fit`` (null without a curve; an
infinite quality or disagreement, which JSON cannot hold, is null with
``quality_infinite`` or ``disagreement_infinite`` true), the images whose curve
changed (``curves_changed``) and the not-detected records dropped
(``dropped_undetected``, as ``[protein id, lane index, band index]``).

Finding a ladder (#58). ``POST /api/images/{image_id}/ladder-proposal``,
``{x, side?}``, finds the ladder whose lane was clicked at ``x`` and answers
``{proposal}``: its ticks, extra peaks, score, gap and whether it is doubtful
(:func:`~proteia.core.operations.proposal_json`), or null where no ladder
stands out. ``POST /api/images/{image_id}/ladder-snap``, ``{x, ys}``, snaps
each tick of a ruler to its band and answers ``{points: [{y, snapped}]}``, in
the order given. Both only read: nothing is changed, logged or saved, and
they answer no project. ``PUT /api/images/{image_id}/calibration/{side}/ladder``,
``{x, points: [{y, mw}], found_at?}``, applies a ruler in one change and one
undo step (:func:`~proteia.core.operations.set_ladder_points`), and answers as
the other calibration routes do, with the ``points`` applied, each with how it
was ``placed`` (and, with ``found_at``, whether it was ``relabelled``), and
``sides_swapped``. A ruler holds at most :data:`MAX_RULER_TICKS` ticks.

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

``GET /api/diagnostics`` lists what a diagnostic file for a bug report would
hold now (:mod:`proteia.web.diagnostics`): the open project's name and open id
(null when none is open), whether its changes are saved, the files it takes
(``files``), the project's image files, taken only when asked for (``images``),
each as ``{name, size}``, the files left out (``left_out``, each ``{name,
size, reason}``), and ``digest``, of the project's files it lists. ``POST
/api/diagnostics`` writes it, with ``{"images", "open_id", "digest"}``:
whether to take the images, and the open id (null: none was open) and digest
the list answered, so neither a project opened since nor a project file made
since is ever written unlisted: another opening is refused as
``project_changed``, or ``no_project``, and other files as ``files_changed``
(the page lists again). It answers the
file's ``name``, ``path`` and ``size``, how many files it holds and how many
were left out. ``POST /api/diagnostics/reveal`` shows the folder it is written
in. These read the open project if one is open, and work with none.

``GET /api/notices`` answers the notices the page shows once per user:
``cloud_sync``, ``{service}``, while the projects folder lies in a folder that
sync service uploads (:mod:`proteia.web.cloudsync`) and the notice is not
dismissed, else null. ``POST /api/notices/cloud_sync/dismiss`` dismisses it for
good, recorded in the per-user state folder. Without a state folder the notice
is never offered, and a dismissal answers ``no_state_folder``.

Every route that reads or edits the open project takes an optional
``Proteia-Opening`` header (:data:`OPENING_HEADER`): the open id of the project
the page shows, as its answers carry it. A request that names another opening
is refused before anything is done, so a page still showing a project opened
before (another tab opened one since, or Proteia read the open one's
``project.json`` again) neither edits nor reads the one open now (#134).
Without the header there is no check. Either way the route runs, and answers,
within the opening it was given: Proteia reads the open project's
``project.json`` again only once no such request runs, and one made meanwhile
waits for that, then is checked (:meth:`Workspace.using`); so no edit is made
in, and no answer is about, an opening the request did not name. An upload
(``POST /api/images``) is checked before any of its body is read, and again
once the body is stored: only its import and answer run within the opening,
so a file that takes minutes to arrive holds up no reopen, and an upload
whose opening ended meanwhile is refused, with nothing imported. A diagnostic
file (``POST /api/diagnostics``) is checked, and planned, within the opening,
then written once the session is released: one that takes minutes (with the
images) holds up no reopen either. ``GET
/api/workspace`` answers which project is open, and its open id, without
reading it: for a page to find out. The routes that list, create or open
projects, the status and quit routes, the one that shows the diagnostics
folder, and the notices', need no opening.

Images handed to the running app by a launch wait in the workspace's inbox
(:mod:`proteia.web.handoff`) until the page imports or discards them. ``POST
/api/incoming?name=<name>`` takes one file's bytes, as ``POST /api/images``
does, into the private staging folder under a name the server makes, and
answers ``{file_id, name, size}``; ``GET /api/incoming/room?name=<name>&size=<bytes>``
answers 204 if it would take that file now, or the refusal it would get before
its body is read, holding nothing; ``POST /api/handoffs`` offers uploaded files
(``files``, their ids) with the arguments the launch refused (``refused``,
``{name, code, message}`` each, bounded, never refusing the offer), and answers
``{handoff_id, merged, files, refused}``: the hand-off they went to, whether it
was pending already, and how many files and refused entries it holds. ``GET
/api/workspace`` lists the pending hand-offs as ``handoffs``, each ``{id, kind,
files: [{file_id, name, size}], suggested_name, refused, more_refused,
more_may_arrive, claimed}``, with the name its project would take now;
``claimed`` while an accept imports it: no other accept or discard takes it
then (``handoff_claimed``), and an accept refused leaves it as it was. ``POST
/api/handoffs/{id}/accept`` imports a hand-off into a new project
(:meth:`Workspace.accept`), with a kind, a polarity and a membrane (``"new"``,
or the index of an earlier file whose membrane it joins) for each file, and
answers as a create does, with ``handoff``: what was imported, what was not and
why, the launch's refused entries and notes. ``POST /api/handoffs/{id}/discard``
drops one as the page shows it. No path crosses HTTP: the page sees file ids,
names and sizes only. These routes need no opening, except the accept: it
closes the open project, so it is refused, as an edit is, for a page that
shows another.

Errors answer JSON ``{"code", "message", "ids"}``: an operation's refusal is 422
with its :class:`~proteia.core.session.ErrorCode` value, and ``detail`` when the
refusal carries one (a row box's: what the detector saw); an unknown id 404, and
an export folder to reveal that does not exist 404 ``folder_not_found``;
``no_project`` 409 before a project is open; ``no_state_folder`` 409 for a
diagnostic file, or a notice's dismissal, when Proteia was served without its
state folder, and
``files_changed`` 409 for one whose project files are not those listed;
``project_changed`` 409 for a
request that names an opening no longer open, with ``detail`` ``{open,
open_id}``: the open project's name and open id; ``invalid_input`` 422 for a
request the routes cannot read, a ``Proteia-Opening`` not in plain digits too.
Every error answer is logged (:mod:`proteia.web.logs`) with the request's method
and path (cut short: :func:`~proteia.web.logs.shorten`), its status, code and
message, and its ids and ``detail`` when it has them; a 500 with its stack
trace, in the log file only. A request no route takes (a 404 or 405, which a
read of the page shell can get without the token) is logged as the guard logs
a refusal: at most :data:`~proteia.web.logs.REFUSALS_LOGGED` a minute one by
one, the rest counted.
The hand-off routes add: ``stopping``, ``too_many_pending``, ``file_claimed``,
``handoff_claimed`` and ``handoff_changed`` (``detail``: the hand-off as it is
listed now) 409, ``handoff_not_found`` 404, and ``nothing_imported`` 422
(``detail.refused``: each file and why). ``unsaved_changes`` carries
``detail.created`` when a switch created its project but then could not save
the one open, which stays open.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler
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
from starlette.exceptions import HTTPException

from proteia.core import ladders, storage
from proteia.core import operations as ops
from proteia.core.analyze import ReduceMethod
from proteia.core.export import DEFAULT_CHART_FORMATS
from proteia.core.model import BoxSize, ImageKind, LadderSide, Polarity, UnknownIdError
from proteia.core.plotspec import ErrorType, PlotSpec
from proteia.core.results import Results
from proteia.core.session import Clock, ErrorCode, OperationError, ProjectSession, utc_now
from proteia.core.storage import ProjectError
from proteia.web import cloudsync, diagnostics, handoff, logs, projects, sample_project
from proteia.web.charts import ChartStore
from proteia.web.handoff import HandoffView, Inbox, Refusal
from proteia.web.results_view import results_payload
from proteia.web.state import has_colour, original_png, preview_png, project_state, revision

_log = logging.getLogger(__name__)

# The request header that names the opening of a project a page shows (its open id).
OPENING_HEADER: Final = "Proteia-Opening"
MAX_UPLOAD_BYTES: Final = 512 * 1024 * 1024
_WRITE_BYTES: Final = 1024 * 1024  # an upload is written to disk in pieces this large
_PREVIEWS_KEPT: Final = 8
# How long a reopen waits for an operation running on the open session before
# answering it without reading its project.json again.
REOPEN_WAIT_S: Final = 5.0

_log = logging.getLogger(__name__)


class NoProjectError(RuntimeError):
    """No project is open."""


class ProjectChangedError(RuntimeError):
    """A request names an opening of a project that is no longer the open one
    (``named``; None: it names none, as a diagnostic file listed with no project
    open does): ``open`` (the open project's name) and ``open_id`` say which is."""

    def __init__(self, named: int | None, name: str, open_id: int) -> None:
        which = "no opening" if named is None else f"opening {named}"
        super().__init__(
            f"the request names {which}, but {name!r} is open now (opening {open_id}):"
            " read the project again"
        )
        self.open = name
        self.open_id = open_id


class UnsavedChangesError(RuntimeError):
    """The open project has changes that could not be saved; it stays open.
    ``created``: the project a switch created before that save failed, which
    exists, complete, and stays closed; None if none was."""

    def __init__(self, message: str, *, created: str | None = None) -> None:
        super().__init__(message)
        self.created = created


class NothingImportedError(RuntimeError):
    """No file of a hand-off could be imported: ``refused`` says why, for each."""

    def __init__(self, refused: tuple[NotImported, ...]) -> None:
        super().__init__("none of the images could be imported")
        self.refused = refused


@dataclass(frozen=True)
class FileChoice:
    """How a file of a hand-off is imported: its kind and polarity, and
    ``membrane``, the index (in the accept's list) of an earlier file whose
    membrane it joins, or None for a new membrane."""

    file_id: str
    kind: ImageKind
    polarity: Polarity
    membrane: int | None = None


@dataclass(frozen=True)
class Imported:
    """A file of a hand-off imported, as the image ``image_id`` on
    ``membrane_id`` (``new_membrane``: one it started)."""

    file_id: str
    name: str
    image_id: str
    membrane_id: str
    new_membrane: bool


@dataclass(frozen=True)
class NotImported:
    """A file of a hand-off not imported: the operation's code and message, or
    ``file_error`` (it could not be read or stored) or ``stopping``."""

    file_id: str
    name: str
    code: str
    message: str


@dataclass(frozen=True)
class Accepted:
    """What an accept did (:meth:`Workspace.accept`): the files imported and not,
    in order; ``notes`` (a file put on a new membrane because the one it was to
    join was not imported); and the arguments the launches refused
    (``launch_refused``, and ``more_refused`` not kept)."""

    imported: tuple[Imported, ...]
    refused: tuple[NotImported, ...]
    notes: tuple[str, ...]
    launch_refused: tuple[Refusal, ...]
    more_refused: int


def _was_created(name: str) -> str:
    return f"{name!r} was created"


def _reason(exc: OSError) -> str:
    """Why a file could not be read or written, naming no path: its strerror."""
    return exc.strerror or type(exc).__name__


def _not_saved(exc: Exception) -> UnsavedChangesError:
    """The open project could not be saved (``exc``), and nothing was made."""
    return UnsavedChangesError(f"the open project could not be saved: {exc}")


class UploadTooLargeError(ValueError):
    """An upload longer than :data:`MAX_UPLOAD_BYTES`."""


class FolderNotFoundError(LookupError):
    """No export folder of the open project has this name."""


class NoStateFolderError(RuntimeError):
    """Proteia was served without its per-user state folder (a launch gives it
    one), so it has no session log to read and no diagnostics folder."""


class FilesChangedError(RuntimeError):
    """A diagnostic file would hold other project files than the page listed
    (:meth:`proteia.web.diagnostics.Plan.digest`): made or removed since, in
    the same opening."""


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
    does not redirect it (so this only matters in a race). A request that names
    the opening its page shows gets the session only while that is the open one
    (:meth:`current`): a page showing a project no longer open, in another tab
    say, neither edits nor reads the one open now.
    Every create, and every open of another project, gives the new session the
    next open id, so answers about different openings never compare equal, even
    at the same revision. Opening the project already open answers its session
    (:meth:`open`), so no two sessions write one folder (#93); that session
    takes the next open id too if it reads a ``project.json`` changed outside
    Proteia again. It never does so while a request uses it (:meth:`using`):
    every request runs, and is answered, wholly within the opening it was
    given, so what a page named is what its request reads or edits.
    The results of the open project's latest revision computed so far are kept
    (with the open id, the revision and the settings they belong to), since
    reading them again is common and computing them is not cheap. So are the
    charts of the answers about the latest opening (:class:`ChartStore`).
    Images handed to the app wait in :attr:`inbox` until an accept imports
    them into a new project (:meth:`accept`), a switch like any other.
    Whether the projects root lies in a folder a sync service uploads is
    checked once, when first asked (:meth:`synced_folder`).

    The locks, in the order they are taken: the switch lock (a create or an
    open holds it throughout), a session's lock (an operation holds it while it
    runs; a reopen takes it to read ``project.json`` again; a switch takes the
    old session's to save it a last time and replace it), this workspace's
    lock (held briefly, never while computing or waiting for another lock),
    then the chart store's. None is taken while one after it is held. Waiting
    adds no cycle: a request waits for a reopen under way (:meth:`using`) before
    it takes any of them, on this lock's condition, which releases it
    meanwhile; a reopen waits for the requests using the session holding only
    the switch lock, which no request takes, and at most :data:`REOPEN_WAIT_S`.
    The inbox's lock, and the lock of the check whether the projects root is
    synced (:meth:`synced_folder`), are taken with none of these held, and none
    is taken while either is.
    """

    def __init__(
        self,
        root: Path,
        *,
        reveal: Callable[[Path], None],
        clock: Clock = utc_now,
        inbox: Inbox | None = None,
        state: Path | None = None,
        sync_check: Callable[[Path], cloudsync.SyncedFolder | None] = cloudsync.check,
    ) -> None:
        self.root = root
        self.reveal = reveal
        self.clock = clock
        # Images handed to the app; a launch gives it its staging folder.
        self.inbox = Inbox() if inbox is None else inbox
        # The per-user state folder, which holds the session log, the
        # diagnostics folder and the notices dismissed; a launch gives it
        # (proteia.web.launch).
        self.state = state
        # Whether the projects root lies in a folder a sync service uploads:
        # checked once, when first asked (synced_folder), under its own lock.
        self._sync_check = sync_check
        self._sync_lock = threading.Lock()
        self._synced: tuple[cloudsync.SyncedFolder | None] | None = None  # None: not checked
        # Guards the open session, the open ids, the sessions in use, the reopen
        # under way, the settings, the previews and the results; never held while
        # computing. Taken before the chart store's own lock, never while holding it.
        self._lock = threading.Lock()
        # Notified whenever a request stops using a session and when a reopen ends.
        self._changed = threading.Condition(self._lock)
        self._switching = threading.Lock()  # one switch at a time; never held by readers
        self._session: ProjectSession | None = None
        self._open_id = 0  # the open id of the latest create or open
        # Each session's open id, for as long as a request still holds that session.
        self._open_ids: weakref.WeakKeyDictionary[ProjectSession, int] = weakref.WeakKeyDictionary()
        # The requests using each session (using, answering): none is read again
        # while it has one.
        self._in_use: dict[ProjectSession, int] = {}
        # The open session while a reopen may read it again: none starts using it then.
        self._reopening: ProjectSession | None = None
        self._settings = ResultSettings()
        self._results: tuple[_ResultsKey, Results] | None = None
        # (image id, SHA-256, original colours) -> preview PNG, least recently shown first.
        self._previews: OrderedDict[tuple[str, str, bool], bytes] = OrderedDict()
        self._charts = ChartStore()

    def current(self, opening: int | None = None) -> ProjectSession:
        """The open project's session; with ``opening``, only if that is its open
        id (:class:`ProjectChangedError` otherwise), checked and answered under
        one lock, so the session answered is the one checked."""
        with self._lock:
            return self._checked(opening)

    def _checked(self, opening: int | None) -> ProjectSession:
        """:meth:`current`'s answer. Called with the lock held."""
        if self._session is None:
            raise NoProjectError("create or open a project first")
        if opening is not None and opening != self._open_id:
            raise ProjectChangedError(opening, self._session.folder.name, self._open_id)
        return self._session

    def current_any(self, opening: int | None = None) -> ProjectSession | None:
        """:meth:`current`, for a request that reads the open project if there is
        one: None when none is open and ``opening`` is None, as
        :meth:`using_any` gives it; checked and answered under one lock."""
        with self._lock:
            return self._checked_any(opening)

    def _checked_any(self, opening: int | None) -> ProjectSession | None:
        """:meth:`current_any`'s answer. Called with the lock held."""
        if self._session is None and opening is None:
            return None
        return self._checked(opening)

    @contextlib.contextmanager
    def using(self, opening: int | None = None) -> Iterator[ProjectSession]:
        """The open project's session, as :meth:`current` gives it, for a request
        that reads or edits it, in use until the block ends. Meanwhile no reopen
        reads its ``project.json`` again (:meth:`open`), so its open id and
        project stay those of the opening checked: an operation the request runs
        is made in the project its page shows, never in one read again after
        the check, and its answer is of that opening. A request made while a
        reopen runs waits for it, then is checked against the opening it leaves.
        A switch while it runs does not redirect it, as the class says."""
        with self._lock:
            self._changed.wait_for(lambda: self._reopening is None)
            session = self._checked(opening)
            self._in_use[session] = self._in_use.get(session, 0) + 1
        try:
            yield session
        finally:
            self._done(session)

    @contextlib.contextmanager
    def using_any(self, opening: int | None = None) -> Iterator[ProjectSession | None]:
        """:meth:`using`, for a request that reads the open project if there is
        one: None, and nothing in use, when none is open and the request names
        no opening; checked and taken under one lock, as :meth:`using` does."""
        with self._lock:
            self._changed.wait_for(lambda: self._reopening is None)
            session = self._checked_any(opening)
            if session is not None:
                self._in_use[session] = self._in_use.get(session, 0) + 1
        try:
            yield session
        finally:
            if session is not None:
                self._done(session)

    def opening_of(self, session: ProjectSession | None) -> int | None:
        """The open id of ``session``, a session a request holds (:meth:`using`),
        or None for none."""
        if session is None:
            return None
        with self._lock:
            return self._open_ids[session]

    def state_folder(self) -> Path:
        """The per-user state folder (:attr:`state`), or
        :class:`NoStateFolderError` without one."""
        if self.state is None:
            raise NoStateFolderError(
                "Proteia was started without its state folder: it has no session log and"
                " nowhere to write a diagnostic file"
            )
        return self.state

    def synced_folder(self) -> cloudsync.SyncedFolder | None:
        """The folder a sync service uploads that the projects root lies in
        (:func:`~proteia.web.cloudsync.check`, or the check this workspace was
        given), or None: checked once, when first asked, and logged by the
        service's name and where it was found, never by path. A check that
        fails is logged, with its stack trace in the log file, and taken as
        none."""
        with self._sync_lock:
            if self._synced is None:
                try:
                    synced = self._sync_check(self.root)
                except Exception:
                    synced = None
                    _log.warning(
                        "could not check whether the projects folder is synced to the cloud",
                        exc_info=True,
                        extra=logs.FILE_ONLY,
                    )
                else:
                    if synced is None:
                        _log.info(
                            "the projects folder is in no folder a sync service Proteia recognises"
                        )
                    else:
                        _log.info(
                            "the projects folder is in a folder %s uploads (found from %s)",
                            synced.service,
                            synced.source,
                        )
                self._synced = (synced,)
            return self._synced[0]

    @contextlib.contextmanager
    def answering(self, session: ProjectSession) -> Iterator[None]:
        """``session`` in use (:meth:`using`) while a create or an open makes its
        answer, so the answer is of one opening: a reopen of it from another
        page, under way or made meanwhile, is waited for or waits."""
        with self._lock:
            self._changed.wait_for(lambda: self._reopening is not session)
            self._in_use[session] = self._in_use.get(session, 0) + 1
        try:
            yield
        finally:
            self._done(session)

    def _done(self, session: ProjectSession) -> None:
        """A request stops using ``session``."""
        with self._lock:
            left = self._in_use.pop(session) - 1
            if left:
                self._in_use[session] = left
            self._changed.notify_all()

    def opened(self) -> tuple[str | None, int | None]:
        """The open project's name and open id, read together; (None, None)
        before a project is open."""
        with self._lock:
            if self._session is None:
                return None, None
            return self._session.folder.name, self._open_id

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
            raise _not_saved(exc) from exc

    def flush(self) -> None:
        """Save the open project if it has unsaved changes (an autosave failed);
        :class:`UnsavedChangesError` if that fails again."""
        session = self._peek()
        if session is not None and session.dirty:
            self._save(session)

    def _switch(
        self,
        make: Callable[[], ProjectSession],
        *,
        created: Callable[[str], str] | None = _was_created,
    ) -> ProjectSession:
        """Replace the open project with ``make()``; the old one is saved first,
        and stays open, with its undo history, if ``make`` fails or its unsaved
        changes cannot be saved. While ``make`` runs the old one is still open,
        and edited if a request asks. So once ``make`` returns, the old one's
        lock is taken, which waits for an operation running on it, and it is
        saved again if an edit's autosave failed meanwhile; it is replaced
        before the lock is released, so no edit falls between. If that save
        fails, the new project is not opened, while the old one stays open with
        its changes. A project ``make`` created stays on disk, closed, and
        :class:`UnsavedChangesError` names it (``created``, and ``created(name)``
        says so in its message); with ``created`` None (an open, which creates
        nothing) the error is the one a failed first save raises. Once replaced,
        the old one is closed (:meth:`~proteia.core.session.ProjectSession.close`):
        its history is gone, and so are the image files only that history
        kept, unless the new session's folder is, or may be, the old one's.
        Then it is replaced without waiting for its lock, and the close deletes
        nothing, since the new session may already be storing files the old one
        does not know (the close waits for any request still running on the old
        one), and the new session's first save or import deletes those files;
        nor is the old one saved again, over the new one's file. After the
        close, an edit that took the old session before it was replaced, and
        whose autosave failed, is saved once more; a failure then is logged.
        Opening the open project never switches (:meth:`open`), but a project
        created where the open one's folder was removed outside Proteia does.
        Saving, opening and closing run outside the lock readers take. Called
        with the switch lock held."""
        old = self._peek()
        if old is not None and old.dirty:
            self._save(old)
        session = make()
        if old is None:
            self._opened(session)
            return session
        shared = _same_folder(old.folder, session.folder, unknown=True)
        if shared:  # without waiting for a request running on the old one
            self._opened(session)
        else:
            with old.lock:
                if old.dirty:
                    try:
                        old.save()
                    except (OSError, ProjectError) as exc:
                        session.close(remove_files=False)
                        if created is None:
                            raise _not_saved(exc) from exc
                        name = old.folder.name
                        raise UnsavedChangesError(
                            f"{name!r} could not be saved: {exc}. {created(session.folder.name)};"
                            f" open it from Projects once {name!r} can be saved",
                            created=session.folder.name,
                        ) from exc
                self._opened(session)
        old.close(remove_files=not shared)
        if old.dirty and not shared:
            try:
                old.save()
            except (OSError, ProjectError) as exc:
                _log.warning(
                    "%r was closed with changes that could not be saved: %s", old.folder.name, exc
                )
        return session

    def _opened(self, session: ProjectSession) -> None:
        """Make ``session`` the open one, under the next open id."""
        with self._lock:
            self._session = session
            self._open_id += 1
            self._open_ids[session] = self._open_id
            self._previews.clear()
            self._results = None
            self._charts.reset(self._open_id)

    def close(self) -> None:
        """Close the open project: its undo history is gone, and so are the image
        files only that history kept. Called when the server stops, after
        :meth:`flush`. First the inbox stops (:meth:`~proteia.web.handoff.Inbox.stop`):
        no upload, offer or accept starts, and an accept under way imports no
        further file, so this waits for one file's import at most; then the
        open project is closed, once no switch runs; then the staged files are
        deleted (:meth:`~proteia.web.handoff.Inbox.close`), once no accept reads
        them (Windows cannot delete a file held open)."""
        self.inbox.stop()
        with self._switching:
            session = self._peek()
            if session is not None:
                session.close()
        self.inbox.close()

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

    def accept(
        self,
        handoff_id: str,
        name: object,
        choices: Sequence[FileChoice],
        *,
        opening: int | None = None,
    ) -> tuple[ProjectSession, Accepted]:
        """Import the hand-off ``handoff_id`` into a new project and open it, as
        a create does (:meth:`_switch`): each file in the order of ``choices``,
        which must name its files, each once, with its kind, polarity and
        membrane, through the ordinary import (each logged and undoable).

        The project is named ``name`` (:func:`~proteia.web.projects.project_name`;
        ``project_exists`` if taken) or, with None, after the first file
        (:func:`~proteia.web.projects.name_from_file`), numbered if taken
        (:func:`~proteia.web.projects.free_name`), under the switch lock, so no
        other create takes the name meanwhile. A file that cannot be imported
        (an operation's refusal, a file that cannot be read) is left out, and
        the answer says why; one to join the membrane of a file left out starts
        a new one, and a note says so. Once Proteia is stopping, the files left
        are not imported (``stopping``).

        ``opening``: the opening the page shows, as a request names it; the
        accept is refused (:class:`ProjectChangedError`) if it is no longer the
        open one, checked under the switch lock, since the open project is
        closed. The hand-off is claimed first (:meth:`~proteia.web.handoff.Inbox.claim`),
        so another accept or a discard of it is refused meanwhile, the listing
        says it is claimed, and files offered meanwhile start another hand-off.
        Afterwards it is gone, and its staged files are deleted, when its files
        were imported; when none could be (:class:`NothingImportedError`: no
        project is left, and the open one stays open), since trying again would
        fail again; and when the open project could not be saved after the
        imports (:class:`UnsavedChangesError` with ``created``), since the
        images are in the project created. Any other refusal changes nothing,
        and it is pending again, listed as it was."""
        typed = None if name is None else projects.project_name(name)
        claimed = self.inbox.claim(handoff_id, [choice.file_id for choice in choices])
        files = {file.file_id: file for file in claimed.files}
        imported: list[Imported] = []
        refused: list[NotImported] = []
        notes: list[str] = []

        def set_up(session: ProjectSession) -> None:
            membranes: dict[int, str] = {}  # choice index -> the membrane its file went on
            for index, choice in enumerate(choices):
                file = files[choice.file_id]
                if self.inbox.stopping:
                    refused.append(
                        NotImported(file.file_id, file.name, "stopping", "Proteia is stopping")
                    )
                    continue
                membrane_id = None
                if choice.membrane is not None:
                    membrane_id = membranes.get(choice.membrane)
                    if membrane_id is None:
                        earlier = files[choices[choice.membrane].file_id].name
                        notes.append(
                            f"{file.name} was put on a new membrane because {earlier} was not"
                            " imported"
                        )
                try:
                    with file.source.open() as stream:
                        image_id = ops.import_image(
                            session,
                            stream,
                            file.name,
                            kind=choice.kind,
                            polarity=choice.polarity,
                            membrane_id=membrane_id,
                            max_bytes=MAX_UPLOAD_BYTES,
                        )
                except OperationError as exc:
                    refused.append(NotImported(file.file_id, file.name, exc.code.value, str(exc)))
                    continue
                except OSError as exc:
                    refused.append(NotImported(file.file_id, file.name, "file_error", _reason(exc)))
                    continue
                membranes[index] = session.project.batch.membrane_of(image_id).id
                imported.append(
                    Imported(
                        file.file_id,
                        file.name,
                        image_id,
                        membranes[index],
                        new_membrane=membrane_id is None,
                    )
                )
            if not imported:
                raise NothingImportedError(tuple(refused))

        def make() -> ProjectSession:
            new = typed
            if new is None:
                first = files[choices[0].file_id].name
                new = projects.free_name(self.root, projects.name_from_file(first))
            return projects.create_set_up(self.root, new, set_up, clock=self.clock)

        def into(new: str) -> str:
            return f"the images were imported into {new!r}"

        try:
            with self._switching:
                if opening is not None:
                    self.current(opening)
                if self.inbox.stopping:
                    raise handoff.StoppingError("Proteia is stopping")
                session = self._switch(make, created=into)
        except NothingImportedError:
            self.inbox.finish(claimed)
            raise
        except UnsavedChangesError as exc:
            if exc.created is None:
                self.inbox.release(claimed)
            else:
                self.inbox.finish(claimed)
            raise
        except BaseException:
            self.inbox.release(claimed)
            raise
        self.inbox.finish(claimed)
        accepted = Accepted(
            imported=tuple(imported),
            refused=tuple(refused),
            notes=tuple(notes),
            launch_refused=tuple(claimed.refused),
            more_refused=claimed.more_refused,
        )
        return session, accepted

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
        replaced by the older file: the next change or quitting saves them. The
        requests using the session (:meth:`using`: an edit, or a read such as a
        preview), and then an operation running on it, are waited for, up to
        :data:`REOPEN_WAIT_S` in all, before the file is checked: after an
        edit, which saved over it, there is nothing to read; after a read, a
        changed file is read. The requests made meanwhile wait for the reopen.
        One that runs longer leaves the file unread, and the session is answered
        as it is; opening it again reads it. When it cannot be told whether the
        two folders are one (the open one is gone), the open is a switch, whose
        close deletes nothing."""
        with self._switching:
            folder = projects.project_folder(self.root, name)
            session = self._peek()
            if session is None or not _same_folder(session.folder, folder, unknown=False):
                return self._switch(
                    lambda: ops.open_project(folder, clock=self.clock), created=None
                )
            deadline = time.monotonic() + REOPEN_WAIT_S
            try:
                with self._lock:
                    self._reopening = session  # no request starts using it now
                    idle = self._changed.wait_for(
                        lambda: session not in self._in_use, timeout=REOPEN_WAIT_S
                    )
                if idle and session.lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
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
                else:
                    _log.info(
                        "%r was opened again while a request still used it after %s s: its"
                        " project.json was not checked for changes made outside Proteia",
                        session.folder.name,
                        REOPEN_WAIT_S,
                    )
            finally:
                with self._lock:
                    self._reopening = None
                    self._changed.notify_all()
            return session

    def view(self, session: ProjectSession) -> tuple[int, ops.ComputedView]:
        """``session``'s open id, and its committed project with the results of
        it (:func:`~proteia.core.operations.compute_view`). The kept results are
        reused while the open id, the revision and the settings match; the
        computation itself runs outside the lock, and its results are kept only
        while ``session`` is still the open one and no later revision's results
        are kept. The routes call it while ``session`` is in use (:meth:`using`,
        :meth:`answering`), when no reopen reads it again: its open id and its
        project, read apart, are of one opening."""
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
        show (:func:`~proteia.web.state.has_colour`; a CMYK file's are its
        colours converted to red, green and blue); otherwise it answers the
        gray one: a gray file's original colours are its gray levels. The last
        few previews shown are kept, gray and colour alike."""
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
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


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


def _plain_int(value: str) -> bool:
    try:
        return str(int(value)) == value
    except ValueError:
        return False


def _url_kda(value: object) -> object:
    """A molecular weight in a URL as a float, only if it is written as Python
    writes the number: ``100``, ``61.5`` or ``100.0``, not ``1e2``, ``+100``,
    ``0100`` or ``61.50``, so one MW has one spelling. Whether it is a positive
    MW is the operation's to check."""
    if not isinstance(value, str):
        return value
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number) or not (repr(number) == value or _plain_int(value)):
        raise ValueError(
            f"must be a molecular weight in kDa as plain digits, such as 100 or 61.5, not {value!r}"
        )
    return number


UrlKda = Annotated[float, BeforeValidator(_url_kda)]


def _side(side: str) -> LadderSide:
    """The ladder side a path names; any other word is no such side (404)."""
    try:
        return LadderSide(side)
    except ValueError:
        raise UnknownIdError(f"no ladder side {side!r}: left or right") from None


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
    # The MW check's tolerance, a share (0.1: ±10%, #58); absent or null: the default.
    mw_tolerance: StrictFloat | None = None


class ProteinEditBody(_Body):
    """The fields a protein edit changes. A field left out keeps its value: only
    the fields the request set are passed on, so these defaults are never used,
    and null is a value only for ``expected_mw`` (no expected MW)."""

    name: str = ""
    role: str = ""
    expected_mw: ExpectedMw = None
    loading_control_ids: list[str] = []
    mw_tolerance: StrictFloat = 0.1  # a share (0.1: ±10%, #58); never null


class BoxSizeBody(_Body):
    width: PositiveInt
    height: PositiveInt


class MarkerBody(_Body):
    marker_image_id: str | None  # required: null unlinks


class LadderBody(_Body):
    ladder: str | None  # required: a preset key or a custom name; null clears it
    kda: list[StrictFloat] | None = None  # the ladder's MWs, top to bottom


class PointBody(_Body):
    """A calibration point, in continuous coordinates of the image's analysis
    array; ``x`` is required: every new point records where it was marked."""

    y: StrictFloat
    mw: StrictFloat
    source: str
    x: StrictFloat
    snap: StrictBool = True


class PointEditBody(_Body):
    """A calibration point's new position (``y``) or label (``mw``). A field
    left out keeps its value: only the fields the request set are passed on,
    so these defaults are never used."""

    y: StrictFloat = 0.0
    mw: StrictFloat = 0.0
    snap: StrictBool = False


# At most this many ticks in a ruler sent to snap or apply: a ladder has a
# dozen bands, and a list is checked tick by tick.
MAX_RULER_TICKS: Final = 200


class LadderProposalBody(_Body):
    """Where a ladder lane was clicked, and which ladder of the register group
    it is (``left`` or ``right``)."""

    x: StrictFloat
    side: str = "left"


class LadderSnapBody(_Body):
    """A ruler's ticks, drawn at ``x``, to snap each to its band."""

    x: StrictFloat
    ys: Annotated[list[StrictFloat], Field(max_length=MAX_RULER_TICKS)]


class RulerTickBody(_Body):
    y: StrictFloat
    mw: StrictFloat


class LadderPointsBody(_Body):
    """A ruler applied: its ticks at ``x``, and the x it was found at, if it was."""

    x: StrictFloat
    points: Annotated[list[RulerTickBody], Field(max_length=MAX_RULER_TICKS)]
    found_at: StrictFloat | None = None


class BoxPaddingBody(_Body):
    """A protein's padding, whole pixels on each side: ``across`` left and right,
    ``along`` above and below. A field left out keeps its value: only the fields
    the request set are passed on, so these defaults are never used, and null is
    no value."""

    across: NonNegativeInt = 0
    along: NonNegativeInt = 0


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


class DiagnosticsBody(_Body):
    images: StrictBool = False  # take the project's image files too
    # The open id GET /api/diagnostics answered, whose files the page listed;
    # required, null when no project was open.
    open_id: StrictInt | None
    # The digest of the project's files it listed, as it answered it (a SHA-256).
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RefusedBody(_Body):
    """An argument a launch refused: its base name, a code and a message, taken
    as sent and bounded by the inbox (:meth:`~proteia.web.handoff.Refusal.bounded`)."""

    name: str
    code: str
    message: str


class OfferBody(_Body):
    files: list[str] = []  # the ids POST /api/incoming answered, in order
    refused: list[RefusedBody] = []


class AcceptFileBody(_Body):
    file_id: str
    kind: str
    polarity: str  # required: the model has no silent default
    # "new", or the index in the accept's list of an earlier file whose membrane it joins.
    membrane: Literal["new"] | NonNegativeInt = "new"


class AcceptBody(_Body):
    name: str | None = None  # None: named after the first file, numbered if taken
    files: list[AcceptFileBody]


class DiscardBody(_Body):
    files: list[str]  # the hand-off's file ids as the page shows them
    refused: NonNegativeInt = 0  # its refused entries as the page counts them


# --- Routes ---


def _workspace(request: Request) -> Workspace:
    return request.app.state.workspace


WorkspaceDep = Annotated[Workspace, Depends(_workspace)]


def _opening(request: Request) -> int | None:
    """The opening of a project the request's page shows: its
    :data:`OPENING_HEADER`, an open id in plain digits (:func:`_url_index`), or
    None without one. ``invalid_input`` for any other text, or for two."""
    given = request.headers.getlist(OPENING_HEADER)
    if not given:
        return None
    try:
        if len(given) > 1:
            raise ValueError("given more than once")
        return int(_url_index(given[0]))  # type: ignore[call-overload]
    except ValueError as exc:
        raise OperationError(ErrorCode.INVALID_INPUT, f"{OPENING_HEADER}: {exc}") from exc


def _open_session(request: Request) -> Iterator[ProjectSession]:
    """The open project's session, for a route that reads or edits it: if the
    request names an opening (:func:`_opening`), only while that is the open
    one; ``project_changed`` otherwise. It stays in use until the route returns
    (:meth:`Workspace.using`), so the route runs, and answers, within the
    opening checked. A dependency runs before the route checks its path, query
    and body (only a body that is not JSON is refused first), so such a request
    is refused before anything is done. A sync generator, so its code on either
    side of the yield runs in the thread pool and never blocks the event loop,
    waiting for a reopen included. Of scope ``function`` (:data:`OpenSession`):
    its session is done with once the route returns, before its answer is
    sent."""
    with _workspace(request).using(_opening(request)) as session:
        yield session


OpenSession = Annotated[ProjectSession, Depends(_open_session, scope="function")]


def _checked_session(request: Request) -> ProjectSession:
    """The open project's session, checked as :func:`_open_session` checks it,
    and as early, but not taken in use: for a route that reads its request's
    body before it uses the session (:func:`import_image`), and then takes it in
    use itself (:meth:`Workspace.using`), checked again. So an upload naming an
    opening no longer open is refused before any of its bytes are read, and one
    under way holds up no reopen. A sync function, so it runs in the thread
    pool and never blocks the event loop."""
    return _workspace(request).current(_opening(request))


CheckedSession = Annotated[ProjectSession, Depends(_checked_session)]


def _any_session(request: Request) -> Iterator[ProjectSession | None]:
    """:func:`_open_session`, for a route that reads the open project if one is
    open and works with none (the diagnostic file's): None when none is open
    and the request names no opening (:meth:`Workspace.using_any`)."""
    with _workspace(request).using_any(_opening(request)) as session:
        yield session


AnySession = Annotated[ProjectSession | None, Depends(_any_session, scope="function")]


def _checked_any_session(request: Request) -> ProjectSession | None:
    """:func:`_checked_session`, for a route that works with no project open
    (:func:`write_diagnostics`): None when none is open and the request names
    no opening, as :func:`_any_session` gives it. Checked as early, but not
    taken in use: the route takes it in use itself, only while it needs it
    (:meth:`Workspace.using_any`), checked again. A sync function, so it runs
    in the thread pool and never blocks the event loop."""
    return _workspace(request).current_any(_opening(request))


def _no_other_opening(request: Request) -> int | None:
    """The opening the request names (:func:`_opening`), or None, checked as
    :func:`_checked_session` checks it, and as early, for a route that closes
    the open project to open another (:func:`accept_handoff`): a page that
    shows a project no longer open does not close the one open now. The route
    checks it again once no other switch can run. A sync function, so it runs
    in the thread pool."""
    opening = _opening(request)
    if opening is not None:
        _workspace(request).current(opening)
    return opening


NamedOpening = Annotated[int | None, Depends(_no_other_opening)]


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


def _size(size: BoxSize) -> dict[str, int]:
    return {"width": size.width, "height": size.height}


def _remeasured(
    remeasured: tuple[tuple[str, float, float], ...], largest: tuple[str, float] | None
) -> dict[str, Any]:
    """The other proteins' bands an edit re-measured, each as ``{band_id,
    net_before, net_after}``, and the largest change as ``{band_id, change}``
    (or None)."""
    return {
        "remeasured": [
            {"band_id": band_id, "net_before": before, "net_after": after}
            for band_id, before, after in remeasured
        ],
        "largest_change": None
        if largest is None
        else {"band_id": largest[0], "change": largest[1]},
    }


def _row_placement(placement: ops.RowPlacement) -> dict[str, Any]:
    """Every field of what a row box did
    (:class:`~proteia.core.operations.RowPlacement`), with lists for tuples, the
    box size as ``{width, height}``, each empty lane as ``{lane_index,
    reason, snr, expected_x}``, and the bands re-measured as
    :func:`_remeasured` gives them."""
    return {
        "band_ids": list(placement.band_ids),
        "box_size": _size(placement.box_size),
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
        **_remeasured(placement.remeasured, placement.largest_change),
        "unlocated_lanes": list(placement.unlocated_lanes),
    }


def _calibration_update(update: ops.CalibrationUpdate) -> dict[str, Any]:
    """What a calibration change did
    (:class:`~proteia.core.operations.CalibrationUpdate`): the point as stored
    with ``snapped`` (or None), the group's fit as JSON (or None), and lists
    for tuples."""
    return {
        "point": update.point,
        "fit": None if update.fit is None else update.fit.as_json(),
        "curves_changed": list(update.curves_changed),
        "dropped_undetected": [list(key) for key in update.dropped_undetected],
    }


def _padding_change(change: ops.PaddingChange) -> dict[str, Any]:
    """Every field of what a padding change did
    (:class:`~proteia.core.operations.PaddingChange`): the sizes as ``{width,
    height}``, the padding as ``box_padding {across, along}``, the net change as
    ``[smallest, largest]`` (or None), lists for tuples, and the bands
    re-measured as :func:`_remeasured` gives them."""
    padding, net_change = change.padding, change.net_change
    return {
        "box_size": _size(change.box_size),
        "fitted_size": _size(change.fitted_size),
        "box_padding": {"across": padding.across, "along": padding.along},
        "net_change": None if net_change is None else list(net_change),
        "edge_shifted": list(change.edge_shifted),
        "overlapping": list(change.overlapping),
        **_remeasured(change.remeasured, change.largest_change),
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


@router.get("/workspace")
def get_workspace(workspace: WorkspaceDep) -> dict[str, Any]:
    """Which project is open, without reading it: the projects root, and the
    open project's name and open id (null before one is open). A page checks it
    to know whether the project it shows is still the one open. And the
    hand-offs pending (:func:`_handoffs`), those an accept is importing too, as
    ``claimed``: that accept may yet be refused, and a page showing one keeps
    its choices until it is gone from the listing."""
    name, open_id = workspace.opened()
    return {
        "root": str(workspace.root),
        "open": name,
        "open_id": open_id,
        "handoffs": _handoffs(workspace.root, workspace.inbox.listing()),
    }


def _handoffs(root: Path, views: list[HandoffView]) -> list[dict[str, Any]]:
    """The hand-offs ``views`` as the page lists them, each with the name its
    project would take now (``suggested_name``, a hint: the accept names it
    again; null for a notice, or when no name is free). The projects root is
    listed once for all of them, and not at all without files."""
    names: list[str] = []
    if any(view.files for view in views):
        with contextlib.suppress(OSError):
            names = projects.names_in(root)
    return [_handoff(root, view, names) for view in views]


def _handoff(root: Path, view: HandoffView, names: list[str]) -> dict[str, Any]:
    suggested = None
    if view.files:
        with contextlib.suppress(projects.ProjectExistsError, projects.ProjectNameError):
            first = projects.name_from_file(view.files[0].name)
            suggested = projects.free_name(root, first, existing=names)
    return {
        "id": view.id,
        "kind": view.kind,
        "files": [
            {"file_id": file.file_id, "name": file.name, "size": file.size} for file in view.files
        ],
        "suggested_name": suggested,
        "refused": [dataclasses.asdict(entry) for entry in view.refused],
        "more_refused": view.more_refused,
        "more_may_arrive": view.more_may_arrive,
        "claimed": view.claimed,
    }


@router.post("/incoming", status_code=201)
async def receive_file(
    request: Request, workspace: WorkspaceDep, name: Annotated[str, Query(min_length=1)]
) -> dict[str, Any]:
    """Stage a file a launch hands to the app: the request body is its bytes,
    ``name`` its original name (percent-encoded in the URL), metadata only. It
    is stored in the private staging folder under a name the server makes
    (:meth:`~proteia.web.handoff.Inbox.begin_upload`), and waits there for an
    offer (``POST /api/handoffs``), which must come within
    :data:`~proteia.web.handoff.UPLOAD_EXPIRY_S`. Answers ``{file_id, name,
    size}``. The name is checked before any of the body is read
    (:func:`~proteia.web.handoff.check_name`), and so are the size the request
    declares and the limits (as ``GET /api/incoming/room`` checks them); an
    empty body is ``invalid_image``. One that cannot be stored is
    ``file_error``, with a message that names no path: the launch passes it on
    to the page. A refused upload leaves no file."""
    given = request.headers.get("content-length")
    declared = int(given) if given is not None and given.isdigit() else None
    _check_incoming(name, declared)
    inbox = workspace.inbox
    upload = await run_in_threadpool(inbox.begin_upload, name, declared)
    try:
        size = 0
        with upload.open() as out:
            pending = bytearray()
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise UploadTooLargeError(f"an image may have at most {MAX_UPLOAD_BYTES} bytes")
                pending += chunk
                if len(pending) >= _WRITE_BYTES:
                    inbox.make_room(upload, size)
                    await run_in_threadpool(out.write, bytes(pending))
                    pending.clear()
            if pending:
                inbox.make_room(upload, size)
                await run_in_threadpool(out.write, bytes(pending))
        if size == 0:
            raise OperationError(ErrorCode.INVALID_IMAGE, f"image {name!r} is empty")
        stored = inbox.upload_stored(upload, size)
    except BaseException as exc:
        inbox.upload_failed(upload)  # closed by now: Windows cannot delete an open file
        if isinstance(exc, OSError):  # the launch shows the page this: no path
            raise OSError(f"{name!r} could not be stored: {_reason(exc)}") from exc
        raise
    return {"file_id": stored.file_id, "name": stored.name, "size": stored.size}


@router.get("/incoming/room", status_code=204)
def incoming_room(
    workspace: WorkspaceDep,
    name: Annotated[str, Query(min_length=1)],
    size: Annotated[int, Query(ge=0)],
) -> Response:
    """Whether ``POST /api/incoming`` would take a file named ``name`` of
    ``size`` bytes now: 204 if so, else the refusal that upload would get
    before reading any of its body (the name, the size, ``stopping``,
    ``too_many_pending``). Nothing is held or stored
    (:meth:`~proteia.web.handoff.Inbox.check_room`). A launch asks before it
    sends a file's bytes. An upload refused before its body is read is still
    sent whole, for nothing: the server answers at once, then reads the rest
    and throws it away; and if the rest stops arriving for a while (the
    server's keep-alive time), it closes the connection, which the launch may
    find reset before it can read the answer."""
    _check_incoming(name, size)
    workspace.inbox.check_room(size)
    return Response(status_code=204)


def _check_incoming(name: str, size: int | None) -> None:
    """The checks of a file a launch hands over, before any of its bytes are
    read: its name (:func:`~proteia.web.handoff.check_name`), and its size, if
    known, against :data:`MAX_UPLOAD_BYTES`."""
    handoff.check_name(name)
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(f"an image may have at most {MAX_UPLOAD_BYTES} bytes")


@router.post("/handoffs", status_code=201)
def offer_handoff(body: OfferBody, workspace: WorkspaceDep) -> dict[str, Any]:
    """Hand off staged files, with the arguments the launch refused, as one
    hand-off or into one pending (:meth:`~proteia.web.handoff.Inbox.offer`).
    Answers ``{handoff_id, merged, files, refused}``: ``merged`` if the
    hand-off was pending already (a launch then opens no tab: one is open on
    it), and how many files and refused entries it holds now."""
    refused = [Refusal(entry.name, entry.code, entry.message) for entry in body.refused]
    offered = workspace.inbox.offer(body.files, refused)
    return dataclasses.asdict(offered)


@router.post("/handoffs/{handoff_id}/accept", status_code=201)
def accept_handoff(
    handoff_id: str, body: AcceptBody, opening: NamedOpening, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Import the hand-off into a new project and open it
    (:meth:`Workspace.accept`); answers as a create does, with ``handoff``:
    ``imported`` (``{file_id, name, image_id, membrane_id, new_membrane}``
    each), ``refused`` (``{file_id, name, code, message}`` each: the files not
    imported), ``launch_refused`` and ``more_refused`` (the arguments the
    launches refused), and ``notes``. Refused, changing nothing, for a page
    that shows another opening than the open one (``project_changed``)."""
    session, accepted = workspace.accept(
        handoff_id, body.name, _file_choices(body.files), opening=opening
    )
    with workspace.answering(session):
        return _answer(workspace, session, handoff=_accepted(accepted))


def _file_choices(files: list[AcceptFileBody]) -> list[FileChoice]:
    """The accept's files as choices: ``invalid_input`` for a kind or polarity
    the model does not know, or a membrane that is not an earlier file's."""
    choices = []
    for index, file in enumerate(files):
        try:
            kind, polarity = ImageKind(file.kind), Polarity(file.polarity)
        except ValueError as exc:
            raise OperationError(ErrorCode.INVALID_INPUT, f"files.{index}: {exc}") from exc
        membrane = None if file.membrane == "new" else file.membrane
        if membrane is not None and membrane >= index:
            raise OperationError(
                ErrorCode.INVALID_INPUT,
                f"files.{index}.membrane: {membrane} is not an earlier file's index",
            )
        choices.append(FileChoice(file.file_id, kind, polarity, membrane))
    return choices


def _accepted(accepted: Accepted) -> dict[str, Any]:
    return {
        "imported": [dataclasses.asdict(file) for file in accepted.imported],
        "refused": [dataclasses.asdict(file) for file in accepted.refused],
        "launch_refused": [dataclasses.asdict(entry) for entry in accepted.launch_refused],
        "more_refused": accepted.more_refused,
        "notes": list(accepted.notes),
    }


@router.post("/handoffs/{handoff_id}/discard", status_code=204)
def discard_handoff(handoff_id: str, body: DiscardBody, workspace: WorkspaceDep) -> Response:
    """Discard the hand-off as the page shows it: its files and its count of
    refused entries (``handoff_changed`` if it holds others, and nothing is
    deleted). Its staged copies are deleted; the originals are never touched."""
    workspace.inbox.discard(handoff_id, body.files, body.refused)
    return Response(status_code=204)


@router.post("/projects", status_code=201)
def create_project(body: NameBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.create(body.name)
    with workspace.answering(session):
        return _answer(workspace, session)


@router.post("/projects/sample", status_code=201)
def create_sample_project(workspace: WorkspaceDep) -> dict[str, Any]:
    """Create the sample project (:mod:`proteia.web.sample_project`), named
    ``Sample blot`` or the next free ``Sample blot (n)``, and open it. Answers as
    a create does, with ``sample`` (:func:`~proteia.web.sample_project.sample_payload`):
    the truth table's name in the project folder and each protein's row."""
    session = workspace.create_sample()
    with workspace.answering(session):
        return _answer(workspace, session, sample=sample_project.sample_payload(session.project))


@router.post("/projects/open")
def open_project(body: NameBody, workspace: WorkspaceDep) -> dict[str, Any]:
    session = workspace.open(body.name)
    with workspace.answering(session):
        return _answer(workspace, session)


@router.get("/project")
def get_project(session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, session)


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
def reveal_project(
    session: OpenSession, workspace: WorkspaceDep, body: RevealBody | None = None
) -> Response:
    """Show the project folder in the system file manager or, with ``folder``,
    one of its export folders, as ``POST /api/export`` answered it."""
    folder = session.folder if body is None else _export_folder(session.folder, body.folder)
    workspace.reveal(folder)
    shown = "its folder" if body is None else body.folder
    _log.info("in %r: showed %s in the file manager", session.folder.name, shown)
    return Response(status_code=204)


@router.post("/images", status_code=201)
async def import_image(
    request: Request,
    checked: CheckedSession,
    workspace: WorkspaceDep,
    name: Annotated[str, Query(min_length=1)],
    kind: str,
    polarity: str,
    membrane_id: str | None = None,
) -> dict[str, Any]:
    """The request body is the file's bytes; ``name`` is its original name
    (percent-encoded in the URL), kept only as metadata.

    The opening the request names is checked before any of its bytes are read
    (:func:`_checked_session`), but the session is taken in use only once the
    body is stored, which for a large file may take minutes: a reopen meanwhile
    does not wait for the upload (:meth:`Workspace.open`). Then it is checked
    again, and the import is made and answered within that opening
    (:meth:`Workspace.using`): an upload whose opening a reopen or a switch
    ended meanwhile is refused ``project_changed``, with nothing imported and
    its temporary file removed; one that names none is imported into the
    project open then."""
    opening = _opening(request)  # as _checked_session read it
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(f"an image may have at most {MAX_UPLOAD_BYTES} bytes")
    # Spooled to a temporary file on the project's drive (not the system drive),
    # never held in memory whole; import_image then copies it into images/.
    with tempfile.TemporaryFile(dir=checked.folder) as spool:
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

        def imported() -> dict[str, Any]:
            with workspace.using(opening) as session:
                image_id = ops.import_image(
                    session,
                    spool,
                    name,
                    kind=kind,
                    polarity=polarity,
                    membrane_id=membrane_id,
                    max_bytes=MAX_UPLOAD_BYTES,
                )
                return _answer(workspace, session, image_id=image_id)

        # Waiting for a reopen under way, importing and computing the results
        # take a while: off the event loop.
        return await run_in_threadpool(imported)


@router.delete("/images/{image_id}")
def remove_image(image_id: str, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    cascade = ops.remove_image(session, image_id)
    return _answer(workspace, session, **_cascade(cascade))


@router.put("/images/{image_id}/polarity")
def set_polarity(
    image_id: str, body: PolarityBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    ops.set_polarity(session, image_id, body.polarity)
    return _answer(workspace, session)


@router.get("/ladders")
def list_ladders() -> dict[str, Any]:
    """The ladder presets, as :mod:`proteia.core.ladders` holds them; no
    project is read."""
    return {
        "ladders": [
            {
                "key": preset.key,
                "product": preset.product,
                "catalog_numbers": list(preset.catalog_numbers),
                "system": preset.system,
                "kda": list(preset.kda),
                "reference": [
                    {"kda": band.kda, "colour": band.colour} for band in preset.reference
                ],
                "source": preset.source,
            }
            for preset in ladders.PRESETS
        ]
    }


@router.put("/images/{image_id}/marker")
def set_marker_image(
    image_id: str, body: MarkerBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Link the chemiluminescence image to its marker image, or unlink it with
    null (:func:`~proteia.core.operations.set_marker_image`)."""
    update = ops.set_marker_image(session, image_id, body.marker_image_id)
    return _answer(workspace, session, **_calibration_update(update))


@router.put("/membranes/{membrane_id}/calibration/ladder")
def set_ladder(
    membrane_id: str, body: LadderBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Choose the membrane's ladder (:func:`~proteia.core.operations.set_ladder`)."""
    ops.set_ladder(session, membrane_id, body.ladder, kda=body.kda)
    return _answer(workspace, session)


@router.post("/images/{image_id}/calibration/{side}/points", status_code=201)
def add_calibration_point(
    image_id: str, side: str, body: PointBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Mark a calibration point on the ladder ``side`` of the image's register
    group (:func:`~proteia.core.operations.add_calibration_point`)."""
    update = ops.add_calibration_point(
        session, image_id, body.y, body.mw, body.source, x=body.x, side=_side(side), snap=body.snap
    )
    return _answer(workspace, session, **_calibration_update(update))


@router.patch("/images/{image_id}/calibration/{side}/points/{mw}")
def edit_calibration_point(
    image_id: str,
    side: str,
    mw: UrlKda,
    body: PointEditBody,
    session: OpenSession,
    workspace: WorkspaceDep,
) -> dict[str, Any]:
    """Move or relabel the point at ``mw``, in one change
    (:func:`~proteia.core.operations.edit_calibration_point`)."""
    given = body.model_dump(exclude_unset=True)
    update = ops.edit_calibration_point(
        session,
        image_id,
        mw,
        side=_side(side),
        y=given.get("y", ops.KEEP),
        new_mw=given.get("mw", ops.KEEP),
        snap=body.snap,
    )
    return _answer(workspace, session, **_calibration_update(update))


@router.delete("/images/{image_id}/calibration/{side}/points/{mw}")
def remove_calibration_point(
    image_id: str, side: str, mw: UrlKda, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Remove the point at ``mw`` (:func:`~proteia.core.operations.remove_calibration_point`)."""
    update = ops.remove_calibration_point(session, image_id, mw, side=_side(side))
    return _answer(workspace, session, **_calibration_update(update))


@router.delete("/images/{image_id}/calibration")
def clear_calibration(
    image_id: str,
    session: OpenSession,
    workspace: WorkspaceDep,
    side: Literal["left", "right"] | None = None,
) -> dict[str, Any]:
    """Remove every point of the image's register group, or those of one
    ``side`` (:func:`~proteia.core.operations.clear_calibration`); none there is
    a no-op."""
    chosen = None if side is None else LadderSide(side)
    update = ops.clear_calibration(session, image_id, side=chosen)
    return _answer(workspace, session, **_calibration_update(update))


@router.post("/images/{image_id}/ladder-proposal")
def propose_ladder(image_id: str, body: LadderProposalBody, session: OpenSession) -> dict[str, Any]:
    """Find the ladder clicked at ``x`` and propose its labels
    (:func:`~proteia.core.operations.propose_ladder`); changes nothing."""
    proposal = ops.propose_ladder(session, image_id, body.x, body.side)
    return {"proposal": None if proposal is None else ops.proposal_json(proposal)}


@router.post("/images/{image_id}/ladder-snap")
def snap_ladder(image_id: str, body: LadderSnapBody, session: OpenSession) -> dict[str, Any]:
    """Snap each tick of a ruler to its band
    (:func:`~proteia.core.operations.snap_ladder`); changes nothing."""
    snapped = ops.snap_ladder(session, image_id, body.x, body.ys)
    return {"points": [{"y": y, "snapped": moved} for y, moved in snapped]}


@router.put("/images/{image_id}/calibration/{side}/ladder")
def set_ladder_points(
    image_id: str,
    side: str,
    body: LadderPointsBody,
    session: OpenSession,
    workspace: WorkspaceDep,
) -> dict[str, Any]:
    """Apply a ruler to the ladder ``side`` of the image's register group, in
    one step (:func:`~proteia.core.operations.set_ladder_points`)."""
    update = ops.set_ladder_points(
        session,
        image_id,
        _side(side),
        [(point.y, point.mw) for point in body.points],
        x=body.x,
        found_at=body.found_at,
    )
    return _answer(
        workspace,
        session,
        **_calibration_update(update),
        points=list(update.points),
        sides_swapped=update.sides_swapped,
    )


@router.get("/images/{image_id}/preview")
def image_preview(
    image_id: str,
    session: OpenSession,
    workspace: WorkspaceDep,
    colour: Literal["original"] | None = None,
) -> Response:
    """The image as the view draws it, a PNG of the image's own size: its gray
    analysis array, which the nets are measured on, or with ``colour=original``
    its stored file in the file's own colours, for display only (the gray one
    for a file without colour to show: ``colour`` in the project state says
    which have it). Either is read after the stored file's SHA-256 is checked:
    ``image_file_changed`` or ``unreadable_image`` (422) otherwise."""
    data = workspace.preview(session, image_id, original=colour == "original")
    return Response(data, media_type="image/png")


@router.get("/charts/{key}.svg", dependencies=[Depends(_open_session, scope="function")])
def chart(key: str, workspace: WorkspaceDep) -> Response:
    """A chart of an answer, at its ``chart_url``; 404 ``unknown_id`` for a key
    not given in the open project's answers, or no longer kept."""
    return Response(workspace.chart(key), media_type="image/svg+xml")


@router.put("/lanes")
def set_lanes(body: LanesBody, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
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
def set_reference_condition(
    body: ReferenceBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    ops.set_reference_condition(session, body.condition)
    return _answer(workspace, session)


@router.post("/proteins", status_code=201)
def add_protein(body: ProteinBody, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
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
        mw_tolerance=body.mw_tolerance,
    )
    return _answer(workspace, session, protein_id=protein_id)


@router.patch("/proteins/{protein_id}")
def edit_protein(
    protein_id: str, body: ProteinEditBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    # Only the fields the request set: the operation keeps the others (KEEP).
    ops.edit_protein(session, protein_id, **body.model_dump(exclude_unset=True))
    return _answer(workspace, session)


@router.delete("/proteins/{protein_id}")
def remove_protein(
    protein_id: str, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    cascade = ops.remove_protein(session, protein_id)
    return _answer(workspace, session, **_cascade(cascade))


@router.put("/proteins/{protein_id}/box-size")
def set_box_size(
    protein_id: str, body: BoxSizeBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Set the protein's fitted size: every box becomes it plus the protein's
    padding on each side (:func:`~proteia.core.operations.set_box_size`)."""
    ops.set_box_size(session, protein_id, BoxSize(width=body.width, height=body.height))
    return _answer(workspace, session)


@router.put("/proteins/{protein_id}/box-padding")
def set_box_padding(
    protein_id: str, body: BoxPaddingBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Set how far every box of the protein extends beyond its fitted size, on
    each side (:func:`~proteia.core.operations.set_box_padding`); answers what
    it did (:func:`_padding_change`). The same padding, or an empty body, is a
    no-op."""
    # Only the fields the request set: the operation keeps the others (KEEP).
    change = ops.set_box_padding(session, protein_id, **body.model_dump(exclude_unset=True))
    return _answer(workspace, session, **_padding_change(change))


@router.delete("/proteins/{protein_id}/undetected/{lane_index}")
def remove_undetected(
    protein_id: str,
    lane_index: UrlIndex,
    session: OpenSession,
    workspace: WorkspaceDep,
    band_index: UrlIndex = 0,
) -> dict[str, Any]:
    """Remove the protein's not-detected record in the lane, for its first band
    unless ``band_index`` names a later one; no record there is a no-op."""
    ops.remove_undetected(session, protein_id, lane_index, band_index=band_index)
    return _answer(workspace, session)


@router.post("/boxes", status_code=201)
def place_box(body: PlaceBody, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    band_id = ops.place_box(
        session, body.protein_id, body.x, body.y, lane_index=body.lane_index, grow=body.grow
    )
    return _answer(workspace, session, band_id=band_id)


@router.post("/boxes/row", status_code=201)
def detect_row_boxes(
    body: RowBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Box the protein's first band in every declared lane from the row box
    dragged over its row (:func:`~proteia.core.operations.detect_row_boxes`),
    in image pixels. Answers what the row did in each lane
    (:func:`_row_placement`); the same drag again changes nothing."""
    placement = ops.detect_row_boxes(session, body.protein_id, body.rect)
    return _answer(workspace, session, **_row_placement(placement))


@router.put("/boxes/{band_id}")
def move_box(
    band_id: str, body: MoveBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    ops.move_box(session, band_id, body.rect)
    return _answer(workspace, session)


@router.delete("/boxes/{band_id}")
def remove_box(band_id: str, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    ops.remove_box(session, band_id)
    return _answer(workspace, session)


@router.put("/boxes/{band_id}/lane")
def set_box_lane(
    band_id: str, body: LaneIndexBody, session: OpenSession, workspace: WorkspaceDep
) -> dict[str, Any]:
    ops.set_box_lane(session, band_id, body.lane_index)
    return _answer(workspace, session)


@router.delete("/proteins/{protein_id}/boxes")
def clear_boxes(protein_id: str, session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    """Remove every box and not-detected record of the protein; it keeps its box
    size. A protein with neither is a no-op."""
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
def requantify(session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    """Switch the project to the local background and re-quantify every band,
    or, on the local background already, re-quantify the images whose bands
    were never assessed for over-exposure (the state's ``unassessed_images``;
    :func:`~proteia.core.operations.requantify`); answers the images
    re-quantified (``images``; empty for a no-op: nothing to do)."""
    images = ops.requantify(session)
    return _answer(workspace, session, images=list(images))


@router.post("/export", status_code=201)
def export_bundle(
    session: OpenSession, workspace: WorkspaceDep, body: ExportBody | None = None
) -> dict[str, Any]:
    """Write the results into a new export folder
    (:func:`~proteia.core.operations.export_bundle`), computed with the settings
    they are shown with, in the chart ``formats`` asked for (by default
    :data:`~proteia.core.export.DEFAULT_CHART_FORMATS`). Answers the folder,
    relative to the project folder (``exports/<name>``, for ``POST
    /api/project/reveal``), and the files in it, by name, in the order written.
    Refusals are 422 with the operation's codes: ``no_lanes``,
    ``image_file_changed``, ``path_too_long`` and ``invalid_input``."""
    formats = DEFAULT_CHART_FORMATS if body is None or body.formats is None else body.formats
    bundle = ops.export_bundle(session, formats=formats, **dataclasses.asdict(workspace.settings))
    return _answer(
        workspace,
        session,
        folder=bundle.folder.relative_to(session.folder).as_posix(),
        files=list(bundle.files),
    )


def _listed(items: Sequence[diagnostics.Item]) -> list[dict[str, Any]]:
    return [{"name": item.name, "size": item.size} for item in items]


@router.get("/diagnostics")
def list_diagnostics(session: AnySession, workspace: WorkspaceDep) -> dict[str, Any]:
    """What a diagnostic file written now would hold
    (:func:`~proteia.web.diagnostics.plan`), for which opening, and the digest
    of its project files: the page shows it before the file is written, and
    names the opening and the digest when it asks."""
    plan = diagnostics.plan(workspace.state_folder(), session)
    return {
        "project": plan.project,
        "open_id": workspace.opening_of(session),
        "saved": plan.saved,
        "files": _listed(plan.items),
        "images": _listed(plan.images),
        "left_out": [
            {"name": item.name, "size": item.size, "reason": item.reason} for item in plan.left_out
        ],
        "digest": plan.digest(),
        "added": [diagnostics.README_FILE, diagnostics.MANIFEST_FILE],
        "kept": diagnostics.KEPT,
    }


@router.post("/diagnostics", status_code=201, dependencies=[Depends(_checked_any_session)])
def write_diagnostics(
    request: Request, body: DiagnosticsBody, workspace: WorkspaceDep
) -> dict[str, Any]:
    """Write a diagnostic file (:func:`~proteia.web.diagnostics.write`) in the
    diagnostics folder of the state folder, with the project's image files if
    ``images``; only for the opening the page listed (``open_id``), and only
    while the project's files are those it listed (``digest``), so the file
    holds no project, and no project file, the page did not show.

    The opening the request names is checked as early as every other route's
    (:func:`_checked_any_session`), but the session is in use only while the
    file is planned (:meth:`Workspace.using_any`): the opening and the files
    are checked then, and ``project.json`` read, within one opening. The plan
    holds what the write needs (``project.json``'s bytes, the other files'
    paths), so the file is written once the session is released, which with
    the images may take minutes: a reopen meanwhile does not wait for it
    (:meth:`Workspace.open`). A file the plan names that is removed meanwhile
    is left out, and the manifest says why."""
    opening = _opening(request)  # as _checked_any_session read it
    with workspace.using_any(opening) as session:
        open_id = workspace.opening_of(session)
        if body.open_id != open_id:
            if session is None or open_id is None:
                raise NoProjectError("the page listed a project, but none is open now")
            raise ProjectChangedError(body.open_id, session.folder.name, open_id)
        state = workspace.state_folder()
        plan = diagnostics.plan(state, session)
        if plan.digest() != body.digest:
            raise FilesChangedError(
                "the project's files are not those listed (an export made or an image imported"
                " or removed since): list them again"
            )
    written = diagnostics.write(
        plan, state / diagnostics.DIAGNOSTICS_DIR, images=body.images, moment=workspace.clock()
    )
    project = "no project" if plan.project is None else repr(plan.project)
    _log.info(
        "wrote the diagnostic file %r: %d files, %d bytes, %d left out; %s%s",
        written.path.name,
        written.files,
        written.size,
        written.left_out,
        project,
        ", with its images" if body.images and plan.project is not None else "",
    )
    return {
        "name": written.path.name,
        "path": str(written.path),
        "size": written.size,
        "files": written.files,
        "left_out": written.left_out,
    }


@router.post("/diagnostics/reveal", status_code=204)
def reveal_diagnostics(workspace: WorkspaceDep) -> Response:
    """Show the folder diagnostic files are written in, in the system file
    manager (made if need be)."""
    folder = workspace.state_folder() / diagnostics.DIAGNOSTICS_DIR
    diagnostics.make_folder(folder)
    workspace.reveal(folder)
    _log.info("showed the diagnostics folder in the file manager")
    return Response(status_code=204)


@router.get("/notices")
def get_notices(workspace: WorkspaceDep) -> dict[str, Any]:
    """The notices the page shows once per user: ``cloud_sync``, the service
    that uploads the projects folder as ``{service}`` (no path), while it lies
    in a folder a sync service uploads (:meth:`Workspace.synced_folder`) and
    the notice is not dismissed (:mod:`proteia.web.cloudsync`); else null. Null
    too without a state folder: its dismissal could not be remembered, and it
    would be shown at every start."""
    synced = workspace.synced_folder()
    state = workspace.state
    shown = (
        synced is not None
        and state is not None
        and cloudsync.CLOUD_SYNC not in cloudsync.dismissed(state)
    )
    return {"cloud_sync": {"service": synced.service} if shown else None}


@router.post("/notices/cloud_sync/dismiss", status_code=204)
def dismiss_cloud_sync(workspace: WorkspaceDep) -> Response:
    """Dismiss the ``cloud_sync`` notice for good: recorded in the per-user
    state folder (:func:`~proteia.web.cloudsync.dismiss`), never in a project.
    A folder that cannot be written is answered as a file error naming no path."""
    try:
        cloudsync.dismiss(workspace.state_folder(), cloudsync.CLOUD_SYNC)
    except OSError as exc:  # the page shows the message: no path
        raise OSError(f"the notice could not be dismissed: {_reason(exc)}") from exc
    _log.info("the notice that the projects folder is synced to the cloud was dismissed for good")
    return Response(status_code=204)


@router.post("/undo")
def undo(session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
    return _answer(workspace, session, **_restored(ops.undo(session)))


@router.post("/redo")
def redo(session: OpenSession, workspace: WorkspaceDep) -> dict[str, Any]:
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
        NoStateFolderError: lambda e: _error(409, "no_state_folder", str(e)),
        FilesChangedError: lambda e: _error(409, "files_changed", str(e)),
        ProjectChangedError: lambda e: _error(
            409, "project_changed", str(e), detail={"open": e.open, "open_id": e.open_id}
        ),
        UnsavedChangesError: lambda e: _error(
            409,
            "unsaved_changes",
            str(e),
            detail=None if e.created is None else {"created": e.created},
        ),
        UploadTooLargeError: lambda e: _error(413, "image_too_large", str(e)),
        NothingImportedError: lambda e: _error(
            422,
            "nothing_imported",
            str(e),
            detail={"refused": [dataclasses.asdict(file) for file in e.refused]},
        ),
        handoff.StoppingError: lambda e: _error(409, "stopping", str(e)),
        handoff.TooManyPendingError: lambda e: _error(409, "too_many_pending", str(e)),
        handoff.FileClaimedError: lambda e: _error(409, "file_claimed", str(e)),
        handoff.HandoffNotFoundError: lambda e: _error(404, "handoff_not_found", str(e)),
        handoff.HandoffClaimedError: lambda e: _error(409, "handoff_claimed", str(e)),
        handoff.HandoffChangedError: lambda e: _error(
            409, "handoff_changed", str(e), detail=_handoffs(workspace.root, [e.view])[0]
        ),
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
    app.add_exception_handler(HTTPException, _http_refusal(logs.Throttle()))


def _handler(answer: Callable[[Exception], JSONResponse]):
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        response = answer(exc)
        _log_refusal(request, response, exc)
        return response

    return handle


def _log_refusal(request: Request, response: JSONResponse, exc: Exception) -> None:
    """Log an error answer: the request's method and path, the status, code and
    message, and the ids and ``detail`` when it has them; a server error (a file
    that cannot be written) as an error, with its stack trace."""
    body = json.loads(response.body)
    parts = [f"{response.status_code} {body['code']}: {body['message']}"]
    if body["ids"]:
        parts.append(f"ids {', '.join(body['ids'])}")
    if body.get("detail") is not None:
        parts.append(f"detail {json.dumps(body['detail'], ensure_ascii=False)}")
    method, path = request.method, logs.shorten(request.url.path)
    if response.status_code >= 500:
        # The file only: the console never showed a refusal.
        _log.error(
            "refused %s %s: %s",
            method,
            path,
            "; ".join(parts),
            exc_info=exc,
            extra=logs.FILE_ONLY,
        )
    else:
        _log.info("refused %s %s: %s", method, path, "; ".join(parts))


def _http_refusal(refusals: logs.Throttle):
    """The handler of a request no route takes (an unknown path, a method a
    route does not take): answered as FastAPI answers it, and logged, at most
    as often as ``refusals`` admits."""

    async def handle(request: Request, exc: HTTPException) -> Response:
        admitted, left_out = refusals.admit()
        if left_out:
            _log.info(
                "refused %d more requests no route takes, not logged one by one"
                " (more than %d a minute)",
                left_out,
                refusals.limit,
            )
        if admitted:
            method, path = request.method, logs.shorten(request.url.path)
            _log.info("refused %s %s: %s %s", method, path, exc.status_code, exc.detail)
        return await http_exception_handler(request, exc)

    return handle


def _describe(error: dict[str, Any]) -> str:
    where = ".".join(str(part) for part in error.get("loc", ()))
    return f"{where}: {error.get('msg', 'invalid')}"
