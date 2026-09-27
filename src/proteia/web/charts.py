# SPDX-License-Identifier: Apache-2.0
"""The charts the browser shows, drawn on the server (ADR 0002).

Every chart in an answer (:func:`~proteia.web.results_view.results_payload`) is
registered here and answered as a URL named by its content, a hash of its
:class:`~proteia.core.plotspec.PlotSpec` (:func:`chart_key`). A chart that did
not change keeps its URL, so the browser does not fetch it again and the server
does not draw it again. A chart is drawn (:func:`~proteia.viz.render_svg`) when
its URL is first fetched, once for all the fetches that come while it is drawn
(a page reloaded meanwhile, a second tab), and the drawing is kept.

The store keeps the specs registered or fetched last and the drawings fetched
last, a few of each. A key it does not keep, like one it never gave, is unknown
(404 ``unknown_id``); the browser then reads the project again, which registers
its charts anew. Creating or opening a project empties the store
(:meth:`ChartStore.reset`), so no chart of another project is served, and an
answer about an earlier opening that finishes late registers nothing.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import Final

from proteia.core.model import UnknownIdError
from proteia.core.plotspec import PlotSpec
from proteia.viz import render_svg

SPECS_KEPT: Final = 256
DRAWINGS_KEPT: Final = 64
KEY_PATTERN: Final = re.compile(r"[0-9a-f]{32}")
# Part of every key: change it when the same spec is drawn differently, so a
# browser never keeps a drawing under the key of a spec drawn anew.
_KEY_VERSION: Final = b"render-v3\n"  # v2: tests named as readers name them; v3: p < 0.0001
_UNKNOWN: Final = "no chart has this key; read the project again for its charts' URLs"


def chart_key(spec: PlotSpec) -> str:
    """The key of the chart drawn from ``spec``: the first 32 hex digits of a
    SHA-256 of the drawing's version and the spec as JSON."""
    return hashlib.sha256(_KEY_VERSION + spec.model_dump_json().encode()).hexdigest()[:32]


def chart_url(spec: PlotSpec) -> str:
    """The URL the chart drawn from ``spec`` is served at."""
    return f"/api/charts/{chart_key(spec)}.svg"


class _Underway:
    """A chart being drawn, for the fetches of it that come meanwhile to wait on:
    ``drawing`` is set, or left None if the drawing failed, before ``done`` is."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.drawing: bytes | None = None


class ChartStore:
    """The specs of the open project's charts by key, and the drawings of those
    fetched last, both least recently used first and behind one lock. A chart is
    drawn outside the lock (a drawing takes tens of milliseconds), once for all
    the fetches of it that come while it is drawn."""

    def __init__(self, *, specs_kept: int = SPECS_KEPT, drawings_kept: int = DRAWINGS_KEPT) -> None:
        self._specs_kept = specs_kept
        self._drawings_kept = drawings_kept
        self._lock = threading.Lock()
        self._open_id = 0  # the opening whose charts are kept
        self._specs: OrderedDict[str, PlotSpec] = OrderedDict()
        self._drawings: OrderedDict[str, bytes] = OrderedDict()  # keys: some of _specs'
        self._underway: dict[str, _Underway] = {}  # the charts being drawn now

    def reset(self, open_id: int) -> None:
        """Forget every chart: the project opened as ``open_id`` is the open one.
        A fetch from now on does not wait for a drawing begun before."""
        with self._lock:
            self._open_id = open_id
            self._specs.clear()
            self._drawings.clear()
            self._underway.clear()

    def register(self, spec: PlotSpec, *, open_id: int) -> str:
        """Keep ``spec`` for the chart URL answered about the opening ``open_id``,
        and give that URL. The spec is kept only while ``open_id`` is the open
        one: an answer about an earlier opening gives the URL, which is unknown."""
        key = chart_key(spec)
        with self._lock:
            if open_id == self._open_id:
                self._specs[key] = spec
                self._specs.move_to_end(key)
                while len(self._specs) > self._specs_kept:
                    evicted, _ = self._specs.popitem(last=False)
                    self._drawings.pop(evicted, None)
        return f"/api/charts/{key}.svg"

    def svg(self, key: str) -> bytes:
        """The chart ``key`` names, as SVG: drawn now if it was not kept drawn,
        or the drawing under way if another fetch is drawing it.
        :class:`~proteia.core.model.UnknownIdError` for a key the store does not
        keep, or that no key could be; that is told before the store is read."""
        if not KEY_PATTERN.fullmatch(key):
            raise UnknownIdError(_UNKNOWN)
        while True:
            with self._lock:
                spec = self._specs.get(key)
                if spec is None:
                    raise UnknownIdError(_UNKNOWN)
                self._specs.move_to_end(key)
                drawing = self._drawings.get(key)
                if drawing is not None:
                    self._drawings.move_to_end(key)
                    return drawing
                underway = self._underway.get(key)
                drawer = underway is None
                if underway is None:
                    underway = self._underway[key] = _Underway()
                open_id = self._open_id
            if drawer:
                return self._draw(key, spec, underway, open_id)
            underway.done.wait()
            if underway.drawing is not None:
                return underway.drawing
            # That drawing failed: this fetch draws the chart for itself.

    def _draw(self, key: str, spec: PlotSpec, underway: _Underway, open_id: int) -> bytes:
        """Draw ``spec``, outside the lock, for this fetch of ``key`` and those
        that wait on ``underway``; keep the drawing if ``open_id`` is still the
        open one."""
        try:
            underway.drawing = render_svg(spec)
            return underway.drawing
        finally:
            with self._lock:
                if self._underway.get(key) is underway:
                    del self._underway[key]
                # Not kept if the store was reset, or the spec went, while it was drawn.
                kept = open_id == self._open_id and key in self._specs
                if underway.drawing is not None and kept:
                    self._drawings[key] = underway.drawing
                    while len(self._drawings) > self._drawings_kept:
                        self._drawings.popitem(last=False)
            underway.done.set()
