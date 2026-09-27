# SPDX-License-Identifier: Apache-2.0
"""The HTTP routes over the project operations (#50): projects in the app-managed
root, image upload and previews, lanes and the reference, proteins, boxes and
not-detected records. Requests go to a real server on a loopback socket; every
edit answers with the stored project state and its live results (#52), in strict
JSON."""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np
import pytest
import tifffile
from fastapi.routing import APIRoute
from PIL import Image
from starlette.routing import Mount

from conftest import (
    MEMBRANE_LEVEL,
    FakeClock,
    make_project,
    make_project_with_clashing_names,
    make_project_with_undetected,
    synthetic_blot,
    write_image_files,
    write_tiff,
)
from proteia import samples
from proteia.core import session as session_module
from proteia.core import storage
from proteia.core.analyze import ReduceMethod
from proteia.core.model import (
    BoxSize,
    Project,
    ProposalSource,
    Region,
    UndetectedBand,
    UndetectedReason,
    apply_change,
)
from proteia.core.plotspec import ErrorType, PlotSpec
from proteia.core.results import compute_results
from proteia.viz import render_svg
from proteia.web import api, charts, handoff, launch, sample_project, server
from proteia.web.results_view import results_payload
from proteia.web.state import project_state, revision
from rowcases import RowCase, adversarial
from test_operations import v1_folder

H, W = 60, 400
ROW = 30
LANE_X = [50, 120, 190, 260, 330]
NAME = "β-actin 10 µM.tif"  # beta-actin 10 micro-molar


def _not_json(constant: str) -> float:
    raise ValueError(f"{constant} is not JSON")


class Client:
    """JSON requests with this launch's token, over http.client; ``workspace`` is
    the server's, for state no route writes yet."""

    def __init__(
        self, port: int, token: str, root: Path, revealed: list[Path], workspace: api.Workspace
    ) -> None:
        self.port, self.token, self.root, self.revealed = port, token, root, revealed
        self.workspace = workspace

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        """``headers`` are sent too, such as the opening the request names."""
        headers = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        data = raw
        if raw is not None:
            headers["Content-Type"] = "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            kind = response.getheader("content-type", "")
        finally:
            conn.close()
        if not kind.startswith("application/json"):
            return response.status, (kind, payload)
        return response.status, json.loads(payload, parse_constant=_not_json)  # strict JSON

    def ok(self, method: str, path: str, body: Any = None, **kw: Any) -> Any:
        status, payload = self.call(method, path, body, **kw)
        assert status in (200, 201), (status, payload)
        return payload

    def refused(self, method: str, path: str, body: Any = None, **kw: Any) -> tuple[int, str, list]:
        status, payload = self.call(method, path, body, **kw)
        assert status >= 400, payload
        return status, payload["code"], payload["ids"]


@pytest.fixture
def client(tmp_path):
    revealed: list[Path] = []
    root = tmp_path / "projects"
    workspace = api.Workspace(root, reveal=revealed.append, clock=FakeClock())
    instance = launch.start(folder=tmp_path / "state", opener=lambda url: True, workspace=workspace)
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(0.01)
    yield Client(instance.port, instance.token, root, revealed, workspace)
    instance.stop()
    thread.join(10)


def blot_bytes(tmp_path: Path, *, clipped_lane: int | None = None) -> bytes:
    """A 16-bit blot with a band under each lane; one saturated if asked."""
    bands = [
        (x, ROW, 5.0, 3.0, MEMBRANE_LEVEL + 5000 if lane == clipped_lane else 30000.0)
        for lane, x in enumerate(LANE_X)
    ]
    return write_tiff(tmp_path / "source.tif", synthetic_blot((H, W), bands)).read_bytes()


def upload(client: Client, data: bytes, name: str = NAME, **query: str) -> tuple[int, Any]:
    params = {"kind": "chemiluminescence", "polarity": "dark_on_light", **query}
    url = f"/api/images?name={quote(name, safe='')}" + "".join(
        f"&{key}={quote(value, safe='')}" for key, value in params.items()
    )
    return client.call("POST", url, raw=data)


def ready(client: Client, tmp_path: Path, **blot: Any) -> tuple[str, str]:
    """A project with the blot, five lanes and one target; (image id, protein id)."""
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, blot_bytes(tmp_path, **blot))
    assert status == 201, answer
    image_id = answer["image_id"]
    lanes = [{"condition": f"c{i}"} for i in range(len(LANE_X))]
    client.ok("PUT", "/api/lanes", {"lanes": lanes})
    answer = client.ok(
        "POST", "/api/proteins", {"name": "β-actin", "role": "target", "image_id": image_id}
    )
    return image_id, answer["protein_id"]


def bands(answer: dict) -> dict[str, dict]:
    return {
        band["id"]: band for protein in answer["project"]["proteins"] for band in protein["bands"]
    }


# --- Projects ---


def test_projects_are_created_listed_and_opened_by_name(client):
    listing = client.ok("GET", "/api/projects")
    assert listing == {"root": str(client.root), "open": None, "projects": []}
    assert client.refused("GET", "/api/project")[:2] == (409, "no_project")

    answer = client.ok("POST", "/api/projects", {"name": "  Blot µ  1 "})
    assert answer["project"]["name"] == "Blot µ 1"  # stored as cleaned text
    assert (client.root / "Blot µ 1" / storage.PROJECT_FILE).is_file()
    assert client.refused("POST", "/api/projects", {"name": "BLOT µ 1"})[:2] == (
        409,
        "project_exists",
    )
    client.ok("POST", "/api/projects", {"name": "Second"})
    listing = client.ok("GET", "/api/projects")
    assert listing["open"] == "Second"
    assert sorted(p["name"] for p in listing["projects"]) == ["Blot µ 1", "Second"]

    answer = client.ok("POST", "/api/projects/open", {"name": "blot µ 1"})
    assert answer["project"]["name"] == "Blot µ 1"  # the folder's own spelling
    assert client.refused("POST", "/api/projects/open", {"name": "Missing"})[:2] == (
        404,
        "project_not_found",
    )
    assert client.call("POST", "/api/project/reveal")[0] == 204
    assert client.revealed == [client.root / "Blot µ 1"]


@pytest.mark.parametrize(
    "name", ["a/b", "a\\b", "..", ".", "CON", "lpt1.txt", "ends.", "x" * 101, "  ", "a\u0007b"]
)
def test_a_project_name_that_is_not_one_plain_folder_name_is_refused(client, name):
    assert client.refused("POST", "/api/projects", {"name": name})[:2] == (
        422,
        "invalid_project_name",
    )
    assert not client.root.exists() or list(client.root.iterdir()) == []


# --- Images ---


def test_an_upload_is_stored_under_its_own_name_and_previewed(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, blot_bytes(tmp_path))
    assert status == 201
    (image,) = answer["project"]["images"]
    assert image["id"] == answer["image_id"]
    assert (image["original_name"], image["width"], image["height"]) == (NAME, W, H)
    assert image["bit_depth"] == 16
    stored = sorted(p.name for p in (client.root / "Blot" / storage.IMAGES_DIR).iterdir())
    assert stored == [f"{image['id']}.tif"]  # a name the server chose

    status, (kind, png) = client.call("GET", f"/api/images/{image['id']}/preview")
    assert status == 200 and kind == "image/png"
    pixels = np.asarray(Image.open(io.BytesIO(png)))
    assert pixels.shape == (H, W) and pixels.dtype == np.uint8


def test_a_16_bit_preview_is_neither_blank_nor_saturated(client, tmp_path):
    # Levels from about 20,000 to 50,000: all above 255, so a naive 8-bit cast
    # would be blank or saturated.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, answer = upload(client, blot_bytes(tmp_path))
    _, (_, png) = client.call("GET", f"/api/images/{answer['image_id']}/preview")
    pixels = np.asarray(Image.open(io.BytesIO(png))).astype(int)
    assert pixels.min() == 0 and pixels.max() == 255
    assert (pixels == 255).mean() < 0.99 and (pixels == 0).mean() < 0.5
    assert len(np.unique(pixels)) > 50
    band = pixels[ROW, LANE_X[0]]
    assert band < pixels[5, LANE_X[0]]  # the band is darker than the membrane above it


def test_a_refused_upload_leaves_no_file(client, tmp_path, monkeypatch):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    folder = client.root / "Blot"
    before = sorted(p.name for p in folder.rglob("*"))
    data = blot_bytes(tmp_path)
    assert upload(client, data, name="notes.txt")[1]["code"] == "unsupported_image_type"
    status, answer = upload(client, data, name="../escape.tif")
    assert status == 422 and answer["code"] in ("invalid_input", "invalid_image")
    assert upload(client, data, polarity="sideways")[1]["code"] == "invalid_input"
    assert upload(client, b"not a tiff", name="x.tif")[1]["code"] == "unreadable_image"
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1000)
    status, answer = upload(client, data)
    assert (status, answer["code"]) == (413, "image_too_large")
    assert sorted(p.name for p in folder.rglob("*")) == before
    assert not (client.root / "escape.tif").exists()


def test_an_import_joins_the_membrane_it_names(client, tmp_path):
    # The page's "Membrane" choice: a new membrane, or the one an earlier image is on.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    data = blot_bytes(tmp_path)
    _, first = upload(client, data)
    (image,) = first["project"]["images"]
    membrane = image["membrane_id"]
    status, reprobe = upload(client, data, name="reprobe α 10 µM.tif", membrane_id=membrane)
    assert status == 201, reprobe
    _, other = upload(client, data, name="other.tif")  # none named: a membrane of its own
    images = {i["id"]: i["membrane_id"] for i in other["project"]["images"]}
    assert images[reprobe["image_id"]] == membrane
    assert images[other["image_id"]] != membrane
    project = storage.load_project(client.root / "Blot")
    assert [[i.id for i in m.images] for m in project.batch.membranes] == [
        [image["id"], reprobe["image_id"]],
        [other["image_id"]],
    ]
    joined = project.log[-2]
    assert (joined.action, joined.params["membrane_id"], joined.params["new_membrane"]) == (
        "import_image",
        membrane,
        False,
    )

    # An unknown membrane is refused before the file is stored: 404, nothing changes.
    before = client.ok("GET", "/api/project")
    stored = sorted(p.name for p in (client.root / "Blot").rglob("*"))
    status, answer = upload(client, data, name="lost.tif", membrane_id="mem-99")
    assert (status, answer["code"]) == (404, "unknown_id")
    assert client.ok("GET", "/api/project") == before
    assert sorted(p.name for p in (client.root / "Blot").rglob("*")) == stored


def test_an_import_answers_the_warnings_found_in_the_file(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, blot_bytes(tmp_path))  # a 16-bit TIFF: nothing to report
    assert status == 201 and answer["project"]["images"][0]["warnings"] == []
    pixels = np.full((H, W), 200, dtype=np.uint8)
    jpeg = io.BytesIO()
    Image.fromarray(pixels).save(jpeg, format="JPEG")
    status, answer = upload(client, jpeg.getvalue(), name="blot β.jpg")
    assert status == 201, answer
    image = next(i for i in answer["project"]["images"] if i["id"] == answer["image_id"])
    (warning,) = image["warnings"]
    assert warning["code"] == "lossy_format"
    assert warning["message"].startswith("JPEG-type compression can change pixel values")
    assert client.ok("GET", "/api/project")["project"]["images"][1] == image  # and it stays


def test_an_import_warns_of_an_image_that_looks_like_a_processed_figure(client):
    # #127: a background levelled to pure white, as a figure's is. A warning
    # only: the page lists it with the others, and the image imports as usual.
    pixels = np.full((H, W), 255, dtype=np.uint8)
    pixels[ROW - 6 : ROW + 6] = 90  # a dark row across it: a fifth of the image
    figure = io.BytesIO()
    Image.fromarray(pixels).save(figure, format="PNG")
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, figure.getvalue(), name="figure β.png")
    assert status == 201, answer
    (image,) = answer["project"]["images"]
    (warning,) = image["warnings"]
    assert warning["code"] == "looks_processed"
    assert warning["message"].startswith(
        "This looks like a processed figure rather than a raw scan: its median level,"
        " the image-wide background, is pure white; 80% of its pixels are pure white."
    )
    assert image["bit_depth"] == 8
    # Its background's end of the range comes from the polarity: light on dark,
    # nothing is at 0, and the page gets the warning dropped.
    path = f"/api/images/{image['id']}/polarity"
    answer = client.ok("PUT", path, {"polarity": "light_on_dark"})
    assert answer["project"]["images"][0]["warnings"] == []
    answer = client.ok("PUT", path, {"polarity": "dark_on_light"})
    assert answer["project"]["images"][0]["warnings"] == [warning]


def test_images_can_be_switched_repolarized_and_removed(client, tmp_path):
    image_id, _ = ready(client, tmp_path)
    answer = client.ok("PUT", f"/api/images/{image_id}/polarity", {"polarity": "light_on_dark"})
    assert answer["project"]["images"][0]["polarity"] == "light_on_dark"
    assert client.refused("PUT", f"/api/images/{image_id}/polarity", {"polarity": "x"})[:2] == (
        422,
        "invalid_input",
    )
    _, second = upload(client, blot_bytes(tmp_path), name="second.tif")
    assert [image["id"] for image in second["project"]["images"]] == [image_id, second["image_id"]]
    answer = client.ok("DELETE", f"/api/images/{image_id}")
    assert image_id in answer["removed"]
    assert [image["id"] for image in answer["project"]["images"]] == [second["image_id"]]
    assert client.refused("GET", f"/api/images/{image_id}/preview")[:2] == (404, "unknown_id")


# --- Original colours (#57) ---

PONCEAU = "Ponceau S α.png"  # alpha


def ponceau(dtype: type = np.uint8) -> np.ndarray:
    """A blot as a Ponceau S stain shows it: red bands on a pale pink membrane,
    so its red, green and blue differ; 8-bit RGB, or 16-bit at 257 times."""
    gray = synthetic_blot((H, W), [(x, ROW, 5.0, 3.0, 30000.0) for x in LANE_X], dtype=float)
    depth = 1 - gray / MEMBRANE_LEVEL  # 0 on the membrane, 0.6 at a band's centre
    rgb = np.stack([245 - 40 * depth, 225 - 330 * depth, 228 - 300 * depth], axis=-1)
    rgb = np.clip(np.round(rgb), 0, 255)
    return (rgb * 257).astype(np.uint16) if dtype is np.uint16 else rgb.astype(np.uint8)


def png_bytes(pixels: np.ndarray | Image.Image) -> bytes:
    """A PNG of pixels (gray, gray with alpha, RGB or RGBA by their channels) or of an image."""
    image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preview_of(client: Client, image_id: str, query: str = "") -> bytes:
    status, (kind, png) = client.call("GET", f"/api/images/{image_id}/preview{query}")
    assert (status, kind) == (200, "image/png")
    return png


def decoded(png: bytes) -> tuple[str, np.ndarray]:
    with Image.open(io.BytesIO(png)) as image:
        return image.mode, np.asarray(image)


def test_a_colour_image_is_previewed_in_its_original_colours_on_request(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    stain = ponceau()
    status, answer = upload(client, png_bytes(stain), name=PONCEAU, kind="visible_marker")
    assert status == 201, answer
    image_id = answer["image_id"]
    (image,) = answer["project"]["images"]
    assert image["colour"] is True
    assert [w["code"] for w in image["warnings"]] == ["color_channels_differ"]

    # The default stays the gray analysis image: the mean the nets are measured on.
    mode, gray = decoded(preview_of(client, image_id))
    assert (mode, gray.shape) == ("L", (H, W))
    # The original colours, on the same pixel grid: 8-bit colour is shown as stored.
    mode, colour = decoded(preview_of(client, image_id, "?colour=original"))
    assert (mode, colour.shape) == ("RGB", (H, W, 3))
    assert np.array_equal(colour, stain)
    band, membrane = colour[ROW, LANE_X[0]].astype(int), colour[5, LANE_X[0]].astype(int)
    assert band[0] > band[1] + 100 and band[0] > band[2] + 100  # a red band
    assert (membrane < 250).all() and membrane[0] > membrane[1]  # on a pink membrane
    # Nothing about the colours reaches the project: it is a view, not an edit.
    assert client.ok("GET", "/api/project")["project"]["revision"] == answer["project"]["revision"]


def test_a_gray_image_answers_its_gray_preview_for_its_original_colours(client, tmp_path):
    # A gray file's original colours are its gray levels: the same PNG as the
    # gray preview, so a page asking with a switch left on still draws the image.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, answer = upload(client, blot_bytes(tmp_path))  # 16-bit gray
    level = np.linspace(40, 220, W).round().astype(np.uint8)
    gray = np.tile(level, (H, 1))
    alpha = np.full_like(gray, 200)
    ids = {
        "16-bit gray": answer["image_id"],
        # Three equal channels: gray stored as RGB, as many scanners save it.
        "gray as RGB": upload(client, png_bytes(np.dstack([gray] * 3)), name="rgb.png")[1],
        "gray with alpha": upload(client, png_bytes(np.dstack([gray, alpha])), name="la.png")[1],
    }
    ids = {what: got if isinstance(got, str) else got["image_id"] for what, got in ids.items()}
    images = {i["id"]: i for i in client.ok("GET", "/api/project")["project"]["images"]}
    for what, image_id in ids.items():
        assert images[image_id]["colour"] is False, what
        plain = preview_of(client, image_id)
        assert preview_of(client, image_id, "?colour=original") == plain, what
        assert decoded(plain)[0] == "L", what


@pytest.mark.parametrize("query", ["?colour=grey", "?colour=ORIGINAL", "?colour="])
def test_an_unknown_colours_choice_is_refused(client, tmp_path, query):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, answer = upload(client, png_bytes(ponceau()), name=PONCEAU)
    path = f"/api/images/{answer['image_id']}/preview{query}"
    assert client.refused("GET", path)[:2] == (422, "invalid_input")


def test_original_colours_need_the_token_and_a_known_image(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, answer = upload(client, png_bytes(ponceau()), name=PONCEAU)
    path = f"/api/images/{answer['image_id']}/preview?colour=original"
    status, headers, body = fetch(client, path, token=False)
    assert (status, headers["www-authenticate"]) == (401, "Bearer")
    assert not body.startswith(b"\x89PNG")
    assert fetch(client, path)[0] == 200
    assert not_found(client, "/api/images/img-99/preview?colour=original") == "unknown_id"


def test_original_colours_are_read_only_from_the_unchanged_stored_file(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, answer = upload(client, png_bytes(ponceau()), name=PONCEAU)
    image_id = answer["image_id"]
    stored = client.root / "Blot" / storage.IMAGES_DIR / f"{image_id}.png"
    # Other bytes of the same size and layout, written outside Proteia.
    stored.write_bytes(png_bytes(ponceau()[:, ::-1].copy()))
    path = f"/api/images/{image_id}/preview?colour=original"
    assert client.refused("GET", path) == (422, "image_file_changed", [image_id])
    stored.unlink()
    assert client.refused("GET", path) == (422, "image_file_changed", [image_id])


def test_original_colours_are_kept_with_the_last_previews_shown(tmp_path, monkeypatch):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A")
    with io.BytesIO(png_bytes(ponceau())) as stream:
        image_id = api.ops.import_image(
            session, stream, "a.png", kind="visible_marker", polarity="dark_on_light"
        )
    session._pixels.clear()  # as if the image had not been worked on in this session
    reads: list[str] = []
    colour_pixels = type(session).colour_pixels

    def counting(self, image_id: str) -> np.ndarray:
        reads.append(image_id)
        return colour_pixels(self, image_id)

    monkeypatch.setattr(type(session), "colour_pixels", counting)
    colour = workspace.preview(session, image_id, original=True)
    assert workspace.preview(session, image_id, original=True) == colour
    assert reads == [image_id]  # read once, then kept

    monkeypatch.setattr(api, "_PREVIEWS_KEPT", 1)
    # The gray one is another preview: kept in place of the colour one, the last shown.
    assert workspace.preview(session, image_id) != colour
    assert image_id not in session._pixels  # neither holds the image in memory
    assert workspace.preview(session, image_id, original=True) == colour
    assert reads == [image_id, image_id]


def test_original_colours_of_other_colour_files(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    stain = ponceau()
    # A palette PNG, read as its palette's colours.
    indices = (np.arange(H * W).reshape(H, W) % 3).astype(np.uint8)
    table = [200, 30, 40, 250, 220, 225, 120, 10, 20] + [0] * (253 * 3)
    palette = Image.frombytes("P", (W, H), indices.tobytes())
    palette.putpalette(table)
    # RGBA: the alpha channel is dropped, as the analysis ignores it.
    rgba = np.dstack([stain, np.full((H, W), 90, dtype=np.uint8)])
    # 16-bit RGB: stretched to 8 bits over the three channels at once.
    ids = {
        "palette": upload(client, png_bytes(palette), name="palette.png")[1]["image_id"],
        "rgba": upload(client, png_bytes(rgba), name="rgba.png")[1]["image_id"],
        "16-bit": upload(
            client, write_tiff(tmp_path / "rgb16.tif", ponceau(np.uint16)).read_bytes()
        )[1]["image_id"],
    }
    images = {i["id"]: i for i in client.ok("GET", "/api/project")["project"]["images"]}
    assert all(images[image_id]["colour"] for image_id in ids.values())
    shown = {what: decoded(preview_of(client, i, "?colour=original")) for what, i in ids.items()}
    assert all(mode == "RGB" and pixels.shape == (H, W, 3) for mode, pixels in shown.values())
    expected = np.array(table[:9], dtype=np.uint8).reshape(3, 3)[indices]
    assert np.array_equal(shown["palette"][1], expected)
    assert np.array_equal(shown["rgba"][1], stain)
    sixteen = shown["16-bit"][1].astype(int)
    assert (sixteen.min(), sixteen.max()) == (0, 255)
    stretched = (stain.astype(float) - stain.min()) * 255 / (int(stain.max()) - int(stain.min()))
    assert np.abs(sixteen - stretched).max() <= 1  # one stretch: the hues keep their balance


def palette_tiff(colormap: np.ndarray) -> tuple[bytes, np.ndarray]:
    """An 8-bit TIFF whose pixels index ``colormap`` (3 x 256, 16-bit), as ImageJ
    saves one with a lookup table, and its indices: a dark band under each lane."""
    gray = synthetic_blot((H, W), [(x, ROW, 5.0, 3.0, 30000.0) for x in LANE_X], dtype=float)
    indices = np.clip(np.round(gray / 257), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, indices, photometric="palette", colormap=colormap)
    return buffer.getvalue(), indices


def test_a_palette_tiff_is_shown_through_its_colour_map(client, tmp_path):
    # The nets are measured on the indices; its own colours are the map's.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    level = np.arange(256, dtype=np.uint16)
    fire = np.stack([level * 257, level * 128, (255 - level) * 64])
    data, indices = palette_tiff(fire)
    status, answer = upload(client, data, name="Fire LUT α.tif")
    assert status == 201, answer
    image_id = answer["image_id"]
    (image,) = answer["project"]["images"]
    assert (image["colour"], image["warnings"]) == (True, [])
    mode, colour = decoded(preview_of(client, image_id, "?colour=original"))
    assert (mode, colour.shape) == ("RGB", (H, W, 3))
    np.testing.assert_array_equal(colour, (fire.T >> 8).astype(np.uint8)[indices])
    assert decoded(preview_of(client, image_id))[0] == "L"
    # A gray lookup table (inverted, say) has no colour to show.
    gray_map = np.stack([(255 - level) * 257] * 3)
    other = upload(client, palette_tiff(gray_map)[0], name="inverted.tif")[1]["image_id"]
    images = {i["id"]: i for i in client.ok("GET", "/api/project")["project"]["images"]}
    assert images[other]["colour"] is False
    assert preview_of(client, other, "?colour=original") == preview_of(client, other)


@pytest.mark.parametrize("fmt", ["TIFF", "JPEG"])
def test_a_cmyk_file_is_imported_converted_and_offered_in_its_converted_colours(
    client, tmp_path, fmt
):
    # #131: its inks are converted to red, green and blue, then to gray as the
    # RGB original's are, and the import says so; its original colours are the
    # converted ones (exact for this TIFF, within JPEG's error for the JPEG).
    client.ok("POST", "/api/projects", {"name": "Blot"})
    stain = ponceau()
    original = upload(client, png_bytes(stain), name=PONCEAU, kind="visible_marker")[1]
    buffer = io.BytesIO()
    quality = {"quality": 95} if fmt == "JPEG" else {}
    Image.fromarray(stain).convert("CMYK").save(buffer, format=fmt, **quality)
    suffix = ".tif" if fmt == "TIFF" else ".jpg"
    status, answer = upload(client, buffer.getvalue(), name=f"cmyk{suffix}", kind="visible_marker")
    assert status == 201, answer
    image_id = answer["image_id"]
    image = next(i for i in answer["project"]["images"] if i["id"] == image_id)
    codes = [w["code"] for w in image["warnings"]]
    lossy = ["lossy_format"] if fmt == "JPEG" else []
    assert codes == [*lossy, "cmyk_converted", "color_channels_differ"]
    assert (
        "converted to red, green and blue with the standard formula"
        in (image["warnings"][len(lossy)]["message"])
    )
    assert image["colour"] is True
    mode, colour = decoded(preview_of(client, image_id, "?colour=original"))
    assert (mode, colour.shape) == ("RGB", (H, W, 3))
    error = 0 if fmt == "TIFF" else 3
    assert np.abs(colour.astype(int) - stain).max() <= error
    # Its gray is the RGB original's: the same preview, stretched alike.
    mode, gray = decoded(preview_of(client, image_id))
    expected = decoded(preview_of(client, original["image_id"]))[1]
    assert mode == "L"
    assert np.abs(gray.astype(int) - expected).max() <= error * 2


def test_an_image_in_a_colour_space_proteia_does_not_convert_is_refused(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    buffer = io.BytesIO()
    Image.fromarray(ponceau()).convert("LAB").save(buffer, format="TIFF")
    status, answer = upload(client, buffer.getvalue(), name="lab µ.tif")
    assert (status, answer["code"]) == (422, "unreadable_image")
    assert "in the CIELAB colour space" in answer["message"]
    assert client.ok("GET", "/api/project")["project"]["images"] == []


def test_the_state_reads_a_file_header_once_and_only_where_colour_can_hide(tmp_path, monkeypatch):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A")
    level = np.linspace(40, 220, W).round().astype(np.uint8)
    files = {
        "gray.png": png_bytes(np.tile(level, (H, 1))),
        "stain.png": png_bytes(ponceau()),
        "gray.tif": blot_bytes(tmp_path),
    }
    ids = {}
    for name, data in files.items():
        with io.BytesIO(data) as stream:
            ids[name] = api.ops.import_image(
                session, stream, name, kind="visible_marker", polarity="dark_on_light"
            )
    reads: list[str] = []
    file_colours = session_module.file_colours

    def counting(path: Path) -> str | None:
        reads.append(Path(path).stem)  # the stored file: the image id
        return file_colours(path)

    monkeypatch.setattr(session_module, "file_colours", counting)
    session._file_colours.clear()
    for _ in range(3):
        state = project_state("A", session, session.project, open_id=1)
        assert {i["id"]: i["colour"] for i in state["images"]} == {
            ids["gray.png"]: False,
            ids["stain.png"]: True,
            ids["gray.tif"]: False,  # a TIFF: read, as a palette would show only there
        }
    # A gray PNG is never read; the colour PNG and the TIFF once each.
    assert sorted(reads) == sorted([ids["stain.png"], ids["gray.tif"]])


# --- Boxes ---


def test_boxes_are_exchanged_in_image_pixels(client, tmp_path):
    _, protein = ready(client, tmp_path)
    first = client.ok(
        "POST",
        "/api/boxes",
        {"protein_id": protein, "x": LANE_X[0], "y": ROW, "lane_index": 0, "grow": True},
    )
    size = first["project"]["proteins"][0]["box_size"]
    answer = client.ok(
        "POST", "/api/boxes", {"protein_id": protein, "x": LANE_X[1], "y": ROW, "lane_index": 1}
    )
    x0, y0, x1, y1 = bands(answer)[answer["band_id"]]["rect"]
    assert (x1 - x0, y1 - y0) == (size["width"], size["height"])
    assert abs((x0 + x1) / 2 - LANE_X[1]) <= 0.5 and abs((y0 + y1) / 2 - ROW) <= 0.5

    target = [x0 + 7, y0 + 3, x1 + 7, y1 + 3]  # exactly the box size, anywhere on the image
    moved = client.ok("PUT", f"/api/boxes/{answer['band_id']}", {"rect": target})
    assert bands(moved)[answer["band_id"]]["rect"] == target
    # What is drawn, and the results computed from the same snapshot.
    assert client.ok("GET", "/api/project") == {
        "project": moved["project"],
        "results": moved["results"],
    }


def test_boxes_are_placed_relaned_and_removed_through_the_operations(client, tmp_path):
    _, protein = ready(client, tmp_path)

    def place(lane: int | None, x: int) -> str:
        body = {"protein_id": protein, "x": x, "y": ROW, "lane_index": lane, "grow": lane == 0}
        return client.ok("POST", "/api/boxes", body)["band_id"]

    a, b = place(0, LANE_X[0]), place(1, LANE_X[1])
    c = place(None, LANE_X[3])  # proposed from the two boxes before
    answer = client.ok("GET", "/api/project")
    assert bands(answer)[c]["lane_index"] == 3

    answer = client.ok("PUT", f"/api/boxes/{b}/lane", {"lane_index": 2})
    assert (bands(answer)[b]["lane_index"], bands(answer)[b]["manually_edited"]) == (2, True)
    assert client.refused("PUT", f"/api/boxes/{b}/lane", {"lane_index": 3}) == (
        422,
        "lane_occupied",
        [c],
    )
    assert client.refused("PUT", f"/api/boxes/{b}/lane", {"lane_index": 9})[:2] == (
        422,
        "lane_out_of_range",
    )
    # The page shows a refusal's message as it is, so it numbers lanes from 1
    # as the page does; the request's lane_index 3 is the stored index.
    _, payload = client.call("PUT", f"/api/boxes/{b}/lane", {"lane_index": 3})
    assert payload["message"] == "'β-actin' already has a box in lane 4"
    rect = bands(answer)[a]["rect"]
    assert client.refused("PUT", f"/api/boxes/{b}", {"rect": rect})[:2] == (422, "overlap")

    answer = client.ok("DELETE", f"/api/boxes/{a}")
    assert set(bands(answer)) == {b, c}
    assert client.refused("DELETE", f"/api/boxes/{a}")[:2] == (404, "unknown_id")
    project = json.loads((client.root / "Blot" / storage.PROJECT_FILE).read_text(encoding="utf-8"))
    actions = [entry["action"] for entry in project["log"]]
    assert actions[-4:] == ["place_box", "place_box", "set_box_lane", "remove_box"]


@pytest.mark.parametrize(
    "body",
    [
        {"x": "50", "y": ROW},  # a number as text
        {"x": 50.5, "y": ROW},
        {"x": 50, "y": ROW, "grow": "yes"},
        {"x": 50, "y": ROW, "color": "red"},  # an unknown field
    ],
)
def test_a_malformed_box_request_is_refused(client, tmp_path, body):
    _, protein = ready(client, tmp_path)
    assert client.refused("POST", "/api/boxes", {"protein_id": protein, **body})[:2] == (
        422,
        "invalid_input",
    )


def test_missing_lanes_and_clipped_bands_are_reported(client, tmp_path):
    _, protein = ready(client, tmp_path, clipped_lane=1)
    for lane in (0, 1, 3):
        body = {
            "protein_id": protein,
            "x": LANE_X[lane],
            "y": ROW,
            "lane_index": lane,
            "grow": lane == 0,
        }
        answer = client.ok("POST", "/api/boxes", body)
    (state,) = answer["project"]["proteins"]
    assert {band["lane_index"]: band["clipped"] for band in state["bands"]} == {
        0: False,
        1: True,
        3: False,
    }
    missing = {entry["lane_index"]: entry for entry in state["missing_lanes"]}
    assert set(missing) == {2, 4}
    for lane, entry in missing.items():
        assert abs(entry["x"] - LANE_X[lane]) <= 2
        assert abs(entry["y"] - ROW) <= 1


def test_missing_lanes_have_no_position_before_two_lanes_are_boxed(client, tmp_path):
    _, protein = ready(client, tmp_path)
    (state,) = client.ok("GET", "/api/project")["project"]["proteins"]
    assert state["missing_lanes"] == [
        {"lane_index": lane, "x": None, "y": None} for lane in range(len(LANE_X))
    ]


def _record(
    lane: int, *, band_index: int = 0, source: ProposalSource = ProposalSource.ROW_BOX
) -> UndetectedBand:
    """A not-detected record over a lane of the blot."""
    x = LANE_X[lane]
    return UndetectedBand(
        lane_index=lane,
        band_index=band_index,
        reason=UndetectedReason.BELOW_DETECTION_LIMIT,
        snr=2.0,
        threshold=6.0,
        region=Region(x0=x - 10, y0=ROW - 8, x1=x + 10, y1=ROW + 8),
        source=source,
    )


def plant_records(client: Client, protein_id: str, *records: UndetectedBand, bands: int) -> None:
    """Store records in the open project as given: a row box writes only its own
    (band index 0, source ``row_box``)."""
    session = client.workspace.current()

    def change(draft: Project) -> None:
        protein = draft.batch.find_protein(protein_id)
        protein.expected_band_count = bands
        protein.undetected.extend(records)

    with session.transaction():
        project, _ = apply_change(session.project, change)
        session._commit(project, action="plant", params={})


def test_records_are_drawn_and_their_lanes_are_not_missing(client, tmp_path):
    _, protein = ready(client, tmp_path)
    for lane in (0, 1):
        body = {"protein_id": protein, "x": LANE_X[lane], "y": ROW, "lane_index": lane}
        client.ok("POST", "/api/boxes", {**body, "grow": lane == 0})
    guided = _record(3, band_index=1, source=ProposalSource.MW_GUIDED)
    plant_records(client, protein, _record(2), guided, bands=2)

    (state,) = client.ok("GET", "/api/project")["project"]["proteins"]
    x2, x3 = LANE_X[2], LANE_X[3]
    common = {"reason": "below_detection_limit", "snr": 2.0, "threshold": 6.0}
    assert state["undetected"] == [
        {
            "lane_index": 2,
            "band_index": 0,
            **common,
            "region": [x2 - 10, ROW - 8, x2 + 10, ROW + 8],
            "source": "row_box",
        },
        {
            "lane_index": 3,
            "band_index": 1,
            **common,
            "region": [x3 - 10, ROW - 8, x3 + 10, ROW + 8],
            "source": "mw_guided",
        },
    ]
    # Lane 2 was examined. Lane 3's record is for the second band: its first is missing.
    missing = {entry["lane_index"]: entry for entry in state["missing_lanes"]}
    assert set(missing) == {3, 4}
    assert abs(missing[3]["x"] - LANE_X[3]) <= 2 and abs(missing[3]["y"] - ROW) <= 1


def test_the_state_lists_every_proteins_records(tmp_path):
    doc = make_project_with_undetected().model_dump(mode="json")
    # Two loading controls, in neither id nor creation order.
    doc["batch"]["proteins"][0]["loading_control_ids"] = ["prot-9", "prot-8"]
    project = Project.model_validate(doc)
    folder = tmp_path / "Blot"
    write_image_files(folder, project)
    storage.save_project(project, folder)
    session = api.ops.open_project(folder, clock=FakeClock())

    answer = project_state("Blot", session, session.project, open_id=4)
    assert (answer["open_id"], answer["revision"]) == (4, 0)  # the fixture has no log
    proteins = {protein["id"]: protein for protein in answer["proteins"]}
    common = {"band_index": 0, "reason": "below_detection_limit", "threshold": 6.0}
    assert proteins["prot-7"]["undetected"] == [
        {**common, "lane_index": 2, "snr": 2.5, "region": [98, 36, 128, 64], "source": "row_box"}
    ]
    assert proteins["prot-8"]["undetected"] == []
    assert proteins["prot-9"]["undetected"] == [
        {
            **common,
            "lane_index": 2,
            "snr": -0.75,
            "region": [98, 93, 128, 121],
            "source": "row_box",
        },
        {
            **common,
            "lane_index": 3,
            "snr": 4.125,
            "region": [138, 93, 168, 121],
            "source": "mw_guided",
        },
    ]
    # Every lane of each protein holds a box or a record: none is offered as missing.
    assert [protein["missing_lanes"] for protein in answer["proteins"]] == [[], [], []]
    assert json.loads(json.dumps(answer, allow_nan=False)) == answer
    # The target's loading controls in series order, its expected MW, each box's source.
    details = {
        protein["id"]: (protein["loading_control_ids"], protein["expected_mw"])
        for protein in answer["proteins"]
    }
    assert details == {
        "prot-7": (["prot-9", "prot-8"], 92.0),
        "prot-8": ([], None),
        "prot-9": ([], None),
    }
    series = compute_results(session.project.batch).series
    assert [s.loading_id for s in series] == proteins["prot-7"]["loading_control_ids"]
    assert [(band["lane_index"], band["source"]) for band in proteins["prot-7"]["bands"]] == [
        (0, "click"),
        (1, "row_box"),
        (3, "mw_guided"),
    ]


def test_edits_need_an_open_project(client):
    body = {"protein_id": "prot-1", "x": 1, "y": 1}
    assert client.refused("POST", "/api/boxes", body)[:2] == (409, "no_project")
    assert client.refused("PUT", "/api/lanes", {"lanes": []})[:2] == (409, "no_project")


# --- Review of #80 ---


def test_a_folder_that_cannot_be_written_answers_json(client):
    client.root.parent.mkdir(parents=True, exist_ok=True)
    client.root.write_text("a file where the projects folder should be", encoding="utf-8")
    status, code, _ = client.refused("POST", "/api/projects", {"name": "Blot"})
    assert (status, code) == (500, "file_error")


def test_an_older_project_whose_backup_cannot_be_kept_is_not_opened(client, monkeypatch):
    # #140: a schema-1 project beside the open one; its project.json cannot be
    # kept before the migration rewrites it.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    folder = v1_folder(client.root)
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    real_open = os.open

    def refuse(path, flags, *args, **kwargs):
        if Path(path).name.startswith("project.schema"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "open", refuse)
        status, answer = client.call("POST", "/api/projects/open", {"name": folder.name})
    # The page shows the message as it is, in the Projects dialog.
    assert (status, answer["code"]) == (500, "file_error")
    assert answer["message"] == (
        f"{folder.name!r} was saved by an older Proteia (schema 1), and a copy of its"
        " project.json could not be kept before updating it (Permission denied);"
        " nothing was changed"
    )
    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before
    assert client.ok("GET", "/api/projects")["open"] == "Blot"  # still open

    opened = client.ok("POST", "/api/projects/open", {"name": folder.name})
    assert opened["project"]["name"] == folder.name
    backup = folder / "project.schema1.json"
    assert backup.read_bytes() == before[folder / storage.PROJECT_FILE]


def test_readers_do_not_wait_for_a_project_switch(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    workspace.create("A")
    with workspace._switching:  # a switch is saving or opening
        assert workspace.current().folder.name == "A"
        assert workspace.open_name == "A"


def test_a_preview_does_not_keep_the_full_image_in_memory(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A")
    with io.BytesIO(blot_bytes(tmp_path)) as stream:
        image_id = api.ops.import_image(
            session, stream, "a.tif", kind="chemiluminescence", polarity="dark_on_light"
        )
    session._pixels.clear()  # as if the image had not been worked on in this session
    assert workspace.preview(session, image_id).startswith(b"\x89PNG")
    assert image_id not in session._pixels


def test_quit_saves_unsaved_changes_first_or_refuses(client, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    project_file = client.root / "Blot" / storage.PROJECT_FILE
    project_file.unlink()
    project_file.mkdir()  # the autosave cannot replace it
    answer = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    assert answer["project"]["saved"] is False and answer["project"]["save_error"]

    assert client.refused("POST", "/api/quit")[:2] == (409, "unsaved_changes")
    assert client.ok("GET", "/api/status")["app"] == "proteia"  # still running

    project_file.rmdir()
    assert client.call("POST", "/api/quit")[0] == 202
    saved = json.loads(project_file.read_text(encoding="utf-8"))
    assert [lane["label"] for lane in saved["batch"]["lanes"]] == ["vehicle"]


@pytest.mark.parametrize("size", [[0, 10], [10, -5], [10], ["10", 10]])
def test_a_bad_box_size_is_refused_as_json(client, tmp_path, size):
    image_id, _ = ready(client, tmp_path)
    body = {"name": "GAPDH", "role": "loading control", "image_id": image_id, "box_size": size}
    assert client.refused("POST", "/api/proteins", body)[:2] == (422, "invalid_input")


# --- Live results (#52) ---

TARGET_ROW, LOADING_ROW, TWO_ROW_H = 30, 75, 100
SIZE = [14, 10]  # every box of both proteins: the same pixels under each lane
DOSES = ["vehicle", "vehicle", "10 µM", "10 µM", "10 µM"]  # vehicle / 10 micro-molar
DEPTHS = (20000.0, 22000.0, 30000.0, 33000.0, 36000.0)


def two_row_bytes(tmp_path: Path, target: tuple[float, ...], loading: tuple[float, ...]) -> bytes:
    """A 16-bit blot with a target row and a loading-control row: one band per
    lane in each, as dark as ``target`` and ``loading`` give per lane."""
    rows = ((TARGET_ROW, target), (LOADING_ROW, loading))
    spots = [
        (x, row, 5.0, 3.0, depth)
        for row, depths in rows
        for x, depth in zip(LANE_X, depths, strict=True)
    ]
    blot = synthetic_blot((TWO_ROW_H, W), spots)
    return write_tiff(tmp_path / "two rows.tif", blot).read_bytes()


def live(
    client: Client,
    tmp_path: Path,
    conditions: list[str],
    *,
    target: tuple[float, ...] = DEPTHS,
    loading: tuple[float, ...] = (25000.0,) * 5,
    boxed: tuple[int, ...] = (0, 1, 2, 3, 4),
    reference: str | None = None,
) -> tuple[str, str, dict]:
    """A project over the two-row blot with the lanes ``conditions`` (there may
    be more lanes than bands), α-tubulin boxed in every band's lane and
    β-catenin in the lanes ``boxed``; (target id, loading id, the last answer)."""
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, two_row_bytes(tmp_path, target, loading))
    assert status == 201, answer
    image_id = answer["image_id"]
    lanes: dict[str, Any] = {"lanes": [{"condition": condition} for condition in conditions]}
    if reference is not None:
        lanes["reference_condition"] = reference
    client.ok("PUT", "/api/lanes", lanes)
    ids = []
    for name, role in (("α-tubulin", "loading control"), ("β-catenin", "target")):
        body = {"name": name, "role": role, "image_id": image_id, "box_size": SIZE}
        ids.append(client.ok("POST", "/api/proteins", body)["protein_id"])
    loading_id, target_id = ids
    for protein, row, lanes_boxed in (
        (loading_id, LOADING_ROW, range(len(LANE_X))),
        (target_id, TARGET_ROW, boxed),
    ):
        for lane in lanes_boxed:
            body = {"protein_id": protein, "x": LANE_X[lane], "y": row, "lane_index": lane}
            answer = client.ok("POST", "/api/boxes", body)
    return target_id, loading_id, answer


def column(answer: dict, protein_id: str) -> dict:
    """A protein's per-lane column of the answer's results."""
    return next(p for p in answer["results"]["proteins"] if p["protein_id"] == protein_id)


def only_series(answer: dict, set_index: int = 0) -> dict:
    (series,) = answer["results"]["sets"][set_index]["series"]
    return series


def bar(series: dict, condition: str) -> dict:
    return next(b for b in series["chart"]["bars"] if b["label"] == condition)


def test_a_jpeg_project_says_its_bands_were_not_checked_for_over_exposure(client, tmp_path):
    # #112: the over-exposure check does not run on a JPEG, and the results say so
    # per protein. The Checks list shows every notice of a set, and a chart card
    # those about its series' proteins, whatever their code.
    spots = [(x, row, 5.0, 3.0, 25000.0) for row in (TARGET_ROW, LOADING_ROW) for x in LANE_X]
    blot = (synthetic_blot((TWO_ROW_H, W), spots) // 257).astype(np.uint8)
    jpeg = io.BytesIO()
    Image.fromarray(blot).save(jpeg, format="JPEG", quality=95)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, jpeg.getvalue(), name="blot β.jpg")
    assert status == 201, answer
    image_id = answer["image_id"]
    (image,) = answer["project"]["images"]
    assert [w["code"] for w in image["warnings"]] == ["lossy_format"]
    assert "over-exposure cannot be checked" in image["warnings"][0]["message"]
    lanes = [{"condition": condition} for condition in DOSES]
    client.ok("PUT", "/api/lanes", {"lanes": lanes, "reference_condition": "vehicle"})
    ids = []
    for name, role in (("α-tubulin", "loading control"), ("β-catenin", "target")):
        body = {"name": name, "role": role, "image_id": image_id, "box_size": SIZE}
        ids.append(client.ok("POST", "/api/proteins", body)["protein_id"])
    loading_id, target_id = ids
    for protein, row, lanes_boxed in (
        (loading_id, LOADING_ROW, (0, 1, 2, 3)),
        (target_id, TARGET_ROW, (0, 2, 4)),
    ):
        for lane in lanes_boxed:
            body = {"protein_id": protein, "x": LANE_X[lane], "y": row, "lane_index": lane}
            answer = client.ok("POST", "/api/boxes", body)
    assert {band["clipped"] for band in bands(answer).values()} == {None}

    (result_set,) = answer["results"]["sets"]
    unchecked = [n for n in result_set["notices"] if n["code"] == "clipping_not_checked"]
    assert [(n["protein_ids"], n["lane_indices"], n["level"]) for n in unchecked] == [
        ([loading_id], [0, 1, 2, 3], "warning"),
        ([target_id], [0, 2, 4], "warning"),
    ]
    loading, target = (n["message"] for n in unchecked)
    assert loading.startswith(
        "'α-tubulin' was not checked for over-exposure in lanes 1, 2, 3, 4: its image has"
        " lossy (JPEG-type) compression"
    )
    assert loading.endswith("which biases every value normalized to it")
    assert target.startswith("'β-catenin' was not checked for over-exposure in lanes 1, 3, 5:")
    # The series' chart card shows both: each is about one of its proteins.
    series = only_series(answer)
    assert series["chart"] is not None
    assert {n["protein_ids"][0] for n in unchecked} == {series["target_id"], series["loading_id"]}


def test_a_jpeg_project_flags_bands_possibly_over_exposed(client, tmp_path):
    # #112: on a JPEG, a box with 5 or more pixels within 2 levels of the limit
    # is flagged, per band in the state and per lane in the results, and each
    # protein gets a warning naming its lanes.
    deep = {("target", 1), ("target", 3), ("loading", 0)}  # saturated at 0
    spots = [
        (x, row, 5.0, 3.0, 70000.0 if (name, lane) in deep else 25000.0)
        for name, row in (("target", TARGET_ROW), ("loading", LOADING_ROW))
        for lane, x in enumerate(LANE_X)
    ]
    blot = np.round(synthetic_blot((TWO_ROW_H, W), spots) / 257).astype(np.uint8)
    jpeg = io.BytesIO()
    Image.fromarray(blot).save(jpeg, format="JPEG", quality=85)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    status, answer = upload(client, jpeg.getvalue(), name="blot β.jpg")
    assert status == 201, answer
    image_id = answer["image_id"]
    lanes = [{"condition": condition} for condition in DOSES]
    client.ok("PUT", "/api/lanes", {"lanes": lanes, "reference_condition": "vehicle"})
    ids = []
    for name, role in (("α-tubulin", "loading control"), ("β-catenin", "target")):
        body = {"name": name, "role": role, "image_id": image_id, "box_size": SIZE}
        ids.append(client.ok("POST", "/api/proteins", body)["protein_id"])
    loading_id, target_id = ids
    for protein, row in ((loading_id, LOADING_ROW), (target_id, TARGET_ROW)):
        for lane, x in enumerate(LANE_X):
            body = {"protein_id": protein, "x": x, "y": row, "lane_index": lane}
            answer = client.ok("POST", "/api/boxes", body)

    # The state: each band's flags (the exact check did not run).
    flagged = {
        (protein["id"], band["lane_index"]): (band["clipped"], band["possibly_clipped"])
        for protein in answer["project"]["proteins"]
        for band in protein["bands"]
    }
    assert flagged == {
        (protein, lane): (None, (name, lane) in deep)
        for name, protein in (("target", target_id), ("loading", loading_id))
        for lane in range(len(LANE_X))
    }
    # The results: the columns, and one warning per protein, loading control first.
    assert column(answer, target_id)["possibly_clipped"] == [False, True, False, True, False]
    assert column(answer, loading_id)["possibly_clipped"] == [True, False, False, False, False]
    (result_set,) = answer["results"]["sets"]
    possibly = [n for n in result_set["notices"] if n["code"] == "possibly_clipped"]
    assert [(n["protein_ids"], n["lane_indices"], n["level"]) for n in possibly] == [
        ([loading_id], [0], "warning"),
        ([target_id], [1, 3], "warning"),
    ]
    loading, target = (n["message"] for n in possibly)
    assert loading.startswith(
        "'α-tubulin' is possibly over-exposed in lane 1: its box holds 5 or more pixels"
        " within 2 grey levels of the detector limit, and its image has lossy (JPEG-type)"
        " compression, so saturation cannot be confirmed;"
    )
    assert "which biases every value normalized to it" in loading
    assert target.startswith("'β-catenin' is possibly over-exposed in lanes 2, 4:")
    # Both reach the series' chart card: each is about one of its proteins.
    series = only_series(answer)
    assert {n["protein_ids"][0] for n in possibly} == {series["target_id"], series["loading_id"]}
    assert answer["project"]["unassessed_images"] == []

    # The same boxes as a project saved before #112: neither flag on them. The
    # state names the image, each not-checked notice says to requantify, and a
    # requantify assesses its bands.
    session = client.workspace.current()

    def forget(draft: Project) -> None:
        for protein in draft.batch.proteins:
            for band in protein.bands:
                band.possibly_clipped = None

    with session.transaction():
        project, _ = apply_change(session.project, forget)
        session._commit(project, action="plant", params={})
    before = client.ok("GET", "/api/project")
    assert before["project"]["unassessed_images"] == [image_id]
    (result_set,) = before["results"]["sets"]
    assert "possibly_clipped" not in notice_codes(before)
    unchecked = [n["message"] for n in result_set["notices"] if n["code"] == "clipping_not_checked"]
    assert len(unchecked) == 2
    assert all(message.endswith(": requantify to look for them") for message in unchecked)

    requantified = client.ok("POST", "/api/requantify")
    assert requantified["images"] == [image_id]
    assert requantified["project"]["unassessed_images"] == []
    assert {
        (protein["id"], band["lane_index"]): (band["clipped"], band["possibly_clipped"])
        for protein in requantified["project"]["proteins"]
        for band in protein["bands"]
    } == flagged
    assert "possibly_clipped" in notice_codes(requantified)
    assert history(requantified)["undo"] == {
        "seq": requantified["project"]["revision"],
        "action": "requantify",
    }


def test_box_edits_answer_with_the_changed_net_and_chart(client, tmp_path):
    target, _, before = live(client, tmp_path, DOSES, boxed=(0, 1, 2, 3))
    assert column(before, target)["nets"][4] is None
    assert bar(only_series(before), "10 µM")["n"] == 2

    body = {"protein_id": target, "x": LANE_X[4], "y": TARGET_ROW, "lane_index": 4}
    placed = client.ok("POST", "/api/boxes", body)
    net = column(placed, target)["nets"][4]  # the net appears
    assert net > 0 and column(placed, target)["band_ids"][4] == placed["band_id"]
    series = only_series(placed)
    assert series["normalized"][4] is not None
    grown = bar(series, "10 µM")
    assert (grown["n"], grown["lane_indices"]) == (3, [2, 3, 4])
    assert series["chart"] != only_series(before)["chart"]

    x0, y0, x1, y1 = bands(placed)[placed["band_id"]]["rect"]
    rect = [x0 + 4, y0 + 3, x1 + 4, y1 + 3]  # off the band's centre
    moved = client.ok("PUT", f"/api/boxes/{placed['band_id']}", {"rect": rect})
    assert 0 < column(moved, target)["nets"][4] < net  # the net changes
    shifted = bar(only_series(moved), "10 µM")
    assert shifted["n"] == 3 and shifted["mean"] < grown["mean"]
    assert shifted["points"][:2] == grown["points"][:2]
    assert shifted["points"][2] < grown["points"][2]

    removed = client.ok("DELETE", f"/api/boxes/{placed['band_id']}")
    assert column(removed, target)["nets"][4] is None  # the net becomes null
    assert only_series(removed)["normalized"][4] is None
    # The boxes are those before the placement again, and so are the results.
    assert removed["results"]["sets"] == before["results"]["sets"]
    assert removed["results"]["revision"] == before["results"]["revision"] + 3


def test_the_answer_equals_the_results_of_the_saved_project(client, tmp_path):
    _, _, answer = live(client, tmp_path, DOSES, reference="vehicle")
    results = answer["results"]
    assert results["sets"][0]["tier"] == "fold_change"
    saved = storage.load_project(client.root / "Blot")
    open_id, revision = results["open_id"], results["revision"]
    expected = results_payload(
        compute_results(saved.batch), open_id=open_id, revision=revision, charts=charts.chart_url
    )
    assert results == expected  # the one computation path (#42)
    assert revision == saved.log[-1].seq == answer["project"]["revision"]

    client.ok("POST", "/api/projects", {"name": "Other"})
    reopened = client.ok("POST", "/api/projects/open", {"name": "Blot"})
    assert reopened["results"] == {**expected, "open_id": open_id + 2}
    assert reopened["project"]["revision"] == revision  # opening logs nothing


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
def test_identical_values_answer_strict_json_without_a_test(client, tmp_path):
    # Every band alike: every ratio is the same, so a t-test would divide by zero.
    same = (30000.0,) * 5
    _, _, answer = live(client, tmp_path, DOSES, target=same, loading=same)
    series = only_series(answer)  # Client.call refused NaN and inf in the body
    assert len(set(series["normalized"])) == 1
    chart = series["chart"]
    assert (chart["test_name"], chart["test_p"], chart["comparisons"]) == (None, None, [])
    assert chart["test_note"].startswith("no test")


def test_an_excluded_lane_with_a_value_gives_two_labelled_sets(client, tmp_path):
    conditions = [*DOSES, "ladder"]  # the last lane has no box of any protein
    _, _, answer = live(client, tmp_path, conditions)
    (only,) = answer["results"]["sets"]
    assert (only["id"], only["label"], only["excluded_lanes"]) == ("applied", None, [])

    def excluding(*lanes: int) -> dict:
        rows = [{"condition": c, "included": i not in lanes} for i, c in enumerate(conditions)]
        return {"lanes": rows}

    answer = client.ok("PUT", "/api/lanes", excluding(5))  # no value: nothing to show twice
    (only,) = answer["results"]["sets"]
    assert (only["label"], only["excluded_lanes"]) == (None, [5])
    assert only_series(answer)["chart"]["subtitle"] is None

    answer = client.ok("PUT", "/api/lanes", excluding(2, 5))
    applied, every = answer["results"]["sets"]
    assert [(s["id"], s["label"], s["excluded_lanes"]) for s in (applied, every)] == [
        ("applied", "Excluding lane 3", [2, 5]),
        ("all_lanes", "All lanes", []),
    ]
    for result_set in (applied, every):
        (series,) = result_set["series"]
        assert series["chart"]["subtitle"] == result_set["label"]
    assert bar(applied["series"][0], "10 µM")["n"] == 2
    assert bar(every["series"][0], "10 µM")["n"] == 3
    lanes = answer["results"]["lanes"]  # once, as the lane table has them
    assert [lane["included"] for lane in lanes] == [True, True, False, True, True, False]
    # The per-lane values keep the excluded lane in both sets.
    assert applied["series"][0]["normalized"][2] == every["series"][0]["normalized"][2] > 0

    answer = client.ok("PUT", "/api/lanes", excluding())
    assert [s["label"] for s in answer["results"]["sets"]] == [None]


def test_an_unusable_reference_gives_no_chart_and_a_notice(client, tmp_path):
    # No β-catenin box in the vehicle lanes: no baseline for a fold-change.
    target, loading, answer = live(client, tmp_path, DOSES, boxed=(2, 3, 4), reference="vehicle")
    (result_set,) = answer["results"]["sets"]
    (series,) = result_set["series"]
    assert (series["chart"], series["chart_url"], series["fold_change"]) == (None, None, None)
    assert series["value_kind"] == "loading_normalized"
    (notice,) = [n for n in result_set["notices"] if n["code"] == "reference_unusable"]
    assert (notice["level"], notice["protein_ids"], notice["conditions"]) == (
        "warning",
        [target, loading],
        ["vehicle"],
    )


def test_the_revision_counts_commits_and_the_open_id_counts_opens(client, tmp_path):
    created = client.ok("POST", "/api/projects", {"name": "Blot"})
    _, imported = upload(client, blot_bytes(tmp_path))
    image_id = imported["image_id"]
    lanes = {"lanes": [{"condition": f"c{i}"} for i in range(len(LANE_X))]}
    declared = client.ok("PUT", "/api/lanes", lanes)
    same = client.ok("PUT", "/api/lanes", lanes)  # a no-op: no log entry
    body = {"name": "α-tubulin", "role": "loading control", "image_id": image_id}
    loading = client.ok("POST", "/api/proteins", body)
    body = {
        "name": "β-catenin",
        "role": "target",
        "image_id": image_id,
        "expected_mw": 92.5,
        "loading_control_ids": [loading["protein_id"]],
    }
    target = client.ok("POST", "/api/proteins", body)
    place = {"protein_id": target["protein_id"], "y": ROW}
    grown = client.ok(
        "POST", "/api/boxes", {**place, "x": LANE_X[0], "lane_index": 0, "grow": True}
    )
    fixed = client.ok("POST", "/api/boxes", {**place, "x": LANE_X[1], "lane_index": 1})
    answers = [created, imported, declared, same, loading, target, grown, fixed]
    assert [a["project"]["revision"] for a in answers] == [1, 2, 3, 3, 4, 5, 6, 7]
    assert all(a["results"]["revision"] == a["project"]["revision"] for a in answers)
    assert {a["project"]["open_id"] for a in answers} == {1}
    refused = client.refused("PUT", f"/api/boxes/{fixed['band_id']}/lane", {"lane_index": 0})
    assert refused[1] == "lane_occupied"

    state = client.ok("GET", "/api/project")["project"]
    assert (state["open_id"], state["revision"]) == (1, 7)  # the refusal committed nothing
    proteins = {protein["name"]: protein for protein in state["proteins"]}
    assert proteins["β-catenin"]["loading_control_ids"] == [loading["protein_id"]]
    assert proteins["β-catenin"]["expected_mw"] == 92.5
    tubulin = proteins["α-tubulin"]
    assert (tubulin["loading_control_ids"], tubulin["expected_mw"]) == ([], None)
    assert [band["source"] for band in proteins["β-catenin"]["bands"]] == ["click", "manual"]

    # Every create, and every open of another project, is a new open id; opening
    # the project already open answers its own (#93).
    assert client.ok("POST", "/api/projects", {"name": "Other"})["project"]["open_id"] == 2
    for name in ("Blot", "BLOT"):
        answer = client.ok("POST", "/api/projects/open", {"name": name})
        assert (answer["project"]["open_id"], answer["project"]["revision"]) == (3, 7)
        assert (answer["results"]["open_id"], answer["results"]["revision"]) == (3, 7)


def test_every_project_answer_carries_its_results(client, tmp_path):
    answers = {"POST /api/projects": client.ok("POST", "/api/projects", {"name": "Blot"})}
    answers["GET /api/project"] = client.ok("GET", "/api/project")
    _, answers["POST /api/images"] = upload(client, blot_bytes(tmp_path))
    image_id = answers["POST /api/images"]["image_id"]
    _, second = upload(client, blot_bytes(tmp_path), name="second.tif")
    answers["PUT /api/images/{image_id}/polarity"] = client.ok(
        "PUT", f"/api/images/{image_id}/polarity", {"polarity": "light_on_dark"}
    )
    lanes = [{"condition": "vehicle"}, {"condition": "10 µM"}, {"condition": "10 µM"}]
    answers["PUT /api/lanes"] = client.ok("PUT", "/api/lanes", {"lanes": lanes})
    body = {"name": "β-catenin", "role": "target", "image_id": image_id, "box_size": SIZE}
    answers["POST /api/proteins"] = client.ok("POST", "/api/proteins", body)
    protein = answers["POST /api/proteins"]["protein_id"]
    body = {"protein_id": protein, "x": LANE_X[0], "y": ROW, "lane_index": 0}
    answers["POST /api/boxes"] = client.ok("POST", "/api/boxes", body)
    band = answers["POST /api/boxes"]["band_id"]
    answers["PUT /api/proteins/{protein_id}/box-padding"] = client.ok(
        "PUT", f"/api/proteins/{protein}/box-padding", {"along": 1}
    )
    x0, y0, x1, y1 = bands(answers["PUT /api/proteins/{protein_id}/box-padding"])[band]["rect"]
    answers["PUT /api/boxes/{band_id}"] = client.ok(
        "PUT", f"/api/boxes/{band}", {"rect": [x0 + 2, y0, x1 + 2, y1]}
    )
    answers["PUT /api/boxes/{band_id}/lane"] = client.ok(
        "PUT", f"/api/boxes/{band}/lane", {"lane_index": 1}
    )
    answers["DELETE /api/boxes/{band_id}"] = client.ok("DELETE", f"/api/boxes/{band}")
    answers["PUT /api/reference"] = client.ok("PUT", "/api/reference", {"condition": "vehicle"})
    answers["PATCH /api/proteins/{protein_id}"] = client.ok(
        "PATCH", f"/api/proteins/{protein}", {"expected_mw": 92.5}
    )
    answers["PUT /api/proteins/{protein_id}/box-size"] = client.ok(
        "PUT", f"/api/proteins/{protein}/box-size", {"width": 16, "height": 12}
    )
    answers["POST /api/undo"] = client.ok("POST", "/api/undo")
    answers["POST /api/redo"] = client.ok("POST", "/api/redo")
    # A row box over the three declared lanes, on the image that kept the blot's polarity.
    body = {"name": "GAPDH", "role": "loading control", "image_id": second["image_id"]}
    other = client.ok("POST", "/api/proteins", body)["protein_id"]
    body = {"protein_id": other, "rect": [15, ROW - 12, LANE_X[2] + 35, ROW + 12]}
    answers["POST /api/boxes/row"] = client.ok("POST", "/api/boxes/row", body)
    plant_records(client, protein, _record(1), bands=1)
    answers["DELETE /api/proteins/{protein_id}/boxes"] = client.ok(
        "DELETE", f"/api/proteins/{protein}/boxes"
    )
    plant_records(client, protein, _record(1), bands=1)
    answers["DELETE /api/proteins/{protein_id}/undetected/{lane_index}"] = client.ok(
        "DELETE", f"/api/proteins/{protein}/undetected/1"
    )
    answers["DELETE /api/proteins/{protein_id}"] = client.ok("DELETE", f"/api/proteins/{protein}")
    answers["DELETE /api/images/{image_id}"] = client.ok(
        "DELETE", f"/api/images/{second['image_id']}"
    )
    answers["POST /api/requantify"] = client.ok("POST", "/api/requantify")  # a no-op here
    answers["POST /api/export"] = client.ok("POST", "/api/export", {"formats": ["svg"]})
    answers["POST /api/projects/open"] = client.ok("POST", "/api/projects/open", {"name": "Blot"})
    # Another project, so another opening, whose revisions start again.
    answers["POST /api/projects/sample"] = client.ok("POST", "/api/projects/sample")
    # And another: images handed to the app, imported into a new project.
    handoff_id, files = handed_off(client, tmp_path, ["handed.tif"])
    answers["POST /api/handoffs/{handoff_id}/accept"] = accept(client, handoff_id, files)

    revisions = []
    blot = answers["POST /api/projects"]["project"]["open_id"]
    for route, answer in answers.items():
        project, results = answer["project"], answer["results"]
        assert (results["open_id"], results["revision"]) == (
            project["open_id"],
            project["revision"],
        ), route
        assert {"lanes", "proteins", "sets", "settings"} <= set(results), route
        if project["open_id"] == blot:
            revisions.append(project["revision"])
    assert len(revisions) == len(answers) - 2
    assert revisions == sorted(revisions)
    # Every route that answers with the project is exercised above.
    others = {
        "GET /api/projects",
        "GET /api/workspace",
        "POST /api/project/reveal",
        "GET /api/images/{image_id}/preview",
        "GET /api/charts/{key}.svg",
        "POST /api/incoming",
        "POST /api/handoffs",
        "POST /api/handoffs/{handoff_id}/discard",
        "GET /api/diagnostics",
        "POST /api/diagnostics",
        "POST /api/diagnostics/reveal",
    }
    routes = {f"{method} {route.path}" for route in api.router.routes for method in route.methods}
    assert routes - others == set(answers)


def _counting(monkeypatch, calls: list) -> None:
    """Record each computation of the results: its session's folder and settings."""
    compute_view = api.ops.compute_view

    def counting(session, **settings):
        calls.append((session.folder.name, settings))
        return compute_view(session, **settings)

    monkeypatch.setattr(api.ops, "compute_view", counting)


def test_a_repeated_read_reuses_the_results_of_its_revision(client, monkeypatch):
    calls: list = []
    _counting(monkeypatch, calls)
    created = client.ok("POST", "/api/projects", {"name": "Blot"})
    assert client.ok("GET", "/api/project") == created
    defaults = {"plot_conditions": None, "error_type": ErrorType.SD, "method": ReduceMethod.MEAN}
    assert calls == [("Blot", defaults)]

    lanes = {"lanes": [{"condition": "vehicle"}]}
    client.ok("PUT", "/api/lanes", lanes)  # a new revision
    client.ok("PUT", "/api/lanes", lanes)  # a no-op: the same revision
    client.ok("GET", "/api/project")
    assert len(calls) == 2
    client.ok("POST", "/api/projects/open", {"name": "Blot"})  # the open project: its session
    assert len(calls) == 2


def test_a_request_keeps_the_open_id_of_the_project_it_started_with(tmp_path, monkeypatch):
    calls: list = []
    _counting(monkeypatch, calls)
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    second = workspace.create("B")  # a switch while a request on A still runs
    open_id, view = workspace.view(first)
    assert open_id == 1 and view.project is first.project
    assert workspace.view(second)[0] == 2
    assert workspace.view(first)[0] == 1  # A's results were not kept in place of B's
    assert workspace.view(second)[1].project is second.project
    assert [name for name, _ in calls] == ["A µ", "B", "A µ"]
    assert workspace.view(workspace.open("A µ"))[0] == 3
    # Both halves of a late answer on A carry A's open id, so a client drops it whole.
    answer = api._answer(workspace, first)
    assert answer["project"]["name"] == "A µ"
    assert (answer["project"]["open_id"], answer["results"]["open_id"]) == (1, 1)


def test_a_commit_while_the_results_are_computed_splits_no_answer(tmp_path, monkeypatch):
    # compute_view takes no lock, so another request may commit while it runs: the
    # answer describes the one project it computed from, and those results are not
    # kept as the committed revision's.
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    compute_view = api.ops.compute_view
    committed = threading.Event()

    def commit_meanwhile(current, **settings):
        view = compute_view(current, **settings)
        if not committed.is_set():  # during the first computation only
            lanes = [api.ops.LaneInput("vehicle"), api.ops.LaneInput("10 µM")]
            other = threading.Thread(target=api.ops.set_lanes, args=(current, lanes))
            other.start()
            other.join(10)
            committed.set()
        return view

    monkeypatch.setattr(api.ops, "compute_view", commit_meanwhile)
    first = api._answer(workspace, session)
    assert revision(session.project) == 2  # the commit landed
    halves = (first["project"], first["results"])
    assert [half["revision"] for half in halves] == [1, 1]
    assert [half["lanes"] for half in halves] == [[], []]

    second = api._answer(workspace, session)
    halves = (second["project"], second["results"])
    assert [half["revision"] for half in halves] == [2, 2]
    conditions = [[lane["condition"] for lane in half["lanes"]] for half in halves]
    assert conditions == [["vehicle", "10 µM"]] * 2


def test_results_that_finish_late_do_not_replace_those_of_a_later_revision(tmp_path, monkeypatch):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    compute_view = api.ops.compute_view
    computed: list[int] = []  # the revision of each computation
    paused, resume = threading.Event(), threading.Event()

    def first_finishes_last(current, **settings):
        view = compute_view(current, **settings)
        computed.append(revision(view.project))
        if len(computed) == 1:
            paused.set()
            resume.wait(10)
        return view

    monkeypatch.setattr(api.ops, "compute_view", first_finishes_last)
    late = threading.Thread(target=workspace.view, args=(session,))  # a read of revision 1
    late.start()
    assert paused.wait(10)
    api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])  # another request: revision 2
    assert workspace.view(session)[1].project is session.project
    resume.set()
    late.join(10)
    assert not late.is_alive()

    _, view = workspace.view(session)  # revision 2 again: its results were kept
    assert computed == [1, 2]
    assert [lane.condition for lane in view.results.lanes] == ["vehicle"]


# --- The reference, protein edits, removals and box size (#52) ---

CASCADE = [field.name for field in dataclasses.fields(api.ops.Cascade)]


def unchanged_refusal(client: Client, method: str, path: str, body: Any = None) -> tuple[str, list]:
    """A refusal's code and ids, checked to have changed nothing: the project, its
    revision and its results answer as before."""
    before = client.ok("GET", "/api/project")
    status, code, ids = client.refused(method, path, body)
    assert status in (404, 422), (status, code)
    assert client.ok("GET", "/api/project") == before
    return code, ids


def logged(client: Client) -> list[str]:
    """The actions in the saved log of the project "Blot"."""
    return [entry.action for entry in storage.load_project(client.root / "Blot").log]


def protein_of(answer: dict, protein_id: str) -> dict:
    """A protein of the answer's project state."""
    return next(p for p in answer["project"]["proteins"] if p["id"] == protein_id)


def notice_codes(answer: dict) -> set[str]:
    return {notice["code"] for s in answer["results"]["sets"] for notice in s["notices"]}


def test_a_box_size_change_recentres_every_box_and_changes_the_nets(client, tmp_path):
    target, loading, before = live(client, tmp_path, DOSES)
    old = {band["id"]: band["rect"] for band in protein_of(before, target)["bands"]}
    path = f"/api/proteins/{target}/box-size"
    answer = client.ok("PUT", path, {"width": 20, "height": 16})
    state = protein_of(answer, target)
    assert state["box_size"] == {"width": 20, "height": 16}
    assert {band["id"] for band in state["bands"]} == set(old)
    for band in state["bands"]:
        x0, y0, x1, y1 = band["rect"]
        ox0, oy0, ox1, oy1 = old[band["id"]]
        assert (x1 - x0, y1 - y0) == (20, 16)
        assert ((x0 + x1) // 2, (y0 + y1) // 2) == ((ox0 + ox1) // 2, (oy0 + oy1) // 2)
    # A larger box takes in more of each band's tails: every net grows, and the chart moves.
    grown = zip(column(answer, target)["nets"], column(before, target)["nets"], strict=True)
    assert all(new > old for new, old in grown)
    assert column(answer, loading)["nets"] == column(before, loading)["nets"]
    assert only_series(answer)["chart"] != only_series(before)["chart"]
    assert answer["project"]["revision"] == before["project"]["revision"] + 1
    assert logged(client)[-1] == "set_box_size"

    same = client.ok("PUT", path, {"width": 20, "height": 16})  # a no-op: no log entry
    assert same == client.ok("GET", "/api/project")
    assert same["project"]["revision"] == answer["project"]["revision"]
    assert logged(client).count("set_box_size") == 1
    body = {"width": 20, "height": 16}
    assert unchanged_refusal(client, "PUT", "/api/proteins/prot-99/box-size", body) == (
        "unknown_id",
        [],
    )


@pytest.mark.parametrize(
    ("size", "code"),
    [
        ({"width": 80, "height": 10}, "size_would_overlap"),  # wider than the lane pitch, 70
        ({"width": W + 1, "height": 10}, "size_out_of_bounds"),
        ({"width": 14, "height": TWO_ROW_H + 1}, "size_out_of_bounds"),
    ],
)
def test_a_box_size_that_does_not_fit_changes_nothing(client, tmp_path, size, code):
    target, _, _ = live(client, tmp_path, DOSES)
    path = f"/api/proteins/{target}/box-size"
    assert unchanged_refusal(client, "PUT", path, size) == (code, [])


@pytest.mark.parametrize(
    "body",
    [
        {"width": 0, "height": 10},
        {"width": 20, "height": -1},
        {"width": "20", "height": 10},  # a number as text
        {"width": 20.0, "height": 10},
        {"width": True, "height": 10},
        {"width": 20},
        {"width": 20, "height": 10, "depth": 1},  # an unknown field
        [20, 10],  # the form POST /api/proteins takes, not this route's
    ],
)
def test_a_malformed_box_size_is_refused(client, tmp_path, body):
    _, protein = ready(client, tmp_path)
    path = f"/api/proteins/{protein}/box-size"
    assert unchanged_refusal(client, "PUT", path, body) == ("invalid_input", [])


def test_the_box_size_route_takes_the_fitted_size_the_state_shows(client, tmp_path):
    # A padding (#57) makes every box the fitted size plus the padding on each
    # side. The state carries both sizes, and the route takes the fitted one:
    # the size the page shows, sent back as it is, changes nothing (it is not
    # padded again), and a new one is padded once.
    target, _, _ = live(client, tmp_path, DOSES)
    api.ops.set_box_padding(client.workspace.current(), target, across=3, along=2)
    before = client.ok("GET", "/api/project")
    state = protein_of(before, target)
    assert state["fitted_size"] == {"width": 14, "height": 10}  # SIZE
    assert state["box_size"] == {"width": 14 + 2 * 3, "height": 10 + 2 * 2}
    path = f"/api/proteins/{target}/box-size"

    same = client.ok("PUT", path, state["fitted_size"])  # a no-op: no log entry
    assert same == client.ok("GET", "/api/project")
    assert same["project"]["revision"] == before["project"]["revision"]
    assert protein_of(same, target) == state
    assert "set_box_size" not in logged(client)

    answer = client.ok("PUT", path, {"width": 16, "height": 12})
    state = protein_of(answer, target)
    assert state["fitted_size"] == {"width": 16, "height": 12}
    assert state["box_size"] == {"width": 16 + 2 * 3, "height": 12 + 2 * 2}
    for band in state["bands"]:
        x0, y0, x1, y1 = band["rect"]
        assert (x1 - x0, y1 - y0) == (22, 16)


# --- Box padding (#57) ---

# What a padding change answers besides the project and its results: every
# field of PaddingChange, its padding as box_padding (the state's name).
PADDING_FIELDS = {
    "box_padding" if field.name == "padding" else field.name
    for field in dataclasses.fields(api.ops.PaddingChange)
}


def padding_path(protein_id: str) -> str:
    return f"/api/proteins/{protein_id}/box-padding"


def lane_nets(answer: dict, protein_id: str) -> dict[str, float]:
    """A protein's nets in the answer's results, by band id."""
    lanes = column(answer, protein_id)
    return {b: net for b, net in zip(lanes["band_ids"], lanes["nets"], strict=True) if b}


def test_a_box_padding_change_answers_sizes_and_net_change(client, tmp_path):
    # Every box of the protein grows by the padding on each side around its own
    # centre; the fitted size is kept, and the nets take in more of the bands'
    # tails. The answer says how the protein's nets moved, and the state
    # carries every protein's padding.
    target, loading, before = live(client, tmp_path, DOSES)
    assert protein_of(before, target)["box_padding"] == {"across": 0, "along": 0}
    old = {band["id"]: band["rect"] for band in protein_of(before, target)["bands"]}
    answer = client.ok("PUT", padding_path(target), {"along": 2})
    assert set(answer) - {"project", "results"} == PADDING_FIELDS
    state = protein_of(answer, target)
    assert state["box_padding"] == answer["box_padding"] == {"across": 0, "along": 2}
    assert state["fitted_size"] == answer["fitted_size"] == {"width": 14, "height": 10}  # SIZE
    assert state["box_size"] == answer["box_size"] == {"width": 14, "height": 10 + 2 * 2}
    for band in state["bands"]:
        x0, y0, x1, y1 = old[band["id"]]
        assert band["rect"] == [x0, y0 - 2, x1, y1 + 2]
    assert protein_of(answer, loading)["box_padding"] == {"across": 0, "along": 0}
    new, was = lane_nets(answer, target), lane_nets(before, target)
    changes = [new[band] / was[band] - 1 for band in was]
    assert min(changes) > 0
    assert answer["net_change"] == pytest.approx([min(changes), max(changes)])
    assert (answer["edge_shifted"], answer["overlapping"]) == ([], [])
    new, was = lane_nets(answer, loading), lane_nets(before, loading)
    changed = [
        {"band_id": band, "net_before": was[band], "net_after": new[band]}
        for band in was
        if new[band] != was[band]
    ]
    assert answer["remeasured"] == changed
    assert answer["project"]["revision"] == before["project"]["revision"] + 1
    assert logged(client)[-1] == "set_box_padding"

    for body in ({"along": 2}, {"across": 0, "along": 2}, {}):  # the same: a no-op
        same = client.ok("PUT", padding_path(target), body)
        assert same["project"] == client.ok("GET", "/api/project")["project"]
        assert same["project"]["revision"] == answer["project"]["revision"]
        assert (same["box_padding"], same["net_change"]) == ({"across": 0, "along": 2}, None)
    assert logged(client).count("set_box_padding") == 1


def test_a_box_padding_field_left_out_keeps_its_value(client, tmp_path):
    # A page sends only the direction it changed: a tab that still shows the
    # padding before another tab set the other direction does not reset it.
    target, _, _ = live(client, tmp_path, DOSES)
    client.ok("PUT", padding_path(target), {"across": 1, "along": 2})
    answer = client.ok("PUT", padding_path(target), {"across": 0})
    assert protein_of(answer, target)["box_padding"] == {"across": 0, "along": 2}
    answer = client.ok("PUT", padding_path(target), {"along": 3})
    assert protein_of(answer, target)["box_padding"] == {"across": 0, "along": 3}
    assert protein_of(answer, target)["box_size"] == {"width": 14, "height": 10 + 2 * 3}


def test_a_box_padding_reports_boxes_the_edge_shifts_and_boxes_it_overlaps(client, tmp_path):
    # A fitted height of 60 puts the target's boxes against the image's top
    # edge (their centre, y 30, is 30 px down), so 6 px more above and below
    # shifts them down, onto 2 of the 10 rows of the loading control's boxes
    # (y 70 to 80): each box then counts part of the other's band, which is
    # allowed and said (more than half of the smaller box is refused, #114).
    target, loading, _ = live(client, tmp_path, DOSES)
    client.ok("PUT", f"/api/proteins/{target}/box-size", {"width": 14, "height": 60})
    answer = client.ok("PUT", padding_path(target), {"along": 6})
    state = protein_of(answer, target)
    assert all(band["rect"][1::2] == [0, 72] for band in state["bands"])
    assert answer["edge_shifted"] == [band["id"] for band in state["bands"]]
    assert answer["overlapping"] == [band["id"] for band in protein_of(answer, loading)["bands"]]


def test_a_box_padding_answers_the_other_proteins_nets_it_changed(client, tmp_path):
    # The loading control boxed just above two of the target's bands: every
    # ring on the image leaves out the target's boxes, so padding them left
    # and right changes those two nets, which the answer names as a row box
    # does (#124), with the largest change.
    target, loading, _ = live(client, tmp_path, DOSES)
    client.ok("DELETE", f"/api/proteins/{loading}/boxes")
    for lane in (1, 2):
        y = TARGET_ROW - 12
        body = {"protein_id": loading, "x": LANE_X[lane], "y": y, "lane_index": lane}
        before = client.ok("POST", "/api/boxes", body)
    answer = client.ok("PUT", padding_path(target), {"across": 7})  # half of 14
    old, new = lane_nets(before, loading), lane_nets(answer, loading)
    assert len(old) == 2
    changed = [
        {"band_id": band_id, "net_before": old[band_id], "net_after": new[band_id]}
        for band_id in old
        if new[band_id] != old[band_id]
    ]
    assert changed and answer["remeasured"] == changed
    shares = {
        c["band_id"]: abs(c["net_after"] - c["net_before"]) / c["net_before"] for c in changed
    }
    largest = max(shares, key=shares.__getitem__)
    assert answer["largest_change"] == {"band_id": largest, "change": shares[largest]}


def _no_box_yet(client: Client, tmp_path: Path) -> str:
    return ready(client, tmp_path)[1]


def _live_target(client: Client, tmp_path: Path) -> str:
    return live(client, tmp_path, DOSES)[0]


def _wide_target(client: Client, tmp_path: Path) -> str:
    """The target's boxes 60 px wide, and 60 px tall, on lanes 70 px apart."""
    target = live(client, tmp_path, DOSES)[0]
    client.ok("PUT", f"/api/proteins/{target}/box-size", {"width": 60, "height": 60})
    return target


@pytest.mark.parametrize(
    ("setup", "body", "code", "words"),
    [
        (_no_box_yet, {"along": 2}, "invalid_input", "place a box of 'β-actin' first"),
        (_live_target, {"along": 6}, "invalid_input", "at most 5 px"),  # half of 10
        (_live_target, {"across": 8}, "invalid_input", "at most 7 px"),  # half of 14
        (_wide_target, {"across": 6}, "size_would_overlap", "at most 5 px fits"),
        (_wide_target, {"along": 21}, "size_out_of_bounds", "exceed"),  # 102 on 100
    ],
)
def test_a_box_padding_that_does_not_fit_changes_nothing(
    client, tmp_path, setup, body, code, words
):
    protein = setup(client, tmp_path)
    before = client.ok("GET", "/api/project")
    status, payload = client.call("PUT", padding_path(protein), body)
    assert (status, payload["code"]) == (422, code)
    assert words in payload["message"]
    assert client.ok("GET", "/api/project") == before
    boxes = [band["id"] for band in protein_of(before, protein)["bands"]]
    assert payload["ids"] == (boxes if code == "size_would_overlap" else [])
    assert unchanged_refusal(client, "PUT", padding_path("prot-99"), body) == ("unknown_id", [])


@pytest.mark.parametrize(
    "body",
    [
        {"along": -1},
        {"across": -1, "along": 2},
        {"along": 2.0},
        {"along": "2"},  # a number as text
        {"along": True},
        {"along": None},  # left out is how a direction is kept
        {"along": 2, "depth": 1},  # an unknown field
        [0, 2],
    ],
)
def test_a_malformed_box_padding_is_refused(client, tmp_path, body):
    target, _, _ = live(client, tmp_path, DOSES)  # it has boxes: a read body would be applied
    assert unchanged_refusal(client, "PUT", padding_path(target), body) == ("invalid_input", [])


def test_a_box_padding_from_a_page_showing_another_opening_changes_nothing(client, tmp_path):
    # #134: the page names the opening it shows; a padding asked for from a
    # page still showing Other is refused before anything is done in Blot.
    answer, shown = opened_elsewhere(client, tmp_path)
    project = answer["project"]
    target = next(p["id"] for p in project["proteins"] if p["role"] == "target")
    status, payload = client.call(
        "PUT", padding_path(target), {"along": 2}, headers={OPENING: str(shown)}
    )
    assert (status, payload["code"]) == (409, "project_changed")
    now = client.ok("GET", "/api/project")
    assert now["project"]["revision"] == project["revision"]
    assert protein_of(now, target)["box_padding"] == {"across": 0, "along": 0}
    named = {OPENING: str(project["open_id"])}
    done = client.ok("PUT", padding_path(target), {"along": 2}, headers=named)
    assert protein_of(done, target)["box_padding"] == {"across": 0, "along": 2}


def test_a_seed_click_grows_a_box_over_the_band_and_answers_its_net(client, tmp_path):
    _, protein = ready(client, tmp_path)
    cx = LANE_X[2]
    body = {"protein_id": protein, "x": cx + 2, "y": ROW - 1, "lane_index": 2, "grow": True}
    answer = client.ok("POST", "/api/boxes", body)
    band = bands(answer)[answer["band_id"]]
    assert band["source"] == "click"
    x0, y0, x1, y1 = band["rect"]
    assert x0 <= cx - 5 and cx + 5 <= x1  # the band's 1/e half-widths are 5 and 3
    assert y0 <= ROW - 3 and ROW + 3 <= y1
    results = column(answer, protein)
    assert results["band_ids"][2] == answer["band_id"]
    assert results["nets"][2] > 0 and results["detected"][2] is True

    between = (LANE_X[0] + LANE_X[1]) // 2
    background = {**body, "x": between, "y": 5, "lane_index": 0}  # the flat membrane
    assert unchanged_refusal(client, "POST", "/api/boxes", background) == ("no_band_found", [])


def test_the_reference_switches_the_series_to_fold_change_and_back(client, tmp_path):
    _, _, before = live(client, tmp_path, DOSES)
    series = only_series(before)
    assert before["results"]["sets"][0]["tier"] == "normalized"
    assert (series["value_kind"], series["chart"]["y_label"]) == (
        "loading_normalized",
        "Normalized signal (target / loading)",
    )

    answer = client.ok("PUT", "/api/reference", {"condition": "vehicle"})
    assert answer["project"]["reference_condition"] == "vehicle"
    assert answer["results"]["reference_condition"] == "vehicle"
    (result_set,) = answer["results"]["sets"]
    series = only_series(answer)
    assert (result_set["tier"], series["value_kind"], series["chart"]["value_kind"]) == (
        "fold_change",
        "fold_change",
        "fold_change",
    )
    assert series["chart"]["y_label"] == "Fold change vs control"
    assert bar(series, "vehicle")["mean"] == pytest.approx(1.0)
    assert logged(client)[-1] == "set_reference_condition"

    cleared = client.ok("PUT", "/api/reference", {"condition": None})
    assert cleared["project"]["reference_condition"] is None
    assert cleared["results"]["sets"] == before["results"]["sets"]
    assert cleared["project"]["revision"] == before["project"]["revision"] + 2

    # Greek mu names the lanes' micro sign: the lanes' own spelling is stored.
    answer = client.ok("PUT", "/api/reference", {"condition": "10 μM"})
    assert answer["project"]["reference_condition"] == "10 µM"


def test_renaming_every_lane_of_the_reference_keeps_it_as_the_reference(client, tmp_path):
    # The lane table renames the reference condition in every lane of it, and
    # sends the new name as the reference in the same change.
    _, _, before = live(client, tmp_path, DOSES, reference="vehicle")
    renamed = {"lanes": [{"condition": c} for c in ["DMSO", "DMSO", *DOSES[2:]]]}
    answer = client.ok("PUT", "/api/lanes", {**renamed, "reference_condition": "DMSO"})
    assert answer["reference_cleared"] is False
    assert answer["project"]["reference_condition"] == "DMSO"
    assert answer["results"]["reference_condition"] == "DMSO"
    series = only_series(answer)
    assert (answer["results"]["sets"][0]["tier"], series["value_kind"]) == (
        "fold_change",
        "fold_change",
    )
    assert bar(series, "DMSO")["mean"] == pytest.approx(1.0)
    assert series["fold_change"] == only_series(before)["fold_change"]  # the same lanes
    assert answer["project"]["revision"] == before["project"]["revision"] + 1  # one change
    assert logged(client)[-1] == "set_lanes"

    # Left out, the reference names no lane after the rename: cleared, and said so.
    client.ok("POST", "/api/undo")
    answer = client.ok("PUT", "/api/lanes", renamed)
    assert answer["reference_cleared"] is True
    assert answer["project"]["reference_condition"] is None
    assert answer["results"]["sets"][0]["tier"] == "normalized"


def test_labels_swapped_between_lanes_keep_the_reference_with_its_label(client, tmp_path):
    # The lane table leaves reference_condition out when the old reference
    # still names a lane after the edit: the server keeps it, on those lanes.
    _, _, before = live(client, tmp_path, DOSES, reference="vehicle")
    swapped = ["10 µM", "10 µM", "vehicle", "vehicle", "vehicle"]
    answer = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": c} for c in swapped]})
    assert answer["reference_cleared"] is False
    assert answer["project"]["reference_condition"] == "vehicle"
    series = only_series(answer)
    assert series["value_kind"] == "fold_change"
    assert bar(series, "vehicle")["mean"] == pytest.approx(1.0)  # lanes 3-5 now
    assert series["fold_change"] != only_series(before)["fold_change"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"condition": "20 µM"}, "unknown_condition"),
        ({"condition": "C0"}, "unknown_condition"),  # conditions keep their case
        ({"condition": "  "}, "blank_text"),
        ({"condition": ""}, "blank_text"),  # not a way to clear the reference
        ({"condition": "c\u00070"}, "control_character"),
        ({"condition": 0}, "invalid_input"),
        ({}, "invalid_input"),  # clearing the reference takes an explicit null
        ({"condition": "c0", "lane": 0}, "invalid_input"),
    ],
)
def test_a_reference_that_names_no_lane_is_refused(client, tmp_path, body, code):
    ready(client, tmp_path)
    assert unchanged_refusal(client, "PUT", "/api/reference", body) == (code, [])


def test_a_protein_edit_keeps_the_fields_it_leaves_out(client, tmp_path):
    target, loading, _ = live(client, tmp_path, DOSES)
    path = f"/api/proteins/{target}"

    def fields(answer: dict) -> tuple:
        state = protein_of(answer, target)
        return state["name"], state["role"], state["expected_mw"], state["loading_control_ids"]

    answer = client.ok("PATCH", path, {"expected_mw": 92.5, "loading_control_ids": [loading]})
    assert fields(answer) == ("β-catenin", "target", 92.5, [loading])
    answer = client.ok("PATCH", path, {"name": "  β-catenin  µ "})  # stored cleaned
    assert fields(answer) == ("β-catenin µ", "target", 92.5, [loading])
    answer = client.ok("PATCH", path, {"expected_mw": None})  # null is a value: no expected MW
    assert fields(answer) == ("β-catenin µ", "target", None, [loading])
    answer = client.ok("PATCH", path, {"loading_control_ids": []})  # the only one, implicitly
    assert fields(answer) == ("β-catenin µ", "target", None, [])
    revision_before = answer["project"]["revision"]
    assert client.ok("PATCH", path, {})["project"]["revision"] == revision_before  # a no-op
    assert logged(client)[-4:] == ["edit_protein"] * 4

    answer = client.ok("PATCH", path, {"role": "loading control"})
    assert fields(answer) == ("β-catenin µ", "loading control", None, [])
    assert answer["results"]["sets"][0]["tier"] == "export_only"  # no target is left


def test_a_protein_edit_is_refused_with_the_operations_codes(client, tmp_path):
    target, loading, _ = live(client, tmp_path, DOSES)
    edit_target, edit_loading = f"/api/proteins/{target}", f"/api/proteins/{loading}"
    cases: list[tuple[str, Any, tuple[str, list]]] = [
        # Greek capital alpha: the same name, ignoring case and look-alikes.
        (edit_target, {"name": "Α-Tubulin"}, ("duplicate_name", [loading])),
        (edit_target, {"name": "Lane"}, ("reserved_name", [])),
        (edit_target, {"name": "α-tubulin clipped"}, ("reserved_name", [loading])),
        (edit_target, {"name": " \t "}, ("blank_text", [])),
        (edit_target, {"name": "β\u0007"}, ("control_character", [])),
        # β-catenin normalizes to it as the batch's only loading control.
        (edit_loading, {"role": "target"}, ("loading_control_in_use", [target])),
        (edit_target, {"role": "enzyme"}, ("invalid_input", [])),
        (edit_target, {"name": None}, ("invalid_input", [])),  # only expected_mw takes null
        (edit_target, {"loading_control_ids": [target]}, ("invalid_input", [target])),
        (edit_target, {"loading_control_ids": "prot-1"}, ("invalid_input", [])),
        (edit_target, {"expected_mw": 0}, ("invalid_input", [])),
        (edit_target, {"colour": "red"}, ("invalid_input", [])),  # an unknown field
        (edit_target, {"loading_control_ids": ["prot-99"]}, ("unknown_id", [])),
        ("/api/proteins/prot-99", {"name": "GAPDH"}, ("unknown_id", [])),
    ]
    for path, body, refusal in cases:
        assert unchanged_refusal(client, "PATCH", path, body) == refusal, body


@pytest.mark.parametrize("mw", [True, "42", "42.5", [42]])
def test_an_expected_mw_that_is_not_a_number_is_refused(client, tmp_path, mw):
    image_id, protein = ready(client, tmp_path)
    body = {"name": "GAPDH", "role": "loading control", "image_id": image_id, "expected_mw": mw}
    assert unchanged_refusal(client, "POST", "/api/proteins", body) == ("invalid_input", [])
    path = f"/api/proteins/{protein}"
    assert unchanged_refusal(client, "PATCH", path, {"expected_mw": mw}) == ("invalid_input", [])


def test_an_expected_mw_may_be_a_whole_number(client, tmp_path):
    image_id, protein = ready(client, tmp_path)
    body = {"name": "GAPDH", "role": "loading control", "image_id": image_id, "expected_mw": 36}
    added = client.ok("POST", "/api/proteins", body)
    assert protein_of(added, added["protein_id"])["expected_mw"] == 36.0
    edited = client.ok("PATCH", f"/api/proteins/{protein}", {"expected_mw": 92})
    assert protein_of(edited, protein)["expected_mw"] == 92.0


def test_removing_a_protein_answers_its_cascade_and_drops_its_series(client, tmp_path):
    target, loading, before = live(client, tmp_path, DOSES)
    assert len(before["results"]["sets"][0]["series"]) == 1
    band_ids = [band["id"] for band in protein_of(before, loading)["bands"]]

    answer = client.ok("DELETE", f"/api/proteins/{loading}")
    assert set(answer) == {*CASCADE, "project", "results"}
    assert {name: answer[name] for name in CASCADE} == {
        "removed": [loading, *band_ids],
        "detached_targets": [target],  # it used the batch's only loading control
        "unpaired_images": [],
        "unfitted_membranes": [],
    }
    assert [p["id"] for p in answer["project"]["proteins"]] == [target]
    assert [p["protein_id"] for p in answer["results"]["proteins"]] == [target]
    (result_set,) = answer["results"]["sets"]
    assert (result_set["tier"], result_set["series"]) == ("export_only", [])
    assert logged(client)[-1] == "remove_protein"
    assert unchanged_refusal(client, "DELETE", f"/api/proteins/{loading}") == ("unknown_id", [])


def test_removing_an_image_answers_the_whole_cascade(client, tmp_path):
    target, loading, before = live(client, tmp_path, DOSES)
    (image,) = before["project"]["images"]
    _, other = upload(client, blot_bytes(tmp_path), name="γ-actin.tif")  # on its own membrane
    body = {"name": "γ-actin", "role": "target", "image_id": other["image_id"]}
    gamma = client.ok("POST", "/api/proteins", body)["protein_id"]
    band_ids = [band["id"] for p in (loading, target) for band in protein_of(before, p)["bands"]]

    answer = client.ok("DELETE", f"/api/images/{image['id']}")
    assert set(answer) == {*CASCADE, "project", "results"}
    assert {name: answer[name] for name in CASCADE} == {
        "removed": [image["id"], loading, target, *band_ids, image["membrane_id"]],
        "detached_targets": [gamma],  # it used α-tubulin, the only loading control
        "unpaired_images": [],
        "unfitted_membranes": [],
    }
    assert [i["id"] for i in answer["project"]["images"]] == [other["image_id"]]
    assert [p["id"] for p in answer["project"]["proteins"]] == [gamma]
    path = f"/api/images/{image['id']}"
    assert unchanged_refusal(client, "DELETE", path) == ("unknown_id", [])


def test_removing_a_marker_image_answers_the_pairing_and_the_fit_it_undid(client):
    # The conftest sample: img-3 is img-2's marker image and holds two of mem-1's
    # three calibration points, so mem-1 loses its fit.
    project = make_project()
    folder = client.root / "Sample µ"
    write_image_files(folder, project)
    storage.save_project(project, folder)
    before = client.ok("POST", "/api/projects/open", {"name": "Sample µ"})

    answer = client.ok("DELETE", "/api/images/img-3")
    assert {name: answer[name] for name in CASCADE} == {
        "removed": ["img-3"],
        "detached_targets": [],
        "unpaired_images": ["img-2"],
        "unfitted_membranes": ["mem-1"],
    }
    assert [i["id"] for i in answer["project"]["images"]] == ["img-2", "img-4", "img-6"]
    assert answer["project"]["revision"] == before["project"]["revision"] + 1
    batch = storage.load_project(folder).batch
    assert batch.find_image("img-2").marker_image_id is None
    (membrane,) = (m for m in batch.membranes if m.id == "mem-1")
    assert membrane.calibration.fit_quality is None


def test_removing_a_not_detected_record_leaves_its_lane_not_measured(client, tmp_path):
    _, protein = ready(client, tmp_path)
    for lane in (0, 1):
        body = {"protein_id": protein, "x": LANE_X[lane], "y": ROW, "lane_index": lane}
        client.ok("POST", "/api/boxes", {**body, "grow": lane == 0})
    second_band = _record(3, band_index=1, source=ProposalSource.MW_GUIDED)
    plant_records(client, protein, _record(2), second_band, bands=2)
    before = client.ok("GET", "/api/project")
    assert column(before, protein)["detected"] == [True, True, False, None, None]
    assert "below_detection" in notice_codes(before)

    path = f"/api/proteins/{protein}/undetected"
    answer = client.ok("DELETE", f"{path}/2")
    state = protein_of(answer, protein)
    assert [(r["lane_index"], r["band_index"]) for r in state["undetected"]] == [(3, 1)]
    assert 2 in {entry["lane_index"] for entry in state["missing_lanes"]}
    assert column(answer, protein)["detected"] == [True, True, None, None, None]
    assert "below_detection" not in notice_codes(answer)
    assert answer["project"]["revision"] == before["project"]["revision"] + 1

    # Lane 3's record is for the second band: band index 0 finds none there, a no-op.
    assert client.ok("DELETE", f"{path}/3") == answer
    answer = client.ok("DELETE", f"{path}/3?band_index=1")
    assert protein_of(answer, protein)["undetected"] == []
    assert logged(client)[-2:] == ["remove_undetected"] * 2

    for route, code in [
        ("/api/proteins/prot-99/undetected/2", "unknown_id"),
        (f"{path}/5", "lane_out_of_range"),
        (f"{path}/-1", "lane_out_of_range"),
        (f"{path}/two", "invalid_input"),
        (f"{path}/2?band_index=-1", "invalid_input"),
        (f"{path}/2?band_index=first", "invalid_input"),
    ]:
        assert unchanged_refusal(client, "DELETE", route) == (code, []), route


@pytest.mark.parametrize(
    "where",
    [
        "0_2",  # not lane 2: underscores are not digits
        "2.0",
        "+2",
        "%202",  # " 2"
        "2%20",
        "02",
        "-0",
        "%D9%A2",  # the Arabic-Indic digit two
        "2?band_index=0_0",
        "2?band_index=0.0",
        "2?band_index=%2B0",  # "+0"
        "2?band_index=00",
        "2?band_index=-0",
    ],
)
def test_an_index_in_a_url_is_read_only_as_plain_digits(client, tmp_path, where):
    _, protein = ready(client, tmp_path)
    plant_records(client, protein, _record(2), bands=1)
    path = f"/api/proteins/{protein}/undetected/{where}"
    assert unchanged_refusal(client, "DELETE", path) == ("invalid_input", [])


def test_an_unpaired_surrogate_in_typed_text_is_refused_as_json(client, tmp_path):
    # JSON can escape half of a UTF-16 pair ("\ud800"), as a string cut inside a
    # pair gives; no UTF-8 file can store it. json.dumps sends it as that escape.
    image_id, protein = ready(client, tmp_path)
    lanes = [{"condition": f"c{i}"} for i in range(len(LANE_X))]
    add = {"name": "GAPDH\udfff", "role": "loading control", "image_id": image_id}
    cases: list[tuple[str, str, Any, str]] = [
        ("PATCH", f"/api/proteins/{protein}", {"name": "β-actin\ud800"}, "control_character"),
        ("POST", "/api/proteins", add, "control_character"),
        ("PUT", "/api/lanes", {"lanes": [{"condition": "c\ud800"}]}, "control_character"),
        (
            "PUT",
            "/api/lanes",
            {"lanes": [{**lanes[0], "sample": "α\udc00"}, *lanes[1:]]},
            "control_character",
        ),
        ("POST", "/api/projects", {"name": "Blot µ\ud800"}, "invalid_project_name"),
        ("POST", "/api/projects/open", {"name": "Blot\ud800"}, "invalid_project_name"),
    ]
    for method, path, body, code in cases:
        assert unchanged_refusal(client, method, path, body) == (code, []), (path, body)
    # A whole pair is one character, a mathematical bold beta here: stored as typed.
    answer = client.ok("PATCH", f"/api/proteins/{protein}", {"name": "\U0001d6c3-actin"})
    assert protein_of(answer, protein)["name"] == "\U0001d6c3-actin"


# --- Undo, redo and clearing a protein's boxes (#52) ---

RESTORED = ["action", "seq", "removed", "restored", "undetected_removed", "undetected_restored"]


def history(answer: dict) -> dict:
    return answer["project"]["history"]


def test_requantify_answers_the_images_and_every_new_net(client, tmp_path):
    image_id, protein = ready(client, tmp_path)
    for lane in (0, 1, 2):
        body = {"protein_id": protein, "x": LANE_X[lane], "y": ROW, "lane_index": lane}
        placed = client.ok("POST", "/api/boxes", body)
    state = placed["project"]
    assert state["background_method"] == "ring_median_v1"  # a new project measures locally
    assert {band["background_mode"] for band in protein_of(placed, protein)["bands"]} == {
        "symmetric"
    }
    ring_nets = placed["results"]["proteins"][0]["nets"]
    assert client.ok("POST", "/api/requantify")["images"] == []  # already local: a no-op
    assert logged(client)[-1] == "place_box"

    # The same boxes as a project quantified before #83 (a migrated one).
    session = client.workspace.current()

    def legacy(draft: Project) -> None:
        draft.background_method = "global_median"
        api.ops._quantify_image(draft, image_id, session.pixels(image_id))

    with session.transaction():
        project, _ = apply_change(session.project, legacy)
        session._commit(project, action="plant", params={})
    before = client.ok("GET", "/api/project")
    assert before["project"]["background_method"] == "global_median"
    assert "legacy_background" in notice_codes(before)

    answer = client.ok("POST", "/api/requantify")
    assert answer["images"] == [image_id]
    assert answer["project"]["background_method"] == "ring_median_v1"
    assert "legacy_background" not in notice_codes(answer)
    assert answer["results"]["proteins"][0]["nets"] == ring_nets
    assert before["results"]["proteins"][0]["nets"] != ring_nets
    assert history(answer)["undo"] == {"seq": answer["project"]["revision"], "action": "requantify"}
    assert logged(client)[-1] == "requantify"


def test_undo_and_redo_answer_what_they_took_back_and_did_again(client, tmp_path):
    created = client.ok("POST", "/api/projects", {"name": "Blot"})
    assert history(created) == {"undo": None, "redo": None}
    assert unchanged_refusal(client, "POST", "/api/undo") == ("nothing_to_undo", [])
    assert unchanged_refusal(client, "POST", "/api/redo") == ("nothing_to_redo", [])
    assert client.refused("POST", "/api/undo")[:2] == (422, "nothing_to_undo")
    assert client.refused("POST", "/api/redo")[:2] == (422, "nothing_to_redo")

    _, imported = upload(client, blot_bytes(tmp_path))
    assert history(imported) == {"undo": {"seq": 2, "action": "import_image"}, "redo": None}
    lanes = [{"condition": f"c{i}"} for i in range(len(LANE_X))]
    client.ok("PUT", "/api/lanes", {"lanes": lanes})
    body = {"name": "β-actin", "role": "target", "image_id": imported["image_id"]}
    protein = client.ok("POST", "/api/proteins", body)["protein_id"]
    body = {"protein_id": protein, "x": LANE_X[0], "y": ROW, "lane_index": 0, "grow": True}
    placed = client.ok("POST", "/api/boxes", body)
    assert history(placed) == {"undo": {"seq": 5, "action": "place_box"}, "redo": None}
    assert column(placed, protein)["nets"][0] is not None

    undone = client.ok("POST", "/api/undo")
    assert set(undone) == {*RESTORED, "project", "results"}
    assert {name: undone[name] for name in RESTORED} == {
        "action": "place_box",
        "seq": 5,
        "removed": [placed["band_id"]],
        "restored": [],
        "undetected_removed": [],
        "undetected_restored": [],
    }
    assert undone["project"]["revision"] == 6  # an undo is a logged change
    assert history(undone) == {
        "undo": {"seq": 4, "action": "add_protein"},
        "redo": {"seq": 5, "action": "place_box"},
    }
    assert column(undone, protein)["nets"][0] is None  # the results follow
    path = f"/api/boxes/{placed['band_id']}/lane"
    assert unchanged_refusal(client, "PUT", path, {"lane_index": 1})[0] == "unknown_id"

    redone = client.ok("POST", "/api/redo")
    assert {name: redone[name] for name in RESTORED} == {
        "action": "place_box",
        "seq": 5,
        "removed": [],
        "restored": [placed["band_id"]],
        "undetected_removed": [],
        "undetected_restored": [],
    }
    assert history(redone) == {"undo": {"seq": 5, "action": "place_box"}, "redo": None}
    assert redone["project"]["proteins"] == placed["project"]["proteins"]
    assert redone["results"]["proteins"] == placed["results"]["proteins"]
    assert unchanged_refusal(client, "POST", "/api/redo") == ("nothing_to_redo", [])
    assert logged(client)[-2:] == ["undo", "redo"]


def test_clearing_a_proteins_boxes_answers_what_went_and_undo_brings_it_back(client, tmp_path):
    _, protein = ready(client, tmp_path)
    for lane in (0, 1):
        body = {"protein_id": protein, "x": LANE_X[lane], "y": ROW, "lane_index": lane}
        client.ok("POST", "/api/boxes", {**body, "grow": lane == 0})
    plant_records(client, protein, _record(2), bands=1)
    before = client.ok("GET", "/api/project")
    band_ids = [band["id"] for band in protein_of(before, protein)["bands"]]

    answer = client.ok("DELETE", f"/api/proteins/{protein}/boxes")
    assert set(answer) == {"removed", "dropped_undetected", "project", "results"}
    assert (answer["removed"], answer["dropped_undetected"]) == (band_ids, [[2, 0]])
    state = protein_of(answer, protein)
    assert (state["bands"], state["undetected"]) == ([], [])
    assert state["box_size"] == protein_of(before, protein)["box_size"]  # kept
    assert column(answer, protein)["nets"] == [None] * len(LANE_X)
    revision = answer["project"]["revision"]
    assert history(answer)["undo"] == {"seq": revision, "action": "clear_boxes"}
    assert logged(client)[-1] == "clear_boxes"
    path = "/api/proteins/prot-99/boxes"
    assert unchanged_refusal(client, "DELETE", path) == ("unknown_id", [])
    again = client.ok("DELETE", f"/api/proteins/{protein}/boxes")  # nothing left: a no-op
    assert (again["removed"], again["dropped_undetected"]) == ([], [])
    assert again["project"]["revision"] == revision

    undone = client.ok("POST", "/api/undo")
    assert (undone["action"], undone["restored"]) == ("clear_boxes", band_ids)
    assert undone["undetected_restored"] == [[protein, 2, 0]]
    assert undone["project"]["proteins"] == before["project"]["proteins"]
    assert undone["results"]["proteins"] == before["results"]["proteins"]


def blot_upload(session: api.ProjectSession, tmp_path: Path, name: str) -> str:
    with io.BytesIO(blot_bytes(tmp_path)) as stream:
        return api.ops.import_image(
            session, stream, name, kind="chemiluminescence", polarity="dark_on_light"
        )


def test_a_switch_closes_the_old_project_and_deletes_what_only_its_history_kept(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    kept = blot_upload(first, tmp_path, "kept α.tif")
    gone = blot_upload(first, tmp_path, "gone β.tif")
    api.ops.remove_image(first, gone)
    images = first.folder / storage.IMAGES_DIR
    assert sorted(p.name for p in images.iterdir()) == [f"{kept}.tif", f"{gone}.tif"]

    with pytest.raises(api.projects.ProjectExistsError):
        workspace.create("a µ")  # the switch fails: A stays open, and so does its history
    assert workspace.current() is first and first.undo_step is not None
    assert (images / f"{gone}.tif").exists()

    workspace.create("B")
    assert sorted(p.name for p in images.iterdir()) == [f"{kept}.tif"]
    assert (first.undo_step, first.redo_step) == (None, None)
    reopened = workspace.open("A µ")
    assert reopened.project == first.project
    assert reopened.undo_step is None  # the history is per session


@pytest.mark.parametrize(
    "name",
    ["Café µ", "CAFÉ µ", "Café µ", "café μ", "  Café  µ "],
    ids=["exact", "case", "decomposed", "look-alike", "spaces"],
)
def test_reopening_the_open_project_answers_its_open_session(client, tmp_path, monkeypatch, name):
    # Any name that resolves to the open project's folder answers the open
    # session as it is (#93): no second session on that folder, the same open
    # id, revision and results, and its undo history kept.
    client.ok("POST", "/api/projects", {"name": "Café µ"})
    _, imported = upload(client, blot_bytes(tmp_path))
    declared = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    session = client.workspace.current()
    calls: list = []
    _counting(monkeypatch, calls)

    reopened = client.ok("POST", "/api/projects/open", {"name": name})
    assert client.workspace.current() is session
    assert reopened == {"project": declared["project"], "results": declared["results"]}
    assert reopened == client.ok("GET", "/api/project")
    assert calls == []  # the open project's kept results
    assert client.ok("GET", "/api/projects")["open"] == "Café µ"

    undone = client.ok("POST", "/api/undo")
    assert (undone["action"], undone["project"]["open_id"]) == ("set_lanes", 1)
    undone = client.ok("POST", "/api/undo")
    assert undone["action"] == "import_image"
    assert imported["image_id"] in undone["removed"]


def test_opening_another_project_still_switches_and_closes_the_open_one(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    other = workspace.create("B")
    api.ops.set_lanes(other, [api.ops.LaneInput("vehicle")])
    assert other.undo_step is not None

    opened = workspace.open("a μ")  # A, spelled otherwise
    assert workspace.current() is opened and opened not in (first, other)
    assert opened.folder == first.folder
    assert (other.undo_step, other.redo_step) == (None, None)  # closed
    assert workspace.view(opened)[0] == 3
    assert workspace.open("A µ") is opened  # now the open one
    assert workspace.view(opened)[0] == 3


def test_reopening_during_an_edit_leaves_one_session_saving_the_folder(tmp_path):
    # #93: a second session on the open folder let an edit still running on the
    # first save over the project.json of the second and delete the file of an
    # image only the second had stored. The reopen answers the one session now,
    # without waiting for the edit, and the import made after it follows the edit.
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    gone = blot_upload(session, tmp_path, "gone β.tif")
    api.ops.remove_image(session, gone)  # only the history keeps its file
    held, release = threading.Event(), threading.Event()

    def edit() -> None:  # a request that holds the lock while it runs, then commits
        with session.lock:
            held.set()
            release.wait(10)
            api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])

    editing = threading.Thread(target=edit)
    editing.start()
    assert held.wait(10)
    answers: list[dict[str, Any]] = []
    reopening = threading.Thread(
        target=lambda: answers.append(api.open_project(api.NameBody(name="a µ"), workspace))
    )
    reopening.start()
    reopening.join(10)
    assert not reopening.is_alive()  # answered while the edit still runs
    (answer,) = answers
    assert workspace.current() is session
    assert (answer["project"]["open_id"], answer["project"]["revision"]) == (1, 3)
    assert history(answer)["undo"] == {"seq": 3, "action": "remove_image"}
    stored: list[str] = []
    importing = threading.Thread(
        target=lambda: stored.append(blot_upload(workspace.current(), tmp_path, "stored α.tif"))
    )
    importing.start()
    release.set()
    for thread in (editing, importing):
        thread.join(10)
        assert not thread.is_alive()

    saved = storage.load_project(session.folder)
    assert saved == session.project
    assert [lane.label for lane in saved.batch.lanes] == ["vehicle"]
    assert [image.id for image in saved.batch.iter_images()] == stored
    images = session.folder / storage.IMAGES_DIR
    assert sorted(p.name for p in images.iterdir()) == [f"{gone}.tif", f"{stored[0]}.tif"]
    assert [entry.action for entry in saved.log[-2:]] == ["set_lanes", "import_image"]
    for action in ("import_image", "set_lanes", "remove_image"):  # the history is whole
        assert api.ops.undo(session).action == action
    assert gone in {image.id for image in session.project.batch.iter_images()}


def test_reopening_reads_the_open_project_again_once_changed_outside_proteia(client, tmp_path):
    # Its folder synced from another copy: the reopen reads project.json again,
    # in the one session and under the next open id, so no edit saves the state
    # it held before over the other copy's changes and files.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    _, imported = upload(client, blot_bytes(tmp_path))
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    session = client.workspace.current()  # as a request still streaming an upload holds it
    other = tmp_path / "copy µ"
    shutil.copytree(session.folder, other)
    remote = api.ops.open_project(other, clock=FakeClock())
    newer = blot_upload(remote, tmp_path, "remote α.tif")
    api.ops.set_lanes(remote, [api.ops.LaneInput("drug")])
    shutil.copytree(other, session.folder, dirs_exist_ok=True)

    reopened = client.ok("POST", "/api/projects/open", {"name": "BLOT"})
    assert client.workspace.current() is session
    project = reopened["project"]
    assert (project["open_id"], project["revision"]) == (2, revision(remote.project))
    results = reopened["results"]
    assert (results["open_id"], results["revision"]) == (2, project["revision"])
    assert [lane["condition"] for lane in project["lanes"]] == ["drug"]
    assert [image["id"] for image in project["images"]] == [imported["image_id"], newer]
    assert history(reopened) == {"undo": None, "redo": None}
    assert client.ok("POST", "/api/projects/open", {"name": "Blot"}) == reopened  # read once

    stored = blot_upload(session, tmp_path, "stored β.tif")
    saved = storage.load_project(session.folder)
    assert saved == session.project
    assert [lane.label for lane in saved.batch.lanes] == ["drug"]
    assert [image.id for image in saved.batch.iter_images()] == [
        imported["image_id"],
        newer,
        stored,
    ]
    images = session.folder / storage.IMAGES_DIR
    assert sorted(p.name for p in images.iterdir()) == sorted(
        f"{image_id}.tif" for image_id in (imported["image_id"], newer, stored)
    )


def test_reopening_after_a_restore_shows_the_older_project_under_a_new_open_id(client):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    project_file = client.root / "Blot" / storage.PROJECT_FILE
    backup = project_file.read_bytes()
    shown = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "drug"}]})["project"]
    project_file.write_bytes(backup)  # a previous version restored

    # An earlier revision, but a later opening: a client keeping the newest
    # answer by (open id, revision) shows it.
    restored = client.ok("POST", "/api/projects/open", {"name": "Blot"})["project"]
    assert (shown["open_id"], shown["revision"]) == (1, 3)
    assert (restored["open_id"], restored["revision"]) == (2, 2)
    assert [lane["condition"] for lane in restored["lanes"]] == ["vehicle"]

    # A file that cannot be read is refused, as opening it is; the session stays.
    project_file.write_bytes(b"{not json")
    refused = client.refused("POST", "/api/projects/open", {"name": "Blot"})
    assert refused[:2] == (422, "unreadable_project")
    assert client.ok("GET", "/api/project")["project"] == restored


def test_charts_of_a_project_read_again_on_reopen_are_served(client, tmp_path):
    live(client, tmp_path, DOSES)
    project_file = client.root / "Blot" / storage.PROJECT_FILE
    backup = project_file.read_bytes()
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": c} for c in reversed(DOSES)]})
    # Another copy, never shown here: its charts have keys never registered.
    outside = backup.decode("utf-8").replace(f'"{DOSES[0]}"', '"mock µ"')
    assert outside != backup.decode("utf-8")
    project_file.write_bytes(outside.encode("utf-8"))

    # The reload takes a new opening: its charts are registered under it.
    reopened = client.ok("POST", "/api/projects/open", {"name": "Blot"})
    assert reopened["project"]["open_id"] == 2
    urls = [url for url in chart_urls(reopened).values() if url is not None]
    assert urls
    for url in urls:
        assert fetch(client, url)[0] == 200


def test_a_reopen_reads_no_file_while_an_operation_runs_or_changes_are_unsaved(
    tmp_path, monkeypatch
):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    project_file = session.folder / storage.PROJECT_FILE
    empty = project_file.read_bytes()
    api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])
    project_file.write_bytes(empty)  # changed outside Proteia
    project = session.project

    # An operation holds the session past the wait: the reopen answers it as it is.
    monkeypatch.setattr(api, "REOPEN_WAIT_S", 0.2)
    held, release = threading.Event(), threading.Event()

    def running() -> None:
        with session.lock:
            held.set()
            release.wait(10)

    operation = threading.Thread(target=running)
    operation.start()
    assert held.wait(10)
    try:
        assert workspace.open("A µ") is session
        assert session.project is project
    finally:
        release.set()
        operation.join(10)
    assert workspace.view(session)[0] == 1
    assert workspace.open("A µ") is session  # nothing holds it now: read again
    assert workspace.view(session)[0] == 2 and not session.project.batch.lanes

    # Unsaved changes: the file is older than the session, not changed outside.
    def refuse(project: Project, folder: Path) -> Path:
        raise PermissionError(13, "held by another process", str(folder))

    monkeypatch.setattr(storage, "save_project", refuse)
    api.ops.set_lanes(session, [api.ops.LaneInput("drug")])
    assert session.dirty
    project = session.project
    assert workspace.open("a µ") is session and session.project is project
    assert workspace.view(session)[0] == 2
    assert project_file.read_bytes() == empty


def test_a_reopen_waits_for_a_short_read_and_then_reads_the_changed_file(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    project_file = session.folder / storage.PROJECT_FILE
    empty = project_file.read_bytes()
    api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])
    project_file.write_bytes(empty)  # changed outside Proteia

    # A read (such as a preview hashing a large image) holds the session briefly.
    held = threading.Event()

    def reading() -> None:
        with session.lock:
            held.set()
            time.sleep(0.3)

    read = threading.Thread(target=reading)
    read.start()
    assert held.wait(10)
    try:
        assert workspace.open("A µ") is session  # waits for the read, then reads the file
    finally:
        read.join(10)
    assert workspace.view(session)[0] == 2 and not session.project.batch.lanes


def test_a_project_created_where_the_open_one_was_removed_keeps_the_files_it_stores(tmp_path):
    # The open project's folder removed outside Proteia, then a project of that
    # name created: the switch closes the old session on the new project's
    # folder, whose files it does not know, so the close deletes none.
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    shutil.rmtree(first.folder)
    held, release = threading.Event(), threading.Event()

    def running() -> None:  # a request on the old session, e.g. a preview
        with first.lock:
            held.set()
            release.wait(10)

    request = threading.Thread(target=running)
    request.start()
    assert held.wait(10)
    creating = threading.Thread(target=workspace.create, args=("A µ",))
    creating.start()
    deadline = time.monotonic() + 10
    while workspace.current() is first:  # the close then waits for the old one's lock
        assert time.monotonic() < deadline
        time.sleep(0.005)
    second = workspace.current()
    stored = blot_upload(second, tmp_path, "stored α.tif")  # an upload that reached it
    release.set()
    for thread in (request, creating):
        thread.join(10)
        assert not thread.is_alive()

    assert (first.undo_step, first.redo_step) == (None, None)  # closed
    images = second.folder / storage.IMAGES_DIR
    assert sorted(p.name for p in images.iterdir()) == [f"{stored}.tif"]
    assert storage.load_project(second.folder) == second.project


def test_the_open_project_renamed_outside_proteia_opens_as_another(tmp_path):
    # Its old folder is gone, so whether it is the one named cannot be told: the
    # open is a switch (whose close deletes nothing), never the session on a
    # folder that no longer exists.
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    first.folder.rename(first.folder.with_name("C"))
    opened = workspace.open("C")
    assert workspace.current() is opened and opened is not first
    assert opened.folder.name == "C"


def test_folders_differing_only_in_case_are_two_projects(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "probe").mkdir()
    if (root / "PROBE").exists():
        pytest.skip("two names differing only in case need a case-sensitive file system")
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("blot")
    api.ops.new_project(root / "Blot")
    opened = workspace.open("Blot")  # the exact name: the other folder
    assert opened is not first and opened.folder.name == "Blot"
    assert workspace.open("blot") is not opened


def test_reading_the_project_never_waits_for_a_running_operation(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    blot_upload(session, tmp_path, "α.tif")
    answers: list[dict[str, Any]] = []
    reader = threading.Thread(
        target=lambda: answers.append(api.get_project(workspace.current(), workspace))
    )
    with session.lock:  # as an operation holds it while it runs, e.g. a large import
        reader.start()
        reader.join(10)
        assert not reader.is_alive()  # answered without waiting for it
    assert history(answers[0]) == {"undo": {"seq": 2, "action": "import_image"}, "redo": None}


def test_stopping_the_server_closes_the_open_project(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    instance = launch.start(folder=tmp_path / "state", opener=lambda url: True, workspace=workspace)
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    session = workspace.create("A µ")
    gone = blot_upload(session, tmp_path, "gone β.tif")
    api.ops.remove_image(session, gone)
    file = session.folder / storage.IMAGES_DIR / f"{gone}.tif"
    assert file.exists()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(0.01)
    instance.stop()
    thread.join(10)
    assert not thread.is_alive()
    assert not file.exists()  # closed after the flush
    assert storage.load_project(session.folder) == session.project


# --- Charts drawn on the server (#52) ---


def fetch(client: Client, path: str, *, token: bool = True) -> tuple[int, dict[str, str], bytes]:
    """A GET with the launch's token or none: its status, headers (by lower-case
    name) and body."""
    headers = {"Authorization": f"Bearer {client.token}"} if token else {}
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    try:
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        body = response.read()
        got = {name.lower(): value for name, value in response.getheaders()}
    finally:
        conn.close()
    return response.status, got, body


def not_found(client: Client, path: str) -> str:
    status, _, body = fetch(client, path)
    assert status == 404, (status, body[:200])
    return json.loads(body)["code"]


def chart_urls(answer: dict) -> dict[tuple[str, str], str | None]:
    """Each series' chart URL by (set id, target id)."""
    return {
        (result_set["id"], series["target_id"]): series["chart_url"]
        for result_set in answer["results"]["sets"]
        for series in result_set["series"]
    }


@pytest.fixture
def drawn(monkeypatch) -> list[str]:
    """The title of every chart the server draws, in order."""
    titles: list[str] = []
    render = charts.render_svg

    def counting(spec: PlotSpec) -> bytes:
        titles.append(spec.title)
        return render(spec)

    monkeypatch.setattr(charts, "render_svg", counting)
    return titles


def test_an_edit_answers_a_chart_url_that_serves_the_chart_as_svg(client, tmp_path, drawn):
    _, _, answer = live(client, tmp_path, DOSES)  # the last answer: a box placed
    series = only_series(answer)
    spec = PlotSpec.model_validate(series["chart"])
    assert series["chart_url"] == charts.chart_url(spec)  # named by the chart it answers
    assert drawn == []  # drawn only when fetched

    status, headers, body = fetch(client, series["chart_url"])
    assert status == 200
    assert headers["content-type"] == "image/svg+xml"
    for name, value in server._SECURITY_HEADERS:  # the guard's, as on every answer
        assert headers[name.decode()] == value.decode(), name
    assert body == render_svg(spec)
    assert fetch(client, series["chart_url"])[2] == body
    assert len(drawn) == 1  # kept once drawn

    status, headers, body = fetch(client, series["chart_url"], token=False)
    assert (status, headers["www-authenticate"]) == (401, "Bearer")
    assert b"<svg" not in body


@pytest.mark.parametrize(
    "condition",
    ["10 µM\uffff", r"$\notacommand$"],  # a noncharacter; not math matplotlib could draw
    ids=ascii,
)
def test_a_chart_of_any_name_is_served_as_well_formed_svg(client, tmp_path, condition):
    _, _, answer = live(client, tmp_path, ["vehicle", "vehicle", condition, condition, condition])
    series = only_series(answer)
    assert condition in [bar["label"] for bar in series["chart"]["bars"]]  # stored as typed
    status, headers, body = fetch(client, series["chart_url"])
    assert (status, headers["content-type"]) == (200, "image/svg+xml")
    assert ET.fromstring(body).tag == "{http://www.w3.org/2000/svg}svg"


def test_moving_a_box_changes_only_the_charts_of_its_series(client, tmp_path, drawn):
    target, _, _ = live(client, tmp_path, DOSES)
    # The other series on a second image, with its own loading control: a box
    # edit changes every net on its image (each band's ring leaves out every box
    # there), so only a series off that image keeps its chart.
    status, answer = upload(client, two_row_bytes(tmp_path, DEPTHS, (25000.0,) * 5), "reprobe.tif")
    assert status == 201, answer
    image_id = answer["image_id"]
    body = {"name": "GAPDH", "role": "loading control", "image_id": image_id, "box_size": SIZE}
    gapdh = client.ok("POST", "/api/proteins", body)["protein_id"]
    body = {"name": "p-ERK", "role": "target", "image_id": image_id, "box_size": SIZE}
    body["loading_control_ids"] = [gapdh]
    other = client.ok("POST", "/api/proteins", body)["protein_id"]
    for protein, row in ((gapdh, LOADING_ROW), (other, TARGET_ROW)):
        for lane, x in enumerate(LANE_X):
            body = {"protein_id": protein, "x": x, "y": row, "lane_index": lane}
            client.ok("POST", "/api/boxes", body)
    rows = [{"condition": condition, "included": lane != 1} for lane, condition in enumerate(DOSES)]
    before = client.ok("PUT", "/api/lanes", {"lanes": rows})  # a lane with values left out
    urls = chart_urls(before)
    assert set(urls) == {(s, t) for s in ("applied", "all_lanes") for t in (target, other)}
    assert None not in urls.values() and len(set(urls.values())) == 4  # both sets' series
    for url in urls.values():
        assert fetch(client, url)[0] == 200
    assert len(drawn) == 4

    band = column(before, target)["band_ids"][4]
    x0, y0, x1, y1 = bands(before)[band]["rect"]
    moved = client.ok("PUT", f"/api/boxes/{band}", {"rect": [x0 + 4, y0 + 3, x1 + 4, y1 + 3]})
    after = chart_urls(moved)
    for set_id in ("applied", "all_lanes"):
        assert after[(set_id, target)] != urls[(set_id, target)]
        assert after[(set_id, other)] == urls[(set_id, other)]
    for url in after.values():
        assert fetch(client, url)[0] == 200
    assert drawn[4:] == ["β-catenin / α-tubulin"] * 2  # only the two changed charts


def test_a_chart_key_is_unknown_unless_given_in_this_opening(client, tmp_path):
    _, _, answer = live(client, tmp_path, DOSES)
    url = only_series(answer)["chart_url"]
    key = url.removeprefix("/api/charts/").removesuffix(".svg")
    assert fetch(client, url)[0] == 200
    for other in ["0" * 32, "f" * 32, key.upper(), key[:-1], key + "0", "g" * 32]:
        assert not_found(client, f"/api/charts/{other}.svg") == "unknown_id", other

    client.ok("POST", "/api/projects", {"name": "Other"})
    assert not_found(client, url) == "unknown_id"  # a key of the project closed
    reopened = client.ok("POST", "/api/projects/open", {"name": "Blot"})
    assert only_series(reopened)["chart_url"] == url  # the same chart: the same URL
    assert fetch(client, url)[0] == 200  # registered again by the answer
    client.ok("POST", "/api/projects/open", {"name": "Other"})
    assert not_found(client, url) == "unknown_id"


def test_an_answer_about_an_earlier_opening_registers_no_chart(client, tmp_path):
    _, _, answer = live(client, tmp_path, DOSES)
    session = client.workspace.current()
    client.ok("POST", "/api/projects", {"name": "Other"})
    late = api._answer(client.workspace, session)  # a request on Blot that finished late
    assert late["results"]["open_id"] == answer["results"]["open_id"]
    url = only_series(late)["chart_url"]
    assert url == only_series(answer)["chart_url"]
    assert not_found(client, url) == "unknown_id"  # its charts are not Other's


def test_a_read_from_the_kept_results_registers_their_charts_again(client, tmp_path, monkeypatch):
    _, _, answer = live(client, tmp_path, DOSES)
    url = only_series(answer)["chart_url"]
    calls: list = []
    _counting(monkeypatch, calls)
    # The store forgets the charts, as when the least recently used are evicted.
    client.workspace._charts.reset(answer["results"]["open_id"])
    assert not_found(client, url) == "unknown_id"
    read = client.ok("GET", "/api/project")
    assert calls == []  # the kept results of this revision, not computed again
    assert only_series(read)["chart_url"] == url
    assert fetch(client, url)[0] == 200


# --- A row of boxes from a dragged row box (#52) ---

ROW_FIELDS = [field.name for field in dataclasses.fields(api.ops.RowPlacement)]
TARGET_ROW_BOX = [15, TARGET_ROW - 12, W - 35, TARGET_ROW + 12]  # every lane of the target row
BOTH_ROWS_BOX = [15, TARGET_ROW - 12, W - 35, LOADING_ROW + 12]  # the target's row and the LC's


def drag(client: Client, protein_id: str, rect: list[int] = TARGET_ROW_BOX) -> dict:
    """The answer to a row box dragged over the protein's row: 201, as any placement."""
    body = {"protein_id": protein_id, "rect": rect}
    status, answer = client.call("POST", "/api/boxes/row", body)
    assert status == 201, (status, answer)
    return answer


def test_a_row_box_boxes_every_lane_and_answers_the_nets_and_the_chart(client, tmp_path):
    target, _, before = live(client, tmp_path, DOSES, boxed=())
    assert column(before, target)["nets"] == [None] * len(LANE_X)

    answer = drag(client, target)
    assert set(answer) == {*ROW_FIELDS, "project", "results"}  # every field of RowPlacement
    band_ids = answer["band_ids"]
    assert None not in band_ids and len(set(band_ids)) == len(LANE_X)
    state = protein_of(answer, target)
    assert [band["id"] for band in state["bands"]] == band_ids  # new ids in lane order
    size = answer["box_size"]
    assert state["box_size"] == size
    for lane, band in enumerate(state["bands"]):
        assert (band["lane_index"], band["source"], band["manually_edited"]) == (
            lane,
            "row_box",
            False,
        )
        x0, y0, x1, y1 = band["rect"]
        assert (x1 - x0, y1 - y0) == (size["width"], size["height"])  # one shared size
        assert x0 < LANE_X[lane] < x1 and y0 < TARGET_ROW < y1
    assert {name: answer[name] for name in ROW_FIELDS if name not in ("band_ids", "box_size")} == {
        "kept_lanes": [],
        "replaced_band_ids": [],
        "removed_band_ids": [],
        "undetected_lanes": [],
        "unmeasured_lanes": [],
        "empty": [],
        "flags": [],
        "notes": [],
        "right_to_left": False,
        # The loading control's row lies beyond every ring the new boxes change.
        "remeasured": [],
        "largest_change": None,
        "unlocated_lanes": [],
    }

    # The nets of the new boxes and the chart they make, in the same answer.
    results = column(answer, target)
    assert results["band_ids"] == band_ids and results["detected"] == [True] * len(LANE_X)
    assert all(net > 0 for net in results["nets"])
    series = only_series(answer)
    assert None not in series["normalized"]
    assert [bar(series, dose)["n"] for dose in ("vehicle", "10 µM")] == [2, 3]
    status, headers, body = fetch(client, series["chart_url"])
    assert (status, headers["content-type"]) == (200, "image/svg+xml")
    assert ET.fromstring(body).tag == "{http://www.w3.org/2000/svg}svg"

    # One change, one entry, autosaved; the answer is what a read gives.
    assert answer["project"]["revision"] == before["project"]["revision"] + 1
    entry = storage.load_project(client.root / "Blot").log[-1]
    assert (entry.action, entry.params["protein_id"], entry.params["row"]) == (
        "detect_row_boxes",
        target,
        TARGET_ROW_BOX,
    )
    read = client.ok("GET", "/api/project")
    assert read == {"project": answer["project"], "results": answer["results"]}

    # The same drag again replaces each box in place with itself: nothing changes.
    again = drag(client, target)
    assert {"project": again["project"], "results": again["results"]} == read
    assert again["band_ids"] == again["replaced_band_ids"] == band_ids
    assert logged(client).count("detect_row_boxes") == 1


def test_a_lane_without_a_band_gets_a_not_detected_record(client, tmp_path):
    lanes = [0, 1, 3, 4]  # lane 2 holds no band
    depths = (20000.0, 22000.0, 0.0, 33000.0, 36000.0)
    target, _, _ = live(client, tmp_path, DOSES, target=depths, boxed=())
    answer = drag(client, target)
    band_ids = answer["band_ids"]
    assert band_ids[2] is None and None not in [band_ids[lane] for lane in lanes]
    state = protein_of(answer, target)
    # The other lanes keep their indices.
    assert {band["id"]: band["lane_index"] for band in state["bands"]} == {
        band_ids[lane]: lane for lane in lanes
    }
    assert (answer["undetected_lanes"], answer["unmeasured_lanes"]) == ([2], [])
    (empty,) = answer["empty"]
    assert list(empty) == ["lane_index", "reason", "snr", "expected_x"]
    assert (empty["lane_index"], empty["reason"]) == (2, "no_band")
    assert 0 <= empty["snr"] < 6 and abs(empty["expected_x"] - LANE_X[2]) <= 1
    (record,) = state["undetected"]
    assert {name: record[name] for name in ("lane_index", "band_index", "reason", "source")} == {
        "lane_index": 2,
        "band_index": 0,
        "reason": "below_detection_limit",
        "source": "row_box",
    }
    assert (record["snr"], record["threshold"]) == (empty["snr"], 6.0)
    assert 2 not in {entry["lane_index"] for entry in state["missing_lanes"]}  # examined

    results = column(answer, target)
    assert results["band_ids"] == band_ids
    assert results["detected"] == [True, True, False, True, True]
    assert results["nets"][2] is None and all(results["nets"][lane] > 0 for lane in lanes)
    assert "below_detection" in notice_codes(answer)
    assert bar(only_series(answer), "10 µM")["lane_indices"] == [3, 4]
    assert logged(client)[-1] == "detect_row_boxes"


def test_one_band_on_an_image_without_lanes_placed_answers_the_lanes_it_cannot_locate(
    client, tmp_path
):
    # The target's band in lane 2 alone and nothing else boxed on the image:
    # the other lanes' slots rest on that one band, so they get no record, and
    # the answer names them apart from the lanes not measured for a reason of
    # the detector's.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    depths = (0.0, 0.0, 30000.0, 0.0, 0.0)
    status, answer = upload(client, two_row_bytes(tmp_path, depths, (25000.0,) * 5))
    assert status == 201, answer
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": c} for c in DOSES]})
    body = {"name": "β-catenin", "role": "target", "image_id": answer["image_id"], "box_size": SIZE}
    target = client.ok("POST", "/api/proteins", body)["protein_id"]
    answer = drag(client, target)
    assert [lane for lane, band_id in enumerate(answer["band_ids"]) if band_id] == [2]
    assert (answer["undetected_lanes"], answer["unlocated_lanes"]) == ([], [0, 1, 3, 4])
    assert answer["unmeasured_lanes"] == [0, 1, 3, 4]
    assert {entry["reason"] for entry in answer["empty"]} == {"no_band"}
    assert protein_of(answer, target)["undetected"] == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
def test_a_second_row_box_corrects_the_first_and_removes_a_box_its_lane_lost(client, tmp_path):
    depths = (20000.0, 22000.0, 0.0, 33000.0, 36000.0)  # lane 2 of the target row: no band
    target, loading, _ = live(client, tmp_path, DOSES, target=depths, boxed=())
    # Dragged over the loading control's row first, by mistake, before that row
    # was boxed (over its boxes the row is refused, #114): a box in every lane.
    client.ok("DELETE", f"/api/proteins/{loading}/boxes")
    first = drag(client, target, [15, LOADING_ROW - 12, W - 35, LOADING_ROW + 12])
    ids = first["band_ids"]
    assert None not in ids

    answer = drag(client, target)  # then over its own row
    # The detector's boxes give way: taken over in place, or removed where no
    # band is found (lane 2, which gets a not-detected record instead).
    assert answer["band_ids"] == [*ids[:2], None, *ids[3:]]
    assert answer["replaced_band_ids"] == [*ids[:2], *ids[3:]]
    assert answer["removed_band_ids"] == [ids[2]]
    assert (answer["kept_lanes"], answer["undetected_lanes"], answer["unmeasured_lanes"]) == (
        [],
        [2],
        [],
    )
    state = protein_of(answer, target)
    assert ids[2] not in bands(answer)
    assert all(band["rect"][1] < TARGET_ROW < band["rect"][3] for band in state["bands"])
    assert [record["lane_index"] for record in state["undetected"]] == [2]
    assert column(answer, target)["detected"] == [True, True, False, True, True]


def test_a_row_box_over_two_rows_answers_the_lanes_off_its_line(client, tmp_path):
    # The target's bands are deeper than the loading control's in lanes 2-4,
    # shallower in lanes 0 and 1: there the row's boxes would lie on the
    # loading control's row (#114).
    target, _, _ = live(client, tmp_path, DOSES, boxed=())
    status, payload = client.call(
        "POST", "/api/boxes/row", {"protein_id": target, "rect": BOTH_ROWS_BOX}
    )
    assert (status, payload["code"], payload["ids"]) == (422, "row_off_line", [])
    assert payload["message"] == (
        "the bands found in lanes 1, 2 lie above or below the row's line through the other"
        " bands, by more than 0.75 of a box's height: the row box covers more than one row,"
        " or those bands lie off the row; draw it over one row only, or box those lanes by"
        " clicking their bands"
    )
    detail = payload["detail"]
    assert detail["cause"] == "off_row_line" and detail["flags"][0] == "off_row_line"
    offsets = [lane["line_offset"] for lane in detail["lanes"]]
    assert [lane for lane, v in enumerate(offsets) if abs(v) > 0.75] == [0, 1]
    assert all(v > 0 for v in offsets[:2])  # below the target's row


def test_a_box_over_another_proteins_box_is_refused_naming_it(client, tmp_path):
    # Clicked, shift+clicked or moved onto the loading control's box in lane 1
    # (#114): refused, naming that box for the page to offer it.
    target, loading, answer = live(client, tmp_path, DOSES, boxed=(0,))
    in_the_way = column(answer, loading)["band_ids"][1]
    moved = column(answer, target)["band_ids"][0]
    click = {"protein_id": target, "x": LANE_X[1], "y": LOADING_ROW, "lane_index": 1}
    over = [LANE_X[1] - 7, LOADING_ROW - 5, LANE_X[1] + 7, LOADING_ROW + 5]
    for method, path, body in (
        ("POST", "/api/boxes", {**click, "grow": True}),
        ("POST", "/api/boxes", {**click, "grow": False}),
        ("PUT", f"/api/boxes/{moved}", {"rect": over}),
    ):
        assert unchanged_refusal(client, method, path, body) == ("overlap", [in_the_way])
        _, payload = client.call(method, path, body)
        assert payload["message"] == (
            "the box would overlap the box of 'α-tubulin' in lane 2 by more than 50% of the"
            " smaller box's area: two proteins' boxes would measure the same band"
        )
        [covered] = payload["detail"]["covered"]
        assert payload["detail"]["cause"] == "other_protein"
        assert (covered["band_id"], covered["protein_id"], covered["lane_index"]) == (
            in_the_way,
            loading,
            1,
        )
        assert 0.5 < covered["share"] <= 1.0
    assert "move_box" not in logged(client)


def test_a_box_size_over_another_proteins_boxes_is_refused_naming_them(client, tmp_path):
    # The loading control's boxes made 90 px high reach up over the target's,
    # 45 px above (#114): refused, naming the target's boxes.
    target, loading, answer = live(client, tmp_path, DOSES)
    covered = column(answer, target)["band_ids"]
    path = f"/api/proteins/{loading}/box-size"
    body = {"width": 14, "height": 90}
    assert unchanged_refusal(client, "PUT", path, body) == ("overlap", covered)
    _, payload = client.call("PUT", path, body)
    assert payload["message"] == (
        "at the box size 14x90, boxes of 'α-tubulin' would overlap the boxes of 'β-catenin' in"
        " lanes 1, 2, 3, 4, 5 by more than 50% of the smaller box's area: two proteins' boxes"
        " would measure the same band"
    )
    assert payload["detail"]["cause"] == "other_protein"
    assert [c["band_id"] for c in payload["detail"]["covered"]] == covered
    assert "set_box_size" not in logged(client)


def adversarial_project(client: Client, tmp_path: Path, key: str) -> tuple[RowCase, str]:
    """A project over an adversarial row (:func:`rowcases.adversarial`, seed
    1000) imported as a 16-bit TIFF, its lanes declared and one target without
    boxes; (the case, the target's id)."""
    case = adversarial(key, 1000)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    data = write_tiff(tmp_path / "row α.tif", case.image.astype(np.uint16)).read_bytes()
    polarity = "dark_on_light" if case.dark_on_light else "light_on_dark"
    status, answer = upload(client, data, polarity=polarity)
    assert status == 201, answer
    lanes = [{"condition": "vehicle" if i % 2 == 0 else "10 µM"} for i in range(case.n_lanes)]
    client.ok("PUT", "/api/lanes", {"lanes": lanes})
    body = {"name": "β-catenin", "role": "target", "image_id": answer["image_id"]}
    return case, client.ok("POST", "/api/proteins", body)["protein_id"]


def test_a_box_placed_in_a_lane_a_row_box_left_unmeasured_is_kept_by_the_next(client, tmp_path):
    case, target = adversarial_project(client, tmp_path, "blotch_empty")  # lane 4: a stain
    first = drag(client, target, list(case.row))
    # A stain is no not-detected claim: the lane is left with neither box nor record.
    assert (first["kept_lanes"], first["undetected_lanes"], first["unmeasured_lanes"]) == (
        [],
        [],
        [4],
    )
    assert first["band_ids"][4] is None
    assert [(empty["lane_index"], empty["reason"]) for empty in first["empty"]] == [(4, "artefact")]
    assert protein_of(first, target)["undetected"] == []
    assert column(first, target)["detected"][4] is None

    # The user boxes the lane by hand; the same drag again keeps that box.
    x, y = round(case.lane_cx[4]), round(case.lane_cy[4])
    body = {"protein_id": target, "x": x, "y": y, "lane_index": 4}
    box = client.ok("POST", "/api/boxes", body)["band_id"]
    answer = drag(client, target, list(case.row))
    assert (answer["kept_lanes"], answer["undetected_lanes"], answer["unmeasured_lanes"]) == (
        [4],
        [],
        [],
    )
    assert answer["band_ids"] == [*first["band_ids"][:4], box, *first["band_ids"][5:]]
    assert answer["replaced_band_ids"] == [band for band in first["band_ids"] if band is not None]
    assert answer["removed_band_ids"] == []
    assert column(answer, target)["detected"][4] is True


def test_a_row_box_answers_the_detectors_warnings_and_notes(client, tmp_path):
    case, target = adversarial_project(client, tmp_path, "tall_band")  # lane 3: far taller
    answer = drag(client, target, list(case.row))
    params = storage.load_project(client.root / "Blot").log[-1].params
    assert answer["flags"] == params["flags"] == ["size_outlier"]
    assert answer["notes"] == params["notes"] != []


def test_a_first_row_box_read_off_answers_its_lanes_doubtful(client, tmp_path):
    # #111: a row box that also covers a neighbouring panel beside the row,
    # with no lanes on the image to check it, is read two lanes off. It is
    # placed, and the answer carries the warning and its note (the page asks
    # to check the lane numbers, with an Undo of the row).
    case, target = adversarial_project(client, tmp_path, "panel_beside")
    answer = drag(client, target, list(case.row))
    assert None not in answer["band_ids"]
    params = storage.load_project(client.root / "Blot").log[-1].params
    assert answer["flags"] == params["flags"] == ["doubtful_lanes"]
    [note] = answer["notes"]
    assert note == params["notes"][0]
    assert note.startswith("lane numbers doubtful: the fitted pitch")
    assert answer["project"]["history"]["undo"]["action"] == "detect_row_boxes"


def test_a_row_box_answers_a_band_it_cuts_through(client, tmp_path):
    # The box's top edge 2 px above the target's band centres (#115).
    target, _, _ = live(client, tmp_path, DOSES, boxed=())
    answer = drag(client, target, [15, TARGET_ROW - 2, W - 35, TARGET_ROW + 12])
    assert None not in answer["band_ids"]
    params = storage.load_project(client.root / "Blot").log[-1].params
    assert answer["flags"] == params["flags"] == ["cut_by_row_box"]
    [note] = answer["notes"]
    assert note.startswith("lanes 1, 2, 3, 4, 5: the row box's top or bottom edge cuts through")


def test_a_refused_row_box_answers_what_the_detector_saw(client, tmp_path):
    # The box's top edge through the target's band centres: every band peaks
    # on the box's top row, so none is kept, and the refusal says the box cuts
    # them (#117) with the detector's raw reason.
    target, _, _ = live(client, tmp_path, DOSES, boxed=())
    body = {"protein_id": target, "rect": [15, TARGET_ROW, W - 35, TARGET_ROW + 12]}
    status, payload = client.call("POST", "/api/boxes/row", body)
    assert (status, payload["code"], payload["ids"]) == (422, "no_band_found", [])
    assert payload["message"] == (
        "the only signal in the row box lies at its top or bottom edge: the box cuts through"
        " the bands or reaches into a neighbouring row; include the whole band height"
    )
    detail = payload["detail"]
    assert set(detail) == {"cause", "flags", "notes", "margin", "membrane_shift", "lanes"}
    assert (detail["cause"], detail["margin"]) == ("edge_signal", None)
    assert [lane["lane_index"] for lane in detail["lanes"]] == list(range(len(LANE_X)))
    assert {lane["reason"] for lane in detail["lanes"]} == {"edge_signal"}
    # The detector names the bands it saw cut, though none was kept (#115).
    assert detail["flags"] == ["cut_by_row_box"]
    assert all(lane["cut"] for lane in detail["lanes"])
    # A refusal that carries no detail answers the three keys only.
    body = {"protein_id": target, "rect": [W + 10, 0, W + 50, 20]}
    status, payload = client.call("POST", "/api/boxes/row", body)
    assert (status, set(payload)) == (422, {"code", "message", "ids"})


def test_a_row_box_answers_the_other_proteins_nets_it_changed(client, tmp_path):
    # The loading control boxed just above two of the target's bands: every
    # ring on the image leaves out the row's boxes, so its nets change with the
    # row (#124).
    target, loading, _ = live(client, tmp_path, DOSES, boxed=())
    client.ok("DELETE", f"/api/proteins/{loading}/boxes")
    for lane in (1, 2):
        y = TARGET_ROW - 12
        body = {"protein_id": loading, "x": LANE_X[lane], "y": y, "lane_index": lane}
        before = client.ok("POST", "/api/boxes", body)

    def nets(answer: dict) -> dict[str, float]:
        lanes = column(answer, loading)
        return {b: net for b, net in zip(lanes["band_ids"], lanes["nets"], strict=True) if b}

    answer = drag(client, target)
    old, new = nets(before), nets(answer)
    changed = [
        {"band_id": band_id, "net_before": old[band_id], "net_after": new[band_id]}
        for band_id in old
        if new[band_id] != old[band_id]
    ]
    assert changed and answer["remeasured"] == changed
    shares = {
        c["band_id"]: abs(c["net_after"] - c["net_before"]) / c["net_before"] for c in changed
    }
    largest = max(shares, key=shares.__getitem__)
    assert answer["largest_change"] == {"band_id": largest, "change": shares[largest]}


def test_a_row_box_reads_the_lanes_the_way_the_image_numbers_them(client, tmp_path):
    target, loading, _ = live(client, tmp_path, DOSES, boxed=())
    client.ok("DELETE", f"/api/proteins/{loading}/boxes")
    for lane in range(len(LANE_X)):  # the loading control's lanes numbered right to left
        body = {"protein_id": loading, "x": LANE_X[-1 - lane], "y": LOADING_ROW, "lane_index": lane}
        client.ok("POST", "/api/boxes", body)
    answer = drag(client, target)
    assert answer["right_to_left"] is True
    assert storage.load_project(client.root / "Blot").log[-1].params["right_to_left"] is True
    for band in protein_of(answer, target)["bands"]:
        x0, _, x1, _ = band["rect"]
        assert x0 < LANE_X[-1 - band["lane_index"]] < x1


def test_a_row_box_may_reach_far_past_the_image_and_is_logged_as_given(client, tmp_path):
    # A drag is clipped to the image. Its corners are held to 32 bits: the log
    # keeps the row as given, and the project file must still read.
    target, _, _ = live(client, tmp_path, DOSES, boxed=())
    rect = [-(2**31), TARGET_ROW - 12, 2**31 - 1, TARGET_ROW + 12]
    answer = drag(client, target, rect)
    assert None not in answer["band_ids"]
    assert storage.load_project(client.root / "Blot").log[-1].params["row"] == rect
    client.ok("POST", "/api/projects", {"name": "Other"})
    reopened = client.ok("POST", "/api/projects/open", {"name": "Blot"})
    assert protein_of(reopened, target)["bands"] == protein_of(answer, target)["bands"]


def _no_lanes(client: Client, target: str, loading: str) -> list[str]:
    client.ok("DELETE", f"/api/proteins/{loading}")  # its boxes hold the lanes
    client.ok("PUT", "/api/lanes", {"lanes": []})
    return []


def _edited_box(client: Client, target: str, size: list[int], lane: int, rect: list[int]) -> str:
    """A box of the target placed in the lane at the size, then moved by hand to ``rect``."""
    width, height = size
    client.ok("PUT", f"/api/proteins/{target}/box-size", {"width": width, "height": height})
    body = {"protein_id": target, "x": LANE_X[lane], "y": TARGET_ROW, "lane_index": lane}
    band_id = client.ok("POST", "/api/boxes", body)["band_id"]
    moved = client.ok("PUT", f"/api/boxes/{band_id}", {"rect": rect})
    assert bands(moved)[band_id]["manually_edited"]
    return band_id


def _lane_1_on_two_columns(client: Client, target: str, loading: str) -> list[str]:
    """The target's box in lane 1, moved by hand onto lane 2's band: lane 1 on
    two columns, the loading control's and the target's."""
    band_id = _edited_box(client, target, SIZE, 1, [LANE_X[2] - 7, 25, LANE_X[2] + 7, 35])
    return [column(client.ok("GET", "/api/project"), loading)["band_ids"][1], band_id]


def _wide_kept_box(client: Client, target: str, loading: str) -> list[str]:
    """A box wider than the lane pitch (70), kept above the row: the row's
    boxes, grown to its width, would overlap each other."""
    _edited_box(client, target, [80, 8], 1, [LANE_X[1] - 39, 6, LANE_X[1] + 41, 14])
    return []


def _kept_box_on_lane_2(client: Client, target: str, loading: str) -> list[str]:
    """A box kept in lane 1, moved by hand toward lane 2 (less than half the
    pitch from its column): the row's box in lane 2 would overlap it."""
    return [_edited_box(client, target, [60, 10], 1, [LANE_X[1], 25, LANE_X[1] + 60, 35])]


def _loading_lanes(client: Client, target: str, loading: str) -> list[str]:
    """Nothing to set up: the loading control's boxes place the lanes the row
    is checked against, and a row that does not line up with them names those
    its bands' lanes are read from. Leaving out lane 0, the row reads its four
    bands a lane off, as lanes 0 to 3: the boxes of those lanes, and lane 4's,
    whose x lane 3's band is measured against."""
    return column(client.ok("GET", "/api/project"), loading)["band_ids"][:5]


def _loading_boxes(client: Client, target: str, loading: str) -> list[str]:
    """Nothing to set up: the loading control's boxes, which a row over its
    row would cover (#114)."""
    return column(client.ok("GET", "/api/project"), loading)["band_ids"]


def _unreadable_pixels(client: Client, target: str, loading: str) -> list[str]:
    """A non-finite pixel in the row, as a damaged file would read."""
    session = client.workspace.current()
    image_id = session.project.batch.find_protein(target).image_id
    pixels = np.array(session.pixels(image_id), dtype=np.float64)
    pixels[TARGET_ROW, LANE_X[0] + 30] = np.nan
    pixels.flags.writeable = False
    session._pixels[image_id] = pixels
    return [image_id]  # the image is at fault, not the row


ROW_BOX_REFUSALS = [
    pytest.param(None, [W - 35, 18, 15, 42], "invalid_input", id="inverted"),
    pytest.param(None, [15, 18, 15, 42], "invalid_input", id="empty"),
    pytest.param(_no_lanes, TARGET_ROW_BOX, "no_lanes", id="no-lanes"),
    pytest.param(None, [W + 10, 0, W + 50, 20], "out_of_image", id="outside"),
    pytest.param(None, [15, 18, 24, 42], "row_too_small", id="too-narrow"),  # < 2 px a lane
    pytest.param(None, [15, 18, W - 35, 20], "row_too_small", id="too-low"),
    pytest.param(_unreadable_pixels, TARGET_ROW_BOX, "unreadable_image", id="unreadable"),
    # Read a lane off: the loading control's boxes show it.
    pytest.param(_loading_lanes, [85, 18, W, 42], "row_lanes_unclear", id="first-lane-left-out"),
    # The row box alone does not show the lanes: nothing on the image is named.
    pytest.param(None, [15, 18, 180, 42], "row_lanes_unclear", id="part-of-the-row"),
    pytest.param(
        _lane_1_on_two_columns, TARGET_ROW_BOX, "row_lanes_unclear", id="numbered-inconsistently"
    ),
    pytest.param(None, [15, 45, W - 35, 60], "no_band_found", id="no-band"),
    pytest.param(_wide_kept_box, TARGET_ROW_BOX, "size_would_overlap", id="size-would-overlap"),
    pytest.param(_kept_box_on_lane_2, TARGET_ROW_BOX, "overlap", id="onto-a-kept-box"),
    # Over both rows: the target's bands in lanes 2-4, the loading control's
    # (deeper there) in lanes 0 and 1, so the boxes would lie on two rows (#114).
    pytest.param(None, BOTH_ROWS_BOX, "row_off_line", id="over-two-rows"),
    # Over the loading control's row: its boxes are in the way (#114).
    pytest.param(
        _loading_boxes,
        [15, LOADING_ROW - 12, W - 35, LOADING_ROW + 12],
        "overlap",
        id="onto-another-proteins-row",
    ),
]


@pytest.mark.parametrize(("setup", "rect", "code"), ROW_BOX_REFUSALS)
def test_a_refused_row_box_changes_nothing(client, tmp_path, setup, rect, code):
    target, loading, _ = live(client, tmp_path, DOSES, boxed=())
    ids = [] if setup is None else setup(client, target, loading)
    body = {"protein_id": target, "rect": rect}
    assert unchanged_refusal(client, "POST", "/api/boxes/row", body) == (code, ids)
    assert "detect_row_boxes" not in logged(client)


def test_a_row_box_is_for_a_known_protein_so_never_on_a_marker_image(client, tmp_path):
    target, _, answer = live(client, tmp_path, DOSES, boxed=())
    membrane = answer["project"]["images"][0]["membrane_id"]
    status, marker = upload(
        client, blot_bytes(tmp_path), "marker α.tif", kind="visible_marker", membrane_id=membrane
    )
    assert status == 201, marker
    marker_id = marker["image_id"]
    # No protein is on a visible-light marker image, and a row box is only ever a protein's.
    body = {"name": "GAPDH", "role": "loading control", "image_id": marker_id}
    assert unchanged_refusal(client, "POST", "/api/proteins", body) == ("marker_image", [marker_id])
    for protein in ("prot-99", marker_id):
        body = {"protein_id": protein, "rect": TARGET_ROW_BOX}
        assert unchanged_refusal(client, "POST", "/api/boxes/row", body) == ("unknown_id", [])
    body = {"protein_id": target, "rect": TARGET_ROW_BOX, "image_id": marker_id}
    assert unchanged_refusal(client, "POST", "/api/boxes/row", body) == ("invalid_input", [])
    assert "detect_row_boxes" not in logged(client)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"rect": [15, 18, 365, True]}, id="bool"),
        pytest.param({"rect": [15, 18, 365.0, 42]}, id="whole-float"),
        pytest.param({"rect": [15, 18, 365.5, 42]}, id="float"),
        pytest.param({"rect": [15, 18, 365]}, id="three-numbers"),
        pytest.param({"rect": [15, 18, 365, 42, 0]}, id="five-numbers"),
        pytest.param({"rect": ["15", 18, 365, 42]}, id="number-as-text"),
        pytest.param({"rect": "15 18 365 42"}, id="text"),
        pytest.param({"rect": {"x0": 15, "y0": 18, "x1": 365, "y1": 42}}, id="object"),
        pytest.param({"rect": None}, id="null"),
        pytest.param({}, id="no-rect"),
        pytest.param({"rect": [15, 18, 365, 42], "protein_id": None}, id="no-protein"),
        pytest.param({"rect": [15, 18, 365, 42], "lane_index": 0}, id="unknown-field"),
        pytest.param({"rect": [-(2**31) - 1, 18, 365, 42]}, id="below-32-bits"),
        pytest.param({"rect": [15, 18, 365, 2**31]}, id="above-32-bits"),
        # JSON reads 4300 digits and a sign, but a project file holding the row
        # as the log keeps it would no longer read.
        pytest.param({"rect": [-(10**4299), 18, 365, 42]}, id="4300-digits"),
    ],
)
def test_a_malformed_row_box_is_refused(client, tmp_path, body):
    _, protein = ready(client, tmp_path)
    body = {"protein_id": protein, **body}
    assert unchanged_refusal(client, "POST", "/api/boxes/row", body) == ("invalid_input", [])


# --- The export bundle (#53) ---

SAMPLE = "Sample µ"
SAMPLE_FILES = [
    "lane-table (Excluding lane 4).csv",
    "lane-table (All lanes).csv",
    "chart β-catenin ÷ α-tubulin (Excluding lane 4).svg",
    "chart β-catenin ÷ α-tubulin (Excluding lane 4).png",
    "chart β-catenin ÷ α-tubulin (All lanes).svg",
    "chart β-catenin ÷ α-tubulin (All lanes).png",
    "README.txt",
    "export.record.json",
]


def open_sample(client: Client) -> dict:
    """The conftest sample project, saved in the projects root and opened."""
    project = make_project()
    folder = client.root / SAMPLE
    write_image_files(folder, project)
    storage.save_project(project, folder)
    return client.ok("POST", "/api/projects/open", {"name": SAMPLE})


def export_folders(client: Client) -> list[str]:
    exports = client.root / SAMPLE / storage.EXPORTS_DIR
    return sorted(path.name for path in exports.iterdir()) if exports.is_dir() else []


def test_an_export_answers_its_folder_and_files(client):
    before = open_sample(client)
    project_json = (client.root / SAMPLE / storage.PROJECT_FILE).read_bytes()
    status, answer = client.call("POST", "/api/export", {})
    assert status == 201, answer
    name = answer["folder"].removeprefix("exports/")
    assert answer["folder"] == f"exports/{name}" and "/" not in name
    assert answer["files"] == SAMPLE_FILES
    folder = client.root / SAMPLE / storage.EXPORTS_DIR / name
    assert sorted(path.name for path in folder.iterdir()) == sorted(SAMPLE_FILES)
    # Not a change: the same project and results, at the same revision.
    assert answer["project"] == before["project"]
    assert answer["results"] == before["results"]
    assert (client.root / SAMPLE / storage.PROJECT_FILE).read_bytes() == project_json

    status, answer = client.call("POST", "/api/export")  # no body: the default formats
    assert status == 201 and answer["files"] == SAMPLE_FILES
    answer = client.ok("POST", "/api/export", {"formats": ["pdf"]})
    assert [name for name in answer["files"] if name.startswith("chart ")] == [
        "chart β-catenin ÷ α-tubulin (Excluding lane 4).pdf",
        "chart β-catenin ÷ α-tubulin (All lanes).pdf",
    ]
    assert len(export_folders(client)) == 3


def test_an_export_of_names_whose_columns_would_clash_answers_201(client):
    # A name the routes take: a protein named like a series column.
    open_sample(client)
    client.ok("PATCH", "/api/proteins/prot-9", {"name": "β-catenin ÷ α-tubulin normalized"})
    status, answer = client.call("POST", "/api/export", {"formats": ["svg"]})
    assert status == 201, answer
    table = client.root / SAMPLE / answer["folder"] / answer["files"][0]
    assert table.read_bytes().decode("utf-8-sig").splitlines()[0].split(",")[8:] == [
        "β-catenin ÷ α-tubulin normalized (2)",
        "β-catenin ÷ α-tubulin normalized (2) clipped",
        "β-catenin ÷ α-tubulin normalized",
        "β-catenin ÷ α-tubulin fold change vs vehicle",
    ]

    # Names a project.json may hold: "GAPDH clipped" next to GAPDH.
    project = make_project_with_clashing_names()
    write_image_files(client.root / "Clash", project)
    storage.save_project(project, client.root / "Clash")
    client.ok("POST", "/api/projects/open", {"name": "Clash"})
    status, answer = client.call("POST", "/api/export", {"formats": ["svg"]})
    assert status == 201, answer
    table = client.root / "Clash" / answer["folder"] / answer["files"][0]
    assert table.read_bytes().decode("utf-8-sig").splitlines()[0].split(",")[4:10] == [
        "GAPDH",
        "GAPDH clipped",
        "α-tubulin",
        "α-tubulin clipped",
        "GAPDH clipped (2)",
        "GAPDH clipped (2) clipped",
    ]


def test_an_export_uses_the_workspaces_result_settings(client):
    open_sample(client)
    client.workspace._settings = api.ResultSettings(
        plot_conditions=("vehicle",),
        error_type=ErrorType.SEM,
        method=ReduceMethod.REPRESENTATIVE,
    )
    answer = client.ok("POST", "/api/export", {"formats": ["svg"]})
    folder = client.root / SAMPLE / answer["folder"]
    doc = json.loads((folder / "export.record.json").read_bytes())
    assert {key: value for key, value in doc["results"].items() if key != "statistics"} == {
        "method": "representative",
        "error_type": "SEM",
        "plot_conditions": ["vehicle"],
        "excluded_lanes": [3],
    }
    # What the screen shows, each setting of it.
    auto = {"family": "auto", "comparisons": "auto", "scale": "auto"}
    assert doc["results"]["statistics"]["setting"] == auto
    settings = {
        "error_type": "SEM",
        "plot_conditions": ["vehicle"],
        "method": "representative",
        "statistics": auto,
    }
    assert answer["results"]["settings"] == settings
    [chart] = answer["results"]["sets"][0]["series"]
    assert chart["chart"]["error_type"] == "SEM"
    assert [bar["label"] for bar in chart["chart"]["bars"]] == ["vehicle"]
    svg = (folder / "chart β-catenin ÷ α-tubulin (Excluding lane 4).svg").read_bytes()
    assert svg == render_svg(PlotSpec.model_validate(chart["chart"]))


def test_an_export_folder_is_revealed_on_request(client):
    open_sample(client)
    answer = client.ok("POST", "/api/export", {"formats": []})
    assert answer["files"] == SAMPLE_FILES[:2] + SAMPLE_FILES[-2:]  # an empty list: no chart
    assert client.call("POST", "/api/project/reveal", {"folder": answer["folder"]})[0] == 204
    assert client.revealed == [client.root / SAMPLE / answer["folder"]]
    assert client.call("POST", "/api/project/reveal")[0] == 204  # the project folder
    assert client.revealed[-1] == client.root / SAMPLE


@pytest.mark.parametrize(
    "folder",
    [
        "",
        "exports",
        "exports/",
        "images",
        "project.json",
        "exports/.",
        "exports/..",
        "exports/../images",
        "../exports/x",
        "exports/a/b",
        "exports\\x",
        "/exports/x",
        "C:/exports/x",
        "exports/C:x",
        "exports/...",  # Windows drops the dots: the exports folder itself
        "exports/ ",
        "exports/x.",
        "exports/a\u0007b",
    ],
)
def test_only_an_export_folder_can_be_revealed(client, folder):
    open_sample(client)
    (client.root / SAMPLE / "exports" / "x").mkdir()  # "exports/x." would reach it on Windows
    assert client.refused("POST", "/api/project/reveal", {"folder": folder})[:2] == (
        422,
        "invalid_input",
    )
    assert client.revealed == []


def test_a_missing_export_folder_is_not_revealed(client):
    open_sample(client)
    (client.root / SAMPLE / "exports" / "a file").write_bytes(b"")
    for folder in ("exports/2026-01-01 0000", "exports/a file"):
        assert client.refused("POST", "/api/project/reveal", {"folder": folder})[:2] == (
            404,
            "folder_not_found",
        )
    assert client.revealed == []


def test_an_export_is_refused_with_a_code_and_writes_nothing(client):
    assert client.refused("POST", "/api/export", {})[:2] == (409, "no_project")
    client.ok("POST", "/api/projects", {"name": "Blot"})
    assert client.refused("POST", "/api/export", {})[:2] == (422, "no_lanes")
    assert list((client.root / "Blot" / "exports").iterdir()) == []

    open_sample(client)
    for body in (
        {"formats": ["svg", "gif"]},
        {"formats": ["SVG"]},
        {"formats": "svg"},
        {"formats": [1]},
        {"format": ["svg"]},
    ):
        assert client.refused("POST", "/api/export", body)[:2] == (422, "invalid_input"), body
    (client.root / SAMPLE / "images" / "img-2.tif").write_bytes(b"other pixels")
    assert client.refused("POST", "/api/export", {}) == (422, "image_file_changed", ["img-2"])
    assert export_folders(client) == []


def test_an_export_under_too_long_a_path_is_refused(client, monkeypatch):
    open_sample(client)
    project = os.path.abspath(client.root / SAMPLE)
    # A path limit that leaves an export's files too little room.
    monkeypatch.setattr(storage, "PATH_LIMIT", len(project) + 60)
    assert client.refused("POST", "/api/export", {})[:2] == (422, "path_too_long")
    assert export_folders(client) == []


# --- The sample project (#55) ---

SAMPLE_BLOT = "Sample blot"
SAMPLE_LOG = [
    "new_project",
    "import_image",
    "import_image",
    "set_lanes",
    "add_protein",
    "add_protein",
]


def open_sample_blot(client: Client) -> dict:
    """The answer to "Open sample project": 201, as a create."""
    status, answer = client.call("POST", "/api/projects/sample")
    assert status == 201, (status, answer)
    return answer


def setup_of(project: dict) -> dict:
    """What the sample project's setup made, without the revision or history."""
    return {key: project[key] for key in ("lanes", "reference_condition", "images", "proteins")}


def test_the_sample_project_is_set_up_up_to_the_row_boxes(client):
    answer = open_sample_blot(client)
    project = answer["project"]
    assert project["name"] == sample_project.SAMPLE_NAME == SAMPLE_BLOT
    assert project["saved"] and client.ok("GET", "/api/projects")["open"] == SAMPLE_BLOT

    # The blot and its marker image, on one membrane; the bytes proteia.samples writes.
    blot, marker = project["images"]
    fields = ("original_name", "kind", "polarity", "width", "height", "bit_depth", "warnings")
    assert [[image[field] for field in fields] for image in (blot, marker)] == [
        ["sample-blot.tif", "chemiluminescence", "dark_on_light", 1200, 500, 16, []],
        ["sample-marker.tif", "visible_marker", "dark_on_light", 1200, 500, 8, []],
    ]
    assert marker["membrane_id"] == blot["membrane_id"]
    files = samples.sample_files()
    batch = client.workspace.current().project.batch
    for image in (blot, marker):
        data = files[image["original_name"]]
        assert batch.find_image(image["id"]).sha256 == hashlib.sha256(data).hexdigest()

    # The design, written out here rather than read back from the module.
    assert [(lane["condition"], lane["sample"], lane["included"]) for lane in project["lanes"]] == [
        ("vehicle", "V1", True),
        ("vehicle", "V2", True),
        ("vehicle", "V3", True),
        ("vehicle", "V4", True),
        ("treatment", "T1", True),
        ("treatment", "T2", True),
        ("treatment", "T3", True),
        ("treatment", "T4", True),
    ]
    assert project["reference_condition"] == "vehicle"
    loading, target = project["proteins"]
    assert [
        (p["name"], p["role"], p["image_id"], p["loading_control_ids"], p["bands"], p["undetected"])
        for p in (loading, target)
    ] == [
        ("α-tubulin", "loading control", blot["id"], [], [], []),
        ("β-catenin", "target", blot["id"], [loading["id"]], [], []),
    ]
    assert column(answer, target["id"])["nets"] == [None] * 8

    # Each step an ordinary logged operation, saved.
    folder = client.root / SAMPLE_BLOT
    assert [entry.action for entry in storage.load_project(folder).log] == SAMPLE_LOG
    assert project["history"]["undo"]["action"] == "add_protein"

    # The truth table next to project.json, never among the exports.
    assert sorted(path.name for path in folder.iterdir()) == [
        "exports",
        "images",
        "project.json",
        "sample-truth.csv",
    ]
    assert list((folder / storage.EXPORTS_DIR).iterdir()) == []
    assert (folder / "sample-truth.csv").read_bytes() == files[samples.TRUTH_FILE]

    # Each protein's row, top to bottom: a drag that spans every lane's band.
    sample = answer["sample"]
    assert sample["truth_file"] == "sample-truth.csv"
    assert [row["protein_id"] for row in sample["rows"]] == [target["id"], loading["id"]]
    (_, _, _, top_y1), (_, low_y0, _, _) = (row["rect"] for row in sample["rows"])
    assert top_y1 <= low_y0  # the rows do not overlap
    half = samples.BAND_WIDTH / 2
    for row, kda in zip(sample["rows"], (92.0, 50.0), strict=True):
        x0, y0, x1, y1 = row["rect"]
        assert x0 < samples.LANE_X[0] - half and samples.LANE_X[-1] + half < x1 <= 1200
        for x in samples.LANE_X:
            assert y0 + 10 < samples.band_y(kda, x) < y1 - 10, (kda, x)


def test_undo_takes_the_sample_setup_back_step_by_step(client):
    made = setup_of(open_sample_blot(client)["project"])
    steps = []
    for _ in SAMPLE_LOG[1:]:
        undone = client.ok("POST", "/api/undo")
        steps.append(undone["action"])
    assert steps == SAMPLE_LOG[:0:-1]  # every step, the last first
    assert setup_of(undone["project"]) == {
        "lanes": [],
        "reference_condition": None,
        "images": [],
        "proteins": [],
    }
    assert undone["project"]["history"]["undo"] is None  # the empty project it was created as
    # The truth table is not part of the project: undo leaves it.
    assert (client.root / SAMPLE_BLOT / "sample-truth.csv").is_file()

    for _ in SAMPLE_LOG[1:]:
        redone = client.ok("POST", "/api/redo")
    assert setup_of(redone["project"]) == made


def test_each_sample_project_takes_the_next_free_name(client):
    first = open_sample_blot(client)
    second = open_sample_blot(client)
    assert (first["project"]["name"], second["project"]["name"]) == (
        SAMPLE_BLOT,
        "Sample blot (2)",
    )
    assert second["project"]["open_id"] > first["project"]["open_id"]
    # Taken ignoring case and look-alikes, as project names are, by any folder.
    (client.root / "SAMPLE BLOT （3）").mkdir()  # fullwidth parentheses
    third = open_sample_blot(client)
    assert third["project"]["name"] == "Sample blot (4)"
    listing = client.ok("GET", "/api/projects")
    assert listing["open"] == "Sample blot (4)"
    names = [SAMPLE_BLOT, "Sample blot (2)", "Sample blot (4)"]
    assert sorted(p["name"] for p in listing["projects"]) == names
    for name in names:  # each a whole sample project of its own
        assert [e.action for e in storage.load_project(client.root / name).log] == SAMPLE_LOG
    assert setup_of(third["project"]) == setup_of(first["project"])


def test_the_sample_rows_box_every_lane_and_give_the_documented_fold_change(client):
    answer = open_sample_blot(client)
    for row in answer["sample"]["rows"]:
        answer = drag(client, row["protein_id"], row["rect"])
        assert len(answer["band_ids"]) == 8 and None not in answer["band_ids"]
        assert (answer["flags"], answer["empty"], answer["notes"]) == ([], [], [])
    assert [len(protein["bands"]) for protein in answer["project"]["proteins"]] == [8, 8]
    [result_set] = answer["results"]["sets"]
    assert result_set["tier"] == "fold_change"
    series = only_series(answer)
    vehicle, treatment = bar(series, "vehicle"), bar(series, "treatment")
    assert (vehicle["n"], treatment["n"]) == (4, 4)
    assert vehicle["mean"] == pytest.approx(1.0)  # the baseline is the vehicle mean
    # The truth: a treatment mean of 2.00 (proteia.samples). measured 2.0088
    assert treatment["mean"] == pytest.approx(2.0, rel=0.03)


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (OSError(28, "No space left on device"), 500, "file_error"),
        (api.ops.OperationError(api.ops.ErrorCode.INVALID_INPUT, "refused"), 422, "invalid_input"),
    ],
)
def test_a_sample_project_that_fails_midway_leaves_no_folder(
    client, monkeypatch, error, status, code
):
    client.ok("POST", "/api/projects", {"name": "Blot"})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    # After the images are stored: their files go with the folder.
    monkeypatch.setattr(sample_project.ops, "set_lanes", fail)
    assert client.refused("POST", "/api/projects/sample")[:2] == (status, code)
    assert sorted(path.name for path in client.root.iterdir()) == ["Blot"]
    assert client.ok("GET", "/api/projects")["open"] == "Blot"  # still open


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (OSError(28, "No space left on device"), "[Errno 28] No space left on device"),
        (api.ops.OperationError(api.ops.ErrorCode.INVALID_INPUT, "refused"), "refused"),
    ],
)
def test_a_sample_folder_that_cannot_be_removed_is_named_in_the_error(
    client, monkeypatch, error, text
):
    client.ok("POST", "/api/projects", {"name": "Blot"})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    def rmtree_but_images(path: Path, ignore_errors: bool = False) -> None:
        # What shutil.rmtree removes while another program holds the stored images open.
        for item in sorted(Path(path).rglob("*"), reverse=True):  # children first
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
            elif item.is_file() and item.parent.name != "images":
                item.unlink()

    with monkeypatch.context() as patch:
        patch.setattr(sample_project.ops, "set_lanes", fail)
        patch.setattr(api.projects, "shutil", SimpleNamespace(rmtree=rmtree_but_images))
        status, answer = client.call("POST", "/api/projects/sample")
    # A file error whatever failed: the folder left needs deleting by hand.
    assert (status, answer["code"]) == (500, "file_error")
    assert answer["message"] == (
        f"the sample project could not be set up ({text}), and its unfinished folder"
        f" '{SAMPLE_BLOT}' could not be removed: delete it from the projects folder"
    )
    folder = client.root / SAMPLE_BLOT
    assert {path.relative_to(folder).parts[0] for path in folder.rglob("*")} == {"images"}
    listing = client.ok("GET", "/api/projects")
    assert (listing["open"], [entry["name"] for entry in listing["projects"]]) == ("Blot", ["Blot"])
    # Its name stays taken until it is deleted: the next sample is numbered.
    assert open_sample_blot(client)["project"]["name"] == f"{SAMPLE_BLOT} (2)"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to delete a file held open")
def test_a_sample_image_held_open_leaves_its_folder_named_in_the_error(client, monkeypatch):
    held: list[io.BufferedReader] = []

    def hold_and_fail(session: Any, *args: Any, **kwargs: Any) -> None:
        held.append(next((session.folder / "images").iterdir()).open("rb"))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sample_project.ops, "set_lanes", hold_and_fail)
    try:
        status, answer = client.call("POST", "/api/projects/sample")
    finally:
        for handle in held:
            handle.close()
    assert (status, answer["code"]) == (500, "file_error")
    assert f"unfinished folder '{SAMPLE_BLOT}' could not be removed" in answer["message"]
    folder = client.root / SAMPLE_BLOT
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*")) == [
        "images",
        f"images/{Path(held[0].name).name}",
    ]
    assert client.ok("GET", "/api/projects")["projects"] == []


def test_a_sample_project_in_a_projects_root_that_cannot_be_written_answers_json(client):
    client.root.parent.mkdir(parents=True, exist_ok=True)
    client.root.write_text("a file where the projects folder should be", encoding="utf-8")
    assert client.refused("POST", "/api/projects/sample")[:2] == (500, "file_error")
    assert client.refused("GET", "/api/project")[:2] == (409, "no_project")


# --- A page that shows a project no longer open (#134) ---

OPENING = "Proteia-Opening"

# The routes that need no opening, since none reads or edits the open project,
# and why. Every other route under /api refuses a request that names another
# opening (found from the app's routes), so a new route is guarded unless it is
# added here.
NEEDS_NO_OPENING = {
    ("GET", "/api/status"): "says which app answers",
    ("POST", "/api/quit"): "stops Proteia, whichever project the page shows",
    ("GET", "/api/projects"): "lists the projects root",
    ("POST", "/api/projects"): "creates the project the request names, and opens it",
    ("POST", "/api/projects/open"): "opens the project the request names",
    ("POST", "/api/projects/sample"): "creates the sample project, and opens it",
    ("GET", "/api/workspace"): "says which opening is open: how a page finds it changed",
    ("POST", "/api/incoming"): "stages a file a launch hands to the app",
    ("POST", "/api/handoffs"): "hands off staged files, for the page to ask about",
    (
        "POST",
        "/api/handoffs/{handoff_id}/discard",
    ): "drops files handed off, as the page shows them",
    ("POST", "/api/diagnostics/reveal"): "shows the diagnostics folder, in the state folder",
}


def _declared(routes: list[Any]) -> Iterator[APIRoute]:
    """The routes in ``routes`` and in the routers included there: FastAPI keeps
    an included router as one route that holds it (``original_router``)."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from _declared(route.original_router.routes)
        else:
            assert isinstance(route, Mount) and route.path == "/static", route


def api_routes() -> set[tuple[str, str]]:
    """Every (method, path) the app serves under /api, as its routes declare them."""
    workspace = api.Workspace(Path("unused"), reveal=lambda folder: None)
    app = server.create_app(
        token="t" * 43, port=8000, on_quit=lambda: None, workspace=workspace
    ).app
    routes = {
        (method, route.path)
        for route in _declared(app.routes)
        if route.path.startswith("/api/")
        for method in route.methods
    }
    described = {
        (method.upper(), path)
        for path, methods in app.openapi()["paths"].items()
        for method in methods
    }
    assert described <= routes  # every route found
    return routes


def files_of(folder: Path) -> dict[str, bytes]:
    """Every file in ``folder``, by its path there, with its bytes."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def opened_elsewhere(client: Client, tmp_path: Path) -> tuple[dict, int]:
    """Blot set up (:func:`live`); then a page opened Other, and another page
    Blot again. Gives the answer that page got, and the opening the first page
    shows (Other's)."""
    live(client, tmp_path, DOSES)
    shown = client.ok("POST", "/api/projects", {"name": "Other"})["project"]["open_id"]
    return client.ok("POST", "/api/projects/open", {"name": "Blot"}), shown


def test_every_route_on_the_open_project_refuses_another_opening(client, tmp_path):
    # A page still showing Other must neither edit nor read Blot, whose ids are
    # Other's too (img-1, prot-1). Each route is sent with the ids Blot has and
    # no body: the refusal comes before the body is read, and changes nothing.
    # So is each route that takes its session in use only later
    # (USES_ITS_SESSION_LATER): it checks the opening as early.
    routes = api_routes()
    assert set(NEEDS_NO_OPENING) <= routes  # none of them has gone
    guarded = sorted(routes - set(NEEDS_NO_OPENING))
    assert {("GET", "/api/project"), ("PUT", "/api/boxes/{band_id}")} <= set(guarded)
    assert set(USES_ITS_SESSION_LATER) <= set(guarded)
    answer, shown = opened_elsewhere(client, tmp_path)
    project = answer["project"]
    protein = next(p for p in project["proteins"] if p["bands"])
    ids = {
        "image_id": project["images"][0]["id"],
        "protein_id": protein["id"],
        "band_id": protein["bands"][0]["id"],
        "lane_index": "0",
        "key": only_series(answer)["chart_url"].removeprefix("/api/charts/").removesuffix(".svg"),
    }
    paths = [
        (method, re.sub(r"\{(\w+)\}", lambda m: ids.get(m[1], "0"), path))
        for method, path in guarded
    ]
    paths.append(("GET", f"/api/images/{ids['image_id']}/preview?colour=original"))
    folder = client.root / "Blot"
    before = files_of(folder)

    answered = {}
    for method, path in paths:
        status, payload = client.call(method, path, headers={OPENING: str(shown)})
        got = (payload["code"], payload.get("detail")) if isinstance(payload, dict) else payload
        answered[f"{method} {path}"] = (status, got)
    status, payload = client.call(  # an upload: refused before its body is stored
        "POST",
        "/api/images?name=a.tif&kind=chemiluminescence&polarity=dark_on_light",
        raw=blot_bytes(tmp_path),
        headers={OPENING: str(shown)},
    )
    answered["POST /api/images with a file"] = (status, (payload["code"], payload.get("detail")))

    refused = (409, ("project_changed", {"open": "Blot", "open_id": project["open_id"]}))
    assert {route: got for route, got in answered.items() if got != refused} == {}
    assert files_of(folder) == before
    assert client.revealed == []
    after = client.ok("GET", "/api/project")["project"]
    assert (after["open_id"], after["revision"]) == (project["open_id"], project["revision"])


def test_a_request_naming_the_open_opening_or_none_is_served(client, tmp_path):
    answer, _ = opened_elsewhere(client, tmp_path)
    project = answer["project"]
    now = {OPENING: str(project["open_id"])}
    read = client.ok("GET", "/api/project", headers=now)["project"]
    assert (read["open_id"], read["revision"]) == (project["open_id"], project["revision"])
    band = next(p for p in project["proteins"] if p["bands"])["bands"][0]
    x0, y0, x1, y1 = band["rect"]
    rect = [x0 + 2, y0, x1 + 2, y1]
    moved = client.ok("PUT", f"/api/boxes/{band['id']}", {"rect": rect}, headers=now)
    assert moved["project"]["revision"] == project["revision"] + 1
    image_id = project["images"][0]["id"]
    status, (kind, _) = client.call("GET", f"/api/images/{image_id}/preview", headers=now)
    assert (status, kind) == (200, "image/png")
    status, (kind, _) = client.call("GET", only_series(moved)["chart_url"], headers=now)
    assert (status, kind) == (200, "image/svg+xml")
    # Without the header a request acts on whichever project is open, as before.
    assert client.ok("POST", "/api/undo")["action"] == "move_box"


@pytest.mark.parametrize(
    "given",
    ["", "one", "01", "+2", "2.0", "2_0", "0x2", pytest.param("9" * 5000, id="5000 digits")],
)
def test_an_opening_not_in_plain_digits_is_refused(client, given):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    client.ok("POST", "/api/projects", {"name": "Other"})  # open id 2
    before = files_of(client.root / "Other")
    lanes = {"lanes": [{"condition": "vehicle"}]}
    status, code, _ = client.refused("PUT", "/api/lanes", lanes, headers={OPENING: given})
    assert (status, code) == (422, "invalid_input")
    assert files_of(client.root / "Other") == before


def test_two_openings_in_one_request_are_refused(client):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    try:
        conn.putrequest("GET", "/api/project")
        conn.putheader("Authorization", f"Bearer {client.token}")
        conn.putheader(OPENING, "1")
        conn.putheader(OPENING, "1")
        conn.endheaders()
        response = conn.getresponse()
        status, body = response.status, json.loads(response.read())
    finally:
        conn.close()
    assert (status, body["code"]) == (422, "invalid_input")


def test_the_routes_that_need_no_opening_ignore_the_one_named(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    stale = {OPENING: "7"}  # an opening that never was
    # The page shell, served without the token, holds no user data.
    for path in ("/", "/static/app.js"):
        assert client.call("GET", path, headers=stale)[0] == 200
    bodies: dict[tuple[str, str], Any] = {
        ("GET", "/api/status"): None,
        ("GET", "/api/projects"): None,
        ("GET", "/api/workspace"): None,
        ("POST", "/api/projects"): {"name": "Other"},
        ("POST", "/api/projects/open"): {"name": "Blot"},
        ("POST", "/api/projects/sample"): None,
        ("POST", "/api/quit"): None,  # last: it stops the server
    }
    handoff_id, files = handed_off(client, tmp_path, ["dropped.tif"])
    incoming = ("POST", "/api/incoming")
    discard = ("POST", "/api/handoffs/{handoff_id}/discard")
    bodies[incoming] = None
    bodies[discard] = {"files": [file["file_id"] for file in files]}
    bodies[("POST", "/api/handoffs")] = {
        "refused": [{"name": "a.bmp", "code": "x", "message": "y"}]
    }
    bodies[("POST", "/api/diagnostics/reveal")] = None
    bodies[("POST", "/api/quit")] = bodies.pop(("POST", "/api/quit"))  # still last
    paths = {incoming: "/api/incoming?name=a.tif", discard: f"/api/handoffs/{handoff_id}/discard"}
    assert set(bodies) == set(NEEDS_NO_OPENING)
    answers = {
        route: client.call(
            route[0],
            paths.get(route, route[1]),
            body,
            raw=blot_bytes(tmp_path) if route == incoming else None,
            headers=stale,
        )
        for route, body in bodies.items()
    }
    assert {route: status for route, (status, _) in answers.items()} == {
        ("GET", "/api/status"): 200,
        ("GET", "/api/projects"): 200,
        ("GET", "/api/workspace"): 200,
        ("POST", "/api/projects"): 201,
        ("POST", "/api/projects/open"): 200,
        ("POST", "/api/projects/sample"): 201,
        incoming: 201,
        discard: 204,
        ("POST", "/api/handoffs"): 201,
        ("POST", "/api/diagnostics/reveal"): 204,
        ("POST", "/api/quit"): 202,
    }
    opened = [
        answers[("POST", path)][1]["project"] for path in ("/api/projects", "/api/projects/open")
    ]
    assert [(p["name"], p["open_id"]) for p in opened] == [("Other", 2), ("Blot", 3)]


def test_the_workspace_says_which_opening_is_open_without_reading_it(client, monkeypatch):
    calls: list = []
    _counting(monkeypatch, calls)
    root = str(client.root)
    none: dict[str, Any] = {"handoffs": []}
    assert client.ok("GET", "/api/workspace") == {
        "root": root,
        "open": None,
        "open_id": None,
        **none,
    }
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    client.ok("POST", "/api/projects", {"name": "Other"})
    computed = len(calls)
    assert client.ok("GET", "/api/workspace") == {
        "root": root,
        "open": "Other",
        "open_id": 2,
        **none,
    }
    client.ok("POST", "/api/projects/open", {"name": "Blot µ"})
    assert client.ok("GET", "/api/workspace") == {
        "root": root,
        "open": "Blot µ",
        "open_id": 3,
        **none,
    }
    assert len(calls) == computed + 1  # the open's answer only


def test_the_open_session_is_given_only_for_its_own_opening(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    with pytest.raises(api.NoProjectError):
        workspace.current(1)
    first = workspace.create("A µ")
    assert workspace.current(1) is first
    second = workspace.create("B")
    with pytest.raises(api.ProjectChangedError) as refused:
        workspace.current(1)
    assert (refused.value.open, refused.value.open_id) == ("B", 2)
    assert workspace.current(2) is workspace.current() is second


def test_a_project_read_again_refuses_its_earlier_opening(client, tmp_path):
    # Its project.json changed outside Proteia (a synced copy): the reopen reads
    # it again under the next open id, and a page showing it as it was before
    # is refused until it reads it again.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    session = client.workspace.current()
    other = tmp_path / "copy µ"
    shutil.copytree(session.folder, other)
    remote = api.ops.open_project(other, clock=FakeClock())
    api.ops.set_lanes(remote, [api.ops.LaneInput("drug")])
    shutil.copytree(other, session.folder, dirs_exist_ok=True)
    assert client.ok("POST", "/api/projects/open", {"name": "Blot"})["project"]["open_id"] == 2

    lanes = {"lanes": [{"condition": "vehicle"}, {"condition": "stale"}]}
    status, payload = client.call("PUT", "/api/lanes", lanes, headers={OPENING: "1"})
    assert (status, payload["code"], payload["detail"]) == (
        409,
        "project_changed",
        {"open": "Blot", "open_id": 2},
    )
    saved = storage.load_project(session.folder)
    assert [lane.label for lane in saved.batch.lanes] == ["drug"]


# --- A request runs within the opening it named (#134's review) ---


def _held_until_released(
    monkeypatch, owner: Any, name: str
) -> tuple[threading.Event, threading.Event]:
    """Hold each call of ``owner.name`` (what a route calls once its request is
    past the check) until released: (entered, release)."""
    entered, release = threading.Event(), threading.Event()
    original = getattr(owner, name)

    def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, held)
    return entered, release


def _reads_recorded(monkeypatch, session: api.ProjectSession) -> threading.Event:
    """Set once ``session`` is asked to read its project.json again."""
    asked = threading.Event()
    reload = session.reload

    def reloading() -> bool:
        asked.set()
        return reload()

    monkeypatch.setattr(session, "reload", reloading)
    return asked


def _in_thread(target: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
    """``target`` started in a thread: (the thread, the list its result goes to)."""
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(target()))
    thread.start()
    return thread, result


def test_an_edit_past_the_check_is_made_in_its_opening_before_a_reopen_reads_the_file(
    client, tmp_path, monkeypatch
):
    # A page shows opening 1 and moves a box; its request passes the check, and
    # before the move runs another page opens the project again, whose
    # project.json changed outside Proteia: in another copy the band's id went
    # to another box (the same next id). Read again first, the move would be
    # made to that box, in a project the page never showed, and its answer, of
    # opening 2, would be the page's first sign of that opening. The reopen
    # waits for the request instead: the move is made to the box the page shows
    # and answered as of opening 1. Saved, it replaces the outside change, as an
    # edit made just before the reopen does, so the reopen finds nothing to read.
    target_id, _, _ = live(client, tmp_path, DOSES, boxed=(0, 1, 2))
    session = client.workspace.current()
    other = tmp_path / "copy µ"
    shutil.copytree(session.folder, other)
    placed = client.ok(
        "POST",
        "/api/boxes",
        {"protein_id": target_id, "x": LANE_X[3], "y": TARGET_ROW, "lane_index": 3},
    )
    band_id, shown = placed["band_id"], placed["project"]["open_id"]
    remote = api.ops.open_project(other, clock=FakeClock())
    elsewhere = api.ops.place_box(
        remote, target_id, LANE_X[4], TARGET_ROW, lane_index=4, grow=False
    )
    assert elsewhere == band_id
    shutil.copytree(other, session.folder, dirs_exist_ok=True)
    band = next(b for p in placed["project"]["proteins"] for b in p["bands"] if b["id"] == band_id)
    x0, y0, x1, y1 = band["rect"]
    rect = [x0 + 2, y0, x1 + 2, y1]

    entered, release = _held_until_released(monkeypatch, api.ops, "move_box")
    asked = _reads_recorded(monkeypatch, session)
    moving, moved = _in_thread(
        lambda: client.call(
            "PUT", f"/api/boxes/{band_id}", {"rect": rect}, headers={OPENING: str(shown)}
        )
    )
    assert entered.wait(20)
    reopening, reopened = _in_thread(
        lambda: client.call("POST", "/api/projects/open", {"name": "Blot"})
    )
    reopening.join(1.0)  # time to read the file, were it not waiting for the move
    read_meanwhile = asked.is_set()
    release.set()
    for thread in (moving, reopening):
        thread.join(30)
        assert not thread.is_alive()

    (status, answer), (reopen_status, reopen) = moved[0], reopened[0]
    assert status == 200, answer
    assert answer["project"]["open_id"] == shown
    box = next(b for p in answer["project"]["proteins"] for b in p["bands"] if b["id"] == band_id)
    assert (box["lane_index"], box["rect"]) == (3, rect)
    assert not read_meanwhile
    assert reopen_status == 200, reopen
    assert (reopen["project"]["open_id"], reopen["project"]["revision"]) == (
        shown,
        answer["project"]["revision"],
    )
    _, saved = storage.load_project(session.folder).batch.find_band(band_id)
    assert (saved.lane_index, list(saved.box.rect(BoxSize(width=SIZE[0], height=SIZE[1])))) == (
        3,
        rect,
    )


def test_a_read_past_the_check_answers_its_opening_before_a_reopen_reads_the_file(
    client, tmp_path, monkeypatch
):
    # The same race with a read: were the file read again while it runs, the
    # answer to a request naming opening 1 would be of opening 2, or of opening
    # 1 with opening 2's project in it. The reopen waits for it, then reads the
    # file: the read answers opening 1 as it was, the reopen opening 2.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    shown = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})["project"]
    session = client.workspace.current()
    other = tmp_path / "copy µ"
    shutil.copytree(session.folder, other)
    api.ops.set_lanes(api.ops.open_project(other, clock=FakeClock()), [api.ops.LaneInput("drug")])
    shutil.copytree(other, session.folder, dirs_exist_ok=True)

    entered, release = _held_until_released(monkeypatch, api, "_answer")
    asked = _reads_recorded(monkeypatch, session)
    reading, read = _in_thread(
        lambda: client.call("GET", "/api/project", headers={OPENING: str(shown["open_id"])})
    )
    assert entered.wait(20)
    reopening, reopened = _in_thread(
        lambda: client.call("POST", "/api/projects/open", {"name": "Blot"})
    )
    reopening.join(1.0)  # time to read the file, were it not waiting for the read
    read_meanwhile = asked.is_set()
    release.set()
    for thread in (reading, reopening):
        thread.join(30)
        assert not thread.is_alive()

    (status, answer), (reopen_status, reopen) = read[0], reopened[0]
    assert status == 200, answer
    assert answer["project"] == shown
    assert answer["results"]["open_id"] == shown["open_id"]
    assert not read_meanwhile
    assert reopen_status == 200, reopen
    assert reopen["project"]["open_id"] == shown["open_id"] + 1
    assert [lane["condition"] for lane in reopen["project"]["lanes"]] == ["drug"]


def test_a_request_made_while_a_reopen_waits_is_checked_against_the_opening_it_leaves(tmp_path):
    # A reopen waits for the requests in use on the session; one made
    # meanwhile waits for the reopen, so it never runs between the two: then it
    # is checked against the opening the reopen leaves (here a new one).
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    project_file = session.folder / storage.PROJECT_FILE
    empty = project_file.read_bytes()
    api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])
    project_file.write_bytes(empty)  # changed outside Proteia

    def later() -> Any:
        try:
            with workspace.using(1):
                return "served"
        except api.ProjectChangedError as refused:
            return refused.open_id

    with workspace.using(1) as used:
        assert used is session
        reopening, reopened = _in_thread(lambda: workspace.open("A µ"))
        deadline = time.monotonic() + 10
        while workspace._reopening is not session:  # it waits for the request in use
            assert reopening.is_alive() and time.monotonic() < deadline
            time.sleep(0.005)
        asking, asked = _in_thread(later)
        asking.join(0.3)
        assert asking.is_alive()  # it waits for the reopen
        assert workspace.view(session)[0] == 1 and session.project.batch.lanes  # nothing read
    for thread in (reopening, asking):
        thread.join(10)
        assert not thread.is_alive()
    assert reopened == [session]
    assert asked == [2]
    assert workspace.view(session)[0] == 2 and not session.project.batch.lanes


def test_a_reopen_leaves_the_file_unread_while_a_request_runs_past_the_wait(tmp_path, monkeypatch):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    project_file = session.folder / storage.PROJECT_FILE
    empty = project_file.read_bytes()
    api.ops.set_lanes(session, [api.ops.LaneInput("vehicle")])
    project_file.write_bytes(empty)  # changed outside Proteia
    project = session.project
    monkeypatch.setattr(api, "REOPEN_WAIT_S", 0.2)

    with workspace.using(1):
        assert workspace.open("A µ") is session  # answered as it is
        assert session.project is project and workspace.view(session)[0] == 1
        with workspace.using(1) as again:  # nothing waits once the reopen has ended
            assert again is session
    assert workspace.open("A µ") is session  # nothing in use now: read again
    assert workspace.view(session)[0] == 2 and not session.project.batch.lanes


# The routes on the open project that check the opening their request names as
# early as every other (before its path, query and body are), but take the
# session in use only later, or only for a while, themselves, and why. Each has
# a test of its own that it holds the session while it uses it.
USES_ITS_SESSION_LATER = {
    ("POST", "/api/images"): "reads its body first, which may take minutes: no reopen waits for it",
    ("POST", "/api/diagnostics"): (
        "writes its file after planning it, which with images may take minutes: no reopen"
        " waits for it"
    ),
}
# The routes on the open project that check the opening their request names as
# early as every other, but never use its session: they close it, and open
# another. Each checks it again once no other switch can run.
REPLACES_ITS_SESSION = {
    ("POST", "/api/handoffs/{handoff_id}/accept"): "imports images into a new project it opens",
}
# The routes that read the open project if one is open, and work with none (a
# diagnostic file with no project, #138): they get the session, or None, from
# _any_session, which checks the opening and keeps the session in use as
# _open_session does; or, when they also use it later
# (USES_ITS_SESSION_LATER), from _checked_any_session, which checks it and no
# more, as _checked_session does.
WORKS_WITH_NO_PROJECT = {
    ("GET", "/api/diagnostics"): "lists the files of a diagnostic file",
    ("POST", "/api/diagnostics"): "writes a diagnostic file",
}


def test_every_route_on_the_open_project_holds_its_session_until_it_returns():
    # The routes as test_every_route_on_the_open_project_refuses_another_opening
    # finds them: each gets its session from _open_session, which keeps it in
    # use (Workspace.using) until the route returns, so no reopen reads the
    # project again while one runs. Those in USES_ITS_SESSION_LATER get it from
    # _checked_session (or _checked_any_session), which checks it and no more,
    # and take it in use themselves
    # (test_an_upload_holds_its_session_while_it_imports_and_answers,
    # test_a_reopen_does_not_wait_for_a_file_being_written).
    workspace = api.Workspace(Path("unused"), reveal=lambda folder: None)
    app = server.create_app(
        token="t" * 43, port=8000, on_quit=lambda: None, workspace=workspace
    ).app

    def given(dependant: Any, call: Callable[..., Any]) -> Iterator[Any]:
        for sub in dependant.dependencies:
            if sub.call is call:
                yield sub
            yield from given(sub, call)

    guarded = api_routes() - set(NEEDS_NO_OPENING)
    special = set(USES_ITS_SESSION_LATER) | set(REPLACES_ITS_SESSION) | set(WORKS_WITH_NO_PROJECT)
    assert special <= guarded
    found = {
        (method, route.path): (
            [sub.scope for sub in given(route.dependant, api._open_session)],
            [sub.scope for sub in given(route.dependant, api._checked_session)],
            [sub.scope for sub in given(route.dependant, api._no_other_opening)],
            [sub.scope for sub in given(route.dependant, api._any_session)],
            [sub.scope for sub in given(route.dependant, api._checked_any_session)],
        )
        for route in _declared(app.routes)
        for method in route.methods
        if (method, route.path) in guarded
    }

    def expected(route: tuple[str, str]) -> tuple[list, list, list, list, list]:
        later = route in USES_ITS_SESSION_LATER
        if route in WORKS_WITH_NO_PROJECT:
            return [], [], [], ([] if later else ["function"]), ([None] if later else [])
        if later:
            return [], [None], [], [], []
        if route in REPLACES_ITS_SESSION:
            return [], [], [None], [], []
        return ["function"], [], [], [], []

    assert found == {route: expected(route) for route in guarded}


# --- An upload holds its session only once its body is stored (#134's review) ---

UPLOAD = "/api/images?name=a.tif&kind=chemiluminescence&polarity=dark_on_light"


def _spools(monkeypatch) -> threading.Event:
    """Set once an upload's route makes the temporary file its body goes to:
    the request is past its check, and its body is read next."""
    made = threading.Event()
    temporary_file = api.tempfile.TemporaryFile

    def spool(*args: Any, **kwargs: Any) -> Any:
        made.set()
        return temporary_file(*args, **kwargs)

    monkeypatch.setattr(api.tempfile, "TemporaryFile", spool)
    return made


def _upload_begun(
    client: Client, size: int, headers: dict[str, str] | None = None
) -> http.client.HTTPConnection:
    """An upload of ``size`` bytes (:data:`UPLOAD`) with its request line and
    headers sent, and none of its body: the caller sends the body, as slowly as
    it likes, reads the answer (:func:`_answered`) and closes the connection."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    conn.putrequest("POST", UPLOAD)
    sent = {
        "Authorization": f"Bearer {client.token}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
        **(headers or {}),
    }
    for name, value in sent.items():
        conn.putheader(name, value)
    conn.endheaders()
    return conn


def _answered(conn: http.client.HTTPConnection) -> tuple[int, Any]:
    response = conn.getresponse()
    return response.status, json.loads(response.read())


def _lanes_set(client: Client, name: str) -> tuple[dict, Path, bytes]:
    """The project ``name`` created, then its lanes set. Gives that answer's
    project (opening 1, as a page shows it), the project folder, and its
    project.json from before the lanes were set: an earlier copy, which, once
    restored outside Proteia, a reopen reads again."""
    client.ok("POST", "/api/projects", {"name": name})
    folder = client.root / name
    earlier = (folder / storage.PROJECT_FILE).read_bytes()
    shown = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})["project"]
    return shown, folder, earlier


def test_a_reopen_does_not_wait_for_an_upload_whose_body_is_on_its_way(
    client, tmp_path, monkeypatch
):
    # A large file over a slow link takes seconds or minutes to arrive. Were
    # the session in use meanwhile, a reopen from another page would hold up
    # every request on the project for REOPEN_WAIT_S, then give up and leave
    # the changed file unread, with nothing to say so. The upload holds nothing
    # until its body is stored: the reopen reads the file at once, and an
    # upload naming no opening is then imported into the project read again.
    shown, folder, earlier = _lanes_set(client, "Blot")
    (folder / storage.PROJECT_FILE).write_bytes(earlier)  # restored outside Proteia
    data = blot_bytes(tmp_path)
    spooled = _spools(monkeypatch)
    conn = _upload_begun(client, len(data))
    try:
        conn.send(data[:1000])
        assert spooled.wait(20)
        started = time.monotonic()
        reopened = client.ok("POST", "/api/projects/open", {"name": "Blot"})["project"]
        took = time.monotonic() - started
        conn.send(data[1000:])
        status, imported = _answered(conn)
    finally:
        conn.close()
    assert (reopened["open_id"], reopened["lanes"]) == (shown["open_id"] + 1, [])
    assert took < api.REOPEN_WAIT_S
    assert status == 201, imported
    project = imported["project"]
    assert (project["open_id"], project["lanes"]) == (reopened["open_id"], [])
    assert [image["id"] for image in project["images"]] == [imported["image_id"]]


@pytest.mark.parametrize(("meanwhile", "now"), [("reopen", "Blot"), ("create", "Other")])
def test_an_upload_whose_opening_ended_on_its_way_is_refused_and_leaves_nothing(
    client, tmp_path, monkeypatch, meanwhile, now
):
    # The page named opening 1. While its file was on its way, another page
    # opened the project again, which was read again (opening 2: perhaps with
    # other images and lanes), or created another project. The import is not
    # made in a project the page does not show: it is refused once the body is
    # stored, and neither the image nor its file is left, nor the temporary
    # file the body went to.
    shown, folder, earlier = _lanes_set(client, "Blot")
    (folder / storage.PROJECT_FILE).write_bytes(earlier)  # restored outside Proteia
    before = files_of(folder)
    data = blot_bytes(tmp_path)
    spooled = _spools(monkeypatch)
    conn = _upload_begun(client, len(data), {OPENING: str(shown["open_id"])})
    try:
        conn.send(data[:1000])
        assert spooled.wait(20)
        if meanwhile == "reopen":
            client.ok("POST", "/api/projects/open", {"name": "Blot"})
        else:
            client.ok("POST", "/api/projects", {"name": "Other"})
        conn.send(data[1000:])
        status, answer = _answered(conn)
    finally:
        conn.close()
    open_id = shown["open_id"] + 1
    assert status == 409, answer
    assert (answer["code"], answer["detail"]) == (
        "project_changed",
        {"open": now, "open_id": open_id},
    )
    assert files_of(folder) == before
    project = client.ok("GET", "/api/project")["project"]
    assert (project["name"], project["open_id"], project["images"]) == (now, open_id, [])


def test_an_upload_naming_another_opening_is_refused_before_its_body_is_read(
    client, tmp_path, monkeypatch
):
    # Refused on its headers alone: none of a file of perhaps hundreds of
    # megabytes is read, or stored even for a while, for a page that shows a
    # project no longer open. The body is never sent: were it waited for, no
    # answer would come.
    client.ok("POST", "/api/projects", {"name": "Blot"})
    client.ok("POST", "/api/projects", {"name": "Other"})  # opening 2
    before = files_of(client.root)
    spooled = _spools(monkeypatch)
    conn = _upload_begun(client, len(blot_bytes(tmp_path)), {OPENING: "1"})
    try:
        status, answer = _answered(conn)
    finally:
        conn.close()
    assert (status, answer["code"], answer["detail"]) == (
        409,
        "project_changed",
        {"open": "Other", "open_id": 2},
    )
    assert not spooled.is_set()
    assert files_of(client.root) == before


@pytest.mark.parametrize("held", ["import", "answer"])
def test_an_upload_holds_its_session_while_it_imports_and_answers(
    client, tmp_path, monkeypatch, held
):
    # Once its body is stored the upload takes the session in use, as every
    # other route on the open project does from its start
    # (USES_ITS_SESSION_LATER): its import is made, and answered, within the
    # opening it named. project.json changes outside Proteia while it runs; a
    # reopen meanwhile waits for it, then finds the import saved over the
    # change (held in the import) or reads the change (held in the answer).
    shown, folder, earlier = _lanes_set(client, "Blot")
    session = client.workspace.current()
    owner, name = {"import": (api.ops, "import_image"), "answer": (api, "_answer")}[held]
    entered, release = _held_until_released(monkeypatch, owner, name)
    asked = _reads_recorded(monkeypatch, session)
    data = blot_bytes(tmp_path)
    uploading, uploaded = _in_thread(
        lambda: client.call("POST", UPLOAD, raw=data, headers={OPENING: str(shown["open_id"])})
    )
    assert entered.wait(20)
    (folder / storage.PROJECT_FILE).write_bytes(earlier)  # restored outside Proteia
    reopening, reopened = _in_thread(
        lambda: client.call("POST", "/api/projects/open", {"name": "Blot"})
    )
    reopening.join(1.0)  # time to read the file, were it not waiting for the upload
    read_meanwhile = asked.is_set()
    release.set()
    for thread in (uploading, reopening):
        thread.join(30)
        assert not thread.is_alive()

    (status, answer), (reopen_status, reopen) = uploaded[0], reopened[0]
    assert status == 201, answer
    project = answer["project"]
    assert (project["open_id"], project["lanes"]) == (shown["open_id"], shown["lanes"])
    assert [image["id"] for image in project["images"]] == [answer["image_id"]]
    assert not read_meanwhile
    assert reopen_status == 200, reopen
    after = reopen["project"]
    if held == "import":  # saved over the change: nothing to read
        assert (after["open_id"], after["revision"]) == (shown["open_id"], project["revision"])
    else:
        assert (after["open_id"], after["lanes"], after["images"]) == (shown["open_id"] + 1, [], [])


# --- Images handed to the app (#57, N3) ---


class Ticks:
    """A monotonic clock for the inbox that moves only when told."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def ticks(client, monkeypatch) -> Ticks:
    """The inbox's clock, stopped: hand-offs stay young until it is advanced."""
    clock = Ticks()
    monkeypatch.setattr(client.workspace.inbox, "_clock", clock)
    return clock


def incoming(name: str) -> str:
    return f"/api/incoming?name={quote(name, safe='')}"


def staged(client: Client, data: bytes, name: str = "blot.tif") -> dict:
    """One file staged as a launch hands it to the app: the answer."""
    status, answer = client.call("POST", incoming(name), raw=data)
    assert status == 201, answer
    return answer


def listed(client: Client, handoff_id: str) -> dict:
    """The hand-off as GET /api/workspace lists it."""
    (entry,) = [h for h in client.ok("GET", "/api/workspace")["handoffs"] if h["id"] == handoff_id]
    return entry


def handed_off(
    client: Client, tmp_path: Path, names: list[str], *, refused: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """A synthetic blot staged under each name, offered at once: the hand-off's
    id and its files as listed."""
    ids = [staged(client, blot_bytes(tmp_path), name)["file_id"] for name in names]
    offered = client.ok("POST", "/api/handoffs", {"files": ids, "refused": refused or []})
    return offered["handoff_id"], listed(client, offered["handoff_id"])["files"]


def choices(
    files: list[dict],
    *,
    kinds: dict[int, str] | None = None,
    membranes: dict[int, int] | None = None,
    polarity: str = "dark_on_light",
) -> list[dict]:
    """An accept's files: chemiluminescence unless ``kinds`` says, each on a new
    membrane unless ``membranes`` names an earlier file's."""
    return [
        {
            "file_id": file["file_id"],
            "kind": (kinds or {}).get(index, "chemiluminescence"),
            "polarity": polarity,
            "membrane": (membranes or {}).get(index, "new"),
        }
        for index, file in enumerate(files)
    ]


def accept(
    client: Client,
    handoff_id: str,
    files: list[dict],
    *,
    name: str | None = None,
    headers: dict[str, str] | None = None,
    **chosen: Any,
) -> dict:
    body = {"name": name, "files": choices(files, **chosen)}
    return client.ok("POST", f"/api/handoffs/{handoff_id}/accept", body, headers=headers)


def staging(client: Client) -> list[str]:
    """The names of the files in the staging folder."""
    folder = client.workspace.inbox.folder
    assert folder is not None
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def in_root(client: Client) -> list[str]:
    return sorted(path.name for path in client.root.iterdir()) if client.root.is_dir() else []


def test_an_incoming_file_is_staged_under_a_name_the_server_makes(client, tmp_path):
    data = blot_bytes(tmp_path)
    answer = staged(client, data, NAME)
    assert (answer["name"], answer["size"]) == (NAME, len(data))
    folder = client.workspace.inbox.folder
    assert folder == tmp_path / "state" / "incoming"
    (stored,) = folder.iterdir()
    assert re.fullmatch(r"[0-9a-f]{32}", stored.name) and stored.read_bytes() == data
    assert answer["file_id"] not in stored.name
    # Nothing in the projects root, and nothing listed until it is offered.
    assert not client.root.exists()
    assert client.ok("GET", "/api/workspace")["handoffs"] == []


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("notes.txt", b"x", 422, "unsupported_image_type"),
        ("blot.bmp", b"x", 422, "unsupported_image_type"),
        ("C:\\x\\a.tif", b"x", 422, "invalid_image"),
        ("../a.tif", b"x", 422, "invalid_image"),
        ("a/b.tif", b"x", 422, "invalid_image"),
        ("x" * 252 + ".tif", b"x", 422, "invalid_image"),  # 256 characters
        ("blot.tif", b"", 422, "invalid_image"),
    ],
    ids=["txt", "bmp", "windows path", "parent", "folder", "256 characters", "empty"],
)
def test_an_incoming_file_that_cannot_be_an_image_is_refused_and_leaves_nothing(
    client, name, data, status, code
):
    assert client.refused("POST", incoming(name), raw=data)[:2] == (status, code)
    assert staging(client) == []
    assert not client.root.exists()


def _stage_chunked(client: Client, chunks: list[bytes], name: str = "a.tif") -> tuple[int, Any]:
    """A file staged in chunks, with no size declared (Transfer-Encoding: chunked)."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    try:
        conn.request(
            "POST",
            incoming(name),
            body=iter(chunks),
            headers={
                "Authorization": f"Bearer {client.token}",
                "Content-Type": "application/octet-stream",
            },
            encode_chunked=True,
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_an_incoming_file_over_the_size_limit_is_refused(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    assert client.refused("POST", incoming("a.tif"), raw=b"x" * 101)[:2] == (
        413,
        "image_too_large",
    )
    status, answer = _stage_chunked(client, [b"x" * 60, b"x" * 60])  # no size declared
    assert (status, answer["code"]) == (413, "image_too_large")
    assert staging(client) == []
    assert staged(client, b"x" * 100)["size"] == 100


def test_an_incoming_file_that_cannot_be_stored_is_refused_naming_no_path(client, tmp_path):
    # The launch passes the message on to the page, which sees no path: the
    # staging folder's is in the per-user state folder.
    folder = client.workspace.inbox.folder
    assert folder is not None
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_bytes(b"")  # the staging folder cannot be made
    status, payload = client.call("POST", incoming("a µ.tif"), raw=b"x" * 10)
    assert (status, payload["code"]) == (500, "file_error")
    assert payload["message"].startswith("'a µ.tif' could not be stored: ")
    for path in (str(tmp_path), str(folder), folder.name):
        assert path not in payload["message"]
    folder.unlink()
    assert staged(client, b"x" * 10)["size"] == 10


def test_uploads_beyond_the_pending_limits_are_refused(client, tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_PENDING_FILES", 2)
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    staged(client, b"x" * 10, "b.tif")  # not offered: waiting too
    status, code, _ = client.refused("POST", incoming("c.tif"), raw=b"x" * 10)
    assert (status, code) == (409, "too_many_pending")
    assert len(staging(client)) == 2
    body = {"files": [file["file_id"] for file in files]}
    assert client.call("POST", f"/api/handoffs/{handoff_id}/discard", body)[0] == 204
    staged(client, b"x" * 10, "c.tif")  # room again

    # The bytes staged: those waiting, and those an upload declares or has sent.
    monkeypatch.setattr(handoff, "MAX_PENDING_FILES", 32)
    monkeypatch.setattr(handoff, "MAX_STAGED_BYTES", 50)
    status, code, _ = client.refused("POST", incoming("d.tif"), raw=b"x" * 31)
    assert (status, code) == (409, "too_many_pending")
    monkeypatch.setattr(api, "_WRITE_BYTES", 10)  # written, and counted, 10 bytes at a time
    status, answer = _stage_chunked(client, [b"x" * 20, b"x" * 20])
    assert (status, answer["code"]) == (409, "too_many_pending")
    assert len(staging(client)) == 2
    assert staged(client, b"x" * 30, "e.tif")["size"] == 30


def test_nothing_is_staged_offered_or_imported_once_proteia_is_stopping(client, tmp_path):
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    waiting = staged(client, blot_bytes(tmp_path), "b.tif")
    client.workspace.inbox.stop()
    assert client.refused("POST", incoming("c.tif"), raw=b"x")[:2] == (409, "stopping")
    body: dict[str, Any] = {"files": [waiting["file_id"]]}
    assert client.refused("POST", "/api/handoffs", body)[:2] == (409, "stopping")
    body = {"files": choices(files)}
    assert client.refused("POST", f"/api/handoffs/{handoff_id}/accept", body)[:2] == (
        409,
        "stopping",
    )
    assert not client.root.exists()


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"files": ["0" * 16]}, 404, "unknown_id"),
        ({"files": ["A", "A"]}, 422, "invalid_input"),
        ({}, 422, "invalid_input"),
        ({"files": ["A"], "more": 1}, 422, "invalid_input"),
        ({"files": ["A"], "refused": [{"name": 1, "code": "x", "message": "y"}]}, 422, None),
        ({"files": ["A"], "refused": [{"name": "a.bmp", "code": "x"}]}, 422, None),
        ({"files": "A"}, 422, "invalid_input"),
    ],
    ids=[
        "unknown id",
        "an id twice",
        "nothing",
        "unknown key",
        "a name not text",
        "no message",
        "files not a list",
    ],
)
def test_an_offer_that_cannot_be_taken_hands_off_nothing(client, tmp_path, body, status, code):
    staged_id = staged(client, blot_bytes(tmp_path), "a.tif")["file_id"]
    body = json.loads(json.dumps(body).replace('"A"', json.dumps(staged_id)))
    assert client.refused("POST", "/api/handoffs", body)[:2] == (status, code or "invalid_input")
    assert client.ok("GET", "/api/workspace")["handoffs"] == []
    assert client.ok("POST", "/api/handoffs", {"files": [staged_id]})["files"] == 1  # still waiting


def test_an_offer_hands_off_uploaded_files_once(client, tmp_path, ticks):
    a = staged(client, blot_bytes(tmp_path), "a.tif")
    b = staged(client, b"x" * 7, "b µ.png")
    offered = client.ok("POST", "/api/handoffs", {"files": [a["file_id"], b["file_id"]]})
    assert offered == {
        "handoff_id": offered["handoff_id"],
        "merged": False,
        "files": 2,
        "refused": 0,
    }
    body = {"files": [a["file_id"]]}
    assert client.refused("POST", "/api/handoffs", body)[:2] == (409, "file_claimed")
    assert client.ok("GET", "/api/workspace")["handoffs"] == [
        {
            "id": offered["handoff_id"],
            "kind": "images",
            "files": [
                {"file_id": a["file_id"], "name": "a.tif", "size": a["size"]},
                {"file_id": b["file_id"], "name": "b µ.png", "size": 7},
            ],
            "suggested_name": "a",
            "refused": [],
            "more_refused": 0,
            "more_may_arrive": True,
        }
    ]


def test_refused_arguments_are_kept_bounded_and_never_refuse_the_offer(client, tmp_path):
    file_id = staged(client, blot_bytes(tmp_path), "a.tif")["file_id"]
    refused = [
        {"name": "β" * 300 + ".tif", "code": "missing", "message": "no such file or folder"},
        {"name": "bad\x07name\u202e.tif", "code": "made_up", "message": "m" * 500},
        {"name": "half\ud800.tif", "code": "unreadable", "message": "cannot be read: \udfff"},
    ]
    offered = client.ok("POST", "/api/handoffs", {"files": [file_id], "refused": refused})
    assert (offered["files"], offered["refused"]) == (1, 3)
    entry = listed(client, offered["handoff_id"])
    assert [file["file_id"] for file in entry["files"]] == [file_id]
    first, second, third = entry["refused"]
    assert first == {
        "name": "β" * 119 + "…",
        "code": "missing",
        "message": "no such file or folder",
    }
    assert second == {
        "name": "bad\ufffdname\ufffd.tif",
        "code": "other",
        "message": "m" * 199 + "…",
    }
    assert third == {
        "name": "half\ufffd.tif",
        "code": "unreadable",
        "message": "cannot be read: \ufffd",
    }


def test_only_the_first_hundred_refused_entries_are_kept(client):
    refused = [
        {"name": f"{i}.bmp", "code": "unsupported_type", "message": "not an image type"}
        for i in range(150)
    ]
    offered = client.ok("POST", "/api/handoffs", {"refused": refused})
    assert (offered["merged"], offered["files"], offered["refused"]) == (False, 0, 150)
    entry = listed(client, offered["handoff_id"])
    assert (entry["kind"], entry["files"], entry["suggested_name"]) == ("notice", [], None)
    assert [r["name"] for r in entry["refused"]] == [f"{i}.bmp" for i in range(100)]
    assert entry["more_refused"] == 50


def test_offers_whose_uploads_began_within_the_window_join_one_hand_off(
    client, tmp_path, ticks, monkeypatch
):
    def offer(name: str) -> dict:
        file_id = staged(client, blot_bytes(tmp_path), name)["file_id"]
        return client.ok("POST", "/api/handoffs", {"files": [file_id]})

    first = offer("a.tif")
    ticks.advance(handoff.MERGE_WINDOW_S - 1)
    second = offer("b.tif")
    assert (second["handoff_id"], second["merged"], second["files"]) == (
        first["handoff_id"],
        True,
        2,
    )
    ticks.advance(handoff.MERGE_WINDOW_S)  # from b.tif's start
    third = offer("c.tif")
    assert third["handoff_id"] != first["handoff_id"] and not third["merged"]
    monkeypatch.setattr(handoff, "MAX_HANDOFF_FILES", 2)
    fourth = offer("d.tif")
    assert fourth["handoff_id"] == third["handoff_id"]
    fifth = offer("e.tif")  # within the window, but the hand-off is full
    assert fifth["handoff_id"] not in (first["handoff_id"], third["handoff_id"])
    handoffs = client.ok("GET", "/api/workspace")["handoffs"]
    assert [[file["name"] for file in h["files"]] for h in handoffs] == [
        ["a.tif", "b.tif"],
        ["c.tif", "d.tif"],
        ["e.tif"],
    ]


def _staging_begun(client: Client, name: str, size: int) -> http.client.HTTPConnection:
    """A file staged with its request line and headers sent, and none of its
    body (as :func:`_upload_begun`)."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    conn.putrequest("POST", incoming(name))
    for header, value in {
        "Authorization": f"Bearer {client.token}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
    }.items():
        conn.putheader(header, value)
    conn.endheaders()
    return conn


def test_a_slow_upload_joins_the_hand_off_its_start_was_near(client, tmp_path, ticks, monkeypatch):
    # An Explorer selection of a large file and a small one: the small one is
    # offered first; the large one, begun as early, arrives a minute later and
    # still joins it, so the selection gives one project. Meanwhile the
    # hand-off says more may arrive.
    inbox = client.workspace.inbox
    began = threading.Event()
    begin = inbox.begin_upload

    def beginning(*args: Any, **kwargs: Any) -> Any:
        upload = begin(*args, **kwargs)
        began.set()
        return upload

    monkeypatch.setattr(inbox, "begin_upload", beginning)
    data = blot_bytes(tmp_path)
    conn = _staging_begun(client, "slow.tif", len(data))
    try:
        conn.send(data[:1000])
        assert began.wait(20)
        ticks.advance(2)
        quick = staged(client, data, "quick.tif")["file_id"]
        offered = client.ok("POST", "/api/handoffs", {"files": [quick]})
        ticks.advance(60)  # the slow file takes a minute more
        assert listed(client, offered["handoff_id"])["more_may_arrive"]
        conn.send(data[1000:])
        status, slow = _answered(conn)
    finally:
        conn.close()
    assert status == 201, slow
    assert listed(client, offered["handoff_id"])["more_may_arrive"]  # staged, not offered
    joined = client.ok("POST", "/api/handoffs", {"files": [slow["file_id"]]})
    assert (joined["handoff_id"], joined["merged"]) == (offered["handoff_id"], True)
    entry = listed(client, offered["handoff_id"])
    assert [file["name"] for file in entry["files"]] == ["quick.tif", "slow.tif"]
    assert not entry["more_may_arrive"]


def test_the_workspace_lists_each_hand_off_with_the_name_its_project_would_take(
    client, tmp_path, ticks, monkeypatch
):
    client.ok("POST", "/api/projects", {"name": "blot"})
    first, _ = handed_off(client, tmp_path, ["Blot.TIF"])
    ticks.advance(handoff.MERGE_WINDOW_S)
    second, _ = handed_off(client, tmp_path, [NAME, "marker α.tif"])
    listings: list[Path] = []
    iterdir = Path.iterdir

    def counting(self: Path) -> Any:
        if self == client.root:
            listings.append(self)
        return iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting)
    handoffs = client.ok("GET", "/api/workspace")["handoffs"]
    assert [(h["id"], h["suggested_name"]) for h in handoffs] == [
        (first, "Blot (2)"),  # blot is taken, ignoring case
        (second, "β-actin 10 µM"),
    ]
    assert len(listings) == 1  # the root, once for both
    assert [h["more_may_arrive"] for h in handoffs] == [False, True]  # the first is older
    ticks.advance(handoff.MERGE_WINDOW_S)
    assert [h["more_may_arrive"] for h in client.ok("GET", "/api/workspace")["handoffs"]] == [
        False,
        False,
    ]

    # A hand-off an accept has claimed is not listed.
    inbox = client.workspace.inbox
    claimed = inbox.claim(first, [file["file_id"] for file in listed(client, first)["files"]])
    assert [h["id"] for h in client.ok("GET", "/api/workspace")["handoffs"]] == [second]
    inbox.release(claimed)
    assert [h["id"] for h in client.ok("GET", "/api/workspace")["handoffs"]] == [first, second]


def test_an_accept_imports_the_images_into_a_new_project_named_after_the_first(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    before = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})["project"]
    old = client.workspace.current()
    handoff_id, files = handed_off(client, tmp_path, [NAME, "marker α.tif"])
    answer = accept(client, handoff_id, files, kinds={1: "visible_marker"}, membranes={1: 0})

    project = answer["project"]
    assert project["name"] == "β-actin 10 µM" and project["open_id"] == before["open_id"] + 1
    images = project["images"]
    assert [(i["original_name"], i["kind"], i["polarity"]) for i in images] == [
        (NAME, "chemiluminescence", "dark_on_light"),
        ("marker α.tif", "visible_marker", "dark_on_light"),
    ]
    membrane = images[0]["membrane_id"]
    assert images[1]["membrane_id"] == membrane
    assert answer["handoff"] == {
        "imported": [
            {
                "file_id": files[0]["file_id"],
                "name": NAME,
                "image_id": images[0]["id"],
                "membrane_id": membrane,
                "new_membrane": True,
            },
            {
                "file_id": files[1]["file_id"],
                "name": "marker α.tif",
                "image_id": images[1]["id"],
                "membrane_id": membrane,
                "new_membrane": False,
            },
        ],
        "refused": [],
        "launch_refused": [],
        "more_refused": 0,
        "notes": [],
    }
    saved = storage.load_project(client.root / "β-actin 10 µM")
    assert [entry.action for entry in saved.log] == ["new_project", "import_image", "import_image"]
    assert staging(client) == []
    workspace = client.ok("GET", "/api/workspace")
    assert (workspace["open"], workspace["handoffs"]) == ("β-actin 10 µM", [])
    # The project open before is saved and closed.
    assert (old.undo_step, old.redo_step) == (None, None)
    assert [lane.label for lane in storage.load_project(old.folder).batch.lanes] == ["vehicle"]


def test_an_accept_says_what_the_launches_refused(client, tmp_path, ticks):
    refused = [{"name": "photo.bmp", "code": "unsupported_type", "message": "not an image type"}]
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"], refused=refused)
    late = [{"name": "gone.tif", "code": "missing", "message": "no such file or folder"}]
    assert client.ok("POST", "/api/handoffs", {"refused": late})["merged"]  # a notice, merged
    handed = accept(client, handoff_id, files)["handoff"]
    assert (handed["launch_refused"], handed["more_refused"]) == (refused + late, 0)


def test_an_accept_takes_the_name_typed_or_refuses_it_changing_nothing(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Taken"})
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    path = f"/api/handoffs/{handoff_id}/accept"
    for name, refused in (
        ("taken", (409, "project_exists")),
        ("a/b", (422, "invalid_project_name")),
    ):
        assert client.refused("POST", path, {"name": name, "files": choices(files)})[:2] == refused
        assert in_root(client) == ["Taken"] and listed(client, handoff_id)["files"] == files
    assert accept(client, handoff_id, files, name="  My  blot ")["project"]["name"] == "My blot"


def _missing_polarity(body: dict) -> None:
    del body["files"][1]["polarity"]


@pytest.mark.parametrize(
    "change",
    [
        _missing_polarity,
        lambda body: body["files"][0].update(kind="photo"),
        lambda body: body["files"][1].update(polarity="dark"),
        lambda body: body["files"][1].update(polarity=None),
        lambda body: body["files"][0].update(membrane=0),  # itself
        lambda body: body["files"][0].update(membrane=1),  # a later file
        lambda body: body["files"][1].update(membrane=-1),
        lambda body: body["files"][1].update(membrane="same"),
        lambda body: body["files"][1].update(membrane=True),
        lambda body: body.update(files="all"),
        lambda body: body.update(extra=1),
    ],
    ids=[
        "no polarity",
        "unknown kind",
        "unknown polarity",
        "null polarity",
        "membrane of itself",
        "membrane of a later file",
        "negative membrane",
        "membrane text",
        "membrane true",
        "files not a list",
        "unknown key",
    ],
)
def test_an_accept_that_cannot_be_read_changes_nothing(client, tmp_path, change):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    handoff_id, files = handed_off(client, tmp_path, ["a.tif", "b.tif"])
    body: dict[str, Any] = {"name": None, "files": choices(files)}
    change(body)
    path = f"/api/handoffs/{handoff_id}/accept"
    assert client.refused("POST", path, body)[:2] == (422, "invalid_input")
    assert in_root(client) == ["Blot"]
    assert client.ok("GET", "/api/workspace")["open"] == "Blot"
    assert listed(client, handoff_id)["files"] == files
    assert len(staging(client)) == 2


def test_an_accept_of_other_files_than_those_waiting_is_refused_with_them(client, tmp_path, ticks):
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    late = staged(client, blot_bytes(tmp_path), "b.tif")["file_id"]
    assert client.ok("POST", "/api/handoffs", {"files": [late]})["merged"]
    path = f"/api/handoffs/{handoff_id}/accept"
    status, payload = client.call("POST", path, {"files": choices(files)})
    assert (status, payload["code"]) == (409, "handoff_changed")
    assert payload["detail"] == listed(client, handoff_id)
    now = payload["detail"]["files"]
    assert [file["name"] for file in now] == ["a.tif", "b.tif"]
    assert not client.root.exists()
    assert client.refused("POST", "/api/handoffs/nothing/accept", {"files": []})[:2] == (
        404,
        "handoff_not_found",
    )
    project = accept(client, handoff_id, now)["project"]
    assert [image["original_name"] for image in project["images"]] == ["a.tif", "b.tif"]


def test_a_notice_has_nothing_to_accept(client):
    refused = [{"name": "photo.bmp", "code": "unsupported_type", "message": "not an image type"}]
    handoff_id = client.ok("POST", "/api/handoffs", {"refused": refused})["handoff_id"]
    path = f"/api/handoffs/{handoff_id}/accept"
    assert client.refused("POST", path, {"files": []})[:2] == (422, "invalid_input")
    assert client.call("POST", f"/api/handoffs/{handoff_id}/discard", {"files": [], "refused": 1})[
        0
    ] == (204)
    assert client.ok("GET", "/api/workspace")["handoffs"] == []


def test_a_file_that_cannot_be_imported_is_left_out_and_said(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    ids = [
        staged(client, b"not an image" * 100, "bad.tif")["file_id"],
        staged(client, blot_bytes(tmp_path), "blot.tif")["file_id"],
        staged(client, blot_bytes(tmp_path), "marker α.tif")["file_id"],
    ]
    offered = client.ok("POST", "/api/handoffs", {"files": ids})
    files = listed(client, offered["handoff_id"])["files"]
    # blot.tif to join bad.tif's membrane, the marker to join blot.tif's.
    answer = accept(client, offered["handoff_id"], files, membranes={1: 0, 2: 1})
    handed = answer["handoff"]
    assert [(f["name"], f["new_membrane"]) for f in handed["imported"]] == [
        ("blot.tif", True),
        ("marker α.tif", False),
    ]
    assert handed["imported"][0]["membrane_id"] == handed["imported"][1]["membrane_id"]
    (refused,) = handed["refused"]
    assert (refused["file_id"], refused["name"], refused["code"]) == (
        ids[0],
        "bad.tif",
        "unreadable_image",
    )
    assert handed["notes"] == [
        "blot.tif was put on a new membrane because bad.tif was not imported"
    ]
    # Named after the first file, though it was left out: the name is chosen first.
    assert answer["project"]["name"] == "bad"
    assert [image["original_name"] for image in answer["project"]["images"]] == [
        "blot.tif",
        "marker α.tif",
    ]
    assert staging(client) == []


def test_an_accept_with_nothing_importable_leaves_no_project(client):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    ids = [staged(client, b"not an image", name)["file_id"] for name in ("a.tif", "b.png")]
    offered = client.ok("POST", "/api/handoffs", {"files": ids})
    files = listed(client, offered["handoff_id"])["files"]
    path = f"/api/handoffs/{offered['handoff_id']}/accept"
    status, payload = client.call("POST", path, {"files": choices(files)})
    assert (status, payload["code"]) == (422, "nothing_imported")
    assert [(r["file_id"], r["name"], r["code"]) for r in payload["detail"]["refused"]] == [
        (ids[0], "a.tif", "unreadable_image"),
        (ids[1], "b.png", "unreadable_image"),
    ]
    assert in_root(client) == ["Blot"]
    workspace = client.ok("GET", "/api/workspace")
    # Discarded: an accept again would fail again.
    assert (workspace["open"], workspace["handoffs"]) == ("Blot", [])
    assert staging(client) == []


def test_an_accept_while_the_open_project_cannot_be_saved_changes_nothing(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    project_file = client.root / "Blot" / storage.PROJECT_FILE
    project_file.unlink()
    project_file.mkdir()  # the autosave cannot replace it
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    path = f"/api/handoffs/{handoff_id}/accept"
    status, payload = client.call("POST", path, {"files": choices(files)})
    assert (status, payload["code"]) == (409, "unsaved_changes") and "detail" not in payload
    assert in_root(client) == ["Blot"]
    assert listed(client, handoff_id)["files"] == files and len(staging(client)) == 1

    project_file.rmdir()
    assert accept(client, handoff_id, files)["project"]["name"] == "a"
    saved = storage.load_project(client.root / "Blot")
    assert [lane.label for lane in saved.batch.lanes] == ["vehicle"]


@pytest.mark.parametrize("switch", ["create", "accept"])
def test_an_edit_made_while_a_switch_runs_is_saved_before_the_old_project_closes(
    client, tmp_path, monkeypatch, switch
):
    # While a switch creates its project (for an accept: and imports into it)
    # the old project is still open, and another page may edit it. If that
    # edit's autosave fails, the old project is saved again before it is
    # closed; if that fails too, the switch is abandoned: the old project stays
    # open with the edit, and the answer names the project created.
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    old = client.workspace.current()
    project_file = old.folder / storage.PROJECT_FILE
    edits: list[str] = []
    fixed = threading.Event()  # the file can be written again once the edit is made
    create_project = api.projects.create_project

    def edited_meanwhile(*args: Any, **kwargs: Any) -> Any:
        project_file.unlink()
        project_file.mkdir()  # the edit's autosave fails
        edits.append(f"c{len(edits) + 1}")
        api.ops.set_lanes(old, [api.ops.LaneInput(condition) for condition in edits])
        assert old.dirty
        if fixed.is_set():
            project_file.rmdir()
        return create_project(*args, **kwargs)

    monkeypatch.setattr(api.projects, "create_project", edited_meanwhile)

    def switch_to(name: str) -> tuple[int, Any]:
        if switch == "create":
            return client.call("POST", "/api/projects", {"name": name})
        handoff_id, files = handed_off(client, tmp_path, [f"{name}.tif"])
        return client.call("POST", f"/api/handoffs/{handoff_id}/accept", {"files": choices(files)})

    status, payload = switch_to("First")
    assert (status, payload["code"], payload["detail"]) == (
        409,
        "unsaved_changes",
        {"created": "First"},
    )
    assert "'First'" in payload["message"] and "'Blot'" in payload["message"]
    assert client.workspace.current() is old and old.dirty
    assert [lane.label for lane in old.project.batch.lanes] == ["c1"]
    assert client.ok("GET", "/api/workspace")["open"] == "Blot"
    created = storage.load_project(client.root / "First")  # complete, and closed
    assert len(list(created.batch.iter_images())) == (1 if switch == "accept" else 0)
    assert client.ok("GET", "/api/workspace")["handoffs"] == []  # its images are in First
    assert staging(client) == []

    project_file.rmdir()  # saved again before the switch, then while it runs
    fixed.set()
    status, payload = switch_to("Second")
    assert status == 201, payload
    assert payload["project"]["name"] == "Second"
    saved = storage.load_project(old.folder)
    assert [lane.label for lane in saved.batch.lanes] == ["c1", "c2"]
    assert not old.dirty


def test_an_open_whose_old_project_cannot_be_saved_after_it_names_nothing_created(
    client, monkeypatch
):
    # An open makes no project: if an edit made while it runs cannot be saved,
    # it is refused as when the open project cannot be saved first. Nothing is
    # named as created, and the old project stays open with the edit.
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Other"})
    other = (client.root / "Other" / storage.PROJECT_FILE).read_bytes()
    client.ok("POST", "/api/projects", {"name": "Blot"})
    old = client.workspace.current()
    project_file = old.folder / storage.PROJECT_FILE
    open_project = api.ops.open_project

    def edited_meanwhile(*args: Any, **kwargs: Any) -> Any:
        project_file.unlink()
        project_file.mkdir()  # the edit's autosave fails
        api.ops.set_lanes(old, [api.ops.LaneInput("c1")])
        assert old.dirty
        return open_project(*args, **kwargs)

    monkeypatch.setattr(api.ops, "open_project", edited_meanwhile)
    status, payload = client.call("POST", "/api/projects/open", {"name": "Other"})
    assert (status, payload["code"]) == (409, "unsaved_changes")
    assert "detail" not in payload and "created" not in payload["message"]
    assert payload["message"].startswith("the open project could not be saved: ")
    assert client.workspace.current() is old and old.dirty
    assert client.ok("GET", "/api/workspace")["open"] == "Blot"
    assert (client.root / "Other" / storage.PROJECT_FILE).read_bytes() == other

    monkeypatch.setattr(api.ops, "open_project", open_project)
    project_file.rmdir()
    assert client.ok("POST", "/api/projects/open", {"name": "Other"})["project"]["name"] == "Other"
    assert [lane.label for lane in storage.load_project(old.folder).batch.lanes] == ["c1"]


def test_a_switch_waits_for_an_edit_running_on_the_old_project(tmp_path, monkeypatch):
    # An edit running on the open project when a create has made its project
    # holds that project's lock and has not changed it yet. The switch waits
    # for it before replacing the project, so an edit whose autosave fails is
    # saved again, or stays open, never left only in a closed session. Here the
    # save fails again: the switch is abandoned, and the edit stays open.
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    old = workspace.create("Blot")
    project_file = old.folder / storage.PROJECT_FILE
    project_file.unlink()
    project_file.mkdir()  # every save of Blot fails
    made = threading.Event()
    create_project = api.projects.create_project

    def creating(*args: Any, **kwargs: Any) -> Any:
        try:
            return create_project(*args, **kwargs)
        finally:
            made.set()

    monkeypatch.setattr(api.projects, "create_project", creating)
    holding, go = threading.Event(), threading.Event()

    def editing() -> None:  # a request's operation on Blot, e.g. a slow detection
        with old.lock:
            holding.set()
            assert go.wait(20)
            api.ops.set_lanes(old, [api.ops.LaneInput("vehicle")])  # its autosave fails

    def switching() -> None:
        with pytest.raises(api.UnsavedChangesError) as refused:
            workspace.create("New")
        errors.append(refused.value)

    errors: list[api.UnsavedChangesError] = []
    edit = threading.Thread(target=editing)
    edit.start()
    assert holding.wait(20)
    switch = threading.Thread(target=switching)
    switch.start()
    assert made.wait(20)
    deadline = time.monotonic() + 0.5  # time enough for a switch that does not wait
    while workspace.current() is old and time.monotonic() < deadline:
        time.sleep(0.005)
    assert workspace.current() is old  # waiting for the edit
    go.set()
    for thread in (edit, switch):
        thread.join(20)
        assert not thread.is_alive()

    (error,) = errors
    assert error.created == "New"
    assert workspace.current() is old and old.dirty
    assert [lane.label for lane in old.project.batch.lanes] == ["vehicle"]
    assert storage.load_project(tmp_path / "root" / "New").batch.lanes == []  # created, closed


@pytest.mark.parametrize("saves", [True, False], ids=["saved", "logged"])
def test_an_edit_made_on_the_old_project_once_replaced_is_saved_after_the_close(
    tmp_path, monkeypatch, caplog, saves
):
    # A request that took the open project before a switch replaced it may
    # edit it after, before the close. If that edit's autosave fails, the old
    # project is saved once more after the close; a failure then is logged.
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    old = workspace.create("Blot")
    project_file = old.folder / storage.PROJECT_FILE
    close = old.close

    def edited_then_closed(**kwargs: Any) -> None:
        assert workspace.current() is not old  # replaced
        project_file.unlink()
        project_file.mkdir()  # the edit's autosave fails
        api.ops.set_lanes(old, [api.ops.LaneInput("vehicle")])
        assert old.dirty
        if saves:
            project_file.rmdir()
        close(**kwargs)

    monkeypatch.setattr(old, "close", edited_then_closed)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        new = workspace.create("New")
    assert workspace.current() is new
    logged = [record.getMessage() for record in caplog.records if record.name == api.__name__]
    if saves:
        assert not old.dirty and logged == []
        assert [lane.label for lane in storage.load_project(old.folder).batch.lanes] == ["vehicle"]
    else:
        assert old.dirty
        (message,) = logged
        assert message.startswith("'Blot' was closed with changes that could not be saved: ")


def test_an_accept_from_a_page_showing_another_opening_changes_nothing(client, tmp_path):
    shown = client.ok("POST", "/api/projects", {"name": "Blot"})["project"]["open_id"]
    now = client.ok("POST", "/api/projects", {"name": "Other"})["project"]["open_id"]
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    path = f"/api/handoffs/{handoff_id}/accept"
    status, payload = client.call(
        "POST", path, {"files": choices(files)}, headers={OPENING: str(shown)}
    )
    assert (status, payload["code"], payload["detail"]) == (
        409,
        "project_changed",
        {"open": "Other", "open_id": now},
    )
    assert in_root(client) == ["Blot", "Other"] and listed(client, handoff_id)["files"] == files
    answer = accept(client, handoff_id, files, headers={OPENING: str(now)})
    assert (answer["project"]["name"], answer["project"]["open_id"]) == ("a", now + 1)


def test_files_offered_while_an_accept_runs_start_another_hand_off(
    client, tmp_path, monkeypatch, ticks
):
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"])
    ids = [file["file_id"] for file in files]
    entered, release = _held_until_released(monkeypatch, api.ops, "import_image")
    path = f"/api/handoffs/{handoff_id}"
    accepting, accepted = _in_thread(
        lambda: client.call("POST", f"{path}/accept", {"files": choices(files)})
    )
    try:
        assert entered.wait(20)
        assert client.ok("GET", "/api/workspace")["handoffs"] == []  # claimed: not listed
        late = staged(client, blot_bytes(tmp_path), "late.tif")["file_id"]
        offered = client.ok("POST", "/api/handoffs", {"files": [late]})
        assert not offered["merged"] and offered["handoff_id"] != handoff_id
        for route, body in (("accept", {"files": choices(files)}), ("discard", {"files": ids})):
            assert client.refused("POST", f"{path}/{route}", body)[:2] == (409, "handoff_claimed")
    finally:
        release.set()
        accepting.join(30)
    status, answer = accepted[0]
    assert status == 201, answer
    assert [image["original_name"] for image in answer["project"]["images"]] == ["a.tif"]
    handoffs = client.ok("GET", "/api/workspace")["handoffs"]
    assert [(h["id"], [f["name"] for f in h["files"]]) for h in handoffs] == [
        (offered["handoff_id"], ["late.tif"])
    ]


def test_a_discard_drops_only_what_the_page_shows(client, tmp_path, ticks):
    refused = [{"name": "photo.bmp", "code": "unsupported_type", "message": "not an image type"}]
    handoff_id, files = handed_off(client, tmp_path, ["a.tif"], refused=refused)
    shown = [file["file_id"] for file in files]
    late = staged(client, blot_bytes(tmp_path), "b.tif")["file_id"]
    assert client.ok("POST", "/api/handoffs", {"files": [late]})["merged"]
    path = f"/api/handoffs/{handoff_id}/discard"
    status, payload = client.call("POST", path, {"files": shown, "refused": 1})
    assert (status, payload["code"]) == (409, "handoff_changed")
    assert payload["detail"] == listed(client, handoff_id)
    assert len(staging(client)) == 2
    now = [file["file_id"] for file in payload["detail"]["files"]]
    status, payload = client.call("POST", path, {"files": now, "refused": 0})  # it holds 1
    assert (status, payload["code"]) == (409, "handoff_changed")

    assert client.call("POST", path, {"files": list(reversed(now)), "refused": 1})[0] == 204
    assert staging(client) == []
    assert client.ok("GET", "/api/workspace")["handoffs"] == []
    assert client.refused("POST", path, {"files": now, "refused": 1})[:2] == (
        404,
        "handoff_not_found",
    )
    assert not client.root.exists()


def test_files_named_on_the_servers_command_line_are_read_where_they_are_never_deleted(
    client, tmp_path
):
    # The first launch registers the files on its own command line: read
    # where they are when imported, never copied into the staging folder, and
    # never deleted, whether imported, discarded, or dropped when Proteia stops.
    originals = tmp_path / "originals α"
    blot = synthetic_blot((H, W), [(50, ROW, 5.0, 3.0, 30000.0)])
    gone = write_tiff(originals / "gone β.tif", blot)
    kept = write_tiff(originals / "kept µ.tif", blot)

    def unchanged() -> tuple[bytes, int]:
        return kept.read_bytes(), kept.stat().st_mtime_ns

    before = unchanged()
    inbox = client.workspace.inbox
    offered = inbox.add_local([(gone, gone.name), (kept, kept.name)])
    assert offered is not None and staging(client) == []
    files = listed(client, offered.handoff_id)["files"]
    assert [(f["name"], f["size"]) for f in files] == [
        (gone.name, gone.stat().st_size),
        (kept.name, kept.stat().st_size),
    ]
    gone.unlink()  # removed before Import is pressed
    handed = accept(client, offered.handoff_id, files)["handoff"]
    assert [f["name"] for f in handed["imported"]] == ["kept µ.tif"]
    (refused,) = handed["refused"]
    assert (refused["name"], refused["code"]) == ("gone β.tif", "file_error")
    assert str(originals) not in json.dumps(handed, ensure_ascii=False)  # no path
    assert unchanged() == before

    offered = inbox.add_local([(kept, kept.name)])
    ids = [f["file_id"] for f in listed(client, offered.handoff_id)["files"]]
    path = f"/api/handoffs/{offered.handoff_id}/discard"
    assert client.call("POST", path, {"files": ids})[0] == 204
    assert unchanged() == before
    inbox.add_local([(kept, kept.name)])
    inbox.close()  # as when Proteia stops
    assert unchanged() == before


def _stage_in(inbox: handoff.Inbox, name: str, data: bytes) -> str:
    """``data`` staged as ``name`` straight into ``inbox``: its file id."""
    upload = inbox.begin_upload(name, len(data))
    with upload.open() as out:
        out.write(data)
    return inbox.upload_stored(upload, len(data)).file_id


def _placed_workspace(tmp_path: Path) -> tuple[api.Workspace, handoff.Inbox]:
    inbox = handoff.Inbox()
    inbox.place(tmp_path / "incoming")
    workspace = api.Workspace(
        tmp_path / "root", reveal=lambda folder: None, clock=FakeClock(), inbox=inbox
    )
    return workspace, inbox


def _chosen(file_ids: list[str]) -> list[api.FileChoice]:
    return [
        api.FileChoice(file_id, api.ImageKind.CHEMILUMINESCENCE, api.Polarity.DARK_ON_LIGHT)
        for file_id in file_ids
    ]


def test_a_stop_during_an_accept_waits_for_the_file_being_imported(tmp_path, monkeypatch):
    # Stopping waits for the accept (it holds the switch lock), but no longer
    # than the file being imported: the rest are not imported, the project is
    # saved with what was, and the staged files are deleted once the accept
    # no longer reads them.
    workspace, inbox = _placed_workspace(tmp_path)
    workspace.create("Blot")
    data = blot_bytes(tmp_path)
    ids = [_stage_in(inbox, name, data) for name in ("a.tif", "b.tif")]
    offered = inbox.offer(ids)
    entered, release = _held_until_released(monkeypatch, api.ops, "import_image")
    accepting, accepted = _in_thread(
        lambda: workspace.accept(offered.handoff_id, None, _chosen(ids))
    )
    try:
        assert entered.wait(20)
        stopping, _ = _in_thread(workspace.close)
        deadline = time.monotonic() + 10
        while not inbox.stopping:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        stopping.join(0.3)
        assert stopping.is_alive()  # it waits for the accept
        assert len(list((tmp_path / "incoming").iterdir())) == 2
    finally:
        release.set()
    for thread in (accepting, stopping):
        thread.join(30)
        assert not thread.is_alive()
    session, done = accepted[0]
    assert [file.name for file in done.imported] == ["a.tif"]
    assert [(file.name, file.code) for file in done.refused] == [("b.tif", "stopping")]
    saved = storage.load_project(session.folder)
    assert [image.original_name for image in saved.batch.iter_images()] == ["a.tif"]
    assert list((tmp_path / "incoming").iterdir()) == []


def test_an_accept_checks_the_opening_again_once_no_other_switch_runs(tmp_path):
    workspace, inbox = _placed_workspace(tmp_path)
    workspace.create("Blot")  # opening 1, as a page shows it
    ids = [_stage_in(inbox, "a.tif", blot_bytes(tmp_path))]
    offered = inbox.offer(ids)
    workspace.create("Other")  # another page's switch, after the route's first check
    with pytest.raises(api.ProjectChangedError):
        workspace.accept(offered.handoff_id, None, _chosen(ids), opening=1)
    assert [view.id for view in inbox.listing()] == [offered.handoff_id]  # pending again
    assert workspace.current().folder.name == "Other"
    session, _ = workspace.accept(offered.handoff_id, None, _chosen(ids), opening=2)
    assert session.folder.name == "a"


def test_the_status_says_proteia_takes_handed_off_files(client):
    assert client.ok("GET", "/api/status")["handoff"] == server.HANDOFF == 1
