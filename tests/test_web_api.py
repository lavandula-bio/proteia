# SPDX-License-Identifier: Apache-2.0
"""The HTTP routes over the project operations (#50): projects in the app-managed
root, image upload and previews, lanes, proteins and boxes. Requests go to a real
server on a loopback socket; every edit answers with the stored project state."""

from __future__ import annotations

import http.client
import io
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

from conftest import MEMBRANE_LEVEL, FakeClock, synthetic_blot, write_tiff
from proteia.core import storage
from proteia.web import api, launch

H, W = 60, 400
ROW = 30
LANE_X = [50, 120, 190, 260, 330]
NAME = "β-actin 10 µM.tif"  # beta-actin 10 micro-molar


class Client:
    """JSON requests with this launch's token, over http.client."""

    def __init__(self, port: int, token: str, root: Path, revealed: list[Path]) -> None:
        self.port, self.token, self.root, self.revealed = port, token, root, revealed

    def call(
        self, method: str, path: str, body: Any = None, *, raw: bytes | None = None
    ) -> tuple[int, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
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
        return response.status, json.loads(payload) if kind.startswith("application/json") else (
            kind,
            payload,
        )

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
    yield Client(instance.port, instance.token, root, revealed)
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
    assert client.ok("GET", "/api/project") == {"project": moved["project"]}  # what is drawn


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
