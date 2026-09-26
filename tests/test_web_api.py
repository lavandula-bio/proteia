# SPDX-License-Identifier: Apache-2.0
"""The HTTP routes over the project operations (#50): projects in the app-managed
root, image upload and previews, lanes, proteins and boxes. Requests go to a real
server on a loopback socket; every edit answers with the stored project state and
its live results (#52), in strict JSON."""

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

from conftest import (
    MEMBRANE_LEVEL,
    FakeClock,
    make_project_with_undetected,
    synthetic_blot,
    write_image_files,
    write_tiff,
)
from proteia.core import storage
from proteia.core.analyze import ReduceMethod
from proteia.core.model import (
    Project,
    ProposalSource,
    Region,
    UndetectedBand,
    UndetectedReason,
    apply_change,
)
from proteia.core.plotspec import ErrorType
from proteia.core.results import compute_results
from proteia.web import api, launch
from proteia.web.results_view import results_payload
from proteia.web.state import project_state, revision

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
    """Store records in the open project: the row-box route that writes them is #51's."""
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
    expected = results_payload(compute_results(saved.batch), open_id=open_id, revision=revision)
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

    # Every create or open is a new open id, even of the project already open.
    assert client.ok("POST", "/api/projects", {"name": "Other"})["project"]["open_id"] == 2
    for open_id in (3, 4):
        answer = client.ok("POST", "/api/projects/open", {"name": "Blot"})
        assert (answer["project"]["open_id"], answer["project"]["revision"]) == (open_id, 7)
        assert (answer["results"]["open_id"], answer["results"]["revision"]) == (open_id, 7)


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
    x0, y0, x1, y1 = bands(answers["POST /api/boxes"])[band]["rect"]
    answers["PUT /api/boxes/{band_id}"] = client.ok(
        "PUT", f"/api/boxes/{band}", {"rect": [x0 + 2, y0, x1 + 2, y1]}
    )
    answers["PUT /api/boxes/{band_id}/lane"] = client.ok(
        "PUT", f"/api/boxes/{band}/lane", {"lane_index": 1}
    )
    answers["DELETE /api/boxes/{band_id}"] = client.ok("DELETE", f"/api/boxes/{band}")
    answers["DELETE /api/images/{image_id}"] = client.ok(
        "DELETE", f"/api/images/{second['image_id']}"
    )
    answers["POST /api/projects/open"] = client.ok("POST", "/api/projects/open", {"name": "Blot"})

    revisions = []
    for route, answer in answers.items():
        project, results = answer["project"], answer["results"]
        assert (results["open_id"], results["revision"]) == (
            project["open_id"],
            project["revision"],
        ), route
        assert {"lanes", "proteins", "sets", "settings"} <= set(results), route
        revisions.append(project["revision"])
    assert revisions == sorted(revisions)
    # Every route that answers with the project is exercised above.
    others = {"GET /api/projects", "POST /api/project/reveal", "GET /api/images/{image_id}/preview"}
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
    client.ok("POST", "/api/projects/open", {"name": "Blot"})  # the same project, opened again
    assert len(calls) == 3


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
