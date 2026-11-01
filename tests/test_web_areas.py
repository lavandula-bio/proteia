# SPDX-License-Identifier: Apache-2.0
"""The membrane-area routes and what the page state says of areas (#243)."""

import math

import pytest

from conftest import make_project, write_tiff
from proteia.core.model import Arena, Outline
from test_areas import STRIPS, two_strips
from test_web_api import client, upload  # noqa: F401 - the fixture

assert client  # the fixture, used by name


def strips_project(client, tmp_path) -> dict:
    """A project with the two strips uploaded and three lanes; its state."""
    client.ok("POST", "/api/projects", {"name": "Strips"})
    data = write_tiff(tmp_path / "strips.tif", two_strips()).read_bytes()
    status, answer = upload(client, data, name=STRIPS)
    assert status == 201, answer
    lanes = [{"condition": c} for c in ("vehicle", "10 µM", "50 µM")]
    return client.ok("PUT", "/api/lanes", {"lanes": lanes})["project"]


def test_a_whole_image_area_in_the_state(client, tmp_path):
    project = strips_project(client, tmp_path)
    [image] = project["images"]
    assert (image["membrane_id"], image["arena_ids"]) == ("mem-2", ["arena-4"])
    assert project["arenas"] == [
        {
            "id": "arena-4",
            "membrane_id": "mem-2",
            "image_id": "img-1",
            "outline": None,
            "angle_deg": 0.0,
            "source": "whole_image",
            "corners": [[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]],
            "bounds": [0, 0, 200, 100],
        }
    ]
    [membrane] = project["membranes"]
    assert (membrane["label"], membrane["arena_ids"]) == ("Membrane 1", ["arena-4"])


def test_area_routes(client, tmp_path):
    strips_project(client, tmp_path)
    status, payload = client.call(
        "POST", "/api/arenas", {"image_id": "img-1", "outline": [0, 50, 200, 100]}
    )
    assert (status, payload["code"]) == (422, "arena_covers_image")
    answer = client.ok("PUT", "/api/arenas/arena-4", {"outline": [0, 0, 200, 50]})
    assert answer["arena_ids"] == ["arena-4"]
    answer = client.ok(
        "POST",
        "/api/arenas",
        {"image_id": "img-1", "outline": [0, 50, 200, 100], "angle_deg": 2, "membrane_id": None},
    )
    assert (answer["arena_ids"], answer["membrane_id"], answer["new_membrane"]) == (
        ["arena-6"],
        "mem-5",
        True,
    )
    project = answer["project"]
    assert project["images"][0]["arena_ids"] == ["arena-4", "arena-6"]
    assert [m["label"] for m in project["membranes"]] == ["Membrane 1", "Membrane 2"]
    drawn = project["arenas"][1]
    assert (drawn["outline"], drawn["angle_deg"], drawn["source"]) == (
        [0, 50, 200, 100],
        2.0,
        "manual",
    )
    # A protein on the area; the image alone is refused.
    status, payload = client.call(
        "POST", "/api/proteins", {"name": "p53", "role": "target", "image_id": "img-1"}
    )
    assert (status, payload["code"]) == (422, "arena_required")
    answer = client.ok(
        "POST", "/api/proteins", {"name": "p53", "role": "target", "arena_id": "arena-6"}
    )
    [protein] = answer["project"]["proteins"]
    assert (protein["arena_id"], protein["image_id"]) == ("arena-6", "img-1")
    # A box outside its area.
    status, payload = client.call(
        "POST", "/api/boxes", {"protein_id": protein["id"], "x": 40, "y": 24, "lane_index": 0}
    )
    assert (status, payload["code"]) == (422, "out_of_arena")
    # The angle alone; then the membrane; then the area goes.
    answer = client.ok("PUT", "/api/arenas/arena-6", {"angle_deg": -1.5})
    assert answer["project"]["arenas"][1]["angle_deg"] == -1.5
    answer = client.ok("PUT", "/api/arenas/arena-6/membrane", {"membrane_id": None})
    assert (answer["membrane_id"], answer["removed_membrane"], answer["renumbered"]) == (
        "mem-9",
        "mem-5",
        None,
    )
    assert answer["moved_proteins"] == [protein["id"]]
    answer = client.ok("DELETE", "/api/arenas/arena-6")
    assert answer["removed"][:2] == ["arena-6", protein["id"]]
    assert answer["renumbered"] is None and answer["remeasured"] == []
    status, payload = client.call("DELETE", "/api/arenas/arena-4")
    assert (status, payload["code"]) == (422, "arena_required")


def test_an_unknown_area_answers_404_then_the_state(client, tmp_path):
    strips_project(client, tmp_path)
    for method, path, body in (
        ("PUT", "/api/arenas/arena-99", {"angle_deg": 1.0}),
        ("DELETE", "/api/arenas/arena-99", None),
        ("PUT", "/api/arenas/arena-99/membrane", {"membrane_id": None}),
        ("PUT", "/api/arenas/img-1", {"angle_deg": 1.0}),
        ("POST", "/api/arenas/arena-99/ladder-proposal", {"x": 5.0}),
    ):
        status, payload = client.call(method, path, body)
        assert status == 404, (path, payload)
        assert payload["code"] == "unknown_id"
        assert payload["message"] == (
            "that membrane area no longer exists; it may have been removed in another window"
        )
        assert payload["detail"] == {"arena_id": path.split("/")[3]}
    # The page reads the state again: the areas as they are.
    assert [a["id"] for a in client.ok("GET", "/api/project")["project"]["arenas"]] == ["arena-4"]


def test_arena_calibration_routes(client, tmp_path):
    strips_project(client, tmp_path)
    client.ok("PUT", "/api/arenas/arena-4", {"outline": [0, 0, 200, 50]})
    client.ok("POST", "/api/arenas", {"image_id": "img-1", "outline": [0, 50, 200, 100]})
    point = {"y": 70.0, "mw": 50.0, "source": "strip_edge", "x": 5.0, "snap": False}
    status, payload = client.call("POST", "/api/images/img-1/calibration/left/points", point)
    assert (status, payload["code"]) == (422, "arena_required")
    answer = client.ok("POST", "/api/arenas/arena-6/calibration/left/points", point)
    assert answer["point"]["image_id"] == "img-1"
    groups = [m["groups"] for m in answer["project"]["membranes"]]
    assert [len(g[0]["points"]) for g in groups] == [0, 1]
    answer = client.ok("PATCH", "/api/arenas/arena-6/calibration/left/points/50", {"y": 80.0})
    assert answer["point"]["y"] == 80.0
    answer = client.ok("POST", "/api/arenas/arena-6/ladder-snap", {"x": 100.0, "ys": [76.0]})
    assert len(answer["points"]) == 1
    client.ok("DELETE", "/api/arenas/arena-6/calibration/left/points/50")
    client.ok("DELETE", "/api/arenas/arena-6/calibration")
    status, payload = client.call("POST", "/api/arenas/arena-6/ladder-proposal", {"x": 5.0})
    assert (status, payload["code"]) == (422, "invalid_input")  # no ladder chosen: as by image
    ruler = {"x": 5.0, "points": [{"y": 60.0, "mw": 100.0}, {"y": 90.0, "mw": 30.0}]}
    answer = client.ok("PUT", "/api/arenas/arena-6/calibration/left/ladder", ruler)
    assert [p["image_id"] for p in answer["points"]] == ["img-1", "img-1"]
    ruler["points"][0]["y"] = 20.0  # on membrane 1's strip
    status, payload = client.call("PUT", "/api/arenas/arena-6/calibration/left/ladder", ruler)
    assert (status, payload["code"]) == (422, "out_of_arena")


def test_marker_link_answers_the_areas_it_adopts(client, tmp_path):
    strips_project(client, tmp_path)
    client.ok("PUT", "/api/arenas/arena-4", {"outline": [0, 0, 200, 50]})
    client.ok("POST", "/api/arenas", {"image_id": "img-1", "outline": [0, 50, 200, 100]})
    data = write_tiff(tmp_path / "m.tif", two_strips()).read_bytes()
    status, answer = upload(
        client, data, name="marker.tif", kind="visible_marker", membrane_id="mem-2"
    )
    assert status == 201, answer
    marker = answer["image_id"]
    answer = client.ok("PUT", "/api/images/img-1/marker", {"marker_image_id": marker})
    on_marker = [a for a in answer["project"]["arenas"] if a["image_id"] == marker]
    assert answer["adopted_arena_ids"] == [a["id"] for a in on_marker]
    assert [a["outline"] for a in on_marker] == [[0, 0, 200, 50], [0, 50, 200, 100]]


@pytest.mark.parametrize("angle", [0.0, 4.0, -3.0, 45.0])
def test_state_corners_match_model_geometry(client, tmp_path, angle):
    strips_project(client, tmp_path)
    answer = client.ok(
        "PUT", "/api/arenas/arena-4", {"outline": [20, 10, 180, 60], "angle_deg": angle}
    )
    [arena] = answer["project"]["arenas"]
    image = (
        make_project().batch.find_image("img-2").model_copy(update={"width": 200, "height": 100})
    )
    model = Arena(
        id="arena-4",
        image_id="img-2",
        outline=Outline(x0=20, y0=10, x1=180, y1=60),
        angle_deg=angle,
        source="manual",
    )
    assert arena["corners"] == [list(corner) for corner in model.corners(image)]
    assert arena["bounds"] == list(model.bounds(image))
    # Each corner lies on the turned outline: the model holds it (closed).
    for x, y in arena["corners"]:
        assert model.contains(image, x, y)
    # Turned by a positive angle, the right side runs lower.
    (_, y_left), (_, y_right) = arena["corners"][0], arena["corners"][1]
    assert math.copysign(1.0, y_right - y_left) == (
        1.0 if angle > 0 else -1.0 if angle < 0 else 1.0
    )
