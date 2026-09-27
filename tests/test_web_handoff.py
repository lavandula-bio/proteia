# SPDX-License-Identifier: Apache-2.0
"""The inbox of images handed to the running app (#57, N3): staged and local
files, offers and their merging, the limits, expiry, claims and discards, and
what is deleted when. No server: the routes over it are tested in
test_web_api.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from proteia.core.model import UnknownIdError
from proteia.core.session import OperationError
from proteia.web import handoff
from proteia.web.handoff import Inbox, LocalFile, Refusal, StagedFile

HEX = "0123456789abcdef" * 2  # a name the server could have made


class Ticks:
    """A monotonic clock that moves only when told."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def ticks() -> Ticks:
    return Ticks()


@pytest.fixture
def inbox(tmp_path, ticks) -> Inbox:
    inbox = Inbox(clock=ticks)
    inbox.place(tmp_path / "incoming")
    return inbox


def stage(inbox: Inbox, name: str = "a.tif", data: bytes = b"pixels") -> str:
    upload = inbox.begin_upload(name, len(data))
    with upload.open() as out:
        out.write(data)
    return inbox.upload_stored(upload, len(data)).file_id


def staged_names(tmp_path: Path) -> list[str]:
    folder = tmp_path / "incoming"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def files_of(inbox: Inbox) -> list[list[str]]:
    return [[file.name for file in view.files] for view in inbox.listing()]


# --- Names and refused entries ---


@pytest.mark.parametrize("name", ["a.tif", "β-actin 10 µM.TIF", "x.jpeg", "中文.png", "a b.tiff"])
def test_an_image_name_is_taken_as_given(name):
    assert handoff.check_name(name) == name


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("a.bmp", "unsupported_image_type"),
        ("a", "unsupported_image_type"),
        ("project.json", "unsupported_image_type"),
        ("a/b.tif", "invalid_image"),
        ("a\\b.tif", "invalid_image"),
        ("C:\\x\\a.tif", "invalid_image"),
        ("../a.tif", "invalid_image"),
        ("a\x00.tif", "invalid_image"),
        ("half\ud800.tif", "invalid_image"),
        ("x" * 252 + ".tif", "invalid_image"),
    ],
)
def test_a_name_that_is_not_an_images_plain_name_is_refused(name, code):
    with pytest.raises(OperationError) as refused:
        handoff.check_name(name)
    assert refused.value.code.value == code


def test_a_refused_entry_is_bounded_whatever_a_launch_sends():
    entry = Refusal.bounded("β" * 300, "missing", "no such file")
    assert entry.name == "β" * 119 + "…" and len(entry.name) == handoff.MAX_REFUSED_NAME
    entry = Refusal.bounded("a\x07\u202e\u2028\ud800b.tif", "made_up", "m" * 500)
    assert entry == Refusal("a\ufffd\ufffd\ufffd\ufffdb.tif", "other", "m" * 199 + "…")
    kept = Refusal.bounded("β-actin 實驗 µ.bmp", "unsupported_type", "not an image type")
    assert kept == Refusal("β-actin 實驗 µ.bmp", "unsupported_type", "not an image type")


# --- What is deleted, and what never is ---


def test_only_a_staged_file_in_the_staging_folder_can_be_deleted(tmp_path, caplog):
    folder = tmp_path / "incoming"
    folder.mkdir()
    staged = folder / HEX
    staged.write_bytes(b"x")
    elsewhere = tmp_path / HEX
    elsewhere.write_bytes(b"x")
    named_otherwise = folder / "blot.tif"
    named_otherwise.write_bytes(b"x")
    StagedFile(elsewhere, folder).delete()  # not in the staging folder
    StagedFile(named_otherwise, folder).delete()  # not a name the server makes
    assert elsewhere.exists() and named_otherwise.exists()
    assert "not a staged file" in caplog.text
    StagedFile(staged, folder).delete()
    assert not staged.exists()
    assert not hasattr(LocalFile(elsewhere), "delete")  # a user's file cannot be deleted


def test_a_new_instance_clears_only_the_staged_files_a_crash_left(tmp_path):
    folder = tmp_path / "incoming"
    (folder / ("f" * 32)).mkdir(parents=True)  # a folder, though named as a staged file
    leftovers = [folder / HEX, folder / ("0" * 32)]
    others = [folder / "blot.tif", folder / ("a" * 31), folder / (HEX + ".tif")]
    for path in leftovers + others:
        path.write_bytes(b"x")
    inbox = Inbox()
    inbox.place(folder)
    assert inbox.folder == folder
    assert sorted(p.name for p in folder.iterdir()) == sorted([p.name for p in others] + ["f" * 32])


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symbolic links")
def test_the_clearing_follows_no_link(tmp_path):
    folder = tmp_path / "incoming"
    folder.mkdir()
    target = tmp_path / "a user's file.tif"
    target.write_bytes(b"x")
    try:
        (folder / HEX).symlink_to(target)
    except OSError:
        pytest.skip("symbolic links need a privilege here")
    Inbox().place(folder)
    assert target.read_bytes() == b"x" and (folder / HEX).is_symlink()


def test_the_staging_folder_is_made_when_the_first_file_comes(tmp_path, inbox):
    assert not (tmp_path / "incoming").exists()
    stage(inbox)
    assert len(staged_names(tmp_path)) == 1
    if os.name != "nt":
        assert (tmp_path / "incoming").stat().st_mode & 0o777 == 0o700


def test_a_file_is_staged_under_a_new_name_never_over_another(tmp_path, inbox):
    upload = inbox.begin_upload("a.tif", 3)
    with upload.open() as out:
        out.write(b"abc")
    with pytest.raises(FileExistsError):
        upload.open()  # never over an existing file
    pending = inbox.upload_stored(upload, 3)
    assert isinstance(pending.source, StagedFile) and pending.source.path.read_bytes() == b"abc"
    assert pending.source.path.name != pending.file_id


def test_local_files_are_never_deleted_or_copied(tmp_path, inbox):
    original = tmp_path / "blot α.tif"
    original.write_bytes(b"pixels")
    for finish in ("discard", "finish", "close"):
        offered = inbox.add_local([(original, original.name)])
        assert offered is not None
        (view,) = [v for v in inbox.listing() if v.id == offered.handoff_id]
        if finish == "discard":
            inbox.discard(view.id, [f.file_id for f in view.files], 0)
        elif finish == "finish":
            inbox.finish(inbox.claim(view.id, [f.file_id for f in view.files]))
        else:
            inbox.close()
        assert original.read_bytes() == b"pixels"
    assert staged_names(tmp_path) == []


def test_a_local_file_that_cannot_be_read_is_refused(tmp_path, inbox):
    offered = inbox.add_local([(tmp_path / "missing.tif", "missing.tif")])
    assert offered is not None
    (view,) = inbox.listing()
    assert (view.kind, view.files) == ("notice", ())
    ((name, code, message),) = [(r.name, r.code, r.message) for r in view.refused]
    assert (name, code) == ("missing.tif", "unreadable") and str(tmp_path) not in message
    assert inbox.add_local([]) is None


# --- Offers and merging ---


def test_offers_within_the_window_of_the_last_upload_start_merge(inbox, ticks):
    first = inbox.offer([stage(inbox, "a.tif")])
    ticks.now += handoff.MERGE_WINDOW_S - 0.5
    second = inbox.offer([stage(inbox, "b.tif")])
    assert (second.handoff_id, second.merged, second.files) == (first.handoff_id, True, 2)
    ticks.now += handoff.MERGE_WINDOW_S  # measured from b.tif's start, not a.tif's
    third = inbox.offer([stage(inbox, "c.tif")])
    assert not third.merged and third.handoff_id != first.handoff_id
    assert files_of(inbox) == [["a.tif", "b.tif"], ["c.tif"]]


def test_an_upload_begun_early_merges_however_long_it_took(inbox, ticks):
    slow = inbox.begin_upload("slow.tif", 4)  # begins first
    ticks.now += 1
    quick = inbox.offer([stage(inbox, "quick.tif")])
    ticks.now += 100  # the slow one takes long
    with slow.open() as out:
        out.write(b"slow")
    inbox.upload_stored(slow, 4)
    offered = inbox.offer([slow.file_id])
    assert (offered.handoff_id, offered.merged) == (quick.handoff_id, True)


def test_a_merge_beyond_the_files_a_hand_off_holds_starts_another(inbox, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_HANDOFF_FILES", 3)
    first = inbox.offer([stage(inbox, "a.tif"), stage(inbox, "b.tif")])
    second = inbox.offer([stage(inbox, "c.tif"), stage(inbox, "d.tif")])  # 4 > 3: not split
    assert not second.merged and second.handoff_id != first.handoff_id
    third = inbox.offer([stage(inbox, "e.tif")])
    assert third.handoff_id == second.handoff_id
    assert files_of(inbox) == [["a.tif", "b.tif"], ["c.tif", "d.tif", "e.tif"]]


def test_an_offer_of_more_files_than_one_hand_off_holds_is_split_in_order(inbox, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_HANDOFF_FILES", 2)
    refused = [Refusal("x.bmp", "unsupported_type", "not an image type")]
    offered = inbox.offer([stage(inbox, f"{n}.tif") for n in "abcde"], refused)
    assert (offered.merged, offered.files, offered.refused) == (False, 2, 1)
    views = inbox.listing()
    assert [[f.name for f in v.files] for v in views] == [
        ["a.tif", "b.tif"],
        ["c.tif", "d.tif"],
        ["e.tif"],
    ]
    assert [len(v.refused) for v in views] == [1, 0, 0]  # with the first files
    assert views[0].id == offered.handoff_id


def test_a_notice_joins_young_images_else_a_young_notice_else_starts_one(inbox, ticks):
    refused = [Refusal("x.bmp", "unsupported_type", "not an image type")]
    images = inbox.offer([stage(inbox)])
    assert inbox.offer([], refused).handoff_id == images.handoff_id
    ticks.now += handoff.MERGE_WINDOW_S
    notice = inbox.offer([], refused)
    assert not notice.merged
    ticks.now += handoff.MERGE_WINDOW_S - 1
    assert inbox.offer([], refused).handoff_id == notice.handoff_id
    ticks.now += handoff.MERGE_WINDOW_S
    later = inbox.offer([], refused)
    assert later.handoff_id not in (images.handoff_id, notice.handoff_id)
    # An images offer never joins a notice.
    assert inbox.offer([stage(inbox)]).handoff_id not in (notice.handoff_id, later.handoff_id)
    assert [(v.kind, len(v.refused)) for v in inbox.listing()] == [
        ("images", 1),
        ("notice", 2),
        ("notice", 1),
        ("images", 0),
    ]


def test_refused_entries_merge_up_to_the_hundred_kept(inbox, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_REFUSED", 3)
    entries = [Refusal(f"{i}.bmp", "unsupported_type", "no") for i in range(5)]
    first = inbox.offer([stage(inbox)], entries[:2])
    merged = inbox.offer([], entries[2:])
    assert (merged.handoff_id, merged.refused) == (first.handoff_id, 5)
    (view,) = inbox.listing()
    assert [r.name for r in view.refused] == ["0.bmp", "1.bmp", "2.bmp"]
    assert view.more_refused == 2


def test_a_claimed_hand_off_takes_no_more_files_and_is_not_listed(inbox):
    offered = inbox.offer([stage(inbox, "a.tif")])
    (view,) = inbox.listing()
    claimed = inbox.claim(offered.handoff_id, [f.file_id for f in view.files])
    assert inbox.listing() == []
    later = inbox.offer([stage(inbox, "b.tif")])
    assert not later.merged and later.handoff_id != offered.handoff_id
    with pytest.raises(handoff.HandoffClaimedError):
        inbox.claim(offered.handoff_id, [f.file_id for f in view.files])
    with pytest.raises(handoff.HandoffClaimedError):
        inbox.discard(offered.handoff_id, [f.file_id for f in view.files], 0)
    inbox.release(claimed)
    assert [v.id for v in inbox.listing()] == [offered.handoff_id, later.handoff_id]


def test_an_offer_refuses_unknown_repeated_or_offered_files_changing_nothing(inbox):
    a = stage(inbox, "a.tif")
    with pytest.raises(UnknownIdError):
        inbox.offer([a, "0" * 16])
    with pytest.raises(OperationError):
        inbox.offer([a, a])
    with pytest.raises(OperationError):
        inbox.offer([])
    assert inbox.listing() == []
    inbox.offer([a])
    with pytest.raises(handoff.FileClaimedError):
        inbox.offer([a])


# --- Limits and expiry ---


def test_the_pending_files_count_uploads_under_way_waiting_and_handed_off(inbox, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_PENDING_FILES", 3)
    inbox.offer([stage(inbox, "a.tif")])
    stage(inbox, "b.tif")
    running = inbox.begin_upload("c.tif", None)
    with pytest.raises(handoff.TooManyPendingError):
        inbox.begin_upload("d.tif", None)
    inbox.upload_failed(running)
    inbox.begin_upload("d.tif", None)


def test_the_staged_bytes_count_what_waits_and_what_uploads_hold(tmp_path, inbox, monkeypatch):
    monkeypatch.setattr(handoff, "MAX_STAGED_BYTES", 10)
    local = tmp_path / "local.tif"
    local.write_bytes(b"x" * 100)
    inbox.add_local([(local, local.name)])  # not staged: takes no room
    stage(inbox, "a.tif", b"x" * 4)
    with pytest.raises(handoff.TooManyPendingError):
        inbox.begin_upload("b.tif", 7)
    upload = inbox.begin_upload("b.tif", None)  # no size declared
    inbox.make_room(upload, 6)
    with pytest.raises(handoff.TooManyPendingError):
        inbox.make_room(upload, 7)
    with pytest.raises(handoff.TooManyPendingError):
        inbox.begin_upload("c.tif", 1)  # the 6 held count


def test_an_upload_no_offer_takes_expires(tmp_path, inbox, ticks):
    old = stage(inbox, "old.tif")
    ticks.now += handoff.UPLOAD_EXPIRY_S - 1
    young = stage(inbox, "young.tif")
    assert len(staged_names(tmp_path)) == 2
    ticks.now += 1
    with pytest.raises(UnknownIdError):
        inbox.offer([old])
    assert len(staged_names(tmp_path)) == 1  # deleted at that offer
    assert inbox.offer([young]).files == 1


def test_more_may_arrive_while_young_or_while_a_near_upload_waits(inbox, ticks):
    offered = inbox.offer([stage(inbox, "a.tif")])

    def more() -> bool:
        (view,) = [v for v in inbox.listing() if v.id == offered.handoff_id]
        return view.more_may_arrive

    assert more()
    ticks.now += handoff.MERGE_WINDOW_S
    assert not more()
    ticks.now -= handoff.MERGE_WINDOW_S + 1  # an upload that began before it
    upload = inbox.begin_upload("b.tif", None)
    ticks.now += handoff.MERGE_WINDOW_S + 100
    assert more()  # under way
    with upload.open() as out:
        out.write(b"b")
    inbox.upload_stored(upload, 1)
    assert more()  # staged, not yet offered
    inbox.offer([upload.file_id])
    assert not more()


def test_an_upload_left_unoffered_holds_off_no_hand_off_for_long(inbox, ticks):
    # A launch uploads its files one by one, then offers them at once. A file
    # uploaded and never offered (its launch exited) waits until it expires,
    # but a hand-off offered later still settles once its window has passed.
    stage(inbox, "orphan.tif")
    ticks.now += 300
    offered = inbox.offer([stage(inbox, "blot.tif")])

    def more() -> bool:
        (view,) = [v for v in inbox.listing() if v.id == offered.handoff_id]
        return view.more_may_arrive

    assert more()
    ticks.now += handoff.MERGE_WINDOW_S
    assert not more()


def test_a_file_waiting_for_its_launchs_next_upload_keeps_a_hand_off_settling(inbox, ticks):
    # A file uploaded near a hand-off's last, and not yet offered, waits while
    # its launch's next file uploads, however late that one began, and until
    # the offer that follows: the offer joins the hand-off.
    offered = inbox.offer([stage(inbox, "a.tif")])

    def more() -> bool:
        (view,) = [v for v in inbox.listing() if v.id == offered.handoff_id]
        return view.more_may_arrive

    ticks.now += 5
    near = stage(inbox, "near.tif")
    ticks.now += handoff.MERGE_WINDOW_S + 5
    upload = inbox.begin_upload("late.tif", None)  # began past the window
    ticks.now += handoff.MERGE_WINDOW_S * 3
    assert more()
    with upload.open() as out:
        out.write(b"late")
    inbox.upload_stored(upload, 4)
    assert more()  # the offer comes at once
    assert inbox.offer([near, upload.file_id]).merged
    ticks.now += handoff.MERGE_WINDOW_S
    assert not more()


# --- Claims, discards and stopping ---


def test_a_claim_or_discard_of_other_files_than_those_held_changes_nothing(tmp_path, inbox):
    offered = inbox.offer([stage(inbox, "a.tif"), stage(inbox, "b.tif")])
    (view,) = inbox.listing()
    ids = [file.file_id for file in view.files]
    for wrong in ([ids[0]], [ids[0], ids[0]], [*ids, "x"], []):
        with pytest.raises(handoff.HandoffChangedError) as changed:
            inbox.claim(offered.handoff_id, wrong)
        assert changed.value.view == view
        with pytest.raises(handoff.HandoffChangedError):
            inbox.discard(offered.handoff_id, wrong, 0)
    with pytest.raises(handoff.HandoffChangedError):
        inbox.discard(offered.handoff_id, ids, 1)  # it holds no refused entry
    assert inbox.listing() == [view] and len(staged_names(tmp_path)) == 2
    with pytest.raises(handoff.HandoffNotFoundError):
        inbox.claim("nothing", ids)
    inbox.discard(offered.handoff_id, list(reversed(ids)), 0)
    assert inbox.listing() == [] and staged_names(tmp_path) == []
    with pytest.raises(handoff.HandoffNotFoundError):
        inbox.discard(offered.handoff_id, ids, 0)


def test_once_stopping_nothing_starts_and_closing_deletes_every_staged_file(tmp_path, inbox):
    waiting = stage(inbox, "a.tif")
    offered = inbox.offer([stage(inbox, "b.tif")])
    running = inbox.begin_upload("c.tif", None)
    with running.open() as out:
        out.write(b"c")
    (view,) = inbox.listing()
    inbox.stop()
    assert inbox.stopping
    with pytest.raises(handoff.StoppingError):
        inbox.begin_upload("d.tif", None)
    with pytest.raises(handoff.StoppingError):
        inbox.offer([waiting])
    with pytest.raises(handoff.StoppingError):
        inbox.claim(offered.handoff_id, [f.file_id for f in view.files])
    with pytest.raises(handoff.StoppingError):
        inbox.upload_stored(running, 1)
    assert len(staged_names(tmp_path)) == 3
    inbox.close()
    assert staged_names(tmp_path) == [] and inbox.listing() == []
