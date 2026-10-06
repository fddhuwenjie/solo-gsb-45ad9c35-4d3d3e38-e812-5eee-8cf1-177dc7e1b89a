"""Integration tests for POST /ingest/preview (read-only ingest dry run)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import KIND_POSITION, SnapshotStore, normalize_observed_at
from tests.conftest import (
    AMBIG_F1A,
    AMBIG_F1B,
    AMBIG_F2,
    INCOMPLETE_F1,
    MULTI_FRAG1,
    MULTI_FRAG2,
    SINGLE_FRAGMENT,
    SINGLE_PAYLOAD,
    make_sentence,
)
from tests.test_conflicts import FAR_SENTENCE

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

# Same MMSI as SINGLE_FRAGMENT (366999999) but a different longitude.
OTHER_POSITION = make_sentence("AIVDM,1,1,,A,15Mwqgh01DreF9HGD@44gkg40000,0")

MMSI = 366999999
NORTHERN = 366000001  # AMBIG_F1A + AMBIG_F2 -> "NORTHERN STAR"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh snapshot database per test."""
    monkeypatch.setenv("AIS_DB_PATH", str(tmp_path / "snapshots.db"))
    monkeypatch.setattr(main, "_store", None)
    yield TestClient(main.app)
    monkeypatch.setattr(main, "_store", None)


def one(source_id: str, sentence: str) -> dict:
    return {"source_id": source_id, "sentence": sentence}


def preview(client: TestClient, sentences: list[dict], observed_at: str):
    response = client.post(
        "/ingest/preview", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def ingest(client: TestClient, sentences: list[dict], observed_at: str):
    response = client.post(
        "/ingest", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def vessels(client: TestClient):
    return client.get("/vessels").json()


def track(client: TestClient, source_id: str = "rx-1", mmsi: int = MMSI):
    response = client.get("/tracks", params={"source_id": source_id, "mmsi": mmsi})
    return response.status_code, response.json()


def conflicts(client: TestClient):
    return client.get("/position-conflicts").json()


# -- the preview reports exactly what /ingest would persist --------------------


def test_preview_reports_snapshot_track_and_leaves_no_writes(client):
    status, body = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert body["observed_at"].startswith("2026-10-05T12:00:00")
    # decode results and summary mirror /ingest
    assert body["summary"]["decoded"] == 1
    assert body["results"][0]["status"] == "decoded"

    (change,) = body["changes"]["snapshots"]
    assert change["source_id"] == "rx-1"
    assert change["mmsi"] == MMSI
    assert change["before"] is None  # nothing stored yet
    assert change["after"]["position"]["observed_at"].startswith(
        "2026-10-05T12:00:00"
    )
    assert change["after"]["position"]["data"]["longitude"] == pytest.approx(
        -73.9835
    )
    assert change["after"]["static"] is None

    (point,) = body["changes"]["track_points"]
    assert point["source_id"] == "rx-1"
    assert point["mmsi"] == MMSI
    assert point["observed_at"].startswith("2026-10-05T12:00:00")
    assert point["data"]["longitude"] == pytest.approx(-73.9835)

    assert body["changes"]["conflicts"] == []

    # the dry run persisted nothing
    assert vessels(client) == []
    assert track(client)[0] == 404
    assert conflicts(client) == []


def test_ingest_after_preview_persists_exactly_the_previewed_changes(client):
    sentences = [one("rx-1", SINGLE_FRAGMENT)]
    _, previewed = preview(client, sentences, T2)

    status, _ = ingest(client, sentences, T2)
    assert status == 200

    # snapshot equals the previewed "after" value
    (snapshot,) = vessels(client)
    (change,) = previewed["changes"]["snapshots"]
    assert snapshot == change["after"]
    # the previewed track point is the stored one
    _, stored_track = track(client)
    assert len(stored_track["points"]) == 1
    assert stored_track["points"][0]["observed_at"] == previewed["changes"][
        "track_points"
    ][0]["observed_at"]
    assert stored_track["points"][0]["data"] == previewed["changes"][
        "track_points"
    ][0]["data"]

    # a second preview of the same batch now reports no changes at all
    _, again = preview(client, sentences, T2)
    assert again["changes"] == {
        "snapshots": [],
        "track_points": [],
        "conflicts": [],
    }


def test_preview_late_point_adds_track_point_but_no_snapshot_change(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)  # newer snapshot

    status, body = preview(client, [one("rx-1", OTHER_POSITION)], T1)  # late
    assert status == 200
    # the older report never rolls back the snapshot: not listed
    assert body["changes"]["snapshots"] == []
    # ...but it would still enter the track
    (point,) = body["changes"]["track_points"]
    assert point["observed_at"].startswith("2026-10-05T11:00:00")
    assert point["data"]["longitude"] == pytest.approx(-73.980087)
    # nothing was written
    _, stored_track = track(client)
    assert len(stored_track["points"]) == 1


def test_preview_same_observed_at_first_saved_wins(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)

    # a different position at the same observed_at loses: no changes at all
    status, body = preview(client, [one("rx-1", OTHER_POSITION)], T2)
    assert status == 200
    assert body["changes"] == {
        "snapshots": [],
        "track_points": [],
        "conflicts": [],
    }


def test_preview_does_not_resurrect_invalidated_point(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T2,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200

    # the same position resubmitted at the invalidated time: skipped entirely
    status, body = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert body["summary"]["decoded"] == 1  # decode contract unchanged
    assert body["changes"] == {
        "snapshots": [],
        "track_points": [],
        "conflicts": [],
    }
    # a newer observation of the same vessel still applies normally
    _, body = preview(client, [one("rx-1", OTHER_POSITION)], T3)
    assert len(body["changes"]["snapshots"]) == 1
    assert len(body["changes"]["track_points"]) == 1


def test_preview_static_message_reports_snapshot_change_only(client):
    status, body = preview(
        client, [one("rx-1", MULTI_FRAG1), one("rx-1", MULTI_FRAG2)], T2
    )
    assert status == 200
    (change,) = body["changes"]["snapshots"]
    assert change["before"] is None
    assert change["after"]["static"]["data"]["name"] == "MV CLAUDE"
    assert change["after"]["position"] is None
    # a message 5 creates no track point
    assert body["changes"]["track_points"] == []
    assert vessels(client) == []


def test_preview_errors_and_incomplete_change_nothing(client):
    bad = SINGLE_FRAGMENT.replace("15Mwqgh", "15Mwqgi")  # checksum mismatch
    status, body = preview(
        client, [one("rx-1", bad), one("rx-3", INCOMPLETE_F1)], T2
    )
    assert status == 200
    assert body["summary"]["errors"] == 1
    assert body["summary"]["incomplete"] == 1
    assert body["changes"] == {
        "snapshots": [],
        "track_points": [],
        "conflicts": [],
    }
    assert vessels(client) == []


# -- 409 semantics: only the unambiguous decodes are previewed ------------------


def test_preview_ambiguous_batch_returns_409_with_clear_decode_changes(client):
    status, body = preview(
        client,
        [
            one("rx-2", SINGLE_FRAGMENT),  # decodes unambiguously
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
        T2,
    )
    assert status == 409
    statuses = {r["status"] for r in body["results"]}
    assert statuses == {"decoded", "ambiguous"}
    # only the unambiguous position report is part of the changes
    (change,) = body["changes"]["snapshots"]
    assert change["mmsi"] == MMSI
    assert change["after"]["position"]["data"]["longitude"] == pytest.approx(
        -73.9835
    )
    (point,) = body["changes"]["track_points"]
    assert point["mmsi"] == MMSI
    # ambiguous candidates appear nowhere; nothing was persisted
    assert vessels(client) == []
    assert track(client, "rx-2", MMSI)[0] == 404


# -- conflicts ------------------------------------------------------------------


def test_preview_reports_new_same_instant_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    status, body = preview(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert status == 200
    (conflict,) = body["changes"]["conflicts"]
    assert conflict["mmsi"] == MMSI
    assert conflict["observed_at"].startswith("2026-10-05T11:00:00")
    assert conflict["source_id_a"] == "rx-1"
    assert conflict["source_id_b"] == "rx-2"
    assert conflict["distance_m"] > 1852
    # the conflict was not persisted
    assert conflicts(client) == []

    # the real ingest records exactly the previewed conflict
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert conflicts(client) == [conflict]


def test_preview_late_point_backfills_historical_conflict(client):
    # rx-1 reported at the older T1; rx-2's snapshot is already at the newer T2
    ingest(client, [one("rx-1", FAR_SENTENCE)], T1)
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T2)

    # rx-2's late T1 report: no snapshot change, but a new track point and
    # a historical conflict with rx-1's T1 point
    status, body = preview(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    assert status == 200
    assert body["changes"]["snapshots"] == []
    (point,) = body["changes"]["track_points"]
    assert point["observed_at"].startswith("2026-10-05T11:00:00")
    (conflict,) = body["changes"]["conflicts"]
    assert conflict["observed_at"].startswith("2026-10-05T11:00:00")
    assert conflict["source_id_a"] == "rx-1"
    assert conflict["source_id_b"] == "rx-2"
    # still nothing persisted
    assert conflicts(client) == []
    _, stored_track = track(client, "rx-2", MMSI)
    assert len(stored_track["points"]) == 1  # only the T2 ingest


def test_preview_existing_conflict_is_not_reported_again(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert len(conflicts(client)) == 1

    # retransmitting both points at the same instant changes nothing
    status, body = preview(
        client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_SENTENCE)], T1
    )
    assert status == 200
    assert body["changes"] == {
        "snapshots": [],
        "track_points": [],
        "conflicts": [],
    }


# -- request validation ---------------------------------------------------------


def test_preview_validates_request_schema(client):
    response = client.post(
        "/ingest/preview", json={"sentences": [one("rx-1", SINGLE_FRAGMENT)]}
    )
    assert response.status_code == 422  # missing observed_at



# -- store-level guarantees ------------------------------------------------------


def test_store_preview_leaves_no_writes_on_success(tmp_path):
    store = SnapshotStore(str(tmp_path / "preview.db"))
    observed = normalize_observed_at(
        datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    )
    message = decode_payload(SINGLE_PAYLOAD, 0).model_dump()

    changes = store.preview_ingest(
        [("rx-1", MMSI, KIND_POSITION, observed, message, 0)]
    )
    assert len(changes.snapshots) == 1
    assert changes.snapshots[0].before is None
    assert len(changes.track_points) == 1

    assert store.get_snapshot("rx-1", MMSI) is None
    assert store.get_track("rx-1", MMSI) == []
    assert store.list_position_conflicts() == []
    store.close()


def test_store_preview_rolls_back_on_failure(tmp_path):
    store = SnapshotStore(str(tmp_path / "preview.db"))
    observed = "2026-10-05T12:00:00.000000Z"
    message = decode_payload(SINGLE_PAYLOAD, 0).model_dump()

    good = ("rx-1", MMSI, KIND_POSITION, observed, message, 0)
    bad = ("rx-1", MMSI, KIND_POSITION, observed, {"oops": object()}, 1)
    with pytest.raises(TypeError):
        store.preview_ingest([good, bad])

    # the failed dry run persisted nothing, and the store stays usable
    assert store.get_snapshot("rx-1", MMSI) is None
    assert store.get_track("rx-1", MMSI) == []
    changes = store.preview_ingest([good])
    assert len(changes.snapshots) == 1
    assert store.get_snapshot("rx-1", MMSI) is None
    store.close()


def test_store_preview_matches_apply_ingest_outcome(tmp_path):
    """The previewed changes equal the state a real ingest then persists."""
    path = str(tmp_path / "preview.db")
    store = SnapshotStore(path)
    t1 = "2026-10-05T11:00:00.000000Z"
    t2 = "2026-10-05T12:00:00.000000Z"
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    far = decode_payload(FAR_SENTENCE.split(",")[5], 0).model_dump()

    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, t1, base, 0)])
    writes = [
        ("rx-2", MMSI, KIND_POSITION, t1, far, 0),  # new vessel + conflict
        ("rx-1", MMSI, KIND_POSITION, t2, base, 0),  # snapshot moves to T2
    ]
    changes = store.preview_ingest(writes)
    assert {(c.source_id, c.mmsi) for c in changes.snapshots} == {
        ("rx-1", MMSI),
        ("rx-2", MMSI),
    }
    assert len(changes.track_points) == 2
    assert len(changes.conflicts) == 1
    # the preview itself wrote nothing
    assert store.get_snapshot("rx-2", MMSI) is None
    assert store.list_position_conflicts() == []

    store.apply_ingest(writes)
    after = store.get_snapshot("rx-1", MMSI)
    previewed = next(c for c in changes.snapshots if c.source_id == "rx-1")
    assert after == previewed.after
    assert store.get_snapshot("rx-2", MMSI) == next(
        c for c in changes.snapshots if c.source_id == "rx-2"
    ).after
    rx2_point = next(
        p for p in changes.track_points if p.source_id == "rx-2"
    )
    assert store.get_track("rx-2", MMSI)[0].observed_at == rx2_point.observed_at
    assert store.list_position_conflicts() == changes.conflicts
    store.close()
