"""HTTP tests for the minimum-stock-cost inventory mode.

Reuses the in-memory FakeStore harness from test_inventory (the fake mirrors
the SQL check-and-set semantics: atomic under asyncio).
"""
import json

import pytest
from fastapi.testclient import TestClient

import app.db as db
import app.main as main

from test_inventory import FakePool, FakeStore


@pytest.fixture
def store(monkeypatch):
    st = FakeStore()

    async def fake_connect(dsn=None):
        return FakePool()

    pairs = [
        ("connect", fake_connect),
        ("get_frozen", st.get_frozen),
        ("freeze", st.freeze),
        ("create_batch", st.create_batch),
        ("get_batch", st.get_batch),
        ("get_batches", st.get_batches),
        ("list_batches", st.list_batches),
        ("get_reservation", st.get_reservation),
        ("reserve_and_freeze", st.reserve_and_freeze),
        ("transition_reservation", st.transition_reservation),
        ("register_step", st.register_step),
        ("complete_stepwise", st.complete_stepwise),
        ("abort_stepwise", st.abort_stepwise),
        ("recall_batch", st.recall_batch),
        ("get_recall", st.get_recall),
    ]
    for name, fn in pairs:
        monkeypatch.setattr(db, name, fn)
        monkeypatch.setattr(main, name, fn)
    main.app.state.db = FakePool()
    return st


@pytest.fixture
def client(store):
    with TestClient(main.app) as c:
        yield c


BODY = {
    "batch_ids": ["B-HI", "B-LO"],
    "diluent": {"available_nl": 10_000_000},
    "wells": [
        {"id": "A1", "final_volume_nl": 1000,
         "final_concentration": "125"},
        {"id": "A2", "final_volume_nl": 1000,
         "final_concentration": "75"},
        {"id": "A3", "final_volume_nl": 1000,
         "final_concentration": "0"},
    ],
    "containers": [],
    "pipette": {"min_nl": 25, "max_nl": 1000, "step_nl": 25},
    "default_well_capacity_nl": 5000,
}

OPT_BLOCK = {
    "mode": "minimum_stock_cost",
    "batch_costs": [
        {"batch_id": "B-HI", "unit_cost_nl": "100"},
        {"batch_id": "B-LO", "unit_cost_nl": "1/10"},
    ],
}


def _register(client):
    r = client.post("/batches", json={"id": "B-HI", "concentration": "200",
                                      "initial_volume_nl": 1_000_000_000})
    assert r.status_code in (200, 201)
    r = client.post("/batches", json={"id": "B-LO", "concentration": "50",
                                      "initial_volume_nl": 1_000_000_000})
    assert r.status_code in (200, 201)


def _body():
    b = json.loads(json.dumps(BODY))
    b["optimization"] = json.loads(json.dumps(OPT_BLOCK))
    return b


# --------------------------------------------------------------------------- #
# Success: whole-plan optimum + cost report + reservation
# --------------------------------------------------------------------------- #


def test_min_cost_plan_optimal_and_reports_cost(client):
    _register(client)
    r = client.post("/inventory/plans", json=_body(),
                    headers={"X-Request-ID": "mc-1"})
    assert r.status_code == 201, r.text
    body = r.json()
    # A1@125 blends with the cheap LO batch: 500 HI + 500 LO (vs 625 HI
    # direct); A2@75 uses 375 HI + diluent; A3 pure diluent.
    assert body["reservations"] == [
        {"batch_id": "B-HI", "reserved_nl": 875},
        {"batch_id": "B-LO", "reserved_nl": 500},
    ]
    opt = body["optimization"]
    assert opt["mode"] == "minimum_stock_cost"
    by_batch = {b["batch_id"]: b for b in opt["batches"]}
    assert by_batch["B-HI"]["used_nl"] == 875
    assert by_batch["B-HI"]["unit_cost_nl"] == "100"
    assert by_batch["B-HI"]["cost"] == "87500"
    assert by_batch["B-LO"]["used_nl"] == 500
    assert by_batch["B-LO"]["unit_cost_nl"] == "1/10"
    assert by_batch["B-LO"]["cost"] == "50"
    assert opt["total_cost"] == "87550"
    # reserved exactly the optimal draw
    hi = client.get("/batches/B-HI").json()
    lo = client.get("/batches/B-LO").json()
    assert hi["reserved_nl"] == 875
    assert lo["reserved_nl"] == 500


def test_min_cost_plan_replays_frozen_optimum(client):
    _register(client)
    r1 = client.post("/inventory/plans", json=_body(),
                     headers={"X-Request-ID": "mc-2"})
    r2 = client.post("/inventory/plans", json=_body(),
                     headers={"X-Request-ID": "mc-2"})
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json() == r1.json()
    assert client.get("/batches/B-HI").json()["reserved_nl"] == 875


def test_min_cost_get_carries_optimization_block(client):
    _register(client)
    client.post("/inventory/plans", json=_body(),
                headers={"X-Request-ID": "mc-3"})
    r = client.get("/inventory/plans/mc-3")
    assert r.status_code == 200
    assert r.json()["optimization"]["total_cost"] == "87550"
    r = client.get("/plans/mc-3")
    assert r.status_code == 200
    assert r.json()["optimization"]["total_cost"] == "87550"


def test_price_change_flips_optimum(client):
    _register(client)
    # LO expensive: every target avoids LO entirely.
    body = _body()
    body["optimization"]["batch_costs"] = [
        {"batch_id": "B-HI", "unit_cost_nl": "1"},
        {"batch_id": "B-LO", "unit_cost_nl": "1000"},
    ]
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-4"})
    assert r.status_code == 201, r.text
    assert r.json()["reservations"] == [
        {"batch_id": "B-HI", "reserved_nl": 1000}]


# --------------------------------------------------------------------------- #
# Cost coverage validation (400, nothing frozen or reserved)
# --------------------------------------------------------------------------- #


def test_missing_batch_cost_rejected(client):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"] = [
        {"batch_id": "B-HI", "unit_cost_nl": "100"}]
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-bad-missing"})
    assert r.status_code == 400
    assert "missing unit cost" in r.json()["detail"]
    assert client.get("/batches/B-HI").json()["reserved_nl"] == 0


def test_extra_batch_cost_rejected(client):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"].append(
        {"batch_id": "B-OTHER", "unit_cost_nl": "1"})
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-bad-extra"})
    assert r.status_code == 400
    assert "unexpected" in r.json()["detail"]


def test_duplicate_batch_cost_rejected(client):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"].append(body["optimization"
                                                    ]["batch_costs"][0])
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-bad-dup"})
    assert r.status_code == 400


@pytest.mark.parametrize("bad", [-1, 0.5, "nope"])
def test_illegal_cost_values_rejected(client, bad):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"][0]["unit_cost_nl"] = bad
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-bad-value"})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_request"


def test_zero_cost_is_allowed(client):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"] = [
        {"batch_id": "B-HI", "unit_cost_nl": 0},
        {"batch_id": "B-LO", "unit_cost_nl": "0"},
    ]
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-zero"})
    assert r.status_code == 201, r.text
    assert r.json()["optimization"]["total_cost"] == "0"


def test_empty_batch_costs_rejected(client):
    _register(client)
    body = _body()
    body["optimization"]["batch_costs"] = []
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-empty"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Retryable resource limit vs definitive infeasibility
# --------------------------------------------------------------------------- #


def test_search_limit_is_retryable_and_freezes_nothing(client):
    _register(client)
    body = _body()
    # Eight identical bracketed wells exhaust the smallest permitted node
    # budget (100) before the optimum is proved.
    body["wells"] = [
        {"id": f"A{i}", "final_volume_nl": 1000,
         "final_concentration": "125"} for i in range(8)]
    body["limits"] = {"max_search_nodes": 100}
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-limit"})
    assert r.status_code == 503
    err = r.json()
    assert err["error"] == "search_limit"
    assert err["retryable"] is True
    assert err["limits"]["max_search_nodes"] == 100
    # nothing frozen: the id replays once a larger budget is allowed
    assert client.get("/plans/mc-limit").status_code == 404
    assert client.get("/batches/B-HI").json()["reserved_nl"] == 0
    body["limits"] = {"max_search_nodes": 200_000}
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-limit"})
    assert r.status_code == 201, r.text


def test_definitive_infeasibility_still_422_and_frozen(client):
    _register(client)
    body = _body()
    body["batch_ids"] = ["B-LO"]  # conc 50 cannot reach 125
    body["optimization"]["batch_costs"] = [
        {"batch_id": "B-LO", "unit_cost_nl": "1"}]
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-infeasible"})
    assert r.status_code == 422
    r2 = client.post("/inventory/plans", json=body,
                     headers={"X-Request-ID": "mc-infeasible"})
    assert r2.status_code == 422  # frozen negative result


# --------------------------------------------------------------------------- #
# Lifecycle + backward compatibility
# --------------------------------------------------------------------------- #


def test_min_cost_confirm_cancel_lifecycle(client):
    _register(client)
    client.post("/inventory/plans", json=_body(),
                headers={"X-Request-ID": "mc-life"})
    r = client.post("/inventory/plans/mc-life/confirm")
    assert r.status_code == 200
    hi = client.get("/batches/B-HI").json()
    assert (hi["reserved_nl"], hi["used_nl"]) == (0, 875)


def test_min_cost_works_with_stepwise_mode(client):
    _register(client)
    body = _body()
    body["execution_mode"] = "stepwise"
    r = client.post("/inventory/plans", json=body,
                    headers={"X-Request-ID": "mc-step"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["execution_mode"] == "stepwise"
    assert out["reservation_status"] == "stepwise"
    assert out["optimization"]["total_cost"] == "87550"
    assert out["progress"]["total_steps"] == len(out["plan"]["transfers"])


def test_request_without_optimization_keeps_old_behavior(client):
    _register(client)
    plain = {k: v for k, v in _body().items() if k != "optimization"}
    r = client.post("/inventory/plans", json=plain,
                    headers={"X-Request-ID": "mc-plain"})
    assert r.status_code == 201, r.text
    assert "optimization" not in r.json()


def test_adding_optimization_to_frozen_id_conflicts(client):
    _register(client)
    plain = {k: v for k, v in _body().items() if k != "optimization"}
    client.post("/inventory/plans", json=plain,
                headers={"X-Request-ID": "mc-conf"})
    r = client.post("/inventory/plans", json=_body(),
                    headers={"X-Request-ID": "mc-conf"})
    assert r.status_code == 409
    assert r.json()["error"] == "idempotency_conflict"
