"""Min-cost objective tests: whole-plan stock-cost optimization.

Solver-level tests exercise ``solve(prob, stock_costs=...)`` directly; HTTP
tests run the inventory endpoint against the in-memory store (same pattern
as tests.test_inventory / tests.test_reviews).

Rules under test:

* the request selects ``objective: "min_cost"`` and gives every candidate
  batch a non-negative rational price per nanoliter; illegal, missing or
  superfluous costs are rejected with 400;
* the search compares the **total** stock cost of the whole plan (wells
  share intermediates), never a per-well greedy choice;
* a feasible answer lists the executable transfers, the per-batch usage
  and cost and the exact rational total cost;
* when the search node budget is exhausted before optimality is proven,
  the answer is a retryable 503 resource-limit error: nothing is frozen,
  nothing is reserved, and the same X-Request-ID stays usable;
* genuine infeasibility keeps the existing frozen 422 semantics, and all
  freezing / retry / confirm-cancel / stepwise conventions are unchanged.
"""
import json
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

import app.db as db
import app.main as main
from app.solver import Problem, Well, solve
from app.solver.core import make_pip
from app.solver.solve import verify_plan
from tests.test_inventory import FakePool, FakeStore


# --------------------------------------------------------------------------- #
# Solver level
# --------------------------------------------------------------------------- #


def _prob(wells, stocks, containers=(20_000, 10_000), pmin=500, pmax=1000,
          step=100, limit=200_000):
    return Problem(
        stocks={sid: {"conc": Fraction(c), "volume": v}
                for sid, c, v in stocks},
        diluent_volume=10_000_000,
        wells=[Well(wid, vol, Fraction(c), cap)
               for wid, vol, c, cap in wells],
        containers=list(containers),
        pip_min=pmin, pip_max=pmax, pip_step=step,
        max_search_nodes=limit,
    )


def _usage(plan):
    usage = {}
    for t in plan["transfers"]:
        src = t["source"]
        if src["kind"] == "stock":
            usage[src["id"]] = usage.get(src["id"], 0) + t["volume_nl"]
    return usage


def test_solver_mincost_prefers_cheaper_batch():
    prob = _prob([("A1", 1000, 400, 2000), ("A2", 1000, 100, 2000)],
                 (("S1", 1000, 10 ** 6), ("S2", 1000, 10 ** 6)))
    costs = {"S1": Fraction(5), "S2": Fraction(1, 2)}
    res = solve(prob, stock_costs=costs)
    assert res.ok, res.failure
    pip = make_pip(prob.pip_min, prob.pip_max, prob.pip_step)
    assert verify_plan(prob, pip, res.plan) is None
    usage = _usage(res.plan)
    # every draw comes from the cheap batch, and one intermediate batch is
    # shared by both wells (600 nl total, not 1100 nl of separate draws)
    assert set(usage) == {"S2"}
    assert usage["S2"] == 600


def test_solver_mincost_is_global_not_per_well_greedy():
    # S2 matches the target concentration directly but is expensive; S1 is
    # cheap but needs an intermediate whose pipette minimum forces a
    # 1000 nl batch. The plan-level optimum shares one intermediate
    # (500 nl of S1, cost 500) instead of two direct fills (1000 nl of
    # S2, cost 3000).
    prob = _prob([("W1", 500, 50, 2000), ("W2", 500, 50, 2000)],
                 (("S1", 100, 10 ** 6), ("S2", 50, 10 ** 6)),
                 containers=(2000,))
    costs = {"S1": Fraction(1), "S2": Fraction(3)}
    res = solve(prob, stock_costs=costs)
    assert res.ok, res.failure
    assert _usage(res.plan) == {"S1": 500}
    assert len(res.plan["intermediates"]) == 1
    for t in res.plan["transfers"]:
        if t["destination"]["kind"] == "well":
            assert t["source"]["kind"] == "intermediate"
    # the default first-feasible search finds a different (pricier) plan
    other = solve(_prob([("W1", 500, 50, 2000), ("W2", 500, 50, 2000)],
                        (("S1", 100, 10 ** 6), ("S2", 50, 10 ** 6)),
                        containers=(2000,)))
    assert other.ok
    assert sum(costs[s] * n for s, n in _usage(other.plan).items()) > 500


def test_solver_mincost_search_limit_is_not_infeasible():
    prob = _prob([(f"W{i}", 1000, c, 2000)
                  for i, c in enumerate([400, 100, 500, 250, 600, 300])],
                 (("S1", 1000, 10 ** 6), ("S2", 1000, 10 ** 6)),
                 containers=(20_000, 20_000, 20_000), limit=100)
    costs = {"S1": Fraction(5), "S2": Fraction(1)}
    res = solve(prob, stock_costs=costs)
    assert not res.ok
    assert res.failure.reason == "search_limit"
    # the same problem with an adequate budget is feasible
    prob = _prob([(f"W{i}", 1000, c, 2000)
                  for i, c in enumerate([400, 100, 500, 250, 600, 300])],
                 (("S1", 1000, 10 ** 6), ("S2", 1000, 10 ** 6)),
                 containers=(20_000, 20_000, 20_000))
    res = solve(prob, stock_costs=costs)
    assert res.ok, res.failure
    assert set(_usage(res.plan)) == {"S2"}


def test_solver_mincost_infeasible_keeps_existing_semantics():
    prob = _prob([("A1", 1000, 2000, 5000)],
                 (("S1", 1000, 10 ** 6), ("S2", 1000, 10 ** 6)))
    costs = {"S1": Fraction(1), "S2": Fraction(1)}
    res = solve(prob, stock_costs=costs)
    assert not res.ok
    assert res.failure.reason == "no_feasible_plan"
    assert "target_above_max_stock_concentration" in res.failure.detail["global"]


def test_solver_mincost_zero_costs():
    prob = _prob([("A1", 1000, 400, 2000)],
                 (("S1", 1000, 10 ** 6), ("S2", 1000, 10 ** 6)))
    res = solve(prob, stock_costs={"S1": Fraction(0), "S2": Fraction(0)})
    assert res.ok, res.failure


# --------------------------------------------------------------------------- #
# HTTP level (in-memory store)
# --------------------------------------------------------------------------- #


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


BATCHES = [
    {"id": "S1", "concentration": "1000", "initial_volume_nl": 1_000_000},
    {"id": "S2", "concentration": "1000", "initial_volume_nl": 1_000_000},
]

MINCOST_BODY = {
    "batch_ids": ["S1", "S2"],
    "objective": "min_cost",
    "batch_costs": {"S1": "5", "S2": "1/2"},
    "diluent": {"available_nl": 10_000_000},
    "wells": [
        {"id": "A1", "final_volume_nl": 1000, "final_concentration": "400"},
        {"id": "A2", "final_volume_nl": 1000, "final_concentration": "100"},
    ],
    "containers": [
        {"id": "tube-1", "capacity_nl": 10_000},
        {"id": "tube-2", "capacity_nl": 20_000},
    ],
    "pipette": {"min_nl": 500, "max_nl": 1000, "step_nl": 100},
    "default_well_capacity_nl": 2000,
}


def _body(**patch):
    body = json.loads(json.dumps(MINCOST_BODY))
    body.update(patch)
    return body


def _register_batches(client):
    for b in BATCHES:
        assert client.post("/batches", json=b).status_code == 201


def _create(client, rid="mc-1", body=None):
    return client.post("/inventory/plans", json=body or _body(),
                       headers={"X-Request-ID": rid})


# --------------------------------------------------------------------------- #
# Feasible min-cost plans
# --------------------------------------------------------------------------- #


def test_mincost_plan_lists_usage_costs_and_total(client):
    _register_batches(client)
    r = _create(client)
    assert r.status_code == 201, r.text
    body = r.json()
    cost = body["cost"]
    assert cost["objective"] == "min_cost"
    # the whole plan draws only from the cheap batch: 600 nl at 1/2 per nl
    assert cost["batches"] == [
        {"batch_id": "S2", "used_nl": 600, "cost_per_nl": "1/2",
         "cost": "300"},
    ]
    assert cost["total_cost"] == "300"
    # reservations mirror the same per-batch usage
    assert body["reservations"] == [{"batch_id": "S2", "reserved_nl": 600}]
    # executable transfers are present and draw only from S2
    transfers = body["plan"]["transfers"]
    assert transfers and all(
        t["source"].get("id") != "S1" for t in transfers
        if t["source"]["kind"] == "stock")
    # the expensive batch was not reserved at all
    assert client.get("/batches/S1").json()["reserved_nl"] == 0
    assert client.get("/batches/S2").json()["reserved_nl"] == 600


def test_mincost_exact_rational_total(client):
    for b in ({"id": "L1", "concentration": "400",
               "initial_volume_nl": 1_000_000},
              {"id": "L2", "concentration": "400",
               "initial_volume_nl": 1_000_000}):
        assert client.post("/batches", json=b).status_code == 201
    body = _body(batch_ids=["L1", "L2"],
                 batch_costs={"L1": "1/3", "L2": "5"},
                 wells=[{"id": "A1", "final_volume_nl": 1000,
                         "final_concentration": "400"}],
                 containers=[])
    r = _create(client, body=body)
    assert r.status_code == 201, r.text
    cost = r.json()["cost"]
    # 1000 nl drawn directly from the cheap batch at 1/3 per nl: the total
    # is the exact rational 1000/3, never a rounded float
    assert cost["batches"] == [
        {"batch_id": "L1", "used_nl": 1000, "cost_per_nl": "1/3",
         "cost": "1000/3"},
    ]
    assert cost["total_cost"] == "1000/3"


def test_mincost_zero_cost_is_legal(client):
    _register_batches(client)
    body = _body(batch_costs={"S1": "0", "S2": 0})
    r = _create(client, body=body)
    assert r.status_code == 201, r.text
    assert r.json()["cost"]["total_cost"] == "0"


def test_default_objective_has_no_cost_block(client):
    _register_batches(client)
    body = _body()
    del body["objective"], body["batch_costs"]
    r = _create(client, body=body)
    assert r.status_code == 201, r.text
    assert "cost" not in r.json()


def test_explicit_default_objective_hashes_like_absent(client):
    # "objective": "feasible" (and no costs) normalizes away, so a request
    # id frozen without the new fields replays identically.
    _register_batches(client)
    plain = _body()
    del plain["objective"], plain["batch_costs"]
    r = _create(client, rid="mc-norm", body=plain)
    assert r.status_code == 201, r.text
    explicit = _body(objective="feasible")
    del explicit["batch_costs"]
    r = _create(client, rid="mc-norm", body=explicit)
    assert r.status_code == 200, r.text
    assert "cost" not in r.json()


# --------------------------------------------------------------------------- #
# Cost validation (400)
# --------------------------------------------------------------------------- #


def test_mincost_costs_must_cover_every_candidate(client):
    _register_batches(client)
    r = _create(client, body=_body(batch_costs={"S1": "1"}))
    assert r.status_code == 400
    assert "S2" in r.json()["detail"]


def test_mincost_costs_reject_non_candidate_batches(client):
    _register_batches(client)
    body = _body(batch_costs={"S1": "1", "S2": "1", "S9": "1"})
    r = _create(client, body=body)
    assert r.status_code == 400
    assert "S9" in r.json()["detail"]


def test_mincost_requires_batch_costs(client):
    _register_batches(client)
    body = _body()
    del body["batch_costs"]
    r = _create(client, body=body)
    assert r.status_code == 400


def test_batch_costs_require_mincost_objective(client):
    _register_batches(client)
    body = _body()
    del body["objective"]
    r = _create(client, body=body)
    assert r.status_code == 400


@pytest.mark.parametrize("bad_cost", ["-1", "-1/3", 0.5, "abc", ""])
def test_mincost_rejects_illegal_cost_values(client, bad_cost):
    _register_batches(client)
    body = _body(batch_costs={"S1": "1", "S2": bad_cost})
    r = _create(client, body=body)
    assert r.status_code == 400


def test_mincost_validation_failure_freezes_nothing(client):
    _register_batches(client)
    r = _create(client, rid="mc-bad", body=_body(batch_costs={"S1": "1"}))
    assert r.status_code == 400
    assert client.get("/inventory/plans/mc-bad").status_code == 404
    assert client.get("/batches/S1").json()["reserved_nl"] == 0


# --------------------------------------------------------------------------- #
# Retryable resource limit (503) vs frozen infeasibility (422)
# --------------------------------------------------------------------------- #


def _big_body(limit):
    body = _body()
    body["wells"] = [
        {"id": f"W{i}", "final_volume_nl": 1000,
         "final_concentration": str(c)}
        for i, c in enumerate([400, 100, 500, 250, 600, 300])
    ]
    body["containers"] = [{"id": f"tube-{i}", "capacity_nl": 20_000}
                          for i in range(3)]
    body["limits"] = {"max_search_nodes": limit}
    return body


def test_mincost_search_limit_is_retryable_and_unfrozen(client):
    _register_batches(client)
    r = _create(client, rid="mc-lim", body=_big_body(100))
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["error"] == "resource_limit"
    assert body["reason"] == "search_limit"
    assert body["retryable"] is True
    # nothing was frozen and nothing was reserved
    assert client.get("/inventory/plans/mc-lim").status_code == 404
    assert client.get("/batches/S1").json()["reserved_nl"] == 0
    assert client.get("/batches/S2").json()["reserved_nl"] == 0
    # the same request id stays usable: retry with an adequate budget
    r = _create(client, rid="mc-lim", body=_big_body(200_000))
    assert r.status_code == 201, r.text
    cost = r.json()["cost"]
    assert cost["batches"] == [
        {"batch_id": "S2", "used_nl": 2200, "cost_per_nl": "1/2",
         "cost": "1100"},
    ]
    assert cost["total_cost"] == "1100"
    assert client.get("/batches/S2").json()["reserved_nl"] == 2200


def test_mincost_infeasible_is_frozen_422(client):
    _register_batches(client)
    body = _body()
    body["wells"] = [{"id": "A1", "final_volume_nl": 1000,
                      "final_concentration": "2000"}]
    r = _create(client, rid="mc-inf", body=body)
    assert r.status_code == 422, r.text
    assert r.json()["status"] == "infeasible"
    # frozen: the same request id replays the 422
    r = _create(client, rid="mc-inf", body=body)
    assert r.status_code == 422
    assert client.get("/inventory/plans/mc-inf").status_code == 422
    assert client.get("/batches/S1").json()["reserved_nl"] == 0


# --------------------------------------------------------------------------- #
# Idempotency and lifecycle conventions
# --------------------------------------------------------------------------- #


def test_mincost_replay_and_conflict(client):
    _register_batches(client)
    r = _create(client)
    assert r.status_code == 201
    again = _create(client)
    assert again.status_code == 200
    assert again.json()["cost"] == r.json()["cost"]
    assert client.get("/batches/S2").json()["reserved_nl"] == 600
    conflict = _create(client, body=_body(batch_costs={"S1": "1",
                                                       "S2": "1"}))
    assert conflict.status_code == 409


def test_mincost_confirm_cancel_and_live_read(client):
    _register_batches(client)
    assert _create(client).status_code == 201
    r = client.post("/inventory/plans/mc-1/confirm")
    assert r.status_code == 200
    assert client.get("/batches/S2").json()["used_nl"] == 600
    # the frozen cost block rides along on the live read
    body = client.get("/inventory/plans/mc-1").json()
    assert body["cost"]["total_cost"] == "300"
    assert body["reservation_status"] == "confirmed"
    # cancel after confirm keeps the existing conflict semantics
    assert client.post("/inventory/plans/mc-1/cancel").status_code == 409


def test_mincost_combines_with_stepwise_mode(client):
    _register_batches(client)
    body = _body(execution_mode="stepwise")
    r = _create(client, body=body)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["execution_mode"] == "stepwise"
    assert created["cost"]["total_cost"] == "300"
    assert created["progress"]["total_steps"] == len(
        created["plan"]["transfers"])
    for t in created["plan"]["transfers"]:
        reg = client.post("/inventory/plans/mc-1/steps", json={
            "step": t["step"], "source": t["source"],
            "destination": t["destination"], "volume_nl": t["volume_nl"]})
        assert reg.status_code == 201, reg.text
    done = client.post("/inventory/plans/mc-1/complete")
    assert done.status_code == 201, done.text
    assert client.get("/batches/S2").json()["used_nl"] == 600
