#!/usr/bin/env python3
"""Minimum-stock-cost inventory-mode demo using only the standard library.

Registers two stock batches of different concentration and per-nanoliter
cost, submits an inventory plan in minimum_stock_cost mode (an exact
non-negative rational unit cost for *every* candidate batch), prints the
frozen optimum (executable transfers, per-batch usage/cost, whole-plan
total cost) and walks the validation/retry rules.

Usage:
    python examples/min_cost_demo.py [base_url]
Default base URL: http://localhost:8000
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

with open("examples/inventory_min_cost.json", encoding="utf-8") as fh:
    PLAN_BODY = json.load(fh)

RID = "mincost-demo-py-001"

BATCHES = [
    {"id": "B-HI", "concentration": "200",
     "initial_volume_nl": 1_000_000_000},
    {"id": "B-LO", "concentration": "50",
     "initial_volume_nl": 1_000_000_000},
]


def call(path, body=None, request_id=None, method=None):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id
    data = json.dumps(body).encode() if body is not None else None
    if method is None:
        method = "POST" if body is not None or "/confirm" in path \
            or "/cancel" in path else "GET"
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


print("1) register batches:")
for batch in BATCHES:
    status, out = call("/batches", batch)
    print("  ", batch["id"], "HTTP", status)

print("2) minimum-stock-cost plan (whole-plan optimum, reserved atomically):")
status, plan = call("/inventory/plans", PLAN_BODY, request_id=RID)
print("   HTTP", status)
opt = plan["optimization"]
print("   mode:", opt["mode"], " total_cost:", opt["total_cost"])
for row in opt["batches"]:
    print(f"   {row['batch_id']:>5}: used {row['used_nl']:>5} nl "
          f"x {row['unit_cost_nl']:>5}/nl = {row['cost']}")
for t in plan["plan"]["transfers"]:
    src = t["source"].get("id") or t["source"]["kind"]
    dst = t["destination"].get("id") or t["destination"]["kind"]
    print(f"   {t['step']:2d}. {src:>5} -> {dst:<4} {t['volume_nl']:>5} nl")
assert plan["reservations"] == [
    {"batch_id": "B-HI", "reserved_nl": 875},
    {"batch_id": "B-LO", "reserved_nl": 500},
]
assert opt["total_cost"] == "87550"

print("3) identical retry replays the frozen optimum (200, no double reserve):")
status, again = call("/inventory/plans", PLAN_BODY, request_id=RID)
print("   HTTP", status, "same payload:", again == plan)

print("4) same id without optimization block -> 409 (different content):")
body = json.loads(json.dumps(PLAN_BODY))
body.pop("optimization")
status, err = call("/inventory/plans", body, request_id=RID)
print("   HTTP", status, err["error"])

print("5) missing unit cost for a candidate batch -> 400:")
body = json.loads(json.dumps(PLAN_BODY))
body["optimization"]["batch_costs"] = [
    {"batch_id": "B-HI", "unit_cost_nl": "100"}]
status, err = call("/inventory/plans", body,
                   request_id="mincost-py-bad-missing")
print("   HTTP", status, err["error"])

print("6) cost for a non-candidate batch -> 400:")
body = json.loads(json.dumps(PLAN_BODY))
body["optimization"]["batch_costs"].append(
    {"batch_id": "B-OTHER", "unit_cost_nl": "1"})
status, err = call("/inventory/plans", body,
                   request_id="mincost-py-bad-extra")
print("   HTTP", status, err["error"])

print("7) negative / float costs -> 400:")
for bad in ("-1", 0.5):
    body = json.loads(json.dumps(PLAN_BODY))
    body["optimization"]["batch_costs"][0]["unit_cost_nl"] = bad
    status, err = call("/inventory/plans", body,
                       request_id="mincost-py-bad-value")
    print(f"    {bad!r:>5} -> HTTP", status, err["error"])

print("8) duplicate cost entry -> 400:")
body = json.loads(json.dumps(PLAN_BODY))
body["optimization"]["batch_costs"].append(body["optimization"]["batch_costs"][0])
status, err = call("/inventory/plans", body,
                   request_id="mincost-py-bad-dup")
print("   HTTP", status, err["error"])

print("9) requests without the optimization block keep the old behavior:")
plain = json.loads(json.dumps(PLAN_BODY))
plain.pop("optimization")
status, out = call("/inventory/plans", plain, request_id="mincost-py-plain")
print("   HTTP", status, "has optimization key:", "optimization" in out)

print("10) confirm the frozen optimum:")
status, res = call(f"/inventory/plans/{RID}/confirm")
print("   HTTP", status, res["status"], "changed:", res["changed"])

print("11) final balances:")
for batch in BATCHES:
    status, out = call(f"/batches/{batch['id']}")
    print("   ", batch["id"], {k: out[k] for k in
                               ("reserved_nl", "used_nl", "available_nl")})
