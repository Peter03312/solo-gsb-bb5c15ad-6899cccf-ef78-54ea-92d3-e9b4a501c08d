#!/usr/bin/env python3
"""Min-cost objective demo using only the standard library.

Registers two batches of the same concentration with different unit
costs, then plans with ``objective: "min_cost"``: the solver compares the
total stock cost of the whole plan (never per-well greedy) and returns
the executable transfers, the per-batch usage/cost and the exact rational
total cost. Also demonstrates cost validation (400), the retryable
resource-limit answer (503 -- nothing frozen, nothing reserved, the same
request id stays usable) and the unchanged idempotency and confirm/cancel
conventions.

Usage:
    python examples/mincost_demo.py [base_url]
Default base URL: http://localhost:8000
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

with open("examples/inventory_mincost.json", encoding="utf-8") as fh:
    PLAN_BODY = json.load(fh)

BATCHES = [
    {"id": "S1", "concentration": "1000", "initial_volume_nl": 1_000_000},
    {"id": "S2", "concentration": "1000", "initial_volume_nl": 1_000_000},
]
RID = "mc-demo-001"


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


print("1) register two batches (same concentration, different unit costs):")
for b in BATCHES:
    status, batch = call("/batches", b)
    print("   HTTP", status, batch["id"], batch["concentration"])

print("2) min-cost plan (costs cover every candidate batch):")
status, plan = call("/inventory/plans", PLAN_BODY, request_id=RID)
print("   HTTP", status)
print("   reservations:", plan["reservations"])
print("   cost:", json.dumps(plan["cost"]))
for t in plan["plan"]["transfers"]:
    src = t["source"]
    label = src.get("id") or src["kind"]
    print(f"   {t['step']:2d}. {label:>12} -> "
          f"{t['destination']['kind']:<13} {t['volume_nl']:>5} nl")

print("3) identical retry returns the original result (no re-reservation):")
status, again = call("/inventory/plans", PLAN_BODY, request_id=RID)
print("   HTTP", status, "same payload:", again == plan)

print("4) same request id with different costs -> 409:")
changed = dict(PLAN_BODY, batch_costs={"S1": "1", "S2": "1"})
status, res = call("/inventory/plans", changed, request_id=RID)
print("   HTTP", status, res["error"])

print("5) cost table must cover every candidate batch -> 400:")
incomplete = dict(PLAN_BODY, batch_costs={"S1": "5"})
status, res = call("/inventory/plans", incomplete, request_id="mc-demo-002")
print("   HTTP", status, res["detail"])

print("6) search budget too small to prove optimality -> 503, retryable:")
big = dict(PLAN_BODY)
big["wells"] = [
    {"id": f"W{i}", "final_volume_nl": 1000, "final_concentration": str(c)}
    for i, c in enumerate([400, 100, 500, 250, 600, 300])
]
big["containers"] = [{"id": f"tube-{i}", "capacity_nl": 20_000}
                     for i in range(3)]
big["limits"] = {"max_search_nodes": 100}
reserved_before = [call(f"/batches/{b}")[1]["reserved_nl"] for b in ("S1", "S2")]
status, res = call("/inventory/plans", big, request_id="mc-demo-003")
print("   HTTP", status, res["error"], "retryable:", res["retryable"])
status, res = call("/inventory/plans/mc-demo-003")
print("   GET after 503 -> HTTP", status, "(nothing frozen)")
reserved_after = [call(f"/batches/{b}")[1]["reserved_nl"] for b in ("S1", "S2")]
print("   reserved S1/S2 unchanged by the 503:",
      reserved_before, "->", reserved_after)

print("7) same request id retried with an adequate budget -> 201:")
big["limits"] = {"max_search_nodes": 200_000}
status, plan = call("/inventory/plans", big, request_id="mc-demo-003")
print("   HTTP", status, "total_cost:", plan["cost"]["total_cost"])
print("   batches:", json.dumps(plan["cost"]["batches"]))

print("8) batch balances: only the cheap batch was reserved:")
for bid in ("S1", "S2"):
    status, batch = call(f"/batches/{bid}")
    print("  ", bid, {k: batch[k] for k in
                       ("reserved_nl", "used_nl", "available_nl")})

print("9) confirm the reservation (reserved -> used), unchanged lifecycle:")
status, res = call(f"/inventory/plans/{RID}/confirm")
print("   HTTP", status, res["status"], "changed:", res["changed"])
