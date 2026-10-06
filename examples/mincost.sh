#!/usr/bin/env bash
# Min-cost objective demo against a locally running service
# (docker compose up).
#
# Registers two batches of the same concentration with different unit
# costs, then plans with objective="min_cost": the solver compares the
# total stock cost of the whole plan (never per-well greedy) and returns
# the executable transfers, the per-batch usage/cost and the exact total
# cost. Also shows cost validation (400), the retryable resource-limit
# answer (503, nothing frozen or reserved) and the unchanged idempotency
# and confirm/cancel conventions.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
RID="mc-demo-001"

echo "== register two batches (same concentration, different unit costs) =="
curl -sS -X POST "$BASE/batches" \
  -H "Content-Type: application/json" \
  -d '{"id": "S1", "concentration": "1000", "initial_volume_nl": 1000000}' \
  | python3 -m json.tool
curl -sS -X POST "$BASE/batches" \
  -H "Content-Type: application/json" \
  -d '{"id": "S2", "concentration": "1000", "initial_volume_nl": 1000000}' \
  | python3 -m json.tool

echo "== min-cost plan: costs cover every candidate batch =="
curl -sS -X POST "$BASE/inventory/plans" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $RID" \
  --data @examples/inventory_mincost.json | python3 -m json.tool

echo "== identical retry -> 200, original cost block, no double reservation =="
curl -sS -o /dev/null -w "%{http_code}\n" -X POST "$BASE/inventory/plans" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $RID" \
  --data @examples/inventory_mincost.json

echo "== same X-Request-ID, different costs -> 409 =="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = json.load(open("examples/inventory_mincost.json"))
body["batch_costs"] = {"S1": "1", "S2": "1"}
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mc-demo-001"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code)
    print(e.read().decode())
PY

echo "== cost table must cover every candidate batch -> 400 =="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = json.load(open("examples/inventory_mincost.json"))
body["batch_costs"] = {"S1": "5"}          # S2 missing
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mc-demo-002"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code)
    print(e.read().decode())
PY

echo "== search budget too small to prove optimality -> 503, retryable =="
python3 - <<'PY'
import json, urllib.request, urllib.error

def reserved():
    out = []
    for bid in ("S1", "S2"):
        with urllib.request.urlopen(f"http://localhost:8000/batches/{bid}") as r:
            out.append(json.loads(r.read())["reserved_nl"])
    return out

body = json.load(open("examples/inventory_mincost.json"))
body["wells"] = [
    {"id": f"W{i}", "final_volume_nl": 1000, "final_concentration": str(c)}
    for i, c in enumerate([400, 100, 500, 250, 600, 300])
]
body["containers"] = [{"id": f"tube-{i}", "capacity_nl": 20000}
                      for i in range(3)]
body["limits"] = {"max_search_nodes": 100}
before = reserved()
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mc-demo-003"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code)
    print(e.read().decode())
print("reserved S1/S2 unchanged by the 503:", before, "->", reserved())

print("== nothing frozen: the same request id retries with a bigger budget ==")
body["limits"] = {"max_search_nodes": 200000}
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mc-demo-003"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print("status:", resp.status)
    plan = json.loads(resp.read())
    print("total_cost:", plan["cost"]["total_cost"])
    print("batches:", json.dumps(plan["cost"]["batches"]))
PY

echo "== batch balances: only the cheap batch was reserved =="
curl -sS "$BASE/batches/S1" | python3 -m json.tool
curl -sS "$BASE/batches/S2" | python3 -m json.tool

echo "== confirm the reservation (reserved -> used), unchanged lifecycle =="
curl -sS -X POST "$BASE/inventory/plans/$RID/confirm" | python3 -m json.tool
