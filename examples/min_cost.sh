#!/usr/bin/env bash
# Minimum-stock-cost inventory-mode demo against a locally running service
# (docker compose up).
#
# Registers two batches of different concentration and per-nanoliter cost,
# then submits an inventory plan with optimization.mode =
# "minimum_stock_cost": every candidate batch carries an exact non-negative
# rational unit cost, and the solver compares *whole plans* and freezes the
# cheapest feasible one (never a per-well greedy choice). The demo also
# shows: same request id + same body replays the frozen optimum (200),
# missing/extra/negative costs are rejected (400), and requests without the
# optimization block keep the original behavior.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
RID="mincost-demo-001"

echo "== register two batches: B-HI (200, expensive) and B-LO (50, cheap) =="
curl -sS -X POST "$BASE/batches" \
  -H "Content-Type: application/json" \
  -d '{"id": "B-HI", "concentration": "200", "initial_volume_nl": 1000000000}' \
  | python3 -m json.tool
curl -sS -X POST "$BASE/batches" \
  -H "Content-Type: application/json" \
  -d '{"id": "B-LO", "concentration": "50", "initial_volume_nl": 1000000000}' \
  | python3 -m json.tool

echo "== minimum-cost plan (whole-plan optimum, atomically reserved) =="
curl -sS -X POST "$BASE/inventory/plans" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $RID" \
  --data @examples/inventory_min_cost.json | python3 -m json.tool

echo "== identical retry -> 200: the same frozen optimum, no double reserve =="
curl -sS -o /dev/null -w "%{http_code}\n" -X POST "$BASE/inventory/plans" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $RID" \
  --data @examples/inventory_min_cost.json

echo "== missing unit cost for one candidate batch -> 400 =="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = json.load(open("examples/inventory_min_cost.json"))
body["optimization"]["batch_costs"] = [
    {"batch_id": "B-HI", "unit_cost_nl": "100"}]
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mincost-demo-bad-missing"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code)
    print(e.read().decode())
PY

echo "== cost entry for a non-candidate batch -> 400 =="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = json.load(open("examples/inventory_min_cost.json"))
body["optimization"]["batch_costs"].append(
    {"batch_id": "B-OTHER", "unit_cost_nl": "1"})
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mincost-demo-bad-extra"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code)
    print(e.read().decode())
PY

echo "== negative / float / non-rational cost -> 400 =="
for bad in '-1' '0.5' 'not-a-number'; do
  python3 - "$bad" <<'PY'
import json, sys, urllib.request, urllib.error
bad = sys.argv[1]
body = json.load(open("examples/inventory_min_cost.json"))
body["optimization"]["batch_costs"][0]["unit_cost_nl"] = bad
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mincost-demo-bad-value"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print(f"cost {bad!r} -> status:", e.code)
PY
done

echo "== same request id without optimization block still replays? no: it is =="
echo "== different content for the frozen id -> 409 (old behavior preserved) =="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = json.load(open("examples/inventory_min_cost.json"))
body.pop("optimization")
req = urllib.request.Request(
    "http://localhost:8000/inventory/plans",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "X-Request-ID": "mincost-demo-001"},
    method="POST",
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("status:", e.code, e.read().decode())
PY

echo "== confirm the frozen minimum-cost plan =="
curl -sS -X POST "$BASE/inventory/plans/$RID/confirm" | python3 -m json.tool

echo "== batch balances reflect exactly the optimal draw =="
curl -sS "$BASE/batches/B-HI" | python3 -m json.tool
curl -sS "$BASE/batches/B-LO" | python3 -m json.tool
