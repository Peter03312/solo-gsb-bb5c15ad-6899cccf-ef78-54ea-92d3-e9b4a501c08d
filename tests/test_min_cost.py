"""Minimum-stock-cost solver tests (no network / database required)."""
from fractions import Fraction

from app.solver import Problem, Well, solve
from app.solver.core import make_pip
from app.solver.solve import verify_plan


def cost_problem(wells, costs, stocks=(("HI", 200), ("LO", 50)),
                 volumes=None, containers=(), pmin=25, pmax=1000, step=25,
                 diluent=10 ** 9, max_search_nodes=200_000):
    volumes = volumes or {sid: 10 ** 9 for sid, _ in stocks}
    return Problem(
        stocks={sid: {"conc": Fraction(c), "volume": volumes[sid]}
                for sid, c in stocks},
        diluent_volume=diluent,
        wells=[Well(wid, vol, Fraction(c), cap)
               for wid, vol, c, cap in wells],
        containers=list(containers),
        pip_min=pmin, pip_max=pmax, pip_step=step,
        max_search_nodes=max_search_nodes,
        stock_costs={sid: Fraction(v) for sid, v in costs.items()},
    )


def stock_draws(plan):
    draws = {}
    for t in plan["transfers"]:
        src = t["source"]
        if src["kind"] == "stock":
            draws[src["id"]] = draws.get(src["id"], 0) + t["volume_nl"]
    return draws


def assert_optimal(p):
    res = solve(p)
    assert res.ok, res.failure
    pip = make_pip(p.pip_min, p.pip_max, p.pip_step)
    assert verify_plan(p, pip, res.plan) is None
    return res.plan


def total_cost(p, draws):
    return sum(p.stock_costs[sid] * nl for sid, nl in draws.items())


def test_picks_blend_when_high_stock_expensive():
    # target 125 in 1000:
    #   direct 200 + diluent -> 625 HI
    #   blend 200/50         -> 500 HI + 500 LO (concentration exactly 125)
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": 100, "LO": "1/10"})
    plan = assert_optimal(p)
    assert stock_draws(plan) == {"HI": 500, "LO": 500}


def test_picks_direct_dilution_when_low_stock_expensive():
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": 1, "LO": 100})
    plan = assert_optimal(p)
    assert stock_draws(plan) == {"HI": 625}


def test_whole_plan_not_per_well_greedy():
    # Two wells at 125. With HI expensive the per-well and whole-plan
    # optima coincide for the direct-vs-blend choice; this case additionally
    # forces the search to compare across wells: a zero-cost LO batch makes
    # the blend strictly better for every well, and the optimizer must
    # choose it for both rather than stopping at the first feasible plan.
    p = cost_problem(
        [("A1", 1000, 125, 5000), ("A2", 1000, 125, 5000)],
        costs={"HI": 10, "LO": 0})
    plan = assert_optimal(p)
    assert stock_draws(plan) == {"HI": 1000, "LO": 1000}


def test_zero_costs_still_solves():
    p = cost_problem(
        [("A1", 1000, 125, 5000), ("A2", 1000, 75, 5000)],
        costs={"HI": 0, "LO": 0})
    assert_optimal(p)


def test_rational_costs_exact():
    # Fraction costs like 1/3 and 2/3 must be compared exactly; the two
    # candidate plans differ by 125 nl of HI (625 vs 500):
    #   direct: 625/3            = 208.333...
    #   blend : 500/3 + 500*2/3  = 500
    # direct is cheaper here despite HI looking nominally cheaper per unit.
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": "1/3", "LO": "2/3"})
    plan = assert_optimal(p)
    assert stock_draws(plan) == {"HI": 625}


def test_inventory_limit_respected():
    # Only 400 nl of the cheap batch available: the 500/500 blend is
    # impossible, so the optimizer falls back to the feasible 625 nl of HI.
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": 100, "LO": 1},
        volumes={"HI": 10 ** 9, "LO": 400})
    plan = assert_optimal(p)
    assert stock_draws(plan) == {"HI": 625}


def test_both_plans_infeasible_when_inventory_short():
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": 1, "LO": 1},
        volumes={"HI": 600, "LO": 400})
    res = solve(p)
    assert not res.ok
    assert res.failure.reason == "no_feasible_plan"


def test_search_limit_is_distinct_from_infeasible():
    # The smallest permitted node budget (100) over eight identical
    # bracketed wells is exhausted before optimality is proved, so the
    # result is a retryable search_limit, never a frozen "no solution".
    p = cost_problem(
        [(f"A{i}", 1000, 125, 5000) for i in range(8)],
        costs={"HI": 1, "LO": 1},
        max_search_nodes=100)
    res = solve(p)
    assert not res.ok
    assert res.failure.reason == "search_limit"
    assert res.failure.detail["max_search_nodes"] == 100


def test_default_mode_without_costs_unchanged():
    # No stock_costs: first-feasible behavior, a node budget never surfaces
    # for a trivially feasible request.
    p = cost_problem(
        [("A1", 1000, 125, 5000)],
        costs={"HI": 1, "LO": 1})
    p.stock_costs = None
    res = solve(p)
    assert res.ok
    pip = make_pip(p.pip_min, p.pip_max, p.pip_step)
    assert verify_plan(p, pip, res.plan) is None


def test_min_cost_serial_dilution_chain():
    # A single 1000-concentration batch reaching a 100 target through
    # serial dilution; cost accounting covers the intermediate's stock draw.
    p = Problem(
        stocks={"S": {"conc": Fraction(1000), "volume": 10 ** 9}},
        diluent_volume=10 ** 9,
        wells=[Well("A1", 1000, 100, 5000)],
        containers=[10_000],
        pip_min=500, pip_max=1000, pip_step=100,
        stock_costs={"S": Fraction("3/2")},
    )
    plan = assert_optimal(p)
    draws = stock_draws(plan)
    assert draws == {"S": 500}  # 5000 nl intermediate at 100 => 500 stock
