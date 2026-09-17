r"""Every proposition in `src/cg2026/pricing.py`, checked numerically.

A derivation in a docstring is a claim. These are the evidence. Each test is
named for the proposition it checks, and the ones that *refute* a historical
statement say which artifact and which line.

Brute force wherever brute force is available: the coordinate problem is
one-dimensional, so a fine grid plus the boundary behaviour settles it, and a
grid does not share a mistake with the closed form it is checking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.cg2026.objective import Penalty, kkt_violation, lambda_max
from src.cg2026.pricing import (
    coordinate_optimum,
    dual_objective,
    pricing_is_bounded,
    unit_ball_pricing_value,
    violations,
)

GRID = np.linspace(-500.0, 500.0, 200_001)


def brute(a: float, l1: float, l2: float) -> float:
    return float(np.min(a * GRID + l1 * np.abs(GRID) + l2 * GRID**2))


# -- P4: the Elastic-Net closed form ------------------------------------------


@pytest.mark.parametrize("a", [-9.0, -4.001, -4.0, -3.9, -1.0, 0.0, 2.5, 4.0, 4.7, 12.0])
@pytest.mark.parametrize("l2", [0.25, 1.0, 7.0])
def test_p4_the_elastic_net_minimiser_matches_a_grid(a: float, l2: float) -> None:
    """The closed form, against a search that shares none of its algebra."""

    l1 = 4.0
    found = coordinate_optimum(a, l1, l2)
    assert found.bounded and found.unique
    assert found.minimiser is not None
    assert found.value == pytest.approx(brute(a, l1, l2), abs=1e-3)
    # And the minimiser really is where the value is attained.
    b = found.minimiser
    assert a * b + l1 * abs(b) + l2 * b * b == pytest.approx(found.value, abs=1e-12)


@pytest.mark.parametrize("a", [-11.0, -5.0, 5.0, 11.0])
def test_p4_the_optimal_value_is_minus_excess_squared_over_four_l2(a: float) -> None:
    """`f(b*) = -(|a| - l1)_+^2 / (4 l2)`, the pricing-improvement score."""

    l1, l2 = 4.0, 1.5
    excess = max(abs(a) - l1, 0.0)
    assert coordinate_optimum(a, l1, l2).value == pytest.approx(-(excess**2) / (4 * l2))


def test_p4_confirms_the_april_2025_note() -> None:
    """`sources/2023-2025/20250408 - Reviewing my past work (Finally).pdf` §1.2.

    It writes `beta_i = (-(psi'X)_i + sign[(psi'X)_i] * lambda_1) / (2 lambda_2)`
    for `lambda_1 < |(psi'X)_i|`, and 0 otherwise. Algebraically the same.
    """

    l1, l2 = 4.0, 1.5
    for a in (-13.0, -4.5, 4.5, 13.0, 3.0, -3.0, 0.0):
        note = (-a + math.copysign(l1, a)) / (2 * l2) if abs(a) > l1 else 0.0
        assert coordinate_optimum(a, l1, l2).minimiser == pytest.approx(note)


def test_p4_refutes_the_whiteboard_closed_form() -> None:
    """`Decomposition_Method_for_Feature_Selection ___.pdf` §3.4.

    It states `beta* = -(1/theta)(2 sqrt(tau kappa) + psi'X)` under the
    assumption that "the equality holds in the conic constraints on the optimal
    point". With `theta = 2 l2` that is `-(l1 + a)/(2 l2)`, which is the
    `a < -l1` branch applied everywhere. It agrees on that branch and nowhere
    else -- wrong sign for `a > l1`, and non-zero where the true answer is 0.
    """

    l1, l2 = 4.0, 1.5
    whiteboard = lambda a: -(l1 + a) / (2 * l2)

    # Agrees where its hidden assumption happens to hold.
    for a in (-13.0, -4.5):
        assert coordinate_optimum(a, l1, l2).minimiser == pytest.approx(whiteboard(a))

    # Disagrees, and is worse, everywhere else.
    for a in (13.0, 4.5, 3.0, 0.0, -3.0):
        true = coordinate_optimum(a, l1, l2)
        assert true.minimiser is not None
        assert true.minimiser != pytest.approx(whiteboard(a))
        claimed = whiteboard(a)
        at_claimed = a * claimed + l1 * abs(claimed) + l2 * claimed * claimed
        assert at_claimed > true.value + 1e-9, (
            "the whiteboard's point is not merely different, it is not optimal"
        )


# -- P2 / P3: the LASSO case and its equality ---------------------------------


@pytest.mark.parametrize("a", [-20.0, -4.0001, 4.0001, 20.0])
def test_p2_strict_violation_is_unbounded(a: float) -> None:
    found = coordinate_optimum(a, 4.0, 0.0)
    assert not found.bounded
    assert found.value == -math.inf
    assert found.recession_direction == -math.copysign(1.0, a)


@pytest.mark.parametrize("a", [-3.9999, -1.0, 0.0, 2.0, 3.9999])
def test_p2_strict_feasibility_is_bounded_with_minimum_zero(a: float) -> None:
    found = coordinate_optimum(a, 4.0, 0.0)
    assert found.bounded and found.unique
    assert found.minimiser == 0.0
    assert found.value == 0.0


@pytest.mark.parametrize("a", [-4.0, 4.0])
def test_p3_the_equality_case_is_bounded_with_a_ray_of_minimisers(a: float) -> None:
    """`|a| = l1`: bounded, value 0, and every opposite-signed `b` attains it.

    The April 2025 note says `beta_i = 0` here, which is *a* minimiser and not
    *the* minimiser. The difference is the whole of §2025-D in the timeline: at
    the LASSO optimum the KKT conditions put every support coordinate on this
    boundary exactly, so a boundedness test with a tolerance on the wrong side
    fires forever.
    """

    l1 = 4.0
    found = coordinate_optimum(a, l1, 0.0)
    assert found.bounded
    assert found.value == 0.0
    assert not found.unique
    assert found.minimiser is None
    for t in (0.0, 1.0, 17.0, 1e6):
        b = -math.copysign(t, a)
        assert a * b + l1 * abs(b) == pytest.approx(0.0, abs=1e-6 * max(1.0, t))


def test_p3_the_implementations_tolerance_reports_unbounded_at_the_optimum() -> None:
    """The defect, as the shipped code writes it.

    `src/cg_models.py`:
        unbounded = (|psi'X| - 2 sqrt(kappa tau) > -cg_lambda_tol).any()

    which is `|a| > l1 - tol`. At the optimum `|a_i| == l1` on the support, so
    this is true there. The pricing test can therefore never certify
    optimality, and the run stops on the dual-stall heuristic instead.
    """

    from pathlib import Path

    # Read the shipped source, because a test that asserts a property of
    # somebody else's code and never opens it is asserting nothing. The
    # previous version computed `(abs(l1) - l1) > -tol` -- that is `0 > -1e-6`
    # -- and would have passed verbatim if the shipped test were correct.
    source = Path("src/cg_models.py").read_text(encoding="utf-8")
    line = next(
        item.strip()
        for item in source.splitlines()
        if item.strip().startswith("unbounded_lagrangian = ")
    )
    assert "> -cg_lambda_tol" in line, line
    assert not line.startswith("#")

    l1, tol = 4.0, 1e-6
    at_the_optimum = l1  # exactly, as KKT makes it on the support
    shipped = (abs(at_the_optimum) - l1) > -tol
    assert shipped is True
    assert coordinate_optimum(at_the_optimum, l1, 0.0).bounded is True, (
        "the truth is that it is bounded; the shipped test says otherwise"
    )


# -- P2 as a statement about the whole vector ---------------------------------


def test_p2_boundedness_is_exactly_dual_feasibility() -> None:
    """Both directions, on random instances, against the coordinate solver."""

    rng = np.random.default_rng(11)
    for _ in range(200):
        n, p = 9, 6
        X = rng.normal(size=(n, p))
        psi = rng.normal(size=n) * rng.choice([0.2, 1.0, 5.0])
        l1 = float(abs(rng.normal()) * 3.0)
        a = X.T @ psi
        by_coordinate = all(coordinate_optimum(ai, l1, 0.0).bounded for ai in a)
        by_dual_feasibility = bool(np.abs(a).max() <= l1)
        assert by_coordinate == by_dual_feasibility
        assert pricing_is_bounded(X, psi, l1, 0.0) == by_dual_feasibility


def test_p2_a_positive_l2_makes_every_pricing_bounded() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 12))
    psi = rng.normal(size=20) * 1000.0
    assert not pricing_is_bounded(X, psi, l1=0.1, l2=0.0)
    assert pricing_is_bounded(X, psi, l1=0.1, l2=1e-9)


def test_p3_violations_are_positive_exactly_on_the_unbounded_coordinates() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 30))
    psi = rng.normal(size=15)
    l1 = float(np.median(np.abs(X.T @ psi)))
    v = violations(X, psi, l1)
    for i, ai in enumerate(X.T @ psi):
        assert (v[i] > 0) == (not coordinate_optimum(ai, l1, 0.0).bounded)


def test_p3_the_violation_is_the_rate_of_decrease_along_its_ray() -> None:
    """`v_i` is the reduced cost per unit step, so ordering by `v` orders by rate."""

    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 8))
    psi = rng.normal(size=12)
    l1 = 0.5 * float(np.abs(X.T @ psi).max())
    a = X.T @ psi
    v = violations(X, psi, l1)
    for i in np.nonzero(v)[0]:
        for t in (1.0, 10.0, 100.0):
            b = -math.copysign(t, a[i])
            assert a[i] * b + l1 * abs(b) == pytest.approx(-v[i] * t, rel=1e-12)


# -- P6: the unit ball is not a lower bound ------------------------------------


def test_p6_the_unit_ball_value_can_exceed_the_primal_optimum() -> None:
    """A two-line counterexample, worked by hand and then checked.

    `X = [[1]]`, `y = [10]`, `l1 = 1`. The primal optimum is at
    `beta = 10 - 1/2 = 9.5`, value `0.25 + 9.5 = 9.75`.

    Take `psi = -1.5`, which is dual-infeasible (`|X'psi| = 1.5 > 1 = l1`), so
    the true `g(psi) = -inf`. The dual objective there is
    `-(1.5^2)/4 + 15 = 14.4375`, and `min_{|b| <= 1} (-1.5 b + |b|) = -0.5` at
    `b = 1`. The unit-ball value is `13.9375`, which is **43 % above** the
    primal optimum.
    """

    X = np.array([[1.0]])
    y = np.array([10.0])
    l1 = 1.0
    penalty = Penalty(lambda_1=l1)

    beta_star = np.array([9.5])
    primal = penalty.value(X, y, beta_star)
    assert primal == pytest.approx(9.75)
    assert kkt_violation(X, y, beta_star, penalty) == pytest.approx(0.0, abs=1e-12)

    psi = np.array([-1.5])
    assert not pricing_is_bounded(X, psi, l1, 0.0)
    assert dual_objective(psi, y) == pytest.approx(14.4375)

    ball = unit_ball_pricing_value(X, y, psi, l1)
    assert ball == pytest.approx(13.9375, abs=1e-6)
    assert ball > primal
    assert ball / primal > 1.4


def test_p6_the_reachable_first_dual_is_the_strong_exhibit() -> None:
    """`psi = -2y` is the dual of the empty master, so every run visits it.

    The hand counterexample uses `psi = -1.5`, which SOC complementarity makes
    unreachable on that instance -- the only reachable duals are `-20` and
    `-1`. An independent review made that objection. The answer is that the
    reachable dual is *worse*: at `psi = -2y` the unit-ball value is 81 against
    a primal optimum of 9.75.
    """

    X = np.array([[1.0]])
    y = np.array([10.0])
    l1 = 1.0
    penalty = Penalty(lambda_1=l1)
    primal = penalty.value(X, y, np.array([9.5]))

    psi = -2.0 * y  # the dual of the master with no columns: beta = 0
    assert not pricing_is_bounded(X, psi, l1, 0.0)
    ball = unit_ball_pricing_value(X, y, psi, l1)
    assert ball == pytest.approx(81.0, abs=1e-6)
    assert ball > 8 * primal


def test_p6_depends_on_standardisation_not_on_dimension() -> None:
    """The question two of this project's own measurements disagreed about.

    An independent review exhibited reachable duals where the unit-ball value
    exceeds the primal optimum; `scripts/adjudicate_pricing.py` found none in
    thirty. Both were right, and the first two explanations attempted here --
    "it is about p", then "it is about p/n" -- were both wrong, measured and
    discarded.

    It is about **scaling**. At the reachable first dual `psi = -2y` the
    unit-ball value is `||y||^2 + min_{||b||<=1}[a'b + l1||b||_1]`, and the
    inner term is bounded below by `-||X'psi||_2`. Standardising sets
    `||y||^2 = n` and every column norm to `sqrt(n)`, which puts the two terms
    on the same scale and drives the value well below `p*`. Leaving the data
    unscaled leaves `||y||^2` far larger than anything the unit ball can
    subtract, and the raw dual objective shows through.

    Measured, six seeds per cell, at the reachable `psi = -2y`:

    ```text
    standardised    (300,10) (300,80) (30,60) (40,12)   0 of 24 exceed
    unstandardised  same shapes                        24 of 24 exceed,
                                                        median 1.26x - 1.55x p*
    ```

    This matters beyond bookkeeping. Both the 2025 driver and
    `src/cg2026/data.py` standardise, so the historical runs would *not* have
    shown a bound obviously above the optimum -- which is consistent with the
    2025 notes reporting oscillation rather than an evidently invalid bound.
    The device is still unjustified; standardisation hides its symptom.
    """

    from src.cg2026.data import InstanceSpec, build

    def exceedances(standardise: bool) -> tuple[int, int]:
        hits = trials = 0
        for n, p in ((300, 10), (300, 80), (30, 60), (40, 12)):
            for seed in range(3):
                X, y, _ = build(
                    InstanceSpec(
                        "sparse",
                        n=n,
                        p=p,
                        k=min(8, p),
                        seed=seed,
                        standardise=standardise,
                    )
                )
                penalty = Penalty(lambda_1=0.2 * lambda_max(X, y))
                psi = -2.0 * y  # the dual of the master with no columns
                if pricing_is_bounded(X, psi, penalty.lambda_1, 0.0):
                    continue
                trials += 1
                primal = penalty.value(X, y, _solve_lasso(X, y, penalty))
                if unit_ball_pricing_value(X, y, psi, penalty.lambda_1) > primal:
                    hits += 1
        return hits, trials

    scaled = exceedances(True)
    raw = exceedances(False)
    assert scaled[1] > 0 and raw[1] > 0
    assert scaled[0] == 0, f"standardised: {scaled}"
    assert raw[0] == raw[1], f"unstandardised: {raw}"


def test_p6_a_dual_feasible_point_does_give_a_valid_bound() -> None:
    """The contrast that makes the counterexample mean something.

    At a dual-*feasible* `psi` the unbounded pricing minimum is 0, the
    unit-ball restriction changes nothing, and the value is a genuine lower
    bound. The unit ball is only ever wrong where it is only ever used: at an
    infeasible point, which is the only place the pricing is unbounded.
    """

    X = np.array([[1.0]])
    y = np.array([10.0])
    l1 = 1.0
    for psi_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        psi = np.array([psi_value])
        assert pricing_is_bounded(X, psi, l1, 0.0)
        assert unit_ball_pricing_value(X, y, psi, l1) == pytest.approx(
            dual_objective(psi, y), abs=1e-7
        )
        assert dual_objective(psi, y) <= 9.75 + 1e-9


def test_p6_the_failure_is_not_a_one_dimensional_artifact() -> None:
    """The same thing on random multi-dimensional instances."""

    rng = np.random.default_rng(23)
    exceedances = 0
    for _ in range(40):
        n, p = 6, 4
        X = rng.normal(size=(n, p))
        y = rng.normal(size=n) * 8.0
        l1 = 0.15 * lambda_max(X, y)
        penalty = Penalty(lambda_1=l1)
        beta = _solve_lasso(X, y, penalty)
        primal = penalty.value(X, y, beta)
        # A dual point the algorithm could actually reach: the residual of a
        # *restricted* fit, scaled up. psi = -2r is the exact relation at
        # optimality, so this is the same object one iteration earlier.
        psi = -2.0 * (y - X @ np.zeros(p))
        if pricing_is_bounded(X, psi, l1, 0.0):
            continue
        if unit_ball_pricing_value(X, y, psi, l1) > primal + 1e-9:
            exceedances += 1
    # 37 of 40 measured. The previous assertion was `> 0`, which is 37 times
    # weaker than the fact and would have passed if 39 of 40 regressed.
    assert exceedances >= 30, f"only {exceedances} of 40 exceeded"


def _solve_lasso(X: np.ndarray, y: np.ndarray, penalty: Penalty) -> np.ndarray:
    from sklearn.linear_model import Lasso

    n = X.shape[0]
    model = Lasso(
        alpha=penalty.sklearn_alpha(n),
        fit_intercept=False,
        max_iter=200_000,
        tol=1e-14,
    )
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float)


# -- the objective convention itself -------------------------------------------


def test_the_sklearn_conversion_agrees_on_objective_values() -> None:
    """The conversion is checked on the *value*, not on the formula.

    This is the test whose absence would have invalidated every timing in the
    project, so it is checked for the LASSO and the Elastic Net, at several
    sample sizes, against scikit-learn's own written objective.
    """

    rng = np.random.default_rng(1)
    for n in (5, 50, 500):
        X = rng.normal(size=(n, 6))
        y = rng.normal(size=n)
        beta = rng.normal(size=6)
        for l1, l2 in ((3.0, 0.0), (3.0, 1.5), (0.0, 2.0), (0.7, 0.2)):
            penalty = Penalty(lambda_1=l1, lambda_2=l2)
            alpha = penalty.sklearn_alpha(n)
            ratio = penalty.sklearn_l1_ratio(n)
            sk = (
                (1 / (2 * n)) * float((y - X @ beta) @ (y - X @ beta))
                + alpha * ratio * float(np.abs(beta).sum())
                + 0.5 * alpha * (1 - ratio) * float(beta @ beta)
            )
            assert penalty.value(X, y, beta) == pytest.approx(2 * n * sk, rel=1e-12)


def test_lambda_max_is_exactly_where_the_solution_becomes_zero() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 10))
    y = rng.normal(size=60)
    lmax = lambda_max(X, y)
    # Tight on both sides. The previous bracket was [0.98, 1.001], which a
    # `lambda_max` overstated by 1.9 % still passed -- the name promised
    # "exactly" and the assertion checked one side.
    assert np.allclose(_solve_lasso(X, y, Penalty(lambda_1=lmax * 1.0001)), 0.0)
    assert not np.allclose(_solve_lasso(X, y, Penalty(lambda_1=lmax * 0.999)), 0.0)


def test_the_kkt_measure_is_zero_at_a_solved_lasso() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 15))
    y = rng.normal(size=80)
    penalty = Penalty(lambda_1=0.2 * lambda_max(X, y))
    beta = _solve_lasso(X, y, penalty)
    assert kkt_violation(X, y, beta, penalty) < 1e-6 * penalty.lambda_1


def test_the_duality_gap_is_a_certificate() -> None:
    """Non-negative always, and shrinking to zero at the solution."""

    from src.cg2026.objective import duality_gap

    rng = np.random.default_rng(6)
    X = rng.normal(size=(70, 12))
    y = rng.normal(size=70)
    penalty = Penalty(lambda_1=0.3 * lambda_max(X, y))
    solved = _solve_lasso(X, y, penalty)
    assert duality_gap(X, y, solved, penalty) >= -1e-9
    assert duality_gap(X, y, solved, penalty) < 1e-6 * penalty.value(X, y, solved)
    sloppy = solved + 0.3 * rng.normal(size=12)
    assert duality_gap(X, y, sloppy, penalty) > duality_gap(X, y, solved, penalty)


# -- the corollary, which had no test at all -----------------------------------
#
# An independent adversarial review's first observation about this file was
# that its most load-bearing claim carried no evidence: the tests checked the
# coordinate solver against itself. These are the missing tests, and each one
# is named for the wrong form of the corollary it rules out.


def restricted_master(X, y, l1, l2, working_set, *, weights="free"):
    """Solve the master restricted to ``span{e_i : i in S}`` and return its dual.

    ``weights`` is the hypothesis under test: ``free`` and ``nonneg`` are cones
    containing zero, ``simplex`` (the historical ``sum_1_comb=True``) is an
    affine set and is the counterexample.
    """

    import cvxpy as cp

    p = X.shape[1]
    index = sorted(working_set)
    columns = np.zeros((p, max(len(index), 1)))
    for column, coordinate in enumerate(index):
        columns[coordinate, column] = 1.0
    pi = cp.Variable(columns.shape[1])
    xi = cp.Variable(1)
    eta = cp.Variable(p)
    beta = columns @ pi
    objective = cp.square(xi) + l1 * cp.sum(eta)
    if l2:
        objective = objective + l2 * cp.sum_squares(beta)
    constraints = [cp.SOC(xi, y - X @ beta), beta <= eta, -beta <= eta]
    if weights == "nonneg":
        constraints.append(pi >= 0)
    elif weights == "simplex":
        constraints.extend([cp.sum(pi) == 1, pi >= 0])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL, tol_gap_rel=1e-12, tol_feas=1e-12)
    dual = constraints[0].dual_value
    return (
        float(problem.value),
        np.ravel(dual[1]).astype(float),
        np.asarray(beta.value, dtype=float).ravel(),
    )


def test_the_corollary_holds_when_every_violated_coordinate_is_in_the_set():
    """The corollary itself, for `l2 = 0` and for `l2 > 0`, on a real master."""

    from src.cg2026.pricing import restricted_master_is_optimal

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 10))
    y = rng.normal(size=40) * 3
    for l2 in (0.0, 1.0, 5.0):
        l1 = 0.5 * lambda_max(X, y)
        optimum, _psi, beta = restricted_master(X, y, l1, l2, set(range(10)))
        support = set(np.flatnonzero(np.abs(beta) > 1e-7).tolist())
        value, psi, _ = restricted_master(X, y, l1, l2, support)
        assert restricted_master_is_optimal(X, psi, l1, l2, working_set=support)
        assert value == pytest.approx(optimum, rel=1e-6), (
            f"l2={l2}: the certificate fired and the master was not optimal"
        )


def test_the_violated_set_is_non_empty_at_optimality_when_l2_is_positive():
    """Why "nothing violates" is the wrong certificate for the Elastic Net.

    With `l2 > 0` the full inner minimum is attained at the soft threshold,
    whose support is exactly the violated set -- so at the optimum the violated
    set equals the support and is not empty. A corollary phrased as
    `violations == 0` would never fire. This was the third error made while
    repairing the first two, and it is why `restricted_master_is_optimal`
    refuses to answer for `l2 > 0` without a working set.
    """

    from src.cg2026.pricing import restricted_master_is_optimal, violations

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 10))
    y = rng.normal(size=40) * 3
    l1 = 0.5 * lambda_max(X, y)

    _v, psi0, _b0 = restricted_master(X, y, l1, 0.0, set(range(10)))
    assert not np.flatnonzero(violations(X, psi0, l1)).size, (
        "with l2 = 0 the violated set IS empty at optimality"
    )

    _v, psi2, beta2 = restricted_master(X, y, l1, 1.0, set(range(10)))
    violated = set(np.flatnonzero(violations(X, psi2, l1)).tolist())
    support = set(np.flatnonzero(np.abs(beta2) > 1e-7).tolist())
    assert violated, "with l2 > 0 the violated set is non-empty at optimality"
    assert violated == support

    with pytest.raises(ValueError, match="never fires"):
        restricted_master_is_optimal(X, psi2, l1, 1.0)


def test_an_affine_master_breaks_the_corollary():
    """`sum_1_comb=True`, which the historical code offers, is a counterexample.

    `X = I_2`, `y = (1,1)`, `l1 = 20 > lambda_max`, so `beta* = 0` and
    `p* = 2`. With `e'pi = 1` the master cannot reach zero, lands at
    `beta = (0.5, 0.5)` with objective 20.5, and its dual violates nothing --
    so a corollary without hypothesis 1 would declare a master ten times worse
    than the optimum to be optimal.
    """

    from src.cg2026.pricing import restricted_master_is_optimal, violations

    X = np.eye(2)
    y = np.ones(2)
    l1 = 20.0
    penalty = Penalty(lambda_1=l1)
    assert l1 > lambda_max(X, y)
    optimum = penalty.value(X, y, np.zeros(2))
    assert optimum == pytest.approx(2.0)

    value, psi, beta = restricted_master(X, y, l1, 0.0, {0, 1}, weights="simplex")
    assert value == pytest.approx(20.5, rel=1e-6)
    assert beta == pytest.approx(np.array([0.5, 0.5]), abs=1e-6)
    assert not np.flatnonzero(violations(X, psi, l1)).size
    # The certificate fires ...
    assert restricted_master_is_optimal(X, psi, l1, 0.0, working_set={0, 1})
    # ... and the master is ten times worse than the optimum. Hypothesis 1 is
    # what stands between those two lines.
    assert value > 10 * optimum


def test_a_cone_master_restores_it():
    """The contrast: free and non-negative weights both satisfy hypothesis 1."""

    X = np.eye(2)
    y = np.ones(2)
    l1 = 20.0
    optimum = Penalty(lambda_1=l1).value(X, y, np.zeros(2))
    for weights in ("free", "nonneg"):
        value, _psi, _beta = restricted_master(X, y, l1, 0.0, {0, 1}, weights=weights)
        assert value == pytest.approx(optimum, rel=1e-6), weights


def test_the_dual_objective_equals_the_master_value_when_l2_is_zero():
    """`d(psi^k) = RMP_k` identically, which is why there is no gap to plot.

    The consequence is the one that condemns the historical bound figure: the
    Lagrangian "lower bound" is either minus infinity or exactly the upper
    bound, so the plot can only show two identical curves or one missing curve.
    """

    from src.cg2026.pricing import dual_objective

    rng = np.random.default_rng(17)
    X = rng.normal(size=(12, 5))
    y = rng.normal(size=12) * 4
    l1 = 0.4 * lambda_max(X, y)
    for size in range(6):
        for _ in range(3):
            working = set(rng.choice(5, size=size, replace=False).tolist())
            value, psi, _ = restricted_master(X, y, l1, 0.0, working)
            assert dual_objective(psi, y) == pytest.approx(value, rel=1e-6), f"S={sorted(working)}"


def test_the_identity_does_not_extend_to_a_positive_l2():
    """Positive homogeneity is what makes it work, and `l2` destroys it."""

    from src.cg2026.pricing import dual_objective

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 6))
    y = rng.normal(size=10) * 5
    value, psi, _ = restricted_master(X, y, 2.0, 1.0, set(range(6)))
    assert dual_objective(psi, y) != pytest.approx(value, rel=1e-3)
    assert dual_objective(psi, y) > value, (
        "and it is above, so reading it as a lower bound is unsafe"
    )
