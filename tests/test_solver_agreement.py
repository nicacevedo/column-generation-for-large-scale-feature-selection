"""Every solver is solving the same problem. Checked on values, not formulas.

This is the test whose absence would make the whole benchmark meaningless and
whose failure nothing else would reveal: two solvers minimising objectives that
differ by a factor of `2n` still both converge, still both report success, and
still produce a table in which one of them looks fast.

The check is: solve the same instance with every solver at a tight tolerance,
then assert the **objective values agree** in this project's convention and the
**supports agree**. Not that the conversion formula looks right.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cg2026.data import InstanceSpec, build
from src.cg2026.objective import Penalty, lambda_max
from src.cg2026.solvers import REGISTRY

ACCURATE = ("sklearn", "celer", "skglm", "lars", "conic_reduced", "conic_thesis")


@pytest.fixture(scope="module")
def instance():
    X, y, _ = build(InstanceSpec("sparse", n=200, p=60, k=8, seed=1))
    return X, y


@pytest.fixture(scope="module")
def penalty(instance):
    X, y = instance
    return Penalty(lambda_1=0.15 * lambda_max(X, y))


@pytest.fixture(scope="module")
def reference(instance, penalty):
    """LARS, which is exact for the LASSO path, as the arbiter."""

    X, y = instance
    return REGISTRY["lars"](X, y, penalty, tol=1e-12)


@pytest.mark.parametrize("name", ACCURATE)
def test_every_solver_reaches_the_same_objective(name, instance, penalty, reference):
    X, y = instance
    found = REGISTRY[name](X, y, penalty, tol=1e-10)
    assert found.objective == pytest.approx(reference.objective, rel=1e-6), (
        f"{name} is not minimising the same function: {found.objective} vs {reference.objective}"
    )


@pytest.mark.parametrize("name", ACCURATE)
def test_every_solver_reaches_the_same_support(name, instance, penalty, reference):
    X, y = instance
    found = REGISTRY[name](X, y, penalty, tol=1e-10)
    threshold = 1e-6 * np.abs(reference.beta).max()
    assert set(np.flatnonzero(np.abs(found.beta) > threshold)) == set(
        np.flatnonzero(np.abs(reference.beta) > threshold)
    ), f"{name} selected a different support"


@pytest.mark.parametrize("name", ACCURATE)
def test_every_solver_reaches_a_small_certified_gap(name, instance, penalty):
    """The relative duality gap, which is the project's primary accuracy measure.

    Threshold-free and certified, unlike the KKT violation, which needs a
    support tolerance to be meaningful for an interior-point solver at all.
    """

    X, y = instance
    found = REGISTRY[name](X, y, penalty, tol=1e-10)
    assert found.gap is not None
    assert found.gap < 1e-5, f"{name} reported success at relative gap {found.gap}"


def test_the_thesis_conic_form_is_less_accurate_than_the_reduced_one(instance, penalty):
    """Same problem, same solver, same requested tolerance, worse answer.

    `conic_thesis` writes the l1 penalty as `m` rotated cones over `(b, z, u)`;
    `conic_reduced` writes it as one epigraph over `eta`. They are the same
    optimisation problem -- `test_the_two_conic_formulations_are_the_same_problem`
    checks that -- but the first gives Clarabel `3m + 1` variables and `m + 1`
    cones where the second gives it `2m + 1` and one.

    The difference shows up as accuracy at a fixed requested tolerance, which
    is worth recording because it is a cheap and never-tried intervention on
    the numerical instability the historical material documents at length:
    the model the 2023 work submitted to MOSEK is not the easiest model of its
    own problem.
    """

    X, y = instance
    thesis = REGISTRY["conic_thesis"](X, y, penalty, tol=1e-10)
    reduced = REGISTRY["conic_reduced"](X, y, penalty, tol=1e-10)
    assert thesis.gap is not None and reduced.gap is not None
    assert thesis.gap > reduced.gap


@pytest.mark.parametrize("name", ACCURATE)
def test_every_solver_satisfies_the_kkt_conditions(name, instance, penalty):
    X, y = instance
    found = REGISTRY[name](X, y, penalty, tol=1e-10)
    assert found.kkt < 1e-3 * penalty.lambda_1, (
        f"{name} reported success at KKT violation {found.kkt} against lambda_1 {penalty.lambda_1}"
    )


def test_the_kkt_measure_needs_its_support_tolerance() -> None:
    """The defect this tolerance exists to prevent, as a regression test.

    Without it, a conic solution whose objective agrees with LARS to seven
    digits scores a violation of roughly `lambda_1` -- because every one of its
    coefficients is `6e-13` rather than `0`, so every coordinate takes the
    support branch. A benchmark using the naive measure would report that
    interior-point solvers never converge.
    """

    from src.cg2026.objective import kkt_violation as measure

    X, y, _ = build(InstanceSpec("sparse", n=120, p=30, k=5, seed=9))
    pen = Penalty(lambda_1=0.2 * lambda_max(X, y))
    exact = REGISTRY["lars"](X, y, pen, tol=1e-12).beta
    dusty = exact + np.where(exact == 0, 6e-13, 0.0)
    assert measure(X, y, dusty, pen, support_rtol=0.0) > 0.5 * pen.lambda_1
    assert measure(X, y, dusty, pen) < 1e-6 * pen.lambda_1


def test_the_two_conic_formulations_are_the_same_problem(instance, penalty, reference):
    """`min tau z + kappa u  s.t.  b^2 <= zu` is `2 sqrt(tau kappa) |b|`.

    The thesis's three-block form with `m` rotated cones and the one-epigraph
    form must reach the same optimum, or the AM-GM identity in `pricing.py`'s
    header is wrong.
    """

    X, y = instance
    thesis = REGISTRY["conic_thesis"](X, y, penalty, tol=1e-10)
    reduced = REGISTRY["conic_reduced"](X, y, penalty, tol=1e-10)
    assert thesis.objective == pytest.approx(reduced.objective, rel=1e-7)
    assert thesis.objective == pytest.approx(reference.objective, rel=1e-6)


def test_the_elastic_net_conversions_also_agree(instance):
    """`lambda_2 > 0` exercises a different branch of every conversion."""

    X, y = instance
    penalty = Penalty(lambda_1=0.1 * lambda_max(X, y), lambda_2=5.0)
    values = {
        name: REGISTRY[name](X, y, penalty, tol=1e-10).objective
        for name in ("sklearn", "skglm", "conic_reduced", "conic_thesis")
    }
    best = min(values.values())
    for name, value in values.items():
        assert value == pytest.approx(best, rel=1e-5), f"{name}: {values}"


def test_the_historical_method_reaches_the_same_optimum(instance, penalty, reference):
    """A PARTIAL_REPRODUCTION that disagreed with LARS would not be one.

    The tolerance here is looser than for the others, deliberately: the
    historical method stops on a dual-stall heuristic rather than an optimality
    certificate, so it is *expected* to land near the optimum rather than on it.
    Quantifying that gap is `HYP-0006`, not a reason to fail this test.
    """

    X, y = instance
    found = REGISTRY["cg_hist"](X, y, penalty, tol=1e-8, time_limit_minutes=2.0)
    assert found.status == "ok", found.status
    assert found.objective == pytest.approx(reference.objective, rel=1e-4)


def test_the_historical_method_stops_short_of_optimality(instance, penalty, reference):
    """`HYP-0006`, as a measurement rather than a reading of the source.

    Every other solver here reaches a KKT violation below `1e-4 * lambda_1`.
    If the historical method did too, the dual-stall criterion would be as good
    as a certificate and HYP-0006 would be false.
    """

    X, y = instance
    found = REGISTRY["cg_hist"](X, y, penalty, tol=1e-8, time_limit_minutes=2.0)
    exact = REGISTRY["lars"](X, y, penalty, tol=1e-12)
    assert found.kkt > 100 * exact.kkt, (
        "the historical method matched an exact solver's optimality, which would refute HYP-0006"
    )


def test_the_lagrangian_bound_is_recorded_only_at_large_lambda(instance):
    """The lower bound is computed -- rarely, and only when the penalty is large.

    **This test previously asserted "never", and that was wrong.** It was named
    `test_the_historical_method_never_records_a_lagrangian_bound`, it asserted
    `== 0`, and it passed only because its fixture sits at `0.15 * lambda_max`.
    `EXP-0001` then recorded a bound on four cells, every one of them at
    `lambda_ratio = 0.5` and `tol = 1e-8`, which rejects `HYP-0006` as stated.

    The mechanism is the unboundedness test at `cg_models.py:446`:
    `|psi'X| > 2*sqrt(kappa*tau) - cg_lambda_tol`. Both `tau` and `kappa` are
    `lambda_1 / 2`, so the threshold grows with the penalty. At a small penalty
    it is cleared on every iteration, the Lagrangian model is skipped, and no
    bound exists. At a large one the test can fail to trigger, the model runs,
    and a bound is recorded.

    So the 2025 note "in the old version we didn't even have a lower bound" is
    right about the regime it was written in and wrong as a universal. The test
    now pins both sides of the boundary rather than the convenient side.
    """

    X, y = instance
    ceiling = lambda_max(X, y)

    weak = REGISTRY["cg_hist"](
        X, y, Penalty(lambda_1=0.15 * ceiling), tol=1e-8, time_limit_minutes=2.0
    )
    assert weak.detail["lagrangian_values_recorded"] == 0, (
        "at a small penalty the pricing is unbounded on every iteration and the "
        "Lagrangian model never runs"
    )

    strong = REGISTRY["cg_hist"](
        X, y, Penalty(lambda_1=0.7 * ceiling), tol=1e-8, time_limit_minutes=2.0
    )
    assert strong.detail["lagrangian_values_recorded"] >= 1, (
        "at a large penalty the method does compute a Lagrangian lower bound, "
        "which is what rejects HYP-0006"
    )
