"""Every solver, behind one interface, at one objective.

The interface is `solve(X, y, penalty, tol) -> Solution`. What the adapters do
is translate this project's convention (`ASM-0001`) into each solver's own and
translate the answer back, and **every translation is checked on an objective
value** in `tests/test_solver_agreement.py`. A benchmark whose parameter
conversion is wrong produces a clean, plausible, meaningless table, and nothing
in the table itself would show it.

What is deliberately *not* here: any attempt to make the solvers stop at the
same place. They cannot -- scikit-learn stops on a scaled duality gap, celer on
its own, an interior-point solver on a complementarity measure -- so instead
each is run over a ladder of its own tolerances and the *achieved* KKT
violation is recorded alongside the time. "Time to reach this accuracy" is then
a question the data can answer, rather than one the configuration decides in
advance. That is the comparison methodology the modern solver papers use and
the historical comparison did not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.cg2026.objective import Penalty, kkt_violation, relative_gap


@dataclass(slots=True)
class Solution:
    """One solve, with everything needed to judge it."""

    solver: str
    beta: np.ndarray
    wall_seconds: float
    cpu_seconds: float
    objective: float
    kkt: float
    gap: float | None
    """Relative duality gap. ``None`` when ``lambda_2 > 0``, where the
    construction in :func:`~src.cg2026.objective.duality_gap` does not apply."""

    nnz: int
    tol: float
    status: str = "ok"
    iterations: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _finish(
    solver: str,
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    penalty: Penalty,
    tol: float,
    wall: float,
    cpu: float,
    *,
    status: str = "ok",
    iterations: int | None = None,
    detail: dict[str, Any] | None = None,
) -> Solution:
    beta = np.asarray(beta, dtype=float).ravel()
    return Solution(
        solver=solver,
        beta=beta,
        wall_seconds=wall,
        cpu_seconds=cpu,
        objective=penalty.value(X, y, beta),
        kkt=kkt_violation(X, y, beta, penalty),
        gap=relative_gap(X, y, beta, penalty) if penalty.lambda_2 == 0 else None,
        # A coefficient below 1e-12 is numerical dust, not a selected feature.
        # Interior-point solvers return dense vectors of tiny values and
        # counting them as support would make every conic solve look like it
        # selected everything.
        nnz=int((np.abs(beta) > 1e-12).sum()),
        tol=tol,
        status=status,
        iterations=iterations,
        detail=detail or {},
    )


# -- the modern baselines ------------------------------------------------------


def solve_sklearn(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    from sklearn.linear_model import ElasticNet, Lasso

    n = X.shape[0]
    alpha = penalty.sklearn_alpha(n)
    if penalty.lambda_2 == 0:
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1_000_000, tol=tol)
    else:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=penalty.sklearn_l1_ratio(n),
            fit_intercept=False,
            max_iter=1_000_000,
            tol=tol,
        )
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(X, y)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return _finish(
        "sklearn",
        model.coef_,
        X,
        y,
        penalty,
        tol,
        wall,
        cpu,
        iterations=int(np.max(model.n_iter_)),
    )


def solve_celer(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    from celer import Lasso

    if penalty.lambda_2 != 0:
        raise NotImplementedError("celer is used for the LASSO only here")
    n = X.shape[0]
    model = Lasso(
        alpha=penalty.sklearn_alpha(n),
        fit_intercept=False,
        tol=tol,
        max_iter=1000,
        max_epochs=100_000,
        verbose=0,
    )
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(X, y)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return _finish("celer", model.coef_, X, y, penalty, tol, wall, cpu)


def solve_skglm(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    from skglm import ElasticNet, Lasso

    n = X.shape[0]
    alpha = penalty.sklearn_alpha(n)
    if penalty.lambda_2 == 0:
        model = Lasso(alpha=alpha, fit_intercept=False, tol=tol, max_iter=100)
    else:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=penalty.sklearn_l1_ratio(n),
            fit_intercept=False,
            tol=tol,
            max_iter=100,
        )
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(X, y)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return _finish("skglm", model.coef_, X, y, penalty, tol, wall, cpu)


def solve_lars(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    """LassoLars: exact to machine precision, and the historical code offered it.

    Its `tol` means something else entirely, so it is run once per instance and
    its recorded `tol` is the one it was asked for, not one it honoured.
    """

    from sklearn.linear_model import LassoLars

    if penalty.lambda_2 != 0:
        raise NotImplementedError("LARS is for the LASSO only")
    n = X.shape[0]
    model = LassoLars(alpha=penalty.sklearn_alpha(n), fit_intercept=False, max_iter=8000)
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(X, y)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return _finish("lars", model.coef_, X, y, penalty, tol, wall, cpu)


# -- the conic reference solves ------------------------------------------------


def solve_conic_reduced(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    """`models.py::L_LASSO`: the model with the l1 term written with one epigraph.

    ``min xi^2 + l1 e'eta  s.t.  ||y - X b||_2 <= xi,  -eta <= b <= eta``.
    `m + m + 1` variables and one second-order cone.
    """

    import cvxpy as cp

    p = X.shape[1]
    beta = cp.Variable(p)
    eta = cp.Variable(p)
    xi = cp.Variable(1)
    objective = cp.square(xi) + penalty.lambda_1 * cp.sum(eta)
    if penalty.lambda_2:
        objective = objective + penalty.lambda_2 * cp.sum_squares(beta)
    problem = cp.Problem(
        cp.Minimize(objective),
        [cp.SOC(xi, y - X @ beta), beta <= eta, -beta <= eta],
    )
    wall, cpu = time.perf_counter(), time.process_time()
    problem.solve(solver=cp.CLARABEL, tol_gap_rel=tol, tol_feas=tol)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    value = beta.value if beta.value is not None else np.zeros(p)
    return _finish(
        "conic_reduced",
        value,
        X,
        y,
        penalty,
        tol,
        wall,
        cpu,
        status=str(problem.status),
    )


def solve_conic_thesis(X, y, penalty: Penalty, tol: float, **_: Any) -> Solution:
    """`models.py::SOCP`: the 2023 formulation, with its `m` rotated cones.

    ``min xi^2 + tau e'z + kappa e'u`` subject to ``||y - Xb|| <= xi`` and
    ``b_i^2 <= z_i u_i``. With `tau = kappa = l1/2` this is the same problem as
    `conic_reduced` -- `min tau z + kappa u  s.t.  b^2 <= zu` is `l1|b|` -- and
    it is included because it is what the historical work actually submitted to
    MOSEK, and because `3m + 1` variables with `m` cones is a materially
    different thing for a solver to see than `2m + 1` with one.
    """

    import cvxpy as cp

    p = X.shape[1]
    tau = kappa = penalty.lambda_1 / 2.0
    beta = cp.Variable(p)
    z = cp.Variable(p, nonneg=True)
    u = cp.Variable(p, nonneg=True)
    xi = cp.Variable(1)
    objective = cp.square(xi) + tau * cp.sum(z) + kappa * cp.sum(u)
    if penalty.lambda_2:
        objective = objective + penalty.lambda_2 * cp.sum_squares(beta)
    constraints = [
        cp.SOC(xi, y - X @ beta),
        cp.SOC(u + z, cp.vstack([u - z, 2 * beta]), axis=0),
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    wall, cpu = time.perf_counter(), time.process_time()
    problem.solve(solver=cp.CLARABEL, tol_gap_rel=tol, tol_feas=tol)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    value = beta.value if beta.value is not None else np.zeros(p)
    return _finish(
        "conic_thesis",
        value,
        X,
        y,
        penalty,
        tol,
        wall,
        cpu,
        status=str(problem.status),
    )


# -- the historical method -----------------------------------------------------


def solve_cg_hist(
    X,
    y,
    penalty: Penalty,
    tol: float,
    *,
    policy: str = "v_solution",
    v: int = 5,
    time_limit_minutes: float = 5.0,
    **_: Any,
) -> Solution:
    """The 2025 code, unchanged except for the compatibility diff.

    `PARTIAL_REPRODUCTION`: original code, original defaults, Clarabel instead
    of MOSEK because MOSEK is not licensed here (`ASM-0002`).
    """

    import contextlib
    import io

    import cvxpy as cp

    from src.cg2026.cg_hist_compat import CG_LASSO_SOC1_v2

    if penalty.lambda_2 != 0:
        raise NotImplementedError("the historical CG is the LASSO one")
    tau = kappa = penalty.lambda_1 / 2.0
    buffer = io.StringIO()
    wall, cpu = time.perf_counter(), time.process_time()
    with contextlib.redirect_stdout(buffer):
        out = CG_LASSO_SOC1_v2(
            X,
            y,
            tau,
            kappa,
            solver=cp.CLARABEL,
            solver_params={},
            solver_verbose=False,
            add_constant=False,
            save_conv_info=True,
            v=v,
            v0=0,
            cg_lambda_tol=tol,
            cg_residuals_tol=tol,
            time_limit=time_limit_minutes,
            unboundedness_policy=policy,
        )
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    if out is None:
        # The historical code returns None when it cannot read a dual value --
        # "Primal-Dual gap is too big, no dual solution". A real failure mode,
        # recorded as one rather than dropped.
        return _finish(
            f"cg_hist_{policy}",
            np.zeros(X.shape[1]),
            X,
            y,
            penalty,
            tol,
            wall,
            cpu,
            status="no_dual_solution",
        )
    beta, _xi, _a, _b, _value, _tmin, _tpmin, info = out
    hit_limit = (wall / 60.0) >= time_limit_minutes
    return _finish(
        f"cg_hist_{policy}",
        beta,
        X,
        y,
        penalty,
        tol,
        wall,
        cpu,
        status="time_limit" if hit_limit else "ok",
        iterations=int(info["k"]),
        detail={
            "k_virtual": int(info["k_v"]),
            "columns_generated": int(info["k_v"]) * v,
            "lagrangian_values_recorded": sum(
                1 for value in info["lagrangian_values"] if value is not None
            ),
        },
    )


REGISTRY = {
    "sklearn": solve_sklearn,
    "celer": solve_celer,
    "skglm": solve_skglm,
    "lars": solve_lars,
    "conic_reduced": solve_conic_reduced,
    "conic_thesis": solve_conic_thesis,
    "cg_hist": solve_cg_hist,
}
