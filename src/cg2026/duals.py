"""Capture the dual points the implemented method actually visits.

Every claim about the pricing subproblem is a claim about ``psi``, and ``psi``
is not a free variable: it is the optimal second-order-cone dual of the
restricted master at some iteration. A statement that holds for *some* ``psi``
in R^n and a statement that holds for the ``psi`` this method reaches are
different statements, and the second is the one that bears on the method.

That distinction is exactly what the autonomous cycle's own proposal flagged
about `HYP-0004`: its existence clause is "there exist **reachable** dual
points at which the unit-ball value exceeds the primal optimum", and
"reachable" is unfalsifiable until it is pinned to something. This module pins
it to *the duals the implemented method produces on a given instance*.

The capture is a wrapper around ``cvxpy.Problem.solve`` rather than a
re-implementation of the master, for the same reason the profiler is: a
re-implementation would be a claim about a different algorithm.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VisitedDual:
    """One master dual, with everything needed to judge the pricing at it."""

    iteration: int
    psi: np.ndarray
    mu: float


@dataclass
class DualTrace:
    """Every dual one run of the method visited, in order."""

    duals: list[VisitedDual] = field(default_factory=list)
    iterations: int = 0
    status: str = "ok"


def capture_duals(
    X: np.ndarray,
    y: np.ndarray,
    lambda_1: float,
    *,
    policy: str = "v_solution",
    v: int = 5,
    tol: float = 1e-6,
    time_limit_minutes: float = 2.0,
) -> DualTrace:
    """Run the historical method and record the SOC dual at every master solve.

    The dual of interest is the one attached to ``cp.SOC(xi, y - X beta_k pi)``
    -- the constraint the Lagrangian relaxation relaxes. It is identified by
    shape: its vector part has one entry per observation. Identifying it by
    position in ``problem.constraints`` would break the moment the historical
    code reorders its constraint list, and identifying it by name is not
    possible because it has none.
    """

    import cvxpy as cp

    from src.cg2026 import cg_hist_compat

    trace = DualTrace()
    original_solve = cp.Problem.solve

    def timed_solve(self, *args, **kwargs):
        result = original_solve(self, *args, **kwargs)
        for constraint in self.constraints:
            if not isinstance(constraint, cp.constraints.second_order.SOC):
                continue
            dual = constraint.dual_value
            if dual is None or len(dual) != 2:
                continue
            mu_part, psi_part = dual
            psi = np.asarray(psi_part, dtype=float).ravel()
            if psi.size != y.size:
                continue
            trace.duals.append(
                VisitedDual(
                    iteration=len(trace.duals),
                    psi=psi.copy(),
                    mu=float(np.asarray(mu_part, dtype=float).ravel()[0]),
                )
            )
            break
        return result

    tau = kappa = lambda_1 / 2.0
    cp.Problem.solve = timed_solve
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = cg_hist_compat.CG_LASSO_SOC1_v2(
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
    finally:
        cp.Problem.solve = original_solve

    if out is None:
        trace.status = "no_dual_solution"
    else:
        trace.iterations = int(out[7]["k"])
    return trace
