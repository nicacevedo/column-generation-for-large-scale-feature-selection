"""Profile the historical method before proposing to make it faster.

The 2025 notes end with a "Future Note: Do a profiling of the method, to see
what are the most time-consuming operations". It was never done, and the ideas
that followed -- minibatch estimation of ``psi'X``, GPU matrix-vector products,
approximate pricing -- all address the cost of forming ``X'psi``. `HYP-0007`
says that is the wrong term.

This measures it, per iteration, by instrumenting the real code path rather
than by re-implementing it: `cvxpy.Problem.solve` is wrapped to accumulate
solver time, and the pricing scan is timed directly. Everything the wrapper
does not attribute is reported as "other", because a profile whose parts do not
sum to the whole is a profile that hides its own gap.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Profile:
    """Where one run's wall time went, in seconds."""

    total: float = 0.0
    master_solve: float = 0.0
    """`cvxpy` solve calls: the restricted master, and the Lagrangian when it runs."""

    pricing_scan: float = 0.0
    """Forming ``X' psi`` and finding the largest violations."""

    column_build: float = 0.0
    residual_check: float = 0.0
    """The least-squares regression of the new column onto the existing ones."""

    iterations: int = 0
    solve_calls: int = 0
    per_call: list[float] = field(default_factory=list)

    @property
    def other(self) -> float:
        return max(
            0.0,
            self.total
            - self.master_solve
            - self.pricing_scan
            - self.column_build
            - self.residual_check,
        )

    def shares(self) -> dict[str, float]:
        if self.total <= 0:
            return {}
        return {
            "master_solve": self.master_solve / self.total,
            "pricing_scan": self.pricing_scan / self.total,
            "column_build": self.column_build / self.total,
            "residual_check": self.residual_check / self.total,
            "other": self.other / self.total,
        }


def profile_cg_hist(
    X: np.ndarray,
    y: np.ndarray,
    lambda_1: float,
    *,
    policy: str = "v_solution",
    v: int = 5,
    tol: float = 1e-6,
    time_limit_minutes: float = 5.0,
) -> tuple[Profile, object]:
    """Run the historical method with every timed region instrumented."""

    import cvxpy as cp
    from sklearn.linear_model import LinearRegression

    from src.cg2026 import cg_hist_compat

    profile = Profile()

    original_solve = cp.Problem.solve

    def timed_solve(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_solve(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            profile.master_solve += elapsed
            profile.solve_calls += 1
            profile.per_call.append(elapsed)

    # `psi' X` happens inside the historical code as a bare numpy expression, so
    # there is no function to wrap. What there *is* is `np.argpartition`, which
    # the code calls exactly once per column-generation step, and `matmul`. Both
    # are wrapped at the numpy level for the duration of the run.
    original_argpartition = np.argpartition
    original_matmul = np.matmul

    def timed_argpartition(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_argpartition(*args, **kwargs)
        finally:
            profile.column_build += time.perf_counter() - started

    def timed_matmul(a, b, *args, **kwargs):
        # Only the products that involve the full design matrix are pricing
        # work. `beta_k @ pi_k` and friends are master bookkeeping.
        involves_design = (
            getattr(a, "shape", None) == X.shape
            or getattr(b, "shape", None) == X.shape
            or getattr(a, "shape", None) == X.T.shape
            or getattr(b, "shape", None) == X.T.shape
        )
        started = time.perf_counter()
        try:
            return original_matmul(a, b, *args, **kwargs)
        finally:
            if involves_design:
                profile.pricing_scan += time.perf_counter() - started

    original_fit = LinearRegression.fit

    def timed_fit(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            profile.residual_check += time.perf_counter() - started

    tau = kappa = lambda_1 / 2.0
    buffer = io.StringIO()
    cp.Problem.solve = timed_solve
    np.argpartition = timed_argpartition
    np.matmul = timed_matmul
    LinearRegression.fit = timed_fit
    try:
        started = time.perf_counter()
        with contextlib.redirect_stdout(buffer):
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
        profile.total = time.perf_counter() - started
    finally:
        cp.Problem.solve = original_solve
        np.argpartition = original_argpartition
        np.matmul = original_matmul
        LinearRegression.fit = original_fit

    if out is not None:
        profile.iterations = int(out[7]["k"])
    return profile, out
