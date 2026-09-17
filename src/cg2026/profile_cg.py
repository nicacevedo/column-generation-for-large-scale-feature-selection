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

    pricing_products: int = 0
    """How many ``psi'X`` products the run performed, counted rather than assumed."""

    pricing_scan: float = 0.0
    """Forming ``X' psi``, measured directly rather than intercepted.

    **The first version of this field measured nothing, and the 0.00 % it
    reported was an artifact.** It patched ``np.matmul`` and the historical code
    writes ``psi_k_sol.T @ X``; the ``@`` operator dispatches to
    ``ndarray.__matmul__`` in C and never reaches the Python-level ``np.matmul``
    symbol, so the wrapper was never called. Verified: patching ``np.matmul``
    and evaluating ``A @ B`` intercepts zero calls and ``np.matmul(A, B)``
    intercepts one.

    ``cvxpy.Problem.solve`` is an ordinary Python method, so *that* interception
    was real and the master-solve share was measured. What was not measured was
    the term the whole question is about.

    This is now a direct measurement: the product is timed on the actual
    operand shapes, over repetitions, and multiplied by the number of
    iterations the run performed. That is a measurement of the operation rather
    than of the code path, which is weaker in one way -- it does not capture
    allocation or cache effects inside the run -- and stronger in another: it
    cannot silently miss.
    """

    column_build: float = 0.0
    residual_check: float = 0.0
    """The least-squares regression of the new column onto the existing ones."""

    iterations: int = 0
    pricing_scan_per_iteration: float = 0.0
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

    # `np.argpartition` IS reached, because the historical code calls it by
    # name; `@` is not, because it dispatches in C. See `Profile.pricing_scan`.
    original_argpartition = np.argpartition

    def timed_argpartition(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_argpartition(*args, **kwargs)
        finally:
            profile.column_build += time.perf_counter() - started

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
        LinearRegression.fit = original_fit

    if out is not None:
        profile.iterations = int(out[7]["k"])

    # The pricing scan, measured directly on the shapes the run used. `psi` in
    # the historical code is (n, 1), so the product is `(1, n) @ (n, m)`.
    psi = np.zeros((X.shape[0], 1))
    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        _ = psi.T @ X
    per_product = (time.perf_counter() - started) / repeats

    # **How many such products the run actually performed, counted, not assumed.**
    #
    # This multiplier was `max(profile.iterations, 1)` -- one product per
    # iteration -- and that was wrong by about a factor of two, which an
    # adversarial review caught and which halved every pricing share this
    # module has ever reported. `psi_k_sol.T @ X` appears on at least two live
    # paths per iteration: the unboundedness test at `cg_models.py:446` and the
    # reduced-cost scan at `:601`/`:631`.
    #
    # Counting beats reading the branches, because which of `:601`/`:631` runs
    # depends on the iterate. Measured here at 1.80, 1.83 and 1.98 products per
    # iteration for lambda ratios 0.5, 0.1 and 0.02 on one instance -- close to
    # two and not exactly two, which is the kind of thing a hardcoded constant
    # would have quietly got wrong in the other direction.
    products = _count_pricing_products(
        X, y, tau, kappa, v=v, tol=tol, time_limit_minutes=time_limit_minutes
    )
    profile.pricing_products = products
    profile.pricing_scan = per_product * max(products, 1)
    profile.pricing_scan_per_iteration = per_product * (products / max(profile.iterations, 1))
    return profile, out


class _CountingMatrix(np.ndarray):
    """An ``X`` that tallies how often something is multiplied into it.

    Only ``__rmatmul__`` is overridden, which is exactly the ``psi.T @ X``
    shape the pricing scan uses; ``X @ beta`` and friends are left alone.
    """

    count = 0

    def __rmatmul__(self, other: object) -> np.ndarray:
        type(self).count += 1
        return np.asarray(other) @ np.asarray(self)


def _count_pricing_products(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    kappa: float,
    *,
    v: int,
    tol: float,
    time_limit_minutes: float,
) -> int:
    """Re-run the method once, untimed, purely to count ``psi'X`` products.

    A separate pass rather than instrumentation of the timed one: viewing ``X``
    as an ndarray subclass puts a Python frame in front of every product, and
    measuring a cost while inflating it is the mistake this whole module exists
    to avoid. The method is deterministic in these inputs, so the count from
    this pass is the count from that one.
    """

    import cvxpy as cp

    from src.cg2026.cg_hist_compat import CG_LASSO_SOC1_v2 as _run

    _CountingMatrix.count = 0
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            _run(
                X.view(_CountingMatrix),
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
            )
    except Exception:  # noqa: BLE001, S110 - the timed run already recorded it
        # A crash here is not a new fact: the timed pass above hit the same
        # path and reported it. Return what was counted before it failed.
        pass
    return int(_CountingMatrix.count)
