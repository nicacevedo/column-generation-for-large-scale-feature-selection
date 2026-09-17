r"""The pricing subproblem, derived from the model rather than from the notes.

Everything here is stated under one convention, `ASM-0001`:

```text
(P)   min_beta  ||y - X beta||_2^2 + l1 ||beta||_1 + l2 ||beta||_2^2
```

The 2023 conic model is

```text
(SOCP) min_{beta,z,u,xi}  xi^2 + tau e'z + kappa e'u
       s.t.  ||y - X beta||_2 <= xi
             beta_i^2 <= z_i u_i          for all i
             z, u >= 0
```

and `min_{z,u >= 0, b^2 <= zu} tau z + kappa u = 2 sqrt(tau kappa) |b|` by
AM-GM, attained at `z = |b| sqrt(kappa/tau)`, `u = |b| sqrt(tau/kappa)`. So
(SOCP) is (P) with `l1 = 2 sqrt(tau kappa)` and `l2 = 0`: an exact
reformulation of the LASSO, not a relaxation of anything.

Two precisions an independent review insisted on. Attainment fails when
*exactly one* of `tau, kappa` is zero -- the infimum is still zero but is
approached only as `z -> inf`, and the stated point is `0/0`; unreachable in
practice because `solvers.py` sets `tau = kappa = l1/2`. And `l2 > 0` is **not**
"the 2025 Elastic-Net extension": `theta` enters `models.py` at commit 471a414
on **2024-01-31** and `CG_SOC1_ElasticNet` at c3c223d on 2024-02-03. In an
archaeology this misdated it by fifteen months.

**The Lagrangian relaxation.** Relax only the residual cone, with dual
`(mu, psi)` in the second-order cone (`||psi||_2 <= mu`):

```text
L(beta, xi; mu, psi) = xi^2 + l1||beta||_1 + l2||beta||_2^2
                       - mu xi - psi'(y - X beta)
```

`min_xi (xi^2 - mu xi) = -mu^2/4` at `xi = mu/2`, so the dual function is

```text
g(mu, psi) = -mu^2/4 - psi'y + min_beta [ psi'X beta + l1||beta||_1
                                                     + l2||beta||_2^2 ]
```

The bracket separates over coordinates. With `a = X'psi`, each coordinate
solves

```text
(L_i)  min_{b}  a_i b + l1 |b| + l2 b^2
```

which is the function `coordinate_optimum` below. Every proposition in this
module is about that one-dimensional problem, and every one of them is checked
numerically in `tests/test_pricing_math.py` -- the derivations here are the
claim, the tests are the evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CoordinateOptimum:
    """The solution of ``min_b a b + l1|b| + l2 b^2``, including when there is none."""

    bounded: bool
    minimiser: float | None
    """``None`` when unbounded, or when bounded with a non-unique minimiser."""

    value: float
    """``-inf`` when unbounded."""

    unique: bool
    recession_direction: float
    """A direction of non-increase: ``-sign(a)`` when one exists, else ``0``.

    Non-zero in two distinct cases and they must not be confused. When
    ``|a| > l1`` it is a direction of *strict* decrease and the column to add.
    When ``|a| == l1`` it is a direction of *zero* change -- a legitimate
    recession direction of the level set that buys the master nothing.
    """


def coordinate_optimum(a: float, l1: float, l2: float) -> CoordinateOptimum:
    r"""Solve ``min_b a b + l1 |b| + l2 b^2`` exactly, equality case included.

    **Proposition P4 (l2 > 0).** The objective is strongly convex, so the
    minimiser is unique and is the soft threshold

    ```text
    b* = -sign(a) (|a| - l1)_+ / (2 l2)
    ```

    with optimal value ``-(|a| - l1)_+^2 / (4 l2)``.

    *Derivation.* For ``b > 0`` the derivative is ``a + l1 + 2 l2 b``, zero at
    ``b = -(a + l1)/(2 l2)``, which is positive iff ``a < -l1``. For ``b < 0``
    it is ``a - l1 + 2 l2 b``, zero at ``b = (l1 - a)/(2 l2)``, negative iff
    ``a > l1``. And ``0`` is optimal iff ``0 in a + l1[-1,1]``, i.e.
    ``|a| <= l1``. The three cases are exactly the displayed formula.
    Substituting with ``d = (|a| - l1)/(2 l2)`` gives
    ``-(|a| - l1) d + l2 d^2 = -(|a| - l1)^2/(4 l2)``.

    This **confirms** the April 2025 note, which writes the same quantity as
    ``(-a + sign(a) l1)/(2 l2)`` -- algebraically identical when ``|a| > l1``.

    It **refutes** the whiteboard's ``beta* = -(1/theta)(l1 + a)``, which is
    the ``a < -l1`` branch applied unconditionally: it has the wrong sign for
    ``a > l1`` and is non-zero on ``|a| <= l1``, where the true minimiser is
    ``0``. The whiteboard states its own assumption -- "assuming the equality
    holds in the conic constraints on the optimal point" -- and that assumption
    is exactly what fails at ``b = 0``.

    **Proposition P2/P3 (l2 == 0).** The objective is piecewise linear.

    - ``|a| < l1`` : bounded, unique minimiser ``0``, value ``0``.
    - ``|a| == l1``: bounded, value ``0``, and **every** ``b`` with
      ``sign(b) = -sign(a)`` attains it. The minimiser is a ray, not a point.
    - ``|a| > l1`` : unbounded below along ``b = -t sign(a)``, ``t -> inf``,
      where the objective is ``-(|a| - l1) t``.

    The equality case is the one the implementation gets wrong, and it is not a
    corner case: at the LASSO optimum the KKT conditions make ``|a_i| = l1``
    *exactly* on the support. A boundedness test written as
    ``|a| - l1 > -tol`` therefore reports "unbounded" at the solution, for
    every support coordinate, forever.
    """

    if l1 < 0 or l2 < 0:
        raise ValueError("penalties are non-negative")
    excess = abs(a) - l1
    if l2 > 0:
        step = max(excess, 0.0) / (2.0 * l2)
        minimiser = -math.copysign(step, a) if step > 0 else 0.0
        return CoordinateOptimum(
            bounded=True,
            minimiser=minimiser,
            value=-(max(excess, 0.0) ** 2) / (4.0 * l2),
            unique=True,
            recession_direction=0.0,
        )
    if excess > 0:
        return CoordinateOptimum(
            bounded=False,
            minimiser=None,
            value=-math.inf,
            unique=False,
            recession_direction=-math.copysign(1.0, a),
        )
    if excess == 0 and a != 0:
        return CoordinateOptimum(
            bounded=True,
            minimiser=None,
            value=0.0,
            unique=False,
            recession_direction=-math.copysign(1.0, a),
        )
    return CoordinateOptimum(
        bounded=True, minimiser=0.0, value=0.0, unique=True, recession_direction=0.0
    )


def pricing_is_bounded(X: np.ndarray, psi: np.ndarray, l1: float, l2: float) -> bool:
    r"""**Proposition P2.** Bounded iff ``l2 > 0`` or ``||X'psi||_inf <= l1``.

    And the right-hand side is **exactly** the feasibility constraint of the
    dual of the full problem. The dual of (P) with ``l2 = 0``, in this
    module's ``psi`` parameterisation, is

    ```text
    max_{psi}  -||psi||_2^2/4 - psi'y    s.t.  ||X'psi||_inf <= l1
    ```

    (complete the square: ``-||psi||^2/4 - psi'y = ||y||^2 - ||psi + 2y||^2/4``,
    and with ``theta = -psi/2`` this is the textbook LASSO dual
    ``max_theta ||y||^2 - ||y - theta||^2`` subject to
    ``||X'theta||_inf <= l1/2``).

    So **pricing boundedness and full-problem dual feasibility are the same
    condition**, and this is not a coincidence to be proved case by case: the
    dual feasible set of a Lagrangian dual *is* by definition the set where the
    inner minimisation is finite. The content of the proposition is the
    identification of that set with ``||X'psi||_inf <= l1``, which is the
    coordinate analysis in :func:`coordinate_optimum`.

    **This function is the wrong optimality test when ``l2 > 0``.** It returns
    ``True`` unconditionally there, because the pricing really is bounded --
    and bounded pricing is then no evidence of anything. The certificate for
    ``l2 > 0`` is ``violations(X, psi, l1) == 0``; see
    :func:`restricted_master_is_optimal` for why, and for a measured
    counterexample to using boundedness instead.
    """

    if l2 > 0:
        return True
    return bool(np.abs(X.T @ psi).max() <= l1)


def restricted_master_is_optimal(
    X: np.ndarray,
    psi: np.ndarray,
    l1: float,
    l2: float = 0.0,
    *,
    working_set: Iterable[int] | None = None,
) -> bool:
    r"""**The corollary**, with the hypotheses that make it true.

    This is the load-bearing claim of the whole decomposition, and an earlier
    version of it -- written as a paragraph inside
    :func:`pricing_is_bounded`'s docstring, with no hypotheses and no scope --
    was **wrong in two ways that an independent adversarial review found with
    counterexamples**, and then wrong in a third way in the first attempt to
    repair it. All three are tests in ``tests/test_pricing_math.py``. A claim
    this central should carry the evidence against its wrong forms.

    **Corollary.** Let ``S`` be the working set, ``B_k = {B^k pi : pi in Pi}``
    the master's reachable set for ``beta``, and ``psi^k`` its optimal
    second-order-cone dual. Assume

    1. ``B_k = span{e_i : i in S}`` or the cone it generates -- satisfied by
       ``Pi = R^{n_k}`` (the default) and by ``Pi = R_+^{n_k}``
       (``pos_linear_comb=True``), and **not** by ``e'pi = 1``
       (``sum_1_comb=True``); and
    2. every violated coordinate is already inside the working set:

       ```text
       {i : (|(X'psi^k)_i| - l1)_+ > 0}  is a subset of  S
       ```

    Then ``RMP_k = p*``: the restricted master is already optimal for the full
    problem. This holds for ``l2 = 0`` and for ``l2 > 0`` alike.

    *Proof.* Strong duality for the restricted master gives
    ``g_R(psi^k) = RMP_k``, where
    ``g_R(psi) = d(psi) + min_{beta in B_k} [psi'X beta + l1||beta||_1 + l2||beta||^2]``
    and ``d(psi) = -||psi||^2/4 - psi'y``. The full inner minimum separates
    over coordinates and is attained at the soft threshold
    ``beta*_i = -sign(a_i)(|a_i| - l1)_+/(2 l2)`` (P4), whose support is exactly
    ``{i : v_i > 0}``. Hypothesis 2 puts that point inside ``B_k``, so the
    restricted and full inner minima coincide and ``g_R(psi^k) = g(psi^k)``.
    Weak duality gives ``g(psi^k) <= p*``, and ``RMP_k >= p*`` because the
    master is a restriction. Hence ``p* <= RMP_k = g(psi^k) <= p*``. For
    ``l2 = 0`` the soft threshold degenerates: the infimum is ``0`` when no
    coordinate violates and ``-inf`` when one does, so hypothesis 2 reduces to
    ``{i : v_i > 0}`` being **empty**, because a violated coordinate inside
    ``S`` would make the master itself unbounded. (Verified: at a master
    optimum no coordinate of ``S`` ever violates.)

    **Why hypothesis 1 is needed.** With ``sum_1_comb=True`` -- an option the
    historical code offers -- ``B_k`` is an **affine** set, the argument above
    fails at "the restricted and full inner minima coincide", and:

    ```text
    X = I_2, y = (1,1), l1 = 20        p* = 2.0   (beta* = 0, l1 > lambda_max)
    RMP with e'pi = 1                  20.5       at beta^k = (0.5, 0.5)
    psi^k = (-1,-1),  ||X'psi^k||_inf = 1 <= 20   ->  no violated coordinate
    ```

    No violation, and the restricted master is **ten times worse** than the
    full optimum. Weak duality survives; the corollary does not. The Slater
    point the first version cited, ``beta = 0`` with ``xi`` large, is not even
    feasible for that master.

    **Why "the pricing is bounded" is the wrong test when ``l2 > 0``.**
    :func:`pricing_is_bounded` returns ``True`` unconditionally there, because
    the pricing really is bounded -- so it certifies nothing. Measured,
    ``n=10, p=6, l1=2, l2=1``, ``p* = 191.606``: at a one-column master the
    pricing is "bounded", ``RMP = 362.640`` (89 % suboptimal) and
    ``d(psi) = 362.651``. Worse, at the *full* master ``d(psi) = 224.506``
    against ``p* = 191.606`` -- the quantity an earlier version called a valid
    lower bound sitting 17 % **above** the optimum.

    And note what hypothesis 2 looks like there: with ``l2 > 0`` the violated
    set is **non-empty at optimality** -- it is exactly the support of
    ``beta*`` -- so "no coordinate violates" would never fire. That is the
    third error, made while repairing the first two: an attempt to state the
    corollary as ``violations == 0`` for all ``l2``. Verified over three values
    of ``l2`` on a 40x10 instance: at ``l2 = 1`` and ``l2 = 5`` the violated set
    at optimality is ``{1,2,5,7,9}``, which is the support, and ``RMP = p*``
    exactly when that set is contained in ``S``.

    **A consequence worth stating, because it condemns the historical bound
    plot.** For ``l2 = 0`` and cone weights the bracket is positively
    homogeneous, so the restricted inner minimum is ``0`` or ``-inf`` and
    therefore ``d(psi^k) = RMP_k`` **identically, at every iteration**. The
    Lagrangian "lower bound" the historical code records is thus either
    ``-inf`` or *exactly equal to its own upper bound*. There is no
    intermediate value and no converging gap to plot. The "Bounds per
    Iteration" figure the code draws can only show two identical curves or one
    missing curve -- and it shows the second, because the pricing test never
    reports bounded (timeline §2025-D). This identity does **not** extend to
    ``l2 > 0``, where homogeneity fails: measured, ``d(psi) = 224.506`` against
    ``RMP = 191.606`` at the full master.

    **Answering the whiteboard's unanswered §2.4, "Why is the pricing bounded
    only at the final iteration?"** Because for ``l2 = 0`` under these
    hypotheses bounded pricing *is* the optimality certificate. It is not an
    inconvenience of the method; it is the method's stopping condition, and the
    implementation does not use it.
    """

    violated = set(np.flatnonzero(violations(X, psi, l1) > 0).tolist())
    if working_set is None:
        # No working set supplied: the only certificate available is the
        # `l2 = 0` one, which is that nothing violates at all. Returning that
        # for `l2 > 0` would be the third error above, so it is refused.
        if l2 > 0:
            raise ValueError(
                "with l2 > 0 the violated set is non-empty at optimality -- it "
                "is the support of beta* -- so 'nothing violates' never fires. "
                "Supply working_set; the certificate is that every violated "
                "coordinate is already in it."
            )
        return not violated
    return violated <= set(working_set)


def violations(X: np.ndarray, psi: np.ndarray, l1: float) -> np.ndarray:
    r"""``v_i = (|(X'psi)_i| - l1)_+`` -- the per-coordinate dual violation.

    **Proposition P3.** ``v_i > 0`` exactly on the coordinates whose pricing
    subproblem is unbounded, and the rate of decrease along the corresponding
    recession direction ``-sign(a_i) e_i`` is ``v_i``. So "the coordinates of
    largest violation" and "the columns with the most negative reduced cost,
    per unit step" are the same set in the same order.

    **Proposition P5 (the novelty gate).** The set ``{i : v_i > 0}`` is empty
    iff the restricted master is optimal for the full problem (by P2 and its
    corollary). ``v`` is therefore the standard **KKT / dual-constraint
    violation** of the LASSO, and selecting its largest entries is the
    **maximum-violation working-set rule**. In the textbook parameterisation
    ``theta = -psi/2`` it reads ``(|X_i'r| - l1/2)_+`` with ``r`` the residual
    of the restricted fit -- the quantity Osborne, Presnell and Turlach's
    active-set method adds on, that glmnet's strong rules approximate, and that
    Gap Safe screening, Blitz, Celer and skglm all compute.

    What this module does *not* claim is that the historical method is
    identical to any one of those. It claims the **selection rule** is the same
    rule. Whether the surrounding method differs enough to matter is an
    empirical question, and it is `HYP-0001`.
    """

    return np.maximum(np.abs(X.T @ psi) - l1, 0.0)


def dual_objective(psi: np.ndarray, y: np.ndarray) -> float:
    """``-||psi||^2/4 - psi'y``, the dual objective at ``psi`` (``l2 = 0``).

    Valid as a lower bound on the primal optimum **only** at dual-feasible
    ``psi``. Callers must check :func:`pricing_is_bounded` first; this function
    deliberately does not, because the whole point of
    :func:`unit_ball_pricing_value` is to show what happens when nobody does.
    """

    return float(-(psi @ psi) / 4.0 - psi @ y)


def unit_ball_pricing_value(X: np.ndarray, y: np.ndarray, psi: np.ndarray, l1: float) -> float:
    r"""The 2025 notes' unit-ball value: ``d(psi) + min_{||b||_2 <= 1} [...]``.

    **Proposition P6. This is not a lower bound on the primal optimum.**

    Adding a constraint to a *minimisation* raises its optimum, so

    ```text
    g_ball(psi) = d(psi) + min_{||b||_2 <= 1} [psi'X b + l1||b||_1]
                >= d(psi) + min_{b} [psi'X b + l1||b||_1] = g(psi)
    ```

    and ``g(psi) <= p*`` is the only inequality weak duality supplies. ``g_ball``
    sits on the wrong side of it. Concretely, when the pricing is unbounded
    ``g(psi) = -inf``, which is a valid and useless bound; ``g_ball(psi)`` is
    finite and has nothing keeping it below ``p*``.

    The primal would have to supply a matching bound ``||beta*||_2 <= 1`` for
    the restriction to be legitimate, and (P) supplies no such bound -- its
    solution's norm is whatever the data makes it. A *scaled* ball of radius
    ``R >= ||beta*||_2`` would be legitimate, and the notes use radius one.

    ``tests/test_pricing_math.py`` exhibits a two-line counterexample where
    ``g_ball`` exceeds the primal optimum by 43 %.

    **That exhibit is correct and it is the wrong one.** Its ``psi = -1.5`` is
    a point in R^n, and the algorithm's ``psi`` is not: it is the optimal cone
    dual of a restricted master, which SOC complementarity pins to
    ``psi^k = -2 r^k`` for the residual ``r^k``. On that one-dimensional
    instance the only reachable duals are ``-20`` and ``-1``, and ``-1.5`` is
    neither. An independent review made exactly that objection and then
    answered it: the *reachable* ``psi = -2y`` -- the dual of the empty initial
    master, which every run visits -- gives ``g_ball = 81`` against
    ``p* = 9.75``, **731 % above**, and the failure reproduces inside the
    historical code with ``unboundedness_policy="unit_ball"``, which records
    ``646.23`` as a "Lower Bound" against ``p* = 529.05`` at iteration 1.

    **And it is dimension-dependent, which neither the counterexample nor the
    review's sweep shows on its own.** Two measurements of this project
    disagreed until the regimes were compared:

    ```text
    p = 1, 5, 12     38 % of reachable unbounded duals exceed p*; worst 731 %
    p = 80, 400       0 of 30 reachable duals exceed p*
    ```

    The reason is scale: the ball has radius one whatever ``p`` is, so as ``p``
    grows the inner minimum ``min_{||b||<=1} [a'b + l1||b||_1]`` reaches
    roughly ``-||a||_2``, which grows with ``p`` and drags ``g_ball`` far below
    ``p*``. At small ``p`` the penalty is too small to do that and the raw
    ``d(psi)`` shows through. `scripts/adjudicate_pricing.py` measures both
    regimes and `docs/2026/UNIT_BALL_VERDICT.md` records the split.

    **This does not condemn the device.** The whiteboard §2.3 uses the same
    construction for a different purpose -- *"it is enough to provide an extreme
    ray of the pricing to the master: this can be done by normalizing it in
    some way"* -- and for that purpose it is sound: any direction of decrease is
    a legitimate column, and normalising an unbounded ray is a reasonable way to
    pick one. The 2025 notes' error is in what they *read off* it, not in what
    they add to the master.

    Verdict: **VALID_COLUMN_OR_DIRECTION_HEURISTIC_ONLY**, never a bound.
    """

    from scipy.optimize import minimize

    a = X.T @ psi

    def objective(b: np.ndarray) -> float:
        return float(a @ b + l1 * np.abs(b).sum())

    p = X.shape[1]
    best = 0.0  # b = 0 is feasible
    for start in (np.zeros(p), -np.sign(a) / max(math.sqrt(p), 1.0)):
        result = minimize(
            objective,
            start,
            constraints=[{"type": "ineq", "fun": lambda b: 1.0 - b @ b}],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.fun < best:
            best = float(result.fun)
    return dual_objective(psi, y) + best
