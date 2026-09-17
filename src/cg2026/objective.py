"""One objective convention, and the conversions into every solver's own.

**The convention.** Everything in this project is stated as

```text
F(beta) = ||y - X beta||_2^2 + lambda_1 ||beta||_1 + lambda_2 ||beta||_2^2
```

with no ``1/n`` and no ``1/2``. That is the convention the 2023 thesis uses --
its conic model minimises ``xi^2 + tau e'z + kappa e'u`` with ``xi >= ||y -
X beta||_2``, and its poster states the LASSO equivalence as ``lambda =
2*sqrt(tau*kappa)``.

**Why this module exists at all.** Every solver compared here minimises
something else. scikit-learn and skglm minimise ``(1/(2n))||y - X beta||^2 +
alpha ||beta||_1``; celer follows scikit-learn; a conic model written directly
minimises the convention above. A factor of ``2n`` between two of them is the
difference between comparing two solvers and comparing two problems, and a
benchmark that gets it wrong produces a clean, plausible, meaningless table.

So the conversions are functions, in one place, with tests that assert the
*objective value* agrees -- not that the formula looks right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A coefficient below this fraction of the largest one is numerical dust.
#:
#: One constant, used by the KKT measure and by the support count, because a
#: benchmark whose "support size" column and whose "optimality" column disagree
#: about what zero means is a benchmark with two answers. Calibrated from a
#: measured tolerance ladder: an interior-point solve at a certified relative
#: gap of 1e-3 leaves dust up to 7e-5 relative, so anything tighter than this
#: reports that solver as having selected every feature and as never having
#: converged. The cost is that a genuinely selected coefficient smaller than
#: 1e-5 of the largest is counted as zero, which for a sparse-regression
#: benchmark is the right trade and is stated rather than assumed.
SUPPORT_RTOL = 1e-5


@dataclass(frozen=True, slots=True)
class Penalty:
    """The project's own penalty parameters."""

    lambda_1: float
    lambda_2: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_1 < 0 or self.lambda_2 < 0:
            raise ValueError("penalties are non-negative")

    @classmethod
    def from_thesis(cls, tau: float, kappa: float, theta: float = 0.0) -> Penalty:
        """The 2023 parameterisation: ``lambda_1 = 2 sqrt(tau kappa)``.

        ``theta`` is the **whiteboard §3.4 and `models.py::SOCP_v2`**
        coefficient on ``(1/2)||beta||_2^2``, so ``lambda_2 = theta/2``.

        It is **not** the April 2025 note's ``lambda_2``, which multiplies
        ``||beta||_2^2`` with no one-half and must be passed as ``lambda_2``
        directly. An earlier version of this docstring attributed ``theta`` to
        that note, which would have cost a reader who followed it exactly half
        the intended ridge penalty -- a factor-of-two error, in the one
        function whose entire job is preventing them. Found by an independent
        review; no caller existed, so it was a trap rather than a bug.
        """

        return cls(lambda_1=2.0 * float(np.sqrt(tau * kappa)), lambda_2=theta / 2.0)

    def sklearn_alpha(self, n_samples: int) -> float:
        """``alpha`` for ``sklearn.linear_model.Lasso`` / ``ElasticNet``.

        scikit-learn minimises

        ```text
        (1/(2n))||y - X b||^2 + alpha*l1_ratio*||b||_1
                              + (alpha*(1 - l1_ratio)/2)*||b||_2^2
        ```

        Multiplying by ``2n``:

        ```text
        ||y - X b||^2 + 2n*alpha*l1_ratio*||b||_1
                      + n*alpha*(1 - l1_ratio)*||b||_2^2
        ```

        so ``lambda_1 = 2n*alpha*l1_ratio`` and
        ``lambda_2 = n*alpha*(1 - l1_ratio)``. Adding the two solved forms
        eliminates ``l1_ratio`` and gives the line below. celer and skglm follow
        scikit-learn's convention, so the same conversion serves all three.
        """

        return self.lambda_1 / (2.0 * n_samples) + self.lambda_2 / float(n_samples)

    def sklearn_l1_ratio(self, n_samples: int) -> float:
        """``l1_ratio`` to go with :meth:`sklearn_alpha`."""

        alpha = self.sklearn_alpha(n_samples)
        if alpha == 0.0:
            return 1.0
        return self.lambda_1 / (2.0 * n_samples * alpha)

    def value(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """``F(beta)`` in this project's convention. The only objective reported."""

        residual = y - X @ beta
        return float(
            residual @ residual + self.lambda_1 * np.abs(beta).sum() + self.lambda_2 * (beta @ beta)
        )


def lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """The smallest ``lambda_1`` whose LASSO solution is exactly zero.

    ``beta = 0`` is optimal iff ``0`` is in the subdifferential at zero, i.e.
    ``|2 X' y|_inf <= lambda_1``. Used to place every experiment on a scale
    that means the same thing across instances: ``lambda_1 = ratio *
    lambda_max`` selects a comparable amount of sparsity whatever the data's
    units are, which a fixed absolute ``lambda_1`` does not.
    """

    return float(2.0 * np.abs(X.T @ y).max())


def kkt_violation(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    penalty: Penalty,
    *,
    support_rtol: float = SUPPORT_RTOL,
) -> float:
    r"""The distance from zero to the subdifferential, in absolute units.

    For ``F(beta) = ||y - Xb||^2 + l1||b||_1 + l2||b||^2`` with
    ``g = -2X'(y - Xb) + 2 l2 b``, the optimality condition at coordinate ``i``
    is ``0 in g_i + l1 d|b_i|``, so the violation is

    ```text
    b_i != 0 :  |g_i + l1 sign(b_i)|
    b_i == 0 :  (|g_i| - l1)_+
    ```

    **``support_rtol`` is not a convenience, it is a correctness requirement,
    and getting it wrong invalidates a whole benchmark.** The two branches
    disagree by about ``l1`` at ``b_i = 0``, and an interior-point solver never
    returns an exact zero -- it returns ``6e-13``. Testing ``beta != 0``
    therefore takes the *support* branch for every coordinate of a conic
    solution and reports a violation of roughly ``l1``, which reads as "this
    solver never converges". Measured here: a Clarabel solve whose objective
    agreed with LARS to seven digits and whose duality gap was ``2e-4`` scored
    a KKT violation of ``27.95`` against ``l1 = 27.87``.

    So a coefficient counts as zero when it is below ``support_rtol`` times the
    largest coefficient.

    **The default was 1e-9 and that was calibrated for one tolerance only.** An
    independent review measured the same conic solve over a tolerance ladder
    and found the dust sitting between 1e-7 and 1e-4 relative at loose
    settings, so at ``tol = 1e-5`` -- a certified relative gap of 2.8e-03 --
    the measure still reported ``kkt = 24.42`` against ``l1 = 21.18``:
    precisely the "this solver never converges" artefact this parameter exists
    to prevent, in the exact regime the paragraph above describes. The default
    is 1e-5, which covers the ladder. It is still a threshold and it is still
    the reason :func:`relative_gap` rather than this is the primary measure.

    For a threshold-free measure, use :func:`duality_gap`, which is also a
    certificate. This one is kept because it is the quantity the pricing rule
    is about (see `src/cg2026/pricing.py`), and because it localises the
    violation to a coordinate.
    """

    grad = -2.0 * (X.T @ (y - X @ beta)) + 2.0 * penalty.lambda_2 * beta
    largest = float(np.abs(beta).max()) if beta.size else 0.0
    nonzero = np.abs(beta) > support_rtol * largest
    violation = 0.0
    if nonzero.any():
        violation = float(np.abs(grad[nonzero] + penalty.lambda_1 * np.sign(beta[nonzero])).max())
    if (~nonzero).any():
        violation = max(
            violation,
            float(np.maximum(np.abs(grad[~nonzero]) - penalty.lambda_1, 0.0).max()),
        )
    return violation


def relative_gap(X: np.ndarray, y: np.ndarray, beta: np.ndarray, penalty: Penalty) -> float:
    """``duality_gap / |primal|`` -- the primary accuracy measure of this project.

    Threshold-free, certified, and comparable across a first-order method, an
    interior-point method and a decomposition. Every solver in the benchmark is
    run over a ladder of its own tolerances and this is what is read off, so
    "time to reach accuracy epsilon" is answered by the data rather than
    decided by each solver's idea of what its `tol` means.
    """

    primal = penalty.value(X, y, beta)
    if primal == 0.0:
        return 0.0
    return duality_gap(X, y, beta, penalty) / abs(primal)


def duality_gap(X: np.ndarray, y: np.ndarray, beta: np.ndarray, penalty: Penalty) -> float:
    """A certified gap for the LASSO case, from a rescaled dual point.

    Only for ``lambda_2 == 0``. The dual of ``min ||y - Xb||^2 + l1||b||_1`` is

    ```text
    max_theta  ||y||^2 - ||y - theta||^2   s.t.  ||X'theta||_inf <= l1/2
    ```

    and the residual ``r = y - X beta`` is dual feasible only at the optimum, so
    it is rescaled by ``min(1, (l1/2)/||X'r||_inf)`` -- the standard
    construction. The returned gap is a genuine certificate: the primal optimum
    lies within it.
    """

    if penalty.lambda_2 != 0.0:
        raise ValueError("this gap is derived for the LASSO case only")
    if penalty.lambda_1 == 0.0:
        # `scale` would be 0, `theta` would be 0, and the gap would be the
        # whole primal -- a true certificate ("p* is somewhere in [0, F]") and
        # a useless one, which would read as 100 % unconverged in a benchmark
        # table at the exact OLS solution. Refused rather than reported.
        raise ValueError(
            "with lambda_1 = 0 the rescaled dual point is the origin and the "
            "gap degenerates to the primal value; use the KKT violation"
        )
    residual = y - X @ beta
    correlation = float(np.abs(X.T @ residual).max())
    scale = 1.0
    if correlation > 0.0:
        scale = min(1.0, (penalty.lambda_1 / 2.0) / correlation)
    theta = scale * residual
    primal = penalty.value(X, y, beta)
    # `2 y'theta - theta'theta`, not `||y||^2 - ||y - theta||^2`.
    #
    # They are identically equal and the second cancels catastrophically. The
    # gap is exactly zero at the optimum, so the subtraction of two nearly
    # equal large numbers is unopposed there, and the certificate goes
    # NEGATIVE -- which is the one thing a certificate must never do.
    #
    # Measured by an independent adversarial review, over 200 000 exactly
    # optimal points with ||y|| in [1e7, 1e11]: the cancelling form returned a
    # negative gap in 49.9 % of them, worst -3.35e6 (relative -1.26e-05). This
    # form returned a negative in 8.3 %, worst -1.22e-04 -- a 10^10 reduction,
    # with the residual being the primal's own rounding and irreducible.
    #
    # Latent rather than live here, because `data.py` standardises by default
    # so ||y||^2 = n. It is fixed anyway: `relative_gap` is this project's
    # primary accuracy measure and a public function has no idea who will call
    # it with what.
    dual = float(2.0 * (y @ theta) - theta @ theta)
    return primal - dual
