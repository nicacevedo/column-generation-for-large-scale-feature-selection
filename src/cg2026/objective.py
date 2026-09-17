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

        ``theta`` is the 2025 notes' Elastic-Net coefficient on ``(1/2)||beta||_2^2``,
        so ``lambda_2 = theta/2``.
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


def kkt_violation(X: np.ndarray, y: np.ndarray, beta: np.ndarray, penalty: Penalty) -> float:
    """The largest violation of the optimality conditions, in absolute units.

    For ``F(beta) = ||y - Xb||^2 + l1||b||_1 + l2||b||^2`` the subdifferential
    condition at coordinate ``i`` with ``g = -2X'(y - Xb) + 2*l2*b`` is

    ```text
    beta_i != 0 :  g_i + l1*sign(beta_i) == 0
    beta_i == 0 :  |g_i| <= l1
    ```

    so this returns ``max_i`` of ``|g_i + l1 sign(b_i)|`` on the support and
    ``(|g_i| - l1)_+`` off it. Absolute rather than relative, and the caller
    normalises: a relative measure needs a denominator, and every candidate
    denominator (``lambda_1``, ``||X'y||_inf``, ``F``) flatters a different
    solver.

    This is the **one** optimality measure every solver in this project is
    compared at, because "converged" means different things to scikit-learn's
    duality gap, celer's, and an interior-point solver's tolerance.
    """

    grad = -2.0 * (X.T @ (y - X @ beta)) + 2.0 * penalty.lambda_2 * beta
    nonzero = beta != 0
    violation = 0.0
    if nonzero.any():
        violation = float(np.abs(grad[nonzero] + penalty.lambda_1 * np.sign(beta[nonzero])).max())
    if (~nonzero).any():
        violation = max(
            violation,
            float(np.maximum(np.abs(grad[~nonzero]) - penalty.lambda_1, 0.0).max()),
        )
    return violation


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
    residual = y - X @ beta
    correlation = float(np.abs(X.T @ residual).max())
    scale = 1.0
    if correlation > 0.0:
        scale = min(1.0, (penalty.lambda_1 / 2.0) / correlation)
    theta = scale * residual
    primal = penalty.value(X, y, beta)
    dual = float(y @ y - (y - theta) @ (y - theta))
    return primal - dual
