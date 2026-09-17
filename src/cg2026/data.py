"""Instances, including the historical ones, generated from a recorded spec.

Three families, and the reason there are three is that the historical family
alone cannot answer the question.

**`historical`** reproduces `synthetic_data.py::original_data_generation`. It is
here for continuity and because it is the design every 2023 claim was measured
on. Reading it is also the single most surprising thing in the archaeology:
`betas = np.random.uniform(-1, 1, n_var)` is drawn over **every** column, `y =
X @ betas` is computed, and only *afterwards* are 10 % of the columns
overwritten with noise. **The ground truth is dense.** 90 % of the coefficients
are non-zero, and the LASSO is being asked to recover a dense signal. That is
not a sparse-regression benchmark, and it explains why the historical claim is
about the regime "40 % of the features non-zero" -- which is where a
sparsity-exploiting solver has the least to exploit.

**`sparse`** is the regime modern sparse regression is about: `k` non-zero
coefficients out of `p`, `k << p`, with a controlled signal-to-noise ratio.

**`correlated`** reproduces `synthetic_data.py::correlated_data_generation`: an
equicorrelated design with `cov(i,j) = s_i s_j rho`, which is what makes
coordinate descent slow and is the historical work's own stress case.

Every instance is a pure function of its `InstanceSpec`, so the spec is the
whole provenance and no data file has to travel with a result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    """Everything that determines an instance, and nothing that does not."""

    family: str
    n: int
    p: int
    seed: int = 0
    #: `sparse` only: the number of non-zero coefficients.
    k: int = 10
    #: `sparse` only: ||X beta|| / ||noise||.
    snr: float = 5.0
    #: `correlated` only: the pairwise correlation.
    rho: float = 0.0
    #: `historical` only: the fraction of columns overwritten with noise.
    noise_fraction: float = 0.1
    #: `illcond` only: the condition number of the design's Gram matrix.
    condition: float = 1.0
    standardise: bool = True

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[
            :16
        ]

    @property
    def label(self) -> str:
        core = f"{self.family}-n{self.n}-p{self.p}"
        if self.family == "sparse":
            core += f"-k{self.k}-snr{self.snr:g}"
        if self.family == "correlated":
            core += f"-rho{self.rho:g}"
        if self.family == "illcond":
            core += f"-cond{self.condition:g}"
        return f"{core}-s{self.seed}"


def build(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(X, y, beta_true)``. ``beta_true`` is ``None`` when undefined."""

    builder = {
        "historical": _historical,
        "sparse": _sparse,
        "correlated": _correlated,
        "illcond": _illcond,
    }[spec.family]
    X, y, truth = builder(spec)
    if spec.standardise:
        # Exactly what `new_main.py` does, and what the whiteboard found helped
        # the conic solves: centre and scale every column, and centre and scale
        # the response. Columns with zero variance would divide by zero, so they
        # are left alone -- the historical code does not guard this and would
        # produce NaN.
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        X = (X - X.mean(axis=0)) / scale
        y_scale = y.std() or 1.0
        y = (y - y.mean()) / y_scale
    return np.ascontiguousarray(X), np.ascontiguousarray(y), truth


def _historical(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray, None]:
    """`synthetic_data.py::original_data_generation`, as it is written.

    Including the part that matters: `betas` is drawn over every column and `y`
    is formed *before* the noise columns are overwritten.

    **There is no ground-truth coefficient vector here, so none is returned.**
    An earlier version returned the drawn `beta` with the overwritten entries
    zeroed, and an adversarial review was right to call it inconsistent with
    `y`. It is not a near-miss. `y` was formed from the pre-overwrite columns,
    so after the overwrite the vector explains a relative residual of
    **0.78** at `n=2000, p=500, noise_fraction=0.1` -- it accounts for
    essentially none of `y` -- while carrying 450 of 500 nonzeros. Feeding that
    to `bench._support_scores` compares a 25-feature LASSO solution against a
    450-nonzero fiction and reports a recall near zero for every solver, which
    says nothing about any of them.

    Returning `None` is the honest encoding: `build` already declares the truth
    optional, and `bench` already skips the support scores when it is absent.
    `X` and `y` are untouched, because reproducing the historical generator is
    the entire point of this family.
    """

    rng = np.random.default_rng(spec.seed)
    X = np.empty((spec.n, spec.p))
    for j in range(spec.p):
        mean = rng.uniform(-10, 10)
        std = rng.uniform(0, 5)
        X[:, j] = rng.normal(mean, std, spec.n)
    X = np.round(X, 3)
    beta = rng.uniform(-1, 1, spec.p)
    y = np.round(X @ beta, 3)
    n_noise = round(spec.p * spec.noise_fraction)
    if n_noise:
        noise_cols = rng.choice(spec.p, n_noise, replace=False)
        X[:, noise_cols] = rng.normal(0, 1, (spec.n, n_noise))
    return X, y, None


def _sparse(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    X = rng.standard_normal((spec.n, spec.p))
    beta = np.zeros(spec.p)
    support = rng.choice(spec.p, spec.k, replace=False)
    beta[support] = rng.choice([-1.0, 1.0], spec.k) * rng.uniform(1.0, 3.0, spec.k)
    signal = X @ beta
    noise = rng.standard_normal(spec.n)
    noise *= np.linalg.norm(signal) / (spec.snr * np.linalg.norm(noise))
    return X, signal + noise, beta


def _correlated(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray, None]:
    """`synthetic_data.py::correlated_data_generation`, without the O(p^2) loop.

    The historical code builds the covariance with a Python double loop and then
    calls `multivariate_normal`, which is O(p^3) in the factorisation. The same
    equicorrelated matrix -- `cov(i,j) = s_i s_j rho`, `cov(i,i) = s_i^2` --
    factorises in closed form: with `z` and `w` standard normal,
    `sqrt(rho) w + sqrt(1 - rho) z_j`, scaled by `s_j`, has exactly that
    covariance. Seconds instead of hours at p = 5000.

    **Same distribution, NOT the same draw.** An earlier version of this
    docstring claimed "same seed semantics", which is false and was caught in
    review: `multivariate_normal` consumes the generator stream in a different
    order and quantity than the closed-form factorisation, so the same seed
    produces a different `X`. What is preserved is the law, not the sample.
    Nothing here reproduces a specific historical instance; for that, see
    :func:`load_historical_artifact`.

    As in :func:`_historical`, no ground truth is returned -- `y` is formed
    before the noise columns are overwritten, leaving the drawn coefficients
    explaining a relative residual of 0.26 at `n=2000, p=500, rho=0.5`.
    """

    if not 0.0 <= spec.rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    rng = np.random.default_rng(spec.seed)
    stds = rng.uniform(0, 5, size=spec.p)
    common = rng.standard_normal((spec.n, 1))
    own = rng.standard_normal((spec.n, spec.p))
    base = np.sqrt(spec.rho) * common + np.sqrt(1.0 - spec.rho) * own
    X = np.round(base * stds, 3)
    beta = rng.uniform(-1, 1, spec.p)
    y = np.round(X @ beta, 3)
    n_noise = round(spec.p * spec.noise_fraction)
    if n_noise:
        noise_cols = rng.choice(spec.p, n_noise, replace=False)
        X[:, noise_cols] = rng.normal(0, 1, (spec.n, n_noise))
    return X, y, None


def _illcond(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A design with a prescribed condition number and a sparse truth.

    Singular values decay geometrically from 1 to `1/condition`, so the Gram
    matrix's condition number is `condition^2`. Absent from the historical work
    entirely, and the brief asks for it because it is where a first-order method
    and an interior-point method diverge most.
    """

    rng = np.random.default_rng(spec.seed)
    q = min(spec.n, spec.p)
    left, _ = np.linalg.qr(rng.standard_normal((spec.n, q)))
    right, _ = np.linalg.qr(rng.standard_normal((spec.p, q)))
    singular = np.geomspace(1.0, 1.0 / spec.condition, q)
    X = (left * singular) @ right.T
    beta = np.zeros(spec.p)
    support = rng.choice(spec.p, spec.k, replace=False)
    beta[support] = rng.choice([-1.0, 1.0], spec.k) * rng.uniform(1.0, 3.0, spec.k)
    signal = X @ beta
    noise = rng.standard_normal(spec.n)
    noise *= np.linalg.norm(signal) / (spec.snr * np.linalg.norm(noise))
    return X, signal + noise, beta


HISTORICAL_ARTIFACT = Path("synthetic-data-error")


def load_historical_artifact() -> tuple[np.ndarray, np.ndarray]:
    """The one committed historical instance: `X (10000, 1000)`, 20 % noise.

    The others are gone. `synthetic-data-correlated/` and the two real-data
    directories are in `.gitignore` and are not on this machine, so the
    historical real-data experiment cannot be reproduced at all -- `usa.csv`,
    which *is* committed, is a scraped list of ticker symbols and industries,
    not the regression matrix.
    """

    X = np.loadtxt(HISTORICAL_ARTIFACT / "synthetic_X_1000_10000_e20.csv", delimiter=",")
    y = np.loadtxt(HISTORICAL_ARTIFACT / "synthetic_y_1000_10000_e20.csv", delimiter=",")
    return X, y
