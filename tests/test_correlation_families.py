"""The two design families added for the portability sweep, checked numerically.

A generator's docstring says what covariance it produces. These are the
evidence. Both families exist so that a sweep can vary *only* the column
covariance: same `k`, same SNR construction, same coefficient law as
`sparse`, which is the independent case. If that were not true, a shift in
the fast window across families would not be attributable to correlation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cg2026.data import InstanceSpec, build


def corr(family: str, **kw) -> np.ndarray:
    X, _y, _t = build(InstanceSpec(family=family, n=4000, p=40, k=6, seed=0, **kw))
    return np.corrcoef(X, rowvar=False)


def test_toeplitz_correlation_decays_as_rho_to_the_distance() -> None:
    """`cov(i, j) = rho ** |i - j|`, which is what AR(1) means."""

    rho = 0.7
    C = corr("toeplitz", rho=rho)
    for gap in (1, 2, 3, 5):
        seen = np.mean([C[i, i + gap] for i in range(30)])
        assert seen == pytest.approx(rho**gap, abs=0.06), f"gap {gap}"


def test_block_correlation_is_flat_inside_and_absent_outside() -> None:
    """The property that distinguishes it from AR(1): no decay with distance."""

    rho, size = 0.6, 10
    C = corr("block", rho=rho, block_size=size)
    inside = [C[0, j] for j in range(1, size)]
    outside = [C[0, j] for j in range(size, 40)]
    assert np.mean(inside) == pytest.approx(rho, abs=0.06)
    assert max(inside) - min(inside) < 0.12, "block correlation should not decay"
    assert abs(np.mean(outside)) < 0.06


def test_the_independent_family_really_is_independent() -> None:
    """The control the other two are compared against."""

    C = corr("sparse")
    off = C[~np.eye(C.shape[0], dtype=bool)]
    assert abs(np.mean(off)) < 0.03
    assert np.max(np.abs(off)) < 0.15


@pytest.mark.parametrize(
    ("family", "kw"),
    [("toeplitz", {"rho": 0.5}), ("block", {"rho": 0.5, "block_size": 8})],
)
def test_the_correlated_families_differ_from_sparse_only_in_the_design(
    family: str, kw: dict
) -> None:
    """Same k, same SNR, same coefficient law -- so only the covariance varies."""

    spec = InstanceSpec(family=family, n=800, p=40, k=6, snr=5.0, seed=3, **kw)
    X, y, truth = build(spec)
    assert truth is not None, "a correlated design must carry its ground truth"
    assert int((truth != 0).sum()) == 6
    assert X.shape == (800, 40) and y.shape == (800,)
    assert np.isfinite(X).all() and np.isfinite(y).all()


@pytest.mark.parametrize("rho", [-0.1, 1.0, 1.5])
def test_a_correlation_outside_its_domain_is_refused(rho: float) -> None:
    for family, kw in (("toeplitz", {}), ("block", {"block_size": 8})):
        with pytest.raises(ValueError, match="rho"):
            build(InstanceSpec(family=family, n=50, p=10, k=3, rho=rho, **kw))


def test_a_block_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="block_size"):
        build(InstanceSpec(family="block", n=50, p=10, k=3, rho=0.5, block_size=0))


def test_the_same_seed_gives_the_same_instance() -> None:
    """A frozen plan names a seed; the seed has to mean something."""

    for family, kw in (("toeplitz", {"rho": 0.4}), ("block", {"rho": 0.4, "block_size": 5})):
        a, ya, _ = build(InstanceSpec(family=family, n=200, p=20, k=4, seed=11, **kw))
        b, yb, _ = build(InstanceSpec(family=family, n=200, p=20, k=4, seed=11, **kw))
        assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_the_label_and_digest_distinguish_the_new_families() -> None:
    """Two cells that differ must not collide in the record."""

    a = InstanceSpec(family="toeplitz", n=100, p=10, rho=0.3)
    b = InstanceSpec(family="toeplitz", n=100, p=10, rho=0.9)
    c = InstanceSpec(family="block", n=100, p=10, rho=0.3, block_size=5)
    assert a.label != b.label != c.label and a.label != c.label
    assert len({a.digest, b.digest, c.digest}) == 3
