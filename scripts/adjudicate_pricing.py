"""Settle HYP-0002 and HYP-0004 on the duals the method actually visits.

Two questions, one cheap experiment.

**HYP-0002** says the pricing subproblem has a finite infimum exactly when
``||X'psi||_inf <= lambda_1``, and that the infimum is then zero. That is a
derivation (`src/cg2026/pricing.py` P2) and this supplies the numerical
certificate: for every dual the implemented method visits, compare the
indicator against the infimum computed coordinate-wise.

**HYP-0004** says the unit-ball-restricted pricing value is not a lower bound
on the primal optimum. Its first clause is a derivation (P6) and a two-line
counterexample. Its second clause -- "there exist reachable dual points at
which it exceeds the primal optimum" -- is the one this experiment exists for,
because "reachable" was undefined and the autonomous cycle's own proposal said
so. Here it means: a dual produced by the implemented method at some iteration
on one of these instances.

The honest possible outcomes are three, and the third is reported as such:
    exceeds the primal optimum on some visited dual -> HYP-0004 supported
    never exceeds it on any visited dual           -> NOT OBSERVED on the
                                                      reachable set, which is
                                                      weaker than refuted
    the method never produces a usable dual        -> inconclusive

Run: python scripts/adjudicate_pricing.py --out results/2026/EXP-0002.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cg2026.data import InstanceSpec, build
from src.cg2026.duals import capture_duals
from src.cg2026.objective import Penalty, lambda_max
from src.cg2026.pricing import (
    coordinate_optimum,
    dual_objective,
    pricing_is_bounded,
    unit_ball_pricing_value,
)

INSTANCES = [
    (InstanceSpec("sparse", n=300, p=80, k=8, seed=0), 0.30),
    (InstanceSpec("sparse", n=300, p=80, k=8, seed=1), 0.10),
    (InstanceSpec("sparse", n=200, p=400, k=8, seed=2), 0.20),
    (InstanceSpec("historical", n=300, p=80, seed=3), 0.20),
    (InstanceSpec("correlated", n=300, p=80, rho=0.7, seed=4), 0.20),
    (InstanceSpec("illcond", n=300, p=80, condition=1e3, k=8, seed=5), 0.20),
]

TOLERANCE = 1e-7


def reference_optimum(X, y, penalty: Penalty) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import LassoLars

    model = LassoLars(alpha=penalty.sklearn_alpha(X.shape[0]), fit_intercept=False, max_iter=8000)
    model.fit(X, y)
    beta = np.asarray(model.coef_, dtype=float)
    return penalty.value(X, y, beta), beta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/2026/EXP-0002.json")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Added to every instance seed. Exists so a run can be a first "
            "observation rather than a repeat of one somebody has already "
            "seen: the pilot that established this script works used offset 0, "
            "and its summary is therefore known. Any other offset draws "
            "instances nobody has looked at."
        ),
    )
    args = parser.parse_args()
    instances = [
        (
            replace(spec, seed=spec.seed + args.seed_offset) if args.seed_offset else spec,
            ratio,
        )
        for spec, ratio in INSTANCES
    ]

    report: dict = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "seed_offset": args.seed_offset,
        "note": (
            "Conic solver is Clarabel; MOSEK is not licensed here (ASM-0002). "
            "'Reachable' means: a dual produced by the implemented method at "
            "some iteration on one of these instances."
        ),
        "instances": [],
    }
    try:
        report["environment"]["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        report["environment"]["git_commit"] = "unknown"

    hyp2_agree = hyp2_total = 0
    hyp2_nonzero_finite = 0
    hyp4_exceed = hyp4_total = 0
    worst_excess = 0.0

    for spec, ratio in instances:
        X, y, _ = build(spec)
        penalty = Penalty(lambda_1=ratio * lambda_max(X, y))
        primal, _beta = reference_optimum(X, y, penalty)
        trace = capture_duals(X, y, penalty.lambda_1)
        entry: dict = {
            "instance": spec.label,
            "instance_digest": spec.digest,
            "lambda_ratio": ratio,
            "lambda_1": penalty.lambda_1,
            "primal_optimum": primal,
            "cg_status": trace.status,
            "cg_iterations": trace.iterations,
            "duals_visited": len(trace.duals),
            "duals": [],
        }
        for visited in trace.duals:
            a = X.T @ visited.psi
            indicator = bool(np.abs(a).max() <= penalty.lambda_1)
            # The infimum, computed coordinate-wise from the closed form rather
            # than from a solver status string.
            per_coordinate = [
                coordinate_optimum(float(value), penalty.lambda_1, 0.0) for value in a
            ]
            finite = all(item.bounded for item in per_coordinate)
            infimum = sum(item.value for item in per_coordinate) if finite else float("-inf")
            hyp2_total += 1
            if finite == indicator:
                hyp2_agree += 1
            if finite and abs(infimum) > TOLERANCE:
                hyp2_nonzero_finite += 1

            ball = unit_ball_pricing_value(X, y, visited.psi, penalty.lambda_1)
            hyp4_total += 1
            excess = ball - primal
            if excess > TOLERANCE * max(abs(primal), 1.0):
                hyp4_exceed += 1
                worst_excess = max(worst_excess, excess / max(abs(primal), 1e-12))
            entry["duals"].append(
                {
                    "iteration": visited.iteration,
                    "max_abs_correlation": float(np.abs(a).max()),
                    "dual_feasible": indicator,
                    "pricing_finite": finite,
                    "pricing_infimum": infimum,
                    "bounded_matches_indicator": finite == indicator,
                    "dual_objective": dual_objective(visited.psi, y),
                    "unit_ball_value": ball,
                    "exceeds_primal_optimum": bool(excess > TOLERANCE * max(abs(primal), 1.0)),
                    "relative_excess": excess / max(abs(primal), 1e-12),
                }
            )
            assert pricing_is_bounded(X, visited.psi, penalty.lambda_1, 0.0) == indicator
        report["instances"].append(entry)
        print(
            f"{spec.label:34} duals={len(trace.duals):3d} "
            f"status={trace.status} primal={primal:.6f}",
            flush=True,
        )

    report["summary"] = {
        "hyp0002_duals_checked": hyp2_total,
        "hyp0002_indicator_agreement": hyp2_agree,
        "hyp0002_agreement_rate": (hyp2_agree / hyp2_total) if hyp2_total else None,
        "hyp0002_finite_but_nonzero_infimum": hyp2_nonzero_finite,
        "hyp0002_verdict": (
            "SUPPORTED"
            if hyp2_total and hyp2_agree == hyp2_total and hyp2_nonzero_finite == 0
            else "REJECTED"
            if hyp2_total
            else "INCONCLUSIVE"
        ),
        "hyp0004_duals_checked": hyp4_total,
        "hyp0004_duals_exceeding_primal": hyp4_exceed,
        "hyp0004_worst_relative_excess": worst_excess,
        "hyp0004_verdict": (
            "SUPPORTED_ON_REACHABLE_SET"
            if hyp4_exceed > 0
            else "NOT_OBSERVED_ON_REACHABLE_SET"
            if hyp4_total
            else "INCONCLUSIVE"
        ),
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("\n" + json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
