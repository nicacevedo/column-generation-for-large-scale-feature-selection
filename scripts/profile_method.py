"""Profile the historical method. Answers `HYP-0007` before anything is optimised.

The 2025 notes' last line is "Future Note: Do a profiling of the method, to see
what are the most time-consuming operations, and analyze how to improve them."
Every idea that followed -- minibatch estimation of ``psi'X``, GPU matrix-vector
products, approximate pricing -- addresses the cost of forming ``X'psi``.

Run: `python scripts/profile_method.py`
Writes: `results/2026/profile.json`
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cg2026.data import InstanceSpec, build
from src.cg2026.objective import Penalty, lambda_max, relative_gap
from src.cg2026.profile_cg import profile_cg_hist

CASES = [
    (InstanceSpec("sparse", n=2000, p=500, k=25, seed=0), 0.10),
    (InstanceSpec("sparse", n=2000, p=2000, k=25, seed=0), 0.10),
    (InstanceSpec("sparse", n=2000, p=5000, k=25, seed=0), 0.10),
    (InstanceSpec("historical", n=2000, p=500, seed=0), 0.10),
    (InstanceSpec("historical", n=2000, p=1000, seed=0), 0.10),
    (InstanceSpec("correlated", n=2000, p=500, rho=0.5, seed=0), 0.10),
    (InstanceSpec("correlated", n=2000, p=500, rho=0.9, seed=0), 0.10),
    (InstanceSpec("sparse", n=2000, p=500, k=25, seed=0), 0.01),
]


def main() -> int:
    out = {
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "note": (
            "Threads pinned to 1 by the caller's environment. Conic solver is "
            "Clarabel: MOSEK is not licensed on this host (ASM-0002), so these "
            "are the timings of an independent solver on the historical code, "
            "never a reproduction of a MOSEK timing."
        ),
        "cases": [],
    }
    for spec, ratio in CASES:
        X, y, _ = build(spec)
        penalty = Penalty(lambda_1=ratio * lambda_max(X, y))
        profile, result = profile_cg_hist(X, y, penalty.lambda_1, time_limit_minutes=3.0)
        entry = {
            "instance": spec.label,
            "instance_digest": spec.digest,
            "n": int(X.shape[0]),
            "p": int(X.shape[1]),
            "lambda_ratio": ratio,
            "total_seconds": profile.total,
            "iterations": profile.iterations,
            "cvxpy_solve_calls": profile.solve_calls,
            "shares": profile.shares(),
            "master_solve_seconds": profile.master_solve,
            "pricing_scan_seconds": profile.pricing_scan,
            "pricing_scan_ms_per_iteration": profile.pricing_scan_per_iteration * 1000,
            "first_master_ms": profile.per_call[0] * 1000 if profile.per_call else None,
            "last_master_ms": profile.per_call[-1] * 1000 if profile.per_call else None,
        }
        if result is not None:
            beta = result[0]
            entry["relative_gap"] = relative_gap(X, y, beta.ravel(), penalty)
        out["cases"].append(entry)
        print(
            f"{spec.label:34} ratio={ratio:<5} total={profile.total:8.3f}s "
            f"iters={profile.iterations:4d} "
            f"master={profile.shares().get('master_solve', 0):6.1%} "
            f"pricing={profile.shares().get('pricing_scan', 0):6.2%}",
            flush=True,
        )
    target = Path("results/2026/profile.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
