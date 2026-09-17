"""The benchmark runner: one frozen plan in, one JSONL of measurements out.

Three decisions that make this a comparison rather than a demonstration.

**Threads are pinned before numpy loads.** Every worker sets
``OMP_NUM_THREADS`` and friends to one *in the environment of a fresh process*,
because setting them after numpy has imported its BLAS does nothing. The
historical timings were taken with ``MSK_IPAR_NUM_THREADS = cpu_count()``
against a single-threaded scikit-learn, which measures the machine's core count
as much as the method.

**Each solver is run over a ladder of its own tolerances.** No two of these
solvers mean the same thing by ``tol``, so asking them all for ``1e-8`` compares
nothing. What the ladder produces is a (time, achieved accuracy) curve per
solver, and "time to reach a relative gap of 1e-6" is then read off the data.

**Each (instance, solver) pair runs in its own process**, which gives isolation
from a solver that corrupts state, a real timeout, and a peak-RSS measurement
that means something.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",  # Clarabel is Rust
}


@dataclass(frozen=True, slots=True)
class Arm:
    """One solver, with the tolerance ladder it will be run over."""

    solver: str
    tolerances: tuple[float, ...]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Plan:
    """A frozen benchmark. Its digest is what a preregistration record binds."""

    instances: tuple[dict[str, Any], ...]
    lambda_ratios: tuple[float, ...]
    arms: tuple[Arm, ...]
    repetitions: int
    timeout_seconds: int
    threads: int = 1

    def to_json(self) -> str:
        return json.dumps(
            {
                "instances": [dict(item) for item in self.instances],
                "lambda_ratios": list(self.lambda_ratios),
                "arms": [asdict(arm) for arm in self.arms],
                "repetitions": self.repetitions,
                "timeout_seconds": self.timeout_seconds,
                "threads": self.threads,
            },
            sort_keys=True,
            indent=2,
        )

    @property
    def digest(self) -> str:
        import hashlib

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> Plan:
        raw = json.loads(text)
        return cls(
            instances=tuple(raw["instances"]),
            lambda_ratios=tuple(raw["lambda_ratios"]),
            arms=tuple(
                Arm(
                    solver=item["solver"],
                    tolerances=tuple(item["tolerances"]),
                    options=item.get("options", {}),
                )
                for item in raw["arms"]
            ),
            repetitions=int(raw["repetitions"]),
            timeout_seconds=int(raw["timeout_seconds"]),
            threads=int(raw.get("threads", 1)),
        )


# -- the worker, run as `python -m src.cg2026.bench worker` --------------------


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    import resource

    import numpy as np

    from src.cg2026.data import InstanceSpec, build
    from src.cg2026.objective import Penalty, lambda_max
    from src.cg2026.solvers import REGISTRY

    spec = InstanceSpec(**payload["instance"])
    X, y, truth = build(spec)
    penalty = Penalty(lambda_1=payload["lambda_ratio"] * lambda_max(X, y))
    solver = REGISTRY[payload["solver"]]
    options = payload.get("options", {})

    # Warm-up on a tiny instance of the same shape family, outside the timing.
    # celer and skglm compile on first call; an uncompiled first call would be
    # attributed to the method.
    warm_X, warm_y, _ = build(InstanceSpec(spec.family, n=40, p=12, k=3, seed=99))
    with _suppressed():
        try:
            solver(
                warm_X,
                warm_y,
                Penalty(lambda_1=0.3 * lambda_max(warm_X, warm_y)),
                1e-6,
                **options,
            )
        except Exception as exc:  # noqa: BLE001 - a warm-up failure is not a result
            del exc

    records = []
    for tol in payload["tolerances"]:
        for repetition in range(payload["repetitions"]):
            started = time.perf_counter()
            try:
                with _suppressed():
                    found = solver(X, y, penalty, tol, **options)
            except Exception as exc:  # noqa: BLE001 - a failure is a measurement
                records.append(
                    {
                        "tol": tol,
                        "repetition": repetition,
                        "status": f"error:{type(exc).__name__}",
                        "detail": str(exc)[:400],
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                continue
            support = None if truth is None else _support_scores(found.beta, truth)
            records.append(
                {
                    "tol": tol,
                    "repetition": repetition,
                    "status": found.status,
                    "wall_seconds": found.wall_seconds,
                    "cpu_seconds": found.cpu_seconds,
                    "objective": found.objective,
                    "kkt": found.kkt,
                    "relative_gap": found.gap,
                    "nnz": found.nnz,
                    "iterations": found.iterations,
                    "detail": found.detail,
                    **(support or {}),
                }
            )
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "instance": payload["instance"],
        "instance_label": spec.label,
        "instance_digest": spec.digest,
        "lambda_ratio": payload["lambda_ratio"],
        "lambda_1": penalty.lambda_1,
        "solver": payload["solver"],
        "options": options,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "peak_rss_mib": peak_kib / 1024.0,
        "records": records,
        "numpy": np.__version__,
    }


def _support_scores(beta, truth) -> dict[str, float]:
    import numpy as np

    threshold = 1e-8 * max(float(np.abs(beta).max()), 1.0)
    selected = np.abs(beta) > threshold
    actual = truth != 0
    true_positive = int((selected & actual).sum())
    return {
        "support_size": int(selected.sum()),
        "true_support_size": int(actual.sum()),
        "support_recall": true_positive / max(int(actual.sum()), 1),
        "support_precision": true_positive / max(int(selected.sum()), 1),
    }


class _suppressed:
    """Silence the historical code's prints without losing real errors."""

    def __enter__(self):
        import contextlib
        import io

        self._buffer = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self._buffer)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        return self._redirect.__exit__(*exc)


# -- the orchestrator ----------------------------------------------------------


def run_plan(plan: Plan, output: Path, *, python: str | None = None) -> Path:
    """Execute every cell of ``plan``, appending one JSON object per cell."""

    python = python or sys.executable
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **SINGLE_THREAD}
    cells = [
        {
            "instance": instance,
            "lambda_ratio": ratio,
            "solver": arm.solver,
            "tolerances": list(arm.tolerances),
            "options": arm.options,
            "repetitions": plan.repetitions,
        }
        for instance in plan.instances
        for ratio in plan.lambda_ratios
        for arm in plan.arms
    ]
    with output.open("w", encoding="utf-8") as handle:
        for index, cell in enumerate(cells, start=1):
            started = time.perf_counter()
            process = subprocess.run(
                [python, "-m", "src.cg2026.bench", "worker"],
                input=json.dumps(cell),
                capture_output=True,
                text=True,
                env=environment,
                timeout=None,
                check=False,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            elapsed = time.perf_counter() - started
            if process.returncode == 0 and process.stdout.strip():
                try:
                    result = json.loads(process.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    result = {
                        **cell,
                        "status": "unparseable",
                        "stdout_tail": process.stdout[-800:],
                    }
            else:
                result = {
                    **cell,
                    "status": f"worker_exit_{process.returncode}",
                    "stderr_tail": process.stderr[-800:],
                }
            result["cell_wall_seconds"] = elapsed
            result["cell_index"] = index
            result["cell_total"] = len(cells)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            label = f"{cell['instance'].get('family')}-{cell['solver']}"
            print(
                f"[{index}/{len(cells)}] {label} ratio={cell['lambda_ratio']} {elapsed:.1f}s",
                flush=True,
            )
    return output


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        payload = json.loads(sys.stdin.read())
        print(json.dumps(_worker(payload), sort_keys=True))
        return 0
    if len(sys.argv) < 3:
        print("usage: bench <plan.json> <output.jsonl>", file=sys.stderr)
        return 2
    plan = Plan.from_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"plan digest {plan.digest}")
    run_plan(plan, Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
