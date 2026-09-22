"""Reduce a benchmark sweep to the numbers a preregistered rule can address.

`run_benchmark.py` writes JSON *Lines*: one object per (instance, lambda,
solver) cell. A decision rule fixed before the run names **one number in one
JSON document**, and a stream of lines is not that. This command is the
missing half: deterministic post-processing, no model, no solving, reading a
sweep and writing a single document whose numeric paths are stable enough to
preregister against.

It computes nothing about whether a hypothesis is true. It computes the
quantities; the thresholds live in the decision rule that was frozen before
the sweep ran, outside this repository. That separation is the point -- a
script that knew the thresholds could be written to clear them.

    python scripts/analyse_sweep.py results/2026/EXP-0002.jsonl \
        --out results/2026/EXP-0002-sweep.json

What it answers, for a sweep over designs x lambda/lambda_max:

    For each design, the "fast window" is the set of lambda/lambda_max at
    which the cost is within `--window-tolerance` of that design's own
    minimum. Its boundaries move as the design changes. The same window,
    mapped through s(lambda), has boundaries in support size that also move.

        R = fold_change_support / fold_change_lambda

    R > 1 means the support-indexed window is the less portable of the two;
    R < 1 means the lambda-indexed one is. A rule fixed beforehand decides
    what magnitude of R counts.

**Failure is never a verdict.** A missing cell, a solver error, a timeout, a
non-finite time, a zero or negative support -- any of these and `portability`
carries no `R` at all, `portability.status` says which, and a decision rule
reading `portability.R` finds nothing to compare. An absent number is an
absent measurement, which is what INSUFFICIENT means; it is not evidence
against anything.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

#: What a cell is summarised by. `iterations` is the pricing-round count, which
#: is the quantity the portability question is posed in; `wall_seconds` is the
#: same question asked of the clock, and the two disagreeing is itself a
#: finding (see `time_per_round` below).
COSTS = ("iterations", "wall_seconds")


def _finite(value: Any) -> float | None:
    """A number, or nothing. `bool` is not a number and `NaN` is not finite."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def summarise_cells(rows: list[dict[str, Any]], solver: str) -> dict[str, Any]:
    """One entry per (design, lambda), median over repetitions.

    Median rather than mean: a single stalled repetition should move the
    summary by one rank, not by its own magnitude.
    """

    cells: dict[tuple[str, float], dict[str, Any]] = {}
    problems: list[str] = []
    for row in rows:
        if row.get("solver") != solver:
            continue
        design = str(row.get("instance_label", "?"))
        ratio = _finite(row.get("lambda_ratio"))
        if ratio is None:
            problems.append(f"{design}: lambda_ratio is not a finite number")
            continue
        good = [r for r in row.get("records", ()) if r.get("status") == "ok"]
        bad = len(row.get("records", ())) - len(good)
        if not good:
            problems.append(f"{design} at lambda={ratio:g}: no repetition succeeded")
            continue
        if bad:
            problems.append(f"{design} at lambda={ratio:g}: {bad} repetition(s) failed")
        entry: dict[str, Any] = {
            "design": design,
            "lambda_ratio": ratio,
            "repetitions_ok": len(good),
            "repetitions_failed": bad,
            "n": row.get("n"),
            "p": row.get("p"),
            "peak_rss_mib": _finite(row.get("peak_rss_mib")),
        }
        for field in (*COSTS, "nnz"):
            values = [v for v in (_finite(r.get(field)) for r in good) if v is not None]
            if len(values) != len(good):
                problems.append(
                    f"{design} at lambda={ratio:g}: {field} missing or non-finite"
                )
            entry[field] = statistics.median(values) if values else None
        rounds, seconds = entry["iterations"], entry["wall_seconds"]
        # The falsifier's secondary control, in a computable form: if wall
        # clock tracks round count, this is flat across designs; if it is not
        # flat, time and rounds are answering different questions.
        entry["time_per_round"] = (
            seconds / rounds if rounds and seconds is not None and rounds > 0 else None
        )
        cells[(design, ratio)] = entry
    return {"cells": cells, "problems": problems}


def fast_window(
    points: list[dict[str, Any]], cost: str, tolerance: float
) -> dict[str, Any] | None:
    """The lambda interval within `tolerance` of this design's own minimum.

    Relative to the design's own best, never to a global best: the question is
    where each design's fast regime *is*, not which design is fastest.
    """

    usable = [pt for pt in points if _finite(pt.get(cost)) is not None]
    if len(usable) < 3:
        return None
    best = min(float(pt[cost]) for pt in usable)
    if best <= 0:
        return None
    inside = [pt for pt in usable if float(pt[cost]) <= best * (1.0 + tolerance)]
    if not inside:
        return None
    lo = min(inside, key=lambda pt: pt["lambda_ratio"])
    hi = max(inside, key=lambda pt: pt["lambda_ratio"])
    if _finite(lo.get("nnz")) is None or _finite(hi.get("nnz")) is None:
        return None
    return {
        "best_cost": best,
        "points_in_window": len(inside),
        "lambda_lower": lo["lambda_ratio"],
        "lambda_upper": hi["lambda_ratio"],
        # The same interval read through s(lambda): the support sizes at the
        # two lambda the window is bounded by.
        "support_lower": float(lo["nnz"]),
        "support_upper": float(hi["nnz"]),
    }


def _fold(values: list[float]) -> float | None:
    """max/min across designs. Undefined if anything is non-positive."""

    if len(values) < 2 or any(v <= 0 for v in values):
        return None
    return max(values) / min(values)


def portability(
    windows: dict[str, dict[str, Any]], *, cost: str, tolerance: float
) -> dict[str, Any]:
    """How far each indexing's window moves across designs, and their ratio.

    Both boundaries are folded and the **larger** movement is taken for each
    indexing. Reporting only the lower boundary would let a window that is
    stable at one end and wild at the other look portable, and the claim under
    test is about the window, not about one of its edges. Both are in the
    document so the choice can be checked rather than believed.
    """

    out: dict[str, Any] = {
        "cost": cost,
        "window_tolerance": tolerance,
        "designs": sorted(windows),
        "design_count": len(windows),
    }
    if len(windows) < 2:
        out["status"] = f"needs at least 2 designs with a window, found {len(windows)}"
        return out
    parts: dict[str, float | None] = {}
    for axis, keys in (
        ("lambda", ("lambda_lower", "lambda_upper")),
        ("support", ("support_lower", "support_upper")),
    ):
        for edge, key in zip(("lower", "upper"), keys, strict=True):
            parts[f"{axis}_{edge}"] = _fold([w[key] for w in windows.values()])
        both = [parts[f"{axis}_lower"], parts[f"{axis}_upper"]]
        parts[axis] = max(both) if all(v is not None for v in both) else None
    out["fold_change"] = parts
    lam, sup = parts["lambda"], parts["support"]
    if lam is None or sup is None:
        out["status"] = (
            "a fold change is undefined: a window boundary was zero, negative "
            "or missing in at least one design"
        )
        return out
    if lam <= 0:
        out["status"] = "the lambda fold change is not positive"
        return out
    ratio = sup / lam
    if not math.isfinite(ratio):
        out["status"] = "the ratio is not finite"
        return out
    out["R"] = ratio
    out["status"] = "ok"
    return out


def analyse(rows: list[dict[str, Any]], *, solver: str, tolerance: float) -> dict[str, Any]:
    """The analysis, and deliberately not a copy of the data.

    **Kept small on purpose.** Research OS shows a designer the numeric paths
    of a committed output so it can name one in a preregistered rule, and it
    shows the first `MAX_SCHEMA_PATHS` of them in sorted order. An earlier
    version of this document also carried a `by_cost` tree and every
    (design, lambda) point -- around 600 paths for the real sweep -- and
    `portability.R`, the only path anybody would ever preregister, sorted
    past the cut. The designer would have seen forty paths under `by_cost.`
    and concluded, correctly, that it had been shown no usable metric.

    So the document holds the analysis and the per-design window boundaries
    that show its working, and the raw curve stays in the JSON Lines stream
    written beside it, which is where a reader who wants to recompute this
    should be looking anyway.
    """

    summary = summarise_cells(rows, solver)
    cells = summary["cells"]
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in cells.values():
        by_design[entry["design"]].append(entry)

    windows: dict[str, dict[str, dict[str, Any]]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for cost in COSTS:
        found = {}
        for design, points in by_design.items():
            window = fast_window(points, cost, tolerance)
            if window is not None:
                found[design] = window
        windows[cost] = found
        verdicts[cost] = portability(found, cost=cost, tolerance=tolerance)

    return {
        "schema": "cg2026.sweep.v1",
        "solver": solver,
        "window_tolerance": tolerance,
        "design_count": len(by_design),
        "cell_count": len(cells),
        "problems": summary["problems"],
        # The preregisterable scalars, at shallow paths that sort early.
        "portability": verdicts["iterations"],
        "portability_wall_clock": verdicts["wall_seconds"],
        # The working: where each design's window sits, on both axes.
        "windows": windows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a benchmark JSON Lines file")
    parser.add_argument("--out", type=Path, required=True, help="the JSON document to write")
    parser.add_argument("--solver", default="cg_hist", help="which arm to summarise")
    parser.add_argument(
        "--window-tolerance",
        type=float,
        default=0.10,
        help="a cost within this fraction of the design's minimum is 'fast'",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.window_tolerance < 10.0:
        parser.error("--window-tolerance must be in (0, 10)")
    rows: list[dict[str, Any]] = []
    with args.results.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                # `parse_constant` refuses NaN and Infinity, which Python's json
                # accepts by default and no other reader does. A sweep that
                # wrote one is malformed, and saying so beats computing with it.
                rows.append(json.loads(line, parse_constant=_reject))
            except ValueError as exc:
                print(f"{args.results}:{number}: {exc}", file=sys.stderr)
                return 2
    document = analyse(rows, solver=args.solver, tolerance=args.window_tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    status = document["portability"].get("status", "?")
    print(f"wrote {args.out}  designs={document['design_count']} status={status}")
    if "R" in document["portability"]:
        print(f"  portability.R = {document['portability']['R']:.4f}")
    for problem in document["problems"]:
        print(f"  problem: {problem}")
    return 0


def _reject(literal: str) -> float:
    raise ValueError(f"{literal} is not a number any JSON reader but Python accepts")


if __name__ == "__main__":
    raise SystemExit(main())
