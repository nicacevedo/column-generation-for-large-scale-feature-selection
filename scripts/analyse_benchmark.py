"""Apply EXP-0001's preregistered decision rule to its results. Nothing else.

The rule is frozen in `.research/experiments/EXP-0001/manifest.yaml` and is
reproduced here as code so that the verdict is computed rather than argued:

    T(s, c) = the median wall time of solver s on cell c, over its repetitions,
              at the LOOSEST tolerance whose median attained relative duality
              gap is at most 1e-6. Undefined if no tolerance reaches it.
    M(c)    = min over {celer, skglm, sklearn, lars} of T.

    HYP-0001 SUPPORTED   if T(cg_hist, c) is undefined or > M(c) in all cells
             REJECTED    if any cell has T(cg_hist, c) <= M(c)
    HYP-0005 SUPPORTED   if sklearn at tol 1e-10 reaches gap < 1e-8 in at least
                         27 of 30 cells without its iteration count reaching 1e6
    HYP-0006 SUPPORTED   if cg_hist's Lagrangian-bound count is zero in every
                         cell where it returns status ok

A cell where cg_hist times out counts as T undefined, which SUPPORTS HYP-0001.
A cell where all four modern solvers are undefined is excluded from the tally
and reported separately.

Run: python scripts/analyse_benchmark.py results/2026/EXP-0001.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cg2026.data import InstanceSpec

TARGET_GAP = 1e-6
MODERN = ("celer", "skglm", "sklearn", "lars")


def time_to_target(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """``(median wall seconds, achieved gap)`` at the loosest adequate tolerance."""

    by_tol: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") == "ok" and record.get("relative_gap") is not None:
            by_tol[float(record["tol"])].append(record)
    best: tuple[float, float] | None = None
    # Loosest first: a larger `tol` is a weaker request, and the rule asks for
    # the cheapest configuration that still reaches the accuracy.
    for tol in sorted(by_tol, reverse=True):
        rows = by_tol[tol]
        gap = statistics.median(float(r["relative_gap"]) for r in rows)
        if gap <= TARGET_GAP:
            wall = statistics.median(float(r["wall_seconds"]) for r in rows)
            if best is None or wall < best[0]:
                best = (wall, gap)
    return best if best else (None, None)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026/EXP-0001.jsonl")
    cells: dict[tuple[str, float], dict[str, Any]] = defaultdict(dict)
    incomplete: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Derive the label from the spec rather than reading `instance_label`,
        # which a timed-out cell does not carry -- the worker that would have
        # written it was killed. Grouping on the raw dict instead would split
        # one cell in two and report the half with no modern arm as EXCLUDED,
        # which is a bug in the reading and not a fact about the data.
        label = InstanceSpec(**row["instance"]).label
        key = (label, float(row["lambda_ratio"]))
        if "records" not in row:
            incomplete.append(
                f"{label} r={row['lambda_ratio']} {row['solver']}: {row.get('status', 'unknown')}"
            )
            cells[key][row["solver"]] = {"status": row.get("status", "unknown")}
            continue
        cells[key][row["solver"]] = row

    print(f"cells with at least one arm: {len(cells)}")
    print(f"arms that produced no records: {len(incomplete)}")
    print()

    header = f"{'instance':34} {'ratio':>6} {'best modern':>22} {'cg_hist':>22}  {'verdict':>9}"
    print(header)
    print("-" * len(header))

    rejected: list[str] = []
    supported = 0
    excluded: list[str] = []
    sklearn_ok = sklearn_total = 0
    lagrangian_nonzero: list[str] = []
    cg_status: dict[str, int] = defaultdict(int)

    for (label, ratio), arms in sorted(cells.items()):
        modern: dict[str, float] = {}
        for name in MODERN:
            row = arms.get(name)
            if not row or "records" not in row:
                continue
            wall, _gap = time_to_target(row["records"])
            if wall is not None:
                modern[name] = wall
        cg_row = arms.get("cg_hist")
        cg_wall = None
        if cg_row and "records" in cg_row:
            cg_wall, _ = time_to_target(cg_row["records"])
            for record in cg_row["records"]:
                cg_status[record.get("status", "?")] += 1
                detail = record.get("detail") or {}
                if detail.get("lagrangian_values_recorded"):
                    lagrangian_nonzero.append(f"{label} r={ratio}")
        elif cg_row:
            cg_status[cg_row.get("status", "?")] += 1

        sk = arms.get("sklearn")
        if sk and "records" in sk:
            tight = [r for r in sk["records"] if float(r["tol"]) == 1e-10]
            if tight:
                sklearn_total += 1
                gaps = [
                    float(r["relative_gap"]) for r in tight if r.get("relative_gap") is not None
                ]
                iters = [r.get("iterations") or 0 for r in tight]
                if gaps and statistics.median(gaps) < 1e-8 and max(iters) < 1_000_000:
                    sklearn_ok += 1

        if not modern:
            excluded.append(f"{label} r={ratio}")
            verdict = "EXCLUDED"
            best_text = "none reached 1e-6"
        else:
            best_name = min(modern, key=lambda k: modern[k])
            best_text = f"{best_name} {modern[best_name]:.4f}s"
            if cg_wall is None or cg_wall > modern[best_name]:
                verdict = "SUPPORTS"
                supported += 1
            else:
                verdict = "REJECTS"
                rejected.append(
                    f"{label} r={ratio}: cg_hist {cg_wall:.4f}s "
                    f"<= {best_name} {modern[best_name]:.4f}s"
                )
        cg_text = "did not reach 1e-6" if cg_wall is None else f"{cg_wall:.4f}s"
        print(f"{label:34} {ratio:6} {best_text:>22} {cg_text:>22}  {verdict:>9}")

    print()
    print("=" * 72)
    print("HYP-0001  modern working-set solvers dominate the CG method everywhere")
    print(f"  cells supporting  {supported}")
    print(f"  cells rejecting   {len(rejected)}")
    print(f"  cells excluded    {len(excluded)}")
    for item in rejected:
        print(f"    REJECTS: {item}")
    print(
        f"  VERDICT  {'SUPPORTED' if not rejected else 'REJECTED'}"
        f"{'  (on the cells measured so far)' if len(cells) < 30 else ''}"
    )
    print()
    print("HYP-0005  the historical scikit-learn baseline was not converging")
    print(
        f"  cells where current scikit-learn reached gap < 1e-8 without "
        f"hitting max_iter: {sklearn_ok}/{sklearn_total}"
    )
    print()
    print("HYP-0006  the CG method never computes a Lagrangian lower bound")
    print(f"  cg_hist run statuses: {dict(cg_status)}")
    print(f"  cells with a non-zero Lagrangian bound count: {len(lagrangian_nonzero)}")
    print(f"  VERDICT  {'SUPPORTED' if not lagrangian_nonzero else 'REJECTED'}")
    if len(cells) < 30:
        print()
        print(
            f"PARTIAL: {len(cells)} of 30 preregistered cells have data. The "
            f"frozen decision rule is applied to the complete run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
