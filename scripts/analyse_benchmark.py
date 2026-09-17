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
    # **The loosest qualifying tolerance, and only that one.**
    #
    # An earlier version of this function looped over every qualifying
    # tolerance and kept the fastest. That is not what the manifest says, and
    # an adversarial review measured the difference: 10 of 40 (cell, solver)
    # pairs affected, up to 1.82x, and always in the direction that lowers
    # M(c) and therefore favours the reported conclusion. Silent per-solver
    # best-case selection.
    #
    # It is restored to the frozen rule rather than the rule amended, because
    # the deviation was discovered after results existed and the whole point of
    # freezing it was that this is when it matters.
    for tol in sorted(by_tol, reverse=True):
        rows = by_tol[tol]
        gap = statistics.median(float(r["relative_gap"]) for r in rows)
        if gap <= TARGET_GAP:
            return statistics.median(float(r["wall_seconds"]) for r in rows), gap
    return None, None


def objective_excess(records: list[dict[str, Any]], reference: float) -> float | None:
    """The best relative objective excess this solver reached, ``(F - F*)/F*``.

    **Not a preregistered metric, and the one a reader actually wants.** The
    preregistered rule is time to a certified relative duality gap of 1e-6, and
    that gap is built by rescaling the residual, so it is *first order* in the
    KKT overshoot while objective suboptimality is *second order*. A solver can
    therefore be disqualified on the certificate while its answer is optimal to
    eleven significant figures -- which an adversarial review showed is exactly
    what happens to `cg_hist` on eight of nine cells.

    Reported alongside the frozen rule, never instead of it, and recorded as a
    post-hoc addition in `.research/decisions/DEC-0001.yaml`.
    """

    best: float | None = None
    for record in records:
        value = record.get("objective")
        if value is None or record.get("status") not in {"ok", "time_limit"}:
            continue
        excess = (float(value) - reference) / abs(reference) if reference else 0.0
        if best is None or excess < best:
            best = excess
    return best


def time_at_matched_objective(
    records: list[dict[str, Any]], reference: float, tolerance: float
) -> float | None:
    """Median wall time at the loosest tolerance reaching ``tolerance`` excess.

    The same shape as the frozen rule, with the objective in place of the
    certificate. Post-hoc; see :func:`objective_excess`.
    """

    by_tol: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") in {"ok", "time_limit"} and record.get("objective"):
            by_tol[float(record["tol"])].append(record)
    for tol in sorted(by_tol, reverse=True):
        rows = by_tol[tol]
        excess = statistics.median(
            (float(r["objective"]) - reference) / abs(reference) for r in rows
        )
        if excess <= tolerance:
            return statistics.median(float(r["wall_seconds"]) for r in rows)
    return None


def best_gap(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """The tightest accuracy a solver reached at all, and what it cost.

    "Did not reach 1e-6" is the decision rule's answer and it is not the
    informative one. A solver that stops at 3e-5 and one that stops at 0.2 are
    both "did not reach it" and they are not the same result.
    """

    best: tuple[float, float] | None = None
    for record in records:
        if record.get("status") not in {"ok", "time_limit"}:
            continue
        gap = record.get("relative_gap")
        if gap is None:
            continue
        if best is None or float(gap) < best[0]:
            best = (float(gap), float(record["wall_seconds"]))
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

    header = (
        f"{'instance':30} {'ratio':>5} {'best modern':>20} {'cg_hist':>24} "
        f"{'cg obj excess':>14}  {'verdict':>9}"
    )
    print(header)
    print("-" * len(header))

    rejected: list[str] = []
    supported = 0
    excluded: list[str] = []
    sklearn_ok = sklearn_total = 0
    lagrangian_nonzero: list[str] = []
    solver_failures: list[str] = []
    cg_status: dict[str, int] = defaultdict(int)

    matched: list[tuple[str, float, float]] = []

    for (label, ratio), arms in sorted(cells.items()):
        # The best objective any arm reached on this cell. Every arm solves the
        # *same* convex problem, so this is a legitimate common reference: the
        # minimum of a set of upper bounds on one minimum.
        reference: float | None = None
        for row in arms.values():
            for record in row.get("records", ()):
                value = record.get("objective")
                if value is None or record.get("status") not in {"ok", "time_limit"}:
                    continue
                if reference is None or float(value) < reference:
                    reference = float(value)

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
                detail = record.get("detail")
                if isinstance(detail, str):
                    # `solve_cg_hist` records a crash as its detail rather than
                    # letting it abort the cell. A failed master solve is a
                    # measurement of the method, so it is counted, not skipped.
                    solver_failures.append(f"{label} r={ratio} tol={record['tol']:g}")
                    continue
                if (detail or {}).get("lagrangian_values_recorded"):
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
        if cg_wall is None and cg_row and "records" in cg_row:
            gap, wall = best_gap(cg_row["records"])
            cg_text = f"best gap {gap:.2g} in {wall:.1f}s" if gap is not None else "no result"
        elif cg_wall is None:
            cg_text = "no result"
        else:
            cg_text = f"{cg_wall:.4f}s"

        # The objective column. A cell where cg_hist misses the certificate but
        # matches the objective to 1e-11 is a *different* result from one where
        # it misses both, and the certificate column cannot tell them apart.
        excess_text = "n/a"
        if cg_row and "records" in cg_row and reference is not None:
            excess = objective_excess(cg_row["records"], reference)
            if excess is not None:
                excess_text = f"{excess:+.2e}"
                cg_matched = time_at_matched_objective(cg_row["records"], reference, 1e-6)
                best_modern_matched = None
                for name in MODERN:
                    row = arms.get(name)
                    if not row or "records" not in row:
                        continue
                    value = time_at_matched_objective(row["records"], reference, 1e-6)
                    if value is not None and (
                        best_modern_matched is None or value < best_modern_matched
                    ):
                        best_modern_matched = value
                if cg_matched and best_modern_matched:
                    matched.append(
                        (f"{label} r={ratio}", cg_matched / best_modern_matched, cg_matched)
                    )
        print(f"{label:30} {ratio:5} {best_text:>20} {cg_text:>24} {excess_text:>14}  {verdict:>9}")

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
    print("SECONDARY (post-hoc, DEC-0001): slowdown at MATCHED OBJECTIVE 1e-6")
    print("  The preregistered rule scores the duality-gap certificate, which is")
    print("  first order in the KKT overshoot while objective suboptimality is")
    print("  second order. This is the same comparison against the answer.")
    if matched:
        factors = sorted(f for _, f, _ in matched)
        print(f"  cells comparable   {len(matched)}")
        print(
            f"  slowdown  min {factors[0]:.1f}x  median "
            f"{statistics.median(factors):.1f}x  max {factors[-1]:.1f}x"
        )
        for name, factor, wall in sorted(matched, key=lambda item: -item[1]):
            print(f"    {name:34} {factor:8.1f}x  (cg_hist {wall:.3f}s)")
    else:
        print("  cells comparable   0")
    print()
    # Also not a verdict on the hypothesis whose id used to head this block.
    # HYP-0005 is about the 2023 runs; this measures 2026 scikit-learn on 2026
    # instances, which cannot confirm or deny anything about them. HYP-0005 was
    # settled separately, and rejected, by reading the committed 2023 CSVs
    # (EVI-0001): two of its three configurations stopped below max_iter.
    print("OBSERVATION  cells where cg_hist's restricted master failed outright")
    print(f"  runs whose conic solve raised: {len(solver_failures)}")
    for item in solver_failures:
        print(f"    {item}")
    print()
    print("OBSERVATION  does current scikit-learn converge on these instances")
    print(f"  cells reaching gap < 1e-8 without hitting max_iter: {sklearn_ok}/{sklearn_total}")
    print("  Context for the modern baseline's credibility, not a test of HYP-0005.")
    print()
    # NOT a verdict on HYP-0006. An earlier version printed this block under
    # HYP-0006's id with the caption "the CG method never computes a Lagrangian
    # lower bound", which is not what HYP-0006 says. HYP-0006 is about whether
    # the method *terminates* through the pricing test rather than through the
    # dual-stall criterion, and nothing recorded here answers that -- the run
    # detail carries no termination reason. Mislabelling a side observation
    # with a hypothesis id is how an unrelated measurement gets read as a
    # result, so the caption is now the observation and the id is gone.
    print("OBSERVATION  when the Lagrangian lower bound gets computed at all")
    print(f"  cg_hist run statuses: {dict(cg_status)}")
    print(f"  runs recording a bound: {len(lagrangian_nonzero)}")
    for item in sorted(set(lagrangian_nonzero)):
        print(f"    {item}")
    print(
        "  Bears on HYP-0006's stated mechanism -- that the boundedness test "
        "always fires -- without settling its termination claim."
    )
    if len(cells) < 30:
        print()
        print(
            f"PARTIAL: {len(cells)} of 30 preregistered cells have data. The "
            f"frozen decision rule is applied to the complete run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
