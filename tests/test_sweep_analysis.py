"""The decision-rule inputs, and what must never become one.

`analyse_sweep.py` computes the numbers a preregistered rule reads. It does
not know the thresholds, so what these tests check is not "does it reach the
right verdict" but the two properties a rule depends on:

1. the three verdicts are all *reachable* -- a measurement instrument that
   can only produce one answer is not an instrument;
2. nothing that is a failure of the run can arrive looking like a number.

The second is the one that matters. A missing cell, a solver error, a NaN, a
zero support -- every one of them must leave `portability.R` absent, because
an absent number is an absent measurement and a decision rule reading it
finds nothing to compare.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyse_sweep import analyse, fast_window, portability

ROOT = Path(__file__).resolve().parents[1]


def cell(design: str, ratio: float, rounds, nnz, seconds=1.0, status="ok", reps=1):
    """One benchmark row, in the shape `run_plan` writes."""

    return {
        "instance_label": design,
        "lambda_ratio": ratio,
        "solver": "cg_hist",
        "n": 100,
        "p": 50,
        "peak_rss_mib": 128.0,
        "records": [
            {
                "status": status,
                "iterations": rounds,
                "nnz": nnz,
                "wall_seconds": seconds,
                "cpu_seconds": seconds,
            }
            for _ in range(reps)
        ],
    }


def sweep(design: str, *, best_at: float, supports: dict[float, int]) -> list[dict]:
    """A design whose cost is minimised at `best_at`, with a chosen s(lambda)."""

    rows = []
    for ratio, nnz in supports.items():
        rounds = 10 if ratio == best_at else 100
        rows.append(cell(design, ratio, rounds, nnz))
    return rows


# -- the three verdicts are reachable -----------------------------------------
def test_a_large_ratio_is_reachable() -> None:
    """SUPPORTS control: the lambda window holds still, the support window moves.

    Both designs are fast at the same lambda, so the lambda fold change is 1;
    the support size at that lambda differs by 8x between them.
    """

    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40}),
        *sweep("B", best_at=0.1, supports={0.02: 40, 0.1: 80, 0.5: 320}),
    ]
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert out["status"] == "ok"
    assert out["fold_change"]["lambda"] == pytest.approx(1.0)
    assert out["R"] == pytest.approx(8.0)


def test_a_small_ratio_is_reachable() -> None:
    """CONTRADICTS control: the support window holds still, lambda moves."""

    rows = [
        *sweep("A", best_at=0.02, supports={0.02: 10, 0.1: 11, 0.5: 12}),
        *sweep("B", best_at=0.5, supports={0.02: 8, 0.1: 9, 0.5: 10}),
    ]
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert out["status"] == "ok"
    assert out["R"] < 1.0


def test_a_ratio_near_one_is_reachable() -> None:
    """INCONCLUSIVE control: both indexings move by the same factor."""

    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 20}),
        *sweep("B", best_at=0.2, supports={0.02: 10, 0.1: 20, 0.2: 20, 0.5: 40}),
    ]
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert out["status"] == "ok"
    assert 0.3 < out["R"] < 3.0


# -- what must never become a number ------------------------------------------
@pytest.mark.parametrize(
    ("label", "broken"),
    [
        ("a NaN round count", float("nan")),
        ("an infinite round count", float("inf")),
        ("a missing round count", None),
        ("a round count that is a string", "12"),
        ("a round count that is a bool", True),
    ],
)
def test_a_cost_that_is_not_a_number_yields_no_ratio(label: str, broken) -> None:
    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40}),
        cell("B", 0.02, broken, 40),
        cell("B", 0.1, broken, 80),
        cell("B", 0.5, broken, 320),
    ]
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert "R" not in out, f"{label} produced a ratio"


def test_a_support_of_zero_yields_no_ratio() -> None:
    """A fold change over a zero boundary is not a large number, it is undefined."""

    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 0, 0.5: 40}),
        *sweep("B", best_at=0.1, supports={0.02: 40, 0.1: 80, 0.5: 320}),
    ]
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert "R" not in out
    assert "undefined" in out["status"]


def test_a_design_whose_every_repetition_failed_is_excluded_and_named() -> None:
    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40}),
        *sweep("B", best_at=0.1, supports={0.02: 40, 0.1: 80, 0.5: 320}),
        cell("C", 0.1, 10, 10, status="error:TimeoutExpired"),
    ]
    document = analyse(rows, solver="cg_hist", tolerance=0.10)
    assert any("no repetition succeeded" in p for p in document["problems"])
    assert "C" not in document["portability"]["designs"]


def test_one_design_alone_yields_no_ratio() -> None:
    rows = sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40})
    out = analyse(rows, solver="cg_hist", tolerance=0.10)["portability"]
    assert "R" not in out
    assert "at least 2 designs" in out["status"]


def test_too_few_lambda_points_is_not_a_window() -> None:
    """Two points cannot show where a window's edges are."""

    assert fast_window(
        [
            {"lambda_ratio": 0.1, "iterations": 10.0, "nnz": 10.0},
            {"lambda_ratio": 0.5, "iterations": 20.0, "nnz": 20.0},
        ],
        "iterations",
        0.10,
    ) is None


def test_a_nan_in_the_results_file_is_refused_rather_than_read(tmp_path: Path) -> None:
    """Python's json accepts `NaN`; no other reader does, and neither does this."""

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"solver": "cg_hist", "lambda_ratio": NaN, "records": []}\n')
    done = subprocess.run(
        [sys.executable, "scripts/analyse_sweep.py", str(bad), "--out", str(tmp_path / "o.json")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2
    assert "not a number" in done.stderr


def test_a_malformed_line_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"solver": "cg_hist"}\n{not json at all}\n')
    done = subprocess.run(
        [sys.executable, "scripts/analyse_sweep.py", str(bad), "--out", str(tmp_path / "o.json")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2


def test_a_missing_results_file_fails_rather_than_writing_an_empty_verdict(
    tmp_path: Path,
) -> None:
    done = subprocess.run(
        [sys.executable, "scripts/analyse_sweep.py", str(tmp_path / "nope.jsonl"),
         "--out", str(tmp_path / "o.json")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode != 0
    assert not (tmp_path / "o.json").exists()


def test_the_document_is_json_every_reader_accepts(tmp_path: Path) -> None:
    """Written with allow_nan=False, so it cannot contain a bare NaN token."""

    source = tmp_path / "in.jsonl"
    rows = [
        *sweep("A", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40}),
        *sweep("B", best_at=0.1, supports={0.02: 40, 0.1: 80, 0.5: 320}),
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "doc.json"
    done = subprocess.run(
        [sys.executable, "scripts/analyse_sweep.py", str(source), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


# -- the load-bearing predicates ----------------------------------------------
def test_the_window_is_relative_to_each_designs_own_minimum() -> None:
    """A slow design still has a fast window; the question is where, not whether.

    If the window were relative to a global best, a design that is uniformly
    slower would have an empty window and drop out -- which would silently
    select the sample on the thing being measured.
    """

    fast = sweep("fast", best_at=0.1, supports={0.02: 5, 0.1: 10, 0.5: 40})
    slow = [cell("slow", r, n * 1000, s)
            for r, n, s in ((0.02, 100, 40), (0.1, 10, 80), (0.5, 100, 320))]
    out = analyse([*fast, *slow], solver="cg_hist", tolerance=0.10)["portability"]
    assert out["design_count"] == 2, "a uniformly slower design was dropped"
    assert out["status"] == "ok"


def test_both_window_edges_are_folded_and_the_larger_is_taken() -> None:
    """A window stable at one end and wild at the other is not portable."""

    windows = {
        "A": {"lambda_lower": 0.1, "lambda_upper": 0.1,
              "support_lower": 10.0, "support_upper": 10.0},
        "B": {"lambda_lower": 0.1, "lambda_upper": 0.9,
              "support_lower": 10.0, "support_upper": 10.0},
    }
    out = portability(windows, cost="iterations", tolerance=0.1)
    assert out["fold_change"]["lambda_lower"] == pytest.approx(1.0)
    assert out["fold_change"]["lambda_upper"] == pytest.approx(9.0)
    assert out["fold_change"]["lambda"] == pytest.approx(9.0), "the stable edge won"


def test_the_single_command_writes_a_document_a_rule_can_read(tmp_path: Path) -> None:
    """`run_sweep` exists because the bridge binds one command per experiment.

    Two commands would need two experiments and a way to pass an artifact
    between them, which that layer deliberately does not have. This checks
    the composition end to end on the smallest real sweep: two designs,
    three lambda, one repetition -- real solves, not fixtures.
    """

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "arms": [{"options": {}, "repetitions": 1, "solver": "cg_hist",
                  "tolerances": [1e-06]}],
        "instances": [
            {"family": "sparse", "k": 5, "n": 200, "p": 40, "seed": 0, "snr": 5.0},
            {"family": "toeplitz", "k": 5, "n": 200, "p": 40, "rho": 0.8,
             "seed": 0, "snr": 5.0},
        ],
        "lambda_ratios": [0.5, 0.25, 0.1],
        "repetitions": 1, "threads": 1, "timeout_seconds": 120,
    }))
    out = tmp_path / "doc.json"
    done = subprocess.run(
        [sys.executable, "scripts/run_sweep.py", "--plan", str(plan), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert out.exists() and out.with_suffix(".jsonl").exists()
    assert out.with_suffix(".manifest.json").exists()
    document = json.loads(out.read_text())
    assert document["schema"] == "cg2026.sweep.v1"
    assert document["design_count"] == 2
    assert "plan_digest" in document
    # The path a preregistered rule names must exist or say why not.
    assert "R" in document["portability"] or document["portability"]["status"]


def test_the_preregisterable_path_survives_the_schema_listing() -> None:
    """The document must stay small enough for Research OS to show the point of it.

    Research OS lets a designer name a metric path only if it can show that
    the path exists, and it shows the first 40 numeric paths of a committed
    output **in sorted order**. An earlier version of this document also
    carried a `by_cost` tree and every (design, lambda) point -- about 600
    paths at the real sweep's size -- and `portability.R` sorted well past
    the cut. The designer would have been shown forty paths under
    `by_cost.` and would have refused the idea again, correctly, for having
    no usable metric. That is the bug this instrument exists to remove, so
    it must not reintroduce it.

    40 is Research OS's constant, restated here rather than imported: this
    repository does not depend on that one, and a silent change there should
    fail here loudly.
    """

    limit = 40
    designs = [f"design-{i}" for i in range(8)]
    lambdas = [0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.09, 0.06, 0.045, 0.03, 0.02]
    rows = [
        cell(design, ratio, 10 if index == position else 50 + index, 5 + index * 3,
             seconds=(10 if index == position else 50 + index) * 0.2, reps=3)
        for position, design in enumerate(designs)
        for index, ratio in enumerate(lambdas)
    ]
    document = analyse(rows, solver="cg_hist", tolerance=0.10)
    paths = sorted(_numeric_paths(document))
    assert "portability.R" in paths[:limit], (
        f"portability.R sorted to {paths.index('portability.R')}, past the "
        f"first {limit} a designer is shown"
    )


def _numeric_paths(node, prefix: str = "") -> list[str]:
    """The same walk Research OS does: dotted paths to every finite number."""

    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += _numeric_paths(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _numeric_paths(value, f"{prefix}.{index}")
    elif isinstance(node, int | float) and not isinstance(node, bool):
        found.append(prefix)
    return found
