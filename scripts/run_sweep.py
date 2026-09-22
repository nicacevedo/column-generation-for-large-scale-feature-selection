"""Run a frozen sweep plan and write the analysed document, in one command.

Research OS binds **one** command to one experiment, and the portability
question needs two steps: execute the plan, then reduce the per-cell stream to
the single number a preregistered rule names. Chaining them from the outside
would need two experiments and a way to pass an artifact between them, which
that layer deliberately does not have.

So this composes the two pieces that already exist -- `bench.run_plan` and
`analyse_sweep.analyse` -- and adds nothing of its own. Both remain usable
separately: `run_benchmark.py` for a plan whose analysis is EXP-0001's, and
`analyse_sweep.py` for re-analysing a stream that is already on disk, which is
what makes the reduction auditable after the fact.

The manifest, the plan digest and the JSON Lines stream are all written too.
The document is the experiment's declared output; the stream beside it is what
lets somebody recompute the document and get the same answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyse_sweep import analyse
from scripts.run_benchmark import environment_record
from src.cg2026.bench import Plan, run_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="a tracked frozen plan")
    parser.add_argument("--out", required=True, help="the analysed JSON document")
    parser.add_argument("--solver", default="cg_hist", help="which arm to summarise")
    parser.add_argument("--window-tolerance", type=float, default=0.10)
    args = parser.parse_args(argv)

    if not 0.0 < args.window_tolerance < 10.0:
        parser.error("--window-tolerance must be in (0, 10)")

    plan_path, out = Path(args.plan), Path(args.out)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan = Plan.from_json(plan_text)
    out.parent.mkdir(parents=True, exist_ok=True)

    stream = out.with_suffix(".jsonl")
    manifest = {
        "plan_path": str(plan_path),
        "plan_digest": plan.digest,
        "plan_file_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
        "environment": environment_record(),
        "results": str(stream),
        "document": str(out),
        "solver": args.solver,
        "window_tolerance": args.window_tolerance,
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"plan digest        {plan.digest}")
    print(f"cells              {len(plan.instances) * len(plan.lambda_ratios) * len(plan.arms)}")

    run_plan(plan, stream)
    rows = [
        json.loads(line, parse_constant=_reject)
        for line in stream.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    document = analyse(rows, solver=args.solver, tolerance=args.window_tolerance)
    document["plan_digest"] = plan.digest
    out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    status = document["portability"].get("status", "?")
    print(f"stream             {stream}")
    print(f"document           {out}")
    print(f"portability.status {status}")
    if "R" in document["portability"]:
        print(f"portability.R      {document['portability']['R']:.6f}")
    for problem in document["problems"]:
        print(f"  problem: {problem}")
    return 0


def _reject(literal: str) -> float:
    raise ValueError(f"{literal} is not a number any JSON reader but Python accepts")


if __name__ == "__main__":
    raise SystemExit(main())
