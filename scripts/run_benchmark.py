"""Execute one frozen benchmark plan. The command Research OS declares.

Deliberately takes the plan as a path and writes to a path: everything that
decides what runs is in the plan file, whose digest the preregistration record
binds, so a run cannot quietly differ from what was registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cg2026.bench import SINGLE_THREAD, Plan, run_plan


def environment_record() -> dict[str, str]:
    import numpy
    import scipy
    import sklearn

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "threads": json.dumps(SINGLE_THREAD, sort_keys=True),
    }
    for name in ("cvxpy", "clarabel", "celer", "skglm"):
        try:
            record[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - absence is the record
            record[name] = "absent"
    try:
        record["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        record["git_dirty"] = str(
            bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        )
    except Exception:  # noqa: BLE001
        record["git_commit"] = "unknown"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan = Plan.from_json(plan_text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "plan_path": str(plan_path),
        "plan_digest": plan.digest,
        "plan_file_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
        "environment": environment_record(),
        "results": str(out),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"plan digest        {plan.digest}")
    print(f"plan file sha256   {manifest['plan_file_sha256']}")
    print(f"manifest           {manifest_path}")

    run_plan(plan, out)
    print(f"results            {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
