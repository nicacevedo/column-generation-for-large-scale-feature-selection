# Column generation for large-scale feature selection

A 2023 master's thesis at Universidad de Chile
([repositorio.uchile.cl/handle/2250/198750](https://repositorio.uchile.cl/handle/2250/198750))
proposed a second-order cone formulation for feature selection in linear
regression and a column-generation method for solving it at scale. Work resumed
at MIT in April 2025 and stopped that June.

`master` holds that work, unchanged.

## The 2026 reassessment

Branch `research/2026-reassessment` reassesses it under the theory, algorithms
and software available as of 2026-09-17. It adds nothing to `master` and
deletes nothing from it.

Start here: **[`docs/2026/SCIENTIFIC_REPORT.md`](docs/2026/SCIENTIFIC_REPORT.md)**.

| document | what it is |
|---|---|
| [`docs/2026/SCIENTIFIC_REPORT.md`](docs/2026/SCIENTIFIC_REPORT.md) | the closure report, A–R, and the paper/no-paper decision |
| [`docs/2026/TIMELINE.md`](docs/2026/TIMELINE.md) | what the 2023–2025 commits and notes actually established |
| [`docs/2026/PROVENANCE.md`](docs/2026/PROVENANCE.md) | the nine source documents, their hashes, and where they disagree |
| [`docs/2026/LITERATURE.md`](docs/2026/LITERATURE.md) | the literature ledger and the novelty matrix |
| [`docs/2026/UNIT_BALL_VERDICT.md`](docs/2026/UNIT_BALL_VERDICT.md) | whether the 2025 unit-ball device is a bound |
| [`docs/2026/REPRODUCTION.md`](docs/2026/REPRODUCTION.md) | what can and cannot be reproduced, and why |

### The three findings that matter

**The model is the LASSO.** The conic formulation starts from ℓ0 with big-M,
passes through the perspective function, and loses the binary restriction on
the way. With `z, u` continuous and `βᵢ² ≤ zᵢuᵢ`, AM–GM makes the penalty
exactly `2√(τκ)|βᵢ|`. The poster's own equivalence box says so. So the method
belongs in the LASSO-solver comparison class, not the best-subset one.

**The pricing rule is the maximum-violation working-set rule.** Selecting the
`v` largest `(|(Xᵀψ)ᵢ| − λ₁)₊` is, with `ψ = −2r`, selecting the largest
`(|Xᵢᵀr| − λ₁/2)₊` — the KKT violation. Osborne, Presnell and Turlach add
exactly that coordinate in 2000.

**It is not competitive.** On a frozen benchmark against Celer, skglm,
scikit-learn and LassoLars, at matched objectives and pinned threads, the
method does not reach a relative duality gap of 1e-6 at all, in any cell
measured. The best modern solver reaches it in 10–210 ms.

The recommendation is **no paper on the computational contribution**. That is a
result, reached with evidence, not a failure to find one.

## Running it

```bash
uv sync --extra conic --extra baselines
uv run python -m pytest -q                      # 96 tests
uv run python scripts/profile_method.py         # answers HYP-0007
uv run python scripts/adjudicate_pricing.py     # answers HYP-0002, HYP-0004
uv run python scripts/run_benchmark.py --plan experiments/EXP-0001-plan.json \
    --out results/2026/EXP-0001.jsonl
uv run python scripts/analyse_benchmark.py results/2026/EXP-0001.jsonl
```

MOSEK is not licensed on this host, so the conic solver is Clarabel. Nothing
produced here reproduces a historical MOSEK timing, and nothing claims to.

## Layout

```text
.research/          the Research Capsule: questions, hypotheses, assumptions,
                    the preregistered experiment. No accepted Claims.
sources/2023-2025/  the nine historical documents, with hashes in PROVENANCE.md
src/cg2026/         the 2026 code: objective conventions, pricing mathematics,
                    instances, solver adapters, the benchmark runner
src/cg_models.py    the 2025 code, unchanged
models.py           the 2024 code, unchanged
tests/              96 tests; every proposition carries its evidence
```
