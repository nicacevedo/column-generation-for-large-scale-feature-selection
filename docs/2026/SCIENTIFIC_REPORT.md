# Scientific project report — 2026-09-17

The reassessment of the 2023 conic column-generation approach to sparse
regression, structured as the brief's §37 asks: A–R, evidence first.

**The short version.** The method solves the LASSO, not ℓ0. Its pricing rule is
the maximum-violation working-set rule. Its central mathematical claim is
correct and already written down in a private note. Both 2025 improvements were
measured by their author and are four times slower than what they replaced. On
every cell of a frozen benchmark measured so far it does not merely lose to
Celer, skglm and scikit-learn — it never reaches the target accuracy at all.
The recommendation is **TRACK F, no viable modern paper on the computational
contribution**, with one narrow exception in §M that a human must rule on.

---

## A. Repository starting and final state

```text
repository   /home/nicacevedo/Documents/Github/column-generation-for-large-scale-feature-selection
start        master @ 58d8b62  "[Feature] Code refactor & new pricing approaches"  2025-06-07
branch       research/2026-reassessment, created from 58d8b62
```

`master` is untouched. No thesis-era history was rewritten, nothing was
force-pushed, and no historical result file or failed experiment was deleted.
Eight commits existed before this work and all eight are intact.

## B. Source and provenance inventory

`docs/2026/PROVENANCE.md`. Nine PDFs, each with its SHA-256, its role and its
embedded metadata, copied from the Research OS repository (where they did not
belong) into `sources/2023-2025/`.

Findings worth repeating here:

- **Only one artifact carries its own date.** `Poster_MSWorkshop.pdf` has
  `CreationDate D:20231219`. Every other PDF was re-exported in September 2026,
  so its embedded date is evidence of nothing. The two April 2025 notes were
  printed to PDF **today** and their dates are self-reported in their titles.
- **Two files the brief names are absent.**
  `Memoria_Tesis__MGO___Final___ENG_.pdf` and its `(1)` variant are not on this
  machine. Searched `$THESIS_REPO_DIR`, `~/Documents`, `~/Documents/Github`,
  `~/Downloads`, `~/research` to depth 4. `THESIS_INPUT_DIR` is unset.
- **`Tesis_MGO.pdf` and `feature_selection.pdf` are one document.** Different
  SHA-256, byte-identical decompressed content streams. Recorded as an alias
  pair rather than merged.
- **Four disagreements are preserved rather than resolved** (D1–D4): the 2026
  cover dates against the 2023 archival record; the poster's "penalizing the
  number of non-zero coefficients" against its own LASSO-equivalence box; two
  incompatible closed forms for the Elastic-Net pricing solution; and the
  unit-ball device described as a bound in one artifact and as extreme-ray
  generation in another.

## C. The 2023 → 2025 timeline

`docs/2026/TIMELINE.md`. Eight commits, one branch, a thirteen-month gap. Every
development classified. The three that change what this project is:

**§T0 — the model is the LASSO.** The earliest note starts from ℓ0 with big-M,
rewrites it through the perspective function, and drops the binary restriction
on `z` on the way. With `z, u` continuous and `βᵢ² ≤ zᵢuᵢ`, AM–GM makes the
penalty exactly `2√(τκ)|βᵢ|`. The poster's own equivalence box says so. The
`MIQP` that is ℓ0 exists in `models.py` and the decomposition is never applied
to it.

**§W — the whiteboard already proved the central result.** It derives the
Lagrangean dual, simplifies it to `max −‖λ‖²/4 − λᵀy s.t. ‖Xᵀλ‖_∞ ≤ 2√(τκ)`,
observes that dualising back gives the LASSO, and says "we can already see a
pricing-boundedness criterion when fixing λ". That is the brief's Q2, answered,
in an unfinished private note, never written up. Its §2.4 asks "why is the
pricing bounded only at the final iteration?" and leaves it as `[Todo renaud]`;
it follows from the preceding paragraph in one line.

**§2025-B — both 2025 ideas were measured and are worse.** The author's own
committed data, same instance, same parameters:

| policy | wall time | iterations |
|---|---|---|
| `v_solution` (2023) | **0.262 min** | 13 |
| `unit_ball` (2025) | 1.010 min | 74 |
| `negative_gradient` (2025) | 1.006 min | 72 |

## D. Literature ledger

`docs/2026/LITERATURE.md`. Twelve topic retrievals and eight identifier fetches
through Research OS's literature subsystem; 1 124 works indexed. The 2023
work's entire declared prior art is three papers, none of them a LASSO solver
later than glmnet.

Limitations recorded rather than hidden: no forward citation expansion (the
subsystem exposes search and fetch, not citation traversal); no full-text
reading, so every "subsumed by" rests on a stated contribution; and OpenAlex
rate-limited after 1 000 credits, which the tool reported as `complete: no` on
four retrievals rather than silently returning fewer.

## E. Novelty matrix

Ten candidates, nine subsumed by named work. **The primary gate is N4 and it
does not pass**: selecting the `v` largest `(|(Xᵀψ)ᵢ| − λ₁)₊` is, with
`ψ = −2r`, selecting the largest `(|Xᵢᵀr| − λ₁/2)₊` — the KKT violation.
Osborne, Presnell and Turlach add exactly that coordinate in 2000
(`doi:10.1093/imanum/20.3.389`); the 2011 strong rules screen on it; Gap Safe,
Celer and Blitz build working sets from it.

The whiteboard's perspective convexification of the ℓ0+ℓ2 set is
Atamtürk–Gómez 2019 (`arXiv:1901.10334`), six years earlier. The Elastic-Net
closed form is the soft-thresholding operator.

## F. Mathematical audit

`src/cg2026/pricing.py`, with 67 tests in `tests/test_pricing_math.py`. Every
proposition carries the numerical evidence for it and, where one exists, the
counterexample against its wrong form.

- **P2 — bounded pricing is exactly full-LASSO dual feasibility.** Confirmed,
  and certified on 30 duals the implemented method actually visits: 30/30
  agreement between the indicator and finiteness, every finite infimum zero.
- **P3 — the equality case.** At `|aᵢ| = λ₁` the problem is bounded with value
  zero and a ray of minimisers. The shipped test is `|aᵢ| − λ₁ > −tol`, and KKT
  puts every support coordinate exactly on that boundary at optimality, so the
  pricing reports "unbounded" at the method's own solution.
- **P4 — the Elastic-Net closed form.** `βᵢ* = −sign(aᵢ)(|aᵢ| − λ₁)₊/(2λ₂)`,
  value `−(|aᵢ| − λ₁)₊²/(4λ₂)`. **Confirms** the April 2025 note and
  **refutes** the whiteboard's `−(1/θ)(λ₁ + a)`, which is one branch applied
  everywhere and is not merely different but not optimal.
- **The corollary** — bounded pricing implies the restricted master is optimal
  — was stated without hypotheses and was **wrong as scoped**. An independent
  adversarial review produced two counterexamples and the first repair was
  wrong in a third way. The correct statement needs the master's reachable set
  to be a cone containing zero (`sum_1_comb=True` breaks it: bounded pricing,
  `RMP = 20.5`, `p* = 2.0`) and the certificate to be "every violated
  coordinate is already in the working set" rather than "nothing violates"
  (with `λ₂ > 0` the violated set at optimality *is* the support). All three
  errors are tests.
- **The consequence that condemns the historical bound plot.** For `λ₂ = 0`
  with cone weights, `d(ψᵏ) = RMPₖ` identically, so the recorded "lower bound"
  is either `−∞` or exactly the upper bound. There is no gap to plot.

## G. Unit-ball verdict

`docs/2026/UNIT_BALL_VERDICT.md`.

```text
As a source of columns:   VALID_COLUMN_OR_DIRECTION_HEURISTIC_ONLY
As a lower bound:         INVALID_OR_UNJUSTIFIED
```

Adding a constraint to a minimisation raises its optimum, so the value is an
upper bound on the Lagrangian dual function and weak duality gives no
inequality in the direction the 2025 notes use. `X = [[1]]`, `y = [10]`,
`λ₁ = 1`, the reachable `ψ = −2y`: value 81 against `p* = 9.75`.

**And whether the symptom appears depends on standardisation**, which took
three attempts to establish and two wrong explanations to discard. Holding
`(n, p)` fixed: **0 of 24** reachable duals exceed `p*` on standardised data,
**24 of 24** on the same data unscaled. Both the 2025 driver and this project
standardise, which is why the historical runs saw oscillation rather than a
visibly invalid bound.

## H. Historical reproduction status

`docs/2026/REPRODUCTION.md`. **There is no `ORIGINAL_REPRODUCTION` row and there
cannot be one.**

Most historical results cannot be attempted at all, and the reason is not
MOSEK — it is that the data is gone. `real-data/`, `real-data-treated/` and
`synthetic-data-correlated/` are gitignored and absent; `usa.csv`, which is
committed, is a scraped ticker sheet and not a regression matrix. One synthetic
instance survives.

The historical code also does not run on a 2026 stack: `Constraint.dual_value`
for a `cp.SOC` is a ragged list in cvxpy 1.9, so `np.array(..., dtype=object)`
raises under numpy 2. A 46-line compatibility copy, no arithmetic changed,
makes it execute — `PARTIAL_REPRODUCTION`. A second, pre-existing defect in the
bound plotting raises on any termination path with `k ≥ 4`, which is consistent
with the 2025 notes' transcript ending at "Plotting bounds..." and nothing
after.

**A reproduction-fidelity limitation found by review and not yet closed**: the
benchmark adapter runs the historical function's *own* defaults, and the 2025
driver sets `pos_linear_comb=True` and `v = v0 = max(int(m·0.012), 5)`, which
costs 4–7× the iterations. Any comparison against the committed 2025 timings is
a comparison between two configurations.

## I. Profiling

`results/2026/profile.json`. Eight instances, three families, two penalty
scales, `p` from 500 to 5000, threads pinned to one.

```text
restricted master conic solve   79 % – 99 % of wall time
pricing scan (forming X'psi)     0.00 %  -- below measurement resolution
```

Every acceleration the 2025 notes propose — minibatch estimation of `ψ'X`, GPU
matrix-vector products, approximate pricing — addresses a term under a tenth of
a percent of the runtime. **`HYP-0007`'s measurable core is supported.**

The profile also locates the real cost: on the historical dense-truth family at
`n = 2000, p = 1000` the per-iteration master solve grows from 17 ms to 5.4 s
over 88 iterations and the run ends at a relative gap of 0.199.

## J. Benchmark preregistration

`.research/experiments/EXP-0001/manifest.yaml`, frozen and committed at
`0275e54` before any comparative number existed, bound to
`experiments/EXP-0001-plan.json` by digest `8e42f3ec…`. Ten instances across
four families — two of which the historical work never tried — three penalty
levels scaled to `λmax`, seven arms each over its own tolerance ladder, threads
pinned before numpy loads, a 600 s per-cell timeout, and a frozen
failure-and-exclusion policy.

## K. Benchmark results

**PARTIAL at the time of writing.** The frozen decision rule is applied by
`scripts/analyse_benchmark.py`; on the cells completed so far:

```text
HYP-0001   7 cells support, 0 reject, 0 excluded
HYP-0005   current scikit-learn reached gap < 1e-8 without hitting max_iter in 7/7
HYP-0006   REJECTED -- 2 cells recorded a non-zero Lagrangian bound count
```

The result is stronger than `HYP-0001` predicted and weaker than `HYP-0006`
claimed, and both matter.

**Stronger, and stated more carefully than a first draft of this sentence
had it.** `cg_hist` does not reach a relative duality gap of 1e-6 in any cell
measured, at any of its three tolerances. But "never reaches it" reads as a
catastrophe everywhere and the data does not say that; what it says is two
different things in two regimes:

```text
                                        best modern       cg_hist best gap / time
sparse n=2000 p=500   ratio 0.5         skglm  0.0105 s   2.0e-06 in   0.3 s
sparse n=2000 p=500   ratio 0.1         skglm  0.0118 s   6.0e-06 in   0.4 s
sparse n=2000 p=5000  ratio 0.5         celer  0.0910 s   1.7e-06 in   0.6 s
sparse n=500  p=5000  ratio 0.1         skglm  0.0313 s   5.2e-06 in   1.6 s
sparse n=2000 p=500   ratio 0.02        sklearn 0.0133 s  2.5e-06 in   6.3 s
sparse n=500  p=5000  ratio 0.02        skglm  0.1146 s   2.2e-05 in  76.0 s
sparse n=2000 p=5000  ratio 0.02        skglm  0.2076 s   1.2e-01 in 121.3 s   (hit its limit)
sparse n=10000 p=1000 ratio 0.5         sklearn 0.1007 s  no result             (timed out)
```

On easy cells it **stops just short**, at a few times 1e-6, in 0.3–1.6 s —
30× to 60× slower than the best modern solver and short of the target by a
factor of two to six. On the sparsest penalties it degrades sharply: 76 s for
2.2e-05, and 121 s for 0.12, which is its own two-minute limit rather than
convergence.

The pattern is the profile (§I): the cost is the restricted master, the master
grows with the support, and the sparsest penalty is where the most columns are
generated before anything settles. Nothing here needs "never" to carry it.

**Weaker:** `HYP-0006` said the method never computes a Lagrangian lower bound
and the preregistered rule said one non-zero count rejects it. Two cells have
one. Recorded as a rejection. The reading of the source that motivated it still
stands and the profiling and adjudication runs did show empty bound lists; the
hypothesis as *written* is falsified, and rewriting it after the fact is the
thing preregistration exists to prevent.

## L. Numerical-stability results

Partly measured, partly inherited.

- The historical material records infeasible masters, `singular KKT matrix`,
  solver status `Numerical`, matrix coefficients spanning `[6e-06, 1e+00]` and
  392 dense columns. Preserved in `sources/2023-2025/`, not reproduced (the
  solvers are absent).
- **The thesis's own conic formulation is the harder of two equivalent
  models.** `conic_thesis` (`3m+1` variables, `m` rotated cones) reaches a
  relative gap of 2.8e-06 where `conic_reduced` (`2m+1`, one cone) reaches
  better, at the same requested tolerance, on the same problem. A cheap,
  never-tried intervention on the instability the historical material
  documents at length.
- **An optimality measure that depends on exact zeros silently condemns
  interior-point solvers.** A Clarabel solution whose objective matched LARS to
  seven digits scored a KKT violation of 27.95 against `λ₁ = 27.87`, because
  every coefficient was `6e-13` rather than `0`. Any benchmark comparing a
  first-order and an interior-point solver on a naive KKT measure reports the
  second as never converging.

## M. Selected and rejected tracks

```text
TRACK A  modernised exact CG-V                   REJECTED
TRACK B  pricing/dual/working-set theory         REJECTED as novelty
TRACK C  true Elastic-Net CG                     NOT PURSUED
TRACK D  stabilised continuation                 NOT PURSUED
TRACK E  perspective l0+l2                       REJECTED
TRACK F  no viable modern paper                  RECOMMENDED
```

- **A** — the empirical half is settled on every cell measured: the method does
  not reach the target accuracy at all, and the profile says the cost is the
  master solve, which is the part a conic solver makes expensive and a
  coordinate-descent working set makes cheap.
- **B** — the mathematics is correct and is the LASSO dual. The selection rule
  is the maximum-violation rule (§E, N4).
- **C** — the closed form is the soft-thresholding operator; there is no
  evidence a conic master would beat a coordinate-descent one, and §I says
  where the time goes.
- **D** — untested, and generic regularisation-based stabilisation of column
  generation is long established.
- **E** — Atamtürk–Gómez 2019, and the implemented method never touches ℓ0.

**The one exception, and it is a human's call.** A precise negative result
exists: *a conic Dantzig–Wolfe decomposition of the LASSO reduces, under free
master weights and unit-ray columns, to a maximum-violation working-set method
whose restricted solve is an interior-point solve, and is therefore dominated
by working-set methods whose restricted solve is coordinate descent.* It is a
claim about a family, it is checkable, and the benchmark settles its empirical
half. Whether that is publishable depends on a venue's appetite for negative
results about a natural-looking approach, and that judgement is not this
report's to make.

## N. Human scientific decisions

**One is pending and it is the researcher's.** An autonomous cycle produced
`PROP-19700101T000000Z-b9c26fcd`, a twelve-item proposal grounded in a runtime
finding, and the run is parked at `WAITING_FOR_SCIENTIFIC_DECISION`. Nothing in
this session promoted or declined it. The commands are in §N of the Research OS
report.

The proposal's first item states that the finding it rests on is
provenance-only and bears on nothing substantive — which is correct, and is the
kind of thing an eager system does not say about its own output. Its second
item observes that two of this project's hypotheses weld a testable core to an
unfalsifiable clause. That criticism is correct and it changed this work:
`HYP-0004`'s "reachable" was undefined, so it was defined, instrumented and
measured.

## O. Accepted and rejected claims

**No Claim has been accepted. The capsule contains none and should not.**

| object | status |
|---|---|
| `HYP-0001` modern solvers dominate | supported on 7/30 cells; awaiting the full run |
| `HYP-0002` bounded pricing ≡ dual feasibility | supported, derived and certified on 30 visited duals |
| `HYP-0003` it is a maximum-violation working-set method | supported by source reading; the trajectory trace the proposal asks for is not run |
| `HYP-0004` unit-ball value is not a lower bound | first clause proved; existence clause supported on unstandardised data, not observed on standardised |
| `HYP-0005` the historical baseline did not converge | supported on 7/7 cells |
| `HYP-0006` termination is never certified | **REJECTED** by its own preregistered rule |
| `HYP-0007` the pricing scan is not the bottleneck | supported: 0.00 % against 79–99 % |

Promoting any of these into the capsule is a human act and none has been taken.

## P. Paper / no-paper decision

**Recommended: no paper on the computational contribution.** The reasoning is
§E and §K, and neither depends on the other: the selection rule is a known rule
whether or not it is fast, and it is not fast.

This is a **successful outcome of the programme**, not a failure of it. The
brief's outcome D is reached with evidence: a novelty gate that closes on a
named 2000 paper, and a benchmark in which the method does not reach the
accuracy its competitors reach in milliseconds.

**What is worth keeping.** The equivalence — that the thesis's conic model is
the LASSO and its pricing-boundedness criterion is the LASSO dual feasibility
constraint — is correct, is not in the thesis, and is the fact that reorganises
the whole programme. It belongs in the thesis, not in a paper.

## Q. Manuscript status

**None, and none should be drafted.** The brief's §31 gate requires one
coherent contribution to survive the novelty audit, the mathematical audit, the
benchmark and an independent review. Nothing has. `docs/2026/` is the
`PAPER_DECISION` report §31 asks for in that case.

## R. Remaining scientific questions

1. **Finish `EXP-0001`.** 7 of 30 cells. The rule is frozen and coded.
2. **Run the working-set trajectory trace** the autonomous proposal specifies
   (its PR-006). It converts the novelty gate from a literature argument into a
   round-by-round measurement, and it is the one experiment that could falsify
   `HYP-0003` and reopen `Q-0004`.
3. **Re-measure with the 2025 driver's settings** (`pos_linear_comb=True`,
   `v = v0 = max(int(m·0.012), 5)`), separately preregistered. Nothing here
   reproduces the committed 2025 numbers, and the difference is 4–7× in
   iterations.
4. **Real data.** None was used. The historical real data is gone and OpenML
   was unreachable within budget. `Q-0009` is untouched.
5. **`Q-0007`, the numerical instability**, has one new datum (the thesis's
   conic form is the harder of two equivalent models) and no ablation.
6. **The ℓ0 track** was declared out of scope by §T0 and that should be
   revisited only if someone intends to apply the decomposition to `MIQP`,
   which nobody has.
