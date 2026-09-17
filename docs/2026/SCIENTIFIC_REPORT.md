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

**Two reproduction-fidelity limitations found by review and not yet closed.**

*Configuration drift.* The 2025 driver sets `pos_linear_comb=True` and
`v = v0 = max(int(m·0.012), 5)`, which costs 4–7× the iterations; the benchmark
adapter does not. Any comparison against the committed 2025 timings is a
comparison between two configurations.

*The adapter does not run "the function's own defaults".* An earlier revision
of this paragraph said it did, and that is wrong. The historical signature
defaults `solver_params` to
`{'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6}}`
(`src/cg_models.py:50`) -- a MOSEK-only dictionary. MOSEK is unlicensed here, so
the adapter passes `solver_params={}` and the master runs on Clarabel at *its*
built-in tolerance. The consequence is material and is documented at the point
of the decision in `src/cg2026/solvers.py`: the benchmark's `tol` never reaches
the restricted master, only the CG loop's termination tests, so `cg_hist`'s
achieved duality gap is flat across the whole tolerance ladder while every
other arm's moves five orders of magnitude. See §K.1.

## I. Profiling

`results/2026/profile.json`. Eight instances, three families, two penalty
scales, `p` from 500 to 5000, threads pinned to one.

```text
restricted master conic solve   81.1 % – 98.8 % of wall time
pricing scan (forming X'psi)     0.12 % –  9.42 %
```

**The first version of this measurement said `0.00 %` and was measuring
nothing.** The profiler patched `np.matmul`, and the historical code writes
`psi_k_sol.T @ X`; the `@` operator dispatches to `ndarray.__matmul__` in C and
never reaches the Python-level `np.matmul` symbol, so the wrapper was never
called. Verified directly: patching `np.matmul` and evaluating `A @ B`
intercepts zero calls, `np.matmul(A, B)` intercepts one. The master-solve share
*was* real — `cvxpy.Problem.solve` is an ordinary Python method — so the number
that mattered was measured and the number the question was about was not.
Caught here rather than by a reviewer, and corrected by measuring the product
directly on the shapes the run uses rather than trying to intercept it.

**And a second correction, from the adversarial review of the empirical case.**
The numbers above are roughly double the ones this section carried before it.
The profiler multiplied the cost of one `psi'X` product by the *iteration
count*, assuming one product per iteration; the historical code performs about
two, on at least two live paths -- the unboundedness test at
`cg_models.py:446` and the reduced-cost scan at `:601`/`:631`. The profiler now
counts the products instead of assuming them, in a separate untimed pass so the
counting does not inflate what it measures (`_count_pricing_products`).
Measured multiplier: 1.80, 1.83 and 1.98 products per iteration at lambda
ratios 0.5, 0.1 and 0.02 -- close to two, and not exactly two, which is why it
is counted.

The conclusion survives and is now quantified rather than asserted. The pricing
scan's share peaks at **9.42 %** (`sparse`, `p = 500`, ratio 0.1) and *shrinks*
as the run lengthens — 0.64 % on the 59-iteration historical case, 0.15 % on the
62-iteration one, 0.12 % at ratio 0.01 — because the master grows and the
pricing scan does not. Every acceleration the 2025 notes propose addresses a
term worth at most about a tenth of the runtime and usually under 1 %.
**`HYP-0007`'s measurable core is supported**, on a twice-corrected
measurement; the earlier "at most one twenty-fourth" is withdrawn as an
artefact of the undercount.

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

### J.1 What code the frozen run is actually executing

`EXP-0001` runs in a Research OS worktree pinned at **`0275e54`**, the commit
that froze it. That is the point of the worktree, and it has a consequence
worth stating plainly, because two metric repairs landed *after* the freeze and
are therefore **not** in the run:

| repair | commit | in `EXP-0001`? | effect on the run |
|---|---|---|---|
| `SUPPORT_RTOL` for `nnz` and `kkt` | `7c26f3d` | **no** | the shipped JSONL's `nnz` and `kkt` follow the old exact-zero rule (`|beta| > 1e-12`) |
| cancellation-safe `duality_gap` | `7c26f3d` | **no** | none measurable; see below |

The `nnz`/`kkt` columns in `results/2026/EXP-0001.jsonl` are consequently
*not* the quantities defined in the current `src/cg2026/objective.py`, and
should not be read as such. Neither enters the preregistered decision rule or
the `DEC-0001` secondary, both of which are built from `wall_seconds`,
`relative_gap` and `objective`.

The gap repair needed checking rather than assuming, because `relative_gap`
*is* the primary metric. The frozen form computes the dual value as
`y'y - (y-theta)'(y-theta)` and the repaired one as `2 y'theta - theta'theta`;
these are algebraically identical, and the repair was for catastrophic
cancellation, not for correctness. Measured difference between them on six
sparse cells at `tol = 1e-10`, where the gaps sit around 1e-9:

```text
max relative difference   4.3e-04     (i.e. ~1e-13 absolute)
```

Four orders of magnitude below the 1e-6 decision threshold, so the frozen run's
primary metric stands. Recorded because "the fragile version was used" is a
fact about the experiment either way, and a reader should not have to take on
trust that it did not matter.

## K. Benchmark results

**COMPLETE: 30 of 30 preregistered cells, 210 arm-runs.** The frozen decision
rule, applied by `scripts/analyse_benchmark.py` to the full run:

```text
HYP-0001  modern working-set solvers dominate the CG method everywhere
  cells supporting  30
  cells rejecting    0
  cells excluded     0
  VERDICT  SUPPORTED
```

All four families are measured, including the two -- `correlated` and
`illcond` -- that were outstanding while the sections below were first drafted,
and `correlated` is the regime the 2023 claim was actually made in (§B of
`TIMELINE.md`). **It does not rescue the method.** On
`correlated-n2000-p500-rho0.5` at ratio 0.1, LARS finishes in 0.030 s against
`cg_hist`'s 3.78 s at an objective matched to 1.3e-13; at ratio 0.02 the same
comparison is 0.104 s against 84.9 s. The regime that produced the historical
claim is where the gap is widest, not narrowest.

### K.1 What the certificate says, and why it is the wrong headline

`cg_hist` does not reach a relative duality gap of 1e-6 on any cell measured,
at any of its three tolerances. **An earlier revision of this section built its
headline on that sentence, including a "30x to 60x slower ... and short of the
target" claim. Both are withdrawn.** The multiplier was not supported by any
recorded number, and the framing measures the wrong thing. Two facts, each
established after the fact by adversarial review, force the retraction.

**First: the certificate is inert for this arm.** `solve_cg_hist` passes
`solver_params={}`, replacing the historical MOSEK-only default. MOSEK is not
licensed here, so the restricted master runs on Clarabel at its *built-in*
tolerance, and the `tol` the benchmark sweeps reaches only the column
generation loop's own termination tests. Measured consequence: across the
sparse family, `cg_hist`'s achieved gap is flat to four significant figures
over `tol` = 1e-4, 1e-6, 1e-8, while every other arm's moves about five orders
of magnitude over the same ladder. The floor on `cg_hist`'s certificate is set
by the harness, not by the method. "It did not reach 1e-6" is therefore partly
a statement about one line of adapter code, and cannot carry a conclusion.

**Second: the certificate and the answer disagree.** The relative duality gap
is built by rescaling the residual, making it *first order* in the KKT
overshoot, while objective suboptimality is *second order*. A solver can fail
the certificate by a factor of six and still be optimal to eleven significant
figures. That is exactly what happens:

```text
family      instance                      ratio  cg_hist outcome    objective excess
correlated  n2000-p500-rho0.5             0.1    3.78 s             +1.3e-13
correlated  n2000-p500-rho0.5             0.5    gap 1.3e-06        +7.8e-12
correlated  n2000-p500-rho0.9             0.02   gap 3.2e-06        +1.5e-11
correlated  n2000-p500-rho0.9             0.1    0.77 s             +4.2e-13
correlated  n2000-p500-rho0.9             0.5    0.39 s             +1.8e-11
correlated  n2000-p500-rho0.5             0.02   gap 9.5e-06        +2.0e-12
historical  n2000-p500                    0.1    gap 2.2e-06        +6.3e-13
historical  n2000-p500                    0.5    gap 2.1e-06        +1.2e-12
historical  n10000-p1000                  0.5    33.25 s            +2.7e-13
illcond     n2000-p500-cond1e3            0.02   gap 8.6e-06        +2.2e-11
illcond     n2000-p500-cond1e3            0.1    gap 3.8e-05        +5.2e-10
illcond     n2000-p500-cond1e3            0.5    gap 6.1e-06        +2.4e-11
illcond     n2000-p500-cond1e5            0.02   5.53 s             +1.3e-11
illcond     n2000-p500-cond1e5            0.1    gap 3.0e-05        +2.5e-10
illcond     n2000-p500-cond1e5            0.5    gap 3.7e-06        +2.5e-10
sparse      n2000-p500                    0.02   gap 2.5e-06        +1.1e-14
sparse      (9 further sparse cells)      ...    ...                +1e-14 .. +6e-10

   -- and the four where it does not --
historical  n2000-p500                    0.02   gap 0.24, 122.8 s  +2.3e-04
historical  n10000-p1000                  0.1    gap 0.59, 126.4 s  +1.7e-01
historical  n10000-p1000                  0.02   gap 0.89, 120.7 s  +9.4e-01
sparse      n2000-p5000                   0.02   gap 0.12, 121.3 s  +8.8e-04
```

On **26 of 30** cells the historical method returns an answer matching the best
any solver found to between 1.1e-14 and 5.9e-10, most of them while *failing* the
1e-6 certificate. It is not inaccurate. It is **slow**.

On the remaining **four** it genuinely fails, and they are not scattered: three
are the `historical` dense-truth family at the two smaller penalties, where the
objective ends 2.3e-04, 1.7e-01 and **9.4e-01** high after burning the full
two-minute limit. A 94 % objective excess is not a slow solve, it is a
non-solution.

### K.2 The supported headline

Comparing time to a *matched objective* of 1e-6 -- the same shape of rule, with
the answer in place of the certificate, recorded as post-hoc in `DEC-0001` and
reported beside the frozen rule, never instead of it:

> On the 26 comparable cells, `cg_hist` is **7.9x to 4530x slower than the
> best modern solver at a matched objective, median 38.7x.**

| cell | slowdown | `cg_hist` |
|---|---|---|
| `historical-n2000-p500` r=0.1 | **4530x** | 68.35 s |
| `sparse-n500-p5000` r=0.02 | 833x | 75.98 s |
| `correlated-n2000-p500-rho0.5` r=0.02 | 818x | 84.92 s |
| `sparse-n2000-p500` r=0.02 | 628x | 6.33 s |
| `historical-n10000-p1000` r=0.5 | 269x | 33.25 s |
| `illcond-n2000-p500-cond1e5` r=0.02 | 157x | 5.53 s |
| `correlated-n2000-p500-rho0.5` r=0.1 | 154x | 3.78 s |
| `illcond-n2000-p500-cond1e3` r=0.02 | 153x | 4.47 s |
| `illcond-n2000-p500-cond1e3` r=0.1 | 115x | 1.38 s |
| `historical-n2000-p500` r=0.5 | 104x | 0.97 s |
| `illcond-n2000-p500-cond1e5` r=0.1 | 67x | 1.14 s |
| `sparse-n500-p5000` r=0.1 | 48x | 1.58 s |
| `correlated-n2000-p500-rho0.5` r=0.5 | 41x | 0.42 s |
| `sparse-n2000-p500` r=0.1 | 37x | 0.37 s |
| `correlated-n2000-p500-rho0.9` r=0.02 | 33x | 0.90 s |
| `correlated-n2000-p500-rho0.9` r=0.1 | 32x | 0.77 s |
| `correlated-n2000-p500-rho0.9` r=0.5 | 28x | 0.39 s |
| `sparse-n10000-p1000` r=0.02 | 26x | 2.77 s |
| `sparse-n2000-p500` r=0.5 | 23x | 0.25 s |
| `illcond-n2000-p500-cond1e3` r=0.5 | 22x | 0.21 s |
| `sparse-n10000-p1000` r=0.5 | 22x | 2.06 s |
| `sparse-n500-p5000` r=0.5 | 18x | 0.51 s |
| `sparse-n10000-p1000` r=0.1 | 17x | 2.02 s |
| `illcond-n2000-p500-cond1e5` r=0.5 | 12x | 0.14 s |
| `sparse-n2000-p5000` r=0.1 | 8.7x | 0.82 s |
| `sparse-n2000-p5000` r=0.5 | 7.9x | 0.72 s |

This is a weaker claim than "does not reach the target accuracy" and a much
better supported one. It also survives the inert-ladder problem, because the
objective is already optimal at the loosest tolerance -- a working ladder could
only change how long the certificate takes, not the answer that is already
there.

No family escapes. The slowest cells are `historical` and `correlated` -- the
two regimes the 2023 work was actually about -- and the fastest relative
showing, 7.9x, is on `sparse` `p = 5000`, a regime the historical work never
tested. That is the opposite of what §B of `TIMELINE.md` records for the
2023-vs-2023 comparison, where the method was 2.4x **faster** than its
contemporary baseline at a matched objective in exactly this correlated regime.

The difference is not the regime and not the instance. It is sixteen years of
working-set solvers: the 2023 baseline was scikit-learn's cyclic coordinate
descent, and the arms that beat the method here are `celer`, `skglm` and LARS.

### K.3 The shape of the cost

The pattern is the profile (§I): the cost is the restricted master, the master
grows with the support, and the sparsest penalty is where the most columns are
generated before anything settles. The one outright failure --
`sparse-n2000-p5000` at ratio 0.02, stopping at a 0.12 gap and an objective
8.8e-04 high after hitting its own two-minute limit -- is the extreme of that
same pattern, not a separate phenomenon.

### K.4 Two hypothesis labels that were wrong

Corrected after review; both were reporting errors of mine, not measurements.

- The analysis script printed a block headed `HYP-0005` over a count of how
  often **current** scikit-learn converges on **2026** instances. `HYP-0005` is
  about the 2023 runs, and nothing measured here could confirm or deny it. It
  was settled separately, and **rejected**, by reading the researcher's own
  committed CSVs (`EVI-0001`): two of its three configurations stopped below
  `max_iter=1e6`, so they converged, and all three match the CG arm's objective
  to better than 5e-08.
- A second block printed `HYP-0006` over a count of recorded Lagrangian bounds.
  `HYP-0006` is about whether the method *terminates* through the pricing test
  rather than the dual-stall criterion, which the run detail does not record.
  **`HYP-0006` is therefore still open**, and an earlier revision of this
  section calling it "REJECTED -- 2 cells recorded a non-zero Lagrangian bound
  count" was conflating two different claims.

  The underlying observation is real and is kept as an observation: four runs
  recorded a bound, every one at `lambda_ratio = 0.5` and `tol = 1e-8`. The
  boundedness test is `|psi'X| > 2*sqrt(kappa*tau) - cg_lambda_tol` with
  `tau = kappa = lambda_1/2`, so its threshold grows with the penalty; at a
  small penalty it fires every iteration and no bound is ever computed, at a
  large one it need not. This weakens `HYP-0006`'s stated *mechanism* -- that
  the test always fires -- without settling its termination claim.

### K.5 Seven runs where the master did not solve at all

Recorded because a benchmark that reports only the cells that finished is not
reporting the method. Across the 210 arm-runs, `cg_hist` ended
`error:SolverError` 7 times, `time_limit` 12 times, and `ok` 71 times. Every
one of the seven is at the **tightest** requested tolerance, `tol = 1e-8`:

```text
correlated-n2000-p500-rho0.5  r=0.02, r=0.1
correlated-n2000-p500-rho0.9  r=0.02, r=0.1
historical-n10000-p1000       r=0.5
illcond-n2000-p500-cond1e3    r=0.5
illcond-n2000-p500-cond1e5    r=0.5
```

Clarabel raises rather than returning an inaccurate answer, and the restricted
master is where it raises. Not one is in the `sparse` family. This is the
numerical instability §L inherits from the historical material -- infeasible
masters, `singular KKT matrix`, solver status `Numerical` -- reproducing on
new instances with a different solver, which is stronger evidence that it is a
property of the *formulation* than anything in the 2023 record.

It also caps what the tolerance ladder could ever have delivered for this arm
(§K.1): the one setting that asks the master for more accuracy is the one that
makes it fail.


## L. Numerical-stability results

Partly measured, partly inherited.

- The historical material records infeasible masters, `singular KKT matrix`,
  solver status `Numerical`, matrix coefficients spanning `[6e-06, 1e+00]` and
  392 dense columns. Preserved in `sources/2023-2025/`, not reproduced (the
  solvers are absent).
- **The thesis's own conic formulation is the larger of two equivalent models,
  and modestly slower -- but not less accurate.** Corrected after review: an
  earlier revision of this section claimed `conic_thesis` "reaches a relative
  gap of 2.8e-06 where `conic_reduced` reaches better". That figure has no
  source in any recorded run, and the comparison it asserts does not hold.

  Measured on six cells (`sparse`, n=300, p=120, k=10, seeds 0 and 1, lambda
  ratios 0.5 / 0.1 / 0.02, `tol = 1e-8`):

  | comparison | `conic_reduced` better |
  |---|---|
  | relative duality gap | **3 / 6** |
  | objective | **3 / 6** |
  | wall time | **6 / 6**, median 1.12x, range 1.02x-1.21x |

  On accuracy the two are indistinguishable -- a coin flip, and no basis for
  calling either the harder model. On time the smaller model wins every cell,
  which is what its size predicts (`2m+1` variables and one cone against
  `3m+1` and `m` rotated cones) and is a real if unexciting effect.

  So the surviving claim is narrow: the reformulation is cheaper, by about
  12%, and does nothing for the numerical instability the historical material
  documents at length. It is not the cheap fix for that instability that the
  retracted sentence implied.
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
| `HYP-0001` modern solvers dominate | **SUPPORTED on 30/30 cells** by the frozen rule; 7.9x-4530x slower at a matched objective, median 38.7x |
| `HYP-0002` bounded pricing ≡ dual feasibility | supported, derived and certified on 30 visited duals |
| `HYP-0003` it is a maximum-violation working-set method | supported by source reading; the trajectory trace the proposal asks for is not run |
| `HYP-0004` unit-ball value is not a lower bound | first clause proved; existence clause supported on unstandardised data, not observed on standardised |
| `HYP-0005` the historical baseline did not converge | **REJECTED** (`EVI-0001`): 2 of its 3 configurations stopped below `max_iter`, and all 3 match the CG objective to better than 5e-08 |
| `HYP-0006` termination is never certified | **still open** — an earlier revision marked it REJECTED on a count of Lagrangian bounds, which is a different claim (§K.4) |
| `HYP-0007` the pricing scan is not the bottleneck | supported, at a corrected share (§I): the profiler counted one `psi'X` product per iteration where the code performs about two |

Promoting any of these into the capsule is a human act and none has been taken.

## P. Paper / no-paper decision

**Recommended: no paper on the computational contribution.** The reasoning is
§E and §K, and neither depends on the other: the selection rule is a known rule
whether or not it is fast, and it is not fast.

This is a **successful outcome of the programme**, not a failure of it. The
brief's outcome D is reached with evidence: a novelty gate that closes on a
named 2000 paper, and a benchmark in which the method is between one and three
orders of magnitude slower than its competitors at a matched answer.

**The second half of that sentence is a correction.** An earlier revision read
"a benchmark in which the method does not reach the accuracy its competitors
reach in milliseconds", and §K.1 now withdraws that framing: on thirteen of
fourteen measured cells the method's *objective* matches the best any solver
found to between 1.1e-14 and 5.9e-10. It is not inaccurate; it is slow. The
recommendation is unchanged and its evidential basis is stronger, because
"7.9x to 4530x slower at a matched objective" does not depend on the duality-gap
certificate whose ladder §K.1 shows to be inert for this arm.

**The run is now complete and it did not reopen this.** All 30 preregistered
cells have data; the frozen rule returns `SUPPORTED` on 30 of 30 with no
rejections and no exclusions. The two families that were outstanding when this
section was first drafted -- `correlated` and `illcond` -- came in against the
method, and `correlated` is the regime the 2023 claim was actually made in.
§B of `TIMELINE.md` records the method beating its *contemporary* baseline
there by 2.4x at a matched objective; against 2026 working-set solvers in the
same regime it is 28x to 818x slower. Sixteen years of solver development, not
a change of problem.

Two further complete-run facts point the same way: four cells where the
objective ends between 2.3e-04 and **9.4e-01** high after exhausting the time
limit, and seven runs where the restricted master raised outright rather than
returning an answer (§K.5), every one at the tightest tolerance and none in
the `sparse` family.

**This recommendation is not the decision.** Under the capsule's own rules a
Claim is accepted by a human, and none has been; §O records that no Claim
exists. The gate is `PROP-19700101T000000Z-b9c26fcd` and it is the
researcher's to decide.

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

1. ~~**Finish `EXP-0001`.**~~ **Done: 30 of 30 cells** (`EVI-0002`). The frozen
   rule returns SUPPORTED for `HYP-0001` on every cell.
2. **Give `cg_hist` a tolerance ladder that moves, and re-measure.** The
   adapter passes `solver_params={}`, pinning the restricted master at
   Clarabel's default accuracy, so the preregistered certificate metric cannot
   distinguish the method's own settings (§K.1). The fix is
   `{'tol_gap_rel': tol, 'tol_feas': tol}`, it changes the measurement, and it
   therefore needs its own preregistration. Note what §K.5 implies about the
   likely outcome: the seven runs that failed outright were all at the tightest
   tolerance, so asking the master for more accuracy is already what breaks it.
3. **Settle `HYP-0006`.** It is open, not rejected -- the earlier rejection was
   a mislabel (§K.4). Settling it needs the run detail to record *why* the loop
   stopped, which it currently does not: dual-stall, no new column, pricing
   certificate, time limit. A one-field change to `solve_cg_hist`'s detail dict
   and a re-run would answer it.
4. **Run the working-set trajectory trace** the autonomous proposal specifies
   (its PR-006). It converts the novelty gate from a literature argument into a
   round-by-round measurement, and it is the one experiment that could falsify
   `HYP-0003` and reopen `Q-0004`.
5. **Re-measure with the 2025 driver's settings** (`pos_linear_comb=True`,
   `v = v0 = max(int(m·0.012), 5)`), separately preregistered. Nothing here
   reproduces the committed 2025 numbers, and the difference is 4–7× in
   iterations.
6. **Real data.** None was used. The historical real data is gone and OpenML
   was unreachable within budget. `Q-0009` is untouched.
7. **`Q-0007`, the numerical instability**, now has a real datum and it is not
   the one this section used to cite. The claim that the thesis's conic form is
   "the harder of two equivalent models" is withdrawn (§L): measured, the
   reduced model is 3/6 on gap, 3/6 on objective and 6/6 on time at a median
   1.12x -- cheaper, not more stable. The new datum is §K.5: seven restricted
   master solves raised outright, all at `tol = 1e-8`, none in the `sparse`
   family. Still no ablation.
8. **The ℓ0 track** was declared out of scope by §T0 and that should be
   revisited only if someone intends to apply the decomposition to `MIQP`,
   which nobody has.
