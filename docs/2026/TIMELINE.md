# The 2023 → 2025 development timeline

Reconstructed on 2026-09-17 from the eight commits of this repository, the code
they contain, the result files they committed, and the nine source documents in
`sources/2023-2025/` (see `docs/2026/PROVENANCE.md`).

Written **before** any new algorithm was implemented, for one reason: several
of the ideas the 2025 notes propose as next steps were already tried, and two of
them were already measured and found worse. Rediscovering them would have cost
weeks.

Classification, as the brief defines it: `IDEA_ONLY`,
`IMPLEMENTED_NOT_TESTED`, `TESTED_FAILED`, `TESTED_MIXED`, `TESTED_SUPPORTED`,
`SUPERSEDED`, `CURRENT`.

---

## The commits

```text
65448c4  2023-12-16  Initial commit                                         (README only)
72c9aca  2024-01-30  [Creation] Moved stuff from the old repo & updating models_simple
471a414  2024-01-31  [Update + Features] Elastic Net SOCP & Scikit-learn. Name change of new_main
c3c223d  2024-02-03  [Feature] Elastic Net w/ CG (only one pricing). Some fo_value differences. Check it
d1c80bc  2024-02-21  [Update] All comparisons: lasso models, alphas for CG_SOC1, and more
bd50256  2024-03-23  [Update] Model comparison, lasso_soc2 update, synthetic databases
eec123d  2025-04-21  Continuing research at MIT on April 2025
58d8b62  2025-06-07  [Feature] Code refactor & new pricing approaches
```

Eight commits, one branch (`master`), no tags, no abandoned branches. The
thirteen-month gap between 2024-03-23 and 2025-04-21 is the gap between the
thesis and its resumption at MIT.

`master` is unchanged by this work; everything here is on
`research/2026-reassessment`.

---

## §T0 — What the model actually is, and where that was decided

This is the single most consequential piece of archaeology, because it decides
which literature the work must be compared against.

`sources/2023-2025/Tesis_MGO.pdf` (the earliest derivation) starts from the ℓ0
problem:

```text
min ‖y − Xβ‖² + κ Σ zᵢ      s.t.  −M zᵢ ≤ βᵢ ≤ M zᵢ,  z binary
```

rewrites it as `min ‖y − Xβ‖² + Σ βᵢ²/zᵢ` — the perspective function — and then
as `min ‖y − Xβ‖² + κ̃ Σ tᵢ` subject to `βᵢ² ≤ tᵢ zᵢ`, finally in rotated-cone
form. Between the first line and the last, **the binary restriction on `z` is
dropped and never restored.**

The consequence is exact and is stated on the poster's own LASSO Equivalence
box: with `z, u ∈ ℝᵐ₊` continuous and `βᵢ² ≤ zᵢuᵢ`,

```text
min_{z,u ≥ 0} τ z + κ u   s.t.  β² ≤ z u     =     2√(τκ) |β|
```

by AM–GM, attained. So the thesis's SOCP is an **exact conic reformulation of
the LASSO** with `λ = 2√(τκ)`. It is not a relaxation of ℓ0, it is not a
perspective formulation of anything (a perspective formulation needs the
binary), and its optimal value is the LASSO's optimal value, not a bound on the
ℓ0 problem's.

The ℓ0 problem does exist in the code — `models.py::MIQP`, big-M = 1e6, `z`
boolean — as a **separate model that the column-generation method is never
applied to**.

Status: `CURRENT`, and it reclassifies the whole programme. The relevant
comparison class is *LASSO solvers*, not best-subset solvers.

---

## §W — The whiteboard, and what it already proves

`sources/2023-2025/Decomposition_Method_for_Feature_Selection ___.pdf`,
undated, multi-author, unfinished. It contains four results that the 2025
notes present as open or new.

**W1. The Lagrangean dual of the SOCP, simplified to the LASSO dual.** §2.1
derives

```text
max_λ  −‖λ‖²/4 − λᵀy      s.t.  ‖Xᵀλ‖_∞ ≤ 2√(τκ)
```

and §2.2 observes that dualising it back gives "exactly a lasso model with
parameter 2√(τκ)" — an independent proof of the equivalence in §T0.
`TESTED_SUPPORTED` (as mathematics: derived, and re-derived independently in
this audit).

**W2. The pricing-boundedness criterion.** Immediately after W1: *"We can
already see a pricing-boundedness criterion when fixing λ."* The criterion is
`‖Xᵀλ‖_∞ ≤ 2√(τκ)`, which **is** the dual feasibility constraint. So the brief's
Q2 — is bounded pricing exactly full-LASSO dual feasibility — was already
answered affirmatively here, in a private note, and never written up.
`IMPLEMENTED_NOT_TESTED` → re-derived and stated as a proposition in the
mathematical audit.

**W3. §2.4 "Why is the pricing bounded only at the final iteration?" —
`[Todo renaud]`, never answered.** It follows in one line from W2 and is
answered in the mathematical audit. `IDEA_ONLY`.

**W4. §2.3 "When the pricing is unbounded, it is enough to provide an extreme
ray of the pricing to the master: this can be done by normalizing it in some
way, maybe some are better than others."** This is the correct framing of the
unbounded case, and it is *not* the framing the 2025 notes use. Extreme-ray
generation is a statement about which **column** to add; the 2025 notes use the
same device to produce a **lower bound**, which is a different claim. See §2025-B.
`IDEA_ONLY` in the whiteboard; implemented in 2025.

**W5. The ℓ0 + ℓ2 track, done properly and separately.** §3 takes
`min ‖Xβ−y‖² + α₁Σzᵢ + α₂‖β‖²` with `z ∈ {0,1}ᵐ`, `|βᵢ| ≤ Mzᵢ`, and derives
`conv(F) = {(β,z,u) : |β| ≤ Mz, β² ≤ uz}` — the standard perspective /
rank-one convexification. §3.3 then shows that **dropping `zᵢ ≤ 1` and the
big-M collapses it to the LASSO**: *"si se guarda zᵢ ≤ 1 o bigM: elasticnet ≠
lasso, pero se tiene la equivalencia si se relajan esas cotas sup. y bigM."*

This is the same collapse as §T0, identified contemporaneously, in writing, and
then not carried into the implemented method. Recorded here because it means
the conflation between "Elastic Net" and "ℓ0 + ℓ2" the brief warns about in
§15 was **present in the source material from the beginning** and was already
understood by at least one of the authors. `TESTED_SUPPORTED` as mathematics,
`SUPERSEDED` as a direction — the implemented method took the other branch.

**W6. A closed form for Elastic-Net pricing that is wrong.** §3.4 adds a
`‖β‖₂ ≤ ξ₂` term and a `½θξ₂²` objective term — the *true* Elastic Net — and
states, "assuming the equality holds in the conic constraints on the optimal
point", `β* = −(1/θ)(2√(τκ) + ψᵀX)`. The assumption is exactly what fails at
`βᵢ = 0`, and the formula has no soft threshold and no sign case. It is correct
only on the coordinates with `(ψᵀX)ᵢ < −λ₁`. Corrected in 2025 (see §2025-A).
`TESTED_FAILED` as stated.

**W7. Numerical failure, recorded in detail.** §5 "Numerical errors" carries a
MOSEK log with `PRIMAL_INFEASIBLE` / `Problem status: unbounded` /
`Optimal value: −inf` on a real-data master at iteration 3; a CVXOPT run
terminating `singular KKT matrix`; and a COPT run finishing with status
`Numerical`, dual infeasibility 1.08e-02 absolute, 392 dense columns, matrix
coefficients spanning `[6e-06, 1e+00]` and cost coefficients `[1e+00, 1e+03]`.
It also records what helped: *"Solved some SOCP problems standardizing X and
y."* `TESTED_MIXED` — the instability is real, reproducible in kind, and
partially addressed by standardisation.

---

## §2023 — The thesis-era method (CG-HIST)

Implemented in `models.py` (4 927 lines) across the 2024 commits and unchanged
in substance since.

| function | what it is | status |
|---|---|---|
| `MIQP` | ℓ0 with big-M = 1e6, `z` boolean | `SUPERSEDED` — never decomposed |
| `MIP_R` | its relaxation | `SUPERSEDED` |
| `SOCP`, `SOCP_v2` | the conic model of §T0, solved whole | `CURRENT` (as a reference solve) |
| `L_LASSO` | the same model in "relaxed"/ℓ1 form, solved whole | `CURRENT` (reference) |
| `L_LASSO_scipy` | L-BFGS-B on the objective | `TESTED_FAILED` — the source comment says *"NOT STABLE SOLUTION. MAYBE DELETE IT"* |
| `CG_SOC1_upgrade` | the original CG, `ψᵀX ≥ τ+κ` condition | `SUPERSEDED` by `CG_LASSO_SOC1_v2` |
| `CG_LASSO_SOC1` | LASSO CG, ℓ1 as a constraint | `SUPERSEDED` |
| `CG_LASSO_SOC1_v2` | LASSO CG, ℓ1 in the objective | `CURRENT` — the method |
| `CG_LASSO_SOC2` | relaxing the ℓ1 cone instead | `IDEA_ONLY` — *"NOT TRIVIAL. THINK ABOUT IT LONGER. NOT READY TO USE"* |
| `CG_SOC1_SOC2_upgrade` | relaxing both cones | `IMPLEMENTED_NOT_TESTED` |
| `CG_SOC2` | relaxing only the second cone | `TESTED_FAILED` — the closing comment states it cannot be done: *"No se puede hacer CG_SOC2 así como así… como no podemos armar la sol sintética entera, entonces no podemos hacer CG_SOC2."* |
| `CG_SOC1_ElasticNet` | Elastic-Net CG, one pricing | `TESTED_MIXED` — commit c3c223d's own message: *"Some fo_value differences. Check it"* |
| `SCIKIT_LASSO`, `SCIKIT_ElasticNet` | the baselines | `CURRENT` — and see §B |

**What CG-HIST does, stated plainly, because the name obscures it.** The master
optimises `β = Bᵏπ` over the span of the generated columns `Bᵏ`. When pricing
is unbounded it adds `v` columns, each a **signed canonical vector `±eᵢ`** on
the `v` coordinates with the largest `|(ψᵀX)ᵢ|`. Since `π` is unrestricted by
default (`sum_1_comb=False`, `pos_linear_comb=False`), the master's reachable
set is `span{eᵢ : i ∈ S}` plus the seed columns — that is, **the LASSO
restricted to the coordinate set `S`**.

So CG-HIST is a working-set method over coordinates, selecting by maximum
dual-constraint violation, whose restricted problem is solved as a conic
program. The consequence for novelty is the subject of the mathematical audit.

One deviation from that description, worth recording because it is unexplained:
with `add_constant=True` the seed column is the **all-ones vector in β-space**,
`np.ones((m,1))`, which is not a coordinate direction and ties every coefficient
together. The 2025 driver sets `add_constant = False`.

---

## §B — The baseline, and why it is the crux

The poster's headline result is a comparison against `LASSO ... solved via
Coordinate Descent`, i.e. `SCIKIT_LASSO`. The wrapper is
`models.py:28`:

```python
clf = Lasso(alpha=tau / n, fit_intercept=False, max_iter=1 / tol, tol=tol)
```

**The λ conversion is correct.** The target objective is
`‖y − Xβ‖² + 2τ‖β‖₁`; scikit-learn minimises `(1/2n)‖y − Xβ‖² + α‖β‖₁`, so
`α = τ/n`. Checked; it is right, and it is the first thing that would have
invalidated everything.

What the committed result files show is a baseline that is **slow, and -- in
two of three configurations -- converged**:

| file | n | m | λ scale | `SCIKIT_LASSO` time | `n_iter_` | hit `max_iter`? |
|---|---|---|---|---|---|---|
| `method-benchmark/…correlated_data.csv` | 1000 | 500 | `tau_exp=1` | 1.07 min | 217 912 | no |
| `method-benchmark/…correlated_data.csv` | 1000 | 500 | `tau_exp=2` | 5.61 min | **1 000 000** | **yes** |
| `method-benchmark/…real_data2.csv` | 10000 | 100 | `tau_exp=3` | 0.72 min | 73 363 | no |

Times are minutes (`models.py` returns `(t1-t0)/60`); the cap is `max_iter=1e6`
(`models.py:99`).

**An earlier revision of this file called all three rows "a solver that did not
reach its tolerance". That was wrong, and an adversarial review caught it.**
Only the `tau_exp=2` row hit the cap. The other two stopped *below* it, which
for scikit-learn's coordinate descent means its own duality-gap test was
satisfied -- they converged. The claim is retracted.

Worse for the retracted reading: the objectives agree. On the same
`tau_exp=1` row, `SCIKIT_LASSO` reports `fo_value = 9.962080731336979` and
`CG_LASSO_SOC1_v2` reports `9.962080551706714` -- a relative difference of
**1.8e-08**. Even the capped `tau_exp=2` row agrees to **4.9e-08**. Whatever
else was happening, the 2023 baseline was returning the right answer.

So the 2023 timing comparison is **not** invalidated by non-convergence, and
the honest reading of that row is the uncomfortable one:

> at `tau_exp=1`, `CG_LASSO_SOC1_v2` took **0.449 min** against
> `SCIKIT_LASSO`'s **1.068 min** at an objective matched to 1.8e-08. In the
> correlated regime the method was designed for, the 2023 method was
> **2.4x faster** than the 2023 baseline.

That is a real datapoint in the historical method's favour, it is recorded
here because it is in the researcher's own committed results, and it is
exactly the regime `EXP-0001`'s `correlated` and `historical` families exist
to re-measure against 2026 working-set solvers. Nothing in the `sparse`-family
numbers reported in `SCIENTIFIC_REPORT.md` §K speaks to it.

What remains true is narrower: cyclic coordinate descent is at its worst here.
The `tau_exp` sweep drives `λ = 2·error_quad_OLS/m^{tau_exp}` toward zero -- at
`tau_exp=2`, `τ = 3.6e-05` -- which is the nearly-OLS, nearly-dense regime, and
the one configuration that did hit the cap is the one with the smallest λ.

And in the one committed regime with `p > n`:

| file | n | m | `SCIKIT_LASSO` | `CG_LASSO_SOC1_v2` |
|---|---|---|---|---|
| `…synthetic_data2.csv` | 1000 | 5000, 241 selected | **0.0036 min** | 0.325 min |

**scikit-learn is 90× faster than the decomposition method there**, in the
committed data, in the high-dimensional regime that modern sparse regression is
about.

Status: `TESTED_MIXED`, and the mixture is the whole question. The historical
speed claim is **regime-specific and rests on a baseline that did not
converge**. It is not refuted by this reading — a non-converged baseline is
slow for a reason, and correlated designs are genuinely hard for CD — but it
cannot stand without a fair modern re-measurement. That is the first
falsification experiment.

---

## §2025-A — The April 2025 reformulation

`sources/2023-2025/20250408 - Reviewing my past work (Finally).pdf`.

Writes the **true Elastic Net** as an SOCP (`r² + λ₁e'z + λ₂e'u`, with
`uᵢ ≥ βᵢ²` a rotated cone), relaxes the least-squares cone only, and reaches

```text
(Lᵢ)  min_{βᵢ}  (ψ'X)ᵢ βᵢ + λ₁|βᵢ| + λ₂βᵢ²
```

with closed form `βᵢ = (−(ψ'X)ᵢ + sign[(ψ'X)ᵢ]·λ₁)/(2λ₂)` when
`λ₁ < |(ψ'X)ᵢ|`, else `0`.

**This is correct**, and it **corrects W6** — the same quantity the whiteboard
got wrong by assuming the conic constraints are tight. Verified independently
in the mathematical audit. `TESTED_SUPPORTED` as mathematics.

It also states the two degenerate cases: `λ₂ = 0` gives LASSO with an unbounded
subproblem when `λ₁ < |(ψ'X)ᵢ|`; `λ₁ = 0` gives Ridge with a single condition.
The LASSO statement is right except at equality, where it says `βᵢ = 0` — true,
but `0` is then one of infinitely many minimisers and there is a zero-cost
recession direction. That edge case is where the implementation's tolerance
lives, and it matters (§2025-D).

Open item it records and does not resolve: *"We also need to verify if it is
worth to use SOC formulations for βᵢ² terms on the master problem or not, in
terms of performance."* `IDEA_ONLY`.

---

## §2025-B — The unbounded-pricing explorations

`sources/2023-2025/202504 - LASSO - Unbounded subproblem case.pdf`, implemented
in `src/cg_models.py` as `unboundedness_policy ∈ {v_solution, unit_ball,
negative_gradient, mixture}`, committed 58d8b62 (2025-06-07).

The note opens by **flagging its own uncertainty about the bound**: *"I am
assuming constraining the master problem to just a set of columns is somewhat
an equivalent idea to having an integer feasible region… But is not the same in
our case."* That uncertainty is well placed and is the subject of the unit-ball
verdict.

**Measured, by the author, and committed** — `results/unboundedness_policy/`,
n = 10000, m = 1000, MOSEK, identical instance and parameters:

| policy | wall time | CG iterations | non-zeros | objective |
|---|---|---|---|---|
| `L_LASSO` (whole model, reference) | 3.594 | 1 | 154 | 8832.4694 |
| **`v_solution`** (thesis-era) | **0.262** | **13** (12 virtual) | 135 | 8832.4685 |
| `unit_ball` (2025) | 1.010 | 74 | 135 | 8832.4685 |
| `negative_gradient` (2025) | 1.006 | 72 | 135 | 8832.4685 |

Times in minutes. **Both 2025 pricing ideas are at least 3.8× slower than the
2023 method they were meant to improve, on the author's own instance**, and
reach the same objective.

**The two 2025 times are censored, and the `converged` column is meaningless.**
Corrected after review. `new_main.py:28` sets `time_limit = 1` minute, and
`cg_models.py:142` breaks the column generation loop as soon as elapsed time
passes it. Both 2025 rows sit just *past* that cap -- 1.0103 and 1.0057 minutes
-- so both were truncated mid-solve, and their wall times are **lower bounds**
on what the policies would have taken to finish. The `status` column says
`converged` for all three rows, but that string is a hardcoded literal written
unconditionally at `new_main.py:373`; it records nothing about how any run
ended.

Two consequences, in opposite directions:

- The multiplier is not "~4x". It is **"at least 3.8x"** (1.0057 / 0.2624 and
  1.0103 / 0.2624), with no upper bound available from this data. The
  conclusion that the 2025 ideas are worse is *strengthened* by the censoring,
  not weakened -- but the specific figure was never measurable and is
  withdrawn.
- The comparison rests on **n = 1** per policy, on a single instance, with no
  repetition and no variance estimate. Whatever it supports, it does not
  support a precise ratio.

What survives is the qualitative ordering, which the censoring makes safe:
`v_solution` finished in 0.262 min while both 2025 policies were still running
at 1.0 min, all three at objectives agreeing to about 1e-9.


The note's own reading agrees: *"The unit ball method (analogous behavior as
minus gradient) have a faster approach to the optimal value, but then it starts
to oscillate instead of converge… I tried to combine them… it did not work
quite well… It improved more doing mixtures of 1 iter of unit_ball and 1 iter
of v_solutions, but still worse than v_solution alone."*

| idea | status |
|---|---|
| unit-ball pricing (`unit_ball`) | **`TESTED_FAILED`** — slower, and oscillates |
| steepest-descent direction (`negative_gradient`) | **`TESTED_FAILED`** — slower |
| mixtures (`mixture`, `mixture2`) | **`TESTED_FAILED`** — *"still worse than v_solution alone"* |
| top-`v` largest divergence coordinates | `CURRENT` — §1.4, and it is what `v_solution` does |
| SGD/minibatch estimation of `ψ'X` | `IDEA_ONLY` — §1.3, never implemented |
| exact direction from `ψ'X` | `IDEA_ONLY` — §1.2, heading only, empty |
| gradient estimation | `IDEA_ONLY` — §1.2.1 is the single character "vi" |
| profiling the method | `IDEA_ONLY` — "Future Note", never done |

**The brief's instruction not to repeat already-failed 2025 ideas without a new
theoretical reason applies to the first three rows.** There is no new
theoretical reason for unit-ball or negative-gradient pricing; the audit
supplies a theoretical reason *against* using either as a bound.

---

## §2025-C — What the 2025 refactor changed operationally

`eec123d` and `58d8b62`. `src/cg_models.py` (824 lines) is a copy of the
relevant part of `models.py` with the four policies added; `models.py` is kept.
`new_main.py` is the current driver.

Changes that matter for any re-measurement:

- **Standardisation is now on**: `X = (X − mean)/std`, `y = (y − mean)/std`.
  The whiteboard had found this helped (W7); in 2025 it is unconditional.
- `add_constant = False` — the all-ones seed column of §2023 is off.
- `tau_exp = -1`, so `τ = κ = error_quad_OLS · m` and `λ₁ = 2√(τκ) = 2τ`.
- MOSEK with `MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-6` and
  `MSK_IPAR_NUM_THREADS = cpu_count()` — **threads uncontrolled**, which makes
  every committed timing a measurement of this machine's core count as much as
  of the method.
- `time_limit = 1` minute, which is why the 2025 note's transcript ends
  "ITERATION 58 … Time limit reached".

---

## §2025-D — Termination, which is not certified by the pricing

Read from `src/cg_models.py`. Two stopping criteria:

1. **dual stall** — `max|λᵏ − λᵏ⁻¹| < 1e-6`;
2. **no new column** — the new pricing solution regresses onto the existing
   columns with residual `< 1e-6`; *only checked when the added column is not a
   virtual solution* (`check_residuals = False` on every `v_solution`
   iteration).

The pricing-boundedness test itself is
`(|ψᵀX| − 2√(τκ) > −1e-6).any()`, i.e. it declares "unbounded" when any
`|aᵢ|` is within tolerance **below** `λ₁`. At optimality the KKT conditions make
`|aᵢ| = λ₁` **exactly** on the support, so this test is true at the solution and
stays true. The algorithm therefore terminates on the dual-stall heuristic, not
on an optimality certificate.

Status: `CURRENT`, and a defect. A method whose termination is a heuristic
cannot claim to solve the problem exactly, and the fix — using the bounded
pricing value as the certificate, which W2 shows is exactly dual feasibility —
is available.

---

## What was never done

Stated because absence is evidence too.

- **No profiling**, ever. The 2025 note lists it as a "Future Note".
- **No modern baseline.** The comparison set is scikit-learn coordinate descent
  and monolithic conic solves. Celer (2018), Blitz, Gap Safe screening,
  skglm and glmnet-class working-set solvers appear nowhere in the code, the
  notes, the poster's three references, or the thesis table of contents.
- **No ℓ0 comparison**, because the method does not solve ℓ0 (§T0). `MIQP`
  exists and is never compared against `L0Learn` or any exact method.
- **No memory measurement**, no KKT residual, no duality gap reported for the
  CG method, and no repetitions — every committed result is a single run.
- **No seed discipline** beyond `np.random.seed(123)` at import time in
  `models.py` and again inside each CG function.
