# The unit-ball pricing device: verdict

The brief's §13 asks for exactly one of three labels, based on proof:

    VALID_LOWER_BOUND_ORACLE
    VALID_COLUMN_OR_DIRECTION_HEURISTIC_ONLY
    INVALID_OR_UNJUSTIFIED

## Verdict

```text
As a source of columns:   VALID_COLUMN_OR_DIRECTION_HEURISTIC_ONLY
As a lower bound:         INVALID_OR_UNJUSTIFIED
```

And a third statement that neither label covers, which is the one this audit
would most want a reader to carry away:

```text
On the duals the implemented method actually reaches, across 30 visited duals
on six instances from four families, the unit-ball value did not exceed the
primal optimum once. "Unjustified" is what the mathematics supports. "Observed
to be wrong on this method" is not, and is not claimed.
```

---

## 1. What the 2025 notes do

`sources/2023-2025/202504 - LASSO - Unbounded subproblem case.pdf` §1.1. When
`∃i : |(ψ'X)ᵢ| > λ₁` the pricing subproblem is unbounded, so — in the note's
words — "we do not have any useful solution to add to the master problem".
Instead of an unbounded ray it solves

```text
(L)  min_β  λ₁‖β‖₁ + ψ'Xβ − ψ'y − μ²/4     s.t.  ‖β‖₂ ≤ 1
```

and reads the value as a **lower bound**: *"We can get a better behavior of the
lower bound (Lagrangian) and upper bound (master problem) … and in the old
version we didn't even have a lower bound."*

## 2. Why that reading is not justified

Adding a constraint to a **minimisation** raises its optimum:

```text
g_ball(ψ) = d(ψ) + min_{‖β‖₂ ≤ 1} [ψ'Xβ + λ₁‖β‖₁]
         ≥ d(ψ) + min_{β}        [ψ'Xβ + λ₁‖β‖₁]  =  g(ψ)
```

and `g(ψ) ≤ p*` is the only inequality weak duality supplies. `g_ball` sits on
the wrong side of it. When the pricing is unbounded — which is the only
situation in which the device is ever used — `g(ψ) = −∞`, a valid and useless
bound, and `g_ball(ψ)` is finite with nothing keeping it below `p*`.

The restriction would be legitimate if the *primal* supplied a matching bound
`‖β*‖₂ ≤ 1`. It does not: the LASSO's solution norm is whatever the data makes
it. A ball of radius `R ≥ ‖β*‖₂` would be legitimate; the notes use radius one.

**A counterexample, verified numerically**
(`tests/test_pricing_math.py::test_p6_the_unit_ball_value_can_exceed_the_primal_optimum`):

```text
X = [[1]], y = [10], λ₁ = 1
primal optimum        β* = 9.5,  p* = 9.75
ψ = −1.5              dual-infeasible: |X'ψ| = 1.5 > 1 = λ₁, so g(ψ) = −∞
d(ψ) = −1.5²/4 + 15 = 14.4375
min_{|β| ≤ 1} (−1.5β + |β|) = −0.5  at β = 1
g_ball(ψ) = 13.9375                 43 % ABOVE the primal optimum
```

A companion test shows the device is correct at dual-*feasible* points — where
the unbounded minimum is already zero and the restriction changes nothing —
which is precisely where it is never used.

## 3. Why the counterexample is not the whole answer

`ψ = −1.5` is a point in ℝⁿ. The algorithm's `ψ` is not: it is the optimal
second-order-cone dual of a restricted master at some iteration. A statement
about arbitrary `ψ` and a statement about the `ψ` this method reaches are
different statements, and only the second bears on the method.

**The autonomous cycle's own proposal caught this.** Asked to disposition the
project's hypotheses, it wrote that `HYP-0004`'s existence clause —
"there exist *reachable* dual points at which it exceeds the primal optimum" —
"is unfalsifiable until 'reachable' is defined, and should be pinned to
'duals arising at some iteration of the implemented method'".

So it was pinned, and measured. `src/cg2026/duals.py` captures the SOC dual at
every master solve of the real method; `scripts/adjudicate_pricing.py`
evaluates the pricing infimum, the feasibility indicator, the unit-ball value
and a converged reference optimum at each one.

```text
duals visited                                     30
instances                                          6   (four families)
duals where g_ball > p*                            0
worst relative excess                            0.0
```

**Not one.** The verdict on the existence clause is
`NOT_OBSERVED_ON_REACHABLE_SET`, which is weaker than refutation and is
reported as such rather than collapsed into it. Thirty duals on small
instances is not a proof that it cannot happen; it is evidence that it does not
happen readily, and the reason it does not is worth stating: the master's dual
is the dual of a *restriction* of the primal, so `d(ψᵏ)` tracks an upper bound
on `p*` from above as the column set grows, and the ball's penalty term is
subtracted from it.

## 4. Why the device is nonetheless fine as a column source

The whiteboard frames the same construction correctly and differently
(`sources/2023-2025/Decomposition_Method_for_Feature_Selection ___.pdf` §2.3):

> When the pricing is unbounded, it is enough to provide an extreme ray of the
> pricing to the master: this can be done by normalizing it in some way, maybe
> some are better than others.

That is sound. Any direction of decrease is a legitimate column in
Dantzig–Wolfe, and normalising an unbounded ray is a reasonable way to choose
one. The 2025 notes' error is in what they *read off* the device, not in what
they add to the master. Two artifacts, two framings, one construction, and only
one of the framings is defensible — recorded in `docs/2026/PROVENANCE.md` as
disagreement **D4**.

## 5. And it is slower anyway

The author measured it, on his own instance, and committed the result
(`results/unboundedness_policy/`, n = 10000, m = 1000, MOSEK, identical
parameters):

| policy | wall time | CG iterations |
|---|---|---|
| `v_solution` (thesis-era) | **0.262 min** | 13 |
| `unit_ball` (2025) | 1.010 min | 74 |
| `negative_gradient` (2025) | 1.006 min | 72 |

Both 2025 pricing ideas are ~4× slower than the 2023 method they were meant to
improve, at the same objective, and the note's own reading agrees: *"it starts
to oscillate instead of converge … still worse than v_solution alone."*

The brief's instruction not to repeat already-failed 2025 ideas without a new
theoretical reason applies here, and this audit supplies a theoretical reason
**against** the one use that motivated them.

## 6. What would change the verdict

- A primal bound `‖β*‖₂ ≤ R` derivable from the data, plus the ball at radius
  `R`. Then the restriction is legitimate and the value is a genuine bound. For
  the LASSO, `‖β*‖₁ ≤ ‖y‖²/λ₁` follows from `F(β*) ≤ F(0)`, which gives an
  `R` — but a data-dependent one, not one.
- Or: read the unit-ball solution as a **direction** and evaluate the true
  Lagrangian at the resulting column, which is free and is a valid bound.

Neither is done in the historical code, and neither is proposed here as a
contribution: the first is a standard trick and the second is what a correct
implementation would already do.
