# State

**As of 2026-09-17.** Programme opened. Sources inventoried, the 2023–2025
development reconstructed, and the frontier written as ten open questions and
the hypotheses that would settle them.

Nothing is accepted. There are no Claims in this capsule and there should not
be until evidence exists.

## Where the frontier is

- `Q-0001` … `Q-0010` — open.
- `HYP-0001` … `HYP-0007` — active and untested.
- `ASM-0001` … `ASM-0004` — the assumptions everything below rests on.

## What is already established, and how

Not as Claims — as reconstruction, recorded in `docs/2026/TIMELINE.md` with the
artifact and line each statement comes from:

- the conic model is an exact LASSO reformulation, `λ = 2√(τκ)` (§T0, and
  independently in the whiteboard §2.2);
- the historical pricing rule selects the `v` coordinates of largest
  `|(ψᵀX)ᵢ|` and adds signed unit vectors, so the restricted master is the
  LASSO on a coordinate subset (§2023);
- the two 2025 pricing ideas were measured by their author and are ~4× slower
  than the method they were meant to improve (§2025-B);
- termination is a dual-stall heuristic, not an optimality certificate
  (§2025-D).

## Next

The two falsification questions, in this order:

1. **Empirical** — does the historical speed regime survive a fair comparison
   against Celer, skglm and current scikit-learn, at matched objectives,
   matched tolerances and controlled threads? (`Q-0001`, `HYP-0001`.)
2. **Theoretical** — is bounded pricing exactly full-LASSO dual feasibility,
   and is top-violation selection already a known working-set rule? (`Q-0002`,
   `Q-0003`, `Q-0004`; `HYP-0002`, `HYP-0003`.)
