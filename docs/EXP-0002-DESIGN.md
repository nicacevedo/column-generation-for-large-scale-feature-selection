# EXP-0002 — Is the fast regime portable in λ/λ_max or in support size?

Recorded **before the sweep is run**. Nothing below may be edited after the
first result exists; a later change belongs in a new EXP.

Instrument added for `PIDEA-20260921T054529Z-7c68ed14 v2`, which the
autonomous portfolio adjudicated `empirical` and then refused to test four
times, correctly, because no declared command could vary the design or report
support size.

## Question

For this ℓ1-conic column-generation solver, when the same synthetic design
(fixed `n, p, k, SNR`) is run under different column-correlation structures,
does the "fast" interval of the rounds/λ curve shift **less** when indexed by
normalised dual margin λ/λ_max than when the same interval is re-expressed in
achieved cardinality `s`?

The thesis reports a fast regime at a support size. If that window is stable in
λ/λ_max and moves in `s`, the support-indexed claim is an artifact of the
indexing variable rather than a property of the method.

## Hypothesis and falsifier

    R = fold_change_support / fold_change_lambda

    R > 3      λ/λ_max is the more portable invariant; the s-indexed claim is
               correlation-structure dependent.          -> SUPPORTS
    R < 1/3    s is the more portable invariant, the opposite of the claim.
                                                         -> CONTRADICTS
    otherwise  both move comparably; neither is shown to be privileged.
                                                         -> INCONCLUSIVE

The three-way rule, including a named null, is the idea's own. The thresholds
live in the Research OS decision rule and **not** in `analyse_sweep.py`; a
script that knew them could be written to clear them.

## Inputs and parameter domain

| axis | values | held fixed |
|---|---|---|
| correlation structure | `sparse` (independent); `toeplitz` ρ ∈ {0.3, 0.6, 0.9}; `block` ρ ∈ {0.6, 0.9}, block 50 | — |
| λ/λ_max | 0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.09, 0.06, 0.045, 0.03, 0.02 | — |
| n, p, k, SNR | — | 2000, 500, 25, 5.0 |
| solver | — | `cg_hist`, tol 1e-6 |
| seed | 0 for every instance | — |
| repetitions | 3 | — |
| threads | 1 | — |

Six designs × twelve λ = **72 cells, 216 solves.** `toeplitz` and `block` are
built to differ from `sparse` in the column covariance and in nothing else —
same `k`, same SNR construction, same coefficient law — which is what makes a
shift attributable to correlation. `tests/test_correlation_families.py` checks
the covariance each one actually produces.

## Metrics

Per cell, median over repetitions: `iterations` (pricing rounds, the primary
cost), `wall_seconds` (the same question asked of the clock), `nnz` (achieved
support size `s`). Also recorded: `peak_rss_mib`, `time_per_round`.

`time_per_round` is the falsifier's own secondary control in computable form:
if wall clock tracks round count it is flat across designs, and if it is not
flat then rounds and time are answering different questions. It is reported
alongside, never substituted for the primary.

## Decision rule input

One number in one JSON document:

    results/2026/EXP-0002-sweep.json    ->    portability.R

`portability_wall_clock.R` carries the same computation over `wall_seconds`
and is secondary.

## Resource ceiling

Per cell 300 s (in the plan). Whole command 5400 s. Measured on this host:
the most expensive probed cell is 14.7 s (`sparse`, λ = 0.02); typical cells
are 1–6 s; the estimate for the full sweep is 10–30 minutes.

## Expected artifacts

    results/2026/EXP-0002.jsonl          one line per cell, from run_benchmark
    results/2026/EXP-0002-sweep.json     the analysed document
    results/2026/EXP-0002.manifest.json  written by run_benchmark

## Failure semantics

An operational failure must stay operational. `analyse_sweep.py` omits
`portability.R` entirely — rather than emitting a number — when any of:

- a cell has no successful repetition (timeout, solver error, crash);
- a cost or support value is missing, `NaN`, infinite, a string or a bool;
- a window boundary is zero or negative, making a fold change undefined;
- fewer than two designs produced a window;
- a design has fewer than three usable λ points.

Every one of these writes a line into `problems[]` and sets
`portability.status`. A decision rule reading `portability.R` then finds
nothing to compare, which the Research OS bridge records as INSUFFICIENT —
an absent measurement, not evidence against the hypothesis.

The document is written with `allow_nan=False`, so it cannot contain a token
only Python can read, and the input is parsed with `parse_constant` refusing
`NaN`/`Infinity` for the same reason.

## What this instrument cannot answer

It cannot speak to the other three refused ideas. `2396c265` needs a
warm-start toggle and a solver-parameter dump, neither of which exists.
`e9e1551d` needs original-era hardware and an original-era solver binary, which
no instrument can supply. `b91e45b7` needs a fitted size × difficulty
interaction, which is a second analysis over a second plan — reachable with
this same machinery and not attempted here.
