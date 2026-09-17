# Charter

## What this project is

A 2023 master's thesis at Universidad de Chile
(`https://repositorio.uchile.cl/handle/2250/198750`) proposed a second-order
cone formulation for feature selection in linear regression and a column
generation method for solving it at scale. Work resumed at MIT in April 2025
and stopped in June 2025.

This programme is the **reassessment** of that work under the theory,
algorithms and software available as of 2026-09-17.

## The objective

> Reassess the 2023 conic column-generation approach for sparse regression
> under the theory, algorithms and software available as of 2026-09-17;
> reconstruct its 2023–2025 development; determine which historical claims
> survive; identify the strongest genuinely novel contribution if one remains;
> perform a modern fair computational evaluation; and produce a
> publication-quality paper only if supported by the evidence.

## What is not assumed

**That there is a paper here.** "Modern literature and algorithms subsume the
contribution and no sufficiently strong paper remains" is a valid outcome of
this programme and, if the evidence supports it, the correct one. Nothing in
this charter commits the project to publishing.

**That the historical conclusions hold.** The poster's claims — that the
decomposition method is fastest between 75 selected features and 40 % of the
feature count, and that execution time does not depend on instance size — are
evidence to reassess, not results to defend. `docs/2026/TIMELINE.md` §B records
why the baseline they rest on is in question.

**That the method solves what it was designed to solve.**
`docs/2026/TIMELINE.md` §T0 establishes that the conic model is an exact
reformulation of the LASSO rather than a relaxation of ℓ0, which changes the
comparison class.

## Scope

In scope: the ℓ1 conic model and its column-generation decomposition; the true
Elastic-Net extension; the pricing/working-set theory; a fair modern
computational comparison; the numerical instability recorded in the historical
material.

Out of scope unless the evidence forces it back in: the ℓ0 / best-subset
track. The method has never been applied to it (`models.py::MIQP` exists and is
never decomposed), and a perspective formulation without a binary variable is
not a perspective formulation.

## Standards this project holds itself to

- An objective-function scaling conversion between two solvers is unit-tested
  before any timing produced with it is reported.
- A timing comparison controls threads and reports them.
- A reproduction is called a reproduction only when the original code, solver
  and seeds ran. MOSEK is not licensed on this host, so nothing here can be an
  original reproduction of a MOSEK number.
- A negative result is a result. A falsified hypothesis is recorded as
  falsified, not quietly dropped.
