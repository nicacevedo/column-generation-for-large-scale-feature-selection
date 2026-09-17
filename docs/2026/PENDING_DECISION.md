# One decision is waiting, and it is yours

An autonomous Research OS cycle over this project produced a proposal and
stopped. Nothing in that session promoted or declined it, and nothing should
have: promoting writes into `.research/`, and declining closes a question the
runtime would otherwise keep asking. Both are scientific decisions.

This file is deliberately **not** in `.research/`. Writing it there would change
the capsule digest, which the runtime watches, and would spawn a successor
cycle before you have decided anything.

## What is waiting

```text
proposal   PROP-19700101T000000Z-b9c26fcd
run        RRUN-20260917T095239Z-42c57607   terminal: WAITING_FOR_SCIENTIFIC_DECISION
project    cg-sparse-regression
written at 9d5474e   (the basis has been re-checked since: unchanged)
items      12, none decided
grounded in FIND-20260917T095202Z-f63a6014
```

## Read it first

```bash
export RESEARCH_OS_RUNTIME_DSN='postgresql://postgres:@/research_os?host=$HOME/.local/state/research-os/devdb/cluster'
cd $HOME/research/research-os-rc

uv run --frozen researchctl propose show PROP-19700101T000000Z-b9c26fcd
uv run --frozen researchctl runtime findings
uv run --frozen researchctl runtime run RRUN-20260917T095239Z-42c57607
```

## Then decide

Promote one item into a DRAFT capsule object:

```bash
uv run --frozen researchctl propose promote PROP-19700101T000000Z-b9c26fcd --item PR-00N
```

Or record that you read it and do not want it:

```bash
uv run --frozen researchctl propose decline PROP-19700101T000000Z-b9c26fcd --reason "..."
```

Both require an interactive terminal and both ask before writing. `--item` is
optional on `decline`; omitting it declines every undecided item.

## What it says, and what this reassessment thinks of it

Its first item states that the finding it rests on is provenance-only and bears
on nothing substantive — which is correct, and is not what an eager system
writes about its own output.

Its second item observes that two of this project's hypotheses weld a testable
core to an unfalsifiable clause. **That criticism was correct and it changed
this work.** `HYP-0004`'s "there exist *reachable* dual points…" was
unfalsifiable until "reachable" was defined, so it was defined as "a dual the
implemented method produces at some iteration", instrumented
(`src/cg2026/duals.py`) and measured
(`scripts/adjudicate_pricing.py`). The answer turned out to depend on
standardisation, which took three attempts; see
`docs/2026/UNIT_BALL_VERDICT.md`.

Its PR-005 asks for exactly the analytic-plus-numerical adjudication of
`HYP-0002` and `HYP-0004` that has since been done. Its PR-006, the
round-by-round working-set equivalence trace, has **not** been done and is the
single most valuable outstanding experiment: it is the one thing that could
falsify `HYP-0003` and reopen the novelty question this reassessment closed.

None of that is a recommendation about how to decide. It is what you should
know before you do.

## What follows either way

Research OS observes the capsule, emits one deduplicated `CAPSULE_CHANGED`
event, and opens a successor cycle with recorded lineage. You do not need to
tell it anything.
