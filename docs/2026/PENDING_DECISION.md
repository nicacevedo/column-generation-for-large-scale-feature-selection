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

## What has changed since this file was written

`EXP-0001` has since completed all 30 preregistered cells and the frozen
decision rule returns **SUPPORTED on 30/30** for `HYP-0001` (`EVI-0002`). That
does not decide the proposal, and it does not touch PR-006 -- the
round-by-round working-set equivalence trace is still the one outstanding
experiment that could falsify `HYP-0003` and reopen the novelty question.

If anything the completed run raises PR-006's value. The benchmark establishes
that the method is slow; it does not establish *that it is the known rule*, and
`HYP-0003` currently rests on source reading alone (§O of
`SCIENTIFIC_REPORT.md`). The no-paper recommendation in §P leans on both legs
independently, so PR-006 is the cheapest way to test the leg that has not been
measured.

An adversarial review of the empirical case also landed, and several of its
findings changed what the benchmark is taken to show -- the headline is now a
matched-objective slowdown rather than a failure to reach accuracy. §K.1 and
§K.4 of `SCIENTIFIC_REPORT.md` list every retraction. None of it bears on
whether to promote or decline any item here.

## What follows either way

Research OS observes the capsule, emits one deduplicated `CAPSULE_CHANGED`
event, and opens a successor cycle with recorded lineage. You do not need to
tell it anything.
