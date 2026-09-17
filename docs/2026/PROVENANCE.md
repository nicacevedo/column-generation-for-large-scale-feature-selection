# Source provenance inventory

Every historical artifact this reassessment reads, with its content hash, where
it came from, what epistemic role it has, and — where they exist — the
disagreements between artifacts. Recorded on 2026-09-17.

Nothing here is treated as established theory. These are **untrusted scientific
source material**: a private note is a note, a poster is a claim snapshot, and a
thesis-shaped PDF whose abstract is the word "Resumen" is a draft.

---

## 1. Where the files are, and where they were

All nine PDFs were found inside the Research OS implementation repository at
`$RESEARCH_OS_DIR/docs2/`, committed there in three commits on
`r5/autonomous-runtime`:

```text
f69e99d  [Test] File update                        docs2/hola.txt
e69eddf  [Update] Files upload for test on thesis   7 PDFs
76882c3  [Update] Final reference files for thesis  2 PDFs (the 2025 notes)
```

That is the wrong repository for them — they are this project's scientific
sources, not Research OS's — so they are copied here, to
`sources/2023-2025/`, byte for byte. The originals are left where they are;
nothing was deleted from either repository.

**Nothing was vendored in the other direction.** No Research OS code, schema or
configuration is copied into this repository. The only relationship recorded is
the Research OS commit used, in `docs/2026/RESEARCH_OS.md`.

---

## 2. Files the brief named that are NOT present

Recorded precisely rather than silently worked around.

| named in the brief | present? |
|---|---|
| `Decomposition_Method_for_Feature_Selection.pdf` | **as `Decomposition_Method_for_Feature_Selection ___.pdf`** (trailing space + three underscores). Same stem; assumed the same artifact |
| `Memoria_Tesis__MGO___Final___ENG_.pdf` | **ABSENT** |
| `Memoria_Tesis__MGO___Final___ENG_ (1).pdf` | **ABSENT** |
| `Poster_MSWorkshop.pdf` | present, exact name |
| `20250408 - Reviewing my past work (Finally).pdf` | present, exact name |
| `202504 - LASSO - Unbounded subproblem case.pdf` | present, exact name |

Searched: `$THESIS_REPO_DIR` (whole tree), `~/Documents`, `~/Documents/Github`,
`~/Downloads`, `~/research`, to depth 4, for `*ENG*`, `Memoria_Tesis*`,
`Decomposition_Method*`, `Poster_MS*`. `THESIS_INPUT_DIR` is unset. The two
English-language thesis derivatives the brief describes as the "detailed
thesis-version scientific source" **do not exist on this machine**, so every
statement below about thesis content is read from the Spanish artifacts that do.

Two artifacts are present that the brief does not name: `PosterEVIC2023 (4).pdf`
and `Tesis_MGO.pdf` / `feature_selection.pdf`.

---

## 3. The inventory

Hashes are SHA-256 of the file as found. Where a PDF's embedded
`CreationDate` is evidence of nothing, that is said.

### 3.1 `Poster_MSWorkshop.pdf`
```text
sha256   c44163062793a7c86c0c886f4e56bf0985c13bcc60d63f23c2d1eca9fb028f2b
bytes    381843          pages 1
created  D:20231219000700Z        modified  D:20260907181228Z
creator  Matplotlib v3.4.3 / cairo
```
**Role: historical result/claim snapshot, and the only artifact whose embedded
date is its own.** Every other PDF here was re-exported in September 2026, so
this is the single file that carries a 2023 timestamp from 2023.

Title *Column Generation-Based Decomposition for Large-Scale Feature Selection
Problems*, Nicolás Acevedo and Fernando Ordóñez, Departamento de Ingeniería
Industrial, Universidad de Chile, Workshop in Management Science. Funded by
ANID–Subdirección de Capital Humano/Magíster Nacional 2022–Folio 22220861.

Claims recorded from it, verbatim in substance, as **claims to reassess**:
- the SOCP (2) and the unconstrained LASSO (1) are equivalent, with
  `λ = 2√(τκ)`;
- `SOCP` solved directly by MOSEK has near-constant time in the penalty;
- coordinate descent (`LASSO`) is fastest at **≤ 75 selected features**;
- the decomposition method is fastest from **≈ 75 features to 40 % of `m`**;
- *"Neither coordinate descent nor the decomposition method execution times
  appear to depend on the size of the instances, as they have a clear
  dependence on the number of selected features only."*

Its three references are the whole declared prior art: Bertsimas–King–Mazumder
2016, **Chicoisne 2023** (the column-generation scheme being applied), and
Küçükyavuz–Shojaie–Manzour–Wei–Wu 2020.

### 3.2 `Decomposition_Method_for_Feature_Selection ___.pdf`
```text
sha256   12a34bc5f204f10c10189ceababf948ce1a31dafd5f83f3b165422c2f2b24d9a
bytes    285286          pages 11
created  D:20260907181222Z  (re-export; says nothing about when it was written)
creator  LaTeX with hyperref / pdfTeX-1.40.25
```
**Role: private research notes. Not accepted proof.** Titled "Whiteboard", part
Spanish part English, with at least two hands in it (blue text added to a black
draft, and marginal comments in Spanish addressed to a collaborator —
`[Todo renaud]`).

It is nonetheless **the most mathematically serious artifact in the set**, and
it already contains derivations the April 2025 notes present as new. Recorded
in `docs/2026/TIMELINE.md` §W.

### 3.3 `Tesis_MGO.pdf` and `feature_selection.pdf`
```text
Tesis_MGO.pdf         sha256 2d84c81249fdd2d6c2ea79dfeebc6991232cc730d0141b34ab3b358a2a1f2e96
feature_selection.pdf sha256 aef0ab5e8efa3b57dfae329111d1109a8b2225118056115bfd95c2179907ecc7
both                  106099 bytes, 1 page, pdfTeX-1.40.23
```
**Two files, one document.** The SHA-256s differ, so a file-hash deduplication
keeps both — and it should, because they are two artifacts. Their *decompressed
content streams are byte-identical* (131 359 bytes each, verified), so they
differ only in PDF metadata and timestamps. Recorded as an alias pair rather
than merged.

**Role: the earliest derivation in the set, and the one that fixes the
lineage.** It starts from the ℓ0 big-M model, rewrites it as
`min ‖y − Xβ‖² + Σ βᵢ²/zᵢ`, and then as `min ‖y − Xβ‖² + κ̃ Σ tᵢ` subject to
`βᵢ² ≤ tᵢ zᵢ`. See `docs/2026/TIMELINE.md` §T0 for why that sequence matters.

### 3.4 `PosterEVIC2023 (4).pdf`
```text
sha256   e6124da0116a8e39c8b84d2ba948ae7f0218bf10c28551c6d4d4f912c099e719
bytes    238785          pages 1        pdfTeX-1.40.25 (re-export)
```
Role: a second poster, not named in the brief. Recorded for completeness.

### 3.5 `Memoria_Tesis__MGO_ (4).pdf`
```text
sha256   08bbaf776fd9832957ad51eeece6f62fb1af27d3b1be6671fb13c413ffbb1c0d
bytes    406605          pages 27       pdfTeX-1.40.24 (re-export)
```
Cover: *DECOMPOSITION METHOD FOR FEATURE SELECTION PROBLEM*, Nicolás Acevedo
Villena, profesor guía Fernando Iván Ordóñez Pizarro, **SANTIAGO DE CHILE
2026**. Abstract page reads `Resumen` / `Resumen`.

**Role: an incomplete thesis draft, in Spanish.** It is *not* the brief's
"2023 presentation".

### 3.6 `Memoria_Tesis__MGO___Final_ (5).pdf`
```text
sha256   d1e96dc5a6fdca6552ca215fc8070650ab7f1a6854ef12d99a24bc22cf68172d
bytes    646364          pages 87       pdfTeX-1.40.24 (re-export)
```
Cover: *CONIC DECOMPOSITION METHOD FOR FEATURE SELECTION PROBLEM*, same author
and advisor, **SANTIAGO DE CHILE 2026**. Abstract page is the heading `Resumen`
and nothing else; acknowledgements page reads `Agradecimientos`.

**Role: the fullest thesis-shaped artifact available, and a draft.** Its table
of contents is the real value: it names §4.2.5 "Sobre la convergencia de la
relajación lagrangiana", §4.2.6 "Soluciones artificiales para problemas
relajados divergentes", §5.3 "Revisión de criterios de convergencia en modelos
relajados", and §5.4 "Ajuste de parámetros del método".

### 3.7 `20250408 - Reviewing my past work (Finally).pdf`
```text
sha256   1acda37b5be15e46f04fa003c611e61c7dbf867d901ec38e90f4a35deafe3bda
bytes    140220          pages 3
created  D:20260917071308+00'00'   (Chrome print-to-PDF, today)
title    "20250408 - Reviewing my past work (Finally)"
```
**Role: 2025 review/reformulation notes — observations and hypotheses, not
established theory.** The 2025-04-08 date is self-reported in the title and
filename and is **not corroborated by the artifact**, which was printed today
from a web notebook.

### 3.8 `202504 - LASSO - Unbounded subproblem case.pdf`
```text
sha256   0d203789d111bfb24a223b666dec99236b9fd095c4cfbaec259d7783a99b2cda
bytes    273468          pages 5
created  D:20260917071337+00'00'   (Chrome print-to-PDF, today)
```
**Role: 2025 unbounded-pricing exploration — observations and hypotheses.** Same
date caveat as §3.7. It is the continuation §3.7 points to.

---

## 4. Disagreements, preserved rather than resolved

**D1 — publication year.** The official archival record for the thesis is
`https://repositorio.uchile.cl/handle/2250/198750`, and the poster is
independently dated 2023-12-19 by its own PDF metadata. Both thesis-shaped PDFs
here print **2026** on their covers. These are drafts that were re-compiled in
2026 with a `\year`-style macro, not evidence that the thesis is a 2026
document. The archival record is authoritative for publication metadata; the
2026 covers are recorded and not corrected.

**D2 — what the model penalises.** The poster's Problem Formulation says the
proposed SOCP "minimizes the sum of squared errors of a linear model, while
penalizing **the number of non-zero coefficients** βᵢ ≠ 0", and its own LASSO
Equivalence box states that for all τ, κ > 0 there is a λ = 2√(τκ) making the
SOCP and the **unconstrained LASSO** equivalent. Both cannot be true. `z` and
`u` are continuous (`z ∈ ℝᵐ₊`, `u ∈ ℝᵐ₊`) and there is no binary variable in the
formulation, so the equivalence box is the correct statement and the
cardinality sentence is not. This is not a quibble: it decides which literature
the work must be compared against, and it is examined in
`docs/2026/TIMELINE.md` §T0 and the mathematical audit.

**D3 — a closed form that two artifacts state differently.** The whiteboard
(§3.3) gives the Elastic-Net pricing solution as
`β* = −(1/θ)(2√(τκ) + ψᵀX)`, stated under the assumption "the equality holds in
the conic constraints at the optimal point". The April 2025 note gives
`βᵢ = (−(ψᵀX)ᵢ + sign[(ψᵀX)ᵢ]·λ₁)/(2λ₂)` with `βᵢ = 0` when `λ₁ ≥ |(ψᵀX)ᵢ|`.
They disagree. The audit resolves which is right; both are preserved here.

**D4 — a lower bound that may not be one.** The April 2025 notes describe the
unit-ball-constrained pricing value as a *lower bound* ("in the old version we
didn't even have a lower bound"). The whiteboard §2.3 describes the same device
differently — as a way to "provide an extreme ray of the pricing to the master".
Those are different claims with different validity, and the difference is the
subject of the unit-ball verdict.

---

## 5. Roles, in one table

| artifact | role | date evidence |
|---|---|---|
| `Poster_MSWorkshop.pdf` | historical claim snapshot | **self-dated 2023-12-19** |
| `Tesis_MGO.pdf` = `feature_selection.pdf` | earliest derivation, private | none (re-export) |
| `Decomposition_Method_for_Feature_Selection ___.pdf` | private notes, multi-author, unfinished | none (re-export) |
| `Memoria_Tesis__MGO_ (4).pdf` | thesis draft, 27 pp, Spanish | none (re-export) |
| `Memoria_Tesis__MGO___Final_ (5).pdf` | thesis draft, 87 pp, Spanish | none (re-export) |
| `PosterEVIC2023 (4).pdf` | second poster | none (re-export) |
| `20250408 - Reviewing my past work (Finally).pdf` | 2025 reformulation, hypotheses | filename only |
| `202504 - LASSO - Unbounded subproblem case.pdf` | 2025 pricing exploration, hypotheses | filename only |
| *(absent)* `Memoria_Tesis__MGO___Final___ENG_.pdf` | — | **not on this machine** |
| *(absent)* `Memoria_Tesis__MGO___Final___ENG_ (1).pdf` | — | **not on this machine** |
