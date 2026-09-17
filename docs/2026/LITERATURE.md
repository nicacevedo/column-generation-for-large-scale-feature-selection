# Literature ledger and novelty matrix

Retrieved through Research OS's literature subsystem (OpenAlex, Crossref,
arXiv) on 2026-09-17 and indexed locally; 1 124 works in the index after this
audit. Every entry below is identified by DOI or arXiv id, and every claim
about what a paper does is a claim about its abstract and stated contribution,
not about a search snippet.

**What this audit is for.** `docs/2026/TIMELINE.md` §T0 establishes that the
thesis's conic model is an exact reformulation of the LASSO. That decides the
comparison class, and the comparison class decides novelty. A method for the
LASSO is measured against LASSO solvers.

---

## 1. What the 2023 work cited

The poster's entire declared prior art, three items:

| id | work | relation |
|---|---|---|
| `doi:10.1214/15-aos1388` | Bertsimas, King, Mazumder, *Best subset selection via a modern optimization lens*, Ann. Statist. 44(2), 2016 | the ℓ0 / MIO line; the thesis's `MIQP` is this model |
| `doi:10.1007/s10589-022-00445-0` | **Chicoisne**, *Computational aspects of column generation for nonlinear and conic optimization: classical and linearized schemes*, COAP 84(3):789–831, 2023 | **the method being applied.** The thesis contributes the application, not the scheme |
| `arXiv:2005.14346` | Küçükyavuz, Shojaie, Manzour, Wei, Wu, *Consistent second-order conic integer programming for learning Bayesian networks* | the conic-IP modelling line |

Nothing in the poster, the thesis drafts, the whiteboard or the 2025 notes
cites a LASSO solver paper later than scikit-learn's implementation of
Friedman–Hastie–Tibshirani coordinate descent.

---

## 2. Convex sparse regression: the line the method is actually in

| id | work | year | what it establishes |
|---|---|---|---|
| `doi:10.1111/j.2517-6161.1996.tb02080.x` | Tibshirani, *Regression Shrinkage and Selection Via the Lasso*, JRSS-B | 1996 | the problem |
| `doi:10.1111/j.1467-9868.2005.00503.x` | Zou, Hastie, *Regularization and Variable Selection Via the Elastic Net*, JRSS-B | 2005 | **true Elastic Net**: ℓ1 + ℓ2². The thesis's TRACK EN target |
| `doi:10.1093/imanum/20.3.389` | Osborne, Presnell, Turlach, *A new approach to variable selection in least squares problems*, IMA J. Numer. Anal. 20(3) | 2000 | **the active-set / homotopy method for the LASSO.** Maintains a working set and adds the coordinate of largest dual-constraint violation |
| `doi:10.18637/jss.v033.i01` | Friedman, Hastie, Tibshirani, *Regularization Paths for Generalized Linear Models via Coordinate Descent*, JSS 33(1) | 2010 | glmnet; the algorithm behind `sklearn.linear_model.Lasso` |
| `doi:10.1111/j.1467-9868.2011.01004.x` | Tibshirani, Bien, Friedman, Hastie, Simon, Taylor, Tibshirani, *Strong Rules for Discarding Predictors in Lasso-Type Problems*, JRSS-B | 2011 | heuristic screening on `|Xᵀr|`; the same quantity as the thesis's pricing scan |
| `arXiv:1505.03410` | Fercoq, Gramfort, Salmon, *Mind the duality gap: safer rules for the Lasso*, ICML | 2015 | **safe** screening from the duality gap |
| `arXiv:1611.05780` | Ndiaye, Fercoq, Gramfort, Salmon, *Gap Safe screening rules for sparsity enforcing penalties*, JMLR | 2016 | the general Gap Safe framework |
| `arXiv:1802.07481` | **Massias, Gramfort, Salmon, *Celer: a Fast Solver for the Lasso with Dual Extrapolation*, ICML** | 2018 | **the state of the art this work must beat.** Working sets built from a dual point, with extrapolation |
| `doi:10.48550/arxiv.1807.08046` | Johnson, Guestrin, *A Fast, Principled Working Set Algorithm for Exploiting Piecewise Linear Structure in Convex Problems* (Blitz) | 2018 | working sets from a principled subproblem-selection rule |
| `doi:10.52202/068431-2823` | Bertrand, Klopfenstein, Bannier, Gidel, Massias, *Beyond L1: Faster and Better Sparse Models with skglm*, NeurIPS 35 | 2022 | the current general working-set + coordinate-descent system |
| `arXiv:2102.10846` | Dantas, Soubies, Févotte, *Expanding boundaries of Gap Safe screening*, JMLR 22(236) | 2021 | how far safe screening now reaches |
| `arXiv:2405.08631` | Yang, Hastie, *A Fast and Scalable Pathwise-Solver for Group Lasso and Elastic Net Penalized Regression via Block-Coordinate Descent* | 2024 | current glmnet-class engineering |

---

## 3. Best subset, ℓ0 and ℓ0 + ℓ2

Included because the thesis *started* here (§T0) and because `MIQP` exists.
Not the comparison class for the CG method, which never touches ℓ0.

| id | work | year |
|---|---|---|
| `doi:10.1214/15-aos1388` | Bertsimas, King, Mazumder, best subset via MIO | 2016 |
| `arXiv:1707.08692` | Hastie, Tibshirani, Tibshirani, *Extended Comparisons of Best Subset Selection, Forward Stepwise Selection, and the Lasso* | 2017 |
| `doi:10.1287/opre.2019.1919` | Hazimeh, Mazumder, *Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms*, Oper. Res. | 2020 |
| `arXiv:2202.04820` | Hazimeh, Mazumder, Nonet, *L0Learn: A Scalable Package for Sparse Learning using L0 Regularization* | 2022 |
| `doi:10.1287/ijoc.2020.1031` | Gómez, Prokopyev, *A Mixed-Integer Fractional Optimization Approach to Best Subset Selection*, IJOC | 2021 |
| `doi:10.1007/s11222-024-10387-8` | Moka, Liquet, Zhu, Muller, *COMBSS: best subset selection via continuous optimization*, Stat. Comput. | 2024 |
| `doi:10.32614/cran.package.combss` | Liquet, Mathur, Moka, `combss` | 2026 |
| `arXiv:2606.27638` | Huang, Muller, Tarr, *Robust Best Subset Selection via Fast Approximate MM-Estimation* | 2026 |

---

## 4. Convexification

| id | work | year | relation |
|---|---|---|---|
| `arXiv:1901.10334` | **Atamtürk, Gómez, *Rank-one Convexification for Sparse Regression*** | 2019 | **the whiteboard's §3.1 derivation is this, for one variable.** `conv{(β,z,u) : |β| ≤ Mz, β² ≤ u, z ∈ {0,1}} = {|β| ≤ Mz, β² ≤ uz}` is the standard perspective set |
| `arXiv:2504.16330` | Choi, Cepeda, Gómez, Küçükyavuz, *Rank-one convexification for quadratic optimization problems with step function penalties* | 2025 | the line is still active |
| `doi:10.1137/15m1012232` | Wu, Sun, Li, Zheng, *Tight MIQP Reformulations for Semi-Continuous Quadratic Programming: Lift-and-Convexification* | 2015 | the same convexification, earlier |

---

## 5. Decomposition

| id | work | year | relation |
|---|---|---|---|
| `doi:10.1007/s10589-022-00445-0` | Chicoisne, conic/nonlinear column generation | 2023 | the scheme the thesis applies |
| `arXiv:2602.22589` | Yamin, van Hoeve, Ralphs, *Dantzig-Wolfe and Arc-Flow Reformulations: A Systematic Comparison* | 2026 | current DW practice |

**Forward citation expansion on Chicoisne 2023 returned nothing relevant in
this index.** Recorded as a limitation, not as a finding: the providers reached
were OpenAlex (rate-limited part-way through this audit), Crossref and arXiv,
and none of them was asked a "cited by" query, because the literature
subsystem exposes search and fetch and not citation traversal. A full forward
expansion is remaining work and is listed as such.

---

## 6. The novelty matrix

Each row is a candidate contribution from the historical material. "Subsumed
by" names the work that already contains it.

| # | candidate | what it is, precisely | status | subsumed by |
|---|---|---|---|---|
| N1 | conic SOCP formulation of feature selection | with `z, u` continuous this is `2√(τκ)‖β‖₁`, an exact conic writing of the LASSO | **NOT NOVEL** | the LASSO is `doi:10.1111/...1996...`; writing `\|β\|` as a rotated cone is textbook conic modelling |
| N2 | column generation applied to it | Chicoisne's scheme, applied to a new model | **application, not method** | `doi:10.1007/s10589-022-00445-0` |
| N3 | pricing-boundedness criterion `‖Xᵀψ‖_∞ ≤ λ₁` | exactly full-LASSO dual feasibility (audit P2) | **NOT NOVEL** | it is the LASSO dual constraint; the whiteboard derives it itself |
| N4 | **top-`v` violation selection** | select the `v` largest `(\|Xᵀψ\|ᵢ − λ₁)₊`; with `ψ = −2r` that is `(\|Xᵢᵀr\| − λ₁/2)₊` | **NOT NOVEL — this is the primary gate and it does not pass** | Osborne–Presnell–Turlach `doi:10.1093/imanum/20.3.389` add exactly the maximum-violation coordinate; strong rules `doi:10.1111/...2011...` screen on it; Gap Safe `arXiv:1611.05780`, Celer `arXiv:1802.07481` and Blitz build working sets from it |
| N5 | the master as a conic solve over generated columns | with signed unit columns and free weights, the reachable set is the coordinate subspace, so this is "solve the restricted LASSO with an interior-point solver" | **not novel; and a design choice, not an advantage** | any working-set method; the restricted solve is normally coordinate descent, which is what makes it fast |
| N6 | artificial / `v`-solution columns for an unbounded pricing | adding an extreme ray when the pricing is unbounded | **standard Dantzig–Wolfe**, and the whiteboard says so in those words | any DW treatment; the specific choice of signed unit rays is what makes N4 true |
| N7 | unit-ball pricing as a lower bound | restricting `‖β‖₂ ≤ 1` and reading the value as a bound | **INVALID** (audit P6) | not applicable — it is not a correct construction |
| N8 | Elastic-Net pricing closed form | `βᵢ* = −sign(aᵢ)(\|aᵢ\| − λ₁)₊/(2λ₂)` | **correct, and it is the soft-thresholding operator** | the proximal operator of the ℓ1 norm; in every coordinate-descent LASSO solver since 2010 |
| N9 | `λ₂ → 0` continuation as stabilisation | not implemented, not tested | **untested idea**, and generic regularisation-based stabilisation of column generation is long established | the CG stabilisation literature |
| N10 | perspective / rank-one convexification of ℓ0 + ℓ2 | whiteboard §3.1 | **NOT NOVEL** | Atamtürk–Gómez `arXiv:1901.10334`, six years earlier |

### What survives

Nothing in the list is a novel *method*. Two things are nonetheless worth
stating, and they are the only candidates this audit leaves open:

**C1 — a negative computational result, stated precisely.** "A conic
Dantzig–Wolfe decomposition of the LASSO reduces, under free master weights
and unit-ray columns, to a maximum-violation working-set method whose
restricted solve is an interior-point solve, and is therefore dominated by
working-set methods whose restricted solve is coordinate descent." That is a
claim about a family, it is checkable, and `EXP-0001` is designed to settle its
empirical half. Negative results about a natural-looking approach are
publishable in the right venue and are worth very little in the wrong one.

**C2 — the equivalence, written down.** The whiteboard's derivation that the
thesis's conic model is the LASSO and that its pricing-boundedness criterion is
the LASSO dual feasibility constraint is correct, is not in the thesis, and is
the fact that makes C1 true. It is not novel mathematics; it is the missing
sentence that reorganises the whole programme.

Both are subject to the mathematical audit's independent review and to
`EXP-0001`'s result. Neither is asserted here.

---

## 7. What this audit did not do

- **No forward citation expansion.** §5.
- **No full-text reading.** The ranking is over metadata and, where arXiv
  supplied it, abstracts. Every "subsumed by" above rests on a stated
  contribution, which is the right granularity for a novelty gate and the wrong
  one for a related-work section.
- **OpenAlex rate-limited part-way through**, after 1 000 credits. Four of the
  twelve topic retrievals completed on arXiv and Crossref only, and the tool
  reported each as `complete: no` rather than silently returning fewer results.
