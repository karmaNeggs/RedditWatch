# V3 Statistical Methodology Protocols
### Monthly inauthentic-activity scoring for 45 Indian subreddits

Status: research reference. Every section is written as **PROTOCOL → RULE → METHOD/FORMULA → REASON → CITATION**, so it can be lifted into code and into the methods appendix of the report.

---

## 0. The estimand comes first

**PROTOCOL 0.1 — Write down the estimand before writing any model code.**

Three different quantities are being conflated in V2. They need different machinery:

| # | Estimand | Statistical object | Right tool |
|---|---|---|---|
| E1 | "Is *this account* inauthentic?" | `P(y=1 \| x)` per account | PU classifier + calibration |
| E2 | "What *fraction* of r/india activity in 2025-06 was inauthentic?" | mixture proportion `α_{s,t}` | **quantification / MPE**, not classification |
| E3 | "Is sub A worse than sub B, and is the trend rising?" | contrast + CI | effect sizes, blocked inference |

The deliverable stated by the team ("a PERCENTAGE of inauthentic activity") is **E2**, and E2 is *not* obtained by thresholding E1 and counting. Classify-and-count is a biased estimator of prevalence (§6). This single reframing is the highest-leverage change in V3.

**REASON.** Under PU labels, `P(y=1|x)` is not even identifiable without a class-prior assumption, whereas the mixture proportion is the thing PU theory is actually built to estimate. Framing the deliverable as MPE makes the whole pipeline defensible.

**CITATION.** Bekker & Davis, *Learning from positive and unlabeled data: a survey*, Mach. Learn. 109 (2020) — https://link.springer.com/article/10.1007/s10994-020-05877-5 · Garg et al., *Mixture Proportion Estimation and PU Learning: A Modern Approach*, NeurIPS 2021 — https://arxiv.org/abs/2111.00980

---

**PROTOCOL 0.2 — Census does not mean "no uncertainty", and it does not mean "everything is significant".**

If the 45×12 panel is the population, then descriptive quantities have **zero sampling error** and p-values are meaningless. If the target is the *generating process* (i.e. "would this hold next month / in a comparable sub"), then uncertainty comes from the process, not from sampling, and must be estimated by **block bootstrap over accounts and over months**, not by a t-test on millions of rows.

Decide once, state it in the methods section, and never mix the two framings.

---

# 1. PROTOCOL FAMILY A — Labels and Positive-Unlabeled learning

## A0. The label diagnosis that must happen before anything else

**PROTOCOL A0.1 — Estimate `(β − α)` first. It is a hard ceiling on every AUC you will ever report.**

Notation (Jain, White & Radivojac, AAAI 2017):
- `α` = proportion of **true positives hiding inside the unlabeled set** (the undetected bad accounts),
- `β` = proportion of **true positives inside your labeled-positive set** (label purity; `β = 1` means every suspended account really is inauthentic).

Train a classifier to separate *labeled positives* from *unlabeled*. Call the AUC you measure `AUC_pu`. Then:

```
AUC_true = [ AUC_pu − (1 − (β − α)) / 2 ] / (β − α)
```

Inverting, the **maximum attainable `AUC_pu`** — achieved by a hypothetically perfect classifier — is:

```
AUC_pu^max = (1 + β − α) / 2
```

Sanity check (β=1): a perfect ranker scores `1 − α/2`, and a coin flip scores 0.5, exactly as the formula gives.

**Ceiling table (β = 1, i.e. perfectly pure labels):**

| α (undetected positives in unlabeled) | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| max observable `AUC_pu` | 0.990 | 0.975 | 0.950 | 0.900 | 0.850 | 0.750 |

**Ceiling table when labels are impure (α = 0.10):**

| β (label purity) | 1.00 | 0.80 | 0.60 | 0.50 | 0.40 |
|---|---|---|---|---|---|
| max observable `AUC_pu` | 0.950 | 0.850 | 0.750 | 0.700 | 0.650 |

**This is the most important number in the V3 project.** V2 plateaued at 0.663. Solving `(1 + β − α)/2 = 0.663` gives `β − α = 0.326`. In other words: **if the effective label quality is around `β − α ≈ 0.33`, then 0.663 is what a *perfect* classifier scores, and the plateau is not a modelling failure at all — it is the label ceiling.** Before spending another sprint on features, the team must bound `β − α`. If it is ≲ 0.8, the stated target of AUC > 0.90 is *arithmetically unreachable* against these labels and the target must be renegotiated (or the reporting metric changed to a quantification error, §6).

**CITATION.** Jain, White & Radivojac, *Recovering True Classifier Performance in Positive-Unlabeled Learning*, AAAI 2017 — https://arxiv.org/abs/1702.00518 (formulas verified algebraically above).

---

**PROTOCOL A0.2 — Treat the five label channels as five separate PU problems until proven otherwise.**

The measurement that "of 34 admin-removed post authors, zero were suspended" is not a curiosity — it is evidence that admin-removal and suspension are **near-disjoint constructs**, or that one mechanically censors the other (a suspended account's posts may be re-attributed to `[deleted]`, so its removals can never be observed → a *competing-risks / censoring* structure, not independence).

**RULE.** For each channel `c ∈ {admin_removal, mod_removal, automod_filter, account_suspension, self_deletion}`:
1. Fit a separate PU model `M_c`.
2. Estimate a separate prior `α_c`.
3. Run the **cross-channel transfer test**: score channel `c'`'s positives with `M_c`. Report the 5×5 matrix of transfer AUCs.

**Decision rule:**
- If off-diagonal transfer AUC ≈ 0.5 → the channels do **not** share a latent "inauthenticity" factor. A single composite score is **not defensible**; publish per-channel scores, or publish only the one channel with a clear construct.
- If off-diagonal transfer AUC ≥ ~0.7 → a shared factor is plausible; you may pool, and you should report the pooled model *and* the transfer matrix.

**REASON.** Self-deletion is overwhelmingly a privacy behaviour of ordinary users; automod filtering is largely keyword/karma-threshold mechanics; mod removal is subreddit-policy-specific and varies wildly by mod team. Pooling them into one "bad" label manufactures a low-`β` label set, which (by A0.1) caps AUC. **This is the single most likely cause of the 0.663 plateau.**

**CITATION.** Bekker & Davis survey §on label mechanisms — https://arxiv.org/abs/1811.04820 · On suspension-as-ground-truth being unreliable: Rauchfleisch & Kaiser, *The False positive problem of automatic bot detection in social science research*, PLOS ONE 2020 — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045

---

**PROTOCOL A0.3 — Record label *timing* and enforce a fixed observation window.**

Suspension status is observed **at collection time**, not at post time. Without a window, an account active in 2025-01 and suspended in 2026-07 is labelled positive for 18 months of history — which is (a) a future-information leak, and (b) a *maturity* confound: recent months have less time to accrue labels, so measured prevalence falls artificially toward the present.

**RULE.** Define a fixed look-forward horizon `h` (e.g. 90 days). Row `(account, month t)` is labelled positive iff the moderation event occurs in `(t, t + h]`. Months with `t + h > collection_date` are **censored** — either drop them or model them explicitly with a maturity offset. Publish the label-maturity curve (cumulative label rate vs months-since-activity) as a figure; it is the honest way to show the censoring.

---

## A1. Which labeling assumption applies

| Assumption | Definition | Applies here? |
|---|---|---|
| **SCAR** (Selected Completely At Random) | `P(s=1 | y=1, x) = c`, constant | **No.** Reddit admin action is strongly non-random in `x` (volume, reports, subreddit, language). |
| **SAR** (Selected At Random) | `P(s=1 | y=1, x) = e(x)`, a propensity function of observed covariates | **Yes, this is your setting.** |
| **PG** (Probabilistic Gap) | labeling propensity increases with `P(y=1|x)` | Plausible as a weaker fallback. |

**PROTOCOL A1.1 — Do not assume SCAR silently. Either (a) restrict to a stratum where SCAR is credible, or (b) model the propensity `e(x)` explicitly, or (c) run SCAR as the primary analysis and report a SAR sensitivity analysis.**

The cheapest defensible option (b'): **stratified SCAR**. Assume SCAR *within* strata of {subreddit × month × activity decile} and estimate `c_k` per stratum. This absorbs the dominant non-randomness (big subs get more admin attention) at low cost.

**CITATION.** Bekker & Davis, *Learning from Positive and Unlabeled Data under the Selected At Random Assumption*, 2018 — https://arxiv.org/abs/1808.08755 · PULSNAR (non-SCAR class-proportion estimation) — https://peerj.com/articles/cs-2451/

---

## A2. Class-prior / mixture-proportion estimation

**PROTOCOL A2.1 — Estimate `π = P(y=1)` (equivalently the label frequency `c`) with ≥3 independent methods and report the spread as the headline uncertainty.**

Methods, ranked for this project:

**(1) TIcE — Tree Induction for c Estimation. USE THIS AS PRIMARY.**
Finds subsets of feature space (via decision-tree induction) where the labeled-positive density is maximal, and uses a lower bound on the label frequency within the best subset:
```
ĉ = max over induced subsets T of  [ (#labeled positives in T) / (#instances in T) ]  , with a
    pessimistic (1−δ) confidence correction subtracted from the ratio.
```
*Why primary:* best average performance and lowest variance across settings, and it is orders of magnitude faster than KM1/KM2 — which matters at millions of rows.
Citation: Bekker & Davis, AAAI 2018 — https://ojs.aaai.org/index.php/AAAI/article/view/11715

**(2) BBE (Best Bin Estimation) — USE AS SECOND, it is the most scalable.**
Score all points with any PU classifier, bin the scores, and pick the bin threshold `z` maximising the estimated ratio of labeled-positive mass to unlabeled mass above `z`, with a finite-sample penalty:
```
α̂ = min over z of  [ q̂_P(z) ] / [ q̂_U(z) ]   where q̂(z) = fraction of that sample scoring ≥ z,
    with a uniform-convergence correction term ~ sqrt(log(4/δ)/(2n)) added/subtracted.
```
Scales to your data because it only needs classifier scores, not an RKHS quadratic program.
Citation: Garg et al., NeurIPS 2021 — https://arxiv.org/abs/2111.00980

**(3) DEDPUL** — difference of estimated densities; strong on low-dimensional score distributions. https://github.com/dimonenka/DEDPUL

**(4) KM1 / KM2** — kernel-embedding MPE, theoretically clean but **does not scale**; run only on a stratified subsample (e.g. 50k rows) as a cross-check.
Citation: Ramaswamy, Scott & Tewari, ICML 2016 — https://arxiv.org/abs/1603.02501

**(5) Elkan–Noto `ĉ`** — cheapest sanity check: train `g(x) ≈ P(s=1|x)` on P vs U, then
```
ĉ = (1/|V|) Σ_{x ∈ V, s(x)=1} g(x)          (V = held-out validation set)
```
Citation: Elkan & Noto, KDD 2008 — https://cseweb.ucsd.edu/~elkan/posonly.pdf

**PROTOCOL A2.2 — Report `π` as an interval, never a point.** The identifiability of `π` requires an **irreducibility / anchor-set** condition (there must exist a region of feature space that is purely negative). That condition is untestable. Without it, only an *upper bound* on `π` is identified. So: report `[min, max]` across methods × bootstrap, and propagate that interval through every downstream number (§6.4).

---

## A3. Training objective

**PROTOCOL A3.1 — Default: cost-sensitive reweighted XGBoost (Elkan–Noto weighting). Escalate to nnPU only if the reweighted model underperforms.**

**Elkan–Noto weighting (simple, works with any weighted learner including XGBoost):**
Under SCAR, `P(y=1|x) = P(s=1|x) / c`. Each unlabeled example is duplicated into a positive copy and a negative copy with weights:
```
w_pos(x) = ( (1 − c) / c ) · ( g(x) / (1 − g(x)) )
w_neg(x) = 1 − w_pos(x)
```
where `g(x) = P̂(s=1|x)` from the non-traditional classifier. Labeled positives keep weight 1 on the positive class.

**nnPU (Kiryo et al., NeurIPS 2017)** — use when the model is flexible enough that the unbiased estimator goes negative (it will, with deep XGBoost):

Unbiased PU risk (uPU, du Plessis et al.):
```
R̂_pu(g) = π_p · R̂_p⁺(g) − π_p · R̂_p⁻(g) + R̂_u⁻(g)

R̂_p⁺(g) = (1/n_p) Σ_{i=1..n_p} ℓ( g(x_i^p), +1 )
R̂_p⁻(g) = (1/n_p) Σ_{i=1..n_p} ℓ( g(x_i^p), −1 )
R̂_u⁻(g) = (1/n_u) Σ_{i=1..n_u} ℓ( g(x_i^u), −1 )
```

Non-negative PU risk (nnPU):
```
R̃_pu(g) = π_p · R̂_p⁺(g) + max{ 0 ,  R̂_u⁻(g) − π_p · R̂_p⁻(g) }
```

Large-scale stochastic algorithm (their Algorithm 1), with hyperparameters `0 ≤ β ≤ π_p·sup ℓ` and `0 ≤ γ ≤ 1`. For minibatch `i`, let
```
r_i = R̂_u⁻(g; X_u^i) − π_p · R̂_p⁻(g; X_p^i)
```
- if `r_i ≥ −β`: step along `∇_θ R̂_pu(g; X_p^i, X_u^i)` with learning rate `η`;
- else: step along `∇_θ ( π_p·R̂_p⁻(g; X_p^i) − R̂_u⁻(g; X_u^i) )` with **discounted** rate `γη` — i.e. deliberately go *backwards* to undo the overfitting that drove the risk negative.

Citation: https://arxiv.org/abs/1703.00593 · reference code https://github.com/kiryor/nnPUlearning

**Bagging-PU (Mordelet & Vert, Pattern Recognition Letters 2014)** — a cheap, very effective ensemble that pairs naturally with XGBoost: repeatedly train `P` vs a *random subsample* of `U`, and average. Use the **out-of-bag** score for each unlabeled point (each point is scored only by the models that did not see it), which gives a nearly free, leakage-clean unlabeled score.
Citation: https://www.sciencedirect.com/science/article/abs/pii/S0167865513002432 · impl. https://github.com/phuijse/bagging_pu

**Two-step / spy method (Liu et al. 2002; S-EM)** — use as a *diagnostic*, not the main model:
1. Plant a random 10–15% of labeled positives ("spies") into `U`.
2. Train P-vs-U; find the score threshold `t` below which e.g. 95% of spies fall.
3. Unlabeled points scoring below `t` are **reliable negatives (RN)**.
4. Train P vs RN.
The spy threshold gives an interpretable, direct read on how deeply the positives are buried in `U` — which is a `α` diagnostic in its own right.
Citation: Liu et al. — https://www.cs.uic.edu/~liub/publications/ijcai03-textClass.pdf

**Which to use when `π` is small and unknown:** small `π` makes uPU/nnPU unstable (the `π_p` multiplier shrinks the positive term into noise) and makes MPE hard. Prefer, in order: **Bagging-PU + OOB scores** (robust, no `π` needed for ranking) → **Elkan–Noto weighting with a stratified `ĉ`** → **nnPU** only with `π` fixed from A2 and a `β`-sweep sensitivity.

---

## A4. Evaluating a PU model — what to report instead of naive AUC

**PROTOCOL A4.1 — Never report a raw PU-AUC as if it were an AUC.** Label it `AUC_pu` everywhere, and always accompany it with the corrected value and its α-sensitivity band (A0.1).

**PROTOCOL A4.2 — What is and is not estimable from PU data:**

| Quantity | Estimable from PU alone? | How |
|---|---|---|
| **Model *ranking*** (which of two models is better) | ✅ Yes | `AUC_pu` is a monotone increasing affine map of `AUC_true` with slope `(β−α) > 0`, so it preserves ordering. **Use `AUC_pu` for model selection — that is valid.** |
| **Recall / TPR** | ✅ Yes (under SCAR) | fraction of held-out labeled positives scoring above threshold |
| **Precision / PPV** | ❌ No | requires `α`; false positives are unobservable |
| **`AUC_true`** | ⚠️ Only given `(α, β)` | Jain correction, A0.1 |
| **Prevalence `α`** | ⚠️ Only under irreducibility | §A2 |
| **Calibrated `P(y=1|x)`** | ⚠️ Only given `c` | `P(y=1|x) = P(s=1|x)/c` |

**PROTOCOL A4.3 — Primary PU model-selection criterion: the Lee–Liu statistic.**
```
PU-F1  ∝  r̂² / P̂r[ f(X) = 1 ]
```
where `r̂` = recall estimated on held-out labeled positives, and `P̂r[f(X)=1]` = fraction of *all* instances predicted positive. This is a monotone proxy for F1 that needs no negatives, and — crucially — its denominator penalises the degenerate "call everything positive" solution that plain recall rewards.
Citation: Lee & Liu, ICML 2003 — https://dblp.org/rec/conf/icml/LeeL03.html

**PROTOCOL A4.4 — Report the following block for every model, every time:**
```
AUC_pu                     0.XXX  [bootstrap CI]
AUC_true | α=0.05, β=1.0   0.XXX      ← sensitivity grid, not a point estimate
AUC_true | α=0.15, β=0.8   0.XXX
AUC_true | α=0.30, β=0.6   0.XXX
Recall @ top-1%            0.XXX
Recall @ top-5%            0.XXX
Lift @ top-1%              X.Xx   ( = recall@1% / 0.01 )
PU-F1 (Lee–Liu)            0.XXX
Estimated π (range over TIcE / BBE / DEDPUL / EN)   [0.0XX, 0.0XX]
```

**PROTOCOL A4.5 — Add a permutation null.** Shuffle the labels within {subreddit × month} strata, refit end-to-end, and record `AUC_pu`. Report the observed value against the null distribution. This is the only defence against "we got 0.72 but the whole pipeline would give 0.68 on noise."
Citation: *Leveraging permutation testing to assess confidence in PU learning*, BMC Bioinformatics 2024 — https://link.springer.com/article/10.1186/s12859-024-05834-2

Broader critique of sloppy PU evaluation across 51 papers: https://arxiv.org/abs/2206.02423

---

# 2. PROTOCOL FAMILY B — Metric hygiene at scale

## B1. The volume confound is the default failure mode

**PROTOCOL B1.1 — Every metric must be volume-invariant *by construction*, then *verified empirically*.**

**Verification test (run for all ~60 metrics, produce a table):**
```python
rho, _ = spearmanr(metric_values, np.log1p(n_underlying))
r2_log = ols(metric ~ log1p(n)).rsquared
```
**Gate:** `|rho| > 0.20` or `r2_log > 0.05` ⇒ the metric is a size proxy. It must be corrected (B1.2) or explicitly relabelled as a volume metric. Metrics that fail this test and are left in are the reason a "bot score" turns out to be a "big subreddit detector".

**PROTOCOL B1.2 — The universal fix: null-model standardisation.**
For any statistic `T` computed on `n` items, compute its distribution under a **volume-matched null** (resample `n` items from the pooled/marginal distribution, `B ≥ 200` times), then report
```
z_T = ( T_obs − E_null[T] ) / SD_null[T]
```
This works uniformly for Gini, HHI, entropy, clustering coefficient, assortativity, burstiness — anything whose small-sample expectation depends on `n`. It is more robust than hunting for a closed-form bias correction per metric, and it is trivially parallelisable.

**PROTOCOL B1.3 — Never compare raw counts across subreddits.** Permitted comparisons only:
1. rate per unit exposure, with empirical-Bayes shrinkage (B5);
2. within-subreddit z-score or rank (removes level, keeps shape);
3. model-based with an explicit exposure offset: `log E[count] = Xβ + log(exposure)`;
4. subreddit fixed effects in the model.

---

## B2. Concentration metrics — the small-sample bias, and the corrections

### Entropy

Plug-in (MLE) Shannon entropy is **biased downward**, and the bias grows as `n` shrinks relative to the number of categories `K`. Small subreddits therefore look artificially *more concentrated*. This is a pure artefact.

**Miller–Madow correction** (first-order, cheap, use as the default):
```
Ĥ_MM = Ĥ_plugin + ( K̂ − 1 ) / ( 2n )
```
where `K̂` = number of categories with non-zero observed frequency, `n` = sample size.

**Chao–Shen** (Horvitz–Thompson + Good–Turing coverage; **use when `n` is small relative to `K`**, which is your long-tail-author case):
```
Ĉ = 1 − f₁ / n                    (f₁ = number of singleton categories)
p̃_k = Ĉ · p̂_k                     (coverage-adjusted probabilities)

Ĥ_CS = − Σ_k  [ p̃_k · log p̃_k ] / [ 1 − (1 − p̃_k)^n ]
```
The denominator is the Horvitz–Thompson inclusion-probability weight (probability that category `k` is seen at least once in `n` draws).

*Empirical guidance:* in the heavily undersampled regime (`n < K`) Chao–Shen substantially outperforms Miller–Madow, shrinkage, and plug-in; all methods converge as `n` grows.

**PROTOCOL B2.1 — Report entropy as a Hill number, not as raw or "normalised" entropy.**
`H / log K` is **not** size-invariant (both `H` and `K̂` move with `n`). Use the **effective number of categories**, `D₁ = exp(H)`, which has the right units (a doubling means twice as many equally-active authors) and behaves linearly.

**PROTOCOL B2.2 — For cross-subreddit comparison, standardise by *coverage*, not by *sample size*.**
Classical rarefaction (subsample everything to the smallest `n`) systematically under-samples the more diverse assemblage and compresses real differences. Coverage-based rarefaction/extrapolation (Chao & Jost) standardises to equal sample **completeness** `Ĉ` and ranks assemblages more faithfully.

**CITATIONS.** Chao & Shen, *Env. Ecol. Stat.* 10:429–443 (2003); Miller (1955)/Madow; Good, *Biometrika* 40:237–264 (1953); Horvitz & Thompson, *JASA* 47:663–685 (1952) — all documented in the R `entropy` package: https://cran.r-project.org/web/packages/entropy/entropy.pdf · Chao & Jost, *Coverage-based rarefaction and extrapolation*, Ecology 2012 — https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-1952.1 · Chao et al., Ecol. Monogr. 2014 — https://esajournals.onlinelibrary.wiley.com/doi/10.1890/13-0133.1 · `iNEXT` — https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.12613

### Herfindahl / Simpson

Raw HHI has an **explicit, analytically removable `1/n` term**. Under a multinomial with `n` draws over `K` equiprobable categories, `E[HHI] = 1/K + (1 − 1/K)/n`. So raw HHI *mechanically* rises as `n` falls.

**PROTOCOL B2.3 — Always use the unbiased Simpson estimator, never the raw sum of squared shares.**
```
λ̂_unbiased = Σ_k  x_k (x_k − 1) / ( n (n − 1) )
```
(with `x_k` = raw count in category `k`, `n = Σ x_k`). Then `HHI* = (λ̂ − 1/K) / (1 − 1/K)` if a `[0,1]` normalisation is wanted, and `D₂ = 1/λ̂` (Hill number of order 2) for the effective-number scale. **Enforce a minimum `n` (e.g. `n ≥ 30`); below that, report NA rather than a number.**

### Gini

The sample Gini is biased **downward** by `O(1/n)`.

**PROTOCOL B2.4 — Apply the Deltas small-sample correction and a jackknife/bootstrap CI.**
```
G_corrected = ( n / (n − 1) ) · G_sample
```
This is exactly unbiased for exponential/gamma populations and materially reduces bias generally; where the distributional assumption is uncomfortable, use the jackknife or bootstrap bias correction instead. **Always attach an `n` to every Gini you publish.**
Citation: Deltas, *The Small-Sample Bias of the Gini Coefficient*, Rev. Econ. Stat. 85(1):226–234 (2003) — https://direct.mit.edu/rest/article/85/1/226/57394 · bias/CI practice: https://cran.r-project.org/web/packages/giniVarCI/vignettes/GiniVarInterval.html

---

## B3. Transforms for severely right-skewed counts

**PROTOCOL B3.1 — For XGBoost, do not transform for skewness. It is a no-op.**
Tree splits depend only on the *ordering* of feature values, so any strictly monotone transform (log1p, Box-Cox, Yeo-Johnson, rank) leaves the fitted tree **identical**. Skewness of 27 or 41 is not a problem for the GBM. Time spent Box-Cox-ing features for XGBoost is wasted.

**PROTOCOL B3.2 — Transform where it actually matters:**

| Use | Transform | Note |
|---|---|---|
| z-score composites, PCA/FA, VIF, correlation pruning | **log1p** or **Yeo-Johnson** | skew of 27 makes a z-score meaningless — one outlier dominates the mean and SD |
| Linear/logistic baselines | log1p / Yeo-Johnson | |
| Distance/kernel methods, k-NN, clustering | **quantile (rank→normal)** | most aggressive; destroys magnitude information but is fully outlier-proof |
| Plots and tables | log1p with labelled original units | |
| Signed or zero-crossing quantities | **asinh**: `asinh(x) = log(x + √(x²+1))` | Box-Cox needs `x>0`; log1p needs `x ≥ 0`; Yeo-Johnson and asinh handle negatives |

**Selection rule:** log1p first (interpretable, one fewer fitted parameter, no leakage risk). Escalate to Yeo-Johnson only if residual skew `|γ₁| > 2` matters for the downstream method. Escalate to quantile transform only when outliers still dominate. **A fitted Box-Cox/Yeo-Johnson `λ` or a quantile mapping is a fitted parameter → it must be fitted inside the CV fold (see D4).**

---

## B4. Winsorisation policy

**PROTOCOL B4.1**
1. Winsorise at **fixed, pre-registered quantiles** (default 0.1% / 99.9%) — not at "3 SD", which is itself corrupted by the outliers on a skew-27 variable.
2. Compute the caps on the **training fold only**, then apply to validation/test. Full-data caps are textbook preprocessing leakage (L1.2).
3. Emit a companion binary feature `was_clipped_<metric>`. Extreme values in bot detection are *signal*, not noise; clipping without an indicator throws away the strongest evidence you have.
4. Log the clip rate per metric per fold. If > 1%, the quantile is wrong or the metric is broken.
5. **Never winsorise the label or the exposure/denominator.**

---

## B5. Rates, zero-inflation, and small denominators

**PROTOCOL B5.1 — Every rate gets empirical-Bayes shrinkage.**
A raw rate `x/n` with `n = 3` is noise; ranked against `n = 30,000`, it will always win or always lose. Fit a Beta prior by method of moments (or MLE) on the pooled `(x_i, n_i)` and report:
```
p̂_EB = ( x + α₀ ) / ( n + α₀ + β₀ )
```
Fit `(α₀, β₀)` **on the training fold only**. Emit both `p_raw` and `p_EB`; use `p_EB` in the model and in all rankings; use `p_raw` only in diagnostics. Report the shrinkage weight `n/(n+α₀+β₀)` as a data-quality column.
Citation: standard beta-binomial empirical Bayes; readable derivation — http://varianceexplained.org/r/empirical_bayes_baseball/

**PROTOCOL B5.2 — Minimum-denominator rule.** Pre-register a minimum `n` per metric family (suggest `n ≥ 30` for concentration metrics, `n ≥ 10` for rates with EB shrinkage). Below it, emit `NA` — do **not** emit 0. XGBoost handles `NaN` natively via default directions; imputing 0 tells the model a lie.

**PROTOCOL B5.3 — Hurdle-decompose zero-inflated features.**
If `P(x = 0) > 0.30`, split into two features:
```
any_x    = 1{ x > 0 }                        (the occurrence process)
mag_x    = log1p(x)  if x > 0 else NaN       (the intensity process | occurrence)
```
The two processes usually have *different* relationships with the label (e.g. "ever posts a link" vs "how many links given that it posts links"). Forcing them through a single `log1p` blends them and weakens both.

---

## B6. METRIC-SANITISATION CHECKLIST (codeable)

Produce one row per metric in a `metric_registry.csv` and assert on it in CI:

```
[ ] 1.  name, one-line definition, unit, numerator field(s), denominator/exposure field
[ ] 2.  PROVENANCE: list every raw field used.  Flag = True if ANY field encodes a
        moderation action (removed_by_category, banned_by, body=='[removed]',
        body=='[deleted]', author=='[deleted]', mod_reason, approved_by, score_hidden,
        distinguished, locked, is_robot_indexable).  Provenance flag MUST NOT
        intersect the provenance of the active label (see D1).
[ ] 3.  TIME VALIDITY: every field is observable at or before month t.  No
        as-of-collection account attributes (current karma, current age, still_exists).
[ ] 4.  n_effective: min / p05 / median / max of the underlying sample size.
[ ] 5.  MIN-N RULE applied; NA (not 0) emitted below threshold.
[ ] 6.  VOLUME TEST: spearman(metric, log1p(n)) reported.  |rho| < 0.20 required,
        else corrected or relabelled as a volume metric.
[ ] 7.  NULL-MODEL z: E_null and SD_null computed by volume-matched resampling
        (B >= 200) for every concentration / network statistic.
[ ] 8.  SMALL-SAMPLE CORRECTION applied where relevant:
        entropy -> Miller-Madow or Chao-Shen, reported as exp(H);
        HHI/Simpson -> sum x(x-1)/(n(n-1));
        Gini -> n/(n-1) * G  + bootstrap CI.
[ ] 9.  ZERO FRACTION; if > 0.30 -> hurdle split into any_x / mag_x.
[ ] 10. SKEWNESS and KURTOSIS before/after transform (only required for features
        feeding composites, PCA, VIF, or linear baselines).
[ ] 11. MISSINGNESS: rate + mechanism note + explicit NA (no silent zero-fill).
[ ] 12. FITTED PARAMETERS listed (winsor caps, YJ lambda, quantile map, EB prior,
        target-encoding means) and marked "fit inside fold".
[ ] 13. TEMPORAL STABILITY: metric mean/SD by month plotted; any step change flagged
        as a possible collection/API artefact, not a finding.
[ ] 14. CORRELATION-CLUSTER ID (from the 1 - |rho| hierarchical clustering, C2).
[ ] 15. DIRECTION: does higher = more suspicious?  Recorded, and consistent with
        the sign of its SHAP contribution (a sign flip is a red flag for confounding).
```

---

# 3. PROTOCOL FAMILY C — Composite metrics

**PROTOCOL C1.1 — Default position: do NOT hand-build composites for the model. Feed the components to XGBoost.**

A gradient-boosted tree ensemble learns arbitrary monotone transforms and interactions of its inputs. A hand-built composite `z̄ = mean(z₁..z_k)`:
- imposes a **linear, equal-weight, fully compensatory** aggregation the data never justified (a high score on one indicator silently offsets a low score on another);
- **discards** the interaction structure the model would otherwise discover;
- is **dominated by whichever family has the most members** (see C3).

Build composites **only** for the human-facing report, where a single interpretable number is the deliverable — and then treat that composite as a *communication artefact*, evaluated separately, never as a model input.

---

**PROTOCOL C1.2 — Prune within families before anything else.**
```python
# 1. Spearman (not Pearson: features are skew-27)
corr = spearmanr(X).correlation
d    = 1 - np.abs(corr)
Z    = hierarchy.ward(squareform(d, checks=False))
clusters = hierarchy.fcluster(Z, t=0.30, criterion='distance')   # t=0.30  <=> |rho| > 0.70
```
Keep **one representative per cluster** (highest univariate PU-signal, or lowest missingness, or most interpretable), or replace the cluster by its first principal component. Record the cluster map in the metric registry (checklist item 14).

**Why not just VIF?** VIF (`VIF_j = 1/(1 − R²_j)`, flag at 5, act at 10) is a *linear*-model diagnostic. It is the right tool for the logistic baseline and for any z-score composite. For the tree model, collinearity does not hurt *prediction* — it wrecks *interpretation* (§8), which is why the clustering is still mandatory.
Citation: threshold conventions — https://towardsdatascience.com/when-predictors-collide-mastering-vif-in-multicollinear-regression/ · AutoSpearman (automated Spearman+VIF pruning) — https://arxiv.org/abs/1806.09791

---

**PROTOCOL C1.3 — PCA vs factor analysis vs z-score averaging vs learned weights.**

| Method | What it assumes | Use when |
|---|---|---|
| **z-score average** | all indicators equally important, fully compensatory, uncorrelated | almost never justified; **only** after cluster-size reweighting (C1.4) |
| **PCA** | no latent-variable model; PC1 = max-variance direction | you want dimension reduction, and you accept that PC1 may just be "volume" |
| **Factor analysis** | there *is* a latent construct generating the indicators | you are willing to *claim* a latent "inauthenticity" factor and to test it |
| **Learned weights (supervised)** | labels are trustworthy | ⚠️ under PU labels, learned weights inherit the label's biases; use only inside the validation protocol |

If you go the FA route, you must run and report the standard battery: KMO measure of sampling adequacy, Bartlett's test of sphericity, **parallel analysis** for the number of factors, factor loadings after rotation, and internal consistency (Cronbach's α or McDonald's ω). Reporting a "factor" without these is not defensible.
Citation: OECD/JRC, *Handbook on Constructing Composite Indicators: Methodology and User Guide* (2008), steps on multivariate analysis, normalisation, weighting and aggregation — https://www.oecd.org/content/dam/oecd/en/publications/reports/2008/08/handbook-on-constructing-composite-indicators-methodology-and-user-guide_g1gh9301/9789264043466-en.pdf

---

**PROTOCOL C1.4 — Guard against cluster-domination (the specific trap V2 fell into).**
If you average 20 z-scored indicators and 12 of them are near-duplicates of "posting volume", the composite **is** posting volume with 40% noise. Two acceptable fixes:
1. **Cluster-size reweighting:** each indicator gets weight `1 / |its cluster|`, so every *construct* contributes equally regardless of how many ways you happened to measure it.
2. **Two-stage aggregation:** average within cluster to get a cluster score, then average (or weight by theory) across clusters.

**Mandatory diagnostic:** regress the final composite on each cluster's first PC and report the R². If any single cluster explains > 50% of the composite's variance, the composite is that cluster, and must be renamed accordingly.

---

**PROTOCOL C1.5 — Uncertainty and sensitivity analysis is not optional for a published index.**
Enumerate the discretionary choices — normalisation method (z / min-max / rank), weighting scheme (equal / cluster-size / PCA / expert), aggregation (arithmetic / geometric), winsorisation cutpoint, imputation — take the full or Latin-hypercube cross-product, recompute the index, and report:
- the **rank interval** for every subreddit-month (e.g. "r/X is ranked 4th [2nd–11th] across 96 specifications");
- the fraction of specifications in which each headline claim survives.

Geometric aggregation is worth including as a spec because it is *less compensatory* than arithmetic — a near-zero on any indicator drags the whole index down, which is often the behaviour you actually want from a "risk" index.
Citation: OECD/JRC Handbook, uncertainty & sensitivity analysis chapter (same URL as C1.3).

---

# 4. PROTOCOL FAMILY D — Data leakage

Base taxonomy (Kapoor & Narayanan, *Patterns* 4(9), 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/):

- **[L1] No clean train/test separation**
  - L1.1 no test set
  - L1.2 pre-processing on train+test
  - L1.3 feature selection on train+test
  - L1.4 duplicates across the split
- **[L2] Model uses illegitimate features** — information not available at prediction time
- **[L3] Test set not drawn from the distribution of scientific interest**
  - L3.1 temporal leakage
  - L3.2 non-independence between train and test samples
  - L3.3 sampling bias in the test distribution

Their civil-war case study: after leakage was corrected, complex ML models performed **no better than logistic regression**; one paper's claimed advantage collapsed from 0.14 AUC to 0.01.

## D1. Target leakage — the specific version that will bite this project

**PROTOCOL D1.1 — Maintain a label-provenance blocklist, and assert `features ∩ blocklist = ∅` in CI.**

If the label is derived from a moderation action, then **any feature touching that action is the label wearing a hat.** The blocklist, by label channel:

| Label channel | Fields that MUST be excluded from features |
|---|---|
| admin removal | `removed_by_category ∈ {admin, ...}`, `banned_by`, `body == '[removed]'`, body length when removed, `is_robot_indexable`, `distinguished` |
| mod removal | `removed_by_category ∈ {moderator}`, `mod_reason_*`, `approved_by`, `locked`, removal-count aggregates |
| automod filter | `removed_by_category ∈ {automod_filtered, reddit}`, filter-reason fields |
| account suspension | account-status snapshot, `author == '[deleted]'`, author-profile availability, `author_is_blocked` |
| self-deletion | `body == '[deleted]'`, `author == '[deleted]'`, edited-then-emptied signals |

**PROTOCOL D1.2 — Leave-one-out aggregation for every account-level and thread-level feature.**
If the row is `(post p, account a, month t)` and the label is "p was removed", then `a`'s removal rate in month `t` **includes p**. That is direct target leakage. Every aggregate must be computed as `(total − this row's contribution) / (count − 1)`. This is the most easily missed leak in panel feature engineering, and it produces exactly the "suspiciously good but doesn't generalise" pattern.

**PROTOCOL D1.3 — Missingness is a feature *of the label*.** `[removed]`/`[deleted]` placeholders mean text-derived features (length, sentiment, readability, embedding) are structurally missing for exactly the positive class. A model that learns "empty body ⇒ positive" has learned nothing. **Mitigation:** either drop text features entirely for the removal-label task, or restrict training to rows where text was captured *before* removal (a pre-removal snapshot), and document which.

## D2. Temporal leakage

**PROTOCOL D2.1 — Feature horizon rule.** Every feature for row `(·, month t)` uses **only** data with timestamp `≤ end of month t`. No account-lifetime totals, no "total karma" as of collection, no "account still exists".

**PROTOCOL D2.2 — Label horizon rule.** The label uses **only** the window `(t, t+h]` (A0.3). Rows whose label window is not fully observed are censored.

**PROTOCOL D2.3 — Purge and embargo.** Because label windows have duration `h`, a training row from month `t` and a test row from month `t'` share information when their windows overlap. **Purge** training rows whose label window intersects the test window; **embargo** an additional gap (≥ 1 month) after the test block.
Citation: López de Prado's purged K-fold with embargo — https://en.wikipedia.org/wiki/Purged_cross-validation · impl. `PurgedKFold` / `PurgedGroupKFold` (`purgedcv`).

## D3. Group / panel leakage

**PROTOCOL D3.1 — The grouping key is `author_id`, not `row_id`.** The same account recurs across months and subreddits; a random KFold puts the same account on both sides and the model memorises accounts rather than behaviour.

**PROTOCOL D3.2 — Secondary grouping keys to consider:** thread/submission id (for comment-level models); near-duplicate content cluster (crossposts, copypasta, template bots — dedupe by MinHash/SimHash and treat each duplicate cluster as one group); and, for the sub-level analysis, subreddit.

**PROTOCOL D3.3 — Deduplicate *before* splitting**, never after. L1.4.

## D4. Preprocessing leakage

**PROTOCOL D4.1 — Everything fitted goes inside the fold.** The full list for this project: scalers, winsor caps, Yeo-Johnson `λ`, quantile maps, EB prior `(α₀, β₀)`, PCA/FA loadings, correlation-cluster assignments, feature selection, target/frequency encodings, hyperparameters, the **class prior `π̂`**, the **calibration map**, and the **decision threshold**. Implement as an `sklearn.Pipeline` (or equivalent) so this is structurally enforced rather than remembered.

## D5. LEAKAGE CHECKLIST (assertions to write)

```
[ ] D-1  assert set(feature_cols) & set(LABEL_PROVENANCE[active_label]) == empty
[ ] D-2  assert all account/thread aggregates computed leave-one-out
[ ] D-3  assert max(feature_timestamp) <= month_end(t)  for every row
[ ] D-4  assert label window == (t, t+h]  and t+h <= collection_date  (else censored)
[ ] D-5  assert len(set(train.author) & set(test.author)) == 0        [P-GROUP, P-BOTH]
[ ] D-6  assert min(test.month) >= max(train.month) + embargo_months  [P-TIME, P-BOTH]
[ ] D-7  assert purge applied: no train row whose label window overlaps test window
[ ] D-8  assert content-hash dedupe ran BEFORE the split; dup clusters share a group id
[ ] D-9  assert every fitted transform's .fit() was called only on train-fold data
         (test: run pipeline twice with shuffled test set -> train-fold artefacts identical)
[ ] D-10 assert no text-derived feature is used when the label is a removal action,
         OR that a pre-removal text snapshot is documented
[ ] D-11 assert the test distribution matches the stated estimand (no silent filtering
         like "accounts with >= 5 posts" applied to test but not to the population claim)
[ ] D-12 record a model info sheet per Kapoor & Narayanan, published with the report
```

---

# 5. PROTOCOL FAMILY E — Validation protocol

## E1. Run four protocols, publish all four

**PROTOCOL E1.1 — The leakage ladder.** Each rung answers a different question and the *gaps between rungs* are the diagnosis:

| Rung | Split | Question answered | Expected relationship |
|---|---|---|---|
| **R0** Random KFold | i.i.d. shuffle | *nothing* — diagnostic only | highest, always |
| **R1** GroupKFold(author) | unseen accounts, mixed time | "does it generalise to new accounts?" | R0 − R1 = account memorisation |
| **R2** Blocked time | train months 1..k → test k+2 | "does it work next month?" | R0 − R2 = temporal drift + drift-memorisation |
| **R3** Grouped **+** blocked + purge + embargo | unseen accounts **and** later months | **the headline number** | lowest; this is what you publish |

**PROTOCOL E1.2 — Publish R3 as *the* result.** R0 goes in the appendix, labelled "optimistically biased, reported for diagnostic contrast". If R0 − R3 > 0.10 AUC, you have a leakage or memorisation problem and must find it before publishing anything.

**PROTOCOL E1.3 — When random-CV and blocked-CV disagree, say so explicitly and say why.**
The honest write-up is: *"Random 5-fold CV gives AUC_pu 0.81; grouped+blocked CV gives 0.69. The 0.12 gap is attributable to account-level recurrence across folds. We report 0.69."* Never report the higher number, never report only a range without saying which protocol produced which end.

Blocking can *also* over-correct: by restricting the range of predictor combinations available for training, blocked CV can induce extrapolation and thereby **over**estimate error. Roberts et al. recommend block CV wherever dependence structures exist — even when residuals look uncorrelated — while acknowledging this two-sided bias. Report both bounds and describe them as such.
Citation: Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 40:913–929 (2017) — https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881

## E2. Nested CV — how much optimism does in-fold selection actually save?

**PROTOCOL E2.1 — Feature selection, hyperparameter tuning, threshold choice, calibration, and class-prior estimation ALL go inside the inner loop. No exceptions.**

**The evidence, which is unusually stark:**
- **Ambroise & McLachlan (PNAS 99:6562–6566, 2002)**: selecting genes on the full dataset before cross-validating produced *near-zero* apparent error rates on microarray data where the honest error was on the order of 30%. Selection bias of this magnitude is not a rounding error; it is the entire result.
- **Varma & Simon (BMC Bioinformatics 7:91, 2006)**: cross-validation used both to select and to evaluate is substantially biased; nested CV is required for an almost-unbiased estimate. — https://www.researchgate.net/publication/7273753
- **Cawley & Talbot (JMLR 11:2079–2107, 2010)**: over-fitting in *model selection* is a distinct and widely underappreciated source of optimism, analogous to feature-selection bias, with the same remedy. — https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
- Modern quantification: mean AUC inflation of **+0.040** at k=10 with positive inflation in **92%** of datasets; for `n < 100` the AUC bias frequently runs 5–10%.
- Even *unsupervised* preprocessing on the pooled data biases CV. — https://academic.oup.com/jrsssb/article-abstract/84/4/1474/7073256

**PROTOCOL E2.2 — Structure.** Outer loop = R3 (grouped + blocked + purged). Inner loop = grouped CV *within the outer training block only*, used for tuning and selection. The outer test block is touched exactly once, at the end, per configuration.

*Pragmatic caveat, worth knowing:* for pure *classifier selection* among a handful of well-behaved candidates, nested CV can be overkill relative to its cost (https://www.sciencedirect.com/science/article/abs/pii/S0957417421006540). But this project does **aggressive feature selection over ~60 metrics**, which is exactly the regime where Ambroise & McLachlan's bias is maximal. Nest it.

## E3. Baselines that must be beaten

**PROTOCOL E3.1 — Report these four baselines beside the XGBoost number, on R3:**
1. **Intercept-only** (the base rate).
2. **Volume-only**: a single feature, `log1p(n_posts)` (or `n_comments`). ⚠️ **If XGBoost does not clearly beat this, you have built a volume detector.** Given the volume confound documented in §B1, this is the most important baseline in the project.
3. **Logistic regression** on 3–5 pre-registered, obviously-motivated features.
4. **Account-age-only.**

Kapoor & Narayanan's central empirical finding is that once leakage is fixed, complex ML routinely fails to beat logistic regression. Pre-committing to these baselines is how you find that out before a reviewer does.

## E4. Uncertainty

**PROTOCOL E4.1 — Block bootstrap over accounts (and separately over months), never over rows.** Rows are not independent; accounts are closer to it. Resample author IDs with replacement, recompute the whole metric, `B ≥ 1000`. Report the 2.5/97.5 percentiles.

**PROTOCOL E4.2 — Report fold-to-fold and seed-to-seed variance.** A model whose R3 AUC ranges 0.61–0.78 across folds has not achieved 0.70; it has achieved "somewhere in a wide band", and that band belongs in the headline.

## E5. VALIDATION-PROTOCOL CHECKLIST (codeable)

```
[ ] V-1  estimand written down (E1/E2/E3 of Protocol 0.1) and matched to the metric
[ ] V-2  label channel fixed; per-channel models fit; 5x5 transfer matrix produced
[ ] V-3  (beta - alpha) bounded; AUC ceiling computed and stated BEFORE modelling
[ ] V-4  content dedupe -> group ids -> split.  In that order.
[ ] V-5  four rungs run: R0 random / R1 grouped / R2 blocked / R3 grouped+blocked+purge+embargo
[ ] V-6  all preprocessing inside sklearn Pipeline; .fit only on train fold
[ ] V-7  nested CV: inner loop does feature selection + hyperparams + threshold
         + calibration + pi-hat.  Outer test block touched once.
[ ] V-8  four baselines reported on R3 (intercept, volume-only, small LR, age-only)
[ ] V-9  label permutation null run end-to-end; observed vs null distribution reported
[ ] V-10 block bootstrap over authors (B >= 1000) -> CI on every headline number
[ ] V-11 fold-level and seed-level dispersion reported, not just the mean
[ ] V-12 leakage-ladder table published with R0-R3 gaps explained in prose
[ ] V-13 PU corrections applied: AUC_true grid over (alpha, beta); recall@k; lift@k; PU-F1
[ ] V-14 calibration fitted on a held-out calibration fold, evaluated on the outer test
[ ] V-15 final test months (most recent 2-3) held out entirely and touched ONCE,
         after the feature list and model family are frozen
```

---

# 6. PROTOCOL FAMILY F — Calibration and quantification

## F1. Ranking ≠ probability

**PROTOCOL F1.1 — Report a discrimination metric AND a calibration metric, always together.**
AUC is invariant to any strictly monotone transform of the score. A model can have AUC 0.95 and produce probabilities that are uniformly 10× too large. "Well-ranked" and "well-calibrated" are orthogonal properties, and only the second one licenses statements like "3.2% of activity is inauthentic".

**PROTOCOL F1.2 — Report the Brier score with its Murphy decomposition.**
```
BS = (1/N) Σ ( p̂_i − y_i )²
BS = REL − RES + UNC
```
- **REL** (reliability / calibration) — mean squared gap between predicted probability and observed frequency within each bin. **Lower is better; this is the calibration term.**
- **RES** (resolution / discrimination) — how far bin-wise observed frequencies depart from the base rate. **Higher is better.**
- **UNC** (uncertainty) — `p̄(1−p̄)`, the irreducible base-rate term; identical for all models on the same data, so it is the benchmark a model must beat.

Reporting the decomposition tells you *which* problem you have: bad REL → recalibrate; bad RES → the features are weak, and no amount of calibration will help.
Citation: Murphy (1973); modern treatment — https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985

**PROTOCOL F1.3 — Reliability diagram + ECE, with the ECE caveat spelled out.** ECE is sensitive to the number of bins and is a biased estimator; use equal-*mass* bins, report the bin count, and treat ECE as a rough summary of the reliability diagram rather than a metric to optimise.

## F2. Which calibrator

**PROTOCOL F2.1 — Decision rule on calibration-set size `n_cal`:**
```
n_cal <  1000   ->  Platt scaling (sigmoid):  p̂ = 1 / (1 + exp(A·f(x) + B))
n_cal >= 1000   ->  Isotonic regression (PAVA), non-parametric monotone fit
```
Platt assumes the distortion is sigmoid-shaped; isotonic assumes only monotonicity, so it is strictly more general — but it overfits on small calibration sets. Empirically, isotonic matches or beats Platt at ≥1000 calibration points, and loses below ~2000 in some regimes. With millions of rows you are comfortably in isotonic territory; use isotonic, and cross-check against Platt.

Boosted trees specifically are known to produce **badly** calibrated posteriors (sigmoid-distorted, pushed away from 0 and 1), and both methods substantially improve them.
Citation: Niculescu-Mizil & Caruana, *Predicting Good Probabilities With Supervised Learning*, ICML 2005 — https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf · *Obtaining Calibrated Probabilities from Boosting* — https://arxiv.org/abs/1207.1403

**PROTOCOL F2.2 — Calibrate on a dedicated held-out fold, never on training data.** Calibrating on training predictions fits the calibrator to the model's overfitting and produces a *worse*-calibrated model that *looks* better.

**PROTOCOL F2.3 — Under PU labels, be explicit about what you calibrated to.**
The calibrator makes `P(s=1|x)` honest — the probability of being **labeled**, not of being **inauthentic**. Under SCAR, convert afterwards:
```
P̂(y=1|x) = P̂(s=1|x) / ĉ            (Elkan & Noto)
```
and clip to `[0,1]`. Because `ĉ` is itself an interval estimate (A2.2), `P̂(y=1|x)` must be reported as an interval too. **Never publish a per-account probability without the `ĉ` band.**

## F3. Quantification — the actual deliverable

**PROTOCOL F3.1 — Do not classify-and-count.** Naive CC, `p̂_CC = (1/N) Σ 1{f(x_i) > τ}`, is a biased estimator of prevalence whenever the classifier is imperfect, and its bias grows with the distance between training and deployment prevalence — which is exactly the drift you are trying to *measure*.

**PROTOCOL F3.2 — Estimator ladder for the monthly percentage.**

**(a) ACC — Adjusted Classify and Count.** From `p_CC = p_true·TPR + (1 − p_true)·FPR`:
```
p̂_ACC = ( p̂_CC − FPR ) / ( TPR − FPR )        , clipped to [0,1]
```
with TPR/FPR estimated on a held-out fold from the training distribution.

**(b) PACC — Probabilistic ACC.** Replace crisp counts with expected soft counts, `p̂_PCC = (1/N) Σ p̂(y=1|x_i)`, and estimate TPR/FPR as the mean posterior on positives and negatives respectively. Same correction formula. Generally lower variance than ACC.
Citation: Forman (2008); Bella et al.; rediscovered as BBSE (Lipton et al. 2018). Review: https://arxiv.org/abs/2103.03223

**(c) SLD / EMQ — EM-based prior correction. USE THIS AS PRIMARY when prior shift is expected month-to-month.**
Initialise `p̂⁽⁰⁾(y=1)` = training prior; let `p⁽⁰⁾(y=1|x_i)` = the model's original posteriors. Iterate to convergence:
```
E-step:   p̂⁽ᵏ⁾(y=1|x_i)  =        [ p̂⁽ᵏ⁾(y=1) / p̂⁽⁰⁾(y=1) ] · p⁽⁰⁾(y=1|x_i)
                             ────────────────────────────────────────────────────────────
                             Σ_{c∈{0,1}} [ p̂⁽ᵏ⁾(y=c) / p̂⁽⁰⁾(y=c) ] · p⁽⁰⁾(y=c|x_i)

M-step:   p̂⁽ᵏ⁺¹⁾(y=1)   =  (1/N) Σ_{i=1..N} p̂⁽ᵏ⁾(y=1|x_i)
```
The fixed point is the prevalence estimate for that month. SLD requires *well-calibrated* input posteriors — so F2 is a prerequisite, not an optional extra.
Citation: Saerens, Latinne & Decaestecker, *Adjusting the outputs of a classifier to new a priori probabilities*, Neural Computation 14(1):21–41 (2002) — https://direct.mit.edu/neco/article/14/1/21/6577 · critical reassessment — https://dl.acm.org/doi/10.1145/3433164

**(d) Direct MPE per month — the cleanest framing for this project.**
The monthly inauthentic share **is** the mixture proportion `α_{s,t}`. Score every account-month, then run TIcE / BBE / DEDPUL on *that month's score distribution* directly. This bypasses thresholds and TPR/FPR entirely, and it is the framing that survives the PU labels most gracefully.

**PROTOCOL F3.3 — Evaluate quantification with quantification metrics, not classification metrics.**
```
AE   = | p̂ − p |                          absolute error (most interpretable; report this)
RAE  = | p̂ − p | / p                      relative absolute error (report when p is small)
NKLD = normalised Kullback-Leibler divergence
```
Validate with the **Artificial Prevalence Protocol (APP)**: construct held-out samples at a grid of true prevalences (e.g. 0.01, 0.02, 0.05, 0.10, 0.20, 0.50) by resampling, and report AE across the grid. A quantifier that is accurate only at the training prevalence is useless for detecting a trend.
Citation: Sebastiani; González et al., *A Review on Quantification Learning*, ACM CSUR — https://dl.acm.org/doi/pdf/10.1145/3117807 · comparative evaluation — https://arxiv.org/abs/2103.03223

**PROTOCOL F3.4 — Publish the monthly percentage as an interval with a decomposed error budget:**
```
r/<sub>, 2025-06:  4.1%  [2.6% – 6.8%]
  contributions to the interval:
    sampling / bootstrap over accounts        ±0.4 pp
    class-prior method spread (TIcE/BBE/DEDPUL/EN)   ±1.3 pp     <- usually dominates
    label-channel choice (admin vs mod vs susp)      ±1.1 pp     <- usually second
    calibration method (isotonic vs Platt)           ±0.2 pp
```
Showing that the method-choice terms dominate the sampling term is honest, is true, and pre-empts the obvious criticism.

---

# 7. PROTOCOL FAMILY G — Multiple testing and effect size

**PROTOCOL G1.1 — Default: Benjamini–Hochberg at `q = 0.05` (or 0.10 for exploratory screens).**
Sort p-values `p₍₁₎ ≤ … ≤ p₍ₘ₎`; find the largest `k` with `p₍ₖ₎ ≤ (k/m)·q`; reject `H₍₁₎…H₍ₖ₎`.
BH controls FDR under independence **and** under **PRDS** (positive regression dependency on a subset). Metrics within a subreddit, and the same metric across adjacent months, are positively dependent — PRDS is the right working assumption, so **plain BH is appropriate here**.
Citation: Benjamini & Hochberg (1995); Benjamini & Yekutieli, *Ann. Statist.* 29(4):1165–1188 (2001).

**PROTOCOL G1.2 — Use Benjamini–Yekutieli only when the dependence structure is genuinely unknown or negatively correlated.**
```
BY threshold:   p₍ₖ₎ ≤ ( k / m ) · q / Σ_{j=1..m} (1/j)
```
For `m ≈ 30,000` tests (45 subs × 12 months × 60 metrics ≈ 32,400): `Σ 1/j ≈ ln(32400) + 0.5772 ≈ 10.96`. **BY is ~11× more conservative than BH at this scale.** Report BY alongside BH only when a specific claim is contested; making BY the default would silently kill nearly every finding.

**PROTOCOL G1.3 — Storey q-values for extra power on the big exploratory screen.**
BH implicitly assumes `π₀ = 1` (all nulls true). Storey estimates `π̂₀` from the flat right tail of the p-value histogram and plugs it in, gaining power whenever a real fraction of tests are non-null:
```
q(p_i) = min over t >= p_i of   π̂₀ · m · t / #{ p_j <= t }
```
`π̂₀` via the Storey–Tibshirani spline smoother or the bootstrap method. **Always plot the p-value histogram before applying it** — a non-uniform, non-flat-tailed histogram means the tests are misspecified and no FDR method will save you.
Citation: Storey (2002, 2003); Storey & Tibshirani (2003) — https://genomics.princeton.edu/storeylab/papers/Storey_FDR_2011.pdf · `qvalue` Bioconductor package.

**PROTOCOL G1.4 — Bonferroni only for a small, pre-registered confirmatory family (≤ 10 headline claims).** Split the analysis explicitly into *confirmatory* (pre-registered, Bonferroni or BH at 0.05) and *exploratory* (everything else, BH at 0.10, labelled "hypothesis-generating"). Do not mix.

## G2. Effect sizes — the actual answer to "everything is significant"

**PROTOCOL G2.1 — Delete p-values from descriptive tables. Report effect size + bootstrap CI, and screen on effect size first.**
With millions of rows, `p` is a deterministic function of `n` and tells you nothing about magnitude. Pre-register a **smallest effect size of interest (SESOI)** per metric family before looking at results, and filter on it.

**PROTOCOL G2.2 — Effect-size selection table:**

| Data type | Effect size | Formula / note |
|---|---|---|
| Proportions / rates | **risk difference** (`p₁ − p₂`) — report first, it is the actionable one | plus **risk ratio** `p₁/p₂` for relative framing |
| Proportions, scale-free | **Cohen's h** | `h = 2·arcsin√p₁ − 2·arcsin√p₂` — variance-stabilised, so it does not distort near 0 or 1 |
| Counts with exposure | **incidence rate ratio (IRR)** | from Poisson/negative-binomial GLM with `offset = log(exposure)`; **always test for overdispersion and fall back to NB** — social-media counts are massively overdispersed |
| Skewed continuous | **Cliff's δ** (= rank-biserial `r`) | `δ = P(X > Y) − P(Y > X)`; equivalently `δ = 2·AUC − 1`; distribution-free, tie-tolerant, immune to skew-27. **Use this, not Cohen's d.** |
| Contingency tables | **Cramér's V** | note V is attenuated at high df; report the table too |
| Network statistics | **ratio to a null model** | `T_obs / E_null[T]` under a configuration model / degree-preserving rewiring, with a permutation CI. Never report a raw clustering coefficient or assortativity without its null. |
| Model comparison | **ΔAUC, ΔBrier, ΔAE** with paired bootstrap CI | not "model A is significant" |

Citations: Cliff's δ / rank-biserial equivalence — https://easystats.github.io/effectsize/reference/rank_biserial.html · practical-vs-statistical-significance at scale — https://www.nature.com/articles/s41598-021-00199-5

**PROTOCOL G2.3 — The CI-width rule.** For every headline number, report the CI width. If the CI is narrower than the SESOI, state plainly: *"the interval is far narrower than any effect we would act on; statistical significance carries no information here, and the finding rests entirely on the point estimate's magnitude."* This one sentence, repeated, is what makes a census-scale analysis credible.

---

# 8. PROTOCOL FAMILY H — Model interpretation

**PROTOCOL H1.1 — TreeSHAP: choose the perturbation mode deliberately and state which you used.**

```python
shap.TreeExplainer(model, data=background_df, feature_perturbation="interventional")
# vs
shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")   # the default
```
- `interventional` (with a background sample): **"true to the model"** — marginal expectations; credit goes only to features the model actually uses. Features with zero model influence get exactly zero SHAP.
- `tree_path_dependent`: approximates *conditional* expectations using the tree's own coverage statistics. **A feature with no influence on the prediction function can receive non-zero SHAP purely by being correlated with an influential one.**

**Recommendation for this project: `interventional`, because the question being asked is "what is the model doing?" not "what does the world look like?"** If you switch to conditional/`TreeSHAP`-path-dependent to answer a data-oriented question, say so and do not mix the two in one figure.
Citation: Aas, Jullum & Løland (2021) on conditional SHAP under dependence · https://christophm.github.io/interpretable-ml-book/shap.html · https://github.com/slundberg/shap/issues/882

**PROTOCOL H1.2 — Aggregate SHAP to *family* level before publishing any ranked list.**
With correlated metric families, SHAP splits credit arbitrarily *within* the family — the split is an artefact of tree-building order and the random seed, not a finding. Sum `mean(|SHAP|)` within each correlation cluster (C1.2) and rank **clusters**, not individual metrics.

**PROTOCOL H1.3 — Never state SHAP causally.** "Account age contributes −0.3 to the score" is a statement about the model. "Older accounts are less likely to be inauthentic" is a causal claim the model cannot support. Enforce this in the writing, not just in the analyst's head.

**PROTOCOL H2.1 — Permutation importance is unreliable here; prefer LOCO at family level.**
Permuting a feature breaks its dependence with the others and forces the model to **extrapolate into regions with no training data**, systematically over-emphasising correlated features. Alternatives, in order of preference for this project:
1. **LOCO (leave-one-covariate-out) at the family level** — drop the whole correlation cluster, **refit**, and re-run the *full R3 validation protocol*. The ΔAUC / ΔAE is the only importance measure that honestly answers "what would we lose without this family?" It is expensive; run it for the top ~6 families only.
2. **Grouped permutation** — permute a whole correlated block jointly, which at least keeps the within-block joint distribution intact.
3. **Conditional permutation importance** (Strobl et al. 2008) — permute within strata defined by correlated covariates.
4. Marginal permutation importance — diagnostic only.
Citation: Hooker & Mentch, *Please Stop Permuting Features: An Explanation and Alternatives* — https://www.semanticscholar.org/paper/e6121c3744b9af235b32d35e87350ffd8b390efd · https://slds-lmu.github.io/iml_methods_limitations/pfi-correlated.html

**PROTOCOL H2.2 — Stability gate on any published importance ranking.** Recompute the ranking across CV folds and ≥5 random seeds. Report rank ranges. If the top-10 ordering is not stable, publish an unordered *set* of important families, not a ranked list.

---

# 9. Honest assessment: what "AUC > 0.90" would and would not mean

**PROTOCOL Z.1 — State the ceiling before stating the target.**
By A0.1, `AUC_pu^max = (1 + β − α)/2`. A target of `AUC_pu > 0.90` implicitly asserts `β − α > 0.80` — i.e. that the labels are ≥80% pure *and* that fewer than a few percent of unlabeled accounts are undetected positives. Given that the five label channels are near-disjoint (A0.2) and that suspension is a documented-unreliable proxy, **that assertion is almost certainly false, and the target should be renegotiated to a quantification-error target instead** (e.g. "monthly prevalence estimated to within ±1.5 pp AE under the APP protocol"), which is both achievable and closer to what the deliverable actually is.

**PROTOCOL Z.2 — The three ways V3 could hit 0.90 that are all artefacts.** Check each explicitly and report the check:
1. **Volume leakage** — the score is a subreddit-size detector. *Test:* the volume-only baseline (E3.1 #2) and the metric-level volume test (B1.1).
2. **Target leakage** — a feature encodes the moderation action. *Test:* the provenance blocklist (D1.1) and leave-one-out aggregation (D1.2).
3. **Account memorisation** — the same accounts sit on both sides of the split. *Test:* the R0-vs-R3 gap (E1.1).

A 0.90 that survives all three, on R3, with a permutation null far below it, is real. A 0.90 that appears after a feature-engineering sprint without those three checks is not.

**PROTOCOL Z.3 — Do not publish per-account classifications.**
The Botometer literature is the cautionary tale: score thresholds proved unstable enough that downstream social-science studies unknowingly counted large numbers of humans as bots. Publish **subreddit-month aggregates with intervals**, plus a documented method; if per-account scores must be shared, share them as deciles with an explicit false-positive discussion, never as a binary "bot" label.
Citation: Rauchfleisch & Kaiser, PLOS ONE 15(10):e0241045 (2020) — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045

**PROTOCOL Z.4 — Publish a model info sheet** (Kapoor & Narayanan) alongside the report: the estimand, the label definition and window, the exact split protocol, every fitted preprocessing step and where it was fitted, the baseline comparisons, and the leakage checklist with each item marked pass/fail/NA.

---

# 10. Priority ordering for V3

Ranked by expected AUC/credibility gain per unit of effort:

1. **A0.1 / A0.2** — bound `(β − α)`; run the 5×5 cross-channel transfer matrix. *This most likely explains the 0.663 plateau outright, and it costs a day.*
2. **D1.1 / D1.2** — provenance blocklist + leave-one-out aggregation. *Removes the most dangerous silent leak.*
3. **E1.1** — build the four-rung leakage ladder into the harness. *Everything downstream is uninterpretable without it.*
4. **B1.1 / B1.2** — volume test on all 60 metrics; null-model standardisation. *Directly attacks the "big sub = bad sub" artefact.*
5. **Protocol 0.1 / F3** — reframe the deliverable as quantification (MPE per subreddit-month). *Changes what "success" means, in a direction you can actually defend.*
6. **B2** — small-sample corrections for entropy/Gini/HHI. *Cheap, and removes a whole class of spurious cross-sub differences.*
7. **A3** — Bagging-PU with OOB scores, or Elkan–Noto reweighting. *Modelling gain, but only worth it after 1–4.*
8. **E2** — nest the CV. *Costs compute, buys honesty.*
9. **F2** — calibration on a held-out fold, isotonic.
10. **C1.1** — delete the hand-built composites from the model inputs; keep at most one for the report, with sensitivity analysis.
11. **G2** — swap p-value tables for effect-size-plus-CI tables.
12. **H1/H2** — family-level SHAP with `interventional` perturbation; LOCO for the top families.

---

# Reference list

**PU learning**
- Bekker & Davis. *Learning from positive and unlabeled data: a survey.* Machine Learning 109:719–760 (2020). https://link.springer.com/article/10.1007/s10994-020-05877-5 · https://arxiv.org/abs/1811.04820
- Kiryo, Niu, du Plessis & Sugiyama. *Positive-Unlabeled Learning with Non-Negative Risk Estimator.* NeurIPS 2017. https://arxiv.org/abs/1703.00593 · https://github.com/kiryor/nnPUlearning
- Elkan & Noto. *Learning Classifiers from Only Positive and Unlabeled Data.* KDD 2008. https://cseweb.ucsd.edu/~elkan/posonly.pdf
- Jain, White & Radivojac. *Recovering True Classifier Performance in Positive-Unlabeled Learning.* AAAI 2017. https://arxiv.org/abs/1702.00518
- Bekker & Davis. *Estimating the Class Prior in Positive and Unlabeled Data Through Decision Tree Induction* (TIcE). AAAI 2018. https://ojs.aaai.org/index.php/AAAI/article/view/11715
- Bekker & Davis. *Learning from Positive and Unlabeled Data under the Selected At Random Assumption.* 2018. https://arxiv.org/abs/1808.08755
- Ramaswamy, Scott & Tewari. *Mixture Proportion Estimation via Kernel Embeddings of Distributions* (KM1/KM2). ICML 2016. https://arxiv.org/abs/1603.02501
- Garg et al. *Mixture Proportion Estimation and PU Learning: A Modern Approach* (BBE, CVIR, TED^n). NeurIPS 2021. https://arxiv.org/abs/2111.00980
- Ivanov. *DEDPUL.* ICMLA 2020. https://github.com/dimonenka/DEDPUL
- Mordelet & Vert. *A bagging SVM to learn from positive and unlabeled examples.* Pattern Recognition Letters 2014. https://www.sciencedirect.com/science/article/abs/pii/S0167865513002432
- Liu et al. *Learning to Classify Texts Using Positive and Unlabeled Documents* (spy / S-EM). https://www.cs.uic.edu/~liub/publications/ijcai03-textClass.pdf
- Lee & Liu. *Learning with Positive and Unlabeled Examples Using Weighted Logistic Regression.* ICML 2003. https://dblp.org/rec/conf/icml/LeeL03.html
- Claesen et al. *Assessing binary classifiers using only positive and unlabeled data.* 2015. https://arxiv.org/abs/1504.06837
- *Evaluating the Predictive Performance of Positive-Unlabelled Classifiers.* 2022. https://arxiv.org/abs/2206.02423
- PULSNAR (non-SCAR prior estimation). https://peerj.com/articles/cs-2451/
- Permutation testing for PU confidence. BMC Bioinformatics 2024. https://link.springer.com/article/10.1186/s12859-024-05834-2

**Estimators for concentration / diversity**
- Chao & Shen (2003), Miller (1955), Good (1953), Horvitz & Thompson (1952) — as implemented in R `entropy`: https://cran.r-project.org/web/packages/entropy/entropy.pdf
- Chao & Jost. *Coverage-based rarefaction and extrapolation.* Ecology 2012. https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-1952.1
- Chao et al. *Rarefaction and extrapolation with Hill numbers.* Ecol. Monographs 2014. https://esajournals.onlinelibrary.wiley.com/doi/10.1890/13-0133.1
- Hsieh et al. *iNEXT.* MEE 2016. https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.12613
- Deltas. *The Small-Sample Bias of the Gini Coefficient.* Rev. Econ. Stat. 85(1) (2003). https://direct.mit.edu/rest/article/85/1/226/57394

**Composite indices**
- OECD/JRC. *Handbook on Constructing Composite Indicators.* 2008. https://www.oecd.org/content/dam/oecd/en/publications/reports/2008/08/handbook-on-constructing-composite-indicators-methodology-and-user-guide_g1gh9301/9789264043466-en.pdf
- AutoSpearman. https://arxiv.org/abs/1806.09791

**Leakage and validation**
- Kapoor & Narayanan. *Leakage and the reproducibility crisis in ML-based science.* Patterns 4(9) (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/ · https://arxiv.org/abs/2207.07048
- Ambroise & McLachlan. *Selection bias in gene extraction on the basis of microarray gene-expression data.* PNAS 99:6562–6566 (2002).
- Varma & Simon. *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics 7:91 (2006).
- Cawley & Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11 (2010). https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
- Roberts et al. *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40:913–929 (2017). https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881
- Purged cross-validation with embargo (López de Prado). https://en.wikipedia.org/wiki/Purged_cross-validation
- *On the Cross-Validation Bias due to Unsupervised Preprocessing.* JRSS-B 84(4) (2022). https://academic.oup.com/jrsssb/article-abstract/84/4/1474/7073256

**Calibration and quantification**
- Niculescu-Mizil & Caruana. *Predicting Good Probabilities With Supervised Learning.* ICML 2005. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- *Obtaining Calibrated Probabilities from Boosting.* https://arxiv.org/abs/1207.1403
- Saerens, Latinne & Decaestecker. *Adjusting the outputs of a classifier to new a priori probabilities.* Neural Computation 14(1) (2002). https://direct.mit.edu/neco/article/14/1/21/6577
- *A Critical Reassessment of the SLD Algorithm.* ACM TOIS. https://dl.acm.org/doi/10.1145/3433164
- González et al. *A Review on Quantification Learning.* ACM CSUR. https://dl.acm.org/doi/pdf/10.1145/3117807
- *A Comparative Evaluation of Quantification Methods.* https://arxiv.org/abs/2103.03223
- Murphy (1973) Brier decomposition; modern form: https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985

**Multiple testing and effect size**
- Benjamini & Hochberg (1995); Benjamini & Yekutieli, Ann. Statist. 29(4):1165–1188 (2001).
- Storey & Tibshirani (2003); Storey FDR review: https://genomics.princeton.edu/storeylab/papers/Storey_FDR_2011.pdf
- Cliff's δ / rank-biserial: https://easystats.github.io/effectsize/reference/rank_biserial.html
- p-values at large n: https://www.nature.com/articles/s41598-021-00199-5

**Interpretation**
- Hooker & Mentch. *Please Stop Permuting Features: An Explanation and Alternatives.* https://www.semanticscholar.org/paper/e6121c3744b9af235b32d35e87350ffd8b390efd
- Aas, Jullum & Løland (2021), conditional SHAP under dependence; Molnar, *Interpretable ML*, ch. 18: https://christophm.github.io/interpretable-ml-book/shap.html
- TreeSHAP conditional-vs-marginal discussion: https://github.com/slundberg/shap/issues/882

**Domain caution**
- Rauchfleisch & Kaiser. *The False positive problem of automatic bot detection in social science research.* PLOS ONE 15(10):e0241045 (2020). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045
- *Unsupervised detection of coordinated information operations in the wild.* https://arxiv.org/abs/2401.06205
