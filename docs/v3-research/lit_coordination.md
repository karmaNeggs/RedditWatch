# Literature Review: Coordination, Brigading, Rally and Community-Level Prevalence Detection

**Scope:** state of the art relevant to a monthly (subreddit, month) verdict on (A) prevalence of inauthentic activity, (B) coordination beyond individual bad accounts, (C) rally / mobilisation events. Written for a Reddit V3 dataset with full comment trees (body, timestamps, `parent_id`), post metadata, and per-account cross-subreddit histories via Arctic Shift. **No vote data.**

**Headline for the V3 design:** the field's centre of gravity has moved from "share the same URL within *k* seconds" (Twitter/Facebook idiom, weak on Reddit) to **bipartite co-action networks validated against a degree-preserving null model**. On Reddit the natural bipartite structure is *accounts × threads* (or accounts × parent-comments, accounts × near-duplicate text clusters). This maps exactly onto the Tumminello / Saracco / Neal statistical-validation machinery, which is the most implementable and defensible thing in this literature. Everything else is decoration on top of that.

---

## 1. Coordination detection — methods with algorithmic detail

### 1.1 The canonical pipeline (Pacheco et al., ICWSM 2021)

The most-cited general framework. Four stages:

1. **Choose a behavioural trace** — a signal that should be *independent across accounts* under the null. Four families: **content** (words, n-grams, hashtags, media hashes, links, mentions), **activity** (timestamps, places), **identity** (names, handles, profile pictures, creation dates), and **combinations**. Filter out nodes with insufficient "support" (low activity / few interactions with the chosen trace).
2. **Build a bipartite network** accounts × trace-features. Optionally weight by strength of association: unweighted, **TF-IDF** (to discount popular features), or custom normalisation.
3. **Project to an account-account network.** Similarity options given explicitly: raw **co-occurrence count**, **Jaccard**, **cosine similarity of TF-IDF vectors**, **mutual information**, **χ²**.
4. **Filter edges, then cluster.** Their filtering was blunt percentile cuts, and the values are worth knowing as anchors: handle-sharing case — no filter (suspicious by construction); image-sharing — **top 1%** of edge weights; co-retweet — **top 0.5%**; synchronised-activity — **top 0.5%**. Clustering: **connected components** where edges are suspicious by design, otherwise **community detection** (k-core, modularity maximisation, label propagation).

Validation was news reports + manual annotation of clusters + Botometer comparison against random accounts (see §6).

<https://ojs.aaai.org/index.php/ICWSM/article/view/18075> · preprint <https://arxiv.org/abs/2001.05658>

**Reddit translation.** Traces available in V3, ranked by expected signal:
- accounts × **thread (link_id)** — co-participation. Cheap, dense, needs a strong null (§2).
- accounts × **parent comment id** — co-reply to the *same specific comment*. Much sparser, much higher signal. This is the Reddit analogue of "co-reply attacks" (§1.7).
- accounts × **near-duplicate text cluster** (MinHash/SimHash over comment bodies, or embedding cluster). Analogue of co-URL/co-retweet; the strongest single signal because organic co-occurrence is rare.
- accounts × **submission the account replied to within Δt of another account** — adds the temporal constraint.
- accounts × **external domain / URL** in comment bodies — direct CLSB analogue.
- accounts × **subreddit** over the account's history — for cross-community flow (§7), *not* for within-subreddit coordination (too dense).

### 1.2 Estimating the coordination interval FROM DATA — CooRnet

This is the only widely used *data-driven* interval estimator. Exact algorithm of `CooRnet::estimate_coord_interval()` (defaults `q = 0.1`, `p = 0.5`):

1. Drop URLs shared only once (cannot evidence coordination).
2. For each URL: total share count `n_u`; first-share timestamp `t_0`; rank each share chronologically; compute `secs_from_first = difftime(t_share, t_0, units="secs")`; compute `perc_of_shares = rank / n_u`.
3. Take the **rank-2 share** of every URL → distribution of "time to second share".
4. Keep only URLs whose time-to-second-share is **≤ quantile(that distribution, q = 0.1)** — i.e. the fastest 10% of URLs.
5. Within that fast subset, for each URL find the *minimum* `secs_from_first` among shares with `perc_of_shares > p = 0.5` — i.e. the time by which half of that URL's shares had happened.
6. **Coordination interval = median of those per-URL times.** If the median is 0, it is set to 1 second.

Then `get_coord_shares()`:
- Bin each URL's shares with `cut(as.POSIXct(date), breaks = coordination_interval)`; any bin containing ≥2 *distinct* accounts marks those shares coordinated.
- Build account–account edges for every co-occurring pair; each edge carries `t_coord_share`, the vector of timestamps at which the pair co-occurred. Edge weight = number of such co-occurrences.
- **`percentile_edge_weight = 0.90` (default)**: retain only edges above the 90th percentile of the edge-weight distribution — this is the *repetition* requirement.
- `build_coord_graph()` extracts connected components and reports per-entity degree/component/metadata; component-level outputs include coordination ratios and domain dispersion.

Source: <https://github.com/fabiogiglietto/CooRnet/> · <https://coornet.org/> · original method paper Giglietto, Righetti, Rossi & Marino, *Information, Communication & Society* 23(6), 2020, <https://www.tandfonline.com/doi/abs/10.1080/1369118X.2020.1739732>

**Critique to carry forward.** Steps 4–6 are a heuristic stack of quantiles with no null model and no error bars; `q` and `p` are themselves hand-set. It answers "how fast is fast for *this* corpus", which is genuinely better than a hardcoded 10s, but it is not a significance test. Treat it as a *scale estimator*, then do significance separately (§2). Also note the 90th-percentile edge filter guarantees you always output ~10% of edges as "coordinated" regardless of whether any coordination exists — a base-rate trap.

**Direct evidence that transplanted thresholds fail.** A 2025 *Scientific Reports* study shows that using a previously published threshold (10 s window, ≥5 repeated shares, from a 2020 COVID-misinformation Facebook study) on 2021 US-politics Facebook pages **missed large amounts of coordination**; the authors argue the method must be recalibrated per corpus and per period, and demonstrate sensitivity analysis across thresholds as the validity argument. <https://www.nature.com/articles/s41598-025-00233-w>

### 1.3 CooRTweet (generalised, validated re-implementation)

Content-agnostic (`object_id` can be a URL, hashtag, image hash, or arbitrary object). Input schema is exactly four columns: `object_id`, `account_id`, `content_id`, `timestamp_share` (UNIX).

- `detect_groups(x, time_window, min_participation)` — `time_window` in seconds (default 10; examples use 30 and 60); `min_participation` = minimum number of accounts to form a group (e.g. 2).
- `generate_coordinated_network(..., edge_weight = 0.5, ...)` — produces an `igraph` object whose edges carry `weight`, `avg_time_delta` (mean interval between the pair's coordinated shares), `edge_symmetry_score` (asymmetry in who acts first — a directionality/leader-follower signal), and `n_content_id` / `n_content_id_y` (per-account content counts, i.e. the volume normalisation terms).
- Explicitly models the "uncoordinated network in which coordination is contextualised" — i.e. it retains the complement graph as a baseline rather than discarding it.

`edge_symmetry_score` is worth stealing: in genuine coordination one account is often reliably first (the seed) and others follow; symmetric co-occurrence is more consistent with shared exogenous stimulus.

<https://cran.r-project.org/web/packages/CooRTweet/vignettes/vignette.html> · <https://github.com/nicolarighetti/CooRTweet> · paper: Righetti & Giglietto, *Computational Communication Research* 2025, <https://www.aup-online.com/content/journals/10.5117/CCR2025.1.7.RIGH>

### 1.4 Similarity + backboning + coordination-as-a-spectrum (Nizzoli et al., ICWSM 2021)

The methodologically cleanest of the "network" family, and the one I'd copy for the (subreddit, month) score.

1. Represent each user as a **TF-IDF-weighted vector over retweeted tweet IDs** (Reddit: over thread IDs, parent-comment IDs, or text-cluster IDs). TF-IDF is what makes co-participation in a giant thread nearly worthless and co-participation in an obscure thread heavily weighted — essential on Reddit.
2. Pairwise **cosine similarity** → weighted undirected user-similarity network `G(V,E,W)`.
3. **Multiscale backbone via the disparity filter** (Serrano, Boguñá & Vespignani, PNAS 2009) — *not* a global threshold. Retains locally statistically-significant edges at every weight scale. Their filtered network retained 276,775 edges.
4. **Coordination is continuous, not binary.** For a moving threshold `t`, the "degree of coordination" is the **percentile rank of `t` in the edge-weight distribution**. "Degree of coordination = 0.9" means the subnetwork of the top-10% strongest edges.
5. **Algorithm 1 (iterative community detection):** start at `t_0` = min edge weight; run **Louvain (resolution 1.5)**; increment `t_i` by `δw`; drop edges with `w(e) < t_i` and drop isolated nodes; re-run community detection *seeded with the previous iteration's partition*; trace how communities evolve across the coordination spectrum. Minimum community size 20 users at `t_0`.
6. Key empirical finding: **coordination and automation are orthogonal** — Botometer scores were uncorrelated with coordination level. Do not conflate your bot score with your coordination score.

<https://ojs.aaai.org/index.php/ICWSM/article/view/18074> · <https://ar5iv.labs.arxiv.org/html/2008.08370> · dataset <https://zenodo.org/records/4647893>

**Disparity filter formula** (implementable in ~15 lines). For node `i` with degree `k_i`, normalise its incident weights `p_ij = w_ij / Σ_j w_ij`. Under the null (weights distributed uniformly at random over the node's `k_i` links), retain edge `(i,j)` iff

```
α_ij = 1 - (k_i - 1) ∫_0^{p_ij} (1 - x)^{k_i - 2} dx  <  α
```

which has closed form `α_ij = (1 - p_ij)^(k_i - 1)`. For directed graphs compute `α_ij^in` and `α_ij^out` separately with `k^in`, `k^out`. Keep an edge if it is significant from *either* endpoint. Typical `α` ∈ [0.01, 0.05]; sweep it and report the curve.
<https://en.wikipedia.org/wiki/Disparity_filter_algorithm_of_weighted_network> · Python: <https://github.com/DerwenAI/disparity_filter>

> **Caveat:** the disparity filter's null is *local* (it only conditions on one node's weight profile). It does **not** condition on the popularity of the shared object. For Reddit thread co-participation, that is the wrong null — a 5,000-comment megathread will manufacture edges. Use it *after* a bipartite-null validation, or use the bipartite nulls in §2 instead.

### 1.5 Latent coordination networks + FSA_V (Weber & Neumann, SNAM 2021)

Detailed and directly reusable, especially the temporal decay.

- Reduce every post to `(author, timestamp, interaction type)`; apply coordination criteria `C = {c_1 … c_q}`: **co-retweet, co-hashtag, co-URL, co-mention, co-conversation** (replies sharing a common root — the Reddit analogue is co-participation under the same `link_id`).
- `β^c_{u,v}` = number of inferred links between accounts `u,v` under criterion `c`; edge weight `w_c(e) = β^c_{u,v}`; multi-edges collapse as **`w(e) = Σ_{c=1}^{q} w_c(e)`**.
- **Time windows tested: γ ∈ {15, 60, 360, 1440} minutes**, justified as human-reaction time / frequent-checker / meal-session / once-daily. Report results at all four rather than picking one.
- **Sliding-window persistence with decay:** `w_{c,t}(e) = Σ_{x=0}^{T-1} w_{c,(t-x)}(e) · α^x` — an exponentially-decayed running edge weight. This is the right primitive for a *monthly* verdict that shouldn't be dominated by one day.
- Filtering options tested: **FSA_V** with mean-edge-weight retention `θ = 0.3`; **kNN** with `k = ln|V|`; simple normalised-weight threshold 0.1.
- **FSA_V** extracts *subsets* rather than partitioning: Louvain decomposition first; grow a candidate from the heaviest edge, attaching next-heaviest edges until the candidate's mean edge weight (MEW) drops below `θ × previous MEW` or below the whole-network MEW; keep only HCCs with MEW ≥ network MEW. Complexity `O(|A| log²|A| + |E|)`.
- Validation battery (see §6): Jaccard and overlap coefficients between account sets, cosine similarity on **5-character n-grams** of concatenated posts, DTW barycentre averaging of activity distributions, and one-class classifiers (Bagging PU, RF, SVM). Plus internal/external ratios `IRR = |RT_int| / (|RT_int| + |RT_ext|)` and `IMR` for mentions — **low** ratios indicate covert behaviour (coordinated accounts avoid boosting each other directly).

<https://pmc.ncbi.nlm.nih.gov/articles/PMC8557266> · <https://link.springer.com/article/10.1007/s13278-021-00815-2> · companion: "Who's in the Gang?" <https://arxiv.org/abs/2010.08180>

### 1.6 Synchronised Action Framework (Magelinski, Ng & Carley, *Journal of Online Trust & Safety* 2022)

Multi-view networks over synchronised actions (co-hashtag, co-URL, co-mention, co-retweet, co-timing), validated on the "Reopen America" Twitter conversation where they surfaced three coordinated campaigns. Its lasting contribution is the **"responsible detection"** framing: with a tiny base rate, even a very specific detector produces a majority-false-positive candidate set, so outputs must be treated as *investigative leads*, not verdicts. Relevant caution for publishing per-subreddit numbers.
<https://arxiv.org/abs/2105.07454> · <https://tsjournal.org/index.php/jots/article/view/30>

### 1.7 Coordinated reply attacks (Pote, Elmas, Flammini & Menczer, ICWSM 2025) — closest analogue to Reddit reply trees

Targets of coordinated reply attacks are influential accounts (journalists, media, officials, politicians). Two supervised models: one classifies **tweets** as targeted by a reply attack (**AUC 0.88**), one classifies **replying accounts** as part of the coordinated attack (**AUC 0.97**). Ground truth = Twitter's Information Operations archive, joined to replies collected via the Academic API. Sampling focused on tweets with **≥5 replies from IO accounts**. Released artefacts include per-reply **time deltas** (minutes between original and reply) and **cosine similarity between reply pairs to the same tweet** — i.e. the co-reply network is built on (same target) × (text similarity) × (timing).

Design lesson: *target-conditioned* coordination (many accounts converging on one post/comment) is far more detectable than free-floating co-occurrence, and V3's `parent_id` gives you exactly this.
<https://arxiv.org/abs/2410.19272> · <https://ojs.aaai.org/index.php/ICWSM/article/view/35889>

### 1.8 Astroturfing with real ground truth (Schoch, Keller, Stier & Yang, *Scientific Reports* 2022)

Simple method, strong evaluation, useful numbers.

- Link two accounts if they **co-tweet** (post the same message) **or co-retweet** (retweet the same message) **within a 1-minute window**; require **≥10 co-retweet instances**; keep only accounts with >10 tweets in the period.
- An account is classified as campaign-participating if it is a **non-isolated node** in either network. No classifier at all.
- Robustness: thresholds swept from seconds to 8+ hours.
- Ground truth: **Twitter Information Operations Hub (46 campaigns; 33 with ≥50k tweets)** plus the 2012 South Korean NIS campaign.
- **Recall ≈ 74%** of known astroturfing accounts, **FPR ≈ 1%** of regular users at the 1-minute threshold.
- **Comparison samples are activity-matched** (see §2.6): (a) random users from the target country matched on activity; (b) users engaging the same campaign hashtags, matched on activity.

<https://www.nature.com/articles/s41598-022-08404-9> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC8930979/>

### 1.9 Survey-level synthesis (Tardelli / Cresci et al., 2024–25)

The reference map of the field. Standard pipeline = *user selection → network construction → network filtering → community discovery*, outputting communities `P`, clusters `C`, or binary labels `B`. Their Table 2 catalogues, per method: co-actions (retweet, tweet, URL, hashtag, mention), similarity function (**cardinality, cosine, TF-IDF, text similarity, Jaccard**), filtering (**threshold, EDO, ADJ, backbone, kNN graph**), community detection (**Louvain, Leiden, modularity clustering, connected components**).

Their central criticism, quoted, is the mandate for this project:

> "the field still lacks a formal and statistically grounded definition of coordination, including well-defined null models against which coordinated behavior can be rigorously assessed"

and

> "many approaches rely on platform- and task-specific heuristics, which complicates cross-domain comparisons, reproducibility, and the estimation of effect sizes."

They also warn about **operational circularity**: the definition of coordination is "influenced by the specific technique employed for its detection", so the method silently defines the phenomenon. And they insist coordination ≠ harm: grassroots movements, fandoms, and disaster-response mutual aid are coordinated and benign; conspiracy communities are harmful but authentic. Report coordination and harm as separate axes.

<https://arxiv.org/abs/2408.01257> · <https://arxiv.org/html/2408.01257v2>

Temporal follow-up (Tardelli, Nizzoli, Tesconi, Conti, Nakov, Da San Martino & Cresci, *PNAS* 2024): static coordination analyses are unreliable because coordinated communities are **temporally unstable**; they build a multiplex temporal network and run dynamic community detection, and identify user archetypes by join/leave behaviour. Directly relevant to a *monthly* cadence: month-to-month membership churn is expected, so score *communities* over time, and use community persistence itself as evidence.
<https://arxiv.org/abs/2301.06774> · <https://www.pnas.org/doi/10.1073/pnas.2307038121>

---

## 2. Null models and significance testing — exact construction

This is the section that determines whether the (B) coordination verdict is defensible. There are five constructions worth implementing, in increasing order of rigour and cost.

### 2.1 Hypergeometric statistically-validated network (Tumminello, Miccichè, Lillo, Piilo & Mantegna, PLOS ONE 2011) — **start here**

The simplest exact null for bipartite co-occurrence. Preserves *both* account activity and object size marginals, one pair at a time.

Setup for accounts `A`, `B` over a set of `N_k` objects (threads in the month):
- `N_A` = number of threads `A` commented in
- `N_B` = number of threads `B` commented in
- `N_AB` = number of threads both commented in (observed co-occurrence)

Under the null "B's threads are a uniformly random `N_B`-subset of the `N_k` threads", the co-occurrence count is hypergeometric:

```
H(X | N_k, N_A, N_B) = [ C(N_A, X) · C(N_k - N_A, N_B - X) ] / C(N_k, N_B)
```

One-tailed p-value for over-representation:

```
p(N_AB) = 1 - Σ_{X=0}^{N_AB - 1} H(X | N_k, N_A, N_B)
```

**Multiple testing.** With `N_t` = number of tested pairs:
- **Bonferroni:** threshold `p_b = p_t / N_t` with `p_t = 0.01`. Very conservative.
- **Benjamini–Hochberg FDR:** sort p-values ascending, find the largest `i` with `p_(i) ≤ i · p_t / N_t`, reject all `j ≤ i`. **The Bonferroni network is always a subgraph of the FDR network.**
Reported effect in their movie data: the Bonferroni network kept **16% of nodes / 1% of links**; FDR kept **47% / 7%**. Expect similar order-of-magnitude sparsification on Reddit.

Also supports **multi-links**: classify a validated pair by *which* condition was significant (e.g. co-occurrence in political threads vs. all threads), giving typed coordination edges.

<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017994> · <https://ar5iv.labs.arxiv.org/html/1008.1414>

**Reddit instantiation.** Objects = threads (`link_id`) within (subreddit, month). Accounts = commenters with ≥ `m` comments (set `m` ≥ 3; below that, tests have no power). `N_k` = number of threads in the month. Run pairwise, BH-correct, keep validated edges → coordination graph. Then community-detect (Leiden) on the validated graph only.

**Refinement for tighter signal:** replace "co-occur in thread" with "co-occur in thread **and** within Δt of each other" or "reply to the **same parent comment**". The null is unchanged in form; only `N_A`, `N_B`, `N_AB` are recounted on the refined incidence matrix. The refined version is far more specific, at the cost of power.

**Known limitation:** the hypergeometric null fixes the marginals of the *pair under test* but implicitly assumes objects are exchangeable — it does not account for the fact that some threads are 5,000 comments and some are 3. If you build the incidence matrix as binary account×thread and use `N_k` = number of threads, the thread-size heterogeneity is *not* preserved. That is what §2.2/§2.3 fix.

### 2.2 Bipartite Configuration Model + Poisson-Binomial (Saracco, Straka, Di Clemente, Gabrielli, Caldarelli & Squartini, *New J. Phys.* 2017) — **the rigorous version**

Maximum-entropy null that preserves **both degree sequences in expectation** — i.e. every account's comment count *and* every thread's commenter count.

**Model.** Biadjacency matrix `M` (rows = accounts `r`, columns = threads `c`). Exponential random graph with degree constraints:

```
P(M) = e^{-H(θ, C(M))} / Z(θ)
```

which factorises to independent Bernoulli edges with

```
p_rc = (x_r · y_c) / (1 + x_r · y_c)
```

**Fitting.** Solve the likelihood equations (Lagrange multipliers `x_r`, `y_c`) so ensemble-average degrees equal the observed ones:

```
⟨k_r⟩ = Σ_c p_rc = k_r*          (for every account r)
⟨k_c⟩ = Σ_r p_rc = k_c*          (for every thread c)
```

Solved by fixed-point iteration / Newton; `bicm` handles it (Python: <https://github.com/tsakim/bicm>, docs <https://bicm.readthedocs.io/>).

**Test statistic — the V-motif.** Shared-neighbour count for accounts `r, r'`:

```
V_{rr'} = Σ_{c=1}^{N_C} m_rc · m_r'c
```

Each term `V^c_{rr'} = m_rc · m_r'c` is a Bernoulli with success probability `p_rc · p_r'c` (independent under the null), so `V_{rr'}` is **Poisson-Binomial**:

```
P(V_{rr'} = n) = Σ_{C_n} Π_{c ∈ C_n} p_rc p_r'c · Π_{c ∉ C_n} (1 - p_rc p_r'c)
```

**p-value** (one-tailed survival function):

```
p-value(V*_{rr'}) = P(V_{rr'} ≥ V*_{rr'}) = Σ_{k = V*_{rr'}}^{N_C} P(V_{rr'} = k)
```

**FDR.** Over all `C(N_R, 2)` pairs, find the largest `î` with

```
p-value_(î)  ≤  î · t / C(N_R, 2)          (t = 0.01 typical)
```

and link every pair with p-value ≤ p-value_(î). This is the **validated monopartite projection**: two accounts are connected *iff* they co-occurred a statistically significant number of times given their own activity and the popularity of the things they co-occurred on.

<https://iopscience.iop.org/article/10.1088/1367-2630/aa6b38> · meta-validation review: <https://www.nature.com/articles/s42005-022-00856-9>

**Computing the Poisson-Binomial.** Exact DFT-characteristic-function method is `O(N_C log N_C)` per pair; for large `N_C` use the refined normal approximation or Le Cam / translated-Poisson bound. In practice: exact for small threads counts, Poisson approximation (mean `λ = Σ_c p_rc p_r'c`) when `λ` is small and `N_C` large, which is the usual regime.

### 2.3 Degree-preserving permutation nulls: SDSM / FDSM (`backbone` R package, Neal et al.)

This is the **volume-matched permutation null**, packaged and battle-tested. `backbone_from_bipartite()` offers five nulls, differing in exactly what they constrain:

| Model | Constrains | Method | Cost |
|---|---|---|---|
| **fixedfill** | total number of 1s only | analytic | trivial, weak |
| **fixedrow** | row (account) marginals | analytic | cheap, weak |
| **fixedcol** | column (thread) marginals | analytic | cheap, weak |
| **SDSM** | row **and** column marginals *in expectation* (canonical ensemble) | **BiCM** cellwise probabilities → Poisson-Binomial p-values (logit/logistic regression on degrees when required/prohibited edges are present) | fast, "often the better choice and is the default" |
| **FDSM** | row **and** column marginals **exactly** (microcanonical ensemble) | Monte Carlo resampling via the **fastball** algorithm (C++ successor to **curveball**) | slow, statistically most powerful |

`sdsm(B, alpha = 0.05, missing.as.zero = FALSE, signed = FALSE, mtc = "none", class = "original", narrative = FALSE, ...)` — `mtc` accepts anything R's `p.adjust()` accepts (use `"fdr"`); one-tailed test for unsigned backbones, two-tailed for signed. With `alpha = NULL` it returns the S3 object with the raw p-value matrices, which is what you want if you're going to do your own thresholding.

**This is exactly the null the task specification asks for**: "preserve account activity counts and thread sizes while shuffling". FDSM with curveball/fastball *is* the volume-matched permutation null — it resamples the account×thread incidence matrix uniformly from the set of matrices with identical row sums (per-account comment counts) and column sums (per-thread comment counts), then recomputes the co-occurrence statistic on each sample to build an empirical null distribution.

**Practical recipe (FDSM):**
1. Build binary incidence `M` (accounts × threads) for the (subreddit, month).
2. Observed statistic for every pair: `V*_{rr'} = Σ_c m_rc m_r'c`.
3. Generate `S` (≥ 1000, ideally 10,000) curveball/fastball-randomised matrices `M^(s)` with identical row and column sums.
4. Empirical p-value: `p_{rr'} = (1 + #{s : V^(s)_{rr'} ≥ V*_{rr'}}) / (S + 1)`.
5. BH-FDR across all tested pairs; retain the significant ones.
6. Community-detect on the retained graph.

The `(1 + ...)/(S + 1)` form is the standard bias-corrected Monte Carlo p-value; it also gives a hard floor of `1/(S+1)`, so `S` must exceed the number of tests you want to resolve at FDR.

<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0349258> · <https://cran.r-project.org/web/packages/backbone/backbone.pdf> · SDSM with edge constraints: <https://arxiv.org/abs/2307.12828>

**Recommendation for V3:** SDSM (BiCM-backed) as the default engine, FDSM on the top candidates as confirmation. They "yield similar backbones" but FDSM is exact.

### 2.4 Temporal permutation nulls (destroy synchrony, preserve volume)

Complementary to the bipartite nulls: these keep *who did what* fixed and destroy *when*, so a surviving signal is specifically **temporal** coordination rather than topical co-participation.

- **Shuffled timestamps:** randomly permute the timestamps of all interactions while keeping node pairs fixed — conserves graph structure and the global set of interaction times, removes temporal correlation.
- **Network-shuffling-timestamps (NTS):** every event stays on its original link between the same nodes; each event's occurrence time is sampled **without replacement** from the pool of all occurrence times in the network.
- **Within-stratum permutation (recommended for Reddit):** permute comment timestamps **within thread** (preserves per-thread comment count and the thread's overall temporal envelope, destroys which *account* was fast) or **within account** (preserves the account's diurnal rhythm, destroys which *thread* it hit early). Running both isolates two different mechanisms.
- **Negative control as published practice:** a 2026 Telegram+Reddit coordination study explicitly randomly permutes message timestamps **within channels, preserving per-channel message volume and textual content**, then re-runs the whole detection pipeline; any coordination the pipeline still "finds" is its false-positive floor. <https://arxiv.org/abs/2602.13333>

Report your pipeline's output on the permuted data as a **calibration curve**, not just a single number. If your detector flags 4.1% of accounts on real data and 3.6% on volume-matched permuted data, you have essentially no signal — and this is the single most useful diagnostic you can produce for a monthly report.

### 2.5 Matched-control design for mobilisation (Kumar, Hamilton, Leskovec & Jurafsky, WWW 2018) — **the Reddit-native null**

The best-specified null model in the Reddit literature, and the one to copy for the (C) rally verdict. Data: 40 months of Reddit (Jan 2014 – Apr 2017), 1.8B comments, 100M users, 36,000 communities; 137,113 cross-links after removing overlaps.

**Setup.** A *cross-link* is a post in source community `S` hyperlinking to a post in target `T`. Three phases: **initiation** (cross-link created) → **interaction** (source members comment in target thread) → **impact** (long-term membership change).

**Post-level matching.** For each post `p` analysed, select a **matched post from the same community, created closest in time to `p`, that has no outgoing or incoming cross-links**. Same-community selection removes community-level confounds; closest-in-time removes hourly/daily/weekly seasonality.

**Member definition (exclusive membership).** For a cross-link made on day `d`, members of `S` (resp. `T`) are users who made **≥1 comment in `S` (resp. `T`) in the 30 days prior to `d`** and who did **not** comment in `T` (resp. `S`) in that period. Users who belong to both are excluded. When computing user history they always ignore comments made within **±3 days of the cross-link** to remove reverse causation.

**User-level matching.** For each source member who comments on the target thread, sample a random comparison member of `S` who did not, **matched on number of comments in the past 30 days**.

**Restriction for comparability.** Keep only pairs where the target post and the matched post had a **near-equal number of comments before the cross-link** (difference < 5). This controls for initial popularity.

**Measurement window.** Compare comment counts by source members in a **12-hour window before vs. after** the cross-link.

**Result and the derived threshold.** Matched threads show a **1.6× after/before increase** (this is the *null* — the baseline expected increase irrespective of cross-links); target threads show **8.8×**. A **mobilisation** is defined as a cross-link producing **>1.6× after/before increase** in comments by source members. This yields **22,075 mobilisations = 16% of cross-link posts**.

**Sentiment layer.** MTurk labels on 1,020 source/target post pairs, inter-rater agreement >0.95; negative vs. neutral (positive folded into neutral); LIWC + VADER + stylistic features (avg word length, readability, punctuation counts) → **Random Forest, 400 trees, accuracy 0.80 (10-fold CV)**; 8% labelled negative → **1,809 negative mobilisations**, 20,266 neutral.

**Selected findings usable as priors:**
- **<0.1% of communities initiate 38% of negative mobilisations; <1% initiate 74%.** Conflict initiation is extremely concentrated.
- Mobilisations occur between **topically similar** communities: mean TF-IDF similarity 0.51 vs 0.34 for random pairs (p<0.001).
- Cross-link creators are **highly active core members** (10% more active than matched), but the **attackers who actually show up are much less active** (fraction of past comments in source 0.30 vs. matched 0.17 — i.e. attackers are *more* concentrated in the source than their matches, while both attackers and defenders are less active than their communities' cores). Defenders: 0.32 vs 0.145.
- Attackers use **1.2× more anger words** (LIWC 0.31 vs 0.26); defenders **2.2×** more.
- Target threads during negative mobilisation: **+44% anger words**, and **25× more likely to have a comment removed by a moderator** (deletion rate 0.205 vs 0.008, p<0.001). *Moderator deletion rate is a free, strong, label-like signal in V3 data.*
- Echo chamber quantified with two PageRank variants (A-PageRank / D-PageRank) on the directed weighted user–user reply network built from the target thread: attackers score high on A-PageRank, defenders on D-PageRank.
- Prediction: a "socially-primed" LSTM combining graph embeddings + user/community/text features achieves **AUC 0.76** vs **0.67** for an expert-feature baseline.

<https://dl.acm.org/doi/10.1145/3178876.3186141> · <https://arxiv.org/abs/1803.03697> · <https://cs.stanford.edu/~srijan/pubs/conflict-paper-www18.pdf>

### 2.6 Activity-matched comparison samples

Cheapest useful null, used by Schoch et al. (§1.8): construct two control cohorts — (a) random accounts from the same population, **matched on activity volume**; (b) accounts engaging the same topic/hashtag, **matched on activity volume**. Then report your detector's flag rate on the treatment cohort *and* on both control cohorts. This directly yields the FPR (≈1% in their case). For Reddit: (a) random commenters in the same subreddit-month matched on comment count; (b) commenters in a *different, topically similar* subreddit matched on comment count.

### 2.7 Summary table — which null answers which question

| Question | Null model | What it holds fixed | Cost |
|---|---|---|---|
| "Do A and B co-occur more than chance?" | Hypergeometric SVN | pair's own marginals | trivial |
| "…given thread popularity too?" | BiCM / SDSM (Poisson-Binomial) | both degree sequences (in expectation) | moderate |
| "…exactly?" | FDSM (curveball/fastball) | both degree sequences (exactly) | high |
| "Is the *timing* coordinated, not just the topic?" | within-thread / within-account timestamp permutation | volumes + content; destroys synchrony | moderate |
| "Was this thread's spike caused by the cross-link?" | Kumar matched post + matched user | community, time, pre-link size, user activity | moderate |
| "What's my detector's FPR?" | activity-matched control cohorts | activity volume | cheap |
| "What's my pipeline's false-positive floor end to end?" | full-pipeline rerun on volume-preserving permuted data | everything except the signal | cheap |

---

## 3. Community-level PREVALENCE estimation — formulas

### 3.1 Why naive averaging / thresholding-and-counting is biased

Two distinct failures, often conflated:

**(a) Classify-and-count is biased under prior shift.** Let `p` be true prevalence, `tpr` and `fpr` the classifier's true/false positive rates *measured on the training distribution*. The expected classify-and-count estimate is

```
E[p̂_CC] = p · tpr + (1 - p) · fpr
```

which equals `p` **only if** `tpr = 1` and `fpr = 0`. For any imperfect classifier, `p̂_CC` is biased toward the point where `p = fpr / (1 - tpr + fpr)`; below it CC over-estimates, above it under-estimates. This is *systematic*, does not vanish with more accounts, and is worst exactly in the regime that matters (low true prevalence).

**(b) The base-rate catastrophe.** Gallwitz & Kreil demonstrate this empirically for bot detection. At the standard Botometer bot-score threshold of 0.5, in April 2018, **~50% of US Congress members on Twitter were classified as bots**, along with **12% of Nobel laureates, 17% of Reuters journalists, and 21.9% of UN Women staff**. Their conclusion is blunt: studies estimating bot prevalence from such tools "have, in reality, just investigated false positives and artifacts of this approach." If the true prevalence is ~1–5% and your FPR on genuine humans is 10–20%, your prevalence estimate is essentially a measurement of your FPR.

<https://arxiv.org/abs/2207.11474> · <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3814191>

**Implication for V3:** you must (i) measure `tpr`/`fpr` on a held-out labelled sample *from Indian subreddits*, not from a Twitter benchmark; (ii) correct the count; (iii) publish an interval, not a point estimate; (iv) if `tpr - fpr` is small, refuse to publish a prevalence number at all.

### 3.2 The quantification-learning toolkit (formulas)

Notation: `U` = unlabelled sample (all accounts in a subreddit-month), `|U| = n`; `h` = hard classifier; `s(x) ∈ [0,1]` = calibrated posterior `P(y=1|x)`; `p` = true positive-class prevalence; `p̂` = estimate.

**CC — Classify and Count**
```
p̂_CC = (1/n) · Σ_{x∈U} 1[h(x) = 1]
```
Biased as above. Baseline only.

**ACC — Adjusted Classify and Count** (Forman 2005; = BBSE, Lipton et al. 2018; = the confusion-matrix approach of Saerens et al. 2002)
```
p̂_ACC = (p̂_CC - fpr) / (tpr - fpr)
```
`tpr` and `fpr` are estimated by **k-fold cross-validation on the training set**. Clip to `[0,1]`. **The denominator `tpr - fpr` is Youden's J** — if it is small, ACC amplifies noise catastrophically; this is the number that decides whether prevalence estimation is feasible at all.

**PCC — Probabilistic Classify and Count**
```
p̂_PCC = (1/n) · Σ_{x∈U} s(x)
```
This is the "naive averaging of per-account bot probabilities" named in the brief. It is unbiased **only if `s` is perfectly calibrated on the target distribution** — which it never is under prior shift, since calibration is itself distribution-dependent. In practice PCC is often *worse* than CC.

**PACC — Probabilistic ACC**
```
p̂_PACC = (p̂_PCC - E_{x|y=0}[s(x)]) / (E_{x|y=1}[s(x)] - E_{x|y=0}[s(x)])
```
i.e. identical in form to ACC but with hard counts and rates replaced by **soft counts** — the expected scores conditional on each class, estimated by cross-validation.

**SLD / EMQ — Saerens-Latinne-Decaestecker EM prior adjustment** (Saerens, Latinne & Decaestecker, *Neural Computation* 2002). Mutually recursive re-estimation of priors and posteriors:

*Initialise:* `p̂^(0)(y=c) = p_train(y=c)`, and `p̂^(0)(y=c|x) = s_c(x)` (the classifier's raw output).

*E-step* (rescale posteriors by the prior ratio, renormalise):
```
                    [ p̂^(k)(y=c) / p_train(y=c) ] · p_train(y=c|x)
p̂^(k)(y=c|x)  =  ────────────────────────────────────────────────────────
                  Σ_{c'} [ p̂^(k)(y=c') / p_train(y=c') ] · p_train(y=c'|x)
```

*M-step* (re-estimate the prior as the mean adjusted posterior):
```
p̂^(k+1)(y=c) = (1/|U|) · Σ_{x∈U} p̂^(k)(y=c|x)
```

*Iterate* until `max_c |p̂^(k+1)(y=c) - p̂^(k)(y=c)| < ε` (e.g. 1e-6) or a max iteration count.

SLD is a genuine EM: the class labels are the latent variables, the priors are the parameters. **Critical caveat** — Esuli, Molinari & Sebastiani's reassessment (*ACM TOIS* 2021) shows SLD's advantage depends heavily on the underlying classifier being well calibrated, and that it can degrade for large numbers of classes and for extreme shift. For a binary bot/not task with a calibrated classifier it is a strong default.
<https://dl.acm.org/doi/10.1145/3433164>

**HDy — Hellinger-distance distribution matching** (González-Castro, Alaiz-Rodríguez & Alegre, *Information Sciences* 2013). Bin the posterior scores into `b` bins. Let `H_test` be the histogram of `s(x)` over `U`; `H_pos`, `H_neg` the histograms over the positive and negative training instances. Find

```
p̂_HDy = argmin_{α ∈ [0,1]}  HD( H_test ,  α·H_pos + (1-α)·H_neg )
```
with the Hellinger distance between normalised histograms
```
HD(P, Q) = sqrt( Σ_{j=1}^{b} ( sqrt(P_j) - sqrt(Q_j) )^2 )
```
Standard practice: solve by 1-D grid/golden-section search over `α`, repeat for `b ∈ {10, 20, …, 110}`, and take the **median** `α` across bin counts (removes binning sensitivity).

**HDx** — the same idea applied to the *raw feature* distributions rather than posterior scores; extends more naturally to multiclass.

**MS — Median Sweep** (Forman): compute `p̂_ACC` at many decision thresholds `t`, each with its own `tpr(t)`, `fpr(t)`, and take the **median** of the resulting estimates. Related threshold-choice policies: **X** (choose `t` where `fpr = 1 - tpr`), **MAX** (choose `t` maximising `tpr - fpr`), **T50** (choose `t` where `tpr = 0.5`). All are attempts to avoid the small-denominator blow-up in ACC.

**Empirical guidance** (Schumacher, Strohmaier & Lemmerich, comparative evaluation of ~24 methods): for **binary** tasks the best performers are **MS, TSMax, HDy, DyS, and Friedman's method**; for multiclass, **HDx, GPAC, readme, ED, EM, Friedman**. "No single algorithm generally outperforms all competitors." Distribution-shift magnitude drives the ranking; **training-set size has limited impact**; classifier hyperparameter tuning has limited impact.
<https://arxiv.org/abs/2103.03223>

**Tooling.** QuaPy implements CC/ACC/PCC/PACC/EMQ/HDy/MS/X/MAX/T50 plus the evaluation protocols. <https://arxiv.org/abs/2106.11057> · <https://github.com/HLT-ISTI/QuaPy>
Open-access textbook with full derivations: Esuli, Fabris, Moreo & Sebastiani, *Learning to Quantify*, Springer IR Series vol. 47, 2023 — <https://link.springer.com/book/10.1007/978-3-031-20467-8> (OAPEN mirror: <https://library.oapen.org/handle/20.500.12657/60679>)

### 3.3 Evaluation protocol and error measures for prevalence

Do **not** report AUC for a prevalence task. Report:

- **Absolute Error** `AE = |p̂ - p|`
- **Relative Absolute Error** `RAE = |p̂ - p| / p` (with smoothing `(|p̂ - p| + ε)/(p + ε)`, `ε = 1/(2n)`, to handle `p = 0`)
- **Normalised KLD / Pearson divergence** for multiclass.

**Artificial Prevalence Protocol (APP)** — the standard evaluation: from a labelled pool, draw many samples of fixed size at *systematically varied* prevalences (e.g. `p ∈ {0.00, 0.05, …, 1.00}`), run the quantifier on each, and report mean error **per prevalence bin**, plus the worst-case bin. This is exactly how you demonstrate that your prevalence estimator does not collapse in the low-prevalence regime you actually operate in. BotPercent uses the analogous protocol: balanced *and* imbalanced test settings with bot percentages swept **10%–90%**.

### 3.4 BotPercent (Tan et al., arXiv 2302.00381)

The one paper that directly does community-level bot *percentage*.

- **Problem statement:** individual-account detectors are miscalibrated, and miscalibration "prevent[s] trivial aggregation for community-level estimation."
- **Calibration:** **temperature scaling** — a single scalar `T` fitted on a held-out validation set, rescaling logits `z → z/T` before softmax. Post-hoc, cheap, preserves the argmax ranking (so it changes prevalence estimates without changing per-account rankings).
- **Ensemble:** three modalities.
  - *feature-based*: Random Forest + AdaBoost over 12 direct + 14 derived features
  - *text-based*: RoBERTa + T5 over user description + latest 20 tweets
  - *graph-based*: SimpleHGN, HGT, BotRGCN, RGT — **distilled into linear layers** via knowledge distillation for scalability
- **Aggregation formula:**
```
p = (1/𝒟) · Σ_{i=1}^{𝒟}  argmax_{{0,1}} (  Σ_{j ∈ {f,t,g}}  Σ_{k=1}^{K_j}  α_{jk} · p^i_{jk}  )
```
where `𝒟` = number of accounts in the community, `f/t/g` = feature/text/graph modalities, `K_j` = number of sub-models in modality `j`, and `α_{jk}` are **learnable mixture weights optimised by negative log-likelihood on validation data**.
- **Community-level metric:** `|p_𝒰 - p̂_𝒰|` — absolute deviation of the estimated from the true bot percentage. (This is AE from §3.3.)
- Finding: bot presence has a strongly heterogeneous **spatial-temporal distribution** — i.e. community-and-month is the right unit of analysis, which validates the V3 framing.

<https://arxiv.org/abs/2302.00381> · <https://ar5iv.labs.arxiv.org/html/2302.00381>

**Honest assessment:** BotPercent's aggregation is still *calibrated-then-hard-count* — the `argmax` inside the sum means it is fundamentally CC applied to a calibrated ensemble. It fixes the calibration half of the problem, not the `E[p̂_CC] = p·tpr + (1-p)·fpr` half. **The strictly better construction is: calibrate (temperature scaling, as BotPercent does) → then apply ACC/PACC/SLD/HDy on top.** Calibration and quantification are complementary, not substitutes; the relationship is worked out in "On the Interconnections of Calibration, Quantification, and Classifier Accuracy Prediction under Dataset Shift" <https://arxiv.org/abs/2505.11380>.

### 3.5 Uncertainty on the prevalence estimate

Minimum viable: propagate binomial sampling error through ACC by the delta method,

```
Var(p̂_ACC) ≈ Var(p̂_CC) / (tpr - fpr)^2 ,      Var(p̂_CC) = p̂_CC (1 - p̂_CC) / n
```

but this **ignores the estimation error in `tpr` and `fpr`**, which typically dominates (they come from a small labelled validation set). Better: bootstrap the *validation set* jointly with the unlabelled sample, recompute `p̂_ACC` per bootstrap replicate, and take percentile intervals. Formal treatment: "Confidence intervals for class prevalences under prior probability shift" <https://arxiv.org/abs/1906.04119>; see also "Estimating prevalence with precision and accuracy" <https://arxiv.org/abs/2507.06061> and Bayesian treatments <https://arxiv.org/abs/2302.09159>.

**Reporting rule to adopt:** publish `p̂ ± CI` and the value of `tpr - fpr` alongside it. If `tpr - fpr < 0.5`, the interval will be wide enough that the honest conclusion is "cannot distinguish from baseline", and that should be the published verdict.

---

## 4. Rally / brigading / mobilisation event detection

### 4.1 Changepoint detection on activity series

| Method | Detects | Notes |
|---|---|---|
| **CUSUM** | **shift in mean** — sustained level change | Good for regime changes (e.g. a subreddit permanently getting busier). Used with a **sliding window, typically 4 weeks, slid every few days**, to find multiple changepoints. |
| **BOCPD** (Bayesian Online Changepoint Detection) | **sudden short surges** | Bayesian run-length posterior; flags an observation as improbable under the current regime. Best match for rally detection. |
| **PELT** (Pruned Exact Linear Time) | **multiple offline changepoints, exact global optimum** | Dynamic programming with pruning; penalty term controls overfitting; linear time. Best for retrospective monthly segmentation. |

A published social-media practice worth copying directly: run **CUSUM and BOCPD together** and declare a changepoint significant when **either** gives a confidence score **> 0.5** — CUSUM catches baseline shifts, BOCPD catches spikes. <https://arxiv.org/abs/2307.10245> · <https://arxiv.org/abs/2401.06275>

PELT background: <https://arxiv.org/abs/1602.01254> (nonparametric variant); implementations `ruptures` (Python), `changepoint` (R).

**Practical note for count data:** subreddit comment counts are overdispersed counts with strong weekly + diurnal seasonality. Either (a) model with a negative-binomial cost function in PELT, or (b) deseasonalise first (STL / day-of-week + hour-of-day fixed effects) and run changepoint detection on residuals. Running PELT on raw counts will find your weekends.

### 4.2 Kleinberg burst detection — the right primitive for "rally within a month"

Kleinberg's infinite-state automaton is the standard for burst detection on event streams and gives you a **hierarchical, nested** set of bursts with intensity levels, which is exactly the "sudden mobilisation event" object.

Model: while in state `i`, inter-event gaps are drawn from an exponential distribution with expected value proportional to **`s^(-i)`** — higher states = faster events. Transition costs: moving **up** `j - i` states costs `(j - i) · γ · ln n` (where `n` is the number of events); moving **down** is **free**. The optimal state sequence minimising (total transition cost − log-likelihood of the observed gaps) is found by a Viterbi dynamic program. A burst is a maximal interval at or above a given state level.

**Parameters and defaults** (R `bursts::kleinberg`):
- **`s = 2`** — base of the exponential state scaling. Higher = stricter.
- **`γ = 1`** — transition-cost coefficient. "Higher values mean roughly that bursts must be sustained over longer periods of time in order for the algorithm to recognize them."
- Input: `offsets`, a vector of event times.

<https://www.cs.cornell.edu/home/kleinber/bhs.pdf> · <https://link.springer.com/article/10.1023/A:1024940629314> · <https://cran.r-project.org/web/packages/bursts/refman/bursts.html> · Python: `pybursts`

Run it per (subreddit, month) on comment timestamps, and separately on the timestamps of comments from **accounts with no prior history** (§4.5). A high-level burst in the newcomer stream that is not present in the incumbent stream is a very clean rally signature.

### 4.3 Distinguishing organic news-driven spikes from coordinated mobilisation — Crane & Sornette (PNAS 2008)

**This is the single most valuable idea in the section.** The *shape* of the relaxation after a burst encodes its origin. Crane & Sornette fit an epidemic branching model with a power-law memory kernel to daily view counts of ~5M YouTube videos and found relaxation exponents cluster into distinct classes:

| Class | Relaxation exponent (modal) | Signature |
|---|---|---|
| **Exogenous, subcritical** | **≈ 1.4** (`= 1 + θ`) | abrupt jump, fast decay, no precursor. Pure external shock with little social propagation. |
| **Exogenous, critical** | **≈ 0.6** (`= 1 - θ`) | abrupt jump, slow power-law decay. External shock amplified by the social network. |
| **Endogenous, critical** | **≈ 0.2** (`= 1 - 2θ`) | **symmetric power-law growth *before* the peak with the same exponent**, then slow decay. Internally generated cascade. |

All with **θ = 0.4** (the exponent of the waiting-time distribution between cause and action). The **precursory growth** is the diagnostic: the branching model predicts significant power-law growth *before* an endogenous peak, centred on exponent `1 - 2θ`, whereas exogenous shocks are step-like.

<https://www.pnas.org/doi/abs/10.1073/pnas.0803685105> · <https://arxiv.org/abs/0803.2189>

**Operational reading for the rally verdict:**
- **Abrupt onset + no precursor + fast decay (steep exponent)** → likely externally driven (news event). Check GDELT (§5).
- **Abrupt onset + no precursor + *sustained* elevated activity** → external trigger with strong amplification. Could be organic virality *or* a campaign riding a news peg.
- **Gradual precursory build-up then peak** → endogenous. On a subreddit this is either genuine community-driven escalation or a warm-up campaign.
- **Onset that is abrupt in *both* directions — tight arrival cluster then near-silence, with no power-law tail at all** → matches *neither* class. This is the coordinated-mobilisation signature: organic attention always leaves a power-law tail because human waiting times are heavy-tailed; a scripted or scheduled push does not.

Fit: bin post-peak activity, regress `log(activity)` on `log(time since peak)`, report the exponent and the R² of the power-law fit. A **poor power-law fit** is itself the alarm.

### 4.4 Hawkes / self-exciting point processes

Standard machinery for cascade modelling (Rizoiu et al., "A Tutorial on Hawkes Processes for Events in Social Media", WWW'17 tutorial):

**Conditional intensity:**
```
λ(t | H_t) = λ_0(t) + Σ_{i : T_i < t} φ(t - T_i)
```

**Kernels:**
- exponential: `φ(x) = α e^{-δx}`, with `α ≥ 0`, `δ > 0`, `α < δ`
- power-law: `φ(x) = α / (x + δ)^{η+1}`, with `α ≥ 0`, `δ, η > 0`, `α < η δ^η`

**Branching factor** (expected direct offspring per event):
```
n* = ∫_0^∞ φ(τ) dτ
n* = α / δ                (exponential kernel)
n* = α / (η δ^η)          (power-law kernel)
```
`n* < 1` subcritical (cascade terminates); `n* > 1` supercritical (explosive).

**Expected final cascade size** (subcritical, single immigrant):
```
E[total events] = 1 / (1 - n*)
```

**Log-likelihood for MLE:**
```
ℓ(θ) = - ∫_0^T λ(t) dt  +  Σ_{i=1}^{N(T)} log λ(T_i)
```
`O(N²)` naive; `O(N)` recursive for the exponential kernel.

**The decomposition that matters:** Hawkes explicitly separates **immigrant (exogenous) events**, arriving via the background rate `λ_0(t)`, from **offspring (endogenous) events**, spawned through `φ`. So:
- Make `λ_0(t)` a function of the **GDELT news volume series** (§5). Then the fitted background intensity *is* the news-explained component, and the residual/offspring structure is the unexplained mobilisation.
- **Branching ratio `n*` is a direct "how much of this was internal amplification" statistic**, comparable across subreddits and months.
- A **multivariate Hawkes process** with an `N × N` infectivity matrix `Φ_{uv}` over accounts turns coordination into "who reliably excites whom"; strong, persistent, asymmetric off-diagonal blocks are the coordination signal. Applied to disinformation link-sharing at scale by the ANU behavioral-ds group. <https://www.behavioral-ds.science/theme2_content/coordinated_disinfo_hawkes/> · <https://arxiv.org/abs/1708.06401>

**Known failure mode to guard against:** Hawkes processes **overestimate self-excitation when burst-like non-stationarities are present** — i.e. an unmodelled exogenous shock gets absorbed into `n*` and looks like endogenous cascading. This is precisely why `λ_0(t)` must carry the news covariate before you interpret `n*`. <https://arxiv.org/abs/1610.05383>

### 4.5 Outsider influx — the measure the brief asks for

Definitions to take verbatim from Kumar et al. (§2.5), because they are careful:
- **Member of community `C` at day `d`** = made ≥1 comment in `C` in the **30 days prior to `d`**.
- **Exclusive membership**: exclude users who were members of both source and target in that window.
- Ignore all activity within **±3 days** of the event when computing histories (prevents the event contaminating its own baseline).

Derived per-(subreddit, month, or per-thread) metrics:
- **Newcomer share** = fraction of commenters with **zero prior comments** in this subreddit in the trailing 30/90 days. Compare against the subreddit's own trailing-12-month distribution of this quantity, not against an absolute threshold — baseline newcomer rates vary enormously by subreddit.
- **Newcomer comment share** (volume-weighted, not account-weighted) — a rally is better characterised by outsiders' share of *comments* than of *accounts*.
- **Provenance concentration**: for newcomers, the distribution over the subreddits where they *were* active in the prior 30 days. Compute normalised entropy; **low entropy = influx from one place = brigade**. High entropy = organic discovery / front-page exposure. This is the discriminator between "r/X brigaded us" and "we hit r/all".
- **Excess relative to matched threads**: apply the Kumar matched-post design — for each spiking thread, find the same-subreddit thread created closest in time with comparable pre-event size, and compare after/before ratios. The matched thread's ratio is the null (Kumar's was **1.6×**; compute your own per subreddit).
- **Moderator deletion rate** on the thread. Kumar: **25× higher** during negative mobilisation (0.205 vs 0.008). Arctic Shift exposes removed/deleted state; this is close to a free label.
- **Anger/toxicity lift** on the thread relative to matched (Kumar: +44% anger words).

### 4.6 Arrival-burst tightness

Metrics on the inter-arrival times `τ_i` of the participating accounts' first comments in the event:

**Burstiness parameter** (Goh & Barabási, *EPL* 2008):
```
B ≡ (r - 1) / (r + 1) = (σ - ⟨τ⟩) / (σ + ⟨τ⟩) ,     r ≡ σ / ⟨τ⟩
```
`B = -1` perfectly regular (σ = 0); `B = 0` Poisson (σ = ⟨τ⟩); `B → 1` extremely bursty.

**Memory coefficient**:
```
M ≡ (1 / (n_τ - 1)) · Σ_{i=1}^{n_τ - 1} [ (τ_i - m_1)(τ_{i+1} - m_2) ] / (σ_1 σ_2)
```
where `m_1, σ_1` are the mean/SD of the first `n_τ - 1` inter-event times and `m_2, σ_2` of the last `n_τ - 1`. `M > 0`: short gaps follow short gaps (clustered). `M < 0`: alternating.

<https://www.semanticscholar.org/paper/Burstiness-and-memory-in-complex-systems-Goh-Barabasi/17afd11e5320840e4ce74a688c21ac8dc1987350>

**Interpretation.** Organic attention: `B` moderately positive, heavy-tailed inter-arrivals, power-law decay, `M` near 0 to slightly positive. Coordinated push: **very high `B` with a short, tight arrival cluster followed by silence**, and — crucially — **inter-arrival times that are too *regular within* the burst** (a scheduled queue can produce near-uniform gaps, which pushes `B` *down* inside the burst window). So compute `B` at two scales: across the whole month (expect high for a rally) and *within* the burst window (anomalously low = scripted).

Robustness note from the literature: use the **median** inter-arrival time rather than the mean when summarising per-pair timing, since the mean is dominated by outliers.

### 4.7 Composite rally score — suggested construction

For each (subreddit, month), a defensible rally verdict combines:
1. A Kleinberg burst of level ≥ 2 exists in the month's comment stream (**event exists**);
2. and the burst's newcomer comment share exceeds the subreddit's trailing-12-month 95th percentile (**outsiders**);
3. and newcomer provenance entropy is in the bottom decile (**concentrated origin**);
4. and the GDELT-conditioned residual (§5) is significantly positive (**unexplained**);
5. and the post-peak decay is a poor power-law fit / decays faster than exponent 0.2–1.4 (**not organic-shaped**);
6. and the moderator deletion rate is elevated versus matched threads (**contested**).

Report these six as an explicit, auditable checklist per (subreddit, month) rather than collapsing to a single opaque score. Any single one of them is weak; the conjunction is strong, and readers can see which leg is carrying the verdict.

---

## 5. Event conditioning — regressing out real-world news

### 5.1 GDELT DOC 2.0 API as the news covariate

Free, no key, JSON/CSV. Modes:
- **`ArtList`** — matching articles with source country, language, publication date
- **`TimelineVol`** — coverage volume as a **percentage of all global coverage monitored by GDELT** per time step (i.e. already normalised against global news volume — use this to avoid confounding with GDELT's own growing corpus)
- **`TimelineVolRaw`** — raw article counts plus the normalisation denominator
- **`TimelineTone`** — mean tone (−100 to +100) of matching coverage
- **`ToneChart`** — tonal histogram

Query operators: quoted phrases (`"farmers protest"`), boolean `(a OR b)`, `domain:ndtv.com`, `sourcecountry:india`, `sourcelang:hindi`, `theme:PROTEST`.

**Hard constraints:** rolling window of the **last 3 months** by default; minimum granularity **15 minutes**; `STARTDATETIME`/`ENDDATETIME` in `YYYYMMDDHHMMSS`. **The 3-month window is a real problem for a monthly historical pipeline — you must harvest and archive GDELT timelines contemporaneously, or fall back to GDELT's BigQuery / raw file exports for history.**

<https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/> · <https://api.gdeltproject.org/api/v2/doc/doc>

### 5.2 Method 1 — counterfactual forecasting (CausalImpact / BSTS)

Brodersen, Gallusser, Koehler, Remy & Scott, *Annals of Applied Statistics* 9(1), 2015. Fit a **Bayesian structural time-series** model to the pre-event response series with **contemporaneous covariates**, then forecast the counterfactual over the event window and integrate the difference.

Structure: local level/trend + seasonality (weekly, and daily if sub-daily) + **regression on control series** (spike-and-slab prior for automatic variable selection over many candidate controls); posterior inference by MCMC; outputs pointwise and cumulative effect with credible intervals.

**Controls to use for a subreddit-month:**
- GDELT `TimelineVol` for the topic keywords (the news signal)
- comment volume in **other subreddits not expected to be affected** (this is the synthetic-control donor pool — e.g. r/india's spike controlled by r/CricketShitpost, r/bangalore, etc.)
- Reddit-wide comment volume (platform-level trend)

**Key assumption, and its failure mode:** the control series must **not themselves be affected by the intervention**. If a coordinated campaign hits several of your 45 subreddits simultaneously, using them as mutual controls will *cancel the effect you are trying to measure*. Mitigation: select donors by pre-period correlation *and* verify that donors show no burst; or use a hold-out set of topically unrelated subreddits as the fixed donor pool.

<https://arxiv.org/abs/1506.00356> · <https://google.github.io/CausalImpact/CausalImpact.html> · Python: `tfcausalimpact`, `pycausalimpact`

### 5.3 Method 2 — count regression with news covariates (simplest, recommended default)

Model daily comment counts `Y_{s,t}` for subreddit `s` on day `t`:

```
Y_{s,t} ~ NegBin(μ_{s,t}, θ_s)

log μ_{s,t} = β_0
            + f_s(t)                              # smooth trend (spline / RW)
            + Σ_k γ_k · dow_k(t)                  # day-of-week
            + Σ_j δ_j · log(1 + GDELT_{j,t-l})    # news volume, topic j, lags l ∈ {0,1,2}
            + η · GDELTtone_{t}                   # optional tone
            + offset( log(ReddIt_wide_volume_t) ) # platform-level exposure
```

Fit on a long baseline (12–24 months), then define the **unexplained mobilisation** as the standardised Pearson residual:

```
r_{s,t} = (Y_{s,t} - μ̂_{s,t}) / sqrt( μ̂_{s,t} + μ̂_{s,t}² / θ̂_s )
```

Flag days with `r_{s,t}` above a high quantile of the subreddit's own residual distribution. Then **run the changepoint/burst detectors of §4 on the residual series `r_{s,t}` rather than on raw counts** — this is the cleanest formulation of "flag only unexplained mobilisation". Negative binomial (rather than Poisson) is essential: comment counts are heavily overdispersed.

Bayesian deep/GLM variants of exactly this on GDELT panels — NB2 and zero-inflated NB2 likelihood heads with posterior predictive simulation giving predictive quantiles and **right-tail probabilities as anomaly scores**, plus a lagged cross-series GLM with shrinkage priors for spillover attribution: <https://arxiv.org/abs/2603.25970>

### 5.4 Method 3 — GDELT's own baseline comparison

GDELT's timeseries anomaly APIs use a **`forecastHistory` parameter** specifying how far into the past to look to establish the baseline against which the search period is compared. A cheap operational analogue: for each (subreddit, day), compute `z = (Y_t - mean(Y_{t-1..t-56})) / sd(Y_{t-1..t-56})` on deseasonalised counts, and require the *Reddit* z-score to substantially exceed the *GDELT topic* z-score for the same window. "Reddit spiked much harder than the news did" is a simple, explainable statement.

### 5.5 What to report

Do not report "was there a news event". Report the **ratio of observed to news-explained activity**, with an interval:
```
Excess ratio = Y_observed / μ̂_news-explained      (with credible interval)
```
An excess ratio of 1.2 with a CI spanning 1 is not a rally. An excess ratio of 6 with a tight CI, concentrated in newcomers from one source subreddit, is.

---

## 6. Evaluation without ground truth

### 6.1 What the field actually does (survey synthesis)

Three strategies, in descending order of rigour:

1. **Public ground-truth proxies.** Platform disclosure archives: **Twitter's Information Operations archive** (Oct 2018 onward; 46 campaigns), Meta's and Google's CIB takedown disclosures, and **Reddit's own 2017 disclosure of 944 Russian-linked accounts** (335 of which TrollMagnifier uses). The survey is explicit that these are **platform-curated samples, not comprehensive ground truth**, "driven primarily by pressing practical regulation needs and by immediate contingencies, rather than by methodological rigor". Note also that platforms disagree about what counts: Reddit has published Spamouflage-related data while **"refraining from considering these activities as rule violations."**
2. **Simulation-based validation.** Evaluate in controlled synthetic environments where coordination is explicitly injected and therefore known. Lets you measure recall as a function of campaign size, coordination tightness, and cover traffic. Under-used and cheap.
3. **Characterisation-as-validation.** The most common and the weakest: inspect the detected groups a posteriori for consistency with known coordination properties — account activity patterns, temporal synchrony, content/socio-linguistic similarity, network structure, bot scores. The survey: characterisation output "can also be leveraged to validate the output of the detection task, as in those frequent cases when a ground-truth of coordinated users is unavailable" — but this "lacks objective ground truth and limits reproducibility and comparability across methods."

<https://arxiv.org/abs/2408.01257>

### 6.2 What they report instead of AUC

- **Precision@K** — of the top K flagged, how many survive manual review. The honest metric when analysts can only review a bounded number of alerts.
- **Precision vs. support curves** — Pacheco et al. plot precision (from manual annotation) as a function of the edge-weight/support threshold, so the reader picks their own operating point.
- **Recall against a disclosed campaign, with an explicit FPR on matched controls** — Schoch et al.: **74% recall, ~1% FPR** on activity-matched regular users. This is the gold standard when a disclosure exists.
- **Enrichment/lift over matched controls** rather than absolute rates.
- **Multi-signal corroboration counts** — "N of the detected accounts satisfy ≥1 of these 4 independent suspicion criteria."
- **Robustness/sensitivity curves** across the free parameters (time window, threshold, α) — the *shape* is the evidence, not one point.

### 6.3 TrollMagnifier's validation battery (arXiv 2112.00443) — the template for Reddit

Because this is the Reddit-specific seed-and-expand paper and its validation design is the most directly copyable, it is worth spelling out fully. **Note: TrollMagnifier is *single-stage*, not iterative** — it pre-filters candidates, extracts 9 features, classifies once. The "expansion" is the pre-filter + classify step, not label propagation.

**Seed set.** 335 known Russian-sponsored troll accounts identified by Reddit, active 2015–2018.

**Candidate pre-filter** (the "magnification" step): 12,143 accounts posting submissions with the **same title** as known trolls + 42,001 accounts **commenting on known-troll submissions**, overlap 381 → **53,763 candidates**.

**The 9 features** (note that 6 of 9 are *relational to the seed set*, and all are **normalised by the account's own activity** — the volume control is built into the feature definition):
1. total comments
2. total submissions
3. account age (years since first post/comment)
4. fraction of the user's submissions whose **title matches** a known-troll submission title
5. fraction of the user's comments on **submissions that troll accounts also commented on**
6. fraction of the user's comments on **submissions authored by troll accounts**
7. fraction of the user's comments that are **direct (top-level) replies** on troll-authored submissions
8. fraction of the user's comments **replying to troll comments**
9. fraction of the user's comments **replying to troll comments within troll-authored submissions**

**Negative class construction — the important design choice.** 335 random accounts sampled **from the 53,763 pre-filtered candidates**, not from Reddit at large, explicitly "to avoid over-fitting": otherwise the classifier trivially learns "interacted with a troll at all" rather than the subtler distinction. Suspended/deleted accounts excluded from negatives.

**Classifier.** Random Forest. 10-fold CV on the training set: **precision 97.8%, recall 97.7%, F1 97.8%, accuracy 97.8%**. Compared against KNN (F1 91.8%), Linear SVM (95.6%), Decision Tree (97.3%). *These are training-set CV numbers on a balanced 335/335 set and should not be read as field performance.*

**Output: 1,248 detected accounts** from 53,763 candidates (a ~3.7× expansion of the 335 seeds).

**Validation of the 1,248 — four independent account-level checks:**
| Check | Method | Result |
|---|---|---|
| Account status | query `reddit.com/u/<name>.json`; 403 = suspended, 404 = deleted | **298 / 1,248** suspended or deleted |
| Deleted content | diff Pushshift historical vs. live PRAW (Apr 14 2021); only counted when PRAW returns <1,000 elements (API cap) | **304 / 1,248** deleted ≥1 comment or submission |
| Creation-date clustering | extract "Cake Day" from profile; cluster by date against known-troll creation clusters | **66 / 1,248** in troll creation clusters |
| Topic overlap | TF-IDF top-10 keywords of the known-troll corpus (*people, money, crypto, bitcoin, country, police, black, news, cop, trump*); check if candidate posted any | **359 / 1,248** |

**Combined: 824 / 1,248 = 66%** satisfy at least one criterion.

**Two group-level checks:**
- **Language similarity:** train word2vec (**CBOW, window = 20**) on three corpora (known trolls / detected trolls / non-trolls); for each keyword compare cosine similarity of the top-100 nearest words; z-test. Significant at p<0.05 for *people* (z = 7.55), *money* (z = 4.38), *crypto* (z = 2.12).
- **Time-series correlation:** Pearson correlation between detected-troll and known-troll daily activity. Submissions: **0.5** (detected–known) vs **0.068** (undetected–known). Comments: **0.553** vs **0.334**. Lag: −23 days vs −99 days for submissions.

**False-negative estimate:** two coders manually annotated 20 random non-detected accounts, found 2 suspicious → **~10% FNR** (an extremely small sample; treat as indicative only).

<https://arxiv.org/abs/2112.00443> · <https://ar5iv.labs.arxiv.org/html/2112.00443>

**Transferability caveat for V3:** TrollMagnifier's power comes entirely from having a *disclosed seed set*. Without Indian-context seed accounts, features 4–9 have nothing to anchor to. Options: (i) use Reddit admin-suspended accounts in your 45 subreddits as a noisy seed set (suspension is observable via the 403 check, and is what TrollMagnifier itself used as validation); (ii) bootstrap seeds from the highest-confidence output of the null-model-validated coordination graph (§2), then expand — but then the validation must be fully independent of the coordination signal, or you are just measuring your own detector twice.

### 6.4 The pair-level vs. account-level lesson (Kumar et al., WWW 2017, "An Army of Me")

The most useful single data point for calibrating expectations on Reddit-like data.

Sockpuppet identification heuristic (their ground truth): accounts posting from the **same IP within a 15-minute window in the same discussion**, repeated across **≥3 different discussions**; excluding the top-5% most-used IPs and accounts using many IPs.

Co-occurrence statistics: sockpuppet pairs shared **6.57 sub-discussions** vs **0.33** for random ordinary pairs (p<0.001); posted in the same discussion within 15 minutes **7.8** times vs **4.28** (p<0.001). *Note how much weaker the temporal signal is than the co-participation signal — 1.8× vs 20×.*

Features: activity (reply-network clustering, reciprocity, post counts, reply proportions, temporal gaps, tenure, common sub-discussions), community (upvote fractions, report/deletion rates, blocking), post (ARI readability, LIWC, sentiment, length, agreement).

**Results:**
- **Pair classification** ("are these two accounts the same person?"): **ROC AUC 0.91** (activity features alone: 0.86).
- **Individual classification** ("is this account a sockpuppet?"): **ROC AUC 0.68** (activity features alone: 0.59).

**Design implication for V3:** *relational* questions are far more answerable than *individual* ones on this kind of data. Prioritise pair/group-level coordination claims (B) over per-account inauthenticity claims (A), and be correspondingly humble about the prevalence number.

<https://arxiv.org/abs/1703.07355> · <https://cs.stanford.edu/~srijan/pubs/sockpuppets-www2017.pdf>

### 6.5 Recommended evaluation protocol for this project

1. **Always publish the permuted-null result next to the real result.** Same pipeline, timestamp-permuted / degree-preserving-randomised input. This is your false-positive floor and it costs one extra run.
2. **Sensitivity curves, not points.** Vary time window (e.g. Weber & Neumann's 15/60/360/1440 min), FDR α, min-activity filter; plot flagged-fraction vs. parameter. A real signal is a plateau; an artefact is monotone in the threshold.
3. **Precision@K by manual review**, on a stratified sample, with two coders and reported inter-rater agreement (Kumar's MTurk step reached >0.95 on a binary sentiment task; expect worse on coordination).
4. **Corroboration counts** with *independent* signals not used by the detector: admin suspension (403), moderator removal, account deletion, creation-date clustering, cross-subreddit footprint.
5. **Synthetic injection.** Insert simulated coordinated groups of known size and tightness into a real subreddit-month and measure recall as a function of group size and window. This is the only way to state "we can detect campaigns of ≥N accounts."
6. **Report coordination and harm separately**, per the survey's insistence. A tightly coordinated fan community and a state campaign look identical to a co-action detector.
7. **Never report AUC for prevalence** (§3.3) and never report a prevalence point estimate without `tpr - fpr` and an interval.
8. **Treat outputs as leads, not verdicts** (the Synchronized Action Framework's "responsible detection" point) — especially for named accounts.

---

## 7. Cross-community / cross-subreddit influence

- **Kumar et al. (WWW 2018)** is the primary reference: directed source→target mobilisation via hyperlinks, with the matched null of §2.5. Concentration finding to reuse as a prior: **<0.1% of communities initiate 38%** and **<1% initiate 74%** of negative mobilisations. Conflicts occur between **topically similar** communities (TF-IDF sim 0.51 vs 0.34), and produce **echo-chamber structure** in the target thread (attackers reply to attackers; quantified by A-PageRank/D-PageRank on the thread's directed reply network). Long-term "colonisation" effect: defenders reduce participation, attackers increase. Prediction AUC 0.76 (socially-primed LSTM) vs 0.67 (expert features). Dataset and code: <https://snap.stanford.edu/conflict/>
- **Directed flow without hyperlinks.** V3's cross-subreddit histories let you compute, for each ordered pair (A, B) and month, the number of accounts whose *first-ever* comment in B falls in the month and who were active in A in the preceding 30 days. Normalise by the expected flow under a null that preserves each account's overall subreddit-visiting propensity and each subreddit's newcomer intake — this is again a bipartite (accounts × subreddits) configuration model, so §2.2/§2.3 apply unchanged with subreddits as the second layer.
- **Cross-Subreddit Behavior as Open-Source Indicators of Coordinated Influence: A Case Study of r/Sino & r/China** (2025) — the closest published Reddit analogue. Builds behavioural profiles of **dual-subreddit users** across ideologically opposed communities, combining topic modelling and sentiment against community baselines, lexical diversity, language consistency, account age, posting frequency, karma distribution, cross-community participation overlap, and position in the subreddit co-participation network; flags users showing **multiple simultaneous behavioural anomalies**. Framed via social-cybersecurity theory around influence beneath overt bot activity (tone modulation, narrative blending, identity-driven engagement). Methodologically modest — no null model, no ground truth — but it is the right *shape* for the Indian-subreddit setting and a good citation for framing. <https://arxiv.org/abs/2507.16857>
- **Unmasking Brigading: A Network-Based Framework for Detecting Coordinated Manipulation on Reddit** (IEEE TEMSMET 2025). Models subreddit interactions as **hyperlink graphs augmented with temporal and sentiment metadata**; detects downvote bursts, sentiment alignment, subreddit centrality; clusters with **Louvain, Girvan-Newman, DBSCAN**. Caveat: its core signal is **vote-based**, which is unavailable to you. Cite for framing only. <https://ieeexplore.ieee.org/document/11467228/>
- **Coordinated Information Dissemination on Telegram and Reddit** (2026) — cross-platform, and the source of the volume-preserving timestamp-permutation negative control described in §2.4. <https://arxiv.org/abs/2602.13333>
- **Note on a stated Reddit-specific limitation** found in the recent literature: cross-community near-duplicate synchronisation under short shared temporal windows is **not directly feasible on Reddit** due to structural sparsity and asynchronous posting. This is why the Twitter-style "same content within 10 seconds" transplant fails, and why the co-participation + statistical-validation route (§2) is the right one here.

---

## 8. Concrete recommendations for V3

**(B) Coordination — the core build.**
1. Per (subreddit, month), build the binary incidence matrix `M` = accounts × threads, restricted to accounts with ≥3 comments.
2. Compute all pairwise co-occurrence counts `V_{rr'}`.
3. Validate with **SDSM** (BiCM cellwise probabilities + Poisson-Binomial p-values) via the `backbone` package or a direct `bicm` implementation; **BH-FDR at α = 0.01**. Confirm the top pairs with **FDSM/fastball** Monte Carlo.
4. Repeat on the refined incidence matrices: accounts × parent-comment-id, and accounts × near-duplicate-text-cluster. Take the **union** of validated edges as a multiplex, weighting by which layers validated (Weber & Neumann's `w(e) = Σ_c w_c(e)`).
5. Community-detect (Leiden) on the validated graph; report per-community size, persistence across months, `edge_symmetry_score`-style leader/follower asymmetry, and internal/external interaction ratios.
6. The (subreddit, month) coordination verdict = size and density of validated communities relative to (a) the same subreddit's trailing 12 months, and (b) the permuted-null run.

**(A) Prevalence — be conservative.**
7. Temperature-scale the per-account model. Then apply **ACC** (and **SLD** and **HDy** as cross-checks; report the spread across methods as an honesty signal). Estimate `tpr`/`fpr` by CV on a hand-labelled sample **drawn from these subreddits**. Bootstrap for intervals. Publish `tpr - fpr` next to every prevalence figure.
8. Validate the quantifier under the **Artificial Prevalence Protocol** across `p ∈ {0.01, 0.05, 0.10, …}` — especially the low-prevalence regime — and report AE/RAE per bin.

**(C) Rally.**
9. Fit the NegBin baseline with GDELT covariates (§5.3); run **Kleinberg** (`s = 2`, `γ = 1`) and **BOCPD/PELT** on the **residual** series.
10. For each detected burst compute: newcomer comment share vs trailing-12-month distribution; newcomer provenance entropy; matched-thread excess ratio (Kumar design); post-peak power-law exponent and fit quality (Crane–Sornette classes: ≈1.4 / ≈0.6 / ≈0.2); burstiness `B` at month and within-burst scales; moderator deletion lift.
11. Publish the six-part checklist of §4.7 rather than a single score.

**Throughout.**
12. Every run also executes on volume-preserving permuted data; both numbers go in the report.
13. Coordination, inauthenticity, and harm are three separate axes and are reported separately (Nizzoli et al.: coordination and automation were *uncorrelated*).

---

## 9. Sources

### Coordination detection — frameworks and tools
- Pacheco, Hui, Torres-Lugo, Truong, Flammini & Menczer (2021), *Uncovering Coordinated Networks on Social Media: Methods and Case Studies*, ICWSM — <https://ojs.aaai.org/index.php/ICWSM/article/view/18075> · preprint <https://arxiv.org/abs/2001.05658>
- Giglietto, Righetti, Rossi & Marino (2020), *It takes a village to manipulate the media: coordinated link sharing behavior during 2018 and 2019 Italian elections*, Information, Communication & Society 23(6) — <https://www.tandfonline.com/doi/abs/10.1080/1369118X.2020.1739732>
- CooRnet R package (source for `estimate_coord_interval`, `get_coord_shares`) — <https://github.com/fabiogiglietto/CooRnet/> · <https://coornet.org/> · Python port <https://github.com/UPB-SS1/PyCooRnet>
- Righetti & Giglietto (2025), *CooRTweet: A Generalized R Software for Coordinated Network Detection*, Computational Communication Research — <https://www.aup-online.com/content/journals/10.5117/CCR2025.1.7.RIGH> · vignette <https://cran.r-project.org/web/packages/CooRTweet/vignettes/vignette.html> · <https://github.com/nicolarighetti/CooRTweet>
- Nizzoli, Tardelli, Avvenuti, Cresci & Tesconi (2021), *Coordinated Behavior on Social Media in 2019 UK General Election*, ICWSM — <https://ojs.aaai.org/index.php/ICWSM/article/view/18074> · <https://ar5iv.labs.arxiv.org/html/2008.08370> · data <https://zenodo.org/records/4647893>
- Weber & Neumann (2021), *Amplifying influence through coordinated behaviour in social networks*, SNAM 11:111 — <https://link.springer.com/article/10.1007/s13278-021-00815-2> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC8557266>
- Weber & Neumann (2021), *Who's in the Gang? Revealing Coordinating Communities in Social Media* — <https://arxiv.org/abs/2010.08180>
- Magelinski, Ng & Carley (2022), *A Synchronized Action Framework for Detection of Coordination on Social Media*, J. Online Trust & Safety — <https://arxiv.org/abs/2105.07454> · <https://tsjournal.org/index.php/jots/article/view/30>
- Schoch, Keller, Stier & Yang (2022), *Coordination patterns reveal online political astroturfing across the world*, Scientific Reports — <https://www.nature.com/articles/s41598-022-08404-9> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC8930979/>
- Tardelli et al. (2024–25), *Detection and Characterization of Coordinated Online Behavior: A Survey* — <https://arxiv.org/abs/2408.01257> · <https://arxiv.org/html/2408.01257v2>
- Tardelli, Nizzoli, Tesconi, Conti, Nakov, Da San Martino & Cresci (2024), *Temporal Dynamics of Coordinated Online Behavior: Stability, Archetypes, and Influence*, PNAS 121 — <https://arxiv.org/abs/2301.06774> · <https://www.pnas.org/doi/10.1073/pnas.2307038121>
- Cima et al. (2025), *Coordinated link sharing on Facebook* (threshold calibration & sensitivity), Scientific Reports — <https://www.nature.com/articles/s41598-025-00233-w>
- Luceri et al. (2023), *Coordinated Information Campaigns on Social Media: A Multifaceted Framework for Detection and Analysis* — <https://arxiv.org/abs/2309.12729>
- Iannucci, Muratore, Matakos & Kivelä (2025), *Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis* — <https://arxiv.org/abs/2512.19677>
- *Multimodal Coordinated Online Behavior: Trade-offs and Strategies* (2025) — <https://arxiv.org/abs/2507.12108>
- *Towards Detecting Inauthentic Coordination in Twitter Likes Data* — <https://arxiv.org/abs/2305.07384>
- *Unsupervised detection of coordinated information operations in the wild* — <https://arxiv.org/abs/2401.06205>

### Null models, projection validation, backboning
- Tumminello, Miccichè, Lillo, Piilo & Mantegna (2011), *Statistically Validated Networks in Bipartite Complex Systems*, PLOS ONE 6(3):e17994 — <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017994> · <https://ar5iv.labs.arxiv.org/html/1008.1414>
- Saracco, Straka, Di Clemente, Gabrielli, Caldarelli & Squartini (2017), *Inferring monopartite projections of bipartite networks: an entropy-based approach*, New J. Phys. 19:053022 — <https://iopscience.iop.org/article/10.1088/1367-2630/aa6b38>
- *Meta-validation of bipartite network projections*, Communications Physics (2022) — <https://www.nature.com/articles/s42005-022-00856-9>
- `bicm` Python package — <https://github.com/tsakim/bicm> · <https://bicm.readthedocs.io/>
- Neal (2025), *Backbone 3.0: An R package for extracting network backbones*, PLOS ONE — <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0349258> · manual <https://cran.r-project.org/web/packages/backbone/backbone.pdf> · `sdsm` reference <https://search.r-project.org/CRAN/refmans/backbone/html/sdsm.html>
- Neal (2023), *Stochastic Degree Sequence Model with Edge Constraints (SDSM-EC)* — <https://arxiv.org/abs/2307.12828>
- Serrano, Boguñá & Vespignani (2009), *Extracting the multiscale backbone of complex weighted networks*, PNAS 106(16) — <https://en.wikipedia.org/wiki/Disparity_filter_algorithm_of_weighted_network> · Python <https://github.com/DerwenAI/disparity_filter>
- Kumar, Hamilton, Leskovec & Jurafsky (2018), *Community Interaction and Conflict on the Web*, WWW — <https://dl.acm.org/doi/10.1145/3178876.3186141> · <https://arxiv.org/abs/1803.03697> · <https://cs.stanford.edu/~srijan/pubs/conflict-paper-www18.pdf> · project <https://snap.stanford.edu/conflict/>
- *Temporal patterns of reciprocity in communication networks* (timestamp-shuffling nulls, NTS) — <https://arxiv.org/abs/2207.03910>

### Seed-and-expand / semi-supervised
- Saeed, Blackburn, De Cristofaro, Zannettou & Stringhini (2022), *TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit*, IEEE S&P — <https://arxiv.org/abs/2112.00443> · <https://ar5iv.labs.arxiv.org/html/2112.00443>
- Kumar, Cheng, Leskovec & Subrahmanian (2017), *An Army of Me: Sockpuppets in Online Discussion Communities*, WWW — <https://arxiv.org/abs/1703.07355> · <https://cs.stanford.edu/~srijan/pubs/sockpuppets-www2017.pdf>
- Pote, Elmas, Flammini & Menczer (2025), *Coordinated Reply Attacks in Influence Operations: Characterization and Detection*, ICWSM — <https://arxiv.org/abs/2410.19272> · <https://ojs.aaai.org/index.php/ICWSM/article/view/35889>

### Prevalence estimation / quantification learning
- Tan, Feng, Zhang, Tsvetkov & Luo (2023), *BotPercent: Estimating Bot Populations in Twitter Communities* — <https://arxiv.org/abs/2302.00381> · <https://ar5iv.labs.arxiv.org/html/2302.00381>
- Esuli, Fabris, Moreo & Sebastiani (2023), *Learning to Quantify*, Springer IR Series 47 (open access) — <https://link.springer.com/book/10.1007/978-3-031-20467-8> · <https://library.oapen.org/handle/20.500.12657/60679>
- Saerens, Latinne & Decaestecker (2002), *Adjusting the outputs of a classifier to new a priori probabilities*, Neural Computation 14(1) — <https://www.researchgate.net/publication/221344970>
- Esuli, Molinari & Sebastiani (2021), *A Critical Reassessment of the Saerens-Latinne-Decaestecker Algorithm for Posterior Probability Adjustment*, ACM TOIS — <https://dl.acm.org/doi/10.1145/3433164>
- González-Castro, Alaiz-Rodríguez & Alegre (2013), *Class distribution estimation based on the Hellinger distance* (HDx/HDy), Information Sciences — <https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069>
- Schumacher, Strohmaier & Lemmerich (2021), *A Comparative Evaluation of Quantification Methods* — <https://arxiv.org/abs/2103.03223>
- Moreo, Esuli & Sebastiani (2021), *QuaPy: A Python-Based Framework for Quantification* — <https://arxiv.org/abs/2106.11057> · <https://github.com/HLT-ISTI/QuaPy>
- *On the Interconnections of Calibration, Quantification, and Classifier Accuracy Prediction under Dataset Shift* — <https://arxiv.org/abs/2505.11380>
- *Confidence intervals for class prevalences under prior probability shift* — <https://arxiv.org/abs/1906.04119>
- *Estimating prevalence with precision and accuracy* — <https://arxiv.org/abs/2507.06061>
- *Bayesian Quantification with Black-Box Estimators* — <https://arxiv.org/abs/2302.09159>
- Gallwitz & Kreil (2022), *Investigating the Validity of Botometer-based Social Bot Studies* — <https://arxiv.org/abs/2207.11474>
- Gallwitz & Kreil (2021), *The Rise and Fall of 'Social Bot' Research* — <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3814191>
- Quantification (machine learning), Wikipedia — <https://en.wikipedia.org/wiki/Quantification_(machine_learning)>

### Bursts, changepoints, point processes
- Kleinberg (2003), *Bursty and Hierarchical Structure in Streams*, DMKD 7:373–397 — <https://www.cs.cornell.edu/home/kleinber/bhs.pdf> · <https://link.springer.com/article/10.1023/A:1024940629314> · R `bursts` <https://cran.r-project.org/web/packages/bursts/refman/bursts.html>
- Crane & Sornette (2008), *Robust dynamic classes revealed by measuring the response function of a social system*, PNAS 105(41):15649 — <https://www.pnas.org/doi/abs/10.1073/pnas.0803685105> · <https://arxiv.org/abs/0803.2189>
- Goh & Barabási (2008), *Burstiness and memory in complex systems*, EPL 81:48002 — <https://www.semanticscholar.org/paper/17afd11e5320840e4ce74a688c21ac8dc1987350>
- Rizoiu, Lee, Mishra & Xie (2017), *A Tutorial on Hawkes Processes for Events in Social Media* — <https://arxiv.org/abs/1708.06401>
- Rizoiu et al. (2017), *Expecting to be HIP: Hawkes Intensity Processes for Social Media Popularity*, WWW — behavioral-ds coordinated-disinformation project page <https://www.behavioral-ds.science/theme2_content/coordinated_disinfo_hawkes/>
- *Detection of intensity bursts using Hawkes processes* (self-excitation overestimation under non-stationarity) — <https://arxiv.org/abs/1610.05383>
- Haque et al. (2016), *A computationally efficient nonparametric approach for changepoint detection* (PELT family) — <https://arxiv.org/abs/1602.01254>
- *Measuring Online Emotional Reactions to Events* (CUSUM + BOCPD combination, 4-week sliding window, 0.5 confidence threshold) — <https://arxiv.org/abs/2307.10245>
- *The Pulse of Mood Online* — <https://arxiv.org/abs/2401.06275>

### Event conditioning
- Brodersen, Gallusser, Koehler, Remy & Scott (2015), *Inferring causal impact using Bayesian structural time-series models*, Ann. Appl. Stat. 9(1) — <https://arxiv.org/abs/1506.00356> · <https://google.github.io/CausalImpact/CausalImpact.html>
- GDELT DOC 2.0 API — <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/> · endpoint <https://api.gdeltproject.org/api/v2/doc/doc>
- *Bayesian Deep Count Regression and Anomaly Detection: Evidence from GDELT Event Panels* — <https://arxiv.org/abs/2603.25970>

### Reddit-specific
- TrollMagnifier — <https://arxiv.org/abs/2112.00443>
- Kumar et al., *Community Interaction and Conflict on the Web* — <https://arxiv.org/abs/1803.03697>
- *Cross-Subreddit Behavior as Open-Source Indicators of Coordinated Influence: A Case Study of r/Sino & r/China* (2025) — <https://arxiv.org/abs/2507.16857>
- *Unmasking Brigading: A Network-Based Framework for Detecting Coordinated Manipulation on Reddit*, IEEE TEMSMET 2025 — <https://ieeexplore.ieee.org/document/11467228/>
- *Coordinated Information Dissemination on Telegram and Reddit During Political Turbulence* (2026) — <https://arxiv.org/abs/2602.13333>
- *Cyberbullying or just Sarcasm? Unmasking Coordinated Networks on Reddit* — <https://arxiv.org/abs/2410.20170>
- *The anatomy of Reddit: An overview of academic research* — <https://arxiv.org/abs/1810.10881>
