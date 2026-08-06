# Account-Level Inauthentic-Activity Detection: Literature Review for V3

**Scope:** state of the art in account-level bot / automation / inauthentic-account detection, with emphasis on feature engineering and *honestly reported* performance. Written to feed an implementation spec for a Reddit-based system with full comment histories (text, timestamps, `parent_id`, reply graph) via Arctic Shift.

**Date of review:** August 2026. Publication years are flagged throughout; the field's assumptions changed materially after ~2023 (LLM-generated content) and after Reddit's 2023 API lockdown.

---

## 0. Executive summary — the five things that matter most

1. **The V2 plateau at ROC-AUC 0.663 is exactly where the literature says an "account-on-paper" feature set lands on a hard label.** Kumar et al. (WWW 2017) got **AUC 0.68** for "is this individual account a sockpuppet?" using activity + community + linguistic features on nine discussion communities — with the classes *matched on activity volume*. That is the canonical account-level ceiling for a genuinely hard target. Your 0.663 is not a bug.

2. **AUC 0.90+ is achievable, but only for three specific target reframings**, none of which is "account is inauthentic, decided per-account, in the wild":
   - **(a) Self-declared / structurally obvious automation** (bot-flair, `_bot` handles, template output): F1 0.90–0.99 routinely (Cresci-15/17, midterm-18, botwiki: F1 96–99). This is nearly a lookup problem.
   - **(b) Pairwise / group-level coordination**, i.e. "do these two accounts share an operator?" or "does this cluster act as one?": **AUC 0.90–0.94** (Kumar et al. pair task AUC **0.91**; Luceri et al. IO detection on fused similarity networks **AUC 0.94, precision 0.96**).
   - **(c) A narrow, mechanically-defined attack signature** (e.g. post-then-delete within 10 min + lexicon): **F1 0.99** (Elmas et al., ephemeral astroturfing).
   The general per-account task sits at **0.60–0.75 AUC / F1 0.35–0.60** on realistic, large, low-prevalence data.

3. **The single most sobering benchmark is TwiBot-22** (1,000,000 users, 14.0% bots, NeurIPS 2022 D&B). Across **35 re-implemented detectors**, the best **F1 is 58.7** and the best accuracy 79.7. Botometer — the field's reference system, >1,000 features — scores **accuracy 49.9, F1 42.8**. Metadata-only random forests (SGBot) drop to **F1 36.6**. The same detectors score F1 96–99 on Cresci-15. *The difference is dataset realism, not model quality.*

4. **Behavioural/temporal features are the highest-yield addition you can make** — but the gain comes from **sequence and session structure**, not from single scalar summaries. Pozzana & Ferrara (2020) show session-position dynamics take AUC from 0.83 → **0.97**. BLOC (Nwala et al., EPJ DS 2023) reaches **F1 0.892 with 197 features** vs Botometer-V4's 0.921 with 1,160 — i.e. a symbolic encoding of action+pause sequences matches a 1,000-feature commercial system.

5. **On Reddit specifically, transferred Twitter detectors collapse.** BotBuster-For-Everyone (2024) reports **34.68% accuracy / macro-F1 33.86 on its Reddit set** — worse than chance on a 75%-bot dataset. Even BotBuster's own Reddit result, on the *easiest possible* label (top-500 crowd-nominated "bad bots" vs hand-verified humans from five thoughtful subreddits), is **F1 60.04–69.77**. There is no published Reddit account-level detector at F1 > 0.75 on non-trivial labels.

**Recommendation in one line:** stop optimising a per-account binary against an enforcement proxy; build V3 as (i) a per-account *automation/regularity* score from temporal-sequence features, and (ii) a separate *pair/cluster coordination* layer, which is where the 0.90+ signal demonstrably lives.

---

## 1. Feature Catalogue

Notation: an account `a` has comments `c_1 … c_n` sorted by `created_utc`, with timestamps `t_1 … t_n`, inter-event times `τ_i = t_{i+1} − t_i` for `i = 1 … n−1`, mean `m_τ`, standard deviation `σ_τ`.

### 1.1 Inter-event-time distribution features

#### 1.1.1 Burstiness parameter B (Goh & Barabási, EPL 2008)

```
B = (σ_τ − m_τ) / (σ_τ + m_τ)          B ∈ [−1, +1]
```
- `B = −1`: perfectly periodic (σ=0) — **the cron signature**.
- `B = 0`: Poisson / memoryless.
- `B → +1`: heavy-tailed, bursty — normal human behaviour.

**Critical implementation caveat.** B is strongly biased by `n`: for finite sequences `max(σ_τ) ≈ m_τ √(n−1)`, so B is compressed toward −1 for short histories. Since your accounts will have wildly varying comment counts, **use the finite-size-corrected form** of Kim & Jo (Phys. Rev. E 94, 032311, 2016):

```
r  = σ_τ / m_τ
A_n(r) = ( √(n+1)·r − √(n−1) ) / ( (√(n+1) − 2)·r + √(n−1) )
```
`A_n ∈ [−1, 1]` and is free of the finite-size artefact. Using raw B across accounts with n ranging 10 → 10,000 will manufacture a spurious correlation between activity volume and "botness". This is a concrete, cheap correctness win over most published work.
- Goh & Barabási: https://arxiv.org/abs/physics/0610233
- Kim & Jo: https://arxiv.org/abs/1604.01125

**Discriminative power reported.** Bot accounts "exhibit highly automated features with large burstiness parameters and small time-interval entropy values" in the microblog-temporal literature; B is never reported as a strong standalone discriminator (it is one of dozens of features). Treat it as a **component**, not a detector.

#### 1.1.2 Memory coefficient M (Goh & Barabási 2008)

```
M = (1/(n−2)) · Σ_{i=1}^{n−2} (τ_i − m_1)(τ_{i+1} − m_2) / (σ_1 σ_2)
```
where `m_1, σ_1` are over `τ_1 … τ_{n−2}` and `m_2, σ_2` over `τ_2 … τ_{n−1}`. Measures lag-1 autocorrelation of inter-event times. Goh & Barabási's finding: **for human dynamics memory is weak** (M ≈ 0) and burstiness comes from the interval distribution, not from correlation. An account with `|M| >> 0` is behaving unlike ordinary human dynamics — e.g. duty-cycled scripts (long gap → long gap → …).

#### 1.1.3 Interval entropy

Two variants used in the literature; compute both.

**(a) Shannon entropy of log-binned intervals.** Bin `τ` into log-spaced buckets (a natural set: `<10s, 10–60s, 1–5m, 5–15m, 15–60m, 1–6h, 6–24h, 1–7d, >7d`), get probabilities `p_k`, then `H = −Σ p_k log p_k`, normalised by `log K`. Low `H` ⇒ mechanical.

**(b) Approximate / Sample Entropy over the interval series.** Reported as a *single-feature* discriminator: **ApEn achieves accuracy 0.8483 (P 0.7686, R 0.9617, F1 0.8679)** and SampEn 0.7926 as standalone bot/human classifiers on a 2,300-account Twitter corpus. (Baseline table in Sci. Rep. 2022 below.) This is the strongest single-scalar temporal result I found — but note the corpus is small and was drawn from *Indian-hashtag* Twitter, which makes it unusually on-point for this project.
- https://www.nature.com/articles/s41598-022-11854-w / https://pmc.ncbi.nlm.nih.gov/articles/PMC9108350/

**Comparison anchor from that same table** (accuracy on 600-profile test set): relative-entropy DNA method **0.9457**, DNA fingerprinting (LCS) 0.9230, ApEn 0.8483, SampEn 0.7926, **Botometer 0.4898**.

#### 1.1.4 Interval quantisation / cron signature

This is the feature family your V2 lacks entirely and is cheap with full histories.

Evidence: bot inter-arrival distributions show "predominant peaks at about 15 and 100 seconds" and generally cluster at round-valued intervals; evasive bot authors are documented as randomising within e.g. 60–120 s windows precisely because fixed intervals are detectable. Human inter-message delays roughly follow a power law with no such spikes.

**Computable definitions (all cheap):**

- **Modular-residue concentration.** For each candidate period `p ∈ {60, 300, 600, 900, 1800, 3600, 86400}` seconds, compute residues `ρ_i = τ_i mod p`. Score by the **Rayleigh z-statistic** of the circular variable `θ_i = 2π ρ_i / p`:
  ```
  R = (1/N)·|Σ_i e^{i θ_i}|          z = N·R²
  ```
  `z` is ~Exp(1) under uniformity; `z > 10` is a strong periodicity flag at that period. Report `max_p z(p)` and `argmax_p`.
- **Grid-snap fraction.** `frac{ i : min_k |τ_i − k·p| ≤ ε }` for `ε = 2 s`, `p ∈ {60, 300, 900, 3600}`. Humans should be near the chance rate `2ε·(count of k in range)/range`; scripts spike.
- **Interval-value concentration (mode mass).** Round `τ` to the nearest second and compute the share of intervals falling in the top-1 and top-5 most common values; and the number of distinct `τ` values divided by `n−1`. Automation with a fixed sleep collapses this.
- **Second-of-minute / minute-of-hour uniformity.** Same Rayleigh test applied directly to `t_i mod 60` and `t_i mod 3600`. Catches "runs at :00 and :30" schedulers even when the account's own interval distribution is noisy because it interleaves multiple jobs.
- **Autocorrelation / FFT of the binned activity series.** Bin activity into 1-hour counts over the observation window, subtract the mean, and take the ACF; a sharp peak at lag 24 h is normal (circadian), a sharp peak at a non-diurnal lag is not. This is the standard "beaconing detection" trick imported from network security.

**Honest note:** I found **no peer-reviewed paper reporting a clean AUC for quantisation features alone** on social media. The evidence is (i) qualitative distributional plots, (ii) their inclusion inside Botometer's "temporal" class, and (iii) the network-security beaconing literature. Treat as a high-precision / low-recall flag: when it fires it is nearly conclusive, but sophisticated operators defeat it trivially with jitter.

#### 1.1.5 Time maps (2-D interarrival representation)

Radziwill & Benton (2016), "Bot or Not? Deciphering Time Maps for Tweet Interarrivals": plot each event as `(τ_{i−1}, τ_i)` on log axes. This separates three classes: pure humans, humans using schedulers (TweetDeck/Hootsuite), and bots — schedulers form distinct straight-line and cluster structures that neither pure humans nor pure bots produce. Useful because **"uses a scheduling tool" is a distinct class you will otherwise mislabel**.
Featurisation: 2-D histogram over the log-log time map (e.g. 8×8 bins) → 64 features; or summary stats (density on diagonal `τ_i ≈ τ_{i−1}`, off-diagonal mass, etc.).
- https://arxiv.org/abs/1605.06555

### 1.2 Circadian / diurnal features

**What to compute (hour-of-day in the *inferred local* timezone, not UTC):**

- **Hour-of-day histogram** `h_0 … h_23` normalised; and its **normalised Shannon entropy** `H_24 / log 24`. Human accounts have `H_24` well below 1; always-on schedulers approach 1.
- **Dead-hours count.** Number of hours with `h_k = 0` (or below `α·mean`, `α = 0.1`) over an observation window long enough that the expected count per hour exceeds ~5. A genuine human almost always has a contiguous 4–8 hour dead band (sleep). **Zero dead hours over months of activity, with sufficient volume, is one of the strongest cheap signals available.**
- **Contiguity of the dead band.** Longest circular run of near-zero hours. Real sleep is contiguous; random gaps from a low-volume human are not. Report `longest_zero_run` and `total_zero_hours`, and require `longest_zero_run / total_zero_hours` to be high for the "human sleep" pattern.
- **Circadian-dip phase → inferred timezone.** `argmin` of a smoothed hour histogram gives the account's local night. Compare with the plausible timezone of the subreddit's audience (for Indian subreddits, IST). A phase implying a working day 8–10 hours displaced from IST across many accounts in a subreddit is a *population*-level coordination signal even if it is weak per-account.
- **Day-of-week profile.** 7-bin histogram + entropy. Commercial/paid operations show weekday-shaped activity; hobbyists show weekend peaks; scripts show flat.
- **Weekday/weekend ratio** and **hour-of-week (168-bin) entropy** for accounts with enough volume.

Reported power: "constant activity throughout the day or systematic regularity of activity at specific times of the day" is a documented strong marker; humans "perform posts at specific daily time intervals, with activity appearing lower during weekends". These features sit inside the Botometer "temporal" class and inside the 99 time-based dimensions of the XGBoost/SHAP US-2020 model (below). No isolated AUC published, but the SHAP work ranks time-based activity patterns 4th of ~335 features.
- SHAP/XGBoost US-2020: https://arxiv.org/abs/2112.04913 — F1 0.916–0.919, **ROC-AUC 0.977–0.980, PR-AUC 0.967** on their own election dataset; **but average ROC-AUC only 0.828 on public datasets**. Feature count 335 (26 profile + 204 context + 99 time + 6 interaction). SHAP top features: `listed_count`, `favourites_by_age`, `friends_by_age`, then time-based activity patterns.

### 1.3 Session structure

**Definition (Pozzana & Ferrara, Front. Phys. 2020):** a *session* is a maximal run of consecutive posts by the same account where every gap is `< T`, with **T = 60 minutes**.

**Their key finding** — the most important behavioural result in the temporal literature: within a session, **humans show monotone trends across the first ~20 posts** (rising fraction of replies, rising mentions, falling text length) while **bots are flat**. The "human warms up / tires out" curve is a genuinely hard-to-fake artefact of cognition and attention.

**Reported power:** classifiers (Decision Tree / Extra Trees / Random Forest) with session-position features reach **AUC 0.97**, vs **0.83** without them — a ~14-point lift.
- https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2020.00125/full

**Reddit translation — this is directly portable and is the single most under-exploited idea for V3:**
For each session, compute per-position (1st, 2nd, …, k-th comment) values of:
- comment length in characters/words,
- whether the comment is a top-level reply to a submission vs a deep reply,
- reply depth in the tree,
- time since the parent comment,
- number of distinct subreddits touched within the session.
Then fit the **slope over position** (OLS over positions 1..min(20, len)) per account and aggregate (mean, sd of slopes across sessions). Human accounts should show a systematic negative slope on length; scripts should have slope ≈ 0 with low variance.

Additional session features: session count, mean/median session length (in posts and in minutes), inter-session gap distribution, and the fraction of activity inside sessions of length 1 (a "drive-by" profile).

### 1.4 Dormancy, reactivation and lifecycle

Sparse formal literature; strong practitioner consensus. What *is* documented:

- **The threat model is explicit in the literature.** The 2026 account-history-features paper states plainly that "a patient adversary who buys aged accounts (which have a market) … will defeat account-age features", and that acquiring aged accounts "carries its own costs and detectable artefacts." The artefacts are what you want to measure.
- **BLOC's pause alphabet** (§1.5) already encodes long gaps as first-class symbols (`t_w` week, `t_m` month, `t_y` year) — meaning "long dormancy followed by a burst" becomes a *word* in the behavioural language and is learnable as an n-gram.
- **Behaviour-change detection** is now an explicit research line: Ariyarathne et al., "Behavior Change as a Signal for Identifying Social Media Manipulation" (ACM WebSci 2026) segments account histories, measures BLOC-distance between consecutive segments, and builds features from the *distribution of change magnitudes*. Finding: **authentic accounts have consistent change distributions; social bots show either minimal or extreme variation; coordinated accounts show similar change patterns within a campaign and divergent ones across campaigns.** (Abstract does not report AUC/F1.)
  - https://arxiv.org/abs/2603.03128 · https://dl.acm.org/doi/10.1145/3795766.3799748

**Computable dormancy features (all trivially available from full histories):**
- `max_gap_days`, `gap_days_p95`, `gap_days_p99`.
- **Dormancy-reactivation flag**: exists a gap `≥ G` days (G = 90, 180, 365) that is followed within `W` days (W = 30) by activity exceeding `k×` the pre-gap rate. Emit the ratio `post_gap_rate / pre_gap_rate` and the gap length.
- **Topic/subreddit discontinuity across the gap**: Jaccard similarity of the subreddit sets before and after the largest gap. Low Jaccard + high rate ratio = the classic repurposed/resold account signature.
- **Style discontinuity across the gap**: cosine distance between character-4-gram TF-IDF profiles of pre-gap and post-gap text (see §1.6.4). A *stylometric break at a dormancy boundary* is the cleanest available proxy for "the account changed hands".
- **Segmented change series**: split the history into fixed-size segments (e.g. 100 comments each), compute a behavioural vector per segment, and emit `mean`, `sd`, `max`, and `n_changepoints` of consecutive-segment distances. This is the direct Reddit port of the WebSci 2026 method.
- **Karma/age mismatch**: activity rate relative to account age, with the pre-gap period excluded. Note the exploratory Reddit r/Sino study flagged "high activity paired with low link karma" and "comment–link karma imbalance" as anomalies (see §1.8 for why their thresholds were badly calibrated).

### 1.5 Behavioural sequence encoding (BLOC and Digital DNA) — recommended core

This is the best-evidenced way to turn "full comment history" into features, and it is platform-agnostic by design.

#### BLOC — Nwala, Flammini, Menczer, *EPJ Data Science* 12, 33 (2023)

Encode each account's history as a **string over three alphabets**, then treat it as text.

**Action alphabet (7 symbols):** `T` post; `P` reply to friend; `p` reply to non-friend; `π` reply to own post; `R` reshare friend's post; `r` reshare non-friend's post; `ρ` reshare own post.

**Pause alphabet — logarithmic discretisation of Δ (the `f₂` function, 6 bins):**
| symbol | condition |
|---|---|
| (none) | Δ < p₁ (session threshold; recommended p₁ ≤ 1 min) |
| `t_h` | p₁ ≤ Δ < 1 hour |
| `t_d` | 1 hour ≤ Δ < 1 day |
| `t_w` | 1 day ≤ Δ < 1 week |
| `t_m` | 1 week ≤ Δ < 1 month |
| `t_y` | 1 month ≤ Δ < 1 year |
| `t_u` | Δ ≥ 1 year |

(A simpler `f₁` emits nothing below p₁ and `.` above it.)

**Content alphabet (8 symbols):** `t` text; `H` hashtag; `M` mention-of-friend; `m` mention-of-non-friend; `q` quote-of-other; `φ` quote-of-own; `E` media; `U` URL.

**Featurisation:** tokenise either by n-gram (`n = 2` used for bot detection) or by pause boundaries (pauses act as word delimiters, individual posts delimit content words), then TF-IDF:
```
w_i(a) = f_i(a) · (1 + log(D / d_i))
```
`f_i(a)` = term frequency of word i in account a; `d_i` = number of accounts containing word i; `D` = total accounts.

**Reported performance:**

| Model | Precision | Recall | F1 | #features |
|---|---|---|---|---|
| **BLOC** | .899 | .884 | **.892** | **197** |
| Botometer-V4 | .929 | .914 | **.921** | 1,160 |
| Digital DNA (b3_type) | .796 | .529 | .636 | – |
| Digital DNA (b3_content) | .866 | .183 | .303 | – |
| DNA-influenced | .499 | 1.000 | .666 | – |

Dataset: 32,056 bots + 42,773 humans pooled from 12 datasets, 5-fold CV.

**Coordination detection (36 Twitter IO campaigns, mean F1 at week 10):** BLOC **0.659** with 108 features; "Activity" baseline 0.648 with 1,680 features; CoRT 0.516; Hashtag 0.316; combined 0.658 with 5,869 features. Per-campaign range is enormous: China_4 **F1 0.996**, China_3 0.995, Iran_6 0.961 — but UAE **0.182**, Russia_2 **0.211**. *Some operations are trivially detectable and some are essentially invisible; report per-community numbers, never a single pooled number.*

- Paper: https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-023-00410-9 · preprint https://arxiv.org/abs/2211.00639
- Reference implementation: https://github.com/anwala/bloc

**Reddit alphabet mapping (proposed).** Reddit lacks retweets and a friend graph, but has a richer reply tree. A workable action alphabet:
`T` = new submission; `C` = top-level comment on someone else's submission; `c` = top-level comment on own submission; `P` = reply to another user's comment; `p` = reply to own comment; `X` = crosspost; `E` = edited comment; `D` = deleted/removed comment (if observable). Content alphabet: `t` text, `U` URL, `E` media/image, `q` blockquote (`>` markdown), `M` u/ mention, `S` r/ mention, `#` markdown heading/list structure. Pause alphabet: use BLOC's `f₂` unchanged.

#### Digital DNA — Cresci et al.

Encode the timeline as a string over a small base alphabet (the canonical Twitter version: `A` = plain tweet, `T` = plain mention, `G` = plain retweet, `C` = tweet with media/URL) and compare accounts via **Longest Common Substring** curves ("Social Fingerprinting"). Group-level, unsupervised.

Reported on Cresci-2017 social spambots (test set #1): **accuracy 0.976, precision 0.982, recall 0.972, F1 0.977, MCC 0.952**. Test set #3: acc 0.929, **precision 1.000**, recall 0.858, F1 0.923.
- https://arxiv.org/abs/1703.04482 (Social Fingerprinting), https://dl.acm.org/doi/10.1145/3041021.3055135 (paradigm-shift)
Caveat: this is a **group** method evaluated on curated spambot groups; it is the "coordination" target (§2b), not per-account.

#### Relative-entropy DNA variant — Sci. Rep. 12, 7654 (2022)

Maps a DNA sequence to a discrete distribution and scores account pairs by symmetrised KL:
```
p_i = (α_i − β⃗_i) / ( ½·n(n+1) − β_n )      with base values T→0.2, A→0.4, G→0.6, C→0.8
Ren(μ₁,μ₂) = Σ_x p₁(x) log(p₁(x)/q₂(x))
d(μ₁,μ₂) = ½·[ Ren(μ₁,μ₂) + Ren(μ₂,μ₁) ]     ;  d ≤ 0.12 ⇒ bot
```
Avg over 5 bootstrap test sets of 600 profiles: **accuracy 0.9457, precision 0.9471, recall 0.9682, F1 0.9511, MCC ~0.91**. Corpus: ~2,300 profiles (1,094 bots, 1,204 humans) collected from **Indian hashtags**, Aug 2020 – Jul 2021. Botometer on the same data: **accuracy 0.4898**.
- https://www.nature.com/articles/s41598-022-11854-w

### 1.6 Content / linguistic features

#### 1.6.1 Self-similarity of an account's own posts

Not well-covered by a single canonical paper, but standard practice and cheap:
- **Mean pairwise cosine similarity** of the account's own comments under (a) character 3–5-gram TF-IDF and (b) a sentence embedding. Emit mean, median, p90, and the **fraction of comment pairs above 0.85**.
- **Near-duplicate rate via MinHash/LSH**: shingle each comment into word 5-grams, compute a 128-permutation MinHash signature, band into LSH buckets, and count the fraction of an account's comments that collide with another of its own comments at estimated Jaccard ≥ 0.7. MinHash was designed for exactly this (duplicate web pages, web-spam detection); SimHash is the Hamming/cosine analogue. Both are O(n) per comment and scale to your volumes.
  - https://arxiv.org/abs/1407.4416 (MinHash vs SimHash comparison)
- **Comment-length entropy**: the Reddit bot literature specifically uses "variability in comment length … bots display repetitive patterns resulting in a comment length distribution characterised by a low value of entropy."

#### 1.6.2 Cross-account near-duplicate / template detection

This is the highest-value content feature *and it is a coordination feature, not an account feature* — which is precisely why it works better (§2b). Build the same MinHash/LSH index across **all** accounts in a subreddit-month, then per account emit:
- number of distinct other accounts sharing a near-duplicate comment,
- max/mean similarity to the nearest non-self comment,
- **time delta to the nearest near-duplicate by another account** (a duplicate posted 8 seconds later is a different animal from one posted 3 weeks later),
- number of distinct submissions on which a near-duplicate pair co-occurs.

The coordination survey (Cinelli/Nizzoli et al. lineage, arXiv 2408.01257) documents that essentially every published coordination detector is built this way: pick a *co-action* (co-retweet, co-URL, co-hashtag-sequence, text similarity), define a **time window** (adjacent, evenly-distributed-overlapping, or action-driven-overlapping), compute user-user similarity as **cardinality of shared actions** or **TF-IDF cosine**, threshold, then cluster (Louvain / connected components).
- https://arxiv.org/abs/2408.01257

Concrete thresholds actually used in the literature: **cosine similarity at the 99.5th percentile** of the edge-weight distribution for co-sharing; **10-second windows** for "fast" co-action; **0.7 cosine** on Sentence-Transformer embeddings for text similarity.

#### 1.6.3 Lexical richness

- **Type-Token Ratio** `TTR = V(N)/N` (distinct tokens / total tokens). **Do not use raw TTR** across accounts with different total text volume — it drifts downward with length by construction.
- **MATTR** (Moving-Average TTR): slide a fixed window (500 tokens is standard; use 100 for Reddit comment volumes) and average within-window TTR. Length-corrected.
- **MTLD**: mean length of sequential token strings maintaining TTR above 0.72. Length-robust.
- Also: hapax-legomena ratio, average word length, contraction rate, emoji/emoticon rate.

**Reported power** — Inuwa-Dutse et al. (2018), "Lexical analysis of automated accounts on Twitter": random forest with **lexical features only** → accuracy 0.65–0.66 / AUC 0.65–0.66 on two datasets, and **accuracy 0.86 / AUC 0.87** on their third (manually inspected) dataset. Combining Gilani's 15 metadata features with the 4 lexical features lifted AUC from **0.71 → 0.95** on Dataset2. Counter-intuitive finding worth noting: **bots used emoticons *more* frequently than humans**, and emoticons were the single most distinctive feature.
- https://arxiv.org/abs/1812.07947

**Interpretation for V3:** lexical richness alone is a ~0.65 AUC feature family. Its value is as a *complement* to behavioural features, and it is the family most vulnerable to LLM rewriting.

#### 1.6.4 Stylometry for sockpuppet / same-operator linkage

The reliable formulation is **pairwise**, not per-account.
- **Character n-grams** (n = 3–5) with TF-IDF are the workhorse: they capture lexical, grammatical and orthographic preference simultaneously and are cheaper and more robust than syntactic parsers. Naive-Bayes over normalised character-bigram frequencies is the classic sockpuppet baseline (Solorio et al., Wikipedia sockpuppet corpus, LREC 2014).
  - http://www.lrec-conf.org/proceedings/lrec2014/pdf/1007_Paper.pdf
- **Punctuation and casing fingerprints:** rates of `, ; : — … ! ?`, multi-punctuation runs (`!!!`, `?!`), space-before-punctuation, straight vs curly quotes, double-space-after-period, ALL-CAPS run frequency, sentence-initial lowercase rate. These survive topic change and are hard to consciously suppress.
- **Emoji fingerprint:** the account's emoji unigram distribution + skin-tone modifier usage + emoji-per-100-tokens.
- **Reddit-specific markers:** markdown habits (`>` quoting style, `*` vs `_` emphasis, list marker choice, use of `^superscript`), edit-note conventions ("Edit:" vs "EDIT:" vs "e:"), and `/s` usage rate.
- Modern practice adds **transformer embeddings**; few-shot authorship attribution on Reddit posts is an active benchmark (https://openreview.net/pdf?id=IuIP-BWC88W). Meta-learning approaches on Wikipedia sockpuppetry: ACL 2025, https://aclanthology.org/2025.acl-long.1083.pdf
- Survey: https://arxiv.org/abs/2103.04673

**Reported power — Kumar et al., "An Army of Me" (WWW 2017), nine Disqus-hosted discussion communities.** This is the closest published analogue to Reddit and the most directly transferable result in this review.

| Task | All features | Activity | Community | Post/linguistic | Baseline |
|---|---|---|---|---|---|
| **Is an account a sockpuppet?** | **0.68** | 0.59 | 0.54 | 0.57 | 0.50 |
| **Are two accounts a sockpuppet pair?** | **0.91** | 0.86 | 0.56 | 0.80 | 0.50 |

Method: random forest, 10-fold CV, ROC-AUC; sockpuppets and ordinary users **matched on total posts and on the discussions they post in**, so the balanced dataset gives chance = 0.50. Their feature sets:
- *Activity*: reply-egonetwork clustering coefficient and reciprocity, number of posts, proportion of reply posts, mean time between consecutive posts, tenure (days since first post); for the pair task, **number of common sub-discussions**.
- *Community*: whether the account is blocked, fraction of upvotes, fraction of posts reported, fraction deleted.
- *Post*: characters/syllables/words/sentences, punctuation and uppercase fractions, syllables-per-word, words-per-sentence, ARI readability, LIWC categories, agreement, sentiment, emotion strength.

Behavioural characterisation: sockpuppets start fewer discussions, write shorter posts, use more first-person singular pronouns, have more clustered ego-networks, get more posts reported (0.06 vs 0.03) and deleted (0.11 vs 0.08), and receive fewer upvotes (0.45 vs 0.53).
- https://arxiv.org/abs/1703.07355 · https://cs.stanford.edu/~srijan/pubs/sockpuppets-www2017.pdf

**Take-away for V3:** the 0.68 → 0.91 jump from the account task to the pair task, on the *same features and the same data*, is the strongest evidence in this literature that you should be building a pairwise/cluster layer.

#### 1.6.5 LLM-generated-text detection

**Status: usable on Reddit, but only for long comments, and only in aggregate.**

The best current large-scale evidence is La Cava, Aiello & Tagarelli (2025), "Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit" — 51 subreddits, 2022–2024, 9,032,003 comments and 2,130,559 submissions after filtering.
- Detector: **Fast-DetectGPT** (zero-shot, curvature/conditional-probability based).
- Operating policy: **threshold τ = 0.99** and **minimum 250 tokens** — both required to keep false positives acceptable. Mean retained comment length 387 ± 180 tokens.
- Findings: MGT is "marginally present" overall but peaks at **6.3% (r/askscience), 7.7% (r/malefashionadvice), 8.5% (r/teenagers)** in individual subreddit-months; concentrated in a small fraction of users; conveys distinct social signals of *warmth* and *status-giving*.
- **Crucially: MGT achieves engagement comparable to, and sometimes higher than, human-authored content.** So reception signals do *not* separate LLM text.
- https://arxiv.org/abs/2510.07226

**Implications you must design around:**
1. The 250-token floor excludes the overwhelming majority of Reddit comments. LLM-detection is an *account-level aggregate* feature (`fraction of the account's ≥250-token comments flagged at τ=0.99`), not a per-comment label.
2. Humans are useless at this: a study of human perception of LLM text in social-media environments reports **42% accuracy with a 49% false-negative rate** — participants systematically read bots as humans. Do not build human-annotated ground truth for LLM-authored content without heavy protocol design.
   - https://arxiv.org/abs/2409.06653
3. Current LLM-powered bots are still *distinguishable in aggregate* but in the "too clean" direction. Ng & Carley (2025) compared LLM-generated bot networks against wild bots and wild humans and found LLM bots are **higher on all pronoun classes (1st 1.38 vs 0.71/0.73; 2nd 0.43 vs 0.20/0.18; 3rd 0.88 vs 0.47/0.50)**, **lower on reading difficulty (0.05 vs 0.12/0.10)**, and dramatically **lower on abusive terms (0.001 vs 0.13/0.09), expletives (0.00 vs 0.12/0.08), negative sentiment (0.01 vs 1.56/1.59) and positive sentiment (0.003 vs 2.88/3.10)** — i.e. sentiment-flat, profanity-free, high-pronoun, easy-to-read. **Higher hashtag use (1.93 vs 0.54/0.49).** These are *computable Reddit features*: sentiment magnitude, expletive rate, Flesch reading ease, pronoun rates.
   - https://arxiv.org/abs/2508.00998
4. **Compression ratio** (`len(text) / len(gzip(text))`) and **Flesch Reading Ease** are used as MGT stylistic discriminators in the Reddit MGT paper and cost nothing.
5. Surveys for the detector landscape: https://aclanthology.org/2025.cl-1.8.pdf (Computational Linguistics 51(1), 2025); adversarial/real-world benchmark https://arxiv.org/abs/2411.04032, https://arxiv.org/abs/2406.12549 (multilingual, includes social media).

**A hard warning specific to Indian subreddits:** LLM-detection tooling is calibrated on English and degrades on non-native English and code-mixed text. Rauchfleisch & Kaiser's language finding for Botometer (ROC-AUC 0.90 on English accounts vs **0.69** on German) is the general shape of this problem. Hinglish / Indian-English comments will produce elevated false positives on any perplexity-based detector. **Measure this before shipping.**

### 1.7 Reception features (signals the operator does not directly control)

This is a real and under-used family, but the literature is **genuinely mixed** and you should not over-promise on it.

**Evidence for:**
- Cheng, Danescu-Niculescu-Mizil & Leskovec (ICWSM 2015), "Antisocial Behavior in Online Discussion Communities" — 1.7M users on CNN, Breitbart, IGN. Predicting *future banned users* from their **first 5–10 posts** reaches **>0.80 AUC**. The features are post content, user activity, **community response**, and moderator action. Moderator/deletion features are the strongest signals. Future-banned users "concentrate their efforts in a small number of threads, are more likely to post irrelevantly, and **are more successful at garnering responses from other users**."
  - https://arxiv.org/abs/1504.00680 · https://cs.stanford.edu/people/jure/pubs/trolls-icwsm15.pdf
- Kumar et al. 2017 (above): sockpuppets receive measurably worse community treatment — more reports (0.06 vs 0.03), more deletions (0.11 vs 0.08), fewer upvotes (0.45 vs 0.53). But note the *community* feature set alone scored only **AUC 0.54** for the account task and 0.56 for the pair task. **Reception features were the weakest family in that study.**
- Reddit troll data point: state-sponsored troll accounts had **mean score per comment 5.7 vs 4.8** for other accounts — trolls got *more* engagement, not less (TrollMagnifier).
- LLM-powered bots "do not get much engagement due to their use of GenAI" per OpenAI's threat-intelligence reporting — but see the contradicting Reddit MGT result below.

**Evidence against / mixed:**
- Reddit MGT (2025): machine-generated comments get engagement **comparable to or higher than** human comments.
- Bot engagement on Twitter: "social bots generate as much engagement, at least in terms of obtained retweets, as humans."
- Feature-level: Mbona & Eloff found **retweet count discriminative but reply count not**; Pozzana & Ferrara found both replies and retweets more prevalent in human interaction. Brazilian-elections study found reply-share correlates **positively** with bot engagement (r = 0.66) while retweet-share correlates negatively (r = −0.55). https://arxiv.org/abs/2310.09051
- Bot/human reply asymmetry does exist: "humans engage in reply interactions significantly more with other humans than with bots, while bots … end up interacting via replies with other bots significantly more than with humans."

**Recommended Reddit reception features (compute them, expect a modest lift, do not build the model on them):**
- `mean_score`, `median_score`, `p10_score`, `p90_score`, **fraction of comments at score exactly 1** (i.e. never voted on — the "talked past" signal), fraction at score ≤ 0.
- **Score distribution shape**: Gini coefficient of the account's score vector; ratio of p90 to median. Organic accounts have a few hits; broadcast accounts have flat, low, unimodal scores.
- **Incoming-reply rate**: fraction of the account's comments that received ≥1 reply; mean number of direct children; mean subtree size below the account's comments.
- **Reply-partner diversity**: number of distinct users who replied to the account / number of the account's comments; and the **entropy** of that distribution. A high-volume account that is only ever replied to by a handful of the same accounts is a coordination signal.
- **Reciprocity**: fraction of the account's replies that go to users who have replied to it.
- **Talked-past ratio**: `(#comments with 0 replies) / (#top-level-eligible comments)`, controlled for the parent submission's overall reply density — otherwise you are measuring subreddit activity, not the account.
- **Controversiality rate**: Reddit exposes a `controversiality` flag on comments; use the rate directly. Also compute score-vs-depth: organic comments deep in a thread earn less.
- **Removal/deletion rate**: fraction of the account's comments whose body is `[removed]` (moderator action) vs `[deleted]` (user action) in the archive. Per Cheng et al., moderator deletion is the *strongest* signal family for enforcement prediction — but this leaks toward your enforcement label, so keep it out of any model whose label is enforcement-derived. It is legitimate for an automation/inauthenticity label.

**Essential normalisation caveat:** every reception feature is dominated by *where and when* the comment was posted. A comment 4 hours into a thread in a 50k-member subreddit at 3am IST has a structurally different score distribution than a top-level comment 90 seconds after submission. **Compute all reception features as residuals against a baseline model of `expected_score | subreddit, hour, thread_age_at_comment, depth, parent_score`.** Without this, reception features encode posting-strategy, not reception.

### 1.8 Time-to-arrival on new content

- **`arrival_lag = comment.created_utc − submission.created_utc`** for top-level comments; distribution over the account (median, p10, fraction under 60 s, fraction under 10 s).
- **`reply_lag = comment.created_utc − parent_comment.created_utc`** for replies.
- **Rank-of-arrival**: the account's comment position among all comments on that submission, normalised.
- **Cross-account fast co-arrival** (this is the coordination version and the stronger signal): number of submissions where the account and another specific account both commented **within a 10-second, 60-second, and 300-second window**. The coordination literature uses exactly this as the "fast retweet" trace — although note the ablation in Luceri et al. found **Fast Retweet was the *least* influential** of their five traces, while **Co-Retweet was the most influential**. On Reddit, the analogous strongest trace is likely **co-submission-participation** (which submissions two accounts both comment on) rather than raw speed.
- Consistently sub-60-second arrival on new submissions across a large sample implies a streaming/monitoring pipeline. Human "first!" behaviour exists but is bursty and confined to a few subreddits.

### 1.9 What V2's ten features actually are, per the literature

Your V2 set (account age, karma/day, link-karma ratio, verified email, sample-local activity counts) is the **"account metadata / user profile"** family — the oldest and weakest family. Direct evidence:

- **Yang, Ferrara & Menczer (AAAI 2020)**, "Scalable and Generalizable Social Bot Detection through Data Selection" — random forest over ~20 metadata features (`statuses_count`, `followers_count`, `friends_count`, `favourites_count`, `listed_count`, `default_profile`, `profile_use_background_image`, `verified`, `screen_name_length`, `num_digits_in_screen_name`, `name_length`, `num_digits_in_name`, `description_length`, plus 7 derived rates: tweet/follower/friend/favourite/listed growth per day of account age, followers-friends ratio, screen-name bigram likelihood).
  - Cross-validation AUC: **0.98**.
  - Cross-dataset AUC on unseen data: **botwiki-verified 0.99, midterm-18 0.99, gilani-17 0.68, cresci-rtbust 0.60**.
  - Botometer on the same: 0.92 / 0.96 / **0.67** / 0.71.
  - The 11×11 cross-dataset matrix contains cells **below 0.5** — "training and test datasets have contradictory labels."
  - https://arxiv.org/abs/1911.09179 · https://ojs.aaai.org/index.php/AAAI/article/view/5460

  **This is the cleanest possible statement of your problem: the same 20-feature metadata RF is 0.98 or 0.60 depending entirely on what "bot" means in the labelled set.** Gilani-17 (human annotators asked "is this account automated?") is the hardest, and that is the target most like yours: **0.67–0.69**.

- **TwiBot-22 (2022):** metadata-only baselines on realistic data — SGBot **acc 75.1 / F1 36.6**; BotHunter acc 72.8 / F1 23.5; Kudugunta acc 65.9 / F1 51.7; Abreu acc 70.7 / F1 53.4; NameBot acc 70.6 / **F1 0.5**. The accuracy figures look respectable only because 86% of the data is human.

- **Counter-evidence you should read sceptically:** Katyal (arXiv 2606.26127, June 2026) reports **ROC-AUC 0.977** for a 24-feature "account-history" random forest vs 0.830 for content features, and shows the behavioural model is invariant to adversarial text rewriting while the content model falls to AUC 0.466. The adversarial-robustness argument is sound and worth taking. **But the headline AUC is not transferable:** the corpus is 2,432 accounts, 43% bots, balanced, from a publicly redistributed GitHub/Kaggle set, and `verified` alone has Cohen's *d* = −1.27 (0.46 of humans verified, 0.01 of bots) with account age *d* = −1.55. That corpus is trivially separable and is not comparable to TwiBot-22 or to your setting. It is a single-author, non-peer-reviewed preprint. Cite it for the *robustness argument*, not for the number.
  - The 24 features, for completeness: 5 account-age rates (count / age-in-days for posts, followers, friends, favourites), 5 follow asymmetries (log counts, friends/followers ratio, listed/followers ratio), 6 profile-completeness binaries, 7 screen-name structure features (length, digit ratio, character entropy, trailing-digit pattern for handle and display name), 1 engagement asymmetry (posts/favourites).

---

## 2. Benchmark / AUC Reality Check — by target type

**Read this table as: what is the ceiling, and for what exact question?**

### 2a. Detecting self-declared / structurally obvious automation

| Study / dataset | Year | Target | Metric | Notes |
|---|---|---|---|---|
| TwiBot-22 baselines on **Cresci-15** | 2022 | traditional spambots | **F1 98.8** (LOBO), 98.6 (Lee), acc 98.4 | 35 methods re-run; most exceed F1 0.95 |
| TwiBot-22 baselines on **midterm-18** | 2022 | election bots | **F1 99.6** (BotHunter), 99.5 (SGBot) | |
| BotBuster-4 on **verified-2019** / **political-bots-2019** | 2023 | verified humans vs known bots | **F1 89–100** | |
| Cresci et al., traditional spambots, **human annotators** | 2017 | traditional spambots | acc 0.9136, **precision 1.000** | humans are near-perfect on *old* bots |
| Ephemeral astroturfing (Elmas et al.) | 2021 | one specific attack signature | **precision 1.000, recall 0.989, F1 0.994**; lexicon-agnostic F1 0.978 | post-then-delete-within-10-min + keyword lexicon |
| DeBot (Chavoshi et al., ICDM 2016) | 2016 | warped-correlation synchrony | **precision 0.94**, unsupervised | detects thousands/day; group-level |

**Verdict: AUC/F1 ≥ 0.95 is routine here.** But this target is nearly self-defining — you are detecting accounts that *announce* their automation through handle patterns, template output, or a mechanically exact signature. **This is not what "inauthentic activity on r/india" means.**

### 2b. Detecting coordination / shared operator / influence operations (group- or pair-level)

| Study | Year | Target | Metric |
|---|---|---|---|
| **Kumar et al., "An Army of Me"** (WWW) | 2017 | are these **two accounts** sockpuppets of one operator? | **AUC 0.91** (activity 0.86, post 0.80, community 0.56) |
| same, "which of the pair is the puppetmaster?" | 2017 | | **AUC 0.91** |
| **Luceri/Cardoso et al., "Unmasking the Web of Deceit"** | 2023 | IO account detection, fused similarity network + node2vec, supervised | **AUC 0.94, F1 0.82, precision 0.96** |
| same, global (cross-country) classification | 2023 | | **AUC 0.92, precision 0.95, recall 0.70, F1 0.78** |
| same, **unsupervised** node pruning on fused network | 2023 | | **AUC 0.83–0.84, F1 0.76–0.77** |
| same, prior-art edge-filtering baselines | 2023 | | **AUC 0.47–0.61** (optimised: 0.52–0.72) |
| **BLOC**, 36 Twitter IO campaigns | 2023 | coordinated campaign membership | **mean F1 0.659**; range **0.182 (UAE) → 0.996 (China_4)** |
| Cresci et al. Social Fingerprinting (Digital DNA/LCS) | 2017–18 | spambot *group* detection | acc 0.976, F1 0.977 (set #1); precision 1.000, F1 0.923 (set #3) |
| ACCD (causal coordination, IRA dataset) | 2026 | coordinated attack detection | F1 0.873 vs CCM baseline precision 0.80 / recall 0.72 / **AUC 0.722** |
| Unsupervised CIO detection (EPJ Data Science) | 2025 | coordinated info operations | **PR-AUC lift of 76×–580×** over naive baseline (absolute PR-AUC low) |
| **TrollMagnifier** (Reddit!) | 2021–22 | expand a seed set of known Reddit trolls | 10-fold CV P/R/F1/acc all **0.978** — **but see caveat** |

**Verdict: AUC 0.90–0.94 is genuinely achievable here.** This is where your 0.90 target is realistic.

**Two mandatory caveats:**
1. **TrollMagnifier's 97.8% is not what it looks like.** Six of its nine features are literally "fraction of the account's comments that interact with a *known troll*": same-title submissions, comments on troll-commented submissions, comments on troll submissions, direct replies on troll submissions, replies to troll comments, nested troll-to-troll replies. The other three are total comments, total submissions, account age. This is guilt-by-association with a labelled seed set — cross-validation on that construction is near-circular. The paper's *real* result is the field validation: of 1,248 newly flagged accounts, **66% showed corroborating signs** of instrumentation. So the honest operational precision is ~0.66, not 0.98. That said, **the architecture is the right one for Reddit** and it is the only published Reddit-native detector of this kind. https://arxiv.org/abs/2112.00443
2. **Per-campaign variance is enormous.** BLOC's 0.182–0.996 range across 36 campaigns means a single pooled F1 is meaningless. Report per-subreddit, per-month.

### 2c. Predicting platform enforcement (suspension / removal / ban) — *your V2 label*

| Study | Year | Target | Metric |
|---|---|---|---|
| **Cheng et al. (ICWSM)** | 2015 | future *banned* user, from first 5–10 posts | **AUC > 0.80** — but features include **moderator deletion actions**, the strongest family |
| **Kumar et al. (WWW)** account task | 2017 | is this account a sockpuppet (IP+time-derived label) | **AUC 0.68** |
| Rauchfleisch & Kaiser (PLOS ONE) | 2020 | Botometer vs curated bot/human sets | AUC **0.93** (US politicians vs bots), 0.86 (Varol), **0.85** (all combined), **0.76–0.78** (German) |
| ICWSM (Elsayed/Mishra et al.) — deleted & suspended accounts, 3 languages | 2022 | account deletion/suspension | best category **F1 0.88, ROC-AUC 0.95** on initial data; **profile features F1 0.79, ROC-AUC 0.90** in the long-term scenario; **language/discourse is the strongest family**, image and affect weakest |
| Regularized logistic regression, suspended-user detection | — | suspended users | **AUC ≈ 0.83**; ~60% of suspended users at 10% FPR |
| **TwiBot-22** (label = expert + Snorkel weak supervision, 14% bots) | 2022 | "is this a bot" at scale | **best F1 58.7**, best acc 79.7; **Botometer acc 49.9 / F1 42.8** |

**Verdict for your exact situation:** the honest ceiling for "predict platform enforcement from account-visible behaviour" is roughly **AUC 0.75–0.85, and only when you are allowed to use moderator-action features** (deletions, reports) which are themselves downstream of enforcement. Without those, and with a low base rate, you should expect **0.65–0.75**. Your controlled experiment (0.59–0.67 across suspension-only and admin-removal labels) is squarely consistent with the literature. **Reasons enforcement is a bad label, all documented:**
- **Enforcement is not inauthenticity.** Reddit suspends for harassment, ban evasion, spam, doxxing, vote manipulation, and age violations. Only some of that is automation.
- **Enforcement is temporally unstable.** Rauchfleisch & Kaiser: over a 3-month window, **27.2% of "new bot" accounts and 22.2% of German bot accounts crossed the classification threshold**, vs 0.6% of US politicians. Bot scores are not stationary; neither is enforcement.
- **Enforcement misses the sophisticated cases by construction** — the accounts you most want to catch are the ones that were not caught.
- **Base-rate/precision collapse.** At the widely-used Botometer threshold 0.76, **41% of accounts classified as bots were human** on the combined dataset, rising to **76% false positives** on the German politicians + bots set. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045

### 2d. Detecting sophisticated/modern social bots per-account (the target you actually want)

| Study | Year | Target | Metric |
|---|---|---|---|
| **Cresci et al. (WWW 2017)** — human crowdworkers on social spambots | 2017 | is this a social spambot? | acc 0.2355 (spambot class), **precision 0.267, recall 0.080, F1 0.123, MCC 0.001** — i.e. **humans are at chance** |
| same, **BotOrNot/Botometer** on social spambots #1 | 2017 | | acc 0.734, precision 0.471, recall 0.208, **F1 0.288, MCC 0.174** |
| same, Yang et al. 2013 | 2017 | | acc 0.506, F1 0.261 |
| same, Miller et al. 2014 | 2017 | | acc 0.526, F1 0.435 |
| **Gilani-17** (human-annotated "automated vs human", κ = 0.77) — best across 35 TwiBot-22 baselines | 2022 | | **acc 78.6 (SGBot), F1 77.4 (Botometer)** |
| Yang et al. 2020 cross-dataset RF on gilani-17 | 2020 | | **AUC 0.68** |
| **TwiBot-22** — best of 35 methods | 2022 | | **acc 79.7 (BotRGCN), F1 58.7 (Kantepe)** |
| BotArtist (XGBoost, 49 features) — per-dataset | 2023–25 | | avg **F1 83.19** across 9 datasets individually; **TwiBot-22 F1 58.2**, gilani-17 F1 76.1 |
| BotArtist — **cross-dataset generalisation** | | | **avg F1 68.5, total F1 63.7** (Botometer 63.5) |
| BotBuster (mixture of experts, ICWSM 2023) | 2023 | | Twitter avg **F1 73.54** vs baselines 45.12; external TwiBot-20 **F1 60.92** |
| Human perception of LLM text in social media | 2024 | | **42% accuracy, 49% FN** |

**Verdict: 0.65–0.80 AUC / 0.55–0.75 F1 is the realistic band for per-account detection of non-obvious automation on realistic data.** Anything above 0.90 reported for this target is, in every case I checked, one of: (i) a small balanced curated corpus, (ii) a label that leaks (interactions with known bad actors), (iii) an old bot generation, or (iv) within-dataset CV with no cross-dataset check.

### 2e. Reddit specifically

| Study | Year | Setup | Result |
|---|---|---|---|
| **BotBuster** (Ng & Carley, ICWSM 2023) | 2023 | 500 top crowd-nominated "bad bots" from BotRank vs manually-verified humans from 5 thoughtful subreddits; **usernames + 20 posts only** (no follower graph, no rich metadata) | **F1 60.04** (merged training) / **69.77** (single-dataset) |
| **BotBuster For Everyone** (2024) ensemble | 2024 | reddit-2022, 667 users, 75% bots | **accuracy 34.68%, micro-F1 51.20, macro-F1 33.86** |
| **TrollMagnifier** (Oakland/S&P) | 2021–22 | 335 Reddit-identified Russian troll accounts → expand | CV F1 0.978; **operational corroboration 66%**; grew the seed set by >300% |
| Hurtado, Ray & Marculescu, "Bot Detection in Reddit Political Discussion" | 2019 | r/The_Donald 2016 election | (ACM DL paywalled; comment/submission-count features + classifiers) |
| Cross-Subreddit Behavior (r/Sino & r/China) | 2025 | 63 dual-subreddit users, 5 heuristic anomaly flags | **exploratory only, no ground truth.** "Low lexical diversity" flagged **51 of 63 users** — i.e. an uncalibrated threshold. Useful as a cautionary example of what not to ship. |
| La Cava et al., Reddit MGT footprint | 2025 | Fast-DetectGPT, 51 subreddits, 9M comments | MGT peaks 6.3–8.5% per subreddit-month; engagement ≥ human |

**Verdict: there is no published Reddit account-level detector above F1 ~0.70 on a non-leaking label.** The field is genuinely open here, which is both the opportunity and the reason not to promise 0.90.

---

## 3. Model Recommendations

### 3.1 Where gradient boosting beats linear models — and where it doesn't

**Use GBDT (XGBoost / LightGBM) as the default.** Direct evidence from this literature:
- **BotArtist** uses XGBoost (chosen over SVM and RandomForest by tuning) over 49 features and reports **avg F1 83.19 per-dataset, 68.5 cross-dataset**, beating Botometer's 63.5 on generalisation. https://arxiv.org/abs/2306.00037
- The **US-2020 explainable framework** uses XGBoost + SHAP over 335 features: F1 0.916–0.919, ROC-AUC 0.977–0.980 in-domain, 0.828 out-of-domain. https://arxiv.org/abs/2112.04913
- **Random forest** is the most-used model in the classic literature (Gilani-17, Yang et al. 2020, Kumar et al. 2017, TrollMagnifier, Cheng et al. 2015, Pozzana & Ferrara) and is consistently competitive.
- Katyal (2026) reports **RF 0.977 vs LogReg 0.951** on the same behavioural features (and RF 0.830 vs LogReg 0.719 on content features) — **the tree/linear gap is ~0.026 AUC on behavioural features and ~0.111 on content features**. Gradient boosting was "almost identical to random forests at this scale."

**Why the gap will be much larger for you than for V2.** Linear logistic regression on 10 monotone-ish metadata features loses very little to trees. But the V3 feature families are exactly the ones where trees win:
- **Non-monotone features.** "Dead hours = 0" and "dead hours = 20" are both suspicious for opposite reasons; a linear model can only pick one direction. Same for burstiness (both B ≈ −1 and B ≈ +1 are anomalous relative to the human band), and for behaviour-change magnitude (the WebSci 2026 finding is literally "bots show either minimal *or* extreme variation").
- **Interactions.** `zero dead hours` is only meaningful *conditional on* sufficient volume; `interval quantisation` is only meaningful conditional on `n ≥ 50`; `arrival lag < 10 s` only matters conditional on subreddit size. Trees get these free; logistic regression needs them hand-crafted.
- **Heavy tails and outliers.** Karma/day, comment counts, and gap lengths span orders of magnitude. Trees are scale-invariant; LR needs careful transforms.

**Keep a regularised logistic regression as a permanent baseline** — it is your instrument for detecting when the GBDT is exploiting a leak.

### 3.2 Feature counts

Published feature counts and what they bought:

| System | #features | Result |
|---|---|---|
| BLOC (bi-gram) | **197** | F1 0.892 |
| BLOC (coordination, pause-tokenised) | **108** | mean F1 0.659 over 36 campaigns |
| Botometer V4 | **1,160** (>1,000 in six classes) | F1 0.921 in-domain; **F1 42.8 on TwiBot-22** |
| BotArtist | **49** | avg F1 83.19; best cross-dataset generalisation |
| Yang et al. 2020 RF | **~20** | AUC 0.98 CV / 0.60–0.99 cross-dataset |
| US-2020 XGBoost | **335** (26 profile / 204 context / 99 time / 6 interaction) | AUC 0.977 in-domain / 0.828 out |
| Katyal 2026 | 24 behavioural + 12 content | AUC 0.977 (easy corpus) |
| Kumar et al. 2017 | ~30 across 3 families | AUC 0.68 / 0.91 |

**Recommendation: target 80–200 features for V3.** Below ~50 you cannot express sequence structure; above ~300 you will overfit month-to-month and lose interpretability, and the evidence (BLOC 197 ≈ Botometer 1,160; BotArtist 49 generalising best) says the marginal feature buys nothing. Botometer's collapse to F1 42.8 on TwiBot-22 despite 1,160 features is the clearest possible refutation of "more features."

Botometer's six-class taxonomy is still the right *organising* schema: **user profile, friends, network, temporal, content & language, sentiment**. Your V2 has family 1 only. V3 should have 1, 4, 5, plus a Reddit-native "reply-graph" analogue of 2 and 3.

### 3.3 Class imbalance

- **Never report accuracy.** TwiBot-22 is 86% human; a majority-class classifier scores 86% accuracy and F1 0. NameBot achieves acc 70.6 / **F1 0.5** on TwiBot-22 — a perfect illustration.
- **Report PR-AUC (average precision) as the headline, ROC-AUC as secondary, plus precision@k.** The IO-detection literature explicitly adopts average precision "for highly imbalanced datasets, as it focuses on the ability to detect the positive class rather than the ROC curve." The unsupervised-CIO paper reports its result as a **PR-AUC lift factor (76×–580×)** precisely because the absolute number is low and the lift is the meaningful quantity. For a monthly product, **precision@100 and precision@1000 per subreddit** is what actually matters — that is the review budget.
- **Also report MCC.** Rauchfleisch & Kaiser explicitly recommend "reporting precision, recall, and Matthews correlation coefficient." MCC is the metric that exposed how bad BotOrNot was on social spambots (MCC 0.174 despite acc 0.734) and how bad human annotators were (**MCC 0.001** despite acc 0.698).
- **Handling:** BotArtist's recipe is the best-documented in this space — stratified splits, undersampling of the majority class *during feature selection only*, repeated (10×) resampling to cover the majority class, **class weights in the final classifier**, and **decision-threshold optimisation via the precision-recall curve**. Prefer `scale_pos_weight` / class weights over SMOTE: synthetic minority oversampling on behavioural features generates accounts with impossible feature combinations (e.g. fractional comment counts, incoherent timestamps).
- Kudugunta & Ferrara combined SMOTE with undersampling; their TwiBot-22 F1 was 51.7 — respectable for a metadata method, but not evidence that SMOTE was the reason.

### 3.4 Calibration

- Report **Brier score** and **reliability diagrams** on a held-out set. Katyal (2026) does this explicitly and motivates it correctly: calibrated probabilities are required "as inputs to threshold-based moderation pipelines."
- Use **isotonic regression** (you will have enough data) or Platt scaling on a dedicated calibration fold. GBDTs trained with class weights are systematically over-confident.
- **Calibrate per subreddit and per month.** The stability evidence is unambiguous: 27.2% of accounts crossed the Botometer decision threshold within three months without any change in the accounts themselves; Botometer's AUC ranges 0.69 (German) to 0.93 (US politicians) *for the same model*. A single global threshold across 45 heterogeneous Indian subreddits will produce wildly different precision per subreddit. **Report a per-subreddit precision estimate or do not report a per-subreddit rate.**
- **Prevalence estimation ≠ classification.** If the product output is "X% of activity in r/foo is inauthentic," you want a *quantification* method (e.g. BotPercent-style, or classify-and-count with a confusion-matrix correction), not the raw positive rate of a thresholded classifier. https://arxiv.org/abs/2302.00381

### 3.5 Evaluation protocol — the part that will save you

1. **LOBO-style testing.** Echeverría et al. (ACSAC 2018) introduced "Leave One Botnet Out": hold out an entire *class* of bot and train only on the others. Their finding: "methods trained and tested on single bot classes or datasets might not be able to generalize to new bot classes." Your analogue: **leave-one-subreddit-out** and **leave-one-month-out**. If AUC collapses under LOSO, you have learned a subreddit, not a behaviour. https://arxiv.org/abs/1809.09684
2. **Always report the cross-domain number next to the in-domain number.** The literature's most honest papers do (Yang et al.: 0.98 vs 0.60; BotArtist: 83.19 vs 68.5; US-2020: 0.977 vs 0.828). Papers that report only the in-domain number are the ones the field has since had to discount.
3. **Manual validation of a stratified sample is mandatory, not optional.** Rauchfleisch & Kaiser's recommendations verbatim: manual validation of classified samples; test stability over time; language-specific validation; report precision/recall/MCC; publish classified account IDs.
4. **Minimum-data threshold.** Ng & Carley found bot classification stabilises after **36 posts** (the gradient of the score change goes to ~0: −7.80E-6 ± 6.86E-3). Set an explicit `min_comments` gate (36 is a defensible published floor; 50 is safer for temporal features, and burstiness/entropy features need ~50–100 intervals to be stable). **Emit "insufficient data" rather than a score** below the gate — this alone will remove a large slice of your false positives, since low-activity accounts are where metadata models fail worst.

---

## 4. Reddit-Specific Notes (vs Twitter/X)

**What Reddit does not have** (and therefore which literature does not transfer):
- **No follower/friend graph.** This kills the entire "friends" and "network" Botometer classes, all follower/following-ratio features (which SHAP analysis ranks among the top on Twitter: `listed_count`, `friends_by_age`), and every follow-graph GNN — which is precisely the family that tops TwiBot-22 (all top-5 models are graph-based, outperforming the global average by 8.2% on TwiBot-22 and 13.8% on TwiBot-20). **You lose the best-performing model family outright.**
- **No `listed_count`, no verified badge equivalent, no profile background image, no per-account bio metadata of substance.** BotBuster explicitly notes "user information for Reddit accounts only consists of user names."
- **No retweet.** Every co-retweet coordination trace — the *most influential* trace in Luceri et al.'s ablation — has no direct analogue. Crossposts and link resubmissions are the nearest thing and are far rarer.
- **No `source`/client field.** Gilani-17's "activity source type" feature (browser / mobile / OSN management / automation / marketing) is one of the strongest classic features and is simply unavailable.

**What Reddit has that Twitter does not** — and where V3's advantage lies:
- **A real conversation tree.** `parent_id` gives you depth, subtree size, sibling position, reply latency to a specific parent, and reciprocity. Twitter's reply graph is flat and noisy by comparison. **This is your best asset**: Kumar et al.'s *activity* family (reply-egonetwork clustering coefficient, reciprocity, proportion of reply posts, common sub-discussions) is the family that carried AUC 0.86 of their 0.91 pair-level result, and it is built from exactly this structure.
- **Explicit community membership** (`subreddit`) as a first-class, low-cardinality, stable partition. Subreddit-set Jaccard between accounts is a strong, cheap co-participation signal — the Reddit analogue of co-hashtag. The r/Sino study's framing (dual-subreddit users as the sampling frame) is methodologically sound even though its thresholds were not.
- **Score and `controversiality` per comment** — a genuine reception signal Twitter lacks (Twitter has no downvote).
- **Long-form text.** Median Reddit comments are far longer than tweets, which makes stylometry, lexical-richness and LLM-detection viable in a way they are not on Twitter — La Cava et al. needed ≥250 tokens and Reddit supplied 9M such comments, but a comparable X/Bluesky study "would require the development of specialised MGT detectors for short-form text — a research challenge in itself."
- **Complete, retrievable comment histories** via Arctic Shift, including deleted-body markers. Twitter research has been effectively closed since 2023. This is a genuine and shrinking-window advantage.
- **Throwaway accounts are a normal, sanctioned Reddit practice** (see https://arxiv.org/abs/2501.17430 on throwaway accounts and moderation). A new, low-karma, single-subreddit account on Reddit is *not* prima facie suspicious the way it is on Twitter. **Your V2's account-age and karma features are weaker on Reddit than the Twitter literature implies**, and this partly explains the 0.663.

**Reddit-native detection layers documented in practice** (from Reddit's own statements and moderation tooling, not peer-reviewed): rate limiting, karma-velocity analysis, subreddit-level AutoModerator rules, community reporting, and site-wide ban-evasion filtering. Reddit's own disclosure in the r/ChangeMyView incident: it detected and removed **21 of 34** undisclosed AI accounts — a recall of ~62% on a case it was specifically alerted to.

**The r/ChangeMyView natural experiment (Nov 2024 – Mar 2025)** is the most important recent Reddit data point and should inform your expectations directly: 34 LLM-driven accounts posted 1,500–1,700 comments, personalised by inferring user demographics from posting histories, and were "consistently well-received" — **>20,000 upvotes and 137 deltas** — while going undetected by the community for four months in *the* subreddit most attentive to argument quality. Coverage: https://www.404media.co/researchers-secretly-ran-a-massive-unauthorized-ai-persuasion-experiment-on-reddit-users/ ; post-hoc analysis: https://arxiv.org/abs/2606.05256
**Implication:** reception features will not catch competent LLM operations on Reddit, and neither will human review of individual comments. Temporal/structural and cross-account features are the only families with a plausible path.

**Data-source note.** Arctic Shift (Arthur Heitmann, https://github.com/ArthurHeitmann/arctic_shift) is the community successor to Pushshift, offering monthly dumps, an API (~120k requests/hour, unauthenticated), and a web interface, covering 2005–present. Known limitation: **full-text search only works within a single subreddit**, no global keyword search — relevant if you plan cross-subreddit near-duplicate detection, which you will need to build from bulk dumps rather than the search API.

---

## 5. Concrete Feature Shortlist for V3 (ranked by expected value per unit of effort)

**Tier 1 — build first (evidence is strongest, cost is lowest):**
1. Session-position slopes (comment length, reply depth, latency) over the first 20 comments per session, T = 60 min. *[Pozzana & Ferrara: +14 AUC points]*
2. BLOC-style symbolic encoding of `(action, pause, content)` triples + bi-gram TF-IDF, ~150–200 features. *[BLOC: F1 0.892 with 197 features]*
3. Circadian: hour-of-day entropy, dead-hour count, longest contiguous zero-run, inferred timezone offset vs IST, day-of-week entropy.
4. Finite-size-corrected burstiness `A_n`, memory coefficient `M`, log-binned interval entropy, ApEn over the interval series. *[ApEn: acc 0.848 standalone]*
5. Cross-account near-duplicate graph (MinHash/LSH at Jaccard 0.7) + co-participation graph (shared submissions) → per-account degree, max-similarity, min-time-delta, distinct-partner count. *[the 0.68 → 0.91 lever]*

**Tier 2 — high value, more work:**
6. Interval quantisation: Rayleigh z-statistic at p ∈ {60, 300, 900, 3600, 86400}, grid-snap fractions, mode-mass of rounded intervals.
7. Dormancy/reactivation: max gap, post/pre-gap rate ratio, subreddit-set Jaccard across the gap, **character-4-gram style distance across the gap**.
8. Segmented behaviour-change distribution (mean/sd/max/n-changepoints of consecutive-segment BLOC distance). *[WebSci 2026]*
9. Reception residuals: score, reply-received rate, talked-past ratio, reply-partner entropy, controversiality — **all as residuals against `expected | subreddit, hour, thread_age, depth`**.
10. Reply-tree structure: ego-network clustering coefficient and reciprocity over the reply graph, proportion of replies vs top-level. *[Kumar et al.: activity family = 0.86 of the 0.91]*

**Tier 3 — worth having, weaker or riskier:**
11. Lexical richness: MATTR/MTLD (not raw TTR), hapax ratio, emoji rate, punctuation fingerprint, compression ratio, Flesch reading ease. *[~0.65 AUC alone; complementary]*
12. Self-similarity: mean/p90 pairwise cosine of own comments, self-near-duplicate rate.
13. LLM-authorship aggregate: fraction of the account's ≥250-token comments flagged by Fast-DetectGPT at τ = 0.99. **Validate the false-positive rate on Indian-English and Hinglish comments before using.**
14. Arrival lag distribution (median, p10, fraction < 60 s / < 10 s), rank-of-arrival.
15. LLM-style markers from Ng & Carley: sentiment magnitude, expletive rate, pronoun rates, reading difficulty.

**Do not build:** anything whose only justification is a marketing blog. Several claims circulating (e.g. "a 2024 Binghamton study got 96% accuracy from posting rhythm, comment depth and subreddit diversity") appear only in SEO content with no traceable paper. I could not verify that study exists; treat it as fiction.

---

## 6. Full Source List

### Benchmarks and datasets
- **TwiBot-22: Towards Graph-Based Twitter Bot Detection** — Feng et al., NeurIPS 2022 Datasets & Benchmarks. 1M users, 139,943 bots, 92.9M nodes / 170.2M edges, 4 entity types / 14 relation types; labels via 5 experts on 1,000 users + 8 labelling functions + 7 models fused with Snorkel, 90.5% accuracy on the expert test set; **35 baselines re-implemented on 9 datasets.** https://arxiv.org/abs/2206.04564 · https://ar5iv.labs.arxiv.org/html/2206.04564 · https://dl.acm.org/doi/10.5555/3600270.3602825
- **TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark** — https://arxiv.org/abs/2106.13088
- **The Paradigm-Shift of Social Spambots** — Cresci et al., WWW 2017 Companion. Human crowdworker and detector performance on social spambots. https://dl.acm.org/doi/10.1145/3041021.3055135 · https://arxiv.org/abs/1701.03017
- **Of Bots and Humans (on Twitter)** — Gilani et al., ASONAM 2017. 4 annotators, 89% agreement, κ = 0.77; RF over 11–15 features. https://www.semanticscholar.org/paper/65f5f73e074f6d3471a7a495e35bc2510288e5d5
- **Bot Repository** — http://botometer.org/bot-repository
- **BotPercent: Estimating Bot Populations in Twitter Communities** — https://arxiv.org/abs/2302.00381

### Feature-based detectors and generalisation
- **Scalable and Generalizable Social Bot Detection through Data Selection** — Yang, Ferrara, Menczer, AAAI 2020. 20-feature RF; CV AUC 0.98; cross-dataset 0.60–0.99. https://arxiv.org/abs/1911.09179 · https://ojs.aaai.org/index.php/AAAI/article/view/5460
- **LOBO — Evaluation of Generalization Deficiencies in Twitter Bot Classifiers** — Echeverría et al., ACSAC 2018. https://arxiv.org/abs/1809.09684 · https://dl.acm.org/doi/10.1145/3274694.3274738
- **BotArtist: Generic Approach for Bot Detection in Twitter via Semi-automatic ML Pipeline** — XGBoost, 49 features; per-dataset avg F1 83.19, cross-dataset 68.5. https://arxiv.org/abs/2306.00037 · https://arxiv.org/html/2306.00037v4
- **Identification of Twitter Bots Based on an Explainable ML Framework: US 2020 Elections** — XGBoost + SHAP, 335 features. https://arxiv.org/abs/2112.04913
- **Account-history features for social bot detection in the era of large language models** — Katyal, June 2026, **arXiv preprint, single author, not peer-reviewed**. 24 behavioural features, RF AUC 0.977 on a 2,432-account balanced corpus; content AUC 0.830 → 0.466 under adversarial rewriting. https://arxiv.org/abs/2606.26127
- **A Decade of Social Bot Detection** — Cresci, CACM 2020. https://arxiv.org/abs/2007.03604
- **Social Media Bot Detection Research: Review of Literature** — Rodič, 2025; 534 → 49 papers. https://arxiv.org/abs/2503.22838

### Temporal / behavioural
- **Burstiness and memory in complex systems** — Goh & Barabási, EPL 81, 48002 (2008). B and M definitions. https://arxiv.org/abs/physics/0610233
- **Measuring burstiness for finite event sequences** — Kim & Jo, Phys. Rev. E 94, 032311 (2016). Finite-size-corrected `A_n`. https://arxiv.org/abs/1604.01125 · https://link.aps.org/doi/10.1103/PhysRevE.94.032311
- **Measuring Bot and Human Behavioral Dynamics** — Pozzana & Ferrara, Front. Phys. 8:125 (2020). Sessions with T = 60 min; **AUC 0.97 with vs 0.83 without session features.** https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2020.00125/full
- **DeBot: Twitter Bot Detection via Warped Correlation** — Chavoshi, Hamooni, Mueen, ICDM 2016. Unsupervised, precision 0.94. https://www.cs.unm.edu/~chavoshi/debot/
- **Temporal Patterns in Bot Activities** — Chavoshi et al., WWW 2017 Companion, 1601–1606. https://dl.acm.org/doi/10.1145/3041021.3051114
- **Discriminating bot accounts based solely on temporal features of microblog behavior** — Physica A 450 (2016). https://www.sciencedirect.com/science/article/abs/pii/S0378437116000388
- **Bot or Not? Deciphering Time Maps for Tweet Interarrivals** — Radziwill & Benton, 2016. Separates humans / scheduler-users / bots. https://arxiv.org/abs/1605.06555
- **DNA-influenced automated behavior detection on Twitter through relative entropy** — Sci. Rep. 12, 7654 (2022). Indian-hashtag corpus; acc 0.9457; ApEn 0.8483; Botometer 0.4898. https://www.nature.com/articles/s41598-022-11854-w
- **Lightweight Early-Warning Bot Detection on X (Twitter): Temporal Patterns and Entropy Insights** — COMPSAC 2025 (binary activity sequences, no content). https://ieeexplore.ieee.org/abstract/document/11126806/
- **RTbust: Exploiting Temporal Patterns for Botnet Detection on Twitter** — https://arxiv.org/abs/1902.04506

### Behavioural sequence encoding
- **A language framework for modeling social media account behavior (BLOC)** — Nwala, Flammini, Menczer, EPJ Data Science 12, 33 (2023). F1 0.892 with 197 features; coordination mean F1 0.659. https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-023-00410-9 · https://arxiv.org/abs/2211.00639 · code https://github.com/anwala/bloc
- **Behavior Change as a Signal for Identifying Social Media Manipulation** — Ariyarathne, Ariyarathne, Flammini, Menczer, Nwala; ACM WebSci 2026. https://arxiv.org/abs/2603.03128 · https://dl.acm.org/doi/10.1145/3795766.3799748
- **Social Fingerprinting: Detection of Spambot Groups Through DNA-Inspired Behavioral Modeling** — Cresci et al. https://arxiv.org/abs/1703.04482

### Sockpuppets, coordination and influence operations
- **An Army of Me: Sockpuppets in Online Discussion Communities** — Kumar et al., WWW 2017. **Account AUC 0.68 / pair AUC 0.91.** https://arxiv.org/abs/1703.07355 · https://cs.stanford.edu/~srijan/pubs/sockpuppets-www2017.pdf
- **Unmasking the Web of Deceit: Uncovering Coordinated Activity to Expose Information Operations on Twitter** — 5 similarity traces; supervised AUC 0.94 / F1 0.82 / P 0.96; unsupervised AUC 0.83–0.84. https://arxiv.org/abs/2310.09884
- **Detection and Characterization of Coordinated Online Behavior: A Survey** — co-action taxonomy, time-window types, thresholding and clustering practice. https://arxiv.org/abs/2408.01257
- **Unsupervised detection of coordinated information operations in the wild** — EPJ Data Science 2025. PR-AUC lift 76×–580×. https://link.springer.com/article/10.1140/epjds/s13688-025-00544-y
- **Ephemeral Astroturfing Attacks: The Case of Fake Twitter Trends** — Elmas et al. Precision 1.000 / recall 0.989 / F1 0.994. https://arxiv.org/abs/1910.07783
- **Sockpuppet Detection in Wikipedia: A Corpus of Real-World Deceptive Writing** — Solorio et al., LREC 2014. http://www.lrec-conf.org/proceedings/lrec2014/pdf/1007_Paper.pdf
- **Detecting Sockpuppetry on Wikipedia Using Meta-Learning** — ACL 2025. https://aclanthology.org/2025.acl-long.1083.pdf
- **Social Media Identity Deception Detection: A Survey** — https://arxiv.org/abs/2103.04673

### Enforcement / moderation prediction
- **Antisocial Behavior in Online Discussion Communities** — Cheng, Danescu-Niculescu-Mizil, Leskovec, ICWSM 2015. **>0.80 AUC from first 5–10 posts.** https://arxiv.org/abs/1504.00680 · https://cs.stanford.edu/people/jure/pubs/trolls-icwsm15.pdf
- **Identifying Effective Signals to Predict Deleted and Suspended Accounts on Twitter Across Languages** — ICWSM 2022. Best category F1 0.88 / ROC-AUC 0.95 initial; profile features F1 0.79 / ROC-AUC 0.90 long-term. https://ojs.aaai.org/index.php/ICWSM/article/view/14874
- **The false positive problem of automatic bot detection in social science research** — Rauchfleisch & Kaiser, PLOS ONE 15(10) 2020. **41–76% false positives; AUC 0.69 German vs 0.90 English; 27.2% threshold instability over 3 months.** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045
- **Textual Analysis and Timely Detection of Suspended Social Media Accounts** — ICWSM 2021. https://cdn.aaai.org/ojs/18091/18091-28-21586-1-2-20210521.pdf
- **Russo-Ukrainian War: Prediction and explanation of Twitter suspension** — https://arxiv.org/abs/2306.03502

### Reddit-specific
- **TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit** — 9 features, RF, CV F1 0.978, **operational corroboration 66%**, seed set grown >300%. https://arxiv.org/abs/2112.00443
- **BotBuster: Multi-Platform Bot Detection Using a Mixture of Experts** — Ng & Carley, ICWSM 2023. **Reddit F1 60.04 / 69.77**; Twitter avg F1 73.54; stabilises after **36 posts**. https://arxiv.org/abs/2207.13658 · https://ojs.aaai.org/index.php/ICWSM/article/view/22179
- **Assembling a Multi-Platform Ensemble Social Bot Detector (BotBuster For Everyone)** — **reddit-2022: accuracy 34.68%, macro-F1 33.86.** https://arxiv.org/abs/2401.14607 · https://arxiv.org/html/2401.14607v2
- **Bot Detection in Reddit Political Discussion** — Hurtado, Ray, Marculescu, 2019. https://dl.acm.org/doi/pdf/10.1145/3313294.3313386
- **Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit** — La Cava, Aiello, Tagarelli, 2025. Fast-DetectGPT, τ=0.99, ≥250 tokens, 51 subreddits, 9M comments; MGT peaks 6.3–8.5%; **engagement ≥ human**. https://arxiv.org/abs/2510.07226
- **Cross-Subreddit Behavior as Open-Source Indicators of Coordinated Influence: r/Sino & r/China** — 2025, exploratory, no ground truth; 51/63 users flagged on "low lexical diversity". https://arxiv.org/abs/2507.16857
- **Throwaway Accounts and Moderation on Reddit** — https://arxiv.org/abs/2501.17430
- **Arctic Shift** (Pushshift successor) — https://github.com/ArthurHeitmann/arctic_shift
- **r/ChangeMyView covert LLM experiment** — https://www.404media.co/researchers-secretly-ran-a-massive-unauthorized-ai-persuasion-experiment-on-reddit-users/ · analysis https://arxiv.org/abs/2606.05256

### Content / LLM era
- **Lexical analysis of automated accounts on Twitter** — Inuwa-Dutse et al., 2018. TTR/lexical diversity/contractions/emoticons; lexical-only AUC 0.65–0.87; F+L 0.95 vs F 0.71. https://arxiv.org/abs/1812.07947
- **Are LLM-Powered Social Media Bots Realistic?** — Ng & Carley, 2025. Cue-level comparison of LLM bots vs wild bots vs wild humans. https://arxiv.org/abs/2508.00998
- **Human Perception of LLM-generated Text Content in Social Media Environments** — 42% accuracy, 49% FN. https://arxiv.org/abs/2409.06653
- **A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions** — Computational Linguistics 51(1), 2025. https://aclanthology.org/2025.cl-1.8.pdf · https://direct.mit.edu/coli/article/51/1/275/127462
- **Fast-DetectGPT** — zero-shot MGT detection via conditional probability curvature (the detector used at Reddit scale).
- **DetectRL / Beemo / MultiSocial** — adversarial and multilingual MGT benchmarks. https://arxiv.org/abs/2411.04032 · https://arxiv.org/abs/2406.12549
- **In Defense of MinHash Over SimHash** — near-duplicate detection primitives. https://arxiv.org/abs/1407.4416
- **Authorship Attribution in the Era of LLMs** — https://arxiv.org/abs/2408.08946
- **Few-shot Authorship Attribution in English Reddit Posts** — https://openreview.net/pdf?id=IuIP-BWC88W

### Other detectors referenced
- BotSSCL (self-supervised contrastive) — https://www.sciencedirect.com/science/article/pii/S2468696425000199
- LGB (LM + GNN) — https://arxiv.org/abs/2406.08762
- MSM-BD (multimodal, TwiBot-22 acc 0.8002 / F1 0.6105) — https://arxiv.org/abs/2501.00204
- Deep Neural Networks for Bot Detection (Kudugunta & Ferrara) — https://arxiv.org/abs/1802.04289
- Botometer 101 — https://arxiv.org/abs/2201.01608
- Adversarial Botometer — https://arxiv.org/abs/2405.02016
