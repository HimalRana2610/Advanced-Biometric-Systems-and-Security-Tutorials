# AI461 — Assignment 1: User Authentication using Biometric Features

Performance evaluation of a 100-user biometric verification system under two matching
criteria — **Euclidean distance** and **cosine similarity** — covering the genuine and
impostor distributions, FAR/FRR curves, the ROC, the Equal Error Rate and the
decidability index.

---

## Headline result

| Metric | Euclidean distance | Cosine similarity | Winner |
|---|---:|---:|:---:|
| **Equal Error Rate (EER)** | 11.511 % | **8.400 %** | cosine |
| **Decidability index (d′)** | 1.7619 | **1.9254** | cosine |
| ROC AUC | 0.949556 | **0.968943** | cosine |
| TAR @ FAR = 1 % | 66.20 % | **83.60 %** | cosine |
| TAR @ FAR = 0.1 % | 54.60 % | **74.20 %** | cosine |
| Rank-1 identification *(supplementary)* | 87.80 % | **90.80 %** | cosine |

**Cosine similarity is the better matching criterion on this data** — see
[Which criterion performs best?](#which-matching-criterion-performs-best) for why.

---

## Running the code

```bash
python biometric_evaluation.py                 # A-F: all figures and metrics
python biometric_evaluation.py --diagnostics   # additionally: why cosine wins (§5)
```

Requirements: Python 3.9+, `numpy`, `matplotlib`. No other dependencies, no RNG
anywhere, so the run is fully deterministic and reproduces the numbers below exactly.
The script recreates `figures/` and `results/` from scratch on every run; every number
quoted in this README is taken from that output, not typed by hand.

## Repository layout

```
Data.csv                     raw feature database (provided)
biometric_evaluation.py      complete implementation (single file)
README.md                    this file
figures/                     all generated plots (PNG)
  A_genuine_distribution.png     A. genuine distribution, one panel per criterion
  B_impostor_distribution.png    B. impostor distribution, one panel per criterion
  AB_score_distributions.png     A+B overlaid — the separability picture
  C_FAR_plot.png                 C. FAR vs decision threshold
  D_FRR_plot.png                 D. FRR vs decision threshold
  CD_FAR_FRR_combined.png        C+D on one axis (log), EER crossing marked
  E_ROC_curves.png               E. ROC, logarithmic and linear FAR axes
  F_EER_decidability.png         F. EER crossing + d′ separation
results/
  metrics.txt                  human-readable summary report
  metrics.csv / metrics.json   the same numbers, machine-readable
  diagnostics.txt              magnitude-channel analysis (--diagnostics only)
  far_frr_euclidean.csv        threshold, FAR, FRR (2000-point sweep)
  far_frr_cosine.csv           threshold, FAR, FRR (2000-point sweep)
  scores_euclidean.npz         raw genuine / impostor score arrays
  scores_cosine.npz            raw genuine / impostor score arrays
```

---

## 1. Reading the data

`Data.csv` is **tab-separated with no header**, and it stores the database
**transposed**: 144 rows (features) × 1000 columns (samples). Every line also ends
with a trailing tab, which `numpy.genfromtxt` turns into a 1001st all-NaN column.
`load_data()` drops that artefact column, transposes to the conventional
`(n_samples × n_features) = (1000 × 144)` orientation, and asserts the shape.

### Verifying the column ordering

The assignment states 100 users × 10 samples but not how the 1000 columns are ordered.
Two orderings are plausible:

* **user-major** — columns 0–9 are user 1, columns 10–19 are user 2, …
* **sample-major** — column *i* belongs to user *i* mod 100 (all users' sample 1 first)

Rather than assume, `verify_layout()` tests both by computing the ratio

```
mean within-user distance / mean between-user distance
```

A correct grouping clusters, giving a ratio well below 1; a wrong grouping is
indistinguishable from random labels, giving a ratio near 1.

| Hypothesis | Ratio |
|---|---:|
| **user-major (adopted)** | **0.4422** |
| sample-major (rejected) | 1.0098 |

The user-major layout is adopted, and the check runs on every execution so the
assumption is visible in the output rather than buried in the code.

## 2. Enrollment / test split and template generation

Per the assignment flowchart, for each of the 100 users:

* **samples 1–5** → enrollment (training) set
* **samples 6–10** → test set

The five enrollment vectors of a user are **averaged into a single 144-dimensional
template**, giving 100 templates. The five test samples of each user stay separate,
giving 500 test probes. The enrollment samples are never scored — all reported
numbers are test-set numbers.

## 3. Matching and score partitioning

Every test probe is matched against **all 100 templates**, producing a 500 × 100
score matrix per criterion:

* **Euclidean distance** — `‖probe − template‖₂`; a **dissimilarity** (smaller = more alike)
* **Cosine similarity** — `(p·t) / (‖p‖‖t‖)`; a **similarity** (larger = more alike)

The 50 000 comparisons split into:

| | count | definition |
|---|---:|---|
| **Genuine** | 500 | probe matched against *its own* user's template |
| **Impostor** | 49 500 | probe matched against the other 99 users' templates |

500 + 49 500 = 50 000 ✓ — exactly the counts given in the assignment flowchart.

### Score polarity is handled explicitly, not by conversion

Euclidean distance and cosine similarity run in **opposite directions**. Rather than
force cosine into a distance (e.g. `1 − cos`) and plot it in artificial units, the code
carries a `similarity` flag through the whole pipeline and applies the correct decision
rule for each:

| Criterion | Accept when | FAR | FRR |
|---|---|---|---|
| Euclidean distance | `score ≤ threshold` | `P(impostor ≤ t)` | `P(genuine > t)` |
| Cosine similarity | `score ≥ threshold` | `P(impostor ≥ t)` | `P(genuine < t)` |

Every cosine plot is therefore in native cosine-similarity units, which is why the
cosine FAR curve *falls* with the threshold while the Euclidean FAR curve *rises*.

---

## 4. How each required quantity is computed

### A & B — Genuine and impostor distributions

Density-normalised histograms (`density=True`) so the 500 genuine and 49 500 impostor
scores are directly comparable on one axis, with a fitted Gaussian overlay and the
reported μ and σ. `figures/AB_score_distributions.png` overlays both on shared bins —
the overlap region between the two histograms *is* the error the system will make.

### C & D — FAR and FRR

Computed **exactly** by sorted-array counting rather than by looping over thresholds:
both score sets are sorted once, and `np.searchsorted` returns the count on either side
of every threshold in O(log n). Two threshold sets are used:

* a **2000-point uniform sweep** across the full score range, for the threshold plots
  (C, D) and the exported CSVs — a smooth, evenly sampled curve;
* the **exact set of observed scores** (each nudged by one ULP via `np.nextafter` so
  both sides of every step are represented), for the ROC and EER — this makes the
  empirical curve exact rather than a grid approximation.

### E — Receiver Operating Characteristic

TAR (= 1 − FRR) against FAR, plotted on both a **logarithmic** FAR axis (the standard
in biometrics — it resolves the low-FAR operating region where a real system is
deployed) and a **linear** axis with the chance diagonal for reference. AUC is the
trapezoidal integral over the exact curve.

### F — Equal Error Rate and decidability index

**EER.** FAR and FRR are step functions, so an exact FAR = FRR crossing rarely lands on
a sampled threshold. `equal_error_rate()` brackets the crossing by the sign change of
`FAR − FRR`, **linearly interpolates inside that bracket**, and reports the mean of the
interpolated FAR and FRR together with the operating threshold. (Simply taking the
`argmin |FAR − FRR|` grid point is the cruder alternative and is only used as a
fallback if no sign change exists.)

**Decidability index.** Daugman's

```
d′ = |μ_genuine − μ_impostor| / sqrt((σ²_genuine + σ²_impostor) / 2)
```

with sample variances (`ddof=1`). d′ is invariant to any **affine** rescaling of the
score axis, which is what makes it meaningful across two criteria whose scores live on
completely different scales (hundreds of units vs. the interval [0, 1]).

It is *not*, however, invariant to a nonlinear monotone remapping, and this data set
contains a clean illustration: Euclidean matching on L2-normalised vectors has a ROC
and EER **identical** to cosine similarity (they are related by the monotone map
`d = √(2 − 2·cos)`), yet its d′ is 2.4770 against cosine's 1.9254. d′ should therefore
be read alongside the rank-based EER and AUC, which are invariant to any monotone
transform, rather than on its own. All three agree on the ranking here.

---

## 5. Results in detail

**Euclidean distance** — genuine μ = 291.62, σ = 135.60; impostor μ = 703.82, σ = 301.79

* EER **11.511 %** at threshold 429.24
* d′ **1.7619**, AUC 0.9496

**Cosine similarity** — genuine μ = 0.98950, σ = 0.010968; impostor μ = 0.96038, σ = 0.018367

* EER **8.400 %** at threshold 0.9790
* d′ **1.9254**, AUC 0.9689

### Which matching criterion performs best?

**Cosine similarity.** It wins on every measure computed — lower EER (8.40 % vs
11.51 %, a **27 % relative reduction in error**), higher decidability (1.93 vs 1.76),
higher AUC, and a substantially better low-FAR operating region, which is where the
difference actually matters: at FAR = 0.1 % cosine accepts **74.2 %** of genuine users
against Euclidean's **54.6 %**.

#### Why — the magnitude channel, demonstrated rather than asserted

The two criteria differ in exactly one thing: Euclidean distance sees both the
**magnitude** and the **direction** of a feature vector, cosine similarity sees only
the direction. The diagnostics (`python biometric_evaluation.py --diagnostics`,
written to `results/diagnostics.txt`) isolate that one difference and show it accounts
for the entire gap.

**Step 1 — magnitude is unstable across the six-month gap.** Per-user feature-vector
norms drift between the enrollment and test halves, with the test/enrollment mean-norm
ratio spanning **0.825 to 1.154** (sd 5.7 %). Within-user norm variation has a
coefficient of variation of 0.0535.

**Step 2 — the Euclidean score is largely a magnitude score.** Across the 49 500
impostor comparisons, the Euclidean distance correlates **0.90** with the plain
absolute difference of the two vectors' norms (0.58 on the genuine side). That is what
produces the long, heavy right tail of the Euclidean impostor distribution
(σ = 301.8, more than twice the genuine σ of 135.6) visible in
`figures/AB_score_distributions.png`.

**Step 3 — removing magnitude closes the gap exactly.** For L2-normalised vectors,
`‖p̂ − t̂‖² = 2 − 2·cos(p, t)` identically (verified to 2.2 × 10⁻¹⁵, i.e. machine
precision), so Euclidean matching on normalised vectors is a strictly monotone
relabelling of cosine similarity and must give the same ROC. It does:

| Matcher | EER | AUC | d′ |
|---|---:|---:|---:|
| Euclidean distance (raw) | 11.5111 % | 0.94955624 | 1.7619 |
| Cosine similarity | 8.4000 % | 0.96894323 | 1.9254 |
| Euclidean on unit-norm vectors | **8.4000 %** | **0.96894323** | 2.4770 |

L2-normalising before Euclidean matching reproduces the cosine EER and AUC to every
digit. **The whole 11.51 % → 8.40 % improvement is therefore attributable to
discarding the magnitude channel, and to nothing else** — not to any other property of
the cosine function.

So cosine wins because the six-month magnitude drift is a large source of *within*-user
variation that Euclidean distance charges in full — inflating genuine distances and
pushing the two distributions together — while the *direction* of the feature vector
is the stabler and more discriminative channel.

#### Two qualifications

**Magnitude is not pure noise.** Norms vary about 3.6× more across users than within
one (CV 0.191 vs 0.0535), and the absolute norm difference used *alone* as a matcher
still reaches d′ = 1.16. Cosine therefore throws away genuinely informative signal — it
wins because the direction channel is so much stronger that the trade pays off, not
because magnitude is worthless. A matcher that *weighted* the two channels instead of
keeping one and discarding the other could plausibly beat both.

**Neither criterion is strong in absolute terms.** An 8.4 % EER is high for a deployed
biometric system. Both matchers are simple, untrained, unnormalised nearest-template
comparisons on raw 144-dimensional features; per-dimension normalisation (z-score or
min-max), a learned metric (LDA, Mahalanobis), or score normalisation would very likely
improve both. The comparison the assignment asks for is a like-for-like one between the
two distance criteria, and on that comparison cosine similarity is clearly better.

---

## 6. Implementation notes

* **Single file, no hidden state.** `biometric_evaluation.py` runs end to end in a few
  seconds; every stage (`load_data` → `split_and_enroll` → scoring → `evaluate` →
  plotting → `write_reports`) is a separate function with an explicit contract.
* **Vectorised throughout.** All 50 000 comparisons are computed with broadcasting; the
  FAR/FRR sweep over 2000 thresholds is done by binary search, not by nested loops.
* **Assumptions are tested, not assumed.** The column ordering is verified numerically
  (§1) and the shape, NaN and comparison-count checks all raise rather than silently
  proceeding on malformed input.
* **Claims are reproducible.** The explanation of *why* cosine wins is not narrative —
  `magnitude_diagnostics()` computes it, and `--diagnostics` prints the supporting
  numbers to `results/diagnostics.txt`.
* **Plot design.** Two-series categorical palette (blue `#2a78d6` / orange `#eb6834`)
  validated for colour-vision-deficiency separation, so genuine vs. impostor and
  FAR vs. FRR stay distinguishable in print and for colour-blind readers; every series
  is also named in a legend or a direct label, so identity never rests on colour alone.
* **Everything is exported.** The raw score arrays are saved to `results/*.npz` and the
  full FAR/FRR tables to `results/*.csv`, so any number in this README can be
  re-derived without re-running the matcher.
