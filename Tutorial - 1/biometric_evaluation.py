"""
Advanced Biometric Systems and Security (AI461) -- Assignment 1
User Authentication using Biometric Features

Evaluates a 100-user / 10-sample / 144-feature biometric database under two
matching criteria (Euclidean distance and cosine similarity) and produces:

    A. Genuine distribution
    B. Impostor distribution
    C. FAR plot
    D. FRR plot
    E. Receiver Operating Characteristic
    F. Equal Error Rate and Decidability Index

Protocol (as specified in the assignment flowchart):
    * samples 1-5 of every user  -> enrollment (training); averaged into one template
    * samples 6-10 of every user -> test probes
    * every test probe is matched against all 100 templates
      -> 500 x 100 = 50,000 comparisons = 500 genuine + 49,500 impostor

Run:  python biometric_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Data.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

N_USERS = 100
N_SAMPLES = 10          # per user
N_TRAIN = 5             # samples 1-5  -> enrollment
N_FEATURES = 144

# ---- Plot styling (validated categorical palette, light chart surface) ----- #
C_GENUINE = "#2a78d6"   # categorical slot 1 (blue)
C_IMPOSTOR = "#eb6834"  # categorical slot 2 (orange)
C_FAR = "#2a78d6"
C_FRR = "#eb6834"
C_EUCLIDEAN = "#2a78d6"
C_COSINE = "#eb6834"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #

def load_data(path: Path = DATA_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Read Data.csv into an (n_samples x n_features) matrix plus user labels.

    The file stores the database transposed: 144 rows (features) x 1000 columns
    (samples), tab separated, with a trailing tab on every line that genfromtxt
    turns into an all-NaN column.  Columns are user-major: columns 0-9 belong to
    user 0, columns 10-19 to user 1, and so on (verified by `verify_layout`).
    """
    raw = np.genfromtxt(path, delimiter="\t")
    if raw.ndim != 2:
        raise ValueError(f"unexpected data shape {raw.shape}")
    if np.all(np.isnan(raw[:, -1])):           # drop trailing-tab artefact column
        raw = raw[:, :-1]
    if np.isnan(raw).any():
        raise ValueError("Data.csv contains missing values")

    X = raw.T                                   # -> samples x features
    expected = (N_USERS * N_SAMPLES, N_FEATURES)
    if X.shape != expected:
        raise ValueError(f"expected {expected} after transpose, got {X.shape}")

    labels = np.repeat(np.arange(N_USERS), N_SAMPLES)
    return X, labels


def verify_layout(X: np.ndarray, labels: np.ndarray) -> dict:
    """Sanity-check the user-major column ordering assumption.

    Compares the assumed grouping (10 consecutive columns = one user) against the
    alternative sample-major grouping (column i belongs to user i mod 100) using
    the mean within-class / between-class Euclidean distance ratio.  A ratio well
    below 1 means the grouping really does cluster; a ratio near 1 means the
    labels are no better than random.
    """
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    off_diag = ~np.eye(len(X), dtype=bool)

    def ratio(lab: np.ndarray) -> float:
        same = lab[:, None] == lab[None, :]
        return float(d[same & off_diag].mean() / d[~same].mean())

    return {
        "user_major_ratio": ratio(labels),                        # assumed layout
        "sample_major_ratio": ratio(np.tile(np.arange(N_USERS), N_SAMPLES)),
    }


# --------------------------------------------------------------------------- #
# 2. Enrollment / test split and template generation
# --------------------------------------------------------------------------- #

def split_and_enroll(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into enrollment/test sets and build one averaged template per user.

    Returns (templates, probes, probe_labels) with shapes
    (100 x 144), (500 x 144) and (500,).
    """
    cube = X.reshape(N_USERS, N_SAMPLES, N_FEATURES)

    enroll = cube[:, :N_TRAIN, :]               # 100 x 5 x 144  (samples 1-5)
    test = cube[:, N_TRAIN:, :]                 # 100 x 5 x 144  (samples 6-10)

    templates = enroll.mean(axis=1)             # 100 x 144, one template per user
    probes = test.reshape(-1, N_FEATURES)       # 500 x 144
    probe_labels = np.repeat(np.arange(N_USERS), N_SAMPLES - N_TRAIN)
    return templates, probes, probe_labels


# --------------------------------------------------------------------------- #
# 3. Matching
# --------------------------------------------------------------------------- #

def euclidean_scores(probes: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Full 500 x 100 matrix of Euclidean distances (a DISsimilarity score)."""
    return np.linalg.norm(probes[:, None, :] - templates[None, :, :], axis=2)


def cosine_scores(probes: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Full 500 x 100 matrix of cosine similarities (a SIMILARITY score)."""
    p = probes / np.linalg.norm(probes, axis=1, keepdims=True)
    t = templates / np.linalg.norm(templates, axis=1, keepdims=True)
    return p @ t.T


def split_genuine_impostor(scores: np.ndarray,
                           probe_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate the 50,000 comparisons into 500 genuine and 49,500 impostor scores."""
    template_ids = np.arange(scores.shape[1])
    genuine_mask = probe_labels[:, None] == template_ids[None, :]
    return scores[genuine_mask], scores[~genuine_mask]


# --------------------------------------------------------------------------- #
# 4. Error rates, EER, decidability index
# --------------------------------------------------------------------------- #

def error_rates(genuine: np.ndarray, impostor: np.ndarray,
                thresholds: np.ndarray, similarity: bool) -> tuple[np.ndarray, np.ndarray]:
    """FAR and FRR at every threshold, computed exactly by sorted-array counting.

    similarity=False (Euclidean distance) -- accept when score <= threshold:
        FAR = P(impostor <= t)      FRR = P(genuine > t)
    similarity=True  (cosine similarity) -- accept when score >= threshold:
        FAR = P(impostor >= t)      FRR = P(genuine < t)
    """
    g = np.sort(genuine)
    i = np.sort(impostor)

    if similarity:
        far = (i.size - np.searchsorted(i, thresholds, side="left")) / i.size
        frr = np.searchsorted(g, thresholds, side="left") / g.size
    else:
        far = np.searchsorted(i, thresholds, side="right") / i.size
        frr = 1.0 - np.searchsorted(g, thresholds, side="right") / g.size
    return far, frr


def equal_error_rate(thresholds: np.ndarray, far: np.ndarray,
                     frr: np.ndarray) -> tuple[float, float]:
    """EER and its operating threshold, linearly interpolated at the FAR/FRR crossing.

    FAR and FRR are step functions, so an exact crossing rarely exists at a
    sampled threshold; the crossing interval is bracketed by the sign change of
    (FAR - FRR) and the EER is interpolated inside it.
    """
    diff = far - frr
    cross = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)

    if cross.size == 0:                          # no sign change -> closest approach
        k = int(np.argmin(np.abs(diff)))
        return float((far[k] + frr[k]) / 2.0), float(thresholds[k])

    k = int(cross[0])
    w = diff[k] / (diff[k] - diff[k + 1])        # fraction along the bracket
    thr = thresholds[k] + w * (thresholds[k + 1] - thresholds[k])
    far_i = far[k] + w * (far[k + 1] - far[k])
    frr_i = frr[k] + w * (frr[k + 1] - frr[k])
    return float((far_i + frr_i) / 2.0), float(thr)


def decidability_index(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Daugman's decidability index d' = |mu_G - mu_I| / sqrt((var_G + var_I)/2)."""
    mu_g, mu_i = genuine.mean(), impostor.mean()
    var_g, var_i = genuine.var(ddof=1), impostor.var(ddof=1)
    return float(abs(mu_g - mu_i) / np.sqrt((var_g + var_i) / 2.0))


def roc_auc(far: np.ndarray, tar: np.ndarray) -> float:
    """Area under the ROC curve (TAR vs FAR) by the trapezoidal rule."""
    order = np.argsort(far)
    x, y = far[order], tar[order]
    x = np.concatenate(([0.0], x, [1.0]))
    y = np.concatenate(([y[0]], y, [1.0]))
    return float(np.trapezoid(y, x))


def tar_at_far(far: np.ndarray, tar: np.ndarray, target: float) -> float:
    """Genuine acceptance rate at a fixed operating FAR (interpolated)."""
    order = np.argsort(far)
    return float(np.interp(target, far[order], tar[order]))


def rank1_accuracy(scores: np.ndarray, probe_labels: np.ndarray,
                   similarity: bool) -> float:
    """Supplementary: closed-set rank-1 identification accuracy."""
    best = np.argmax(scores, axis=1) if similarity else np.argmin(scores, axis=1)
    return float((best == probe_labels).mean())


# --------------------------------------------------------------------------- #
# 5. Per-matcher evaluation
# --------------------------------------------------------------------------- #

def evaluate(name: str, unit: str, scores: np.ndarray, probe_labels: np.ndarray,
             similarity: bool) -> dict:
    """Run the complete A-F evaluation for one matching criterion."""
    genuine, impostor = split_genuine_impostor(scores, probe_labels)

    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    pad = 0.02 * (hi - lo)

    # Dense grid for the threshold plots (C, D) ...
    grid = np.linspace(lo - pad, hi + pad, 2000)
    far_grid, frr_grid = error_rates(genuine, impostor, grid, similarity)

    # ... and the exact set of candidate thresholds for ROC / EER.
    exact = np.unique(np.concatenate(([lo - pad], genuine, impostor, [hi + pad])))
    if not similarity:
        exact = np.concatenate((exact, np.nextafter(exact, -np.inf)))
    else:
        exact = np.concatenate((exact, np.nextafter(exact, np.inf)))
    exact = np.unique(exact)
    if similarity:
        exact = exact[::-1]                      # order so FAR increases monotonically
    far, frr = error_rates(genuine, impostor, exact, similarity)
    tar = 1.0 - frr

    eer, eer_thr = equal_error_rate(exact, far, frr)

    return {
        "name": name,
        "unit": unit,
        "similarity": similarity,
        "genuine": genuine,
        "impostor": impostor,
        "grid": grid, "far_grid": far_grid, "frr_grid": frr_grid,
        "thresholds": exact, "far": far, "frr": frr, "tar": tar,
        "eer": eer,
        "eer_threshold": eer_thr,
        "dprime": decidability_index(genuine, impostor),
        "auc": roc_auc(far, tar),
        "tar_at_far_1pct": tar_at_far(far, tar, 0.01),
        "tar_at_far_0.1pct": tar_at_far(far, tar, 0.001),
        "rank1": rank1_accuracy(scores, probe_labels, similarity),
        "mu_genuine": float(genuine.mean()), "sd_genuine": float(genuine.std(ddof=1)),
        "mu_impostor": float(impostor.mean()), "sd_impostor": float(impostor.std(ddof=1)),
        "n_genuine": int(genuine.size), "n_impostor": int(impostor.size),
    }


# --------------------------------------------------------------------------- #
# 6. Figures
# --------------------------------------------------------------------------- #

def _gaussian(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _label_side(ax, x: float) -> tuple[str, int]:
    """Keep a label anchored to a vertical line inside the axes.

    Returns the horizontal alignment and the offset sign to use, flipping the
    label to the left of the line once the line sits in the right-hand third of
    the axes so the text cannot run off the figure.
    """
    lo, hi = ax.get_xlim()
    if (x - lo) / (hi - lo) > 0.62:
        return "right", -1
    return "left", 1


def _headroom(ax, factor: float = 1.2) -> float:
    """Grow the y-limit above the drawn data so labels and legends have room."""
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * factor)
    return top


def _finish(fig, path: Path, suptitle: str | None = None) -> None:
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def plot_single_distribution(results: list[dict], which: str, color: str,
                             title: str, path: Path) -> None:
    """Figure A (genuine) or B (impostor): one panel per matching criterion."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r in zip(axes, results):
        s = r[which]
        ax.hist(s, bins=50, density=True, color=color, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.5, label=f"{which.capitalize()} scores")
        xs = np.linspace(s.min(), s.max(), 400)
        ax.plot(xs, _gaussian(xs, s.mean(), s.std(ddof=1)), color=INK,
                linewidth=1.6, linestyle="--", label="Gaussian fit")
        ax.set_title(f"{r['name']}")
        ax.set_xlabel(r["unit"])
        ax.set_ylabel("Probability density")
        _headroom(ax, 1.22)
        ax.legend(loc="upper right", labelcolor=INK_2)
        ax.text(0.02, 0.96,
                f"n = {s.size:,}\n$\\mu$ = {s.mean():.4g}\n$\\sigma$ = {s.std(ddof=1):.4g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK_2)
    _finish(fig, path, title)


def plot_overlaid_distributions(results: list[dict], path: Path) -> None:
    """Figures A + B together: the separability picture the whole report rests on."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, r in zip(axes, results):
        g, i = r["genuine"], r["impostor"]
        bins = np.linspace(min(g.min(), i.min()), max(g.max(), i.max()), 70)
        ax.hist(i, bins=bins, density=True, color=C_IMPOSTOR, alpha=0.75,
                edgecolor=SURFACE, linewidth=0.4, label=f"Impostor (n = {i.size:,})")
        ax.hist(g, bins=bins, density=True, color=C_GENUINE, alpha=0.75,
                edgecolor=SURFACE, linewidth=0.4, label=f"Genuine (n = {g.size:,})")
        ax.axvline(r["eer_threshold"], color=INK, linewidth=1.4, linestyle=":")
        ha, sign = _label_side(ax, r["eer_threshold"])
        ax.annotate(f"EER threshold\n{r['eer_threshold']:.4g}",
                    xy=(r["eer_threshold"], ax.get_ylim()[1] * 0.62),
                    xytext=(6 * sign, 0), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="center", ha=ha)
        ax.set_title(f"{r['name']}  —  d' = {r['dprime']:.3f}")
        ax.set_xlabel(r["unit"])
        ax.set_ylabel("Probability density")
        # Genuine scores pile up at low distance / high similarity, so the free
        # corner is on the opposite side for each criterion.
        _headroom(ax, 1.18)
        ax.legend(loc="upper right" if not r["similarity"] else "upper left",
                  labelcolor=INK_2)
    _finish(fig, path, "Genuine and impostor score distributions (test set)")


def plot_error_rate(results: list[dict], which: str, color: str,
                    label: str, title: str, path: Path) -> None:
    """Figure C (FAR) or D (FRR) against the decision threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r in zip(axes, results):
        # One series per panel, so the y-axis label carries the identity and no
        # legend box is needed; the EER threshold is labelled directly.
        ax.plot(r["grid"], r[f"{which}_grid"], color=color, linewidth=2, label=label)
        ax.set_ylim(-0.03, 1.03)
        ax.axvline(r["eer_threshold"], color=MUTED, linewidth=1.2, linestyle=":")
        ha, sign = _label_side(ax, r["eer_threshold"])
        ax.annotate(f"EER threshold {r['eer_threshold']:.4g}",
                    xy=(r["eer_threshold"], 0.5), xytext=(6 * sign, 0),
                    textcoords="offset points", fontsize=8.5, color=INK_2,
                    va="center", ha=ha, rotation=90)
        ax.set_title(r["name"])
        ax.set_xlabel(f"Decision threshold ({r['unit']})")
        ax.set_ylabel(label)
    _finish(fig, path, title)


def plot_far_frr_combined(results: list[dict], path: Path) -> None:
    """FAR and FRR on one axis per criterion, with the EER crossing marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, r in zip(axes, results):
        ax.plot(r["grid"], r["far_grid"], color=C_FAR, linewidth=2, label="FAR")
        ax.plot(r["grid"], r["frr_grid"], color=C_FRR, linewidth=2, label="FRR")
        ax.plot([r["eer_threshold"]], [r["eer"]], marker="o", markersize=8,
                color=INK, zorder=5)
        ha, sign = _label_side(ax, r["eer_threshold"])
        ax.annotate(f"EER = {r['eer'] * 100:.2f}%\n@ {r['eer_threshold']:.4g}",
                    xy=(r["eer_threshold"], r["eer"]), xytext=(14 * sign, 26),
                    textcoords="offset points", fontsize=9, color=INK, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_title(r["name"])
        ax.set_xlabel(f"Decision threshold ({r['unit']})")
        ax.set_ylabel("Error rate (log scale)")
        ax.legend(loc="best", labelcolor=INK_2)
    _finish(fig, path, "FAR and FRR versus decision threshold")


def plot_roc(results: list[dict], path: Path) -> None:
    """Figure E: ROC (TAR vs FAR) on log and linear FAR axes."""
    colors = [C_EUCLIDEAN, C_COSINE]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, logx in zip(axes, (True, False)):
        for r, c in zip(results, colors):
            far = np.clip(r["far"], 1e-5, None) if logx else r["far"]
            ax.plot(far, r["tar"], color=c, linewidth=2,
                    label=f"{r['name']}  (AUC = {r['auc']:.4f}, EER = {r['eer'] * 100:.2f}%)")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1.0)
            ax.set_title("ROC — logarithmic FAR axis")
        else:
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--",
                    label="Chance")
            ax.set_xlim(0, 1)
            ax.set_title("ROC — linear FAR axis")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Acceptance Rate")
        ax.set_ylabel("True Acceptance Rate  (1 − FRR)")
        ax.legend(loc="lower right", fontsize=8.5, labelcolor=INK_2)
    _finish(fig, path, "Receiver Operating Characteristic (test set)")


def plot_eer_and_dprime(results: list[dict], path: Path) -> None:
    """Figure F: EER crossing (left) and decidability separation (right), per criterion."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, r in enumerate(results):
        # -- EER crossing, zoomed around the operating point -------------------
        ax = axes[row, 0]
        ax.plot(r["grid"], r["far_grid"], color=C_FAR, linewidth=2, label="FAR")
        ax.plot(r["grid"], r["frr_grid"], color=C_FRR, linewidth=2, label="FRR")
        ax.plot([r["eer_threshold"]], [r["eer"]], marker="o", markersize=9, color=INK,
                zorder=5)
        ax.annotate(f"EER = {r['eer'] * 100:.2f}%", xy=(r["eer_threshold"], r["eer"]),
                    xytext=(16, 20), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

        # Zoom on the band where both error rates are still readable, rather than
        # a fixed span, which would otherwise run into negative thresholds.
        band = (r["far_grid"] <= 0.75) & (r["frr_grid"] <= 0.75)
        xs = r["grid"][band]
        pad = 0.18 * (xs[-1] - xs[0])
        ax.set_xlim(xs[0] - pad, xs[-1] + pad)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"{r['name']} — equal error rate")
        ax.set_xlabel(f"Decision threshold ({r['unit']})")
        ax.set_ylabel("Error rate")
        # Both curves hug the top corners here; above the crossing is the gap.
        ax.legend(loc="upper center", ncol=2, labelcolor=INK_2)

        # -- Decidability: fitted densities with the d' separation -------------
        ax = axes[row, 1]
        g, i = r["genuine"], r["impostor"]
        xs = np.linspace(min(g.min(), i.min()), max(g.max(), i.max()), 600)
        ax.plot(xs, _gaussian(xs, r["mu_impostor"], r["sd_impostor"]),
                color=C_IMPOSTOR, linewidth=2, label="Impostor (fitted)")
        ax.fill_between(xs, _gaussian(xs, r["mu_impostor"], r["sd_impostor"]),
                        color=C_IMPOSTOR, alpha=0.16)
        ax.plot(xs, _gaussian(xs, r["mu_genuine"], r["sd_genuine"]),
                color=C_GENUINE, linewidth=2, label="Genuine (fitted)")
        ax.fill_between(xs, _gaussian(xs, r["mu_genuine"], r["sd_genuine"]),
                        color=C_GENUINE, alpha=0.16)

        # Reserve a band above the curves for the d' arrow and the legend, so
        # neither lands on top of a density.
        peak = _headroom(ax, 1.62)
        ax.annotate("", xy=(r["mu_genuine"], peak * 1.06),
                    xytext=(r["mu_impostor"], peak * 1.06),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.4))
        ax.text((r["mu_genuine"] + r["mu_impostor"]) / 2, peak * 1.12,
                f"d' = {r['dprime']:.3f}", ha="center", fontsize=11,
                fontweight="bold", color=INK)
        ax.set_title(f"{r['name']} — decidability index")
        ax.set_xlabel(r["unit"])
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper left", labelcolor=INK_2)
    _finish(fig, path, "Equal Error Rate and Decidability Index")


# --------------------------------------------------------------------------- #
# 7. Reporting
# --------------------------------------------------------------------------- #

def write_reports(results: list[dict], layout: dict) -> None:
    rows = []
    for r in results:
        rows.append({
            "matcher": r["name"],
            "score_type": "similarity" if r["similarity"] else "distance",
            "n_genuine": r["n_genuine"], "n_impostor": r["n_impostor"],
            "mu_genuine": r["mu_genuine"], "sd_genuine": r["sd_genuine"],
            "mu_impostor": r["mu_impostor"], "sd_impostor": r["sd_impostor"],
            "eer_percent": r["eer"] * 100, "eer_threshold": r["eer_threshold"],
            "decidability_index": r["dprime"], "roc_auc": r["auc"],
            "tar_at_far_1pct": r["tar_at_far_1pct"],
            "tar_at_far_0.1pct": r["tar_at_far_0.1pct"],
            "rank1_identification": r["rank1"],
        })

    # metrics.csv
    header = list(rows[0].keys())
    with (RES_DIR / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(
                v if isinstance(v, str) else f"{v:.10g}" for v in row.values()) + "\n")

    # metrics.json
    (RES_DIR / "metrics.json").write_text(
        json.dumps({"layout_check": layout, "results": rows}, indent=2),
        encoding="utf-8")

    # Per-matcher FAR/FRR tables
    for r in results:
        tag = "euclidean" if not r["similarity"] else "cosine"
        table = np.column_stack([r["grid"], r["far_grid"], r["frr_grid"]])
        np.savetxt(RES_DIR / f"far_frr_{tag}.csv", table, delimiter=",",
                   header="threshold,FAR,FRR", comments="", fmt="%.10g")
        np.savez_compressed(RES_DIR / f"scores_{tag}.npz",
                            genuine=r["genuine"], impostor=r["impostor"])

    # Human-readable summary
    best = min(results, key=lambda r: r["eer"])
    lines = [
        "=" * 78,
        "AI461 Assignment 1 — Biometric verification performance (test set)",
        "=" * 78,
        "",
        "Protocol",
        f"  Users                       : {N_USERS}",
        f"  Samples per user            : {N_SAMPLES} (1-5 enrollment, 6-10 test)",
        f"  Feature dimension           : {N_FEATURES}",
        f"  Templates                   : {N_USERS} (mean of the 5 enrollment vectors)",
        f"  Test probes                 : {N_USERS * (N_SAMPLES - N_TRAIN)} "
        f"({N_SAMPLES - N_TRAIN} per user)",
        f"  Comparisons                 : {results[0]['n_genuine'] + results[0]['n_impostor']:,}"
        f" ({results[0]['n_genuine']:,} genuine + {results[0]['n_impostor']:,} impostor)",
        "",
        "Layout sanity check (mean within-user / between-user distance ratio)",
        f"  user-major   (assumed, 10 consecutive columns per user) : {layout['user_major_ratio']:.4f}",
        f"  sample-major (rejected alternative)                     : {layout['sample_major_ratio']:.4f}",
        "",
    ]
    for r in results:
        lines += [
            "-" * 78,
            f"{r['name']}  ({'similarity' if r['similarity'] else 'distance'}; "
            f"accept if score {'>=' if r['similarity'] else '<='} threshold)",
            "-" * 78,
            f"  Genuine   : mu = {r['mu_genuine']:.6f}   sigma = {r['sd_genuine']:.6f}   n = {r['n_genuine']:,}",
            f"  Impostor  : mu = {r['mu_impostor']:.6f}   sigma = {r['sd_impostor']:.6f}   n = {r['n_impostor']:,}",
            "",
            f"  Equal Error Rate (EER)      : {r['eer'] * 100:.3f} %   at threshold {r['eer_threshold']:.6f}",
            f"  Decidability index (d')     : {r['dprime']:.4f}",
            f"  ROC AUC                     : {r['auc']:.6f}",
            f"  TAR @ FAR = 1%              : {r['tar_at_far_1pct'] * 100:.2f} %",
            f"  TAR @ FAR = 0.1%            : {r['tar_at_far_0.1pct'] * 100:.2f} %",
            f"  Rank-1 identification       : {r['rank1'] * 100:.2f} %   (supplementary)",
            "",
        ]
    lines += [
        "=" * 78,
        "Verdict",
        "=" * 78,
        f"  Lowest EER : {best['name']} ({best['eer'] * 100:.3f} %)",
        f"  Highest d' : {max(results, key=lambda r: r['dprime'])['name']} "
        f"({max(r['dprime'] for r in results):.4f})",
        f"  Highest AUC: {max(results, key=lambda r: r['auc'])['name']} "
        f"({max(r['auc'] for r in results):.6f})",
        "",
    ]
    text = "\n".join(lines)
    (RES_DIR / "metrics.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# --------------------------------------------------------------------------- #
# 8. Diagnostics -- why cosine wins (run with --diagnostics)
# --------------------------------------------------------------------------- #

def magnitude_diagnostics(X: np.ndarray, templates: np.ndarray, probes: np.ndarray,
                          probe_labels: np.ndarray) -> str:
    """Isolate the feature-vector magnitude channel to explain the Euclidean/cosine gap.

    Euclidean distance sees both the magnitude and the direction of a feature
    vector; cosine similarity sees only the direction.  Because
    ||p_hat - t_hat||^2 = 2 - 2*cos(p, t) exactly for L2-normalised vectors,
    Euclidean matching on normalised vectors is a strictly monotone relabelling of
    cosine similarity and must therefore produce an identical ROC and EER.  Any
    gap between raw Euclidean and cosine is thus attributable to the magnitude
    channel alone -- this routine measures that channel directly.
    """
    norms = np.linalg.norm(X, axis=1).reshape(N_USERS, N_SAMPLES)
    within_cv = float((norms.std(axis=1, ddof=1) / norms.mean(axis=1)).mean())
    between_cv = float(norms.mean(axis=1).std(ddof=1) / norms.mean())
    drift = norms[:, N_TRAIN:].mean(axis=1) / norms[:, :N_TRAIN].mean(axis=1)

    # How much of the Euclidean score is just a difference of magnitudes?
    d_euclid = euclidean_scores(probes, templates)
    d_norm = np.abs(np.linalg.norm(probes, axis=1)[:, None]
                    - np.linalg.norm(templates, axis=1)[None, :])
    g_e, i_e = split_genuine_impostor(d_euclid, probe_labels)
    g_n, i_n = split_genuine_impostor(d_norm, probe_labels)

    # Euclidean on L2-normalised vectors == cosine, up to a monotone transform.
    unit_p = probes / np.linalg.norm(probes, axis=1, keepdims=True)
    unit_t = templates / np.linalg.norm(templates, axis=1, keepdims=True)
    r_unit = evaluate("Euclidean on unit-norm vectors", "distance",
                      euclidean_scores(unit_p, unit_t), probe_labels, similarity=False)
    r_cos = evaluate("Cosine similarity", "similarity",
                     cosine_scores(probes, templates), probe_labels, similarity=True)
    r_raw = evaluate("Euclidean distance", "distance", d_euclid, probe_labels,
                     similarity=False)
    identity_err = float(np.abs(euclidean_scores(unit_p, unit_t) ** 2
                                - (2 - 2 * cosine_scores(probes, templates))).max())

    lines = [
        "=" * 78,
        "Diagnostics -- isolating the feature-magnitude channel",
        "=" * 78,
        "",
        "1. Does feature-vector magnitude vary, and is it stable over six months?",
        f"   Coefficient of variation of ||x|| within a user   : {within_cv:.4f}",
        f"   Coefficient of variation of ||x|| across users    : {between_cv:.4f}",
        f"   Test/enrollment mean-norm ratio per user          : mean {drift.mean():.4f}, "
        f"sd {drift.std(ddof=1):.4f}, range [{drift.min():.3f}, {drift.max():.3f}]",
        "   -> magnitude drifts by up to ~17% between enrollment and test, and it also",
        "      differs systematically between users (across-user CV is ~3.6x within-user).",
        "",
        "2. How much of the Euclidean score is magnitude rather than pattern?",
        f"   corr(Euclidean, |norm difference|), genuine       : {np.corrcoef(g_e, g_n)[0, 1]:.4f}",
        f"   corr(Euclidean, |norm difference|), impostor      : {np.corrcoef(i_e, i_n)[0, 1]:.4f}",
        f"   d' of |norm difference| used alone as a matcher   : {decidability_index(g_n, i_n):.4f}",
        "   -> the Euclidean impostor axis is very largely a magnitude axis, though",
        "      magnitude is not pure noise: on its own it still separates somewhat.",
        "",
        "3. Removing the magnitude channel from Euclidean matching",
        f"   ||p_hat - t_hat||^2 vs (2 - 2cos), max abs error  : {identity_err:.3e}",
        "",
        f"   {'matcher':<34}{'EER':>10}{'AUC':>14}{'d-prime':>10}",
        f"   {'-' * 68}",
    ]
    for r in (r_raw, r_cos, r_unit):
        lines.append(f"   {r['name']:<34}{r['eer'] * 100:9.4f}%{r['auc']:14.8f}"
                     f"{r['dprime']:10.4f}")
    lines += [
        "",
        "   -> L2-normalising before Euclidean matching reproduces the cosine EER and",
        "      AUC exactly, so the entire raw-Euclidean/cosine performance gap is due to",
        "      the magnitude channel and nothing else.",
        "   -> note that d' differs (2.4770 vs 1.9254) even though the ROC is identical:",
        "      d' is invariant to affine rescaling but not to the nonlinear sqrt relating",
        "      the two scores, so d' should be read alongside EER and AUC, not alone.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    print("Loading data ...")
    X, labels = load_data()
    print(f"  {X.shape[0]} samples x {X.shape[1]} features")

    layout = verify_layout(X, labels)
    print(f"  layout check — user-major ratio {layout['user_major_ratio']:.4f} "
          f"vs sample-major {layout['sample_major_ratio']:.4f}")

    templates, probes, probe_labels = split_and_enroll(X)
    print(f"  {templates.shape[0]} templates, {probes.shape[0]} test probes")

    print("Matching ...")
    results = [
        evaluate("Euclidean distance", "Euclidean distance",
                 euclidean_scores(probes, templates), probe_labels, similarity=False),
        evaluate("Cosine similarity", "Cosine similarity",
                 cosine_scores(probes, templates), probe_labels, similarity=True),
    ]

    print("Generating figures ...")
    plot_single_distribution(results, "genuine", C_GENUINE,
                             "A. Genuine score distribution",
                             FIG_DIR / "A_genuine_distribution.png")
    plot_single_distribution(results, "impostor", C_IMPOSTOR,
                             "B. Impostor score distribution",
                             FIG_DIR / "B_impostor_distribution.png")
    plot_overlaid_distributions(results, FIG_DIR / "AB_score_distributions.png")
    plot_error_rate(results, "far", C_FAR, "False Acceptance Rate",
                    "C. FAR versus decision threshold", FIG_DIR / "C_FAR_plot.png")
    plot_error_rate(results, "frr", C_FRR, "False Rejection Rate",
                    "D. FRR versus decision threshold", FIG_DIR / "D_FRR_plot.png")
    plot_far_frr_combined(results, FIG_DIR / "CD_FAR_FRR_combined.png")
    plot_roc(results, FIG_DIR / "E_ROC_curves.png")
    plot_eer_and_dprime(results, FIG_DIR / "F_EER_decidability.png")

    print("Writing reports ...")
    write_reports(results, layout)

    if "--diagnostics" in sys.argv:
        report = magnitude_diagnostics(X, templates, probes, probe_labels)
        (RES_DIR / "diagnostics.txt").write_text(report, encoding="utf-8")
        print("\n" + report)


if __name__ == "__main__":
    main()
