"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from medunc import plotting
from medunc.data import SHORT, TITLE
from medunc.plotting import DATASET_COLOR, INK, INK_2, MUTED, SERIES, method_style

CORRUPTION_LABEL = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "contrast": "Contrast loss",
                    "brightness": "Brightness shift"}
MAIN = ["Single model", "MC dropout + TS", "Deep ensemble + TS"]
SIX = ["Single model", "Single model + TS", "MC dropout", "MC dropout + TS", "Deep ensemble", "Deep ensemble + TS"]


def _suptitle(fig: plt.Figure, text: str, gap_in: float = 0.3) -> None:
    """Left-aligned takeaway title placed ``gap_in`` inches above the top of the axes."""
    fig.suptitle(text, x=0.01, y=fig.subplotpars.top + gap_in / fig.get_figheight(), ha="left", va="bottom", fontsize=12,
                 fontweight="bold", color=INK)


def _mosaic(images: np.ndarray, gap: int = 2) -> np.ndarray:
    """Images (k, h, w, 3) side by side with a light gap."""
    k, h, w, _ = images.shape
    out = np.full((h, k * w + (k - 1) * gap, 3), 252, dtype=np.uint8)
    for i in range(k):
        out[:, i * (w + gap):i * (w + gap) + w] = images[i]
    return out


def fig_data(classes: pd.DataFrame, ex: dict, path: Path, title: str) -> Path:
    """One training image per class and the class distribution of both data sets."""
    dsets = list(dict.fromkeys(classes.dataset))
    fig, axes = plt.subplots(2, len(dsets), figsize=(12, 5.6), gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.35})
    for j, ds in enumerate(dsets):
        imgs, labels = ex[f"{ds}_samples"], ex[f"{ds}_sample_labels"]
        first = [int(np.flatnonzero(labels == c)[0]) for c in np.unique(labels)]
        ax = axes[0, j]
        ax.imshow(_mosaic(imgs[first]))
        w = imgs.shape[2] + 2
        ax.set_xticks([i * w + imgs.shape[2] / 2 for i in range(len(first))])
        ax.set_xticklabels([SHORT[ds][int(labels[i])] for i in first], fontsize=8.5)
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{TITLE[ds]}: one 64 x 64 training image per class", loc="left")
        c = classes[classes.dataset == ds].sort_values("train")
        ax = axes[1, j]
        ax.barh(c.short, c.train, color=DATASET_COLOR[ds], height=0.65)
        for y, (n, sh) in enumerate(zip(c.train, c.train_share)):
            ax.text(n, y, f" {n:,} ({sh:.0%})", va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, c.train.max() * 1.35)
        ax.set_xlabel("Training images")
        ax.grid(axis="y", visible=False)
        tot = classes[classes.dataset == ds][["train", "val", "test"]].sum()
        ax.set_title(f"Class distribution (train / val / test = {tot.train:,} / {tot.val:,} / {tot.test:,})", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_risk_coverage(rc: pd.DataFrame, res: pd.DataFrame, path: Path, title: str) -> Path:
    """Selective risk (error rate of the accepted images) against coverage, with the oracle bound."""
    dsets = list(dict.fromkeys(rc.dataset))
    fig, axes = plt.subplots(1, len(dsets), figsize=(12, 4.8), gridspec_kw={"wspace": 0.22})
    for ax, ds in zip(np.atleast_1d(axes), dsets):
        for m in MAIN:
            g = rc[(rc.dataset == ds) & (rc.method == m)]
            r = res[(res.dataset == ds) & (res.method == m)].iloc[0]
            ax.plot(g.coverage, 100 * g.risk, **method_style(m), lw=2, label=f"{m} (AURC {100 * r.aurc:.2f})")
        g = rc[(rc.dataset == ds) & (rc.method == "Deep ensemble + TS")]
        ax.plot(g.coverage, 100 * g.oracle_risk, color=INK, lw=1, ls=":", label="Oracle ranking (ensemble + TS errors)")
        tgt = res[res.dataset == ds].target_accuracy.iloc[0]
        ax.axhline(100 * (1 - tgt), color=MUTED, lw=1, ls="-.")
        ax.text(0.02, 100 * (1 - tgt), f" target {tgt:.0%} accuracy", color=INK_2, fontsize=8.5, va="bottom")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, None)
        ax.set_xlabel("Coverage (share of test images the model answers)")
        ax.set_ylabel("Selective risk: error rate on answered images (%)")
        b = res[(res.dataset == ds) & (res.method == "Single model")].iloc[0]
        e = res[(res.dataset == ds) & (res.method == "Deep ensemble + TS")].iloc[0]
        ax.set_title(f"{TITLE[ds]}: {tgt:.0%} accuracy is reached while answering\n"
                     f"{100 * b.coverage_at_target:.0f}% of images (single model) or {100 * e.coverage_at_target:.0f}% "
                     f"(deep ensemble + TS)", loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8.5)
    _suptitle(fig, title, gap_in=0.62)
    plotting.save(fig, path)
    return path


def fig_calibration(rel: pd.DataFrame, res: pd.DataFrame, path: Path, title: str) -> Path:
    """Reliability diagrams of three methods per data set and the ECE of all methods with 95% CIs."""
    dsets = list(dict.fromkeys(rel.dataset))
    fig, axes = plt.subplots(1, len(dsets) + 1, figsize=(14.5, 4.6),
                             gridspec_kw={"width_ratios": [1] * len(dsets) + [1.1], "wspace": 0.55})
    for ax, ds in zip(axes, dsets):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
        for m in ["Single model", "Single model + TS", "Deep ensemble + TS"]:
            g = rel[(rel.dataset == ds) & (rel.method == m) & (rel["count"] >= 10)]
            r = res[(res.dataset == ds) & (res.method == m)].iloc[0]
            ax.plot(g.confidence, g.accuracy, marker="o", ms=4, **method_style(m), lw=1.6,
                    label=f"{m} (ECE {100 * r.ece:.1f}%)")
        lo = rel[rel.dataset == ds].lower[(rel[rel.dataset == ds]["count"] >= 10).to_numpy()].min()
        ax.set_xlim(max(0.0, lo - 0.05), 1.0)
        ax.set_ylim(max(0.0, lo - 0.05) - 0.1, 1.02)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Accuracy in bin")
        ax.set_title(f"{TITLE[ds]} reliability\n(15 bins, bins with >= 10 images shown)", loc="left")
        ax.legend(loc="lower right", fontsize=8)
    ax = axes[-1]
    order = SIX + ["Linear probe (frozen ImageNet features)"]
    for k, ds in enumerate(dsets):
        for i, m in enumerate(order):
            r = res[(res.dataset == ds) & (res.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            yv = i + (k - 0.5) * 0.3
            ax.errorbar(100 * r.ece, yv, xerr=[[100 * max(0.0, r.ece - r.ece_ci_low)], [100 * max(0.0, r.ece_ci_high - r.ece)]],
                        fmt="o", color=DATASET_COLOR[ds], ms=5, lw=1.2, capsize=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([m.replace(" (frozen ImageNet features)", "") for m in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Expected calibration error (%), 95% bootstrap CI")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=DATASET_COLOR[d], marker="o", ls="", label=TITLE[d]) for d in dsets],
              loc="lower right", fontsize=8.5)
    ax.set_title("ECE of every method on the test set\n", loc="left")
    _suptitle(fig, title, gap_in=0.62)
    plotting.save(fig, path)
    return path


def fig_corruptions(cor: pd.DataFrame, path: Path, title: str) -> Path:
    """Accuracy, ECE, answered share and accuracy of the answered images against severity (mean over corruptions)."""
    dsets = list(dict.fromkeys(cor.dataset))
    metrics = [("accuracy", "Accuracy, all images (%)"), ("ece", "Expected calibration error (%)"),
               ("answered_share", "Images answered (%)"), ("answered_accuracy", "Accuracy on answered images (%)")]
    fig, axes = plt.subplots(len(dsets), len(metrics), figsize=(16, 3.9 * len(dsets)), squeeze=False,
                             gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    for i, ds in enumerate(dsets):
        g = cor[cor.dataset == ds].groupby(["method", "severity"])[[m for m, _ in metrics]].mean().reset_index()
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i, j]
            for meth in SIX:
                h = g[g.method == meth].sort_values("severity")
                ax.plot(h.severity, 100 * h[m], marker="o", ms=3.5, lw=1.6, **method_style(meth), label=meth)
            ax.set_xticks(range(0, int(g.severity.max()) + 1))
            ax.set_xlabel("Corruption severity (0 = clean)")
            ax.set_ylabel(lab)
            ax.set_title(f"{TITLE[ds]}: {lab.split(' (')[0].lower()}", loc="left", fontsize=9.5)
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower left", bbox_to_anchor=(0.01, fig.subplotpars.top + 0.045), ncol=6, fontsize=8.5)
    fig.text(0.01, -0.02, "Answered = 1 - max probability below the threshold that answers 80% of clean validation images "
             "(fixed before deployment).", fontsize=8.5, color=INK_2)
    _suptitle(fig, title, gap_in=0.8)
    plotting.save(fig, path)
    return path


def fig_corruption_types(cor: pd.DataFrame, path: Path, title: str) -> Path:
    """Accuracy and ECE by corruption type and severity for the single model and the ensemble with TS."""
    dsets = list(dict.fromkeys(cor.dataset))
    kinds = [k for k in CORRUPTION_LABEL if k in set(cor.corruption)]
    fig, axes = plt.subplots(len(dsets), len(kinds), figsize=(13, 3.6 * len(dsets)), squeeze=False, sharey="row",
                             gridspec_kw={"hspace": 0.5})
    for i, ds in enumerate(dsets):
        clean = cor[(cor.dataset == ds) & (cor.severity == 0)]
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            for meth in ["Single model", "Deep ensemble + TS"]:
                h = pd.concat([clean[clean.method == meth], cor[(cor.dataset == ds) & (cor.corruption == kind) & (cor.method == meth)]])
                h = h.sort_values("severity")
                st = method_style(meth)
                ax.plot(h.severity, 100 * h.accuracy, marker="o", ms=3.5, lw=1.8, color=st["color"], ls="-",
                        label=f"{meth}: accuracy")
                ax.plot(h.severity, 100 * h.ece, marker="s", ms=3.5, lw=1.2, color=st["color"], ls=":",
                        label=f"{meth}: ECE")
            ax.set_title(f"{TITLE[ds]}: {CORRUPTION_LABEL[kind]}", loc="left", fontsize=9.5)
            ax.set_xticks(range(0, int(cor.severity.max()) + 1))
            ax.set_xlabel("Severity")
            if j == 0:
                ax.set_ylabel("Accuracy / ECE (%)")
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower left", bbox_to_anchor=(0.01, fig.subplotpars.top + 0.05), ncol=4, fontsize=8.5)
    _suptitle(fig, title, gap_in=0.8)
    plotting.save(fig, path)
    return path


def fig_ood(ood: pd.DataFrame, path: Path, title: str) -> Path:
    """Out-of-distribution detection: AUROC by method and score, and the share of foreign images still accepted."""
    dsets = list(dict.fromkeys(ood.dataset))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.08})
    score_marker = {"1 - max prob": "o", "entropy": "s", "mutual information": "^"}
    ax = axes[0]
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.5, -0.6, " chance", color=INK_2, fontsize=8, va="bottom")
    for k, ds in enumerate(dsets):
        for i, m in enumerate(SIX):
            for q, (sc, mk) in enumerate(score_marker.items()):
                r = ood[(ood.dataset == ds) & (ood.method == m) & (ood.score == sc)]
                if r.empty:
                    continue
                ax.plot(r.auroc.iloc[0], i + (k - 0.5) * 0.42 + (q - 1) * 0.11, marker=mk, color=DATASET_COLOR[ds], ls="",
                        ms=5, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(SIX)))
    ax.set_yticklabels(SIX, fontsize=8.5)
    ax.set_ylim(len(SIX) - 0.4, -0.7)
    ax.set_xlabel("AUROC: own test images vs the other data set's test images")
    ax.grid(axis="y", visible=False)
    ax.set_title("Does higher uncertainty flag images from the other data set?", loc="left")
    ax = axes[1]
    sub = ood[ood.score == "1 - max prob"]
    width = 0.38
    for k, ds in enumerate(dsets):
        g = sub[sub.dataset == ds].set_index("method").reindex(SIX)
        ax.barh(np.arange(len(SIX)) + (k - 0.5) * width, 100 * g.ood_accepted, height=width, color=DATASET_COLOR[ds])
        for i, v in enumerate(g.ood_accepted):
            ax.text(100 * v, i + (k - 0.5) * width, f" {100 * v:.0f}%", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(SIX)))
    ax.set_yticklabels([])
    ax.set_ylim(len(SIX) - 0.4, -0.7)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Foreign images answered at the threshold that answers 80% of validation images (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of the other data set's images answered (1 - max prob)", loc="left")
    handles = [Line2D([], [], color=DATASET_COLOR[d], marker="o", ls="",
                      label=f"{TITLE[d]} model shown {TITLE[next(o for o in dsets if o != d)]} images") for d in dsets]
    handles += [Line2D([], [], color=MUTED, marker=mk, ls="", label=f"score: {sc}") for sc, mk in score_marker.items()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, fig.subplotpars.top + 0.08), ncol=5, fontsize=8.5)
    _suptitle(fig, title, gap_in=0.75)
    plotting.save(fig, path)
    return path


def fig_gradcam(ex: dict, gce: pd.DataFrame, san: pd.DataFrame, path: Path, title: str) -> Path:
    """Grad-CAM overlays for confident-correct, confident-wrong and most uncertain test images."""
    dsets = list(dict.fromkeys(gce.dataset))
    n = max(int((gce.dataset == ds).sum()) for ds in dsets)
    fig = plt.figure(figsize=(1.45 * n + 0.5, 2.55 * len(dsets) + 0.3))
    subs = np.atleast_1d(fig.subfigures(len(dsets), 1, hspace=0.04))
    for sub, ds in zip(subs, dsets):
        g = gce[gce.dataset == ds].sort_values("order")
        imgs, maps = ex[f"{ds}_gc_images"], ex[f"{ds}_gc_maps"].astype(np.float32)
        s = san[(san.dataset == ds) & (san.randomized == "fc")]
        c = san[(san.dataset == ds) & (san.randomized.str.startswith("none"))]
        rho = (f": rank correlation {s.mean_spearman.iloc[0]:.2f} with the maps after re-initialising the last layer; "
               f"{100 * c.mean_centre_share.iloc[0]:.0f}% of map mass in the central quarter (uniform: 25%)"
               if len(s) and len(c) else "")
        sub.suptitle(f"{TITLE[ds]}{rho}", x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
        axes = sub.subplots(1, n)
        sub.subplots_adjust(top=0.62, bottom=0.02, wspace=0.08, left=0.01, right=0.99)
        for j, ax in enumerate(axes):
            ax.axis("off")
            if j >= len(g):
                continue
            r = g.iloc[j]
            ax.imshow(imgs[j])
            ax.imshow(maps[j], cmap="magma", alpha=0.45, vmin=0, vmax=1)
            ok = r.label == r.prediction
            ax.set_title(f"{r.label} -> {r.prediction}\nconf {r.confidence_single:.2f}", fontsize=7.5,
                         color=INK if ok else SERIES[7], fontweight="normal")
            if j == 0 or g.iloc[j - 1].category != r.category:
                ax.text(0, 1.36, r.category, transform=ax.transAxes, fontsize=8, color=INK_2, ha="left", va="bottom",
                        fontweight="bold")
    fig.suptitle(title, x=0.01, y=1.04, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
    plotting.save(fig, path)
    return path


def fig_imbalance(imb: pd.DataFrame, imbc: pd.DataFrame, defer: pd.DataFrame, path: Path, title: str) -> Path:
    """DermaMNIST per-class recall under three imbalance treatments, and per-class deferral at 80% coverage."""
    ds = imb.dataset.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    meths = [("Single model", SERIES[1]), ("Single model + logit adjustment", SERIES[3]), ("Class-weighted loss", SERIES[6])]
    order = imbc[imbc.method == "Single model"].sort_values("train_share")
    x = np.arange(len(order))
    w = 0.27
    for k, (m, col) in enumerate(meths):
        g = imbc[imbc.method == m].set_index("class").reindex(order["class"])
        r = imb[imb.method == m].iloc[0]
        ax.bar(x + (k - 1) * w, 100 * g.recall, width=w, color=col,
               label=f"{m}: accuracy {100 * r.accuracy:.1f}%, balanced {100 * r.balanced_accuracy:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{s:.1%}" for c, s in zip(order["class"], order.train_share)], fontsize=8.5)
    ax.set_xlabel("Class (share of training images)")
    ax.set_ylabel("Recall on test images (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{TITLE[ds]}: recall per class", loc="left")
    ax = axes[1]
    d = defer[defer.dataset == ds]
    for k, (m, col) in enumerate([("Single model", SERIES[1]), ("Deep ensemble + TS", SERIES[0])]):
        g = d[d.method == m].set_index("class").reindex(order["class"])
        ax.bar(x + (k - 0.5) * 0.38, 100 * g.deferred_share, width=0.38, color=col, label=m)
    cov = d.coverage.iloc[0]
    ax.axhline(100 * (1 - cov), color=MUTED, lw=1, ls="-.")
    ax.text(len(x) - 0.5, 100 * (1 - cov), f"overall {100 * (1 - cov):.0f}% deferred", ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(order["class"], fontsize=8.5)
    ax.set_ylabel("Test images deferred to a clinician (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{TITLE[ds]}: who gets deferred when the model answers {cov:.0%} of images", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_ensemble_size(size: pd.DataFrame, path: Path, title: str) -> Path:
    """NLL, ECE and AURC against the number of ensemble members or MC-dropout passes."""
    dsets = list(dict.fromkeys(size.dataset))
    metrics = [("nll", "Negative log-likelihood"), ("ece", "ECE (%)"), ("aurc", "AURC (%)")]
    fig, axes = plt.subplots(len(dsets), 3, figsize=(13, 3.8 * len(dsets)), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    for i, ds in enumerate(dsets):
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i, j]
            for fam, col in [("Deep ensemble", SERIES[0]), ("MC dropout", SERIES[2])]:
                for ts in (False, True):
                    g = size[(size.dataset == ds) & (size.family == fam) & (size.temperature_scaled == ts)].sort_values("n")
                    scale = 1 if m == "nll" else 100
                    ax.plot(g.n, scale * g[m], marker="o", ms=3.5, lw=1.6, color=col, ls="--" if ts else "-",
                            label=f"{fam}{' + TS' if ts else ''}")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 50])
            ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
            ax.set_xlabel("Members (ensemble) or forward passes (MC dropout)")
            ax.set_ylabel(lab)
            ax.set_title(f"{TITLE[ds]}: {lab.split(' (')[0]}", loc="left")
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower left", bbox_to_anchor=(0.01, fig.subplotpars.top + 0.05), ncol=4, fontsize=8.5)
    _suptitle(fig, title, gap_in=0.8)
    plotting.save(fig, path)
    return path


# Takeaway titles, written after reading the results (keys = figure files).
TITLES = {
    "00_data.png": "DermaMNIST is dominated by one class (67% nevi); BloodMNIST's eight classes range from 7% to 19%",
    "01_risk_coverage.png": "A 5-member ensemble ranks its own errors best: 90% DermaMNIST accuracy while answering 89% of images (single model: 84%)",
    "02_calibration.png": "Temperature scaling halves the single model's DermaMNIST calibration error (5.1% to 2.3%); BloodMNIST is calibrated already",
    "03_corruptions.png": "Under corruption, the BloodMNIST models mostly abstain; the DermaMNIST models keep answering about 70% of images at falling accuracy",
    "04_corruption_types.png": "Contrast loss breaks both models at the mildest level (BloodMNIST 99% to about 40%); blur is the gentlest shift",
    "05_ood.png": "The skin model is more confident on blood-cell images than on skin and answers 99% of them; the blood ensemble rejects 92% of skin images",
    "06_gradcam.png": "Grad-CAM passes the randomisation check; BloodMNIST maps sit on the central cell, DermaMNIST maps often light up the background",
    "07_imbalance.png": "Rebalancing lifts minority recall at a cost in accuracy; answering 80% of images defers 43% of melanomas but 9% of nevi",
    "08_ensemble_size.png": "Each added member lowers DermaMNIST NLL and AURC; 50 MC-dropout passes improve NLL slightly but rank errors no better",
}


def make_all(results: Path, figures: Path) -> list[Path]:
    """Redraw every figure from the CSV/JSON tables in ``results``."""
    plotting.apply_style()
    ex = dict(np.load(results / "image_examples.npz"))
    res = pd.read_csv(results / "method_results.csv")
    cor = pd.read_csv(results / "corruption_results.csv")
    out = [
        fig_data(pd.read_csv(results / "class_counts.csv"), ex, figures / "00_data.png", TITLES["00_data.png"]),
        fig_risk_coverage(pd.read_csv(results / "risk_coverage.csv"), res, figures / "01_risk_coverage.png",
                          TITLES["01_risk_coverage.png"]),
        fig_calibration(pd.read_csv(results / "reliability.csv"), res, figures / "02_calibration.png", TITLES["02_calibration.png"]),
        fig_corruptions(cor, figures / "03_corruptions.png", TITLES["03_corruptions.png"]),
        fig_corruption_types(cor, figures / "04_corruption_types.png", TITLES["04_corruption_types.png"]),
        fig_ood(pd.read_csv(results / "ood_detection.csv"), figures / "05_ood.png", TITLES["05_ood.png"]),
        fig_gradcam(ex, pd.read_csv(results / "gradcam_examples.csv"), pd.read_csv(results / "gradcam_sanity.csv"),
                    figures / "06_gradcam.png", TITLES["06_gradcam.png"]),
        fig_imbalance(pd.read_csv(results / "imbalance.csv"), pd.read_csv(results / "imbalance_per_class.csv"),
                      pd.read_csv(results / "deferral_by_class.csv"), figures / "07_imbalance.png", TITLES["07_imbalance.png"]),
        fig_ensemble_size(pd.read_csv(results / "ensemble_size.csv"), figures / "08_ensemble_size.png",
                          TITLES["08_ensemble_size.png"]),
    ]
    return out
