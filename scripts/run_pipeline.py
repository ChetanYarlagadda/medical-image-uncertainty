"""Run every experiment end to end.

Trains the ensemble members (cached weights), stores the logits of every member and the MC-dropout samples on the
validation, test, corrupted-test and out-of-distribution images (cached predictions), then computes every table in
reports/results/ and every figure in reports/figures/. Re-runs only recompute what is missing from data/cache/.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on small subsets with tiny budgets; outputs go to data/cache/smoke/
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from scipy.special import log_softmax, softmax  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from medunc import corruptions, gradcam  # noqa: E402
from medunc.data import (  # noqa: E402
    CLASSES, LICENSE, MD5, PUBLISHED, SHORT, class_table, crosscheck, filename, load, medmnist_info_check,
)
from medunc.figures import make_all  # noqa: E402
from medunc.metrics import (  # noqa: E402
    BOOT_METRICS, binary_auc, bootstrap_indices, bootstrap_metrics, holm, paired_p, per_class_recall, reliability,
    risk_coverage, summarize,
)
from medunc.models import Classifier, TrainConfig, normalize, predict, to_float, train  # noqa: E402
from medunc.tracking import init_mlflow, log_result  # noqa: E402
from medunc.uncertainty import (  # noqa: E402
    fit_temperature, logit_adjust, mc_dropout_logits, method_samples, mutual_information, residual_mass, scores,
)

torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
VERSION = "v1"  # bump to invalidate every cached weight and prediction
BASE_METHODS = ["Single model", "MC dropout", "Deep ensemble"]
METHODS = [m + s for m in BASE_METHODS for s in ("", " + TS")]
BASELINE = "Single model"
PROBE = "Linear probe (frozen ImageNet features)"

# MedMNIST v2 benchmark (Yang et al., 2023, Table 3; medmnist.com): test AUC and accuracy.
PUBLISHED_BENCHMARK = {
    ("dermamnist", "ResNet-18 (28)"): (0.917, 0.735), ("dermamnist", "ResNet-18 (224)"): (0.920, 0.754),
    ("dermamnist", "ResNet-50 (28)"): (0.913, 0.735), ("dermamnist", "ResNet-50 (224)"): (0.912, 0.731),
    ("dermamnist", "Google AutoML Vision"): (0.914, 0.768),
    ("bloodmnist", "ResNet-18 (28)"): (0.998, 0.958), ("bloodmnist", "ResNet-18 (224)"): (0.998, 0.963),
    ("bloodmnist", "ResNet-50 (28)"): (0.997, 0.956), ("bloodmnist", "ResNet-50 (224)"): (0.997, 0.950),
    ("bloodmnist", "Google AutoML Vision"): (0.998, 0.966),
}
# Paired comparisons on the clean test set (Holm-corrected together over both data sets).
COMPARISONS = [(m, BASELINE) for m in METHODS if m != BASELINE] + [
    ("Deep ensemble + TS", "Single model + TS"), ("MC dropout + TS", "Single model + TS")]
TEST_METRICS = ("accuracy", "auc", "nll", "ece", "aurc")


@dataclass(frozen=True)
class Settings:
    """Data, training and evaluation settings of one run."""

    size: int = 64
    datasets: tuple[str, ...] = ("dermamnist", "bloodmnist")
    members: int = 5
    train: TrainConfig = field(default_factory=TrainConfig)
    weighted: TrainConfig = field(default_factory=lambda: TrainConfig(class_weight_power=0.5))
    weighted_datasets: tuple[str, ...] = ("dermamnist",)
    mc_samples: int = 50
    severities: tuple[int, ...] = (1, 2, 3, 4, 5)
    n_boot: int = 1000
    target_accuracy: tuple[tuple[str, float], ...] = (("dermamnist", 0.90), ("bloodmnist", 0.99))
    coverage: float = 0.8  # operating point for the deferral and out-of-distribution tables
    gradcam_layer: str = "layer3"
    gradcam_n: int = 300
    probe_c: tuple[float, ...] = (0.01, 0.1, 1.0)
    subsample: tuple[tuple[str, int], ...] = ()  # quick mode: images kept per split


QUICK = Settings(members=2, train=TrainConfig(max_epochs=2), weighted=TrainConfig(max_epochs=2, class_weight_power=0.5),
                 mc_samples=10, severities=(2, 5), n_boot=50, gradcam_n=40,
                 subsample=(("train", 800), ("val", 300), ("test", 400)))


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock seconds per stage; GPU stages are listed separately."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.gpu: set[str] = set()

    def run(self, name: str, fn: Callable, gpu: bool = False):
        t0 = time.perf_counter()
        out = fn()
        if gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.gpu.add(name)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out


def key_of(spec: dict) -> str:
    """Short hash of everything that changes an output."""
    return hashlib.sha1(json.dumps({**spec, "version": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached(name: str, spec: dict, fn: Callable[[], dict]) -> dict:
    """Arrays returned by ``fn``, stored as .npz under a hash of ``spec``."""
    path = CACHE / "preds" / f"{name}-{key_of(spec)}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on small subsets with tiny budgets")
    args = ap.parse_args()
    S = QUICK if args.quick else Settings()
    results = CACHE / "smoke" / "results" if args.quick else ROOT / "reports" / "results"
    figures = CACHE / "smoke" / "figures" if args.quick else ROOT / "reports" / "figures"
    results.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(CACHE / "torch"))  # ImageNet weights are downloaded once into data/cache/torch
    timer = Timer()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = datetime.now(timezone.utc)
    targets = dict(S.target_accuracy)

    # ------------------------------------------------------------------ data and checks
    data = {ds: timer.run("data", lambda ds=ds: load(ds, S.size, RAW)) for ds in S.datasets}
    if S.subsample:
        rng = np.random.default_rng(0)
        for ds in S.datasets:
            for split, n in S.subsample:
                data[ds][split] = data[ds][split].take(np.sort(rng.permutation(len(data[ds][split]))[:n]))
    checks = pd.concat([crosscheck(ds, data[ds]).assign(source="MedMNIST v2 paper (Table 1) vs loaded arrays")
                        for ds in S.datasets] + [medmnist_info_check(ds, S.size) for ds in S.datasets], ignore_index=True)
    classes = pd.concat([class_table(ds, data[ds]) for ds in S.datasets], ignore_index=True)
    log(f"published statistics matched: {int(checks.match.sum())}/{len(checks)}")
    checks.to_csv(results / "data_crosscheck.csv", index=False)
    classes.to_csv(results / "class_counts.csv", index=False)

    rows: dict[str, list] = {k: [] for k in ["members", "history", "methods", "corruption", "boot", "sig", "scores", "rc",
                                             "rel", "size", "deferral", "ood", "imbalance", "imbalance_class",
                                             "published", "temps", "gc_sanity", "gc_examples"]}
    examples: dict[str, np.ndarray] = {}
    conds = [f"{c}-{s}" for c in corruptions.SEVERITY for s in S.severities]

    for ds in S.datasets:
        n_classes = len(CLASSES[ds])
        other = next(o for o in S.datasets if o != ds)
        d = data[ds]
        gpu = {s: torch.from_numpy(d[s].x).to(dev) for s in ("train", "val", "test")}
        gpu["ood"] = torch.from_numpy(data[other]["test"].x).to(dev)
        y_tr, y_val, y_te = (torch.from_numpy(d[s].y).to(dev) for s in ("train", "val", "test"))
        data_spec = {"dataset": ds, "size": S.size, "md5": MD5[(ds, S.size)], "subsample": S.subsample}
        prior = np.bincount(d["train"].y, minlength=n_classes) / len(d["train"])

        # -------------------------------------------------------------- linear-probe baseline on frozen ImageNet features
        def probe() -> dict:
            net = Classifier(n_classes, pretrained=True).to(dev).to(memory_format=torch.channels_last)
            f = {s: predict(net, gpu[s])[1] for s in ("train", "val", "test")}
            sc = StandardScaler().fit(f["train"])
            best = None
            for c in S.probe_c:
                lr = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(f["train"]), d["train"].y)
                pv = lr.predict_proba(sc.transform(f["val"]))
                v = float(-np.log(pv[np.arange(len(pv)), d["val"].y] + 1e-12).mean())
                if best is None or v < best[0]:
                    best = (v, c, lr.predict_proba(sc.transform(f["test"])))
            return {"test_prob": best[2], "c": np.array(best[1]), "val_nll": np.array(best[0])}

        pr = timer.run("linear probe", lambda: cached(f"probe-{ds}", {**data_spec, "c": S.probe_c}, probe), gpu=True)
        log(f"{ds}: linear probe (C={float(pr['c'])}) test accuracy {(pr['test_prob'].argmax(1) == d['test'].y).mean():.3f}")

        # -------------------------------------------------------------- ensemble members and their predictions
        def member(tag: str, seed: int, cfg: TrainConfig, conditions: list[str]) -> dict[str, dict]:
            spec = {**data_spec, "seed": seed, "cfg": asdict(cfg)}
            if cfg.class_weight_power:
                spec["early_stopping"] = "class-weighted validation NLL"
            mkey = key_of(spec)
            wpath = CACHE / "weights" / f"{ds}-{tag}-s{seed}-{mkey}.pt"
            model = None

            def get_model() -> Classifier:
                nonlocal model
                if model is not None:
                    return model
                model = Classifier(n_classes, cfg.dropout, cfg.pretrained)
                if wpath.exists():
                    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=False)["state"])
                else:
                    log(f"{ds}: training {tag} member, seed {seed}")
                    t0 = time.perf_counter()
                    state, hist = timer.run("training", lambda: train(model, gpu["train"], y_tr, gpu["val"], y_val, cfg, seed, log),
                                            gpu=True)
                    wpath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": state, "history": hist.to_dict("records"), "spec": spec,
                                "seconds": time.perf_counter() - t0}, wpath)
                return model.to(dev).to(memory_format=torch.channels_last).eval()

            if not wpath.exists():
                get_model()
            saved = torch.load(wpath, map_location="cpu", weights_only=False)
            hist = pd.DataFrame(saved["history"]).assign(dataset=ds, member=tag, seed=seed)
            rows["history"].append(hist)
            best = hist.loc[hist.val_nll.idxmin()]
            rows["members"].append({"dataset": ds, "member": tag, "seed": seed, "epochs_run": len(hist),
                                    "best_epoch": int(best.epoch), "val_nll": best.val_nll, "val_acc": best.val_acc,
                                    "train_seconds": saved["seconds"]})
            out = {}
            for cond in conditions:
                want_mc = tag == "plain" and seed == 0

                def run(cond: str = cond, want_mc: bool = want_mc) -> dict:
                    m = get_model()
                    if cond in ("val", "test", "ood"):
                        lg, ft = predict(m, gpu[cond])
                    else:
                        cname, sev = cond.rsplit("-", 1)
                        lg, ft = predict(m, gpu["test"], transform=corruptions.make(cname, int(sev)))
                    res = {"logits": lg}
                    if want_mc:
                        res["mc_logits"] = mc_dropout_logits(ft, m.fc.weight.detach().cpu().numpy(),
                                                             m.fc.bias.detach().cpu().numpy(), cfg.dropout, S.mc_samples, seed=0)
                    return res

                out[cond] = timer.run("inference", lambda run=run, cond=cond, want_mc=want_mc: cached(
                    f"pred-{ds}-{tag}-s{seed}-{cond}", {"member": mkey, "cond": cond, "mc": S.mc_samples if want_mc else 0,
                                                        "severity": corruptions.SEVERITY}, run), gpu=True)
            if model is not None and tag == "plain" and seed == 0:
                out["_model"] = model
            return out

        all_conds = ["val", "test", "ood"] + conds
        preds = [member("plain", s, S.train, all_conds) for s in range(S.members)]
        model0 = preds[0].pop("_model", None)
        logits = {c: np.stack([p[c]["logits"] for p in preds]) for c in all_conds}  # (M, n, C)
        mc = {c: preds[0][c]["mc_logits"] for c in all_conds}  # (S, n, C)
        weighted = (member("weighted", 0, S.weighted, ["val", "test"]) if ds in S.weighted_datasets else None)
        for r in rows["members"]:
            if r["dataset"] == ds:
                lg = logits["test"][r["seed"]] if r["member"] == "plain" else weighted["test"]["logits"]
                lp = log_softmax(lg, axis=1)
                r["test_acc"] = float((lg.argmax(1) == d["test"].y).mean())
                r["test_nll"] = float(-lp[np.arange(len(lg)), d["test"].y].mean())
        log(f"{ds}: {S.members} members ready; member test accuracies "
            + ", ".join(f"{(logits['test'][m].argmax(1) == d['test'].y).mean():.3f}" for m in range(S.members)))

        # -------------------------------------------------------------- methods: temperatures on validation, predictive distributions
        yv, yt = d["val"].y, d["test"].y
        temps = {}
        for base, lg in method_samples(logits["val"], mc["val"]).items():
            temps[base] = fit_temperature(lg, yv)
            rows["temps"].append({"dataset": ds, "method": base, "temperature": temps[base]})
        log(f"{ds}: temperatures " + ", ".join(f"{k} {v:.3f}" for k, v in temps.items()))

        def methods(cond: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Mean probabilities (n, C) and per-sample probabilities (S, n, C) of the six methods on a condition."""
            out = {}
            for base, lg in method_samples(logits[cond], mc[cond]).items():
                for suffix, t in (("", 1.0), (" + TS", temps[base])):
                    sp = softmax(lg / t, axis=-1)
                    out[base + suffix] = (sp.mean(axis=0), sp)
            return out

        target = targets[ds]
        clean = methods("test")

        def evaluate() -> None:
            draws = bootstrap_indices(len(yt), S.n_boot, seed=0)
            boot = {}
            for name, (p, sp) in {**clean, PROBE: (pr["test_prob"], pr["test_prob"][None])}.items():
                u = scores(p, sp)["1 - max prob"]
                row = {"dataset": ds, "method": name, **summarize(p, yt, u, target),
                       "mean_entropy": float(scores(p)["entropy"].mean()), "target_accuracy": target}
                boot[name] = bootstrap_metrics(p, yt, u, draws)
                for m in BOOT_METRICS:
                    row[f"{m}_ci_low"], row[f"{m}_ci_high"] = np.percentile(boot[name][m], [2.5, 97.5])
                rows["methods"].append(row)
                if name == PROBE:
                    continue
                # risk-coverage curve on a 1% grid, reliability bins
                correct = p.argmax(1) == yt
                cov, risk = risk_coverage(u, correct)
                oracle = risk_coverage((~correct).astype(float), correct)[1]  # errors ranked least certain
                grid = np.arange(1, 101) / 100
                idx = np.clip(np.ceil(grid * len(yt)).astype(int) - 1, 0, len(yt) - 1)
                rows["rc"].append(pd.DataFrame({"dataset": ds, "method": name, "coverage": cov[idx], "risk": risk[idx],
                                                "oracle_risk": oracle[idx]}))
                rows["rel"].append(reliability(p, yt).assign(dataset=ds, method=name))
                # every uncertainty score of this method
                for sname, su in scores(p, sp).items():
                    s = summarize(p, yt, su, target)
                    rows["scores"].append({"dataset": ds, "method": name, "score": sname, "aurc": s["aurc"],
                                           "error_detection_auroc": s["error_detection_auroc"],
                                           "selective_acc_80": s["selective_acc_80"], "coverage_at_target": s["coverage_at_target"],
                                           "oracle_aurc": s["oracle_aurc"], "error_rate": 1 - correct.mean()})
            for a, b in COMPARISONS:
                for m in TEST_METRICS:
                    diff = boot[a][m] - boot[b][m]
                    pa = next(r for r in rows["methods"] if r["dataset"] == ds and r["method"] == a)[m]
                    pb = next(r for r in rows["methods"] if r["dataset"] == ds and r["method"] == b)[m]
                    rows["sig"].append({"dataset": ds, "method": a, "versus": b, "metric": m, "value": pa, "versus_value": pb,
                                        "diff": pa - pb, "diff_ci_low": np.percentile(diff, 2.5),
                                        "diff_ci_high": np.percentile(diff, 97.5), "p": paired_p(diff)})

        timer.run("evaluation", evaluate)

        # -------------------------------------------------------------- ensemble size and number of MC samples
        def sizes() -> None:
            for k in range(1, S.members + 1):
                for ts in (False, True):
                    t = fit_temperature(logits["val"][:k], yv) if ts else 1.0
                    p = softmax(logits["test"][:k] / t, axis=-1).mean(0)
                    rows["size"].append({"dataset": ds, "family": "Deep ensemble", "n": k, "temperature_scaled": ts,
                                         **summarize(p, yt, residual_mass(p), target)})
            for k in [s for s in (1, 2, 5, 10, 20, 50) if s <= S.mc_samples]:
                for ts in (False, True):
                    t = fit_temperature(mc["val"][:k], yv) if ts else 1.0
                    p = softmax(mc["test"][:k] / t, axis=-1).mean(0)
                    rows["size"].append({"dataset": ds, "family": "MC dropout", "n": k, "temperature_scaled": ts,
                                         **summarize(p, yt, residual_mass(p), target)})

        timer.run("evaluation", sizes)

        # -------------------------------------------------------------- deferral by class at the operating coverage
        for name in (BASELINE, "Deep ensemble + TS"):
            p = clean[name][0]
            u = residual_mass(p)
            k = int(round(S.coverage * len(yt)))
            accepted = np.zeros(len(yt), bool)
            accepted[np.argsort(u, kind="stable")[:k]] = True
            correct = p.argmax(1) == yt
            for c in range(n_classes):
                m = yt == c
                rows["deferral"].append({"dataset": ds, "method": name, "class_id": c, "class": SHORT[ds][c],
                                         "images": int(m.sum()), "deferred_share": float((~accepted[m]).mean()),
                                         "accuracy_all": float(correct[m].mean()),
                                         "accuracy_accepted": float(correct[m & accepted].mean()) if (m & accepted).any() else np.nan,
                                         "coverage": S.coverage})

        # -------------------------------------------------------------- corrupted test images
        val_m = methods("val")
        # deployment threshold: answer an image if 1 - max prob is below the value that answers S.coverage of validation images
        thresholds = {name: float(np.quantile(residual_mass(p), S.coverage)) for name, (p, _) in val_m.items()}

        def corrupted() -> None:
            for cname, sev, cond in [("none", 0, "test")] + [(*c.rsplit("-", 1), c) for c in conds]:
                for name, (p, sp) in (clean if cond == "test" else methods(cond)).items():
                    u = residual_mass(p)
                    answered = u <= thresholds[name]
                    correct = p.argmax(1) == yt
                    rows["corruption"].append({
                        "dataset": ds, "corruption": cname, "severity": int(sev), "method": name,
                        **summarize(p, yt, u, target),
                        "mean_entropy": float(scores(p)["entropy"].mean()),
                        "mean_mutual_information": float(mutual_information(sp).mean()) if len(sp) > 1 else np.nan,
                        "answered_share": float(answered.mean()),
                        "answered_accuracy": float(correct[answered].mean()) if answered.any() else np.nan})

        timer.run("evaluation", corrupted)

        # -------------------------------------------------------------- out-of-distribution images (the other data set)
        ood_m = methods("ood")
        for name, (p_in, sp_in) in clean.items():
            p_ood, sp_ood = ood_m[name]
            s_in, s_ood, s_val = scores(p_in, sp_in), scores(p_ood, sp_ood), scores(*val_m[name])
            top = np.bincount(p_ood.argmax(1), minlength=n_classes)
            for sname in s_in:
                thr = np.quantile(s_val[sname], S.coverage)
                lab = np.r_[np.zeros(len(s_in[sname]), bool), np.ones(len(s_ood[sname]), bool)]
                rows["ood"].append({"dataset": ds, "ood_dataset": other, "method": name, "score": sname,
                                    "auroc": binary_auc(np.r_[s_in[sname], s_ood[sname]], lab),
                                    "id_accepted": float((s_in[sname] <= thr).mean()),
                                    "ood_accepted": float((s_ood[sname] <= thr).mean()),
                                    "ood_mean_confidence": float(p_ood.max(1).mean()),
                                    "ood_top_class": SHORT[ds][int(top.argmax())],
                                    "ood_top_class_share": float(top.max() / len(p_ood))})

        # -------------------------------------------------------------- class imbalance (logit adjustment, weighted loss)
        if weighted is not None:
            tw = fit_temperature(weighted["val"]["logits"], yv)
            variants = {
                "Single model": clean[BASELINE][0],
                "Single model + logit adjustment": logit_adjust(clean[BASELINE][0], prior),
                "Class-weighted loss": softmax(weighted["test"]["logits"], axis=-1),
                "Class-weighted loss + TS": softmax(weighted["test"]["logits"] / tw, axis=-1),
                "Deep ensemble + TS": clean["Deep ensemble + TS"][0],
                "Deep ensemble + TS + logit adjustment": logit_adjust(clean["Deep ensemble + TS"][0], prior),
            }
            rows["temps"].append({"dataset": ds, "method": "Class-weighted loss", "temperature": tw})
            for name, p in variants.items():
                rows["imbalance"].append({"dataset": ds, "method": name, **summarize(p, yt, residual_mass(p), target)})
                rec = per_class_recall(p, yt, n_classes)
                for c in range(n_classes):
                    rows["imbalance_class"].append({"dataset": ds, "method": name, "class_id": c, "class": SHORT[ds][c],
                                                    "train_share": prior[c], "recall": rec[c],
                                                    "predicted_share": float((p.argmax(1) == c).mean())})

        # -------------------------------------------------------------- published benchmark
        for (pds, model), (auc_, acc_) in PUBLISHED_BENCHMARK.items():
            if pds == ds:
                rows["published"].append({"dataset": ds, "model": model, "source": "MedMNIST v2 (Yang et al., 2023)",
                                          "resolution": model.split("(")[-1].rstrip(")") if "(" in model else "28",
                                          "auc": auc_, "accuracy": acc_})
        for name in (PROBE, BASELINE, "Deep ensemble + TS"):
            r = next(r for r in rows["methods"] if r["dataset"] == ds and r["method"] == name)
            rows["published"].append({"dataset": ds, "model": f"{name}, this project", "source": "this project",
                                      "resolution": str(S.size), "auc": r["auc"], "accuracy": r["accuracy"]})

        # -------------------------------------------------------------- Grad-CAM on member 0 (examples and sanity check)
        def explain() -> None:
            m = model0
            if m is None:
                m = Classifier(n_classes, S.train.dropout, S.train.pretrained)
                wkey = key_of({**data_spec, "seed": 0, "cfg": asdict(S.train)})
                m.load_state_dict(torch.load(CACHE / "weights" / f"{ds}-plain-s0-{wkey}.pt", map_location="cpu",
                                             weights_only=False)["state"])
            m = m.to(dev).float().eval()
            p_ens = clean["Deep ensemble + TS"][0]
            p0 = clean[BASELINE][0]
            pred0, conf0 = p0.argmax(1), p0.max(1)
            correct0 = pred0 == yt
            u_ens = residual_mass(p_ens)
            rng = np.random.default_rng(0)
            confident_ok = np.flatnonzero(correct0 & (conf0 >= np.quantile(conf0, 0.5)))
            pick = {
                "confident and correct": rng.choice(confident_ok, size=min(3, len(confident_ok)), replace=False),
                "confident but wrong": np.flatnonzero(~correct0)[np.argsort(-conf0[~correct0])][:3],
                "most uncertain (ensemble)": np.argsort(-u_ens)[:3],
            }
            idx = np.concatenate(list(pick.values()))
            cat = np.concatenate([[k] * len(v) for k, v in pick.items()])
            xb = normalize(to_float(gpu["test"][torch.from_numpy(idx).to(dev)]))
            cams = gradcam.gradcam(m, xb, torch.from_numpy(pred0[idx]).to(dev), S.gradcam_layer)
            examples[f"{ds}_gc_images"] = d["test"].x[idx]
            examples[f"{ds}_gc_maps"] = gradcam.upsample(cams, S.size).astype(np.float16)
            for i, j in enumerate(idx):
                rows["gc_examples"].append({"dataset": ds, "order": i, "category": cat[i], "test_index": int(j),
                                            "label": SHORT[ds][yt[j]], "prediction": SHORT[ds][pred0[j]],
                                            "confidence_single": float(conf0[j]),
                                            "ensemble_prediction": SHORT[ds][p_ens[j].argmax()],
                                            "confidence_ensemble_ts": float(p_ens[j].max())})
            # sanity check: cascading randomisation of fc, layer4, layer3 on random test images
            sidx = np.sort(rng.choice(len(yt), size=min(S.gradcam_n, len(yt)), replace=False))
            xs = normalize(to_float(gpu["test"][torch.from_numpy(sidx).to(dev)]))
            tgt = torch.from_numpy(pred0[sidx]).to(dev)
            base_maps = np.concatenate([gradcam.gradcam(m, xs[i:i + 100], tgt[i:i + 100], S.gradcam_layer)
                                        for i in range(0, len(sidx), 100)])
            centre = gradcam.centre_share(gradcam.upsample(base_maps, S.size))
            rows["gc_sanity"].append({"dataset": ds, "randomized": "none (trained model)", "images": len(sidx),
                                      "mean_spearman": 1.0, "median_spearman": 1.0, "share_spearman_above_0_5": 1.0,
                                      "mean_centre_share": float(np.nanmean(centre)), "uniform_centre_share": 0.25})
            for blocks in (["fc"], ["fc", "layer4"], ["fc", "layer4", "layer3"]):
                r = gradcam.randomized(m, blocks, seed=0).to(dev).eval()
                maps = np.concatenate([gradcam.gradcam(r, xs[i:i + 100], tgt[i:i + 100], S.gradcam_layer)
                                       for i in range(0, len(sidx), 100)])
                rho = gradcam.rank_similarity(base_maps, maps)
                c_r = gradcam.centre_share(gradcam.upsample(maps, S.size))
                rows["gc_sanity"].append({"dataset": ds, "randomized": " + ".join(blocks), "images": len(sidx),
                                          "mean_spearman": float(np.nanmean(rho)), "median_spearman": float(np.nanmedian(rho)),
                                          "share_spearman_above_0_5": float(np.nanmean(rho > 0.5)),
                                          "mean_centre_share": float(np.nanmean(c_r)), "uniform_centre_share": 0.25})
            # sample training images per class for the data figure
            pick_tr = np.concatenate([np.flatnonzero(d["train"].y == c)[:2] for c in range(n_classes)])
            examples[f"{ds}_samples"] = d["train"].x[pick_tr]
            examples[f"{ds}_sample_labels"] = d["train"].y[pick_tr]

        timer.run("grad-cam", explain, gpu=True)
        gpu.clear()  # free the GPU copies before the next data set
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"{ds}: evaluation done")

    # ------------------------------------------------------------------ tables
    sig = pd.DataFrame(rows["sig"])
    sig["p_holm"] = holm(sig.p.tolist())
    out = {
        "members.csv": pd.DataFrame(rows["members"]),
        "training_history.csv": pd.concat(rows["history"], ignore_index=True),
        "method_results.csv": pd.DataFrame(rows["methods"]),
        "significance.csv": sig,
        "uncertainty_scores.csv": pd.DataFrame(rows["scores"]),
        "risk_coverage.csv": pd.concat(rows["rc"], ignore_index=True),
        "reliability.csv": pd.concat(rows["rel"], ignore_index=True),
        "ensemble_size.csv": pd.DataFrame(rows["size"]),
        "deferral_by_class.csv": pd.DataFrame(rows["deferral"]),
        "corruption_results.csv": pd.DataFrame(rows["corruption"]),
        "ood_detection.csv": pd.DataFrame(rows["ood"]),
        "imbalance.csv": pd.DataFrame(rows["imbalance"]),
        "imbalance_per_class.csv": pd.DataFrame(rows["imbalance_class"]),
        "published_comparison.csv": pd.DataFrame(rows["published"]),
        "temperatures.csv": pd.DataFrame(rows["temps"]),
        "gradcam_sanity.csv": pd.DataFrame(rows["gc_sanity"]),
        "gradcam_examples.csv": pd.DataFrame(rows["gc_examples"]),
    }
    for name, df in out.items():
        df.to_csv(results / name, index=False)
    np.savez_compressed(results / "image_examples.npz", **examples)

    init_mlflow(ROOT if not args.quick else CACHE / "smoke")
    for r in out["members.csv"].itertuples():
        log_result(f"member | {r.dataset} | {r.member} | seed {r.seed}",
                   {"dataset": r.dataset, "member": r.member, "seed": r.seed, "size": S.size,
                    **asdict(S.weighted if r.member == "weighted" else S.train)},
                   {"val_nll": r.val_nll, "val_acc": r.val_acc, "best_epoch": r.best_epoch, "train_seconds": r.train_seconds})
    for r in out["method_results.csv"].itertuples():
        log_result(f"method | {r.dataset} | {r.method}", {"dataset": r.dataset, "method": r.method, "members": S.members,
                                                           "mc_samples": S.mc_samples},
                   {k: getattr(r, k) for k in ["accuracy", "balanced_accuracy", "auc", "nll", "ece", "aurc", "coverage_at_target"]})

    meta = {
        "started_utc": started.isoformat(timespec="seconds"),
        "quick": args.quick,
        "settings": {k: v for k, v in asdict(S).items() if k not in ("train", "weighted")},
        "train_config": asdict(S.train), "weighted_config": asdict(S.weighted),
        "data": {ds: {"file": filename(ds, S.size), "md5": MD5[(ds, S.size)], "license": LICENSE[ds],
                      "published": PUBLISHED[ds], "classes": CLASSES[ds],
                      "observed": {s: len(data[ds][s]) for s in ("train", "val", "test")}} for ds in S.datasets},
        "corruption_severities": corruptions.SEVERITY,
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
        "gpu_stages": sorted(timer.gpu),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
    }
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    for p in make_all(results, figures):
        log(f"wrote {p.name}")
    log(f"done; stage seconds: {meta['timings_s']}")


if __name__ == "__main__":
    main()
