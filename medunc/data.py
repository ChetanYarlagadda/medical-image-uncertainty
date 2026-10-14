"""Download, verify and load the MedMNIST+ subsets used here (DermaMNIST and BloodMNIST).

The arrays come from the MedMNIST+ Zenodo record 10519652 (version 3.0, the files the ``medmnist`` package
downloads). Each file is checked against the MD5 published in ``medmnist.info.INFO``, and the split sizes are
compared with the numbers in the MedMNIST v2 paper (Yang et al., 2023).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ZENODO = "https://zenodo.org/records/10519652/files"

# MD5 digests from medmnist.info.INFO (medmnist 3.0.2), identical to the Zenodo record's checksums.
MD5 = {
    ("dermamnist", 28): "0744692d530f8e62ec473284d019b0c7",
    ("dermamnist", 64): "b70a2f5635c6199aeaa28c31d7202e1f",
    ("dermamnist", 128): "2defd784463fa5243564e855ed717de1",
    ("bloodmnist", 28): "7053d0359d879ad8a5505303e11de1dc",
    ("bloodmnist", 64): "2b94928a2ae4916078ca51e05b6b800b",
    ("bloodmnist", 128): "adace1e0ed228fccda1f39692059dd4c",
}

# Split sizes and class counts published in the MedMNIST v2 paper (Table 1) and medmnist.info.INFO.
PUBLISHED = {
    "dermamnist": {"train": 7_007, "val": 1_003, "test": 2_005, "classes": 7},
    "bloodmnist": {"train": 11_959, "val": 1_712, "test": 3_421, "classes": 8},
}

CLASSES = {
    "dermamnist": ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
                   "melanoma", "melanocytic nevus", "vascular lesion"],
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocyte", "lymphocyte",
                   "monocyte", "neutrophil", "platelet"],
}
SHORT = {
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "bloodmnist": ["baso", "eosi", "eryth", "ig", "lymph", "mono", "neut", "plt"],
}
TITLE = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}
LICENSE = {"dermamnist": "CC BY-NC 4.0", "bloodmnist": "CC BY 4.0"}
SPLITS = ("train", "val", "test")


@dataclass
class Split:
    """Images of one split as uint8 (n, h, w, 3) and integer labels (n,)."""

    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def take(self, idx: np.ndarray) -> "Split":
        """Rows ``idx`` of the split."""
        return Split(self.x[idx], self.y[idx])


def md5(path: Path) -> str:
    """Hex MD5 of a file (the digest MedMNIST publishes)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filename(name: str, size: int) -> str:
    """Zenodo file name of a subset at one resolution."""
    return f"{name}.npz" if size == 28 else f"{name}_{size}.npz"


def download(name: str, size: int, raw_dir: Path) -> Path:
    """Download one MedMNIST+ file once (streamed) and verify its MD5."""
    path = raw_dir / filename(name, size)
    expected = MD5[(name, size)]
    if path.exists() and md5(path) == expected:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(f"{ZENODO}/{filename(name, size)}?download=1", stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    got = md5(tmp)
    if got != expected:
        tmp.unlink()
        raise ValueError(f"MD5 mismatch for {path.name}: {got} != {expected}")
    tmp.replace(path)
    return path


def load(name: str, size: int, raw_dir: Path) -> dict[str, Split]:
    """Train, validation and test splits of a downloaded subset."""
    z = np.load(download(name, size, raw_dir))
    out = {}
    for s in SPLITS:
        x = z[f"{s}_images"]
        if x.ndim == 3:  # grey-scale subsets: repeat to three channels
            x = np.repeat(x[..., None], 3, axis=-1)
        out[s] = Split(np.ascontiguousarray(x, dtype=np.uint8), z[f"{s}_labels"].reshape(-1).astype(np.int64))
    return out


def class_table(name: str, data: dict[str, Split]) -> pd.DataFrame:
    """Images per class and split, with the class share in the training split."""
    rows = []
    n_classes = len(CLASSES[name])
    for c in range(n_classes):
        row = {"dataset": name, "class_id": c, "class": CLASSES[name][c], "short": SHORT[name][c]}
        for s in SPLITS:
            row[s] = int((data[s].y == c).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t["train_share"] = t.train / t.train.sum()
    return t


def crosscheck(name: str, data: dict[str, Split]) -> pd.DataFrame:
    """Compare split sizes and the number of classes with the published values."""
    pub = PUBLISHED[name]
    obs = {s: len(data[s]) for s in SPLITS}
    obs["classes"] = int(max(int(data[s].y.max()) for s in SPLITS) + 1)
    rows = [{"dataset": name, "statistic": k, "published": v, "observed": obs[k], "match": v == obs[k]} for k, v in pub.items()]
    return pd.DataFrame(rows)


def medmnist_info_check(name: str, size: int) -> pd.DataFrame:
    """Compare the pinned digest, licence, split sizes and class names with ``medmnist.info.INFO`` (if installed)."""
    try:
        from medmnist.info import INFO
    except ImportError:
        return pd.DataFrame()
    info = INFO[name]
    pairs = {
        f"MD5 of {filename(name, size)}": (info["MD5" if size == 28 else f"MD5_{size}"], MD5[(name, size)]),
        "license": (info["license"], LICENSE[name]),
        "classes": (len(info["label"]), len(CLASSES[name])),
        **{f"{s} images": (info["n_samples"][s], PUBLISHED[name][s]) for s in SPLITS},
    }
    return pd.DataFrame([{"dataset": name, "statistic": k, "published": a, "observed": b, "match": a == b,
                          "source": "medmnist.info.INFO vs this project's pinned values"} for k, (a, b) in pairs.items()])
