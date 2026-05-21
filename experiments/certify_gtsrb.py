#!/usr/bin/env python3
"""Small-scale robustness certification for traffic-sign-like classifiers.

The script is intentionally lightweight:
- supports the official GTSRB archive if it is available locally,
- falls back to sklearn digits for a sanity-check run,
- trains a tiny 1-hidden-layer ReLU MLP using sklearn,
- certifies samples with IBP, LP relaxation, and MILP.

The MILP formulation uses scipy.optimize.milp, so no external solver is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
GTSRB_URLS = [
    "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_file(url: str, out_path: Path) -> bool:
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False


def maybe_download_gtsrb(root: Path) -> Path:
    """Return a zip archive path containing GTSRB data.

    Preferred source in this project is a Kaggle-style archive named `data.zip`
    with Train.csv/Test.csv plus Train/ and Test/ folders. If not present,
    fallback to downloading the official archive.
    """
    kaggle_archive = root / "data.zip"
    if kaggle_archive.exists():
        return kaggle_archive

    archive = root / "GTSRB_Final_Training_Images.zip"
    if archive.exists():
        return archive

    for url in GTSRB_URLS:
        if download_file(url, archive):
            return archive
    raise RuntimeError("failed to download GTSRB archive")


def _read_zip_csv(zf: zipfile.ZipFile, name: str) -> List[dict]:
    with zf.open(name) as f:
        text = TextIOWrapper(f, encoding="utf-8")
        reader = csv.DictReader(text)
        return list(reader)


def load_gtsrb_subset(
    root: Path,
    classes: Sequence[int],
    max_per_class: int,
    image_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a small balanced subset from a zip archive."""

    archive = maybe_download_gtsrb(root)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    rng = np.random.default_rng(0)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if "Train.csv" in names:
            rows = _read_zip_csv(zf, "Train.csv")
            grouped = {}
            for row in rows:
                cls = int(row["ClassId"])
                if cls in classes:
                    grouped.setdefault(cls, []).append(row)
            for cls in classes:
                cls_rows = grouped.get(cls, [])
                if len(cls_rows) > max_per_class:
                    idx = rng.choice(len(cls_rows), size=max_per_class, replace=False)
                    cls_rows = [cls_rows[i] for i in sorted(idx)]
                for row in cls_rows:
                    rel_path = row["Path"].replace("/code/", "")
                    with zf.open(rel_path) as f:
                        img = Image.open(f).convert("L").resize((image_size, image_size))
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                    xs.append(arr.reshape(-1))
                    ys.append(int(row["ClassId"]))
        else:
            zf.extractall(root / "GTSRB_Final_Training_Images")
            images_dir = root / "GTSRB_Final_Training_Images" / "GTSRB" / "Final_Training" / "Images"
            for cls in classes:
                class_dir = images_dir / f"{cls:05d}"
                csv_path = class_dir / f"GT-{cls:05d}.csv"
                rows: List[dict] = []
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    rows = list(reader)
                if len(rows) > max_per_class:
                    idx = rng.choice(len(rows), size=max_per_class, replace=False)
                    rows = [rows[i] for i in sorted(idx)]
                for row in rows:
                    img_path = class_dir / row["Filename"]
                    img = Image.open(img_path).convert("L").resize((image_size, image_size))
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    xs.append(arr.reshape(-1))
                    ys.append(int(row["ClassId"]))

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def load_digits_subset(
    classes: Sequence[int],
    image_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    x = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    mask = np.isin(y, np.asarray(classes))
    x = x[mask]
    y = y[mask]
    return x, y


@dataclass
class TinyMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.scaler_mean) / self.scaler_scale
        z1 = xs @ self.w1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        return a1 @ self.w2.T + self.b2

    def hidden_bounds(self, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo_s = (lo - self.scaler_mean) / self.scaler_scale
        hi_s = (hi - self.scaler_mean) / self.scaler_scale
        wpos = np.maximum(self.w1, 0.0)
        wneg = np.minimum(self.w1, 0.0)
        z_lo = lo_s @ wpos.T + hi_s @ wneg.T + self.b1
        z_hi = hi_s @ wpos.T + lo_s @ wneg.T + self.b1
        return z_lo, z_hi


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, hidden: int, seed: int) -> Tuple[TinyMLP, StandardScaler]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=seed,
        batch_size=64,
    )
    clf.fit(x_train_s, y_train)
    w1 = clf.coefs_[0].T.copy()
    b1 = clf.intercepts_[0].copy()
    w2 = clf.coefs_[1].T.copy()
    b2 = clf.intercepts_[1].copy()
    model = TinyMLP(w1=w1, b1=b1, w2=w2, b2=b2, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    return model, scaler


def ibp_margin_lower_bound(model: TinyMLP, x0: np.ndarray, eps: float, y: int, j: int) -> float:
    lo = np.clip(x0 - eps, 0.0, 1.0)
    hi = np.clip(x0 + eps, 0.0, 1.0)
    z_lo, z_hi = model.hidden_bounds(lo, hi)
    a_lo = np.maximum(z_lo, 0.0)
    a_hi = np.maximum(z_hi, 0.0)
    wy = model.w2[y]
    wj = model.w2[j]
    margin_w = wy - wj
    margin_b = model.b2[y] - model.b2[j]
    wpos = np.maximum(margin_w, 0.0)
    wneg = np.minimum(margin_w, 0.0)
    return float(wpos @ a_lo + wneg @ a_hi + margin_b)


def build_lp_relaxation(model: TinyMLP, x0: np.ndarray, eps: float, y: int, j: int):
    lo = np.clip(x0 - eps, 0.0, 1.0)
    hi = np.clip(x0 + eps, 0.0, 1.0)
    d = x0.size
    h = model.w1.shape[0]

    z_lo, z_hi = model.hidden_bounds(lo, hi)

    n = d + 2 * h
    x_idx = slice(0, d)
    z_idx = slice(d, d + h)
    a_idx = slice(d + h, d + 2 * h)

    c = np.zeros(n, dtype=np.float64)
    margin_w = model.w2[y] - model.w2[j]
    c[a_idx] = margin_w
    const = float(model.b2[y] - model.b2[j])

    # z = W1 * x_scaled + b1
    # x_scaled = (x - mean) / scale
    W_eff = model.w1 / model.scaler_scale[np.newaxis, :]
    b_eff = model.b1 - (model.w1 @ (model.scaler_mean / model.scaler_scale))
    A_eq = np.zeros((h, n), dtype=np.float64)
    A_eq[:, x_idx] = -W_eff
    A_eq[:, z_idx] = np.eye(h)
    b_eq = b_eff.copy()

    A_ub = []
    b_ub = []
    bounds = [(float(lo[i]), float(hi[i])) for i in range(d)] + [(-np.inf, np.inf)] * (2 * h)
    for i in range(h):
        l = float(z_lo[i])
        u = float(z_hi[i])
        # a >= 0
        row = np.zeros(n, dtype=np.float64)
        row[d + h + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        if u <= 0:
            # stable inactive
            row = np.zeros(n, dtype=np.float64)
            row[d + h + i] = 1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 0.0)
            continue
        if l >= 0:
            row = np.zeros(n, dtype=np.float64)
            row[d + h + i] = 1.0
            row[d + i] = -1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 0.0)
            continue
        # a >= z
        row = np.zeros(n, dtype=np.float64)
        row[d + i] = 1.0
        row[d + h + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # a <= alpha (z - l)
        alpha = u / (u - l)
        row = np.zeros(n, dtype=np.float64)
        row[d + i] = -alpha
        row[d + h + i] = 1.0
        A_ub.append(row)
        b_ub.append(-alpha * l)
    return c, const, np.asarray(A_ub), np.asarray(b_ub), A_eq, b_eq, bounds


def lp_margin_lower_bound(model: TinyMLP, x0: np.ndarray, eps: float, y: int, j: int) -> float:
    c, const, A_ub, b_ub, A_eq, b_eq, bounds = build_lp_relaxation(model, x0, eps, y, j)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return float("nan")
    return float(res.fun + const)


def milp_margin_lower_bound(model: TinyMLP, x0: np.ndarray, eps: float, y: int, j: int) -> float:
    lo = np.clip(x0 - eps, 0.0, 1.0)
    hi = np.clip(x0 + eps, 0.0, 1.0)
    d = x0.size
    h = model.w1.shape[0]
    z_lo, z_hi = model.hidden_bounds(lo, hi)

    n = d + 3 * h
    x_idx = slice(0, d)
    z_idx = slice(d, d + h)
    a_idx = slice(d + h, d + 2 * h)
    s_idx = slice(d + 2 * h, d + 2 * h + h)

    c = np.zeros(n, dtype=np.float64)
    c[a_idx] = model.w2[y] - model.w2[j]
    const = float(model.b2[y] - model.b2[j])

    W_eff = model.w1 / model.scaler_scale[np.newaxis, :]
    b_eff = model.b1 - (model.w1 @ (model.scaler_mean / model.scaler_scale))
    A_eq = np.zeros((h, n), dtype=np.float64)
    A_eq[:, x_idx] = -W_eff
    A_eq[:, z_idx] = np.eye(h)
    b_eq = b_eff.copy()

    A_ub = []
    b_ub = []
    bounds = [(float(lo[i]), float(hi[i])) for i in range(d)]
    bounds.extend([(-np.inf, np.inf)] * h)  # pre-activation z
    bounds.extend([(0.0, np.inf)] * h)      # activation a
    bounds.extend([(0.0, 1.0)] * h)         # binary selector s
    integrality = np.zeros(n, dtype=int)
    for i in range(h):
        l = float(z_lo[i])
        u = float(z_hi[i])
        if u <= 0:
            # stable inactive: a = 0, s = 0
            row = np.zeros(n, dtype=np.float64)
            row[d + h + i] = 1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 0.0)
            row = np.zeros(n, dtype=np.float64)
            row[d + 2 * h + i] = 1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 0.0)
            continue
        if l >= 0:
            # stable active: a = z, s = 1
            row = np.zeros(n, dtype=np.float64)
            row[d + h + i] = 1.0
            row[d + i] = -1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 0.0)
            row = np.zeros(n, dtype=np.float64)
            row[d + 2 * h + i] = 1.0
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, 1.0)
            continue
        # unstable
        integrality[d + 2 * h + i] = 1
        # a >= z
        row = np.zeros(n, dtype=np.float64)
        row[d + i] = 1.0
        row[d + h + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # a >= 0
        row = np.zeros(n, dtype=np.float64)
        row[d + h + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # a <= z - l(1-s)
        row = np.zeros(n, dtype=np.float64)
        row[d + i] = -1.0
        row[d + h + i] = 1.0
        row[d + 2 * h + i] = -l
        A_ub.append(row)
        b_ub.append(-l)
        # a <= u s
        row = np.zeros(n, dtype=np.float64)
        row[d + h + i] = 1.0
        row[d + 2 * h + i] = -u
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = np.asarray(A_eq)
    A_ub = np.asarray(A_ub)
    constraints = []
    if A_eq.size:
        constraints.append(LinearConstraint(A_eq, np.asarray(b_eq), np.asarray(b_eq)))
    if A_ub.size:
        constraints.append(LinearConstraint(A_ub, -np.inf, np.asarray(b_ub)))
    lower = np.asarray([b[0] for b in bounds], dtype=np.float64)
    upper = np.asarray([b[1] for b in bounds], dtype=np.float64)
    res = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints)
    if not res.success:
        return float("nan")
    return float(res.fun + const)


def certify_sample(model: TinyMLP, x0: np.ndarray, y: int, eps: float) -> dict:
    logits = model.forward(x0[None, :])[0]
    pred = int(np.argmax(logits))
    out = {
        "y": int(y),
        "pred": pred,
        "clean_correct": pred == y,
        "ibp": True,
        "lp": True,
        "milp": True,
        "ibp_bounds": {},
        "lp_bounds": {},
        "milp_bounds": {},
    }
    for j in range(logits.size):
        if j == y:
            continue
        lb1 = ibp_margin_lower_bound(model, x0, eps, y, j)
        lb2 = lp_margin_lower_bound(model, x0, eps, y, j)
        lb3 = milp_margin_lower_bound(model, x0, eps, y, j)
        out["ibp_bounds"][j] = lb1
        out["lp_bounds"][j] = lb2
        out["milp_bounds"][j] = lb3
        out["ibp"] = out["ibp"] and (lb1 > 0)
        out["lp"] = out["lp"] and (lb2 > 0)
        out["milp"] = out["milp"] and (lb3 > 0)
    return out


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.dataset == "gtsrb":
        classes = [int(c) for c in args.classes.split(",")]
        x, y = load_gtsrb_subset(DATA_ROOT, classes=classes, max_per_class=args.max_per_class, image_size=args.image_size)
    else:
        classes = [int(c) for c in args.classes.split(",")]
        x, y = load_digits_subset(classes=classes)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    model, _ = train_mlp(x_train, y_train, hidden=args.hidden, seed=args.seed)

    rows = []
    times = []
    for i in range(min(args.n_eval, len(x_test))):
        x0 = x_test[i]
        yy = int(y_test[i])
        t0 = time.perf_counter()
        res = certify_sample(model, x0, yy, args.eps)
        t1 = time.perf_counter()
        res["time"] = t1 - t0
        rows.append(res)
        times.append(res["time"])
        print(json.dumps(res, ensure_ascii=False))

    clean_acc = float(np.mean([r["clean_correct"] for r in rows]))
    ibp_acc = float(np.mean([r["ibp"] for r in rows]))
    lp_acc = float(np.mean([r["lp"] for r in rows]))
    milp_acc = float(np.mean([r["milp"] for r in rows]))
    avg_time = float(np.mean(times))

    summary = {
        "dataset": args.dataset,
        "classes": classes,
        "n_eval": len(rows),
        "eps": args.eps,
        "clean_accuracy": clean_acc,
        "ibp_certified_accuracy": ibp_acc,
        "lp_certified_accuracy": lp_acc,
        "milp_certified_accuracy": milp_acc,
        "avg_verify_time_sec": avg_time,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.out_dir / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sweep_eps(args: argparse.Namespace) -> None:
    eps_values = [float(x) for x in args.eps_list.split(",")]
    all_rows = []
    for eps in eps_values:
        args2 = argparse.Namespace(**vars(args))
        args2.eps = eps
        run_experiment(args2)
        summary_path = args2.out_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["eps"] = eps
        all_rows.append(summary)
    with open(args.out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    try:
        import matplotlib.pyplot as plt
        eps = [r["eps"] for r in all_rows]
        ibp = [r["ibp_certified_accuracy"] for r in all_rows]
        lp = [r["lp_certified_accuracy"] for r in all_rows]
        milp = [r["milp_certified_accuracy"] for r in all_rows]
        plt.figure(figsize=(5, 3.2))
        plt.plot(eps, ibp, marker="o", label="IBP")
        plt.plot(eps, lp, marker="o", label="LP")
        plt.plot(eps, milp, marker="o", label="MILP")
        plt.xlabel(r"$\varepsilon$")
        plt.ylabel("certified accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "certified_accuracy_curve.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"plot failed: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["gtsrb", "digits"], default="gtsrb")
    p.add_argument("--classes", default="0,1,2,3,4")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--max-per-class", type=int, default=150)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--eps", type=float, default=0.08)
    p.add_argument("--eps-list", default="0.005,0.01,0.02,0.04")
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "sweep":
        sweep_eps(args)
    else:
        run_experiment(args)
