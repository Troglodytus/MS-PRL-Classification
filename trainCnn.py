#!/usr/bin/env python3
"""
trainCNN.py
-----------
Multi-modal 3D patch classifier for MS lesions (iron vs non-iron vs no_lesion).

Dependencies:
  - Python 3.9+
  - numpy, pandas, nibabel, torch, tqdm

Project files (same folder):
  - Excel_Parser.py         (from your save)
  - Class_CNN.py            (from your save)
  - MS_Data_Catalogue.xlsx  (Excel; in same folder unless --excel is given)

Lesion annotations (CSV or JSON):
  Required columns/fields: study, patient_id, TP, z, y, x, cls
  where cls ∈ {"iron", "non-iron"}; 'no_lesion' is generated as hard negatives.

Example:
  python trainCNN.py \
    --excel MS_Data_Catalogue.xlsx \
    --lesions lesions.csv \
    --allowed-modalities T2 FLAIR SWI Paramagnetic Diamagnetic \
    --patch-d 12 --patch-h 24 --patch-w 24 \
    --epochs 40 --batch-size 24 --lr 2e-4 --outdir ./runs/run1
"""

import os
import re
import json
import csv
import math
import random
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- local project modules ---
from Excel_Parser import build_patient_modalities_from_excel
from Class_CNN import SmallResNet3D


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_case_id(case_id: str):
    """
    case_id looks like: "{study}_{patient_id}_TP{num}"
    Returns (study, patient_id(str), tp_int)
    """
    m = re.match(r"(.+?)_(\d+)_TP(\d+)", case_id)
    if not m:
        # fallback: try to extract TP and last number
        study = case_id.split("_")[0]
        pid = re.findall(r"_(\d+)_", case_id)
        pid = pid[0] if pid else "0"
        tp = re.findall(r"TP(\d+)", case_id)
        tp = int(tp[0]) if tp else 0
        return study, pid, tp
    return m.group(1), m.group(2), int(m.group(3))


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Volume Cache (simple LRU)
# ----------------------------
class VolumeCache:
    def __init__(self, max_items: int = 64):
        self.max_items = max_items
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str):
        key = str(path)
        if key in self.cache:
            self.cache.move_to_end(key, last=True)
            return self.cache[key]
        # load and insert
        vol = np.asarray(nib.load(key).get_fdata(), dtype=np.float32)
        self.cache[key] = vol
        self.cache.move_to_end(key, last=True)
        # evict if needed
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return vol


# ----------------------------
# Lesion I/O
# ----------------------------
def load_lesions(path: str | None, cases: list[str]) -> dict[str, list[dict]]:
    """
    Load lesions from CSV or JSON and convert to:
      lesions[case_id] = [{"centroid": (z,y,x), "cls": "iron"|"non-iron"}, ...]

    Expected fields: study, patient_id, TP, z, y, x, cls
    """
    if path is None:
        raise RuntimeError("No lesions file provided. Please pass --lesions <csv|json>.")

    path = str(path)
    ext = Path(path).suffix.lower()
    rows = []

    if ext == ".csv":
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    elif ext == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
            if isinstance(payload, dict) and "lesions" in payload:
                rows = payload["lesions"]
            elif isinstance(payload, list):
                rows = payload
            else:
                raise ValueError("JSON lesions must be a list or have key 'lesions'.")
    else:
        raise ValueError("Lesions must be .csv or .json")

    lesions: dict[str, list[dict]] = defaultdict(list)
    # Build case_id the same way as Excel_Parser: f"{study}_{pid}_TP{TP}"
    case_set = set(cases)
    missing_case_ids = set()

    for r in rows:
        study = str(r.get("study"))
        pid = str(r.get("patient_id"))
        tp = int(r.get("TP"))
        z = float(r.get("z"))
        y = float(r.get("y"))
        x = float(r.get("x"))
        cls = str(r.get("cls")).strip().lower()
        if cls not in {"iron", "non-iron"}:
            # skip invalid classes
            continue
        case_id = f"{study}_{pid}_TP{tp}"
        if case_id not in case_set:
            missing_case_ids.add(case_id)
            continue
        lesions[case_id].append({"centroid": (int(round(z)), int(round(y)), int(round(x))), "cls": cls})

    if missing_case_ids:
        print(f"[warn] {len(missing_case_ids)} lesion entries refer to unknown case_ids (ignored). Examples: "
              f"{list(sorted(missing_case_ids))[:5]}")

    return lesions


# ----------------------------
# Dataset
# ----------------------------
class PatchDataset(Dataset):
    """
    Lesion-centric patches + hard negatives.

    Returns:
        x: FloatTensor [C, D, H, W]
        y: LongTensor scalar in {0:no_lesion, 1:iron, 2:non-iron}
    """
    def __init__(
        self,
        cases: list[str],
        modalities: dict[str, dict[str, str]],
        lesions: dict[str, list[dict]],
        patch: tuple[int, int, int] = (12, 24, 24),  # (D,H,W)
        neg_per_case: int = 80,
        min_neg_dist: float = 15.0,
        cache_max_items: int = 64,
        per_channel_zscore: bool = True,
        rng_seed: int = 123,
    ):
        self.cases = list(cases)
        self.modalities = modalities
        self.lesions = lesions
        self.patch = patch
        self.neg_per_case = neg_per_case
        self.min_neg_dist = float(min_neg_dist)
        self.vol_cache = VolumeCache(max_items=cache_max_items)
        self.per_channel_zscore = per_channel_zscore
        self.rng = np.random.default_rng(rng_seed)

        # Build item list: positives + negatives per case
        self.items: list[tuple[str, tuple[int,int,int], str]] = []

        # positives
        for case in self.cases:
            for L in self.lesions.get(case, []):
                self.items.append((case, tuple(map(int, L["centroid"])), L["cls"]))

        # negatives (far from lesions)
        for case in self.cases:
            # pick a reference modality just for shape
            ref_path = next(iter(self.modalities[case].values()))
            vol = self.vol_cache.get(ref_path)
            D, H, W = vol.shape
            cents = [np.array(l["centroid"]) for l in self.lesions.get(case, [])]
            cents_stack = np.vstack(cents) if len(cents) else None

            for _ in range(self.neg_per_case):
                for _attempt in range(200):
                    cz = self.rng.integers(self.patch[0]//2, D - self.patch[0]//2)
                    cy = self.rng.integers(self.patch[1]//2, H - self.patch[1]//2)
                    cx = self.rng.integers(self.patch[2]//2, W - self.patch[2]//2)
                    ok = True
                    if cents_stack is not None:
                        dist = np.min(np.linalg.norm(np.array([cz, cy, cx]) - cents_stack, axis=1))
                        ok = dist > self.min_neg_dist
                    if ok:
                        self.items.append((case, (int(cz), int(cy), int(cx)), "no_lesion"))
                        break

        # Shuffle for randomness
        random.shuffle(self.items)

        # Map labels
        self.label_map = {"no_lesion": 0, "iron": 1, "non-iron": 2}

    def __len__(self):
        return len(self.items)

    def _crop(self, vol: np.ndarray, center: tuple[int,int,int], size: tuple[int,int,int]) -> np.ndarray:
        d, h, w = size
        cz, cy, cx = center
        z0 = max(0, cz - d // 2); z1 = z0 + d
        y0 = max(0, cy - h // 2); y1 = y0 + h
        x0 = max(0, cx - w // 2); x1 = x0 + w
        # clamp if exceeding bounds
        D, H, W = vol.shape
        if z1 > D: z0 = D - d; z1 = D
        if y1 > H: y0 = H - h; y1 = H
        if x1 > W: x0 = W - w; x1 = W
        return vol[z0:z1, y0:y1, x0:x1]

    def __getitem__(self, idx):
        case, center, cls = self.items[idx]

        # stack channels in fixed modality order per case
        chans = []
        for mod_name, path in self.modalities[case].items():
            vol = self.vol_cache.get(path)  # [D,H,W], float32
            patch = self._crop(vol, center, self.patch)
            chans.append(patch)

        x = np.stack(chans, axis=0).astype(np.float32)  # [C,D,H,W]

        if self.per_channel_zscore:
            # z-score per channel inside the patch (robust to intensity shifts)
            # avoid div by zero
            c_mean = x.mean(axis=(2,3,4), keepdims=True)
            c_std  = x.std(axis=(2,3,4), keepdims=True) + 1e-6
            x = (x - c_mean) / c_std

        y = self.label_map[cls]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(model, loader, opt, device, scaler=None, class_weights=None):
    model.train()
    n, loss_sum, correct = 0, 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return loss_sum / max(n,1), correct / max(n,1)


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return loss_sum / max(n,1), correct / max(n,1)


# ----------------------------
# Patient-wise split
# ----------------------------
def patientwise_split(cases: list[str], val_fraction: float = 0.2, seed: int = 42):
    # split by patient_id to avoid leakage
    by_patient = defaultdict(list)
    for cid in cases:
        _, pid, _ = parse_case_id(cid)
        by_patient[pid].append(cid)

    pids = list(by_patient.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_fraction)))
    val_pids = set(pids[:n_val])

    train_cases, val_cases = [], []
    for pid, cids in by_patient.items():
        (val_cases if pid in val_pids else train_cases).extend(cids)

    return sorted(train_cases), sorted(val_cases)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train 3D CNN for MS lesion iron classification")
    parser.add_argument("--excel", type=str, default="MS_Data_Catalogue.xlsx")
    parser.add_argument("--image-type", type=str, default="crosscorrelated")
    parser.add_argument("--allowed-modalities", type=str, nargs="*", default=None,
                        help="Subset of modalities to use (e.g., T2 FLAIR SWI Paramagnetic Diamagnetic). If omitted, uses all found.")
    parser.add_argument("--te-policy", type=str, choices=["max","min","closest"], default="max")
    parser.add_argument("--te-closest-ms", type=float, default=None)

    parser.add_argument("--lesions", type=str, required=True,
                        help="CSV or JSON with lesion annotations (study, patient_id, TP, z, y, x, cls).")
    parser.add_argument("--val-frac", type=float, default=0.2)

    parser.add_argument("--patch-d", type=int, default=12)
    parser.add_argument("--patch-h", type=int, default=24)
    parser.add_argument("--patch-w", type=int, default=24)
    parser.add_argument("--neg-per-case", type=int, default=80)
    parser.add_argument("--min-neg-dist", type=float, default=15.0)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="./runs/run1")

    args = parser.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) Build cases & modalities from Excel
    cases, modalities = build_patient_modalities_from_excel(
        excel_name=args.excel,
        image_type_required=args.image_type,
        allowed_modalities=args.allowed_modalities,
        te_policy=args.te_policy,
        te_closest_ms=args.te_closest_ms,
        drop_missing_files=True,
    )

    if len(cases) == 0:
        raise RuntimeError("No cases found after filtering. Check --excel and --image-type.")

    # 2) Load lesions
    lesions = load_lesions(args.lesions, cases)
    total_pos = sum(len(v) for v in lesions.values())
    if total_pos == 0:
        raise RuntimeError("No positive lesions found in provided annotations. "
                           "Training requires iron/non-iron lesion entries.")

    # 3) Split by patient
    train_cases, val_cases = patientwise_split(cases, val_fraction=args.val_frac, seed=args.seed)

    # 4) Datasets / Loaders
    patch = (args.patch_d, args.patch_h, args.patch_w)

    train_ds = PatchDataset(
        train_cases, modalities, lesions,
        patch=patch, neg_per_case=args.neg_per_case,
        min_neg_dist=args.min_neg_dist, cache_max_items=64,
        per_channel_zscore=True, rng_seed=args.seed
    )
    val_ds = PatchDataset(
        val_cases, modalities, lesions,
        patch=patch, neg_per_case=max(1, args.neg_per_case // 2),
        min_neg_dist=args.min_neg_dist, cache_max_items=64,
        per_channel_zscore=True, rng_seed=args.seed + 1
    )

    # Class weights (downweight abundant 'no_lesion')
    # Simple heuristic: inverse frequency from training items
    labels = [lbl for _,_,cls in train_ds.items for lbl in [ {"no_lesion":0,"iron":1,"non-iron":2}[cls] ]]
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    with np.errstate(divide="ignore"):
        inv = 1.0 / np.clip(counts, 1.0, None)
    weights = inv / inv.sum() * 3.0
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # 5) Model, Optim, AMP
    in_ch = len(next(iter(modalities.values())))
    model = SmallResNet3D(in_channels=in_ch, num_classes=3)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if device.type == "cuda" else None
    cw = class_weights.to(device)

    # 6) Train
    best_val_acc = 0.0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device, scaler, cw)
        val_loss, val_acc = evaluate(model, val_loader, device, cw)

        log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        log_rows.append(log)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        # save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "args": vars(args),
            "class_weights": class_weights.cpu().numpy(),
        }, outdir / "checkpoint_latest.pt")

        # save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), outdir / "model_best.pt")

        # write log csv
        pd.DataFrame(log_rows).to_csv(outdir / "train_log.csv", index=False)

    print(f"Training finished. Best val acc: {best_val_acc:.3f}")
    print(f"Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
