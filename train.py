"""
Class-discriminative single-link WiFi domain generalization for CSI gesture recognition.
Training engine (Phase 3). Signal processing lives in preprocess.py, data preparation in
prepare_data.py, the preprocessing gate in diagnose.py, tables/figures in analyze.py.

- Model: dual-branch ConvNet (GroupNorm) + 2-weight softmax gate, L2-normalised 64-d embedding (<100k params)
- C0-C5: episodic prototype training, class-consistency loss, augmentation curriculum
- B1-B4: ERM, Deep CORAL, DANN, Mixup on the same fused encoder with a linear head
- Job queue: static sharding over worker processes, resumable; final / quick / pilot outputs kept apart
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.autograd import Function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("single_link_dg")

GESTURE_NAMES = {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide", 5: "Draw-O(H)", 6: "Draw-Zigzag(H)"}
NUM_CLASSES = 6
TASKS = {
    "O1": {"train_orientations": [3, 4, 5], "val_orientation": 2, "target_orientation": 1},
    "O3": {"train_orientations": [1, 2, 5], "val_orientation": 4, "target_orientation": 3},
    "O5": {"train_orientations": [2, 3, 4], "val_orientation": 1, "target_orientation": 5},
}
# Queue order: main comparison first, then the remaining ablation arms, then external baselines.
CONFIGS = ["C1", "C5", "C0", "C2", "C3", "C4", "B1", "B2", "B3", "B4"]
SEEDS = [42, 43, 44]
TEMPERATURE = 0.1          # prototype logits = cosine / 0.1; the linear head of B1-B4 sees z / 0.1 (same scale)
CLIP_PCT = (0.5, 99.5)     # train-only clipping percentiles per channel
CORAL_WEIGHT = float(os.environ.get("CORAL_WEIGHT", "1000.0"))   # B2 loss weight: unweighted CORAL on z/T is ~1e-3 of the CE, so it is scaled to be comparable (Sun & Saenko 2016)
PARTS = ("train", "id_test", "val", "target")
CLAP, SLIDE = 2, 3         # 0-indexed gesture 3 (Clap) and gesture 4 (Slide)


# ==========================================
# 1. SPLITS
# ==========================================

def generate_splits_for_task(manifest_df: pd.DataFrame, task_name: str, val_fraction: float = 0.20,
                             seed: int = 0) -> Dict[str, np.ndarray]:
    """Leave-one-orientation-out split. Target = one orientation, source-val = another, the other three
    are train orientations; 20% of the trials of every (gesture, train orientation) cell form the locked
    in-domain test. Only valid trials are assigned."""
    spec = TASKS[task_name]
    valid = manifest_df["valid"].to_numpy()
    ori = manifest_df["orientation"].to_numpy()
    idx = {"target": np.where(valid & (ori == spec["target_orientation"]))[0],
           "val": np.where(valid & (ori == spec["val_orientation"]))[0]}
    cand = manifest_df[valid & np.isin(ori, spec["train_orientations"])]
    rng = np.random.RandomState(seed)
    train, id_test = [], []
    for _, group in cand.groupby(["gesture", "orientation"]):
        ix = group.index.to_numpy().copy()
        rng.shuffle(ix)
        n_test = int(np.round(len(ix) * val_fraction))
        id_test.extend(ix[:n_test]); train.extend(ix[n_test:])
    idx["train"] = np.sort(np.array(train, dtype=np.int64))
    idx["id_test"] = np.sort(np.array(id_test, dtype=np.int64))
    verify_splits(manifest_df, task_name, idx)
    return idx


def verify_splits(manifest: pd.DataFrame, task: str, idx: Dict[str, np.ndarray]) -> None:
    """Leakage invariants: partitions are disjoint by row and by trial_id, contain only valid trials,
    and each contains only the orientations it is allowed to contain."""
    spec = TASKS[task]
    tid = manifest["trial_id"].to_numpy()
    assert len(set(tid)) == len(tid), "duplicate trial_id in manifest"
    for i, a in enumerate(PARTS):
        assert len(idx[a]) > 0, f"[{task}] empty partition {a}"
        assert manifest["valid"].to_numpy()[idx[a]].all(), f"[{task}] invalid trial in {a}"
        for b in PARTS[i + 1:]:
            assert not (set(idx[a].tolist()) & set(idx[b].tolist())), f"[{task}] leakage: {a}/{b} share rows"
            assert not (set(tid[idx[a]]) & set(tid[idx[b]])), f"[{task}] leakage: {a}/{b} share trial_id"
    o = manifest["orientation"].to_numpy()
    assert np.isin(o[idx["train"]], spec["train_orientations"]).all()
    assert np.isin(o[idx["id_test"]], spec["train_orientations"]).all()
    assert (o[idx["val"]] == spec["val_orientation"]).all()
    assert (o[idx["target"]] == spec["target_orientation"]).all()
    assert not np.isin(spec["target_orientation"], o[np.concatenate([idx["train"], idx["id_test"], idx["val"]])])


def load_manifest(manifest_csv: str) -> pd.DataFrame:
    m = pd.read_csv(manifest_csv, keep_default_na=False)
    m["valid"] = m["valid"].astype(str).str.lower().eq("true")
    return m


def load_task_splits(manifest: pd.DataFrame, splits_csv: str, task: str) -> Dict[str, np.ndarray]:
    """Read the saved split assignment and re-verify every leakage invariant."""
    sp = pd.read_csv(splits_csv, keep_default_na=False)
    assert len(sp) == len(manifest) and (sp["trial_id"].values == manifest["trial_id"].values).all(), \
        "splits.csv does not align with manifest.csv"
    col = sp[f"split_{task}"].values
    idx = {p: np.where(col == p)[0] for p in PARTS}
    verify_splits(manifest, task, idx)
    return idx


def subsample_parts(manifest: pd.DataFrame, idx: Dict[str, np.ndarray], fraction: float) -> Dict[str, np.ndarray]:
    """Quick-mode subset: keep `fraction` of each partition, stratified by (gesture, orientation), seed 0."""
    if fraction >= 1.0:
        return idx
    rng = np.random.RandomState(0)
    out = {}
    for p, ix in idx.items():
        keep = []
        for _, g in manifest.loc[ix, ["gesture", "orientation"]].groupby(["gesture", "orientation"]):
            gi = g.index.to_numpy()
            keep.extend(rng.choice(gi, size=min(len(gi), max(3, int(round(len(gi) * fraction)))), replace=False))
        out[p] = np.sort(np.array(keep, dtype=np.int64))
    return out


# ==========================================
# 2. MODEL
# ==========================================

class ConvBlock(nn.Module):
    """Conv2d 3x3 -> GroupNorm(8) -> ReLU -> MaxPool2d(2)."""
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.gn(self.conv(x))))


class BranchEncoder(nn.Module):
    """3 conv blocks (16, 32, 64) + global average pooling: (B, 1, 64, 128) -> (B, 64)."""
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.gap(self.block3(self.block2(self.block1(x)))), 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class SingleLinkCSIModel(nn.Module):
    """Phase branch (+ Doppler branch and a learnable 2-weight softmax gate unless config == C0)
    -> L2-normalised 64-d embedding. B1-B4 add a linear head on the embedding / TEMPERATURE."""
    def __init__(self, config: str = "C5", num_classes: int = NUM_CLASSES):
        super().__init__()
        self.config = config
        self.phase_encoder = BranchEncoder()
        self.doppler_encoder = BranchEncoder() if config != "C0" else None
        self.gate_logits = nn.Parameter(torch.zeros(2)) if config != "C0" else None
        self.classifier = nn.Linear(64, num_classes) if config.startswith("B") else None
        n_params = count_params(self)
        assert n_params < 100_000, f"{config}: {n_params} trainable params exceeds the 100k budget"

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = self.phase_encoder(x[:, 0:1])
        if self.doppler_encoder is not None:
            w = F.softmax(self.gate_logits, dim=0)
            z = w[0] * z + w[1] * self.doppler_encoder(x[:, 1:2])
        return F.normalize(z, p=2, dim=-1)

    def head(self, z: torch.Tensor) -> torch.Tensor:
        assert self.classifier is not None, "linear head exists only for B1-B4"
        return self.classifier(z / TEMPERATURE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_embedding(x)
        return self.head(z) if self.classifier is not None else z


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """B3 (DANN): predicts which of the 3 train orientations an embedding comes from."""
    def __init__(self, in_features: int = 64, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(inplace=True), nn.Linear(32, num_domains))

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(GradReverse.apply(x, alpha))


def consistency_loss(z_sup: torch.Tensor, z_qry: torch.Tensor) -> torch.Tensor:
    """Class-consistency on the clean embeddings of one episode. z_sup (C, O, 64), z_qry (C, O, Q, 64).
    Per-class per-orientation centre -> class centre -> mean cosine distance of orientation centres to it."""
    allz = torch.cat([z_sup.unsqueeze(2), z_qry], dim=2).float()
    ori_c = F.normalize(allz.mean(dim=2), p=2, dim=-1)
    cls_c = F.normalize(ori_c.mean(dim=1, keepdim=True), p=2, dim=-1)
    return (1.0 - (ori_c * cls_c).sum(-1)).mean()


# ==========================================
# 3. DATA ON DEVICE, NORMALISATION, AUGMENTATION
# ==========================================

class GPUData:
    """Raw fp16 features resident on the device. Clipping limits (0.5 / 99.5 percentile) and z-score
    statistics are computed per channel from the TRAIN partition only and then frozen."""
    def __init__(self, features: np.ndarray, train_idx: np.ndarray, device: torch.device):
        self.raw = torch.from_numpy(np.ascontiguousarray(features)).to(device)
        tr = np.asarray(features[np.sort(train_idx)], dtype=np.float32)
        lo, hi = np.percentile(tr, CLIP_PCT, axis=(0, 2, 3))
        assert (hi > lo).all(), "degenerate train feature range"
        trc = np.clip(tr, lo[None, :, None, None], hi[None, :, None, None])
        as_t = lambda v: torch.tensor(v, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.lo, self.hi = as_t(lo), as_t(hi)
        self.mean, self.std = as_t(trc.mean(axis=(0, 2, 3))), as_t(trc.std(axis=(0, 2, 3)) + 1e-6)
        self.channels_last = device.type == "cuda"

    def norm(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = (torch.maximum(torch.minimum(x_raw, self.hi), self.lo) - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    def get(self, idx) -> torch.Tensor:
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.raw.device)
        return self.raw[idx].float()

    def stats(self) -> Dict[str, List[float]]:
        return {k: getattr(self, k).flatten().tolist() for k in ("lo", "hi", "mean", "std")}


def augment(x_raw: torch.Tensor, data: GPUData, s: float, m: int, f: int, g_db: float,
            doppler_scale: str, gen: torch.Generator) -> torch.Tensor:
    """Label-preserving query augmentation, vectorised on the device. x_raw: (B, 2, 64, 128) raw maps
    (channel 0 phase, channel 1 Doppler). Returns NORMALISED augmented maps.
      1. time-stretch by U(1-s, 1+s), both channels (centred crop with random shift / edge replication)
      2. Doppler gain U(-g, g) dB applied to POWER, Doppler channel only, before clipping and z-scoring:
         log1p maps  x' = log1p(expm1(x) * 10^(g/10));  dB maps  x' = x + g
      3. time mask: one block of U{1..m} frames, both channels, set to the train mean (0 after z-scoring)
      4. Doppler-bin mask: one block of U{1..f} bins, Doppler channel only, set to the train mean
    """
    assert doppler_scale in ("log1p", "db")
    B, C, H, W = x_raw.shape
    dev = x_raw.device
    rand = lambda *shape: torch.rand(*shape, generator=gen, device=dev)
    x = x_raw
    if s > 0.0:
        fac = 1.0 + (2.0 * rand(B) - 1.0) * s
        shift = (2.0 * rand(B) - 1.0) * torch.clamp(1.0 - 1.0 / fac, min=0.0)
        u = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) * 2.0 / W - 1.0
        v = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) * 2.0 / H - 1.0
        gx = (u[None, :] / fac[:, None] + shift[:, None])[:, None, :].expand(B, H, W)
        gy = v[None, :, None].expand(B, H, W)
        x = F.grid_sample(x, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border",
                          align_corners=False)
    if g_db > 0.0 and C > 1:
        g = (2.0 * rand(B) - 1.0) * g_db
        if doppler_scale == "log1p":
            d = torch.log1p(torch.expm1(x[:, 1].clamp_min(0)) * (10.0 ** (g / 10.0)).view(B, 1, 1))
        else:
            d = x[:, 1] + g.view(B, 1, 1)
        x = torch.stack([x[:, 0], d], dim=1)
    x = data.norm(x)
    if m > 0:
        L = 1 + torch.floor(rand(B) * m).long()
        a = torch.floor(rand(B) * (W - L + 1).float()).long()
        cols = torch.arange(W, device=dev)[None, :]
        x = x.masked_fill(((cols >= a[:, None]) & (cols < (a + L)[:, None]))[:, None, None, :], 0.0)
    if f > 0 and C > 1:
        L = 1 + torch.floor(rand(B) * f).long()
        a = torch.floor(rand(B) * (H - L + 1).float()).long()
        rows = torch.arange(H, device=dev)[None, :]
        mask = torch.zeros(B, C, H, W, dtype=torch.bool, device=dev)
        mask[:, 1] = ((rows >= a[:, None]) & (rows < (a + L)[:, None]))[:, :, None]
        x = x.masked_fill(mask, 0.0)
    return x


AUG_NONE = (0.0, 0, 0, 0.0)
AUG_LEVEL1 = (0.05, 8, 4, 1.0)      # (stretch s, time-mask frames m, Doppler-bin mask f, gain dB)
AUG_LEVEL2 = (0.10, 16, 8, 2.0)
AUG_FIXED = (0.08, 13, 6, 1.6)      # epoch-weighted mean of the curriculum over epochs 6-30 (10 x L1, 15 x L2)
# schedule name -> (last warm-up epoch, last level-1 epoch). "compressed" exists only so that a 3-epoch
# quick-mode run exercises every augmentation level; final runs must use "full".
SCHEDULES = {"full": (5, 15), "compressed": (1, 2)}


def get_curriculum_aug_params(epoch: int, config: str, schedule: str = "full") -> Tuple[float, int, int, float]:
    """C2/C5: curriculum. C4: clean warm-up, then the constant level AUG_FIXED. Others: none. Epochs are 1-indexed."""
    warm, l1_end = SCHEDULES[schedule]
    if config not in ("C2", "C4", "C5") or epoch <= warm:
        return AUG_NONE
    if config == "C4":
        return AUG_FIXED
    return AUG_LEVEL1 if epoch <= l1_end else AUG_LEVEL2


# ==========================================
# 4. SAMPLERS, PROTOTYPES, METRICS
# ==========================================

class CellPools:
    """Train trial indices per (class, train orientation), on the device."""
    def __init__(self, labels: np.ndarray, orientations: np.ndarray, train_idx: np.ndarray,
                 device: torch.device, min_per_cell: int):
        self.device = device
        self.orients = sorted(set(orientations[train_idx].tolist()))
        assert len(self.orients) == 3, f"expected 3 train orientations, got {self.orients}"
        self.pools = {}
        for c in range(1, NUM_CLASSES + 1):
            for o in self.orients:
                pool = train_idx[(labels[train_idx] == c) & (orientations[train_idx] == o)]
                assert len(pool) >= min_per_cell, f"class {c} orientation {o}: only {len(pool)} train trials"
                self.pools[(c, o)] = torch.as_tensor(pool, dtype=torch.long, device=device)
        self.cells = list(self.pools)

    def draw(self, cell, k: int) -> torch.Tensor:
        pool = self.pools[cell]
        return pool[torch.randperm(len(pool), device=self.device)[:k]]

    def episode(self, n_orient: int = 2, n_sup: int = 1, n_qry: int = 2):
        """6 classes x 2 of the 3 train orientations x (1 support + 2 query) distinct trials.
        Layout: support (C, O), query (C, O, Q), flattened in that order."""
        oris = np.random.choice(self.orients, size=n_orient, replace=False)
        sup, qry = [], []
        for c in range(1, NUM_CLASSES + 1):
            for o in oris:
                ch = self.draw((c, int(o)), n_sup + n_qry)
                sup.append(ch[:n_sup]); qry.append(ch[n_sup:])
        q_lbl = torch.arange(NUM_CLASSES, device=self.device).repeat_interleave(n_orient * n_qry)
        return torch.cat(sup), torch.cat(qry), q_lbl

    def balanced_batch(self, size: int = 64) -> torch.Tensor:
        """Balanced mini-batch for B1-B4: floor(size / 18) trials from every (class, orientation) cell,
        the remainder from distinct randomly chosen cells."""
        base, rem = divmod(size, len(self.cells))
        extra = set(np.random.choice(len(self.cells), size=rem, replace=False).tolist())
        return torch.cat([self.draw(cell, base + (i in extra)) for i, cell in enumerate(self.cells)])


@torch.no_grad()
def embed(model: nn.Module, data: GPUData, idx: np.ndarray, bs: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model.forward_embedding(data.norm(data.get(idx[i:i + bs]))).float()
                      for i in range(0, len(idx), bs)], dim=0)


def prototype_bank(z: torch.Tensor, lab: np.ndarray, ori: np.ndarray) -> torch.Tensor:
    """Class prototype = normalised mean over train orientations of the normalised per-orientation
    centroid of clean train embeddings (equal weight per orientation)."""
    protos = []
    for c in range(1, NUM_CLASSES + 1):
        cents = []
        for o in sorted(set(ori.tolist())):
            mk = torch.as_tensor((lab == c) & (ori == o), device=z.device)
            assert mk.any(), f"no train embeddings for class {c} orientation {o}"
            cents.append(F.normalize(z[mk].mean(0), dim=-1))
        protos.append(F.normalize(torch.stack(cents).mean(0), dim=-1))
    return torch.stack(protos)


@torch.no_grad()
def logits_of(model: SingleLinkCSIModel, z: torch.Tensor, prototypes: Optional[torch.Tensor]) -> torch.Tensor:
    """Evaluation-only logits (prototype cosine / temperature, or the linear head); never used in training."""
    return (z @ prototypes.T / TEMPERATURE) if prototypes is not None else model.head(z).float()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    n_clap, n_slide = int(cm[CLAP].sum()), int(cm[SLIDE].sum())
    assert n_clap > 0 and n_slide > 0, "clap/slide absent from evaluation split"
    c2s, s2c = int(cm[CLAP, SLIDE]), int(cm[SLIDE, CLAP])
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(per_c.mean()), "per_class_f1": per_c.tolist(), "worst_f1": float(per_c.min()),
            "clap_slide_error": 0.5 * (c2s / n_clap + s2c / n_slide),
            "clap_as_slide": c2s, "n_clap": n_clap, "slide_as_clap": s2c, "n_slide": n_slide, "cm": cm.tolist()}


# ==========================================
# 5. TRAINING
# ==========================================

def _atomic_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def make_job_id(config: str, task: str, seed: int, lambda_cons: float) -> str:
    jid = f"{config}_{task}_s{seed}"
    if config in ("C3", "C4", "C5") and abs(lambda_cons - 0.10) > 1e-9:
        jid += f"_lam{lambda_cons:g}"
    return jid


def load_features_meta(features_path: str) -> Dict[str, Any]:
    """features_meta.json is written next to features.npy by prepare_data.py and records the locked
    preprocessing configuration and the SHA-256 of the feature file."""
    meta_path = Path(features_path).with_name("features_meta.json")
    assert meta_path.exists(), f"{meta_path} missing: rebuild the cache with prepare_data.py --assemble"
    return json.load(open(meta_path))


def train_single_run(
    config: str, task_name: str, seed: int, features_path: str, manifest_csv: str, splits_csv: str,
    out_dir: str, run_kind: str, lambda_cons: float = 0.10, epochs: int = 30, lr: float = 1e-3,
    weight_decay: float = 1e-4, data_fraction: float = 1.0, schedule: str = "full",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """One (config, task, seed) run. Writes OUT_DIR/runs/<job_id>/{metrics.json, preds.npz, best.pt, last.pt}.
    run_kind in {final, quick, pilot} is stored and checked on resume so that smoke-test and pilot outputs
    can never be mistaken for final results."""
    assert config in CONFIGS and task_name in TASKS and run_kind in ("final", "quick", "pilot")
    assert schedule in SCHEDULES
    if run_kind == "final":
        assert epochs == 30 and data_fraction == 1.0 and schedule == "full", "final runs use the full protocol"
    job_id = make_job_id(config, task_name, seed, lambda_cons)
    run_dir = Path(out_dir) / "runs" / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "metrics.json"
    if results_file.exists():
        saved = json.load(open(results_file))
        for k, v in (("epochs", epochs), ("data_fraction", data_fraction), ("run_kind", run_kind),
                     ("lambda_cons", lambda_cons), ("schedule", schedule)):
            assert saved[k] == v, f"{job_id}: existing result has {k}={saved[k]}, requested {v}"
        logger.info(f"[{job_id}] already complete, skipping.")
        return saved

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    aug_gen = torch.Generator(device=device); aug_gen.manual_seed(seed + 1000)

    fmeta = load_features_meta(features_path)
    doppler_scale = fmeta["preprocessing"]["doppler_variant"].split("_")[-1]
    manifest = load_manifest(manifest_csv)
    splits = subsample_parts(manifest, load_task_splits(manifest, splits_csv, task_name), data_fraction)
    feats = np.load(features_path, mmap_mode="r")
    assert feats.shape[0] == len(manifest) == fmeta["n_trials"] and feats.shape[1:] == (2, 64, 128) \
        and feats.dtype == np.float16, "feature cache does not match manifest / metadata"
    labels = manifest["gesture"].to_numpy()
    orientations = manifest["orientation"].to_numpy()
    train_idx = splits["train"]
    data = GPUData(feats, train_idx, device)          # normalisation statistics: train partition only

    model = SingleLinkCSIModel(config=config).to(device)
    params = list(model.parameters())
    disc = None
    if config == "B3":                                # discriminator created once, trained by the main optimizer
        disc = DomainDiscriminator(64, num_domains=3).to(device)
        params += list(disc.parameters())
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_params = count_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    episodic = config.startswith("C")
    n_train = len(train_idx)
    steps_per_epoch = math.ceil(n_train / 36.0) if episodic else math.ceil(n_train / 64.0)
    pools = CellPools(labels, orientations, train_idx, device, min_per_cell=3 if episodic else 4)
    dom_lbl_all = torch.as_tensor([{o: i for i, o in enumerate(pools.orients)}.get(o, -1) for o in orientations],
                                  device=device)
    lab_all = torch.as_tensor(labels - 1, device=device)
    use_cons = config in ("C3", "C4", "C5") and lambda_cons > 0
    tr_lab, tr_ori = labels[train_idx], orientations[train_idx]

    def evaluate(parts):
        """Frozen evaluation: prototypes (C configs) are rebuilt from clean train embeddings only."""
        z_tr = embed(model, data, train_idx)
        protos = prototype_bank(z_tr, tr_lab, tr_ori) if episodic else None
        out = {"_train_acc": float((logits_of(model, z_tr, protos).argmax(1).cpu().numpy() == tr_lab - 1).mean()),
               "_protos": protos}
        for part in parts:
            z = embed(model, data, splits[part])
            conf, pred = F.softmax(logits_of(model, z, protos), dim=-1).max(dim=-1)
            out[part] = (compute_metrics(labels[splits[part]] - 1, pred.cpu().numpy()),
                         pred.cpu().numpy(), conf.cpu().numpy(), z.cpu().numpy())
        return out

    curves = {k: [] for k in ("train_loss", "ce", "aux", "train_acc", "val_macro_f1", "epoch_time_s", "eval_time_s")}
    best_val, best_epoch = -1.0, -1
    best_ckpt, last_ckpt = run_dir / "best.pt", run_dir / "last.pt"
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        aug = get_curriculum_aug_params(epoch, config, schedule)
        aug_on = aug != AUG_NONE
        losses, ces, auxs = [], [], []
        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if episodic:
                    s_idx, q_idx, q_lbl = pools.episode()
                    z_s = model.forward_embedding(data.norm(data.get(s_idx)))            # supports always clean
                    P = F.normalize(z_s.float().view(NUM_CLASSES, 2, 64).mean(1), dim=-1)  # equal weight per orientation
                    q_raw = data.get(q_idx)
                    z_q = model.forward_embedding(data.norm(q_raw))                      # clean queries
                    ce = F.cross_entropy(z_q.float() @ P.T / TEMPERATURE, q_lbl)
                    if aug_on:
                        z_qa = model.forward_embedding(augment(q_raw, data, *aug, doppler_scale, aug_gen))
                        ce = 0.5 * (ce + F.cross_entropy(z_qa.float() @ P.T / TEMPERATURE, q_lbl))
                    loss = ce
                    if use_cons:
                        cons = consistency_loss(z_s.view(NUM_CLASSES, 2, 64), z_q.view(NUM_CLASSES, 2, 2, 64))
                        loss = ce + lambda_cons * cons
                        auxs.append(cons.item())
                else:
                    b_idx = pools.balanced_batch(64)
                    bx, by = data.norm(data.get(b_idx)), lab_all[b_idx]
                    if config == "B4":
                        lam = float(np.random.beta(0.2, 0.2))
                        pm = torch.randperm(bx.size(0), device=device)
                        logits = model(lam * bx + (1 - lam) * bx[pm]).float()
                        ce = lam * F.cross_entropy(logits, by) + (1 - lam) * F.cross_entropy(logits, by[pm])
                        loss = ce
                    else:
                        z = model.forward_embedding(bx)
                        ce = F.cross_entropy(model.head(z).float(), by)
                        loss = ce
                        if config == "B2":            # Deep CORAL on the classifier-input features (z / T), fp32 covariance
                            zf, dl = z.float() / TEMPERATURE, dom_lbl_all[b_idx]
                            covs = [torch.cov(zf[dl == d].T) for d in range(3)]
                            pairs = [(0, 1), (0, 2), (1, 2)]
                            coral = sum(((covs[i] - covs[j]) ** 2).sum() / (4 * 64 * 64) for i, j in pairs) / 3
                            loss = ce + CORAL_WEIGHT * coral
                            auxs.append(coral.item())
                        elif config == "B3":                                              # DANN
                            prog = ((epoch - 1) + step / steps_per_epoch) / epochs
                            alpha = 2.0 / (1.0 + math.exp(-10.0 * prog)) - 1.0
                            dom = F.cross_entropy(disc(z.float(), alpha), dom_lbl_all[b_idx])
                            loss = ce + dom
                            auxs.append(dom.item())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"[{job_id}] non-finite loss at epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item()); ces.append(ce.item())
        train_time = time.time() - t_ep

        t_ev = time.time()
        ev = evaluate(("val",))
        val_f1 = ev["val"][0]["macro_f1"]
        curves["train_loss"].append(float(np.mean(losses))); curves["ce"].append(float(np.mean(ces)))
        curves["aux"].append(float(np.mean(auxs)) if auxs else None)
        curves["train_acc"].append(ev["_train_acc"]); curves["val_macro_f1"].append(val_f1)
        curves["epoch_time_s"].append(train_time); curves["eval_time_s"].append(time.time() - t_ev)
        if val_f1 > best_val:                          # first epoch wins ties
            best_val, best_epoch = val_f1, epoch
            torch.save({"model": model.state_dict(), "prototypes": ev["_protos"], "norm": data.stats(),
                        "epoch": epoch}, best_ckpt)
        logger.info(f"[{job_id}] ep {epoch:02d}/{epochs} loss {curves['train_loss'][-1]:.4f} "
                    f"train-acc {ev['_train_acc']:.4f} val-F1 {val_f1:.4f} aug {aug} "
                    f"train {train_time:.1f}s eval {curves['eval_time_s'][-1]:.1f}s")
    train_time_total = time.time() - t_start

    eval_parts = ("id_test", "val", "target")
    last = evaluate(eval_parts)
    torch.save({"model": model.state_dict(), "prototypes": last["_protos"], "norm": data.stats(), "epoch": epochs},
               last_ckpt)
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
    best = evaluate(eval_parts)

    # batch-1 latency of the frozen model (embedding + prototype / linear head); excludes feature extraction
    model.eval()
    x1 = data.norm(data.get(splits["target"][:1]))
    lat = []
    with torch.no_grad():
        for i in range(150):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = logits_of(model, model.forward_embedding(x1).float(), best["_protos"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 50:
                lat.append((time.perf_counter() - t0) * 1000.0)

    gate = F.softmax(model.gate_logits.detach().float(), dim=0).tolist() if model.gate_logits is not None else None
    results = {
        "job_id": job_id, "config": config, "task": task_name, "seed": seed, "lambda_cons": lambda_cons,
        "epochs": epochs, "data_fraction": data_fraction, "run_kind": run_kind, "schedule": schedule,
        "preprocessing": fmeta["preprocessing"], "features_sha256": fmeta["sha256"],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "n_params": n_params,
        "n_train": int(n_train), "n_id_test": int(len(splits["id_test"])), "n_val": int(len(splits["val"])),
        "n_target": int(len(splits["target"])), "steps_per_epoch": steps_per_epoch,
        "train_time_s": train_time_total,
        "mean_epoch_time_s": float(np.mean(np.array(curves["epoch_time_s"]) + np.array(curves["eval_time_s"]))),
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "latency_median_ms": float(np.median(lat)), "latency_p95_ms": float(np.percentile(lat, 95)),
        "best_epoch": best_epoch, "best_source_val_f1": best_val, "gate_weights_phase_doppler": gate,
        "norm_stats_train_only": data.stats(),
        "best": {p: best[p][0] for p in eval_parts}, "last": {p: last[p][0] for p in eval_parts},
        "train_acc_best": best["_train_acc"],
        "mean_conf_target": float(best["target"][2].mean()),
        "training_curves": curves,
    }
    tids = manifest["trial_id"].to_numpy()
    np.savez_compressed(
        run_dir / "preds.npz",
        **{f"{part}_{k}": v for part in eval_parts for k, v in (
            ("trial_id", tids[splits[part]]), ("y_true", labels[splits[part]]),
            ("y_pred", best[part][1] + 1), ("conf", best[part][2]))},
        target_embedding=best["target"][3], target_y_pred_last=last["target"][1] + 1,
    )
    _atomic_json(results, results_file)
    logger.info(f"[{job_id}] DONE {train_time_total:.0f}s | target acc {results['best']['target']['accuracy']:.4f} "
                f"F1 {results['best']['target']['macro_f1']:.4f} | id-test F1 {results['best']['id_test']['macro_f1']:.4f} "
                f"| best epoch {best_epoch}")
    return results


# ==========================================
# 6. UNIT TESTS
# ==========================================

def run_unit_tests():
    """Shapes, L2 norm, parameter budget, prototype logit scale, consistency loss, augmentation levels,
    curriculum schedules, balanced batches, split leakage checks."""
    torch.manual_seed(0); np.random.seed(0)
    x = torch.randn(4, 2, 64, 128)
    counts = {}
    for cfg in ["C0", "C1", "C5", "B1"]:
        m = SingleLinkCSIModel(config=cfg)
        z = m.forward_embedding(x)
        assert z.shape == (4, 64) and torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)
        counts[cfg] = count_params(m)
    assert counts["C1"] == counts["C5"] < 100_000 and counts["C0"] < counts["C1"] < counts["B1"]
    logger.info(f"  trainable parameters: {counts}")

    # prototype logits: cosine / 0.1 -> self-similarity gives 10
    P = F.normalize(torch.randn(6, 64), dim=-1)
    assert torch.isclose((P[0:1] @ P.T / TEMPERATURE)[0, 0], torch.tensor(10.0), atol=1e-4)
    z = F.normalize(torch.randn(60, 64), dim=-1)
    lab, ori = np.repeat(np.arange(1, 7), 10), np.tile(np.array([3, 4] * 5), 6)
    bank = prototype_bank(z, lab, ori)
    assert bank.shape == (6, 64) and torch.allclose(bank.norm(dim=-1), torch.ones(6), atol=1e-5)

    # consistency: zero when every orientation has the same centre, positive otherwise
    base = F.normalize(torch.randn(6, 1, 64), dim=-1)
    assert consistency_loss(base.expand(6, 2, 64), base.unsqueeze(2).expand(6, 2, 2, 64)).item() < 1e-6
    assert consistency_loss(F.normalize(torch.randn(6, 2, 64), dim=-1),
                            F.normalize(torch.randn(6, 2, 2, 64), dim=-1)).item() > 1e-3

    # augmentation, every curriculum level, both Doppler scales
    dev = torch.device("cpu")
    for scale in ("log1p", "db"):
        raw = F.interpolate(torch.randn(24, 2, 4, 8), size=(64, 128), mode="bicubic", align_corners=False)
        raw[:, 1] = raw[:, 1].abs() + 0.5
        if scale == "db":
            raw[:, 1] = raw[:, 1] * 10 - 30
        feats = raw.numpy().astype(np.float16)
        data = GPUData(feats, np.arange(24), dev)
        xr = data.get(np.arange(24))
        gen = torch.Generator(device=dev); gen.manual_seed(0)
        clean = data.norm(xr)
        assert torch.equal(augment(xr, data, *AUG_NONE, scale, gen), clean), "no-op augmentation changed the input"
        for name, lvl in (("level1", AUG_LEVEL1), ("level2", AUG_LEVEL2), ("fixed", AUG_FIXED)):
            s, m_, f_, g_ = lvl
            out = augment(xr, data, *lvl, scale, gen)
            assert out.shape == clean.shape and torch.isfinite(out).all(), name
            # label preservation proxy: every augmented sample is still closest to its own clean sample
            sim = F.normalize(out.flatten(1), dim=1) @ F.normalize(clean.flatten(1), dim=1).T
            assert (sim.argmax(1) == torch.arange(24)).all(), f"{name}: augmentation changed sample identity"
            # time mask only: at most m columns differ, identical columns in both channels
            o = augment(xr, data, 0.0, m_, 0, 0.0, scale, gen)
            changed = (o != clean).flatten(1, 2).any(1)
            assert (changed.sum(1) <= m_).all() and (changed.sum(1) >= 1).any(), name
            # Doppler-bin mask only: phase channel untouched, at most f Doppler rows differ
            o = augment(xr, data, 0.0, 0, f_, 0.0, scale, gen)
            assert torch.equal(o[:, 0], clean[:, 0]), f"{name}: Doppler-bin mask leaked into the phase channel"
            rows = (o[:, 1] != clean[:, 1]).any(2).sum(1)
            assert (rows <= f_).all() and (rows >= 1).any(), name
            # gain only: phase channel untouched, Doppler changes by at most g dB of power
            big = GPUData(feats, np.arange(24), dev); big.lo -= 1e4; big.hi += 1e4    # disable clipping for the check
            o = augment(xr, big, 0.0, 0, 0, g_, scale, gen)
            assert torch.equal(o[:, 0], big.norm(xr)[:, 0]), f"{name}: gain leaked into the phase channel"
            d_raw = o[:, 1] * big.std[0, 1] + big.mean[0, 1]
            if scale == "db":
                shift = d_raw - xr[:, 1]
                assert torch.allclose(shift, shift[:, :1, :1].expand_as(shift), atol=1e-2)
                assert shift.abs().max() <= g_ + 1e-2
            else:
                k = torch.expm1(d_raw) / torch.expm1(xr[:, 1])
                assert torch.allclose(k, k[:, :1, :1].expand_as(k), rtol=2e-2)
                assert (10 * torch.log10(k[:, 0, 0])).abs().max() <= g_ + 0.05
            # stretch only: shape kept, finite, not identical
            o = augment(xr, data, s, 0, 0, 0.0, scale, gen)
            assert o.shape == clean.shape and not torch.equal(o, clean)

    # curriculum schedules
    assert [get_curriculum_aug_params(e, "C5") for e in (1, 5, 6, 15, 16, 30)] == \
        [AUG_NONE, AUG_NONE, AUG_LEVEL1, AUG_LEVEL1, AUG_LEVEL2, AUG_LEVEL2]
    assert [get_curriculum_aug_params(e, "C4") for e in (5, 6, 30)] == [AUG_NONE, AUG_FIXED, AUG_FIXED]
    assert [get_curriculum_aug_params(e, "C5", "compressed") for e in (1, 2, 3)] == [AUG_NONE, AUG_LEVEL1, AUG_LEVEL2]
    assert all(get_curriculum_aug_params(e, c) == AUG_NONE for e in (1, 30) for c in ("C0", "C1", "C3", "B1"))
    mean_lvl = tuple((10 * a + 15 * b) / 25 for a, b in zip(AUG_LEVEL1, AUG_LEVEL2))
    assert all(abs(a - b) <= 0.5 for a, b in zip(AUG_FIXED, mean_lvl)) and abs(AUG_FIXED[0] - mean_lvl[0]) < 1e-9

    # samplers and split leakage checks on a synthetic manifest (metadata only; no synthetic signals are trained on)
    rows = [dict(trial_id=f"u{u}-{g}-{o}-{r}", user=u, gesture=g, orientation=o, valid=True)
            for u in (1, 2) for g in range(1, 7) for o in range(1, 6) for r in range(10)]
    man = pd.DataFrame(rows)
    for task in TASKS:
        sp = generate_splits_for_task(man, task)
        assert sum(len(v) for v in sp.values()) == len(man)
        bad = dict(sp); bad["train"] = np.concatenate([sp["train"], sp["target"][:1]])
        try:
            verify_splits(man, task, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("verify_splits accepted a split with target leakage")
    sp = generate_splits_for_task(man, "O1")
    pools = CellPools(man["gesture"].to_numpy(), man["orientation"].to_numpy(), sp["train"], dev, 4)
    b = pools.balanced_batch(64).numpy()
    assert len(b) == 64 and len(set(b.tolist())) == 64 and set(b.tolist()) <= set(sp["train"].tolist())
    cell_counts = man.loc[b].groupby(["gesture", "orientation"]).size()
    assert len(cell_counts) == 18 and cell_counts.min() >= 3 and cell_counts.max() <= 4
    s_idx, q_idx, q_lbl = pools.episode()
    assert len(s_idx) == 12 and len(q_idx) == 24 and not (set(s_idx.tolist()) & set(q_idx.tolist()))
    assert (man["gesture"].to_numpy()[q_idx.numpy()] - 1 == q_lbl.numpy()).all()
    assert len(build_jobs()) == 96
    logger.info("All unit tests PASSED.")


# ==========================================
# 7. JOB QUEUE, PILOT / ETA, CLI
# ==========================================

def build_jobs(configs=CONFIGS, tasks=tuple(TASKS), seeds=SEEDS, sensitivity=True) -> List[Dict[str, Any]]:
    """10 configs x 3 tasks x 3 seeds = 90, plus C5 with lambda in {0.05, 0.3} x 3 tasks (seed 42) = 96
    trainings. Order: C1 and C5 first, then C0/C2/C3/C4, then B1-B4, then the lambda runs."""
    jobs = [dict(config=c, task=t, seed=s, lambda_cons=0.10) for c in configs for t in tasks for s in seeds]
    if sensitivity:
        jobs += [dict(config="C5", task=t, seed=42, lambda_cons=lam) for t in tasks for lam in (0.05, 0.30)]
    return jobs


def run_queue(jobs, worker: int, num_workers: int, **kw) -> None:
    """Static sharding (job i -> worker i % num_workers); resumable because finished jobs are skipped."""
    mine = [j for i, j in enumerate(jobs) if i % num_workers == worker]
    logger.info(f"Worker {worker}/{num_workers}: {len(mine)} of {len(jobs)} jobs")
    for k, j in enumerate(mine, 1):
        logger.info(f"--- worker {worker} job {k}/{len(mine)}: {j}")
        train_single_run(config=j["config"], task_name=j["task"], seed=j["seed"], lambda_cons=j["lambda_cons"], **kw)


def eta_from_pilot(out_dir: str, n_workers: int, jobs: List[Dict[str, Any]], epochs: int = 30) -> Dict[str, float]:
    """ETA for the queue from measured per-epoch times of pilot runs (C5 for episodic, B3 for baselines)."""
    runs = {}
    for p in Path(out_dir).glob("runs/*/metrics.json"):
        r = json.load(open(p))
        runs[r["config"]] = r
    assert "C5" in runs and "B3" in runs, "the pilot needs one C5 and one B3 run"
    t_c, t_b = runs["C5"]["mean_epoch_time_s"], runs["B3"]["mean_epoch_time_s"]
    tot = sum((t_c if j["config"].startswith("C") else t_b) * epochs for j in jobs)
    return {"sec_per_epoch_C5": t_c, "sec_per_epoch_B3": t_b, "n_jobs": len(jobs),
            "total_hours_1_worker": tot / 3600, "n_workers": n_workers,
            "wall_hours_if_workers_scale_linearly": tot / 3600 / n_workers}


def main():
    ap = argparse.ArgumentParser(description="Single-link WiFi DG: training engine")
    ap.add_argument("--features", default="cache/features.npy")
    ap.add_argument("--manifest_csv", default="cache/manifest.csv")
    ap.add_argument("--splits_csv", default="cache/splits.csv")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--run_kind", choices=["final", "quick", "pilot"], default="final")
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_cons", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data_fraction", type=float, default=1.0)
    ap.add_argument("--schedule", choices=list(SCHEDULES), default="full")
    ap.add_argument("--queue", action="store_true", help="run the job queue (96 trainings unless filtered)")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--no_sensitivity", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index for this worker")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (CPU runs only)")
    ap.add_argument("--eta", type=int, default=0, help="print the queue ETA from pilot runs in --out_dir for N workers")
    ap.add_argument("--unit_tests", action="store_true")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    if a.unit_tests:
        import preprocess
        preprocess.run_unit_tests()
        run_unit_tests()
        return
    out_dir = a.out_dir or {"final": "results/final", "quick": "results_test/quick", "pilot": "results_test/pilot"}[a.run_kind]
    order = {c: i for i, c in enumerate(CONFIGS)}
    jobs = build_jobs(configs=sorted(a.configs or CONFIGS, key=order.get), tasks=a.tasks or list(TASKS),
                      seeds=a.seeds or SEEDS, sensitivity=not a.no_sensitivity)
    if a.eta:
        print(json.dumps(eta_from_pilot(out_dir, a.eta, jobs), indent=2)); return
    kw = dict(features_path=a.features, manifest_csv=a.manifest_csv, splits_csv=a.splits_csv, out_dir=out_dir,
              run_kind=a.run_kind, epochs=a.epochs, data_fraction=a.data_fraction, schedule=a.schedule,
              device_str=f"cuda:{a.gpu}" if a.gpu is not None else None)
    if a.queue:
        run_queue(jobs, a.worker, a.num_workers, **kw)
        return
    assert a.config and a.task, "--config and --task are required for a single run"
    train_single_run(config=a.config, task_name=a.task, seed=a.seed, lambda_cons=a.lambda_cons, **kw)


if __name__ == "__main__":
    main()
