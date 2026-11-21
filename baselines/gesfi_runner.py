"""
Runs the base-paper method, GesFi (Zhang et al., IMWUT 2026), on the single-link Widar3.0 benchmark.

What is GesFi's own code (imported unchanged from https://github.com/CamLoPA/GesFiCode at a pinned commit, GPL-3.0;
it is cloned at run time and not copied into this repository):
  the GeneFi model (ImageNet-pretrained ResNet18 + bottlenecks + latent-domain classifier + two GRL discriminators),
  its three update steps (update_a: pre-learning, update_d: latent-domain characterisation, update: domain-invariant
  learning), set_dlabel (latent-domain clustering), its image augmentations and its optimiser/scheduler settings.
What this runner adds:
  1. Input rendering. GesFi reads 224x224 RGB images of phase and Doppler maps produced elsewhere. Our phase and Doppler
     maps (cache/features.npy) are clipped with train-partition percentiles, mapped through a fixed colour map, stacked
     (phase above Doppler, 128x128) and resized to 224x224. Rendered in memory for each task; no target statistic used.
  2. Our splits (cache/splits.csv) and metrics (train.compute_metrics), so results are directly comparable with B1-B4/C*.
  3. Three model-selection rules, reported side by side:
       sourceval  epoch with the best source-validation macro-F1 (the strict rule used for every other method)
       last       final epoch
       oracle     epoch with the best target accuracy: the rule implemented in GesFi's released trainer
                  (bestac = max test accuracy). Reported only to measure how much it changes the result; it uses
                  target labels and is not a valid domain-generalization protocol.
  4. Per-epoch checkpoints so a disconnected Colab session resumes where it stopped.

Training loop and hyperparameters follow GesFi's released csimain.py/utils.py: Adam(lr 2e-4, wd 5e-4, betas (0.5, 0.9)),
MultiStepLR(milestones [10, 20], gamma 0.1), batch 32, 50 epochs, 3 latent domains, alpha = alpha1 = 0.1, lam = 0;
pre-learning runs two passes of update_a in rounds 0 and 1. The released code uses lr 2e-4; the paper text states 0.002.

  python baselines/gesfi_runner.py --gesfi_dir <clone of GesFiCode> --task O1 --seed 42 --out_dir results/gesfi
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as T  # noqa: E402

GESFI_REPO = "https://github.com/CamLoPA/GesFiCode"
GESFI_COMMIT = "421e2d21f980737da31c017725134cdce9253672"   # 2025-04-26, the only published version at the time of writing
SELECTION_RULES = ("sourceval", "last", "oracle")


# --------------------------------------------------------------------------- GesFi code
def fetch_gesfi(dest: Path) -> Path:
    """Clone GesFi at the pinned commit (or verify an existing clone) and return its path."""
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "-q", GESFI_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", GESFI_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == GESFI_COMMIT, f"GesFi clone is at {head}, expected {GESFI_COMMIT}"
    return dest


def import_gesfi(gesfi_dir: Path):
    sys.path.insert(0, str(gesfi_dir))
    import algorithm as galg          # GeneFi
    import mytransforms as gtf        # RandomSpi, RandomComPre
    import torchvision.transforms as transforms
    return galg, gtf, transforms


def gesfi_args(num_classes: int = 6) -> SimpleNamespace:
    """Default arguments of GesFi's csimain.get_args() (values copied from the released code)."""
    return SimpleNamespace(alpha=0.1, alpha1=0.1, batch_size=32, beta1=0.5, bottleneck=256, classifier="linear",
                           dis_hidden=256, layer="bn", lam=0.0, latent_domain_num=3, local_epoch=1, lr=0.0002,
                           max_epoch=50, weight_decay=5e-4, num_classes=num_classes)


# --------------------------------------------------------------------------- input rendering
def jet_lut() -> np.ndarray:
    import matplotlib
    return (matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def render_images(features: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """(N, 2, 64, 128) maps -> (N, 224, 224, 3) uint8 images. Clipping limits per channel from the train partition only."""
    from PIL import Image
    tr = np.asarray(features[np.sort(train_idx)], dtype=np.float32)
    lo, hi = np.percentile(tr, T.CLIP_PCT, axis=(0, 2, 3))
    lut = jet_lut()
    out = np.empty((len(features), 224, 224, 3), dtype=np.uint8)
    for s in range(0, len(features), 512):
        x = np.asarray(features[s:s + 512], dtype=np.float32)
        x = (np.clip(x, lo[None, :, None, None], hi[None, :, None, None]) - lo[None, :, None, None]) / (hi - lo)[None, :, None, None]
        q = np.round(x * 255).astype(np.uint8)
        stacked = lut[np.concatenate([q[:, 0, ::-1], q[:, 1, ::-1]], axis=1)]       # phase above Doppler, low freq at bottom
        for i, img in enumerate(stacked):
            out[s + i] = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
    return out


class ImageSet(torch.utils.data.Dataset):
    """Same item interface as GesFi's datatrcsie/datatecsie: (image, label, pseudo-domain label, index)."""
    def __init__(self, images: np.ndarray, idx: np.ndarray, labels: np.ndarray, transform):
        self.images, self.idx, self.labels, self.transform = images, idx, labels, transform
        self.pdlabels = np.zeros(len(idx), dtype=np.int64)

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type="domain_label"):
        if label_type == "pdlabel":
            self.pdlabels[np.asarray(tindex).astype(int)] = np.asarray(tlabels).astype(int)

    def __getitem__(self, item):
        from PIL import Image
        x = self.transform(Image.fromarray(self.images[self.idx[item]]))
        return x, int(self.labels[self.idx[item]]), int(self.pdlabels[item]), item

    def __len__(self):
        return len(self.idx)


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_all(net, loader, device):
    net.eval()
    preds, conf = [], []
    for x, _, _, _ in loader:
        p = torch.softmax(net.predict(x.to(device).float()), dim=1)
        c, k = p.max(1)
        preds.append(k.cpu().numpy()); conf.append(c.cpu().numpy())
    net.train()
    return np.concatenate(preds), np.concatenate(conf)


# --------------------------------------------------------------------------- run
def run(task: str, seed: int, out_dir: Path, gesfi_dir: Path, features: str, manifest_csv: str, splits_csv: str,
        run_kind: str, epochs: int, data_fraction: float, workers: int) -> dict:
    assert run_kind in ("final", "quick", "pilot")
    if run_kind == "final":
        assert epochs == 50 and data_fraction == 1.0, "final runs use GesFi's full schedule on all data"
    job_id = f"GesFi_{task}_s{seed}"
    run_dir = out_dir / "runs" / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.json").exists():
        saved = json.load(open(run_dir / "metrics.json"))
        assert saved["run_kind"] == run_kind and saved["epochs"] == epochs and saved["data_fraction"] == data_fraction
        print(f"[{job_id}] already complete, skipping.")
        return saved

    galg, gtf, transforms = import_gesfi(gesfi_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":                     # GesFi hard-codes .cuda(); identity on CPU for local smoke tests only
        assert run_kind != "final", "final GesFi runs need a GPU"
        torch.Tensor.cuda = lambda self, *a, **k: self
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    fmeta = T.load_features_meta(features)
    manifest = T.load_manifest(manifest_csv)
    splits = T.subsample_parts(manifest, T.load_task_splits(manifest, splits_csv, task), data_fraction)
    feats = np.load(features, mmap_mode="r")
    labels = manifest["gesture"].to_numpy() - 1
    t0 = time.time()
    images = render_images(feats, splits["train"])
    render_s = time.time() - t0

    # GesFi's transforms (csimain.py)
    tf_train = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.RandomCrop(224, padding=(32, 0), padding_mode="reflect")], p=0.2),
        gtf.RandomSpi(p=0.2), gtf.RandomComPre(p=0.2),
        transforms.Resize([224, 224]), transforms.ToTensor()])
    tf_test = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    g = torch.Generator(); g.manual_seed(seed)
    train_set = ImageSet(images, splits["train"], labels, tf_train)
    drop = len(train_set) % 32 == 1                # BatchNorm1d cannot take a final batch of one sample
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=workers,
                                               drop_last=drop, generator=g, persistent_workers=workers > 0)
    eval_loaders = {p: torch.utils.data.DataLoader(ImageSet(images, splits[p], labels, tf_test), batch_size=128,
                                                   shuffle=False, num_workers=workers) for p in ("val", "target", "id_test")}

    args = gesfi_args()
    args.max_epoch = epochs
    net = galg.GeneFi(args).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10, 20], gamma=0.1)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_params_inference = sum(p.numel() for m in (net.featurizer, net.bottleneck, net.classifier) for p in m.parameters())

    hist, start, ckpt = [], 0, run_dir / "checkpoint.pt"
    best = {r: {"score": -1.0, "epoch": -1} for r in SELECTION_RULES}
    if ckpt.exists():                              # resume after a disconnect
        s = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        train_set.pdlabels = s["pdlabels"]; hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        random.setstate(s["py_rng"]); np.random.set_state(s["np_rng"]); torch.set_rng_state(s["torch_rng"])
        print(f"[{job_id}] resumed at epoch {start}")

    t_train = time.time()
    for rnd in range(start, epochs):
        te = time.time()
        net.train()
        if rnd <= 1:                               # pre-learning, as in GesFi's trainer
            for _ in range(2):
                for x, y, pd_, _ in train_loader:
                    net.update_a(x, y, pd_, opt)
        for x, y, pd_, _ in train_loader:          # latent-domain characterisation
            ld = net.update_d(x, y, pd_, opt)
        cpd = net.set_dlabel(train_loader)         # latent-domain clustering
        for x, y, pd_, _ in train_loader:          # domain-invariant feature learning
            lu = net.update(x, y, pd_, opt, cpd)
        sched.step()

        row = {"epoch": rnd + 1, "train_s": time.time() - te, "loss_class": lu["class"], "loss_dis": lu["dis"],
               "loss_d_total": ld["total"], "latent_domain_sizes": {int(k): int(v) for k, v in cpd.items()}}
        preds = {}
        for p, loader in eval_loaders.items():
            pr, cf = predict_all(net, loader, device)
            m = T.compute_metrics(labels[splits[p]], pr)
            row[f"{p}_acc"], row[f"{p}_f1"] = m["accuracy"], m["macro_f1"]
            preds[p] = (pr, cf, m)
        hist.append(row)
        scores = {"sourceval": row["val_f1"], "last": float(rnd), "oracle": row["target_acc"]}
        for r in SELECTION_RULES:
            if scores[r] > best[r]["score"]:
                best[r] = {"score": scores[r], "epoch": rnd + 1}
                best[r]["metrics"] = {p: preds[p][2] for p in preds}
                np.savez_compressed(run_dir / f"preds_{r}.npz", **{
                    f"{p}_{k}": v for p in preds for k, v in (
                        ("trial_id", manifest["trial_id"].to_numpy()[splits[p]]), ("y_true", labels[splits[p]] + 1),
                        ("y_pred", preds[p][0] + 1), ("conf", preds[p][1]))})
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "pdlabels": train_set.pdlabels,
                    "hist": hist, "best": best, "epoch": rnd, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, ckpt)
        print(f"[{job_id}] ep {rnd + 1:02d}/{epochs} val-F1 {row['val_f1']:.4f} target-acc {row['target_acc']:.4f} "
              f"id-acc {row['id_test_acc']:.4f} ({row['train_s']:.0f}s)", flush=True)

    res = {
        "job_id": job_id, "config": "GesFi", "task": task, "seed": seed, "run_kind": run_kind, "epochs": epochs,
        "data_fraction": data_fraction, "gesfi_commit": GESFI_COMMIT, "gesfi_args": vars(args),
        "preprocessing": fmeta["preprocessing"], "features_sha256": fmeta["sha256"],
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__,
        "n_params_training": n_params, "n_params_inference": n_params_inference, "render_s": render_s,
        "train_time_s": time.time() - t_train, "n_train": int(len(splits["train"])), "n_val": int(len(splits["val"])),
        "n_target": int(len(splits["target"])), "n_id_test": int(len(splits["id_test"])),
        "selection": {r: {"epoch": best[r]["epoch"], **{p: best[r]["metrics"][p] for p in ("val", "target", "id_test")}}
                      for r in SELECTION_RULES},
        "history": hist,
    }
    T._atomic_json(res, run_dir / "metrics.json")
    ckpt.unlink()
    s = res["selection"]
    print(f"[{job_id}] DONE target acc: sourceval {s['sourceval']['target']['accuracy']:.4f} | "
          f"last {s['last']['target']['accuracy']:.4f} | oracle {s['oracle']['target']['accuracy']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="GesFi on the single-link benchmark")
    ap.add_argument("--gesfi_dir", default="third_party/GesFiCode")
    ap.add_argument("--fetch", action="store_true", help="clone GesFi at the pinned commit into --gesfi_dir")
    ap.add_argument("--features", default="cache/features.npy")
    ap.add_argument("--manifest_csv", default="cache/manifest.csv")
    ap.add_argument("--splits_csv", default="cache/splits.csv")
    ap.add_argument("--out_dir", default="results/gesfi")
    ap.add_argument("--run_kind", default="final", choices=["final", "quick", "pilot"])
    ap.add_argument("--tasks", nargs="*", default=list(T.TASKS))
    ap.add_argument("--seeds", nargs="*", type=int, default=T.SEEDS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--data_fraction", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    gdir = Path(a.gesfi_dir)
    if a.fetch:
        fetch_gesfi(gdir)
    assert (gdir / "algorithm.py").exists(), f"GesFi code not found in {gdir}; run with --fetch"
    for t in a.tasks:
        for s in a.seeds:
            run(t, s, Path(a.out_dir), gdir, a.features, a.manifest_csv, a.splits_csv, a.run_kind, a.epochs,
                a.data_fraction, a.workers)


if __name__ == "__main__":
    main()
