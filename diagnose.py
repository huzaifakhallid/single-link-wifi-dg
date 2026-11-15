"""
Preprocessing diagnostic gate (run before any DG experiment).

Question: do the single-link features carry gesture information at all, and which candidate
implementation of each preprocessing stage is best?

Protocol (target-free): only trials recorded at orientations 2 and 4 are used. These two
orientations are never a target domain in tasks O1/O3/O5, so choosing the pipeline on them
uses no target-domain information. Trials are split 80/20 at random, stratified by
(user, gesture, orientation). A plain ERM classifier (the project's branch encoder + linear
head, 15 epochs) is fitted per variant; train and held-out accuracy are logged per epoch.

  python diagnose.py features          # variant maps for the o2/o4 trials (parallel)
  python diagnose.py fit --set single  # one ERM per (hampel mode, single-channel variant)
  python diagnose.py fit --set fused   # fused ERMs: Phase-2 pipeline vs chosen pipeline
  python diagnose.py table             # results_diag/gate_table.csv
  python diagnose.py maps              # before/after maps per gesture
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import preprocess as pp

DIAG_ORIENTATIONS = (2, 4)
OUT = Path("results_diag")
CHANNELS = [f"phase:{v}" for v in pp.PHASE_VARIANTS] + [f"doppler:{v}" for v in pp.DOPPLER_VARIANTS]
PHASE2 = ("antenna_complex", "phase:unwrap", "doppler:mean_avg_log1p")   # pipeline as specified in Phase 2
EPOCHS = 15


# --------------------------------------------------------------------------- features
def _one(path):
    import torch
    torch.set_num_threads(1)
    try:
        csi, ts, _ = pp.load_trial(path)
    except Exception as e:
        return None, f"parse_error: {type(e).__name__}: {e}"
    out = np.zeros((len(pp.HAMPEL_MODES), len(CHANNELS), *pp.MAP_SHAPE), dtype=np.float16)
    for hi, h in enumerate(pp.HAMPEL_MODES):
        R, meta = pp.ratio_200hz(csi, ts, h)
        if R is None:
            return None, meta["exclusion_reason"]
        for ci, ch in enumerate(CHANNELS):
            kind, v = ch.split(":")
            m = pp.resize_map(pp.phase_map(R, v) if kind == "phase" else pp.doppler_map(R, v))
            m16 = pp.to_fp16(m)
            if m16 is None:
                return None, f"non_finite_or_fp16_overflow in {ch}"
            out[hi, ci] = m16
    return out, ""


def build_features(data_root: Path, workers: int) -> None:
    from prepare_data import scan_files
    files = scan_files(data_root)
    files = files[files["orientation"].isin(DIAG_ORIENTATIONS)].sort_values("trial_id").reset_index(drop=True)
    assert files["user"].nunique() == 9, f"expected 9 users, got {sorted(files['user'].unique())}"
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_one, files["path"].tolist(), chunksize=8)
    files["valid"] = [r[0] is not None for r in res]
    files["exclusion_reason"] = [r[1] for r in res]
    X = np.stack([r[0] for r in res if r[0] is not None])        # (N, hampel, channel, 64, 128)
    meta = files[files["valid"]].drop(columns=["path"]).reset_index(drop=True)
    # stratified 80/20 split by (user, gesture, orientation), fixed seed
    rng = np.random.RandomState(0)
    part = np.full(len(meta), "train", dtype=object)
    for _, g in meta.groupby(["user", "gesture", "orientation"]):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        part[idx[:int(round(0.2 * len(idx)))]] = "heldout"
    meta["part"] = part
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for hi, h in enumerate(pp.HAMPEL_MODES):
        np.save(OUT / "cache" / f"feat_{h}.npy", X[:, hi])
    meta.to_csv(OUT / "diag_trials.csv", index=False)
    files[~files["valid"]].drop(columns=["path"]).to_csv(OUT / "diag_exclusions.csv", index=False)
    print(f"[features] {len(files)} o2/o4 trials, {len(meta)} valid, {int((~files['valid']).sum())} excluded, "
          f"train={int((part == 'train').sum())} heldout={int((part == 'heldout').sum())}, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- ERM
def fit_one(job):
    """job = (name, hampel, [channel names], head_scale, threads, epochs, lr)."""
    name, hampel, chans, head_scale, threads, epochs, lr = job
    out_file = OUT / "fits" / f"{name}.json"
    if out_file.exists():
        return json.load(open(out_file))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from train import BranchEncoder
    torch.set_num_threads(threads)
    torch.manual_seed(0); np.random.seed(0)

    meta = pd.read_csv(OUT / "diag_trials.csv")
    X = np.load(OUT / "cache" / f"feat_{hampel}.npy", mmap_mode="r")
    x = torch.from_numpy(np.stack([np.asarray(X[:, CHANNELS.index(c)], dtype=np.float32) for c in chans], axis=1))
    y = torch.from_numpy(meta["gesture"].to_numpy() - 1)
    tr = torch.from_numpy(np.where(meta["part"] == "train")[0])
    ho = torch.from_numpy(np.where(meta["part"] == "heldout")[0])
    # train-only clipping (0.5 / 99.5 percentile per channel) and z-scoring
    for c in range(x.shape[1]):
        sample = x[tr, c].flatten()[::17].numpy()
        lo, hi = np.percentile(sample, [0.5, 99.5])
        x[:, c].clamp_(float(lo), float(hi))
        mu, sd = x[tr, c].mean(), x[tr, c].std() + 1e-6
        x[:, c] = (x[:, c] - mu) / sd

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.enc = nn.ModuleList([BranchEncoder() for _ in range(n)])
            self.gate = nn.Parameter(torch.zeros(n))
            self.head = nn.Linear(64, 6)

        def forward(self, xb):
            w = F.softmax(self.gate, dim=0)
            z = sum(w[i] * e(xb[:, i:i + 1]) for i, e in enumerate(self.enc))
            return self.head(F.normalize(z, dim=-1) * head_scale)

    net = Net(len(chans))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    @torch.no_grad()
    def acc(idx):
        net.eval()
        pred = torch.cat([net(x[idx[i:i + 256]]).argmax(1) for i in range(0, len(idx), 256)])
        return float((pred == y[idx]).float().mean())

    curve = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        losses = []
        for i in range(0, len(perm) - 63, 64):
            b = perm[i:i + 64]
            loss = F.cross_entropy(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epochs <= EPOCHS or ep % 3 == 0 or ep == epochs:      # long fits are scored every third epoch
            curve.append({"epoch": ep, "loss": float(np.mean(losses)), "train_acc": acc(tr), "heldout_acc": acc(ho)})
    res = {"name": name, "hampel_mode": hampel, "channels": chans, "head_scale": head_scale,
           "n_train": int(len(tr)), "n_heldout": int(len(ho)), "epochs": epochs, "lr": lr,
           "optimizer_steps": epochs * (len(tr) // 64),
           "train_acc": curve[-1]["train_acc"], "heldout_acc": curve[-1]["heldout_acc"],
           "best_heldout_acc": max(c["heldout_acc"] for c in curve), "fit_seconds": time.time() - t0,
           "curve": curve}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_file, "w"), indent=1)
    print(f"[fit] {name}: train {res['train_acc']:.3f} heldout {res['heldout_acc']:.3f} ({res['fit_seconds']:.0f}s)",
          flush=True)
    return res


def short(ch):
    return ch.split(":")[1]


def jobs_for(which: str, chosen=None, threads=2, epochs=EPOCHS, lr=1e-3):
    jobs = []
    if which == "single":
        for h in pp.HAMPEL_MODES:
            for ch in CHANNELS:
                jobs.append((f"single__{h}__{ch.replace(':', '-')}", h, [ch], 10.0, threads, EPOCHS, 1e-3))
    elif which == "fused":
        h0, p0, d0 = PHASE2
        tag = f"__ep{epochs}" + ("" if lr == 1e-3 else f"__lr{lr:g}")
        if not chosen:
            jobs.append((f"fused__{h0}__{short(p0)}__{short(d0)}{tag}", h0, [p0, d0], 10.0, threads, epochs, lr))
        for h, p, d in (chosen or []):
            jobs.append((f"fused__{h}__{p}__{d}{tag}", h, [f"phase:{p}", f"doppler:{d}"], 10.0, threads, epochs, lr))
    elif which == "head":
        h, p, d = chosen[0]
        jobs.append((f"fused__{h}__{p}__{d}__head1__ep{epochs}", h, [f"phase:{p}", f"doppler:{d}"], 1.0, threads, epochs, 1e-3))
    return jobs


def table() -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / "fits").glob("*.json")):
        r = json.load(open(f))
        rows.append({"fit": r["name"], "hampel_mode": r["hampel_mode"],
                     "channels": " + ".join(r["channels"]), "head_scale": r["head_scale"],
                     "epochs": r["epochs"], "lr": r.get("lr", 1e-3), "optimizer_steps": r.get("optimizer_steps", r["epochs"] * (r["n_train"] // 64)),
                     "train_acc": round(r["train_acc"], 4), "heldout_acc": round(r["heldout_acc"], 4),
                     "n_train": r["n_train"], "n_heldout": r["n_heldout"],
                     "maps_file": "results_diag/maps_before_after.png"})
    df = pd.DataFrame(rows).sort_values("heldout_acc", ascending=False)
    df.to_csv(OUT / "gate_table.csv", index=False)
    print(df.drop(columns=["maps_file", "fit"]).to_string(index=False))
    return df


def maps(user: int = 5) -> None:
    """One training-partition trial per gesture: every phase and Doppler variant under the locked Hampel mode,
    plus the Phase-2 pipeline, drawn from the cached diagnostic features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train import GESTURE_NAMES
    meta = pd.read_csv(OUT / "diag_trials.csv")
    cols = [(PHASE2[0], PHASE2[1]), (pp.LOCKED["hampel_mode"], f"phase:{pp.LOCKED['phase_variant']}"),
            (PHASE2[0], PHASE2[2]), (pp.LOCKED["hampel_mode"], f"doppler:{pp.LOCKED['doppler_variant']}")]
    titles = ["phase, Phase-2 spec", "phase, locked", "Doppler, Phase-2 spec", "Doppler, locked"]
    X = {h: np.load(OUT / "cache" / f"feat_{h}.npy", mmap_mode="r") for h in pp.HAMPEL_MODES}
    fig, axes = plt.subplots(6, 4, figsize=(15, 14), constrained_layout=True)
    for g in range(1, 7):
        sel = meta[(meta.user == user) & (meta.gesture == g) & (meta.part == "train")]
        assert len(sel), f"no diagnostic train trial for user {user} gesture {g}"
        i = sel.index[0]
        for k, ((h, ch), ttl) in enumerate(zip(cols, titles)):
            ax = axes[g - 1, k]
            im = ax.imshow(np.asarray(X[h][i, CHANNELS.index(ch)], dtype=np.float32), aspect="auto", origin="lower",
                           cmap="viridis" if ch.startswith("phase") else "magma")
            ax.set_title(f"G{g} {GESTURE_NAMES[g]}: {ttl}\n{short(ch)} [{h}] {meta.trial_id[i]}", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax)
    fig.savefig(OUT / "maps_before_after.png", dpi=200)
    plt.close(fig)
    print(f"[maps] {OUT / 'maps_before_after.png'}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "fit", "table", "maps"])
    ap.add_argument("--data_root", default="dataset/widar_r1")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--set", default="single", choices=["single", "fused", "head"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chosen", nargs="*", default=[], help="hampel,phase,doppler triples for fused fits")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stage == "features":
        build_features(Path(a.data_root), a.workers)
    elif a.stage == "fit":
        jobs = jobs_for(a.set, [tuple(c.split(",")) for c in a.chosen], a.threads, a.epochs, a.lr)
        with Pool(a.procs) as pool:
            pool.map(fit_one, jobs, chunksize=1)
        table()
    elif a.stage == "table":
        table()
    elif a.stage == "maps":
        maps()


if __name__ == "__main__":
    sys.exit(main())
