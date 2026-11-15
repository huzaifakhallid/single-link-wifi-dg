"""
Data preparation for the single-link Widar3.0 Room-1 (20181130) experiments.

Stages (run in this order; each can be run alone):
  --extract     pull userX-g-l-o-rep-r1.dat (gestures 1-6) out of the zips into DATA_ROOT/user<X>/.
                zipfile verifies the CRC-32 of every member it reads; a mismatch aborts.
  --provenance  SHA-256 + member counts of every zip -> CACHE_DIR/data_provenance.json
  --audit       assert 9 users x 750 files, report empty files
  --preprocess  locked pipeline (preprocess.LOCKED) for every trial -> CACHE_DIR/features.npy (N,2,64,128 fp16),
                manifest.csv (one row per trial, exclusions kept with their reason), splits.csv (O1/O3/O5,
                leakage-checked), features_meta.json (pipeline settings, SHA-256 of features.npy, timing)

  python prepare_data.py --zips "dataset/widar_r1/*.zip" --all
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import preprocess as pp

FNAME_RE = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d)\.dat$")
KEEP_GESTURES = range(1, 7)
ROOM1_USERS = [5, 10, 11, 12, 13, 14, 15, 16, 17]       # session 20181130 = Room 1 (classroom)
EXPECTED_PER_USER = 6 * 5 * 5 * 5                         # gestures x locations x orientations x repetitions


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- extract / provenance
def extract_zip(zip_path: Path, data_root: Path) -> dict:
    assert zip_path.exists(), f"Zip not found: {zip_path}"
    if Path(str(zip_path) + ".aria2").exists():
        raise RuntimeError(f"{zip_path.name} is still downloading (.aria2 present); refusing to open.")
    marker = data_root / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"[extract] {zip_path.name}: already extracted ({marker.read_text()}), skipping.")
        return {}
    counts = Counter()
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            m = FNAME_RE.match(base)
            if not m:
                continue
            user, gesture, rx = int(m.group(1)), int(m.group(2)), int(m.group(6))
            if rx != 1 or gesture not in KEEP_GESTURES:
                continue
            out = data_root / f"user{user}" / base      # path built from parsed integers only
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src:                   # raises BadZipFile on a CRC-32 mismatch
                payload = src.read()
            assert len(payload) == info.file_size, f"{base}: size mismatch"
            out.write_bytes(payload)
            counts[user] += 1
    marker.write_text(",".join(f"user{u}:{n}" for u, n in sorted(counts.items())))
    print(f"[extract] {zip_path.name}: {dict(counts)} r1 g1-6 files in {time.time() - t0:.0f}s (CRC verified)")
    return dict(counts)


def provenance(zips, data_root: Path, cache_dir: Path) -> None:
    rec = {"dataset": "Widar3.0 (Tsinghua mirror), session 20181130 = Room 1", "zips": []}
    for zp in map(Path, zips):
        t0 = time.time()
        with zipfile.ZipFile(zp) as z:
            names = [i.filename.rsplit("/", 1)[-1] for i in z.infolist()]
        ms = [FNAME_RE.match(n) for n in names]
        ms = [m for m in ms if m]
        kept = [m for m in ms if int(m.group(6)) == 1 and int(m.group(2)) in KEEP_GESTURES]
        marker = data_root / f".extracted_{zp.stem}"
        rec["zips"].append({
            "file": zp.name, "bytes": zp.stat().st_size, "sha256": sha256_file(zp),
            "n_dat_members": len(ms), "users": sorted({int(m.group(1)) for m in ms}),
            "gestures": sorted({int(m.group(2)) for m in ms}), "receivers": sorted({int(m.group(6)) for m in ms}),
            "n_r1_gesture1to6_members": len(kept),
            "extracted": marker.read_text() if marker.exists() else None,
            "integrity": "per-member CRC-32 verified by zipfile during extraction",
        })
        print(f"[provenance] {zp.name}: sha256 {rec['zips'][-1]['sha256'][:16]}..., {len(ms)} .dat members, "
              f"{len(kept)} kept ({time.time() - t0:.0f}s)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(cache_dir / "data_provenance.json", "w"), indent=2)


# --------------------------------------------------------------------------- audit
def scan_files(data_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(data_root.glob("user*/*-r1.dat")):
        m = FNAME_RE.match(p.name)
        assert m, f"Unexpected filename {p}"
        u, g, l, o, r, rx = map(int, m.groups())
        assert p.parent.name == f"user{u}", f"File {p} in wrong user dir"
        rows.append(dict(trial_id=p.stem, path=str(p), user=u, gesture=g, location=l,
                         orientation=o, rep=r, file_bytes=p.stat().st_size))
    df = pd.DataFrame(rows)
    assert len(df) > 0, f"No *-r1.dat files under {data_root}"
    assert df["gesture"].between(1, 6).all() and df["orientation"].between(1, 5).all()
    assert df["location"].between(1, 5).all() and df["rep"].between(1, 5).all()
    assert not df["trial_id"].duplicated().any()
    return df


def audit(df: pd.DataFrame) -> None:
    per_user = df.groupby("user").size()
    print("[audit] files per user:", per_user.to_dict())
    assert sorted(per_user.index.tolist()) == ROOM1_USERS, f"users found: {per_user.index.tolist()}"
    assert (per_user == EXPECTED_PER_USER).all(), f"expected {EXPECTED_PER_USER} r1 trials per user"
    cells = df.groupby(["user", "gesture", "location", "orientation"]).size()
    assert len(cells) == 9 * 6 * 5 * 5 and (cells == 5).all(), "incomplete (user, gesture, location, orientation) grid"
    empty = df[df["file_bytes"] == 0]
    print(f"[audit] total={len(df)} (9 users x {EXPECTED_PER_USER}), zero-byte files={len(empty)}")
    if len(empty):
        print(empty[["trial_id", "file_bytes"]].to_string(index=False))


# --------------------------------------------------------------------------- preprocess
def _process_one(path):
    import torch
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    try:
        csi, ts, info = pp.load_trial(path)
    except Exception as e:      # unparseable file -> recorded exclusion, never silently dropped
        return None, {"exclusion_reason": f"parse_error: {type(e).__name__}: {e}"}, None
    x, meta = pp.preprocess_trial(csi, ts)
    meta.update(info)
    meta.pop("feature_ms", None)
    return x, meta, (time.perf_counter() - t0) * 1000


def preprocess_all(files: pd.DataFrame, cache_dir: Path, workers: int) -> pd.DataFrame:
    from train import TASKS, generate_splits_for_task
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = files.sort_values(["user", "trial_id"]).reset_index(drop=True)
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_process_one, files["path"].tolist(), chunksize=8)
    X = np.zeros((len(files), 2, *pp.MAP_SHAPE), dtype=np.float16)
    metas = []
    for i, (x, m, _) in enumerate(res):
        m["valid"] = x is not None
        m["exclusion_reason"] = m.get("exclusion_reason") or ""
        assert m["valid"] == (m["exclusion_reason"] == ""), files.loc[i, "trial_id"]
        if x is not None:
            X[i] = x
        metas.append(m)
    cols = ["valid", "exclusion_reason", "n_packets_raw", "n_dropped_nrx", "n_packets", "n_wraps", "n_dups",
            "duration_s", "pkt_rate_hz", "max_gap_ms"]
    manifest = pd.concat([files.drop(columns=["path"]), pd.DataFrame(metas).reindex(columns=cols)], axis=1)
    np.save(cache_dir / "features.npy", X)
    manifest.to_csv(cache_dir / "manifest.csv", index=False, float_format="%.6g")

    split_cols = {}
    for t in TASKS:
        sp = generate_splits_for_task(manifest, t, val_fraction=0.20, seed=0)
        col = np.full(len(manifest), "", dtype=object)
        for part, idx in sp.items():
            col[idx] = part
        split_cols[f"split_{t}"] = col
        print(f"[splits] {t}: " + ", ".join(f"{p}={len(ix)}" for p, ix in sp.items()) + " (leakage asserts passed)")
    pd.DataFrame({"trial_id": manifest["trial_id"], **split_cols}).to_csv(cache_dir / "splits.csv", index=False)

    ms = np.array([r[2] for r in res if r[2] is not None])
    meta = {"preprocessing": dict(pp.LOCKED), "n_trials": int(len(manifest)), "n_valid": int(manifest["valid"].sum()),
            "shape": list(X.shape), "dtype": "float16", "sha256": sha256_file(cache_dir / "features.npy"),
            "sample_rate_hz": pp.FS, "stft": {"window": pp.STFT_WIN, "hop": pp.STFT_HOP, "n_fft": pp.STFT_NFFT,
                                              "max_hz": pp.DOPPLER_MAX_HZ},
            "band_hz": list(pp.BAND_HZ), "max_gap_s": pp.MAX_GAP_S, "min_duration_s": pp.MIN_DUR_S,
            "decode_plus_feature_ms_median": float(np.median(ms)), "decode_plus_feature_ms_p95": float(np.percentile(ms, 95)),
            "wall_seconds": time.time() - t0, "workers": workers}
    json.dump(meta, open(cache_dir / "features_meta.json", "w"), indent=2)
    print(f"[preprocess] {len(manifest)} trials, valid={meta['n_valid']}, features {X.shape} "
          f"{X.nbytes / 2**20:.0f} MB, sha256 {meta['sha256'][:16]}..., {meta['wall_seconds']:.0f}s")
    exc = manifest[~manifest["valid"]]
    print(f"[exclusions] {len(exc)} / {len(manifest)} trials excluded")
    if len(exc):
        print(exc[["trial_id", "exclusion_reason"]].to_string(index=False))
    return manifest


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="*", default=["dataset/widar_r1/*.zip"])
    ap.add_argument("--data_root", default="dataset/widar_r1")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for flag in ("extract", "provenance", "audit", "preprocess", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    data_root, cache_dir = Path(a.data_root), Path(a.cache_dir)
    zips = sorted(z for pat in a.zips for z in glob.glob(pat))
    if a.extract or a.all:
        assert zips, f"no zip matches {a.zips}"
        for z in zips:
            extract_zip(Path(z), data_root)
    if a.provenance or a.all:
        assert zips, f"no zip matches {a.zips}"
        provenance(zips, data_root, cache_dir)
    files = scan_files(data_root)
    if a.audit or a.all:
        audit(files)
    if a.preprocess or a.all:
        audit(files)
        preprocess_all(files, cache_dir, a.workers)


if __name__ == "__main__":
    sys.exit(main())
