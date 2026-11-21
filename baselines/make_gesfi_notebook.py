"""Writes baselines/gesfi_colab.ipynb (Colab orchestration for the GesFi base-paper runs)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# GesFi (base paper) on the single-link benchmark

Runs the released GesFi code (pinned commit) on our tasks O1/O3/O5 with seeds 42/43/44, reporting three model-selection
rules. Runtime: GPU (T4). Upload two files to `/content/` with the Files pane before running:
`single-link-wifi-dg.zip` (made with `git archive --format=zip -o ../single-link-wifi-dg.zip HEAD`) and `features.npy`.
Results are written to Google Drive, and every run saves a checkpoint after each epoch, so after a disconnect simply
run all cells again: finished runs are skipped and an interrupted run resumes from its last epoch.
""")
code("""
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
REPO = Path("/content/single-link-wifi-dg")
RESULTS = Path("/content/drive/MyDrive/single_link_dg_results/gesfi")
with zipfile.ZipFile("/content/single-link-wifi-dg.zip") as z:
    COMMIT = z.comment.decode().strip()
    assert len(COMMIT) == 40, "upload the zip made with git archive"
    if REPO.exists():
        shutil.rmtree(REPO)
    z.extractall(REPO)
os.chdir(REPO); sys.path.insert(0, str(REPO))
feat = REPO / "cache" / "features.npy"
if not feat.exists():
    shutil.copy("/content/features.npy", feat)
meta = json.load(open(REPO / "cache" / "features_meta.json"))
h = hashlib.sha256()
with open(feat, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest() == meta["sha256"], "features.npy does not match features_meta.json"
import torch
assert torch.cuda.is_available(), "select a T4 GPU runtime"
print("code", COMMIT[:7], "| GPU", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| results", RESULTS)
""")
md("Fetch GesFi at the pinned commit (GPL-3.0; kept outside the repository) and run a one-epoch timing pilot.")
code("""
subprocess.run([sys.executable, "baselines/gesfi_runner.py", "--fetch", "--run_kind", "pilot", "--tasks", "O1",
                "--seeds", "42", "--epochs", "1", "--out_dir", str(RESULTS / "pilot")], check=True)
m = json.load(open(RESULTS / "pilot/runs/GesFi_O1_s42/metrics.json"))
per_epoch = m["history"][0]["train_s"]
print(f"first epoch {per_epoch:.0f} s (includes two pre-learning passes); estimated 9 runs x 50 epochs: "
      f"{9 * 50 * per_epoch * 0.75 / 3600:.1f} h")
""")
md("Full queue: 3 tasks x 3 seeds, 50 epochs each. Re-run this cell after a disconnect.")
code("""
subprocess.run([sys.executable, "baselines/gesfi_runner.py", "--run_kind", "final", "--out_dir", str(RESULTS / "final")],
               check=True)
""")
code("""
import pandas as pd
rows = []
for f in sorted((RESULTS / "final/runs").glob("*/metrics.json")):
    m = json.load(open(f))
    for r, v in m["selection"].items():
        rows.append({"task": m["task"], "seed": m["seed"], "rule": r, "epoch": v["epoch"],
                     "target_acc": v["target"]["accuracy"], "target_f1": v["target"]["macro_f1"],
                     "id_acc": v["id_test"]["accuracy"]})
df = pd.DataFrame(rows)
display(df.groupby("rule")[["target_acc", "target_f1", "id_acc"]].agg(["mean", "std"]).round(4))
""")
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["accelerator"] = "GPU"
out = Path(__file__).with_name("gesfi_colab.ipynb")
nbf.write(nb, out)
print("wrote", out)
