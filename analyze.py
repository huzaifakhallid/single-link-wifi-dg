"""
Tables (CSV + LaTeX) and figures (PNG, 300 dpi) for the Phase-3 evaluation.

Every number is computed from saved run files: <results>/runs/<job>/{metrics.json, preds.npz},
cache/{manifest.csv, splits.csv, features_meta.json, features.npy} and results_diag/gate_table.csv.
The only constants are (a) the experimental design and (b) LITERATURE, which quotes numbers reported
by other papers under different protocols (context only, never ranked against ours).

  python analyze.py --results results/final --run_kind final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train import CLAP, CONFIGS, GESTURE_NAMES, NUM_CLASSES, SEEDS, SLIDE, TASKS

C_FAMILY = ["C0", "C1", "C2", "C3", "C4", "C5"]
B_FAMILY = ["B1", "B2", "B3", "B4"]
ORDER = C_FAMILY + B_FAMILY
LABELS = {"C0": "C0 phase only", "C1": "C1 fused (matched baseline)", "C2": "C2 fused + curriculum",
          "C3": "C3 fused + consistency", "C4": "C4 consistency + fixed aug", "C5": "C5 consistency + curriculum (proposed)",
          "B1": "B1 ERM", "B2": "B2 Deep CORAL", "B3": "B3 DANN", "B4": "B4 Mixup"}
CONTRASTS = [("C5", "C1", "full method vs matched fused baseline"),
             ("C5", "C2", "class consistency (curriculum held fixed)"),
             ("C5", "C3", "curriculum (consistency held fixed)"),
             ("C5", "C4", "curriculum schedule vs fixed-level control"),
             ("C1", "C0", "Doppler branch (capacity also changes)")]
METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("clap_slide_error", "Clap/slide error")]
GEST = [GESTURE_NAMES[i] for i in range(1, 7)]

# Numbers REPORTED BY OTHER PAPERS (quoted as given in the project brief / Phase-2 review). Different receivers,
# inputs, splits and target-data access: context only.
LITERATURE = [
    dict(system="Widar3.0 (MobiSys'19)", receivers="6 receivers", input="BVP (multi-link)", target_data="none",
         in_domain="92.7", cross_orientation="82.6-92.4 (cross-domain range)",
         fairness="Six synchronised receivers and body-velocity profiles; not single-link."),
    dict(system="GLSDA baseline (ICPADS'25, Table II)", receivers="1 receiver location", input="CSI-ratio branch",
         target_data="none", in_domain="95.56", cross_orientation="89.92",
         fairness="Trains on one location/orientation and tests on the others; different split and model size."),
    dict(system="GLSDA (ICPADS'25, Table II)", receivers="1 receiver location", input="CSI-ratio branch",
         target_data="none", in_domain="97.78", cross_orientation="92.80",
         fairness="As above, plus a large-model semantic teacher."),
    dict(system="Wi-CBR (AAAI'26)", receivers="multi-receiver Widar features", input="phase + Doppler images",
         target_data="none", in_domain="-", cross_orientation="96.30",
         fairness="Published Widar pipeline uses multiple receiver inputs; ResNet-18 backbone."),
    dict(system="Wi-SFDAGR (IEEE IoT-J'25)", receivers="multi-device input", input="CSI-ratio phase heatmaps",
         target_data="unlabelled target data", in_domain="-", cross_orientation="97.17",
         fairness="Source-free adaptation on unlabelled target samples; not strict domain generalization."),
]

# dataviz reference palette (light surface)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ROLE_COLOR = {"proposed": CAT[0], "matched baseline": CAT[1], "external DG baseline": CAT[2], "ablation arm": MUTED}
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def role(cfg):
    return "proposed" if cfg == "C5" else "matched baseline" if cfg == "C1" else \
        "external DG baseline" if cfg.startswith("B") else "ablation arm"


# --------------------------------------------------------------------------- loading
def load_runs(results: Path, run_kind: str):
    rows, preds = [], {}
    for mf in sorted(results.glob("runs/*/metrics.json")):
        r = json.load(open(mf))
        assert r["run_kind"] == run_kind, f"{mf}: run_kind {r['run_kind']} != {run_kind}"
        row = {k: r[k] for k in ("job_id", "config", "task", "seed", "lambda_cons", "epochs", "data_fraction",
                                 "schedule", "n_params", "n_train", "n_id_test", "n_val", "n_target",
                                 "train_time_s", "peak_gpu_mem_mb", "latency_median_ms", "latency_p95_ms",
                                 "best_epoch", "best_source_val_f1", "features_sha256", "device", "train_acc_best",
                                 "mean_conf_target")}
        for part in ("id_test", "val", "target"):
            for m in ("accuracy", "macro_f1", "worst_f1", "clap_slide_error"):
                row[f"{part}_{m}"] = r["best"][part][m]
        for m in ("accuracy", "macro_f1", "clap_slide_error"):
            row[f"last_target_{m}"] = r["last"]["target"][m]
        row["gap_macro_f1"] = row["id_test_macro_f1"] - row["target_macro_f1"]
        row["gap_accuracy"] = row["id_test_accuracy"] - row["target_accuracy"]
        row["_metrics"] = r
        rows.append(row)
        preds[r["job_id"]] = mf.with_name("preds.npz")
    assert rows, f"no runs under {results}"
    df = pd.DataFrame(rows)
    assert df["features_sha256"].nunique() == 1, "runs were trained on different feature caches"
    assert df["job_id"].is_unique
    return df, preds


def core(df):
    return df[np.isclose(df["lambda_cons"], 0.10)]


def pm(vals, scale=100.0, nd=2):
    v = np.asarray(list(vals), dtype=float) * scale
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f} (n=1)"


def write_table(df: pd.DataFrame, out: Path, name: str, caption: str) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / f"{name}.csv", index=False)
    esc = lambda s: str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%") \
        .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("↔", r"$\leftrightarrow$") \
        .replace("→", r"$\rightarrow$").replace("λ", r"$\lambda$").replace("−", "$-$")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
             r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\hline",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\hline"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (out / "tables" / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- tables
def table_data(manifest, splits, fmeta, out):
    rows = []
    for t, spec in TASKS.items():
        c = splits[f"split_{t}"].value_counts()
        rows.append({"Task": t, "Train orientations": ", ".join(map(str, spec["train_orientations"])),
                     "Source-val orientation": spec["val_orientation"], "Target orientation": spec["target_orientation"],
                     "Train": int(c.get("train", 0)), "In-domain test": int(c.get("id_test", 0)),
                     "Source val": int(c.get("val", 0)), "Target": int(c.get("target", 0)),
                     "Unassigned (excluded)": int(c.get("", 0))})
    write_table(pd.DataFrame(rows), out, "T1a_splits", "Trial-level splits per task (9 users, Room 1, receiver r1).")
    exc = manifest[~manifest["valid"]]
    summ = pd.DataFrame([
        {"Quantity": "Users (Room 1, session 20181130)", "Value": ", ".join(map(str, sorted(manifest.user.unique())))},
        {"Quantity": "r1 trials found (gestures 1-6)", "Value": len(manifest)},
        {"Quantity": "Trials per user", "Value": ", ".join(f"{u}:{n}" for u, n in manifest.groupby("user").size().items())},
        {"Quantity": "Zero-byte files", "Value": int((manifest.file_bytes == 0).sum())},
        {"Quantity": "Usable trials", "Value": int(manifest.valid.sum())},
        {"Quantity": "Excluded trials", "Value": len(exc)},
        {"Quantity": "Exclusions by reason", "Value": "; ".join(
            f"{k}: {v}" for k, v in exc.exclusion_reason.str.split(":").str[0].value_counts().items()) or "none"},
        {"Quantity": "Excluded trial IDs", "Value": ", ".join(exc.trial_id) or "none"},
        {"Quantity": "Packets per usable trial (median)", "Value": f"{manifest[manifest.valid].n_packets.median():.0f}"},
        {"Quantity": "Trial duration s (min / median / max)", "Value": "{:.2f} / {:.2f} / {:.2f}".format(
            *manifest[manifest.valid].duration_s.agg(["min", "median", "max"]))},
        {"Quantity": "Packets dropped for Nrx != 3", "Value": int(manifest.n_dropped_nrx.fillna(0).sum())},
        {"Quantity": "Duplicate timestamps removed", "Value": int(manifest.n_dups.fillna(0).sum())},
        {"Quantity": "Locked preprocessing", "Value": json.dumps(fmeta["preprocessing"])},
        {"Quantity": "Feature cache SHA-256", "Value": fmeta["sha256"]},
    ])
    write_table(summ, out, "T1b_data_audit", "Dataset audit and exclusions.")
    return summ


def table_main(df, out):
    d = core(df)
    rows = []
    for scope in list(TASKS) + ["Core mean"]:
        for cfg in ORDER:
            s = d[d.config == cfg] if scope == "Core mean" else d[(d.config == cfg) & (d.task == scope)]
            if s.empty:
                continue
            if scope == "Core mean":
                if s.groupby("seed").task.nunique().min() < len(TASKS):
                    continue
                s = s.groupby("seed").mean(numeric_only=True)       # tasks weighted equally within each seed
            rows.append({"Task": scope, "Config": cfg, "Seeds": len(s),
                         "Target acc (%)": pm(s.target_accuracy), "Target macro-F1 (%)": pm(s.target_macro_f1),
                         "Worst-class F1 (%)": pm(s.target_worst_f1), "Clap↔slide error (%)": pm(s.target_clap_slide_error),
                         "In-domain macro-F1 (%)": pm(s.id_test_macro_f1), "ID − target gap (pp)": pm(s.gap_macro_f1),
                         "Source-val macro-F1 (%)": pm(s.val_macro_f1)})
    t = pd.DataFrame(rows)
    write_table(t, out, "T2_main_results", "Main results: mean ± sample SD over seeds; the core mean weights tasks equally.")
    return t


def best_baseline(df):
    d = core(df)
    d = d[d.config.isin(B_FAMILY)]
    if d.empty:
        return None, pd.DataFrame()
    s = d.groupby("config").agg(mean_source_val_macro_f1=("best_source_val_f1", "mean"), runs=("job_id", "count"))
    s = s.reindex([b for b in B_FAMILY if b in s.index])
    return s.mean_source_val_macro_f1.idxmax(), s.reset_index()


def paired(df, a, b, col, scope):
    d = core(df)
    pa = d[d.config == a].set_index(["task", "seed"])[col]
    pb = d[d.config == b].set_index(["task", "seed"])[col]
    diff = (pa - pb).dropna()
    if scope != "Core mean":
        return diff[diff.index.get_level_values("task") == scope].to_numpy()
    by_seed = diff.groupby("seed")
    return by_seed.mean()[by_seed.count() == len(TASKS)].to_numpy()


def table_ablation(df, out, extra):
    rows = []
    for a, b, meaning in CONTRASTS + extra:
        for scope in list(TASKS) + ["Core mean"]:
            vals = {m: paired(df, a, b, f"target_{m}", scope) for m, _ in METRICS}
            if len(vals["macro_f1"]) == 0:
                continue
            rows.append({"Contrast": f"{a} − {b}", "Tests": meaning, "Task": scope, "Paired seeds": len(vals["macro_f1"]),
                         "Δ accuracy (pp)": pm(vals["accuracy"]), "Δ macro-F1 (pp)": pm(vals["macro_f1"]),
                         "Δ clap↔slide error (pp)": pm(vals["clap_slide_error"])})
    v = {m: paired(df, "C5", "C4", f"last_target_{m}", "Core mean") for m, _ in METRICS}
    if len(v["macro_f1"]):
        rows.append({"Contrast": "C5 − C4 (epoch 30)", "Tests": "schedule, equal augmentation exposure", "Task": "Core mean",
                     "Paired seeds": len(v["macro_f1"]), "Δ accuracy (pp)": pm(v["accuracy"]),
                     "Δ macro-F1 (pp)": pm(v["macro_f1"]), "Δ clap↔slide error (pp)": pm(v["clap_slide_error"])})
    t = pd.DataFrame(rows)
    write_table(t, out, "T3_ablation", "Paired ablation deltas (same task and seed), mean ± SD over seeds; pp = percentage points.")
    return t


def cm_metrics(cm):
    """cm: (..., 6, 6) -> accuracy, macro-F1, clap/slide error (vectorised over leading axes)."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    row, col = cm.sum(-1), cm.sum(-2)
    f1 = np.where(row + col > 0, 2 * tp / np.maximum(row + col, 1e-12), 0.0)
    cs = 0.5 * (cm[..., CLAP, SLIDE] / np.maximum(row[..., CLAP], 1e-12) + cm[..., SLIDE, CLAP] / np.maximum(row[..., SLIDE], 1e-12))
    return {"accuracy": tp.sum(-1) / cm.sum((-2, -1)), "macro_f1": f1.mean(-1), "clap_slide_error": cs}


def user_cms(pred_file, user_of):
    p = np.load(pred_file, allow_pickle=True)
    users = np.array([user_of[t] for t in p["target_trial_id"]])
    ulist = sorted(set(user_of.values()))
    cms = np.zeros((len(ulist), NUM_CLASSES, NUM_CLASSES))
    np.add.at(cms, (np.searchsorted(ulist, users), p["target_y_true"] - 1, p["target_y_pred"] - 1), 1)
    return cms, p


def table_bootstrap(df, preds, manifest, out, extra, n_boot=1000, seed=0):
    """Paired user-cluster bootstrap. One draw = 9 users sampled with replacement; the SAME draw is applied to
    both models, every task and every seed. Deltas are averaged over seeds, and over tasks for the core mean."""
    user_of = dict(zip(manifest.trial_id, manifest.user))
    n_users = manifest.user.nunique()
    rng = np.random.RandomState(seed)
    W = rng.multinomial(n_users, np.ones(n_users) / n_users, size=n_boot).astype(float)     # (n_boot, users)
    d = core(df).set_index(["config", "task", "seed"])
    cache, num = {}, {}

    def get(cfg, task, s):
        key = (cfg, task, s)
        if key not in cache:
            cache[key] = user_cms(preds[d.loc[key, "job_id"]], user_of)[0]
        return cache[key]

    rows = []
    for a, b, meaning in CONTRASTS[:1] + extra + CONTRASTS[1:]:
        per_task = {}
        for task in TASKS:
            seeds = [s for s in SEEDS if (a, task, s) in d.index and (b, task, s) in d.index]
            if not seeds:
                continue
            obs = {m: [] for m, _ in METRICS}; boot = {m: [] for m, _ in METRICS}
            for s in seeds:
                ca, cb = get(a, task, s), get(b, task, s)
                assert ca.sum() == cb.sum(), "paired runs evaluated on different target trials"
                ma, mb = cm_metrics(ca.sum(0)), cm_metrics(cb.sum(0))
                ba, bb = cm_metrics(np.einsum("nu,uij->nij", W, ca)), cm_metrics(np.einsum("nu,uij->nij", W, cb))
                for m, _ in METRICS:
                    obs[m].append(ma[m] - mb[m]); boot[m].append(ba[m] - bb[m])
            per_task[task] = ({m: np.mean(v) for m, v in obs.items()}, {m: np.mean(v, axis=0) for m, v in boot.items()}, len(seeds))
        scopes = dict(per_task)
        if len(per_task) == len(TASKS):
            scopes["Core mean"] = ({m: np.mean([per_task[t][0][m] for t in TASKS]) for m, _ in METRICS},
                                   {m: np.mean([per_task[t][1][m] for t in TASKS], axis=0) for m, _ in METRICS},
                                   min(per_task[t][2] for t in TASKS))
        for scope, (o, bt, ns) in scopes.items():
            row = {"Contrast": f"{a} − {b}", "Tests": meaning, "Task": scope, "Seeds": ns, "Users": n_users, "Replicates": n_boot}
            for m, lab in METRICS:
                lo, hi = np.percentile(bt[m], [2.5, 97.5]) * 100
                row[f"Δ {lab} (pp)"] = f"{o[m] * 100:+.2f}"
                row[f"95% CI {lab} (pp)"] = f"[{lo:+.2f}, {hi:+.2f}]"
                num.setdefault(f"{a}-{b}", {}).setdefault(scope, {})[m] = {"delta_pp": float(o[m] * 100), "ci_lo_pp": float(lo),
                                                                              "ci_hi_pp": float(hi), "seeds": ns}
            rows.append(row)
    t = pd.DataFrame(rows)
    write_table(t, out, "T4_bootstrap", "Paired user-cluster bootstrap (1,000 replicates, same draw for both models). "
                "95% percentile intervals are EXPLORATORY: 9 users and 3 orientation shifts.")
    return t, num


def table_cost(df, fmeta, out):
    d = core(df)
    rows = []
    for cfg in ORDER:
        s = d[d.config == cfg]
        if s.empty:
            continue
        rows.append({"Config": cfg, "Runs": len(s), "Trainable params": int(s.n_params.iloc[0]),
                     "Train time per run (min)": pm(s.train_time_s, scale=1 / 60),
                     "Peak GPU memory (MB)": "n/a (CPU)" if s.peak_gpu_mem_mb.isna().all() else f"{s.peak_gpu_mem_mb.max():.0f}",
                     "Batch-1 latency median (ms)": f"{s.latency_median_ms.median():.2f}",
                     "Batch-1 latency p95 (ms)": f"{s.latency_p95_ms.median():.2f}",
                     "Device": s.device.iloc[0]})
    t = pd.DataFrame(rows)
    write_table(t, out, "T5_cost", "Computational cost. Latency is the frozen model only (embedding + classifier); feature "
                f"extraction (decode + maps) took a median of {fmeta['decode_plus_feature_ms_median']:.0f} ms per trial on one CPU core.")
    return t


def table_literature(df, out):
    rows = [{"System": r["system"], "Receivers": r["receivers"], "Input": r["input"], "Target data used": r["target_data"],
             "In-domain acc (%)": r["in_domain"], "Cross-orientation acc (%)": r["cross_orientation"],
             "Source": "reported by the paper", "Fairness note": r["fairness"]} for r in LITERATURE]
    d = core(df)
    for cfg in ("C1", "C5"):
        s = d[d.config == cfg]
        if s.empty or s.groupby("seed").task.nunique().min() < len(TASKS):
            continue
        g = s.groupby("seed").mean(numeric_only=True)
        rows.append({"System": f"This work, {cfg}", "Receivers": "1 receiver (r1), 2 antennas",
                     "Input": "CSI-ratio phase + Doppler (same link)", "Target data used": "none (strict DG)",
                     "In-domain acc (%)": pm(g.id_test_accuracy), "Cross-orientation acc (%)": pm(g.target_accuracy),
                     "Source": "measured here", "Fairness note": f"{int(s.n_params.iloc[0]):,} parameters; leave-one-orientation-out "
                     "over 9 users; trained on 3 orientations; mean ± SD over seeds of the 3-task mean."})
    t = pd.DataFrame(rows)
    write_table(t, out, "T6_literature_context", "Literature context. Published numbers come from different protocols and are "
                "not ranked against ours.")
    return t


def table_lambda(df, out):
    d = df[(df.config == "C5") & (df.seed == 42)]
    rows = []
    for lam, s in d.groupby("lambda_cons"):
        if s.task.nunique() < len(TASKS):
            continue
        row = {"λ": f"{lam:g}", "Seed": 42}
        for t in TASKS:
            row[f"{t} macro-F1 (%)"] = f"{s[s.task == t].target_macro_f1.iloc[0] * 100:.2f}"
        row["Core macro-F1 (%)"] = f"{s.target_macro_f1.mean() * 100:.2f}"
        row["Core clap↔slide error (%)"] = f"{s.target_clap_slide_error.mean() * 100:.2f}"
        row["Core source-val macro-F1 (%)"] = f"{s.val_macro_f1.mean() * 100:.2f}"
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        write_table(t, out, "T7_lambda_sensitivity", "Sensitivity of C5 to the consistency weight λ (seed 42, target domain).")
    return t


def pooled_cm(df, cfg, task, part="target"):
    s = core(df)
    s = s[(s.config == cfg) & (s.task == task)]
    return np.sum([np.array(m["best"][part]["cm"]) for m in s["_metrics"]], axis=0) if len(s) else None


def table_clap_slide(df, out):
    rows = []
    for task in TASKS:
        for cfg in ORDER:
            cm = pooled_cm(df, cfg, task)
            if cm is None:
                continue
            c2s, nc, s2c, ns = cm[CLAP, SLIDE], cm[CLAP].sum(), cm[SLIDE, CLAP], cm[SLIDE].sum()
            rows.append({"Task": task, "Config": cfg, "Clap→slide / actual clap": f"{c2s}/{nc}", "Rate (%)": f"{100 * c2s / nc:.2f}",
                         "Slide→clap / actual slide": f"{s2c}/{ns}", "Rate (%) ": f"{100 * s2c / ns:.2f}",
                         "Pair error (%)": f"{50 * (c2s / nc + s2c / ns):.2f}"})
    t = pd.DataFrame(rows)
    write_table(t, out, "T8_clap_slide", "Directional clap/slide errors on the target domain. Counts pool the seeds, so "
                "denominators count trial-seed decisions.")
    return t


def table_per_class(df, out):
    d = core(df)
    rows = []
    for cfg in ORDER:
        s = d[d.config == cfg]
        if s.empty or s.groupby("seed").task.nunique().min() < len(TASKS):
            continue
        f1 = np.array([[np.mean([m["best"]["target"]["per_class_f1"][k] for m in s[s.seed == sd]["_metrics"]])
                        for k in range(NUM_CLASSES)] for sd in sorted(s.seed.unique())])
        rows.append({"Config": cfg, **{GEST[k]: pm(f1[:, k]) for k in range(NUM_CLASSES)}})
    t = pd.DataFrame(rows)
    write_table(t, out, "T9_per_class_f1", "Per-class target F1 (%), core mean: tasks averaged within seed, mean ± SD over seeds.")
    return t


def table_failures(df, preds, manifest, out):
    d = core(df)
    c5 = d[d.config == "C5"]
    if c5.empty or d[d.config == "C1"].empty:
        return pd.DataFrame(), None
    worst = c5.groupby("task").target_macro_f1.mean().idxmin()
    seed = int(c5[c5.task == worst].seed.min())
    key = lambda c: d[(d.config == c) & (d.task == worst) & (d.seed == seed)].job_id.iloc[0]
    p5, p1 = np.load(preds[key("C5")], allow_pickle=True), np.load(preds[key("C1")], allow_pickle=True)
    assert (p5["target_trial_id"] == p1["target_trial_id"]).all()
    t = pd.DataFrame({"Trial": p5["target_trial_id"], "Actual": [GEST[i - 1] for i in p5["target_y_true"]],
                      "C5 prediction": [GEST[i - 1] for i in p5["target_y_pred"]], "C5 confidence": np.round(p5["target_conf"], 3),
                      "C1 prediction": [GEST[i - 1] for i in p1["target_y_pred"]], "C1 confidence": np.round(p1["target_conf"], 3)})
    t = t[t["Actual"] != t["C5 prediction"]].merge(
        manifest[["trial_id", "user", "location", "duration_s", "max_gap_ms"]].rename(columns={"trial_id": "Trial"}), on="Trial")
    t = t.sort_values("C5 confidence", ascending=False).head(10).round({"duration_s": 3, "max_gap_ms": 1})
    write_table(t, out, "T10_failure_cases", f"Ten most confident C5 errors on the worst C5 task ({worst}, seed {seed}), with the "
                "C1 prediction on the same trial. Softmax confidence is not a calibrated probability.")
    return t, worst


def table_per_user(df, preds, manifest, out):
    user_of = dict(zip(manifest.trial_id, manifest.user))
    ulist = sorted(set(user_of.values()))
    d = core(df)
    rows = []
    for cfg in ("C1", "C5"):
        s = d[d.config == cfg]
        if s.empty:
            continue
        acc = np.mean([(lambda c: np.diagonal(c, axis1=1, axis2=2).sum(1) / c.sum((1, 2)))(user_cms(preds[j], user_of)[0])
                       for j in s.job_id], axis=0)
        rows.append({"Config": cfg, **{f"user{u}": f"{100 * a:.1f}" for u, a in zip(ulist, acc)}})
    t = pd.DataFrame(rows)
    if len(t):
        write_table(t, out, "T11_per_user_accuracy", "Target accuracy (%) per participant, averaged over tasks and seeds.")
    return t


def table_gate(out, diag: Path):
    f = diag / "gate_table.csv"
    if not f.exists():
        return pd.DataFrame()
    g = pd.read_csv(f)
    t = pd.DataFrame({"Hampel placement": g.hampel_mode, "Input channel(s)": g.channels, "Head scale": g.head_scale, "Epochs": g.epochs,
                      "Optimizer steps": g.optimizer_steps,
                      "Train acc (%)": (100 * g.train_acc).round(2), "Held-out acc (%)": (100 * g.heldout_acc).round(2),
                      "Train / held-out trials": g.n_train.astype(str) + " / " + g.n_heldout.astype(str)})
    write_table(t, out, "T0_preprocessing_gate", "Preprocessing gate: plain ERM on orientations 2 and 4 only (never target "
                "domains), random 80/20 split stratified by user, gesture and orientation.")
    return t


# --------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "legend.frameon": False, "axes.axisbelow": True})
    return plt


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", BLUES)


def _save(fig, out, name):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _role_legend(ax_or_fig, cfgs, **kw):
    from matplotlib.patches import Patch
    roles = [r for r in ROLE_COLOR if r in {role(c) for c in cfgs}]
    ax_or_fig.legend(handles=[Patch(color=ROLE_COLOR[r], label=r) for r in roles], **kw)


def fig_bars(df, out, col, ylabel, name, title, lower_better=False):
    plt = _plt()
    d = core(df)
    cfgs = [c for c in ORDER if c in set(d.config)]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        s = d[d.task == task]
        mean = [100 * s[s.config == c][col].mean() for c in cfgs]
        sd = [100 * s[s.config == c][col].std(ddof=1) if (s.config == c).sum() > 1 else 0 for c in cfgs]
        ax.bar(range(len(cfgs)), mean, yerr=sd, width=0.72, color=[ROLE_COLOR[role(c)] for c in cfgs],
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for i, c in enumerate(cfgs):
            if c in ("C1", "C5"):
                ax.text(i, mean[i] + sd[i] + 0.8, f"{mean[i]:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs)
        ax.set_title(f"Task {task} (target orientation {TASKS[task]['target_orientation']})")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    _role_legend(fig, cfgs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title + (" (lower is better)" if lower_better else ""), fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, name)


def fig_ablation(df, out, extra):
    plt = _plt()
    cons = [c for c in CONTRASTS + extra if len(paired(df, c[0], c[1], "target_macro_f1", "Core mean"))]
    if not cons:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(cons) + 1.6), sharey=True)
    for ax, (m, lab) in zip(axes, [("macro_f1", "Δ target macro-F1 (pp)"), ("clap_slide_error", "Δ clap/slide error (pp)")]):
        for i, (a, b, _) in enumerate(cons):
            per_task = [100 * paired(df, a, b, f"target_{m}", t).mean() for t in TASKS]
            v = 100 * paired(df, a, b, f"target_{m}", "Core mean")
            ax.barh(i, v.mean(), xerr=v.std(ddof=1) if len(v) > 1 else 0, height=0.5, color=CAT[0],
                    error_kw=dict(ecolor=INK2, lw=1, capsize=2))
            ax.scatter(per_task, [i] * len(TASKS), s=22, color=SURFACE, edgecolor=INK, linewidth=0.9, zorder=3)
            ax.text(ax.get_xlim()[1], i, "", fontsize=7)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(cons))); ax.set_yticklabels([f"{a} − {b}" for a, b, _ in cons])
        ax.set_xlabel(lab); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    axes[0].set_title("Higher is better")
    axes[1].set_title("Lower is better")
    fig.suptitle("Ablation deltas: bar = core mean ± SD over seeds, circles = individual tasks",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F3_ablation_deltas")


def fig_confusion(df, out):
    plt = _plt()
    fig, axes = plt.subplots(2, len(TASKS), figsize=(11.5, 7.2))
    short = ["Push", "Sweep", "Clap", "Slide", "Draw-O", "Zigzag"]
    drew = False
    for r, cfg in enumerate(("C1", "C5")):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            cm = pooled_cm(df, cfg, task)
            if cm is None:
                ax.axis("off"); continue
            drew = True
            pct = 100 * cm / cm.sum(1, keepdims=True)
            ax.imshow(pct, cmap=_seq_cmap(), vmin=0, vmax=100)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if pct[i, j] > 55 else INK)
            for (i, j) in ((CLAP, SLIDE), (SLIDE, CLAP)):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor=CAT[1], lw=1.8))
            ax.set_xticks(range(6)); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
            ax.set_yticks(range(6)); ax.set_yticklabels(short, fontsize=7.5)
            ax.set_title(f"{cfg}, task {task}"); ax.grid(False)
            if c == 0:
                ax.set_ylabel("Actual gesture")
            if r == 1:
                ax.set_xlabel("Predicted gesture")
    if not drew:
        plt.close(fig); return
    fig.suptitle("Target-domain confusion (row %, seeds pooled); outlined cells are clap/slide confusions",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F4_confusion_C1_vs_C5")


def fig_gap(df, out):
    plt = _plt()
    d = core(df)
    cfgs = [c for c in ORDER if c in set(d.config)]
    g = d.groupby("config")[["id_test_macro_f1", "target_macro_f1"]].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(cfgs) + 1.5))
    for i, c in enumerate(cfgs):
        ax.plot([g.loc[c, "target_macro_f1"], g.loc[c, "id_test_macro_f1"]], [i, i], color=AXIS, lw=2, zorder=1)
        ax.text(g.loc[c].max() + 1.0, i, f"gap {g.loc[c, 'id_test_macro_f1'] - g.loc[c, 'target_macro_f1']:+.1f} pp",
                va="center", fontsize=7.5, color=INK2)
    ax.scatter(g.loc[cfgs, "id_test_macro_f1"], range(len(cfgs)), s=46, color=CAT[0], label="In-domain test (seen orientations)", zorder=2)
    ax.scatter(g.loc[cfgs, "target_macro_f1"], range(len(cfgs)), s=46, color=CAT[1], label="Target (unseen orientation)", zorder=2)
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(cfgs); ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (%), mean over tasks and seeds"); ax.grid(axis="y", visible=False)
    ax.set_xlim(right=g.values.max() + 12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title("Generalization gap: in-domain vs unseen-orientation macro-F1", loc="left", pad=28)
    fig.tight_layout()
    _save(fig, out, "F6_generalization_gap")


def fig_per_class(df, out):
    plt = _plt()
    d = core(df)
    cfgs = [c for c in ORDER if c in set(d.config)]
    M = np.array([[100 * np.mean([m["best"]["target"]["per_class_f1"][k] for m in d[d.config == c]["_metrics"]])
                   for k in range(NUM_CLASSES)] for c in cfgs])
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(cfgs) + 1.4))
    im = ax.imshow(M, cmap=_seq_cmap(), vmin=0, vmax=100, aspect="auto")
    for i in range(len(cfgs)):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="#ffffff" if M[i, j] > 55 else INK)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(GEST, rotation=25, ha="right")
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(cfgs); ax.grid(False)
    fig.colorbar(im, ax=ax, label="Target F1 (%)")
    ax.set_title("Per-class target F1, mean over tasks and seeds", loc="left")
    fig.tight_layout()
    _save(fig, out, "F7_per_class_f1")


def fig_tsne(df, preds, out, task):
    from sklearn.manifold import TSNE
    plt = _plt()
    d = core(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, cfg in zip(axes, ("C1", "C5")):
        s = d[(d.config == cfg) & (d.task == task)]
        if s.empty:
            plt.close(fig); return
        s = s[s.seed == s.seed.min()]
        p = np.load(preds[s.job_id.iloc[0]], allow_pickle=True)
        z, y, yp = p["target_embedding"], p["target_y_true"], p["target_y_pred"]
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
        for k in range(NUM_CLASSES):
            mk = y == k + 1
            ax.scatter(xy[mk & (y == yp), 0], xy[mk & (y == yp), 1], s=10, color=CAT[k], label=GEST[k], linewidths=0)
            ax.scatter(xy[mk & (y != yp), 0], xy[mk & (y != yp), 1], s=14, color=CAT[k], marker="x", linewidths=0.9)
        ax.set_title(f"{cfg}, task {task}, seed {int(s.seed.iloc[0])} (target accuracy {100 * (y == yp).mean():.1f}%)")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Actual gesture\n(x = misclassified)", markerscale=2)
    fig.suptitle("t-SNE of frozen target embeddings (diagnostic only; each panel has its own axes)",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F8_tsne_C1_vs_C5")


def fig_curves(df, out):
    plt = _plt()
    d = core(df)
    show = [c for c in ("C1", "C5", "B1") if c in set(d.config)]
    if not show:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, lab) in zip(axes, [("train_loss", "Training loss"), ("train_acc", "Train accuracy (clean)"),
                                     ("val_macro_f1", "Source-validation macro-F1")]):
        for c in show:
            cur = np.array([m["training_curves"][key] for m in d[d.config == c]["_metrics"]], dtype=float)
            ep = np.arange(1, cur.shape[1] + 1)
            scale = 1 if key == "train_loss" else 100
            ax.plot(ep, scale * cur.mean(0), color=ROLE_COLOR[role(c)], lw=2, label=f"{c} ({role(c)})")
            ax.fill_between(ep, scale * cur.min(0), scale * cur.max(0), color=ROLE_COLOR[role(c)], alpha=0.15, lw=0)
        ax.set_xlabel("Epoch"); ax.set_title(lab + ("" if key == "train_loss" else " (%)"))
    axes[0].legend(loc="upper right")
    fig.suptitle("Training curves: mean over tasks and seeds, band = min-max over runs",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F9_training_curves")


def fig_lambda(df, out):
    d = df[(df.config == "C5") & (df.seed == 42)]
    lams = sorted(l for l, s in d.groupby("lambda_cons") if s.task.nunique() == len(TASKS))
    if len(lams) < 2:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, (col, lab) in zip(axes, [("target_macro_f1", "Target macro-F1 (%)"), ("target_clap_slide_error", "Clap/slide error (%)")]):
        for k, t in enumerate(TASKS):
            y = [100 * d[(d.task == t) & np.isclose(d.lambda_cons, l)][col].iloc[0] for l in lams]
            ax.plot(range(len(lams)), y, marker="o", ms=5, lw=1.5, color=CAT[k + 2], label=f"Task {t}")
        y = [100 * d[np.isclose(d.lambda_cons, l)][col].mean() for l in lams]
        ax.plot(range(len(lams)), y, marker="o", ms=7, lw=2.5, color=CAT[0], label="Core mean")
        ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.set_xlabel("Consistency weight λ"); ax.set_title(lab)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Sensitivity of C5 to λ (seed 42)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F10_lambda_sensitivity")


def fig_example_maps(manifest, splits, features, fmeta, out):
    plt = _plt()
    X = np.load(features, mmap_mode="r")
    tr = splits["split_O1"].to_numpy() == "train"
    fig, axes = plt.subplots(2, 6, figsize=(15, 4.6))
    for g in range(1, 7):
        i = int(np.where(tr & (manifest.gesture.to_numpy() == g))[0][0])
        for r, (ttl, cm) in enumerate([("phase", "viridis"), ("Doppler", "magma")]):
            ax = axes[r, g - 1]
            im = ax.imshow(np.asarray(X[i, r], dtype=np.float32), aspect="auto", origin="lower", cmap=cm)
            ax.set_title(f"{GESTURE_NAMES[g]}: {ttl}\n{manifest.trial_id[i]}", fontsize=8)
            ax.tick_params(labelsize=6); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        axes[1, g - 1].set_xlabel("time (resized to 128)", fontsize=7)
    axes[0, 0].set_ylabel("subcarrier (resized to 64)", fontsize=7)
    axes[1, 0].set_ylabel("Doppler −60..+60 Hz (64 bins)", fontsize=7)
    p = fmeta["preprocessing"]
    fig.suptitle(f"Locked feature maps, one O1 training trial per gesture (phase: {p['phase_variant']}, Doppler: "
                 f"{p['doppler_variant']}, Hampel on: {p['hampel_mode']}); no target-domain trial is shown",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "F1_example_maps")


def fig_gate(out, diag: Path):
    f = diag / "gate_table.csv"
    if not f.exists():
        return
    plt = _plt()
    g = pd.read_csv(f).sort_values("heldout_acc")
    lab = g.channels.str.replace("phase:", "phase ").str.replace("doppler:", "Doppler ") + "  [Hampel: " + g.hampel_mode + \
        np.where(g.head_scale != 10, ", head scale " + g.head_scale.astype(str), "") + ", " + g.epochs.astype(str) + " ep]"
    fig, ax = plt.subplots(figsize=(8.5, 0.27 * len(g) + 1.4))
    y = np.arange(len(g))
    ax.barh(y, 100 * g.heldout_acc, height=0.62, color=np.where(g.channels.str.contains(r"\+"), CAT[0], MUTED))
    ax.scatter(100 * g.train_acc, y, s=18, color=SURFACE, edgecolor=INK, linewidth=0.9, zorder=3, label="train accuracy")
    ax.axvline(100 / 6, color=INK2, lw=0.8, ls=":"); ax.text(100 / 6 + 0.5, -0.9, "chance", fontsize=7, color=INK2)
    ax.axvline(70, color=CAT[1], lw=1); ax.text(70.5, -0.9, "70% gate", fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7); ax.set_xlim(0, 100)
    ax.set_xlabel("ERM accuracy at the last epoch (%)  |  bar = held-out, circle = train; blue = fused two-branch model")
    ax.grid(axis="y", visible=False)
    ax.set_title("Preprocessing gate on orientations 2 and 4 (never target domains)", loc="left")
    fig.tight_layout()
    _save(fig, out, "F0_preprocessing_gate")


# --------------------------------------------------------------------------- main
def expected_jobs(df):
    have = set(zip(df.config, df.task, df.seed, df.lambda_cons.round(4)))
    miss = [(c, t, s) for c in CONFIGS for t in TASKS for s in SEEDS if (c, t, s, 0.1) not in have]
    miss += [("C5", t, 42, lam) for t in TASKS for lam in (0.05, 0.3) if ("C5", t, 42, lam) not in have]
    return miss


def run(results: Path, run_kind: str, cache: Path, diag: Path, require: str, n_boot: int = 1000) -> dict:
    df, preds = load_runs(results, run_kind)
    manifest = pd.read_csv(cache / "manifest.csv", keep_default_na=False)
    manifest["valid"] = manifest["valid"].astype(str).str.lower().eq("true")
    for c in ("n_packets", "duration_s", "max_gap_ms", "n_dropped_nrx", "n_dups"):
        manifest[c] = pd.to_numeric(manifest[c], errors="coerce")
    splits = pd.read_csv(cache / "splits.csv", keep_default_na=False)
    fmeta = json.load(open(cache / "features_meta.json"))
    assert df.features_sha256.iloc[0] == fmeta["sha256"], "runs were not trained on the cache in " + str(cache)

    missing = expected_jobs(df)
    core_missing = [m for m in missing if m[0] in C_FAMILY and len(m) == 3]
    if require == "all":
        assert not missing, f"{len(missing)} of 96 runs missing, e.g. {missing[:5]}"
    elif require == "core":
        assert not core_missing, f"core C0-C5 runs missing: {core_missing[:5]}"
    out = results
    bestB, bsel = best_baseline(df)
    extra = [("C5", bestB, f"full method vs best external baseline ({bestB}, chosen by source-validation macro-F1)")] if bestB else []
    if len(bsel):
        write_table(bsel.round(4), out, "T2b_best_baseline_selection", "Selection of the best external DG baseline by mean "
                    "source-validation macro-F1 (target data not used).")

    table_gate(out, diag)
    table_data(manifest, splits, fmeta, out)
    main = table_main(df, out)
    table_ablation(df, out, extra)
    boot, boot_num = table_bootstrap(df, preds, manifest, out, extra, n_boot=n_boot)
    table_cost(df, fmeta, out)
    table_literature(df, out)
    table_lambda(df, out)
    table_clap_slide(df, out)
    table_per_class(df, out)
    _, worst = table_failures(df, preds, manifest, out)
    table_per_user(df, preds, manifest, out)

    fig_gate(out, diag)
    if (cache / "features.npy").exists():
        fig_example_maps(manifest, splits, cache / "features.npy", fmeta, out)
    fig_bars(df, out, "target_macro_f1", "Target macro-F1 (%)", "F2_macro_f1_bars",
             "Target macro-F1 per configuration and task (mean ± SD over seeds)")
    fig_ablation(df, out, extra)
    fig_confusion(df, out)
    fig_bars(df, out, "target_clap_slide_error", "Clap/slide error (%)", "F5_clap_slide_error",
             "Clap/slide confusion on the target domain", lower_better=True)
    fig_gap(df, out)
    fig_per_class(df, out)
    if worst:
        fig_tsne(df, preds, out, worst)
    fig_curves(df, out)
    fig_lambda(df, out)

    runs_csv = df.drop(columns=["_metrics"]).sort_values(["config", "task", "seed", "lambda_cons"])
    runs_csv.to_csv(out / "tables" / "all_runs.csv", index=False)
    d = core(df)
    cm = lambda c, col: float(d[d.config == c].groupby("seed")[col].mean().mean()) if (d.config == c).any() else None
    summary = {"run_kind": run_kind, "n_runs": int(len(df)), "n_missing_of_96": len(missing), "missing": [list(map(str, m)) for m in missing],
               "best_external_baseline": bestB, "worst_c5_task": worst, "features_sha256": fmeta["sha256"],
               "preprocessing": fmeta["preprocessing"], "device": sorted(df.device.unique().tolist()),
               "torch": sorted({m["torch"] for m in df["_metrics"]}), "n_boot": n_boot, "contrasts": boot_num,
               "per_task": {c: {t: {k: float(d[(d.config == c) & (d.task == t)][k].mean()) for k in ("target_accuracy", "target_macro_f1", "target_clap_slide_error", "id_test_macro_f1")}
                                for t in TASKS if ((d.config == c) & (d.task == t)).any()} for c in ORDER if (d.config == c).any()},
               "n_params": {c: int(d[d.config == c].n_params.iloc[0]) for c in ORDER if (d.config == c).any()},
               "total_train_hours": float(df.train_time_s.sum() / 3600),
               "core": {c: {k: cm(c, k) for k in ("target_accuracy", "target_macro_f1", "target_clap_slide_error", "target_worst_f1",
                                                    "id_test_accuracy", "id_test_macro_f1", "gap_macro_f1", "val_macro_f1",
                                                    "mean_conf_target", "train_acc_best")}
                        for c in ORDER if (d.config == c).any()}}
    json.dump(summary, open(out / "tables" / "summary.json", "w"), indent=2)
    print(f"[analyze] {len(df)} runs ({len(missing)} of 96 missing) -> {out / 'tables'}, {out / 'figures'}")
    print(main[main.Task == "Core mean"].to_string(index=False))
    if len(boot):
        print(boot[boot.Task == "Core mean"].to_string(index=False))
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final")
    ap.add_argument("--run_kind", default="final", choices=["final", "quick", "pilot"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--diag", default="results_diag")
    ap.add_argument("--require", default="all", choices=["all", "core", "none"],
                    help="fail unless all 96 runs / the 54 core runs are present")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    run(Path(a.results), a.run_kind, Path(a.cache), Path(a.diag), a.require, a.n_boot)


if __name__ == "__main__":
    sys.exit(main())
