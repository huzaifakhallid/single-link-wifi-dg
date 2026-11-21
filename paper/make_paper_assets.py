"""
Generates every number, table and figure used by paper/main.tex from saved result files.

Inputs (all produced by the experiment pipeline):
  docs/outputs/tables/*.csv, docs/outputs/tables/summary.json   (analyze.py on the 96 final runs)
  docs/outputs/figures/*.png                                     (analyze.py)
  docs/diagnostics/gate_table.csv                                (diagnose.py)
  cache/features_meta.json, cache/manifest.csv                   (prepare_data.py)
  results_diag/cache/feat_*.npy, results_diag/diag_trials.csv    (diagnose.py; local only, for Fig. 2)
Outputs:
  paper/numbers.tex          LaTeX macros holding every number quoted in the text
  paper/tables/*.tex         table bodies
  paper/figures/*            figures

  python paper/make_paper_assets.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "outputs"
PAPER = ROOT / "paper"
TAB, FIG = PAPER / "tables", PAPER / "figures"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5", "B1", "B2", "B3", "B4"]


def pm_split(s):
    """'61.54 ± 1.47' -> ('61.54', '1.47')."""
    a, b = str(s).split("±")
    return a.strip(), b.strip()


def numbers():
    S = json.load(open(OUT / "tables" / "summary.json"))
    fm = json.load(open(ROOT / "cache" / "features_meta.json"))
    man = pd.read_csv(ROOT / "cache" / "manifest.csv", keep_default_na=False)
    gate = pd.read_csv(ROOT / "docs" / "diagnostics" / "gate_table.csv")
    boot = pd.read_csv(OUT / "tables" / "T4_bootstrap.csv")
    cost = pd.read_csv(OUT / "tables" / "T5_cost.csv")
    pc = pd.read_csv(OUT / "tables" / "T9_per_class_f1.csv")
    sel = pd.read_csv(OUT / "tables" / "T2b_best_baseline_selection.csv")
    lam = pd.read_csv(OUT / "tables" / "T7_lambda_sensitivity.csv")
    splits = pd.read_csv(OUT / "tables" / "T1a_splits.csv")
    m = {}

    def put(name, value):
        assert name.isalpha(), name
        m[name] = value

    letters = {"C0": "Czero", "C1": "Cone", "C2": "Ctwo", "C3": "Cthree", "C4": "Cfour", "C5": "Cfive",
               "B1": "Bone", "B2": "Btwo", "B3": "Bthree", "B4": "Bfour"}
    keys = {"target_accuracy": "Acc", "target_macro_f1": "Fone", "target_clap_slide_error": "CS", "target_worst_f1": "Worst",
            "id_test_accuracy": "IDAcc", "id_test_macro_f1": "IDFone", "gap_macro_f1": "Gap", "val_macro_f1": "Val",
            "train_acc_best": "TrainAcc"}
    for c, L in letters.items():
        for k, K in keys.items():
            put(f"{L}{K}", f"{S['core'][c][k] * 100:.2f}")
        put(f"{L}Params", f"{S['n_params'][c]:,}".replace(",", "{,}"))
    for c in ("C1", "C5", "B1", "B3"):
        for t, T in (("O1", "Oone"), ("O3", "Othree"), ("O5", "Ofive")):
            put(f"{letters[c]}{T}Acc", f"{S['per_task'][c][t]['target_accuracy'] * 100:.1f}")
            put(f"{letters[c]}{T}Fone", f"{S['per_task'][c][t]['target_macro_f1'] * 100:.2f}")
            put(f"{letters[c]}{T}CS", f"{S['per_task'][c][t]['target_clap_slide_error'] * 100:.2f}")
    # paired bootstrap, core mean
    cmap = {"C5 − C1": "FiveOne", "C5 − B1": "FiveBest", "C5 − C2": "FiveTwo", "C5 − C3": "FiveThree", "C5 − C4": "FiveFour",
            "C1 − C0": "OneZero"}
    core = boot[boot.Task == "Core mean"]
    for _, r in core.iterrows():
        if r["Contrast"] not in cmap:
            continue
        K = cmap[r["Contrast"]]
        for col, short in (("Accuracy", "Acc"), ("Macro-F1", "Fone"), ("Clap/slide error", "CS")):
            d = float(r[f"Δ {col} (pp)"])
            lo, hi = [float(x) for x in r[f"95% CI {col} (pp)"].strip("[]").split(",")]
            put(f"D{K}{short}", f"{d:+.2f}".replace("+", "{+}").replace("-", "$-$"))
            put(f"D{K}{short}Lo", f"{lo:+.2f}".replace("+", "{+}").replace("-", "$-$"))
            put(f"D{K}{short}Hi", f"{hi:+.2f}".replace("+", "{+}").replace("-", "$-$"))
    rel = (S["core"]["C1"]["target_clap_slide_error"] - S["core"]["C5"]["target_clap_slide_error"]) / S["core"]["C1"]["target_clap_slide_error"]
    put("CSRelReduction", f"{rel * 100:.1f}")
    # gate
    fused = gate[gate.channels.str.contains(r"\+")]
    lk = fused[fused.hampel_mode == "ratio"].iloc[0]
    p2 = fused[fused.hampel_mode == "antenna_complex"].iloc[0]
    put("GateLocked", f"{lk.heldout_acc * 100:.1f}"); put("GateLockedTrain", f"{lk.train_acc * 100:.1f}")
    put("GateOrig", f"{p2.heldout_acc * 100:.1f}"); put("GateOrigTrain", f"{p2.train_acc * 100:.1f}")
    put("GateEpochs", str(int(lk.epochs)))
    single = gate[~gate.channels.str.contains(r"\+")]
    g = lambda h, ch: f"{single[(single.hampel_mode == h) & (single.channels == ch)].heldout_acc.iloc[0] * 100:.1f}"
    put("GateUnwrapAnt", g("antenna_complex", "phase:unwrap")); put("GateUnwrapRatio", g("ratio", "phase:unwrap"))
    put("GateDiffRatio", g("ratio", "phase:diff")); put("GateDopOrigAnt", g("antenna_complex", "doppler:mean_avg_log1p"))
    put("GateDopOrigRatio", g("ratio", "doppler:mean_avg_log1p")); put("GateDopLockedRatio", g("ratio", "doppler:bp_pca_log1p"))
    put("GateNtrain", str(int(gate.n_train.iloc[0]))); put("GateNheld", str(int(gate.n_heldout.iloc[0])))
    put("GateNfits", str(len(gate)))
    # data
    put("NTrials", f"{len(man):,}".replace(",", "{,}")); put("NValid", f"{int(man.valid.astype(str).str.lower().eq('true').sum()):,}".replace(",", "{,}"))
    put("NExcluded", str(int((~man.valid.astype(str).str.lower().eq("true")).sum())))
    put("NTrain", str(int(splits.Train.min()))); put("NTrainMax", str(int(splits.Train.max())))
    put("NIDTest", str(int(splits["In-domain test"].iloc[0])))
    put("NVal", f"{int(splits['Source val'].min()):,}".replace(",", "{,}"))
    put("NTarget", f"{int(splits['Target'].min()):,}".replace(",", "{,}"))
    put("FeatMs", f"{fm['decode_plus_feature_ms_median']:.0f}")
    put("ShaShort", fm["sha256"][:12])
    # cost and environment
    c = cost.set_index("Config")
    for k in ("C1", "C5", "B1"):
        put(f"{letters[k]}Minutes", pm_split(c.loc[k, "Train time per run (min)"])[0])
        put(f"{letters[k]}Latency", f"{c.loc[k, 'Batch-1 latency median (ms)']:.2f}")
    put("TrainHours", f"{S['total_train_hours']:.2f}"); put("TorchVersion", S["torch"][0].replace("+", "{+}"))
    put("NRuns", str(S["n_runs"])); put("NBoot", f"{S['n_boot']:,}".replace(",", "{,}"))
    # per class
    f = lambda cfg, g_: pm_split(pc[pc.Config == cfg][g_].iloc[0])[0]
    for cfg in ("C1", "C5"):
        put(f"{letters[cfg]}SlideFone", f(cfg, "Slide")); put(f"{letters[cfg]}ClapFone", f(cfg, "Clap"))
        put(f"{letters[cfg]}ZigzagFone", f(cfg, "Draw-Zigzag(H)")); put(f"{letters[cfg]}PushFone", f(cfg, "Push&Pull"))
    # baseline selection
    s = sel.set_index("config")["mean_source_val_macro_f1"]
    for b in ("B1", "B2", "B3", "B4"):
        put(f"{letters[b]}SelVal", f"{s[b] * 100:.2f}")
    # lambda
    for _, r in lam.iterrows():
        L = {"0.05": "LamLow", "0.1": "LamMid", "0.3": "LamHigh"}[f"{float(r['λ']):g}"]
        put(f"{L}Fone", f"{float(r['Core macro-F1 (%)']):.2f}"); put(f"{L}CS", f"{float(r['Core clap↔slide error (%)']):.2f}")
    # clap/slide counts O1
    cs = pd.read_csv(OUT / "tables" / "T8_clap_slide.csv")
    for cfg in ("C1", "C5"):
        r = cs[(cs.Task == "O1") & (cs.Config == cfg)].iloc[0]
        a, n = r["Slide→clap / actual slide"].split("/")
        put(f"{letters[cfg]}OoneSlideClap", a); put(f"{letters[cfg]}OoneSlideN", n)
    # O1 slide row of the pooled target confusion matrix (needs the run files of the final queue)
    runs = ROOT / "results" / "colab" / "single_link_dg_results" / "final" / "runs"
    for cfg in ("C1", "C5"):
        files = sorted(runs.glob(f"{cfg}_O1_s4[234]/metrics.json"))
        assert len(files) == 3, f"run files for {cfg} O1 not found under {runs}"
        cm = sum(np.array(json.load(open(f))["best"]["target"]["cm"]) for f in files)
        put(f"{letters[cfg]}OoneSlideCorrect", str(int(cm[3, 3])))
        put(f"{letters[cfg]}OoneSlidePush", str(int(cm[3, 0])))
        assert int(cm[3, 2]) == int(m[f"{letters[cfg]}OoneSlideClap"])
    # Values REPORTED by the base papers (quoted, not reproduced): source table/figure given per line.
    reported = {
        "RepGesFiSixCO": 96.45,   # GesFi, Table 2, 6 antenna pairs, cross-orientation
        "RepGesFiOneID": 97.29,   # GesFi, Table 3, 1 antenna pair, in-domain
        "RepGesFiOneCO": 86.66,   # GesFi, Table 3, 1 antenna pair, cross-orientation
        "RepGesFiThreeOri": 92.60,  # GesFi, Table 7, 3 training orientations
        "RepGesFiOneOri": 66.35,  # GesFi, Table 7, 1 training orientation
        "RepGesFiTwoOri": 82.46,  # GesFi, Table 7, 2 training orientations
        "RepWiSROneCO": 88.80,    # WiSR single pair, as reported in GesFi Table 3
        "RepWiZeOneID": 98.32,    # WiZeCSi, Table 6, 1 link in-domain
        "RepWiZeOneCO": 96.57,    # WiZeCSi, Table 3/6, 1 link cross-orientation
        "RepWiZeAblID": 88.39,    # WiZeCSi, Table 8, 1 link without multi-ratio and link mixing, in-domain
        "RepWiZeAblCRU": 76.67,   # WiZeCSi, Table 8, same, cross-room-user
        "RepKooID": 99.31,        # Koo and Park, Table 6, in-domain
        "RepKooCU": 54.43,        # Koo and Park, Sec. 4.2, cross-user
        "RepKooCD": 56.61,        # Koo and Park, Sec. 4.2, cross-day
        "RepGLSDABaseCO": 89.92,  # GLSDA, Table II, CSI-ratio baseline, cross-orientation
        "RepGLSDACO": 92.80,      # GLSDA, Table II, CSI-ratio, cross-orientation
    }
    for k, v in reported.items():
        put(k, f"{v:.2f}")
    c5co, c5id = S["core"]["C5"]["target_accuracy"] * 100, S["core"]["C5"]["id_test_accuracy"] * 100
    sgn = lambda x: f"{x:+.2f}".replace("+", "{+}").replace("-", "$-$")
    put("DiffGesFiOneCO", sgn(c5co - reported["RepGesFiOneCO"]))
    put("DiffWiZeOneCO", sgn(c5co - reported["RepWiZeOneCO"]))
    put("DiffWiZeAblID", sgn(c5id - reported["RepWiZeAblID"]))
    put("DiffGesFiOneID", sgn(c5id - reported["RepGesFiOneID"]))
    put("RelGesFiOneCO", f"{(c5co - reported['RepGesFiOneCO']) / reported['RepGesFiOneCO'] * 100:+.1f}".replace("-", "$-$"))
    put("RelWiZeOneCO", f"{(c5co - reported['RepWiZeOneCO']) / reported['RepWiZeOneCO'] * 100:+.1f}".replace("-", "$-$"))
    lines = ["% Numerical results quoted in the text."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[numbers] {len(m)} macros")


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def tables():
    TAB.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(OUT / "tables" / "T2_main_results.csv", keep_default_na=False)
    core = main[main.Task == "Core mean"].set_index("Config")
    rows = []
    for c in CONFIGS:
        r = core.loc[c]
        cells = [r["Target acc (%)"], r["Target macro-F1 (%)"], r["Worst-class F1 (%)"], r["Clap↔slide error (%)"],
                 r["In-domain macro-F1 (%)"], r["ID − target gap (pp)"]]
        cells = [" $\\pm$ ".join(pm_split(x)) for x in cells]
        name = {"C1": "C1 (matched)", "C5": "C5 (proposed)"}.get(c, c)
        if c == "B1":
            rows.append("\\midrule")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    (TAB / "main_results.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    per = main[main.Task != "Core mean"]
    rows = []
    for c in ("C1", "C5", "B1", "B3"):
        cells = []
        for t in ("O1", "O3", "O5"):
            r = per[(per.Task == t) & (per.Config == c)].iloc[0]
            cells += [pm_split(r["Target acc (%)"])[0], pm_split(r["Clap↔slide error (%)"])[0]]   # means only (single column)
        rows.append(f"{c} & " + " & ".join(cells) + r" \\")
    (TAB / "per_task.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    boot = pd.read_csv(OUT / "tables" / "T4_bootstrap.csv")
    core = boot[boot.Task == "Core mean"].set_index("Contrast")
    rows = []
    # single-column layout: one row with the differences, a second row with the intervals
    order = [("C5 − C1", "full vs.\\ matched"), ("C5 − C2", "consistency"), ("C5 − C3", "curriculum"),
             ("C5 − C4", "schedule"), ("C1 − C0", "Doppler$^{\\dagger}$"), ("C5 − B1", "vs.\\ ERM")]
    cols = ("Accuracy", "Macro-F1", "Clap/slide error")
    for i, (key, label) in enumerate(order):
        r = core.loc[key]
        d = [f"{float(r[f'Δ {c} (pp)']):+.2f}".replace("-", "$-$") for c in cols]
        ci = [r[f"95% CI {c} (pp)"].replace("-", "$-$") for c in cols]
        rows.append(f"{key.replace('−', '$-$')} & " + " & ".join(d) + r" \\")
        rows.append(f"\\quad {label} & " + " & ".join(f"\\scriptsize {x}" for x in ci) + r" \\")
        if i < len(order) - 1:
            rows.append(r"\addlinespace[2pt]")
    (TAB / "ablation.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    gate = pd.read_csv(ROOT / "docs" / "diagnostics" / "gate_table.csv")
    name = {"phase:unwrap": "phase, unwrapped", "phase:unwrap_hp": "phase, unwrapped + 2\\,Hz HP", "phase:rel": "phase, relative to static",
            "phase:diff": "phase derivative", "doppler:mean_avg_log1p": "Doppler, mean-removed, avg., log",
            "doppler:mean_avg_db": "Doppler, mean-removed, avg., dB", "doppler:bp_avg_log1p": "Doppler, band-pass, avg., log",
            "doppler:bp_avg_db": "Doppler, band-pass, avg., dB", "doppler:bp_pca_log1p": "Doppler, band-pass, PCA, log",
            "doppler:bp_pca_db": "Doppler, band-pass, PCA, dB"}
    single = gate[~gate.channels.str.contains(r"\+")]
    rows = []
    for ch, label in name.items():
        a = single[(single.hampel_mode == "antenna_complex") & (single.channels == ch)].heldout_acc.iloc[0] * 100
        b = single[(single.hampel_mode == "ratio") & (single.channels == ch)].heldout_acc.iloc[0] * 100
        star = lambda v, best: f"\\textbf{{{v:.1f}}}" if best else f"{v:.1f}"
        rows.append(f"{label} & {a:.1f} & {star(b, ch in ('phase:diff', 'doppler:bp_pca_log1p'))} \\\\")
    (TAB / "gate.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cost = pd.read_csv(OUT / "tables" / "T5_cost.csv").set_index("Config")
    rows = []
    for c in ("C0", "C1", "C5", "B1", "B2", "B3", "B4"):
        r = cost.loc[c]
        rows.append(f"{c} & {int(r['Trainable params']):,} & {' $\\pm$ '.join(pm_split(r['Train time per run (min)']))} & "
                    f"{r['Batch-1 latency median (ms)']:.2f} & {r['Batch-1 latency p95 (ms)']:.2f} \\\\".replace(",", "{,}"))
    (TAB / "cost.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # complete tabular environments (header/footer from tables/<name>_template.tex, rows generated above)
    for name in ("gate", "main_results", "per_task", "cost", "ablation"):
        tpl = (TAB / f"{name}_template.tex").read_text(encoding="utf-8")
        rows = (TAB / f"{name}.tex").read_text(encoding="utf-8")
        assert tpl.count("%ROWS%") == 1, name
        (TAB / f"{name}_full.tex").write_text(tpl.replace("%ROWS%\n", rows), encoding="utf-8")
    print("[tables] main_results, per_task, ablation, gate, cost")


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for name in ("F2_macro_f1_bars", "F3_ablation_deltas", "F4_confusion_C1_vs_C5"):
        shutil.copy(OUT / "figures" / f"{name}.png", FIG / f"{name}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                         "mathtext.fontset": "stix"})

    # ---- Fig. 1: architecture
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", fs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", linestyle=ls, shrinkA=0, shrinkB=0))

    grey, light = "#e6e6e6", "#f4f4f4"
    box(0.5, 16, 10.5, 12, "Receiver r1\nantennas 1, 2\nIntel 5300\n30 subcarriers", fc=grey)
    box(13.5, 16, 13, 12, "Timestamp repair\nHampel filter\non $R=H_1/H_2$\nresample, 200 Hz", fc=light)
    arrow(11, 22, 13.5, 22)
    box(29, 28, 14, 11, "Phase map\n(time derivative)\n$\\angle(\\tilde R_t\\,\\tilde R^{*}_{t-1})$\n$64\\times128$", fc=light)
    box(29, 5, 14, 11, "Doppler map\nband-pass 2-60 Hz,\nPCA, STFT, log\n$64\\times128$", fc=light)
    arrow(26.5, 24, 29, 33.5); arrow(26.5, 20, 29, 10.5)
    box(45.5, 28, 11.5, 11, "CNN branch\n3$\\times$(Conv, GN,\nReLU, Pool)\n+ GAP", fc="#ffffff")
    box(45.5, 5, 11.5, 11, "CNN branch\n3$\\times$(Conv, GN,\nReLU, Pool)\n+ GAP", fc="#ffffff")
    arrow(43, 33.5, 45.5, 33.5); arrow(43, 10.5, 45.5, 10.5)
    box(59.5, 16, 11, 12, "Softmax gate\n$w_p\\mathbf{h}_p+w_d\\mathbf{h}_d$\nL2 norm\n$\\mathbf{z}\\in\\mathbb{R}^{64}$", fc=grey)
    arrow(57, 33.5, 59.5, 25); arrow(57, 10.5, 59.5, 19)
    box(73.5, 16, 26, 12, "Cosine prototype classifier\n$p(c\\mid\\mathbf{x})\\propto\\exp(\\cos(\\mathbf{z},\\mathbf{p}_c)/\\tau)$\n"
        "prototypes frozen from\nsource training data", fc=grey)
    arrow(70.5, 22, 73.5, 22)
    box(59.5, 32.5, 40, 10.5, "Training only: episodes of 6 classes $\\times$ 2 source orientations\n$\\times$ (1 support + 2 queries); "
        "$\\mathcal{L}=\\mathrm{CE}+\\lambda\\,\\mathcal{L}_{\\mathrm{cons}}$", fc="#ffffff", fs=6.3)
    box(59.5, 1, 40, 10.5, "Training only: query augmentation curriculum (time-stretch,\ntime mask, Doppler-bin mask, Doppler gain);"
        "\nepochs 1-5 none, 6-15 level 1, 16-30 level 2", fc="#ffffff", fs=6.3)
    arrow(86.5, 32.5, 86.5, 28, ls="--"); arrow(86.5, 11.5, 86.5, 16, ls="--")
    fig.savefig(FIG / "architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- Fig. 2: clap and slide maps, initial vs locked preprocessing (training-partition trials of orientation 2/4)
    import diagnose as dg
    from train import GESTURE_NAMES
    meta = pd.read_csv(ROOT / "results_diag" / "diag_trials.csv")
    X = {h: np.load(ROOT / "results_diag" / "cache" / f"feat_{h}.npy", mmap_mode="r") for h in ("antenna_complex", "ratio")}
    cols = [("antenna_complex", "phase:unwrap", "Phase, initial"), ("ratio", "phase:diff", "Phase, locked"),
            ("antenna_complex", "doppler:mean_avg_log1p", "Doppler, initial"), ("ratio", "doppler:bp_pca_log1p", "Doppler, locked")]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.9))
    for i, g in enumerate((3, 4)):
        idx = meta[(meta.user == 5) & (meta.gesture == g) & (meta.part == "train")].index[0]
        for j, (h, ch, title) in enumerate(cols):
            a = axes[i, j]
            im = a.imshow(np.asarray(X[h][idx, dg.CHANNELS.index(ch)], dtype=np.float32), aspect="auto", origin="lower",
                          cmap="gray_r" if ch.startswith("phase") else "magma")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(title, fontsize=8)
            if j == 0:
                a.set_ylabel(GESTURE_NAMES[g], fontsize=8)
            cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    for a in axes[1]:
        a.set_xlabel("time", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "maps_clap_slide.png", dpi=300)
    plt.close(fig)
    print("[figures] architecture, maps_clap_slide + copies")


if __name__ == "__main__":
    numbers()
    tables()
    figures()
