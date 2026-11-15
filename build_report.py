"""
Builds the Phase-3 report (.docx) from saved result files only.

Inputs: <results>/tables/*.csv, <results>/tables/summary.json, <results>/figures/*.png,
        cache/features_meta.json, docs/diagnostics/{gate_table.csv, maps_before_after.png}.
Every number in the text is read from those files; sentences that interpret a result are chosen
by rules applied to the saved numbers (sign of a delta, whether a bootstrap interval contains zero).

  python build_report.py --results results/final --out docs/Phase3_Report_23I-2621_23I-2508.docx
"""
import argparse
import sys
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

TITLE = "Class-discriminative single-link WiFi domain generalization for CSI gesture recognition"
AUTHORS = "23I-2621, 23I-2508 | Computer Networks research project, FAST-NUCES"


class Report:
    def __init__(self, results: Path, cache: Path, diag: Path):
        self.res, self.diag = results, diag
        self.S = json.load(open(results / "tables" / "summary.json"))
        self.fm = json.load(open(cache / "features_meta.json"))
        self.gate = pd.read_csv(diag / "gate_table.csv")
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = "Calibri", Pt(10)
        for s in self.doc.sections:
            s.left_margin = s.right_margin = Cm(1.9)
            s.top_margin = s.bottom_margin = Cm(1.8)
        self.nfig = self.ntab = 0
        self.deferred = []          # items moved to the appendix to keep the main body short

    # ---- primitives
    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, text, bold=False, italic=False, size=None):
        para = self.doc.add_paragraph()
        run = para.add_run(text)
        run.bold, run.italic = bold, italic
        if size:
            run.font.size = Pt(size)
        para.paragraph_format.space_after = Pt(4)
        return para

    def bullets(self, items):
        for it in items:
            self.doc.add_paragraph(it, style="List Bullet").paragraph_format.space_after = Pt(1)

    def csv(self, name):
        f = self.res / "tables" / f"{name}.csv"
        return pd.read_csv(f, keep_default_na=False) if f.exists() else None

    def table(self, df, caption, font=7.5, widths=None, appendix=False):
        if df is None or len(df) == 0:
            return
        if appendix:
            self.deferred.append(lambda: self.table(df, caption, font, widths))
            return
        self.ntab += 1
        cap = self.p(f"Table {self.ntab}. {caption}", bold=True, size=8.5)
        cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        for j, c in enumerate(df.columns):
            cell = t.rows[0].cells[j]
            cell.text = str(c)
            for r in cell.paragraphs[0].runs:
                r.font.size, r.bold = Pt(font), True
        for row in df.itertuples(index=False):
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = str(v)
                for r in cells[j].paragraphs[0].runs:
                    r.font.size = Pt(font)
        if widths:
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path, caption, width=17.0, appendix=False):
        path = Path(path)
        if not path.exists():
            return
        if appendix:
            self.deferred.append(lambda: self.figure(path, caption, width))
            return
        self.nfig += 1
        self.doc.add_picture(str(path), width=Cm(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.p(f"Figure {self.nfig}. {caption}", italic=True, size=8.5).alignment = WD_ALIGN_PARAGRAPH.CENTER

    def fig(self, name, caption, width=17.0, appendix=False):
        self.figure(self.res / "figures" / f"{name}.png", caption, width, appendix)

    # ---- numbers
    def core(self, cfg, key, scale=100.0):
        v = self.S["core"].get(cfg, {}).get(key)
        return None if v is None else v * scale

    def con(self, name, metric, scope="Core mean"):
        return self.S["contrasts"].get(name, {}).get(scope, {}).get(metric)

    def delta_sentence(self, name, metric, label, higher_is_better=True, scope="Core mean"):
        c = self.con(name, metric, scope)
        if c is None:
            return f"{label}: not available (runs missing)."
        d, lo, hi = c["delta_pp"], c["ci_lo_pp"], c["ci_hi_pp"]
        good = (d > 0) == higher_is_better
        if lo > 0 or hi < 0:
            verdict = ("the exploratory interval excludes zero in the favourable direction" if good
                       else "the exploratory interval excludes zero in the unfavourable direction")
        else:
            verdict = "the exploratory interval contains zero, so the sign of the effect is not established"
        return f"{label}: {d:+.2f} pp (95% interval {lo:+.2f} to {hi:+.2f} pp); {verdict}."

    # ---- sections
    def build(self):
        S, fm = self.S, self.fm
        doc = self.doc
        t = doc.add_heading(TITLE, level=0)
        self.p("Phase 3: Research Implementation and Experimental Evaluation", bold=True, size=12)
        self.p(AUTHORS, size=9)

        gate = self.gate
        fused = gate[gate.channels.str.contains(r"\+")]
        locked_name = f"phase:{fm['preprocessing']['phase_variant']} + doppler:{fm['preprocessing']['doppler_variant']}"
        p2 = fused[(fused.channels == "phase:unwrap + doppler:mean_avg_log1p") & (fused.head_scale == 10)]
        lk = fused[(fused.channels == locked_name) & (fused.hampel_mode == fm["preprocessing"]["hampel_mode"]) & (fused.head_scale == 10)]
        lk_best = lk.sort_values("heldout_acc", ascending=False).head(1)
        p2_best = p2.sort_values("heldout_acc", ascending=False).head(1)
        gate_acc = float(lk_best.heldout_acc.iloc[0]) * 100 if len(lk_best) else None
        p2_acc = float(p2_best.heldout_acc.iloc[0]) * 100 if len(p2_best) else None

        c1f, c5f = self.core("C1", "target_macro_f1"), self.core("C5", "target_macro_f1")
        c1a, c5a = self.core("C1", "target_accuracy"), self.core("C5", "target_accuracy")
        c1e, c5e = self.core("C1", "target_clap_slide_error"), self.core("C5", "target_clap_slide_error")
        c1id, c5id = self.core("C1", "id_test_accuracy"), self.core("C5", "id_test_accuracy")
        met = (c5a > c1a) and (c5f > c1f) and (c5e <= c1e)
        n_runs, n_missing = S["n_runs"], S["n_missing_of_96"]
        scope_txt = ("all 96 planned trainings" if n_missing == 0 else f"{n_runs} of the 96 planned trainings ({n_missing} not run; see Section 9)")

        # ---------------- abstract
        self.h("Abstract", 1)
        self.p(
            f"We implement and evaluate a compact single-link WiFi gesture recogniser under strict domain generalization: one "
            f"transmitter and one receiver of Widar3.0 (Room 1, nine participants, six gestures), trained on three face orientations "
            f"and tested on an unseen one with no target-domain calibration. The model fuses a CSI-ratio phase map and a Doppler map "
            f"from the same link ({S['n_params'].get('C5', 0):,} trainable parameters) and is trained episodically with source-built class "
            f"prototypes, a class-consistency loss and an augmentation curriculum (C5). It is compared with five ablation arms (C0-C4) "
            f"and four external DG baselines (ERM, Deep CORAL, DANN, Mixup) over three leave-one-orientation-out tasks and three seeds; "
            f"this report covers {scope_txt}. "
            f"Averaged over tasks and seeds, the matched fused baseline C1 reaches {c1a:.2f}% target accuracy and {c1f:.2f}% macro-F1, "
            f"and the proposed C5 reaches {c5a:.2f}% and {c5f:.2f}%; the clap/slide pair error is {c1e:.2f}% for C1 and {c5e:.2f}% for C5. "
            f"In-domain accuracy on held-out trials of the training orientations is {c1id:.2f}% (C1) and {c5id:.2f}% (C5). "
            + ("C5 therefore meets the directional criterion declared in Phase 2 (higher accuracy and macro-F1 than C1 with clap/slide error not increased). "
               if met else
               "C5 therefore does not meet every part of the directional criterion declared in Phase 2 (higher accuracy and macro-F1 than C1 with clap/slide error not increased). ")
            + "Bootstrap intervals over the nine participants are reported as exploratory. All numbers in this report are generated from saved run files.")

        # ---------------- 1 implementation + refinements
        self.h("1. Implementation of the proposed methodology and refinements since Phase 2", 1)
        self.p(
            "The implementation follows the twelve-step methodology of Phase 2: receiver r1 with antennas 1 and 2, CSI-ratio phase and Doppler maps "
            "of 64 x 128, two three-block convolutional branches (16/32/64 channels, 3 x 3, GroupNorm(8), ReLU, max-pooling, global average pooling), "
            "a two-weight softmax gate, an L2-normalised 64-dimensional embedding and a cosine prototype classifier with temperature 0.1. "
            "Training is episodic: six classes x two of the three training orientations x (one support + two query trials); prototypes weight the "
            "two orientations equally. The loss is the query cross-entropy (the mean of the clean and augmented cross-entropies when augmentation "
            "is active) plus 0.1 x the class-consistency penalty, i.e. the mean cosine distance of per-orientation class centres to their common "
            "class centre, computed on clean embeddings. AdamW (learning rate 1e-3, weight decay 1e-4), 30 epochs, ceil(N_train/36) episodes per epoch, "
            "seeds 42/43/44. After each epoch the prototype bank is rebuilt from clean training embeddings and the checkpoint with the best "
            "source-validation macro-F1 is kept; epoch 30 is also saved.")
        self.p("No instructor feedback on Phase 2 had been received when this phase was completed, so no feedback-driven change is claimed. "
               "The table lists the methodological refinements made during implementation and the reason for each.")
        pre = fm["preprocessing"]
        ref = pd.DataFrame([
            ["Query augmentation", "Gaussian feature noise, SD 0.01 then 0.02 (training-standardised units)",
             "Label-preserving signal augmentations on queries: time-stretch, time mask, Doppler-bin mask (Doppler map only), Doppler gain. "
             "Level 1 (epochs 6-15): 5%, 8 frames, 4 bins, 1 dB. Level 2 (epochs 16-30): 10%, 16 frames, 8 bins, 2 dB. Warm-up unchanged (epochs 1-5 clean).",
             "Models nuisance variation that occurs in CSI (execution speed, missing packets, narrow-band interference, link gain) instead of generic noise."],
            ["Fixed-level control (C4)", "Constant noise SD 0.0167 after warm-up", "Constant augmentation level (8%, 13 frames, 6 bins, 1.6 dB) after the same warm-up",
             "Epoch-weighted mean of the curriculum levels, the same matching rule as in Phase 2."],
            ["Hampel filter placement", "On the complex antenna streams before the ratio", f"Selected by the gate: '{pre['hampel_mode']}'",
             "Per-antenna phase is random from packet to packet, so outliers are better defined on the CSI ratio; both placements were tested."],
            ["Phase map", "Unwrapped ratio phase minus its per-trial offset (initial design setting)", f"Selected by the gate: '{pre['phase_variant']}'",
             "Phase 2 declared the map definitions as initial design settings; alternatives were compared on source data only (Section 2.3)."],
            ["Doppler map", "Mean-subtracted ratio, STFT power averaged over subcarriers, log(1+p) (initial design setting)",
             f"Selected by the gate: '{pre['doppler_variant']}'", "Same reason; STFT settings (Hann 64, hop 16, 256-point FFT, +/-60 Hz) are unchanged."],
            ["Preprocessing selection", "Fixed in advance", "Diagnostic gate: plain ERM on orientations 2 and 4 only, random 80/20 split",
             "Verifies that the features carry gesture information before any DG experiment; orientations 2 and 4 are never target domains."],
            ["Baselines", "C0-C5 (54 trainings)", "C0-C5 plus ERM, Deep CORAL, DANN and Mixup on the same fused encoder with a linear head; C5 with lambda 0.05 and 0.3",
             "Adds external DG reference points and a sensitivity check; 96 trainings in total."],
            ["Baseline head and batches", "Not specified", "Linear head on the embedding divided by the prototype temperature (0.1); balanced batches of 64 over class x orientation",
             "Gives the linear head the same logit scale as the prototype classifier and every domain enough samples for CORAL/DANN."],
            ["Cohort", "Participant/location intersection of Room 1", "All nine Room-1 participants (5, 10-17), 750 r1 trials each",
             "The full grid is available for every participant."],
            ["Room/user extension task", "Planned extension if the core runs are secure", "Not run in this phase",
             "Only the Room-1 session was prepared; the three core orientation tasks are the predeclared minimum viable study."],
        ], columns=["Aspect", "Phase 2 design", "Phase 3 implementation", "Reason"])
        self.table(ref, "Methodological refinements between the Phase-2 design and the Phase-3 implementation.", font=7, widths=[2.4, 3.6, 6.0, 5.2])

        # ---------------- 2 dataset
        self.h("2. Dataset, preprocessing and experimental setup", 1)
        self.h("2.1 Dataset and audit", 2)
        self.p("Secondary data: Widar3.0 raw CSI, session 20181130 (Room 1, classroom), Intel 5300, one transmit and three receive antennas, 30 subcarriers, "
               "about 1,000 packets/s at 5.825 GHz. Only receiver r1 and gestures 1-6 (push&pull, sweep, clap, slide, draw-O, draw-zigzag) are used; "
               "files of the other five receivers, gestures 7-9, BVP and the dataset's DFS files (which fuse six receivers) are never read. "
               "Archives were verified by per-member CRC-32 during extraction and their SHA-256 digests are stored in cache/data_provenance.json.")
        self.table(self.csv("T1b_data_audit"), "Dataset audit and exclusions (generated from the trial manifest).", font=7.5, widths=[5.5, 11.5])
        self.h("2.2 Splits", 2)
        self.p("Splits are made once at trial level, saved in cache/splits.csv and reused by every run. For each task, 20% of the trials of every "
               "(gesture, training orientation) cell form a locked in-domain test set. Assertions verify that partitions are disjoint by trial and that "
               "the target orientation never appears in training, validation or in-domain data. Clipping limits (0.5th/99.5th percentile) and z-score "
               "statistics are computed per channel on the training partition only.")
        self.table(self.csv("T1a_splits"), "Leave-one-orientation-out tasks and partition sizes.", font=8)
        self.h("2.3 Preprocessing and the diagnostic gate", 2)
        self.p(f"Pipeline: decode r1 (csiread), keep packets with three RF chains, unwrap the 32-bit microsecond timestamps, sort and de-duplicate, "
               f"Hampel filter (window 7, 3 scaled MAD), CSI ratio R = H1/(H2 + eps), linear interpolation to a 1 kHz grid, polyphase resampling to "
               f"{fm['sample_rate_hz']:.0f} Hz. Trials shorter than {fm['min_duration_s']} s or with a packet gap above {fm['max_gap_s'] * 1000:.0f} ms are excluded and logged. "
               f"The phase map (30 subcarriers x time) and the Doppler map (STFT: Hann {fm['stft']['window']}, hop {fm['stft']['hop']}, {fm['stft']['n_fft']}-point FFT, "
               f"+/-{fm['stft']['max_hz']:.0f} Hz) are resized to 64 x 128 and cached as float16. Feature extraction took a median of "
               f"{fm['decode_plus_feature_ms_median']:.0f} ms per trial (decode + both maps, one CPU core).")
        n_tr, n_ho = int(gate.n_train.iloc[0]), int(gate.n_heldout.iloc[0])
        self.p(f"Before any domain-generalization run, each candidate implementation of the Hampel placement, the phase map and the Doppler map was "
               f"scored with a plain ERM classifier (the project's branch encoder with a linear head) on trials from orientations 2 and 4 only "
               f"({n_tr} training and {n_ho} held-out trials, random 80/20 split stratified by participant, gesture and orientation). "
               f"Orientations 2 and 4 are never target domains in O1/O3/O5, so this selection uses no target-domain information; the held-out trials do "
               f"overlap with source-training and source-validation trials of the later tasks, which is permitted for source-side model selection. "
               + (f"With the Phase-2 map definitions the fused model reached {p2_acc:.1f}% held-out accuracy; " if p2_acc is not None else "")
               + (f"with the locked pipeline it reached {gate_acc:.1f}% (chance is 16.7%). " if gate_acc is not None else "")
               + "Table and figure below give train and held-out accuracy of every variant.")
        g = gate.copy()
        gt = pd.DataFrame({"Hampel on": g.hampel_mode, "Input": g.channels, "Head scale": g.head_scale, "Epochs": g.epochs, "LR": g.lr,
                           "Train acc (%)": (100 * g.train_acc).round(1), "Held-out acc (%)": (100 * g.heldout_acc).round(1)})
        self.table(gt[gt.Input.str.contains(r"\+")], "Diagnostic gate, fused two-branch fits (all single-channel fits are in the appendix).", font=7.5)
        self.fig("F0_preprocessing_gate", "Diagnostic gate: held-out (bar) and train (circle) accuracy of every preprocessing variant.", 15.5)
        self.figure(self.diag / "maps_before_after.png", "One training trial per gesture: Phase-2 map definitions next to the locked maps.", 16.5)
        self.fig("F1_example_maps", "Locked phase and Doppler maps from the final cache, one training trial per gesture.", 17, appendix=True)

        # ---------------- 3 baselines + results
        self.h("3. Baseline evaluation and main results", 1)
        self.p("Controlled arms: C0 phase only; C1 phase/Doppler fusion (main matched baseline); C2 = C1 + curriculum; C3 = C1 + class consistency; "
               "C4 = consistency + fixed-level augmentation; C5 = consistency + curriculum (proposed). External DG baselines on the same fused encoder "
               "with a linear head, the same optimiser, epochs and source-validation checkpoint selection: B1 ERM, B2 Deep CORAL, B3 DANN (domains = "
               "training orientations), B4 Mixup (alpha 0.2). Metrics: accuracy, macro-F1, worst-class F1, clap/slide pair error (mean of "
               "P(slide|clap) and P(clap|slide)), in-domain minus target gap, and cost.")
        main = self.csv("T2_main_results")
        if main is not None:
            self.table(main[main.Task == "Core mean"].drop(columns=["Task"]),
                       "Main results, core mean over O1/O3/O5 (tasks weighted equally within each seed; mean ± sample SD over seeds).", font=7.5)
        best = S.get("best_external_baseline")
        if best:
            bf, ba = self.core(best, "target_macro_f1"), self.core(best, "target_accuracy")
            self.p(f"The best external baseline by mean source-validation macro-F1 (target data not used for this choice) is {best}: "
                   f"{ba:.2f}% target accuracy and {bf:.2f}% macro-F1, against {c5a:.2f}% and {c5f:.2f}% for C5 and {c1a:.2f}% and {c1f:.2f}% for C1.")
            self.table(self.csv("T2b_best_baseline_selection"), "Selection of the best external baseline on source validation.", font=8, appendix=True)
        self.table(self.csv("T5_cost"), "Computational cost per configuration (same device and software for all runs).", font=7.5)

        # ---------------- 4 figures
        self.h("4. Graphical results", 1)
        self.fig("F2_macro_f1_bars", "Target macro-F1 per configuration and task (mean ± SD over seeds).")
        self.fig("F4_confusion_C1_vs_C5", "Target confusion matrices of C1 and C5 (row percentages, seeds pooled); clap/slide cells outlined.", 16)
        self.fig("F5_clap_slide_error", "Clap/slide pair error per configuration and task (lower is better).")
        self.fig("F6_generalization_gap", "In-domain versus target macro-F1 per configuration.", 12)
        self.fig("F7_per_class_f1", "Per-class target F1 per configuration.", 12)
        self.fig("F9_training_curves", "Training loss, clean train accuracy and source-validation macro-F1.", appendix=True)
        if main is not None:
            self.table(main[main.Task != "Core mean"], "Results per task (mean ± sample SD over seeds).", font=7, appendix=True)

        # ---------------- 5 literature
        self.h("5. Comparison with existing research", 1)
        self.p("The table places the measured single-link results next to numbers reported by related papers on Widar3.0. The protocols differ in "
               "the number of receivers, the input representation, the split, the model size and whether target data are used, so the published rows "
               "are context and are not ranked against ours. The closest setting is GLSDA's CSI-ratio branch at one receiver location, which still "
               "uses a different split (train on one location/orientation, test on the others) and a large-model semantic teacher.")
        self.table(self.csv("T6_literature_context"), "Literature context (published values as reported by the papers; fairness limits per row).",
                   font=7, widths=[3.0, 2.4, 2.6, 1.8, 1.7, 2.0, 1.5, 4.2])

        # ---------------- 6 ablation
        self.h("6. Ablation studies", 1)
        self.p("Each contrast is paired on task and seed. The paired user-cluster bootstrap resamples the nine participants with replacement "
               f"({S['n_boot']} replicates) and applies the same draw to both models, all tasks and all seeds; with nine participants and three "
               "orientation shifts its 95% percentile intervals are exploratory and do not support population-level significance claims.")
        self.bullets([
            self.delta_sentence("C5-C2", "macro_f1", "Class consistency (C5 − C2), macro-F1"),
            self.delta_sentence("C5-C2", "clap_slide_error", "Class consistency (C5 − C2), clap/slide error", higher_is_better=False),
            self.delta_sentence("C5-C3", "macro_f1", "Curriculum (C5 − C3), macro-F1"),
            self.delta_sentence("C5-C3", "clap_slide_error", "Curriculum (C5 − C3), clap/slide error", higher_is_better=False),
            self.delta_sentence("C5-C4", "macro_f1", "Schedule versus fixed level (C5 − C4), macro-F1"),
            self.delta_sentence("C1-C0", "macro_f1", "Doppler branch (C1 − C0; capacity also doubles), macro-F1"),
        ])
        self.table(self.csv("T3_ablation"), "Paired ablation deltas per task and for the core mean (mean ± SD over seeds).", font=7, appendix=True)
        self.fig("F3_ablation_deltas", "Ablation deltas: core mean with SD over seeds (bars) and individual tasks (circles).", 15.5)
        self.table(self.csv("T4_bootstrap"), "Paired user-cluster bootstrap for every contrast and task; 95% intervals are exploratory.", font=6.5, appendix=True)
        lam = self.csv("T7_lambda_sensitivity")
        if lam is not None and len(lam):
            self.table(lam, "Sensitivity of C5 to the consistency weight (seed 42).", font=8)
            self.fig("F10_lambda_sensitivity", "C5 target macro-F1 and clap/slide error against the consistency weight.", 13)

        # ---------------- 7 discussion
        self.h("7. Results analysis and discussion", 1)
        self.h("7.1 Does the proposed method address the research gap?", 2)
        self.p("The research question asks whether source-trained class consistency, same-link phase/Doppler fusion and an augmentation curriculum "
               "reduce similar-gesture confusion and improve cross-orientation accuracy without target calibration. Against the matched baseline C1 "
               "(core mean, paired on participants, tasks and seeds):")
        self.bullets([self.delta_sentence("C5-C1", "accuracy", "Accuracy (C5 − C1)"),
                      self.delta_sentence("C5-C1", "macro_f1", "Macro-F1 (C5 − C1)"),
                      self.delta_sentence("C5-C1", "clap_slide_error", "Clap/slide error (C5 − C1)", higher_is_better=False)])
        self.p(("The Phase-2 directional criterion (higher accuracy and macro-F1 than C1 with clap/slide error reduced or unchanged) is met on the core mean. "
                if met else
                "The Phase-2 directional criterion (higher accuracy and macro-F1 than C1 with clap/slide error reduced or unchanged) is not met in full on the core mean. ")
               + "Phase 2 stated that the combination is a testable intervention rather than a promised gain and that a negative or uncertain effect is a "
                 "valid result; the per-task rows and the intervals above are the evidence for how far the gap is addressed in this setting.")
        per = S["per_task"]
        if "C1" in per and "C5" in per:
            self.bullets([f"Task {t}: C1 {per['C1'][t]['target_macro_f1'] * 100:.2f}% and C5 {per['C5'][t]['target_macro_f1'] * 100:.2f}% target macro-F1; "
                          f"clap/slide error {per['C1'][t]['target_clap_slide_error'] * 100:.2f}% and {per['C5'][t]['target_clap_slide_error'] * 100:.2f}%."
                          for t in per["C5"] if t in per["C1"]])
        pc, cs = self.csv("T9_per_class_f1"), self.csv("T8_clap_slide")
        if pc is not None and cs is not None and {"C1", "C5"} <= set(pc.Config):
            f = lambda cfg, g: pc[pc.Config == cfg][g].iloc[0]
            rows = cs[cs.Config.isin(["C1", "C5"])]
            self.p("The pair metric needs care. It counts only clap-to-slide and slide-to-clap errors, so it falls whenever slide or clap "
                   "trials are misread as a third gesture instead. Per-class F1 (core mean, %): clap "
                   f"{f('C1', 'Clap')} for C1 and {f('C5', 'Clap')} for C5; slide {f('C1', 'Slide')} for C1 and {f('C5', 'Slide')} for C5. "
                   "Directional counts per task are: " + "; ".join(
                       f"{r.Task} {r.Config}: clap→slide {r['Clap→slide / actual clap']}, slide→clap {r['Slide→clap / actual slide']}"
                       for _, r in rows.iterrows()) + ". "
                   "The confusion matrices (Figure 4 in Section 4) show where the remaining slide errors go. A lower pair error is therefore evidence "
                   "of less clap/slide mixing, not by itself of better recognition of either gesture.")
        if best:
            self.p("Against the best external baseline (" + best + "): " + self.delta_sentence(f"C5-{best}", "macro_f1", "macro-F1 (C5 − " + best + ")")
                   + " " + self.delta_sentence(f"C5-{best}", "clap_slide_error", "Clap/slide error", higher_is_better=False))
        self.h("7.2 Generalization gap and feature ceiling", 2)
        g1, g5 = self.core("C1", "gap_macro_f1"), self.core("C5", "gap_macro_f1")
        self.p(f"The in-domain minus target macro-F1 gap is {g1:+.2f} pp for C1 and {g5:+.2f} pp for C5. In-domain accuracy "
               f"({c1id:.2f}% for C1, {c5id:.2f}% for C5) is the ceiling that this compact single-link model reaches on orientations it has seen, "
               f"and clean training accuracy at the selected checkpoint is {self.core('C1', 'train_acc_best'):.2f}% (C1) and {self.core('C5', 'train_acc_best'):.2f}% (C5). "
               "Target results must be read against that ceiling: the published numbers in Section 5 come from different models, inputs and "
               "splits, and the diagnostic gate shows how strongly accuracy depends on the map definitions and on the amount of optimisation.")
        self.h("7.3 Error and failure analysis", 2)
        self.p(f"The lowest C5 macro-F1 is on task {S['worst_c5_task']}. The table below lists per-class F1; the appendix lists per-participant accuracy, a t-SNE view and the most confident C5 errors "
               "on that task with the C1 prediction for the same trial; softmax confidence is not a calibrated probability, and an association between "
               "duration or packet gaps and an error does not establish its cause. The t-SNE plot is a diagnostic view only.")
        self.table(self.csv("T9_per_class_f1"), "Per-class target F1 (%), core mean.", font=7.5)
        self.table(self.csv("T11_per_user_accuracy"), "Target accuracy (%) per participant.", font=8, appendix=True)
        self.table(self.csv("T10_failure_cases"), "Most confident C5 errors on its worst task.", font=7, appendix=True)
        self.fig("F8_tsne_C1_vs_C5", "t-SNE of frozen target embeddings, C1 and C5 (diagnostic only).", 16, appendix=True)
        self.h("7.4 Strengths and limitations", 2)
        self.bullets([
            "Strengths: strict source-only protocol with saved splits and leakage assertions; one receiver and two antennas only; matched ablation arms that "
            "share data, sampler, classifier, optimiser and seeds; external DG baselines on the same encoder; every reported number regenerated from run files.",
            "Scope: one room, one recording session and nine participants. The three tasks share participants and locations, so the study measures "
            "orientation shift only; it does not establish participant-, room-, device- or dataset-independent generalization.",
            "Statistics: three seeds and nine participants; bootstrap intervals are exploratory. Seeds and trials are not independent domain replications.",
            "Preprocessing selection used held-out source trials from orientations 2 and 4. These are never target domains, but the selected pipeline was "
            "therefore tuned on data that later serve as source training or source validation.",
            "Model capacity: the encoder is deliberately small (under 100k parameters) and ends in global average pooling, which may weaken the use of "
            "where Doppler energy lies on the frequency axis; whether this limits the in-domain ceiling is untested and left for future work.",
            "The C1 − C0 contrast changes capacity as well as the input, so it is supporting evidence about the Doppler branch, not a pure causal test.",
            "The planned room/user extension task was not run in this phase.",
        ])

        # ---------------- 8 reproducibility
        self.h("8. Reproducibility", 1)
        self.p(f"Code: preprocess.py (signal pipeline), prepare_data.py (extraction, audit, cache, splits), diagnose.py (gate), train.py (models, training, "
               f"job queue), analyze.py (tables and figures), build_report.py (this document), single_link_dg.ipynb (Colab orchestrator). "
               f"Feature cache SHA-256: {S['features_sha256']}. Locked preprocessing: {json.dumps(S['preprocessing'])}. "
               f"Training device: {', '.join(S['device'])}; PyTorch {', '.join(S['torch'])}; total training time {S['total_train_hours']:.2f} hours over {n_runs} runs. "
               "Each run stores its settings, curves, selected and final-epoch metrics (metrics.json), per-trial predictions (preds.npz) and checkpoints with "
               "the source prototype bank. Smoke-test and pilot runs are written to separate folders and tagged so that the analysis refuses to mix them "
               "with final runs.")
        if n_missing:
            self.p(f"Runs not completed ({n_missing}): " + "; ".join("/".join(m) for m in S["missing"]) + ".")

        # ---------------- appendix
        self.h("Appendix", 1)
        self.p("Supporting tables and figures referred to in the main text.", italic=True)
        for item in self.deferred:
            item()
        self.table(gt, "Diagnostic gate, all fits.", font=7)
        self.table(self.csv("T8_clap_slide"), "Directional clap/slide errors with denominators (seeds pooled).", font=7)

    def save(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(out))
        print(f"[report] {out} ({self.ntab} tables, {self.nfig} figures)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--diag", default="docs/diagnostics")
    ap.add_argument("--out", default="docs/Phase3_Report_23I-2621_23I-2508.docx")
    a = ap.parse_args()
    r = Report(Path(a.results), Path(a.cache), Path(a.diag))
    r.build()
    r.save(Path(a.out))


if __name__ == "__main__":
    main()
