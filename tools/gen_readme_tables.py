"""Regenerates the result tables in README.md directly from experiment_results.json (no manual transcription)."""
import json, re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "experiment_results.json").read_text(encoding="utf-8"))
README = ROOT / "README.md"
ORDER = ["RMSprop", "Adam", "Steepest Descent (Armijo)", "Block Coordinate Descent", "Conjugate Gradients (FR)"]
NAME = {
    "RMSprop": "**RMSprop (baseline)**", "Adam": "**Adam (baseline)**",
    "Steepest Descent (Armijo)": "**Steepest Descent (Armijo)**",
    "Block Coordinate Descent": "**Block Coordinate Descent**",
    "Conjugate Gradients (FR)": "**Conjugate Gradients (FR)**",
}
GE = {"RMSprop": "80 (mini)", "Adam": "80 (mini)", "Steepest Descent (Armijo)": "80 (full)",
      "Block Coordinate Descent": "240 (block)", "Conjugate Gradients (FR)": "80 (full)"}


def bold_best(vals, fmt, best):
    shown = [fmt(v) for v in vals]
    ref = (min if best == "min" else max)(float(x) for x in shown)
    return [f"**{s}**" if float(s) == ref else s for s in shown]


def table(header, aligns, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(aligns) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])


def replace_table(text, first_header_cell, new_tbl):
    pat = re.compile(r"^\| " + re.escape(first_header_cell) + r".*?(?=\n\n|\n\*|\Z)", re.S | re.M)
    assert len(pat.findall(text)) >= 1, first_header_cell
    return pat.sub(lambda m: new_tbl, text, count=1)


text = README.read_text(encoding="utf-8")
runs = R["multi_seed_runs"]; seeds = list(runs)
agg = R["multi_seed_aggregate"]
f2 = lambda x: f"{x:.2f}"

# --- 2.3 computational complexity (5-seed func evals / backtracking; epoch time seed 42) ---
rows = []
regime = {"RMSprop": ("4 mini-batches / epoch", "1,530"), "Adam": ("4 mini-batches / epoch", "1,530"),
          "Steepest Descent (Armijo)": ("4 full-batch steps / epoch", "6,120"),
          "Block Coordinate Descent": ("4 sweeps × 3 blocks / epoch", "6,120"),
          "Conjugate Gradients (FR)": ("4 full-batch steps / epoch", "6,120")}
for k in ORDER:
    fe = np.array([runs[s][k]["total_func_evals"] for s in seeds], float)
    bt = np.array([runs[s][k]["total_backtracking_steps"] for s in seeds], float)
    rows.append([NAME[k], regime[k][0], regime[k][1], GE[k], f"{R[k]['grad_samples_processed']:,}",
                 f"{fe.mean():.1f} ± {fe.std(ddof=1):.1f}", f"{bt.mean():.1f} ± {bt.std(ddof=1):.1f}",
                 f"{R[k]['avg_epoch_ms']:.1f}"])
text = replace_table(text, "Optimization Method | Update Regime".split(" | ")[0] + " | Update Regime",
                     table(["Optimization Method", "Update Regime", "Batch Size", "Grad Evals", "Grad Samples Processed",
                            "Func Evals (mean ± std)", "Backtracking Steps (mean ± std)", "Epoch Time, ms (seed=42)"],
                           [":---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"], rows))

# --- 3.1 single seed (seed 42) ---
cols = [("train_loss", 1, 4, "min"), ("train_acc", 100, 2, "max"), ("val_loss", 1, 4, "min"), ("val_acc", 100, 2, "max"),
        ("test_loss", 1, 4, "min"), ("test_acc", 100, 2, "max"), ("eer", 100, 2, "min")]
colvals = {}
for key, mult, nd, best in cols:
    colvals[key] = bold_best([R[k][key] * mult for k in ORDER], lambda v, nd=nd: f"{v:.{nd}f}", best)
far = bold_best([R[k]["far_at_val_frr_10"] * 100 for k in ORDER], f2, "min")
frr = bold_best([R[k]["frr_at_val_frr_10"] * 100 for k in ORDER], f2, "min")
auc = bold_best([R[k]["roc_auc"] for k in ORDER], lambda v: f"{v:.4f}", "max")
rows = []
for i, k in enumerate(ORDER):
    ue = f"{R[k]['user_eer_mean']*100:.2f} ± {R[k]['user_eer_std']*100:.2f}"
    rows.append([NAME[k], GE[k], str(R[k]["total_func_evals"])] + [colvals[c[0]][i] for c in cols[:6]] +
                [colvals["eer"][i], ue, far[i], frr[i], auc[i], f"{R[k]['avg_epoch_ms']:.1f}"])
text = replace_table(text, "Optimization Method | Grad Evals | Func Evals",
                     table(["Optimization Method", "Grad Evals", "Func Evals", "Train Loss", "Train Acc, %", "Val Loss", "Val Acc, %",
                            "Test Loss", "Test Acc, %", "Pooled EER, %", "Per-User EER, %", "FAR @ Val FRR=10%, %", "FRR @ Val FRR=10%, %",
                            "ROC AUC", "Epoch Time, ms"], [":---"] + [":---:"] * 14, rows))


def agg_table(A, first_cell, with_sess=False, paired=None):
    eer = bold_best([A[k]["eer_mean_pct"] for k in ORDER], f2, "min")
    ue = bold_best([A[k]["user_eer_mean_pct"] for k in ORDER], f2, "min")
    fv = bold_best([A[k]["far_at_val_frr_10_mean_pct"] for k in ORDER], f2, "min")
    rv = bold_best([A[k]["frr_at_val_frr_10_mean_pct"] for k in ORDER], f2, "min")
    au = bold_best([A[k]["roc_auc_mean"] for k in ORDER], lambda v: f"{v:.4f}", "max")
    rows = []
    for i, k in enumerate(ORDER):
        a = A[k]
        def wrap(s, extra=None):
            return s
        cells = [NAME[k]]
        e_txt = eer[i].replace("**", "")
        cells.append(("**%s ± %.2f [±%.2f]**" if "**" in eer[i] else "%s ± %.2f [±%.2f]") % (e_txt, a["eer_std_pct"], a["eer_ci95_pct"]))
        u_txt = ue[i].replace("**", "")
        cells.append(("**%s ± %.2f [±%.2f]**" if "**" in ue[i] else "%s ± %.2f [±%.2f]") % (u_txt, a["user_eer_std_pct"], a["user_eer_ci95_pct"]))
        if not with_sess:
            cells.append(f"{a['far_05_mean_pct']:.2f} ± {a['far_05_std_pct']:.2f}")
        for lst, mk, sk in ((fv, "far_at_val_frr_10_mean_pct", "far_at_val_frr_10_std_pct"), (rv, "frr_at_val_frr_10_mean_pct", "frr_at_val_frr_10_std_pct")):
            t = lst[i].replace("**", ""); s_ = f"{t} ± {a[sk]:.2f}"
            cells.append(f"**{s_}**" if "**" in lst[i] else s_)
        t = au[i].replace("**", ""); s_ = f"{t} ± {a['roc_auc_std']:.4f}"
        cells.append(f"**{s_}**" if "**" in au[i] else s_)
        if with_sess:
            se = a["session_eers_aggregate"]
            cells.append(" -> ".join(f"{se[f'session_{n}']['mean_pct']:.2f}" for n in (5, 6, 7, 8)))
            pk = {"RMSprop": "RMSprop vs Conjugate Gradients (FR)", "Adam": "Adam vs Conjugate Gradients (FR)",
                  "Steepest Descent (Armijo)": "Steepest Descent (Armijo) vs Conjugate Gradients (FR)",
                  "Block Coordinate Descent": "Block Coordinate Descent vs Conjugate Gradients (FR)"}.get(k)
            if pk:
                p = paired[pk]
                cells.append(f"+{p['mean_delta_eer_pct']:.2f} ± {p['std_delta_eer_pct']:.2f} (p = {p['p_value_raw']:.4f}; Holm {p['p_value_holm']:.4f})")
            else:
                cells.append("— (Reference)")
        rows.append(cells)
    return rows


# --- 3.3 within_session 5-seed ---
rows = agg_table(agg, "")
hdr = ["Optimization Method", "Pooled EER, % (mean ± std [95% CI])", "Per-User EER, % (mean ± std [95% CI])", "FAR @ 0.5, % (mean ± std)",
       "FAR @ Val FRR=10%, % (mean ± std)", "FRR @ Val FRR=10%, % (mean ± std)", "ROC AUC (mean ± std)"]
text = replace_table(text, "Optimization Method | Pooled EER, % (mean ± std [95% CI]) | Per-User EER, % (mean ± std [95% CI]) | FAR @ 0.5",
                     table(hdr, [":---"] + [":---:"] * 6, rows))

# --- 3.5 strict ---
sb = R["strict_session_protocol_benchmark"]
rows = agg_table(sb["multi_seed_aggregate"], "", with_sess=True, paired=sb["paired_comparisons_vs_fr_cg"])
hdr = ["Optimization Method", "Pooled EER, % (mean ± std [95% CI])", "Per-User EER, % (mean ± std [95% CI])",
       "FAR @ Val FRR=10%, % (mean ± std)", "FRR @ Val FRR=10%, % (mean ± std)", "ROC AUC (mean ± std)",
       "Session 5 -> 6 -> 7 -> 8 EER, % (mean)", "Paired ΔEER vs. FR-CG, pp (raw p; Holm p)"]
text = replace_table(text, "Optimization Method | Pooled EER, % (mean ± std [95% CI]) | Per-User EER, % (mean ± std [95% CI]) | FAR @ Val FRR=10%",
                     table(hdr, [":---"] + [":---:"] * 7, rows))
README.write_text(text, encoding="utf-8")
print("README tables regenerated")
