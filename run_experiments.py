#!/usr/bin/env python3
"""
Adaptive First-Order Gradient Optimization for Continuous Keystroke Behavioral Authentication
Under Concept Drift — Benchmark & Experiments Runner.

Conference: MTUCI International Scientific and Technical Forum "TiVS-2026"
Section 3: Information Security (indexed in RINC / РИНЦ).
Author: Ilsaf Sharibzhanov (Nortero)
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import prepare_concept_drift_dataset
from src.metrics import compute_biometric_metrics, student_t_quantile_95
from src.models import BiometricMLP, clone_model_with_init
from src.optimizers import (
    evaluate_model,
    pretrain_base_model,
    train_adam,
    train_block_coordinate_descent_armijo,
    train_conjugate_gradient_fletcher_reeves,
    train_rmsprop,
    train_steepest_descent_armijo,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data/DSL-StrongPasswordData.csv"
FIG_DIR = ROOT_DIR / "figures"
RESULTS_JSON = ROOT_DIR / "experiment_results.json"


def generate_figures(summary: dict, out_dir: Path, lang: str = "en"):
    """Generates high-resolution publication figures (300 DPI)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
    })

    epochs = np.arange(1, len(summary["RMSprop"]["history"]["val_loss"]) + 1)
    if lang == "ru":
        styles = {
            "RMSprop": {"color": "#d62728", "marker": "o", "ls": "-", "label": "RMSprop (базовый)"},
            "Steepest Descent (Armijo)": {"color": "#1f77b4", "marker": "s", "ls": "--", "label": "Наискорейший спуск (Армихо)"},
            "Block Coordinate Descent": {"color": "#ff7f0e", "marker": "^", "ls": "-.", "label": "Блочный координатный спуск"},
            "Conjugate Gradients (FR)": {"color": "#2ca02c", "marker": "D", "ls": "-", "label": "Сопряжённые градиенты (ФР)"},
        }
    else:
        styles = {
            "RMSprop": {"color": "#d62728", "marker": "o", "ls": "-", "label": "RMSprop (baseline)"},
            "Steepest Descent (Armijo)": {"color": "#1f77b4", "marker": "s", "ls": "--", "label": "Steepest Descent (Armijo)"},
            "Block Coordinate Descent": {"color": "#ff7f0e", "marker": "^", "ls": "-.", "label": "Block Coordinate Descent"},
            "Conjugate Gradients (FR)": {"color": "#2ca02c", "marker": "D", "ls": "-", "label": "Conjugate Gradients (FR)"},
        }

    # Figure 1: Convergence Curves (Val Loss & Val Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    for name, st in styles.items():
        hist = summary[name]["history"]
        ax1.plot(
            epochs,
            hist["val_loss"],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            linewidth=2.0,
            markersize=5,
            label=st["label"],
        )
        ax2.plot(
            epochs,
            np.array(hist["val_accuracy"]) * 100.0,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            linewidth=2.0,
            markersize=5,
            label=st["label"],
        )

    if lang == "ru":
        ax1.set_title("а) Динамика функции потерь на валидации", fontweight="bold")
        ax1.set_xlabel("Эпоха адаптивного дообучения")
        ax1.set_ylabel("Бинарная кросс-энтропия (Val Loss)")
        ax2.set_title("б) Динамика точности верификации на валидации", fontweight="bold")
        ax2.set_xlabel("Эпоха адаптивного дообучения")
        ax2.set_ylabel("Точность верификации (Val Accuracy), %")
    else:
        ax1.set_title("a) Validation Loss Convergence Dynamics", fontweight="bold")
        ax1.set_xlabel("Adaptation Epoch")
        ax1.set_ylabel("Binary Cross-Entropy Loss")
        ax2.set_title("b) Validation Verification Accuracy Dynamics", fontweight="bold")
        ax2.set_xlabel("Adaptation Epoch")
        ax2.set_ylabel("Verification Accuracy, %")

    ax1.set_xticks(range(2, len(epochs) + 1, 2))
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper left", frameon=True)

    ax2.set_xticks(range(2, len(epochs) + 1, 2))
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig1_path = out_dir / "fig1_convergence_loss_acc.png"
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: ROC Curves and Biometric Errors (EER and FAR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    for name, st in styles.items():
        roc = summary[name]["roc"]
        fpr = np.array(roc["fpr"]) * 100.0
        tpr = np.array(roc["tpr"]) * 100.0
        auc_val = summary[name]["roc_auc"]
        eer_val = summary[name]["eer"] * 100.0
        ax1.plot(
            fpr,
            tpr,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=2.2,
            label=f"{st['label']} (AUC={auc_val:.3f}, EER={eer_val:.2f}%)",
        )

    eer_line_label = "Линия равной ошибки (FAR = FRR)" if lang == "ru" else "Equal Error Line (FAR = FRR)"
    ax1.plot([0, 100], [100, 0], color="gray", linestyle=":", linewidth=1.2, label=eer_line_label)
    ax1.set_xlim(0, 35)
    ax1.set_ylim(65, 100)
    if lang == "ru":
        ax1.set_title("а) ROC-кривые на отложенных сессиях с дрейфом", fontweight="bold")
        ax1.set_xlabel("Вероятность ложного допуска (FAR), %")
        ax1.set_ylabel("Вероятность верного допуска (TAR = 1 − FRR), %")
        method_labels = ["RMSprop", "Наискорейший\nспуск", "Блочный\nкоорд. спуск", "Сопряжённые\nградиенты (ФР)"]
        eer_bar_lbl = "Равная ошибка (EER), %"
        far_bar_lbl = "Ложный допуск (FAR), %"
        ax2.set_title("б) Профильные биометрические ошибки на тесте", fontweight="bold")
        ax2.set_ylabel("Уровень ошибки, %")
    else:
        ax1.set_title("a) ROC Curves on Delayed Concept Drift Sessions", fontweight="bold")
        ax1.set_xlabel("False Acceptance Rate (FAR), %")
        ax1.set_ylabel("True Acceptance Rate (TAR = 1 - FRR), %")
        method_labels = ["RMSprop", "Steepest\nDescent", "Block\nCoord.", "Conjugate\nGrad (FR)"]
        eer_bar_lbl = "Equal Error Rate (EER), %"
        far_bar_lbl = "FAR @ Threshold 0.5, %"
        ax2.set_title("b) Biometric Verification Errors on Test Split", fontweight="bold")
        ax2.set_ylabel("Error Rate, %")

    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.8)

    keys = list(styles.keys())
    eer_vals = [summary[k]["eer"] * 100.0 for k in keys]
    far_vals = [summary[k]["far_05"] * 100.0 for k in keys]

    x_pos = np.arange(len(keys))
    width = 0.36
    b1 = ax2.bar(x_pos - width / 2, eer_vals, width, color="#2b5c8f", label=eer_bar_lbl)
    b2 = ax2.bar(x_pos + width / 2, far_vals, width, color="#d95f02", label=far_bar_lbl)

    for bar in b1:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_labels)
    ax2.set_ylim(0, max(eer_vals) * 1.25)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig2_path = out_dir / "fig2_roc_and_biometric_errors.png"
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Figures ({lang}) successfully saved to: {out_dir}")


def run_single_seed(
    df: pd.DataFrame,
    seed: int,
    epochs: int,
    batch_size: int,
    l2_reg: float = 0.009,
    include_adam: bool = False,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    X_train, y_train, X_val, y_val, X_test, y_test, meta = prepare_concept_drift_dataset(
        df, seed=seed, return_subjects=True
    )
    X_pre, y_pre = meta["pretrain_xy"]
    sess_splits = meta["sessions"]

    X_pre_t = torch.tensor(X_pre)
    y_pre_t = torch.tensor(y_pre)
    X_tr_t = torch.tensor(X_train)
    y_tr_t = torch.tensor(y_train)
    X_vl_t = torch.tensor(X_val)
    y_vl_t = torch.tensor(y_val)
    X_te_t = torch.tensor(X_test)
    y_te_t = torch.tensor(y_test)

    input_dim = X_train.shape[1]
    torch.manual_seed(seed)
    init_model = BiometricMLP(input_dim)
    # Pre-train base model w_0 on initial enrollment sessions (Sessions 1-4, rep <= 20)
    _, pretrain_stats = pretrain_base_model(
        init_model, X_pre_t, y_pre_t, steps=2, lr=0.12, l2_reg=l2_reg, return_metrics=True
    )
    print(
        f"[Seed {seed}] Base model w_0 pre-training on T0: "
        f"Loss {pretrain_stats['pretrain_init_loss']:.4f} -> {pretrain_stats['pretrain_post_loss']:.4f}, "
        f"Acc {pretrain_stats['pretrain_init_acc'] * 100:.2f}% -> {pretrain_stats['pretrain_post_acc'] * 100:.2f}%"
    )
    base_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}

    methods = {
        "RMSprop": lambda m: train_rmsprop(
            m, X_tr_t, y_tr_t, X_vl_t, y_vl_t, epochs=epochs, batch_size=batch_size, lr=0.003, l2_reg=l2_reg, seed=seed
        ),
        "Steepest Descent (Armijo)": lambda m: train_steepest_descent_armijo(
            m, X_tr_t, y_tr_t, X_vl_t, y_vl_t, epochs=epochs, steps_per_epoch=4, initial_alpha=0.58, l2_reg=l2_reg
        ),
        "Block Coordinate Descent": lambda m: train_block_coordinate_descent_armijo(
            m, X_tr_t, y_tr_t, X_vl_t, y_vl_t, epochs=epochs, sweeps_per_epoch=4, initial_alpha=0.58, l2_reg=l2_reg
        ),
        "Conjugate Gradients (FR)": lambda m: train_conjugate_gradient_fletcher_reeves(
            m, X_tr_t, y_tr_t, X_vl_t, y_vl_t, epochs=epochs, steps_per_epoch=4, initial_alpha=0.55, restart_every=4, l2_reg=l2_reg
        ),
    }
    if include_adam:
        methods["Adam"] = lambda m: train_adam(
            m, X_tr_t, y_tr_t, X_vl_t, y_vl_t, epochs=epochs, batch_size=batch_size, lr=0.003, l2_reg=l2_reg, seed=seed
        )

    summary = {}
    for name, train_fn in methods.items():
        model = clone_model_with_init(input_dim, base_state)
        hist = train_fn(model)
        _, _, vl_scores = evaluate_model(model, X_vl_t, y_vl_t)
        te_loss, te_acc, te_scores = evaluate_model(model, X_te_t, y_te_t)

        val_bio = compute_biometric_metrics(y_val, vl_scores, subjects=meta["val"], sessions=sess_splits["val"])
        bio = compute_biometric_metrics(
            y_test,
            te_scores,
            subjects=meta["test"],
            sessions=sess_splits["test"],
            fixed_threshold_frr_10=val_bio["threshold_frr_10"],
        )
        avg_epoch_ms = float(np.mean(hist["epoch_time_ms"]))

        all_session_eers = {**val_bio["session_eers"], **bio["session_eers"]}

        summary[name] = {
            "train_loss": hist["loss"][-1],
            "train_acc": hist["accuracy"][-1],
            "val_loss": hist["val_loss"][-1],
            "val_acc": hist["val_accuracy"][-1],
            "test_loss": te_loss,
            "test_acc": te_acc,
            "eer": bio["eer"],
            "user_eer_mean": bio["user_eer_mean"],
            "user_eer_std": bio["user_eer_std"],
            "user_eer_ci95": bio["user_eer_ci95"],
            "far_05": bio["far_05"],
            "frr_05": bio["frr_05"],
            "far_at_frr_10": bio["far_at_frr_10"],
            "val_calibrated_threshold_frr_10": val_bio["threshold_frr_10"],
            "far_at_val_frr_10": bio["far_at_val_frr_10"],
            "frr_at_val_frr_10": bio["frr_at_val_frr_10"],
            "roc_auc": bio["roc_auc"],
            "session_eers": all_session_eers,
            "avg_epoch_ms": avg_epoch_ms,
            "total_grad_evals": hist["grad_evals"][-1] if hist.get("grad_evals") else 0,
            "grad_eval_type": (
                "block-wise (3 layers/sweep)"
                if "Block Coordinate" in name
                else ("mini-batch (B=1530)" if name in ("RMSprop", "Adam") else "full-batch (N=6120)")
            ),
            "history": hist,
            "roc": {
                "fpr": bio["fpr"],
                "tpr": bio["tpr"],
                "fnr": bio["fnr"],
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Keystroke Drift Optimization Benchmark")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to DSL-StrongPasswordData.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=20, help="Number of adaptation epochs")
    parser.add_argument("--batch-size", type=int, default=1530, help="Batch size for stochastic baselines (1530 = 4 updates/epoch)")
    parser.add_argument("--l2-reg", type=float, default=0.009, help="Uniform L2 regularization coefficient")
    parser.add_argument("--include-adam", action="store_true", help="Include Adam optimizer in benchmark table")
    parser.add_argument("--multi-seed", action="store_true", help="Evaluate and persist mean +/- std across 5 seeds (42..46)")
    args = parser.parse_args()

    print(f"[*] Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)

    summary = run_single_seed(
        df,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2_reg=args.l2_reg,
        include_adam=args.include_adam,
    )

    print("\n" + "=" * 126)
    print(
        f"{'Method':28s} | {'TrLoss':6s} {'TrAcc,%':7s} | {'ValLoss':7s} {'ValAcc,%':8s} | "
        f"{'TeLoss':6s} {'TeAcc,%':7s} | {'EER,%':5s} {'UsrEER':6s} {'FAR,%':5s} {'FAR@Val10':9s} {'AUC':6s} | {'GradEv':6s} {'Time,ms':7s}"
    )
    print("=" * 126)

    for name, res in summary.items():
        print(
            f"{name:28s} | {res['train_loss']:.4f} {res['train_acc']*100:6.2f}% | "
            f"{res['val_loss']:.4f} {res['val_acc']*100:7.2f}% | "
            f"{res['test_loss']:.4f} {res['test_acc']*100:6.2f}% | "
            f"{res['eer']*100:5.2f}% {res['user_eer_mean']*100:5.2f}% {res['far_05']*100:5.2f}% "
            f"{res['far_at_val_frr_10']*100:8.2f}% {res['roc_auc']:.4f} | {res['total_grad_evals']:6d} {res['avg_epoch_ms']:7.1f}"
        )
    print("=" * 126 + "\n")

    print("Chronological Per-Session EER Progression (Sessions 5 -> 6 -> 7 -> 8):")
    for name, res in summary.items():
        se = res["session_eers"]
        print(
            f"  {name:28s} | S5: {se.get('session_5', 0)*100:5.2f}% | S6: {se.get('session_6', 0)*100:5.2f}% | "
            f"S7: {se.get('session_7', 0)*100:5.2f}% | S8: {se.get('session_8', 0)*100:5.2f}%"
        )
    print()

    output_payload = dict(summary)
    output_payload["experiment_config"] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "uniform_l2_reg": args.l2_reg,
        "protocol": (
            "Two-tier Causal Protocol: Within-session split (S1-4, rep<=20 pretrain w_0, "
            "21<=rep<=35 adapt, single scaler_pre) -> Forward Val (S5-6) -> Delayed Test (S7-8)"
        ),
        "scaler": "Fixed enrollment scaler_pre fit strictly on pretrain (rep<=20)",
        "model_architecture": "BiometricMLP (338 -> 16 [ReLU] -> 16 [ReLU] -> 1 [Logits], 5,713 params)",
    }

    if args.multi_seed:
        seeds = [42, 43, 44, 45, 46]
        multi_runs_by_seed = {}
        multi_res = {k: [] for k in summary.keys()}
        for s in seeds:
            s_sum = (
                summary
                if s == args.seed
                else run_single_seed(
                    df,
                    seed=s,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    l2_reg=args.l2_reg,
                    include_adam=args.include_adam,
                )
            )
            multi_runs_by_seed[f"seed_{s}"] = {
                k: {
                    m_key: s_sum[k][m_key]
                    for m_key in [
                        "train_loss", "train_acc", "val_loss", "val_acc",
                        "test_loss", "test_acc", "eer", "user_eer_mean", "user_eer_ci95",
                        "far_05", "far_at_frr_10", "far_at_val_frr_10", "frr_at_val_frr_10",
                        "roc_auc", "session_eers", "total_grad_evals", "avg_epoch_ms",
                        "pretrain_stats",
                    ]
                    if m_key in s_sum[k]
                }
                for k in s_sum
            }
            for k in multi_res:
                multi_res[k].append(s_sum[k])

        aggregate_stats = {}
        n_s = len(seeds)
        t_crit_seeds = student_t_quantile_95(n_s - 1)  # t_{0.975, 4} = 2.776445
        ci_factor = t_crit_seeds / np.sqrt(n_s)

        print(f"Multi-Seed Statistical Summary (Seeds 42..46, Student-t df={n_s-1} 95% CI, t={t_crit_seeds:.4f}):")
        for k, runs in multi_res.items():
            eers = np.array([r["eer"] * 100.0 for r in runs])
            ueers = np.array([r["user_eer_mean"] * 100.0 for r in runs])
            fars = np.array([r["far_05"] * 100.0 for r in runs])
            far10s = np.array([r["far_at_frr_10"] * 100.0 for r in runs])
            far_val10s = np.array([r["far_at_val_frr_10"] * 100.0 for r in runs])
            aucs = np.array([r["roc_auc"] for r in runs])

            aggregate_stats[k] = {
                "eer_mean_pct": float(np.mean(eers)),
                "eer_std_pct": float(np.std(eers, ddof=1)),
                "eer_ci95_pct": float(ci_factor * np.std(eers, ddof=1)),
                "user_eer_mean_pct": float(np.mean(ueers)),
                "user_eer_std_pct": float(np.std(ueers, ddof=1)),
                "user_eer_ci95_pct": float(ci_factor * np.std(ueers, ddof=1)),
                "far_05_mean_pct": float(np.mean(fars)),
                "far_05_std_pct": float(np.std(fars, ddof=1)),
                "far_at_frr_10_mean_pct": float(np.mean(far10s)),
                "far_at_frr_10_std_pct": float(np.std(far10s)),
                "far_at_val_frr_10_mean_pct": float(np.mean(far_val10s)),
                "far_at_val_frr_10_std_pct": float(np.std(far_val10s)),
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=1)),
                "roc_auc_ci95": float(ci_factor * np.std(aucs, ddof=1)),
                "student_t_df": int(n_s - 1),
                "student_t_crit_95": float(t_crit_seeds),
            }
            st = aggregate_stats[k]
            print(
                f"  {k:28s} | EER: {st['eer_mean_pct']:.2f} +/- {st['eer_std_pct']:.2f}% "
                f"(95% CI: +/-{st['eer_ci95_pct']:.2f}%) | "
                f"UserEER: {st['user_eer_mean_pct']:.2f} +/- {st['user_eer_std_pct']:.2f}% | "
                f"FAR: {st['far_05_mean_pct']:.2f} +/- {st['far_05_std_pct']:.2f}% | "
                f"AUC: {st['roc_auc_mean']:.4f} +/- {st['roc_auc_std']:.4f}"
            )
        print()

        # Paired seed-by-seed comparison against Fletcher-Reeves Conjugate Gradients
        from scipy import stats
        ref_key = "Conjugate Gradients (FR)"
        paired_comparisons = {}
        if ref_key in multi_res:
            ref_eers = np.array([r["eer"] * 100.0 for r in multi_res[ref_key]])
            ref_ueers = np.array([r["user_eer_mean"] * 100.0 for r in multi_res[ref_key]])
            print("Paired Seed-by-Seed Differences vs. Conjugate Gradients (FR) (d_i = EER_method,i - EER_FR,i):")

            temp_comps = []
            for k, runs in multi_res.items():
                if k == ref_key:
                    continue
                k_eers = np.array([r["eer"] * 100.0 for r in runs])
                k_ueers = np.array([r["user_eer_mean"] * 100.0 for r in runs])
                diff_eer = k_eers - ref_eers
                diff_ueer = k_ueers - ref_ueers
                mean_d = float(np.mean(diff_eer))
                std_d = float(np.std(diff_eer, ddof=1))
                ci_d = float(ci_factor * std_d)
                t_stat = float(mean_d / (std_d / np.sqrt(n_s) + 1e-12))
                p_val = 2.0 * float(stats.t.sf(abs(t_stat), df=n_s - 1))
                temp_comps.append({
                    "method": k,
                    "diff_eer": diff_eer,
                    "mean_d": mean_d,
                    "std_d": std_d,
                    "ci_d": ci_d,
                    "mean_ueer": float(np.mean(diff_ueer)),
                    "std_ueer": float(np.std(diff_ueer, ddof=1)),
                    "t_stat": t_stat,
                    "p_val": p_val,
                })

            # Holm-Bonferroni step-down correction
            m_tests = len(temp_comps)
            sorted_indices = np.argsort([c["p_val"] for c in temp_comps])
            running_max = 0.0
            holm_p_vals = [0.0] * m_tests
            for rank, idx in enumerate(sorted_indices):
                mult = m_tests - rank
                p_adj = min(1.0, temp_comps[idx]["p_val"] * mult)
                running_max = max(running_max, p_adj)
                holm_p_vals[idx] = min(1.0, running_max)

            for idx, c in enumerate(temp_comps):
                c["p_holm"] = holm_p_vals[idx]
                k = c["method"]
                paired_comparisons[f"{k} vs {ref_key}"] = {
                    "paired_eer_diffs_pct": [float(x) for x in c["diff_eer"]],
                    "mean_delta_eer_pct": c["mean_d"],
                    "std_delta_eer_pct": c["std_d"],
                    "ci95_delta_eer_pct": c["ci_d"],
                    "mean_delta_user_eer_pct": c["mean_ueer"],
                    "std_delta_user_eer_pct": c["std_ueer"],
                    "paired_t_stat": c["t_stat"],
                    "p_value_raw": c["p_val"],
                    "p_value_holm": c["p_holm"],
                    "df": n_s - 1,
                }
                print(
                    f"  {k:28s} - FR-CG | dEER: +{c['mean_d']:.2f} +/- {c['std_d']:.2f}% "
                    f"(95% CI: +/-{c['ci_d']:.2f}%, t={c['t_stat']:.2f}, p={c['p_val']:.4f}, p_holm={c['p_holm']:.4f}) | "
                    f"dUserEER: +{c['mean_ueer']:.2f}%"
                )
            print()

        output_payload["multi_seed_runs"] = multi_runs_by_seed
        output_payload["multi_seed_aggregate"] = aggregate_stats
        output_payload["paired_comparisons_vs_fr_cg"] = paired_comparisons

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"[+] Complete results and multi-seed statistics saved to: {RESULTS_JSON}")

    generate_figures(summary, FIG_DIR, lang="en")


if __name__ == "__main__":
    main()
