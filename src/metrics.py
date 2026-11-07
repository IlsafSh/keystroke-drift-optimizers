"""
Biometric error metrics, DET/ROC curve calculation, Student-t confidence intervals,
validation-calibrated operating thresholds, and per-session / per-subject statistical
evaluation for cybersecurity benchmarks.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import auc, roc_curve

try:
    from scipy.stats import t as _student_t
except ImportError:
    _student_t = None

# Exact two-sided 95% Student-t critical values t_{0.975, df} for common degrees of freedom
_T95_LOOKUP = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    9: 2.262157,
    10: 2.228139,
    49: 2.009575,
    50: 2.008559,
}


def student_t_quantile_95(df: int) -> float:
    """
    Returns the exact two-sided 95% Student-t critical value t_{0.975, df}.
    """
    if df <= 0:
        return 0.0
    if _student_t is not None:
        return float(_student_t.ppf(0.975, df=df))
    if df in _T95_LOOKUP:
        return _T95_LOOKUP[df]
    # Peizer-Pratt / Cornish-Fisher high-accuracy approximation for arbitrary df
    z = 1.959964
    return float(z + (z ** 3 + z) / (4.0 * df) + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * df ** 2))


def _compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.absolute(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold, fpr, tpr, fnr, thresholds


def compute_biometric_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    subjects: Optional[np.ndarray] = None,
    sessions: Optional[np.ndarray] = None,
    fixed_threshold_frr_10: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes profile biometric cybersecurity metrics:
    - EER (Equal Error Rate): operating point where FAR == FRR
    - user_eer_mean / user_eer_std / user_eer_ci95: per-subject EER across all 51 subjects
      with Student-t 95% confidence interval (df = N_users - 1 = 50)
    - FAR @ 0.5 and FRR @ 0.5: error rates at fixed threshold 0.5
    - FAR @ FRR=10% (ROC curve point) and threshold_frr_10 (calibrated threshold where FRR <= 10%)
    - far_at_val_frr_10 / frr_at_val_frr_10: independent test evaluation at a validation-calibrated threshold
    - session_eers: EER broken down by chronological session index (when provided)
    - ROC AUC (Area Under ROC Curve) and full ROC/DET curves (FPR, TPR, FNR)
    """
    eer, eer_threshold, fpr, tpr, fnr, thresholds = _compute_eer(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    preds_05 = (y_scores >= 0.5).astype(int)
    neg_mask = y_true == 0
    pos_mask = y_true == 1

    far_05 = float(np.mean(preds_05[neg_mask] == 1)) if np.any(neg_mask) else 0.0
    frr_05 = float(np.mean(preds_05[pos_mask] == 0)) if np.any(pos_mask) else 0.0

    # Operating point where FRR <= 10% (or closest to 10%)
    valid_frr10 = np.where(fnr <= 0.10)[0]
    if len(valid_frr10) > 0:
        frr_10_idx = int(valid_frr10[0])
    else:
        frr_10_idx = int(np.nanargmin(np.abs(fnr - 0.10)))
    far_at_frr_10 = float(fpr[frr_10_idx])
    threshold_frr_10 = float(thresholds[frr_10_idx])

    if fixed_threshold_frr_10 is not None:
        preds_val_thr = (y_scores >= fixed_threshold_frr_10).astype(int)
        far_at_val_frr_10 = float(np.mean(preds_val_thr[neg_mask] == 1)) if np.any(neg_mask) else 0.0
        frr_at_val_frr_10 = float(np.mean(preds_val_thr[pos_mask] == 0)) if np.any(pos_mask) else 0.0
    else:
        far_at_val_frr_10 = far_at_frr_10
        frr_at_val_frr_10 = float(fnr[frr_10_idx])

    per_user_eers = []
    if subjects is not None:
        for s in np.unique(subjects):
            mask = subjects == s
            if len(np.unique(y_true[mask])) < 2:
                continue
            u_eer, _, _, _, _, _ = _compute_eer(y_true[mask], y_scores[mask])
            per_user_eers.append(u_eer)

    if per_user_eers:
        n_u = len(per_user_eers)
        user_eer_mean = float(np.mean(per_user_eers))
        user_eer_std = float(np.std(per_user_eers, ddof=1)) if n_u > 1 else 0.0
        t_crit = student_t_quantile_95(n_u - 1) if n_u > 1 else 0.0
        user_eer_ci95 = float(t_crit * user_eer_std / np.sqrt(n_u))
    else:
        user_eer_mean = eer
        user_eer_std = 0.0
        user_eer_ci95 = 0.0

    session_eers: Dict[str, float] = {}
    if sessions is not None:
        for sess_id in sorted(np.unique(sessions)):
            smask = sessions == sess_id
            if len(np.unique(y_true[smask])) < 2:
                continue
            s_eer, _, _, _, _, _ = _compute_eer(y_true[smask], y_scores[smask])
            session_eers[f"session_{int(sess_id)}"] = s_eer

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "user_eer_mean": user_eer_mean,
        "user_eer_std": user_eer_std,
        "user_eer_ci95": user_eer_ci95,
        "per_user_eers": per_user_eers,
        "far_05": far_05,
        "frr_05": frr_05,
        "far_at_frr_10": far_at_frr_10,
        "threshold_frr_10": threshold_frr_10,
        "far_at_val_frr_10": far_at_val_frr_10,
        "frr_at_val_frr_10": frr_at_val_frr_10,
        "roc_auc": roc_auc,
        "session_eers": session_eers,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "fnr": fnr.tolist(),
    }
