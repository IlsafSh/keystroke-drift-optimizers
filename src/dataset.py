"""
Dataset loading, schema validation, and strictly intra-group biometric feature engineering
for the CMU Keystroke Dynamics Benchmark (DSL-StrongPasswordData.csv) under a strictly
causal temporal protocol (Enrollment/Pre-train -> Adaptation -> Validation -> Test).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Canonical physical grouping of the 31 raw timing measurements in CMU Keystroke Dynamics:
# - 11 Hold times (H): key press to key release for each of the 11 keys in '.tie5Roanl' + Return
# - 10 Down-Down latencies (DD): key press i to key press i+1 for the 10 adjacent digraphs
# - 10 Up-Down flight times (UD): key release i to key press i+1 for the 10 adjacent digraphs
HOLD_COLS: List[str] = [
    "H.period", "H.t", "H.i", "H.e", "H.five",
    "H.Shift.r", "H.o", "H.a", "H.n", "H.l", "H.Return",
]
DD_COLS: List[str] = [
    "DD.period.t", "DD.t.i", "DD.i.e", "DD.e.five", "DD.five.Shift.r",
    "DD.Shift.r.o", "DD.o.a", "DD.a.n", "DD.n.l", "DD.l.Return",
]
UD_COLS: List[str] = [
    "UD.period.t", "UD.t.i", "UD.i.e", "UD.e.five", "UD.five.Shift.r",
    "UD.Shift.r.o", "UD.o.a", "UD.a.n", "UD.n.l", "UD.l.Return",
]
CANONICAL_TIMING_COLS: List[str] = HOLD_COLS + DD_COLS + UD_COLS
REQUIRED_META_COLS: List[str] = ["subject", "sessionIndex", "rep"]


def validate_and_sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the input DataFrame contains all required metadata columns
    and all 31 physical timing columns (H, DD, UD), then returns a chronologically
    sorted copy ordered by (subject, sessionIndex, rep).
    """
    missing_meta = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {missing_meta}")

    missing_timing = [c for c in CANONICAL_TIMING_COLS if c not in df.columns]
    if missing_timing:
        raise ValueError(f"Missing required keystroke timing columns: {missing_timing}")

    return df.sort_values(["subject", "sessionIndex", "rep"], kind="mergesort").reset_index(drop=True)


def intra_group_adjacent_products(arr31: np.ndarray) -> np.ndarray:
    """
    Computes strictly intra-group adjacent products within H (11), DD (10), and UD (10)
    without cyclic wrap-around (no np.roll) and without crossing H/DD/UD boundaries.
    """
    H = arr31[:, :11]
    DD = arr31[:, 11:21]
    UD = arr31[:, 21:31]

    h_adj = H[:, :-1] * H[:, 1:]                        # (N, 10)
    h_mean = np.mean(h_adj, axis=1, keepdims=True)      # (N, 1)

    dd_adj = DD[:, :-1] * DD[:, 1:]                     # (N, 9)
    dd_mean = np.mean(dd_adj, axis=1, keepdims=True)    # (N, 1)

    ud_adj = UD[:, :-1] * UD[:, 1:]                     # (N, 9)
    ud_mean = np.mean(ud_adj, axis=1, keepdims=True)    # (N, 1)

    return np.hstack([h_adj, h_mean, dd_adj, dd_mean, ud_adj, ud_mean])


def intra_digraph_couplings(arr31: np.ndarray) -> np.ndarray:
    """
    Computes cross-feature intra-digraph couplings for each of the 10 consecutive key transitions
    i -> i+1 (i = 0..9), capturing joint deviations within the same digraph:
      - C1_i = H_i * UD_{i,i+1}       (10 features)
      - C2_i = H_{i+1} * DD_{i,i+1}   (10 features)
      - C3_i = DD_{i,i+1} * UD_{i,i+1} (10 features)
      - Mean coupling intensity       (1 feature)
    Total: 31 features, strictly aligned by digraph index i without cyclic shifts.
    """
    H = arr31[:, :11]
    DD = arr31[:, 11:21]
    UD = arr31[:, 21:31]

    c1 = H[:, :-1] * UD                                 # (N, 10)
    c2 = H[:, 1:] * DD                                  # (N, 10)
    c3 = DD * UD                                        # (N, 10)
    c_mean = np.mean(c3, axis=1, keepdims=True)         # (N, 1)

    return np.hstack([c1, c2, c3, c_mean])


def intra_group_first_differences(arr31: np.ndarray) -> np.ndarray:
    """
    Computes first-order finite differences strictly within each homogeneous physical group:
      - dH_i = H_{i+1} - H_i (10 features) + mean |dH| (1 feature)
      - dDD_i = DD_{i+1} - DD_i (9 features) + mean |dDD| (1 feature)
      - dUD_i = UD_{i+1} - UD_i (9 features) + mean |dUD| (1 feature)
    Total: 31 features, with zero subtraction across H/DD/UD boundaries.
    """
    H = arr31[:, :11]
    DD = arr31[:, 11:21]
    UD = arr31[:, 21:31]

    dh = np.diff(H, axis=1)                             # (N, 10)
    dh_norm = np.mean(np.abs(dh), axis=1, keepdims=True)

    ddd = np.diff(DD, axis=1)                           # (N, 9)
    ddd_norm = np.mean(np.abs(ddd), axis=1, keepdims=True)

    dud = np.diff(UD, axis=1)                           # (N, 9)
    dud_norm = np.mean(np.abs(dud), axis=1, keepdims=True)

    return np.hstack([dh, dh_norm, ddd, ddd_norm, dud, dud_norm])


def sample_stratified_impostors(
    pool_df: pd.DataFrame,
    subject: str,
    target_df: pd.DataFrame,
    base_seed: int,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Samples negative (impostor) observations from other subjects (subject != `subject`)
    with exact 1-to-1 matching for each (sessionIndex, rep) pair present in `target_df`.
    Guarantees that both the marginal and joint 2D distributions of (sessionIndex, rep)
    are 100% identical between genuine and impostor classes.

    If `strict=True` (default), raises a ValueError if any required (sessionIndex, rep)
    combination is missing from `pool_df` for other subjects.
    """
    pool_sub = pool_df[pool_df["subject"] != subject]
    idx_dict = pool_sub.groupby(["sessionIndex", "rep"]).groups

    sampled_indices = []
    for (sess_id, rep_val), grp in target_df.groupby(["sessionIndex", "rep"], sort=False):
        k = len(grp)
        pool_idx = idx_dict.get((sess_id, rep_val))
        if pool_idx is None or len(pool_idx) == 0:
            if strict:
                raise ValueError(
                    f"Strict stratification violated: no impostor candidates found for "
                    f"subject != {subject} at (sessionIndex={sess_id}, rep={rep_val})."
                )
            pool_idx = pool_sub[pool_sub["sessionIndex"] == sess_id].index.values
            if len(pool_idx) == 0:
                raise ValueError(f"No impostor candidates found in sessionIndex={sess_id}.")
        rng = np.random.default_rng(int((base_seed + int(sess_id) * 1000 + int(rep_val) * 7) % (2**31 - 1)))
        chosen = rng.choice(pool_idx, size=k, replace=len(pool_idx) < k)
        sampled_indices.extend(chosen)

    return pool_sub.loc[sampled_indices]


def prepare_concept_drift_dataset(
    csv_path_or_df: Union[str, Path, pd.DataFrame],
    seed: int = 42,
    return_subjects: bool = False,
    protocol: str = "within_session",
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
]:
    """
    Constructs the binary verification dataset from the CMU Keystroke Dynamics Benchmark
    (51 subjects, 8 sessions per subject, 50 repetitions of '.tie5Roanl' per session)
    supporting two temporal evaluation protocols:

    Protocol A (`protocol="within_session"`, default — Within-Session Enrollment/Adaptation Split
    with Forward Session Evaluation):
      1. Initial Enrollment & Pre-training Split (T0: Sessions 1–4, rep <= 20, N_pretrain = 8,160):
         - Individual biometric template (mu_u, sigma_u), base model w_0 pre-training pool,
           and single feature standardizer `scaler_pre` are fit strictly on T0.
      2. Adaptation / Training Split (T1: Sessions 1–4, 21 <= rep <= 35, N_train = 6,120).
      3. Validation Split (T2: Sessions 5–6, 21 <= rep <= 50, N_val = 6,120).
      4. Test Split (T3: Sessions 7–8, 21 <= rep <= 50, N_test = 6,120).

    Protocol B (`protocol="strict_session"` — Strict Session-Sequential Causal Protocol, T0 < T1 < T2 < T3):
      1. Enrollment & Pre-training Split (T0: Sessions 1–2, rep <= 40, N_pretrain = 8,160):
         - Individual biometric template (mu_u, sigma_u), base model w_0 pre-training pool,
           and single feature standardizer `scaler_pre` are fit exclusively on Sessions 1–2
           (zero observations from Sessions 3–8 used during enrollment or pre-training).
      2. Adaptation / Training Split (T1: Sessions 3–4, 21 <= rep <= 50, N_train = 6,120).
      3. Validation Split (T2: Sessions 5–6, 21 <= rep <= 50, N_val = 6,120).
      4. Test Split (T3: Sessions 7–8, 21 <= rep <= 50, N_test = 6,120).

    Feature Vector (D = 338 Subject-Conditioned Enrollment-Normalized Biometric Dimensions, ordered as [11 H, 10 DD, 10 UD]):
    - 31 relative rhythmic timings normalized by total inter-key press span T_DD = sum(DD): r_k = x_k / T_DD
    - 31 unsigned rhythm z-deviations from enrollment template: |r_k - mu_r| / sigma_r
    - 31 unsigned raw timing z-deviations from enrollment template: |x_k - mu_x| / sigma_x
    - 31 rhythm-dominant combined deviations: z_comb = 0.20 * z_raw_abs + 0.80 * z_rhy_abs
    - 31 truncated quadratic rhythm deviations: min(z_rhy_abs^2, 16.0)
    - 31 truncated quadratic combined deviations: min(z_comb^2, 16.0)
    - 31 intra-group adjacent rhythm products (within H, DD, and UD separately, no np.roll)
    - 31 cross-feature intra-digraph deviation couplings on z_comb (H_i*UD_i, H_{i+1}*DD_i, DD_i*UD_i)
    - 31 signed temporal drift deviations from initial session baseline: (x_k - mu_x^(s1)) / sigma_x
    - 31 intra-group first-order finite differences (within H, DD, and UD separately)
    - 11 relative key hold-time deviations: |H_i / mean(H) - mu_hrel| / sigma_hrel
    - 10 key-rollover overlap indicators: I(UD_{i,i+1} < 0)
    - 7 statistical summary norms (L1, L2, and 75th-percentile deviations)
    Total: 31 * 10 + 11 + 10 + 7 = 338 deterministic biometric features.
    """
    if protocol not in ("within_session", "strict_session"):
        raise ValueError(f"Unsupported protocol '{protocol}'. Choose 'within_session' or 'strict_session'.")

    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df

    df = validate_and_sort_dataframe(df)
    rng = np.random.default_rng(seed)

    subjects = sorted(df["subject"].unique())
    subj_groups = {s: group for s, group in df.groupby("subject", sort=False)}

    if protocol == "within_session":
        t0_enroll_pool = df[(df["sessionIndex"].isin([1, 2, 3, 4])) & (df["rep"] <= 20)]
        t1_adapt_pool = df[(df["sessionIndex"].isin([1, 2, 3, 4])) & (df["rep"] > 20) & (df["rep"] <= 35)]
    else:
        t0_enroll_pool = df[(df["sessionIndex"].isin([1, 2])) & (df["rep"] <= 40)]
        t1_adapt_pool = df[(df["sessionIndex"].isin([3, 4])) & (df["rep"] > 20)]

    s56_pool = df[(df["sessionIndex"].isin([5, 6])) & (df["rep"] > 20)]
    s78_pool = df[(df["sessionIndex"].isin([7, 8])) & (df["rep"] > 20)]

    splits: Dict[str, Tuple[list, list, list, list, list]] = {
        k: ([], [], [], [], []) for k in ["pretrain", "train", "val", "test"]
    }

    for idx, subj in enumerate(subjects):
        sdf = subj_groups[subj]

        if protocol == "within_session":
            enroll = sdf[(sdf["sessionIndex"].isin([1, 2, 3, 4])) & (sdf["rep"] <= 20)]
            gen_tr = sdf[(sdf["sessionIndex"].isin([1, 2, 3, 4])) & (sdf["rep"] > 20) & (sdf["rep"] <= 35)]
        else:
            enroll = sdf[(sdf["sessionIndex"].isin([1, 2])) & (sdf["rep"] <= 40)]
            gen_tr = sdf[(sdf["sessionIndex"].isin([3, 4])) & (sdf["rep"] > 20)]

        enroll_s1 = sdf[(sdf["sessionIndex"] == 1) & (sdf["rep"] <= 20)]

        raw_e = enroll[CANONICAL_TIMING_COLS].values
        raw_s1 = enroll_s1[CANONICAL_TIMING_COLS].values
        H_e = raw_e[:, :11]
        DD_e = raw_e[:, 11:21]

        tot_e = np.sum(DD_e, axis=1, keepdims=True) + 1e-4
        rhy_e = raw_e / tot_e
        hrel_e = H_e / (np.mean(H_e, axis=1, keepdims=True) + 1e-4)

        mean_raw_s1 = raw_s1.mean(axis=0)
        mean_raw = raw_e.mean(axis=0)
        std_raw = raw_e.std(axis=0) + 1e-4
        mean_rhy = rhy_e.mean(axis=0)
        std_rhy = rhy_e.std(axis=0) + 1e-4
        mean_hrel = hrel_e.mean(axis=0)
        std_hrel = hrel_e.std(axis=0) + 1e-4

        gen_pre = enroll
        gen_vl = sdf[(sdf["sessionIndex"].isin([5, 6])) & (sdf["rep"] > 20)]
        gen_te = sdf[(sdf["sessionIndex"].isin([7, 8])) & (sdf["rep"] > 20)]

        imp_pre = sample_stratified_impostors(t0_enroll_pool, subj, gen_pre, seed + 900 + idx)
        imp_tr = sample_stratified_impostors(t1_adapt_pool, subj, gen_tr, seed + idx)
        imp_vl = sample_stratified_impostors(s56_pool, subj, gen_vl, seed + 200 + idx)
        imp_te = sample_stratified_impostors(s78_pool, subj, gen_te, seed + 400 + idx)

        def extract_338(part_df: pd.DataFrame) -> np.ndarray:
            raw = part_df[CANONICAL_TIMING_COLS].values  # (N, 31) ordered as [11 H, 10 DD, 10 UD]
            H = raw[:, :11]
            DD = raw[:, 11:21]
            UD = raw[:, 21:31]

            t_dd = np.sum(DD, axis=1, keepdims=True) + 1e-4
            rhy = raw / t_dd                                      # 31
            hrel = H / (np.mean(H, axis=1, keepdims=True) + 1e-4) # 11

            # 1. Tempo-invariant rhythmic & raw deviations
            abs_rhy = np.abs((rhy - mean_rhy) / std_rhy)          # 31
            abs_hrel = np.abs((hrel - mean_hrel) / std_hrel)      # 11
            abs_raw = np.abs((raw - mean_raw) / std_raw)          # 31

            z_comb = 0.20 * abs_raw + 0.80 * abs_rhy              # 31
            sq_rhy = np.clip(abs_rhy ** 2, 0.0, 16.0)             # 31
            sq_comb = np.clip(z_comb ** 2, 0.0, 16.0)             # 31

            # 2. Strictly intra-group adjacent products & intra-digraph deviation couplings
            adj_rhy = intra_group_adjacent_products(abs_rhy)      # 31
            dig_comb = intra_digraph_couplings(z_comb)            # 31

            # 3. Signed temporal drift & intra-group finite differences
            z_raw_signed = (raw - mean_raw_s1) / std_raw          # 31
            diff_raw = intra_group_first_differences(z_raw_signed) # 31
            rollover = (UD < 0).astype(np.float32)                # 10

            norms = np.hstack([
                np.mean(abs_rhy, axis=1, keepdims=True),
                np.mean(abs_hrel, axis=1, keepdims=True),
                np.mean(z_comb, axis=1, keepdims=True),
                np.sqrt(np.mean(sq_rhy, axis=1, keepdims=True)),
                np.sqrt(np.mean(sq_comb, axis=1, keepdims=True)),
                np.percentile(abs_rhy, 75, axis=1, keepdims=True),
                np.percentile(z_comb, 75, axis=1, keepdims=True),
            ])  # 7

            return np.hstack([
                rhy, abs_rhy, abs_raw, z_comb, sq_rhy, sq_comb,
                adj_rhy, dig_comb, z_raw_signed, diff_raw,
                abs_hrel, rollover, norms,
            ])

        for key, gp, ip in [
            ("pretrain", gen_pre, imp_pre),
            ("train", gen_tr, imp_tr),
            ("val", gen_vl, imp_vl),
            ("test", gen_te, imp_te),
        ]:
            fg = extract_338(gp)
            fi = extract_338(ip)
            splits[key][0].append(np.vstack([fg, fi]))
            splits[key][1].append(np.concatenate([np.ones(len(fg)), np.zeros(len(fi))]))
            splits[key][2].append(np.full(len(fg) + len(fi), idx, dtype=np.int32))
            splits[key][3].append(
                np.concatenate([gp["sessionIndex"].values, ip["sessionIndex"].values]).astype(np.int32)
            )
            splits[key][4].append(
                np.concatenate([gp["rep"].values, ip["rep"].values]).astype(np.int32)
            )

    # Strictly causal feature standardization:
    # scaler_pre is fit exclusively on initial enrollment/pretrain split (rep <= 20).
    # All downstream splits (pretrain, train, val, test) are standardized in this exact same
    # fixed coordinate space, ensuring behavioral concept drift is evaluated without
    # artificial coordinate shifts between pretraining and adaptation.
    X_pre_raw = np.vstack(splits["pretrain"][0]).astype(np.float32)
    scaler_pre = StandardScaler().fit(X_pre_raw)

    out_arrays = {}
    subj_arrays: Dict[str, Any] = {}
    session_arrays: Dict[str, np.ndarray] = {}
    rep_arrays: Dict[str, np.ndarray] = {}

    for key in ["pretrain", "train", "val", "test"]:
        X = scaler_pre.transform(np.vstack(splits[key][0]).astype(np.float32)).astype(np.float32)
        y = np.concatenate(splits[key][1]).astype(np.float32)
        s = np.concatenate(splits[key][2])
        sess = np.concatenate(splits[key][3])
        rep = np.concatenate(splits[key][4])
        if key == "train":
            perm = rng.permutation(len(X))
            X, y, s, sess, rep = X[perm], y[perm], s[perm], sess[perm], rep[perm]
        out_arrays[key] = (X, y)
        subj_arrays[key] = s
        session_arrays[key] = sess
        rep_arrays[key] = rep

    subj_arrays["sessions"] = session_arrays
    subj_arrays["reps"] = rep_arrays
    subj_arrays["pretrain_xy"] = out_arrays["pretrain"]
    subj_arrays["scaler_pre_mean"] = scaler_pre.mean_.copy()
    subj_arrays["scaler_adapt_mean"] = scaler_pre.mean_.copy()
    subj_arrays["protocol"] = protocol

    X_train, y_train = out_arrays["train"]
    X_val, y_val = out_arrays["val"]
    X_test, y_test = out_arrays["test"]

    if return_subjects:
        return X_train, y_train, X_val, y_val, X_test, y_test, subj_arrays
    return X_train, y_train, X_val, y_val, X_test, y_test
