# Keystroke Drift Optimizers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/IlsafSh/keystroke-drift-optimizers/actions/workflows/ci.yml/badge.svg)](https://github.com/IlsafSh/keystroke-drift-optimizers/actions/workflows/ci.yml)
[![Conference: TiVS-2026](https://img.shields.io/badge/MTUCI-TiVS--2026-purple.svg)](https://mtuci.ru/science/nir_students/tivs/)

> **Reference code and benchmark implementation for the paper:**  
> *«Адаптивное обучение нейросетевых моделей непрерывной поведенческой аутентификации пользователей на основе методов градиентной оптимизации первого порядка»*  
> *(Adaptive Training of Continuous Behavioral Authentication Neural Models Based on First-Order Gradient Optimization Methods)*  
> **Author:** Ilsaf T. Sharibzhanov (Шарибжанов Илсаф Тагирович), MTUCI, Moscow, Russia  
> **Scientific Advisor:** Stanislav S. Zvezhinsky (Звежинский Станислав Сигизмундович), Dr. Sci. (Eng.), Prof., Dept. of Information Security, MTUCI  
> **Proceedings:** International Scientific and Technical Forum *«Telecommunications and Computing Systems — 2026»* (TiVS-2026, MTUCI), Section 3: Information Security (indexed in RINC / РИНЦ).

---

## 1. Problem Statement & Motivation

In continuous behavioral authentication systems (aligned with Zero Trust Architecture, **NIST SP 800-207**, and **GOST R 70262.1–2022**), user keystroke dynamics naturally drift over time (**Concept Drift**) due to operator fatigue, circadian rhythms, or hardware keyboard changes:

$$
P_t(\mathbf{x}, y) \neq P_0(\mathbf{x}, y)
$$

When performing adaptive model fine-tuning on small session batches ($`N \approx 20 \dots 50`$ repetitions) from a pre-trained base model $`\mathbf{w}_0`$ under identical Tikhonov $`L_2`$ regularization ($`\lambda = 0.009`$), standard stochastic optimizers with coordinate-wise gradient scaling (**RMSprop**, **Adam**) suffer from gradient distortion:

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \frac{\eta}{\sqrt{\mathbb{E}[\mathbf{g}_k^2] + \varepsilon}} \odot \mathbf{g}_k
$$

Dividing each partial derivative by running root-mean-square averages may artificially amplify low-amplitude jitter across noisy coordinates, leading to gradient distortion and degrading verification accuracy on delayed sessions (`Sessions 5–8`).

To preserve the true gradient geometry during session fine-tuning while guaranteeing monotonic decrease of the regularized objective, this repository investigates and benchmarks three deterministic first-order optimization schemes equipped with **Armijo backtracking line search**:

### 1.1. Steepest Descent with Armijo Line Search
Updates parameters strictly along the full negative Euclidean gradient $`\mathbf{p}_k = -\mathbf{g}_k`$:

$$
\mathbf{w}_{k+1} = \mathbf{w}_k + \alpha_k \mathbf{p}_k
$$

where step size $`\alpha_k > 0`$ is found via backtracking line search ($`\alpha \leftarrow \rho \alpha`$, $`\rho = 0.5`$) satisfying the Armijo condition of sufficient decrease:

$$
\mathcal{L}(\mathbf{w}_k + \alpha_k \mathbf{p}_k) \le \mathcal{L}(\mathbf{w}_k) + c \, \alpha_k \, \mathbf{g}_k^\top \mathbf{p}_k, \quad c = 10^{-4}
$$

If no step satisfies the condition within maximum iterations, $`\alpha_k = 0`$ and parameter state $`\mathbf{w}_k`$ is restored intact (no unverified update is ever applied).

### 1.2. Layer-Wise Gauss–Seidel Block Coordinate Descent (BCD)
Decomposes network parameters into $`B = 3`$ topological layer blocks $`\mathbf{w} = [(\mathbf{W}_1, \mathbf{b}_1), (\mathbf{W}_2, \mathbf{b}_2), (\mathbf{W}_3, \mathbf{b}_3)]`$, performing sequential block-wise updates:

$$
\mathbf{w}_{k+1}^{(b)} = \mathbf{w}_k^{(b)} - \alpha_k^{(b)} \nabla_{\mathbf{w}^{(b)}} \mathcal{L}(\mathbf{w}_k^{(1)}, \dots, \mathbf{w}_k^{(b)}, \dots, \mathbf{w}_k^{(B)})
$$

with individual per-layer Armijo line search $`\alpha_k^{(b)}`$ for selective layer adaptation.

### 1.3. Nonlinear Conjugate Gradients (Fletcher–Reeves, FR-CG)
Maintains search momentum along conjugate gradient directions:

$$
\mathbf{p}_k = -\mathbf{g}_k + \beta_k^{\text{FR}} \mathbf{p}_{k-1}, \quad \beta_k^{\text{FR}} = \frac{\|\mathbf{g}_k\|_2^2}{\|\mathbf{g}_{k-1}\|_2^2 + \varepsilon}
$$

with sufficient descent verification ($`\mathbf{g}_k^\top \mathbf{p}_k < 0`$), automatic fallback to $`-\mathbf{g}_k`$ upon non-descent directions, and periodic restarts every $`R = 4`$ steps.

---

## 2. Experimental Protocol & Feature Engineering

Experiments are conducted on the **CMU Keystroke Dynamics Benchmark** (`DSL-StrongPasswordData.csv`, Killourhy & Maxion, 2009), comprising **20,400** typing samples of the 10-character strong password `.tie5Roanl` across **51 subjects** (8 sessions per subject, 50 repetitions per session).

### 2.1. Temporal Evaluation Protocols & Exact Stratification
The benchmark supports two reproducible temporal evaluation protocols (`--protocol within_session` and `--protocol strict_session`), both producing identical split cardinalities (`N_pretrain = 8,160`, `N_train = 6,120`, `N_val = 6,120`, `N_test = 6,120`):

- **Protocol A (Default, `--protocol within_session`) — Within-Session Enrollment/Adaptation Split with Forward Session Evaluation**:
  - **Tier 1 — Within-Session Repetition Split (`Sessions 1–4`)**:
    - **Stage 1 — Initial Registration & Warm-Start Pre-training (`T0`: Sessions 1–4, rep ≤ 20, `N_pretrain = 8,160`)**: Each subject's reference template $`(\boldsymbol{\mu}_u, \boldsymbol{\sigma}_u)`$ and standardization parameters (`scaler_pre`) are computed exclusively from initial registration repetitions (`rep ≤ 20`, 80 genuine + 80 impostor samples per subject). A shared base initialization $`\mathbf{w}_0`$ is obtained via 2 full-batch warm-start steps ($`\text{lr} = 0.12`$, $`\lambda = 0.009`$, reducing `T0` loss from `0.7051` to `0.6773`, increasing accuracy from `51.80%` to `58.95%`, and lowering `T0` EER from `57.45%` to `31.32%`), intentionally leaving primary convergence to the session adaptation phase. All five optimizers start from this exact `state_dict` $`\mathbf{w}_0`$.
    - **Stage 2 — Session Adaptation / Fine-Tuning (`T1`: Sessions 1–4, 21 ≤ rep ≤ 35, `N_train = 6,120`)**: 60 genuine + 60 impostor samples per subject (later repetitions `21..35` within each of `Sessions 1–4`), standardized in the fixed `scaler_pre` coordinate space. Note that `T0` (`rep ≤ 20`) and `T1` (`21 ≤ rep ≤ 35`) partition each session's typing sequence chronologically by repetition index within `Sessions 1–4`.
  - **Tier 2 — Chronological Forward Session Evaluation (`Sessions 5–8`)**:
    - **Stage 3 — Intermediate Drift Validation (`T2`: Sessions 5–6, 21 ≤ rep ≤ 50, `N_val = 6,120`)**: Used for monitoring convergence dynamics and calibrating the fixed operating threshold $`\tau^*`$ satisfying empirical $`\text{FRR}_{\text{val}}(\tau^*) \le 10\%`$.
    - **Stage 4 — Delayed Drift Testing (`T3`: Sessions 7–8, 21 ≤ rep ≤ 50, `N_test = 6,120`)**: Final evaluation on the most temporally distant calendar sessions (`Sessions 7–8`), reporting Pooled EER, Per-User EER, ROC AUC, and both `FAR @ Val FRR=10%` and `FRR @ Val FRR=10%` at the validation-calibrated threshold $`\tau^*`$.
- **Protocol B (`--protocol strict_session`) — Strict Session-Sequential Causal Protocol**:
  - Enforces strictly disjoint calendar sessions across all four stages ($`s_{T_0} < s_{T_1} < s_{T_2} < s_{T_3}`$, i.e. $`\max s(T_0) < \min s(T_1) \le \max s(T_1) < \min s(T_2) \le \max s(T_2) < \min s(T_3)`$):
    - **`T0` (Initial Enrollment & Base Pre-training)**: `Sessions 1–2, rep ≤ 40` (`N_pretrain = 8,160`; subject templates $`(\boldsymbol{\mu}_u, \boldsymbol{\sigma}_u)`$ and `scaler_pre` fit strictly on `Sessions 1–2`).
    - **`T1` (Session Adaptation)**: `Sessions 3–4, 21 ≤ rep ≤ 50` (`N_train = 6,120`).
    - **`T2` (Validation & Threshold Calibration)**: `Sessions 5–6, 21 ≤ rep ≤ 50` (`N_val = 6,120`).
    - **`T3` (Delayed Drift Testing)**: `Sessions 7–8, 21 ≤ rep ≤ 50` (`N_test = 6,120`).
- **Exact 1-to-1 Session & Repetition Stratified Impostor Sampling**:
  - In every split (`pretrain`, `train`, `val`, `test`) under both protocols, each genuine observation $`(u, s, r)`$ for subject $`u`$, session $`s`$, and repetition $`r`$ is matched 1-to-1 with an impostor observation from another subject $`u' \neq u`$ with the **exact same session index $`s`$ and repetition number $`r`$** (`strict=True` raises a `ValueError` if any `(sessionIndex, rep)` pair is unavailable). This guarantees 100% identical 2D joint `(sessionIndex, rep)` contingency tables across genuine and impostor classes.

### 2.2. Subject-Conditioned Enrollment-Normalized Feature Space ($`D = 338`$)
The 31 raw timing columns are validated and ordered into three physically homogeneous blocks: **11 Hold times** (H1…H11), **10 Down-Down latencies** (DD1…DD10), and **10 Up-Down flight times** (UD1…UD10). All finite differences and adjacent products are computed **strictly inside each homogeneous group** without cyclic wrap-around (`np.roll` is never used):

| Feature Group | Dim | Mathematical Definition |
| :--- | :---: | :--- |
| 1. Relative rhythmic timings | 31 | $`r_k = x_k / T_{\text{DD}}`$, where $`T_{\text{DD}} = \sum_{i=1}^{10} \text{DD}_{i,i+1}`$ is total inter-key press span |
| 2. Unsigned rhythm z-deviations | 31 | $`z_{\text{rhy}, k} = \lvert r_k - \mu_{r, k} \rvert / \sigma_{r, k}`$ against subject enrollment template |
| 3. Unsigned raw timing z-deviations | 31 | $`z_{\text{raw}, k} = \lvert x_k - \mu_{x, k} \rvert / \sigma_{x, k}`$ against subject enrollment template |
| 4. Rhythm-dominant combined deviations | 31 | $`z_{\text{comb}, k} = 0.20\,z_{\text{raw}, k} + 0.80\,z_{\text{rhy}, k}`$ |
| 5. Clipped quadratic rhythm deviations | 31 | $`\min(z_{\text{rhy}, k}^2,\,16.0)`$ |
| 6. Clipped quadratic combined deviations | 31 | $`\min(z_{\text{comb}, k}^2,\,16.0)`$ |
| 7. Intra-group adjacent products | 31 | Within H (10+1 mean), DD (9+1 mean), and UD (9+1 mean) separately |
| 8. Intra-digraph deviation couplings | 31 | $`z_{\text{H},i} z_{\text{UD},i,i+1}`$ (10), $`z_{\text{H},i+1} z_{\text{DD},i,i+1}`$ (10), $`z_{\text{DD},i,i+1} z_{\text{UD},i,i+1}`$ (10) on $`z_{\text{comb}}`$ + mean (1) |
| 9. Signed session drift deviations | 31 | $`(x_k - \mu_{x, k}^{(s_1)}) / \sigma_{x, k}`$ relative to initial Session 1 baseline |
| 10. Intra-group first differences | 31 | $`\Delta \text{H}_i`$ (10+1 norm), $`\Delta \text{DD}_i`$ (9+1 norm), $`\Delta \text{UD}_i`$ (9+1 norm) without crossing group boundaries |
| 11. Relative hold-time deviations | 11 | $`\lvert h_i - \mu_{h, i} \rvert / \sigma_{h, i}`$, where normalized hold time $`h_i = H_i / \text{mean}(H)`$ |
| 12. Key-rollover indicators | 10 | Binary indicator $`I(\text{UD}_{i,i+1} < 0)`$ for key rollover (press before previous release) |
| 13. Statistical summary norms | 7 | Summary norms: mean, $`L_2`$-norm, and 75th percentile of absolute deviations |
| **Total** | **338** | Subject-conditioned, enrollment-normalized biometric representation |

### 2.3. Neural Network Architecture & Computational Budget Accounting
- **Classifier (`BiometricMLP`)**: `338 -> 16 (ReLU) -> 16 (ReLU) -> 1 (Logits / Sigmoid)`, comprising exactly $`m = (338 \times 16 + 16) + (16 \times 16 + 16) + (16 \times 1 + 1) = 5\,713`$ trainable parameters partitioned into $`B = 3`$ topological layer blocks.
- **Unified Objective Function**: Numerically stable binary cross-entropy on logits (`BCEWithLogitsLoss`) with identical Tikhonov $`L_2`$ weight decay ($`\lambda = 0.009`$) across **all five** optimizers:

$$
\mathcal{L}(\mathbf{w}) = \mathcal{L}_{\text{BCE}}(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2
$$

- **Computational Complexity & Evaluations Breakdown (20 Adaptation Epochs, `N_train = 6,120`, Seeds 42–46)**:

| Optimization Method | Update Regime | Batch Size | Grad Evals | Grad Samples Processed | Func Evals (mean ± std) | Backtracking Steps (mean ± std) | Epoch Time, ms (seed=42) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSprop (baseline)** | 4 mini-batches / epoch | 1,530 | 80 (mini) | 122,400 | 80.0 ± 0.0 | 0.0 ± 0.0 | 18.7 |
| **Adam (baseline)** | 4 mini-batches / epoch | 1,530 | 80 (mini) | 122,400 | 80.0 ± 0.0 | 0.0 ± 0.0 | 21.0 |
| **Steepest Descent (Armijo)** | 4 full-batch steps / epoch | 6,120 | 80 (full) | 489,600 | 168.4 ± 1.7 | 8.4 ± 1.7 | 43.2 |
| **Block Coordinate Descent** | 4 sweeps × 3 blocks / epoch | 6,120 | 240 (block) | 1,468,800 | 482.4 ± 0.5 | 2.4 ± 0.5 | 103.4 |
| **Conjugate Gradients (FR)** | 4 full-batch steps / epoch | 6,120 | 80 (full) | 489,600 | 165.4 ± 2.3 | 5.4 ± 2.3 | 46.4 |

*Note:* For Armijo line-search methods, each step reuses the forward pass loss $`\mathcal{L}(\mathbf{w}_k)`$ computed during gradient evaluation and performs 1 trial evaluation $`\mathcal{L}(\mathbf{w}_k + \alpha_0 \mathbf{p}_k)`$ plus any step-halving backtracking evaluations (`Func Evals = Grad Evals + Steps + Backtracking Steps`).

---

## 3. Experimental Results

### 3.1. Single-Seed Benchmark (`within_session`, seed = 42, 20 Adaptation Epochs, λ = 0.009)

| Optimization Method | Grad Evals | Func Evals | Train Loss | Train Acc, % | Val Loss | Val Acc, % | Test Loss | Test Acc, % | Pooled EER, % | Per-User EER, % | FAR @ Val FRR=10%, % | FRR @ Val FRR=10%, % | ROC AUC | Epoch Time, ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSprop (baseline)** | 80 (mini) | 80 | 0.1272 | 95.46 | 0.3676 | 86.18 | 0.6201 | 78.56 | 17.68 | 15.67 ± 14.72 | 15.26 | 21.18 | 0.9047 | 18.7 |
| **Adam (baseline)** | 80 (mini) | 80 | 0.1296 | 95.20 | 0.3438 | 86.37 | 0.5924 | 78.97 | 18.38 | 15.95 ± 14.58 | 14.77 | 22.71 | 0.9041 | 21.0 |
| **Steepest Descent (Armijo)** | 80 (full) | 166 | 0.1253 | 95.34 | **0.3264** | 87.16 | **0.5359** | **81.03** | **16.21** | 14.75 ± 13.41 | 14.31 | **19.51** | **0.9144** | 43.2 |
| **Block Coordinate Descent** | 240 (block) | 482 | 0.1217 | 95.39 | 0.3285 | 87.21 | 0.5455 | 80.88 | 16.27 | 14.87 ± 13.47 | 14.08 | 19.67 | 0.9140 | 103.4 |
| **Conjugate Gradients (FR)** | 80 (full) | 167 | **0.0855** | **97.24** | 0.3510 | **87.34** | 0.6084 | 80.46 | 16.50 | **14.72 ± 13.42** | **12.78** | 21.37 | 0.9134 | 46.4 |

### 3.2. Chronological Per-Session EER Progression (`Sessions 5 -> 6 -> 7 -> 8`, Seeds 42–46, mean ± std)

Across all 5 random seeds (`42–46`), **Fletcher–Reeves Conjugate Gradients (FR-CG)** achieves the **lowest mean EER on every individual delayed session (`Session 5`, `Session 6`, `Session 7`, and `Session 8`)**, while layer-wise **Block Coordinate Descent (BCD)** exhibits the smallest relative drift increment between Session 5 and Session 8 (`+5.87 ± 0.58 pp`):

| Optimization Method | Session 5 (Val), % (mean ± std [seed=42]) | Session 6 (Val), % (mean ± std [seed=42]) | Session 7 (Test), % (mean ± std [seed=42]) | Session 8 (Test), % (mean ± std [seed=42]) | Drift Degradation (S8 − S5), pp (mean ± std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSprop (baseline)** | 11.88 ± 0.34 [12.19] | 12.54 ± 0.39 [13.10] | 16.68 ± 0.39 [16.60] | 18.24 ± 0.48 [18.66] | +6.36 ± 0.49 |
| **Adam (baseline)** | 12.07 ± 0.39 [11.96] | 12.59 ± 0.51 [13.33] | 17.07 ± 0.44 [17.61] | 18.26 ± 0.61 [18.95] | +6.20 ± 0.61 |
| **Steepest Descent (Armijo)** | 11.52 ± 0.41 [11.27] | 12.36 ± 0.33 [12.78] | 16.35 ± 0.27 [15.92] | 17.63 ± 0.69 [16.73] | +6.12 ± 0.66 |
| **Block Coordinate Descent** | 11.47 ± 0.39 [11.21] | 12.22 ± 0.33 [12.55] | 16.24 ± 0.32 [15.92] | 17.34 ± 0.54 [16.76] | **+5.87 ± 0.58** |
| **Conjugate Gradients (FR)** | **10.86 ± 0.33 [10.62]** | **11.97 ± 0.56 [12.75]** | **15.81 ± 0.35 [15.82]** | **17.10 ± 0.59 [17.25]** | +6.24 ± 0.60 |

### 3.3. Multi-Seed Statistical Evaluation (`within_session`, Seeds 42–46, Student-t 95% CI, df = 4)

Confidence intervals are calculated using the two-tailed Student-$`t`$ distribution with $`df = 4`$ degrees of freedom ($`t_{0.975, 4} = 2.7764`$):

| Optimization Method | Pooled EER, % (mean ± std [95% CI]) | Per-User EER, % (mean ± std [95% CI]) | FAR @ 0.5, % (mean ± std) | FAR @ Val FRR=10%, % (mean ± std) | FRR @ Val FRR=10%, % (mean ± std) | ROC AUC (mean ± std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSprop (baseline)** | 17.41 ± 0.27 [±0.34] | 15.18 ± 0.60 [±0.74] | **5.65 ± 0.90** | 13.79 ± 0.96 | 21.89 ± 0.64 | 0.9088 ± 0.0029 |
| **Adam (baseline)** | 17.66 ± 0.41 [±0.51] | 15.29 ± 0.51 [±0.63] | 6.53 ± 0.52 | 13.88 ± 0.75 | 22.09 ± 0.70 | 0.9091 ± 0.0030 |
| **Steepest Descent (Armijo)** | 16.97 ± 0.47 [±0.58] | 15.00 ± 0.39 [±0.48] | 6.38 ± 0.68 | 13.57 ± 0.84 | **20.73 ± 1.06** | 0.9117 ± 0.0021 |
| **Block Coordinate Descent** | 16.78 ± 0.36 [±0.45] | 14.72 ± 0.54 [±0.68] | 6.52 ± 0.57 | 13.24 ± 0.83 | 20.79 ± 1.10 | 0.9127 ± 0.0027 |
| **Conjugate Gradients (FR)** | **16.41 ± 0.29 [±0.36]** | **14.25 ± 0.27 [±0.33]** | 5.76 ± 0.67 | **12.08 ± 0.81** | 21.39 ± 0.73 | **0.9168 ± 0.0030** |

*Note on Operating Point Shift under Concept Drift:* Because the decision threshold $`\tau^*`$ is calibrated on `T2` (`Sessions 5–6`) for $`\text{FRR}_{\text{val}}(\tau^*) \le 10\%`$ without looking at test labels, temporal concept drift between `Sessions 5–6` and `Sessions 7–8` shifts genuine score distributions downward, resulting in test `FRR @ Val FRR=10%` of `20.73%–22.09%`. At this fixed threshold $`\tau^*`$, **FR-CG** achieves the lowest false acceptance rate (`FAR = 12.08 ± 0.81%`), while **Steepest Descent** and **BCD** yield the lowest test false rejection rates (`20.73%` and `20.79%`).

### 3.4. Paired Seed-by-Seed Differences vs. Fletcher–Reeves CG (`within_session`, Seeds 42–46)

Because all methods share the exact same data splits and pre-trained initialization $`\mathbf{w}_0`$ within each seed $`i \in \{42, 43, 44, 45, 46\}`$, paired differences isolate optimizer variance from dataset sampling variance. Multiple comparisons are evaluated using Student-$`t`$ tests ($`df = 4`$) with **Holm–Bonferroni step-down correction**:

| Baseline Method vs. FR-CG | Mean Paired ΔEER, pp | Paired Std $`s_d`$, pp | Student-t 95% CI (df=4), pp | Paired t-stat | p-value (raw) | p-value (Holm) | Mean Paired ΔUser EER, pp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adam − FR-CG** | **+1.25** | 0.43 | ±0.53 ([+0.72, +1.78]) | t = 6.53 | p = 0.0028 | **p = 0.0085** | **+1.04** |
| **RMSprop − FR-CG** | **+1.00** | 0.16 | ±0.20 ([+0.80, +1.20]) | t = 14.18 | p = 0.0001 | **p = 0.0006** | **+0.93** |
| **Steepest Descent − FR-CG** | **+0.56** | 0.52 | ±0.64 ([-0.08, +1.20]) | t = 2.40 | p = 0.0746 | p = 0.1492 | **+0.75** |
| **Block Coordinate Descent − FR-CG** | **+0.37** | 0.41 | ±0.51 ([-0.14, +0.88]) | t = 2.04 | p = 0.1114 | p = 0.1492 | **+0.47** |

### 3.5. Strict Session-Sequential Causal Protocol Benchmark (`--protocol strict_session`, Seeds 42–46)

Under **Protocol B (`strict_session`)**, where all four temporal splits occupy strictly disjoint calendar sessions (`T0: Sessions 1–2, rep ≤ 40` → `T1: Sessions 3–4, rep 21..50` → `T2: Sessions 5–6, rep 21..50` → `T3: Sessions 7–8, rep 21..50`), adapting on the chronologically closer `Sessions 3–4` reduces delayed test error across all methods while **preserving the exact same optimizer ranking**:

| Optimization Method | Pooled EER, % (mean ± std [95% CI]) | Per-User EER, % (mean ± std [95% CI]) | FAR @ Val FRR=10%, % (mean ± std) | FRR @ Val FRR=10%, % (mean ± std) | ROC AUC (mean ± std) | Session 5 -> 6 -> 7 -> 8 EER, % (mean) | Paired ΔEER vs. FR-CG, pp (raw p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSprop (baseline)** | 15.90 ± 0.58 [±0.73] | 13.71 ± 0.33 [±0.41] | 12.63 ± 0.70 | 20.12 ± 1.00 | 0.9215 ± 0.0044 | 11.49 -> 11.31 -> 15.05 -> 16.65 | +1.13 ± 0.93 (p = 0.0533) |
| **Adam (baseline)** | 15.77 ± 0.53 [±0.66] | 13.81 ± 0.45 [±0.55] | 12.29 ± 0.50 | 20.38 ± 1.38 | 0.9215 ± 0.0027 | 11.53 -> 11.27 -> 14.95 -> 16.69 | **+1.01 ± 0.65 (p = 0.0261)** |
| **Steepest Descent (Armijo)** | 15.77 ± 0.29 [±0.35] | 13.74 ± 0.39 [±0.49] | 11.88 ± 0.57 | 20.41 ± 0.91 | 0.9216 ± 0.0021 | 11.07 -> 11.25 -> 14.99 -> 16.43 | **+1.01 ± 0.72 (p = 0.0352)** |
| **Block Coordinate Descent** | 15.46 ± 0.35 [±0.43] | 13.46 ± 0.39 [±0.49] | 11.69 ± 0.45 | 20.11 ± 0.89 | 0.9236 ± 0.0019 | 10.97 -> 11.09 -> 14.73 -> 16.12 | +0.69 ± 0.57 (p = 0.0545) |
| **Conjugate Gradients (FR)** | **14.77 ± 0.58 [±0.72]** | **12.90 ± 0.65 [±0.81]** | **10.58 ± 0.78** | **19.63 ± 1.46** | **0.9293 ± 0.0030** | **10.36 -> 10.89 -> 14.10 -> 15.48** | — (Reference) |

---

## 4. Convergence & Biometric Error Visualizations

### Figure 1: Validation Convergence Dynamics
![Validation Convergence Dynamics](figures/fig1_convergence_loss_acc.png)

### Figure 2: ROC Curves & Biometric Verification Errors on Delayed Drift Sessions
![ROC Curves and Biometric Errors](figures/fig2_roc_and_biometric_errors.png)

---

## 5. Repository Structure & Reproducibility

```text
keystroke-drift-optimizers/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Automated GitHub Actions CI workflow (Python 3.10–3.12)
├── data/
│   └── DSL-StrongPasswordData.csv   # CMU Keystroke Dynamics Benchmark dataset (20,400 samples)
├── figures/
│   ├── fig1_convergence_loss_acc.png
│   └── fig2_roc_and_biometric_errors.png
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # Schema validation, within_session & strict_session splits, 338-dim features
│   ├── models.py                    # BiometricMLP (338->16->16->1) with 3 topological layer blocks
│   ├── optimizers.py                # Pre-training, Armijo line search, 3-block BCD, FR-CG, RMSprop, Adam
│   └── metrics.py                   # Pooled/Per-User/Per-Session EER, Student-t CI, val-calibrated FAR/FRR@Val10
├── tests/
│   └── test_pipeline.py             # 7 automated unit tests for mathematical, 2D stratification & causal invariants
├── run_experiments.py               # Main CLI runner (within_session & strict_session + 5-seed statistical suite)
├── experiment_results.json          # Saved metrics, pretrain stats, complexity counts & 5-seed paired statistics
├── requirements.txt                 # Minimum version constraints
├── requirements-lock.txt            # Exact pinned package versions for bit-exact reproduction
└── LICENSE
```

### Quick Start
```bash
# Install dependencies (use requirements-lock.txt for exact pinned environment)
pip install -r requirements.txt

# Run automated unit test suite (7 tests)
pytest -v

# Run full benchmark (both within_session and strict_session 5-seed statistical suites)
python run_experiments.py --include-adam --multi-seed

# Or select a specific temporal protocol explicitly
python run_experiments.py --protocol strict_session --include-adam --multi-seed
```

---

## 📖 Citation

If you find this code or research useful, please cite our paper:

```bibtex
@inproceedings{sharibzhanov2026adaptive,
  title     = {Адаптивное обучение нейросетевых моделей непрерывной поведенческой аутентификации пользователей на основе методов градиентной оптимизации первого порядка},
  author    = {Шарибжанов, И. Т. and Звежинский, С. С.},
  booktitle = {Труды Международного научно-технического форума <<Телекоммуникационные и вычислительные системы -- 2026>> (ТиВС-2026)},
  year      = {2026},
  address   = {Москва},
  publisher = {МТУСИ},
  note      = {Секция 3: Информационная безопасность}
}
```

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
