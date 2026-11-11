"""
Automated unit tests for verifying mathematical invariants of the optimization algorithms,
strictly intra-group feature boundaries, causal normalization, session/repetition
stratification between genuine and impostor splits, and Student-t biometric metrics.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    intra_digraph_couplings,
    intra_group_adjacent_products,
    intra_group_first_differences,
    prepare_concept_drift_dataset,
)
from src.metrics import compute_biometric_metrics, student_t_quantile_95
from src.models import BiometricMLP, clone_model_with_init
from src.optimizers import (
    armijo_line_search_full,
    bce_loss,
    pretrain_base_model,
    train_adam,
    train_block_coordinate_descent_armijo,
    train_conjugate_gradient_fletcher_reeves,
    train_rmsprop,
    train_steepest_descent_armijo,
)


def test_armijo_rejects_ascent_and_restores_parameters():
    torch.manual_seed(42)
    model = BiometricMLP(input_dim=16)
    X = torch.randn(32, 16)
    y = torch.randint(0, 2, (32,)).float()

    orig_weights = [p.detach().clone() for p in model.parameters()]
    loss = bce_loss(model, X, y, l2_reg=0.01)
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]

    # Ascent direction (+grads) must be rejected with step size 0.0 and intact weights
    ascent_dir = [g.clone() for g in grads]
    alpha = armijo_line_search_full(model, X, y, ascent_dir, grads, initial_alpha=0.5, l2_reg=0.01)
    assert alpha == 0.0
    for p, orig in zip(model.parameters(), orig_weights):
        assert torch.allclose(p, orig)


def test_armijo_sufficient_decrease_and_model_layer_blocks():
    torch.manual_seed(42)
    model = BiometricMLP(input_dim=16)
    blocks = model.layer_blocks()
    assert len(blocks) == 3
    assert all(len(b) == 2 for b in blocks)

    X = torch.randn(64, 16)
    y = torch.randint(0, 2, (64,)).float()

    f_before = float(bce_loss(model, X, y, l2_reg=0.01).item())
    loss = bce_loss(model, X, y, l2_reg=0.01)
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    descent_dir = [-g for g in grads]

    alpha = armijo_line_search_full(model, X, y, descent_dir, grads, initial_alpha=0.5, l2_reg=0.01)
    assert alpha > 0.0
    with torch.no_grad():
        for p, d in zip(model.parameters(), descent_dir):
            p.add_(alpha * d)
    f_after = float(bce_loss(model, X, y, l2_reg=0.01).item())
    assert f_after < f_before


def test_shared_pretrained_init_and_grad_eval_accounting():
    torch.manual_seed(42)
    X = torch.randn(80, 24)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).float()

    base = BiometricMLP(input_dim=24)
    pretrain_base_model(base, X, y, steps=2, lr=0.12, l2_reg=0.009)
    init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    m_rms = clone_model_with_init(24, init_state)
    m_adam = clone_model_with_init(24, init_state)
    m_sd = clone_model_with_init(24, init_state)
    m_bcd = clone_model_with_init(24, init_state)
    m_cg = clone_model_with_init(24, init_state)

    # Verify identical starting state_dict across all 5 cloned models
    for m in (m_rms, m_adam, m_sd, m_bcd, m_cg):
        for k, v in init_state.items():
            assert torch.equal(m.state_dict()[k], v)

    h_rms = train_rmsprop(m_rms, X, y, X, y, epochs=3, batch_size=20, l2_reg=0.009)
    h_adam = train_adam(m_adam, X, y, X, y, epochs=3, batch_size=20, l2_reg=0.009)
    h_sd = train_steepest_descent_armijo(m_sd, X, y, X, y, epochs=3, steps_per_epoch=4, l2_reg=0.009)
    h_bcd = train_block_coordinate_descent_armijo(m_bcd, X, y, X, y, epochs=3, sweeps_per_epoch=2, l2_reg=0.009)
    h_cg = train_conjugate_gradient_fletcher_reeves(m_cg, X, y, X, y, epochs=3, steps_per_epoch=4, l2_reg=0.009)

    assert h_rms["grad_evals"][-1] == 3 * 4       # 4 mini-batches/epoch * 3 epochs = 12
    assert h_adam["grad_evals"][-1] == 3 * 4      # 4 mini-batches/epoch * 3 epochs = 12
    assert h_sd["grad_evals"][-1] == 3 * 4        # 4 full-batch steps/epoch * 3 epochs = 12
    assert h_cg["grad_evals"][-1] == 3 * 4        # 4 full-batch steps/epoch * 3 epochs = 12
    assert h_bcd["grad_evals"][-1] == 3 * 2 * 3   # 2 sweeps/epoch * 3 layer blocks * 3 epochs = 18
    assert h_bcd["full_sweeps"][-1] == 3 * 2      # 2 full sweeps/epoch * 3 epochs = 6

    assert h_sd["loss"][-1] < h_sd["loss"][0]
    assert h_bcd["loss"][-1] < h_bcd["loss"][0]
    assert h_cg["loss"][-1] < h_cg["loss"][0]


def test_intra_group_feature_boundaries():
    """
    Verifies that H (0..10), DD (11..20), and UD (21..30) never mix across group boundaries
    in intra_group_adjacent_products or intra_group_first_differences, and never wrap around.
    """
    X = np.zeros((2, 31), dtype=np.float64)
    # Set only UD (indices 21..30) to non-zero values; H and DD remain strictly zero
    X[:, 21:31] = np.arange(1, 11, dtype=np.float64)

    adj = intra_group_adjacent_products(X)
    diffs = intra_group_first_differences(X)
    dig = intra_digraph_couplings(X)

    assert adj.shape == (2, 31)
    assert diffs.shape == (2, 31)
    assert dig.shape == (2, 31)

    # Since H and DD are zero, all H products (0..10) and DD products (11..20) must be strictly 0
    assert np.all(adj[:, :21] == 0.0)
    assert np.all(diffs[:, :21] == 0.0)
    # Since H and DD are zero, all digraph products H*UD, H*DD, DD*UD must be strictly 0
    assert np.all(dig == 0.0)


def test_student_t_confidence_intervals_and_val_calibrated_threshold():
    # Verify exact Student-t critical values for df=4 (5 seeds) and df=50 (51 subjects)
    assert np.isclose(student_t_quantile_95(4), 2.776445, atol=1e-4)
    assert np.isclose(student_t_quantile_95(50), 2.008559, atol=1e-4)

    y_val = np.array([0, 0, 1, 1] * 25, dtype=np.float32)
    s_val = np.array([0.1, 0.3, 0.7, 0.9] * 25, dtype=np.float32)
    subjects = np.repeat(np.arange(10), 10)
    sessions = np.tile(np.array([5, 6]), 50)

    val_m = compute_biometric_metrics(y_val, s_val, subjects=subjects, sessions=sessions)
    assert "threshold_frr_10" in val_m

    y_test = np.array([0, 0, 1, 1] * 25, dtype=np.float32)
    s_test = np.array([0.15, 0.35, 0.65, 0.85] * 25, dtype=np.float32)
    test_m = compute_biometric_metrics(
        y_test,
        s_test,
        subjects=subjects,
        sessions=np.tile(np.array([7, 8]), 50),
        fixed_threshold_frr_10=val_m["threshold_frr_10"],
    )
    assert 0.0 <= test_m["eer"] <= 1.0
    assert 0.0 <= test_m["far_at_val_frr_10"] <= 1.0
    assert 0.0 <= test_m["frr_at_val_frr_10"] <= 1.0
    assert "session_7" in test_m["session_eers"]
    assert "session_8" in test_m["session_eers"]


def test_chronological_causality_and_stratified_impostors():
    csv_path = Path(__file__).resolve().parent.parent / "data/DSL-StrongPasswordData.csv"
    if not csv_path.exists():
        pytest.skip(f"Dataset file not found at {csv_path}")

    X_tr, y_tr, X_vl, y_vl, X_te, y_te, meta = prepare_concept_drift_dataset(
        csv_path, seed=42, return_subjects=True
    )
    X_pre, y_pre = meta["pretrain_xy"]

    assert X_pre.shape == (8160, 338)
    assert X_tr.shape == (6120, 338)
    assert X_vl.shape == (6120, 338)
    assert X_te.shape == (6120, 338)

    # Verify that scaler_pre was fit strictly on X_pre (zero mean on pretrain split)
    assert np.allclose(np.mean(X_pre, axis=0), 0.0, atol=1e-4)
    assert np.allclose(meta["scaler_pre_mean"], meta["scaler_adapt_mean"])

    # Verify chronological session isolation
    sess = meta["sessions"]
    reps = meta["reps"]
    assert set(np.unique(sess["pretrain"])) == {1, 2, 3, 4}
    assert set(np.unique(sess["train"])) == {1, 2, 3, 4}
    assert set(np.unique(sess["val"])) == {5, 6}
    assert set(np.unique(sess["test"])) == {7, 8}

    # Verify repetition isolation: pretrain uses rep <= 20, train uses 21..35, val/test use 21..50
    assert np.all(reps["pretrain"] <= 20)
    assert np.all((reps["train"] >= 21) & (reps["train"] <= 35))
    assert np.all((reps["val"] >= 21) & (reps["val"] <= 50))
    assert np.all((reps["test"] >= 21) & (reps["test"] <= 50))

    # Verify identical session and exact repetition counts between genuine (y==1) and impostor (y==0) in every split
    for split_name, y_arr in [("pretrain", y_pre), ("train", y_tr), ("val", y_vl), ("test", y_te)]:
        s_arr = sess[split_name]
        r_arr = reps[split_name]
        pos_sess, pos_cnt = np.unique(s_arr[y_arr == 1], return_counts=True)
        neg_sess, neg_cnt = np.unique(s_arr[y_arr == 0], return_counts=True)
        assert np.array_equal(pos_sess, neg_sess)
        assert np.array_equal(pos_cnt, neg_cnt)

        # Verify exact 1-to-1 repetition histogram matching
        pos_reps, pos_r_cnt = np.unique(r_arr[y_arr == 1], return_counts=True)
        neg_reps, neg_r_cnt = np.unique(r_arr[y_arr == 0], return_counts=True)
        assert np.array_equal(pos_reps, neg_reps)
        assert np.array_equal(pos_r_cnt, neg_r_cnt)

    # Verify determinism under fixed seed
    X_tr2, y_tr2, _, _, _, _ = prepare_concept_drift_dataset(csv_path, seed=42, return_subjects=False)
    assert np.array_equal(X_tr, X_tr2)
    assert np.array_equal(y_tr, y_tr2)


def test_model_architecture_and_parameter_accounting():
    """
    Verifies that BiometricMLP(338) conforms strictly to the specified architecture:
    338 -> 16 (ReLU) -> 16 (ReLU) -> 1, comprising exactly 5,713 trainable parameters,
    and exposes 3 layer blocks for Block Coordinate Descent.
    """
    model = BiometricMLP(input_dim=338)
    assert model.fc1.in_features == 338
    assert model.fc1.out_features == 16
    assert model.fc2.in_features == 16
    assert model.fc2.out_features == 16
    assert model.fc3.in_features == 16
    assert model.fc3.out_features == 1
    assert model.num_parameters == 5713

    # Check forward logits with dummy tensor
    x = torch.randn(8, 338)
    logits = model.forward_logits(x)
    probs = model(x)
    assert logits.shape == (8,)
    assert probs.shape == (8,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))

    # Verify 3 topological layer blocks
    blocks = model.layer_blocks()
    assert len(blocks) == 3
    assert blocks[0][0].shape == (16, 338)
    assert blocks[0][1].shape == (16,)
    assert blocks[1][0].shape == (16, 16)
    assert blocks[1][1].shape == (16,)
    assert blocks[2][0].shape == (1, 16)
    assert blocks[2][1].shape == (1,)
