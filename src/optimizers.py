"""
First-order gradient optimization methods with Armijo backtracking line search,
3-block layer-wise Gauss-Seidel Coordinate Descent, and Fletcher-Reeves Conjugate Gradients
for adaptive neural network fine-tuning under uniform L2 Tikhonov regularization.
"""

import time
from typing import Dict, Generator, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

_BCE_LOGITS = nn.BCEWithLogitsLoss()


def bce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, l2_reg: float = 0.009) -> torch.Tensor:
    """
    Numerically stable binary cross-entropy loss computed on raw logits via BCEWithLogitsLoss,
    with uniform L2 Tikhonov weight regularization:
        L(w) = BCEWithLogits(w; x, y) + 0.5 * l2_reg * ||w||_2^2
    """
    if hasattr(model, "forward_logits"):
        logits = model.forward_logits(x)
        loss = _BCE_LOGITS(logits, y)
    else:
        preds = torch.clamp(model(x), 1e-7, 1.0 - 1e-7)
        loss = nn.functional.binary_cross_entropy(preds, y)

    if l2_reg > 0.0:
        l2_penalty = sum(torch.sum(p * p) for p in model.parameters())
        loss = loss + 0.5 * l2_reg * l2_penalty
    return loss


@torch.no_grad()
def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, np.ndarray]:
    """Computes pure unregularized evaluation BCE loss, binary accuracy, and predicted probabilities."""
    model.eval()
    if hasattr(model, "forward_logits"):
        logits = model.forward_logits(x)
        loss = float(_BCE_LOGITS(logits, y).item())
        probs = torch.sigmoid(logits)
    else:
        probs = model(x)
        probs_clamped = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        loss = float(nn.functional.binary_cross_entropy(probs_clamped, y).item())

    acc = float(((probs >= 0.5).float() == y).float().mean().item())
    return loss, acc, probs.cpu().numpy()


def pretrain_base_model(
    model: nn.Module,
    X_pretrain: torch.Tensor,
    y_pretrain: torch.Tensor,
    steps: int = 2,
    lr: float = 0.12,
    l2_reg: float = 0.009,
    return_metrics: bool = False,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, float]]]:
    """
    Pre-trains initial biometric verification model w_0 on initial registration
    observations (Sessions 1-4, rep <= 20, standardized strictly with scaler_pre)
    prior to adaptive session fine-tuning. All compared optimizers subsequently
    start from this identical pre-trained state w_0.
    """
    with torch.no_grad():
        init_loss, init_acc, _ = evaluate_model(model, X_pretrain, y_pretrain)

    model.train()
    for _ in range(steps):
        model.zero_grad()
        loss = bce_loss(model, X_pretrain, y_pretrain, l2_reg=l2_reg)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(-lr * p.grad)

    with torch.no_grad():
        post_loss, post_acc, _ = evaluate_model(model, X_pretrain, y_pretrain)

    metrics = {
        "pretrain_init_loss": float(init_loss),
        "pretrain_init_acc": float(init_acc),
        "pretrain_post_loss": float(post_loss),
        "pretrain_post_acc": float(post_acc),
        "pretrain_steps": steps,
        "pretrain_lr": lr,
    }
    if return_metrics:
        return model, metrics
    return model


def make_shuffled_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yields shuffled mini-batches of tensors for stochastic optimization."""
    n = len(x)
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield x[idx], y[idx]


def armijo_line_search_full(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    direction: List[torch.Tensor],
    grads: List[torch.Tensor],
    initial_alpha: float = 0.50,
    c: float = 1e-4,
    rho: float = 0.5,
    max_iters: int = 20,
    l2_reg: float = 0.009,
    return_stats: bool = False,
    f_w: float = None,
) -> Union[float, Tuple[float, int, int]]:
    """
    Backtracking line search satisfying the Armijo sufficient decrease condition:
        f(w + alpha * p) <= f(w) + c * alpha * <grad, p>

    If the direction is not a descent direction (<grad, p> >= 0) or if the Armijo condition
    is not satisfied within `max_iters` halvings, the original parameters are restored and
    step size 0.0 is returned (no unverified parameter update is ever applied).
    """
    params = list(model.parameters())
    orig_params = [p.detach().clone() for p in params]
    ls_func_evals = 0
    bt_halvings = 0

    with torch.no_grad():
        if f_w is None:
            f_w = float(bce_loss(model, x_batch, y_batch, l2_reg=l2_reg).item())
            ls_func_evals += 1
        grad_dot_p = sum(float(torch.sum(g * d).item()) for g, d in zip(grads, direction))
        if grad_dot_p >= -1e-12:
            return (0.0, ls_func_evals, bt_halvings) if return_stats else 0.0

        alpha = initial_alpha
        for _ in range(max_iters):
            for p, orig, d in zip(params, orig_params, direction):
                p.copy_(orig + alpha * d)
            f_new = float(bce_loss(model, x_batch, y_batch, l2_reg=l2_reg).item())
            ls_func_evals += 1
            if f_new <= f_w + c * alpha * grad_dot_p:
                for p, orig in zip(params, orig_params):
                    p.copy_(orig)
                return (alpha, ls_func_evals, bt_halvings) if return_stats else alpha
            alpha *= rho
            bt_halvings += 1

        for p, orig in zip(params, orig_params):
            p.copy_(orig)
        return (0.0, ls_func_evals, bt_halvings) if return_stats else 0.0


def train_rmsprop(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 1530,
    lr: float = 0.003,
    l2_reg: float = 0.009,
    seed: int = 42,
) -> Dict[str, list]:
    """
    Baseline stochastic optimizer with coordinate-wise root-mean-square gradient scaling,
    evaluated under the exact same L2-regularized objective as the deterministic methods.
    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
    gen = torch.Generator().manual_seed(seed)
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        "epoch_time_ms": [], "grad_evals": [], "func_evals": [],
        "backtracking_steps": [], "grad_samples_processed": [],
    }
    grad_count = 0
    func_count = 0
    samples_count = 0

    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in make_shuffled_batches(X_train, y_train, batch_size, gen):
            optimizer.zero_grad()
            loss = bce_loss(model, xb, yb, l2_reg=l2_reg)
            loss.backward()
            optimizer.step()
            grad_count += 1
            func_count += 1
            samples_count += int(len(xb))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        tr_loss, tr_acc, _ = evaluate_model(model, X_train, y_train)
        vl_loss, vl_acc, _ = evaluate_model(model, X_val, y_val)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_accuracy"].append(vl_acc)
        history["epoch_time_ms"].append(dt_ms)
        history["grad_evals"].append(grad_count)
        history["func_evals"].append(func_count)
        history["backtracking_steps"].append(0)
        history["grad_samples_processed"].append(samples_count)

    return history


def train_adam(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 1530,
    lr: float = 0.003,
    l2_reg: float = 0.009,
    seed: int = 42,
) -> Dict[str, list]:
    """
    Stochastic adaptive moment estimation (Adam) baseline, evaluated under the exact same
    L2-regularized objective as the deterministic methods.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        "epoch_time_ms": [], "grad_evals": [], "func_evals": [],
        "backtracking_steps": [], "grad_samples_processed": [],
    }
    grad_count = 0
    func_count = 0
    samples_count = 0

    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in make_shuffled_batches(X_train, y_train, batch_size, gen):
            optimizer.zero_grad()
            loss = bce_loss(model, xb, yb, l2_reg=l2_reg)
            loss.backward()
            optimizer.step()
            grad_count += 1
            func_count += 1
            samples_count += int(len(xb))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        tr_loss, tr_acc, _ = evaluate_model(model, X_train, y_train)
        vl_loss, vl_acc, _ = evaluate_model(model, X_val, y_val)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_accuracy"].append(vl_acc)
        history["epoch_time_ms"].append(dt_ms)
        history["grad_evals"].append(grad_count)
        history["func_evals"].append(func_count)
        history["backtracking_steps"].append(0)
        history["grad_samples_processed"].append(samples_count)

    return history


def train_steepest_descent_armijo(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    steps_per_epoch: int = 4,
    initial_alpha: float = 0.58,
    l2_reg: float = 0.009,
) -> Dict[str, list]:
    """
    Deterministic full-batch Steepest Descent with Armijo backtracking line search
    along the true unscaled negative gradient p_k = -nabla f(w_k).
    """
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        "epoch_time_ms": [], "grad_evals": [], "func_evals": [],
        "backtracking_steps": [], "grad_samples_processed": [],
    }
    params = list(model.parameters())
    grad_count = 0
    func_count = 0
    bt_count = 0
    n_samples = int(len(X_train))

    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for _ in range(steps_per_epoch):
            model.zero_grad()
            loss = bce_loss(model, X_train, y_train, l2_reg=l2_reg)
            loss.backward()
            grad_count += 1
            func_count += 1
            grads = [p.grad.detach().clone() for p in params]
            direction = [-g for g in grads]

            alpha, ls_evals, ls_bt = armijo_line_search_full(
                model,
                X_train,
                y_train,
                direction,
                grads,
                initial_alpha=initial_alpha,
                c=1e-4,
                rho=0.5,
                l2_reg=l2_reg,
                return_stats=True,
                f_w=float(loss.item()),
            )
            func_count += ls_evals
            bt_count += ls_bt
            if alpha > 0.0:
                with torch.no_grad():
                    for p, d in zip(params, direction):
                        p.add_(alpha * d)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        tr_loss, tr_acc, _ = evaluate_model(model, X_train, y_train)
        vl_loss, vl_acc, _ = evaluate_model(model, X_val, y_val)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_accuracy"].append(vl_acc)
        history["epoch_time_ms"].append(dt_ms)
        history["grad_evals"].append(grad_count)
        history["func_evals"].append(func_count)
        history["backtracking_steps"].append(bt_count)
        history["grad_samples_processed"].append(grad_count * n_samples)

    return history


def train_block_coordinate_descent_armijo(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    sweeps_per_epoch: int = 4,
    initial_alpha: float = 0.58,
    l2_reg: float = 0.009,
) -> Dict[str, list]:
    """
    Deterministic Gauss-Seidel Block Coordinate Descent (BCD) over the B = 3 topological
    layer blocks w = [(W_1, b_1), (W_2, b_2), (W_3, b_3)] with per-layer Armijo line search.
    Note: Each sweep updates all B = 3 layer blocks sequentially (B partial block-gradient
    evaluations per full sweep -> `sweeps_per_epoch * B` block-gradient evaluations per epoch).
    """
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        "epoch_time_ms": [], "grad_evals": [], "full_sweeps": [],
        "func_evals": [], "backtracking_steps": [], "grad_samples_processed": [],
    }
    if hasattr(model, "layer_blocks"):
        blocks: List[List[nn.Parameter]] = model.layer_blocks()
    else:
        blocks = [[p] for p in model.parameters()]

    all_params = list(model.parameters())
    grad_count = 0
    sweep_count = 0
    func_count = 0
    bt_count = 0
    n_samples = int(len(X_train))

    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for _ in range(sweeps_per_epoch):
            sweep_count += 1
            for block in blocks:
                block_ids = {id(p) for p in block}
                for p in all_params:
                    p.requires_grad_(id(p) in block_ids)
                model.zero_grad()
                f_w_tensor = bce_loss(model, X_train, y_train, l2_reg=l2_reg)
                f_w_tensor.backward()
                grad_count += 1
                func_count += 1
                grads = [p.grad.detach().clone() for p in block]
                f_w = float(f_w_tensor.item())
                grad_dot_p = -sum(float(torch.sum(g * g).item()) for g in grads)
                if grad_dot_p >= -1e-12:
                    continue

                orig_block = [p.detach().clone() for p in block]
                alpha = initial_alpha
                accepted = False
                with torch.no_grad():
                    for _ in range(15):
                        for p, orig, g in zip(block, orig_block, grads):
                            p.copy_(orig - alpha * g)
                        f_new = float(bce_loss(model, X_train, y_train, l2_reg=l2_reg).item())
                        func_count += 1
                        if f_new <= f_w + 1e-4 * alpha * grad_dot_p:
                            accepted = True
                            break
                        alpha *= 0.5
                        bt_count += 1
                    if not accepted:
                        for p, orig in zip(block, orig_block):
                            p.copy_(orig)
        for p in all_params:
            p.requires_grad_(True)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        tr_loss, tr_acc, _ = evaluate_model(model, X_train, y_train)
        vl_loss, vl_acc, _ = evaluate_model(model, X_val, y_val)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_accuracy"].append(vl_acc)
        history["epoch_time_ms"].append(dt_ms)
        history["grad_evals"].append(grad_count)
        history["full_sweeps"].append(sweep_count)
        history["func_evals"].append(func_count)
        history["backtracking_steps"].append(bt_count)
        history["grad_samples_processed"].append(grad_count * n_samples)

    return history


def train_conjugate_gradient_fletcher_reeves(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    steps_per_epoch: int = 4,
    initial_alpha: float = 0.55,
    restart_every: int = 4,
    l2_reg: float = 0.009,
) -> Dict[str, list]:
    """
    Deterministic full-batch nonlinear Conjugate Gradient method using the exact
    Fletcher-Reeves formula:
        beta_k^FR = ||g_k||_2^2 / (||g_{k-1}||_2^2 + eps)
        p_k = -g_k + beta_k^FR * p_{k-1}
    with Armijo backtracking line search, sufficient descent verification (<g_k, p_k> < 0),
    and periodic restarts to the negative gradient direction.
    """
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        "epoch_time_ms": [], "grad_evals": [], "func_evals": [],
        "backtracking_steps": [], "grad_samples_processed": [],
    }
    params = list(model.parameters())
    prev_grads = None
    prev_dirs = None
    step_count = 0
    grad_count = 0
    func_count = 0
    bt_count = 0
    n_samples = int(len(X_train))

    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for _ in range(steps_per_epoch):
            model.zero_grad()
            loss = bce_loss(model, X_train, y_train, l2_reg=l2_reg)
            loss.backward()
            grad_count += 1
            func_count += 1
            grads = [p.grad.detach().clone() for p in params]

            if prev_grads is None or prev_dirs is None or (step_count % restart_every == 0):
                direction = [-g for g in grads]
            else:
                norm_g_new = sum(float(torch.sum(g * g).item()) for g in grads)
                norm_g_old = sum(float(torch.sum(pg * pg).item()) for pg in prev_grads) + 1e-12
                beta_fr = norm_g_new / norm_g_old
                direction = [-g + beta_fr * pd for g, pd in zip(grads, prev_dirs)]

                # Verify that the conjugate direction remains a valid descent direction
                grad_dot_p = sum(float(torch.sum(g * d).item()) for g, d in zip(grads, direction))
                if grad_dot_p >= -1e-12:
                    direction = [-g for g in grads]

            alpha, ls_evals, ls_bt = armijo_line_search_full(
                model,
                X_train,
                y_train,
                direction,
                grads,
                initial_alpha=initial_alpha,
                c=1e-4,
                rho=0.5,
                l2_reg=l2_reg,
                return_stats=True,
                f_w=float(loss.item()),
            )
            func_count += ls_evals
            bt_count += ls_bt
            if alpha == 0.0:
                direction = [-g for g in grads]
                alpha, ls_evals2, ls_bt2 = armijo_line_search_full(
                    model,
                    X_train,
                    y_train,
                    direction,
                    grads,
                    initial_alpha=initial_alpha,
                    c=1e-4,
                    rho=0.5,
                    l2_reg=l2_reg,
                    return_stats=True,
                    f_w=float(loss.item()),
                )
                func_count += ls_evals2
                bt_count += ls_bt2

            if alpha > 0.0:
                with torch.no_grad():
                    for p, d in zip(params, direction):
                        p.add_(alpha * d)
                prev_grads = grads
                prev_dirs = direction
                step_count += 1
            else:
                prev_grads = None
                prev_dirs = None

        dt_ms = (time.perf_counter() - t0) * 1000.0

        tr_loss, tr_acc, _ = evaluate_model(model, X_train, y_train)
        vl_loss, vl_acc, _ = evaluate_model(model, X_val, y_val)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_accuracy"].append(vl_acc)
        history["epoch_time_ms"].append(dt_ms)
        history["grad_evals"].append(grad_count)
        history["func_evals"].append(func_count)
        history["backtracking_steps"].append(bt_count)
        history["grad_samples_processed"].append(grad_count * n_samples)

    return history
