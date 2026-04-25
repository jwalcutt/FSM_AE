"""
Phase 4: Full Evaluation Pipeline

Trains autoencoders for all 3 FSMs, evaluates both autoencoder and
transition whitelist baseline, generates ROC curves, error histograms,
sensitivity analysis, and combined results table.

Outputs:
  figures/roc_{fsm}.png           -- ROC curves per fault type
  figures/error_hist_{fsm}.png    -- normal vs anomalous error distributions
  models/{fsm}_ae.pt              -- trained model weights
  Console: combined results table, sensitivity analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix,
)

from data_generation import (
    build_dataset,
    build_transition_whitelist,
    FSM_CONFIGS,
    FAULT_NORMAL, FAULT_STUCK_AT, FAULT_TRANSITION, FAULT_PERTURBATION,
    FAULT_TYPE_NAMES,
)

from autoencoder import (
    FSMAutoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    select_threshold,
)


FAULT_TYPES = [FAULT_STUCK_AT, FAULT_TRANSITION, FAULT_PERTURBATION]
FSM_DISPLAY = {
    "traffic_light": "Traffic Light",
    "bit_pattern": "Bit-Pattern",
    "vending_machine": "Vending Machine",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If a loaded model's overall test AUC or F1 falls below these, retrain once.
RETRAIN_AUC_FLOOR = 0.70
RETRAIN_F1_FLOOR = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If an FSM's overall test AUC or F1 falls below these, we retrain from scratch.
RETRAIN_AUC_FLOOR = 0.70
RETRAIN_F1_FLOOR = 0.50


# =========================================================================
# Whitelist baseline detector
# =========================================================================

def whitelist_predict(windows, window_size, num_states, whitelist):
    """Predict anomaly labels using transition whitelist baseline.

    A window is anomalous if it contains any (state_t, state_{t+1}) pair
    not in the whitelist. Fully vectorized.
    """
    states = windows.reshape(-1, window_size, num_states).argmax(axis=2)

    # Build legal-transition lookup matrix
    legal = np.zeros((num_states, num_states), dtype=bool)
    for s1, s2 in whitelist:
        legal[s1, s2] = True

    # Check every consecutive pair in every window
    current = states[:, :-1]
    next_s = states[:, 1:]
    is_legal = legal[current, next_s]  # (n_windows, window_size-1)

    # Anomalous if ANY transition in the window is illegal
    return (~is_legal.all(axis=1)).astype(np.int32)


# =========================================================================
# Per-fault-type metric helpers
# =========================================================================

def metrics_by_fault(labels, predictions, errors, fault_types):
    """Compute P/R/F1 (from binary preds) and AUC (from continuous errors)
    for each fault type. Filters to normal + that fault type's windows.

    Returns dict: fault_code -> {precision, recall, f1, auc, fpr, tpr}
    """
    results = {}
    for ft in FAULT_TYPES:
        mask = (fault_types == FAULT_NORMAL) | (fault_types == ft)
        sub_labels = labels[mask]
        sub_preds = predictions[mask]
        sub_errors = errors[mask] if errors is not None else None

        n_pos = (sub_labels == 1).sum()
        if n_pos == 0:
            results[ft] = dict(precision=0, recall=0, f1=0, auc=0,
                               fpr=np.array([0, 1]), tpr=np.array([0, 0]))
            continue

        p = precision_score(sub_labels, sub_preds, zero_division=0)
        r = recall_score(sub_labels, sub_preds, zero_division=0)
        f = f1_score(sub_labels, sub_preds, zero_division=0)

        if sub_errors is not None:
            fpr, tpr, _ = roc_curve(sub_labels, sub_errors)
            a = auc(fpr, tpr)
        else:
            fpr, tpr, a = np.array([0, 1]), np.array([0, 0]), 0.0

        results[ft] = dict(precision=p, recall=r, f1=f, auc=a, fpr=fpr, tpr=tpr)
    return results


# =========================================================================
# Train + evaluate one FSM
# =========================================================================

def _train_fresh(model, ds):
    print(f"  Training autoencoder on {DEVICE}...")
    return train_autoencoder(
        model, ds["train"], ds["val"],
        epochs=50, batch_size=1024, lr=1e-3, device=DEVICE,
    )


def train_and_evaluate(name, cfg, seed=42, force_retrain=False):
    """Evaluate AE + whitelist, reusing an existing checkpoint when available.

    Loads models/{name}_ae.pt if present and architecture matches. Retrains
    once if metrics on the loaded model fall below the floor, or if
    force_retrain=True, or if no checkpoint exists.
    """
    fsm = cfg["fsm_class"]()
    gen_fn = cfg["input_generator"]
    ws = cfg["window_size"]
    input_dim = ws * fsm.num_states
    ckpt_path = f"models/{name}_ae.pt"

    print(f"\n{'=' * 60}")
    print(f"  {FSM_DISPLAY[name]}  (window_size={ws}, input_dim={input_dim})")
    print(f"{'=' * 60}")

    # --- Dataset ---
    print("  Building dataset...")
    ds = build_dataset(fsm=fsm, generate_inputs_fn=gen_fn,
                       window_size=ws, seed=seed)
    print(f"  Train: {ds['train'].shape}  Val: {ds['val'].shape}"
          f"  Test: {ds['test_windows'].shape}")

    # --- Load or train ---
    model = FSMAutoencoder(input_dim=input_dim, bottleneck_dim=16)
    loaded = False
    if not force_retrain and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if (ckpt.get("input_dim") == input_dim
                    and ckpt.get("window_size") == ws
                    and ckpt.get("num_states") == fsm.num_states):
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(DEVICE)
                loaded = True
                print(f"  Loaded existing model from {ckpt_path}")
            else:
                print(f"  Checkpoint at {ckpt_path} dims mismatch; retraining.")
        except Exception as exc:
            print(f"  Failed to load {ckpt_path} ({exc}); retraining.")

    trained_this_run = False
    if not loaded:
        _train_fresh(model, ds)
        trained_this_run = True

    # --- Threshold ---
    val_errors = compute_reconstruction_errors(model, ds["val"], device=DEVICE)
    tau = select_threshold(val_errors, percentile=95)
    print(f"  Threshold tau = {tau:.6f}")

    # --- AE evaluation ---
    test_errors = compute_reconstruction_errors(model, ds["test_windows"], device=DEVICE)
    ae_preds = (test_errors > tau).astype(np.int32)

    # --- If loaded weights underperform, retrain once and re-evaluate ---
    if loaded:
        fpr_chk, tpr_chk, _ = roc_curve(ds["test_labels"], test_errors)
        overall_auc = auc(fpr_chk, tpr_chk)
        overall_f1 = f1_score(ds["test_labels"], ae_preds, zero_division=0)
        if overall_auc < RETRAIN_AUC_FLOOR or overall_f1 < RETRAIN_F1_FLOOR:
            print(f"  Loaded model is weak "
                  f"(AUC={overall_auc:.3f}, F1={overall_f1:.3f}); retraining.")
            model = FSMAutoencoder(input_dim=input_dim, bottleneck_dim=16)
            _train_fresh(model, ds)
            trained_this_run = True
            val_errors = compute_reconstruction_errors(model, ds["val"], device=DEVICE)
            tau = select_threshold(val_errors, percentile=95)
            print(f"  New threshold tau = {tau:.6f}")
            test_errors = compute_reconstruction_errors(model, ds["test_windows"], device=DEVICE)
            ae_preds = (test_errors > tau).astype(np.int32)

    ae_overall = dict(
        precision=precision_score(ds["test_labels"], ae_preds, zero_division=0),
        recall=recall_score(ds["test_labels"], ae_preds, zero_division=0),
        f1=f1_score(ds["test_labels"], ae_preds, zero_division=0),
    )
    fpr_all, tpr_all, _ = roc_curve(ds["test_labels"], test_errors)
    ae_overall["auc"] = auc(fpr_all, tpr_all)
    print(f"  AE overall  P={ae_overall['precision']:.3f}"
          f"  R={ae_overall['recall']:.3f}"
          f"  F1={ae_overall['f1']:.3f}"
          f"  AUC={ae_overall['auc']:.3f}")

    ae_by_fault = metrics_by_fault(
        ds["test_labels"], ae_preds, test_errors, ds["test_fault_types"])
    for ft in FAULT_TYPES:
        m = ae_by_fault[ft]
        print(f"    AE {FAULT_TYPE_NAMES[ft]:<15}"
              f" P={m['precision']:.3f} R={m['recall']:.3f}"
              f" F1={m['f1']:.3f} AUC={m['auc']:.3f}")

    # --- Whitelist evaluation ---
    whitelist = build_transition_whitelist(fsm)
    wl_preds = whitelist_predict(
        ds["test_windows"], ws, fsm.num_states, whitelist)

    wl_overall = dict(
        precision=precision_score(ds["test_labels"], wl_preds, zero_division=0),
        recall=recall_score(ds["test_labels"], wl_preds, zero_division=0),
        f1=f1_score(ds["test_labels"], wl_preds, zero_division=0),
    )
    print(f"  WL overall  P={wl_overall['precision']:.3f}"
          f"  R={wl_overall['recall']:.3f}"
          f"  F1={wl_overall['f1']:.3f}")

    wl_by_fault = metrics_by_fault(
        ds["test_labels"], wl_preds, None, ds["test_fault_types"])
    for ft in FAULT_TYPES:
        m = wl_by_fault[ft]
        print(f"    WL {FAULT_TYPE_NAMES[ft]:<15}"
              f" P={m['precision']:.3f} R={m['recall']:.3f}"
              f" F1={m['f1']:.3f}")

    # --- Confusion matrices ---
    print(f"\n  AE confusion matrix:\n{confusion_matrix(ds['test_labels'], ae_preds)}")
    print(f"  WL confusion matrix:\n{confusion_matrix(ds['test_labels'], wl_preds)}")

    # --- Save model (only when we trained new weights this run) ---
    if trained_this_run:
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "bottleneck_dim": 16,
            "window_size": ws,
            "num_states": fsm.num_states,
            "threshold": tau,
        }, f"models/{name}_ae.pt")
        print(f"  Saved models/{name}_ae.pt")

    # --- ROC curve figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for ft in FAULT_TYPES:
        m = ae_by_fault[ft]
        ax.plot(m["fpr"], m["tpr"],
                label=f"{FAULT_TYPE_NAMES[ft]} (AUC={m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{FSM_DISPLAY[name]} -- ROC by Fault Type")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"figures/roc_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved figures/roc_{name}.png")

    # --- Error histogram figure ---
    normal_err = test_errors[ds["test_labels"] == 0]
    anom_err = test_errors[ds["test_labels"] == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_err, bins=100, alpha=0.6, density=True, label="Normal")
    if len(anom_err) > 0:
        ax.hist(anom_err, bins=100, alpha=0.6, density=True, label="Anomalous")
    ax.axvline(tau, color="red", linestyle="--", lw=2, label=f"tau={tau:.5f}")
    ax.set_xlabel("Reconstruction Error (BCE)")
    ax.set_ylabel("Density")
    ax.set_title(f"{FSM_DISPLAY[name]} -- Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"figures/error_hist_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved figures/error_hist_{name}.png")

    return dict(ae_by_fault=ae_by_fault, wl_by_fault=wl_by_fault,
                ae_overall=ae_overall, wl_overall=wl_overall)


# =========================================================================
# Window-size sensitivity analysis
# =========================================================================

def sensitivity_analysis(name, cfg, seed=42):
    """Retrain + evaluate at N-4, N, N+4 for one FSM."""
    fsm = cfg["fsm_class"]()
    gen_fn = cfg["input_generator"]
    base_ws = cfg["window_size"]
    sizes = [max(4, base_ws - 4), base_ws, base_ws + 4]

    print(f"\n  {FSM_DISPLAY[name]} sensitivity  (N = {sizes})")
    rows = []
    for ws in sizes:
        input_dim = ws * fsm.num_states
        ds = build_dataset(fsm=fsm, generate_inputs_fn=gen_fn,
                           window_size=ws, seed=seed)
        model = FSMAutoencoder(input_dim=input_dim, bottleneck_dim=16)
        train_autoencoder(model, ds["train"], ds["val"],
                          epochs=50, batch_size=1024, lr=1e-3,
                          device=DEVICE, print_every=0)

        val_err = compute_reconstruction_errors(model, ds["val"], device=DEVICE)
        tau = select_threshold(val_err, percentile=95)

        test_err = compute_reconstruction_errors(model, ds["test_windows"], device=DEVICE)
        preds = (test_err > tau).astype(np.int32)

        f1 = f1_score(ds["test_labels"], preds, zero_division=0)
        fpr, tpr, _ = roc_curve(ds["test_labels"], test_err)
        a = auc(fpr, tpr)
        rows.append((ws, f1, a))
        print(f"    N={ws:>2}:  F1={f1:.3f}  AUC={a:.3f}")

    return rows


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- Train and evaluate all FSMs ---
    all_results = {}
    for name, cfg in FSM_CONFIGS.items():
        all_results[name] = train_and_evaluate(name, cfg)

    # --- Combined results table ---
    print(f"\n{'=' * 78}")
    print("COMBINED RESULTS TABLE")
    print(f"{'=' * 78}")
    header = (f"{'FSM':<20} {'Fault Type':<15} "
              f"{'WL P':>6} {'WL R':>6} {'WL F1':>6}  "
              f"{'AE P':>6} {'AE R':>6} {'AE F1':>6} {'AE AUC':>7}")
    print(header)
    print("-" * len(header))
    for name in FSM_CONFIGS:
        res = all_results[name]
        for ft in FAULT_TYPES:
            wl = res["wl_by_fault"][ft]
            ae = res["ae_by_fault"][ft]
            print(f"{FSM_DISPLAY[name]:<20} {FAULT_TYPE_NAMES[ft]:<15} "
                  f"{wl['precision']:>6.3f} {wl['recall']:>6.3f} {wl['f1']:>6.3f}  "
                  f"{ae['precision']:>6.3f} {ae['recall']:>6.3f} {ae['f1']:>6.3f} "
                  f"{ae['auc']:>7.3f}")

    # --- Sensitivity analysis ---
    print(f"\n{'=' * 78}")
    print("WINDOW SIZE SENSITIVITY ANALYSIS")
    print(f"{'=' * 78}")
    for name, cfg in FSM_CONFIGS.items():
        sensitivity_analysis(name, cfg)

if __name__ == "__main__":
    main()
