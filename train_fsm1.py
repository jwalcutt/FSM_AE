"""
Phase 3: Train autoencoder on FSM 1 (Traffic Light) and validate the
full pipeline end-to-end before scaling to other FSMs.

Outputs:
  - figures/fsm1_training_loss.png    — loss curve
  - figures/fsm1_val_error_hist.png   — validation error histogram with threshold
  - models/traffic_light_ae.pt       — trained model weights
  - Console: threshold value, sanity check results
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fsm import TrafficLightFSM
from data_generation import (
    build_dataset,
    generate_traffic_light_inputs,
    inject_stuck_at,
    inject_transition_fault,
    inject_perturbation,
    extract_windows,
    DEFAULT_WINDOW_SIZES,
)
from autoencoder import (
    FSMAutoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    select_threshold,
)


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    fsm = TrafficLightFSM()
    window_size = DEFAULT_WINDOW_SIZES["traffic_light"]  # 8
    input_dim = window_size * fsm.num_states              # 8 * 4 = 32

    # --- Build dataset ---
    print("Building FSM 1 dataset...")
    dataset = build_dataset(
        fsm=fsm,
        generate_inputs_fn=generate_traffic_light_inputs,
        window_size=window_size,
        seed=42,
    )
    train_data = dataset["train"]
    val_data = dataset["val"]
    print(f"  Train: {train_data.shape}  Val: {val_data.shape}")

    # --- Train autoencoder ---
    print("\nTraining autoencoder (bottleneck=16, epochs=50)...")
    model = FSMAutoencoder(input_dim=input_dim, bottleneck_dim=16)
    train_losses, val_losses = train_autoencoder(
        model, train_data, val_data,
        epochs=50, batch_size=64, lr=1e-3,
    )

    # --- Plot training loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train BCE")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("FSM 1 (Traffic Light) - Autoencoder Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fsm1_training_loss.png", dpi=150)
    plt.close(fig)
    print("\n  Saved figures/fsm1_training_loss.png")

    # --- Compute validation reconstruction errors ---
    print("\nComputing validation reconstruction errors...")
    val_errors = compute_reconstruction_errors(model, val_data)
    print(f"  Val error stats: mean={val_errors.mean():.6f}"
          f"  std={val_errors.std():.6f}"
          f"  min={val_errors.min():.6f}"
          f"  max={val_errors.max():.6f}")

    # --- Set threshold ---
    tau = select_threshold(val_errors, percentile=95)
    print(f"\n  Threshold (95th percentile): tau = {tau:.6f}")

    # --- Plot validation error histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(val_errors, bins=100, alpha=0.7, label="Normal (val)")
    ax.axvline(tau, color="red", linestyle="--", linewidth=2,
               label=f"tau (95th pct) = {tau:.5f}")
    ax.set_xlabel("Reconstruction Error (BCE)")
    ax.set_ylabel("Window Count")
    ax.set_title("FSM 1 - Validation Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fsm1_val_error_hist.png", dpi=150)
    plt.close(fig)
    print("  Saved figures/fsm1_val_error_hist.png")

    # --- Sanity check: anomalous windows should exceed threshold ---
    print("\n--- Sanity Check: Anomalous vs Normal Errors ---")
    rng = np.random.default_rng(99)

    # Generate a normal trace and an anomalous trace
    normal_trace = fsm.run([None] * 40)

    # Stuck-at fault at step 10 for 4 steps
    stuck_trace, _ = inject_stuck_at(normal_trace, 10, 4)
    # Transition fault at step 15
    trans_trace, _ = inject_transition_fault(normal_trace, 15, fsm.num_states, rng)
    # Perturbation with p=0.2
    perturb_trace, _ = inject_perturbation(normal_trace, 0.2, fsm.num_states, rng)

    traces = {
        "Normal": normal_trace,
        "Stuck-at(t=10,k=4)": stuck_trace,
        "Transition(t=15)": trans_trace,
        "Perturbation(p=0.2)": perturb_trace,
    }

    for name, trace in traces.items():
        windows = extract_windows(trace, window_size, fsm.num_states)
        errors = compute_reconstruction_errors(model, windows)
        max_err = errors.max()
        mean_err = errors.mean()
        exceeds = (errors > tau).sum()
        print(f"  {name:<25} mean_err={mean_err:.6f}"
              f"  max_err={max_err:.6f}"
              f"  windows>{tau:.4f}: {exceeds}/{len(errors)}")

    # --- Save model ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "bottleneck_dim": 16,
        "window_size": window_size,
        "num_states": fsm.num_states,
        "threshold": tau,
    }, "models/traffic_light_ae.pt")
    print("\n  Saved models/traffic_light_ae.pt")


if __name__ == "__main__":
    main()
