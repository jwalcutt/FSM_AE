"""
Confirm data generation pipeline produces correct,
well-distributed datasets for all three FSMs.

Checks:
  1. State frequency tables (all states appear with non-negligible frequency)
  2. Fault-injected traces differ from normal traces (spot checks)
  3. Window counts >= 20,000 training windows per FSM
  4. Dataset shapes and label distributions
"""

import numpy as np
from collections import Counter

from fsm import TrafficLightFSM, BitPatternDetectorFSM, VendingMachineFSM
from data_generation import (
    build_dataset,
    generate_traffic_light_inputs,
    generate_bit_pattern_inputs,
    generate_vending_machine_inputs,
    generate_traces,
    inject_stuck_at,
    inject_transition_fault,
    inject_perturbation,
    build_transition_whitelist,
    DEFAULT_WINDOW_SIZES,
    FAULT_TYPE_NAMES,
)


def print_state_frequencies(traces, fsm):
    """Print state frequency table for a list of traces."""
    counter = Counter()
    total = 0
    for trace in traces:
        for s in trace:
            counter[s] += 1
            total += 1

    print(f"  {'State':<15} {'Count':>8} {'Frequency':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10}")
    for state_id in range(fsm.num_states):
        count = counter[state_id]
        freq = count / total if total > 0 else 0
        print(f"  {fsm.state_names[state_id]:<15} {count:>8} {freq:>10.4f}")
    print(f"  {'TOTAL':<15} {total:>8}")

    # Check no state has zero frequency
    for state_id in range(fsm.num_states):
        assert counter[state_id] > 0, \
            f"FAIL: state {fsm.state_names[state_id]} never appears in training data"
    print("  PASS: All states appear with non-zero frequency\n")


def spot_check_faults(fsm, rng):
    """Show a few fault-injected traces alongside their originals."""
    # Generate a short trace for readability
    if isinstance(fsm, TrafficLightFSM):
        inputs = [None] * 16
    elif isinstance(fsm, BitPatternDetectorFSM):
        inputs = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    else:
        inputs = [5, 10, 5, 0, 0, 10, 10, 0, 0, 5, 5, 5, 5, 0, 0, 10]

    trace = fsm.run(inputs)
    names = fsm.trace_names

    print(f"  Original:     {names(trace)}")

    # Stuck-at
    mod, steps = inject_stuck_at(trace, 4, 3)
    print(f"  Stuck-at(4,3):{names(mod)}")
    print(f"    Fault steps: {sorted(steps)}")
    if steps:
        print(f"    PASS: stuck-at altered the trace")
    else:
        print(f"    NOTE: stuck-at did not alter trace (state was already stuck)")

    # Transition fault
    mod2, steps2 = inject_transition_fault(trace, 6, fsm.num_states, rng)
    print(f"  Trans(t=6):   {names(mod2)}")
    print(f"    Fault steps: {sorted(steps2)}")
    assert len(steps2) == 1, "Transition fault should affect exactly 1 step"
    print(f"    PASS: transition fault changed state at t=6")

    # Perturbation
    mod3, steps3 = inject_perturbation(trace, 0.3, fsm.num_states, rng)
    print(f"  Perturb(0.3): {names(mod3)}")
    print(f"    Fault steps: {sorted(steps3)}")
    print(f"    PASS: perturbation altered {len(steps3)} timesteps\n")


def validate_fsm_dataset(name, fsm, generate_inputs_fn, window_size):
    """Run all validation checks for one FSM."""
    print("=" * 60)
    print(f"Validating: {name}")
    print("=" * 60)

    rng = np.random.default_rng(123)

    # Build dataset
    dataset = build_dataset(
        fsm=fsm,
        generate_inputs_fn=generate_inputs_fn,
        window_size=window_size,
        seed=42,
    )

    train = dataset["train"]
    val = dataset["val"]
    test_w = dataset["test_windows"]
    test_l = dataset["test_labels"]
    test_ft = dataset["test_fault_types"]
    train_traces = dataset["train_traces"]

    # 1. State frequency table
    print("\n-- State Frequencies (Training Traces) --")
    print_state_frequencies(train_traces, fsm)

    # 2. Fault injection spot checks
    print("-- Fault Injection Spot Checks --")
    spot_check_faults(fsm, rng)

    # 3. Window counts
    print("-- Window Counts --")
    print(f"  Training windows:   {len(train):>8}")
    print(f"  Validation windows: {len(val):>8}")
    print(f"  Test windows:       {len(test_w):>8}")
    total_train = len(train) + len(val)
    print(f"  Total normal:       {total_train:>8}")
    assert len(train) >= 20000, \
        f"FAIL: need >= 20,000 training windows, got {len(train)}"
    print(f"  PASS: training windows >= 20,000")

    # 4. Dataset shapes
    feat_dim = window_size * fsm.num_states
    print(f"\n-- Dataset Shapes --")
    print(f"  train:        {train.shape}  (expected: (*, {feat_dim}))")
    print(f"  val:          {val.shape}")
    print(f"  test_windows: {test_w.shape}")
    print(f"  test_labels:  {test_l.shape}")
    print(f"  test_faults:  {test_ft.shape}")
    assert train.shape[1] == feat_dim
    assert val.shape[1] == feat_dim
    assert test_w.shape[1] == feat_dim
    assert len(test_l) == len(test_w)
    assert len(test_ft) == len(test_w)
    print(f"  PASS: all shapes consistent")

    # 5. Label distribution
    print(f"\n-- Test Label Distribution --")
    num_normal = int((test_l == 0).sum())
    num_anomalous = int((test_l == 1).sum())
    print(f"  Normal windows:    {num_normal:>8}")
    print(f"  Anomalous windows: {num_anomalous:>8}")
    print(f"  Anomaly ratio:     {num_anomalous / len(test_l):.3f}")

    # Fault type breakdown
    print(f"\n-- Test Fault Type Breakdown --")
    for ft_code, ft_name in FAULT_TYPE_NAMES.items():
        mask = test_ft == ft_code
        count = int(mask.sum())
        anomalous_in_type = int((test_l[mask] == 1).sum()) if count > 0 else 0
        print(f"  {ft_name:<15} windows: {count:>8}  anomalous: {anomalous_in_type:>8}")

    # 6. Transition whitelist
    whitelist = build_transition_whitelist(fsm)
    print(f"\n-- Transition Whitelist --")
    print(f"  Legal transitions: {len(whitelist)}")
    print(f"  {sorted(whitelist)}")

    # 7. Data value checks
    assert train.min() >= 0.0 and train.max() <= 1.0, \
        "FAIL: training data values outside [0, 1]"
    print(f"\n  PASS: all values in [0, 1] range (valid for BCE loss)")

    # Check one-hot structure: each timestep in a window should sum to 1
    sample = train[:100].reshape(-1, window_size, fsm.num_states)
    timestep_sums = sample.sum(axis=2)
    assert np.allclose(timestep_sums, 1.0), \
        "FAIL: one-hot encoding broken -- timestep sums != 1"
    print(f"  PASS: one-hot encoding verified (timestep sums == 1.0)")

    print()


def main():
    configs = [
        ("Traffic Light", TrafficLightFSM(),
         generate_traffic_light_inputs, DEFAULT_WINDOW_SIZES["traffic_light"]),
        ("Bit-Pattern Detector", BitPatternDetectorFSM(),
         generate_bit_pattern_inputs, DEFAULT_WINDOW_SIZES["bit_pattern"]),
        ("Vending Machine", VendingMachineFSM(),
         generate_vending_machine_inputs, DEFAULT_WINDOW_SIZES["vending_machine"]),
    ]

    for name, fsm, gen_fn, ws in configs:
        validate_fsm_dataset(name, fsm, gen_fn, ws)

    print("=" * 60)
    print("ALL PHASE 2 VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
