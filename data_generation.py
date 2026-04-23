"""
Data generation pipeline for FSM anomaly detection.

Provides:
  - Biased input sequence generators (one per FSM)
  - Trace generation (run FSM on input sequences)
  - Sliding window extraction with one-hot encoding
  - Three fault injectors (stuck-at, transition, perturbation)
  - Per-window labeling for fault-injected traces
  - Full dataset builder (train/val/test splits)
  - Transition whitelist builder for the baseline detector
"""

import numpy as np
from fsm import TrafficLightFSM, BitPatternDetectorFSM, VendingMachineFSM


# Default window sizes per FSM 
DEFAULT_WINDOW_SIZES = {
    "traffic_light": 8,
    "bit_pattern": 10,
    "vending_machine": 10,
}

# Fault type codes stored in test_fault_types arrays
FAULT_NORMAL = 0
FAULT_STUCK_AT = 1
FAULT_TRANSITION = 2
FAULT_PERTURBATION = 3
FAULT_TYPE_NAMES = {
    FAULT_NORMAL: "normal",
    FAULT_STUCK_AT: "stuck_at",
    FAULT_TRANSITION: "transition",
    FAULT_PERTURBATION: "perturbation",
}


# =========================================================================
# Input Generators
# =========================================================================

def generate_traffic_light_inputs(num_sequences, rng, min_len=20, max_len=100):
    """Generate tick sequences for the traffic light (no meaningful input).

    Each sequence is a list of None values of random length [min_len, max_len].
    """
    sequences = []
    for _ in range(num_sequences):
        length = int(rng.integers(min_len, max_len + 1))
        sequences.append([None] * length)
    return sequences


def generate_bit_pattern_inputs(num_sequences, rng, min_len=20, max_len=50):
    """Generate binary input sequences biased to include '1011' at least once.

    Ensures states S3 and S4 are well-represented in the resulting traces.
    """
    sequences = []
    for _ in range(num_sequences):
        length = int(rng.integers(min_len, max_len + 1))
        bits = rng.integers(0, 2, size=length).tolist()
        # Insert "1011" at a random position to guarantee the pattern appears
        if length >= 4:
            pos = int(rng.integers(0, length - 3))
            bits[pos:pos + 4] = [1, 0, 1, 1]
        sequences.append(bits)
    return sequences


def generate_vending_machine_inputs(num_sequences, rng, min_len=20, max_len=80):
    """Generate coin input sequences with equal nickel/dime probability.

    DISPENSE and CHANGE states auto-transition regardless of input,
    so random coin sequences produce natural vending machine cycles.
    """
    sequences = []
    for _ in range(num_sequences):
        length = int(rng.integers(min_len, max_len + 1))
        coins = rng.choice(
            [VendingMachineFSM.NICKEL, VendingMachineFSM.DIME], size=length
        ).tolist()
        sequences.append(coins)
    return sequences


# =========================================================================
# Trace Generation
# =========================================================================

def generate_traces(fsm, input_sequences):
    """Run FSM on each input sequence and return a list of state traces.

    Each trace has length len(input_sequence) + 1 (includes initial state).
    """
    traces = []
    for inp_seq in input_sequences:
        traces.append(fsm.run(inp_seq))
    return traces


# =========================================================================
# One-Hot Encoding & Sliding Window Extraction
# =========================================================================

def one_hot_encode_trace(trace, num_states):
    """Convert a state trace to a one-hot encoded 2D array.

    Args:
        trace: list of integer state IDs
        num_states: total number of states in the FSM

    Returns:
        np.ndarray of shape (len(trace), num_states), dtype float32
    """
    arr = np.array(trace, dtype=np.int32)
    return np.eye(num_states, dtype=np.float32)[arr]


def extract_windows(trace, window_size, num_states):
    """Extract sliding windows from a state trace.

    Each window is one-hot encoded and flattened to a 1D vector.
    Stride is 1 (fully overlapping).

    Args:
        trace: list of integer state IDs
        window_size: number of timesteps per window
        num_states: total number of states in the FSM

    Returns:
        np.ndarray of shape (num_windows, window_size * num_states), dtype float32
    """
    n = len(trace)
    if n < window_size:
        return np.empty((0, window_size * num_states), dtype=np.float32)

    encoded = one_hot_encode_trace(trace, num_states)  # (n, num_states)

    num_windows = n - window_size + 1
    # Use stride tricks for efficient window extraction
    stride_t, stride_s = encoded.strides
    windows_3d = np.lib.stride_tricks.as_strided(
        encoded,
        shape=(num_windows, window_size, num_states),
        strides=(stride_t, stride_t, stride_s),
    )
    # Flatten each window and copy to own memory (stride_tricks shares memory)
    return windows_3d.reshape(num_windows, window_size * num_states).copy()


# =========================================================================
# Fault Injection
# =========================================================================

def inject_stuck_at(trace, t, k):
    """Freeze the state at timestep t for k consecutive steps.

    The state at position t persists through positions t+1 ... t+k-1.
    Position t itself is unchanged (the FSM reached that state normally).

    Returns:
        (modified_trace, fault_steps) where fault_steps is a set of
        timestep indices where the trace was altered.
    """
    modified = list(trace)
    fault_steps = set()
    stuck_state = trace[t]
    for i in range(t + 1, min(t + k, len(trace))):
        if modified[i] != stuck_state:
            fault_steps.add(i)
        modified[i] = stuck_state
    return modified, fault_steps


def inject_transition_fault(trace, t, num_states, rng):
    """Replace the state at timestep t with a random *different* state.

    Simulates a corrupted state register or combinational logic error.

    Returns:
        (modified_trace, fault_steps)
    """
    modified = list(trace)
    current = trace[t]
    choices = [s for s in range(num_states) if s != current]
    modified[t] = int(rng.choice(choices))
    return modified, {t}


def inject_perturbation(trace, p, num_states, rng):
    """Randomly corrupt each timestep with probability p.

    At each step, with probability p, the state is replaced with a
    uniformly random state (which may happen to be the same state).

    Returns:
        (modified_trace, fault_steps) where fault_steps contains indices
        where the state was actually changed.
    """
    modified = list(trace)
    fault_steps = set()
    for i in range(len(trace)):
        if rng.random() < p:
            new_state = int(rng.integers(0, num_states))
            if new_state != modified[i]:
                fault_steps.add(i)
            modified[i] = new_state
    return modified, fault_steps


# =========================================================================
# Window Labeling
# =========================================================================

def label_windows(trace_length, window_size, fault_steps):
    """Label each sliding window as normal (0) or anomalous (1).

    A window is anomalous if it overlaps with any fault-injected timestep.

    Args:
        trace_length: total length of the trace
        window_size: sliding window size
        fault_steps: set of timestep indices affected by fault injection

    Returns:
        np.ndarray of shape (num_windows,), dtype int32
    """
    num_windows = trace_length - window_size + 1
    labels = np.zeros(num_windows, dtype=np.int32)
    for step in fault_steps:
        # This fault step affects windows [max(0, step-window_size+1), step]
        win_start = max(0, step - window_size + 1)
        win_end = min(step + 1, num_windows)
        labels[win_start:win_end] = 1
    return labels


# =========================================================================
# Transition Whitelist
# =========================================================================

def build_transition_whitelist(fsm):
    """Extract the set of all legal (state_t, state_{t+1}) pairs from an FSM.

    Used as the baseline detector: any window containing an illegal
    transition is flagged as anomalous.
    """
    return fsm.get_legal_transitions()


# =========================================================================
# Dataset Builder
# =========================================================================

def build_dataset(fsm, generate_inputs_fn, window_size,
                  num_train_sequences=7000, num_test_sequences=2000,
                  seed=42):
    """Build complete train/val/test datasets for one FSM.

    Training data: normal traces only, split 80/20 into train/val.
    Test data: 50% normal traces + 50% fault-injected traces (evenly
    split across stuck-at, transition, and perturbation faults).

    Args:
        fsm: an FSM instance
        generate_inputs_fn: callable(num_sequences, rng) -> list of input sequences
        window_size: sliding window length
        num_train_sequences: number of training input sequences to generate
        num_test_sequences: total number of test sequences (half normal, half faulty)
        seed: random seed for reproducibility

    Returns:
        dict with keys:
            'train': np.ndarray (num_train_windows, window_size * num_states)
            'val': np.ndarray (num_val_windows, window_size * num_states)
            'test_windows': np.ndarray (num_test_windows, window_size * num_states)
            'test_labels': np.ndarray (num_test_windows,) — 0=normal, 1=anomalous
            'test_fault_types': np.ndarray (num_test_windows,) — fault type codes
            'train_traces': list of state traces (for validation checks)
    """
    rng = np.random.default_rng(seed)

    # --- Training data (normal traces only) ---
    train_inputs = generate_inputs_fn(num_train_sequences, rng)
    train_traces = generate_traces(fsm, train_inputs)

    train_windows_list = []
    for trace in train_traces:
        windows = extract_windows(trace, window_size, fsm.num_states)
        if len(windows) > 0:
            train_windows_list.append(windows)
    all_train_windows = np.concatenate(train_windows_list)

    # Shuffle and split 80% train / 20% validation
    perm = rng.permutation(len(all_train_windows))
    split_idx = int(0.8 * len(all_train_windows))
    train_data = all_train_windows[perm[:split_idx]]
    val_data = all_train_windows[perm[split_idx:]]

    # --- Test data ---
    num_normal_test = num_test_sequences // 2
    num_fault_test = num_test_sequences - num_normal_test

    test_windows_list = []
    test_labels_list = []
    test_fault_types_list = []

    # Normal test traces
    normal_test_inputs = generate_inputs_fn(num_normal_test, rng)
    normal_test_traces = generate_traces(fsm, normal_test_inputs)
    for trace in normal_test_traces:
        windows = extract_windows(trace, window_size, fsm.num_states)
        if len(windows) > 0:
            test_windows_list.append(windows)
            test_labels_list.append(np.zeros(len(windows), dtype=np.int32))
            test_fault_types_list.append(
                np.full(len(windows), FAULT_NORMAL, dtype=np.int32)
            )

    # Fault-injected test traces (cycle through 3 fault types)
    fault_test_inputs = generate_inputs_fn(num_fault_test, rng)
    fault_test_traces = generate_traces(fsm, fault_test_inputs)
    fault_cycle = [FAULT_STUCK_AT, FAULT_TRANSITION, FAULT_PERTURBATION]

    for i, trace in enumerate(fault_test_traces):
        fault_type = fault_cycle[i % 3]

        if fault_type == FAULT_STUCK_AT:
            t = int(rng.integers(1, max(2, len(trace) - 1)))
            k = int(rng.integers(2, 6))  # 2-5 steps
            modified, fault_steps = inject_stuck_at(trace, t, k)

        elif fault_type == FAULT_TRANSITION:
            t = int(rng.integers(1, max(2, len(trace) - 1)))
            modified, fault_steps = inject_transition_fault(
                trace, t, fsm.num_states, rng
            )

        else:  # FAULT_PERTURBATION
            p = float(rng.uniform(0.1, 0.2))
            modified, fault_steps = inject_perturbation(
                trace, p, fsm.num_states, rng
            )

        windows = extract_windows(modified, window_size, fsm.num_states)
        labels = label_windows(len(modified), window_size, fault_steps)

        if len(windows) > 0:
            test_windows_list.append(windows)
            test_labels_list.append(labels)
            test_fault_types_list.append(
                np.full(len(windows), fault_type, dtype=np.int32)
            )

    test_windows = np.concatenate(test_windows_list)
    test_labels = np.concatenate(test_labels_list)
    test_fault_types = np.concatenate(test_fault_types_list)

    return {
        "train": train_data,
        "val": val_data,
        "test_windows": test_windows,
        "test_labels": test_labels,
        "test_fault_types": test_fault_types,
        "train_traces": train_traces,
    }


# =========================================================================
# Convenience: build all three FSM datasets
# =========================================================================

FSM_CONFIGS = {
    "traffic_light": {
        "fsm_class": TrafficLightFSM,
        "input_generator": generate_traffic_light_inputs,
        "window_size": DEFAULT_WINDOW_SIZES["traffic_light"],
    },
    "bit_pattern": {
        "fsm_class": BitPatternDetectorFSM,
        "input_generator": generate_bit_pattern_inputs,
        "window_size": DEFAULT_WINDOW_SIZES["bit_pattern"],
    },
    "vending_machine": {
        "fsm_class": VendingMachineFSM,
        "input_generator": generate_vending_machine_inputs,
        "window_size": DEFAULT_WINDOW_SIZES["vending_machine"],
    },
}


def build_all_datasets(seed=42):
    """Build datasets for all three FSMs. Returns dict keyed by FSM name."""
    datasets = {}
    for name, cfg in FSM_CONFIGS.items():
        fsm = cfg["fsm_class"]()
        datasets[name] = build_dataset(
            fsm=fsm,
            generate_inputs_fn=cfg["input_generator"],
            window_size=cfg["window_size"],
            seed=seed,
        )
    return datasets
