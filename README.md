# FSM Anomaly Detection with a Dense Autoencoder

This repository contains the full pipeline, trained models, evaluation
results, and figures for a project on **unsupervised runtime fault detection
in finite state machines (FSMs)**. A dense autoencoder is trained on traces
of normal FSM behavior and used to flag anomalous state sequences whose
reconstruction error exceeds a learned threshold. The learned detector is
compared head-to-head against a stateless rule-based baseline (a transition
whitelist) across three FSMs and three fault classes.

Headline quantitative results are in
`[figures/results_table.csv](figures/results_table.csv)`.

---

## Problem

Finite state machines control sequential behavior throughout digital
hardware (traffic controllers, protocol implementations, pipeline control,
on-chip security primitives). At runtime the state register can be
corrupted by stuck-at faults, single-event upsets, fault-injection attacks,
and random perturbation, and these failures often go undetected by
conventional offline test methods such as ATPG and BIST. This project
investigates whether a small autoencoder, trained only on traces of normal
state behavior, can flag anomalous state sequences as they occur, and how
that learned detector compares to a stateless rule that simply checks every
consecutive (statet, statet+1) pair against the FSM's legal transition
table.

## What this project accomplished

1. **End-to-end implementation** of three reference FSMs of increasing
  complexity, a fully synthetic data-generation pipeline, three fault
   injectors, a dense autoencoder, training, threshold selection, and a
   full evaluation harness.
2. **Per-FSM trained models** at `models/{traffic_light,bit_pattern,vending_machine}_ae.pt`,
  each evaluated on a labeled test set of tens of thousands of windows.
3. **Quantitative comparison** against a transition whitelist baseline,
  broken down by FSM and fault type, plus a window-size sensitivity
   analysis at three values of N per FSM.
4. **Paper-ready figures and tables** in `figures/`, including FSM state
  diagrams, the autoencoder block diagram, reconstruction-error
   histograms, ROC curves, the combined results table, and the sensitivity
   table (in CSV, plain-text, and LaTeX form).
5. **Paper-ready documentation** covering the complete method, quantitative
  results, and comparison against the baseline, with figures and tables
   suitable for a formal writeup.

## The three FSMs


| FSM                                | States                                         | Inputs               | Cycle behavior                    |
| ---------------------------------- | ---------------------------------------------- | -------------------- | --------------------------------- |
| Traffic Light Controller           | 4 (RED, REDYELLOW, GREEN, YELLOW)              | none (timer tick)    | fixed 4-step cycle                |
| Serial Bit-Pattern Detector "1011" | 5 (S0–S4)                                      | binary (0/1)         | variable; reaches S4 on match     |
| Vending Machine Controller         | 6 (IDLE, FIVE, TEN, FIFTEEN, DISPENSE, CHANGE) | nickel / dime / none | accumulates to 20¢ then dispenses |


Each is implemented as a deterministic Moore machine and verified by hand
against its transition table. State diagrams for all three are at
`figures/fsm_state_diagrams.png`.

## Method

### Data generation

- **7,000 input sequences per FSM**, run through the simulator to produce
state traces.
- **Biased sampling** so that all states appear with non-negligible
frequency. The bit-pattern generator inserts `1011` at a random offset in
every sequence so deeper match states (S3, S4) are well represented; the
vending generator draws nickels and dimes equiprobably.
- **One-hot encoding** of states (no integer encoding, which would imply
false ordinality).
- **Sliding windows** of length N with stride 1, flattened to a vector of
length N · |S|. Defaults: N = 8 (Traffic Light), N = 10 (Bit-Pattern,
Vending Machine).
- **80 / 20 split** of normal windows into training and validation; the
validation set is reserved exclusively for threshold selection.
- **Test set**: 2,000 sequences (1,000 normal, 1,000 fault-injected, with
the faulty half cycled evenly across the three fault types). After
windowing this yields 50k–110k labeled test windows per FSM.

### Fault injection (evaluation only)


| Fault        | Description                                                       | Parameters     |
| ------------ | ----------------------------------------------------------------- | -------------- |
| Stuck-at     | freeze state at random t for k consecutive steps                  | k ∈ {2,3,4,5}  |
| Transition   | replace state at random t with a uniformly random different state | one timestep   |
| Perturbation | independently replace each timestep with prob. p                  | p ∈ [0.1, 0.2] |


Per-window labels are computed at injection time from the recorded set of
altered timesteps. Labels are used for evaluation only; the model never
sees them during training.

### Autoencoder

Symmetric dense network:

```
input (N · |S|)  →  Dense 128 (ReLU)  →  Dense 64 (ReLU)
                 →  Dense 16 (bottleneck, ReLU)
                 →  Dense 64 (ReLU)  →  Dense 128 (ReLU)
                 →  Dense (N · |S|, sigmoid)
```

- **Loss**: Binary Cross-Entropy (BCE) per output unit, mean over the
window. BCE is the correct probabilistic loss for one-hot categorical
inputs.
- **Optimizer**: Adam, lr 1e-3, batch size 1024, 50 epochs.
- **One model per FSM**; the architecture is identical, only the
input/output dimension changes with N · |S|.
- Block diagram at `figures/autoencoder_architecture.png`.

### Threshold selection

The detection threshold τ is the **99th percentile of the validation
reconstruction-error distribution**. This targets a ~1% expected false
positive rate under normal operation, which is appropriate for a hardware
monitor. Any test window whose mean BCE exceeds τ is flagged anomalous.

### Baseline: transition whitelist

The whitelist is the set of all legal (statet, statet+1) pairs from
each FSM's transition table. A window is flagged if any consecutive pair
in it falls outside the whitelist. By construction the whitelist achieves
**perfect precision** but cannot detect faults that produce only legal
pairs (which is the dominant failure mode for stuck-at faults on FSMs with
self-loops).

## Headline results

From `figures/results_table.csv`. F1 is at the validation-selected τ; AUC
is the autoencoder's threshold-independent score.


| FSM                  | Fault Type   | WL F1     | AE F1     | AE AUC |
| -------------------- | ------------ | --------- | --------- | ------ |
| Traffic Light        | Stuck-At     | 1.000     | **1.000** | 1.000  |
| Traffic Light        | Transition   | 1.000     | **1.000** | 1.000  |
| Traffic Light        | Perturbation | 1.000     | **1.000** | 1.000  |
| Bit-Pattern Detector | Stuck-At     | **0.868** | 0.806     | 0.873  |
| Bit-Pattern Detector | Transition   | **0.957** | 0.890     | 0.960  |
| Bit-Pattern Detector | Perturbation | **0.978** | 0.956     | 0.979  |
| Vending Machine      | Stuck-At     | 0.853     | **0.970** | 0.991  |
| Vending Machine      | Transition   | **0.952** | 0.945     | 0.971  |
| Vending Machine      | Perturbation | 0.966     | **0.967** | 0.983  |


Key observations:

- The autoencoder achieves **perfect detection on the Traffic Light**
because its 4-state cycle is short and fully periodic, so the model
reconstructs every legal window with near-zero error.
- The autoencoder **closes the recall gap on stuck-at faults for the
Vending Machine**: the whitelist recovers only 74.4% of stuck-at faults
(because frozen legal states are still legal pairs), while the
autoencoder recovers 94.1% by learning the credit-accumulation
invariant.
- On the **Bit-Pattern Detector** the whitelist's perfect precision keeps
it ahead of the autoencoder on F1 across every fault class, but the
autoencoder's AUC remains high (0.87–0.98), which means the F1 deficit
is partly a threshold-choice effect rather than a discrimination
failure.

A window-size sensitivity analysis (`figures/sensitivity_table.csv`) shows
that the Traffic Light is insensitive to N, the Bit-Pattern Detector
benefits monotonically from longer windows (F1 0.814 → 0.871 → 0.893 over
N = 6, 10, 14), and the Vending Machine is largely flat over the same
range.

## Repository layout

```
.
├── README.md                       this file
│
├── fsm.py                          three FSM implementations + base class
├── data_generation.py              input generators, fault injection, dataset builder
├── autoencoder.py                  dense autoencoder model + training loop + threshold
│
├── train_fsm1.py                   standalone trainer for FSM 1 (sanity check)
├── evaluate.py                     full evaluation pipeline (all FSMs)
├── generate_figures.py             paper-ready figure and table generation
├── verify_fsms.py                  hand-verifies each FSM's transitions
├── verify_data.py                  validates dataset construction
│
├── figures/                        all generated PNG figures and table outputs
│   ├── fsm_state_diagrams.png
│   ├── autoencoder_architecture.png
│   ├── error_histograms.png
│   ├── roc_combined.png
│   ├── error_hist_{traffic_light,bit_pattern,vending_machine}.png
│   ├── roc_{traffic_light,bit_pattern,vending_machine}.png
│   ├── fsm1_training_loss.png
│   ├── fsm1_val_error_hist.png
│   ├── results_table.{csv,tex,txt}
│   └── sensitivity_table.{csv,tex,txt}
│
├── models/                         trained checkpoints
│   ├── traffic_light_ae.pt
│   ├── bit_pattern_ae.pt
│   └── vending_machine_ae.pt
│
└── eval_output.txt                 captured stdout from the latest evaluate.py run
```

## Reproducing the results

### Setup

```bash
python -m venv venv
source venv/Scripts/activate    # or venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### End-to-end pipeline

```bash
# 1. Verify FSM correctness 
python verify_fsms.py

# 2. Verify data generation 
python verify_data.py

# 3. Train + evaluate all three FSMs and run the window-size sweep.
#    Reuses an existing checkpoint in models/ if dimensions match;
#    retrains automatically if AUC < 0.70 or F1 < 0.50 on load.
python evaluate.py

# 4. Generate all paper figures and the combined results table.
python generate_figures.py
```

`evaluate.py` writes to `models/`, `figures/`, and stdout (mirrored in
`eval_output.txt`). `generate_figures.py` reads the checkpoints and  
produces every relevant figure and table.

## Tech stack

- **Python 3.x**
- **PyTorch** for the autoencoder and training loop
- **NumPy** for trace handling, one-hot encoding, sliding windows
- **scikit-learn** for precision/recall/F1, ROC curves, AUC
- **Matplotlib** for all figures (no external diagram tools)

## Limitations

The trained model is **specific to the FSM whose traces it observed**;
this is a per-circuit deployment model, analogous to per-asset anomaly
detectors in industrial settings. The method generalizes; individual
trained models do not. All experiments were run on **simulated traces**,
which lack the electrical noise, jitter, and metastability of real state
registers. Sliding-window detection introduces a detection latency of up
to N timesteps, and the percentile-based threshold trades off false
positive rate against detection sensitivity.