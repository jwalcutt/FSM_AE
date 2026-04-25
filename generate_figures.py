"""
Phase 5: Paper-ready figure and table generation.

Produces all figures and tables required by the IEEE paper:
  figures/fsm_state_diagrams.png     -- Fig 1 (3 FSM state diagrams)
  figures/autoencoder_architecture.png -- Fig 2 (block diagram)
  figures/error_histograms.png       -- Fig 3 (normal vs anomalous, per FSM)
  figures/roc_combined.png           -- Fig 4 (ROC per FSM, curve per fault type)
  figures/results_table.csv          -- Table 1 data (machine-readable)
  figures/results_table.tex          -- Table 1 as IEEE LaTeX source
  figures/results_table.txt          -- Table 1 pretty-printed for console

Reads trained models from models/{fsm}_ae.pt. If a model is missing the script
errors out -- run evaluate.py first.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc,
)

from fsm import TrafficLightFSM, BitPatternDetectorFSM, VendingMachineFSM
from data_generation import (
    build_dataset, build_transition_whitelist, FSM_CONFIGS,
    FAULT_NORMAL, FAULT_STUCK_AT, FAULT_TRANSITION, FAULT_PERTURBATION,
    FAULT_TYPE_NAMES,
)
from autoencoder import (
    FSMAutoencoder, compute_reconstruction_errors, select_threshold,
)
from evaluate import whitelist_predict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FSM_DISPLAY = {
    "traffic_light":   "Traffic Light",
    "bit_pattern":     "Bit-Pattern Detector",
    "vending_machine": "Vending Machine",
}

FAULT_TYPES = [FAULT_STUCK_AT, FAULT_TRANSITION, FAULT_PERTURBATION]
FAULT_DISPLAY = {
    FAULT_STUCK_AT:     "Stuck-At",
    FAULT_TRANSITION:   "Transition",
    FAULT_PERTURBATION: "Perturbation",
}

# Consistent paper styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# =========================================================================
# Fig 1 -- FSM State Diagrams
# =========================================================================

import math

NODE_HEIGHT = 0.54   # vertical diameter, same for every node
NODE_FONT = 9
NODE_CHAR_W = 0.092  # empirical: data-unit width per char at NODE_FONT
NODE_PAD_X = 0.18


def _node_size(label):
    """Return (a, b) ellipse semi-axes sized to fit `label`."""
    b = NODE_HEIGHT / 2
    a = max(b, len(label) * NODE_CHAR_W / 2 + NODE_PAD_X)
    return a, b


def _draw_node(ax, xy, label, color, size):
    a, b = size
    ax.add_patch(Ellipse(
        xy, 2 * a, 2 * b,
        facecolor=color, edgecolor="black",
        linewidth=1.4, zorder=3,
    ))
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=NODE_FONT, fontweight="bold", zorder=4)


def _ellipse_radius_toward(cx, cy, a, b, target):
    """Distance from ellipse center to its boundary along (target - center)."""
    dx, dy = target[0] - cx, target[1] - cy
    d = math.hypot(dx, dy)
    if d == 0:
        return a
    c = dx / d
    s = dy / d
    return a * b / math.sqrt((b * c) ** 2 + (a * s) ** 2)


def _edge_endpoints(p1, p2, size1, size2):
    """Trim chord from p1 to p2 to the two ellipse boundaries."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return p1, p2
    ux, uy = dx / dist, dy / dist
    r1 = _ellipse_radius_toward(p1[0], p1[1], size1[0], size1[1], p2)
    r2 = _ellipse_radius_toward(p2[0], p2[1], size2[0], size2[1], p1)
    start = (p1[0] + ux * r1, p1[1] + uy * r1)
    end = (p2[0] - ux * r2, p2[1] - uy * r2)
    return start, end


def _draw_edge(ax, p1, p2, size1, size2, label, rad=0.0, color="black"):
    """Draw a directed edge from node at p1 to node at p2, trimmed to borders."""
    start, end = _edge_endpoints(p1, p2, size1, size2)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        # matplotlib arc3 places its Bezier control point perpendicular to the
        # chord, rotated 90 degrees CLOCKWISE; the curve peak sits at half that
        # offset. Label rides the peak so it tracks the arc.
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(dx, dy) or 1.0
        px, py = dy / norm, -dx / norm
        bend = rad * norm * 0.5
        lx = mx + px * bend
        ly = my + py * bend
        # Fully opaque so the arrow is not visible under the label.
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="none", alpha=1.0),
                zorder=5)


def _draw_self_loop(ax, center, size, label, angle_deg=90, spread_deg=32):
    """Self-loop attached to the node border.

    Tail emerges at (angle_deg + spread_deg) on the ellipse, arcs outward
    away from the node center, and the arrowhead returns at
    (angle_deg - spread_deg), pointing back into the bubble.
    """
    cx, cy = center
    a, b = size

    t_tail = math.radians(angle_deg + spread_deg)
    t_head = math.radians(angle_deg - spread_deg)
    tail = (cx + a * math.cos(t_tail), cy + b * math.sin(t_tail))
    head = (cx + a * math.cos(t_head), cy + b * math.sin(t_head))

    # Bow the arc outward from the node. arc3 with rad<0 bows left-of-direction;
    # going from tail(+spread) to head(-spread) the chord direction rotates with
    # angle_deg, but for angle_deg in (0, 180) the outward side is also the
    # left-of-direction, so rad<0 always bows away from the node centre here.
    rad = -2.4

    arrow = FancyArrowPatch(
        tail, head,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color="black",
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)

    # Compute the arc peak so we can place the label inside the C-shape of the
    # loop (between the node's border and the arc peak). This is empty space
    # -- the arc itself stays outside it -- so the label cannot overlap the
    # arc, and it stays close to the node rather than drifting into other
    # transitions above.
    mx = (tail[0] + head[0]) / 2
    my = (tail[1] + head[1]) / 2
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    norm = math.hypot(dx, dy) or 1.0
    px, py = dy / norm, -dx / norm
    bend = rad * norm * 0.5
    peak_x = mx + px * bend
    peak_y = my + py * bend

    bubble_edge_x = cx + a * math.cos(math.radians(angle_deg))
    bubble_edge_y = cy + b * math.sin(math.radians(angle_deg))
    lx = (bubble_edge_x + peak_x) / 2
    ly = (bubble_edge_y + peak_y) / 2
    ax.text(lx, ly, label, ha="center", va="center", fontsize=8,
            zorder=5)


def _draw_traffic_light(ax):
    fsm = TrafficLightFSM()
    color = "#ffd1d1"
    # Wider diamond to accommodate the long RED_YELLOW label on the right
    pos = {
        "RED":        (0.0,  1.6),
        "RED_YELLOW": (1.8,  0.0),
        "GREEN":      (0.0, -1.6),
        "YELLOW":     (-1.8, 0.0),
    }
    sizes = {n: _node_size(n) for n in pos}
    for name, xy in pos.items():
        _draw_node(ax, xy, name, color, sizes[name])
    for s, ns in fsm._transitions.items():
        u = fsm.state_names[s]
        v = fsm.state_names[ns]
        _draw_edge(ax, pos[u], pos[v], sizes[u], sizes[v], "tick", rad=0.0)
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("(a) Traffic Light Controller (4 states)",
                 fontsize=11, pad=8)


def _draw_bit_pattern(ax):
    fsm = BitPatternDetectorFSM()
    color = "#d1e9ff"
    pos = {
        "S0": (0.0, 0.0),
        "S1": (1.8, 0.0),
        "S2": (3.6, 0.0),
        "S3": (5.4, 0.0),
        "S4": (7.2, 0.0),
    }
    sizes = {n: _node_size(n) for n in pos}
    for name, xy in pos.items():
        _draw_node(ax, xy, name, color, sizes[name])

    # Merge parallel edges by input
    pair_inputs = {}
    for s, trans in fsm._transitions.items():
        for inp, ns in trans.items():
            pair_inputs.setdefault(
                (fsm.state_names[s], fsm.state_names[ns]), []
            ).append(str(inp))

    # Backward edges (right-to-left) bend downward with rad<0. Nest deeper rad
    # for longer spans so the arcs don't cross.
    edge_bend = {
        ("S0", "S1"): 0.0,
        ("S1", "S2"): 0.0,
        ("S2", "S3"): 0.0,
        ("S3", "S4"): 0.0,
        ("S2", "S0"): -0.35,
        ("S3", "S2"): -0.35,
        ("S4", "S2"): -0.50,
        ("S4", "S1"): -0.65,
    }

    for (u, v), inps in pair_inputs.items():
        if u == v:
            continue
        label = ",".join(sorted(set(inps)))
        rad = edge_bend.get((u, v), 0.22)
        _draw_edge(ax, pos[u], pos[v], sizes[u], sizes[v], label, rad=rad)

    if ("S0", "S0") in pair_inputs:
        lbl = ",".join(sorted(set(pair_inputs[("S0", "S0")])))
        _draw_self_loop(ax, pos["S0"], sizes["S0"], lbl, angle_deg=90)
    if ("S1", "S1") in pair_inputs:
        lbl = ",".join(sorted(set(pair_inputs[("S1", "S1")])))
        _draw_self_loop(ax, pos["S1"], sizes["S1"], lbl, angle_deg=90)

    ax.set_xlim(-0.8, 8.0)
    ax.set_ylim(-2.6, 1.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("(b) Bit-Pattern Detector \"1011\" (5 states)",
                 fontsize=11, pad=8)


def _draw_vending_machine(ax):
    fsm = VendingMachineFSM()
    color = "#d1ffd9"
    input_symbol = {
        VendingMachineFSM.NONE: "-",
        VendingMachineFSM.NICKEL: "N",
        VendingMachineFSM.DIME: "D",
    }
    # Top row: accumulators (IDLE -> FIVE -> TEN -> FIFTEEN)
    # Bottom row: DISPENSE (right) -> CHANGE (left) -> back up to IDLE
    pos = {
        "IDLE":     (0.0,  1.3),
        "FIVE":     (2.2,  1.3),
        "TEN":      (4.4,  1.3),
        "FIFTEEN":  (6.6,  1.3),
        "DISPENSE": (6.6, -1.3),
        "CHANGE":   (0.0, -1.3),
    }
    sizes = {n: _node_size(n) for n in pos}
    for name, xy in pos.items():
        _draw_node(ax, xy, name, color, sizes[name])

    pair_inputs = {}
    for s, trans in fsm._transitions.items():
        for inp, ns in trans.items():
            pair_inputs.setdefault(
                (fsm.state_names[s], fsm.state_names[ns]), []
            ).append(input_symbol[inp])

    # Forward skip-edges on the top row bow upward (rad<0 in matplotlib's CW
    # perpendicular convention). Bow them high enough to clear the self-loops
    # on the intermediate states (FIVE under IDLE->TEN, TEN under FIVE->FIFTEEN).
    edge_bend = {
        ("IDLE", "FIVE"):        0.0,
        ("IDLE", "TEN"):         -0.55,
        ("FIVE", "TEN"):         0.0,
        ("FIVE", "FIFTEEN"):     -0.55,
        ("TEN", "FIFTEEN"):      0.0,
        ("TEN", "DISPENSE"):     0.10,
        ("FIFTEEN", "DISPENSE"): 0.0,
        ("DISPENSE", "CHANGE"):  0.0,
        ("CHANGE", "IDLE"):      0.0,
    }

    for (u, v), inps in pair_inputs.items():
        if u == v:
            continue
        label = ",".join(sorted(set(inps)))
        rad = edge_bend.get((u, v), 0.18)
        _draw_edge(ax, pos[u], pos[v], sizes[u], sizes[v], label, rad=rad)

    # Self-loops (no-coin tick): IDLE, FIVE, TEN, FIFTEEN
    for state in ["IDLE", "FIVE", "TEN", "FIFTEEN"]:
        if (state, state) in pair_inputs:
            lbl = ",".join(sorted(set(pair_inputs[(state, state)])))
            _draw_self_loop(ax, pos[state], sizes[state], lbl, angle_deg=90)

    ax.set_xlim(-1.1, 7.7)
    ax.set_ylim(-2.3, 3.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("(c) Vending Machine Controller (6 states)   "
                 "[N=nickel, D=dime, -=no coin]",
                 fontsize=11, pad=8)


def fig1_state_diagrams(out_path):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    _draw_traffic_light(ax1)
    _draw_bit_pattern(ax2)
    _draw_vending_machine(ax3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =========================================================================
# Fig 2 -- Autoencoder Architecture
# =========================================================================

def fig2_autoencoder_architecture(out_path):
    """Block diagram: input -> 128 -> 64 -> 16 -> 64 -> 128 -> output."""
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.8)
    ax.set_axis_off()

    # Layers: (x_center, width, height, label_top, label_bottom, color)
    layers = [
        (1.0,  1.0, 3.8, "Input",       "N * num_states", "#eeeeee"),
        (3.0,  0.9, 3.2, "Dense",       "128 (ReLU)",     "#b3d9ff"),
        (4.8,  0.8, 2.6, "Dense",       "64 (ReLU)",      "#b3d9ff"),
        (6.4,  0.7, 1.6, "Bottleneck",  "16 (ReLU)",      "#ffcc80"),
        (8.0,  0.8, 2.6, "Dense",       "64 (ReLU)",      "#b3e6b3"),
        (9.8,  0.9, 3.2, "Dense",       "128 (ReLU)",     "#b3e6b3"),
        (12.0, 1.0, 3.8, "Output",      "Sigmoid",        "#eeeeee"),
    ]

    # Encoder bracket ends BEFORE the bottleneck; decoder bracket starts AFTER.
    # Both group labels sit above the dashed border (not on top of it).
    enc_x0, enc_x1 = 0.45, 5.35
    dec_x0, dec_x1 = 7.45, 12.55
    box_y0, box_y1 = 0.2, 4.8

    ax.add_patch(FancyBboxPatch(
        (enc_x0, box_y0), enc_x1 - enc_x0, box_y1 - box_y0,
        boxstyle="round,pad=0.05", linewidth=1.2,
        linestyle="--", edgecolor="#4682b4", facecolor="none",
    ))
    ax.text((enc_x0 + enc_x1) / 2, box_y1 + 0.25,
            "Encoder", color="#4682b4",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.add_patch(FancyBboxPatch(
        (dec_x0, box_y0), dec_x1 - dec_x0, box_y1 - box_y0,
        boxstyle="round,pad=0.05", linewidth=1.2,
        linestyle="--", edgecolor="#2e8b57", facecolor="none",
    ))
    ax.text((dec_x0 + dec_x1) / 2, box_y1 + 0.25,
            "Decoder", color="#2e8b57",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Layer rectangles
    for x, w, h, top, bot, col in layers:
        y = 2.5 - h / 2
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y), w, h,
            boxstyle="round,pad=0.02", linewidth=1.2,
            edgecolor="black", facecolor=col,
        ))
        ax.text(x, 2.5 + h / 2 + 0.15, top,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x, 2.5 - h / 2 - 0.18, bot,
                ha="center", va="top", fontsize=9)

    # Arrows between layers
    for i in range(len(layers) - 1):
        x1 = layers[i][0] + layers[i][1] / 2 + 0.05
        x2 = layers[i + 1][0] - layers[i + 1][1] / 2 - 0.05
        ax.annotate(
            "", xy=(x2, 2.5), xytext=(x1, 2.5),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"),
        )

    ax.set_title(
        "Autoencoder Architecture: Dense Encoder-Decoder for One-Hot FSM Windows",
        fontsize=12, pad=14,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =========================================================================
# Load models + evaluate (for Fig 3, Fig 4, Table 1)
# =========================================================================

def _load_model_and_errors(name, cfg, seed=42):
    """Load trained autoencoder, rebuild dataset, compute test errors and preds.

    Returns a dict with everything downstream figures/tables need.
    """
    fsm = cfg["fsm_class"]()
    ws = cfg["window_size"]
    input_dim = ws * fsm.num_states
    ckpt_path = f"models/{name}_ae.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Missing {ckpt_path} -- run evaluate.py first.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = FSMAutoencoder(input_dim=input_dim, bottleneck_dim=16).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = build_dataset(fsm=fsm, generate_inputs_fn=cfg["input_generator"],
                       window_size=ws, seed=seed)

    val_err = compute_reconstruction_errors(model, ds["val"], device=DEVICE)
    tau = select_threshold(val_err, percentile=95)

    test_err = compute_reconstruction_errors(
        model, ds["test_windows"], device=DEVICE)
    ae_preds = (test_err > tau).astype(np.int32)

    whitelist = build_transition_whitelist(fsm)
    wl_preds = whitelist_predict(
        ds["test_windows"], ws, fsm.num_states, whitelist)

    return dict(
        fsm=fsm, ws=ws,
        labels=ds["test_labels"],
        fault_types=ds["test_fault_types"],
        test_err=test_err,
        tau=tau,
        ae_preds=ae_preds,
        wl_preds=wl_preds,
    )


def _metrics_for_subset(labels, ae_preds, wl_preds, errors, mask):
    """Compute per-subset P/R/F1 for AE+WL and AUC for AE. Returns dict."""
    y = labels[mask]
    ae = ae_preds[mask]
    wl = wl_preds[mask]
    err = errors[mask]
    out = {}
    out["ae_p"] = precision_score(y, ae, zero_division=0)
    out["ae_r"] = recall_score(y, ae, zero_division=0)
    out["ae_f1"] = f1_score(y, ae, zero_division=0)
    out["wl_p"] = precision_score(y, wl, zero_division=0)
    out["wl_r"] = recall_score(y, wl, zero_division=0)
    out["wl_f1"] = f1_score(y, wl, zero_division=0)
    if (y == 1).sum() > 0 and (y == 0).sum() > 0:
        fpr, tpr, _ = roc_curve(y, err)
        out["ae_auc"] = auc(fpr, tpr)
        out["fpr"] = fpr
        out["tpr"] = tpr
    else:
        out["ae_auc"] = float("nan")
        out["fpr"] = np.array([0, 1])
        out["tpr"] = np.array([0, 0])
    return out


# =========================================================================
# Fig 3 -- Error histograms
# =========================================================================

def fig3_error_histograms(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, r) in zip(axes, results.items()):
        normal = r["test_err"][r["labels"] == 0]
        anom = r["test_err"][r["labels"] == 1]

        # Use matched bins across both distributions
        all_err = np.concatenate([normal, anom]) if len(anom) else normal
        lo, hi = np.quantile(all_err, [0.001, 0.999])
        bins = np.linspace(lo, hi, 80)

        ax.hist(normal, bins=bins, alpha=0.6, density=True,
                color="#3b8ed0", label=f"Normal (n={len(normal):,})")
        if len(anom):
            ax.hist(anom, bins=bins, alpha=0.6, density=True,
                    color="#d9534f", label=f"Anomalous (n={len(anom):,})")
        ax.axvline(r["tau"], color="black", linestyle="--", lw=1.8,
                   label=rf"$\tau$ = {r['tau']:.3g}")
        # Log-scale y makes the two overlapping distributions readable --
        # normal is concentrated near zero; anomalous is long-tailed.
        ax.set_yscale("log")
        ax.set_xlabel("Reconstruction Error (BCE)")
        ax.set_ylabel("Density (log scale)")
        ax.set_title(FSM_DISPLAY[name])
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Reconstruction Error Distributions (Normal vs. Anomalous)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =========================================================================
# Fig 4 -- ROC combined
# =========================================================================

def fig4_roc_combined(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {
        FAULT_STUCK_AT:     "#1f77b4",
        FAULT_TRANSITION:   "#ff7f0e",
        FAULT_PERTURBATION: "#2ca02c",
    }

    for ax, (name, r) in zip(axes, results.items()):
        for ft in FAULT_TYPES:
            mask = (r["fault_types"] == FAULT_NORMAL) | (r["fault_types"] == ft)
            y = r["labels"][mask]
            err = r["test_err"][mask]
            if (y == 1).sum() == 0 or (y == 0).sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y, err)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[ft], lw=1.8,
                    label=f"{FAULT_DISPLAY[ft]} (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(FSM_DISPLAY[name])
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Autoencoder ROC Curves by FSM and Fault Type",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =========================================================================
# Table 1 -- combined results
# =========================================================================

def table1_results(results, out_csv, out_tex, out_txt):
    rows = []  # (fsm_display, fault_display, wl_p, wl_r, wl_f1, ae_p, ae_r, ae_f1, ae_auc)
    for name, r in results.items():
        for ft in FAULT_TYPES:
            mask = (r["fault_types"] == FAULT_NORMAL) | (r["fault_types"] == ft)
            m = _metrics_for_subset(
                r["labels"], r["ae_preds"], r["wl_preds"],
                r["test_err"], mask,
            )
            rows.append((
                FSM_DISPLAY[name], FAULT_DISPLAY[ft],
                m["wl_p"], m["wl_r"], m["wl_f1"],
                m["ae_p"], m["ae_r"], m["ae_f1"], m["ae_auc"],
            ))

    # --- CSV ---
    with open(out_csv, "w") as f:
        f.write("FSM,FaultType,WL_Precision,WL_Recall,WL_F1,"
                "AE_Precision,AE_Recall,AE_F1,AE_AUC\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},"
                    f"{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},"
                    f"{row[5]:.4f},{row[6]:.4f},{row[7]:.4f},{row[8]:.4f}\n")
    print(f"  Saved {out_csv}")

    # --- Pretty-printed TXT ---
    with open(out_txt, "w") as f:
        header = (f"{'FSM':<20} {'Fault Type':<14} "
                  f"{'WL P':>6} {'WL R':>6} {'WL F1':>6}  "
                  f"{'AE P':>6} {'AE R':>6} {'AE F1':>6} {'AE AUC':>7}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in rows:
            f.write(f"{row[0]:<20} {row[1]:<14} "
                    f"{row[2]:>6.3f} {row[3]:>6.3f} {row[4]:>6.3f}  "
                    f"{row[5]:>6.3f} {row[6]:>6.3f} {row[7]:>6.3f} "
                    f"{row[8]:>7.3f}\n")
    print(f"  Saved {out_txt}")

    # --- LaTeX (IEEE style, drop-in for a two-column paper) ---
    with open(out_tex, "w") as f:
        f.write("% Auto-generated by generate_figures.py -- do not edit by hand.\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Detection performance of the transition whitelist "
                "baseline (WL) and the dense autoencoder (AE) across three "
                "FSMs and three fault types.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{llcccccccc}\n")
        f.write("\\hline\n")
        f.write("FSM & Fault Type & WL P & WL R & WL F1 & "
                "AE P & AE R & AE F1 & AE AUC \\\\\n")
        f.write("\\hline\n")
        prev_fsm = None
        for row in rows:
            fsm_col = row[0] if row[0] != prev_fsm else ""
            prev_fsm = row[0]
            f.write(f"{fsm_col} & {row[1]} & "
                    f"{row[2]:.3f} & {row[3]:.3f} & {row[4]:.3f} & "
                    f"{row[5]:.3f} & {row[6]:.3f} & {row[7]:.3f} & "
                    f"{row[8]:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"  Saved {out_tex}")

    # Echo to stdout so the run log contains the table
    print()
    with open(out_txt) as f:
        print(f.read())


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("  Phase 5: generating paper-ready figures and tables")
    print("=" * 60)

    print("\n[Fig 1] FSM state diagrams...")
    fig1_state_diagrams("figures/fsm_state_diagrams.png")

    print("\n[Fig 2] Autoencoder architecture...")
    fig2_autoencoder_architecture("figures/autoencoder_architecture.png")

    print("\n[Loading models and computing errors for all FSMs]")
    results = {}
    for name, cfg in FSM_CONFIGS.items():
        print(f"  {FSM_DISPLAY[name]}...")
        results[name] = _load_model_and_errors(name, cfg)

    print("\n[Fig 3] Reconstruction error histograms...")
    fig3_error_histograms(results, "figures/error_histograms.png")

    print("\n[Fig 4] ROC curves by FSM and fault type...")
    fig4_roc_combined(results, "figures/roc_combined.png")

    print("\n[Table 1] Combined results...")
    table1_results(
        results,
        out_csv="figures/results_table.csv",
        out_tex="figures/results_table.tex",
        out_txt="figures/results_table.txt",
    )

    print("Done.")


if __name__ == "__main__":
    main()
