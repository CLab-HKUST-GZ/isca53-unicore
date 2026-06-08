import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FONT_FAMILY = "Times New Roman"
FONT_WEIGHT_BOLD = "bold"

FONT_SIZE_AXIS_LABEL = 32
FONT_SIZE_TICK_LABEL = 24
FONT_SIZE_BASELINE_LABEL = 28
FONT_SIZE_BAR_VALUE = 24
FONT_SIZE_MODEL_LABEL = 28
FONT_SIZE_PRECISION_LABEL = 28
FONT_SIZE_LEGEND = 28


plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = FONT_FAMILY
plt.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
plt.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results_ddr4_8Gb_x8_3200_ctx_sweep_decode_bs128"
)


PRECISIONS = [
    ("w4a4", "W4A4"),
    ("w4a8", "W4A8"),
    ("w8a8", "W8A8"),
]

MODELS = [
    ("meta-llama/Llama-2-7b-hf", "Llama-2-7B"),
    ("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
]

CATEGORIES = [
    ("OliVe", "olive"),
    ("Tender", "tender"),
    ("M-ANT", "mant"),
    ("UniCore", "unicore"),
]


COLOR_CORE = "#1a759f"
COLOR_SRAM = "#5fa8d3"
COLOR_DRAM = "#cae9ff"
COLOR_SPEEDUP = "#1f2a44"


def _to_float(row, key):
    value = row.get(key, 0)
    return float(value) if value not in (None, "") else 0.0


def _to_int(row, key):
    value = row.get(key, 0)
    return int(value) if value not in (None, "") else 0


def load_results(results_dir, cycle_key="total_cycle"):
    data_by_cxt = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                config_name = row["config_name"]
                prefix, precision = config_name.rsplit("_", 1)
                model_name = row["model_name"]
                cxt_len = _to_int(row, "cxt_len")

                if cxt_len not in data_by_cxt:
                    data_by_cxt[cxt_len] = {}

                data_by_cxt[cxt_len][(prefix, precision, model_name)] = {
                    "is_generation": _to_int(row, "is_generation"),
                    "compute_uJ": _to_float(row, "compute_uJ"),
                    "sram_uJ": _to_float(row, "sram_uJ"),
                    "dram_uJ": _to_float(row, "dram_uJ"),
                    "total_uJ": _to_float(row, "total_uJ"),
                    "total_cycle": _to_float(row, cycle_key),
                }
    return data_by_cxt


def _required_keys():
    keys = []
    for precision, _ in PRECISIONS:
        for model_name, _ in MODELS:
            for _, prefix in CATEGORIES:
                keys.append((prefix, precision, model_name))
    return keys


def validate_results_for_plot(data, cxt_len):
    missing = [key for key in _required_keys() if key not in data]
    if missing:
        preview = ", ".join([str(item) for item in missing[:5]])
        if len(missing) > 5:
            preview += ", ..."
        raise ValueError(
            f"Missing {len(missing)} result entries for cxt_len={cxt_len}: {preview}"
        )

    non_decode = [
        key for key in _required_keys() if int(data[key].get("is_generation", 0)) != 1
    ]
    if non_decode:
        preview = ", ".join([str(item) for item in non_decode[:5]])
        if len(non_decode) > 5:
            preview += ", ..."
        raise ValueError(
            "plot/figure19-20.py expects decode data with is_generation=1. "
            f"Found {len(non_decode)} non-decode entries: {preview}"
        )


def build_plot_arrays(data):
    energy_stacks = []
    speedup_values = []
    model_labels_per_group = []

    for precision, _ in PRECISIONS:
        for model_name, model_label in MODELS:
            olive_total = data[("olive", precision, model_name)]["total_uJ"]
            olive_cycle = data[("olive", precision, model_name)]["total_cycle"]
            group_energy = []
            group_speedup = []

            for _, prefix in CATEGORIES:
                metrics = data[(prefix, precision, model_name)]
                dram_norm = metrics["dram_uJ"] / olive_total
                sram_norm = metrics["sram_uJ"] / olive_total
                core_norm = metrics["compute_uJ"] / olive_total
                group_energy.append([dram_norm, sram_norm, core_norm])

                cur_cycle = metrics["total_cycle"]
                group_speedup.append(olive_cycle / cur_cycle if cur_cycle > 0 else np.nan)

            energy_stacks.append(group_energy)
            speedup_values.append(group_speedup)
            model_labels_per_group.append(model_label)

    return np.array(energy_stacks), np.array(speedup_values), model_labels_per_group


def make_positions(num_groups, num_categories, group_spacing=0.8):
    positions = []
    offset = 0.0
    for _ in range(num_groups):
        x_group = np.arange(num_categories, dtype=float) + offset
        positions.append(x_group)
        offset += num_categories + group_spacing
    return positions


def draw_figure(
    energy_stacks,
    speedup_values,
    model_labels_per_group,
    output_pdf,
    output_png,
    y_label_prefix="",
):
    num_groups = energy_stacks.shape[0]
    num_categories = energy_stacks.shape[1]

    x_groups = make_positions(num_groups, num_categories, group_spacing=0.8)
    x_flat = np.concatenate(x_groups)

    fig, ax = plt.subplots(1, 1, figsize=(16, 5.5))

    # Draw energy bars
    bar_width = 0.75
    total_values = []
    for g, x_group in enumerate(x_groups):
        for c, x in enumerate(x_group):
            dram, sram, core = energy_stacks[g, c]
            total = dram + sram + core
            total_values.append(total)

            ax.bar(
                x,
                dram,
                width=bar_width,
                color=COLOR_DRAM,
                edgecolor="black",
                linewidth=0.8,
                hatch="",
            )
            ax.bar(
                x,
                sram,
                width=bar_width,
                bottom=dram,
                color=COLOR_SRAM,
                edgecolor="black",
                linewidth=0.8,
                hatch="\\",
            )
            ax.bar(
                x,
                core,
                width=bar_width,
                bottom=dram + sram,
                color=COLOR_CORE,
                edgecolor="black",
                linewidth=0.8,
                hatch="///",
            )
            ax.text(
                x,
                total + 0.02,
                f"{total:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=FONT_SIZE_BAR_VALUE,
                fontweight=FONT_WEIGHT_BOLD,
            )

    top_ylim = max(1.28, max(total_values) + 0.42)
    ax.set_ylim(0, top_ylim)
    ax.set_xlim(x_flat[0] - 0.8, x_flat[-1] + 0.8)

    ax.set_ylabel(
        f"{y_label_prefix} Normalized Energy" if y_label_prefix else "Normalized Energy",
        fontsize=FONT_SIZE_AXIS_LABEL,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Precision labels
    for p_idx, (_, precision_label) in enumerate(PRECISIONS):
        start_group = p_idx * len(MODELS)
        end_group = start_group + len(MODELS) - 1
        left = x_groups[start_group][0]
        right = x_groups[end_group][-1]
        center = 0.5 * (left + right)
        ax.text(
            center,
            top_ylim - 0.03,
            precision_label,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_PRECISION_LABEL,
            fontweight=FONT_WEIGHT_BOLD,
        )

    # Group separators
    for i in range(num_groups - 1):
        boundary = 0.5 * (x_groups[i][-1] + x_groups[i + 1][0])
        lw = 1.2 if (i % 2 == 0) else 1.0
        ax.axvline(boundary, color="black", linewidth=lw)

    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

    # X-axis labels
    ax.set_xticks(x_flat)
    ax.set_xticklabels(
        [cat for _ in range(num_groups) for cat, _ in CATEGORIES],
        rotation=90,
        fontsize=FONT_SIZE_BASELINE_LABEL,
    )

    # Model labels
    for g, x_group in enumerate(x_groups):
        center = 0.5 * (x_group[0] + x_group[-1])
        ax.text(
            center,
            -0.60,
            model_labels_per_group[g],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FONT_SIZE_MODEL_LABEL,
            fontweight=FONT_WEIGHT_BOLD,
            clip_on=False,
        )

    # Extend group boundaries below the axis so the baseline labels are grouped
    # under each model, matching the label treatment in Figures 17/18.
    x_left, x_right = ax.get_xlim()
    model_boundaries = [x_left]
    model_boundaries.extend(
        0.5 * (x_groups[i][-1] + x_groups[i + 1][0])
        for i in range(num_groups - 1)
    )
    model_boundaries.append(x_right)
    for boundary in model_boundaries:
        ax.axvline(
            boundary,
            color="black",
            linewidth=1.2,
            ymin=-0.70,
            ymax=0.0,
            clip_on=False,
        )

    # Speedup line on secondary y-axis
    ax_speed = ax.twinx()
    for g, x_group in enumerate(x_groups):
        y_group = speedup_values[g]
        ax_speed.plot(
            x_group,
            y_group,
            color=COLOR_SPEEDUP,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            markerfacecolor=COLOR_SPEEDUP,
            markeredgecolor=COLOR_SPEEDUP,
            zorder=10,
        )

    valid = speedup_values[np.isfinite(speedup_values)]
    speedup_ymax = max(1.2, float(valid.max()) + 0.2) if valid.size else 1.2
    ax_speed.set_ylim(0, speedup_ymax)
    ax_speed.set_ylabel(
        "Speedup", fontsize=FONT_SIZE_AXIS_LABEL, rotation=270, labelpad=34
    )
    ax_speed.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)
    ax_speed.grid(False)

    ax_speed.spines["right"].set_visible(True)
    ax_speed.spines["right"].set_edgecolor("black")
    ax_speed.spines["right"].set_linewidth(0.8)
    ax_speed.spines["top"].set_visible(False)
    ax_speed.spines["left"].set_visible(False)
    ax_speed.spines["bottom"].set_visible(False)

    # Legend
    legend_handles = [
        Patch(facecolor=COLOR_CORE, edgecolor="black", hatch="///", label="Core"),
        Patch(facecolor=COLOR_SRAM, edgecolor="black", hatch="\\", label="SRAM"),
        Patch(facecolor=COLOR_DRAM, edgecolor="black", hatch="", label="DRAM"),
        Line2D(
            [0],
            [0],
            color=COLOR_SPEEDUP,
            marker="o",
            linewidth=1.8,
            markersize=5.0,
            label="Speedup",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        frameon=True,
        edgecolor="black",
        fontsize=FONT_SIZE_LEGEND,
        borderpad=0.25,
    )

    fig.subplots_adjust(left=0.08, right=0.995, top=0.85, bottom=0.34)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the combined decode energy + speedup figure for Figure 19/20."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory containing CSV files.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path stem. If omitted, infer a figure name from the results directory.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="",
        help="Optional prefix for y-axis label (e.g., 'DDR4' -> 'DDR4 Normalized Energy').",
    )
    parser.add_argument(
        "--cxt-len",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sequence length(s) to plot. If omitted, generate one plot for each "
            "cxt_len found in CSV files."
        ),
    )
    parser.add_argument(
        "--cycle-key",
        type=str,
        default="total_cycle",
        help="CSV column used to compute speedup (ANT cycle / current cycle).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    plot_dir = Path(__file__).resolve().parent

    data_by_cxt = load_results(results_dir, cycle_key=args.cycle_key)

    if not data_by_cxt:
        raise ValueError(f"No valid rows found in {results_dir}")

    target_cxt_lens = args.cxt_len if args.cxt_len else sorted(data_by_cxt.keys())

    for cxt_len in target_cxt_lens:
        if cxt_len not in data_by_cxt:
            print(f"Warning: cxt_len={cxt_len} not found in results, skipping.")
            continue

        data = data_by_cxt[cxt_len]
        validate_results_for_plot(data, cxt_len)

        if args.output_stem is not None:
            output_stem = args.output_stem.resolve()
        else:
            dir_name = results_dir.name
            if "ddr4" in dir_name:
                output_stem = plot_dir.parent / "figure" / "figure19"
            elif "hbm2" in dir_name:
                output_stem = plot_dir.parent / "figure" / "figure20"
            else:
                output_stem = plot_dir.parent / "figure" / f"figure19-20_{dir_name}_cxt{cxt_len}"

        output_pdf = output_stem.with_suffix(".pdf")
        output_png = output_stem.with_suffix(".png")
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        energy_stacks, speedup_values, model_labels_per_group = build_plot_arrays(data)
        draw_figure(
            energy_stacks,
            speedup_values,
            model_labels_per_group,
            output_pdf,
            output_png,
            y_label_prefix=args.y_label,
        )


if __name__ == "__main__":
    main()
