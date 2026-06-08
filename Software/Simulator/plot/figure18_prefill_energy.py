import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


FONT_CONFIG = {
    'title': 30,
    'axis_label': 50,
    'legend': 48,
    'tick_label': 43,
    'bar_label': 38,
    'group_label': 44,
    'extra_label': 44,
}

STYLE_CONFIG = {
    'figsize': (24, 13.6),
    'bar_width': 0.4,
    'bar_spacing': 0.2,
    'label_offset': 0.02,
    'group_label_offset_x': 0.5,
    'group_label_offset_y': -0.68,
    'group_spacing': 0.2,
    'margin': 0.2,
    'boundary_linewidth': 1.5,
}

MODEL_ORDER = [
    ('meta-llama/Llama-2-7b-hf', 'Llama-2-7B'),
    ('meta-llama/Meta-Llama-3-8B', 'Llama-3-8B'),
]
PRECISION_ORDER = [
    ('w4a4', 'W4A4'),
    ('w4a8', 'W4A8'),
    ('w8a8', 'W8A8'),
]
CATEGORY_ORDER = [
    ('OliVe', 'olive'),
    ('Tender', 'tender'),
    ('M-ANT', 'mant'),
    ('UniCore', 'unicore'),
]

ENERGY_TEXTURES = {
    'DRAM': {'color': '#cae9ff', 'hatch': ''},
    'SRAM': {'color': '#5fa8d3', 'hatch': '\\\\'},
    'Core': {'color': '#1a759f', 'hatch': '/'},
}
TOPSW_TEXTURES = {
    'TOPS/W': {'color': '#236AB3', 'hatch': '|'},
}
TOPSW_CATEGORY_TEXTURES = {
    'OliVe': {'color': '#dbe8f4', 'hatch': '|', 'edgecolor': '#1f2a44'},
    'Tender': {'color': '#dbe8f4', 'hatch': '|', 'edgecolor': '#1f2a44'},
    'M-ANT': {'color': '#dbe8f4', 'hatch': '|', 'edgecolor': '#1f2a44'},
    'UniCore': {'color': '#3d7db8', 'hatch': '|', 'edgecolor': 'black'},
}


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    plot_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            'Generate the legacy prefill energy figure (Figure 18 style) '
            'from uni_sim CSV results.'
        )
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=repo_root / 'results',
        help='Directory containing prefill CSV results.',
    )
    parser.add_argument(
        '--cxt-len',
        type=int,
        default=None,
        help='Sequence length to plot. If omitted, use the only cxt_len or the max one.',
    )
    parser.add_argument(
        '--output-stem',
        type=Path,
        default=repo_root / 'figure' / 'figure18',
        help='Output path stem. The script writes both .pdf and .png.',
    )
    return parser.parse_args()


def _to_float(row, key):
    value = row.get(key, 0)
    return float(value) if value not in (None, '') else 0.0


def _to_int(row, key):
    value = row.get(key, 0)
    return int(value) if value not in (None, '') else 0


def load_results(results_dir):
    data_by_cxt = {}
    for csv_path in sorted(results_dir.glob('*.csv')):
        with csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config_name = row['config_name']
                prefix, precision = config_name.rsplit('_', 1)
                model_name = row['model_name']
                cxt_len = _to_int(row, 'cxt_len')

                if cxt_len not in data_by_cxt:
                    data_by_cxt[cxt_len] = {}

                data_by_cxt[cxt_len][(prefix, precision, model_name)] = {
                    'is_generation': _to_int(row, 'is_generation'),
                    'compute_uJ': _to_float(row, 'compute_uJ'),
                    'sram_uJ': _to_float(row, 'sram_uJ'),
                    'dram_uJ': _to_float(row, 'dram_uJ'),
                    'total_uJ': _to_float(row, 'total_uJ'),
                }
    return data_by_cxt


def resolve_target_cxt_len(data_by_cxt, requested_cxt_len):
    available = sorted(data_by_cxt.keys())
    if not available:
        raise ValueError('No valid rows found in the results directory.')

    if requested_cxt_len is not None:
        if requested_cxt_len not in data_by_cxt:
            raise ValueError(
                f'cxt_len={requested_cxt_len} not found. Available values: {available}'
            )
        return requested_cxt_len

    if len(available) == 1:
        return available[0]

    chosen = available[-1]
    print(
        f'No --cxt-len specified. Multiple values found {available}; '
        f'using the largest cxt_len={chosen}.'
    )
    return chosen


def validate_results(data):
    missing = []
    non_prefill = []
    for precision, _ in PRECISION_ORDER:
        for model_name, _ in MODEL_ORDER:
            for _, prefix in CATEGORY_ORDER:
                key = (prefix, precision, model_name)
                if key not in data:
                    missing.append(key)
                elif int(data[key].get('is_generation', 0)) != 0:
                    non_prefill.append(key)

    if missing:
        preview = ', '.join(str(item) for item in missing[:6])
        if len(missing) > 6:
            preview += ', ...'
        raise ValueError(f'Missing {len(missing)} result entries: {preview}')

    if non_prefill:
        preview = ', '.join(str(item) for item in non_prefill[:6])
        if len(non_prefill) > 6:
            preview += ', ...'
        raise ValueError(
            'plot/figure18_prefill_energy.py expects prefill data with is_generation=0. '
            f'Found {len(non_prefill)} non-prefill entries: {preview}'
        )


def build_plot_arrays(data):
    groups = []
    extra_labels = [label for _, label in PRECISION_ORDER]
    categories = [label for label, _ in CATEGORY_ORDER]

    energy_values = []
    topsw_values = []

    for precision, _ in PRECISION_ORDER:
        for model_name, model_label in MODEL_ORDER:
            groups.append(model_label)
            olive_metrics = data[('olive', precision, model_name)]
            olive_total = olive_metrics['total_uJ']

            energy_group = []
            topsw_group = []
            for _, prefix in CATEGORY_ORDER:
                metrics = data[(prefix, precision, model_name)]
                dram_norm = metrics['dram_uJ'] / olive_total
                sram_norm = metrics['sram_uJ'] / olive_total
                core_norm = metrics['compute_uJ'] / olive_total
                energy_group.append([dram_norm, sram_norm, core_norm])
                topsw_group.append([olive_total / metrics['total_uJ']])

            energy_values.append(energy_group)
            topsw_values.append(topsw_group)

    return groups, extra_labels, categories, np.array(energy_values), np.array(topsw_values)


def create_stacked_subplot(
    ax,
    groups,
    categories,
    label_ax,
    extra_labels,
    stacked_values,
    ylabel,
    sub_textures,
    ylabel_position=None,
    ylimit=(0, 1),
    label_limit=1,
    fraction_bits=2,
    show_labels=True,
    show_group_labels=True,
    show_extra_labels=True,
    show_extra_lines=True,
    bar_width=0.6,
    bar_spacing=0.8,
    group_spacing=1,
    margin=0.02,
    label_offset=0.05,
    group_label_offset_x=0.42,
    group_label_offset_y=0.15,
    y_ticks=None,
    category_textures=None,
):
    x_positions = []
    offset = 0
    for _ in groups:
        x = np.arange(len(categories)) * (bar_width + bar_spacing) + offset
        x_positions.append(x)
        offset += len(categories) * (bar_width + bar_spacing) + group_spacing
    x_positions_flat = np.concatenate(x_positions)

    for x_group, val_group in zip(x_positions, stacked_values):
        for cat_idx, (x_bar, val_bar) in enumerate(zip(x_group, val_group)):
            bottom = 0
            for i, sub_val in enumerate(val_bar):
                if category_textures is not None:
                    texture = category_textures[categories[cat_idx]]
                else:
                    sub_cat = list(sub_textures.keys())[i]
                    texture = sub_textures[sub_cat]
                ax.bar(
                    x_bar,
                    sub_val,
                    width=bar_width,
                    bottom=bottom,
                    color=texture['color'],
                    hatch=texture['hatch'],
                    edgecolor=texture.get('edgecolor', 'black'),
                    linewidth=0.8,
                )
                bottom += sub_val

            if bottom < label_limit:
                format_str = f'{{:.{fraction_bits}f}}'
                ax.text(
                    x_bar,
                    bottom + label_offset * (ylimit[1] - ylimit[0]),
                    format_str.format(bottom),
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=FONT_CONFIG['bar_label'],
                    rotation=90,
                )

    ax.set_xlim(
        x_positions_flat[0] - bar_width / 2 - margin,
        x_positions_flat[-1] + bar_width / 2 + margin,
    )

    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'], labelpad=14)
    if ylabel_position is not None:
        ax.yaxis.set_label_coords(ylabel_position[0], ylabel_position[1])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label'])
    ax.set_ylim(ylimit[0], ylimit[1])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if show_labels:
        ax.set_xticks(x_positions_flat)
        repeated_labels = [cat for _ in groups for cat in categories]
        ax.set_xticklabels(
            repeated_labels,
            fontsize=FONT_CONFIG['tick_label'],
            rotation=90,
        )
        ax.tick_params(axis='x', length=0, pad=5)
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if show_group_labels:
        x_min = [x.min() for x in x_positions]
        x_range = [x[-1] - x[0] for x in x_positions]
        y_min, y_max = ax.get_ylim()
        label_y = y_min + (y_max - y_min) * group_label_offset_y

        for x_start, width, group in zip(x_min, x_range, groups):
            label_x = x_start + (width * group_label_offset_x)
            ax.text(
                label_x,
                label_y,
                group,
                ha='center',
                va='top',
                fontsize=FONT_CONFIG['group_label'],
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', pad=0),
            )

        for i in range(len(groups) - 1):
            boundary = (x_positions[i][-1] + x_positions[i + 1][0]) / 2
            ax.axvline(
                x=boundary,
                color='black',
                linewidth=STYLE_CONFIG['boundary_linewidth'],
                ymin=-1,
                ymax=0,
                clip_on=False,
            )
        x_left, x_right = ax.get_xlim()
        ax.axvline(
            x=x_left,
            color='black',
            linewidth=STYLE_CONFIG['boundary_linewidth'],
            ymin=-1,
            ymax=0,
            clip_on=False,
        )
        ax.axvline(
            x=x_right,
            color='black',
            linewidth=STYLE_CONFIG['boundary_linewidth'],
            ymin=-1,
            ymax=0,
            clip_on=False,
        )

    label_ax.axis('off')
    label_ax.set_xlim(
        x_positions_flat[0] - bar_width / 2 - margin,
        x_positions_flat[-1] + bar_width / 2 + margin,
    )
    for i, x_group in enumerate(x_positions):
        if i < len(x_positions) - 1:
            x_extra_label = (x_group[-1] + x_positions[i + 1][0]) / 2
        else:
            x_extra_label = x_group[-1] + (bar_width + group_spacing + margin) / 2
        width = x_group[-1] - x_group[0]

        if show_extra_labels:
            if i % 2 == 0 and i != len(x_positions) - 1:
                label_ax.text(
                    x_extra_label,
                    0.5,
                    extra_labels[i // 2],
                    fontsize=FONT_CONFIG['extra_label'],
                    fontweight='bold',
                    ha='center',
                    va='top',
                )

        if show_extra_lines and i % 2 == 1:
            ax.axvline(
                x=x_extra_label,
                color='black',
                linewidth=STYLE_CONFIG['boundary_linewidth'],
                ymin=0,
                ymax=1,
                clip_on=False,
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(STYLE_CONFIG['boundary_linewidth'])


def draw_figure(groups, extra_labels, categories, energy_values, topsw_values, output_stem):
    fig = plt.figure(figsize=STYLE_CONFIG['figsize'])
    gs = GridSpec(5, 1, height_ratios=[0.03, 0.08, 0.50, 0.03, 0.36], figure=fig)
    legend_ax = fig.add_subplot(gs[0])
    label_ax = fig.add_subplot(gs[1])
    main_ax0 = fig.add_subplot(gs[2])
    space_ax = fig.add_subplot(gs[3])
    main_ax1 = fig.add_subplot(gs[4])

    legend_ax.axis('off')
    space_ax.axis('off')

    legend_elements = []
    for sub_cat, texture in ENERGY_TEXTURES.items():
        legend_elements.append(
            Patch(
                facecolor=texture['color'],
                hatch=texture['hatch'],
                edgecolor='black',
                linewidth=0.8,
                label=sub_cat,
            )
        )
    legend_elements.reverse()

    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=3,
        frameon=True,
        fontsize=FONT_CONFIG['legend'],
        title_fontsize=FONT_CONFIG['legend'],
        labelspacing=0.2,
        handlelength=1.8,
        handletextpad=0.6,
        edgecolor='black',
        columnspacing=2.05,
        borderpad=0.25,
    )

    create_stacked_subplot(
        main_ax0,
        groups=groups,
        categories=categories,
        label_ax=label_ax,
        extra_labels=extra_labels,
        stacked_values=energy_values,
        ylabel='Normalized Energy',
        sub_textures=ENERGY_TEXTURES,
        ylimit=(0, 1.74),
        label_limit=1.74,
        fraction_bits=2,
        show_labels=False,
        show_group_labels=False,
        show_extra_labels=True,
        show_extra_lines=True,
        bar_width=STYLE_CONFIG['bar_width'],
        bar_spacing=STYLE_CONFIG['bar_spacing'],
        group_spacing=STYLE_CONFIG['group_spacing'],
        margin=STYLE_CONFIG['margin'],
        label_offset=STYLE_CONFIG['label_offset'],
        group_label_offset_x=STYLE_CONFIG['group_label_offset_x'],
        group_label_offset_y=STYLE_CONFIG['group_label_offset_y'],
        y_ticks=[0.0, 0.5, 1.0, 1.5],
    )

    create_stacked_subplot(
        main_ax1,
        groups=groups,
        categories=categories,
        label_ax=label_ax,
        extra_labels=extra_labels,
        stacked_values=topsw_values,
        ylabel='Normalized TOPS/W',
        sub_textures=TOPSW_TEXTURES,
        ylabel_position=(-0.040, 0.12),
        ylimit=(0, 4.0),
        label_limit=4.0,
        fraction_bits=2,
        show_labels=True,
        show_group_labels=True,
        show_extra_labels=False,
        show_extra_lines=True,
        bar_width=STYLE_CONFIG['bar_width'],
        bar_spacing=STYLE_CONFIG['bar_spacing'],
        group_spacing=STYLE_CONFIG['group_spacing'],
        margin=STYLE_CONFIG['margin'],
        label_offset=STYLE_CONFIG['label_offset'],
        group_label_offset_x=STYLE_CONFIG['group_label_offset_x'],
        group_label_offset_y=STYLE_CONFIG['group_label_offset_y'],
        y_ticks=[0, 2, 4],
        category_textures=TOPSW_CATEGORY_TEXTURES,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.10)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_stem.with_suffix('.pdf')
    output_png = output_stem.with_suffix('.png')
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)
    print(f'Figure saved to: {output_pdf}')
    print(f'Figure saved to: {output_png}')


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_stem = args.output_stem.resolve()

    data_by_cxt = load_results(results_dir)
    cxt_len = resolve_target_cxt_len(data_by_cxt, args.cxt_len)
    data = data_by_cxt[cxt_len]
    validate_results(data)

    groups, extra_labels, categories, energy_values, topsw_values = build_plot_arrays(data)
    print(f'Using results_dir={results_dir}')
    print(f'Using cxt_len={cxt_len}')
    draw_figure(groups, extra_labels, categories, energy_values, topsw_values, output_stem)


if __name__ == '__main__':
    main()
