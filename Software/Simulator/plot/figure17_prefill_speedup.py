import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Keep the legacy prefill style exactly aligned with the original prefill speedup plot.
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Font configuration copied from the legacy script.
font_config = {
    'title': 24,
    'axis_label': 30,
    'legend': 30,
    'tick_label': 28,
    'bar_label': 25,
    'group_label': 28,
    'extra_label': 32,
}

# Layout / category ordering copied from the legacy script.
groups_single = ['W4A4', 'W4A8', 'W8A8', 'W16A16', 'Avg.']
categories = ['OliVe', 'Tender', 'M-ANT', 'UniCore']
category_prefixes = ['olive', 'tender', 'mant', 'unicore']
w16_visible_prefixes = ['tender', 'unicore']
model_order = [
    ('meta-llama/Llama-2-7b-hf', 'Llama-2-7B'),
    ('meta-llama/Meta-Llama-3-8B', 'Llama-3-8B'),
]
base_precisions = ['w4a4', 'w4a8', 'w8a8']
w16_precision = 'w16a16'

style_config = {
    'figsize': (15, 8),
    'bar_width': 0.55,
    'bar_spacing': 0.2,
    'label_offset': 0.3,
    'group_label_offset_x': 0.5,
    'group_label_offset_y': -0.40,
    'group_spacing': 0.2,
    'margin': 0.2,
    'boundary_linewidth': 1.5,
}

texture_normal = {'color': '#cae9ff', 'hatch': '//'}
texture_unicore = {'color': '#5fa8d3', 'hatch': '\\\\'}


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    plot_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            'Generate the legacy prefill speedup figure (Figure 17 style) '
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
        default=repo_root / 'figure' / 'figure17',
        help='Output path stem. The script writes both .pdf and .png.',
    )
    parser.add_argument(
        '--w16a16-reference',
        choices=['tender', 'unicore'],
        default='tender',
        help=(
            'Reference method for W16A16 speedup values. Default: tender, '
            'because only Tender and UniCore have W16A16 simulator configs.'
        ),
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
                    'total_cycle': _to_float(row, 'total_cycle'),
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

    for model_name, _ in model_order:
        for precision in base_precisions:
            for prefix in category_prefixes:
                key = (prefix, precision, model_name)
                if key not in data:
                    missing.append(key)
                elif data[key]['is_generation'] != 0:
                    non_prefill.append(key)

        for prefix in ['tender', 'unicore']:
            key = (prefix, w16_precision, model_name)
            if key not in data:
                missing.append(key)
            elif data[key]['is_generation'] != 0:
                non_prefill.append(key)

    if missing:
        preview = ', '.join(str(item) for item in missing[:8])
        if len(missing) > 8:
            preview += ', ...'
        raise ValueError(f'Missing {len(missing)} result entries: {preview}')

    if non_prefill:
        preview = ', '.join(str(item) for item in non_prefill[:8])
        if len(non_prefill) > 8:
            preview += ', ...'
        raise ValueError(
            'plot/figure17_prefill_speedup.py expects prefill data with is_generation=0. '
            f'Found {len(non_prefill)} non-prefill entries: {preview}'
        )


def build_speedup_matrix(data, model_name, w16a16_reference):
    cycle_discrim = []
    for prefix in category_prefixes:
        cycle_discrim.append(
            np.array([data[(prefix, precision, model_name)]['total_cycle'] for precision in base_precisions])
        )
    cycle_discrim = np.array(cycle_discrim)

    cycle_olive = np.array([
        data[('olive', precision, model_name)]['total_cycle'] for precision in base_precisions
    ])
    speedup_discrim = cycle_olive / cycle_discrim

    w16_row = np.full(len(category_prefixes), np.nan)
    # W16A16 only has Tender / UniCore simulator data, so keep the existing
    # reference behavior for that column instead of forcing an unavailable
    # OliVe baseline.
    ref_cycle = data[(w16a16_reference, w16_precision, model_name)]['total_cycle']
    for prefix in w16_visible_prefixes:
        idx = category_prefixes.index(prefix)
        cur_cycle = data[(prefix, w16_precision, model_name)]['total_cycle']
        w16_row[idx] = ref_cycle / cur_cycle if cur_cycle > 0 else np.nan

    speedup_discrim = np.concatenate(
        (speedup_discrim, w16_row.reshape(-1, 1)),
        axis=1,
    )

    geomean_col = []
    for i, prefix in enumerate(category_prefixes):
        if prefix in w16_visible_prefixes:
            geomean_col.append(np.nanmean(speedup_discrim[i, :]))
        else:
            geomean_col.append(np.mean(speedup_discrim[i, :3]))
    speedup_discrim = np.concatenate(
        (speedup_discrim, np.array(geomean_col).reshape(-1, 1)),
        axis=1,
    )
    return speedup_discrim.T


def create_stacked_subplot(
    ax,
    data_matrix,
    subplot_title,
    ylabel,
    is_bottom_plot=True,
    bar_width=0.6,
    bar_spacing=0.8,
    group_spacing=1,
    label_offset=0.05,
    group_label_offset_x=0.42,
    group_label_offset_y=0.15,
    margin=0.02,
):
    x_positions = []
    offset = 0
    for group in groups_single:
        if group == 'W16A16':
            x = np.full(len(categories), offset, dtype=float)
            for visible_idx, prefix in enumerate(w16_visible_prefixes):
                category_idx = category_prefixes.index(prefix)
                x[category_idx] = offset + visible_idx * (bar_width + bar_spacing)
            x_positions.append(x)
            w16_actual_width = (
                len(w16_visible_prefixes) * bar_width
                + (len(w16_visible_prefixes) - 1) * bar_spacing
            )
            # Keep the two available W16A16 bars centered between neighboring groups.
            offset += w16_actual_width + group_spacing + bar_spacing
        else:
            x = np.arange(len(categories)) * (bar_width + bar_spacing) + offset
            x_positions.append(x)
            offset += len(categories) * (bar_width + bar_spacing) + group_spacing

    x_positions_flat = np.concatenate(x_positions)

    for x_group, val_group in zip(x_positions, data_matrix):
        for i, (x_bar, val_bar) in enumerate(zip(x_group, val_group)):
            if np.isnan(val_bar):
                continue

            if categories[i] == 'UniCore':
                current_color = texture_unicore['color']
                current_hatch = texture_unicore['hatch']
            else:
                current_color = texture_normal['color']
                current_hatch = texture_normal['hatch']

            bottom = 0
            ax.bar(
                x_bar,
                val_bar,
                width=bar_width,
                bottom=bottom,
                color=current_color,
                hatch=current_hatch,
                edgecolor='black',
                linewidth=0.8,
            )
            bottom += val_bar

            ax.text(
                x_bar,
                bottom + label_offset,
                f'{bottom:.2f}',
                ha='center',
                va='bottom',
                fontsize=font_config['bar_label'],
                rotation=90,
            )

    first_visible_x = x_positions[0][0]
    last_visible_x = x_positions_flat[-1]
    ax.set_xlim(
        first_visible_x - bar_width / 2 - margin,
        last_visible_x + bar_width / 2 + margin,
    )

    xticklabels = []
    xticks_filtered = []
    for i, group in enumerate(groups_single):
        x_group = x_positions[i]
        if group == 'W16A16':
            for j, cat in enumerate(categories):
                if category_prefixes[j] in w16_visible_prefixes:
                    xticklabels.append(cat)
                    xticks_filtered.append(x_group[j])
        else:
            xticklabels.extend(categories)
            xticks_filtered.extend(x_group)

    ax.set_xticks(xticks_filtered)

    if is_bottom_plot:
        ax.set_xticklabels(xticklabels, fontsize=font_config['tick_label'], rotation=90)
        ax.tick_params(axis='x', length=0, pad=15)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    ax.xaxis.set_ticks_position('bottom')

    ax.set_ylabel(ylabel, fontsize=font_config['axis_label'], labelpad=12)
    ax.tick_params(axis='y', labelsize=font_config['tick_label'])
    ax.set_ylim(0, 9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    vline_ymin = -0.7 if is_bottom_plot else 0
    for i in range(len(groups_single) - 1):
        boundary = (x_positions[i][-1] + x_positions[i + 1][0]) / 2
        ax.axvline(
            x=boundary,
            color='black',
            linewidth=style_config['boundary_linewidth'],
            ymin=vline_ymin,
            ymax=1,
            clip_on=False,
        )

    x_min, x_max = ax.get_xlim()
    ax.axvline(
        x=x_min,
        color='black',
        linewidth=style_config['boundary_linewidth'],
        ymin=vline_ymin,
        ymax=1,
        clip_on=False,
    )
    ax.axvline(
        x=x_max,
        color='black',
        linewidth=style_config['boundary_linewidth'],
        ymin=vline_ymin,
        ymax=1,
        clip_on=False,
    )

    if is_bottom_plot:
        x_min_pos = [x.min() for x in x_positions]
        x_range = [x[-1] - x[0] for x in x_positions]
        label_y = -6

        for idx, (x, width, group) in enumerate(zip(x_min_pos, x_range, groups_single)):
            label_x = x + (width * group_label_offset_x)
            ax.text(
                label_x,
                label_y,
                group,
                ha='center',
                va='top',
                fontsize=font_config['group_label'],
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', pad=3),
            )

    ax.text(
        (x_min + x_max) / 2,
        8,
        subplot_title,
        fontsize=font_config['extra_label'],
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(facecolor='white', edgecolor='none', pad=5),
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(style_config['boundary_linewidth'])


def draw_figure(data_top, data_bottom, output_stem):
    fig = plt.figure(figsize=style_config['figsize'])
    gs = fig.add_gridspec(2, 1, hspace=0.05)

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    create_stacked_subplot(
        ax_top,
        data_matrix=data_top,
        subplot_title='Llama-2-7B',
        ylabel='Speedup',
        is_bottom_plot=False,
        bar_width=style_config['bar_width'],
        bar_spacing=style_config['bar_spacing'],
        label_offset=style_config['label_offset'],
        group_label_offset_x=style_config['group_label_offset_x'],
        group_label_offset_y=style_config['group_label_offset_y'],
        group_spacing=style_config['group_spacing'],
        margin=style_config['margin'],
    )

    create_stacked_subplot(
        ax_bottom,
        data_matrix=data_bottom,
        subplot_title='Llama-3-8B',
        ylabel='Speedup',
        is_bottom_plot=True,
        bar_width=style_config['bar_width'],
        bar_spacing=style_config['bar_spacing'],
        label_offset=style_config['label_offset'],
        group_label_offset_x=style_config['group_label_offset_x'],
        group_label_offset_y=style_config['group_label_offset_y'] - 0.1,
        group_spacing=style_config['group_spacing'],
        margin=style_config['margin'],
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_stem.with_suffix('.png')
    output_pdf = output_stem.with_suffix('.pdf')
    plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f'Figure saved to: {output_pdf}')
    print(f'Figure saved to: {output_png}')


def main():
    args = parse_args()
    data_by_cxt = load_results(args.results_dir)
    cxt_len = resolve_target_cxt_len(data_by_cxt, args.cxt_len)
    data = data_by_cxt[cxt_len]
    validate_results(data)

    data_top = build_speedup_matrix(data, model_order[0][0], args.w16a16_reference)
    data_bottom = build_speedup_matrix(data, model_order[1][0], args.w16a16_reference)

    print(f'Using results_dir={args.results_dir.resolve()}')
    print(f'Using cxt_len={cxt_len}')
    print(f'Using w16a16_reference={args.w16a16_reference}')
    draw_figure(data_top, data_bottom, args.output_stem)


if __name__ == '__main__':
    main()
