"""Plotting functions for boxplots and oncoplots."""

import os
import datetime
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pyoncoprint
from matplotlib.backends.backend_pdf import PdfPages

from features import rows_for_block
from utils import sig_figs


def _safe_filename(s):
    """Sanitize string for use in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _plot_title_cancer_label(cancer_types):
    """Strip ``column=value`` to ``value`` only; join multiple with ``; ``."""
    if not cancer_types:
        return "Cohort"
    parts = []
    for c in cancer_types:
        s = str(c)
        if "=" in s:
            parts.append(s.split("=", 1)[1].strip())
        else:
            parts.append(s.strip())
    return "; ".join(parts) if parts else "Cohort"


def _build_plot_path(
    target_gene, ko_datatype, logic_type, cancer_types, suffix, output_root=".", extension=".pdf"
):
    """Build an output path under ``output_root`` (caller supplies e.g. ``.../TARGET/DATASET/``)."""
    plot_dir = output_root
    os.makedirs(plot_dir, exist_ok=True)
    cancer_types = cancer_types or []
    cancer_str = "_".join(_safe_filename(c) for c in cancer_types) if cancer_types else "all"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    pdf_path = os.path.join(
        plot_dir,
        f"{date_str}_{target_gene}_{ko_datatype}_{logic_type}_{cancer_str}_{suffix}{extension}",
    )
    return plot_dir, pdf_path


def summary_csv_path(target_gene, ko_datatype, logic_type, cancer_types, output_root="."):
    """Same name as the boxplot PDF, with ``_summary.csv`` in place of ``_boxplots.pdf``."""
    _, pdf_path = _build_plot_path(
        target_gene, ko_datatype, logic_type, cancer_types, "boxplots", output_root=output_root
    )
    return pdf_path[: -len("_boxplots.pdf")] + "_summary.csv"


def create_boxplot(
    tree=None,
    fdf=None,
    tdf=None,
    ko_datatype=None,
    target_gene=None,
    logic_type="AND",
    cancer_types=None,
    output_root=".",
    plot_title_cancer: str | None = None,
):
    """Create boxplots for feature progression. Saves to PDF under ``output_root``.

    Title uses ``plot_title_cancer`` if set, else values from ``cancer_types`` without
    the ``oncotree_column=`` prefix (see ``_plot_title_cancer_label``).
    """
    _, pdf_path = _build_plot_path(
        target_gene, ko_datatype, logic_type, cancer_types, "boxplots", output_root=output_root
    )

    cohort_cols = sorted(
        set(tdf["cellline"].astype(str).unique()).intersection(fdf.columns.astype(str))
    )
    fdf_plot = fdf.reindex(columns=cohort_cols)

    sig_feature_list_of_list = tree.convert_to_node_paths()

    with PdfPages(pdf_path) as pdf:
        for set_idx, cur_set in enumerate(sig_feature_list_of_list):
            feature_nodes = cur_set[1:]
            if not feature_nodes:
                continue
            plot_data = []
            p_values = {}
            logic_map = {}

            if logic_type.upper() == "AND":
                cumulative_mask = pd.Series(True, index=fdf_plot.columns)
            else:
                cumulative_mask = pd.Series(False, index=fdf_plot.columns)

            first_block_feats = feature_nodes[0].feature
            wt_mask = ~rows_for_block(fdf_plot, first_block_feats).fillna(0).any()
            wt_series = tdf.loc[tdf["cellline"].isin(fdf_plot.columns[wt_mask]), ko_datatype]
            n_wt = int(len(wt_series))
            wt_group_label = f"WT (n={n_wt})"
            for val in wt_series:
                plot_data.append({"Group": wt_group_label, "Value": val, "Kind": "WT"})
            logic_map[wt_group_label] = f"Baseline (WT), n={n_wt} cell lines"
            previous_mask = None

            for step_idx, node in enumerate(feature_nodes):
                block_string = node.feature
                is_resistant = getattr(node, "sens_res", "sensitive") == "resistant"
                block_mask = rows_for_block(fdf_plot, block_string).fillna(0).any()
                if logic_type.upper() == "AND":
                    cumulative_mask = cumulative_mask & block_mask
                else:
                    cumulative_mask = cumulative_mask | block_mask
                group_celllines = fdf_plot.columns[cumulative_mask]
                group_series = tdf.loc[tdf["cellline"].isin(group_celllines), ko_datatype]
                features = [part.strip() for part in str(block_string).split(";") if part.strip()]
                short_label = f"Step {step_idx + 1}\n(n={len(group_series)})"
                step_key = f"Step {step_idx + 1}"
                logic_map[step_key] = " | ".join(features)
                kind = "resistant" if is_resistant else "sensitive"
                for val in group_series:
                    plot_data.append({"Group": short_label, "Value": val, "Kind": kind})
                if is_resistant and previous_mask is not None:
                    reference = tdf.loc[tdf["cellline"].isin(fdf_plot.columns[previous_mask]), ko_datatype]
                    if not group_series.empty and not reference.empty:
                        _, p_val = stats.ranksums(group_series, reference, alternative="greater")
                        p_values[short_label] = sig_figs(p_val, 3)
                elif is_resistant and not group_series.empty and not wt_series.empty:
                    _, p_val = stats.ranksums(group_series, wt_series, alternative="greater")
                    p_values[short_label] = sig_figs(p_val, 3)
                elif not group_series.empty and not wt_series.empty:
                    _, p_val = stats.ranksums(group_series, wt_series, alternative="less")
                    p_values[short_label] = sig_figs(p_val, 3)
                previous_mask = cumulative_mask.copy()

            if not plot_data:
                continue
            df_plot = pd.DataFrame(plot_data)
            fig = plt.figure(figsize=(14, 7))
            ax = sns.boxplot(
                data=df_plot, x="Group", y="Value", hue="Group", palette="viridis", legend=False
            )
            sns.stripplot(
                data=df_plot,
                x="Group",
                y="Value",
                hue="Kind",
                hue_order=["WT", "sensitive", "resistant"],
                palette={"WT": "0.15", "sensitive": "#d62728", "resistant": "#1f77b4"},
                dodge=False,
                legend=False,
                alpha=0.45,
                size=4,
            )
            for i, group in enumerate(df_plot["Group"].unique()):
                if group in p_values:
                    y_max = df_plot[df_plot["Group"] == group]["Value"].max()
                    ax.text(
                        i,
                        y_max + (ax.get_ylim()[1] * 0.02),
                        f"p={p_values[group]}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=9,
                    )
            legend_info = (
                f"STEP ADDITIONS (cumulative mask: {logic_type.upper()} between blocks)\n"
                + "Red points: sensitive. Blue points: resistant.\n"
                + "Resistant p is the upper tail versus the combined parent cell lines.\n"
                + "-" * 25
                + "\n"
            )
            for label, logic in logic_map.items():
                wrapped_logic = "\n    ".join(textwrap.wrap(logic, width=40))
                legend_info += f"{label}:\n    {wrapped_logic}\n\n"
            plt.annotate(
                legend_info,
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                verticalalignment="center",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.5", alpha=0.9),
            )
            cancer_title = (
                plot_title_cancer
                if plot_title_cancer is not None
                else _plot_title_cancer_label(cancer_types)
            )
            plt.title(
                f"{target_gene} {cancer_title}: {logic_type.upper()} Progression",
                fontsize=14,
            )
            plt.ylabel(f"{target_gene} {ko_datatype} Essentiality")
            plt.tight_layout(rect=[0, 0, 0.70, 1])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved boxplots to {pdf_path}")


def create_oncoplot(
    target_gene=None,
    ko_datatype=None,
    fdf=None,
    tdf=None,
    tree=None,
    vmin=-2,
    vmax=2,
    logic_type="OR",
    cancer_types=None,
    output_root=".",
):
    """Create oncoplots for each path in the feature tree. Saves to PDF under ``output_root``."""
    if logic_type.upper() not in ("AND", "OR"):
        raise ValueError(f"logic_type must be 'AND' or 'OR'. Got: {logic_type}")

    tdf_clean = tdf.dropna(subset=[ko_datatype]).copy()
    plot_samples = sorted(
        set(tdf_clean["cellline"].astype(str).unique()).intersection(fdf.columns.astype(str))
    )
    fdf_plot = fdf.reindex(columns=plot_samples)
    tdf_plot = tdf_clean[tdf_clean["cellline"].astype(str).isin(plot_samples)].copy()

    ohd = tdf_plot[["cellline", ko_datatype]].rename(
        columns={"cellline": "sample", ko_datatype: target_gene + "_" + ko_datatype}
    )
    ohd = (
        pd.pivot_table(
            ohd, columns=["sample"], values=target_gene + "_" + ko_datatype
        )
        .rename_axis("track_name", axis=1)
    )
    ohd.sort_values(by=target_gene + "_" + ko_datatype, axis=1, inplace=True)

    heatmaps = {
        target_gene + " " + ko_datatype: {
            "heatmap": ohd.iloc[:1],
            "cmap": "RdBu_r",
            "vmin": vmin,
            "vmax": vmax,
        }
    }
    mutation_markers = {"MUT": dict(marker="fill", color="black", zindex=0)}

    _, pdf_path = _build_plot_path(
        target_gene, ko_datatype, logic_type, cancer_types, "oncoplots", output_root=output_root
    )

    with PdfPages(pdf_path) as pdf:
        for path_idx, path in enumerate(tree.convert_to_node_paths()):
            feature_nodes = path[1:]
            if not feature_nodes:
                continue
            path_label = " -> ".join(node.feature for node in feature_nodes).replace(";", " ")
            block_rows = []
            block_labels = []
            group_defs = []
            for node in feature_nodes:
                block = [part.strip() for part in node.feature.split(";") if part.strip()]
                if not block:
                    continue
                block_label = "Group " + block[0]
                block_row = rows_for_block(fdf_plot, node.feature).fillna(0).max(axis=0)
                block_rows.append(block_row)
                block_labels.append(block_label)
                group_defs.append(f"{block_label} = {', '.join(block)}")
            cur_muts_df = pd.DataFrame(block_rows, index=block_labels)
            cur_muts_df.index.name = "track_name"
            union_label = (
                "Feature Intersection" if logic_type.upper() == "AND" else "Feature Union"
            )
            cur_muts_df.loc[union_label] = (
                cur_muts_df.min(axis=0) if logic_type.upper() == "AND" else cur_muts_df.max(axis=0)
            )
            counts = (cur_muts_df == 1).sum(axis=1)
            cur_muts_df.index = [f"{idx} (n={int(c)})" for idx, c in counts.items()]
            cur_muts_df.replace(1, "MUT", inplace=True)
            cur_muts_df.replace(0, np.nan, inplace=True)
            cur_muts_df = cur_muts_df.reindex(columns=ohd.columns)
            path_line = "\n".join(
                textwrap.wrap(f"{target_gene} - Path {path_idx + 1}: {path_label}", width=100)
            )
            group_lines = "\n".join(group_defs)
            path_title = path_line + "\n" + group_lines
            op = pyoncoprint.OncoPrint(cur_muts_df)
            fig, axes = op.oncoprint(
                mutation_markers,
                heatmaps=heatmaps,
                figsize=[40, 30],
                gap=[0.3, 0.02],
                title=path_title,
                sample_sort_method="unsorted",
                gene_sort_method="unsorted",
                topplot=False,
                rightplot=False,
                legend=False,
            )
            ax_main, ax_right_labels = axes[0], axes[1]
            ax_main.set_xticklabels([])
            ax_main.tick_params(axis="y", labelsize=14)
            ax_right_labels.tick_params(axis="y", labelsize=14)
            fig.suptitle(path_title, fontsize=24)
            fig.patch.set_facecolor("white")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved oncoplots to {pdf_path}")
