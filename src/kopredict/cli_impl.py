"""kopredict command-line implementation.

Flow:
  * Parse args, optionally set ``KOPREDICT_DATA_DIR`` and clear parquet caches.
  * Load the features matrix and strip broad/unwanted feature name patterns.
  * Load the target's dependency (CRISPR or RNAi) for ``-t`` / ``--target``.
  * If ``--cohort-csv``: for each model row, resolve cell lines, run
    ``_run_single_cohort_analysis`` (deferred plot write until significant), then
    write plots per cohort and a combined prediction summary CSV.
  * Else (``-c``): resolve one cohort from ``--oncotree-column`` + cancer type strings, run
    a single analysis under ``{output_dir}/{target}/{dataset}/`` with plots.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from kopredict.cohort_summary import build_cohort_summary_rows, safe_cohort_folder_name
from kopredict.database import celllines_for_single_cohort_row, return_celllines_matching_oncotree_filter
from kopredict.features import get_target_df
from kopredict.permutation import estimate_significant_pvalue_cutoff
from kopredict.pipeline import run_prediction_for_biomarker_given_target_dataset
from kopredict.local_data import get_data_dir
from kopredict.plotting import create_boxplot, create_oncoplot
from kopredict.sources import FEATURES_FILENAME


def _count_essential_cell_lines(
    target_df: pd.DataFrame,
    dataset: str,
    cell_lines: list[str],
    essentiality_threshold: float,
) -> int:
    """Count overlapping cell lines whose raw dependency score meets essentiality (<= threshold)."""
    ds = str(dataset).lower()
    score_col = ds
    if score_col not in target_df.columns:
        return 0
    sub = target_df[target_df["cellline"].astype(str).isin(set(str(x) for x in cell_lines))].copy()
    vals = pd.to_numeric(sub[score_col], errors="coerce")
    return int(((vals <= essentiality_threshold) & vals.notna()).sum())


def _resolved_summary_cancer(cur_cts, summary_cancertype, summary_oncotree_column, pipeline_oncotree_column):
    """Return (cancertype, oncotree_column) for summary CSV."""
    if summary_cancertype is not None:
        soc = summary_oncotree_column if summary_oncotree_column is not None else ""
        return summary_cancertype, soc
    if cur_cts and len(cur_cts) == 1 and "=" in cur_cts[0]:
        a, b = cur_cts[0].split("=", 1)
        return b.strip(), a.strip()
    return ";".join(cur_cts or []), (pipeline_oncotree_column or "")


def _prediction_summary_csv_basename(
    *, target: str, dataset: str, progression: str, ct_token: str
) -> str:
    """``YYMMDD_TARGET_DATASET_PROGRESSION_CT_prediction_summary.csv`` stem + extension."""
    date_part = datetime.now().strftime("%y%m%d")

    def sf(s):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s)) or "x"

    stem = f"{date_part}_{sf(target)}_{sf(dataset)}_{sf(progression)}_{sf(ct_token)}_prediction_summary"
    return f"{stem}.csv"


def parse_cancer_type_values(s):
    """Split ``-c`` on commas only; do not strip or change inner/outer spacing."""
    if s is None or s == "":
        return []
    if str(s).strip().lower() in ("none", "null"):
        return []
    return [x for x in str(s).split(",") if x != ""]


def _run_single_cohort_analysis(
    *,
    cur_target,
    cur_dataset,
    final_features_df,
    target_df,
    cell_raw,
    cur_cts,
    group_options,
    n_permutations,
    n_jobs,
    cur_or_flag,
    oncotree_column,
    cohort_csv_path,
    cohort_celllines_override,
    debug_wilcox,
    output_root,
    write_summary_rows,
    vmin,
    vmax,
    min_essential_cell_lines,
    essentiality_threshold,
    max_next_feature_subsets,
    write_plots=True,
    summary_cancertype=None,
    summary_oncotree_column=None,
):
    """Permutation null, z-score + feature search, and optional box/oncoplots for one cohort.

    ``cell_raw`` lists model cell lines for the cohort; they are intersected with both
    feature matrix columns and target dependency rows. Fails the biomarker search with a
    summary-only path if overlap or essential-line counts (see ``min_essential_cell_lines``)
    are too small.

    Returns ``(feature_tree, target_df_after_pipeline, summary_rows)``.
    """
    sum_ct, sum_otc = _resolved_summary_cancer(
        cur_cts, summary_cancertype, summary_oncotree_column, oncotree_column
    )
    celllines_of_interests = list(
        set(final_features_df.columns.astype(str).tolist()).intersection(
            set(str(x) for x in cell_raw)
        )
    )
    final_cell_line_list = list(
        set(target_df["cellline"].astype(str).tolist()).intersection(set(celllines_of_interests))
    )
    target_df_filtered = target_df[target_df["cellline"].astype(str).isin(final_cell_line_list)]

    if not final_cell_line_list:
        print("WARNING: No cell lines overlap features and target data for this cohort.", file=sys.stderr)
        logic_type = "OR" if cur_or_flag else "AND"
        summary_rows = (
            build_cohort_summary_rows(
                feature_tree=None,
                fdf=final_features_df,
                tdf_filtered=target_df_filtered,
                target=cur_target,
                dataset=cur_dataset,
                cancertype=sum_ct,
                oncotree_column=sum_otc,
                logic_type=logic_type,
                ko_datatype=cur_dataset,
                no_tree_group="No overlapping cell lines",
            )
            if write_summary_rows
            else []
        )
        return None, target_df_filtered, summary_rows

    n_essential = _count_essential_cell_lines(
        target_df, cur_dataset, final_cell_line_list, essentiality_threshold
    )
    if min_essential_cell_lines > 0 and n_essential < min_essential_cell_lines:
        print(
            f"WARNING: Only {n_essential} essential cell line(s) in overlap "
            f"(need >= {min_essential_cell_lines}; threshold {essentiality_threshold}). Skipping analysis.",
            file=sys.stderr,
        )
        logic_type = "OR" if cur_or_flag else "AND"
        summary_rows = (
            build_cohort_summary_rows(
                feature_tree=None,
                fdf=final_features_df,
                tdf_filtered=target_df_filtered,
                target=cur_target,
                dataset=cur_dataset,
                cancertype=sum_ct,
                oncotree_column=sum_otc,
                logic_type=logic_type,
                ko_datatype=cur_dataset,
                no_tree_group="Not enough essential cell lines",
            )
            if write_summary_rows
            else []
        )
        return None, target_df_filtered, summary_rows

    print("Estimating significant p-value cutoffs...")
    result = estimate_significant_pvalue_cutoff(
        target=cur_target,
        dataset=cur_dataset,
        fdf=final_features_df,
        cts=cur_cts,
        background_options=group_options,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        oncotree_column=oncotree_column,
        cohort_csv_path=cohort_csv_path,
        cohort_celllines_override=cohort_celllines_override,
        target_df=target_df,
    )
    print(f"P-value cutoff: {result['pvalue_cutoff']}")
    print(f"Q-value cutoff: {result['qvalue_cutoff']}")

    print("Running biomarker prediction pipeline...")
    feature_tree, target_df_pipeline = run_prediction_for_biomarker_given_target_dataset(
        target=cur_target,
        dataset=cur_dataset,
        fdf=final_features_df,
        cts=cur_cts,
        background_options=group_options,
        min_wilcox_pval=result["pvalue_cutoff"],
        min_wilcox_qval=result["qvalue_cutoff"],
        min_geom_pval=0.000000000000001,
        or_flag=cur_or_flag,
        oncotree_column=oncotree_column,
        cohort_csv_path=cohort_csv_path,
        cohort_celllines_override=cohort_celllines_override,
        debug_wilcox_top_n=debug_wilcox,
        max_next_feature_subsets=max_next_feature_subsets,
        target_df=target_df,
    )

    logic_type = "OR" if cur_or_flag else "AND"

    summary_rows: list[dict] = []
    if write_summary_rows:
        summary_rows = build_cohort_summary_rows(
            feature_tree=feature_tree,
            fdf=final_features_df,
            tdf_filtered=target_df_pipeline,
            target=cur_target,
            dataset=cur_dataset,
            cancertype=sum_ct,
            oncotree_column=sum_otc,
            logic_type=logic_type,
            ko_datatype=cur_dataset,
        )

    if feature_tree is None:
        print("No significant features found.")
        return feature_tree, target_df_pipeline, summary_rows

    print("Feature paths:", feature_tree.convert_to_list_of_lists())

    if write_plots:
        print("Creating boxplots...")
        create_boxplot(
            tree=feature_tree,
            fdf=final_features_df,
            tdf=target_df_pipeline,
            ko_datatype=cur_dataset,
            target_gene=cur_target,
            logic_type=logic_type,
            cancer_types=cur_cts,
            output_root=output_root,
            plot_title_cancer=sum_ct,
        )
        print("Creating oncoplots...")
        create_oncoplot(
            target_gene=cur_target,
            ko_datatype=cur_dataset,
            fdf=final_features_df,
            tdf=target_df_pipeline,
            tree=feature_tree,
            vmin=vmin,
            vmax=vmax,
            logic_type=logic_type,
            cancer_types=cur_cts,
            output_root=output_root,
        )
    return feature_tree, target_df_pipeline, summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="kopredict: Predict best biomarker features for a target gene/dataset."
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Target gene (e.g. IPPK, BRCA2)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=("crispr", "rnai"),
        help="Dependency matrix to use: crispr or rnai (local parquet inputs).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with local parquet inputs (default: ./data or KOPREDICT_DATA_DIR env).",
    )
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument(
        "-c",
        "--cancer-types",
        default=None,
        help=(
            "Comma-separated strings matched exactly in the model column given by "
            "--oncotree-column (no trimming or case changes)."
        ),
    )
    cohort.add_argument(
        "--cohort-csv",
        default=None,
        metavar="PATH",
        help=(
            "CSV with columns oncotree_column and name. Runs one analysis per row. Writes "
            "YYMMDD_*_prediction_summary.csv under {--output-dir}/{target}/{dataset}/. PDFs are "
            "written only when a cohort has significant features, under "
            ".../{dataset}/{oncotree_column}_{name}/."
        ),
    )
    parser.add_argument(
        "-O",
        "--oncotree-column",
        default="oncotreeprimarydisease",
        help=(
            "Column in Model_and_OmicsSignatures used with -c (exact header after file load). "
            "Ignored when --cohort-csv is set."
        ),
    )
    parser.add_argument(
        "-p",
        "--progression",
        type=lambda s: str(s).strip().upper(),
        choices=("AND", "OR"),
        default="AND",
        metavar="AND|OR",
        help="Feature progression between blocks: AND (intersection) or OR (union). Default: AND.",
    )
    parser.add_argument(
        "-g",
        "--group-options",
        default=None,
        help="Background filter options string (semicolon-separated)",
    )
    parser.add_argument(
        "-f",
        "--features-path",
        default=None,
        help="Path to features parquet (default: data/final_depmap_features_26Q1.parquet.gz).",
    )
    parser.add_argument(
        "-n",
        "--n-permutations",
        type=int,
        default=50,
        help="Number of permutations for p-value cutoff estimation (default: 50)",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for permutation (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help=(
            "Base directory; PDFs under {output-dir}/{target}/{dataset}/; with --cohort-csv, "
            "per-cohort PDFs under .../{target}/{dataset}/{oncotree_column}_{name}/ when significant."
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-1,
        help="Oncoplot heatmap vmin (default: -1)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1,
        help="Oncoplot heatmap vmax (default: 1)",
    )
    parser.add_argument(
        "--debug-wilcox",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="N",
        help="Print top N features by Wilcoxon p-value (with BH q) after mean filter; default N=10 if flag is given.",
    )
    parser.add_argument(
        "--min-essential-cell-lines",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Minimum number of overlapping cell lines with raw dependency <= essentiality threshold "
            "required to run permutations and the biomarker pipeline (default: 1). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--essentiality-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Raw crispr/rnai score threshold for calling a line essential (<= T counts). "
            "Default: -1 for crispr, -0.5 for rnai."
        ),
    )
    parser.add_argument(
        "--max-next-feature-subsets",
        type=int,
        default=3,
        metavar="K",
        help=(
            "After the initial biomarker groups (determine_best_biomrker_sets), each "
            "add_next_best_features_to_tree step keeps at most K feature subsets that pass "
            "Wilcoxon/q filters, ranked by best (smallest) Wilcoxon p within each subset. "
            "Use 0 for no limit (may be slow)."
        ),
    )
    parser.add_argument(
        "--keep-lowexp-highexp",
        action="store_true",
        default=False,
        help="Keep features whose row names contain 'lowexp' or 'highexp' (default: drop them).",
    )
    parser.add_argument(
        "--keep-depmapprot-sangerprot",
        action="store_true",
        default=False,
        help="Keep protein features (depmapprot, sangerprot, rppa). Default: drop them.",
    )

    args = parser.parse_args()

    if args.data_dir:
        os.environ["KOPREDICT_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
        from kopredict import local_data as _local_data

        _local_data.clear_load_caches()

    output_root = str(Path(args.output_dir).expanduser().resolve())

    cur_target = args.target
    cur_dataset = str(args.dataset).lower()
    cur_or_flag = args.progression == "OR"
    group_options = args.group_options
    essentiality_threshold = (
        float(args.essentiality_threshold)
        if args.essentiality_threshold is not None
        else (-1.0 if cur_dataset == "crispr" else -0.5)
    )
    min_essential_cell_lines = int(args.min_essential_cell_lines)
    if min_essential_cell_lines < 0:
        print("ERROR: --min-essential-cell-lines must be >= 0 (0 disables the filter).", file=sys.stderr)
        sys.exit(2)
    max_next_feature_subsets = int(args.max_next_feature_subsets)

    # Feature matrix: rows = feature IDs, columns = cellline IDs (see README).
    # Drop broad / noisy name families by default; --keep-* flags opt out of subsets.
    if not args.features_path:
        args.features_path = str(get_data_dir() / FEATURES_FILENAME)

    print("Loading features...")
    final_features_df = pd.read_parquet(args.features_path)
    final_features_df = final_features_df[~final_features_df.index.str.contains("absolute")]
    final_features_df = final_features_df[~final_features_df.index.str.contains("sensitive")]
    final_features_df = final_features_df[~final_features_df.index.str.contains("resistant")]
    if not args.keep_depmapprot_sangerprot:
        final_features_df = final_features_df[~final_features_df.index.str.contains("depmapprot")]
        final_features_df = final_features_df[~final_features_df.index.str.contains("sangerprot")]
        final_features_df = final_features_df[~final_features_df.index.str.contains("rppa")]
    if not args.keep_lowexp_highexp:
        final_features_df = final_features_df[~final_features_df.index.str.contains("lowexp")]
        final_features_df = final_features_df[~final_features_df.index.str.contains("highexp")]

    # One column per dep score: ``crispr`` or ``rnai`` plus derived z-scores in-pipeline.
    target_df = get_target_df(cur_target, cur_dataset)

    if args.cohort_csv:
        raw = pd.read_csv(args.cohort_csv, dtype=str, keep_default_na=False)
        for req in ("oncotree_column", "name"):
            if req not in raw.columns:
                print(f"ERROR: Cohort CSV must contain column {req!r}. Found: {list(raw.columns)}", file=sys.stderr)
                sys.exit(2)

        batch_root = os.path.join(output_root, cur_target, cur_dataset)
        os.makedirs(batch_root, exist_ok=True)

        summary_all: list[dict] = []
        logic_type = "OR" if cur_or_flag else "AND"
        for idx, row in raw.iterrows():
            col = row["oncotree_column"]
            name = row["name"]
            if name == "":
                continue
            cancer_type_label = f"{col}={name}" if col != "" else name
            print(f"\n--- Cohort row {idx}: {cancer_type_label} ---")
            try:
                cell_raw = celllines_for_single_cohort_row(col, name)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                summary_all.extend(
                    build_cohort_summary_rows(
                        feature_tree=None,
                        fdf=final_features_df,
                        tdf_filtered=target_df,
                        target=cur_target,
                        dataset=cur_dataset,
                        cancertype=name,
                        oncotree_column=col,
                        logic_type=logic_type,
                        ko_datatype=cur_dataset,
                    )
                )
                continue

            feature_tree, target_df_pipeline, srows = _run_single_cohort_analysis(
                cur_target=cur_target,
                cur_dataset=cur_dataset,
                final_features_df=final_features_df,
                target_df=target_df,
                cell_raw=cell_raw,
                cur_cts=[cancer_type_label],
                group_options=group_options,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                cur_or_flag=cur_or_flag,
                oncotree_column=args.oncotree_column,
                cohort_csv_path=None,
                cohort_celllines_override=cell_raw,
                debug_wilcox=args.debug_wilcox,
                output_root=batch_root,
                write_summary_rows=True,
                write_plots=False,
                vmin=args.vmin,
                vmax=args.vmax,
                min_essential_cell_lines=min_essential_cell_lines,
                essentiality_threshold=essentiality_threshold,
                max_next_feature_subsets=max_next_feature_subsets,
                summary_cancertype=name,
                summary_oncotree_column=col,
            )
            summary_all.extend(srows)

            if feature_tree is not None:
                cohort_dir = os.path.join(batch_root, safe_cohort_folder_name(col, name))
                os.makedirs(cohort_dir, exist_ok=True)
                print("Creating boxplots...")
                create_boxplot(
                    tree=feature_tree,
                    fdf=final_features_df,
                    tdf=target_df_pipeline,
                    ko_datatype=cur_dataset,
                    target_gene=cur_target,
                    logic_type=logic_type,
                    cancer_types=[cancer_type_label],
                    output_root=cohort_dir,
                    plot_title_cancer=name,
                )
                print("Creating oncoplots...")
                create_oncoplot(
                    target_gene=cur_target,
                    ko_datatype=cur_dataset,
                    fdf=final_features_df,
                    tdf=target_df_pipeline,
                    tree=feature_tree,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    logic_type=logic_type,
                    cancer_types=[cancer_type_label],
                    output_root=cohort_dir,
                )

        summary_csv_name = _prediction_summary_csv_basename(
            target=cur_target,
            dataset=cur_dataset,
            progression=args.progression,
            ct_token="cohort",
        )
        out_csv = os.path.join(batch_root, summary_csv_name)
        cols = [
            "Target",
            "Dataset",
            "cancertype",
            "oncotree_column",
            "Logic_type",
            "Group",
            "Biomarker_set",
            "Group_tree",
            "Number_mutant",
            "Number_WT",
            "Mutant_mean_essentiality",
            "Mutant_median_essentiality",
            "WT_mean_essentiality",
            "WT_median_essentiality",
            "Wilcoxon_pval",
        ]
        summary_df = pd.DataFrame(summary_all)[cols]
        before_n = len(summary_df)
        summary_df = summary_df.drop_duplicates(keep="first").reset_index(drop=True)
        after_n = len(summary_df)
        if after_n < before_n:
            print(f"Removed {before_n - after_n} duplicate summary row(s).")
        summary_df.to_csv(out_csv, index=False)
        print(f"\nWrote summary: {out_csv}")
        print("Done.")
        return

    # Non-batch: one cohort from -c / --cancer-types and --oncotree-column
    cur_cts = parse_cancer_type_values(args.cancer_types)
    if not cur_cts:
        print("ERROR: -c / --cancer-types must contain at least one value.", file=sys.stderr)
        sys.exit(2)
    cell_raw = return_celllines_matching_oncotree_filter(args.oncotree_column, cur_cts)

    nested_out = os.path.join(output_root, cur_target, cur_dataset)

    feature_tree, _, _ = _run_single_cohort_analysis(
        cur_target=cur_target,
        cur_dataset=cur_dataset,
        final_features_df=final_features_df,
        target_df=target_df,
        cell_raw=cell_raw,
        cur_cts=cur_cts,
        group_options=group_options,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        cur_or_flag=cur_or_flag,
        oncotree_column=args.oncotree_column,
        cohort_csv_path=None,
        cohort_celllines_override=cell_raw,
        debug_wilcox=args.debug_wilcox,
        output_root=nested_out,
        write_summary_rows=False,
        vmin=args.vmin,
        vmax=args.vmax,
        min_essential_cell_lines=min_essential_cell_lines,
        essentiality_threshold=essentiality_threshold,
        max_next_feature_subsets=max_next_feature_subsets,
        summary_cancertype=";".join(cur_cts),
        summary_oncotree_column=args.oncotree_column,
    )

    if feature_tree is None:
        sys.exit(0)

    print("Done.")
