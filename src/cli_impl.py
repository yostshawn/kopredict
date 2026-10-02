"""kopredict command-line implementation.

Flow:
  * Load a YAML run file (default: metadata/kopredict_default.yaml).
  * Optionally set ``KOPREDICT_DATA_DIR`` and clear parquet caches.
  * Load the features matrix and strip broad/unwanted feature name patterns.
  * Load the target's dependency (CRISPR or RNAi).
  * If ``cohort_csv`` is set: for each row, resolve cell lines, run
    ``_run_single_cohort_analysis`` (deferred plot write until significant), then
    write plots per cohort and a combined prediction summary CSV.
  * Otherwise resolve one cohort from ``oncotree_column`` and ``cancer_types``, and run
    a single analysis under ``{output_dir}/{target}/{dataset}/`` with plots.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from cohort_summary import (
    SUMMARY_COLUMNS,
    build_cohort_summary_rows,
    safe_cohort_folder_name,
)
from database import (
    celllines_for_cohort_names,
    parse_cohort_name_list,
    return_celllines_matching_oncotree_filter,
    validate_cohort_table,
)
from features import get_target_df
from local_data import get_data_dir
from permutation import estimate_significant_pvalue_cutoff
from pipeline import run_prediction_for_biomarker_given_target_dataset
from plotting import create_boxplot, summary_csv_path
from prepare import FEATURES_FILENAME
from sources import metadata_dir

_CONFIG_NAME = "kopredict_default.yaml"
_RUN_HELP = """\
Find genomic features linked to one gene's CRISPR or RNAi sensitivity.

  kopredict
  kopredict path/to/run.yaml
  kopredict --help

Settings are a YAML file. With no path, metadata/kopredict_default.yaml is used.
Copy that file and change target, dataset, and cancer_types for another search.
A custom file can omit keys; missing keys are filled from the default file.
Allowed cancer_types, and what SOLID and PANCANCER include, are in metadata/oncotree.yaml.
"""


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
    """Split a comma-separated cohort string. A YAML list is used as given."""
    if s is None or s == "":
        return []
    if isinstance(s, (list, tuple)):
        return [str(item) for item in s if str(item) != ""]
    if str(s).strip().lower() in ("none", "null"):
        return []
    return [x for x in str(s).split(",") if x != ""]


def _default_config_path() -> Path:
    return metadata_dir() / _CONFIG_NAME


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a YAML mapping of settings.")
    return loaded


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def load_run_config(path: Path | None) -> SimpleNamespace:
    """Load a run file, filling any omitted keys from metadata/kopredict_default.yaml."""
    default_path = _default_config_path()
    defaults = _read_yaml(default_path)
    if path is None:
        loaded = defaults
        source = default_path
    else:
        loaded = _read_yaml(path)
        unknown = sorted(set(loaded) - set(defaults))
        if unknown:
            raise SystemExit(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
        merged = dict(defaults)
        merged.update(loaded)
        loaded = merged
        source = path
    dataset = str(loaded.get("dataset", "")).strip().lower()
    if dataset not in {"crispr", "rnai"}:
        raise SystemExit(f"dataset must be crispr or rnai (got {loaded.get('dataset')!r}).")
    progression = str(loaded.get("progression", "AND")).strip().upper()
    if progression not in {"AND", "OR"}:
        raise SystemExit(f"progression must be AND or OR (got {loaded.get('progression')!r}).")
    cancer_types = None if _blank(loaded.get("cancer_types")) else loaded.get("cancer_types")
    cohort_csv = None if _blank(loaded.get("cohort_csv")) else str(loaded.get("cohort_csv"))
    if cancer_types and cohort_csv:
        raise SystemExit("Set cancer_types or cohort_csv, not both.")
    if not cancer_types and not cohort_csv:
        raise SystemExit("Set cancer_types or cohort_csv in the YAML file.")
    debug = loaded.get("debug_wilcox")
    if isinstance(debug, bool):
        debug = 10 if debug else None
    elif debug is not None:
        debug = int(debug)
    if not 0 <= float(loaded.get("null_percentile", 5)) <= 100:
        raise SystemExit("null_percentile must be between 0 and 100.")
    return SimpleNamespace(
        target=str(loaded["target"]).strip(),
        dataset=dataset,
        cancer_types=cancer_types,
        cohort_csv=cohort_csv,
        oncotree_column=str(loaded.get("oncotree_column") or "oncotreeprimarydisease"),
        progression=progression,
        group_options=None if _blank(loaded.get("group_options")) else str(loaded.get("group_options")),
        data_dir=None if _blank(loaded.get("data_dir")) else str(loaded.get("data_dir")),
        features_path=None if _blank(loaded.get("features_path")) else str(loaded.get("features_path")),
        output_dir=None if _blank(loaded.get("output_dir")) else str(loaded.get("output_dir")),
        n_permutations=int(loaded.get("n_permutations", 50)),
        n_jobs=int(loaded.get("n_jobs", 1)),
        null_percentile=float(loaded.get("null_percentile", 5)),
        min_effect_size=float(loaded.get("min_effect_size", -0.2)),
        resistance_min_effect_size=float(loaded.get("resistance_min_effect_size", 0.2)),
        strong_min_effect_size=float(loaded.get("strong_min_effect_size", -0.2)),
        child_min_effect_size=float(loaded.get("child_min_effect_size", -0.1)),
        child_resistance_min_effect_size=float(loaded.get("child_resistance_min_effect_size", 0.1)),
        vmin=float(loaded.get("vmin", -1)),
        vmax=float(loaded.get("vmax", 1)),
        debug_wilcox=debug,
        min_essential_cell_lines=int(loaded.get("min_essential_cell_lines", 1)),
        essentiality_threshold=None if _blank(loaded.get("essentiality_threshold")) else float(loaded.get("essentiality_threshold")),
        max_next_feature_subsets=int(loaded.get("max_next_feature_subsets", 3)),
        keep_expression_features=bool(loaded.get("keep_expression_features", False)),
        keep_proteiomic_features=bool(loaded.get("keep_proteiomic_features", False)),
    )


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
    null_percentile,
    min_effect_size,
    resistance_min_effect_size,
    strong_min_effect_size,
    child_min_effect_size,
    child_resistance_min_effect_size,
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
        percentile=null_percentile,
        min_effect_size=min_effect_size,
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
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        percentile=null_percentile,
        min_effect_size=min_effect_size,
        resistance_min_effect_size=resistance_min_effect_size,
        strong_min_effect_size=strong_min_effect_size,
        child_min_effect_size=child_min_effect_size,
        child_resistance_min_effect_size=child_resistance_min_effect_size,
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
        # Oncoplots are paused while resistance markers are in the tree.
        # print("Creating oncoplots...")
        # create_oncoplot(
        #     target_gene=cur_target,
        #     ko_datatype=cur_dataset,
        #     fdf=final_features_df,
        #     tdf=target_df_pipeline,
        #     tree=feature_tree,
        #     vmin=vmin,
        #     vmax=vmax,
        #     logic_type=logic_type,
        #     cancer_types=cur_cts,
        #     output_root=output_root,
        # )
    return feature_tree, target_df_pipeline, summary_rows


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=_RUN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="YAML run file (default: metadata/kopredict_default.yaml).",
    )
    parsed = parser.parse_args(argv)
    config_path = Path(parsed.config).expanduser() if parsed.config else None
    args = load_run_config(config_path)

    if args.data_dir:
        os.environ["KOPREDICT_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
        import local_data as _local_data

        _local_data.clear_load_caches()

    if args.output_dir:
        output_root = str(Path(args.output_dir).expanduser().resolve())
    else:
        output_root = str(get_data_dir() / "output")

    cur_target = args.target
    cur_dataset = str(args.dataset).lower()
    cur_or_flag = args.progression == "OR"
    print(f"Run: target={cur_target}  dataset={cur_dataset}  logic={args.progression}")
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
    if not args.keep_proteiomic_features:
        final_features_df = final_features_df[~final_features_df.index.str.contains("depmapprot")]
        final_features_df = final_features_df[~final_features_df.index.str.contains("sangerprot")]
        final_features_df = final_features_df[~final_features_df.index.str.contains("rppa")]
    if not args.keep_expression_features:
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
        validate_cohort_table(raw[["oncotree_column", "name"]])

        batch_root = os.path.join(output_root, cur_target, cur_dataset)
        os.makedirs(batch_root, exist_ok=True)

        summary_all: list[dict] = []
        logic_type = "OR" if cur_or_flag else "AND"
        for idx, row in raw.iterrows():
            col = row["oncotree_column"]
            name = row["name"]
            output_name = str(row["output_name"]).strip() if "output_name" in row.index else ""
            display_name = output_name or name
            names = parse_cohort_name_list(name)
            if not names:
                continue
            cancer_type_label = f"{col}={' + '.join(names)}" if col != "" else " + ".join(names)
            print(f"\n--- Cohort row {idx}: {cancer_type_label} ---")
            try:
                cell_raw = celllines_for_cohort_names(col, names)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                print("ERROR: Cohort CSV was not run.", file=sys.stderr)
                sys.exit(2)

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
                null_percentile=args.null_percentile,
                min_effect_size=args.min_effect_size,
                resistance_min_effect_size=args.resistance_min_effect_size,
                strong_min_effect_size=args.strong_min_effect_size,
                child_min_effect_size=args.child_min_effect_size,
                child_resistance_min_effect_size=args.child_resistance_min_effect_size,
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
                summary_cancertype=display_name,
                summary_oncotree_column=col,
            )
            summary_all.extend(srows)

            if feature_tree is not None:
                folder = output_name or safe_cohort_folder_name(col, name)
                cohort_dir = os.path.join(batch_root, folder)
                os.makedirs(cohort_dir, exist_ok=True)
                print("Creating boxplots...")
                create_boxplot(
                    tree=feature_tree,
                    fdf=final_features_df,
                    tdf=target_df_pipeline,
                    ko_datatype=cur_dataset,
                    target_gene=cur_target,
                    logic_type=logic_type,
                    cancer_types=[display_name],
                    output_root=cohort_dir,
                    plot_title_cancer=display_name,
                )
                # Oncoplots are paused while resistance markers are in the tree.
                # print("Creating oncoplots...")
                # create_oncoplot(
                #     target_gene=cur_target,
                #     ko_datatype=cur_dataset,
                #     fdf=final_features_df,
                #     tdf=target_df_pipeline,
                #     tree=feature_tree,
                #     vmin=args.vmin,
                #     vmax=args.vmax,
                #     logic_type=logic_type,
                #     cancer_types=[display_name],
                #     output_root=cohort_dir,
                # )

        summary_csv_name = _prediction_summary_csv_basename(
            target=cur_target,
            dataset=cur_dataset,
            progression=args.progression,
            ct_token="cohort",
        )
        out_csv = os.path.join(batch_root, summary_csv_name)
        summary_df = pd.DataFrame(summary_all, columns=SUMMARY_COLUMNS)[SUMMARY_COLUMNS]
        before_n = len(summary_df)
        summary_df = summary_df.drop_duplicates(keep="first").reset_index(drop=True)
        after_n = len(summary_df)
        if after_n < before_n:
            print(f"Removed {before_n - after_n} duplicate summary row(s).")
        summary_df.to_csv(out_csv, index=False)
        print(f"\nWrote summary: {out_csv}")
        print("Done.")
        return

    # Non-batch: one cohort from cancer_types and oncotree_column.
    cur_cts = parse_cancer_type_values(args.cancer_types)
    if not cur_cts:
        print("ERROR: cancer_types in the YAML file must contain at least one value.", file=sys.stderr)
        sys.exit(2)
    cell_raw = return_celllines_matching_oncotree_filter(args.oncotree_column, cur_cts)

    nested_out = os.path.join(output_root, cur_target, cur_dataset)

    feature_tree, _, summary_rows = _run_single_cohort_analysis(
        cur_target=cur_target,
        cur_dataset=cur_dataset,
        final_features_df=final_features_df,
        target_df=target_df,
        cell_raw=cell_raw,
        cur_cts=cur_cts,
        group_options=group_options,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        null_percentile=args.null_percentile,
        min_effect_size=args.min_effect_size,
        resistance_min_effect_size=args.resistance_min_effect_size,
        strong_min_effect_size=args.strong_min_effect_size,
        child_min_effect_size=args.child_min_effect_size,
        child_resistance_min_effect_size=args.child_resistance_min_effect_size,
        cur_or_flag=cur_or_flag,
        oncotree_column=args.oncotree_column,
        cohort_csv_path=None,
        cohort_celllines_override=cell_raw,
        debug_wilcox=args.debug_wilcox,
        output_root=nested_out,
        write_summary_rows=True,
        vmin=args.vmin,
        vmax=args.vmax,
        min_essential_cell_lines=min_essential_cell_lines,
        essentiality_threshold=essentiality_threshold,
        max_next_feature_subsets=max_next_feature_subsets,
        summary_cancertype=";".join(cur_cts),
        summary_oncotree_column=args.oncotree_column,
    )
    logic_type = "OR" if cur_or_flag else "AND"
    summary_path = summary_csv_path(cur_target, cur_dataset, logic_type, cur_cts, nested_out)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df = summary_df.drop_duplicates(keep="first").reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    if feature_tree is None:
        sys.exit(0)

    print("Done.")
