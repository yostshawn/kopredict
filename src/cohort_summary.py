"""Build summary tables for cohort-batch runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import rows_for_block, rows_for_features, wilcoxon_test


def _flag_group_message(group: str | None) -> str | None:
    """Prefix warning-style group labels with ``FLAG - ``."""
    if group is None:
        return None
    g = str(group).strip()
    if g == "":
        return g
    normalized = g.lower()
    warning_labels = {
        "no overlapping cell lines",
        "no significant features",
        "not enough essential cell lines",
    }
    if normalized in warning_labels and not g.startswith("FLAG - "):
        return f"FLAG - {g}"
    return g


def _cohort_fdf_tdf(fdf, tdf, ko_datatype: str):
    cohort_cols = sorted(
        set(tdf["cellline"].astype(str).unique()).intersection(fdf.columns.astype(str))
    )
    fdf_plot = fdf.reindex(columns=cohort_cols)
    tdf_plot = tdf[tdf["cellline"].astype(str).isin(cohort_cols)].copy()
    score_col = str(ko_datatype).lower()
    return fdf_plot, tdf_plot, score_col, len(cohort_cols)


def _cumulative_mask(fdf_plot, feature_blocks: list, logic_type: str, up_to_k: int):
    logic_u = logic_type.upper()
    if logic_u == "AND":
        m = pd.Series(True, index=fdf_plot.columns)
    else:
        m = pd.Series(False, index=fdf_plot.columns)
    for i in range(up_to_k + 1):
        bm = rows_for_block(fdf_plot, feature_blocks[i]).fillna(0).any(axis=0)
        if logic_u == "AND":
            m = m & bm
        else:
            m = m | bm
    return m


def _best_feature_wilcox_p_in_block(fdf_plot, tdf_plot, score_col: str, block_string: str):
    parts = [part.strip() for part in str(block_string).split(";") if part.strip()]
    if len(parts) <= 1:
        candidates = parts or [str(block_string)]
    else:
        candidates = parts
    best_f, best_p = None, np.nan
    for feature in candidates:
        try:
            row = rows_for_features(fdf_plot, [feature])
        except KeyError:
            continue
        if row.empty:
            continue
        p = wilcoxon_test(row.iloc[0], tdf_plot, score_col)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        if best_f is None or p < best_p:
            best_f, best_p = feature, p
    return best_f, best_p


def _format_biomarker_set(block_string: str) -> str | None:
    parts = [x.strip() for x in block_string.split(";") if x.strip()]
    if not parts:
        return None
    return "; ".join(parts)


def _mutant_wt_essentiality_stats(
    tdf_plot: pd.DataFrame, score_col: str, m: pd.Series
) -> dict[str, float | None]:
    """Mean and median raw dependency (``crispr`` / ``rnai``) in mutant vs WT cell lines.

    ``m`` is a boolean series indexed by cell line ID: True = mutant (cumulative
    feature-positive), False = wild-type for that feature pattern.
    """
    empty = {
        "Mutant_mean_essentiality": None,
        "Mutant_median_essentiality": None,
        "WT_mean_essentiality": None,
        "WT_median_essentiality": None,
    }
    if tdf_plot.empty or score_col not in tdf_plot.columns:
        return empty
    one = tdf_plot.drop_duplicates(subset=["cellline"], keep="first")
    if one.empty:
        return empty
    cl = one["cellline"].astype(str)
    scores = pd.to_numeric(one[score_col], errors="coerce")
    scores.index = cl
    m = m.copy()
    m.index = m.index.astype(str)
    m2 = m.reindex(scores.index)
    if m2.isna().any():
        m2 = m2.fillna(False)
    m2 = m2.astype(bool)
    mut = scores[m2].dropna()
    wt = scores[~m2].dropna()

    def _mm(s: pd.Series) -> tuple[float | None, float | None]:
        if s.empty:
            return None, None
        return (float(s.mean()), float(s.median()))

    mm_m, med_m = _mm(mut)
    mm_w, med_w = _mm(wt)
    return {
        "Mutant_mean_essentiality": mm_m,
        "Mutant_median_essentiality": med_m,
        "WT_mean_essentiality": mm_w,
        "WT_median_essentiality": med_w,
    }


SUMMARY_COLUMNS = [
    "Target",
    "Dataset",
    "sens_res",
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


def build_cohort_summary_rows(
    *,
    feature_tree,
    fdf,
    tdf_filtered,
    target: str,
    dataset: str,
    cancertype: str,
    oncotree_column: str,
    logic_type: str,
    ko_datatype: str,
    no_tree_group: str | None = None,
    sens_res: str = "sensitive",
) -> list[dict]:
    """Rows for batch prediction summary CSV (one row per step along each tree path).

    ``cancertype`` is the cancer label (e.g. cohort CSV ``name`` or cancer_types values).
    ``oncotree_column`` is the model column used (e.g. ``oncotreecode``), or empty if N/A.
    ``sens_res`` is ``sensitive`` for the current search. A resistance search should pass ``resistant``.

    When ``feature_tree`` is None, ``no_tree_group`` overrides the default
    ``Group`` text (default: ``No Significant Features``).

    For each step, **mutant** / **WT** follow the cumulative feature mask on cell lines;
    mean/median essentiality use the raw dependency column (``crispr`` or ``rnai``)
    on ``tdf_filtered``, split by that mask.
    """
    logic_u = logic_type.upper()
    base = {
        "Target": target,
        "Dataset": dataset,
        "sens_res": sens_res,
        "cancertype": cancertype,
        "oncotree_column": oncotree_column,
        "Logic_type": logic_u,
    }
    if feature_tree is None:
        group = no_tree_group if no_tree_group else "No Significant Features"
        return [
            {
                **base,
                "Group": _flag_group_message(group),
                "Biomarker_set": None,
                "Group_tree": None,
                "Number_mutant": None,
                "Number_WT": None,
                "Mutant_mean_essentiality": None,
                "Mutant_median_essentiality": None,
                "WT_mean_essentiality": None,
                "WT_median_essentiality": None,
                "Wilcoxon_pval": None,
            }
        ]

    fdf_plot, tdf_plot, score_col, n_total = _cohort_fdf_tdf(fdf, tdf_filtered, ko_datatype)
    paths = feature_tree.convert_to_list_of_lists()
    rows: list[dict] = []
    any_path_blocks = False

    for path in paths:
        blocks = path[1:] if len(path) > 1 else []
        if not blocks:
            continue
        any_path_blocks = True
        for k in range(len(blocks)):
            m = _cumulative_mask(fdf_plot, blocks, logic_type, k)
            n_mut = int(m.sum())
            n_wt = int(n_total - n_mut)
            block_k = blocks[k]
            grp_f, grp_p = _best_feature_wilcox_p_in_block(fdf_plot, tdf_plot, score_col, block_k)
            tree_parts = blocks[: k + 1]
            group_tree = " -> ".join(tree_parts)
            biomarker_set = _format_biomarker_set(block_k)
            ess = _mutant_wt_essentiality_stats(tdf_plot, score_col, m)
            wp = None
            if grp_p is not None and not (isinstance(grp_p, float) and np.isnan(grp_p)):
                wp = float(grp_p)
            rows.append(
                {
                    **base,
                    "Group": grp_f if grp_f is not None else "",
                    "Biomarker_set": biomarker_set,
                    "Group_tree": group_tree,
                    "Number_mutant": n_mut,
                    "Number_WT": n_wt,
                    **ess,
                    "Wilcoxon_pval": wp,
                }
            )

    if not any_path_blocks:
        rows.append(
            {
                **base,
                "Group": _flag_group_message("No Significant Features"),
                "Biomarker_set": None,
                "Group_tree": None,
                "Number_mutant": None,
                "Number_WT": None,
                "Mutant_mean_essentiality": None,
                "Mutant_median_essentiality": None,
                "WT_mean_essentiality": None,
                "WT_median_essentiality": None,
                "Wilcoxon_pval": None,
            }
        )
    return rows


def safe_cohort_folder_name(oncotree_column: str, name: str) -> str:
    """Filesystem-safe folder segment ``{CANCERTYPE}_{NAME}`` from cohort CSV columns."""

    def _sf(s):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))

    col = _sf(str(oncotree_column))[:80]
    nm = _sf(str(name))[:80]
    if col and nm:
        out = f"{col}_{nm}"
    elif nm:
        out = nm
    elif col:
        out = col
    else:
        out = "cohort"
    return out[:200]
