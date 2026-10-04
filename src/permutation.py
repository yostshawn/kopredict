"""Permutation testing for p-value cutoff estimation."""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from features import (
    get_target_df,
    add_zscore_to_df,
    cleanup_featureset,
    prepare_permutation_layout,
    permutation_null_for_seed,
)
from database import (
    load_cohort_definition_from_csv,
    return_celllines_matching_oncotree_filter,
)
from background import Background


def run_permutation_null_pvalues(
    tdf=None,
    fdf=None,
    filter_score_column=None,
    wilcox_score_column=None,
    random_state=None,
    min_effect_size=-0.2,
    direction="sensitivity",
    effect_vs_wt=False,
):
    """Run null permutation: shuffle cell line labels, return unfiltered p/q values."""
    layout = prepare_permutation_layout(fdf, tdf, wilcox_score_column)
    return permutation_null_for_seed(
        layout, random_state, min_effect_size, direction, effect_vs_wt=effect_vs_wt
    )


def permutation_cutoffs_from_frames(
    tdf=None,
    fdf=None,
    filter_score_column=None,
    wilcox_score_column=None,
    n_permutations=50,
    percentile=5,
    n_jobs=1,
    min_effect_size=-0.2,
    direction="sensitivity",
    effect_vs_wt=False,
):
    """Estimate p-value and q-value cutoffs by shuffling labels on one feature matrix.

    ``fdf`` must already be restricted to the cell lines under test, with constant
    rows removed and identical rows merged. ``tdf`` must contain those cell lines,
    the raw score column, and the z-score column used by the mean filter.
    """
    min_pvals_per_perm = []
    min_qvals_per_perm = []
    all_pvals_per_perm = []
    all_qvals_per_perm = []
    layout = prepare_permutation_layout(fdf, tdf, wilcox_score_column)

    def _run_one_perm(i):
        pvals, qvals = permutation_null_for_seed(
            layout,
            i,
            min_effect_size=min_effect_size,
            direction=direction,
            effect_vs_wt=effect_vs_wt,
        )
        min_p = min(pvals) if pvals else 1.0
        min_q = min(qvals) if qvals else 1.0
        return pvals, qvals, min_p, min_q

    _n_jobs = n_jobs if n_jobs is not None else 1
    if _n_jobs <= 1:
        results = [_run_one_perm(i) for i in range(n_permutations)]
    else:
        try:
            _n_workers = min(_n_jobs, os.cpu_count() or 4, n_permutations)
            with ThreadPoolExecutor(max_workers=_n_workers) as ex:
                results = list(ex.map(_run_one_perm, range(n_permutations)))
        except Exception as e:
            warnings.warn(f"Parallelization failed ({e}), falling back to sequential execution.")
            results = [_run_one_perm(i) for i in range(n_permutations)]

    for pvals, qvals, min_p, min_q in results:
        all_pvals_per_perm.append(pvals)
        all_qvals_per_perm.append(qvals)
        min_pvals_per_perm.append(min_p)
        min_qvals_per_perm.append(min_q)

    return {
        "pvalue_cutoff": np.percentile(min_pvals_per_perm, percentile),
        "qvalue_cutoff": np.percentile(min_qvals_per_perm, percentile),
        "min_pvals_per_permutation": min_pvals_per_perm,
        "min_qvals_per_permutation": min_qvals_per_perm,
        "all_pvals": all_pvals_per_perm,
        "all_qvals": all_qvals_per_perm,
    }


def estimate_significant_pvalue_cutoff(
    target=None,
    dataset=None,
    fdf=None,
    cts=None,
    background_options=None,
    min_feature_count=3,
    n_permutations=100,
    percentile=5,
    n_jobs=None,
    oncotree_column="oncotreelineage",
    cohort_csv_path=None,
    cohort_celllines_override=None,
    target_df=None,
    min_effect_size=-0.2,
):
    """Estimate p-value and q-value cutoffs from permutation null distribution.

    If ``target_df`` is provided (e.g. from ``get_target_df``), it is used instead of
    loading dependency data again. Must contain ``cellline`` and the raw score column
    (``crispr`` or ``rnai``) for ``dataset``. Rows are filtered and z-scored internally.
    """
    if str(dataset).lower() == "drug":
        raise NotImplementedError("Dataset 'drug' is disabled in the local-only build.")
    if target_df is None:
        target_df = get_target_df(target, dataset)

    if cohort_celllines_override is not None:
        celllines_of_interests = list(cohort_celllines_override)
    elif cohort_csv_path:
        celllines_of_interests, _ = load_cohort_definition_from_csv(cohort_csv_path)
    else:
        celllines_of_interests = return_celllines_matching_oncotree_filter(oncotree_column, cts)
    celllines_of_interests = list(
        set(fdf.columns.astype(str).tolist()).intersection(set(celllines_of_interests))
    )

    if background_options is not None:
        from database import pull_cell_line_full_genomics

        background = Background(background_options, celllines_of_interests)
        background.create_background_df()
        background.fill_in_gene_list()
        background.cancer_types = cts
        omics_df = pull_cell_line_full_genomics(genes=background.gene_list)
        background.update_celllines_given_options(omics_df)
        background.check_background_df(omics_df, perc=25)
        background.update_background_celllines(omics_df)
        celllines_of_interests = background.background_celllines

    final_cell_line_list = list(
        set(target_df["cellline"].values.tolist()).intersection(set(celllines_of_interests))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered_features_df = cleanup_featureset(fdf[final_cell_line_list], mindiff=min_feature_count)

    target_df = target_df[target_df["cellline"].isin(final_cell_line_list)].copy()
    add_zscore_to_df(target_df, dataset)
    cur_filter_score_column = str(dataset).lower() + "_zscore"
    cur_wilcox_score_column = str(dataset).lower()

    return permutation_cutoffs_from_frames(
        tdf=target_df,
        fdf=filtered_features_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        n_permutations=n_permutations,
        percentile=percentile,
        n_jobs=n_jobs,
        min_effect_size=min_effect_size,
        direction="sensitivity",
    )
