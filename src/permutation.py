"""Permutation testing for p-value cutoff estimation."""

from __future__ import annotations

import os
import threading
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

# Cutoff reuse: same direction / effect mode / threshold, n_cells within 5%.
_CUTOFF_CACHE = []
_CUTOFF_CACHE_LOCK = threading.Lock()
_CELL_TOLERANCE = 0.05


def _cache_key_parts(n_cells, direction, effect_vs_wt, min_effect_size):
    return (
        int(n_cells),
        str(direction),
        bool(effect_vs_wt),
        round(float(min_effect_size), 6),
    )


def clear_cutoff_cache():
    """Drop reused permutation cutoffs (call between independent runs if needed)."""
    with _CUTOFF_CACHE_LOCK:
        _CUTOFF_CACHE.clear()


def lookup_cutoff_cache(n_cells, direction, effect_vs_wt, min_effect_size):
    """Return a cached cutoff dict when n_cells is within 5% of a prior subset."""
    n_cells = int(n_cells)
    if n_cells <= 0:
        return None
    with _CUTOFF_CACHE_LOCK:
        for entry in _CUTOFF_CACHE:
            cached_n, d, vs_wt, mes, payload = entry
            if d != str(direction) or vs_wt != bool(effect_vs_wt):
                continue
            if round(float(mes), 6) != round(float(min_effect_size), 6):
                continue
            denom = max(cached_n, 1)
            if abs(n_cells - cached_n) / denom <= _CELL_TOLERANCE:
                return dict(payload)
    return None


def store_cutoff_cache(n_cells, direction, effect_vs_wt, min_effect_size, payload):
    with _CUTOFF_CACHE_LOCK:
        _CUTOFF_CACHE.append(
            (
                int(n_cells),
                str(direction),
                bool(effect_vs_wt),
                round(float(min_effect_size), 6),
                dict(payload),
            )
        )


def _limit_blas_threads():
    """Avoid oversubscribing CPU when shuffle workers already use multiple threads."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def smoothed_null_cutoff(minima, percentile=5):
    """Mean of the lowest ``k`` minima, with ``k`` from the percentile of ``n``.

    For 50 permutations and percentile 5, ``k = ceil(50 * 0.05) = 3``.
    """
    values = np.asarray(minima, dtype=float)
    if values.size == 0:
        return 1.0
    k = max(1, int(np.ceil(values.size * float(percentile) / 100.0)))
    k = min(k, values.size)
    return float(np.mean(np.sort(values)[:k]))


def run_permutation_null_pvalues(
    tdf=None,
    fdf=None,
    filter_score_column=None,
    wilcox_score_column=None,
    random_state=None,
    min_effect_size=-0.2,
    direction="sensitivity",
    effect_vs_wt=False,
    min_group_count=3,
):
    """Run null permutation: shuffle cell line labels, return unfiltered p/q values."""
    layout = prepare_permutation_layout(fdf, tdf, wilcox_score_column)
    return permutation_null_for_seed(
        layout,
        random_state,
        min_effect_size,
        direction,
        effect_vs_wt=effect_vs_wt,
        min_group_count=min_group_count,
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
    min_group_count=3,
    use_cache=True,
):
    """Estimate p-value and q-value cutoffs by shuffling labels on one feature matrix.

    ``fdf`` must already be restricted to the cell lines under test, with constant
    rows removed and identical rows merged. ``tdf`` must contain those cell lines,
    the raw score column, and the z-score column used by the mean filter.

    The cutoff is the mean of the lowest ``ceil(n * percentile / 100)`` per-shuffle
    minima (smoother than a raw percentile). Matching subsets reuse a prior cutoff
    when ``n_cells`` is within 5%.
    """
    n_cells = 0 if fdf is None else int(fdf.shape[1])
    if use_cache:
        cached = lookup_cutoff_cache(n_cells, direction, effect_vs_wt, min_effect_size)
        if cached is not None:
            print(
                f"Reusing permutation cutoffs for ~{n_cells} cell lines "
                f"({direction}, effect_vs_wt={effect_vs_wt})."
            )
            return cached

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
            min_group_count=min_group_count,
        )
        min_p = min(pvals) if pvals else 1.0
        min_q = min(qvals) if qvals else 1.0
        return pvals, qvals, min_p, min_q

    _n_jobs = n_jobs if n_jobs is not None else 1
    if _n_jobs <= 1:
        results = [_run_one_perm(i) for i in range(n_permutations)]
    else:
        _limit_blas_threads()
        try:
            _n_workers = min(int(_n_jobs), os.cpu_count() or 4, n_permutations)
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

    payload = {
        "pvalue_cutoff": smoothed_null_cutoff(min_pvals_per_perm, percentile),
        "qvalue_cutoff": smoothed_null_cutoff(min_qvals_per_perm, percentile),
        "min_pvals_per_permutation": min_pvals_per_perm,
        "min_qvals_per_permutation": min_qvals_per_perm,
        "all_pvals": all_pvals_per_perm,
        "all_qvals": all_qvals_per_perm,
    }
    if use_cache:
        store_cutoff_cache(n_cells, direction, effect_vs_wt, min_effect_size, payload)
    return payload


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
    min_group_count=3,
):
    """Estimate p-value and q-value cutoffs from permutation null distribution.

    Prefer ``permutation_cutoffs_from_frames`` on already-prepared cohort frames.
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
        min_group_count=min_group_count,
    )
