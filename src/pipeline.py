"""Biomarker discovery pipeline.

Exposes the main callable that the CLI uses: z-score the target dependency column,
search for candidate feature sets (Wilcoxon + hypergeometric grouping), then grow a
``FeatureNode`` tree (AND/OR steps). See ``run_prediction_for_biomarker_given_target_dataset``.
"""

import warnings

import pandas as pd

from background import Background
from database import (
    load_cohort_definition_from_csv,
    return_celllines_matching_oncotree_filter,
)
from features import (
    FeatureNode,
    add_zscore_to_df,
    base_feature_name,
    cleanup_featureset,
    debug_print_top_wilcoxon_features,
    expanded_feature_labels,
    feature_is_inverse,
    generaate_overlap_matrix,
    get_target_df,
    group_features_by_pvalue_v2,
    materialize_inverse_features,
    restrict_features_to_celllines,
    return_possible_features,
    run_hypergeom_tests,
)
from permutation import permutation_cutoffs_from_frames


def _limit_next_tree_subsets(cur_list_of_lists, filtered_feature_set: pd.DataFrame, max_subsets: int):
    """Keep at most ``max_subsets`` groups, ranked by best (min) Wilcoxon p in each group.

    ``max_subsets <= 0`` means no limit (all groups from ``group_features_by_pvalue_v2``).
    """
    if not cur_list_of_lists or max_subsets <= 0:
        return cur_list_of_lists
    if filtered_feature_set.empty or "feature" not in filtered_feature_set.columns:
        return cur_list_of_lists[:max_subsets]
    pmap = filtered_feature_set.drop_duplicates(subset=["feature"]).set_index("feature")["p_value"]

    def _best_p(group):
        ps = []
        for f in group:
            if f in pmap.index:
                v = pmap.loc[f]
                ps.append(float(v) if pd.notna(v) else 1.0)
        return min(ps) if ps else 1.0

    ordered = sorted(cur_list_of_lists, key=_best_p)
    return ordered[:max_subsets]


def determine_best_biomrker_sets(
    fdf,
    tdf,
    filter_score_column,
    wilcox_score_column,
    wilcox_pval_cutoff=0.0001,
    wilcox_qval_cutoff=0.01,
    geom_pval_cutoff=0.0000000001,
    min_effect_size=-0.2,
    direction="sensitivity",
):
    """Run feature pipeline: Wilcoxon filter -> overlap matrix -> hypergeom -> group."""
    filtered_feature_set = return_possible_features(
        fdf,
        tdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        wilcox_cutoff=wilcox_pval_cutoff,
        qval_cutoff=wilcox_qval_cutoff,
        min_effect_size=min_effect_size,
        direction=direction,
    )
    if filtered_feature_set.empty:
        return [], filtered_feature_set

    fdf = materialize_inverse_features(fdf, filtered_feature_set["feature"].tolist())
    overlaps = generaate_overlap_matrix(
        fdf,
        filtered_feature_set["feature"].values.tolist(),
        filtered_feature_set["feature"].values.tolist(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlaping_feature_pvals = run_hypergeom_tests(fdf, overlaps)
    return group_features_by_pvalue_v2(
        overlaping_feature_pvals,
        feature_list=filtered_feature_set["feature"].values.tolist(),
        pvalue_threshold=geom_pval_cutoff,
        fdf=fdf,
        tdf=tdf,
        score_column=wilcox_score_column,
        feature_wilcox_pvals=filtered_feature_set,
    ), filtered_feature_set


def prepare_prediction_inputs(
    target,
    dataset,
    fdf,
    cts,
    background_options=None,
    min_feature_count=3,
    oncotree_column="oncotreelineage",
    cohort_csv_path=None,
    cohort_celllines_override=None,
    target_df=None,
):
    """Align feature matrix and target dependency (with z-score) as at pipeline start.

    If ``target_df`` is provided, skip ``get_target_df`` (same contract as
    ``estimate_significant_pvalue_cutoff``).
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
        background = Background(background_options, celllines_of_interests)
        background.create_background_df()
        background.fill_in_gene_list()
        background.cancer_types = cts
        from database import pull_cell_line_full_genomics

        omics_df = pull_cell_line_full_genomics(genes=background.gene_list)
        background.update_celllines_given_options(omics_df)
        background.check_background_df(omics_df, perc=25)
        print(background)
        background.update_background_celllines(omics_df)
        print(len(background.background_celllines))
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
    return filtered_features_df, target_df, cur_filter_score_column, cur_wilcox_score_column


def run_prediction_for_biomarker_given_target_dataset(
    target=None,
    dataset=None,
    fdf=None,
    cts=None,
    background_options=None,
    min_feature_count=3,
    min_wilcox_pval=0.001,
    min_wilcox_qval=0.01,
    min_geom_pval=0.0000000001,
    or_flag=True,
    oncotree_column="oncotreelineage",
    cohort_csv_path=None,
    cohort_celllines_override=None,
    debug_wilcox_top_n=None,
    max_next_feature_subsets=3,
    target_df=None,
    n_permutations=50,
    n_jobs=1,
    percentile=5,
    min_effect_size=-0.2,
    resistance_min_effect_size=0.2,
    strong_min_effect_size=-0.2,
    child_min_effect_size=-0.1,
    child_resistance_min_effect_size=0.1,
):
    """Run full biomarker prediction pipeline for target/dataset.

    Returns ``(feature_tree, target_df)`` where ``target_df`` is the cohort-filtered
    dependency table with raw scores and ``{dataset}_zscore`` (same rows/columns as
    used for Wilcoxon tests). ``feature_tree`` is ``None`` when no root feature groups
    are found.
    """
    filtered_features_df, target_df, cur_filter_score_column, cur_wilcox_score_column = prepare_prediction_inputs(
        target,
        dataset,
        fdf,
        cts,
        background_options=background_options,
        min_feature_count=min_feature_count,
        oncotree_column=oncotree_column,
        cohort_csv_path=cohort_csv_path,
        cohort_celllines_override=cohort_celllines_override,
        target_df=target_df,
    )
    if debug_wilcox_top_n is not None:
        print("[debug] Initial cohort Wilcoxon ranking (same inputs as first determine_best_biomrker_sets step):")
        debug_print_top_wilcoxon_features(
            filtered_features_df,
            target_df,
            filter_score_column=cur_filter_score_column,
            wilcox_score_column=cur_wilcox_score_column,
            top_n=int(debug_wilcox_top_n),
            min_effect_size=min_effect_size,
        )

    sens_scored = return_possible_features(
        filtered_features_df,
        target_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        wilcox_cutoff=min_wilcox_pval,
        qval_cutoff=min_wilcox_qval,
        min_effect_size=min_effect_size,
        direction="sensitivity",
    )
    print(
        f"Root resistance shuffles on {filtered_features_df.shape[1]} cell lines and "
        f"{filtered_features_df.shape[0]} features ({n_permutations} permutations)..."
    )
    resistance_cutoffs = permutation_cutoffs_from_frames(
        tdf=target_df,
        fdf=filtered_features_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        n_permutations=n_permutations,
        percentile=percentile,
        n_jobs=n_jobs,
        min_effect_size=resistance_min_effect_size,
        direction="resistance",
    )
    print(f"Resistance p-value cutoff: {resistance_cutoffs['pvalue_cutoff']}")
    print(f"Resistance q-value cutoff: {resistance_cutoffs['qvalue_cutoff']}")
    res_scored = return_possible_features(
        filtered_features_df,
        target_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        wilcox_cutoff=resistance_cutoffs["pvalue_cutoff"],
        qval_cutoff=resistance_cutoffs["qvalue_cutoff"],
        min_effect_size=resistance_min_effect_size,
        direction="resistance",
    )
    sens_scored, res_scored = _keep_smaller_root_pair(sens_scored, res_scored, filtered_features_df)
    sens_groups, filtered_features_df = _group_scored_features(
        filtered_features_df, target_df, sens_scored, min_geom_pval, cur_wilcox_score_column
    )
    res_groups, filtered_features_df = _group_scored_features(
        filtered_features_df, target_df, res_scored, min_geom_pval, cur_wilcox_score_column
    )

    root_features = FeatureNode(target)

    if not sens_groups and not res_groups:
        return None, target_df

    root_passes = (
        ("sensitive", sens_groups, sens_scored),
        ("resistant", res_groups, res_scored),
    )
    for sens_res, list_of_list_features, scored_features in root_passes:
        excluded = []
        for cur_features in list_of_list_features:
            # Sibling k cannot see itself or any earlier sibling on this row.
            excluded.extend(cur_features)
            cur_feature_string = ";".join(cur_features)
            node_label = "resistant" if sens_res == "resistant" else "sensitive"
            cur_feature_node = FeatureNode(cur_feature_string, sens_res=node_label)
            root_features.add_child(cur_feature_node)
            if or_flag:
                cur_cl = list(
                    set(target_df["cellline"].values.tolist()).intersection(
                        set(
                            filtered_features_df.columns[
                                filtered_features_df.loc[cur_features].max(axis=0) == 0
                            ].values.tolist()
                        )
                    )
                )
            else:
                cur_cl = list(
                    set(target_df["cellline"].values.tolist()).intersection(
                        set(
                            filtered_features_df.columns[
                                filtered_features_df.loc[cur_features].max(axis=0) > 0
                            ].values.tolist()
                        )
                    )
                )

            if sens_res == "resistant":
                add_next_best_features_to_tree(
                    source_fdf=fdf,
                    excluded_features=list(cur_features),
                    tdf=target_df,
                    cl_list=cur_cl,
                    cur_node=cur_feature_node,
                    filter_score_column=cur_filter_score_column,
                    wilcox_score_column=cur_wilcox_score_column,
                    min_feature_count=min_feature_count,
                    min_geom_pval=min_geom_pval,
                    parent_feature=cur_feature_string,
                    or_flag=or_flag,
                    max_next_feature_subsets=max_next_feature_subsets,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                    percentile=percentile,
                    min_effect_size=child_resistance_min_effect_size,
                    resistance_min_effect_size=resistance_min_effect_size,
                    strong_min_effect_size=strong_min_effect_size,
                    child_min_effect_size=child_min_effect_size,
                    child_resistance_min_effect_size=child_resistance_min_effect_size,
                    direction="resistance",
                    sensitivity_min_effect_size=min_effect_size,
                    path_features=list(cur_features),
                )
                continue

            add_next_best_features_to_tree(
                source_fdf=fdf,
                excluded_features=list(excluded),
                tdf=target_df,
                cl_list=cur_cl,
                cur_node=cur_feature_node,
                filter_score_column=cur_filter_score_column,
                wilcox_score_column=cur_wilcox_score_column,
                min_feature_count=min_feature_count,
                min_geom_pval=min_geom_pval,
                parent_feature=cur_feature_string,
                or_flag=or_flag,
                max_next_feature_subsets=max_next_feature_subsets,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                percentile=percentile,
                min_effect_size=child_min_effect_size,
                resistance_min_effect_size=resistance_min_effect_size,
                strong_min_effect_size=strong_min_effect_size,
                child_min_effect_size=child_min_effect_size,
                child_resistance_min_effect_size=child_resistance_min_effect_size,
                direction="sensitivity",
                sensitivity_min_effect_size=min_effect_size,
                path_features=list(cur_features),
            )
            if _group_effect(cur_features, scored_features) <= strong_min_effect_size:
                add_next_best_features_to_tree(
                    source_fdf=fdf,
                    excluded_features=list(cur_features),
                    tdf=target_df,
                    cl_list=cur_cl,
                    cur_node=cur_feature_node,
                    filter_score_column=cur_filter_score_column,
                    wilcox_score_column=cur_wilcox_score_column,
                    min_feature_count=min_feature_count,
                    min_geom_pval=min_geom_pval,
                    parent_feature=cur_feature_string,
                    or_flag=or_flag,
                    max_next_feature_subsets=max_next_feature_subsets,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                    percentile=percentile,
                    min_effect_size=resistance_min_effect_size,
                    resistance_min_effect_size=resistance_min_effect_size,
                    strong_min_effect_size=strong_min_effect_size,
                    child_min_effect_size=child_min_effect_size,
                    child_resistance_min_effect_size=child_resistance_min_effect_size,
                    direction="resistance",
                    sensitivity_min_effect_size=min_effect_size,
                    path_features=list(cur_features),
                )
    return root_features, target_df


def _selected_cell_count(name: str, fdf: pd.DataFrame) -> int:
    """Cell lines marked by a root feature. Inverse names count the base feature's zeros."""
    base = base_feature_name(name)
    if base not in fdf.index:
        return 10**12
    row = fdf.loc[base]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if feature_is_inverse(name):
        return int((row == 0).sum())
    return int((row == 1).sum())


def _keep_smaller_root_pair(sens_df: pd.DataFrame, res_df: pd.DataFrame, fdf: pd.DataFrame):
    """If a feature and its inverse both pass, keep the selected group with fewer cell lines.

    A tie keeps the non-inverse name when one of the two is inverse, and otherwise the
    resistant name.
    """
    if sens_df is None or sens_df.empty or res_df is None or res_df.empty:
        return sens_df, res_df
    sens_by_base: dict[str, list[str]] = {}
    for name in sens_df["feature"].astype(str):
        sens_by_base.setdefault(base_feature_name(name), []).append(name)
    res_by_base: dict[str, list[str]] = {}
    for name in res_df["feature"].astype(str):
        res_by_base.setdefault(base_feature_name(name), []).append(name)
    drop_sens: set[str] = set()
    drop_res: set[str] = set()
    for base, sens_names in sens_by_base.items():
        res_names = res_by_base.get(base, [])
        for sens_name in sens_names:
            for res_name in res_names:
                sens_n = _selected_cell_count(sens_name, fdf)
                res_n = _selected_cell_count(res_name, fdf)
                if sens_n < res_n:
                    drop_res.add(res_name)
                    kept = sens_name
                elif res_n < sens_n:
                    drop_sens.add(sens_name)
                    kept = res_name
                elif feature_is_inverse(res_name) and not feature_is_inverse(sens_name):
                    drop_res.add(res_name)
                    kept = sens_name
                else:
                    drop_sens.add(sens_name)
                    kept = res_name
                print(
                    f"Root pair {base}: keep {kept} "
                    f"(sensitive n={sens_n}, resistant n={res_n})"
                )
    if drop_sens:
        sens_df = sens_df.loc[~sens_df["feature"].astype(str).isin(drop_sens)].copy()
    if drop_res:
        res_df = res_df.loc[~res_df["feature"].astype(str).isin(drop_res)].copy()
    return sens_df, res_df


def _group_scored_features(fdf, tdf, scored, geom_pval_cutoff, wilcox_score_column):
    """Overlap-group one scored feature table. Returns groups and the materialized matrix."""
    if scored is None or scored.empty:
        return [], fdf
    fdf = materialize_inverse_features(fdf, scored["feature"].tolist())
    overlaps = generaate_overlap_matrix(
        fdf,
        scored["feature"].values.tolist(),
        scored["feature"].values.tolist(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlapping = run_hypergeom_tests(fdf, overlaps)
    groups = group_features_by_pvalue_v2(
        overlapping,
        feature_list=scored["feature"].values.tolist(),
        pvalue_threshold=geom_pval_cutoff,
        fdf=fdf,
        tdf=tdf,
        score_column=wilcox_score_column,
        feature_wilcox_pvals=scored,
    )
    return groups, fdf


def _group_effect(group, feature_set: pd.DataFrame) -> float:
    """Most negative effect size in a sensitivity group. Missing effects do not qualify."""
    if feature_set is None or feature_set.empty or "effect_size" not in feature_set.columns:
        return 0.0
    values = []
    for feature in group:
        hit = feature_set.loc[feature_set["feature"] == feature, "effect_size"]
        if len(hit):
            values.append(float(hit.iloc[0]))
    return min(values) if values else 0.0


def _feature_labels(names) -> list[str]:
    """Feature names, their base rows, and pieces joined when rows were merged."""
    return expanded_feature_labels(names)


def add_next_best_features_to_tree(
    source_fdf=None,
    excluded_features=None,
    tdf=None,
    cl_list=None,
    cur_node=None,
    filter_score_column=None,
    wilcox_score_column=None,
    min_feature_count=3,
    min_geom_pval=0.0000000001,
    parent_feature=None,
    or_flag=True,
    max_next_feature_subsets=3,
    n_permutations=50,
    n_jobs=1,
    percentile=5,
    min_effect_size=-0.2,
    resistance_min_effect_size=0.2,
    strong_min_effect_size=-0.2,
    child_min_effect_size=-0.1,
    child_resistance_min_effect_size=0.1,
    direction="sensitivity",
    sensitivity_min_effect_size=None,
    path_features=None,
):
    """Recursively add best feature groups as children to cur_node."""
    if sensitivity_min_effect_size is None:
        sensitivity_min_effect_size = min_effect_size if direction == "sensitivity" else -0.2
    path_features = list(path_features if path_features is not None else (excluded_features or []))
    blocked = _feature_labels(excluded_features or [])
    available = source_fdf.drop(blocked, errors="ignore")
    cur_fdf = restrict_features_to_celllines(available, cl_list)
    cur_tdf = tdf[tdf["cellline"].isin(cur_fdf.columns)].copy()
    if cur_fdf.empty:
        return
    if cur_tdf.empty:
        return
    if cur_fdf.shape[1] < min_feature_count:
        return

    print(
        f"Subset {direction} shuffles on {cur_fdf.shape[1]} cell lines and {cur_fdf.shape[0]} features "
        f"({n_permutations} permutations)..."
    )
    cutoffs = permutation_cutoffs_from_frames(
        tdf=cur_tdf,
        fdf=cur_fdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        n_permutations=n_permutations,
        percentile=percentile,
        n_jobs=n_jobs,
        min_effect_size=min_effect_size,
        direction=direction,
    )
    min_wilcox_pval = cutoffs["pvalue_cutoff"]
    min_wilcox_qval = cutoffs["qvalue_cutoff"]
    print(f"P-value cutoff: {min_wilcox_pval}")
    print(f"Q-value cutoff: {min_wilcox_qval}")

    filtered_feature_set = return_possible_features(
        cur_fdf,
        cur_tdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        wilcox_cutoff=min_wilcox_pval,
        qval_cutoff=min_wilcox_qval,
        min_effect_size=min_effect_size,
        direction=direction,
    )
    if filtered_feature_set.empty:
        return

    cur_fdf = materialize_inverse_features(cur_fdf, filtered_feature_set["feature"].tolist())
    overlaps = generaate_overlap_matrix(
        cur_fdf,
        filtered_feature_set["feature"].values.tolist(),
        filtered_feature_set["feature"].values.tolist(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlaping_feature_pvals = run_hypergeom_tests(cur_fdf, overlaps)

    cur_list_of_lists = group_features_by_pvalue_v2(
        overlaping_feature_pvals,
        feature_list=filtered_feature_set["feature"].values.tolist(),
        pvalue_threshold=min_geom_pval,
        fdf=cur_fdf,
        tdf=cur_tdf,
        score_column=wilcox_score_column,
        feature_wilcox_pvals=filtered_feature_set,
    )
    cur_list_of_lists = _limit_next_tree_subsets(
        cur_list_of_lists, filtered_feature_set, max_next_feature_subsets
    )
    grouped_names = [feature for group in cur_list_of_lists for feature in group]
    cur_fdf = materialize_inverse_features(cur_fdf, grouped_names)

    excluded = list(excluded_features or [])
    for cur_features in cur_list_of_lists:
        # Sibling k cannot see itself, earlier siblings, or features already excluded by the parent.
        excluded.extend(cur_features)
        cur_feature_string = ";".join(cur_features)
        print("Parent - " + parent_feature)
        print("Child - " + cur_feature_string)
        child_label = "resistant" if direction == "resistance" else "sensitive"
        cur_feature_node = FeatureNode(cur_feature_string, sens_res=child_label)
        cur_node.add_child(cur_feature_node)
        if or_flag:
            cur_cl_list = list(
                set(cur_tdf["cellline"].values.tolist()).intersection(
                    set(cur_fdf.columns[cur_fdf.loc[cur_features].max(axis=0) == 0].values.tolist())
                )
            )
        else:
            cur_cl_list = list(
                set(cur_tdf["cellline"].values.tolist()).intersection(
                    set(cur_fdf.columns[cur_fdf.loc[cur_features].max(axis=0) > 0].values.tolist())
                )
            )

        child_path = list(path_features) + list(cur_features)
        child_effect = _group_effect(cur_features, filtered_feature_set)
        if direction != "resistance":
            add_next_best_features_to_tree(
                source_fdf=source_fdf,
                excluded_features=list(excluded),
                tdf=cur_tdf,
                cl_list=cur_cl_list,
                cur_node=cur_feature_node,
                filter_score_column=filter_score_column,
                wilcox_score_column=wilcox_score_column,
                min_feature_count=min_feature_count,
                min_geom_pval=min_geom_pval,
                parent_feature=cur_feature_string,
                or_flag=or_flag,
                max_next_feature_subsets=max_next_feature_subsets,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                percentile=percentile,
                min_effect_size=child_min_effect_size,
                resistance_min_effect_size=resistance_min_effect_size,
                strong_min_effect_size=strong_min_effect_size,
                child_min_effect_size=child_min_effect_size,
                child_resistance_min_effect_size=child_resistance_min_effect_size,
                direction="sensitivity",
                sensitivity_min_effect_size=sensitivity_min_effect_size,
                path_features=child_path,
            )
        resistance_allowed = direction == "resistance" or child_effect <= strong_min_effect_size
        if resistance_allowed:
            add_next_best_features_to_tree(
                source_fdf=source_fdf,
                excluded_features=child_path,
                tdf=cur_tdf,
                cl_list=cur_cl_list,
                cur_node=cur_feature_node,
                filter_score_column=filter_score_column,
                wilcox_score_column=wilcox_score_column,
                min_feature_count=min_feature_count,
                min_geom_pval=min_geom_pval,
                parent_feature=cur_feature_string,
                or_flag=or_flag,
                max_next_feature_subsets=max_next_feature_subsets,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                percentile=percentile,
                min_effect_size=(
                    child_resistance_min_effect_size
                    if direction == "resistance"
                    else resistance_min_effect_size
                ),
                resistance_min_effect_size=resistance_min_effect_size,
                strong_min_effect_size=strong_min_effect_size,
                child_min_effect_size=child_min_effect_size,
                child_resistance_min_effect_size=child_resistance_min_effect_size,
                direction="resistance",
                sensitivity_min_effect_size=sensitivity_min_effect_size,
                path_features=child_path,
            )
