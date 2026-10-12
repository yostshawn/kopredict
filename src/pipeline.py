"""Biomarker discovery pipeline.

Exposes the main callable that the CLI uses: z-score the target dependency column,
search for candidate feature sets (Wilcoxon + Jaccard grouping), then grow
AND and OR ``FeatureNode`` trees from the same root markers. See
``run_prediction_for_biomarker_given_target_dataset``.
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
    get_target_df,
    group_features_by_jaccard,
    materialize_inverse_features,
    restrict_features_to_celllines,
    return_possible_features,
)
from permutation import _limit_blas_threads, clear_cutoff_cache, permutation_cutoffs_from_frames


def _limit_next_tree_subsets(cur_list_of_lists, filtered_feature_set: pd.DataFrame, max_subsets: int):
    """Keep at most ``max_subsets`` groups, ranked by best (min) Wilcoxon p in each group.

    ``max_subsets <= 0`` means no limit (all Jaccard groups).
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


def determine_best_biomarker_sets(
    fdf,
    tdf,
    filter_score_column,
    wilcox_score_column,
    wilcox_pval_cutoff=0.0001,
    wilcox_qval_cutoff=0.01,
    min_effect_size=-0.2,
    direction="sensitivity",
    min_group_count=3,
    jaccard_threshold=0.8,
    effect_vs_wt=False,
):
    """Run feature pipeline: Wilcoxon filter -> Jaccard group."""
    filtered_feature_set = return_possible_features(
        fdf,
        tdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        wilcox_cutoff=wilcox_pval_cutoff,
        qval_cutoff=wilcox_qval_cutoff,
        min_effect_size=min_effect_size,
        direction=direction,
        effect_vs_wt=effect_vs_wt,
        min_group_count=min_group_count,
    )
    if filtered_feature_set.empty:
        return [], filtered_feature_set

    filtered_feature_set = _keep_smaller_direction_pair(filtered_feature_set, fdf)
    if filtered_feature_set.empty:
        return [], filtered_feature_set
    fdf = materialize_inverse_features(fdf, filtered_feature_set["feature"].tolist())
    groups = group_features_by_jaccard(
        fdf,
        filtered_feature_set["feature"].values.tolist(),
        feature_wilcox_pvals=filtered_feature_set,
        jaccard_threshold=jaccard_threshold,
    )
    return groups, filtered_feature_set


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


def _cell_lines_for_logic(tdf, fdf, cur_features, or_flag):
    """Cell lines for the next step: AND keeps carriers; OR keeps non-carriers."""
    present = fdf.loc[cur_features].max(axis=0)
    keep = present == 0 if or_flag else present > 0
    return list(
        set(tdf["cellline"].values.tolist()).intersection(
            set(fdf.columns[keep].values.tolist())
        )
    )


def _node_from_group(group, scored_features, sens_res):
    """Build a FeatureNode with discovery stats from the best-p feature in the group."""
    cur_feature_string = ";".join(group)
    best_name = group[0]
    best_p = None
    best_effect = None
    n_mut = None
    n_ref = None
    if scored_features is not None and not scored_features.empty:
        rows = scored_features.loc[scored_features["feature"].astype(str).isin(group)]
        if not rows.empty:
            top = rows.sort_values("p_value", ascending=True).iloc[0]
            best_name = str(top["feature"])
            best_p = float(top["p_value"])
            if "effect_size" in top.index and pd.notna(top["effect_size"]):
                best_effect = float(top["effect_size"])
            if "n_mutant" in top.index and pd.notna(top["n_mutant"]):
                n_mut = int(top["n_mutant"])
            if "n_reference" in top.index and pd.notna(top["n_reference"]):
                n_ref = int(top["n_reference"])
    return FeatureNode(
        cur_feature_string,
        sens_res=sens_res,
        discovery_p=best_p,
        discovery_effect=best_effect,
        discovery_feature=best_name,
        n_mutant=n_mut,
        n_reference=n_ref,
    )


def _grow_tree_from_root_groups(
    *,
    target,
    root_passes,
    source_fdf,
    filtered_features_df,
    target_df,
    or_flag,
    cur_filter_score_column,
    cur_wilcox_score_column,
    min_feature_count,
    max_next_feature_subsets,
    n_permutations,
    n_jobs,
    percentile,
    min_effect_size,
    resistance_min_effect_size,
    strong_min_effect_size,
    child_min_effect_size,
    child_resistance_min_effect_size,
    min_group_count,
    jaccard_threshold,
):
    """Attach shared root groups and grow later steps under one AND or OR rule."""
    root_features = FeatureNode(target)
    logic = "OR" if or_flag else "AND"
    # OR same-direction children: ±0.2 vs WT. OR direction switches: ±0.2 vs parent.
    # AND children: parent-relative ±0.1 same-direction / ±0.2 on a switch.
    if or_flag:
        same_sens_es = min_effect_size
        same_res_es = resistance_min_effect_size
        same_vs_wt = True
    else:
        same_sens_es = child_min_effect_size
        same_res_es = child_resistance_min_effect_size
        same_vs_wt = False
    print(f"Growing {logic} tree from shared root markers...")
    shared_kwargs = dict(
        source_fdf=source_fdf,
        tdf=target_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        min_feature_count=min_feature_count,
        or_flag=or_flag,
        max_next_feature_subsets=max_next_feature_subsets,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        percentile=percentile,
        resistance_min_effect_size=resistance_min_effect_size,
        strong_min_effect_size=strong_min_effect_size,
        child_min_effect_size=child_min_effect_size,
        child_resistance_min_effect_size=child_resistance_min_effect_size,
        sensitivity_min_effect_size=min_effect_size,
        min_group_count=min_group_count,
        jaccard_threshold=jaccard_threshold,
    )
    for sens_res, list_of_list_features, scored_features in root_passes:
        excluded = []
        for cur_features in list_of_list_features:
            # Sibling k cannot see itself or any earlier sibling on this row.
            excluded.extend(cur_features)
            cur_feature_string = ";".join(cur_features)
            node_label = "resistant" if sens_res == "resistant" else "sensitive"
            cur_feature_node = _node_from_group(cur_features, scored_features, node_label)
            root_features.add_child(cur_feature_node)
            cur_cl = _cell_lines_for_logic(target_df, filtered_features_df, cur_features, or_flag)
            if len(cur_cl) < min_feature_count:
                continue

            if sens_res == "resistant":
                add_next_best_features_to_tree(
                    excluded_features=list(excluded),
                    cl_list=cur_cl,
                    cur_node=cur_feature_node,
                    parent_feature=cur_feature_string,
                    min_effect_size=same_res_es,
                    direction="resistance",
                    path_features=list(cur_features),
                    effect_vs_wt=same_vs_wt,
                    **shared_kwargs,
                )
                continue

            add_next_best_features_to_tree(
                excluded_features=list(excluded),
                cl_list=cur_cl,
                cur_node=cur_feature_node,
                parent_feature=cur_feature_string,
                min_effect_size=same_sens_es,
                direction="sensitivity",
                path_features=list(cur_features),
                effect_vs_wt=same_vs_wt,
                **shared_kwargs,
            )
            if _group_effect(cur_features, scored_features) <= strong_min_effect_size:
                add_next_best_features_to_tree(
                    excluded_features=list(excluded),
                    cl_list=cur_cl,
                    cur_node=cur_feature_node,
                    parent_feature=cur_feature_string,
                    min_effect_size=resistance_min_effect_size,
                    direction="resistance",
                    path_features=list(cur_features),
                    effect_vs_wt=False,
                    **shared_kwargs,
                )
    return root_features


def run_prediction_for_biomarker_given_target_dataset(
    target=None,
    dataset=None,
    fdf=None,
    cts=None,
    background_options=None,
    min_feature_count=3,
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
    min_group_count=3,
    jaccard_threshold=0.8,
):
    """Run full biomarker prediction pipeline for target/dataset.

    Root scoring and grouping run once. AND and OR trees then grow from the same
    first-level markers. Returns ``(and_tree, or_tree, target_df)``. Either tree is
    ``None`` when no root feature groups are found.
    """
    clear_cutoff_cache()
    if n_jobs is not None and int(n_jobs) > 1:
        _limit_blas_threads()

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
        print("[debug] Initial cohort Wilcoxon ranking (same inputs as first determine_best_biomarker_sets step):")
        debug_print_top_wilcoxon_features(
            filtered_features_df,
            target_df,
            filter_score_column=cur_filter_score_column,
            wilcox_score_column=cur_wilcox_score_column,
            top_n=int(debug_wilcox_top_n),
            min_effect_size=min_effect_size,
        )

    print(
        f"Root sensitivity shuffles on {filtered_features_df.shape[1]} cell lines and "
        f"{filtered_features_df.shape[0]} features ({n_permutations} permutations)..."
    )
    sensitivity_cutoffs = permutation_cutoffs_from_frames(
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
    print(f"Sensitivity p-value cutoff: {sensitivity_cutoffs['pvalue_cutoff']}")
    print(f"Sensitivity q-value cutoff: {sensitivity_cutoffs['qvalue_cutoff']}")
    sens_scored = return_possible_features(
        filtered_features_df,
        target_df,
        filter_score_column=cur_filter_score_column,
        wilcox_score_column=cur_wilcox_score_column,
        wilcox_cutoff=sensitivity_cutoffs["pvalue_cutoff"],
        qval_cutoff=sensitivity_cutoffs["qvalue_cutoff"],
        min_effect_size=min_effect_size,
        direction="sensitivity",
        min_group_count=min_group_count,
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
        min_group_count=min_group_count,
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
        min_group_count=min_group_count,
    )
    sens_scored, res_scored = _keep_smaller_root_pair(sens_scored, res_scored, filtered_features_df)
    sens_groups, filtered_features_df = _group_scored_features(
        filtered_features_df, sens_scored, jaccard_threshold=jaccard_threshold
    )
    res_groups, filtered_features_df = _group_scored_features(
        filtered_features_df, res_scored, jaccard_threshold=jaccard_threshold
    )

    if not sens_groups and not res_groups:
        return None, None, target_df

    root_passes = (
        ("sensitive", sens_groups, sens_scored),
        ("resistant", res_groups, res_scored),
    )
    grow_kwargs = dict(
        target=target,
        root_passes=root_passes,
        source_fdf=fdf,
        filtered_features_df=filtered_features_df,
        target_df=target_df,
        cur_filter_score_column=cur_filter_score_column,
        cur_wilcox_score_column=cur_wilcox_score_column,
        min_feature_count=min_feature_count,
        max_next_feature_subsets=max_next_feature_subsets,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        percentile=percentile,
        min_effect_size=min_effect_size,
        resistance_min_effect_size=resistance_min_effect_size,
        strong_min_effect_size=strong_min_effect_size,
        child_min_effect_size=child_min_effect_size,
        child_resistance_min_effect_size=child_resistance_min_effect_size,
        min_group_count=min_group_count,
        jaccard_threshold=jaccard_threshold,
    )
    and_tree = _grow_tree_from_root_groups(or_flag=False, **grow_kwargs)
    or_tree = _grow_tree_from_root_groups(or_flag=True, **grow_kwargs)
    return and_tree, or_tree, target_df


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


def _keep_smaller_direction_pair(scored_df: pd.DataFrame, fdf: pd.DataFrame) -> pd.DataFrame:
    """Within one scored table, drop the larger of a feature / inverse pair."""
    if scored_df is None or scored_df.empty:
        return scored_df
    by_base: dict[str, list[str]] = {}
    for name in scored_df["feature"].astype(str):
        by_base.setdefault(base_feature_name(name), []).append(name)
    drop: set[str] = set()
    for base, names in by_base.items():
        forwards = [n for n in names if not feature_is_inverse(n)]
        inverses = [n for n in names if feature_is_inverse(n)]
        if not forwards or not inverses:
            continue
        for fwd in forwards:
            for inv in inverses:
                fwd_n = _selected_cell_count(fwd, fdf)
                inv_n = _selected_cell_count(inv, fdf)
                if fwd_n < inv_n:
                    drop.add(inv)
                    kept = fwd
                elif inv_n < fwd_n:
                    drop.add(fwd)
                    kept = inv
                else:
                    drop.add(inv)
                    kept = fwd
                print(
                    f"Direction pair {base}: keep {kept} "
                    f"(forward n={fwd_n}, inverse n={inv_n})"
                )
    if drop:
        scored_df = scored_df.loc[~scored_df["feature"].astype(str).isin(drop)].copy()
    return scored_df


def _group_scored_features(fdf, scored, jaccard_threshold=0.8):
    """Jaccard-group one scored feature table. Returns groups and the materialized matrix."""
    if scored is None or scored.empty:
        return [], fdf
    scored = _keep_smaller_direction_pair(scored, fdf)
    if scored.empty:
        return [], fdf
    fdf = materialize_inverse_features(fdf, scored["feature"].tolist())
    groups = group_features_by_jaccard(
        fdf,
        scored["feature"].values.tolist(),
        feature_wilcox_pvals=scored,
        jaccard_threshold=jaccard_threshold,
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


def add_next_best_features_to_tree(
    source_fdf=None,
    excluded_features=None,
    tdf=None,
    cl_list=None,
    cur_node=None,
    filter_score_column=None,
    wilcox_score_column=None,
    min_feature_count=3,
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
    effect_vs_wt=False,
    min_group_count=3,
    jaccard_threshold=0.8,
):
    """Recursively add best feature groups as children to cur_node."""
    if sensitivity_min_effect_size is None:
        sensitivity_min_effect_size = min_effect_size if direction == "sensitivity" else -0.2
    if not cl_list or len(cl_list) < min_feature_count:
        return
    path_features = list(path_features if path_features is not None else (excluded_features or []))
    blocked = expanded_feature_labels(excluded_features or [])
    available = source_fdf.drop(blocked, errors="ignore")
    cur_fdf = restrict_features_to_celllines(available, cl_list)
    cur_tdf = tdf[tdf["cellline"].isin(cur_fdf.columns)].copy()
    if cur_fdf.empty or cur_tdf.empty or cur_fdf.shape[1] < min_feature_count:
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
        effect_vs_wt=effect_vs_wt,
        min_group_count=min_group_count,
    )
    min_wilcox_pval = cutoffs["pvalue_cutoff"]
    min_wilcox_qval = cutoffs["qvalue_cutoff"]
    print(f"P-value cutoff: {min_wilcox_pval}")
    print(f"Q-value cutoff: {min_wilcox_qval}")

    cur_list_of_lists, filtered_feature_set = determine_best_biomarker_sets(
        cur_fdf,
        cur_tdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        wilcox_pval_cutoff=min_wilcox_pval,
        wilcox_qval_cutoff=min_wilcox_qval,
        min_effect_size=min_effect_size,
        direction=direction,
        min_group_count=min_group_count,
        jaccard_threshold=jaccard_threshold,
        effect_vs_wt=effect_vs_wt,
    )
    if not cur_list_of_lists:
        return

    cur_list_of_lists = _limit_next_tree_subsets(
        cur_list_of_lists, filtered_feature_set, max_next_feature_subsets
    )
    grouped_names = [feature for group in cur_list_of_lists for feature in group]
    cur_fdf = materialize_inverse_features(cur_fdf, grouped_names)

    excluded = list(excluded_features or [])
    shared_child_kwargs = dict(
        source_fdf=source_fdf,
        tdf=cur_tdf,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
        min_feature_count=min_feature_count,
        or_flag=or_flag,
        max_next_feature_subsets=max_next_feature_subsets,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        percentile=percentile,
        resistance_min_effect_size=resistance_min_effect_size,
        strong_min_effect_size=strong_min_effect_size,
        child_min_effect_size=child_min_effect_size,
        child_resistance_min_effect_size=child_resistance_min_effect_size,
        sensitivity_min_effect_size=sensitivity_min_effect_size,
        min_group_count=min_group_count,
        jaccard_threshold=jaccard_threshold,
    )
    for cur_features in cur_list_of_lists:
        # Sibling k cannot see itself, earlier siblings, or features already excluded by the parent.
        excluded.extend(cur_features)
        cur_feature_string = ";".join(cur_features)
        print("Parent - " + parent_feature)
        print("Child - " + cur_feature_string)
        child_label = "resistant" if direction == "resistance" else "sensitive"
        cur_feature_node = _node_from_group(cur_features, filtered_feature_set, child_label)
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
        if len(cur_cl_list) < min_feature_count:
            continue

        child_path = list(path_features) + list(cur_features)
        child_effect = _group_effect(cur_features, filtered_feature_set)
        if or_flag:
            next_sens_es = sensitivity_min_effect_size
            next_sens_vs_wt = True
            next_same_res_es = resistance_min_effect_size
            next_same_res_vs_wt = True
            next_switch_res_es = resistance_min_effect_size
            next_switch_res_vs_wt = False
        else:
            next_sens_es = child_min_effect_size
            next_sens_vs_wt = False
            next_same_res_es = child_resistance_min_effect_size
            next_same_res_vs_wt = False
            next_switch_res_es = resistance_min_effect_size
            next_switch_res_vs_wt = False
        if direction != "resistance":
            add_next_best_features_to_tree(
                excluded_features=list(excluded),
                cl_list=cur_cl_list,
                cur_node=cur_feature_node,
                parent_feature=cur_feature_string,
                min_effect_size=next_sens_es,
                direction="sensitivity",
                path_features=child_path,
                effect_vs_wt=next_sens_vs_wt,
                **shared_child_kwargs,
            )
        resistance_allowed = direction == "resistance" or child_effect <= strong_min_effect_size
        if resistance_allowed:
            add_next_best_features_to_tree(
                excluded_features=list(excluded),
                cl_list=cur_cl_list,
                cur_node=cur_feature_node,
                parent_feature=cur_feature_string,
                min_effect_size=(
                    next_same_res_es if direction == "resistance" else next_switch_res_es
                ),
                direction="resistance",
                path_features=child_path,
                effect_vs_wt=(
                    next_same_res_vs_wt if direction == "resistance" else next_switch_res_vs_wt
                ),
                **shared_child_kwargs,
            )
