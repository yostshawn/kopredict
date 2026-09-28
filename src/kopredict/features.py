"""Feature processing and selection (from 20_find_best_union_features).

Typical use: align the binary feature matrix (rows = feature IDs, columns = cellline)
with the target dependency table, z-score dep scores, filter features by *mean* dep
(using z-scored dep), rank by Wilcoxon on **raw** dep where required, then hypergeom
overlap and grouping (see ``pipeline.determine_best_biomrker_sets``).
"""

import sys
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from .utils import sig_figs
from .database import pull_target_gene_dependency_scores


class FeatureNode:
    """Tree node for feature groups. convert_to_list_of_lists() returns paths as list of lists."""

    def __init__(self, f):
        self.feature = f
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def convert_to_list_of_lists(self, current_path=None):
        if current_path is None:
            current_path = []
        if not self.children:
            return [current_path + [self.feature]]
        all_paths = []
        for child in self.children:
            all_paths.extend(child.convert_to_list_of_lists(current_path + [self.feature]))
        return all_paths


def cleanup_featureset(fdf, mindiff=3):
    """Clean up feature dataframe: remove duplicates, filter by min ones/zeros."""
    cur_fdf = fdf.fillna(0)
    ones = (cur_fdf == 1).sum(axis=1)
    zeros = (cur_fdf == 0).sum(axis=1)
    valid_rows = (ones >= mindiff) & (zeros >= mindiff)
    cur_fdf = cur_fdf.loc[valid_rows]

    is_duplicate = cur_fdf.duplicated(keep=False)
    unique_features_df = cur_fdf[~is_duplicate]
    duplicate_features_df = cur_fdf[is_duplicate]

    if duplicate_features_df.empty:
        return unique_features_df

    dup = duplicate_features_df.reset_index()
    index_col = dup.columns[0]
    dup["_hash"] = pd.util.hash_pandas_object(duplicate_features_df, index=False).values
    merged = dup.groupby("_hash", sort=False).agg(
        new_idx=(index_col, lambda x: ";".join(x.astype(str))),
        **{c: (c, "first") for c in duplicate_features_df.columns},
    ).set_index("new_idx")
    merged.index.name = None
    return pd.concat([unique_features_df, merged])


def _hash_feature_rows(encoded: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(encoded)
    pad = (-data.shape[1]) % 8
    if pad:
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
    folded = data.view(np.uint64)
    hashes = np.zeros(data.shape[0], dtype=np.uint64)
    for column in range(folded.shape[1]):
        hashes ^= folded[:, column] * np.uint64(0x9E3779B97F4A7C15 + column)
    return hashes


def _merge_exact_feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine rows that are identical on 0, 1, and missing cells."""
    if len(frame) < 2:
        return frame
    values = frame.to_numpy(dtype=np.float32, copy=False)
    encoded = np.zeros(values.shape, dtype=np.uint8)
    encoded[np.equal(values, 1)] = 1
    encoded[np.isnan(values)] = 255
    hashes = _hash_feature_rows(encoded)
    groups: dict[int, list[int]] = {}
    name_groups: dict[int, list[str]] = {}
    representatives: list[int] = []
    names = frame.index.astype(str).tolist()
    for position, key in enumerate(hashes.tolist()):
        row = encoded[position]
        match = None
        for representative in groups.get(int(key), ()):
            if np.array_equal(row, encoded[representative]):
                match = representative
                break
        if match is None:
            groups.setdefault(int(key), []).append(position)
            name_groups[position] = [names[position]]
            representatives.append(position)
        else:
            name_groups[match].append(names[position])
    chosen = frame.iloc[representatives].copy()
    chosen.index = [";".join(dict.fromkeys(name_groups[rep])) for rep in representatives]
    chosen.index.name = None
    return chosen


def restrict_features_to_celllines(fdf: pd.DataFrame, celllines) -> pd.DataFrame:
    """Subset columns, drop constant rows, and merge identical feature rows.

    A row is constant when the selected cell lines contain no 0, no 1, or only
    missing values. Missing cells are not filled in as 0 before that check or
    before identical rows are combined.
    """
    columns = fdf.columns.intersection(pd.Index(list(celllines)))
    subset = fdf.loc[:, columns]
    if subset.empty:
        return subset
    values = subset.to_numpy(dtype=np.float32, copy=False)
    has_one = np.nansum(values == 1, axis=1) > 0
    has_zero = np.nansum(values == 0, axis=1) > 0
    subset = subset.loc[has_one & has_zero]
    if subset.empty:
        return subset
    return _merge_exact_feature_rows(subset)


def cellline_interesect(tdf, fdf, input_list=None):
    """Intersect cell lines between target and feature dataframes."""
    target_cl = pd.Index(tdf["cellline"].unique())
    if input_list is not None:
        target_cl = target_cl.intersection(pd.Index(input_list))
    return fdf.columns.intersection(target_cl).tolist()


def filter_features_by_mean(score_column=None, fdf=None, tdf=None):
    """Return feature index names that pass the mean *z-scored* dependency filter.

    For each feature row, the weighted mean of per-cellline ``score_column`` (expected
    ``*_zscore``) is computed; features with a negative mean are retained (enrichment
    in essential lines). Returns a list of feature index strings.
    """
    if fdf.empty:
        return []
    if tdf.empty:
        sys.exit("ERROR: Need to provide target dataframe 'tdf = PANDAS'")
    if score_column is None or score_column not in tdf.columns:
        sys.exit("ERROR: Provide score column and make sure it is in the target dataframe")

    cellline_scores = tdf.groupby("cellline")[score_column].sum()
    common_celllines = fdf.columns.intersection(cellline_scores.index)
    fdf_aligned = fdf[common_celllines].fillna(0)
    scores_aligned = cellline_scores.reindex(common_celllines).fillna(0)
    feature_sums = (fdf_aligned * scores_aligned).sum(axis=1)
    return feature_sums[feature_sums < 0].index.tolist()


def get_target_df(gene, dataset):
    """Load one target gene's dependency scores as a long table ``cellline`` × ``crispr`` or ``rnai``."""
    ds = str(dataset).lower()
    if ds not in ("crispr", "rnai"):
        sys.exit(
            "ERROR: kopredict only supports dataset 'crispr' or 'rnai'. "
            "Drug mode is disabled."
        )
    try:
        target_df = pull_target_gene_dependency_scores(gene, ds)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if target_df.empty:
        sys.exit(f"ERROR: Gene {gene!r} has no {ds} scores.")
    return target_df


# Drug / PGX target mode (Redshift s3_pgx_*) — disabled in local-only build.
# def get_drug_target_df(gene):
#     cur_drugs_df = drugs_given_target(gene)
#     ...


def get_drug_target_df(gene):
    raise NotImplementedError(
        "Drug-target mode is disabled. Use dataset 'crispr' or 'rnai' with local dependency files."
    )


def add_zscore_to_df(df, value_col):
    """Add z-scored column (value_col+'_zscore') to df in-place."""
    df[value_col + "_zscore"] = stats.zscore(df[value_col])


def return_minmax_series(df, value_col, mean_scale=True):
    """Min-max scale a column to [-1, 1], optionally mean-centered."""
    min_val = df[value_col].min()
    max_val = df[value_col].max()
    scaled_series = 2 * ((df[value_col] - min_val) / (max_val - min_val)) - 1
    if mean_scale:
        return scaled_series - scaled_series.mean()
    return scaled_series


def wilcoxon_test(group, tdf, score_column):
    """Wilcoxon rank-sum test: feature-positive vs feature-negative cell lines."""
    g1 = tdf.loc[tdf["cellline"].isin(group[group == 1].index.tolist()), score_column].tolist()
    g2 = tdf.loc[tdf["cellline"].isin(group[group == 0].index.tolist()), score_column].tolist()
    if len(g1) > 0 and len(g2) > 0:
        try:
            return sig_figs(stats.ranksums(g1, g2, alternative="less")[1], 3)
        except ValueError:
            return np.nan
    return np.nan


def _sig_figs_array(values, precision=3):
    """Round finite nonzero values to ``precision`` significant figures."""
    out = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(out) & (out != 0)
    magnitude = np.abs(out[valid])
    scale = 10.0 ** (np.floor(np.log10(magnitude)) - (precision - 1))
    out[valid] = np.round(out[valid] / scale) * scale
    return out


def ranksums_less_pvalues(features: pd.DataFrame, scores: pd.Series) -> pd.Series:
    """One-sided Wilcoxon rank-sum p-values for every feature row.

    Matches ``scipy.stats.ranksums(..., alternative='less')`` on the cell lines
    where the feature is 0 or 1. Ranks are computed once for each shared
    missingness pattern, then applied to every row with that pattern.
    """
    if features.empty:
        return pd.Series(dtype=float, index=features.index)
    columns = features.columns.astype(str)
    score_values = scores.reindex(columns).to_numpy(dtype=float)
    usable = np.isfinite(score_values)
    matrix = features.to_numpy(dtype=np.float32, copy=False)
    if matrix.shape[1] != len(usable):
        matrix = features.reindex(columns=columns).to_numpy(dtype=np.float32, copy=False)
    matrix = matrix[:, usable]
    score_values = score_values[usable]
    pvalues = np.full(matrix.shape[0], np.nan, dtype=float)
    if matrix.size == 0 or matrix.shape[1] == 0:
        return pd.Series(pvalues, index=features.index)

    observed = np.isfinite(matrix)
    packed = np.packbits(observed, axis=1)
    row_keys = np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, group_ids = np.unique(row_keys, return_inverse=True)
    is_one = matrix == 1
    is_zero = matrix == 0
    for group in range(int(group_ids.max()) + 1):
        rows = np.flatnonzero(group_ids == group)
        present = observed[rows[0]]
        if not present.any():
            continue
        ranks = stats.rankdata(score_values[present])
        ones = is_one[rows][:, present]
        zeros = is_zero[rows][:, present]
        n1 = ones.sum(axis=1)
        n0 = zeros.sum(axis=1)
        rank_sum = ones @ ranks
        total = n1 + n0
        expected = n1 * (total + 1) / 2.0
        variance = n1 * n0 * (total + 1) / 12.0
        ok = (n1 > 0) & (n0 > 0) & (variance > 0)
        z = np.full(len(rows), np.nan, dtype=float)
        z[ok] = (rank_sum[ok] - expected[ok]) / np.sqrt(variance[ok])
        pvalues[rows[ok]] = stats.norm.cdf(z[ok])
    return pd.Series(_sig_figs_array(pvalues, 3), index=features.index)


def debug_wilcoxon_feature_ranking(
    fdf,
    tdf,
    score_column=None,
    top_n=10,
    filter_score_column=None,
    wilcox_score_column=None,
):
    """Wilcoxon + BH q-values for mean-filtered features (same as ``return_possible_features``), top ``top_n`` by p-value.

    Does not apply Wilcoxon p/q significance cutoffs—only ranks for inspection.
    """
    if filter_score_column is None:
        filter_score_column = score_column
    if wilcox_score_column is None:
        wilcox_score_column = score_column
    filtered_feature_list = filter_features_by_mean(
        score_column=filter_score_column, fdf=fdf, tdf=tdf
    )
    if not filtered_feature_list:
        return pd.DataFrame(columns=["feature", "p_value", "q_value"])
    scores = tdf.drop_duplicates("cellline").set_index("cellline")[wilcox_score_column]
    wilcox_pvals = ranksums_less_pvalues(fdf.loc[filtered_feature_list], scores)
    wilcox_pvals = wilcox_pvals.rename_axis("feature").reset_index(name="p_value")
    if wilcox_pvals.empty:
        return wilcox_pvals.assign(q_value=pd.Series(dtype=float))
    _, q_values, _, _ = multipletests(wilcox_pvals["p_value"], method="fdr_bh")
    wilcox_pvals["q_value"] = q_values
    return wilcox_pvals.sort_values("p_value", ascending=True).head(int(top_n))


def debug_print_top_wilcoxon_features(
    fdf,
    tdf,
    score_column=None,
    top_n=10,
    file=None,
    filter_score_column=None,
    wilcox_score_column=None,
):
    """Print top ``top_n`` mean-filtered features ranked by Wilcoxon ``p_value`` (with ``q_value``)."""
    file = file or sys.stdout
    tbl = debug_wilcoxon_feature_ranking(
        fdf,
        tdf,
        score_column=score_column,
        top_n=top_n,
        filter_score_column=filter_score_column,
        wilcox_score_column=wilcox_score_column,
    )
    show_filter_col = filter_score_column if filter_score_column is not None else score_column
    show_wilcox_col = wilcox_score_column if wilcox_score_column is not None else score_column
    if tbl.empty:
        file.write(
            f"[debug] Wilcoxon: no features after mean filter "
            f"(filter_score_column={show_filter_col!r}, wilcox_score_column={show_wilcox_col!r}).\n"
        )
        return
    file.write(
        f"[debug] Top {len(tbl)} features by Wilcoxon p-value "
        f"(mean filter + BH-FDR q-value; "
        f"filter_score_column={show_filter_col!r}, wilcox_score_column={show_wilcox_col!r}):\n"
    )
    for _, row in tbl.iterrows():
        file.write(f"  {row['feature']!s}\tp={row['p_value']}\tq={row['q_value']}\n")


def return_possible_features(
    fdf,
    tdf,
    score_column=None,
    wilcox_cutoff=0.0001,
    qval_cutoff=0.01,
    filter_score_column=None,
    wilcox_score_column=None,
):
    """Filter features by mean then run Wilcoxon/FDR.

    ``filter_score_column`` controls ``filter_features_by_mean`` (typically ``*_zscore``).
    ``wilcox_score_column`` controls Wilcoxon p-values (typically raw ``crispr``/``rnai``).
    ``score_column`` remains as a backward-compatible fallback when the two explicit
    columns are not supplied.
    """
    if filter_score_column is None:
        filter_score_column = score_column
    if wilcox_score_column is None:
        wilcox_score_column = score_column
    filtered_feature_list = filter_features_by_mean(
        score_column=filter_score_column, fdf=fdf, tdf=tdf
    )
    if not filtered_feature_list:
        return pd.DataFrame(columns=["feature", "p_value", "q_value"])
    scores = tdf.drop_duplicates("cellline").set_index("cellline")[wilcox_score_column]
    wilcox_pvals = ranksums_less_pvalues(fdf.loc[filtered_feature_list], scores)
    wilcox_pvals = wilcox_pvals.rename_axis("feature").reset_index(name="p_value")

    if not wilcox_pvals.empty:
        reject, q_values, _, _ = multipletests(wilcox_pvals["p_value"], method="fdr_bh")
        wilcox_pvals["q_value"] = q_values
        if qval_cutoff < 1:
            final_df = wilcox_pvals[
                (wilcox_pvals["q_value"] < qval_cutoff) & (wilcox_pvals["p_value"] < wilcox_cutoff)
            ]
        else:
            final_df = wilcox_pvals[wilcox_pvals["p_value"] < wilcox_cutoff]
        return final_df.sort_values(by="p_value")
    return pd.DataFrame(columns=["feature", "p_value", "q_value"])


def generaate_overlap_matrix(df, feature_set1, feature_set2):
    """Compute overlap matrix between two feature sets."""
    missing1 = [f for f in feature_set1 if f not in df.index]
    missing2 = [f for f in feature_set2 if f not in df.index]
    if missing1:
        raise KeyError(f"Features not found in DataFrame: {missing1}")
    if missing2:
        raise KeyError(f"Features not found in DataFrame: {missing2}")

    needed = list(set(feature_set1) | set(feature_set2))
    df_sub = df.loc[needed].fillna(0)
    subset_df1 = df_sub.loc[feature_set1]
    subset_df2 = df_sub.loc[feature_set2]
    overlap_values = subset_df1.values @ subset_df2.values.T
    return pd.DataFrame(overlap_values, index=subset_df1.index, columns=subset_df2.index)


def run_hypergeom_tests(feature_df, overlap_matrix):
    """Run hypergeometric tests for feature overlaps."""
    features_to_test = overlap_matrix.columns.tolist()
    feature_data = {f: feature_df.loc[f].dropna() for f in features_to_test}

    rows, k_vals, M_vals, n_vals, N_vals = [], [], [], [], []
    for feature1, feature2 in itertools.combinations(features_to_test, 2):
        f1_series = feature_data[feature1]
        f2_series = feature_data[feature2]
        common_cell_lines = f1_series.index.intersection(f2_series.index)
        f1_data = f1_series[common_cell_lines]
        f2_data = f2_series[common_cell_lines]
        M = len(common_cell_lines)
        n = (f1_data == 1).sum()
        N = (f2_data == 1).sum()
        k = overlap_matrix.loc[feature1, feature2]
        rows.append((feature1, feature2))
        k_vals.append(k)
        M_vals.append(M)
        n_vals.append(n)
        N_vals.append(N)

    k_arr = np.array(k_vals, dtype=float)
    M_arr = np.array(M_vals, dtype=float)
    n_arr = np.array(n_vals, dtype=float)
    N_arr = np.array(N_vals, dtype=float)
    valid = (M_arr > 0) & (n_arr > 0) & (N_arr > 0)
    p_vals = np.where(valid, hypergeom.sf(k_arr - 1, M_arr, n_arr, N_arr), 1.0)
    return pd.DataFrame(
        [(*r, p) for r, p in zip(rows, p_vals)], columns=["feature1", "feature2", "pvalue"]
    )


def group_features_by_pvalue(df, feature_list, pvalue_threshold=0.0000000001):
    """Group features by hypergeometric p-value (connected components)."""
    significant_df = df[df["pvalue"] < pvalue_threshold]
    G = nx.Graph()
    if not significant_df.empty:
        G.add_edges_from(zip(significant_df["feature1"], significant_df["feature2"]))
    result_list = [list(component) for component in nx.connected_components(G)]
    features_in_groups = set(itertools.chain.from_iterable(result_list))
    orphans = [f for f in dict.fromkeys(feature_list) if f not in features_in_groups]
    result_list.extend([f] for f in orphans)
    return result_list


def group_features_by_pvalue_v2(
    df,
    feature_list,
    pvalue_threshold=0.0000000001,
    fdf=None,
    tdf=None,
    score_column=None,
    feature_wilcox_pvals=None,
    wilcox_factor=10,
):
    """Group features by p-value with Wilcoxon-based ordering."""
    significant_df = df[df["pvalue"] < pvalue_threshold]
    geom_pairs = set()
    for _, row in significant_df.iterrows():
        f1, f2 = row["feature1"], row["feature2"]
        geom_pairs.add((f1, f2))
        geom_pairs.add((f2, f1))
    wilcox_check = (
        fdf is not None
        and tdf is not None
        and score_column is not None
        and feature_wilcox_pvals is not None
    )
    if wilcox_check and isinstance(feature_wilcox_pvals, pd.DataFrame):
        feature_wilcox_pvals = (
            feature_wilcox_pvals.set_index("feature")["p_value"]
            if "feature" in feature_wilcox_pvals.columns
            else feature_wilcox_pvals.iloc[:, 0]
        )
    sorted_features = sorted(
        dict.fromkeys(feature_list),
        key=lambda x: feature_wilcox_pvals.get(x, 1) if wilcox_check else 0,
    )
    result_list = []
    for feat in sorted_features:
        added = False
        for group in result_list:
            has_geom = any((feat, g) in geom_pairs for g in group)
            if not has_geom:
                continue
            if wilcox_check:
                min_wilcox = min(
                    feature_wilcox_pvals.get(feat, np.nan),
                    min(feature_wilcox_pvals.get(g, np.nan) for g in group),
                )
                union_row = (
                    (fdf.loc[feat].fillna(0) == 1) | (fdf.loc[group].fillna(0) == 1)
                ).any(axis=0).astype(int)
                union_wilcox = wilcoxon_test(union_row, tdf, score_column)
                if (
                    np.isnan(union_wilcox)
                    or np.isnan(min_wilcox)
                    or min_wilcox <= 0
                ):
                    pass
                elif union_wilcox >= wilcox_factor * min_wilcox:
                    continue
            group.append(feat)
            added = True
            break
        if not added:
            result_list.append([feat])
    orphans = [
        f
        for f in dict.fromkeys(feature_list)
        if f not in set(itertools.chain.from_iterable(result_list))
    ]
    result_list.extend([f] for f in orphans)
    return result_list


def get_p_and_q_cutoffs(x):
    """Return P and Q cutoffs for sample size N where X <= N."""
    pval_thresholds_dict = {
        "N": [10, 25, 50, 100, 500, 200000],
        "P": [0.05, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        "Q": [0.5, 0.1, 0.05, 0.01, 0.005, 0.0001],
    }
    pval_thresholds = pd.DataFrame(pval_thresholds_dict)
    try:
        if not isinstance(x, int):
            raise TypeError(f"Input X must be an integer, got {type(x).__name__}")
        lower_rows = pval_thresholds[pval_thresholds["N"] <= x]
        if lower_rows.empty:
            return f"Error: {x} is smaller than or equal to the minimum N ({pval_thresholds['N'].min()})."
        match = lower_rows.sort_values("N").tail(1)
        if match.empty:
            return f"Error: Value {x} exceeds the maximum sample size in the table."
        return match[["P", "Q"]].values.flatten().tolist()
    except TypeError as e:
        return f"Input Error: {e}"
