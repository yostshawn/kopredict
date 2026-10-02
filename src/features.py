"""Feature processing and selection (from 20_find_best_union_features).

Typical use: align the binary feature matrix (rows = feature IDs, columns = cellline)
with the target dependency table, z-score dep scores, filter features by *mean* dep
(using z-scored dep), rank by Wilcoxon on **raw** dep where required, then hypergeom
overlap and grouping (see ``pipeline.determine_best_biomrker_sets``).
"""

import sys
import itertools
import re
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from utils import sig_figs
from database import pull_target_gene_dependency_scores


class FeatureNode:
    """Tree node for feature groups. convert_to_list_of_lists() returns paths as list of lists."""

    def __init__(self, f, sens_res="sensitive"):
        self.feature = f
        self.sens_res = sens_res
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

    def convert_to_node_paths(self, current_path=None):
        """Paths of nodes, including ``sens_res`` on each child."""
        if current_path is None:
            current_path = []
        path = current_path + [self]
        if not self.children:
            return [path]
        all_paths = []
        for child in self.children:
            all_paths.extend(child.convert_to_node_paths(path))
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
    """Feature names whose 1s have a large enough negative mean-score gap."""
    if fdf.empty:
        return []
    if tdf.empty:
        sys.exit("ERROR: Need to provide target dataframe 'tdf = PANDAS'")
    if score_column is None or score_column not in tdf.columns:
        sys.exit("ERROR: Provide score column and make sure it is in the target dataframe")

    forward, _inverse = mean_filter_directions(score_column=score_column, fdf=fdf, tdf=tdf)
    return forward


INVERSE_SUFFIX = "_inverse"
_FEATURE_SEPARATORS = {";", ":"}
_FEATURE_ATOM_SPLIT = re.compile(r"([;:])")


def inverse_feature_name(name: str) -> str:
    """Add ``_inverse`` to every ``;`` or ``:`` piece that does not already have it."""
    pieces = []
    for piece in _FEATURE_ATOM_SPLIT.split(str(name)):
        if piece in _FEATURE_SEPARATORS or piece == "":
            pieces.append(piece)
        elif piece.endswith(INVERSE_SUFFIX):
            pieces.append(piece)
        else:
            pieces.append(piece + INVERSE_SUFFIX)
    return "".join(pieces)


def base_feature_name(name: str) -> str:
    """Remove ``_inverse`` from every ``;`` or ``:`` piece."""
    pieces = []
    for piece in _FEATURE_ATOM_SPLIT.split(str(name)):
        if piece in _FEATURE_SEPARATORS or piece == "":
            pieces.append(piece)
        elif piece.endswith(INVERSE_SUFFIX):
            pieces.append(piece[: -len(INVERSE_SUFFIX)])
        else:
            pieces.append(piece)
    return "".join(pieces)


def feature_is_inverse(name: str) -> bool:
    """True when any piece of a combined feature name is an inverse."""
    return any(
        piece.endswith(INVERSE_SUFFIX)
        for piece in _FEATURE_ATOM_SPLIT.split(str(name))
        if piece not in _FEATURE_SEPARATORS and piece != ""
    )


def expanded_feature_labels(names) -> list[str]:
    """Names to drop so a chosen feature and its other direction stay excluded."""
    labels = []
    for name in names:
        text = str(name)
        labels.append(text)
        labels.append(base_feature_name(text))
        for piece in _FEATURE_ATOM_SPLIT.split(text):
            if piece in _FEATURE_SEPARATORS or piece == "":
                continue
            labels.append(piece)
            labels.append(base_feature_name(piece))
    return labels


def _score_group_means(score_column, fdf, tdf):
    """Mean and count of the 1s and the 0s. Missing feature cells and scores are left out."""
    if fdf is None or fdf.empty or tdf is None or tdf.empty:
        return None
    if score_column is None or score_column not in tdf.columns:
        sys.exit("ERROR: Provide score column and make sure it is in the target dataframe")
    cellline_scores = tdf.groupby("cellline")[score_column].sum()
    common_celllines = fdf.columns.intersection(cellline_scores.index)
    if len(common_celllines) == 0:
        return None
    values = fdf.loc[:, common_celllines].to_numpy(dtype=np.float32, copy=False)
    scores = cellline_scores.reindex(common_celllines).to_numpy(dtype=np.float64)
    finite_score = np.isfinite(scores)
    values = values[:, finite_score]
    scores = scores[finite_score]
    n_ones = (values == 1).sum(axis=1)
    n_zeros = (values == 0).sum(axis=1)
    sum_ones = (values == 1) @ scores
    sum_zeros = (values == 0) @ scores
    mean_ones = np.full(len(n_ones), np.nan, dtype=np.float64)
    mean_zeros = np.full(len(n_zeros), np.nan, dtype=np.float64)
    np.divide(sum_ones, n_ones, out=mean_ones, where=n_ones > 0)
    np.divide(sum_zeros, n_zeros, out=mean_zeros, where=n_zeros > 0)
    return mean_ones, mean_zeros, n_ones.astype(np.float64), n_zeros.astype(np.float64), fdf.index


def _mean_differences(score_column, fdf, tdf) -> pd.Series:
    """Mean raw score of the 1s minus the mean of the 0s. Missing cells are left out."""
    parts = _score_group_means(score_column, fdf, tdf)
    if parts is None:
        return pd.Series(dtype=float)
    mean_ones, mean_zeros, _n_ones, _n_zeros, index = parts
    return pd.Series(mean_ones - mean_zeros, index=index)


def _parent_relative_shift(mean_ones, mean_zeros, n_ones, n_zeros):
    """Named-group mean minus the mean of every 0 and 1 in that feature's parent subset."""
    total = n_ones + n_zeros
    pooled = np.full(len(total), np.nan, dtype=np.float64)
    weighted = n_ones * mean_ones + n_zeros * mean_zeros
    np.divide(weighted, total, out=pooled, where=(total > 0) & np.isfinite(weighted))
    return mean_ones - pooled, mean_zeros - pooled


def mean_filter_directions(score_column=None, fdf=None, tdf=None, min_effect_size=-0.2, direction="sensitivity"):
    """Features whose named group moves ``min_effect_size`` from the parent mean.

    The parent mean is every cell line in the subset being tested. A sensitive hit
    must fall at least that far below it. A resistant hit must rise at least that far
    above it. Missing cells are left out. Returns ``(forward_names, inverse_names)``.
    """
    if fdf is None or fdf.empty:
        return [], []
    if tdf is None or tdf.empty:
        sys.exit("ERROR: Need to provide target dataframe 'tdf = PANDAS'")
    if score_column is None or score_column not in tdf.columns:
        sys.exit("ERROR: Provide score column and make sure it is in the target dataframe")

    parts = _score_group_means(score_column, fdf, tdf)
    if parts is None:
        return [], []
    mean_ones, mean_zeros, n_ones, n_zeros, index = parts
    forward_effect, inverse_effect = _parent_relative_shift(mean_ones, mean_zeros, n_ones, n_zeros)
    if direction == "resistance":
        limit = min_effect_size if min_effect_size > 0 else abs(min_effect_size)
        forward_keep = np.isfinite(forward_effect) & (forward_effect >= limit)
        inverse_keep = np.isfinite(inverse_effect) & (inverse_effect >= limit)
    else:
        limit = min_effect_size if min_effect_size < 0 else -abs(min_effect_size)
        forward_keep = np.isfinite(forward_effect) & (forward_effect <= limit)
        inverse_keep = np.isfinite(inverse_effect) & (inverse_effect <= limit)
    names = index.to_numpy()
    return names[forward_keep].tolist(), names[inverse_keep].tolist()


def _swap_feature_row(row: pd.Series) -> pd.Series:
    """Swap 0 and 1. Missing cells stay missing."""
    swapped = row.copy()
    swapped = swapped.mask(row == 0, 1)
    swapped = swapped.mask(row == 1, 0)
    return swapped


def _lookup_feature_row(fdf: pd.DataFrame, text: str) -> pd.Series:
    if text in fdf.index:
        row = fdf.loc[text]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    base = base_feature_name(text)
    if feature_is_inverse(text) and base in fdf.index:
        row = fdf.loc[base]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return _swap_feature_row(row)
    raise KeyError(text)


def rows_for_features(fdf: pd.DataFrame, names) -> pd.DataFrame:
    """Rows for ``names``. Inverse pieces swap 0 and 1 on the base feature."""
    columns = fdf.columns
    selected = []
    index = []
    for name in names:
        text = str(name).strip()
        if text == "":
            continue
        row = _lookup_feature_row(fdf, text)
        selected.append(row.to_numpy())
        index.append(text)
    if not selected:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(np.vstack(selected), index=index, columns=columns)


def rows_for_block(fdf: pd.DataFrame, block_string: str) -> pd.DataFrame:
    """Resolve one tree block, including a merged name such as ``A_inverse;B_inverse``."""
    text = str(block_string).strip()
    try:
        return rows_for_features(fdf, [text])
    except KeyError:
        parts = [part.strip() for part in text.split(";") if part.strip()]
        if len(parts) <= 1:
            raise
        return rows_for_features(fdf, parts)


def materialize_inverse_features(fdf: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Append swapped rows for inverse names that are not already in ``fdf``."""
    extras = []
    seen = set()
    for name in feature_names:
        text = str(name)
        if not feature_is_inverse(text) or text in fdf.index or text in seen:
            continue
        base = base_feature_name(text)
        if base not in fdf.index:
            continue
        seen.add(text)
        row = fdf.loc[base]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        extras.append(_swap_feature_row(row).rename(text))
    if not extras:
        return fdf
    added = pd.DataFrame(extras)
    added = added.reindex(columns=fdf.columns)
    return pd.concat([fdf, added])


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


def wilcoxon_test(group, tdf, score_column, alternative="less"):
    """Wilcoxon rank-sum test: feature-positive vs feature-negative cell lines."""
    g1 = tdf.loc[tdf["cellline"].isin(group[group == 1].index.tolist()), score_column].tolist()
    g2 = tdf.loc[tdf["cellline"].isin(group[group == 0].index.tolist()), score_column].tolist()
    if len(g1) > 0 and len(g2) > 0:
        try:
            return sig_figs(stats.ranksums(g1, g2, alternative=alternative)[1], 3)
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


def ranksums_less_pvalues(features: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Wilcoxon p-values for the 1s (``p_forward``) and the 0s (``p_inverse``).

    Both columns use ``alternative='less'`` on that group. Ranks are computed once
    for each shared missingness pattern. ``p_inverse`` is the other tail of the same z.
    """
    if features.empty:
        return _directional_frame(features.index, np.array([]), np.array([]))
    columns = features.columns.astype(str)
    score_values = scores.reindex(columns).to_numpy(dtype=float)
    usable = np.isfinite(score_values)
    matrix = features.to_numpy(dtype=np.float32, copy=False)
    if matrix.shape[1] != len(usable):
        matrix = features.reindex(columns=columns).to_numpy(dtype=np.float32, copy=False)
    matrix = matrix[:, usable]
    score_values = score_values[usable]
    p_forward = np.full(matrix.shape[0], np.nan, dtype=float)
    p_inverse = np.full(matrix.shape[0], np.nan, dtype=float)
    if matrix.size == 0 or matrix.shape[1] == 0:
        return _directional_frame(features.index, p_forward, p_inverse)

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
        p_forward[rows[ok]] = stats.norm.cdf(z[ok])
        # The 0s are more essential when the 1s' z is positive.
        p_inverse[rows[ok]] = stats.norm.sf(z[ok])
    return _directional_frame(features.index, p_forward, p_inverse)


def _directional_frame(index, p_forward, p_inverse) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p_forward": _sig_figs_array(p_forward, 3),
            "p_inverse": _sig_figs_array(p_inverse, 3),
        },
        index=index,
    )


def directional_wilcoxon_table(
    fdf, tdf, filter_score_column, wilcox_score_column, min_effect_size=-0.2, direction="sensitivity"
) -> pd.DataFrame:
    """Wilcoxon rows for both group directions, with one BH correction across them.

    ``direction`` is ``sensitivity`` (lower scores) or ``resistance`` (higher scores).
    ``effect_size`` is the named group's mean minus the mean of the parent subset.
    """
    empty = pd.DataFrame(columns=["feature", "p_value", "q_value", "effect_size"])
    if fdf is None or fdf.empty:
        return empty
    effects = _mean_differences(wilcox_score_column, fdf, tdf)
    parent_shift = None
    parts = _score_group_means(wilcox_score_column, fdf, tdf)
    if parts is not None:
        mean_ones, mean_zeros, n_ones, n_zeros, index = parts
        forward_shift, inverse_shift = _parent_relative_shift(mean_ones, mean_zeros, n_ones, n_zeros)
        parent_shift = (
            pd.Series(forward_shift, index=index),
            pd.Series(inverse_shift, index=index),
        )
    forward, inverse = mean_filter_directions(
        score_column=wilcox_score_column,
        fdf=fdf,
        tdf=tdf,
        min_effect_size=min_effect_size,
        direction=direction,
    )
    ordered = list(dict.fromkeys([*forward, *inverse]))
    if not ordered:
        return empty
    scores = tdf.drop_duplicates("cellline").set_index("cellline")[wilcox_score_column]
    both = ranksums_less_pvalues(fdf.loc[ordered], scores)
    rows = []
    forward_set = set(map(str, forward))
    inverse_set = set(map(str, inverse))
    for name in ordered:
        key = str(name)
        raw_effect = effects.get(name, np.nan)
        if isinstance(raw_effect, pd.Series):
            raw_effect = raw_effect.iloc[0]
        effect = float(raw_effect) if pd.notna(raw_effect) else np.nan
        if key in forward_set:
            p_value = both.at[name, "p_inverse" if direction == "resistance" else "p_forward"]
            if parent_shift is not None:
                effect = float(parent_shift[0].get(name, np.nan))
            if np.isfinite(p_value):
                rows.append((key, float(p_value), effect))
        if key in inverse_set:
            p_value = both.at[name, "p_forward" if direction == "resistance" else "p_inverse"]
            if parent_shift is not None:
                effect = float(parent_shift[1].get(name, np.nan))
            else:
                effect = -effect
            if np.isfinite(p_value):
                rows.append((inverse_feature_name(key), float(p_value), effect))
    table = pd.DataFrame(rows, columns=["feature", "p_value", "effect_size"])
    if table.empty:
        return empty
    _reject, q_values, _, _ = multipletests(table["p_value"], method="fdr_bh")
    table["q_value"] = q_values
    return table


class PermutationLayout:
    """Feature matrix and missingness groups reused by every label shuffle.

    Scores are the only thing that changes between permutations. The binary matrix,
    ones/zeros counts, and shared missingness groups are built once.
    """

    def __init__(self, matrix, row_labels, row_scores, columns, group_ids, present_masks):
        self.matrix = matrix
        self.row_labels = np.asarray(row_labels)
        self.row_scores = np.asarray(row_scores, dtype=np.float64)
        self.columns = list(columns)
        self.column_pos = {column: i for i, column in enumerate(self.columns)}
        self.group_ids = group_ids
        self.present_masks = present_masks
        self.n_ones = (matrix == 1).sum(axis=1).astype(np.float64)
        self.n_zeros = (matrix == 0).sum(axis=1).astype(np.float64)
        self.is_one = matrix == 1
        self.is_zero = matrix == 0
        self.scores_all_finite = bool(np.isfinite(self.row_scores).all())


def _missingness_groups(observed):
    """Group feature rows that share one missingness pattern."""
    if observed.size == 0:
        return np.zeros(observed.shape[0], dtype=int), []
    packed = np.packbits(observed, axis=1)
    if packed.shape[1] == 0:
        packed = np.zeros((observed.shape[0], 1), dtype=np.uint8)
    row_keys = np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, group_ids = np.unique(row_keys, return_inverse=True)
    present_masks = []
    for group in range(int(group_ids.max()) + 1 if len(group_ids) else 0):
        member = np.flatnonzero(group_ids == group)[0]
        present_masks.append(observed[member])
    return group_ids, present_masks


def prepare_permutation_layout(fdf, tdf, score_column):
    """Align features and scores once for a series of label shuffles."""
    empty = PermutationLayout(
        np.zeros((0, 0), dtype=np.float32),
        np.array([]),
        np.array([]),
        [],
        np.zeros(0, dtype=int),
        [],
    )
    if fdf is None or fdf.empty or tdf is None or tdf.empty or score_column not in tdf.columns:
        return empty
    row_labels = tdf["cellline"].to_numpy()
    row_scores = tdf[score_column].to_numpy(dtype=np.float64, copy=True)
    common = fdf.columns.intersection(pd.unique(row_labels))
    if len(common) == 0:
        return empty
    matrix = np.ascontiguousarray(fdf.loc[:, common].to_numpy(dtype=np.float32, copy=False))
    observed = np.isfinite(matrix)
    group_ids, present_masks = _missingness_groups(observed)
    return PermutationLayout(matrix, row_labels, row_scores, common, group_ids, present_masks)


def _assign_shuffled_scores(layout, shuffled_labels):
    """Map a label permutation onto Wilcoxon scores and mean-filter scores.

    A cell line keeps the first score in row order, matching ``drop_duplicates``.
    The mean filter uses a group sum, and a group of only missing scores sums to 0.
    """
    n_col = len(layout.columns)
    score_w = np.full(n_col, np.nan, dtype=np.float64)
    score_m = np.zeros(n_col, dtype=np.float64)
    seen = np.zeros(n_col, dtype=bool)
    pos = layout.column_pos
    for label, score in zip(shuffled_labels, layout.row_scores):
        column = pos.get(label)
        if column is None:
            continue
        if not seen[column]:
            score_w[column] = score
            seen[column] = True
        if score == score:
            score_m[column] += score
    return score_w, score_m


def _effect_keep_mask(mean_ones, mean_zeros, n_ones, n_zeros, min_effect_size, direction):
    forward_effect, inverse_effect = _parent_relative_shift(mean_ones, mean_zeros, n_ones, n_zeros)
    if direction == "resistance":
        limit = min_effect_size if min_effect_size > 0 else abs(min_effect_size)
        forward = np.isfinite(forward_effect) & (forward_effect >= limit)
        inverse = np.isfinite(inverse_effect) & (inverse_effect >= limit)
    else:
        limit = min_effect_size if min_effect_size < 0 else -abs(min_effect_size)
        forward = np.isfinite(forward_effect) & (forward_effect <= limit)
        inverse = np.isfinite(inverse_effect) & (inverse_effect <= limit)
    return forward, inverse


def _group_means_for_layout(layout, score_m):
    sum_ones = layout.is_one @ score_m
    sum_zeros = layout.is_zero @ score_m
    mean_ones = np.full(len(layout.n_ones), np.nan, dtype=np.float64)
    mean_zeros = np.full(len(layout.n_zeros), np.nan, dtype=np.float64)
    np.divide(sum_ones, layout.n_ones, out=mean_ones, where=layout.n_ones > 0)
    np.divide(sum_zeros, layout.n_zeros, out=mean_zeros, where=layout.n_zeros > 0)
    return mean_ones, mean_zeros, layout.n_ones, layout.n_zeros


def _ranksum_tails(matrix, scores, group_ids, present_masks, selected):
    """Rounded one-sided p-values for ``selected`` rows. Other rows stay NaN."""
    p_forward = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    p_inverse = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    if not np.any(selected):
        return p_forward, p_inverse
    chosen = np.flatnonzero(selected)
    is_one = matrix == 1
    is_zero = matrix == 0
    for group in np.unique(group_ids[chosen]):
        rows = chosen[group_ids[chosen] == group]
        present = present_masks[group]
        if not np.any(present):
            continue
        ranks = stats.rankdata(scores[present])
        ones = is_one[rows][:, present]
        zeros = is_zero[rows][:, present]
        n1 = ones.sum(axis=1)
        n0 = zeros.sum(axis=1)
        rank_sum = ones @ ranks
        total = n1 + n0
        expected = n1 * (total + 1) / 2.0
        variance = n1 * n0 * (total + 1) / 12.0
        ok = (n1 > 0) & (n0 > 0) & (variance > 0)
        z = np.full(len(rows), np.nan, dtype=np.float64)
        z[ok] = (rank_sum[ok] - expected[ok]) / np.sqrt(variance[ok])
        p_forward[rows[ok]] = stats.norm.cdf(z[ok])
        p_inverse[rows[ok]] = stats.norm.sf(z[ok])
    return _sig_figs_array(p_forward), _sig_figs_array(p_inverse)


def permutation_null_for_seed(layout, random_state, min_effect_size=-0.2, direction="sensitivity"):
    """P-values and q-values for one label shuffle on a prepared layout.

    Same order and rounding as ``directional_wilcoxon_table`` followed by ``p < 1``.
    """
    if layout.matrix.size == 0 or len(layout.columns) == 0:
        return [], []
    rng = np.random.default_rng(random_state)
    shuffled_labels = rng.permutation(layout.row_labels)
    score_w, score_m = _assign_shuffled_scores(layout, shuffled_labels)
    forward, inverse = _effect_keep_mask(
        *_group_means_for_layout(layout, score_m), min_effect_size, direction
    )
    selected = forward | inverse
    if not np.any(selected):
        return [], []
    finite_scores = np.isfinite(score_w)
    if layout.scores_all_finite and finite_scores.all():
        p_forward, p_inverse = _ranksum_tails(
            layout.matrix, score_w, layout.group_ids, layout.present_masks, selected
        )
    else:
        matrix = layout.matrix[:, finite_scores]
        observed = np.isfinite(matrix)
        group_ids, present_masks = _missingness_groups(observed)
        p_forward, p_inverse = _ranksum_tails(
            matrix, score_w[finite_scores], group_ids, present_masks, selected
        )
    resistance = direction == "resistance"
    p_values = []
    for rows, tail in (
        (np.flatnonzero(forward), p_inverse if resistance else p_forward),
        (np.flatnonzero(inverse), p_forward if resistance else p_inverse),
    ):
        for row in rows:
            p_value = tail[row]
            if np.isfinite(p_value):
                p_values.append(float(p_value))
    if not p_values:
        return [], []
    _reject, q_values, _, _ = multipletests(p_values, method="fdr_bh")
    kept_p = []
    kept_q = []
    for p_value, q_value in zip(p_values, q_values):
        if p_value < 1:
            kept_p.append(p_value)
            kept_q.append(float(q_value))
    return kept_p, kept_q


def debug_wilcoxon_feature_ranking(
    fdf,
    tdf,
    score_column=None,
    top_n=10,
    filter_score_column=None,
    wilcox_score_column=None,
    min_effect_size=-0.2,
    direction="sensitivity",
):
    """Wilcoxon + BH q-values for mean-filtered features (same as ``return_possible_features``), top ``top_n`` by p-value.

    Does not apply Wilcoxon p/q significance cutoffs—only ranks for inspection.
    """
    if filter_score_column is None:
        filter_score_column = score_column
    if wilcox_score_column is None:
        wilcox_score_column = score_column
    wilcox_pvals = directional_wilcoxon_table(
        fdf, tdf, filter_score_column, wilcox_score_column,
        min_effect_size=min_effect_size, direction=direction,
    )
    if wilcox_pvals.empty:
        return pd.DataFrame(columns=["feature", "p_value", "q_value"])
    return wilcox_pvals.sort_values("p_value", ascending=True).head(int(top_n))


def debug_print_top_wilcoxon_features(
    fdf,
    tdf,
    score_column=None,
    top_n=10,
    file=None,
    filter_score_column=None,
    wilcox_score_column=None,
    min_effect_size=-0.2,
    direction="sensitivity",
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
        min_effect_size=min_effect_size,
        direction=direction,
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
    min_effect_size=-0.2,
    direction="sensitivity",
):
    """Filter features by mean then run Wilcoxon/FDR on both group directions.

    ``wilcox_score_column`` is the raw score used for the effect-size filter and the Wilcoxon test.
    ``min_effect_size`` is the required mean difference (sensitive group minus the other group).
    ``score_column`` remains as a backward-compatible fallback when the two explicit
    columns are not supplied. Inverse hits are named ``{feature}_inverse``.
    """
    if filter_score_column is None:
        filter_score_column = score_column
    if wilcox_score_column is None:
        wilcox_score_column = score_column
    wilcox_pvals = directional_wilcoxon_table(
        fdf,
        tdf,
        filter_score_column,
        wilcox_score_column,
        min_effect_size=min_effect_size,
        direction=direction,
    )
    if wilcox_pvals.empty:
        return pd.DataFrame(columns=["feature", "p_value", "q_value"])
    if qval_cutoff < 1:
        final_df = wilcox_pvals[
            (wilcox_pvals["q_value"] < qval_cutoff) & (wilcox_pvals["p_value"] < wilcox_cutoff)
        ]
    else:
        final_df = wilcox_pvals[wilcox_pvals["p_value"] < wilcox_cutoff]
    return final_df.sort_values(by="p_value")


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
