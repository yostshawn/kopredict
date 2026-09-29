"""Wilcoxon tails, the mean filter, and which side of a feature is kept."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from features import (
    directional_wilcoxon_table,
    mean_filter_directions,
    ranksums_less_pvalues,
    return_possible_features,
    wilcoxon_test,
)
from utils import sig_figs


def _target(celllines, scores):
    return pd.DataFrame(
        {
            "cellline": celllines,
            "crispr": scores,
            "crispr_zscore": stats.zscore(scores),
        }
    )


def test_wilcoxon_test_matches_ranksums_and_drops_missing_cells():
    celllines = ["a", "b", "c", "d", "e"]
    group = pd.Series([1, 1, 0, 0, np.nan], index=celllines)
    tdf = _target(celllines, [-2.0, -1.0, 1.0, 2.0, -100.0])
    expected = sig_figs(stats.ranksums([-2.0, -1.0], [1.0, 2.0], alternative="less").pvalue, 3)
    assert wilcoxon_test(group, tdf, "crispr") == expected
    assert np.isnan(wilcoxon_test(pd.Series([1, 1], index=["a", "b"]), tdf, "crispr"))


def test_ranksums_less_pvalues_matches_the_row_test_with_ties_and_empty_groups():
    scores = pd.Series([0.0, 0.0, 1.0, 2.0], index=list("abcd"))
    features = pd.DataFrame(
        [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 1, np.nan],
        ],
        index=["tied", "zeros_essential", "all_ones"],
        columns=list("abcd"),
    )
    both = ranksums_less_pvalues(features, scores)
    tied = features.loc["tied"]
    tdf = pd.DataFrame({"cellline": list("abcd"), "crispr": scores.to_numpy()})
    assert both.loc["tied", "p_forward"] == pytest.approx(wilcoxon_test(tied, tdf, "crispr"))
    swapped = pd.Series([1, 1, 0, 0], index=list("abcd"))
    assert both.loc["zeros_essential", "p_inverse"] == pytest.approx(
        wilcoxon_test(swapped, tdf, "crispr")
    )
    assert np.isnan(both.loc["all_ones", "p_forward"])
    assert np.isnan(both.loc["all_ones", "p_inverse"])


def test_effect_size_is_the_mean_gap_not_the_sum_and_flips_for_inverse():
    wide = [f"c{i}" for i in range(40)]
    features = pd.DataFrame([[1] * 20 + [0] * 20], index=["wide_small"], columns=wide)
    # Mean gap is -0.05. The sum gap is -1, which would pass a -0.2 sum cutoff.
    tiny = _target(wide, [-0.05] * 20 + [0.0] * 20)
    forward, inverse = mean_filter_directions("crispr", features, tiny, min_effect_size=-0.2)
    assert forward == []
    assert inverse == []

    groups = ["c1", "c2", "c3", "c4"]
    exact = pd.DataFrame([[1, 1, 0, 0]], index=["at_threshold"], columns=groups)
    forward, _ = mean_filter_directions(
        "crispr", exact, _target(groups, [-0.2, -0.2, 0.0, 0.0]), min_effect_size=-0.2
    )
    assert forward == ["at_threshold"]

    flipped = pd.DataFrame([[0, 0, 1, 1]], index=["wt"], columns=groups)
    forward, inverse = mean_filter_directions(
        "crispr", flipped, _target(groups, [-0.4, -0.4, 0.0, 0.0]), min_effect_size=-0.2
    )
    assert forward == []
    assert inverse == ["wt"]
    celllines = ["c1", "c2", "c3", "c4"]
    features = pd.DataFrame(
        [[1, 1, 0, 0], [1, 1, 0, 0]],
        index=["small_gap", "large_gap"],
        columns=celllines,
    )
    small = _target(celllines, [-0.1, -0.1, 0.0, 0.0])
    large = _target(celllines, [-0.3, -0.3, 0.0, 0.0])
    small_forward, _ = mean_filter_directions("crispr", features.loc[["small_gap"]], small, min_effect_size=-0.2)
    large_forward, _ = mean_filter_directions("crispr", features.loc[["large_gap"]], large, min_effect_size=-0.2)
    assert small_forward == []
    assert large_forward == ["large_gap"]


def test_mean_filter_ignores_missing_cells_and_keeps_either_negative_group():
    celllines = ["c1", "c2", "c3"]
    features = pd.DataFrame(
        [
            [1, 1, 0],
            [0, 0, 1],
            [np.nan, np.nan, 0],
        ],
        index=["ones_essential", "zeros_essential", "missing_not_a_zero"],
        columns=celllines,
    )
    tdf = pd.DataFrame(
        {"cellline": celllines, "crispr_zscore": [-5.0, -5.0, 5.0]}
    )
    forward, inverse = mean_filter_directions("crispr_zscore", features, tdf)
    assert forward == ["ones_essential"]
    assert inverse == ["zeros_essential"]


def test_return_possible_features_keeps_only_the_passing_direction():
    celllines = [f"c{i}" for i in range(12)]
    scores = np.array([-3.0] * 6 + [3.0] * 6)
    features = pd.DataFrame(
        [
            [1] * 6 + [0] * 6,
            [0] * 6 + [1] * 6,
            [1, 0] * 6,
        ],
        index=["ko", "wt", "noise"],
        columns=celllines,
    )
    tdf = _target(celllines, scores)
    table = directional_wilcoxon_table(features, tdf, "crispr_zscore", "crispr")
    names = set(table["feature"])
    assert "ko" in names
    assert "wt_inverse" in names
    assert "ko_inverse" not in names
    kept = return_possible_features(
        features,
        tdf,
        filter_score_column="crispr_zscore",
        wilcox_score_column="crispr",
        wilcox_cutoff=0.05,
        qval_cutoff=1,
    )
    assert set(kept["feature"]) >= {"ko", "wt_inverse"}
    assert "ko_inverse" not in set(kept["feature"])
    assert kept["q_value"].notna().all()
