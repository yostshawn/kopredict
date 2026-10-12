"""Inverse feature names must cover every combined piece, and the swapped row."""

import numpy as np
import pandas as pd
from scipy import stats

from features import (
    base_feature_name,
    directional_wilcoxon_table,
    expanded_feature_labels,
    inverse_feature_name,
    materialize_inverse_features,
    rows_for_block,
    rows_for_features,
)


def test_inverse_name_marks_every_semicolon_and_colon_piece():
    merged = "TP53_mutdamaging;TP53_likelylof:TP53_ko"
    assert inverse_feature_name(merged) == (
        "TP53_mutdamaging_inverse;TP53_likelylof_inverse:TP53_ko_inverse"
    )
    assert base_feature_name(inverse_feature_name(merged)) == merged
    assert inverse_feature_name("TP53_ko_inverse;TP53_altered") == (
        "TP53_ko_inverse;TP53_altered_inverse"
    )


def test_merged_inverse_row_is_swapped_not_mixed_with_the_forward_call():
    columns = ["c1", "c2", "c3", "c4"]
    merged_name = "TP53_mutdamaging;TP53_likelylof"
    merged = pd.DataFrame([[1, 1, 0, 0]], index=[merged_name], columns=columns)
    inverse_name = inverse_feature_name(merged_name)
    stored = materialize_inverse_features(merged, [inverse_name])
    assert list(stored.loc[inverse_name]) == [0, 0, 1, 1]
    assert np.isnan(
        materialize_inverse_features(
            pd.DataFrame([[1, np.nan, 0]], index=[merged_name], columns=["a", "b", "c"]),
            [inverse_name],
        ).loc[inverse_name, "b"]
    )

    separate = pd.DataFrame(
        [[1, 1, 0, 0], [1, 1, 0, 0]],
        index=["TP53_mutdamaging", "TP53_likelylof"],
        columns=columns,
    )
    pure = rows_for_block(separate, inverse_name).fillna(0).any()
    old_bug_name = "TP53_mutdamaging;TP53_likelylof_inverse"
    mixed = rows_for_features(separate, old_bug_name.split(";")).fillna(0).any()
    assert list(pure) == [False, False, True, True]
    assert list(mixed) == [True, True, True, True]


def test_inverse_wilcoxon_matches_ranksums_on_the_zero_group():
    columns = [f"c{i}" for i in range(40)]
    scores = np.array([1.5] * 20 + [-1.5] * 20)
    row = [1] * 20 + [0] * 15 + [np.nan] * 5
    fdf = pd.DataFrame([row], index=["TP53_mutdamaging;TP53_likelylof"], columns=columns)
    tdf = pd.DataFrame(
        {"cellline": columns, "crispr": scores, "crispr_zscore": stats.zscore(scores)}
    )
    table = directional_wilcoxon_table(fdf, tdf, "crispr_zscore", "crispr")
    assert list(table["feature"]) == [
        "TP53_mutdamaging_inverse;TP53_likelylof_inverse"
    ]
    raw = stats.ranksums(scores[20:35], scores[:20], alternative="less").pvalue
    stored = float(table["p_value"].iloc[0])
    assert abs(stored - raw) / raw < 0.01
    labels = expanded_feature_labels(table["feature"])
    assert "TP53_mutdamaging" in labels
    assert "TP53_likelylof" in labels
