"""Overlap counts, the hypergeometric grouping, and subset feature filtering."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from features import (
    expanded_feature_labels,
    feature_jaccard,
    generate_overlap_matrix,
    group_features_by_jaccard,
    group_features_by_pvalue_v2,
    restrict_features_to_celllines,
    run_hypergeom_tests,
)
from pipeline import _limit_next_tree_subsets


def test_overlap_matrix_counts_shared_ones_and_skips_missing_cells():
    features = pd.DataFrame(
        [
            [1, 1, 0, np.nan],
            [1, 0, 1, 1],
        ],
        index=["A", "B"],
        columns=list("abcd"),
    )
    overlap = generate_overlap_matrix(features, ["A"], ["B"])
    assert overlap.loc["A", "B"] == 1
    with pytest.raises(KeyError):
        generate_overlap_matrix(features, ["missing"], ["B"])


def test_hypergeometric_p_value_matches_scipy():
    features = pd.DataFrame(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        index=["A", "B"],
        columns=list("abcd"),
    )
    overlap = generate_overlap_matrix(features, ["A", "B"], ["A", "B"])
    result = run_hypergeom_tests(features, overlap)
    pair = result[(result["feature1"] == "A") & (result["feature2"] == "B")].iloc[0]
    # 4 shared lines, 2 ones in A, 2 ones in B, 1 shared one.
    expected = hypergeom.sf(1 - 1, 4, 2, 2)
    assert pair["pvalue"] == pytest.approx(expected)


def test_group_features_by_pvalue_v2_orders_by_p_and_keeps_directions():
    pairs = pd.DataFrame(
        [
            {"feature1": "CDKN2A_ko", "feature2": "TP53_altered_inverse", "pvalue": 1e-20},
            {"feature1": "CDKN2A_ko", "feature2": "MTAP_homdel", "pvalue": 0.5},
        ]
    )
    pvals = pd.DataFrame(
        {
            "feature": ["MTAP_homdel", "CDKN2A_ko", "TP53_altered_inverse"],
            "p_value": [1e-8, 1e-4, 1e-3],
        }
    )
    groups = group_features_by_pvalue_v2(
        pairs,
        pvals["feature"].tolist(),
        pvalue_threshold=1e-15,
        feature_wilcox_pvals=pvals,
    )
    assert groups[0] == ["MTAP_homdel"]
    assert set(groups[1]) == {"CDKN2A_ko", "TP53_altered_inverse"}


def test_group_features_by_pvalue_v2_rejects_a_weak_union():
    features = pd.DataFrame(
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        index=["A", "B"],
        columns=list("abcd"),
    )
    tdf = pd.DataFrame({"cellline": list("abcd"), "crispr": [1.0, 2.0, 3.0, 4.0]})
    pairs = pd.DataFrame([{"feature1": "A", "feature2": "B", "pvalue": 1e-20}])
    pvals = pd.DataFrame({"feature": ["A", "B"], "p_value": [1e-6, 1e-5]})
    groups = group_features_by_pvalue_v2(
        pairs,
        ["A", "B"],
        pvalue_threshold=1e-15,
        fdf=features,
        tdf=tdf,
        score_column="crispr",
        feature_wilcox_pvals=pvals,
        wilcox_factor=10,
    )
    assert groups == [["A"], ["B"]]


def test_restrict_features_to_celllines_drops_constants_and_merges_exact_rows():
    columns = ["c1", "c2", "c3", "c4"]
    features = pd.DataFrame(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 1],
            [np.nan, np.nan, np.nan, 1],
            [1, np.nan, 1, 0],
            [0, np.nan, 0, 1],
            [1, 0, np.nan, 1],
            [1, 0, np.nan, 1],
        ],
        index=["all1", "all0", "all_missing", "ones_and_missing", "zeros_and_missing", "vary_a", "vary_b"],
        columns=columns,
    )
    kept = restrict_features_to_celllines(features, ["c1", "c2", "c3"])
    assert "vary_a;vary_b" in kept.index
    assert set(kept.index) == {"vary_a;vary_b"}
    row = kept.loc["vary_a;vary_b"].to_numpy()
    assert row[0] == 1
    assert row[1] == 0
    assert np.isnan(row[2])


def test_expanded_feature_labels_include_the_base_feature():
    labels = expanded_feature_labels(["TP53_ko_inverse;CDKN2A_ko:MTAP_homdel_inverse"])
    for name in ("TP53_ko", "CDKN2A_ko", "MTAP_homdel"):
        assert name in labels


def test_limit_next_tree_subsets_keeps_the_best_groups():
    groups = [["b_only"], ["a_and_c", "c_weak"], ["d_mid"]]
    pvals = pd.DataFrame(
        {
            "feature": ["a_and_c", "b_only", "c_weak", "d_mid"],
            "p_value": [0.01, 0.001, 0.2, 0.05],
        }
    )
    kept = _limit_next_tree_subsets(groups, pvals, max_subsets=2)
    assert kept == [["b_only"], ["a_and_c", "c_weak"]]


def test_jaccard_groups_near_identical_features():
    columns = list("abcdefghij")
    # A and B: intersection 9, union 10 -> Jaccard 0.9. C is disjoint.
    features = pd.DataFrame(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        index=["A", "B", "C"],
        columns=columns,
    )
    assert feature_jaccard(features, "A", "B") == pytest.approx(0.9)
    pvals = pd.DataFrame({"feature": ["A", "B", "C"], "p_value": [1e-6, 1e-5, 1e-4]})
    groups = group_features_by_jaccard(
        features, ["A", "B", "C"], feature_wilcox_pvals=pvals, jaccard_threshold=0.9
    )
    assert groups[0] == ["A", "B"]
    assert groups[1] == ["C"]
