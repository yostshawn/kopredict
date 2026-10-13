"""Background rules read OmicsFull columns only."""

import pandas as pd
import pytest

from background import Background


def _omics():
    return pd.DataFrame(
        {
            "cellline": ["A", "B", "C", "A"],
            "gene": ["KRAS", "KRAS", "KRAS", "TP53"],
            "muthotspot": [2.0, 0.0, float("nan"), 1.0],
            "mutdamaging": [0.0, 0.0, 0.0, 3.0],
            "tpmlogp1_zscore": [3.0, 0.2, 1.0, 0.0],
        }
    )


def test_hotspot_rule_keeps_lines_with_a_count():
    background = Background("KRAS,omics,muthotspot,greater,0,false", ["A", "B", "C"])
    background.create_background_df()
    background.update_celllines_given_options(_omics())
    background.update_background_celllines(_omics())
    assert background.background_celllines == ["A"]


def test_rules_combine_with_and():
    text = "KRAS,omics,tpmlogp1_zscore,greater,2.5,false;TP53,omics,mutdamaging,greater,0,false"
    background = Background(text, ["A", "B"])
    background.create_background_df()
    background.update_celllines_given_options(_omics())
    background.update_background_celllines(_omics())
    assert background.background_celllines == ["A"]


def test_unknown_column_is_rejected():
    background = Background("KRAS,omics,protein,greater,1,false", ["A"])
    with pytest.raises(SystemExit, match="OmicsFull_processed"):
        background.create_background_df()


def test_only_omics_is_accepted():
    background = Background("KRAS,proteomics,tpmlogp1_zscore,greater,1,false", ["A"])
    with pytest.raises(SystemExit, match="must be omics"):
        background.create_background_df()
