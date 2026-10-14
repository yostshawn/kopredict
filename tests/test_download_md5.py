"""Blank catalog checksums are filled from known_md5_hashes.csv."""

import pandas as pd

from download import fill_blank_md5


def test_fill_blank_md5_for_demeter2():
    catalog = pd.DataFrame(
        [
            {
                "release": "DEMETER2 Data v6",
                "filename": "D2_combined_gene_dep_scores.csv",
                "url": "https://example.com/d2.csv",
                "md5_hash": "",
            }
        ]
    )
    filled = fill_blank_md5(catalog)
    assert filled.loc[0, "md5_hash"] == "e82566a48993753c20ea5f480f1867ec"


def test_database_imports_on_python39_style_annotations():
    import database  # noqa: F401
