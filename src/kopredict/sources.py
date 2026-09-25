"""Pinned public inputs and literature citations for kopredict ingest.

Signed download URLs are not stored for DepMap catalog files. Those URLs expire
and are read from a catalog CSV the user saves in a browser. The HRD workbook
URL below is a CloudFront link and will also expire; the DOI stays the stable
citation.
"""

HRD_DOI = "https://doi.org/10.1158/2767-9764.CRC-24-0316"
PETLJAK_URL = "https://www.cell.com/cell/fulltext/S0092-8674(19)30161-8"

NO_CAPTCHA_CATALOG_URL = "https://depmap.org/portal/api/no-captcha/download/files"

# CloudFront signature expires (Expires=1793322017). Re-download Table S5 from HRD_DOI if this fails.
HRD_XLSX_URL = (
    "https://aacr.silverchair-cdn.com/aacr/content_public/journal/cancerrescommun/4/12/"
    "10.1158_2767-9764.crc-24-0316/3/crc-24-0316_table_s5_suppst5.xlsx"
    "?Expires=1793322017"
    "&Signature=VVi7au32va6IG5eXuchODZtHeCRzyYJaODCHqt-fhLWUQvXyfv36QpbpYDN62tcns87zoYcnJZADJt9cuCGXqEhyor~gWaAy7k69mx88cNFGzMjMB5-OIAdjuqwhhHmo2yVNbX1~2zBrA7ezlnWRDwGumEsdlPNsr5~Ltrv7tv8HZ1f8cJy~JVUk86E25Nxt1cC2R4agexYzNuK3i2w8psnA7OSFWkRxGzEb7yhPNMD-3mkpEVux1OcuG5y~MgDkP4d0LXX7kCHz4GApNJCCpZo8QX2PD98VhclBPpJil6eZmO9c~omu03exB-Zr1cwSI-Vt22MxVSRVJ-CJwuGEew__"
    "&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA"
)

HRD_RAW_NAME = "crc-24-0316_table_s5_suppst5.xlsx"
PETLJAK_RAW_NAME = "petljak2019_mmc1.xlsx"
FEATURES_FILENAME = "final_depmap_features_26Q1.parquet.gz"

HRD_SCORE_CUTOFF = 42
ZSCORE_CUTOFF = 1.64
LOF_CLASSES = (
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
)

# (release, filename) pairs verified against the DepMap no-captcha catalog.
PINNED_FILES = (
    ("DepMap Public 26Q1", "Model.csv"),
    ("DepMap Public 26Q1", "Gene.csv"),
    ("DepMap Public 26Q1", "OmicsProfiles.csv"),
    ("DepMap Public 26Q1", "CRISPRGeneEffect.csv"),
    ("DepMap Public 26Q1", "OmicsCNGeneWGS.csv"),
    ("DepMap Public 26Q1", "OmicsSomaticMutations.csv"),
    ("DepMap Public 26Q1", "OmicsSomaticMutationsMatrixHotspot.csv"),
    ("DepMap Public 26Q1", "OmicsSomaticMutationsMatrixDamaging.csv"),
    ("DepMap Public 26Q1", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"),
    ("DepMap Public 26Q1", "OmicsFusionFiltered.csv"),
    ("DepMap Public 26Q1", "OmicsGlobalSignatures.csv"),
    ("DepMap Public 26Q1", "OmicsCNGeneMC_WES.csv"),
    ("DEMETER2 Data v6", "D2_combined_gene_dep_scores.csv"),
    ("Sanger GDSC1 and GDSC2", "sanger-dose-response.csv"),
    ("Sanger GDSC1 and GDSC2", "sanger-viability.csv"),
    ("Harmonized Public Proteomics 26Q1", "harmonized_Sanger_MS_2022.csv"),
    ("Harmonized Public Proteomics 26Q1", "harmonized_MS_CCLE_Gygi.csv"),
    ("Harmonized Public Proteomics 26Q1", "harmonized_RPPA_CCLE.csv"),
)
