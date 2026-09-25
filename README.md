# kopredict

kopredict finds genomic features linked to CRISPR or RNAi sensitivity. The search uses a Wilcoxon test, a hypergeometric overlap step, and a feature tree. `kopredict-download` and `kopredict-prepare` rebuild the local inputs from public files so a clone can run without a private database.

## Install

```bash
pip install -e .
```

Python 3.9 or newer. The same commands work inside the Docker image.

## Data

Processed files go in `data/` (override with `KOPREDICT_DATA_DIR`). Raw downloads go in `data/raw/`. Cell-line ids are DepMap `ModelID` values such as `ACH-000001`, stored in a column named `cellline`.

### 1. DepMap catalog

Signed URLs expire and the URL list sits behind the DepMap portal check. In a browser, open https://depmap.org/portal/api/download/files and save the CSV (it must include a `url` column). Then:

```bash
kopredict-download --catalog catalog.csv
```

That fetches the open filename catalog, downloads only the pinned files, and checks `md5_hash`.

Pinned files:

- DepMap Public 26Q1: `Model.csv`, `Gene.csv`, `OmicsProfiles.csv`, `CRISPRGeneEffect.csv`, `OmicsCNGeneWGS.csv`, `OmicsSomaticMutations.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, `OmicsSomaticMutationsMatrixDamaging.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, `OmicsFusionFiltered.csv`, `OmicsGlobalSignatures.csv`, `OmicsCNGeneMC_WES.csv`
- DEMETER2 Data v6: `D2_combined_gene_dep_scores.csv`
- Sanger GDSC1 and GDSC2: `sanger-dose-response.csv`, `sanger-viability.csv`
- Harmonized Public Proteomics 26Q1: `harmonized_Sanger_MS_2022.csv`, `harmonized_MS_CCLE_Gygi.csv`, `harmonized_RPPA_CCLE.csv`

It also tries to download the HRD workbook to `data/raw/crc-24-0316_table_s5_suppst5.xlsx`. If that signed link has expired, save Supplementary Table S5 from [10.1158/2767-9764.CRC-24-0316](https://doi.org/10.1158/2767-9764.CRC-24-0316) at that path. The score used is `hrdsum_summary` when it is present, otherwise `hrdsum_broad`. A score of 42 or higher is the high-HRD feature.

### 2. Mutational signatures

Save the Petljak et al. supplementary workbook ([Cell 2019](https://www.cell.com/cell/fulltext/S0092-8674(19)30161-8)) as `data/raw/petljak2019_mmc1.xlsx`. PMC blocks a scripted download. An exposure greater than 0 becomes a `{signature}_True` feature. Sample names are matched to DepMap models by stripped cell-line name.

GDSC dose-response and viability are downloaded and left in `data/raw/`. Drug-response search is not implemented yet.

### 3. Build the matrices

```bash
kopredict-prepare
```

This writes:

| File | Role |
| --- | --- |
| `Model_and_OmicsSignatures.parquet.gz` | Model metadata, global signatures, HRD, and SBS exposures |
| `OmicsFull_processed.parquet.gz` | Long per-gene table used by `-g` background filters |
| `CRISPRGeneEffect_processed.parquet.gz` | CRISPR gene effect |
| `D2_combined_gene_dep_scores_processed.parquet.gz` | DEMETER2 RNAi scores |
| `final_depmap_features_26Q1.parquet.gz` | Binary features (rows) by cell line (columns) |

Non-cancer models (`OncotreePrimaryDisease` of `Non-Cancerous`) are omitted from the feature matrix. Copy-number amplification is log2 ratio at least 1, and homozygous deletion is log2 ratio at most -1. Expression, Sanger MS, Gygi MS, and RPPA use a per-gene z-score cutoff of 1.64. Likely loss-of-function uses the variant classes Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, and Nonstop_Mutation.

These old DataHub feature families are not rebuilt, because the pinned files do not contain them: `has_hessdriver`, `has_oncogenehighimpact`, `has_tumorsuppressorhighimpact`.

## Run a search

```bash
kopredict -t IPPK -d crispr -c SOLID -O oncotreeprimarydisease
```

`-d` is `crispr` or `rnai`. `-p AND|OR` chooses how feature blocks narrow the cohort (default `AND`). `-O` is the model column matched by `-c` (default `oncotreeprimarydisease`). `SOLID` and `PANCANCER` are special cohort names. `-f` defaults to `data/final_depmap_features_26Q1.parquet.gz`.

By default the search drops feature names containing `absolute`, `sensitive`, `resistant`, `lowexp`, `highexp`, `depmapprot`, `sangerprot`, or `rppa`. `--keep-lowexp-highexp` and `--keep-depmapprot-sangerprot` keep those families (`rppa` is kept with the protein flag).

Plots are written under `{output-dir}/{target}/{dataset}/`.

## Docker

The image does not contain DepMap files. Mount a `data/` directory:

```bash
docker build -t kopredict .
docker run --rm --entrypoint kopredict-download -v "$PWD/data:/data" -v "$PWD/catalog.csv:/catalog.csv:ro" kopredict --catalog /catalog.csv
docker run --rm --entrypoint kopredict-prepare -v "$PWD/data:/data" kopredict
docker run --rm -v "$PWD/data:/data" -v "$PWD/results:/results" kopredict -t IPPK -d crispr -c SOLID -O oncotreeprimarydisease --output-dir /results
```

`KOPREDICT_DATA_DIR` inside the image is `/data`.
