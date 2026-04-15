# Historical Air Pollution Data Digitization Pipeline
### OCR Recovery of 1950s Urban Air Quality Records

---

## Project Overview

This project digitizes a 228-page scanned historical document containing U.S. air pollution measurements from the early 1950s (years 1953–1957). The source is a physical reference book that was later scanned to PDF. The book contains 101 structured statistical tables recording particulate air quality data across approximately 259 monitoring stations spanning urban, suburban, and nonurban sites across the United States.

The core challenge is converting those printed tables — which exist only as scanned images — into clean, machine-readable CSV datasets that can be used for statistical analysis and historical research.

The pipeline is fully automated and runs end-to-end from raw page images to cleaned, station-resolved output tables.

---

## The Source Data

### What the Document Contains

The source PDF (`UCData.pdf`) is a historical air quality report. It contains 101 numbered tables organized into two broad structural types:

**Type A — Annual Summary Tables (16-column format)**
These appear 50 times in the book. Each row represents one monitoring station and one year, and the columns are:

| Column | Meaning |
|--------|---------|
| Station Location | City/station name |
| Site | Monitoring site number (multi-site cities only) |
| Years | Year(s) of measurement (e.g. `53`, `53-57`) |
| Num Samples | Number of measurements taken |
| Min | Minimum reading |
| Max | Maximum reading |
| Avg | Annual average |
| P10–P90 | Percentile distribution (10th through 90th) |

**Type B — Monthly Breakdown Tables (60-column format)**
These appear 51 times. They expand each station's data across all 12 months, recording N (sample count), Max, Avg, and Min separately per month — producing 48 data columns plus annual summaries.

### The Monitoring Station Network

The master station reference (`master_stations.csv`) contains **259 stations** organized as follows:

- **Urban stations** — major cities grouped by region and state
- **Suburban stations** — surrounding areas of major cities
- **Nonurban stations** — background/rural monitoring sites

Several stations operated multiple monitoring sites within the same city (e.g., Chicago had up to 5 sites). Stations are annotated with availability flags: some appear only in the first 4 tables (`**`), others in the first 8 (`*`), and the remainder appear across all 31 reporting tables.

---

## Pipeline Architecture

The pipeline has four sequential stages. Each stage produces output consumed by the next.

```
PDF → PNG Images → OCR Extraction → Table Reconstruction → Cleaning & Resolution → Final CSVs
       (manual)     TableOCR.py      TableReconstruct.py    TableCleaner.py
                                                             StationResolver.py
```

---

### Stage 1 — Image Preparation (Manual Pre-processing)

The PDF was converted to individual page images at **400 DPI**, saved as PNG files in an `./Image/` directory. High DPI was chosen deliberately to improve Tesseract's character recognition on small printed numerals. Pages were named sequentially (`001.png`, `002.png`, etc.) to allow automatic page number inference.

---

### Stage 2 — OCR Extraction (`TableOCR.py`)

**Class:** `EnhancedOCRExtractor`

This is the core scanning engine. For each page image it performs the following steps:

**2a. Document Alignment**
Attempts perspective correction by detecting the document's four corners using Canny edge detection and contour analysis. If a clean four-sided contour covering at least 30% of the image area is found, a perspective transform is applied to flatten any page tilt. If no such contour is found, the full image is used as-is.

**2b. Image Preprocessing**
The image is enhanced before OCR runs:
- Bilateral filtering removes noise while preserving text edges
- CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts local contrast in dim areas
- A morphological gradient sharpens character outlines
- Adaptive Gaussian thresholding binarizes the image (converts to pure black and white)
- A light dilation pass reinforces thin strokes

**2c. Table Structure Detection**
Column separators (vertical lines) and row separators (horizontal lines) are detected using morphological operations:
- Vertical lines must span at least **30% of page height** to count as column separators
- Horizontal lines must span at least **20% of page width** to count as row separators
- Nearby duplicate lines are merged using 15-pixel (vertical) and 10-pixel (horizontal) thresholds

The detected line positions are saved to text files (`page_NNN_structure.txt`) for use in reconstruction. These files record the exact pixel coordinates of every column and row separator found.

**2d. Dual-Pass OCR**
OCR runs twice per page using Tesseract:
- **Pass 1 (Alphanumeric):** `--psm 6` with character whitelist `0-9A-Za-z.,- /`. Targets station names and general text.
- **Pass 2 (Numeric Recovery):** `--psm 6` with numeric-only whitelist `0-9.,- `. Targets cells that Pass 1 misread as non-numeric. Recovered an additional **46,213 items** (31.8% of total output).

Each extracted text element is tagged with: page number, table number, pixel position (left, top, width, height), column index, row index, OCR confidence score (0–100), and which pass produced it.

**2e. Special Case — Side-by-Side Tables**
Pages 223–228 contain two tables printed side by side. These are handled as a single full-page extraction (rather than attempting to split them) to avoid misalignment artifacts.

**Output:** `combined_ocr_results.csv` — 145,195 extracted text elements across 228 pages.

---

### Stage 3 — Table Reconstruction (`TableReconstruct.py`)

**Class:** `TableReconstructor`

The raw OCR output is a flat list of text fragments with pixel coordinates. This stage converts it back into structured table rows.

**Key logic:**
- Loads each page's structure file to know where column boundaries fall
- Assigns each text element to a column based on which vertical-line interval its horizontal center falls into
- Groups text elements into rows by clustering their vertical (top) coordinates — elements within **25 pixels** vertically are considered part of the same row
- Rows within the same logical table (same column count across consecutive pages) are merged into a single output table
- A confidence threshold of **60** is applied to all columns except Column 1 (station names), which passes through regardless of confidence
- Rows are annotated with average OCR confidence and their source page number

Pages are grouped by column count so that multi-page tables (e.g. Table 1 spanning pages 1–8) stay together as a single output file.

**Output:** One CSV per detected table group, named by table number and page range (e.g., `Table_1_Pages_1-8.csv`). The combined `All_Tables_Combined.csv` contains all 11,997 raw reconstructed rows.

---

### Stage 4 — Cleaning & Station Resolution (`TableCleaner.py` + `StationResolver.py`)

This is the most complex stage. It handles domain-specific knowledge about the data structure, corrects OCR errors in known fields, and links every row to a canonical station identity.

#### TableCleaner

The cleaner identifies each table's type by its column count and applies type-specific rules:

**For 16-column annual tables:**
- Validates and standardizes the `Years` field against known valid values: `53`, `54`, `55`, `56`, `57`, `53-57`, and multi-year combinations
- Validates and casts the `Site` column (expects a single digit 1–9 or blank)
- Casts all numeric columns (`Num_Samples`, `Min`, `Max`, `Avg`, `P10`–`P90`) to float, coercing non-numeric OCR garbage to NaN
- Flags aggregate rows (rows that summarize multiple years or sites) using `Is_Year_Aggregate` and `Is_Site_Aggregate` boolean columns
- Detects ordering violations in percentile columns (P10 must be ≤ P20 ≤ … ≤ P90) and clears implausible sequences

**For 60-column monthly tables:**
- Validates all 12 × 4 monthly cells (N, Max, Avg, Min per month)
- Checks that within each month, Max ≥ Avg ≥ Min, and clears cells that violate this ordering
- Tracks which months are naturally empty (no measurements) vs. OCR misses
- Computes an adjusted fill rate that excludes naturally empty cells from the denominator

**General cleaning (all types):**
- Removes blank rows and header-repetition rows (rows where the station name cell contains column header text)
- Forward-fills station names down multi-row station blocks (OCR sometimes only captures the name once for a multi-year block)
- Applies a "smart fill" — uses median values from high-confidence neighboring cells to impute plausible values for low-confidence cells

#### StationResolver

After cleaning, every row's station name string is resolved to a canonical entry in the master station list.

The resolver uses a three-tier matching strategy:

| Match type | Threshold | Action |
|------------|-----------|--------|
| Exact | 100.0 | Accept immediately, no flag |
| High fuzzy | ≥ 85.0 | Auto-replace with canonical name |
| Low fuzzy | 60.0–84.9 | Replace but flag with `match_type = 'fuzzy_low'` |
| No match | < 60.0 | Leave original text, flag as `'unmatched'` |

Matching uses Python's `difflib.SequenceMatcher` after normalizing both strings (lowercasing, removing punctuation, collapsing whitespace). When a `Site` column is present, site number is used as an additional constraint to narrow candidates among multi-site cities.

Every resolved row receives four new columns: `Resolved_Name`, `Station_Order`, `Match_Score`, and `Match_Type`.

**Output:** 101 cleaned CSVs in `Cleaned_Tables/`, plus per-table cleaning reports and the `comprehensive_cleaning_summary.txt`.

---

## Supporting Files

| File | Purpose |
|------|---------|
| `master_stations.csv` | Ground-truth list of 259 stations with canonical names, city, state, region, urban type, and ordering |
| `codebook.json` | Column definitions and value encoding for all table types |
| `historical_context.json` | Metadata about the source document and measurement methodology |
| `context_utils.py` | Shared helper functions for reading context files across modules |
| `summary.py` | Generates the comprehensive cleaning summary report |
| `page_NNN_structure.txt` | Per-page structure files (10 samples present) recording detected column and row line positions |

---

## Performance Results

### OCR Extraction (Stage 2)

| Metric | Value |
|--------|-------|
| Pages processed | 228 |
| Total text elements extracted | 145,195 |
| Pass 1 items | 98,982 (68.2%) |
| Pass 2 recovered items | 46,213 (31.8%) |
| Average OCR confidence | 90.5 / 100 |
| High confidence (≥ 80) | 131,385 items — **90.5%** |
| Medium confidence (60–79) | 6,970 items — 4.8% |
| Low confidence (40–59) | 3,633 items — 2.5% |
| Very low confidence (< 40) | 3,207 items — 2.2% |
| Total processing time | 20.4 minutes |
| Average time per page | 5.4 seconds |

The dual-pass approach was critical: nearly one-third of all extracted values came from the numeric-only second pass, which recovered digits that the first pass had misclassified as letters.

### Table Reconstruction (Stage 3)

| Metric | Value |
|--------|-------|
| Raw reconstructed rows | 11,997 |
| Average columns per page detected | 14.8 |
| Tables spanning multiple pages | correctly grouped |

### Cleaning & Resolution (Stage 4)

| Metric | Value |
|--------|-------|
| Tables processed | 101 of 102 (98.0%) |
| Total data rows after cleaning | 23,674 |
| Rows removed as junk/blank | 5 |
| **Overall fill rate** | **78.3%** |
| **Average OCR confidence** | **92.4 / 100** |

**By table type:**

| Type | Tables | Rows | Fill Rate | Avg Confidence |
|------|--------|------|-----------|----------------|
| 16-column annual | 50 | 3,845 | **90.3%** | 92.8 |
| 60-column monthly | 51 | ~19,829 | ~72% | ~91.9 |

**Top-performing individual tables** (fill rate ≥ 95%):

| Table | Pages | Fill Rate | Confidence |
|-------|-------|-----------|------------|
| Table_1 | 1–8 | 96.7% | 92.9 |
| Table_9 | 31–36 | 96.0% | 92.9 |
| Table_26 | 81–83 | 95.6% | 93.5 |
| Table_58 | 146–148 | 95.6% | 93.4 |
| Table_5 | 23–25 | 95.8% | 93.1 |
| Table_30 | 91–93 | 96.3% | 93.5 |

**Challenging tables** (fill rate ≤ 60%):

| Table | Pages | Fill Rate | Likely cause |
|-------|-------|-----------|-------------|
| Table_40 | 111 | 51.2% | Very short table (8 rows), few data points |
| Table_65 | 158 | 53.3% | Monthly table with sparse station coverage |
| Table_25 | 80 | 60.1% | Monthly table format, partial page |

The 16-column annual tables perform significantly better than the 60-column monthly tables. This is expected: monthly tables have 4× more numeric cells per station row, giving OCR more opportunities to miss or misalign individual values.

---

## Output File Structure

After the full pipeline, the output is organized as:

```
Cleaned_Tables/
├── Table_1_Pages_1-8_cleaned.csv         # Annual summary, 548 rows, 96.7% fill
├── Table_2_Pages_9-10_cleaned.csv
├── ...
├── Table_101_Pages_219-222_cleaned.csv
├── comprehensive_cleaning_summary.txt
├── comprehensive_cleaning_summary.csv
└── All_Tables_Combined_cleaning_report.txt

Processed/
├── combined_ocr_results.csv              # 145,195 raw OCR elements
├── All_Tables_Combined.csv              # 11,997 reconstructed rows (pre-cleaning)
├── comprehensive_report.txt             # OCR performance stats
├── Structures/
│   ├── page_001_structure.txt
│   └── ...
└── Visualizations/
    └── page_NNN.png                     # Debug visualizations
```

### Columns in Cleaned 16-Column CSVs

`Station_Location`, `Site`, `Years`, `Num_Samples`, `Min`, `Max`, `Avg`, `P10`, `P20`, `P30`, `P40`, `P50`, `P60`, `P70`, `P80`, `P90`, `Avg_Confidence`, `Page_Num`, `Aggregate_Type`, `Resolved_Name`, `Station_Order`, `Match_Score`, `Match_Type`, `Is_Year_Aggregate`, `Is_Site_Aggregate`

### Columns in Cleaned 60-Column CSVs

`Station_Location`, `Resolved_Name`, `Station_Order`, `Years`, `Total_Samples`, `Annual_Max`, `Annual_Avg`, `Annual_Min`, then for each month Jan–Dec: `{Mon}_N`, `{Mon}_Max`, `{Mon}_Avg`, `{Mon}_Min` (48 columns), plus `Avg_Confidence`, `Match_Score`, `Match_Type`, `Page_Num`

---

## Limitations and Known Issues

**Row detection undercount.** The OCR report shows only 143 unique row zones across 228 pages (averaging 3.2 per page), which is far fewer than the actual data rows per page. This reflects a limitation in how horizontal line detection was used: the row zone index groups text by detected printed rules, not by individual data row. Each "zone" contains multiple data rows, and the reconstruction step re-clusters by pixel proximity. This does not affect the final output but means the OCR-level row count is not a meaningful data-row count.

**Monthly table fill gap.** The 60-column monthly tables average roughly 72% fill compared to 90% for annual tables. Some of this gap is genuine absence of data (not all stations recorded measurements every month), and the cleaner attempts to exclude these from the denominator. However, some fraction represents real OCR misses that could theoretically be recovered.

**51 tables classified as "Unknown" type.** The cleaner could not confirm the schema of 51 tables from column count alone. These were still cleaned using best-effort logic, but their fill rates may be slightly understated due to schema ambiguity.

**1 table unprocessed.** One of the 102 detected tables (Table 15) did not produce a cleaned output file during the run captured in this report.

**Side-by-side pages.** Pages 223–228 contain two tables printed side by side. The pipeline handles these with a single-pass extraction and flags them, but column assignment accuracy on these pages is lower than single-table pages (average confidence 86.8 vs. 90.5+).

---

## Dependencies

```
Python 3.8+
opencv-python
numpy
pandas
pytesseract
Tesseract OCR (system install, version 4+)
difflib (standard library)
```

---

## How to Run

```bash
# Stage 2: OCR extraction (requires images in ./Image/)
python TableOCR.py

# Stage 3: Table reconstruction
python TableReconstruct.py

# Stage 4: Cleaning and station resolution
python TableCleaner.py   # processes all tables in batch

# Generate summary report
python summary.py
```

---

*Pipeline developed for the digitization of historical U.S. air pollution monitoring records, 1953–1957.*

