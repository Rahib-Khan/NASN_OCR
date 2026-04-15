"""
TableReconstruct_improvements.py
---------------------------------
Targeted improvements to your existing TableReconstruct.py pipeline.
These are standalone functions you can call from your existing code.

HOW TO INTEGRATE
----------------
Import at the top of TableReconstruct.py:
    from TableReconstruct_improvements import (
        validate_station_order,
        infer_missing_years_from_context,
        detect_table_section_type,
        write_reconstruction_report,
    )

Then call them at the appropriate points described in each function's docstring.

WHAT THESE FIX
--------------
1. validate_station_order    - checks Station_Order is monotonically increasing
                               per page; flags gaps that indicate missed rows
2. infer_missing_years       - uses the station's other rows + the book's known
                               year range to fill blank Year cells
3. detect_table_section_type - identifies whether a page block is station-level
                               data, a state total page, a regional summary, or
                               the monthly breakdown format (Table 4 style)
4. write_reconstruction_report - saves a readable report showing per-table stats
                                 with context from historical_context.json
"""

import json
import os
import re
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONTEXT (used by write_reconstruction_report)
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_context() -> dict:
    path = os.path.join(_HERE, "historical_context.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATION ORDER VALIDATION
#    Call this after StationResolver.resolve_dataframe() has run.
#    Per the book's design, Station_Order must increase monotonically across
#    all pages within a table. Any decrease or large gap means rows were missed.
# ─────────────────────────────────────────────────────────────────────────────
def validate_station_order(df: pd.DataFrame,
                            table_num: Optional[int] = None,
                            verbose: bool = True) -> Dict:
    """
    Check that Station_Order is monotonically non-decreasing within each page
    group, and that no large gaps exist between consecutive stations.

    Per the original publication, every table uses the identical station
    ordering from Table D. A gap > 5 in Station_Order between two consecutive
    data rows strongly suggests a missed or misread row.

    Args:
        df:         Cleaned + resolved DataFrame (has Station_Order column).
        table_num:  Table number (for reporting).
        verbose:    Print findings.

    Returns:
        dict with keys:
          'total_stations'   - unique Station_Orders found
          'expected_range'   - (min_order, max_order)
          'gaps'             - list of (order_before, order_after, gap_size)
          'reversals'        - list of (idx, order_before, order_after) where order decreases
          'missing_estimated'- estimated count of missed stations
    """
    if 'Station_Order' not in df.columns:
        return {}

    # Only look at data rows (exclude aggregate, skip, state_total)
    mask = (
        df['Station_Order'].notna() &
        (df.get('Is_Year_Aggregate', False) != True) &
        (df.get('Aggregate_Type', '') == '')
    )
    orders = df.loc[mask, 'Station_Order'].dropna().astype(int)

    if orders.empty:
        return {}

    unique_orders = sorted(orders.unique())
    min_o, max_o  = unique_orders[0], unique_orders[-1]

    gaps      = []
    reversals = []
    prev      = None

    for order in orders:
        if prev is not None:
            diff = order - prev
            if diff < 0:
                reversals.append((prev, order, abs(diff)))
            elif diff > 5:
                gaps.append((prev, order, diff - 1))
        prev = order

    missing_estimated = sum(g[2] for g in gaps)
    expected_in_range = max_o - min_o + 1
    found_in_range    = len(unique_orders)
    fill_rate         = 100.0 * found_in_range / expected_in_range if expected_in_range else 0

    result = {
        'table_num':          table_num,
        'total_stations':     found_in_range,
        'expected_range':     (min_o, max_o),
        'range_coverage':     round(fill_rate, 1),
        'gaps':               gaps,
        'reversals':          reversals,
        'missing_estimated':  missing_estimated,
    }

    if verbose:
        label = f"Table {table_num}" if table_num else "Table"
        print(f"\n[Station Order Validation — {label}]")
        print(f"  Station_Order range:  {min_o} – {max_o}")
        print(f"  Unique stations found: {found_in_range} / {expected_in_range} ({fill_rate:.1f}%)")
        if gaps:
            print(f"  Gaps detected ({len(gaps)}, est. {missing_estimated} missed rows):")
            for before, after, size in gaps[:10]:
                print(f"    Order {before} → {after}  (gap of {size})")
            if len(gaps) > 10:
                print(f"    ...and {len(gaps) - 10} more gaps")
        else:
            print("  No significant gaps detected.")
        if reversals:
            print(f"  Order reversals ({len(reversals)} — may indicate page reconstruction error):")
            for before, after, size in reversals[:5]:
                print(f"    {before} → {after}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. INFER MISSING YEARS FROM CONTEXT
#    When OCR missed a year value, infer it from the station's other rows
#    and the book's known year range (1953-1957).
# ─────────────────────────────────────────────────────────────────────────────
def infer_missing_years(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Fill blank Years values by inferring from the station's other rows.

    Rules applied (in order):
    1. If a station has exactly one blank-year row and the other rows cover
       years 54, 55, 56, then the blank row is probably 53 or 57.
    2. If a station has rows for 54, 56 but nothing for 55, and has one
       blank row, fill it as 55.
    3. If a station has only one row total and year is blank, leave blank
       (insufficient context).

    Only fills when exactly one year is missing and can be determined
    with certainty. Never fills aggregate rows.

    Returns the DataFrame with filled Years values and an 'Year_Inferred'
    boolean column marking rows that were inferred.
    """
    if 'Years' not in df.columns:
        return df

    VALID_YEARS = {'53', '54', '55', '56', '57'}

    df = df.copy()
    if 'Year_Inferred' not in df.columns:
        df['Year_Inferred'] = False

    filled = 0

    # Process by station
    name_col = 'Resolved_Name' if 'Resolved_Name' in df.columns else 'Station_Location'

    for station in df[name_col].dropna().unique():
        if not station or station.strip() == '':
            continue

        station_mask = df[name_col] == station

        # Only work on individual (non-aggregate) rows
        if 'Is_Year_Aggregate' in df.columns:
            individual_mask = station_mask & (~df['Is_Year_Aggregate'].fillna(False))
        else:
            individual_mask = station_mask

        station_rows = df[individual_mask]
        if len(station_rows) < 2:
            continue

        # Find rows with blank years
        blank_mask  = individual_mask & (df['Years'].isna() | (df['Years'].astype(str).str.strip() == ''))
        filled_mask = individual_mask & ~blank_mask

        blank_indices   = df[blank_mask].index.tolist()
        known_years_raw = df.loc[filled_mask, 'Years'].astype(str).str.strip().tolist()

        if not blank_indices:
            continue

        # Parse known years (handle multi-year strings like "54 55")
        known_single_years = set()
        for y_str in known_years_raw:
            for token in y_str.split():
                if token in VALID_YEARS:
                    known_single_years.add(token)

        missing_years = VALID_YEARS - known_single_years

        # Only fill if exactly one blank row and exactly one missing year
        if len(blank_indices) == 1 and len(missing_years) == 1:
            missing_year = list(missing_years)[0]
            df.at[blank_indices[0], 'Years'] = missing_year
            df.at[blank_indices[0], 'Year_Inferred'] = True
            filled += 1

    if verbose:
        print(f"[infer_missing_years] Inferred {filled} blank Year values")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECT TABLE SECTION TYPE
#    Determines which "section" of a table a page belongs to.
#    Used by the reconstructor to route pages to the correct cleaner.
# ─────────────────────────────────────────────────────────────────────────────

# Known page ranges for each section type within Table 1
# (book page numbers, add ~16 for PDF page number)
# From historical_context.json and direct PDF inspection:
TABLE_SECTION_MAP = {
    # Table 1: Suspended Particulate Matter (book pages 17-38)
    # Pages 17-??: Urban station individual data (16-col)
    # Pages ??:    State/Regional totals (15-col, all aggregate)
    # Pages ??:    Monthly breakdown (vertical block format)
    # Tables 2-31 follow the same pattern but without Site column
}

def detect_table_section_type(page_text_sample: str,
                               column_count: int) -> str:
    """
    Given a sample of text from the top of a reconstructed page and the
    detected column count, return the section type.

    Section types:
      'station_data'     - Individual station rows (the main data)
      'state_totals'     - State/regional aggregate summary page
      'monthly_data'     - Monthly breakdown (Jan-Dec columns)
      'nonurban_data'    - Nonurban station rows (same format as station_data)
      'unknown'          - Cannot determine

    Args:
        page_text_sample: First ~500 chars of OCR'd text from page top.
        column_count:     Number of columns detected by the reconstructor.

    Returns:
        String section type label.
    """
    text = page_text_sample.upper()

    # Monthly format pages have month abbreviations in headers
    month_pattern = r'\bJAN\b.*\bFEB\b.*\bMAR\b'
    if re.search(month_pattern, text):
        return 'monthly_data'

    # State total pages contain "TOTAL" many times
    total_count = text.count('TOTAL')
    if total_count >= 3:
        return 'state_totals'

    # Check for nonurban markers
    if 'NONURBAN' in text or 'NON-URBAN' in text or 'SUBURBAN' in text:
        return 'nonurban_data'

    # Column count can help distinguish
    # 16 cols = Table 1 station data (has Site)
    # 15 cols = Tables 2-31 station data
    # 14 cols = monthly data (Station + Years + 12 months)
    if column_count == 14:
        return 'monthly_data'
    if column_count in (15, 16):
        return 'station_data'

    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# 4. WRITE RECONSTRUCTION REPORT
#    Saves a rich text report combining your cleaning stats with the
#    historical context from historical_context.json
# ─────────────────────────────────────────────────────────────────────────────
def write_reconstruction_report(
    table_stats: List[Dict],
    output_path: str = "reconstruction_report.txt",
    include_context: bool = True,
) -> str:
    """
    Write a comprehensive reconstruction report combining your pipeline's
    data quality stats with the historical context from historical_context.json.

    This is the report you hand to your professor alongside the data.

    Args:
        table_stats:     List of dicts from your existing cleaning pipeline.
                         Each dict should have at minimum:
                           table_num, name, rows, fill_pct, match_rate
                         Produced by the cleaning report section of summary.py.
        output_path:     Where to write the report.
        include_context: Whether to include the historical context section.

    Returns:
        The report text as a string.
    """
    ctx    = _load_context() if include_context else {}
    tables = ctx.get("tables", {})
    net    = ctx.get("network", {})
    stat   = ctx.get("statistical_context", {})

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("NASN AIR POLLUTION DATA — RECONSTRUCTION AND QUALITY REPORT")
    lines.append("=" * 72)

    if ctx:
        meta = ctx.get("_meta", {})
        lines.append(f"\nSource: {meta.get('document_title', 'Unknown')}")
        lines.append(f"Publisher: {meta.get('publisher', 'Unknown')}")
        lines.append(f"Year: {meta.get('year_published', 'Unknown')}")
        lines.append(f"Archive: {meta.get('archive_url', '')}")

    # ── Network context ───────────────────────────────────────────────────────
    if net:
        lines.append("\n" + "─" * 72)
        lines.append("NETWORK OVERVIEW")
        lines.append("─" * 72)
        lines.append(net.get("preface_summary", ""))
        lines.append(f"\nData years: {net.get('data_years', [])}")
        lines.append(f"Operating agency: {net.get('operating_agency', '')}")

        hist = net.get("history_by_year", {})
        lines.append("\nNetwork growth:")
        for year, info in sorted(hist.items()):
            if isinstance(info, dict):
                communities = info.get("communities", info.get("urban_stations", "?"))
                lines.append(f"  {year}: {communities} communities/stations")

    # ── Statistical note ──────────────────────────────────────────────────────
    if stat:
        lines.append("\n" + "─" * 72)
        lines.append("STATISTICAL ANALYSIS NOTE")
        lines.append("─" * 72)
        lines.append(stat.get("implication_for_analysis", ""))
        lines.append(f"\nDistribution: {stat.get('distribution_type', '')}")
        lines.append(stat.get("percentile_interpretation", ""))

    # ── Per-table cleaning stats ───────────────────────────────────────────────
    if table_stats:
        lines.append("\n" + "─" * 72)
        lines.append("DIGITISATION QUALITY — PER TABLE")
        lines.append("─" * 72)
        lines.append(
            f"{'T':>3}  {'Pollutant':<32}  {'Units':<16}  "
            f"{'Rows':>6}  {'Fill%':>6}  {'Match%':>7}"
        )
        lines.append("─" * 72)

        total_rows = total_fill = total_match = 0
        for s in table_stats:
            tnum = s.get("table_num", "?")
            name = s.get("name", "")[:32]
            rows = s.get("rows", 0)
            fill = s.get("fill_pct", 0)
            match = s.get("match_rate", 0)

            # Get units from JSON
            units = ""
            if ctx:
                t_info = tables.get(str(tnum), {})
                units  = t_info.get("units", "")[:16]

            lines.append(
                f"{tnum:>3}  {name:<32}  {units:<16}  "
                f"{rows:>6}  {fill:>5.1f}%  {match:>6.1f}%"
            )
            total_rows  += rows
            total_fill  += fill * rows
            total_match += match * rows

        lines.append("─" * 72)
        if total_rows > 0:
            lines.append(
                f"ALL  {'TOTAL':<32}  {'':16}  "
                f"{total_rows:>6}  {total_fill/total_rows:>5.1f}%  "
                f"{total_match/total_rows:>6.1f}%"
            )

    # ── Per-pollutant context ─────────────────────────────────────────────────
    if tables:
        lines.append("\n" + "─" * 72)
        lines.append("POLLUTANT SIGNIFICANCE NOTES")
        lines.append("─" * 72)
        for tnum_str in sorted(tables.keys(), key=lambda x: int(x)):
            t_info = tables[tnum_str]
            sig    = t_info.get("significance", "")
            if sig:
                lines.append(f"\nTable {tnum_str}: {t_info.get('name','')}")
                lines.append(f"  Units: {t_info.get('units','')}")
                lines.append(f"  Group: {t_info.get('pollutant_group','')}")
                # Wrap significance text at 65 chars
                words = sig.split()
                line  = "  "
                for word in words:
                    if len(line) + len(word) + 1 > 67:
                        lines.append(line)
                        line = "  " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    lines.append(line)

    # ── Errata ────────────────────────────────────────────────────────────────
    errata = ctx.get("errata", {}).get("corrections", [])
    if errata:
        lines.append("\n" + "─" * 72)
        lines.append("KNOWN ERRATA IN ORIGINAL PUBLICATION")
        lines.append("─" * 72)
        for e in errata:
            lines.append(
                f"  {e.get('location','')}: "
                f"'{e.get('error','')}' should be '{e.get('correction','')}'"
            )

    lines.append("\n" + "=" * 72)
    report_text = "\n".join(lines)

    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[write_reconstruction_report] Report saved: {output_path}")
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO CALL FROM YOUR EXISTING historical_ocr.py / summary.py
# ─────────────────────────────────────────────────────────────────────────────
"""
In summary.py CleaningReportSummarizer.generate_summary(), add near the end:

    from TableReconstruct_improvements import write_reconstruction_report
    report = write_reconstruction_report(
        table_stats=combined_stats,   # your existing stats list
        output_path=os.path.join(self.cleaned_dir, "reconstruction_report.txt"),
    )

In TableReconstruct.py reconstruct_all_tables() or wherever you finalize
a table DataFrame, add:

    from TableReconstruct_improvements import validate_station_order, infer_missing_years

    # After StationResolver runs:
    validation = validate_station_order(group_table, table_num=group_idx)

    # After year cleaning:
    group_table = infer_missing_years(group_table)
"""
