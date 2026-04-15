"""
TableCleaner_improvements.py
----------------------------
Targeted additions to your existing TableCleaner.py pipeline.
These are self-contained methods/functions you add to your
ImprovedTableCleaner class and call from its run() method.

HOW TO INTEGRATE
----------------
1. Copy the STATE_TOTAL_MAP and OCR_KNOWN_FIXES dicts to the top of
   TableCleaner.py (after the imports).

2. Add the four new methods to the ImprovedTableCleaner class.

3. In ImprovedTableCleaner.run(), add the new calls in this order
   (after rename_columns, before fill_station_locations):

       df = self.fix_known_ocr_errors(df)          # NEW
       df = self.normalise_aggregate_names(df)      # NEW
       df = self.fix_table_footer_rows(df)          # NEW
       # ...your existing steps...
       df = self.attach_json_context(df, table_num) # NEW - at end

4. For Table 4 (monthly format), call the standalone function:
       df = pivot_monthly_table(df)
   at the end of run() before saving, but only when the table has a
   Row_Type column.
"""

import re
import json
import os
import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWN OCR ERROR CORRECTIONS
#    Fixes specific high-frequency OCR errors that fuzzy matching can't catch
#    because the strings are too short or too corrupted.
#    Key = normalised OCR text (lower, stripped)
#    Value = canonical station name to use
# ─────────────────────────────────────────────────────────────────────────────
OCR_KNOWN_FIXES = {
    # "8 T MO" / "8T MO" = St. Louis, MO — 'S' → '8', space eaten
    "8 t mo":          "St. Louis",
    "8t mo":           "St. Louis",
    "st mo":           "St. Louis",
    # "IDF" = Idaho Falls — 3-char fragment too short for fuzzy match
    "idf":             "Idaho Falls",
    # "PH PA" = Philadelphia PA
    "ph pa":           "Philadelphia",
    # "DEL" alone = Wilmington, Delaware (state abbr fragment)
    "del":             "Wilmington",
    # "1NO" = Indiana (OCR 'I' → '1', rest dropped)
    "1no":             "Indianapolis",
    # "MUM" = Minimum row header leaking into station column
    "mum":             None,   # None = mark as skip/noise
    "mi nemum":        None,
    "maximum ber of":  None,
    # "GRAND TOTAL" in individual station tables = table footer, not Grand Rapids
    "grand total":     "__FOOTER__",
    # Perrys Memorial Ohio = Perry's Victory and International Peace Memorial
    "perrys memorial ohio": "Perry's Victory",
    # CNTY CONN (Table 3) - incomplete OCR, likely Litchfield County
    "cnty conn":       "Litchfield County",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE TOTAL NAME MAP
#    Maps the OCR'd fragments of state aggregate rows to canonical names.
#    These appear in Table 1 pages 9-10 (and similar summary pages in other
#    tables). Format: "STATE TOTAL" rows are legitimate data — they should
#    be preserved with a clean name, not matched to individual stations.
#
#    Pattern: the OCR breaks the state name with spaces and/or char swaps
#    so "CONNECTICUT TOTAL" becomes "CONNECTICUT T OTAL" etc.
# ─────────────────────────────────────────────────────────────────────────────
STATE_TOTAL_MAP = {
    # New England
    "connecticut": "Connecticut Total",
    "maine":       "Maine Total",
    "massachusetts": "Massachusetts Total",
    "massachusets": "Massachusetts Total",
    "new hampshire": "New Hampshire Total",
    "rhode island":  "Rhode Island Total",
    "vermont":       "Vermont Total",
    # Mid Atlantic
    "delaware":      "Delaware Total",
    "new jersey":    "New Jersey Total",
    "new york":      "New York Total",
    "pennsylvania":  "Pennsylvania Total",
    "pennsylva":     "Pennsylvania Total",
    # Mid East
    "district of columbia": "D.C. Total",
    "dis columbia":  "D.C. Total",
    "kentucky":      "Kentucky Total",
    "maryland":      "Maryland Total",
    "north carolina": "North Carolina Total",
    "n carolina":    "North Carolina Total",
    "virginia":      "Virginia Total",
    "west virginia": "West Virginia Total",
    "west verg":     "West Virginia Total",
    # South East
    "alabama":       "Alabama Total",
    "florida":       "Florida Total",
    "georgia":       "Georgia Total",
    "mississippi":   "Mississippi Total",
    "south carolina": "South Carolina Total",
    "s carolina":    "South Carolina Total",
    "tennessee":     "Tennessee Total",
    # Mid West
    "illinois":      "Illinois Total",
    "indiana":       "Indiana Total",
    "michigan":      "Michigan Total",
    "ohio":          "Ohio Total",
    "wisconsin":     "Wisconsin Total",
    # Great Plains
    "iowa":          "Iowa Total",
    "kansas":        "Kansas Total",
    "minnesota":     "Minnesota Total",
    "missouri":      "Missouri Total",
    "nebraska":      "Nebraska Total",
    "north dakota":  "North Dakota Total",
    "south dakota":  "South Dakota Total",
    # Gulf South
    "arkansas":      "Arkansas Total",
    "louisiana":     "Louisiana Total",
    "new mexico":    "New Mexico Total",
    "oklahoma":      "Oklahoma Total",
    "texas":         "Texas Total",
    # Rocky Mountain
    "colorado":      "Colorado Total",
    "idaho":         "Idaho Total",
    "montana":       "Montana Total",
    "utah":          "Utah Total",
    "wyoming":       "Wyoming Total",
    # Pacific Coast
    "alaska":        "Alaska Total",
    "arizona":       "Arizona Total",
    "california":    "California Total",
    "hawaii":        "Hawaii Total",
    "nevada":        "Nevada Total",
    "oregon":        "Oregon Total",
    "washington":    "Washington Total",
    # Regional totals
    "new england":   "New England Total",
    "mid atlantic":  "Mid Atlantic Total",
    "mid east":      "Mid East Total",
    "south east":    "South East Total",
    "mid west":      "Mid West Total",
    "great plains":  "Great Plains Total",
    "gulf south":    "Gulf South Total",
    "rocky mountain": "Rocky Mountain Total",
    "pacific coast": "Pacific Coast Total",
    # National
    "grand total":   "Grand Total",
    "national total": "National Total",
    "urban":         "Urban Total",
    "nonurban":      "Nonurban Total",
    "suburban":      "Suburban Total",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: extract canonical state/region from a noisy OCR aggregate string
# ─────────────────────────────────────────────────────────────────────────────
def _match_state_total(raw: str) -> Optional[str]:
    """
    Given a raw OCR string like "CONNECTICUT T OTAL" or "MASSACHUSETS TOT AL",
    return the canonical "Connecticut Total" or None if no match.

    Uses two strategies:
    1. Clean the string (strip "total", punctuation) then substring match keys
    2. Direct OCR-fragment lookup for severely corrupted state names where the
       first characters were dropped or mangled beyond substring matching
    """
    if not raw:
        return None

    # ── Direct lookup for severely corrupted OCR forms ──────────────────────
    # When OCR drops the first letter(s) or mangles the name beyond recognition
    DIRECT_LOOKUP = {
        "hode isl":    "Rhode Island Total",
        "hode":        "Rhode Island Total",
        "mio atlntc":  "Mid Atlantic Total",
        "mio atl":     "Mid Atlantic Total",
        "est verg":    "West Virginia Total",
        "ll inois":    "Illinois Total",
        "c hi gan":    "Michigan Total",
        "hi gan":      "Michigan Total",
        "tex as":      "Texas Total",
        "lori da":     "Florida Total",
        "grand":       "Grand Total",
        "cal":         "California Total",
        "entucky":     "Kentucky Total",
        "aryland":     "Maryland Total",
        "elaware":     "Delaware Total",
        "ermont":      "Vermont Total",
        "ew jersey":   "New Jersey Total",
        "ew york":     "New York Total",
        "nd 1a nan":   "Indiana Total",
        "nd ian":      "Indiana Total",
        "nd a nan":    "Indiana Total",
        "ew jerse":    "New Jersey Total",        "ennsylva":    "Pennsylvania Total",
        "laba ma":     "Alabama Total",
        "eorg la":     "Georgia Total",
        "enne ssee":   "Tennessee Total",
        "ow a":        "Iowa Total",
        "an sa s":     "Kansas Total",
        "in ne sota":  "Minnesota Total",
        "a so uri":    "Missouri Total",
        "neb ra ska":  "Nebraska Total",
        "ark an sas":  "Arkansas Total",
        "lou lana":    "Louisiana Total",
        "okl ahoma":   "Oklahoma Total",
        "col or ado":  "Colorado Total",
        "mon ta nan":  "Montana Total",
        "uta h":       "Utah Total",
        "ala ska":     "Alaska Total",
        "ari zona":    "Arizona Total",
        "haw all":     "Hawaii Total",
        "is colum":    "D.C. Total",
        "irgt wia":    "Virginia Total",
        "p t ota":     "Puerto Rico Total",
        "ca rolin":    "South Carolina Total",
        "ccarolt":     "North Carolina Total",
        "verto rt":    "Vermont Total",
        "1c hi":       "Michigan Total",
        "nd 1a nan":   "Indiana Total",
        "1 ll":        "Illinois Total",
        "0 hd 0":      "Ohio Total",
        "is consin":   "Wisconsin Total",
        "mio east":    "Mid East Total",
        "mio west":    "Mid West Total",
        "grt plains":  "Great Plains Total",
        "rocky mn tn": "Rocky Mountain Total",
        "pac coast":   "Pacific Coast Total",
        "new englno":  "New England Total",
        "gulf south":  "Gulf South Total",
        "south east":  "South East Total",
    }

    # Normalise: lowercase, collapse spaces, remove digits/punctuation
    s = raw.lower()
    s = re.sub(r'\bt\s*0\s*tal\b', '', s)   # "T 0 TAL" variant
    s = re.sub(r'\bt\s*otal\b', '', s)       # "T OTAL" variant
    s = re.sub(r'\bota\b', '', s)
    s = re.sub(r'\botal\b', '', s)
    s = re.sub(r'\btotal\b', '', s)
    s = re.sub(r'[^a-z\s]', ' ', s)         # remove digits, punctuation
    s = re.sub(r'\s+', ' ', s).strip()

    # Try direct key match on cleaned string
    if s in STATE_TOTAL_MAP:
        return STATE_TOTAL_MAP[s]

    # Try direct lookup for corrupted fragments
    for fragment, canonical in DIRECT_LOOKUP.items():
        if fragment in s:
            return canonical

    # Substring match against STATE_TOTAL_MAP keys (longest wins)
    best_key   = None
    best_score = 0
    for key in STATE_TOTAL_MAP:
        if key in s and len(key) > best_score:
            best_key   = key
            best_score = len(key)

    if best_key:
        return STATE_TOTAL_MAP[best_key]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW METHOD 1: fix_known_ocr_errors
# Add to ImprovedTableCleaner class
# ─────────────────────────────────────────────────────────────────────────────
def fix_known_ocr_errors(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the OCR_KNOWN_FIXES dictionary to Station_Location.

    Catches high-frequency specific errors that fuzzy matching misses because
    the strings are too short, too corrupted, or are common English words.

    Adds column: OCR_Fixed (True/False) to track which rows were corrected.
    """
    self.log("\nApplying known OCR error corrections...")

    if 'Station_Location' not in df.columns:
        return df

    fixed_count = 0
    skip_count  = 0

    for idx in df.index:
        raw = str(df.at[idx, 'Station_Location']).strip()
        key = raw.lower().strip()

        if key in OCR_KNOWN_FIXES:
            correction = OCR_KNOWN_FIXES[key]
            if correction is None:
                # Mark as noise/skip
                df.at[idx, 'Station_Location'] = ''
                skip_count += 1
            elif correction == '__FOOTER__':
                # Table footer row — blank out so it drops later
                df.at[idx, 'Station_Location'] = ''
                skip_count += 1
            else:
                df.at[idx, 'Station_Location'] = correction
                fixed_count += 1

    self.log(f"  Fixed {fixed_count} known OCR errors, blanked {skip_count} noise rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW METHOD 2: normalise_aggregate_names
# Add to ImprovedTableCleaner class
# ─────────────────────────────────────────────────────────────────────────────
def normalise_aggregate_names(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise OCR'd state/regional total rows to clean canonical names.

    These rows ("CONNECTICUT T OTAL", "MIO ATLNTC TOTAL", etc.) appear on
    the summary pages of each table. They are legitimate data but should NOT
    be fed to the individual station matcher — they represent state or regional
    aggregates.

    Adds column: Aggregate_Type ('state_total', 'region_total', 'national_total',
    or blank for normal station rows).
    """
    self.log("\nNormalising state/regional aggregate row names...")

    if 'Station_Location' not in df.columns:
        return df

    if 'Aggregate_Type' not in df.columns:
        df['Aggregate_Type'] = ''

    normalised = 0

    for idx in df.index:
        raw = str(df.at[idx, 'Station_Location']).strip()
        if not raw:
            continue

        # Only process rows that look like aggregate rows
        # (contain "total", "otal", "tota", or are very long state-name fragments)
        raw_lower = raw.lower()
        is_total_like = (
            'total' in raw_lower or
            ' t otal' in raw_lower or
            'tota' in raw_lower or
            'ota l' in raw_lower
        )
        if not is_total_like:
            continue

        canonical = _match_state_total(raw)
        if canonical:
            df.at[idx, 'Station_Location'] = canonical

            # Classify aggregate type
            if 'Total' in canonical:
                name_part = canonical.replace(' Total', '').strip()
                if name_part in ['New England','Mid Atlantic','Mid East','South East',
                                  'Mid West','Great Plains','Gulf South','Rocky Mountain',
                                  'Pacific Coast']:
                    df.at[idx, 'Aggregate_Type'] = 'region_total'
                elif name_part in ['Grand','National','Urban','Nonurban','Suburban']:
                    df.at[idx, 'Aggregate_Type'] = 'national_total'
                else:
                    df.at[idx, 'Aggregate_Type'] = 'state_total'
            normalised += 1

    self.log(f"  Normalised {normalised} aggregate rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW METHOD 3: fix_table_footer_rows
# Add to ImprovedTableCleaner class
# ─────────────────────────────────────────────────────────────────────────────
def fix_table_footer_rows(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove or flag rows that are table footers/headers rather than data.

    These appear when the reconstructor captures the printed text at the
    bottom of multi-page tables. Examples:
      "GRAND TOTAL" appearing in a station-level table (not a total page)
      Rows with no numeric data AND a name that is a known non-station string
    """
    self.log("\nFlagging table footer rows...")

    if 'Station_Location' not in df.columns:
        return df

    FOOTER_PATTERNS = [
        r'^grand\s+total$',
        r'^national\s+total$',
        r'^urban\s+stations\s+total$',
        r'^see\s+note',
        r'^continued',
        r'^\*+$',           # rows of asterisks
        r'^-+$',            # rows of dashes
    ]

    footer_count = 0
    for idx in df.index:
        raw = str(df.at[idx, 'Station_Location']).strip().lower()
        for pat in FOOTER_PATTERNS:
            if re.match(pat, raw):
                # Check: if this row has real numeric data, keep it (it might
                # be a legitimate national total row, not a spurious footer)
                num_cols = ['Min','Max','Avg','Num_Samples']
                has_data = any(
                    pd.notna(df.at[idx, c]) and df.at[idx, c] != ''
                    for c in num_cols if c in df.columns
                )
                if not has_data:
                    df.at[idx, 'Station_Location'] = ''
                    footer_count += 1
                break

    self.log(f"  Blanked {footer_count} footer rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW METHOD 4: attach_json_context
# Add to ImprovedTableCleaner class
# ─────────────────────────────────────────────────────────────────────────────
def attach_json_context(self, df: pd.DataFrame, table_num: int) -> pd.DataFrame:
    """
    Attach table_name, pollutant_units, unit_multiplier, and pollutant_group
    columns from historical_context.json.

    These columns travel with the data so any downstream analysis knows
    the pollutant and units without looking anything up.

    Requires historical_context.json and context_utils.py to be in the
    same directory as TableCleaner.py.
    """
    self.log("\nAttaching JSON context columns...")

    try:
        from context_utils import attach_table_context
        df = attach_table_context(df, table_num=table_num, inplace=True)
        self.log(f"  Added: table_name, pollutant_units, unit_multiplier, pollutant_group")
    except ImportError:
        self.log("  Skipped: context_utils.py not found (add it to use this feature)")
    except Exception as e:
        self.log(f"  Skipped: {e}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE FUNCTION: pivot_monthly_table
# Call this for Table 4 (and any other monthly-format tables)
# Converts 5-row vertical blocks → 1 wide row per station-year
# ─────────────────────────────────────────────────────────────────────────────
def pivot_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot Table 4's vertical block structure into a wide analytical format.

    INPUT structure (5 rows per station-year block):
      Row_Type='Name'        → station identifier row (months = NaN)
      Row_Type='Num_Samples' → Jan...Dec sample counts
      Row_Type='Maximum'     → Jan...Dec maximum values
      Row_Type='Average'     → Jan...Dec average values
      Row_Type='Minimum'     → Jan...Dec minimum values

    OUTPUT structure (1 row per station-year):
      Station_Location | Resolved_Name | Station_Order | Years |
      Jan_N | Feb_N | ... | Dec_N |    (Num_Samples per month)
      Jan_Max | Feb_Max | ... | Dec_Max |
      Jan_Avg | Feb_Avg | ... | Dec_Avg |
      Jan_Min | Feb_Min | ... | Dec_Min |
      Annual_Avg | Annual_Max | Annual_Min | Total_Samples |
      Avg_Confidence | Match_Type | Match_Score | table_name | pollutant_units

    Also calculates:
      Annual_Avg   = mean of monthly averages (ignoring NaN)
      Annual_Max   = max of monthly maximums
      Annual_Min   = min of monthly minimums
      Total_Samples = sum of monthly sample counts

    Only processes rows where Row_Type is set (i.e. Table 4 format).
    Returns the original df unchanged if Row_Type column is absent.
    """
    if 'Row_Type' not in df.columns:
        return df

    MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_present = [m for m in MONTHS if m in df.columns]

    if not months_present:
        return df

    records = []
    # Group by consecutive Name rows — each Name row starts a new block
    block_start_indices = df.index[df['Row_Type'] == 'Name'].tolist()

    for i, start_idx in enumerate(block_start_indices):
        # Determine the end of this block
        if i + 1 < len(block_start_indices):
            end_idx = block_start_indices[i + 1]
            block = df.loc[start_idx:end_idx - 1]
        else:
            block = df.loc[start_idx:]

        # Extract the Name row for station metadata
        name_row = block[block['Row_Type'] == 'Name']
        if name_row.empty:
            continue
        name_row = name_row.iloc[0]

        rec = {
            'Station_Location': name_row.get('Station_Location', ''),
            'Resolved_Name':    name_row.get('Resolved_Name', ''),
            'Station_Order':    name_row.get('Station_Order', np.nan),
            'Years':            name_row.get('Years', ''),
            'Avg_Confidence':   name_row.get('Avg_Confidence', np.nan),
            'Match_Score':      name_row.get('Match_Score', np.nan),
            'Match_Type':       name_row.get('Match_Type', ''),
            'Page_Num':         name_row.get('Page_Num', np.nan),
        }

        # Add any context columns that are present
        for ctx_col in ['table_name','pollutant_units','unit_multiplier','pollutant_group']:
            if ctx_col in block.columns:
                rec[ctx_col] = name_row.get(ctx_col, '')

        # Extract each Row_Type into month columns
        for row_type, suffix in [
            ('Num_Samples', '_N'),
            ('Maximum',     '_Max'),
            ('Average',     '_Avg'),
            ('Minimum',     '_Min'),
        ]:
            type_row = block[block['Row_Type'] == row_type]
            if not type_row.empty:
                type_row = type_row.iloc[0]
                for month in months_present:
                    rec[f'{month}{suffix}'] = type_row.get(month, np.nan)

        # Calculate annual summary values
        avg_vals = [rec.get(f'{m}_Avg', np.nan) for m in months_present]
        max_vals = [rec.get(f'{m}_Max', np.nan) for m in months_present]
        min_vals = [rec.get(f'{m}_Min', np.nan) for m in months_present]
        n_vals   = [rec.get(f'{m}_N',   np.nan) for m in months_present]

        avg_clean = [v for v in avg_vals if pd.notna(v)]
        max_clean = [v for v in max_vals if pd.notna(v)]
        min_clean = [v for v in min_vals if pd.notna(v)]
        n_clean   = [v for v in n_vals   if pd.notna(v)]

        rec['Annual_Avg']     = round(np.mean(avg_clean),  1) if avg_clean else np.nan
        rec['Annual_Max']     = max(max_clean)                if max_clean else np.nan
        rec['Annual_Min']     = min(min_clean)                if min_clean else np.nan
        rec['Total_Samples']  = sum(n_clean)                  if n_clean   else np.nan

        records.append(rec)

    if not records:
        return df

    result = pd.DataFrame(records)

    # Reorder columns: identifiers → annual summaries → monthly detail
    id_cols      = ['Station_Location','Resolved_Name','Station_Order','Years']
    ctx_cols     = ['table_name','pollutant_units','unit_multiplier','pollutant_group']
    summary_cols = ['Total_Samples','Annual_Max','Annual_Avg','Annual_Min']
    month_detail = (
        [f'{m}_N'   for m in months_present] +
        [f'{m}_Max' for m in months_present] +
        [f'{m}_Avg' for m in months_present] +
        [f'{m}_Min' for m in months_present]
    )
    meta_cols = ['Avg_Confidence','Match_Score','Match_Type','Page_Num']

    final_col_order = (
        [c for c in id_cols      if c in result.columns] +
        [c for c in ctx_cols     if c in result.columns] +
        [c for c in summary_cols if c in result.columns] +
        [c for c in month_detail if c in result.columns] +
        [c for c in meta_cols    if c in result.columns]
    )
    extra = [c for c in result.columns if c not in final_col_order]
    result = result[final_col_order + extra]

    print(f"[pivot_monthly_table] Pivoted {len(df)} rows → {len(result)} station-year records")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO CALL pivot_monthly_table FROM YOUR TableCleaner.run()
# ─────────────────────────────────────────────────────────────────────────────
"""
Add this block near the END of ImprovedTableCleaner.run(), just before saving:

    # Pivot monthly-format tables (Table 4 and similar)
    if 'Row_Type' in df.columns:
        self.log("\\nPivoting monthly table format to wide rows...")
        from TableCleaner_improvements import pivot_monthly_table
        df = pivot_monthly_table(df)
        self.log(f"  Result: {len(df)} station-year rows")

"""


# ─────────────────────────────────────────────────────────────────────────────
# WHERE TO ADD THE NEW METHOD CALLS IN ImprovedTableCleaner.run()
# ─────────────────────────────────────────────────────────────────────────────
"""
In your existing run() method, after rename_columns() and remove_column_zero(),
add these calls in this exact order:

    df = self.fix_known_ocr_errors(df)          # ← NEW: fixes "8 T MO", "IDF", etc.
    df = self.normalise_aggregate_names(df)      # ← NEW: fixes "CONNECTICUT T OTAL"
    df = self.fix_table_footer_rows(df)          # ← NEW: removes spurious footer rows
    df = self.clean_station_names(df)            # existing
    df = self.extract_site_numbers(df)           # existing
    df = self.clean_years(df)                    # existing
    df = self.fill_station_locations(df)         # existing
    df = self.identify_aggregate_rows(df)        # existing
    df = self.fill_missing_years_smart(df)       # existing
    df = self.clean_numeric_columns(df)          # existing
    df = self.remove_header_noise(df)            # existing
    df = self.validate_min_avg_max(df)           # existing
    df = self.validate_percentile_ordering(df)   # existing
    df = self.finalize_types(df)                 # existing
    df = self.attach_json_context(df, table_num) # ← NEW: adds name/units/group cols
    # Pivot monthly tables last
    if 'Row_Type' in df.columns:
        from TableCleaner_improvements import pivot_monthly_table
        df = pivot_monthly_table(df)

The table_num parameter needs to be passed into run():
    def run(self, table_num: int = None):
And called as:
    cleaner.run(table_num=1)   # for Table 1
    cleaner.run(table_num=4)   # for Table 4
"""
