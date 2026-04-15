"""
StationResolver.py
------------------
Resolves OCR'd station name strings to canonical station names using the
master_stations.csv reference file derived from the document's Table D.

Key facts from the source document:
  - Every data table uses the SAME station row order as Table D.
  - Stations are grouped: Urban → Suburban → Nonurban, each by region then state.
  - Multi-site cities (e.g. Chicago sites 1-5) appear as consecutive rows.
  - The 'Site' column (Col_2 in 16-col tables) gives the site number.
  - Stations marked ** appear only in Tables 1-4.
  - Stations marked *  appear only in Tables 1-8.
  - Unmarked stations appear in all 31 tables.

How to use standalone:
    resolver = StationResolver("master_stations.csv")
    result = resolver.resolve("Chicagp", site_num=3)
    print(result)
    # {'canonical_name': 'Chicago', 'city': 'Chicago', 'state_abbr': 'IL',
    #  'station_order': 101, 'site_num': 3, 'match_score': 93.3,
    #  'match_type': 'fuzzy', 'validated': True}

How to use in TableCleaner:
    resolver = StationResolver("master_stations.csv")
    df = resolver.resolve_dataframe(df, name_col='Station_Location', site_col='Site')
    # Adds columns: Resolved_Name, Match_Score, Match_Type, Station_Order
"""

import csv
import difflib
import re
import os
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
EXACT_THRESHOLD   = 100.0   # Perfect after normalization
HIGH_THRESHOLD    =  85.0   # High-confidence fuzzy match → auto-replace
LOW_THRESHOLD     =  60.0   # Plausible match → replace but flag
# Below LOW_THRESHOLD → leave as-is, flag as 'unmatched'


class StationResolver:
    """
    Loads master_stations.csv and resolves OCR station name strings to
    canonical names with confidence scores.
    """

    def __init__(self, master_csv_path: str):
        """
        Args:
            master_csv_path: Path to master_stations.csv
        """
        if not os.path.exists(master_csv_path):
            raise FileNotFoundError(f"Master stations file not found: {master_csv_path}")

        self.master_csv_path = master_csv_path
        self.stations: List[Dict] = []          # all rows from CSV
        self._lookup_by_order: Dict[int, List[Dict]] = {}   # station_order → rows
        self._canonical_names: List[str] = []   # for difflib matching
        self._name_to_rows: Dict[str, List[Dict]] = {}      # normalized name → rows

        self._load_stations()
        self._build_indexes()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_stations(self):
        """Read master_stations.csv into self.stations."""
        with open(self.master_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields
                row['station_order'] = int(row['station_order'])
                row['max_table'] = int(row['max_table'])
                row['site_num'] = int(row['site_num']) if row['site_num'].strip() else None
                self.stations.append(row)

        print(f"[StationResolver] Loaded {len(self.stations)} station records "
              f"from {self.master_csv_path}")

    def _build_indexes(self):
        """Build lookup structures for fast matching."""
        for row in self.stations:
            order = row['station_order']
            if order not in self._lookup_by_order:
                self._lookup_by_order[order] = []
            self._lookup_by_order[order].append(row)

            # Index by normalized canonical name
            norm = self._normalize(row['canonical_name'])
            if norm not in self._name_to_rows:
                self._name_to_rows[norm] = []
            self._name_to_rows[norm].append(row)

            # Index by normalized city name (catches "Ft. Worth" vs "Fort Worth")
            norm_city = self._normalize(row['city'])
            if norm_city not in self._name_to_rows:
                self._name_to_rows[norm_city] = []
            if row not in self._name_to_rows[norm_city]:
                self._name_to_rows[norm_city].append(row)

            # Index by normalized display_name (e.g. "waterbury conn" so OCR
            # text like "Waterbury, Conn." matches directly)
            display = row.get('display_name', '')
            if display:
                norm_display = self._normalize(display)
                if norm_display not in self._name_to_rows:
                    self._name_to_rows[norm_display] = []
                if row not in self._name_to_rows[norm_display]:
                    self._name_to_rows[norm_display].append(row)

        # Unique list of all matchable name forms for difflib.
        # Include both canonical_name and display_name so fuzzy matching
        # works against whichever form the OCR text is closer to.
        seen = set()
        for row in self.stations:
            for name in (row['canonical_name'], row.get('display_name', '')):
                if name and name not in seen:
                    self._canonical_names.append(name)
                    seen.add(name)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Lowercase, strip punctuation/whitespace, collapse spaces.
        Handles common OCR artifacts.
        """
        if not text:
            return ''
        t = text.lower()
        # Common OCR substitutions: 0→o, 1→l, etc. are intentionally NOT done
        # here — we want normalization only, not correction.
        # Remove trailing/leading punctuation and normalize internal spaces.
        t = re.sub(r'[^\w\s]', ' ', t)   # punctuation → space
        t = re.sub(r'\s+', ' ', t)        # collapse spaces
        t = t.strip()
        return t

    @staticmethod
    def _ocr_clean(text: str) -> str:
        """
        Additional cleaning for raw OCR input before matching.
        Removes noise characters that OCR often inserts.
        """
        if not text:
            return ''
        # Strip leading/trailing noise
        t = text.strip()
        # Remove OCR artifacts: asterisks, bullets, pipes, leading digits alone
        t = re.sub(r'^[\*\|\.\-\d]+\s*', '', t)
        t = re.sub(r'\s*[\*\|]+$', '', t)
        # Normalize multiple spaces
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------

    def _score(self, raw: str, candidate: str) -> float:
        """
        Return a 0-100 similarity score between raw OCR text and a
        candidate canonical name, using SequenceMatcher.
        """
        norm_raw = self._normalize(raw)
        norm_cand = self._normalize(candidate)
        if not norm_raw or not norm_cand:
            return 0.0
        return difflib.SequenceMatcher(None, norm_raw, norm_cand).ratio() * 100.0

    def _find_best_match(
        self,
        raw: str,
        site_num: Optional[int] = None,
        state_hint: Optional[str] = None,
        table_num: Optional[int] = None,
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching station record for a raw OCR string.

        Args:
            raw:        The OCR'd station name string.
            site_num:   Site number from the Site column (if available).
            state_hint: Two-letter state abbreviation (if known from context).
            table_num:  The data table number (1-31) to filter by max_table.

        Returns:
            (best_row, score) — row is None if no match above LOW_THRESHOLD.
        """
        cleaned = self._ocr_clean(raw)
        if not cleaned:
            return None, 0.0

        # --- Step 1: Exact match after normalization ---
        norm = self._normalize(cleaned)
        if norm in self._name_to_rows:
            candidates = self._name_to_rows[norm]
            candidates = self._filter_candidates(candidates, site_num, state_hint, table_num)
            if candidates:
                return candidates[0], 100.0

        # --- Step 2: Fuzzy match against all canonical names ---
        # difflib.get_close_matches is fast for moderate list sizes
        close = difflib.get_close_matches(
            self._normalize(cleaned),
            [self._normalize(n) for n in self._canonical_names],
            n=5,
            cutoff=LOW_THRESHOLD / 100.0
        )

        if not close:
            return None, 0.0

        # Map normalized names back to rows and score them
        best_row = None
        best_score = 0.0

        for norm_match in close:
            # Find the original canonical name for this normalized form
            matching_rows = self._name_to_rows.get(norm_match, [])
            if not matching_rows:
                # Try to find via canonical_names list
                for orig_name in self._canonical_names:
                    if self._normalize(orig_name) == norm_match:
                        matching_rows = self._name_to_rows.get(self._normalize(orig_name), [])
                        break

            score = difflib.SequenceMatcher(None, norm, norm_match).ratio() * 100.0

            # Apply bonuses for contextual hints
            filtered = self._filter_candidates(matching_rows, site_num, state_hint, table_num)
            if filtered:
                if state_hint and filtered[0]['state_abbr'] == state_hint:
                    score = min(score * 1.05, 100.0)   # 5% bonus for state match
                if site_num and filtered[0]['site_num'] == site_num:
                    score = min(score * 1.05, 100.0)   # 5% bonus for site match

                if score > best_score:
                    best_score = score
                    best_row = filtered[0]
            elif matching_rows and score > best_score:
                # No filtered candidates but found something — use unfiltered
                best_score = score
                best_row = matching_rows[0]

        return best_row, best_score

    def _filter_candidates(
        self,
        rows: List[Dict],
        site_num: Optional[int],
        state_hint: Optional[str],
        table_num: Optional[int],
    ) -> List[Dict]:
        """
        Filter candidate rows by site number, state hint, and table number.
        Returns filtered list (may be empty if constraints too strict).
        """
        result = rows[:]

        # Filter by table number (respect the asterisk rules)
        if table_num is not None:
            result = [r for r in result if r['max_table'] >= table_num]

        # Filter by state
        if state_hint and result:
            state_filtered = [r for r in result if r['state_abbr'] == state_hint]
            if state_filtered:
                result = state_filtered

        # Filter by site number
        if site_num is not None and result:
            site_filtered = [r for r in result if r['site_num'] == site_num]
            if site_filtered:
                result = site_filtered

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        raw_name: str,
        site_num: Optional[int] = None,
        state_hint: Optional[str] = None,
        table_num: Optional[int] = None,
    ) -> Dict:
        """
        Resolve a single OCR'd station name to its canonical form.

        Args:
            raw_name:   The raw string from the Station_Location column.
            site_num:   Site number from the adjacent Site column (int or None).
            state_hint: Two-letter state abbr if known from context.
            table_num:  Data table number (1-31) to filter asterisk stations.

        Returns:
            dict with keys:
                raw_name        - original input
                canonical_name  - resolved canonical city name (or raw if unmatched)
                city            - city field from master list
                state           - full state name
                state_abbr      - two-letter abbreviation
                region          - geographic region
                urban_type      - Urban / Suburban / Nonurban
                station_order   - sequential row number in every data table
                site_num        - site number (int or None)
                max_table       - highest table number this station appears in
                match_score     - 0-100 similarity score
                match_type      - 'exact', 'high_confidence', 'low_confidence', 'unmatched'
                validated       - True if match_score >= HIGH_THRESHOLD
        """
        best_row, score = self._find_best_match(raw_name, site_num, state_hint, table_num)

        if best_row is None or score < LOW_THRESHOLD:
            return {
                'raw_name':       raw_name,
                'canonical_name': raw_name,   # keep original
                'city':           '',
                'state':          '',
                'state_abbr':     '',
                'region':         '',
                'urban_type':     '',
                'station_order':  None,
                'site_num':       None,
                'max_table':      None,
                'match_score':    round(score, 1),
                'match_type':     'unmatched',
                'validated':      False,
            }

        if score >= EXACT_THRESHOLD:
            match_type = 'exact'
        elif score >= HIGH_THRESHOLD:
            match_type = 'high_confidence'
        else:
            match_type = 'low_confidence'

        return {
            'raw_name':       raw_name,
            'canonical_name': best_row['canonical_name'],
            'city':           best_row['city'],
            'state':          best_row['state'],
            'state_abbr':     best_row['state_abbr'],
            'region':         best_row['region'],
            'urban_type':     best_row['urban_type'],
            'station_order':  best_row['station_order'],
            'site_num':       best_row['site_num'],
            'max_table':      best_row['max_table'],
            'match_score':    round(score, 1),
            'match_type':     match_type,
            'validated':      score >= HIGH_THRESHOLD,
        }

    def resolve_dataframe(
        self,
        df,                            # pandas DataFrame
        name_col: str = 'Station_Location',
        site_col: Optional[str] = 'Site',
        state_hint: Optional[str] = None,
        table_num: Optional[int] = None,
        inplace: bool = True,
    ):
        """
        Resolve station names for an entire cleaned table DataFrame.

        Adds four new columns:
            Resolved_Name   - canonical station name (or original if unmatched)
            Station_Order   - sequential order from master list (key for cross-table joins)
            Match_Score     - 0-100 similarity
            Match_Type      - 'exact' | 'high_confidence' | 'low_confidence' | 'unmatched'

        Args:
            df:          The cleaned table DataFrame (post TableCleaner).
            name_col:    Column containing raw station names.
            site_col:    Column containing site numbers (None to skip).
            state_hint:  Optional two-letter state abbreviation for all rows.
            table_num:   Data table number for asterisk filtering.
            inplace:     If True, modify df in place. If False, return a copy.

        Returns:
            DataFrame with added resolution columns.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for resolve_dataframe()")

        if not inplace:
            df = df.copy()

        if name_col not in df.columns:
            print(f"[StationResolver] Warning: '{name_col}' column not found. Skipping.")
            return df

        resolved_names   = []
        station_orders   = []
        match_scores     = []
        match_types      = []

        total = len(df)
        exact_count      = 0
        high_count       = 0
        low_count        = 0
        unmatched_count  = 0

        for idx, row in df.iterrows():
            raw = str(row.get(name_col, '')).strip()

            # Extract site number from Site column if present
            site_num = None
            if site_col and site_col in df.columns:
                site_val = str(row.get(site_col, '')).strip()
                if site_val.isdigit():
                    site_num = int(site_val)

            # ── State aggregate rows (15-col tables: "Connecticut Total") ──
            if self._is_state_aggregate_row(raw):
                resolved_names.append(raw)
                station_orders.append(None)
                match_scores.append(None)
                match_types.append('state_aggregate')
                continue

            # ── Skip obvious non-station rows (headers, separators) ──
            if self._is_non_station_row(raw):
                resolved_names.append(raw)
                station_orders.append(None)
                match_scores.append(None)
                match_types.append('skip')
                continue

            result = self.resolve(raw, site_num=site_num,
                                  state_hint=state_hint, table_num=table_num)

            resolved_names.append(result['canonical_name'])
            station_orders.append(result['station_order'])
            match_scores.append(result['match_score'])
            match_types.append(result['match_type'])

            t = result['match_type']
            if t == 'exact':            exact_count += 1
            elif t == 'high_confidence': high_count += 1
            elif t == 'low_confidence':  low_count += 1
            else:                        unmatched_count += 1

        df['Resolved_Name']  = resolved_names
        df['Station_Order']  = station_orders
        df['Match_Score']    = match_scores
        df['Match_Type']     = match_types

        # Summary report
        skipped    = match_types.count('skip')
        aggregates = match_types.count('state_aggregate')
        data_rows  = total - skipped - aggregates
        print(f"\n[StationResolver] Resolution complete for {total} rows")
        print(f"  Station rows resolved: {data_rows}")
        print(f"  State aggregate rows:  {aggregates}  (e.g. 'Connecticut Total')")
        print(f"  Skipped (headers):     {skipped}")
        print(f"  Exact match:           {exact_count:>5}  ({exact_count/max(data_rows,1)*100:.1f}%)")
        print(f"  High confidence:       {high_count:>5}  ({high_count/max(data_rows,1)*100:.1f}%)")
        print(f"  Low confidence:        {low_count:>5}  ({low_count/max(data_rows,1)*100:.1f}%)")
        print(f"  Unmatched:             {unmatched_count:>5}  ({unmatched_count/max(data_rows,1)*100:.1f}%)")
        validated = exact_count + high_count
        print(f"  Validated total:       {validated:>5}  ({validated/max(data_rows,1)*100:.1f}%)")

        return df

    # Full state names as they appear in 15-col state total rows
    # e.g. "Connecticut Total", "New Hampshire Total"
    _STATE_NAMES = {
        'connecticut', 'maine', 'massachusetts', 'new hampshire',
        'rhode island', 'vermont', 'delaware', 'new jersey', 'new york',
        'pennsylvania', 'district of columbia', 'kentucky', 'maryland',
        'north carolina', 'puerto rico', 'virginia', 'west virginia',
        'alabama', 'florida', 'georgia', 'mississippi', 'south carolina',
        'tennessee', 'illinois', 'indiana', 'michigan', 'ohio', 'wisconsin',
        'iowa', 'kansas', 'minnesota', 'missouri', 'nebraska',
        'north dakota', 'south dakota', 'arkansas', 'louisiana',
        'new mexico', 'oklahoma', 'texas', 'colorado', 'idaho', 'montana',
        'utah', 'wyoming', 'alaska', 'arizona', 'california', 'hawaii',
        'nevada', 'oregon', 'washington',
    }

    @staticmethod
    def _is_state_aggregate_row(text: str) -> bool:
        """
        Return True if the text is a 15-col state total row such as
        'Connecticut Total' or 'Maine Total'.

        These are legitimate data rows (they hold state-level aggregates)
        but they don't map to any individual station in the master list,
        so they get match_type='state_aggregate' rather than 'unmatched'.
        """
        if not text:
            return False
        t = text.strip().lower()

        # Pattern: "<state name> total" — with or without leading/trailing noise
        # Also catch OCR variants like "ConnecticutTotal" (missing space)
        t_clean = re.sub(r'\s+', ' ', t).strip()
        if t_clean.endswith('total'):
            # Strip "total" and check if what's left is a known state name
            prefix = t_clean[:-5].strip()
            if prefix in StationResolver._STATE_NAMES:
                return True
        # Also handle "Total" alone on a row (summary line)
        if t_clean == 'total':
            return True

        return False

    @staticmethod
    def _is_non_station_row(text: str) -> bool:
        """
        Return True if the text is clearly a table header, section header,
        separator, or year aggregate — i.e. rows that should be entirely
        skipped with no resolution attempted.

        Note: State aggregate rows like 'Connecticut Total' are NOT caught
        here — they are handled separately by _is_state_aggregate_row so
        they get their own match_type rather than being silently skipped.
        """
        if not text or text.strip() == '':
            return True
        t = text.strip().lower()

        # Section / column headers
        header_keywords = [
            'urban', 'nonurban', 'suburban', 'station', 'location',
            'new england', 'mid atlantic', 'mid east', 'south east',
            'mid west', 'great plains', 'gulf south', 'rocky mountain',
            'pacific coast', 'region', '===',
        ]
        for kw in header_keywords:
            if kw in t:
                return True

        # Year aggregate rows (e.g. "53-57", "1953-1957", "All Years")
        if re.match(r'^[\d\-–]+$', t):
            return True
        if 'all years' in t or 'combined' in t:
            return True

        return False

    def generate_unmatched_report(self, df, name_col: str = 'Station_Location') -> str:
        """
        Generate a text report of all unmatched or low-confidence station names,
        useful for manually reviewing OCR errors.

        Returns the report as a string (also prints it).
        """
        if 'Match_Type' not in df.columns:
            return "No resolution data found. Run resolve_dataframe() first."

        unmatched = df[df['Match_Type'] == 'unmatched'][name_col].value_counts()
        low_conf  = df[df['Match_Type'] == 'low_confidence'][[name_col, 'Resolved_Name', 'Match_Score']]

        lines = []
        lines.append("=" * 60)
        lines.append("STATION RESOLUTION REPORT")
        lines.append("=" * 60)

        lines.append(f"\nUNMATCHED NAMES ({len(unmatched)} unique):")
        lines.append("-" * 40)
        if unmatched.empty:
            lines.append("  None — all stations resolved!")
        else:
            for name, count in unmatched.items():
                lines.append(f"  [{count:>3}x]  '{name}'")

        lines.append(f"\nLOW CONFIDENCE MATCHES ({len(low_conf)} rows):")
        lines.append("-" * 40)
        if low_conf.empty:
            lines.append("  None")
        else:
            for _, r in low_conf.drop_duplicates(subset=[name_col]).iterrows():
                lines.append(
                    f"  '{r[name_col]}' → '{r['Resolved_Name']}'  "
                    f"(score: {r['Match_Score']:.1f})"
                )

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        return report


# ---------------------------------------------------------------------------
# Convenience function for use inside TableCleaner.run()
# ---------------------------------------------------------------------------

def apply_station_resolution(
    df,
    master_csv_path: str,
    name_col: str = 'Station_Location',
    site_col: Optional[str] = 'Site',
    table_num: Optional[int] = None,
) -> Tuple:
    """
    One-call helper: create resolver, apply to df, return (df, report_text).

    Intended to be dropped into TableCleaner.run() with minimal changes:

        from StationResolver import apply_station_resolution
        df, station_report = apply_station_resolution(df, "master_stations.csv")

    Args:
        df:              Cleaned DataFrame (post rename_columns step).
        master_csv_path: Path to master_stations.csv.
        name_col:        Station name column (default 'Station_Location').
        site_col:        Site number column (default 'Site', None to skip).
        table_num:       Table number for asterisk filtering (optional).

    Returns:
        (df_with_resolution, report_string)
    """
    try:
        resolver = StationResolver(master_csv_path)
        df = resolver.resolve_dataframe(df, name_col=name_col,
                                        site_col=site_col, table_num=table_num)
        report = resolver.generate_unmatched_report(df, name_col=name_col)
        return df, report
    except Exception as e:
        print(f"[StationResolver] Warning: resolution failed — {e}")
        return df, f"Resolution failed: {e}"


# ---------------------------------------------------------------------------
# Optional: cross-table consistency check
# ---------------------------------------------------------------------------

def check_cross_table_consistency(
    dataframes: Dict[str, any],     # {table_label: df} dict of resolved DataFrames
    name_col: str = 'Station_Location',
) -> str:
    """
    Given multiple resolved DataFrames (one per pollutant table), check that
    the station_order sequence is consistent across all of them.

    Because the PDF guarantees all tables share the same station order, any
    disagreement indicates an OCR or reconstruction error.

    Args:
        dataframes: Dict mapping a label (e.g. 'Table_1_Particulates') to a
                    resolved DataFrame. Each df must have 'Station_Order' column.
        name_col:   The station name column.

    Returns:
        A text report of any inconsistencies.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CROSS-TABLE STATION ORDER CONSISTENCY CHECK")
    lines.append("=" * 60)

    if len(dataframes) < 2:
        lines.append("Need at least 2 tables to compare.")
        return "\n".join(lines)

    # Use the first table as the reference order
    ref_label, ref_df = next(iter(dataframes.items()))
    if 'Station_Order' not in ref_df.columns:
        lines.append(f"Reference table '{ref_label}' has no Station_Order column.")
        return "\n".join(lines)

    ref_order = ref_df['Station_Order'].dropna().tolist()
    lines.append(f"Reference table: {ref_label} ({len(ref_order)} stations)")
    lines.append("")

    inconsistencies = 0
    for label, df in list(dataframes.items())[1:]:
        if 'Station_Order' not in df.columns:
            lines.append(f"  {label}: No Station_Order column — skipped")
            continue

        other_order = df['Station_Order'].dropna().tolist()
        mismatches = []

        for i, (r, o) in enumerate(zip(ref_order, other_order)):
            if r != o:
                ref_name = ref_df[ref_df['Station_Order'] == r][name_col].iloc[0] \
                    if not ref_df[ref_df['Station_Order'] == r].empty else '?'
                other_name = df[df['Station_Order'] == o][name_col].iloc[0] \
                    if not df[df['Station_Order'] == o].empty else '?'
                mismatches.append(
                    f"    Row {i+1}: ref={r} ({ref_name})  vs  {label}={o} ({other_name})"
                )

        if mismatches:
            lines.append(f"  {label}: {len(mismatches)} mismatch(es)")
            lines.extend(mismatches[:10])   # Show first 10 only
            if len(mismatches) > 10:
                lines.append(f"    ... and {len(mismatches)-10} more")
            inconsistencies += len(mismatches)
        else:
            lines.append(f"  {label}: ✓ Consistent ({len(other_order)} stations)")

    lines.append("")
    lines.append(f"Total inconsistencies found: {inconsistencies}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# CLI: quick test / diagnostic
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print("\n" + "=" * 60)
    print("StationResolver — Diagnostic Mode")
    print("=" * 60 + "\n")

    csv_path = input("Path to master_stations.csv [master_stations.csv]: ").strip()
    if not csv_path:
        csv_path = "master_stations.csv"

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    resolver = StationResolver(csv_path)

    print("\nEnter OCR station name strings to test (blank to quit):\n")
    while True:
        raw = input("  Test name: ").strip()
        if not raw:
            break

        site_input = input("  Site number (blank for none): ").strip()
        site_num = int(site_input) if site_input.isdigit() else None

        result = resolver.resolve(raw, site_num=site_num)

        print(f"\n  Raw:            '{result['raw_name']}'")
        print(f"  Resolved:       '{result['canonical_name']}'")
        print(f"  City:           {result['city']}")
        print(f"  State:          {result['state_abbr']}")
        print(f"  Region:         {result['region']}")
        print(f"  Station Order:  {result['station_order']}")
        print(f"  Match Score:    {result['match_score']}")
        print(f"  Match Type:     {result['match_type']}")
        print(f"  Validated:      {result['validated']}")
        print()