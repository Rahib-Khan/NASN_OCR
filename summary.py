"""
Cleaning Report Summarizer
Aggregates all cleaning reports from cleaned tables and provides comprehensive statistics.
"""

import pandas as pd
import os
import glob
import re
from typing import List, Dict, Tuple


class CleaningReportSummarizer:
    """
    Summarizes cleaning reports from all cleaned tables.
    """
    
    def __init__(self, cleaned_dir: str, original_tables_dir: str = None):
        """
        Initialize summarizer.
        
        Args:
            cleaned_dir: Directory containing cleaned tables and reports
            original_tables_dir: Directory containing original tables (to count uncleaned)
        """
        self.cleaned_dir = cleaned_dir
        self.original_tables_dir = original_tables_dir
        
        self.report_files = []
        self.cleaned_tables = []
        self.table_stats = []
        
    def find_report_files(self):
        """Find all cleaning report text files."""
        pattern = os.path.join(self.cleaned_dir, "*_cleaning_report.txt")
        self.report_files = sorted(glob.glob(pattern))
        
        print(f"Found {len(self.report_files)} cleaning reports")
        return self.report_files
    
    def find_cleaned_tables(self):
        """Find all cleaned table CSV files."""
        pattern = os.path.join(self.cleaned_dir, "Table_*_cleaned.csv")
        self.cleaned_tables = sorted(glob.glob(pattern))
        
        print(f"Found {len(self.cleaned_tables)} cleaned tables")
        return self.cleaned_tables
    
    def parse_report_file(self, report_path: str) -> Dict:
        """
        Parse a cleaning report file to extract statistics.
        Handles 16-column, 15-column, and 14-column monthly tables.
        
        Returns:
            Dictionary with extracted stats
        """
        stats = {
            'report_file': os.path.basename(report_path),
            'table_file': None,
            'table_type': None,
            'input_rows': None,
            'output_rows': None,
            'removed_rows': None,
            'filled_cells': None,
            'total_cells': None,
            'fill_percentage': None,
            'initial_confidence': None,
            'final_confidence': None,
            'confidence_change': None,
            # 14-col specific fields
            'anomalies_detected': None,
            'rows_repaired': None,
            'monthly_cells_cleared': None,
            'ordering_violations_cleared': None,
        }
        
        # Extract table filename from report filename
        # "Table_1_Pages_1-8_cleaning_report.txt" -> "Table_1_Pages_1-8"
        table_name = report_path.replace('_cleaning_report.txt', '')
        table_name = os.path.basename(table_name)
        stats['table_file'] = table_name
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # ── Detect table type ─────────────────────────────────────────────
            if '14-column monthly table' in content or '[14-COL PIPELINE]' in content:
                stats['table_type'] = '14-column'
            elif '16-column table' in content or '16 columns' in content:
                stats['table_type'] = '16-column'
            elif '15-column table' in content or '15 columns' in content:
                stats['table_type'] = '15-column'
            else:
                # Fallback: infer from column names mentioned in the report
                if 'Col_16' in content or 'P90' in content:
                    if 'Site' in content and 'Station_Location' in content:
                        stats['table_type'] = '16-column'
                    else:
                        stats['table_type'] = '15-column'
                elif any(m in content for m in ['Jan', 'Feb', 'Mar', 'Row_Type']):
                    stats['table_type'] = '14-column'
            
            # ── Common statistics ─────────────────────────────────────────────

            # Data completeness
            match = re.search(r'Filled Cells:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(([\d.]+)%\)', content)
            if match:
                stats['filled_cells'] = int(match.group(1).replace(',', ''))
                stats['total_cells'] = int(match.group(2).replace(',', ''))
                stats['fill_percentage'] = float(match.group(3))
            
            # Row counts
            match = re.search(r'Input rows:\s*(\d+)', content)
            if match:
                stats['input_rows'] = int(match.group(1))
            
            match = re.search(r'Output rows:\s*(\d+)', content)
            if match:
                stats['output_rows'] = int(match.group(1))
            
            match = re.search(r'Removed:\s*(\d+)', content)
            if match:
                stats['removed_rows'] = int(match.group(1))
            
            # Confidence
            match = re.search(r'Initial:\s*([\d.]+)', content)
            if match:
                stats['initial_confidence'] = float(match.group(1))
            
            match = re.search(r'Final:\s*([\d.]+)', content)
            if match:
                stats['final_confidence'] = float(match.group(1))
            
            match = re.search(r'Change:\s*([+-]?[\d.]+)', content)
            if match:
                stats['confidence_change'] = float(match.group(1))
            
            # ── 14-column specific statistics ─────────────────────────────────

            match = re.search(r'Anomalies detected:\s*(\d+)', content)
            if match:
                stats['anomalies_detected'] = int(match.group(1))
            
            match = re.search(r'Rows repaired:\s*(\d+)', content)
            if match:
                stats['rows_repaired'] = int(match.group(1))
            
            # "Monthly cells cleared: N (M with non-numeric text warnings)"
            match = re.search(r'Monthly cells cleared:\s*(\d+)', content)
            if match:
                stats['monthly_cells_cleared'] = int(match.group(1))
            
            # "Cells cleared due to ordering violations: N"
            match = re.search(r'Cells cleared due to ordering violations:\s*(\d+)', content)
            if match:
                stats['ordering_violations_cleared'] = int(match.group(1))
        
        except Exception as e:
            print(f"  Warning: Error parsing {report_path}: {e}")
        
        return stats
    
    def analyze_cleaned_table(self, table_path: str) -> Dict:
        """
        Analyze a cleaned table CSV to get additional statistics.
        Detects 14-column monthly tables by looking for month columns and Row_Type.
        
        Returns:
            Dictionary with table analysis
        """
        stats = {
            'table_file': os.path.basename(table_path).replace('_cleaned.csv', ''),
            'num_columns': 0,
            'num_rows': 0,
            'has_site_column': False,
            'table_type': None,
            # 14-col extras (populated by analyze_14col_detail)
            'station_count': None, 'year_values': None, 'row_type_counts': None,
            'month_fill_rates': None, 'naturally_empty_slots': 0,
            'total_applicable_slots': 0,
            'month_means_max': None, 'month_means_avg': None, 'month_means_min': None,
        }
        
        try:
            df = pd.read_csv(table_path, nrows=5)
            
            # Count columns (excluding metadata)
            metadata_cols = ['Avg_Confidence', 'Page_Num', 'Is_Year_Aggregate',
                             'Is_Site_Aggregate', 'Row_Type']
            data_cols = [col for col in df.columns if col not in metadata_cols]
            stats['num_columns'] = len(data_cols)
            
            # Check for Site column
            stats['has_site_column'] = 'Site' in df.columns

            # ── Detect table type ─────────────────────────────────────────────
            month_cols = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'}
            has_month_cols = any(c in df.columns for c in month_cols)
            has_row_type = 'Row_Type' in df.columns

            if has_month_cols or has_row_type:
                stats['table_type'] = '14-column'
            elif stats['has_site_column'] and stats['num_columns'] >= 15:
                stats['table_type'] = '16-column'
            elif not stats['has_site_column'] and stats['num_columns'] == 15:
                stats['table_type'] = '15-column'
            elif stats['num_columns'] == 16:
                stats['table_type'] = '16-column'
            elif stats['num_columns'] == 15:
                stats['table_type'] = '15-column'
            
            # Get row count (read full file)
            df_full = pd.read_csv(table_path)
            stats['num_rows'] = len(df_full)
            
            # ── Extra deep analysis for 14-col tables ─────────────────────────
            if stats['table_type'] == '14-column':
                detail = self.analyze_14col_detail(df_full)
                stats.update(detail)
            
        except Exception as e:
            print(f"  Warning: Error analyzing {table_path}: {e}")
        
        return stats

    def analyze_14col_detail(self, df: 'pd.DataFrame') -> Dict:
        """
        Deep analysis of a 14-column monthly table DataFrame.

        Fill-rate logic (per station-group × month):
          - Name row cells in month columns are structurally always blank → excluded.
          - If ALL four data rows (Num_Samples, Maximum, Average, Minimum) are empty
            → the original page had no data there (natural absence) → excluded.
          - Otherwise → count all 4 data rows in denominator; credit whichever are filled.
        """
        MONTHS     = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
        DATA_ROWS  = ['Num_Samples', 'Maximum', 'Average', 'Minimum']
        existing_months = [m for m in MONTHS if m in df.columns]

        result = {
            'station_count': None, 'year_values': None, 'row_type_counts': None,
            'month_fill_rates': None, 'naturally_empty_slots': 0,
            'total_applicable_slots': 0,
            'month_means_max': None, 'month_means_avg': None, 'month_means_min': None,
        }

        if 'Station_Location' in df.columns:
            result['station_count'] = (
                df[df['Row_Type'] == 'Name']['Station_Location'].nunique()
                if 'Row_Type' in df.columns
                else df['Station_Location'].nunique()
            )

        if 'Years' in df.columns:
            all_years = set()
            for v in df['Years'].dropna():
                all_years.update(str(v).split())
            result['year_values'] = sorted(y for y in all_years if y.strip())

        if 'Row_Type' in df.columns:
            result['row_type_counts'] = df['Row_Type'].value_counts().to_dict()

        if not existing_months:
            return result

        if 'Row_Type' not in df.columns:
            # No group structure — plain per-column fill rate
            result['month_fill_rates'] = {
                m: round(pd.to_numeric(df[m], errors='coerce').notna().mean() * 100, 1)
                for m in existing_months
            }
            return result

        # Pre-build data-row slices once per group (avoid re-slicing per month)
        name_indices = df.index[df['Row_Type'] == 'Name'].tolist()
        groups = []
        for i, idx in enumerate(name_indices):
            nxt = name_indices[i + 1] if i + 1 < len(name_indices) else df.index[-1] + 1
            grp = df.loc[idx:nxt - 1]
            groups.append(grp[grp['Row_Type'].isin(DATA_ROWS)])

        fill_rates = {}
        nat_empty  = 0
        applicable = 0

        for m in existing_months:
            m_filled = m_denom = 0
            for grp in groups:
                if grp.empty:
                    continue
                vals = pd.to_numeric(grp[m], errors='coerce')
                if vals.isna().all():
                    nat_empty += 1          # whole group blank for this month → natural
                else:
                    m_denom  += len(vals)
                    m_filled += int(vals.notna().sum())
                    applicable += 1
            fill_rates[m] = round(m_filled / m_denom * 100 if m_denom else 0.0, 1)

        result['month_fill_rates']       = fill_rates
        result['naturally_empty_slots']  = nat_empty
        result['total_applicable_slots'] = applicable

        # Per-row-type monthly means
        for row_type, key in [('Maximum',  'month_means_max'),
                               ('Average',  'month_means_avg'),
                               ('Minimum',  'month_means_min')]:
            subset = df[df['Row_Type'] == row_type]
            result[key] = {
                m: (round(pd.to_numeric(subset[m], errors='coerce').mean(), 1)
                    if pd.to_numeric(subset[m], errors='coerce').notna().any() else None)
                for m in existing_months
            }

        return result
    
    def count_uncleaned_tables(self) -> Tuple[int, int, int]:
        """
        Count how many original tables still need cleaning.
        
        Returns:
            (total_original, total_cleaned, remaining)
        """
        if not self.original_tables_dir or not os.path.exists(self.original_tables_dir):
            return 0, 0, 0
        
        # Find original table files
        original_pattern = os.path.join(self.original_tables_dir, "Table_*.csv")
        original_files = glob.glob(original_pattern)
        
        # Exclude combined file
        original_files = [f for f in original_files if 'Combined' not in f]
        
        total_original = len(original_files)
        total_cleaned = len(self.cleaned_tables)
        remaining = total_original - total_cleaned
        
        return total_original, total_cleaned, remaining
    
    @staticmethod
    def _smart_fill_pct(s: Dict) -> float:
        """
        Return the smart fill rate for a single table dict.
        For 14-col tables this is the mean of month_fill_rates (natural-absence-excluded).
        For 15/16-col tables fall back to the raw fill_percentage from the cleaning report.
        """
        if s.get('table_type') == '14-column' and s.get('month_fill_rates'):
            rates = list(s['month_fill_rates'].values())
            return round(sum(rates) / len(rates), 1) if rates else 0.0
        return s.get('fill_percentage', 0.0) or 0.0

    def generate_summary(self, output_file: str = None):
        """
        Generate comprehensive summary report covering 14-, 15-, and 16-column tables.
        """
        print("\n" + "="*80)
        print("CLEANING REPORT SUMMARIZER")
        print("="*80)
        print(f"Cleaned tables directory: {self.cleaned_dir}")
        if self.original_tables_dir:
            print(f"Original tables directory: {self.original_tables_dir}")
        print("="*80)

        # Find files
        self.find_report_files()
        self.find_cleaned_tables()

        if not self.report_files and not self.cleaned_tables:
            print("\n⚠ No cleaned tables or reports found!")
            return

        print("\nParsing reports...")

        # Parse all reports
        report_stats = []
        for report_file in self.report_files:
            stats = self.parse_report_file(report_file)
            report_stats.append(stats)

        # Analyze all cleaned tables
        table_stats = []
        for table_file in self.cleaned_tables:
            stats = self.analyze_cleaned_table(table_file)
            table_stats.append(stats)

        # Merge stats by table name
        combined_stats = []
        for report in report_stats:
            table_name = report['table_file']
            matching_table = next(
                (t for t in table_stats if t['table_file'] == table_name), None)

            if matching_table:
                combined = {**report, **matching_table}
                if not combined.get('table_type') or combined.get('table_type') == 'unknown':
                    combined['table_type'] = matching_table.get('table_type')
                combined_stats.append(combined)
            else:
                combined_stats.append(report)

        # Separate by table type
        stats_16col   = [s for s in combined_stats if s.get('table_type') == '16-column']
        stats_15col   = [s for s in combined_stats if s.get('table_type') == '15-column']
        stats_14col   = [s for s in combined_stats if s.get('table_type') == '14-column']
        stats_unknown = [s for s in combined_stats
                         if s.get('table_type') not in ('16-column', '15-column', '14-column')
                         or not s.get('table_type')]

        # Count uncleaned tables
        total_original, total_cleaned, remaining = self.count_uncleaned_tables()

        # ── Build report text ─────────────────────────────────────────────────
        lines = []
        W = 80  # line width

        def section(title, subtitle=None):
            lines.append("=" * W)
            lines.append(title)
            if subtitle:
                lines.append(subtitle)
            lines.append("=" * W)

        def note(text):
            """Wrap a plain-English note at W chars."""
            import textwrap
            for ln in textwrap.wrap(text, W - 2):
                lines.append("  " + ln)

        lines.append("=" * W)
        lines.append("COMPREHENSIVE CLEANING SUMMARY REPORT")
        lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * W)
        lines.append("")
        note("This report summarises how much of the original historical air-pollution "
             "data was successfully extracted from the scanned document pages.  "
             "Each section below explains what was found and how to read the numbers.")
        lines.append("")

        # ── SECTION 1: Progress ───────────────────────────────────────────────
        section("SECTION 1 — CLEANING PROGRESS",
                "How many of the original scanned tables have been processed so far.")
        lines.append("")
        if self.original_tables_dir:
            completion_pct = (total_cleaned / total_original * 100) if total_original > 0 else 0
            lines.append(f"  Original tables found   : {total_original}")
            lines.append(f"  Tables cleaned so far   : {total_cleaned}")
            lines.append(f"  Tables still to process : {remaining}")
            lines.append(f"  Overall completion      : {completion_pct:.1f}%")
            if remaining == 0:
                lines.append("  ✓ All tables have been cleaned!")
        else:
            lines.append(f"  Tables cleaned so far: {len(combined_stats)}")
            note("(Original tables directory not provided — completion % cannot be calculated.)")
        lines.append("")

        # ── SECTION 2: Per-table overview ─────────────────────────────────────
        section("SECTION 2 — INDIVIDUAL TABLE OVERVIEW",
                "One row per cleaned table.  Fill % = share of expected cells that contain a value.")
        lines.append("")
        note("NOTE — 14-column monthly tables: the Fill % shown here is the ADJUSTED "
             "rate.  It only counts cells from months where measurements actually existed "
             "in the original document.  Station-name rows and months with no measurements "
             "are excluded so the percentage reflects true OCR capture quality, not "
             "structural blanks in the original pages.")
        lines.append("")
        lines.append(
            f"  {'Table':<32} {'Type':<12} {'Rows':<7} {'Fill %':<8} {'Confidence'}"
        )
        lines.append("  " + "-" * 68)
        for s in combined_stats:
            tname = (s['table_file'][:30] if len(s['table_file']) > 30 else s['table_file'])
            ttype = (s.get('table_type') or 'Unknown')[:10]
            rows  = s.get('output_rows', 0) or 0
            fpct  = self._smart_fill_pct(s)
            conf  = s.get('final_confidence', 0.0) or 0.0
            flag  = "  [adjusted]" if s.get('table_type') == '14-column' else ""
            lines.append(
                f"  {tname:<32} {ttype:<12} {rows:<7} {fpct:<8.1f} {conf:.1f}{flag}"
            )
        lines.append("")

        # ── SECTION 3: 14-column monthly tables ───────────────────────────────
        if stats_14col:
            MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']

            section("SECTION 3 — 14-COLUMN MONTHLY TABLES",
                    "Tables where each column is a month (January through December).")
            lines.append("")
            note("HOW TO READ THESE TABLES: Each station (city) has five rows — the "
                 "station name, number of samples, maximum reading, average reading, "
                 "and minimum reading.  Not every station collected data every month; "
                 "when an entire month is blank for a station it means no measurements "
                 "were taken that month, not that the scanner missed something.  The "
                 "fill rate below only counts months where at least one measurement "
                 "existed in the original document.")
            lines.append("")

            # Aggregate using smart fill rates
            smart_rates_14 = [self._smart_fill_pct(s) for s in stats_14col
                              if s.get('month_fill_rates')]
            avg_smart_14   = round(sum(smart_rates_14) / len(smart_rates_14), 1) if smart_rates_14 else 0.0
            raw_rates_14   = [s.get('fill_percentage', 0.0) or 0.0 for s in stats_14col]
            avg_raw_14     = round(sum(raw_rates_14) / len(raw_rates_14), 1) if raw_rates_14 else 0.0
            total_rows_14  = sum(s.get('output_rows', 0) or 0 for s in stats_14col)
            s_conf_14      = [s['final_confidence'] for s in stats_14col if s.get('final_confidence')]
            avg_conf_14    = round(sum(s_conf_14) / len(s_conf_14), 1) if s_conf_14 else 0.0

            total_empty_slots  = sum(s.get('naturally_empty_slots', 0) for s in stats_14col)
            total_applic_slots = sum(s.get('total_applicable_slots', 0) for s in stats_14col)

            lines.append(f"  Number of tables   : {len(stats_14col)}")
            lines.append(f"  Total data rows    : {total_rows_14:,}")
            lines.append(f"  Average confidence : {avg_conf_14:.1f}  "
                         f"(how clearly the scanner read the characters — higher is better)")
            lines.append("")
            lines.append(f"  FILL RATE (adjusted) : {avg_smart_14:.1f}%")
            note(f"This is the percentage of cells that contained a value in the original "
                 f"document and were successfully captured.  {total_applic_slots:,} "
                 f"station-month slots were measured; {total_empty_slots:,} slots were "
                 f"skipped because all four data rows were blank in the original "
                 f"(meaning no measurements were taken that month for that station).")
            lines.append("")
            lines.append(f"  Raw fill rate (for reference only) : {avg_raw_14:.1f}%")
            note("The raw rate is lower because it treats station-name row blanks and "
                 "months with no measurements as missing data.  It is kept here only "
                 "for reference — the adjusted rate above is the meaningful number.")
            lines.append("")

            # Station / year coverage
            total_stations = sum(s.get('station_count') or 0 for s in stats_14col)
            all_years = set()
            for s in stats_14col:
                all_years.update(s.get('year_values') or [])
            lines.append(f"  Station entries in these tables : {total_stations}")
            lines.append(
                f"  Year values recorded            : "
                f"{', '.join(sorted(all_years)) if all_years else 'N/A'}"
            )
            lines.append("")

            # Row-type distribution
            type_totals: Dict[str, int] = {}
            for s in stats_14col:
                for rt, cnt in (s.get('row_type_counts') or {}).items():
                    type_totals[rt] = type_totals.get(rt, 0) + cnt
            if type_totals:
                lines.append("  Row counts by type (across all 14-column tables):")
                note("Each station block should have exactly 5 rows: Name, "
                     "Num_Samples, Maximum, Average, Minimum.")
                for rt in ['Name', 'Num_Samples', 'Maximum', 'Average', 'Minimum']:
                    lines.append(f"    {rt:<15}: {type_totals.get(rt, 0):,} rows")
                for rt, cnt in sorted(type_totals.items()):
                    if rt not in ('Name','Num_Samples','Maximum','Average','Minimum'):
                        lines.append(f"    {rt:<15}: {cnt:,}  ← unexpected — review needed")
                lines.append("")

            # OCR quality indicators
            total_anomalies   = sum(s.get('anomalies_detected') or 0 for s in stats_14col)
            total_repairs     = sum(s.get('rows_repaired') or 0 for s in stats_14col)
            total_cells_clear = sum(s.get('monthly_cells_cleared') or 0 for s in stats_14col)
            total_violations  = sum(s.get('ordering_violations_cleared') or 0 for s in stats_14col)
            lines.append("  Cleaning / OCR quality checks:")
            lines.append(f"    Station-block order problems found   : {total_anomalies}"
                         + ("  ← review recommended" if total_anomalies else "  ✓ none"))
            lines.append(f"    Station rows automatically repaired  : {total_repairs}")
            lines.append(f"    Bad-text cells cleared               : {total_cells_clear}"
                         + ("  (scanner read letters instead of numbers)" if total_cells_clear else ""))
            lines.append(f"    Max/Avg/Min order violations cleared : {total_violations}"
                         + ("  (a max value was less than its average, etc.)" if total_violations else ""))
            lines.append("")

            # Per-month fill rate grid
            tables_with_fill = [s for s in stats_14col if s.get('month_fill_rates')]
            if tables_with_fill:
                lines.append("  Adjusted fill rate by month (average across all 14-column tables):")
                note("A low month does not necessarily indicate a scanning problem — "
                     "some months simply had fewer stations reporting data in the "
                     "original documents.")
                hdr = f"    {'':6}" + "".join(f" {m:<7}" for m in MONTHS)
                lines.append(hdr)
                fill_row = f"    {'Fill %':<6}"
                for m in MONTHS:
                    rates = [s['month_fill_rates'][m] for s in tables_with_fill
                             if s.get('month_fill_rates') and m in s['month_fill_rates']]
                    fill_row += f" {(sum(rates)/len(rates) if rates else 0.0):<7.1f}"
                lines.append(fill_row)
                lines.append("")

                # Typical pollutant level grid
                for label, key, desc in [
                    ('Max',  'month_means_max', 'Highest reading recorded that month (avg across stations)'),
                    ('Avg',  'month_means_avg', 'Average reading'),
                    ('Min',  'month_means_min', 'Lowest reading'),
                ]:
                    twk = [s for s in stats_14col if s.get(key)]
                    if not twk:
                        continue
                    row_str = f"    {label:<6}"
                    for m in MONTHS:
                        vals = [s[key][m] for s in twk if s.get(key) and s[key].get(m) is not None]
                        row_str += f" {(f'{sum(vals)/len(vals):.0f}' if vals else 'N/A'):<7}"
                    lines.append(row_str)
                lines.append(f"    (values are means across all stations and tables)")
                lines.append("")

            # Per-table breakdown
            lines.append("  Per-table detail:")
            lines.append(
                f"    {'Table':<32} {'Stations':<10} {'Adj.Fill%':<11} "
                f"{'Empty slots':<13} {'Confidence'}"
            )
            lines.append("    " + "-" * 75)
            for s in stats_14col:
                tname    = s['table_file'][:30]
                stations = s.get('station_count') if s.get('station_count') is not None else '-'
                fpct     = self._smart_fill_pct(s)
                empty_sl = s.get('naturally_empty_slots', 0)
                conf     = s.get('final_confidence', 0.0) or 0.0
                lines.append(
                    f"    {tname:<32} {str(stations):<10} {fpct:<11.1f} "
                    f"{str(empty_sl):<13} {conf:.1f}"
                )
            lines.append("")

        # ── SECTION 4: 16-column tables ───────────────────────────────────────
        if stats_16col:
            section("SECTION 4 — 16-COLUMN TABLES",
                    "Annual summary tables with a Site column and percentile columns (P10–P90).")
            lines.append("")
            note("These tables record one row per station with annual statistics: "
                 "number of samples, min, max, average, and pollution percentiles.  "
                 "Fill % here is straightforward — it is the share of expected cells "
                 "that have a value.")
            lines.append("")

            total_filled_16 = sum(s.get('filled_cells', 0) or 0 for s in stats_16col)
            total_cells_16  = sum(s.get('total_cells', 0) or 0 for s in stats_16col)
            avg_fill_16     = (total_filled_16 / total_cells_16 * 100) if total_cells_16 > 0 else 0
            total_rows_16   = sum(s.get('output_rows', 0) or 0 for s in stats_16col)
            s_conf_16       = [s['final_confidence'] for s in stats_16col if s.get('final_confidence')]
            avg_conf_16     = round(sum(s_conf_16) / len(s_conf_16), 1) if s_conf_16 else 0.0

            lines.append(f"  Number of tables   : {len(stats_16col)}")
            lines.append(f"  Total data rows    : {total_rows_16:,}")
            lines.append(f"  Cells with values  : {total_filled_16:,} / {total_cells_16:,}")
            lines.append(f"  Fill rate          : {avg_fill_16:.1f}%")
            lines.append(f"  Average confidence : {avg_conf_16:.1f}")
            lines.append("")
            lines.append(f"  {'Table':<32} {'Rows':<7} {'Fill %':<9} {'Confidence'}")
            lines.append("  " + "-" * 60)
            for s in stats_16col:
                lines.append(
                    f"  {s['table_file'][:30]:<32} "
                    f"{s.get('output_rows',0) or 0:<7} "
                    f"{s.get('fill_percentage',0.0) or 0.0:<9.1f} "
                    f"{s.get('final_confidence',0.0) or 0.0:.1f}"
                )
            lines.append("")

        # ── SECTION 5: 15-column tables ───────────────────────────────────────
        if stats_15col:
            section("SECTION 5 — 15-COLUMN TABLES",
                    "Annual summary tables without a Site column (otherwise same as 16-column).")
            lines.append("")
            note("These are structured identically to the 16-column tables except that "
                 "individual monitoring sites within a station are not separated out.  "
                 "Fill % is the share of expected cells that have a value.")
            lines.append("")

            total_filled_15 = sum(s.get('filled_cells', 0) or 0 for s in stats_15col)
            total_cells_15  = sum(s.get('total_cells', 0) or 0 for s in stats_15col)
            avg_fill_15     = (total_filled_15 / total_cells_15 * 100) if total_cells_15 > 0 else 0
            total_rows_15   = sum(s.get('output_rows', 0) or 0 for s in stats_15col)
            s_conf_15       = [s['final_confidence'] for s in stats_15col if s.get('final_confidence')]
            avg_conf_15     = round(sum(s_conf_15) / len(s_conf_15), 1) if s_conf_15 else 0.0

            lines.append(f"  Number of tables   : {len(stats_15col)}")
            lines.append(f"  Total data rows    : {total_rows_15:,}")
            lines.append(f"  Cells with values  : {total_filled_15:,} / {total_cells_15:,}")
            lines.append(f"  Fill rate          : {avg_fill_15:.1f}%")
            lines.append(f"  Average confidence : {avg_conf_15:.1f}")
            lines.append("")
            lines.append(f"  {'Table':<32} {'Rows':<7} {'Fill %':<9} {'Confidence'}")
            lines.append("  " + "-" * 60)
            for s in stats_15col:
                lines.append(
                    f"  {s['table_file'][:30]:<32} "
                    f"{s.get('output_rows',0) or 0:<7} "
                    f"{s.get('fill_percentage',0.0) or 0.0:<9.1f} "
                    f"{s.get('final_confidence',0.0) or 0.0:.1f}"
                )
            lines.append("")

        # ── SECTION 6: Overall summary ────────────────────────────────────────
        section("SECTION 6 — OVERALL SUMMARY",
                "Combined totals across all cleaned tables.")
        lines.append("")
        note("The fill rates below use the adjusted calculation for 14-column tables "
             "(natural absences excluded) and the raw calculation for 14/15-column tables.")
        lines.append("")

        total_tables  = len(combined_stats)
        total_rows_all = sum(s.get('output_rows', 0) or 0 for s in combined_stats)
        total_removed  = sum(s.get('removed_rows', 0) or 0 for s in combined_stats)
        all_smart      = [self._smart_fill_pct(s) for s in combined_stats]
        avg_smart_all  = round(sum(all_smart) / len(all_smart), 1) if all_smart else 0.0
        s_conf_all     = [s['final_confidence'] for s in combined_stats if s.get('final_confidence')]
        avg_conf_all   = round(sum(s_conf_all) / len(s_conf_all), 1) if s_conf_all else 0.0

        lines.append(f"  Total tables cleaned      : {total_tables}")
        lines.append(f"    14-column monthly tables : {len(stats_14col)}")
        lines.append(f"    16-column annual tables  : {len(stats_16col)}")
        lines.append(f"    15-column annual tables  : {len(stats_15col)}")
        if stats_unknown:
            lines.append(f"    Unrecognised type        : {len(stats_unknown)}")
        lines.append("")
        lines.append(f"  Total data rows processed : {total_rows_all:,}")
        lines.append(f"  Rows removed (junk/blank) : {total_removed:,}")
        lines.append(f"  Average fill rate         : {avg_smart_all:.1f}%")
        lines.append(f"  Average OCR confidence    : {avg_conf_all:.1f}")
        lines.append("")

        # ── SECTION 7: Quality assessment ─────────────────────────────────────
        section("SECTION 7 — QUALITY ASSESSMENT",
                "A plain-English verdict on how complete and accurate the extracted data is.")
        lines.append("")

        if avg_smart_all >= 85:
            q_fill = "Excellent"
            q_note = "Nearly all data that existed in the original documents was captured."
        elif avg_smart_all >= 75:
            q_fill = "Good"
            q_note = "Most data was captured.  Spot-check a sample of tables to confirm."
        elif avg_smart_all >= 65:
            q_fill = "Fair"
            q_note = ("A noticeable portion of cells is missing.  Review the per-table "
                      "breakdown to identify which tables need manual checking.")
        else:
            q_fill = "Needs Review"
            q_note = ("A significant share of cells could not be read.  Manual review "
                      "of the lower-confidence tables is recommended before using the data.")

        if avg_conf_all >= 85:
            q_conf = "High"
        elif avg_conf_all >= 75:
            q_conf = "Good"
        elif avg_conf_all >= 65:
            q_conf = "Moderate"
        else:
            q_conf = "Low — review recommended"

        lines.append(f"  Data completeness : {q_fill} ({avg_smart_all:.1f}%)")
        note(q_note)
        lines.append("")
        lines.append(f"  OCR read quality  : {q_conf} (avg confidence {avg_conf_all:.1f} / 100)")
        note("Confidence measures how clearly the scanner could read each character.  "
             "Values above 80 are reliable; values below 70 suggest the original page "
             "may have been faded, smudged, or printed in an unusual font.")
        lines.append("")

        if stats_14col:
            total_anom_all = sum(s.get('anomalies_detected') or 0 for s in stats_14col)
            total_viol_all = sum(s.get('ordering_violations_cleared') or 0 for s in stats_14col)
            lines.append("  Monthly-table specific checks:")
            lines.append(
                f"    Station-order anomalies : {total_anom_all}"
                + ("  ✓ none found" if total_anom_all == 0
                   else "  ⚠  review individual cleaning reports")
            )
            lines.append(
                f"    Max/Avg/Min violations  : {total_viol_all}"
                + ("  ✓ none found" if total_viol_all == 0
                   else "  ⚠  cells were auto-cleared")
            )
            lines.append("")

        if remaining > 0:
            section("SECTION 8 — NEXT STEPS")
            lines.append("")
            lines.append(f"  • {remaining} table(s) still need to be cleaned.")
            lines.append(f"  • Estimated time remaining: ~{remaining * 2} minutes (approx. 2 min per table)")
            lines.append("")

        lines.append("=" * W)

        # ── Output ────────────────────────────────────────────────────────────
        report_text = '\n'.join(lines)
        print("\n" + report_text)

        if output_file is None:
            output_file = os.path.join(self.cleaned_dir, "comprehensive_cleaning_summary.txt")
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Summary saved to: {output_file}")

        # CSV detail export
        flat_stats = []
        for s in combined_stats:
            row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
            row['smart_fill_pct'] = self._smart_fill_pct(s)
            for k in ('year_values', 'row_type_counts', 'month_fill_rates',
                      'month_means_max', 'month_means_avg', 'month_means_min'):
                if s.get(k) is not None:
                    row[k] = str(s[k])
            flat_stats.append(row)
        csv_file = output_file.replace('.txt', '.csv')
        pd.DataFrame(flat_stats).to_csv(csv_file, index=False)
        print(f"✓ Detailed statistics saved to: {csv_file}")

        from TableReconstruct_improvements import write_reconstruction_report
        write_reconstruction_report(
            table_stats=combined_stats,
            output_path=os.path.join(self.cleaned_dir, "reconstruction_report.txt"),
)

        return combined_stats


def main():
    """
    Command-line interface for report summarizer.
    """
    print("\n" + "="*80)
    print("CLEANING REPORT SUMMARIZER")
    print("Aggregates statistics from all cleaned tables")
    print("="*80 + "\n")
    
    # Get directories
    cleaned_dir = input("Enter cleaned tables directory [default=Processed/Tables/Cleaned_Tables]: ").strip()
    if not cleaned_dir:
        cleaned_dir = "Processed/Tables/Cleaned_Tables"
    
    if not os.path.exists(cleaned_dir):
        print(f"❌ Error: Directory not found: {cleaned_dir}")
        return
    
    original_dir = input("Enter original tables directory (optional, for progress tracking) [default=Processed/Tables]: ").strip()
    if not original_dir:
        original_dir = "Processed/Tables"
    
    if not os.path.exists(original_dir):
        print(f"⚠ Warning: Original directory not found: {original_dir}")
        print("  Progress tracking will be limited")
        original_dir = None
    
    output_file = input("Enter output file path (optional) [default=auto]: ").strip()
    if not output_file:
        output_file = None
    
    # Create summarizer and run
    summarizer = CleaningReportSummarizer(cleaned_dir, original_dir)
    summarizer.generate_summary(output_file)
    
    print("\n✓ Report generation complete!\n")


if __name__ == "__main__":
    main()