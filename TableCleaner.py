import pandas as pd
import numpy as np
import os
import re
import glob
from StationResolver import apply_station_resolution
from typing import List, Dict, Tuple


class ImprovedTableCleaner:
    """
    Enhanced table cleaner based on actual patterns observed in 1950s air pollution data.
    Handles multi-site stations, site numbering, year patterns, and aggregate rows intelligently.
    
    NEW: Batch processing for tables with same column structure
    """
    
    def __init__(self, input_csv, output_dir=None, master_stations_path=None, table_num=None):
        self.input_csv = input_csv
        
        # Auto-create Cleaned_Tables directory
        if output_dir:
            self.output_dir = output_dir
        else:
            base_dir = os.path.dirname(input_csv) or '.'
            self.output_dir = os.path.join(base_dir, 'Cleaned_Tables')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output filename
        basename = os.path.basename(input_csv).replace('.csv', '_cleaned.csv')
        self.output_csv = os.path.join(self.output_dir, basename)
        
        self.report_lines = []
        
        # Valid years
        self.valid_years = {'53', '54', '55', '56', '57'}
        
        # Column mapping for 16-column tables (with Site)
        self.column_mapping_16 = {
            'Col_1': 'Station_Location',
            'Col_2': 'Site', 
            'Col_3': 'Years',
            'Col_4': 'Num_Samples',
            'Col_5': 'Min',
            'Col_6': 'Max', 
            'Col_7': 'Avg',
            'Col_8': 'P10',
            'Col_9': 'P20',
            'Col_10': 'P30',
            'Col_11': 'P40',
            'Col_12': 'P50',
            'Col_13': 'P60',
            'Col_14': 'P70',
            'Col_15': 'P80',
            'Col_16': 'P90'
        }
        
        # Column mapping for 15-column tables (without Site)
        self.column_mapping_15 = {
            'Col_1': 'Station_Location',
            'Col_2': 'Years',
            'Col_3': 'Num_Samples',
            'Col_4': 'Min',
            'Col_5': 'Max', 
            'Col_6': 'Avg',
            'Col_7': 'P10',
            'Col_8': 'P20',
            'Col_9': 'P30',
            'Col_10': 'P40',
            'Col_11': 'P50',
            'Col_12': 'P60',
            'Col_13': 'P70',
            'Col_14': 'P80',
            'Col_15': 'P90'
        }

        # Column mapping for 14-column monthly tables
        # Structure: Station_Location | Years | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec
        self.column_mapping_14 = {
            'Col_1': 'Station_Location',
            'Col_2': 'Years',
            'Col_3': 'Jan',
            'Col_4': 'Feb',
            'Col_5': 'Mar',
            'Col_6': 'Apr',
            'Col_7': 'May',
            'Col_8': 'Jun',
            'Col_9': 'Jul',
            'Col_10': 'Aug',
            'Col_11': 'Sep',
            'Col_12': 'Oct',
            'Col_13': 'Nov',
            'Col_14': 'Dec',
        }

        # Month columns list (used throughout 14col cleaning)
        self.month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Station location row types for 14-col tables (patterns to match in OCR text)
        # Order matters: Name row is first, then these four in sequence
        self.row_type_patterns = {
            'Num_Samples': re.compile(r'numb|sampl|num|no\.?\s*of|#', re.IGNORECASE),
            'Maximum':     re.compile(r'max', re.IGNORECASE),
            'Average':     re.compile(r'av[eg]|mean|Anchorage', re.IGNORECASE),
            'Minimum':     re.compile(r'min', re.IGNORECASE),
        }

        # Will be set after detecting column count
        self.column_mapping = None
        self.table_type = None  # Will be '16col', '15col', or '14col'

        self.master_stations_path = master_stations_path
        self.table_num = table_num
        
    def log(self, message):
        """Add message to report and print."""
        print(message)
        self.report_lines.append(message)
    
    def load_data(self):
        """Load CSV with proper type handling and detect table type."""
        self.log(f"Loading: {self.input_csv}")
        df = pd.read_csv(self.input_csv, dtype=str)
        self.log(f"  Loaded {len(df)} rows")
        
        # Detect table type based on column count
        data_cols = [col for col in df.columns if col.startswith('Col_')]
        num_data_cols = len(data_cols)
        
        if num_data_cols == 18:
            self.column_mapping = self.column_mapping_16
            self.table_type = '16col'
            self.log(f"  Detected: 16-column table (with Site column)")
        elif num_data_cols == 17:
            self.column_mapping = self.column_mapping_15
            self.table_type = '15col'
            self.log(f"  Detected: 15-column table (without Site column)")
        elif num_data_cols == 16:
            self.column_mapping = self.column_mapping_14
            self.table_type = '14col'
            self.log(f"  Detected: 14-column monthly table (Station | Years | Jan-Dec)")
        else:
            # Default to 16-column mapping
            self.column_mapping = self.column_mapping_16
            self.table_type = 'unknown'
            self.log(f"  Warning: {num_data_cols} columns detected (expected 14, 15, or 16)")
        
        return df
    
    def remove_empty_trailing_columns(self, df):
        """
        Remove empty columns that appear right before metadata columns.
        These are padding columns from the combined CSV that have no data.
        
        IMPORTANT: Only removes trailing empty columns working backwards from metadata.
        Stops at the first non-empty column to preserve sparse data columns.
        """
        self.log("\nRemoving empty trailing columns...")
        
        # Identify metadata columns
        metadata_cols = ['Avg_Confidence', 'Page_Num']
        
        # Get all columns
        all_cols = df.columns.tolist()
        
        # Find where metadata starts
        metadata_start_idx = len(all_cols)
        for meta_col in metadata_cols:
            if meta_col in all_cols:
                idx = all_cols.index(meta_col)
                metadata_start_idx = min(metadata_start_idx, idx)
        
        # Get data columns (everything before metadata)
        data_cols = all_cols[:metadata_start_idx]
        
        if not data_cols:
            self.log("  No data columns found")
            return df
        
        # Work BACKWARDS from the last data column to find trailing empty columns
        empty_cols = []
        
        for col in reversed(data_cols):
            # Check if column is completely empty
            is_empty = df[col].isna().all() or (df[col].astype(str).str.strip() == '').all()
            
            if is_empty:
                # Add to list of empty columns
                empty_cols.append(col)
            else:
                # Stop at first non-empty column
                # This preserves sparse middle columns
                break
        
        # Remove empty trailing columns
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.log(f"  Removed {len(empty_cols)} trailing empty columns: {', '.join(empty_cols)}")
        else:
            self.log("  No trailing empty columns found")
        
        return df
    
    def rename_columns(self, df):
        """Rename columns to meaningful names."""
        self.log("\nRenaming columns...")
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        
        # Handle Page_Num and Avg_Confidence if present
        if 'Page_Num' in df.columns:
            rename_dict['Page_Num'] = 'Page_Num'
        if 'Avg_Confidence' in df.columns:
            rename_dict['Avg_Confidence'] = 'Avg_Confidence'
            
        df = df.rename(columns=rename_dict)
        self.log(f"  Renamed {len(rename_dict)} columns")
        return df
    
    def remove_column_zero(self, df):
        """
        Remove Col_0 entirely if present.
        Col_0 is typically OCR positional noise and not part of the table schema.
        """
        self.log("\nRemoving Column 0 if present...")

        if 'Col_0' in df.columns:
            df = df.drop(columns=['Col_0'])
            self.log("  Col_0 removed")
        else:
            self.log("  Col_0 not present")

        return df
    
    def remove_page_footer_rows(self, df):
        """
        Remove footer rows at the end of each page.
        A footer row is defined as:
        - Last row of a Page_Num group
        - No Station_Location
        - No numeric or meaningful table data
        """
        self.log("\nRemoving footer rows at page ends...")

        if 'Page_Num' not in df.columns:
            self.log("  Skipping: Page_Num not found")
            return df

        data_columns = [
            'Site', 'Years', 'Num_Samples',
            'Min', 'Max', 'Avg',
            'P10', 'P20', 'P30', 'P40', 'P50',
            'P60', 'P70', 'P80', 'P90'
        ]

        rows_to_drop = []

        for page_num in df['Page_Num'].dropna().unique():
            page_df = df[df['Page_Num'] == page_num]

            if page_df.empty:
                continue

            last_idx = page_df.index[-1]
            last_row = df.loc[last_idx]

            station = str(last_row.get('Station_Location', '')).strip()

            # Footer must have no station name
            if station and station.lower() != 'nan':
                continue

            # Check for ANY real data in the row
            has_data = False
            for col in data_columns:
                if col in df.columns:
                    val = last_row.get(col)
                    if not pd.isna(val) and str(val).strip() != '':
                        has_data = True
                        break

            if not has_data:
                rows_to_drop.append(last_idx)

        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)
            self.log(f"  Removed {len(rows_to_drop)} footer rows")
        else:
            self.log("  No footer rows detected")

        return df
    
    def clean_station_names(self, df):
        """
        Clean station location names.
        Pattern: "WATERBURY CONN", "NEW YORK N Y", etc.
        """
        self.log("\nCleaning station names...")
        
        if 'Station_Location' not in df.columns:
            return df
        
        cleaned = 0
        for idx in df.index:
            station = str(df.at[idx, 'Station_Location']).strip()
            
            if station and station not in ['', 'nan']:
                # Remove extra spaces
                station = ' '.join(station.split())
                # Uppercase
                station = station.upper()
                df.at[idx, 'Station_Location'] = station
                cleaned += 1
        
        self.log(f"  Cleaned {cleaned} station names")
        return df
    
    def extract_site_numbers(self, df):
        """
        Extract site numbers intelligently.
        Sites can be: '1', '2', '1 2 3 4 5' (multi-site aggregate)
        """
        self.log("\nExtracting site numbers...")
        
        if 'Site' not in df.columns:
            return df
        
        for idx in df.index:
            site_val = str(df.at[idx, 'Site']).strip()
            
            # Extract all digits
            digits = re.findall(r'\d+', site_val)
            
            if digits:
                # Join multiple sites with space
                df.at[idx, 'Site'] = ' '.join(digits)
            else:
                df.at[idx, 'Site'] = ''
        
        self.log("  Site numbers extracted")
        return df
    
    def clean_years(self, df):
        """
        Clean years column with validation.
        Valid: '54', '55 56', '53 54 55 56 57'
        """
        self.log("\nCleaning years...")
        
        if 'Years' not in df.columns:
            return df
        
        cleaned = 0
        for idx in df.index:
            years_val = str(df.at[idx, 'Years']).strip()
            
            # Extract 2-digit numbers
            years = re.findall(r'\b\d{2}\b', years_val)
            
            # Filter to valid years only
            valid = sorted(set([y for y in years if y in self.valid_years]))
            
            if valid:
                df.at[idx, 'Years'] = ' '.join(valid)
                cleaned += 1
            else:
                df.at[idx, 'Years'] = ''
        
        self.log(f"  Cleaned {cleaned} year entries")
        return df
    
    def fill_station_locations(self, df):
        """
        Forward-fill station locations.
        Station persists until new station name appears.
        """
        self.log("\nFilling station locations...")
        
        if 'Station_Location' not in df.columns:
            return df
        
        current_station = None
        filled = 0
        
        for idx in df.index:
            station = str(df.at[idx, 'Station_Location']).strip()
            
            if station and station not in ['', 'nan']:
                current_station = station
            elif current_station:
                df.at[idx, 'Station_Location'] = current_station
                filled += 1
        
        self.log(f"  Filled {filled} empty stations")
        return df
    
    def identify_aggregate_rows(self, df):
        """
        Identify aggregate rows based on:
        1. Multiple years (e.g., "54 55 56")
        2. Multiple sites (e.g., "1 2 3")
        """
        self.log("\nIdentifying aggregates...")
        
        df['Is_Year_Aggregate'] = False
        df['Is_Site_Aggregate'] = False
        
        year_agg = 0
        site_agg = 0
        
        for idx in df.index:
            # Check years
            if 'Years' in df.columns:
                years = str(df.at[idx, 'Years']).split()
                if len(years) > 1:
                    df.at[idx, 'Is_Year_Aggregate'] = True
                    year_agg += 1
            
            # Check sites
            if 'Site' in df.columns:
                sites = str(df.at[idx, 'Site']).split()
                if len(sites) > 1:
                    df.at[idx, 'Is_Site_Aggregate'] = True
                    site_agg += 1
        
        self.log(f"  Year aggregates: {year_agg}")
        self.log(f"  Site aggregates: {site_agg}")
        return df
    
    def fill_missing_years_smart(self, df):
        """
        Smart year filling based on aggregate patterns.
        For 15-column tables (no Site), operates without site grouping.
        
        Logic:
        - For 16-column tables: Group by station-site, use aggregate rows
        - For 15-column tables: Group by station only, use aggregate rows
        """
        self.log("\nSmart filling missing years...")
        
        # Check if we have required columns based on table type
        if self.table_type == '15col':
            required = ['Station_Location', 'Years', 'Is_Year_Aggregate']
        else:
            required = ['Station_Location', 'Site', 'Years', 'Is_Year_Aggregate']
        
        if not all(c in df.columns for c in required):
            self.log("  Skipping: Required columns not found")
            return df
        
        filled = 0
        
        if self.table_type == '15col':
            # 15-column table: No site grouping
            for station in df['Station_Location'].unique():
                if pd.isna(station):
                    continue
                    
                station_df = df[df['Station_Location'] == station]
                
                # Find year aggregate rows
                agg_rows = station_df[station_df['Is_Year_Aggregate'] == True]
                
                if len(agg_rows) == 0:
                    continue
                
                # Get expected years from aggregate
                expected_years = set()
                for _, agg_row in agg_rows.iterrows():
                    agg_years = str(agg_row['Years']).split()
                    expected_years.update(agg_years)
                
                # Get individual year rows (non-aggregates)
                individual_rows = station_df[station_df['Is_Year_Aggregate'] == False]
                
                present_years = set()
                blank_indices = []
                
                for idx in individual_rows.index:
                    years_str = str(df.at[idx, 'Years']).strip()
                    
                    if not years_str or years_str == '':
                        blank_indices.append(idx)
                    else:
                        years = years_str.split()
                        if len(years) == 1:
                            present_years.add(years[0])
                
                # Calculate missing years
                missing_years = expected_years - present_years
                
                # Only fill if counts match exactly
                if len(blank_indices) == len(missing_years) and len(missing_years) > 0:
                    sorted_blanks = sorted(blank_indices)
                    sorted_missing = sorted(missing_years)
                    
                    for idx, year in zip(sorted_blanks, sorted_missing):
                        df.at[idx, 'Years'] = year
                        filled += 1
        
        else:
            # 16-column table: Original logic with site grouping
            for station in df['Station_Location'].unique():
                if pd.isna(station):
                    continue
                    
                station_df = df[df['Station_Location'] == station]
                
                # Get unique sites for this station
                for site_val in station_df['Site'].unique():
                    if pd.isna(site_val) or site_val == '':
                        continue
                    
                    # Only process single sites, not aggregates
                    site_parts = str(site_val).split()
                    if len(site_parts) > 1:
                        continue
                        
                    site_df = station_df[station_df['Site'] == site_val]
                    
                    # Find year aggregate
                    agg_rows = site_df[site_df['Is_Year_Aggregate'] == True]
                    
                    if len(agg_rows) == 0:
                        continue
                    
                    # Get expected years from aggregate
                    expected_years = set()
                    for _, agg_row in agg_rows.iterrows():
                        agg_years = str(agg_row['Years']).split()
                        expected_years.update(agg_years)
                    
                    # Get individual year rows (non-aggregates)
                    individual_rows = site_df[site_df['Is_Year_Aggregate'] == False]
                    
                    present_years = set()
                    blank_indices = []
                    
                    for idx in individual_rows.index:
                        years_str = str(df.at[idx, 'Years']).strip()
                        
                        if not years_str or years_str == '':
                            blank_indices.append(idx)
                        else:
                            years = years_str.split()
                            if len(years) == 1:
                                present_years.add(years[0])
                    
                    # Calculate missing years
                    missing_years = expected_years - present_years
                    
                    # Only fill if counts match exactly
                    if len(blank_indices) == len(missing_years) and len(missing_years) > 0:
                        sorted_blanks = sorted(blank_indices)
                        sorted_missing = sorted(missing_years)
                        
                        for idx, year in zip(sorted_blanks, sorted_missing):
                            df.at[idx, 'Years'] = year
                            filled += 1
        
        self.log(f"  Total years filled: {filled}")
        return df
    
    def clean_numeric_value(self, val):
        """Clean and convert numeric value."""
        if pd.isna(val) or val == '':
            return np.nan
        
        val_str = str(val).strip()
        
        # Remove non-numeric except dots and dashes
        val_str = re.sub(r'[^0-9\.\-]', '', val_str)
        
        if not val_str or val_str == '-':
            return np.nan
        
        try:
            num = float(val_str)
            # Remove values > 5 digits (OCR errors)
            if abs(num) >= 100000:
                return np.nan
            return num
        except:
            return np.nan
    
    def clean_numeric_columns(self, df):
        """Clean all numeric columns."""
        self.log("\nCleaning numeric columns...")
        
        numeric_cols = ['Num_Samples', 'Min', 'Max', 'Avg',
                       'P10', 'P20', 'P30', 'P40', 'P50', 
                       'P60', 'P70', 'P80', 'P90']
        
        cells_cleaned = 0
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            for idx in df.index:
                original = df.at[idx, col]
                cleaned = self.clean_numeric_value(original)
                df.at[idx, col] = cleaned
                
                if pd.isna(cleaned) and not pd.isna(original):
                    cells_cleaned += 1
        
        self.log(f"  Cleaned {cells_cleaned} cells")
        return df
    
    def validate_min_avg_max(self, df):
        """
        Validate Min <= Avg <= Max ordering.
        Instead of removing rows, clear the offending cells.
        """
        self.log("\nValidating Min/Avg/Max...")
        
        required = ['Min', 'Avg', 'Max']
        if not all(c in df.columns for c in required):
            return df
        
        cells_cleared = 0
        
        for idx in df.index:
            min_val = df.at[idx, 'Min']
            avg_val = df.at[idx, 'Avg']
            max_val = df.at[idx, 'Max']
            
            # Skip if missing
            if pd.isna(min_val) or pd.isna(avg_val) or pd.isna(max_val):
                continue
            
            # Check violations and clear problematic cells
            violations = []
            
            if min_val > avg_val:
                violations.append('Min > Avg')
            if avg_val > max_val:
                violations.append('Avg > Max')
            if min_val > max_val:
                violations.append('Min > Max')
            
            if violations:
                # Clear all three if severely wrong
                if 'Min > Max' in violations:
                    df.at[idx, 'Min'] = np.nan
                    df.at[idx, 'Avg'] = np.nan
                    df.at[idx, 'Max'] = np.nan
                    cells_cleared += 3
                # Otherwise clear the problematic one
                elif 'Min > Avg' in violations:
                    df.at[idx, 'Min'] = np.nan
                    cells_cleared += 1
                elif 'Avg > Max' in violations:
                    df.at[idx, 'Max'] = np.nan
                    cells_cleared += 1
        
        self.log(f"  Cleared {cells_cleared} cells with ordering violations")
        return df
    
    def validate_percentile_ordering(self, df):
        """
        Validate P10 <= P20 <= ... <= P90.
        Clear cells that violate ordering instead of removing entire rows.
        """
        self.log("\nValidating percentile ordering...")
        
        p_cols = ['P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90']
        existing = [c for c in p_cols if c in df.columns]
        
        if len(existing) < 3:
            return df
        
        cells_cleared = 0
        
        for idx in df.index:
            values = [(col, df.at[idx, col]) for col in existing]
            
            # Find violations
            for i in range(len(values) - 1):
                col_i, val_i = values[i]
                col_j, val_j = values[i + 1]
                
                if not pd.isna(val_i) and not pd.isna(val_j):
                    if val_i > val_j:
                        # Clear the higher percentile that's wrong
                        df.at[idx, col_j] = np.nan
                        cells_cleared += 1
        
        self.log(f"  Cleared {cells_cleared} cells with percentile violations")
        return df
    
    def remove_header_noise(self, df):
        """
        Remove rows that are clearly header repetitions.
        Pattern: No station, but has percentage values like 10, 20, 30, 40...
        """
        self.log("\nRemoving header noise...")
        
        if 'Station_Location' not in df.columns:
            return df
        
        rows_removed = []
        
        for idx in df.index:
            station = str(df.at[idx, 'Station_Location']).strip()
            
            # No station name
            if not station or station == '' or station == 'nan':
                # Check if row has sequential percentage-like values
                p_cols = ['P10', 'P20', 'P30', 'P40', 'P50']
                vals = []
                
                for col in p_cols:
                    if col in df.columns:
                        val = df.at[idx, col]
                        if not pd.isna(val):
                            vals.append(val)
                
                # If has 3+ percentage values, likely header
                if len(vals) >= 3:
                    rows_removed.append(idx)
        
        if rows_removed:
            df = df.drop(rows_removed)
            self.log(f"  Removed {len(rows_removed)} header rows")
        
        return df.reset_index(drop=True)
    
    def remove_header_rows_from_pages(self, df):
        """
        Remove rows at the start of new pages that contain data but no station location.
        These are typically OCR errors that captured the table header percentages.
        """
        self.log("\nRemoving header rows from page boundaries...")
        
        if 'Page_Num' not in df.columns or 'Station_Location' not in df.columns:
            self.log("  Skipping: Required columns not found")
            return df
        
        rows_to_drop = []
        
        # Group by page
        for page_num in df['Page_Num'].unique():
            page_df = df[df['Page_Num'] == page_num]
            
            if len(page_df) == 0:
                continue
            
            # Check first row of the page
            first_idx = page_df.index[0]
            first_row = df.loc[first_idx]
            
            # If first row has no station location but has numeric data in other columns
            station = str(first_row.get('Station_Location', '')).strip()
            
            if not station or station == '' or station == 'nan':
                # Check if row has data in percentage columns
                has_data = False
                for col in ['Num_Samples', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90']:
                    if col in df.columns:
                        val = str(first_row.get(col, '')).strip()
                        if val and val != '' and val != 'nan':
                            has_data = True
                            break
                
                if has_data:
                    rows_to_drop.append(first_idx)
        
        if rows_to_drop:
            df = df.drop(rows_to_drop)
            self.log(f"  Removed {len(rows_to_drop)} header rows")
        else:
            self.log("  No header rows found to remove")
        
        return df.reset_index(drop=True)
    
    def finalize_types(self, df):
        """Convert to proper data types."""
        self.log("\nFinalizing data types...")
        
        # Numeric columns
        num_cols = ['Num_Samples', 'Min', 'Max', 'Avg',
                   'P10', 'P20', 'P30', 'P40', 'P50',
                   'P60', 'P70', 'P80', 'P90']
        
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Page number
        if 'Page_Num' in df.columns:
            df['Page_Num'] = pd.to_numeric(df['Page_Num'], errors='coerce').astype('Int64')
        
        # Confidence
        if 'Avg_Confidence' in df.columns:
            df['Avg_Confidence'] = pd.to_numeric(df['Avg_Confidence'], errors='coerce')
        
        self.log("  Types finalized")
        return df
    
    def calculate_data_completeness(self, df):
        """
        Calculate data completeness statistics excluding metadata columns.
        Returns: (filled_cells, total_cells, percentage)
        """
        # Identify data columns (exclude metadata)
        metadata_cols = ['Avg_Confidence', 'Page_Num', 'Is_Year_Aggregate', 'Is_Site_Aggregate']
        data_cols = [col for col in df.columns if col not in metadata_cols]
        
        if not data_cols:
            return 0, 0, 0.0
        
        # Count filled cells in data columns only
        data_subset = df[data_cols]
        total_cells = data_subset.size
        filled_cells = data_subset.count().sum()
        percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0.0
        
        return filled_cells, total_cells, percentage
    
    # =========================================================================
    # 14-COLUMN MONTHLY TABLE METHODS
    # =========================================================================

    def classify_station_row_type(self, text):
        """
        Classify a cell in the Station_Location column of a 14-col table.

        Returns one of:
          'Num_Samples'  - "NUMBER OF SAMPLES", "NO. OF SAMPLES", etc.
          'Maximum'      - "MAXIMUM", "MAX", etc.
          'Average'      - "AVERAGE", "AVG", etc.
          'Minimum'      - "MINIMUM", "MIN", etc.
          'Name'         - station name row (text that doesn't match any keyword)
          'Empty'        - blank / NaN

        The ordering in the raw data is always:
          [station name]
          [Num_Samples row]
          [Maximum row]
          [Average row]
          [Minimum row]
        """
        if pd.isna(text) or str(text).strip() in ('', 'nan'):
            return 'Empty'

        t = str(text).strip()
        for row_type, pattern in self.row_type_patterns.items():
            if pattern.search(t):
                return row_type

        return 'Name'

    def clean_station_location_14col(self, df):
        """
        Process the Station_Location column for 14-column monthly tables.

        Each location block in the OCR output has 5 rows:
          1. Station name  (e.g. "ELIZABETH N J")
          2. Number of samples label
          3. Maximum label
          4. Average label
          5. Minimum label

        This method:
          - Adds a 'Row_Type' column: Name | Num_Samples | Maximum | Average | Minimum
          - Forward-fills the station name so every data row knows which station it belongs to
          - Cleans up the Station_Location cell to contain only the name (not the label text)
          - Flags rows where the expected sequence is broken (reports them)
          - Tries to repair common OCR mis-classifications using positional context

        Reports every anomaly found in the sequence pattern.
        """
        self.log("\nCleaning Station_Location column for 14-col monthly table...")

        if 'Station_Location' not in df.columns:
            self.log("  Skipping: Station_Location column not found")
            return df

        # ── Step 1: raw classification ───────────────────────────────────────
        df['Row_Type'] = df['Station_Location'].apply(self.classify_station_row_type)

        # ── Step 2: sequence validation & repair ─────────────────────────────
        EXPECTED_SEQUENCE = ['Name', 'Num_Samples', 'Maximum', 'Average', 'Minimum']
        anomalies = 0
        repaired = 0

        # We walk through in windows of 5.  When we see a 'Name' row we expect
        # the next 4 rows to follow EXPECTED_SEQUENCE[1:].  Any deviation is
        # logged and, where unambiguous, repaired.

        indices = df.index.tolist()
        i = 0
        while i < len(indices):
            idx = indices[i]
            row_type = df.at[idx, 'Row_Type']

            if row_type == 'Name':
                # Check the next 4 rows
                block = [idx]
                for j in range(1, 5):
                    if i + j < len(indices):
                        block.append(indices[i + j])

                if len(block) < 5:
                    # Incomplete block at end of file – nothing to repair
                    i += len(block)
                    continue

                block_types = [df.at[b, 'Row_Type'] for b in block]
                station_name = str(df.at[idx, 'Station_Location']).strip()

                if block_types != EXPECTED_SEQUENCE:
                    anomalies += 1
                    self.log(
                        f"  [ANOMALY] Station '{station_name}' (rows {block}): "
                        f"expected {EXPECTED_SEQUENCE}, got {block_types}"
                    )

                    # ── Repair pass: re-assign by position if any row is misclassified
                    # Strategy: if a row's text is Empty or ambiguous for its expected
                    # position, force-assign the expected type.
                    for pos, (b_idx, expected_type) in enumerate(
                            zip(block[1:], EXPECTED_SEQUENCE[1:]), start=1):
                        actual_type = df.at[b_idx, 'Row_Type']

                        if actual_type != expected_type:
                            cell_text = str(df.at[b_idx, 'Station_Location']).strip()

                            # Only repair if the cell is empty or looks like a label
                            # (i.e. NOT a station name with real geographic text)
                            is_label_or_empty = (
                                actual_type in ('Empty', 'Num_Samples',
                                                'Maximum', 'Average', 'Minimum')
                                or not cell_text
                            )

                            if is_label_or_empty:
                                df.at[b_idx, 'Row_Type'] = expected_type
                                repaired += 1
                                self.log(
                                    f"    Repaired row {b_idx}: "
                                    f"'{actual_type}' → '{expected_type}'"
                                )

                i += 5  # move to next block
            else:
                # Not a Name row where one was expected – skip ahead one
                i += 1

        self.log(f"  Anomalies detected: {anomalies}")
        self.log(f"  Rows repaired: {repaired}")

        # ── Step 3: forward-fill station name ────────────────────────────────
        current_name = None
        filled = 0

        for idx in df.index:
            rt = df.at[idx, 'Row_Type']
            cell = str(df.at[idx, 'Station_Location']).strip()

            if rt == 'Name' and cell and cell != 'nan':
                current_name = cell
                # Normalise the name cell: uppercase, collapse spaces
                df.at[idx, 'Station_Location'] = ' '.join(cell.upper().split())
            elif rt in ('Num_Samples', 'Maximum', 'Average', 'Minimum'):
                # Replace the label text with the station name
                if current_name:
                    df.at[idx, 'Station_Location'] = ' '.join(current_name.upper().split())
                    filled += 1
                # Clear leftover label text (it is now captured in Row_Type)
            # 'Empty' rows keep whatever is there

        self.log(f"  Station names forward-filled into {filled} label rows")
        return df

    def clean_years_14col(self, df):
        """
        Clean the Years column for 14-column monthly tables.

        Valid years are 53-57.  The Years column is shared across all 5 rows
        in a station block, but typically only the *name row* carries the year
        value (e.g. '55 56').  We forward-fill within each station block so
        every row (Num_Samples, Max, Avg, Min) knows its year range.

        Also strips any non-numeric noise that the OCR may have introduced.
        """
        self.log("\nCleaning Years column for 14-col table...")

        if 'Years' not in df.columns:
            self.log("  Skipping: Years column not found")
            return df

        cleaned = 0
        blanked = 0

        for idx in df.index:
            raw = str(df.at[idx, 'Years']).strip()

            # Extract 2-digit tokens
            tokens = re.findall(r'\b\d{2}\b', raw)
            valid = sorted(set(t for t in tokens if t in self.valid_years))

            if valid:
                df.at[idx, 'Years'] = ' '.join(valid)
                cleaned += 1
            else:
                df.at[idx, 'Years'] = ''
                if raw and raw != 'nan':
                    blanked += 1
                    self.log(f"  [WARN] Row {idx}: invalid year value '{raw}' → blanked")

        # Forward-fill within station blocks so all 5 rows share the same year
        current_year = ''
        for idx in df.index:
            rt = df.at[idx, 'Row_Type'] if 'Row_Type' in df.columns else 'Unknown'
            yr = df.at[idx, 'Years']

            if rt == 'Name':
                current_year = yr  # reset on new station block
            elif rt in ('Num_Samples', 'Maximum', 'Average', 'Minimum'):
                if not yr or yr == '':
                    df.at[idx, 'Years'] = current_year

        self.log(f"  Valid year entries: {cleaned}, Blanked invalid: {blanked}")
        return df

    def clean_monthly_columns(self, df):
        """
        Clean the 12 monthly data columns (Jan-Dec) for 14-col tables.

        Rules:
          - Each cell must contain a non-negative integer (air-pollution reading
            in µg/m³).  Readings above 9999 are almost certainly OCR artefacts.
          - Any cell that cannot be resolved to a plain integer is cleared.
          - Non-numeric characters (letters, punctuation, stray symbols) are
            stripped before trying to parse; if stripping leaves nothing useful
            the cell is cleared with a warning.
        """
        self.log("\nCleaning monthly data columns (Jan-Dec)...")

        existing_months = [m for m in self.month_cols if m in df.columns]
        if not existing_months:
            self.log("  Skipping: No month columns found")
            return df

        cells_cleaned = 0
        cells_warned = 0

        for col in existing_months:
            for idx in df.index:
                raw = df.at[idx, col]

                if pd.isna(raw) or str(raw).strip() in ('', 'nan'):
                    continue  # already empty – leave as-is

                raw_str = str(raw).strip()

                # Strip anything that is not a digit or a decimal point or minus
                numeric_str = re.sub(r'[^0-9\.\-]', '', raw_str)

                if not numeric_str or numeric_str in ('-', '.'):
                    # Nothing numeric left – clear and warn
                    df.at[idx, col] = np.nan
                    cells_cleaned += 1
                    cells_warned += 1
                    self.log(
                        f"  [WARN] Row {idx} col {col}: "
                        f"non-numeric '{raw_str}' → cleared"
                    )
                    continue

                try:
                    val = float(numeric_str)
                except ValueError:
                    df.at[idx, col] = np.nan
                    cells_cleaned += 1
                    self.log(
                        f"  [WARN] Row {idx} col {col}: "
                        f"parse failure '{raw_str}' → cleared"
                    )
                    continue

                # Sanity-check range
                if val < 0:
                    df.at[idx, col] = np.nan
                    cells_cleaned += 1
                    self.log(
                        f"  [WARN] Row {idx} col {col}: "
                        f"negative value {val} → cleared"
                    )
                elif val >= 10000:
                    df.at[idx, col] = np.nan
                    cells_cleaned += 1
                    self.log(
                        f"  [WARN] Row {idx} col {col}: "
                        f"implausibly large value {val} (OCR artefact?) → cleared"
                    )
                else:
                    # Store as string – finalize_types_14col will cast to numeric later
                    df.at[idx, col] = str(val)

        self.log(
            f"  Monthly cells cleared: {cells_cleaned} "
            f"({cells_warned} with non-numeric text warnings)"
        )
        return df

    def validate_monthly_ordering_14col(self, df):
        """
        For each station + year combination, validate that across every month:
            Maximum >= Average >= Minimum

        The four row types within a block are:
            Num_Samples, Maximum, Average, Minimum

        If a violation is found the offending cell is cleared (set to NaN)
        and the anomaly is reported.
        """
        self.log("\nValidating Max >= Avg >= Min for each month (14-col)...")

        required_cols = ['Station_Location', 'Years', 'Row_Type']
        if not all(c in df.columns for c in required_cols):
            self.log("  Skipping: Required columns not found")
            return df

        existing_months = [m for m in self.month_cols if m in df.columns]
        if not existing_months:
            return df

        cells_cleared = 0

        for station in df['Station_Location'].dropna().unique():
            s_df = df[df['Station_Location'] == station]

            for year in s_df['Years'].dropna().unique():
                if not year or str(year).strip() == '':
                    continue

                block = s_df[s_df['Years'] == year]

                max_rows = block[block['Row_Type'] == 'Maximum']
                avg_rows = block[block['Row_Type'] == 'Average']
                min_rows = block[block['Row_Type'] == 'Minimum']

                if max_rows.empty or avg_rows.empty or min_rows.empty:
                    continue

                max_idx = max_rows.index[0]
                avg_idx = avg_rows.index[0]
                min_idx = min_rows.index[0]

                for col in existing_months:
                    max_val = df.at[max_idx, col]
                    avg_val = df.at[avg_idx, col]
                    min_val = df.at[min_idx, col]

                    # Convert to float for comparison
                    try:
                        max_f = float(max_val) if not pd.isna(max_val) else None
                        avg_f = float(avg_val) if not pd.isna(avg_val) else None
                        min_f = float(min_val) if not pd.isna(min_val) else None
                    except (ValueError, TypeError):
                        continue

                    if max_f is None or avg_f is None or min_f is None:
                        continue

                    violations = []
                    if min_f > max_f:
                        violations.append('Min > Max')
                    if avg_f > max_f:
                        violations.append('Avg > Max')
                    if min_f > avg_f:
                        violations.append('Min > Avg')

                    if violations:
                        self.log(
                            f"  [VIOLATION] Station '{station}' Year '{year}' "
                            f"Month {col}: {violations} "
                            f"(Max={max_f}, Avg={avg_f}, Min={min_f})"
                        )
                        if 'Min > Max' in violations:
                            # Severe – clear all three
                            df.at[max_idx, col] = np.nan
                            df.at[avg_idx, col] = np.nan
                            df.at[min_idx, col] = np.nan
                            cells_cleared += 3
                        elif 'Avg > Max' in violations:
                            df.at[max_idx, col] = np.nan
                            cells_cleared += 1
                        elif 'Min > Avg' in violations:
                            df.at[min_idx, col] = np.nan
                            cells_cleared += 1

        self.log(f"  Cells cleared due to ordering violations: {cells_cleared}")
        return df

    def finalize_types_14col(self, df):
        """Convert 14-col table columns to proper data types."""
        self.log("\nFinalizing data types (14-col)...")

        for col in self.month_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Page_Num' in df.columns:
            df['Page_Num'] = pd.to_numeric(df['Page_Num'], errors='coerce').astype('Int64')

        if 'Avg_Confidence' in df.columns:
            df['Avg_Confidence'] = pd.to_numeric(df['Avg_Confidence'], errors='coerce')

        self.log("  Types finalized")
        return df

    def run_14col(self, df, initial_rows, has_conf, initial_conf):
        """
        Cleaning pipeline for 14-column monthly tables.

        Separate from the standard run() pipeline because the table structure
        (5 sub-rows per station instead of 1 data-row per station-year) requires
        completely different logic.
        """
        self.log("\n[14-COL PIPELINE]")

        # ── 1. Clean Station_Location: classify row types, forward-fill names ─
        df = self.clean_station_location_14col(df)

        # ── 2. Clean Years column ─────────────────────────────────────────────
        df = self.clean_years_14col(df)

        # ── 3. Remove page header / footer noise ─────────────────────────────
        # (reuse existing helpers; they check for Station_Location/Page_Num)
        df = self.remove_header_rows_from_pages(df)
        df = self.remove_page_footer_rows(df)

        # ── 4. Clean monthly numeric columns ─────────────────────────────────
        df = self.clean_monthly_columns(df)

        # ── 5. Validate Max >= Avg >= Min per month ───────────────────────────
        df = self.validate_monthly_ordering_14col(df)

        # ── 6. Finalize types ─────────────────────────────────────────────────
        df = self.finalize_types_14col(df)

        return df

    def run(self, table_num: int = None):
        """Execute cleaning pipeline."""
        self.log("="*70)
        self.log("IMPROVED AIR POLLUTION TABLE CLEANER v2")
        self.log("="*70)
        
        # Load
        df = self.load_data()
        initial_rows = len(df)
        
        # Check confidence
        has_conf = 'Avg_Confidence' in df.columns
        initial_conf = None
        if has_conf:
            initial_conf = pd.to_numeric(df['Avg_Confidence'], errors='coerce').mean()
            self.log(f"Initial Avg Confidence: {initial_conf:.2f}")
        
        # Remove empty trailing columns FIRST
        df = self.remove_empty_trailing_columns(df)
        
        # Rename
        df = self.rename_columns(df)

        from TableCleaner_improvements import (
            fix_known_ocr_errors, normalise_aggregate_names,
            fix_table_footer_rows, attach_json_context, pivot_monthly_table,
            OCR_KNOWN_FIXES, STATE_TOTAL_MAP
        )
        df = fix_known_ocr_errors(self, df)
        df = normalise_aggregate_names(self, df)
        df = fix_table_footer_rows(self, df)

        df, station_report = apply_station_resolution(df, "master_stations.csv", table_num=1)       

        # Remove Col_0 noise
        df = self.remove_column_zero(df)

        # ── Dispatch to the correct pipeline ─────────────────────────────────
        if self.table_type == '14col':
            df = self.run_14col(df, initial_rows, has_conf, initial_conf)
        else:
            # Original pipeline for 15-col and 16-col tables
            # Remove page-start header junk
            df = self.remove_header_rows_from_pages(df)

            # Remove page-end footer junk
            df = self.remove_page_footer_rows(df)
                    
            # Clean station names
            df = self.clean_station_names(df)
            
            # Extract sites
            df = self.extract_site_numbers(df)
            
            # Clean years
            df = self.clean_years(df)
            
            # Fill stations
            df = self.fill_station_locations(df)
            
            # Identify aggregates
            df = self.identify_aggregate_rows(df)
            
            # Fill missing years
            df = self.fill_missing_years_smart(df)
            
            # Clean numeric data
            df = self.clean_numeric_columns(df)
            
            # Remove header noise
            df = self.remove_header_noise(df)
            
            # Validate ordering
            df = self.validate_min_avg_max(df)
            df = self.validate_percentile_ordering(df)
            
            # Finalize types
            df = self.finalize_types(df)

            # ── Station resolution against master list ────────────────────────
            if self.master_stations_path and os.path.exists(self.master_stations_path):
                self.log("\nRunning station name resolution...")
                df, station_report = apply_station_resolution(
                    df,
                    master_csv_path=self.master_stations_path,
                    name_col='Station_Location',
                    site_col='Site' if self.table_type == '16col' else None,
                    table_num=self.table_num,
                )
                self.log(station_report)
            else:
                self.log("\nSkipping station resolution (no master_stations.csv provided)")

        # ── Common summary & save ─────────────────────────────────────────────
        # Calculate data completeness (excluding metadata)
        filled_cells, total_cells, fill_percentage = self.calculate_data_completeness(df)
        
        # Final stats
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        if has_conf and initial_conf is not None and 'Avg_Confidence' in df.columns:
            final_conf = df['Avg_Confidence'].mean()
            conf_change = final_conf - initial_conf
            
            self.log("")
            self.log("Confidence Statistics:")
            self.log(f"  Initial: {initial_conf:.2f}")
            self.log(f"  Final: {final_conf:.2f}")
            self.log(f"  Change: {conf_change:+.2f}")
        
        self.log("")
        self.log("="*70)
        self.log("SUMMARY")
        self.log("="*70)
        self.log(f"Data Completeness (excluding metadata):")
        self.log(f"  Filled Cells: {filled_cells:,} / {total_cells:,} ({fill_percentage:.1f}%)")
        self.log(f"Input rows: {initial_rows}")
        self.log(f"Output rows: {final_rows}")
        self.log(f"Removed: {removed} ({removed/initial_rows*100:.1f}%)")

        df = attach_json_context(self, df, table_num)   # adds table_name, units, group
        if 'Row_Type' in df.columns:                    # Table 4 pivot
            df = pivot_monthly_table(df)

        # Save
        df.to_csv(self.output_csv, index=False)
        self.log(f"\nSaved: {self.output_csv}")
        
        # Save report
        report_file = self.output_csv.replace('_cleaned.csv', '_cleaning_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(self.report_lines))
        self.log(f"Report: {report_file}")
        self.log("="*70)
        
        return df


def batch_clean_tables(table_directory: str, output_dir: str = None, column_count: int = 16):
    """
    Batch clean all tables with the same column structure.
    
    Args:
        table_directory: Directory containing table CSVs
        output_dir: Where to save cleaned tables
        column_count: Only process tables with this many data columns
                      (14 = monthly, 15 = percentile no-site, 16 = percentile with-site)
    """
    print("="*70)
    print("BATCH TABLE CLEANER")
    print("="*70)
    print(f"Directory: {table_directory}")
    print(f"Target column count: {column_count}")
    if column_count == 14:
        print(f"Note: Processing 14-column monthly tables (Station | Years | Jan-Dec)")
    else:
        print(f"Note: Handles both 16-column (with Site) and 15-column (without Site) tables")
    print("="*70)
    
    # Find all table CSV files
    table_files = glob.glob(os.path.join(table_directory, "Table_*.csv"))
    
    if not table_files:
        print("No table files found!")
        return
    
    print(f"\nFound {len(table_files)} table files")
    
    # Filter by column count
    matching_tables = []
    
    for table_file in table_files:
        try:
            # Quick column count check
            df_test = pd.read_csv(table_file, nrows=1)
            
            # Count data columns (Col_1, Col_2, etc., excluding metadata)
            data_cols = [col for col in df_test.columns 
                        if col.startswith('Col_') and col not in ['Avg_Confidence', 'Page_Num']]
            
            if len(data_cols) == column_count:
                matching_tables.append(table_file)
                print(f"  ✓ {os.path.basename(table_file)}: {len(data_cols)} columns (match)")
            else:
                print(f"  ✗ {os.path.basename(table_file)}: {len(data_cols)} columns (skip)")
        except Exception as e:
            print(f"  ⚠ {os.path.basename(table_file)}: Error reading ({e})")
    
    if not matching_tables:
        print(f"\n⚠ No tables found with {column_count} columns!")
        return
    
    print(f"\n{len(matching_tables)} table(s) match the criteria")
    print("="*70)
    
    # Process each matching table
    results = []
    
    for i, table_file in enumerate(matching_tables, 1):
        print(f"\n[{i}/{len(matching_tables)}] Processing: {os.path.basename(table_file)}")
        print("-"*70)
        
        try:
            cleaner = ImprovedTableCleaner(table_file, output_dir, master_stations_path="master_stations.csv")
            cleaned_df = cleaner.run()
            results.append({
                'file': os.path.basename(table_file),
                'status': 'Success',
                'rows': len(cleaned_df)
            })
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(table_file)}: {e}")
            results.append({
                'file': os.path.basename(table_file),
                'status': f'Error: {e}',
                'rows': 0
            })
    
    # Summary
    print("\n" + "="*70)
    print("BATCH CLEANING SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'Success')
    total_rows = sum(r['rows'] for r in results)
    
    print(f"Total tables processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total rows cleaned: {total_rows:,}")
    
    print("\nPer-file results:")
    for result in results:
        status_icon = "✓" if result['status'] == 'Success' else "✗"
        print(f"  {status_icon} {result['file']}: {result['status']} ({result['rows']} rows)")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    print("\nIMPROVED AIR POLLUTION TABLE CLEANER v2")
    print("Optimized for 1950s historical data patterns\n")
    
    # Mode selection
    print("Select mode:")
    print("  1. Clean single table (auto-detect type)")
    print("  2. Batch clean tables (16-column tables with Site)")
    print("  3. Batch clean tables (15-column tables without Site)")
    print("  4. Batch clean tables (14-column monthly tables: Jan-Dec)")
    
    mode = input("\nEnter mode (1, 2, 3, or 4) [default=1]: ").strip()
    
    if mode in ('2', '3', '4'):
        # Batch mode
        table_dir = input("\nEnter tables directory [default=Processed/Tables]: ").strip()
        if not table_dir:
            table_dir = "Processed/Tables"
        
        if not os.path.exists(table_dir):
            print(f"Error: Directory not found: {table_dir}")
            sys.exit(1)
        
        output_dir = input("Enter output directory (optional, creates Cleaned_Tables in table dir): ").strip()
        if not output_dir:
            output_dir = None
        
        # Set column count based on mode
        col_count_map = {'2': 18, '3': 17, '4': 16}
        col_count = col_count_map[mode]
        label_map = {
            '2': "16-column tables (with Site column)",
            '3': "15-column tables (without Site column)",
            '4': "14-column monthly tables (Station | Years | Jan-Dec)",
        }
        print(f"\nProcessing {label_map[mode]}")
        
        batch_clean_tables(table_dir, output_dir, col_count)
    
    else:
        # Single file mode
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
        else:
            input_file = input("Enter CSV path: ").strip()
        
        if not input_file:
            print("Error: No file specified")
            sys.exit(1)
        
        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)
        
        # Run cleaner
        cleaner = ImprovedTableCleaner(input_file)
        result = cleaner.run()
        
        # Preview
        print("\n" + "="*70)
        print("PREVIEW (first 15 rows)")
        print("="*70)
        
        if cleaner.table_type == '14col':
            display_cols = ['Station_Location', 'Row_Type', 'Years',
                            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        else:
            display_cols = ['Station_Location', 'Site', 'Years', 'Num_Samples', 
                           'Min', 'Max', 'Avg', 'P50', 'P90']
        available = [c for c in display_cols if c in result.columns]
        
        print(result[available].head(15).to_string())
        print("="*70)