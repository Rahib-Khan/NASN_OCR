"""
Enhanced Table Reconstruction System
Takes OCR results and rebuilds the original table structure using column/row metadata.

Main features:
- Groups pages by column count so tables that span multiple pages stay together
- Uses horizontal/vertical line positions to place text correctly
- Confidence threshold filtering (except for first column which is station names)
- Groups rows based on 25-pixel vertical threshold
- Tracks average confidence and page number for each row
"""

import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict


class TableReconstructor:
    """
    Rebuilds tables from OCR results using the structure metadata we saved during OCR.
    """
    
    def __init__(self, 
                 ocr_csv_path: str,
                 metadata_dir: str,
                 output_dir: str = "./Processed/Tables",
                 min_confidence: int = 60,
                 row_threshold: int = 25):
        """
        Set up the reconstructor.
        
        Args:
            ocr_csv_path: Path to combined_ocr_results.csv
            metadata_dir: Directory with page structure files
            output_dir: Where to save the reconstructed tables
            min_confidence: Minimum confidence for text (except column 0)
            row_threshold: How many pixels apart vertically before we consider it a new row
        """
        self.ocr_csv_path = ocr_csv_path
        self.metadata_dir = metadata_dir
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.row_threshold = row_threshold
        
        # Load the OCR data
        self.ocr_df = pd.read_csv(ocr_csv_path)
        self.page_structures = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_page_structure(self, page_num: int) -> Dict:
        """
        Load the column and row line positions for a specific page.
        
        Returns dict with:
            'column_lines': [x1, x2, ...],
            'row_lines': [y1, y2, ...],
            'num_columns': int
        """
        structure_file = os.path.join(
            self.metadata_dir, 
            f"page_{page_num:03d}_structure.txt"
        )
        
        if not os.path.exists(structure_file):
            print(f"  Warning: No structure file for page {page_num}")
            return {'column_lines': [], 'row_lines': [], 'num_columns': 1}
        
        column_lines = []
        row_lines = []
        
        with open(structure_file, 'r') as f:
            lines = f.readlines()
            
            in_col_section = False
            in_row_section = False
            
            for line in lines:
                line = line.strip()
                
                if 'Vertical lines' in line:
                    in_col_section = True
                    in_row_section = False
                    continue
                elif 'Horizontal lines' in line:
                    in_col_section = False
                    in_row_section = True
                    continue
                
                if in_col_section and 'col_line_' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        column_lines.append(int(parts[1].strip()))
                
                if in_row_section and 'row_line_' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        row_lines.append(int(parts[1].strip()))
        
        return {
            'column_lines': sorted(column_lines),
            'row_lines': sorted(row_lines),
            'num_columns': len(column_lines) + 1
        }
    
    def get_column_for_position(self, x_pos: int, column_lines: List[int]) -> int:
        """
        Figure out which column an x-position belongs to.
        Returns -1 if it's outside the boundaries.
        Column numbering starts at 1 (between first and second line).
        """
        if not column_lines:
            return -1
        
        # Left of first line
        if x_pos < column_lines[0]:
            return -1
        
        # Right of last line
        if x_pos > column_lines[-1]:
            return -1
        
        # Find which column (between lines)
        for i in range(len(column_lines) - 1):
            if column_lines[i] <= x_pos < column_lines[i + 1]:
                return i + 1  # Column 1, 2, 3, etc.
        
        return -1
    
    def is_in_bottom_rows(self, y_pos: int, row_lines: List[int]) -> bool:
        """
        Check if y-position is between the second-to-last and last horizontal lines.
        This helps us filter out header rows (above) and footer junk (below).
        """
        if len(row_lines) < 2:
            return True  # Can't filter if we don't have enough lines
        
        # Keep stuff between these two lines
        return row_lines[-2] < y_pos < row_lines[-1]
    
    def group_pages_by_structure(self) -> List[Dict]:
        """
        Group consecutive pages that have the same column structure.
        This keeps tables that span multiple pages together.
        
        Returns list of groups like:
            [{'pages': [1,2,3], 'num_columns': 5}, ...]
        """
        pages = sorted(self.ocr_df['page_num'].unique())
        
        groups = []
        current_group = None
        
        for page in pages:
            structure = self.load_page_structure(page)
            self.page_structures[page] = structure
            
            num_cols = structure['num_columns']
            
            if current_group is None or current_group['num_columns'] != num_cols:
                # New table group
                if current_group is not None:
                    groups.append(current_group)
                
                current_group = {
                    'pages': [page],
                    'num_columns': num_cols
                }
            else:
                # Add to current group
                current_group['pages'].append(page)
        
        # Don't forget the last group
        if current_group is not None:
            groups.append(current_group)
        
        return groups
    
    def reconstruct_table_group(self, group: Dict) -> pd.DataFrame:
        """
        Rebuild a table from a group of pages.
        
        Args:
            group: {'pages': [1,2,3], 'num_columns': 5}
        
        Returns:
            DataFrame with columns: Col_1, Col_2, ..., Avg_Confidence, Page_Num
        """
        pages = group['pages']
        num_columns = group['num_columns']
        
        print(f"\n  Reconstructing table from pages {pages[0]}-{pages[-1]} ({num_columns} columns)")
        
        # Column names (starting from Col_1 since Col_0 doesn't exist)
        col_names = [f"Col_{i}" for i in range(1, num_columns + 1)]
        
        all_rows = []
        
        for page in pages:
            print(f"    Processing page {page}...")
            
            page_data = self.ocr_df[self.ocr_df['page_num'] == page].copy()
            
            if page_data.empty:
                print(f"      Warning: No data for page {page}")
                continue
            
            structure = self.page_structures[page]
            column_lines = structure['column_lines']
            row_lines = structure['row_lines']
            
            # Filter by confidence (but always keep column 1 - the station names)
            filtered_data = []
            for _, item in page_data.iterrows():
                x_center = item['left'] + (item['width'] / 2)
                col_idx = self.get_column_for_position(x_center, column_lines)
                
                # Skip if outside boundaries
                if col_idx == -1:
                    continue
                
                # Column 1 (station names) gets a free pass on confidence
                if col_idx == 1:
                    filtered_data.append(item)
                elif item['conf'] >= self.min_confidence:
                    filtered_data.append(item)
            
            if not filtered_data:
                print(f"      No data passed confidence filter")
                continue
            
            filtered_df = pd.DataFrame(filtered_data)
            
            # Filter by vertical position (keep stuff above last horizontal line)
            vertically_filtered = []
            for _, item in filtered_df.iterrows():
                y_center = item['top'] + (item['height'] / 2)
                if self.is_in_bottom_rows(y_center, row_lines):
                    vertically_filtered.append(item)
            
            if not vertically_filtered:
                print(f"      No data above last horizontal line")
                continue
            
            valid_df = pd.DataFrame(vertically_filtered)
            
            # Sort top to bottom
            valid_df = valid_df.sort_values('top').reset_index(drop=True)
            
            # Group into rows
            rows = self.group_into_rows(valid_df, column_lines, num_columns, page)
            
            all_rows.extend(rows)
            print(f"      Extracted {len(rows)} rows")
        
        # Build the final table
        if not all_rows:
            print(f"    Warning: No rows extracted for this table group")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_rows)
        
        # Make sure all columns exist
        final_cols = col_names + ['Avg_Confidence', 'Page_Num']
        for col in final_cols:
            if col not in result_df.columns:
                result_df[col] = ''
        
        result_df = result_df[final_cols]
        
        return result_df
    
    def group_into_rows(self, 
                       page_data: pd.DataFrame, 
                       column_lines: List[int],
                       num_columns: int,
                       page_num: int) -> List[Dict]:
        """
        Group text elements into rows based on vertical position.
        
        Args:
            page_data: Filtered OCR data for this page
            column_lines: Column separator x-positions
            num_columns: Number of columns
            page_num: Current page number
        
        Returns:
            List of row dictionaries
        """
        rows = []
        
        if page_data.empty:
            return rows
        
        page_data = page_data.sort_values('top').reset_index(drop=True)
        
        # Track current row
        current_row = {f"Col_{i}": [] for i in range(1, num_columns + 1)}
        current_row['confidences'] = []
        current_row['page_num'] = page_num
        current_row_y = page_data.iloc[0]['top']
        
        for idx, item in page_data.iterrows():
            item_y = item['top']
            x_center = item['left'] + (item['width'] / 2)
            
            # Check if we've moved to a new row (more than threshold pixels down)
            if abs(item_y - current_row_y) > self.row_threshold:
                # Save current row if it has content
                if any(current_row[f"Col_{i}"] for i in range(1, num_columns + 1)):
                    rows.append(self.finalize_row(current_row, num_columns))
                
                # Start new row
                current_row = {f"Col_{i}": [] for i in range(1, num_columns + 1)}
                current_row['confidences'] = []
                current_row['page_num'] = page_num
                current_row_y = item_y
            
            # Figure out which column this goes in
            col_idx = self.get_column_for_position(x_center, column_lines)
            
            if col_idx == -1:
                continue
            
            if col_idx <= num_columns:
                # Store text with its x-position so we can sort left-to-right later
                current_row[f"Col_{col_idx}"].append({
                    'text': str(item['text']),
                    'left': item['left']
                })
                current_row['confidences'].append(item['conf'])
        
        # Don't forget the last row
        if any(current_row[f"Col_{i}"] for i in range(1, num_columns + 1)):
            rows.append(self.finalize_row(current_row, num_columns))
        
        return rows
    
    def finalize_row(self, row_dict: Dict, num_columns: int) -> Dict:
        """
        Convert accumulated row data into final format.
        Sorts text within each column left-to-right.
        
        Args:
            row_dict: Dict with Col_1: [{'text': 'foo', 'left': 100}], ...
        
        Returns:
            Finalized row with text and confidence
        """
        final_row = {}
        
        # Join text in each column (sorted left to right)
        for i in range(1, num_columns + 1):
            col_key = f"Col_{i}"
            if row_dict[col_key]:
                # Sort by x-position
                sorted_items = sorted(row_dict[col_key], key=lambda x: x['left'])
                texts = [item['text'] for item in sorted_items]
                final_row[col_key] = ' '.join(texts).strip()
            else:
                final_row[col_key] = ''
        
        # Calculate average confidence
        if row_dict['confidences']:
            final_row['Avg_Confidence'] = round(np.mean(row_dict['confidences']), 1)
        else:
            final_row['Avg_Confidence'] = 0.0
        
        final_row['Page_Num'] = row_dict['page_num']
        
        return final_row
    
    def calculate_table_stats(self, table_df: pd.DataFrame, num_columns: int, 
                            table_num: int, pages: List[int]) -> Dict:
        """
        Calculate how complete the data is for this table.
        
        Returns dict with stats like fill percentage, number of cells, etc.
        """
        num_rows = len(table_df)
        total_cells = num_rows * num_columns
        
        # Count filled cells (non-empty strings)
        filled_cells = 0
        for col_num in range(1, num_columns + 1):
            col_name = f"Col_{col_num}"
            if col_name in table_df.columns:
                filled_cells += (table_df[col_name].astype(str).str.strip() != '').sum()
        
        fill_percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        page_range = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
        
        return {
            'table_num': table_num,
            'page_range': page_range,
            'num_columns': num_columns,
            'num_rows': num_rows,
            'total_cells': total_cells,
            'filled_cells': filled_cells,
            'fill_percentage': fill_percentage
        }
    
    def save_table(self, table_df: pd.DataFrame, group_num: int, pages: List[int], num_columns: int):
        """
        Save a reconstructed table to CSV.
        Only includes the columns that actually exist for this table.
        """
        # Create column list for this specific table
        col_names = [f"Col_{i}" for i in range(1, num_columns + 1)]
        col_names.extend(['Avg_Confidence', 'Page_Num'])
        
        table_df = table_df[col_names]
        
        filename = f"Table_{group_num}_Pages_{pages[0]}-{pages[-1]}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        table_df.to_csv(filepath, index=False)
        print(f"  ✓ Saved: {filename} ({len(table_df)} rows, {num_columns} columns)")
        
        return filepath
    
    def reconstruct_all_tables(self):
        """
        Main method: reconstruct all tables and save them.
        """
        print("="*80)
        print("TABLE RECONSTRUCTION SYSTEM")
        print("="*80)
        print(f"OCR Data: {self.ocr_csv_path}")
        print(f"Metadata: {self.metadata_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Min Confidence: {self.min_confidence} (except Col_1)")
        print(f"Row Threshold: {self.row_threshold} pixels")
        print("="*80)
        
        # Group pages by structure
        print("\nGrouping pages by table structure...")
        groups = self.group_pages_by_structure()
        
        print(f"\nFound {len(groups)} table group(s):")
        for i, group in enumerate(groups, 1):
            print(f"  Group {i}: Pages {group['pages'][0]}-{group['pages'][-1]} ({group['num_columns']} columns, {len(group['pages'])} pages)")
        
        # Find max columns for the combined file
        max_columns = max(group['num_columns'] for group in groups)
        
        # Reconstruct each group
        all_tables = []
        saved_files = []
        table_stats = []
        
        for i, group in enumerate(groups, 1):
            print(f"\n{'='*80}")
            print(f"TABLE GROUP {i}")
            print(f"{'='*80}")
            
            table_df = self.reconstruct_table_group(group)
            
            if not table_df.empty:
                # Calculate stats
                stats = self.calculate_table_stats(table_df, group['num_columns'], i, group['pages'])
                table_stats.append(stats)
                
                # Save individual table
                filepath = self.save_table(table_df, i, group['pages'], group['num_columns'])
                saved_files.append(filepath)
                
                # Pad with extra columns for the combined file
                padded_df = table_df.copy()
                for col_num in range(group['num_columns'] + 1, max_columns + 1):
                    padded_df[f"Col_{col_num}"] = ''
                
                # Add separator row
                separator_dict = {f"Col_{j}": '' for j in range(1, max_columns + 1)}
                separator_dict['Col_1'] = f"=== TABLE {i}: Pages {group['pages'][0]}-{group['pages'][-1]} ==="
                separator_dict['Avg_Confidence'] = ''
                separator_dict['Page_Num'] = ''
                separator = pd.DataFrame([separator_dict])
                
                all_tables.append(separator)
                all_tables.append(padded_df)
            else:
                print(f"  ⚠ No data for this table group")
        
        # Create combined file
        if all_tables:
            print(f"\n{'='*80}")
            print("CREATING COMBINED TABLE FILE")
            print(f"{'='*80}")
            
            combined_df = pd.concat(all_tables, ignore_index=True)
            
            # Reorder columns
            combined_cols = [f"Col_{i}" for i in range(1, max_columns + 1)]
            combined_cols.extend(['Avg_Confidence', 'Page_Num'])
            combined_df = combined_df[combined_cols]
            
            combined_path = os.path.join(self.output_dir, "All_Tables_Combined.csv")
            combined_df.to_csv(combined_path, index=False)
            
            print(f"✓ Combined table saved: All_Tables_Combined.csv")
            print(f"  Total rows: {len(combined_df):,}")
            print(f"  Maximum columns: {max_columns}")
            
            # Print summary stats
            print(f"\n{'='*80}")
            print("RECONSTRUCTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total table groups: {len(groups)}")
            print(f"Total files saved: {len(saved_files) + 1}")
            
            # Data completeness table
            print(f"\nPER-TABLE DATA COMPLETENESS:")
            print(f"{'-'*80}")
            print(f"{'Table':<8} {'Pages':<12} {'Cols':<6} {'Rows':<6} {'Filled':<10} {'Total':<10} {'Fill %':<8}")
            print(f"{'-'*80}")
            
            total_filled = 0
            total_cells = 0
            
            for stats in table_stats:
                print(f"{stats['table_num']:<8} {stats['page_range']:<12} {stats['num_columns']:<6} "
                      f"{stats['num_rows']:<6} {stats['filled_cells']:<10} {stats['total_cells']:<10} "
                      f"{stats['fill_percentage']:<8.1f}%")
                total_filled += stats['filled_cells']
                total_cells += stats['total_cells']
            
            print(f"{'-'*80}")
            overall_fill = (total_filled / total_cells * 100) if total_cells > 0 else 0
            print(f"{'TOTAL':<8} {'':<12} {'':<6} {'':<6} {total_filled:<10} {total_cells:<10} {overall_fill:<8.1f}%")
            print(f"{'-'*80}")
            
            # Confidence stats
            data_rows = combined_df[combined_df['Avg_Confidence'] != '']
            if not data_rows.empty:
                avg_conf = data_rows['Avg_Confidence'].astype(float).mean()
                print(f"\nAverage row confidence: {avg_conf:.1f}")
            
            print(f"\nOutput directory: {self.output_dir}")
            print("="*80)
        else:
            print("\n⚠ No tables could be reconstructed")


def main():
    """
    Command-line interface for the table reconstructor.
    """
    print("\n" + "="*80)
    print("ENHANCED TABLE RECONSTRUCTION")
    print("Rebuilds tables from OCR results using structure metadata")
    print("="*80 + "\n")
    
    # Get paths
    ocr_csv = input("Enter path to combined_ocr_results.csv [Processed/combined_ocr_results.csv]: ").strip()
    if not ocr_csv:
        ocr_csv = "Processed/combined_ocr_results.csv"
    
    if not os.path.exists(ocr_csv):
        print(f"❌ Error: File not found: {ocr_csv}")
        return
    
    metadata_dir = input("Enter metadata directory [Processed/Column_Metadata]: ").strip()
    if not metadata_dir:
        metadata_dir = "Processed/Column_Metadata"
    
    if not os.path.exists(metadata_dir):
        print(f"❌ Error: Directory not found: {metadata_dir}")
        return
    
    output_dir = input("Enter output directory [Processed/Tables]: ").strip()
    if not output_dir:
        output_dir = "Processed/Tables"
    
    # Get parameters
    min_conf_input = input("Enter minimum confidence threshold [60]: ").strip()
    min_confidence = int(min_conf_input) if min_conf_input else 60
    
    row_threshold_input = input("Enter row grouping threshold in pixels [25]: ").strip()
    row_threshold = int(row_threshold_input) if row_threshold_input else 25
    
    # Run it
    reconstructor = TableReconstructor(
        ocr_csv_path=ocr_csv,
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        min_confidence=min_confidence,
        row_threshold=row_threshold
    )
    
    reconstructor.reconstruct_all_tables()
    
    print("\n✓ Table reconstruction complete!")
    print(f"Check the '{output_dir}' directory for results.\n")


if __name__ == "__main__":
    main()