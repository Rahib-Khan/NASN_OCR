"""
context_utils.py
----------------
Lightweight utilities for loading and using historical_context.json and
codebook.json alongside the cleaned NASN table CSVs.

Designed to slot into the existing pipeline (TableCleaner.py, StationResolver.py)
without changing any of those files. Import what you need, ignore the rest.

Usage examples
--------------
    # 1. Get a table's name and units from its number
    from context_utils import get_table_info
    info = get_table_info(1)
    print(info['name'])   # Suspended Particulate Matter
    print(info['units'])  # ug/m3

    # 2. Get a column description
    from context_utils import get_column_info
    col = get_column_info('P50')
    print(col['description'])

    # 3. Attach a table_name and units column to a cleaned DataFrame
    from context_utils import attach_table_context
    df = attach_table_context(df, table_num=1)

    # 4. Write a JSON sidecar alongside a cleaned CSV
    from context_utils import write_table_sidecar
    write_table_sidecar('Cleaned_Tables/Table_1_Pages_1-8_cleaned.csv', table_num=1)

    # 5. Print a human-readable summary for a table
    from context_utils import print_table_summary
    print_table_summary(15)   # Lead - prints significance, units, filtering advice

    # 6. Load both JSONs in R:
    #    ctx  <- jsonlite::fromJSON('historical_context.json')
    #    book <- jsonlite::fromJSON('codebook.json')
"""

import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Paths  (relative to this file)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT_JSON_PATH  = os.path.join(_HERE, "historical_context.json")
CODEBOOK_JSON_PATH = os.path.join(_HERE, "codebook.json")


# ---------------------------------------------------------------------------
# JSON loading  (cached after first load)
# ---------------------------------------------------------------------------
_context_cache  = None
_codebook_cache = None


def load_context() -> dict:
    """Load and cache historical_context.json. Returns the full dict."""
    global _context_cache
    if _context_cache is None:
        if not os.path.exists(CONTEXT_JSON_PATH):
            raise FileNotFoundError(
                f"historical_context.json not found at {CONTEXT_JSON_PATH}\n"
                "Make sure it is in the same directory as context_utils.py"
            )
        with open(CONTEXT_JSON_PATH, encoding="utf-8") as f:
            _context_cache = json.load(f)
    return _context_cache


def load_codebook() -> dict:
    """Load and cache codebook.json. Returns the full dict."""
    global _codebook_cache
    if _codebook_cache is None:
        if not os.path.exists(CODEBOOK_JSON_PATH):
            raise FileNotFoundError(
                f"codebook.json not found at {CODEBOOK_JSON_PATH}\n"
                "Make sure it is in the same directory as context_utils.py"
            )
        with open(CODEBOOK_JSON_PATH, encoding="utf-8") as f:
            _codebook_cache = json.load(f)
    return _codebook_cache


# ---------------------------------------------------------------------------
# Table-level context
# ---------------------------------------------------------------------------

def get_table_info(table_num: int) -> dict:
    """
    Return the context dict for a specific table number (1-31).

    Keys: name, units, unit_label, pollutant_group, has_site, significance.

    Example:
        info = get_table_info(15)
        print(info['name'])          # Lead
        print(info['significance'])  # HISTORICALLY CRITICAL: leaded gasoline...
    """
    ctx = load_context()
    key = str(table_num)
    if key not in ctx.get("tables", {}):
        raise KeyError(f"Table {table_num} not found in historical_context.json")
    return ctx["tables"][key]


def get_unit_multiplier(table_num: int) -> float:
    """
    Return the multiplier needed to convert this table's concentration values
    to a common ug/m3 base unit for cross-table comparisons.

    Tables 1-3  -> 1.0    (already ug/m3)
    Tables 5-31 -> 0.001  (ug/m3 x10-3 = nanograms/m3)
    Table 4     -> 0.001  (radioactivity, different physical quantity)
    """
    info = get_table_info(table_num)
    units = info.get("units", "")
    ctx = load_context()
    for unit_key, unit_info in ctx.get("units", {}).items():
        if unit_key == units:
            return unit_info.get("conversion_to_ug_m3", 1.0)
    return 1.0


# ---------------------------------------------------------------------------
# Column-level context
# ---------------------------------------------------------------------------

def get_column_info(column_name: str) -> dict:
    """
    Return the codebook entry for a specific column name.

    Example:
        info = get_column_info('Match_Type')
        for v, desc in info['valid_values'].items():
            print(f"  {v}: {desc}")
    """
    book = load_codebook()
    columns = book.get("columns", {})
    if column_name not in columns:
        raise KeyError(
            f"Column '{column_name}' not in codebook.json. "
            f"Available: {sorted(columns.keys())}"
        )
    return columns[column_name]


def get_all_column_names() -> list:
    """Return sorted list of all column names documented in the codebook."""
    return sorted(load_codebook().get("columns", {}).keys())


# ---------------------------------------------------------------------------
# DataFrame enrichment  (drops into TableCleaner post-processing)
# ---------------------------------------------------------------------------

def attach_table_context(df, table_num: int, inplace: bool = True):
    """
    Add table_num, table_name, pollutant_units, and unit_multiplier columns
    to a cleaned DataFrame. Useful when combining tables or doing cross-table work.

    Does NOT change any existing column. Only adds columns that are not already present.

    Example:
        # At the end of TableCleaner.run():
        from context_utils import attach_table_context
        df = attach_table_context(df, table_num=1)
    """
    if not inplace:
        df = df.copy()

    try:
        info = get_table_info(table_num)
        mult = get_unit_multiplier(table_num)
    except (KeyError, FileNotFoundError):
        return df   # silently skip if JSON not found

    if "table_num"       not in df.columns: df["table_num"]       = table_num
    if "table_name"      not in df.columns: df["table_name"]      = info["name"]
    if "pollutant_units" not in df.columns: df["pollutant_units"]  = info["units"]
    if "unit_multiplier" not in df.columns: df["unit_multiplier"]  = mult
    if "pollutant_group" not in df.columns: df["pollutant_group"]  = info.get("pollutant_group", "")

    return df


# ---------------------------------------------------------------------------
# Sidecar JSON writer
# ---------------------------------------------------------------------------

def write_table_sidecar(csv_path: str, table_num: int) -> str:
    """
    Write a JSON sidecar file next to a cleaned CSV that embeds the full
    table context. The sidecar is named identically to the CSV but with
    a _context.json suffix.

    Example:
        write_table_sidecar('Cleaned_Tables/Table_1_Pages_1-8_cleaned.csv', 1)
        # Creates: Cleaned_Tables/Table_1_Pages_1-8_cleaned_context.json

    The sidecar contains:
      - table info (name, units, significance, etc.)
      - unit multiplier for cross-table normalisation
      - network summary (years, sampling method)
      - recommended filter strings
      - all column descriptions

    This means anyone opening the CSV has the context sitting right next
    to it without opening another file.
    """
    try:
        ctx  = load_context()
        book = load_codebook()
        info = get_table_info(table_num)
    except (KeyError, FileNotFoundError) as e:
        print(f"[context_utils] Could not write sidecar: {e}")
        return ""

    sidecar = {
        "csv_file":    os.path.basename(csv_path),
        "table_num":   table_num,
        "table_info":  info,
        "unit_multiplier": get_unit_multiplier(table_num),
        "cross_table_normalisation_note": (
            f"Multiply Min, Max, Avg, P10-P90 by {get_unit_multiplier(table_num)}"
            f" to convert to ug/m3 base unit for cross-table comparisons."
        ),
        "document": {
            "title":      ctx["_meta"]["document_title"] if "_meta" in ctx else "",
            "publisher":  ctx["_meta"].get("publisher", ""),
            "year":       ctx["_meta"].get("year_published", ""),
            "archive":    ctx["_meta"].get("archive_url", ""),
        },
        "data_years":   ctx.get("network", {}).get("data_years", []),
        "sampling_method_summary": (
            ctx.get("sampling", {}).get("equipment", "") + ". " +
            ctx.get("sampling", {}).get("filter_size", "") + ". " +
            ctx.get("sampling", {}).get("sample_duration_hours", "") + " hours."
        ),
        "statistical_note": ctx.get("statistical_context", {}).get("implication_for_analysis", ""),
        "recommended_filters": book.get("recommended_filters", {}),
        "columns": book.get("columns", {}),
    }

    out_path = csv_path.replace(".csv", "_context.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    print(f"[context_utils] Sidecar written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Human-readable summary printer
# ---------------------------------------------------------------------------

def print_table_summary(table_num: int) -> None:
    """
    Print a concise human-readable summary of a table's context.
    Useful for quick reference at the top of a notebook or analysis script.

    Example:
        print_table_summary(1)
        print_table_summary(15)  # Lead - prints historical significance
    """
    try:
        ctx  = load_context()
        info = get_table_info(table_num)
        book = load_codebook()
    except (KeyError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    net = ctx.get("network", {})
    filt = book.get("recommended_filters", {})

    print("=" * 65)
    print(f"TABLE {table_num}: {info['name'].upper()}")
    print("=" * 65)
    print(f"  Units:            {info['units']} ({info.get('unit_label', '')})")
    print(f"  Pollutant group:  {info.get('pollutant_group', 'N/A')}")
    print(f"  Has Site column:  {info.get('has_site', False)} (Table 1 only)")
    mult = get_unit_multiplier(table_num)
    if mult != 1.0:
        print(f"  Cross-table note: Multiply by {mult} to convert to ug/m3")
    print()
    print(f"  Significance:")
    sig = info.get('significance', 'No significance note.')
    for line in _wrap(sig, 60):
        print(f"    {line}")
    print()
    print(f"  Network years:    {net.get('data_years', [])}")
    print(f"  Stations:         ~{net.get('history_by_year',{}).get('1957',{}).get('urban_stations','?')} urban + {net.get('history_by_year',{}).get('1957',{}).get('nonurban_stations','?')} nonurban (1957)")
    print()
    print("  Recommended analysis filter:")
    print(f"    {filt.get('individual_station_year_analysis', 'See codebook.json')}")
    print()
    print("  Statistical note:")
    stat = ctx.get("statistical_context", {}).get("implication_for_analysis", "")
    for line in _wrap(stat, 60):
        print(f"    {line}")
    print("=" * 65)


def _wrap(text: str, width: int) -> list:
    """Simple word wrapper for print_table_summary."""
    words = text.split()
    lines = []
    current = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines


# ---------------------------------------------------------------------------
# Network-level helpers
# ---------------------------------------------------------------------------

def get_network_summary() -> dict:
    """Return just the network section of historical_context.json."""
    return load_context().get("network", {})


def get_errata() -> list:
    """Return list of known errata from the original publication."""
    return load_context().get("errata", {}).get("corrections", [])


def get_region_states() -> dict:
    """Return the NASN region -> list of states mapping."""
    return load_context().get("geographic_context", {}).get("regions", {})


def tables_by_pollutant_group() -> dict:
    """
    Return a dict mapping pollutant_group -> list of table numbers.
    Useful for looping over related tables.

    Example:
        groups = tables_by_pollutant_group()
        for tnum in groups['Trace Metals']:
            df = pd.read_csv(f'Cleaned_Tables/table_{tnum:02d}_cleaned.csv')
            ...
    """
    ctx = load_context()
    groups = {}
    for tnum_str, info in ctx.get("tables", {}).items():
        grp = info.get("pollutant_group", "Unknown")
        if grp not in groups:
            groups[grp] = []
        groups[grp].append(int(tnum_str))
    # Sort table numbers within each group
    for grp in groups:
        groups[grp].sort()
    return groups


# ---------------------------------------------------------------------------
# CLI: quick check that JSONs are valid and paired with a cleaned CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 65)
    print("context_utils — Diagnostic Mode")
    print("=" * 65 + "\n")

    # Validate both JSON files
    print("Checking historical_context.json ...")
    try:
        ctx = load_context()
        ntables = len(ctx.get("tables", {}))
        print(f"  OK  ({ntables} tables defined)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("Checking codebook.json ...")
    try:
        book = load_codebook()
        ncols = len(book.get("columns", {}))
        print(f"  OK  ({ncols} columns documented)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("Pollutant groups:")
    for grp, tables in tables_by_pollutant_group().items():
        print(f"  {grp}: tables {tables}")

    print()
    print("Enter a table number (1-31) to see its summary, or press Enter to quit:")
    while True:
        inp = input("  Table number: ").strip()
        if not inp:
            break
        try:
            print_table_summary(int(inp))
        except (ValueError, KeyError) as e:
            print(f"  Error: {e}")
