# !/usr/bin/env python3
"""
Robust Problem1 pipeline (updated)
- Better future name parsing (robust regex & normalization)
- Proper expiry computation (last Thursday of month)
- Merge using merge_asof to align CM/FUT1/FUT2 with tolerance
- Logging of row counts before/after joins
- Median + percentile bands in plots
- Stratified sampling for scatter to preserve DTE distribution
- Saves two-page PDF of figures and prints coverage stats

Usage:
    python assignment01_fixed.py

Assumptions:
- Input CSVs live under /mnt/data and end with ".data.csv"
- Each CSV contains timestamp-like column (named 'timestamp' or similar),
  instrument name column (named 'instrument' or 'name'), price and volume fields.
- If your column names differ, update COL_NAMES mapping below.
"""

import re
import glob
import os
from datetime import datetime, timedelta, date
from collections import defaultdict
import math
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------------------
# Configuration / constants
# ---------------------------
DATA_DIR = "./data"
CSV_GLOB = os.path.join(DATA_DIR, "*.data.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "problem1_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column mapping: change if your CSVs use different column names
COL_NAMES = {
    "timestamp": None,   # if None, will infer common time-like names
    "instrument": None,  # if None, will infer common names like 'symbol','name'
    "price": None,       # infer 'last' or 'price'
    "volume": None,      # infer 'volume' or 'qty'
}

# Merge tolerance for aligning instruments (use 60s)
MERGE_TOLERANCE = pd.Timedelta("60s")

# DTE filter range
MIN_DTE = 0
MAX_DTE = 40

# Sampling fraction for scatter (but stratified by dte bins)
SCATTER_SAMPLE_PER_BIN = 300  # cap per dte bin to avoid overplot

# Minimum samples per DTE bin to show percentile ribbons
MIN_SAMPLES_PER_BIN = 5

# ---------------------------
# Utilities
# ---------------------------

def infer_columns(df):
    """Infer timestamp, instrument, price, volume column names if not provided."""
    cols = [c.lower() for c in df.columns]
    mapping = {}
    # timestamp
    if COL_NAMES["timestamp"]:
        mapping["timestamp"] = COL_NAMES["timestamp"]
    else:
        for candidate in ["timestamp", "time", "datetime", "date"]:
            if candidate in cols:
                mapping["timestamp"] = df.columns[cols.index(candidate)]
                break
    # instrument
    if COL_NAMES["instrument"]:
        mapping["instrument"] = COL_NAMES["instrument"]
    else:
        for candidate in ["instrument", "symbol", "name", "contract"]:
            if candidate in cols:
                mapping["instrument"] = df.columns[cols.index(candidate)]
                break
    # price
    if COL_NAMES["price"]:
        mapping["price"] = COL_NAMES["price"]
    else:
        for candidate in ["last", "price", "close"]:
            if candidate in cols:
                mapping["price"] = df.columns[cols.index(candidate)]
                break
    # volume
    if COL_NAMES["volume"]:
        mapping["volume"] = COL_NAMES["volume"]
    else:
        for candidate in ["volume", "qty", "size", "trade_volume"]:
            if candidate in cols:
                mapping["volume"] = df.columns[cols.index(candidate)]
                break
    return mapping

# Robust future-name parser
# Accepts many variants, optional separators and case-insensitive
FUT_RE = re.compile(
    r"""
    (?P<root>[A-Za-z0-9]+)     # root symbol (letters / numbers)
    [\-_ ]*                    # optional separator
    (?P<yy>\d{2})              # two-digit year in many tickers or month code may be here
    (?P<mon>[A-Za-z]{3})?      # optional 3-letter month (some tickers embed it)
    .*?FUT\b                   # FUT marker (suffix)
    """,
    re.I | re.X,
)

# Alternative format like ROOTMMMYYFUT or ROOT_YYMMM_FUT etc.
FUT_RE_ALT = re.compile(r"(?P<root>[A-Za-z0-9]+).{0,3}(?P<mon>[A-Za-z]{3}).{0,3}(?P<yy>\d{2}).*FUT", re.I)

MONTH_ABBR_TO_NUM = {m.lower(): i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

def parse_future_name(name):
    """
    Parse instrument/future name into root, year, month (numerical), and contract_code.
    Returns dict with fields or None if parsing fails.
    """
    if not isinstance(name, str):
        return None
    s = name.strip()
    s_upper = s.upper()
    # try primary regex
    m = FUT_RE.search(s_upper)
    if m:
        root = m.group("root")
        yy = m.group("yy")
        mon = m.group("mon") or ""
        # if month missing, try to find 3-letter month inside original name (case-insensitive)
        if not mon:
            alt = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", s_upper)
            mon = alt.group(0) if alt else ""
        mon = mon.title()
        try:
            year_full = 2000 + int(yy)
        except Exception:
            year_full = None
        month_num = MONTH_ABBR_TO_NUM.get(mon.lower()) if mon else None
        return {"root": root, "year": year_full, "month": month_num, "contract": s}
    # try alt
    m2 = FUT_RE_ALT.search(s_upper)
    if m2:
        root = m2.group("root")
        mon = m2.group("mon").title()
        yy = m2.group("yy")
        try:
            year_full = 2000 + int(yy)
        except Exception:
            year_full = None
        month_num = MONTH_ABBR_TO_NUM.get(mon.lower())
        return {"root": root, "year": year_full, "month": month_num, "contract": s}
    # No FUT marker: maybe it's CM or plain symbol
    return {"root": s, "year": None, "month": None, "contract": s}

def last_thursday_of_month(year:int, month:int):
    """Return date of last Thursday of given year-month."""
    # start at first day of next month, go back
    if month == 12:
        nxt = date(year+1,1,1)
    else:
        nxt = date(year, month+1, 1)
    last_day = nxt - timedelta(days=1)
    # backtrack to Thursday
    offset = (last_day.weekday() - 3) % 7  # weekday: Mon=0 ... Sun=6, Thu=3
    return last_day - timedelta(days=offset)

def compute_expiry_from_year_month(year, month):
    if year is None or month is None:
        return None
    return datetime.combine(last_thursday_of_month(year, month), datetime.min.time())

# ---------------------------
# Read and canonicalize CSVs
# ---------------------------
def load_and_canonicalize(filepath):
    """
    Load a CSV and return canonical DataFrame with columns:
    ['timestamp','contract','root','price','volume','is_future','future_year','future_month','expiry']
    """
    df = pd.read_csv(filepath)
    if df.empty:
        return pd.DataFrame()
    mapping = infer_columns(df)
    # if fail to detect mapping, raise
    if "timestamp" not in mapping or "instrument" not in mapping or "price" not in mapping or "volume" not in mapping:
        # Try tolerant names by index positions
        possible = {}
        for k in mapping:
            if mapping.get(k) is None:
                # attempt fallbacks
                pass
        # proceed but may fail downstream
    ts_col = mapping["timestamp"]
    inst_col = mapping["instrument"]
    price_col = mapping["price"]
    vol_col = mapping["volume"]

    # Normalize timestamp parsing: try common formats
    # If timestamp column is numeric (epoch), coerce
    if np.issubdtype(df[ts_col].dtype, np.number):
        df['timestamp'] = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
    else:
        df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce', infer_datetime_format=True)
    # drop rows with NaT timestamp
    df = df.dropna(subset=['timestamp'])
    # Normalize instrument
    df['contract'] = df[inst_col].astype(str).str.strip()
    df['price'] = pd.to_numeric(df[price_col], errors='coerce')
    df['volume'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0).astype(float)
    # Parse contract names
    parsed = df['contract'].apply(parse_future_name)
    parsed_df = pd.DataFrame(parsed.tolist())
    for col in ['root','year','month','contract']:
        if col in parsed_df:
            df[col if col!='year' else 'future_year'] = parsed_df[col]
        else:
            df[col if col!='year' else 'future_year'] = None
    # Mark futures if 'FUT' appears in contract (case-insensitive)
    df['is_future'] = df['contract'].str.upper().str.contains('FUT')
    # compute expiry if we have year & month
    def calc_exp(row):
        y = row.get('future_year')
        m = row.get('month')
        if pd.isna(y) or pd.isna(m) or y is None or m is None:
            return pd.NaT
        return compute_expiry_from_year_month(int(y), int(m))
    df['expiry'] = df.apply(calc_exp, axis=1)
    # Create root symbol column fallback
    df['root_symbol'] = df['root'].fillna(df['contract'])
    # Keep minimal columns
    df = df[['timestamp','contract','root_symbol','price','volume','is_future','expiry','future_year','month']]
    return df

# ---------------------------
# Processing across files
# ---------------------------
def load_all_files(glob_pattern=CSV_GLOB, limit_files=None):
    """Loads, canonicalizes, and concatenates all CSVs found."""
    files = sorted(glob.glob(glob_pattern))
    if limit_files:
        files = files[:limit_files]
    frames = []
    for f in files:
        try:
            df = load_and_canonicalize(f)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    # ensure sorted
    full = full.sort_values('timestamp').reset_index(drop=True)
    return full

# ---------------------------
# Align instruments and compute spreads
# ---------------------------

def build_instrument_tables(full_df):
    """
    From full DF, create three tables per root_symbol:
      - cash (CM): is_future == False
      - futures: is_future == True (keep expiry to rank)
    Return dicts mapping root -> cm_df, futures_df
    """
    cm = full_df[~full_df['is_future']].copy()
    futures = full_df[full_df['is_future']].copy()
    # sanity: ensure timestamp sorted
    cm = cm.sort_values(['root_symbol','timestamp']).reset_index(drop=True)
    futures = futures.sort_values(['root_symbol','timestamp','expiry']).reset_index(drop=True)
    return cm, futures

def pick_fut1_fut2_per_timestamp(futures):
    """
    For each root_symbol and timestamp, pick FUT1 (nearest expiry >= now) and FUT2 (next expiry)
    Strategy:
      - For each root_symbol, for each timestamp, find the two futures with the smallest positive (expiry - timestamp),
        if none exist, fall back to nearest (could be previous month).
    We will do this by:
      - For each root_symbol, pivot futures by contract+expiry and forward-fill last observed price to canonical minute grid
      - Simpler and robust approach: for each root_symbol produce a futures table sorted by timestamp & expiry,
        then at each CM timestamp we will select nearest-in-time future rows via merge_asof and then rank by expiry.
    """
    # This function will not be used directly; selection happens during merge stage using expiry-based ranking.
    return

def align_and_compute_spreads(cm_df, fut_df, tolerance=MERGE_TOLERANCE):
    """
    For each root_symbol:
      - merge_asof cm with fut rows to find nearest FUT1 & FUT2 by timestamp, using expiry ranking
      - compute cm_fut1, fut1_fut2 spreads and volume ratios
    Returns aligned DataFrame with one row per CM timestamp (where a FUT1 match exists)
    """
    outputs = []
    roots = sorted(set(cm_df['root_symbol'].unique()) | set(fut_df['root_symbol'].unique()))
    stats = defaultdict(int)
    for root in roots:
        cm_r = cm_df[cm_df['root_symbol']==root].copy()
        fut_r = fut_df[fut_df['root_symbol']==root].copy()
        if cm_r.empty and fut_r.empty:
            continue
        # ensure sorted
        cm_r = cm_r.sort_values('timestamp').reset_index(drop=True)
        fut_r = fut_r.sort_values('timestamp').reset_index(drop=True)
        # If fut_r has expiry info, rank contracts by expiry to determine which is near/far at any point.
        # Create a helper column 'exp_numeric' for ranking; NaT -> far past
        fut_r['exp_numeric'] = fut_r['expiry'].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
        # We'll perform merge_asof from cm -> futures but we need for each timestamp to identify which expiry is nearest in future.
        # Approach:
        # 1) For each unique expiry for this root, create a time-series of that contract's latest quoted price (forward fill).
        # 2) Create a wide table with columns for each expiry-contract price & volume.
        if fut_r.empty:
            # no futures for this root -> skip
            continue
        # identify unique contract+expiry pairs
        contracts = fut_r[['contract','expiry']].drop_duplicates().sort_values('expiry')
        # build per-contract time series
        per_contract_series = []
        for _, row in contracts.iterrows():
            contract_name = row['contract']
            expiry_val = row['expiry']
            sub = fut_r[fut_r['contract']==contract_name][['timestamp','price','volume']].copy()
            if sub.empty:
                continue
            sub = sub.sort_values('timestamp').set_index('timestamp')
            # resample to 1min grid covering CM timestamps to allow forward-fill
            # We'll reindex later; keep as is and use merge_asof per contract
            sub = sub.reset_index().rename(columns={'price':f'price__{contract_name}','volume':f'vol__{contract_name}'})
            # attach contract + expiry metadata
            sub['contract'] = contract_name
            sub['expiry'] = expiry_val
            per_contract_series.append(sub)
        if not per_contract_series:
            continue
        # Concatenate series and then for each CM timestamp we will pick the nearest-by-time row per contract via merge_asof
        fut_all = pd.concat(per_contract_series, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        # For merge_asof, we need one row per contract per timestamp; create a join loop over unique contracts
        # Build a DataFrame equal to cm_r repeated per contract, then merge_asof with fut rows filtered by contract.
        contract_names = sorted(fut_all['contract'].unique())
        # Build base DF by repeating cm_r for each contract
        base = pd.concat([cm_r.assign(__contract=cn) for cn in contract_names], ignore_index=True)
        # Prepare futures lookup table
        fut_all = fut_all.rename(columns={'contract':'__contract'})
        # merge_asof on timestamp joining base -> fut_all per contract grouping using by='__contract'
        base = base.sort_values('timestamp').reset_index(drop=True)
        fut_all = fut_all.sort_values('timestamp').reset_index(drop=True)
        merged = pd.merge_asof(base, fut_all,
                               left_on='timestamp', right_on='timestamp',
                               left_by='__contract', right_by='__contract',
                               direction='nearest', tolerance=tolerance)
        # merged now has, for each CM timestamp and contract, nearest future price & vol with that contract
        # drop rows where no match found
        matched_count = merged['price__' + contract_names[0]].notna().sum() if contract_names else 0
        # Now for each cm timestamp, choose FUT1 (contract with soonest expiry greater/equal to timestamp OR smallest positive time-to-expiry)
        # We'll compute time_to_expiry = (expiry - timestamp). If expiry missing, set large positive
        merged['time_to_expiry'] = (merged['expiry'] - merged['timestamp']).dt.total_seconds() / 86400.0
        # If time_to_expiry is negative (already expired), still keep but it will rank lower.
        # For each timestamp, pick the two contracts with smallest absolute time_to_expiry >= 0 if possible, else nearest positive.
        def choose_top2(group):
            g = group.copy()
            # prefer positive time_to_expiry (not yet expired); rank by time_to_expiry ascending
            g['pos_tte'] = g['time_to_expiry'].apply(lambda x: x if x >= 0 else np.inf)
            g_sorted = g.sort_values(['pos_tte','time_to_expiry']).reset_index(drop=True)
            # choose first two unique contracts with non-null price
            chosen = g_sorted[g_sorted['price__' + g_sorted['__contract'].iloc[0].split()[0].split('-')[0]].notna() if False else True]
            # simpler: take first two rows
            top = g_sorted.head(2)
            return top

        # Instead of the above slow group-based selection, we'll pivot merged into wide per timestamp and then select top2 by expiry.
        # Pivot: for each timestamp and __contract, keep price and vol and expiry
        pivot_price = merged.pivot_table(index=['timestamp','root_symbol','contract','__contract'], values='price', aggfunc='first')
        # The above pivot is messy given the chain; to simplify: for each timestamp, select the two rows with smallest pos_tte
        top_rows = merged.loc[merged.groupby(['timestamp']).apply(lambda g: g.nsmallest(2, ['pos_tte','time_to_expiry'])).index.get_level_values(1)]
        # top_rows now contains up to two rows per timestamp; we need to create a row per timestamp with columns for fut1/fut2 price/vol/expiry
        def make_row(group):
            group_sorted = group.sort_values(['pos_tte','time_to_expiry']).reset_index(drop=True)
            r = group_sorted.iloc[0] if len(group_sorted) > 0 else None
            r2 = group_sorted.iloc[1] if len(group_sorted) > 1 else None
            out = {}
            # base CM fields from first row in group
            cm_row = group_sorted.iloc[0]
            for c in ['timestamp','contract','root_symbol','price','volume']:
                out['cm_' + c] = cm_row.get(c)
            # FUT1
            if r is not None:
                out['fut1_contract'] = r.get('__contract')
                out['fut1_price'] = r.get('price__' + r.get('__contract')) if ('price__' + r.get('__contract')) in r.index else r.get('price')
                out['fut1_vol'] = r.get('vol__' + r.get('__contract')) if ('vol__' + r.get('__contract')) in r.index else r.get('volume')
                out['fut1_expiry'] = r.get('expiry')
                out['fut1_time_to_expiry'] = r.get('time_to_expiry')
            else:
                out.update({'fut1_contract':None,'fut1_price':np.nan,'fut1_vol':np.nan,'fut1_expiry':pd.NaT,'fut1_time_to_expiry':np.nan})
            # FUT2
            if r2 is not None:
                out['fut2_contract'] = r2.get('__contract')
                out['fut2_price'] = r2.get('price__' + r2.get('__contract')) if ('price__' + r2.get('__contract')) in r2.index else r2.get('price')
                out['fut2_vol'] = r2.get('vol__' + r2.get('__contract')) if ('vol__' + r2.get('__contract')) in r2.index else r2.get('volume')
                out['fut2_expiry'] = r2.get('expiry')
                out['fut2_time_to_expiry'] = r2.get('time_to_expiry')
            else:
                out.update({'fut2_contract':None,'fut2_price':np.nan,'fut2_vol':np.nan,'fut2_expiry':pd.NaT,'fut2_time_to_expiry':np.nan})
            return pd.Series(out)
        # Build grouped top_rows
        if top_rows.empty:
            continue
        grouped = top_rows.groupby('timestamp')
        assembled = grouped.apply(make_row)
        assembled = assembled.reset_index(drop=True)
        # Convert cm_timestamp to proper dtype
        assembled['timestamp'] = pd.to_datetime(assembled['cm_timestamp'])
        assembled['dte'] = (assembled['fut1_expiry'] - assembled['timestamp']).dt.total_seconds() / 86400.0
        # filter by dte range
        assembled = assembled[(assembled['dte']>=MIN_DTE) & (assembled['dte']<=MAX_DTE)]
        # spreads and ratios
        assembled['spread_cm_fut1'] = assembled['cm_price'] - assembled['fut1_price']
        assembled['spread_fut1_fut2'] = assembled['fut1_price'] - assembled['fut2_price']
        # volume ratios: handle zeros
        assembled['vol_ratio_cm_fut1'] = assembled['cm_volume'] / assembled['fut1_vol'].replace(0, np.nan)
        assembled['vol_ratio_fut1_fut2'] = assembled['fut1_vol'] / assembled['fut2_vol'].replace(0, np.nan)
        # attach root symbol
        assembled['root_symbol'] = root
        outputs.append(assembled)
        # stats
        stats['root_%s_rows' % root] = len(cm_r)
        stats['root_%s_matched' % root] = assembled.shape[0]
    if not outputs:
        return pd.DataFrame(), {}
    result = pd.concat(outputs, ignore_index=True)
    # Ensure numeric dtypes
    for c in ['spread_cm_fut1','spread_fut1_fut2','vol_ratio_cm_fut1','vol_ratio_fut1_fut2','dte']:
        if c in result:
            result[c] = pd.to_numeric(result[c], errors='coerce')
    return result, stats

# ---------------------------
# Aggregation & plotting
# ---------------------------

def stratified_sample(df, dte_col='dte', max_per_bin=SCATTER_SAMPLE_PER_BIN, bins=None):
    """Stratified sample by integer dte bins to preserve tail behaviour."""
    if bins is None:
        df['dte_bin'] = df[dte_col].fillna(0).astype(int).clip(0, MAX_DTE)
    else:
        df['dte_bin'] = pd.cut(df[dte_col], bins, labels=False)
    parts = []
    for b, g in df.groupby('dte_bin'):
        n = min(len(g), max_per_bin)
        parts.append(g.sample(n=n, random_state=42) if n>0 else g)
    sampled = pd.concat(parts).drop(columns=['dte_bin'])
    return sampled

def plot_problem1(result_df, out_pdf_path):
    """
    Create the multi-panel plots similar to user's original output but:
      - show median + 25/75 and 10/90 percentile ribbons
      - stratified sampling for scatter
    """
    # Prepare dte-binned aggregated stats
    # Use integer dte bin (0..MAX_DTE)
    result_df['dte_int'] = result_df['dte'].dropna().apply(lambda x: int(math.floor(x)) if pd.notnull(x) else -1)
    result_df = result_df[(result_df['dte_int']>=MIN_DTE)&(result_df['dte_int']<=MAX_DTE)]
    agg = result_df.groupby('dte_int').agg(
        med_spread_cm_fut1 = ('spread_cm_fut1', 'median'),
        mean_spread_cm_fut1 = ('spread_cm_fut1', 'mean'),
        p25_spread_cm_fut1 = ('spread_cm_fut1', lambda s: np.percentile(s.dropna(),25) if len(s.dropna())>0 else np.nan),
        p75_spread_cm_fut1 = ('spread_cm_fut1', lambda s: np.percentile(s.dropna(),75) if len(s.dropna())>0 else np.nan),
        p10_spread_cm_fut1 = ('spread_cm_fut1', lambda s: np.percentile(s.dropna(),10) if len(s.dropna())>0 else np.nan),
        p90_spread_cm_fut1 = ('spread_cm_fut1', lambda s: np.percentile(s.dropna(),90) if len(s.dropna())>0 else np.nan),
        med_roll = ('spread_fut1_fut2', 'median'),
        med_volratio_cm_fut1 = ('vol_ratio_cm_fut1', 'median'),
        med_volratio_fut1_fut2 = ('vol_ratio_fut1_fut2', 'median'),
        count = ('spread_cm_fut1', 'count')
    ).reset_index().rename(columns={'dte_int':'dte'}).sort_values('dte',ascending=False)

    # Create Figure
    plt.close('all')
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(4,1, figsize=(10, 16), sharex=True)
    # Top 1: scatter + median trend for CM-FUT1
    ax1 = axes[0]
    sampled = stratified_sample(result_df[['dte','spread_cm_fut1']].dropna())
    ax1.scatter(sampled['dte'], sampled['spread_cm_fut1'], alpha=0.12, s=8, color='grey', label='Raw Data')
    # Median trend
    ax1.plot(agg['dte'], agg['med_spread_cm_fut1'], color='red', linewidth=2, label='Median Trend')
    # Percentile ribbons
    ax1.fill_between(agg['dte'], agg['p25_spread_cm_fut1'], agg['p75_spread_cm_fut1'], color='red', alpha=0.15, label='25-75 pct')
    ax1.fill_between(agg['dte'], agg['p10_spread_cm_fut1'], agg['p90_spread_cm_fut1'], color='red', alpha=0.08, label='10-90 pct')
    ax1.axhline(0, color='k', linestyle='--', linewidth=1)
    ax1.set_title('Convergence: CM - FUT1 Spread')
    ax1.set_ylabel('spread_cm_fut1')
    ax1.invert_xaxis()

    # Top 2: FUT1-FUT2 median line (roll)
    ax2 = axes[1]
    ax2.plot(agg['dte'], agg['med_roll'], color='blue', linewidth=2)
    ax2.set_title('Roll Cost: FUT1 - FUT2 Spread')
    ax2.set_ylabel('spread_fut1_fut2')
    ax2.invert_xaxis()

    # Top 3: Liquidity ratio CM/FUT1
    ax3 = axes[2]
    ax3.plot(agg['dte'], agg['med_volratio_cm_fut1'], color='green', linewidth=2)
    ax3.set_title('Liquidity Ratio: CM / FUT1 Volume')
    ax3.set_ylabel('vol_ratio_cm_fut1')
    ax3.invert_xaxis()

    # Top 4: Rollover activity FUT1/FUT2
    ax4 = axes[3]
    ax4.plot(agg['dte'], agg['med_volratio_fut1_fut2'], color='purple', linewidth=2)
    ax4.set_title('Rollover Activity: FUT1 / FUT2 Volume')
    ax4.set_ylabel('vol_ratio_fut1_fut2')
    ax4.set_xlabel('dte')
    ax4.invert_xaxis()

    # Tighten layout and save
    plt.tight_layout()
    fig.savefig(out_pdf_path, dpi=200)
    plt.close(fig)
    return agg

# ---------------------------
# Main runner
# ---------------------------

def main():
    print("Scanning for CSV files:", CSV_GLOB)
    full = load_all_files()
    if full.empty:
        print("No data loaded. Ensure CSV files exist under", DATA_DIR)
        return
    print("Total rows loaded:", len(full))
    cm_df, fut_df = build_instrument_tables(full)
    print("CM rows:", len(cm_df), "Futures rows:", len(fut_df))
    result_df, stats = align_and_compute_spreads(cm_df, fut_df)
    if result_df is None or result_df.empty:
        print("No aligned result. Check input data / parsing.")
        return
    print("Aligned rows (after choosing fut1/fut2):", len(result_df))
    # Print per-root stats snippet
    print("Per-root matched counts summary (first 10):")
    printed = 0
    for k,v in stats.items():
        print(k, v)
        printed += 1
        if printed > 20:
            break
    # Basic validation metrics
    overall_cm_rows = cm_df.shape[0]
    overall_matched_rows = result_df.shape[0]
    coverage = 100.0 * overall_matched_rows / overall_cm_rows if overall_cm_rows>0 else np.nan
    print(f"Coverage: matched {overall_matched_rows} / cm rows {overall_cm_rows} = {coverage:.2f}%")
    # Aggregation + plot
    out_pdf = os.path.join(OUTPUT_DIR, "problem1_plots.pdf")
    agg = plot_problem1(result_df, out_pdf)
    print("Saved plot to:", out_pdf)
    # Save aggregated table & a small sample of aligned rows
    agg_csv = os.path.join(OUTPUT_DIR, "problem1_aggregated.csv")
    agg.to_csv(agg_csv, index=False)
    aligned_csv = os.path.join(OUTPUT_DIR, "problem1_aligned_sample.csv")
    result_df.sample(n=min(2000,len(result_df)), random_state=42).to_csv(aligned_csv, index=False)
    print("Saved aggregated and sample CSVs to", OUTPUT_DIR)
    # Print a few diagnostics: dte bins with low sample count
    low_bins = agg[agg['count'] < MIN_SAMPLES_PER_BIN]
    if not low_bins.empty:
        print("Warning: following DTE bins have low sample counts (< {}):".format(MIN_SAMPLES_PER_BIN))
        print(low_bins[['dte','count']].to_string(index=False))
    print("Done.")

if __name__ == "__main__":
    main()
