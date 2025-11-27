
import pandas as pd
import numpy as np
import re
import glob
import os
from math import floor

# --- new knobs ---
ROLL_WINDOW_DAYS = 60
MIN_PERIODS = 120            # require some history before signals
TRIM_Q_LOW = 1.0             # keep central 99% => drop bottom 0.5% …
TRIM_Q_HIGH = 99.0           # … and top 0.5%
WINS_Q_LOW  = 1.0     # clip bottom 0.5%
WINS_Q_HIGH = 99.0    # clip top 0.5%
NO_TRADE_WARMUP_DAYS = 30    # don't trade for first 30 calendar days


# ---------------- Parsing helpers ----------------

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
FUT_REGEX = re.compile(r'^([A-Z0-9&.\-]+?)(\d{2})([A-Z]{3})FUT$')

from dataclasses import dataclass
from itertools import product

@dataclass(frozen=True)
class RelZParams:
    entry: float      # e.g., 1.5   (enter when |z| >= entry)
    tp_off: float     # e.g., 0.5   (exit if moves in favour by >= tp_off from entry)
    stop_off: float   # e.g., 2.0   (exit if moves against by >= stop_off from entry)

def decide_delta_relative(z: float, zmul: float, cost: float, spread_cost: float, lots: int, ready: bool, p: RelZParams) -> int:
    """
    Returns {-1,0,+1} delta lots.
    - Flat: enter when |z| >= p.entry  (short if z>=+E, long if z<=-E)
    - In position: exit 1 lot on either TP or SL, computed relative to the entry threshold.
      (No scale-ins; you still throttle to 1 lot/min elsewhere.)
    """

    _null = (0, 0)
    if not ready:
        return _null

    E, TP, SL = p.entry, p.tp_off, p.stop_off

    if lots > 0:  # long spread (entered at z <= -E)
        if z >= -E + TP:    # moved in favour by >= TP
            return -1, "TP"       # take profit
        elif z <= -E - SL:    # moved against by >= SL
            return -1, "SL"       # stop loss
        elif z <= -E and z >= -E - (SL/2):
            #if abs(TP * zmul) - cost < 1.0 * (2*cost + 2*spread_cost):
            if abs(TP * zmul) - cost < 1.0 * (6*cost):
                return _null
            return +1, "LONG"
       
        return _null

    if lots < 0:  # short spread (entered at z >= +E)
        if z <= +E - TP:    # moved in favour by >= TP
            return +1, "TP"       # take profit
        elif z >= +E + SL:    # moved against by >= SL
            return +1, "SL"      # stop loss
        elif z >= +E and z <= +E + (SL/2):
            #if abs(TP * zmul) - cost < 1.0 * (2*cost + 2*spread_cost):
            if abs(TP * zmul) - cost < 1.0 * (6*cost):
                return _null
            return -1, "SHRT"

        return _null

    # flat -> entry
    if z >= +E and z <= +E + (SL/2):
        #if abs(TP * zmul) - cost < 1.0*(2*cost + 2*spread_cost):
        if abs(TP * zmul) - cost < 1.0 * (6*cost):
            return _null
        return -1, "SHRT"  # enter short
    if z <= -E and z >= -E - (SL/2):
        #if abs(TP * zmul) - cost < 1.0*(2*cost + 2*spread_cost):
        if abs(TP * zmul) - cost < 1.0 * (6*cost):
            return _null
        return +1, "LONG"  # enter long
    return _null

def mark_fut1_expiry_eod(uni: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean column 'is_fut1_expiry_eod' to 'uni':
      True iff this row is the last timestamp of the last trading day
      observed for the current FUT1 contract (name_FUT1) for that underlying.
    Works purely from observed data (no calendar needed).
    """
    df = uni.copy()

    # Ensure we have a 'date' column
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.date

    # Last trading DATE for each (UNDERLYING, FUT1 contract name)
    last_date_per_contract = (
        df.groupby(['underlying', 'name_FUT1'], as_index=False)['date']
          .max()
          .rename(columns={'date': 'last_fut1_date'})
    )

    # Attach last date per current contract to each row
    df = df.merge(last_date_per_contract, on=['underlying', 'name_FUT1'], how='left')

    # Is this row on the last trading DATE for its current FUT1 contract?
    df['is_fut1_last_date'] = (df['date'] == df['last_fut1_date'])

    # EOD timestamp per (UNDERLYING, DATE)
    eod_ts_per_day = (
        df.groupby(['underlying', 'date'], as_index=False)['timestamp']
          .max()
          .rename(columns={'timestamp': 'eod_ts'})
    )
    df = df.merge(eod_ts_per_day, on=['underlying', 'date'], how='left')
    df['is_eod_row'] = (df['timestamp'] == df['eod_ts'])

    # Final flag: last day for this FUT1 contract AND end-of-day
    df['is_fut1_expiry_eod'] = df['is_fut1_last_date'] & df['is_eod_row']

    # Clean up helper columns if you like
    df = df.drop(columns=['eod_ts'])
    return df


def add_global_stats(uni: pd.DataFrame) -> pd.DataFrame:
    uni = uni.sort_values('timestamp')
    s = uni.set_index('timestamp')['spread']
    roll = s.rolling("60D", min_periods=120)

    uni['global_mean'] = roll.mean().values
    uni['global_sd']   = roll.std(ddof=0).values
    return uni.dropna(subset=['global_mean','global_sd']).reset_index(drop=True)

def parse_future_name(name: str):
    if not isinstance(name, str):
        return (None, None, None)
    m = FUT_REGEX.match(name.strip())
    if not m:
        return (None, None, None)
    underlying, yy, mon = m.groups()
    year = 2000 + int(yy)
    month = MONTHS.get(mon, None)
    return (underlying, year, month)

def load_one_file(path: str) -> pd.DataFrame:
    """
    Dual-Key Loader: 
    - Creates 'underlying' AND 'UNDERLYING' to match inconsistent script logic.
    - Creates 'cash_ltp' AND 'cm_price'.
    - Preserves all other fixes (mids, spreads, names).
    """
    try:
        # 1. Read Raw CSV
        df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
        
        # 2. Clean Columns
        df.columns = [c.strip().lower() for c in df.columns] 
        
        for col in ['ltp', 'total_trade_qty', 'mid']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        base_name = os.path.basename(path).split('.')[0]
        df['timestamp'] = pd.to_datetime(base_name + ' ' + df['time'])
        df['date'] = df['timestamp'].dt.normalize()
        
        # 3. Pivot Logic
        df_cm = df[df['exchange'] == 'NSECM'].copy()
        df_fo = df[df['exchange'] == 'NSEFO'].copy()
        
        if df_cm.empty or df_fo.empty:
            return pd.DataFrame()

        def get_expiry_sort_key(name):
            try:
                match = re.search(r"(\d{2})([A-Z]{3})FUT", name)
                if match:
                    yy, mon = match.groups()
                    mon_num = MONTHS.get(mon, 0)
                    return int(yy)*100 + mon_num 
                return 9999
            except:
                return 9999

        df_fo['expiry_key'] = df_fo['name'].apply(get_expiry_sort_key)
        df_fo['root_symbol'] = df_fo['name'].str.extract(r'^([A-Z0-9&]+)\d{2}[A-Z]{3}FUT')[0]
        df_fo = df_fo.dropna(subset=['root_symbol'])
        
        df_fo['rank'] = df_fo.groupby(['timestamp', 'root_symbol'])['expiry_key'].rank(method='dense')
        
        # --- FUT1 ---
        fut1 = df_fo[df_fo['rank'] == 1].rename(columns={
            'ltp': 'fut1_price', 
            'total_trade_qty': 'ttq_FUT1',
            'name': 'name_FUT1',
            'mid': 'mid_FUT1'
        })
        
        # --- FUT2 ---
        fut2 = df_fo[df_fo['rank'] == 2].rename(columns={
            'ltp': 'fut2_price', 
            'total_trade_qty': 'ttq_FUT2',
            'name': 'name_FUT2',
            'mid': 'mid_FUT2'
        })
        
        # --- CM ---
        df_cm = df_cm.rename(columns={
            'name': 'root_symbol', 
            'ltp': 'cash_ltp', 
            'total_trade_qty': 'ttq'
        })
        
        # 4. MERGE
        merged = pd.merge(df_cm, fut1[['timestamp', 'root_symbol', 'fut1_price', 'ttq_FUT1', 'name_FUT1', 'mid_FUT1']], 
                          on=['timestamp', 'root_symbol'], how='inner')
        
        merged = pd.merge(merged, fut2[['timestamp', 'root_symbol', 'fut2_price', 'ttq_FUT2', 'name_FUT2', 'mid_FUT2']], 
                          on=['timestamp', 'root_symbol'], how='left')
        
        # 5. CALCULATIONS
        merged['spread'] = merged['cash_ltp'] - merged['fut1_price']
        merged['spread_FUT1'] = merged['spread']
        merged['spread_FUT2'] = merged['fut1_price'] - merged['fut2_price']
        
        # FIX: Provide duplicates to satisfy all parts of the script
        merged['cm_price'] = merged['cash_ltp']
        merged['underlying'] = merged['root_symbol']  # For line 373
        merged['UNDERLYING'] = merged['root_symbol']  # For line 652
        
        # Return strict column list with ALL variants
        cols_to_return = [
            'date', 'timestamp', 'underlying', 'UNDERLYING', 'cash_ltp', 'cm_price',
            'spread', 'spread_FUT1', 'spread_FUT2', 
            'ttq', 'ttq_FUT1', 'ttq_FUT2',
            'name_FUT1', 'name_FUT2',
            'mid_FUT1', 'mid_FUT2'
        ]
        
        return merged[[c for c in cols_to_return if c in merged.columns]]

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

def prepare_spread_frame(df):
    """
    BYPASS: The spread is already calculated in load_one_file.
    Just return the dataframe as is.
    """
    return df

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# (Optional) avoid thread oversubscription inside each worker
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

def _process_one_file(path: str) -> pd.DataFrame:
    """Top-level worker: read one CSV and return the prepared per-file frame."""
    df = load_one_file(path)            # your existing function
    return prepare_spread_frame(df)     # your existing function

# 60 calendar-day rolling SMA/STD per underlying (time-based window)
def _rolling_winsor_stats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('timestamp')
    s = g.set_index('timestamp')['spread']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)

    def _wmean(a):
        if a.size == 0 or np.all(np.isnan(a)):
            return np.nan
        lo, hi = np.nanpercentile(a, [WINS_Q_LOW, WINS_Q_HIGH])
        b = np.clip(a, lo, hi)
        return np.nanmean(b)

    def _wstd(a):
        if a.size == 0 or np.all(np.isnan(a)):
            return np.nan
        lo, hi = np.nanpercentile(a, [WINS_Q_LOW, WINS_Q_HIGH])
        b = np.clip(a, lo, hi)
        return np.nanstd(b, ddof=0)

    def _pos_delta(a):
        # max(last - first, 0), skipping NaNs at edges
        if a.size < 2: return 0.0
        # trim NaNs from both ends
        i0 = 0
        while i0 < a.size and np.isnan(a[i0]): i0 += 1
        i1 = a.size - 1
        while i1 >= 0 and np.isnan(a[i1]): i1 -= 1
        if i1 <= i0: return 0.0
        d = a[i1] - a[i0]
        return float(d) if d > 0 else 0.0

    g['sma'] = roll.apply(_wmean, raw=True).values
    g['sd']  = roll.apply(_wstd,  raw=True).values

    s = g.set_index('timestamp')['spread_FUT1']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
    g['sma_spread_FUT1'] = roll.apply(_wmean, raw=True).values
    g['sd_spread_FUT1']  = roll.apply(_wstd,  raw=True).values

    s = g.set_index('timestamp')['spread_FUT2']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
    g['sma_spread_FUT2'] = roll.apply(_wmean, raw=True).values
    g['sd_spread_FUT2']  = roll.apply(_wstd,  raw=True).values


    # ---------- ttq deltas & spreadable volume ----------
    # Expect cumulative quantities: ttq_FUT1 / ttq_FUT2
    ttq1 = g.set_index('timestamp')['ttq_FUT1']
    ttq2 = g.set_index('timestamp')['ttq_FUT2']

    for mins in (5, 15, 30):
        win = f'{mins}T'
        # delta over window: max(last - first, 0)
        d1 = ttq1.rolling(win, min_periods=2).apply(_pos_delta, raw=True)
        d2 = ttq2.rolling(win, min_periods=2).apply(_pos_delta, raw=True)

        g[f'ttq_FUT1_{mins}'] = d1.values
        g[f'ttq_FUT2_{mins}'] = d2.values

        # spreadable volume = min of the two legs
        g[f'ttq_SPD_{mins}'] = np.minimum(g[f'ttq_FUT1_{mins}'], g[f'ttq_FUT2_{mins}'])

        # 60D winsorized stats on spreadable volume
        s_spd = g.set_index('timestamp')[f'ttq_SPD_{mins}']
        r_spd = s_spd.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
        g[f'ttq_SPD_{mins}_sma'] = r_spd.apply(_wmean, raw=True).values
        g[f'ttq_SPD_{mins}_sd']  = r_spd.apply(_wstd,  raw=True).values

    return g

def build_rolling_winsor_stats_per_underlying(uni: pd.DataFrame, ncores: int = 1, u_list: list = [], window: int = 1) -> pd.DataFrame:
    # Split once by underlying to avoid groupby inside workers
    groups = [g for under, g in uni.groupby('underlying') if ((not u_list) or (under in u_list)) ]

    if not groups:
        return uni

    results = []
    with ProcessPoolExecutor(max_workers=ncores) as ex:
        futures = {ex.submit(_rolling_winsor_stats, g): g for g in groups}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rolling (winsor) per underlying"):
            results.append(fut.result())

    uni_out = pd.concat(results, ignore_index=True)
    uni_out = uni_out.sort_values(['underlying','timestamp']).reset_index(drop=True)
    return uni_out


def build_universe(folder: str, ncores: int = 1, u_list: list = [], window: int = 1) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {folder}")

    frames = []

    with ProcessPoolExecutor(max_workers=ncores) as ex:
        # submit jobs
        futures = {ex.submit(_process_one_file, f): f for f in files}
        #for fut in as_completed(futures):
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building universe"):
            frames.append(fut.result())

    uni = pd.concat(frames, ignore_index=True)

    # Sort for rolling calcs
    uni = uni.sort_values(['underlying','timestamp']).reset_index(drop=True)

    uni = build_rolling_winsor_stats_per_underlying(uni, ncores, u_list)
    #uni = uni.groupby('underlying', group_keys=False).apply(_rolling_winsor_stats)

    uni['sd_spread_FUT1'] = uni['sd_spread_FUT1'].replace(0.0, np.nan)
    uni['sd_spread_FUT2'] = uni['sd_spread_FUT2'].replace(0.0, np.nan)
    uni['sd'] = uni['sd'].replace(0.0, np.nan)
    uni['z']  = (uni['spread'] - uni['sma']) / uni['sd']
    uni = uni.dropna(subset=['sma','sd','z', 'sma_spread_FUT1', 'sma_spread_FUT2', 'sd_spread_FUT1', 'sd_spread_FUT2']).reset_index(drop=True)
    
    return uni.dropna(subset=['sma','sd','z']).reset_index(drop=True)

# ---------------- Trading simulation ----------------

#def simulate_mean_reversion(uni: pd.DataFrame,
def simulate_mean_reversion(group: (str, pd.DataFrame),
                            zparams: RelZParams,
                            lot_notional_fixed: float = 800000.0,
                            max_lots: int = 40, _min: int = 15):
    """
    Robust simulate_mean_reversion:
      - Accepts either (under, df) tuple OR a DataFrame for a single underlying.
      - Returns (perf, summary, tlog) always (empty DataFrames allowed).
    """
    results = []
    trades = []
    shares_per_lot_dict = {}

    # Normalize input to a list of (under, g) pairs
    groups_list = []
    if isinstance(group, tuple) and len(group) == 2 and isinstance(group[1], pd.DataFrame):
        groups_list = [group]
    elif isinstance(group, pd.DataFrame):
        # If a full uni DataFrame passed, split by 'underlying'
        if 'underlying' in group.columns:
            groups_list = [(u, g.copy()) for u, g in group.groupby('underlying')]
        else:
            # Try uppercase variant
            if 'UNDERLYING' in group.columns:
                groups_list = [(u, g.copy()) for u, g in group.groupby('UNDERLYING')]
            else:
                # Nothing sensible to do
                raise ValueError("simulate_mean_reversion got a DataFrame without 'underlying'/'UNDERLYING' column")
    else:
        raise ValueError("simulate_mean_reversion expects either (under, DataFrame) or a DataFrame; got type: %s" % type(group))

    # Process each (under, g)
    for under, g in groups_list:
        # Ensure DataFrame has expected column names and dtypes
        g = g.sort_values('timestamp').copy()

        if 'date' not in g.columns and 'timestamp' in g.columns:
            g['date'] = g['timestamp'].dt.date

        # Warmup
        start_trade_ts = g['timestamp'].min() + pd.Timedelta(days=NO_TRADE_WARMUP_DAYS)
        start_time = pd.Timestamp('09:20').time()
        end_time = pd.Timestamp('15:20').time()

        lots = 0
        realized = 0.0
        unrealized = 0.0
        gross_pnl  = 0.0
        net_pnl = 0.0
        cost_pnl = 0.0
        own_amount = 0.0
        spr_slpg_pnl1 = 0.0
        spr_slpg_pnl2 = 0.0
        minutes_since_last_trade = 0

        rows = []
        tlog = []
        # cost per notional (kept same as before)
        cost = 0.01 * ((0.02 + (0.00173 + 0.00010) * 2 + 0.002) / 2)

        for _, row in g.iterrows():
            minutes_since_last_trade += 1
            ts   = row['timestamp']
            f1   = row.get('mid_FUT1', np.nan)
            s1   = row.get('spread_FUT1', np.nan)
            f2   = row.get('mid_FUT2', np.nan)
            s2   = row.get('spread_FUT2', np.nan)
            cash = row.get('cash_ltp', np.nan)
            z    = row.get('z', np.nan)
            sma  = row.get('sma', np.nan)
            sd   = row.get('sd', np.nan)
            gmean = row.get('global_mean', np.nan)
            gsd   = row.get('global_sd', np.nan)

            ttq_spd = row.get(f'ttq_SPD_{_min}', 0)
            ttq_spd_sma = row.get(f'ttq_SPD_{_min}_sma', np.nan)
            ttq_spd_sd  = row.get(f'ttq_SPD_{_min}_sd', np.nan)

            # determine shares per lot lazily
            try:
                shares_per_lot = shares_per_lot_dict[under]
                lot_notional = cash * shares_per_lot
            except Exception:
                lot_notional = lot_notional_fixed
                if pd.isna(cash) or cash == 0:
                    shares_per_lot = 1
                else:
                    shares_per_lot = max(1, int(floor(lot_notional / max(1e-6, cash))))
                shares_per_lot_dict[under] = shares_per_lot

            # compute ttq_limit
            if np.isnan(ttq_spd_sma) or np.isnan(ttq_spd):
                ttq_limit = 0
            else:
                ttq_limit = int(max(0, min(ttq_spd_sma, ttq_spd) * 0.25 / max(1, shares_per_lot)))

            # skip rows missing critical data
            if pd.isna(f1) or pd.isna(f2) or pd.isna(s1) or pd.isna(s2) or pd.isna(cash) or pd.isna(z) or pd.isna(sd):
                # still append an EOD row for bookkeeping but skip trading logic
                rows.append({
                    'timestamp': ts, 'date': row.get('date', pd.NaT),
                    'UNDERLYING': under, 'lots': int(lots), 'shares_per_lot': int(shares_per_lot),
                    'FUT1': row.get('name_FUT1', None), 'FUT2': row.get('name_FUT2', None),
                    'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash,
                    'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash,
                    'spread_norm': row.get('spread', np.nan), 'z': z, 'sma': sma, 'sd': sd,
                    'gsma': gmean, 'gsd': gsd,
                    'ttq_spd': ttq_spd, 'ttq_spd_sma': ttq_spd_sma,
                    'spr_raw': (f2 - f1) if (not pd.isna(f1) and not pd.isna(f2)) else np.nan,
                    'value': lots * shares_per_lot * (f2 - f1) if (not pd.isna(f1) and not pd.isna(f2)) else 0.0,
                    'realized': realized, 'unrealized': unrealized, 'equity': realized + unrealized,
                    'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'cost_pnl': cost_pnl,
                    'spr_slpg_pnl1': spr_slpg_pnl1, 'spr_slpg_pnl2': spr_slpg_pnl2, 'own_amount': own_amount,
                    'ttq_FUT2': row.get('ttq_FUT2', np.nan)
                })
                continue

            # compute variables used by strategy
            spr = (f2 - f1)
            zmul = sd * ((f1 + f2) / 2.0) * shares_per_lot
            cost_spr = (abs(s1) + abs(s2)) / 2.0
            curr_value = lots * shares_per_lot * (spr)

            # force close at FUT1 expiry EOD
            if bool(row.get('is_fut1_expiry_eod', False)) and lots != 0:
                delta = -lots
                lots_new = 0
                realized += curr_value
                own_amount += -delta * shares_per_lot * spr
                gross_pnl = own_amount + lots_new * shares_per_lot * spr
                cost_pnl += abs(delta * lot_notional) * cost * 2
                spr_slpg_pnl1 += abs(s1/2) * shares_per_lot * abs(delta)
                spr_slpg_pnl2 += abs(s2/2) * shares_per_lot * abs(delta)
                net_pnl = gross_pnl - cost_pnl
                minutes_since_last_trade = 0

                tlog.append({
                    'timestamp': ts, 'date': row.get('date', pd.NaT), 'UNDERLYING': under,
                    'action': 'FORCE_CLOSE_EXPIRY_EOD', 'tag': 'FORCE_CLOSE', 'delta_lots': int(delta),
                    'lots_after': int(lots_new), 'spr_price': spr, 'shares_per_lot': int(shares_per_lot),
                    'FUT1': row.get('name_FUT1', None), 'FUT2': row.get('name_FUT2', None),
                    'entry_E': zparams.entry, 'tp_off': zparams.tp_off, 'stop_off': zparams.stop_off
                })
                lots = lots_new
                curr_value = 0.0
                unrealized = 0.0
                equity = realized
                rows.append({
                    'timestamp': ts, 'date': row.get('date', pd.NaT), 'UNDERLYING': under, 'lots': int(lots),
                    'shares_per_lot': int(shares_per_lot), 'FUT1': row.get('name_FUT1', None),
                    'FUT2': row.get('name_FUT2', None), 'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash,
                    'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash, 'spread_norm': row.get('spread', np.nan),
                    'z': z, 'sma': sma, 'sd': sd, 'gsma': gmean, 'gsd': gsd, 'ttq_spd': ttq_spd,
                    'ttq_spd_sma': ttq_spd_sma, 'spr_raw': spr, 'value': curr_value,
                    'realized': realized, 'unrealized': unrealized, 'equity': equity,
                    'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'cost_pnl': cost_pnl, 'spr_slpg_pnl1': spr_slpg_pnl1,
                    'spr_slpg_pnl2': spr_slpg_pnl2, 'own_amount': own_amount, 'ttq_FUT2': row.get('ttq_FUT2', np.nan),
                    'entry_E': zparams.entry, 'tp_off': zparams.tp_off, 'stop_off': zparams.stop_off
                })
                continue

            # decide delta using existing logic
            ready = (ts >= start_trade_ts) and (ts.time() >= start_time and ts.time() <= end_time)
            if s1 <= 0 or s2 <= 0:
                # invalid spreads -> skip
                rows.append({
                    'timestamp': ts, 'date': row.get('date', pd.NaT), 'UNDERLYING': under, 'lots': int(lots),
                    'shares_per_lot': int(shares_per_lot), 'FUT1': row.get('name_FUT1', None),
                    'FUT2': row.get('name_FUT2', None), 'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash,
                    'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash, 'spread_norm': row.get('spread', np.nan),
                    'z': z, 'sma': sma, 'sd': sd, 'gsma': gmean, 'gsd': gsd, 'ttq_spd': ttq_spd,
                    'ttq_spd_sma': ttq_spd_sma, 'spr_raw': spr, 'value': curr_value,
                    'realized': realized, 'unrealized': unrealized, 'equity': realized + unrealized,
                    'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'cost_pnl': cost_pnl, 'spr_slpg_pnl1': spr_slpg_pnl1,
                    'spr_slpg_pnl2': spr_slpg_pnl2, 'own_amount': own_amount, 'ttq_FUT2': row.get('ttq_FUT2', np.nan)
                })
                continue

            delta, tag = decide_delta_relative(z, zmul, cost * lot_notional * 2, cost_spr * shares_per_lot, lots, ready, zparams)
            # decide_delta_relative sometimes returns tuple with tag; normalize
            if isinstance(delta, tuple):
                delta, tag = delta

            # throttle & bounds
            delta = int(max(-1, min(1, int(delta))))
            if lots + delta >  max_lots: delta = max_lots - lots
            if lots + delta < -max_lots: delta = -max_lots - lots

            # some exit logic (retain your existing gates)
            spr_norm = spr / max(1.0, cash)
            if (spr_norm < gmean - 2*gsd) and ready:
                delta = -lots
                tag = "EXIT_G_LIMIT"

            if (spr_norm < gmean - 1.75*gsd) and ready:
                if delta * lots >= 0:
                    delta = 0

            delta = int(max(-1, min(1, delta)))

            if delta != 0:
                if ttq_limit == 0 or minutes_since_last_trade < max(1, _min // max(1, ttq_limit)):
                    delta = 0

            if delta != 0:
                minutes_since_last_trade = 0
                trade_side = 'BUY_SPREAD' if delta > 0 else 'SELL_SPREAD'
                lots_new = lots + delta

                if np.sign(lots) != 0 and np.sign(lots) != np.sign(lots_new):
                    realized += curr_value
                elif abs(lots_new) < abs(lots):
                    realized += (abs(lots) - abs(lots_new)) * shares_per_lot * spr * np.sign(lots)

                own_amount += -delta * shares_per_lot * spr
                gross_pnl = own_amount + lots_new * shares_per_lot * spr
                cost_pnl += abs(delta * lot_notional) * cost * 2
                spr_slpg_pnl1 +=  abs(s1/2) * shares_per_lot * abs(delta)
                spr_slpg_pnl2 +=  abs(s2/2) * shares_per_lot * abs(delta)
                net_pnl = gross_pnl - cost_pnl

                tlog.append({
                    'timestamp': ts, 'date': row.get('date', pd.NaT), 'UNDERLYING': under,
                    'action': trade_side, 'tag': tag, 'delta_lots': int(delta), 'lots_after': int(lots_new),
                    'spr_price': spr, 'shares_per_lot': int(shares_per_lot), 'FUT1': row.get('name_FUT1', None),
                    'FUT2': row.get('name_FUT2', None), 'entry_E': zparams.entry, 'tp_off': zparams.tp_off,
                    'stop_off': zparams.stop_off
                })
                lots = lots_new
                curr_value = lots * shares_per_lot * spr

            unrealized = curr_value
            equity = realized + unrealized

            rows.append({
                'timestamp': ts, 'date': row.get('date', pd.NaT), 'UNDERLYING': under, 'lots': int(lots),
                'shares_per_lot': int(shares_per_lot), 'FUT1': row.get('name_FUT1', None), 'FUT2': row.get('name_FUT2', None),
                'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash, 'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash,
                'spread_norm': row.get('spread', np.nan), 'z': z, 'sma': sma, 'sd': sd, 'gsma': gmean, 'gsd': gsd,
                'ttq_spd': ttq_spd, 'ttq_spd_sma': ttq_spd_sma, 'spr_raw': spr, 'value': curr_value,
                'realized': realized, 'unrealized': unrealized, 'equity': equity,
                'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'cost_pnl': cost_pnl, 'spr_slpg_pnl1': spr_slpg_pnl1,
                'spr_slpg_pnl2': spr_slpg_pnl2, 'own_amount': own_amount, 'ttq_FUT2': row.get('ttq_FUT2', np.nan)
            })

        # append dataframes for this underlying
        if rows:
            results.append(pd.DataFrame(rows))
        else:
            results.append(pd.DataFrame(columns=[
                'timestamp','date','UNDERLYING','lots','shares_per_lot','FUT1','FUT2','ltp_FUT1','ltp_FUT2',
                'cash_ltp','mid_FUT1','mid_FUT2','cash_mid','spread_norm','z','sma','sd','gsma','gsd',
                'ttq_spd','ttq_spd_sma','spr_raw','value','realized','unrealized','equity','gross_pnl','net_pnl','cost_pnl'
            ]))
        trades.append(pd.DataFrame(tlog) if tlog else pd.DataFrame(columns=['timestamp','date','UNDERLYING','action','tag','delta_lots','lots_after','spr_price','shares_per_lot','FUT2','FUT1']))

    # combine
    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    perf = pd.concat(results, ignore_index=True)

    # Ensure UNDERLYING exists
    if 'UNDERLYING' not in perf.columns:
        if 'underlying' in perf.columns:
            perf['UNDERLYING'] = perf['underlying']
        elif 'root_symbol' in perf.columns:
            perf['UNDERLYING'] = perf['root_symbol']
        else:
            perf['UNDERLYING'] = 'UNKNOWN'

    # Sort safely
    if 'UNDERLYING' in perf.columns and 'timestamp' in perf.columns:
        perf = perf.sort_values(['UNDERLYING','timestamp']).reset_index(drop=True)
    else:
        perf = perf.sort_values('timestamp').reset_index(drop=True)

    tlog = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if not tlog.empty and 'UNDERLYING' not in tlog.columns:
        if 'underlying' in tlog.columns:
            tlog['UNDERLYING'] = tlog['underlying']
        else:
            tlog['UNDERLYING'] = 'UNKNOWN'

    # Build summary -- guard against missing columns
    # If perf is empty return empties
    if perf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # aggregation
    try:
        summary = (
            perf.groupby('UNDERLYING', as_index=False)
                .agg(
                    start=('timestamp', 'min'),
                    end=('timestamp', 'max'),
                    final_realized=('realized', 'last'),
                    final_unrealized=('unrealized', 'last'),
                    final_equity=('equity', 'last'),
                    final_gross_pnl=('gross_pnl', 'last'),
                    final_net_pnl=('net_pnl', 'last'),
                    final_cost_pnl=('cost_pnl', 'last'),
                    final_spr_slpg1=('spr_slpg_pnl1', 'last'),
                    final_spr_slpg2=('spr_slpg_pnl2', 'last'),
                    final_own_amount=('own_amount', 'last'),
                    max_drawdown=('net_pnl', lambda s: (s.cummax() - s).max())
                )
        )
    except Exception as e:
        # fallback to simple summary if aggregation fails
        summary = pd.DataFrame([{
            'UNDERLYING': perf['UNDERLYING'].iloc[0],
            'start': perf['timestamp'].min(),
            'end': perf['timestamp'].max(),
            'final_realized': perf['realized'].iloc[-1] if 'realized' in perf.columns else 0,
            'final_unrealized': perf['unrealized'].iloc[-1] if 'unrealized' in perf.columns else 0,
            'final_equity': perf['equity'].iloc[-1] if 'equity' in perf.columns else 0,
            'final_gross_pnl': perf['gross_pnl'].iloc[-1] if 'gross_pnl' in perf.columns else 0,
            'final_net_pnl': perf['net_pnl'].iloc[-1] if 'net_pnl' in perf.columns else 0,
            'final_cost_pnl': perf['cost_pnl'].iloc[-1] if 'cost_pnl' in perf.columns else 0,
            'final_spr_slpg1': perf.get('spr_slpg_pnl1', pd.Series([0])).iloc[-1],
            'final_spr_slpg2': perf.get('spr_slpg_pnl2', pd.Series([0])).iloc[-1],
            'final_own_amount': perf.get('own_amount', pd.Series([0])).iloc[-1],
            'max_drawdown': (perf['net_pnl'].cummax() - perf['net_pnl']).max() if 'net_pnl' in perf.columns else 0
        }])

    # compute trade_stats from tlog
    if not tlog.empty:
        trade_stats = (
            tlog.groupby('UNDERLYING', as_index=False)
                .agg(
                    total_contracts_traded=('delta_lots', lambda x: int(np.abs(x).sum())),
                    total_shares_traded=('delta_lots',
                        lambda x: int(np.abs((2*abs(tlog.loc[x.index, 'delta_lots']) *
                                              tlog.loc[x.index, 'shares_per_lot']).sum()))),
                    max_delta_lots=('lots_after', 'max'),
                    min_delta_lots=('lots_after', 'min')
                )
        )
    else:
        trade_stats = pd.DataFrame(columns=[
            'UNDERLYING','total_contracts_traded','total_shares_traded',
            'max_delta_lots','min_delta_lots'
        ])

    # merge
    try:
        summary = summary.merge(trade_stats, on='UNDERLYING', how='left').fillna({
            'total_contracts_traded': 0,
            'total_shares_traded': 0,
            'max_delta_lots': 0,
            'min_delta_lots': 0
        })
    except Exception:
        # If merge fails, just attach zeros
        summary['total_contracts_traded'] = 0
        summary['total_shares_traded'] = 0
        summary['max_delta_lots'] = 0
        summary['min_delta_lots'] = 0

    return perf, summary, tlog


# ---------------- New: Daily breakdown ----------------

def daily_breakdown(perf: pd.DataFrame, tlog: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day, per-underlying with carry-forward of unrealized into next day's realized:
      - daily_realized_cf: realized-to-realized P&L with carry-forward
      - eod_unrealized, eod_equity
      - daily_pnl_total = eod_equity - equity(prev day)
      - lots_traded, fut2_total_contracts, lots_traded_pct_of_fut2
    """
    # Base daily aggregates
    base = (
        perf.groupby(['UNDERLYING','date'], as_index=False)
            .agg(realized_start=('realized','first'),
                 realized_end=('realized','last'),
                 eod_unrealized=('unrealized','last'),
                 eod_equity=('equity','last'))
            .sort_values(['UNDERLYING','date'])
    )

    # Lag (yesterday) values per underlying
    base['eod_unrealized_prev'] = base.groupby('UNDERLYING')['eod_unrealized'].shift(1).fillna(0.0)
    base['eod_equity_prev']     = base.groupby('UNDERLYING')['eod_equity'].shift(1).fillna(0.0)

    # Carry-forward: treat yesterday's EOD unrealized as realized at today's open
    base['realized_start_cf'] = base['realized_start'] + base['eod_unrealized_prev']
    base['daily_realized_cf'] = base['realized_end'] - base['realized_start_cf']

    # Total daily MTM P&L (equity change)
    base['daily_pnl_total'] = base['eod_equity'] - base['eod_equity_prev']

    # Lots traded today
    lots_day = (
        tlog.groupby(['UNDERLYING','date'], as_index=False)
            .agg(lots_traded=('delta_lots', lambda x: int(np.abs(x).sum())))
    )

    # FUT2 EOD contracts (use daily max of cumulative ttq_FUT2)
    fut2_eod = (
        perf.groupby(['UNDERLYING','date'], as_index=False)
            .agg(fut2_total_contracts=('ttq_FUT2','max'))
    )

    daily = (base
             .merge(lots_day, on=['UNDERLYING','date'], how='left')
             .merge(fut2_eod, on=['UNDERLYING','date'], how='left'))

    daily['lots_traded'] = daily['lots_traded'].fillna(0).astype(int)
    daily['lots_traded_pct_of_fut2'] = np.where(
        (daily['fut2_total_contracts'] > 0) & (~daily['fut2_total_contracts'].isna()),
        100.0 * daily['lots_traded'] / daily['fut2_total_contracts'],
        np.nan
    )

    return daily[['UNDERLYING','date',
                  'daily_realized_cf','eod_unrealized','eod_equity','daily_pnl_total',
                  'lots_traded','fut2_total_contracts','lots_traded_pct_of_fut2']] \
           .sort_values(['UNDERLYING','date'])

# ---------------- Driver ----------------


def run_relz_grid(uni: pd.DataFrame,
                  entry_list=(1.0, 1.25, 1.5, 1.75, 2.0),
                  tp_off_list=(0.5, 0.75, 1.0),
                  stop_off_list=(1.5, 2.0, 2.5),
                  lot_notional=800000.0, max_lots=40,
                  save_dir=None, prefix="mr_spread"):
    all_perf, all_summary, all_trades = [], [], []

    for E, TP, SL in product(entry_list, tp_off_list, stop_off_list):
        zp = RelZParams(entry=E, tp_off=TP, stop_off=SL)
        perf, summary, tlog = simulate_mean_reversion(
            uni, zparams=zp, lot_notional_fixed=lot_notional, max_lots=max_lots
        )
        cfg = f"E{E}_TP{TP}_SL{SL}"
        for df in (perf, summary, tlog):
            df['cfg'] = cfg

        all_perf.append(perf)
        all_summary.append(summary)
        all_trades.append(tlog)

        if save_dir:
            #perf.to_csv(f"{save_dir}/{prefix}_perminute_{cfg}.csv", index=False)
            tlog.to_csv(f"{save_dir}/{prefix}_trades_{cfg}.csv", index=False)
            summary.to_csv(f"{save_dir}/{prefix}_summary_{cfg}.csv", index=False)

    perf_cat = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()
    tlog_cat = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summ_cat = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    if save_dir:
        #perf_cat.to_csv(f"{save_dir}/{prefix}_perminute_ALL.csv", index=False)
        tlog_cat.to_csv(f"{save_dir}/{prefix}_trades_ALL.csv", index=False)
        summ_cat.to_csv(f"{save_dir}/{prefix}_summary_ALL.csv", index=False)

    return perf_cat, summ_cat, tlog_cat

def run_sim(folder: str,
            out_prefix: str = "mr_spread",
            save_csv: bool = True):
    uni = build_universe(folder)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)
    perf, summary, trades = simulate_mean_reversion(uni, RelZParams(entry=1.5, tp_off=0.5, stop_off=1.5))

    # New daily breakdown
    daily = daily_breakdown(perf, trades)

    if save_csv:
        perf.to_csv(os.path.join(folder, f"{out_prefix}_perminute.csv"), index=False)
        summary.to_csv(os.path.join(folder, f"{out_prefix}_summary.csv"), index=False)
        trades.to_csv(os.path.join(folder, f"{out_prefix}_trades.csv"), index=False)
        daily.to_csv(os.path.join(folder, f"{out_prefix}_daily.csv"), index=False)

    return perf, summary, trades, daily

def run_sim_grid(folder: str,
            out_prefix: str = "mr_spread_rel",
            save_csv: bool = True):
    uni = build_universe(folder)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)

    perf_all, summary_all, trades_all = run_relz_grid(
        uni,
        entry_list=[1.0,1.25,1.5,1.75,2.0],
        tp_off_list=[0.25,0.5,0.75,1.0,1.25,1.5],
        stop_off_list=[0.75,1.0,1.25,1.5],
        save_dir="./results/",
        prefix=out_prefix
    )

    return perf_all, summary_all, trades_all

# Example:
#perf, summary, trades = run_sim_grid("./customdata")

# ==== CLI driver (single-config run) =========================================
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Mean-reversion spread sim (single zparam run).")
    parser.add_argument("--entry", type=float, required=True, help="Entry z threshold E (e.g., 1.5)")
    parser.add_argument("--tp_off", type=float, required=True, help="TP offset from entry (e.g., 0.5)")
    parser.add_argument("--stop_off", type=float, required=True, help="STOP offset from entry (e.g., 2.0)")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to input CSV folder")
    parser.add_argument("--result_folder", type=str, required=True, help="Folder to write result CSVs")
    # Optional knobs (keep defaults from your script):
    parser.add_argument("--lot_notional", type=float, default=800000.0)
    parser.add_argument("--max_lots", type=int, default=40)
    parser.add_argument("--prefix", type=str, default="mr_spread_rel")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--underlying", type=str, default="")
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.result_folder, exist_ok=True)

    # === Build universe (reuse your existing pipeline) ===
    # uni = build_universe(args.data_folder)
    # uni = add_underlying_ema_stats(uni)   # EMA halflife=5D per earlier patch
    # uni = add_global_ema_stats(uni)       # global EMA stats
    # uni = mark_fut1_expiry_eod(uni)       # expiry day EOD flag
    # NOTE: Keep your existing implementations; the above line-up is illustrative.

    # If your code already has a helper that builds uni, call it here:

    import datetime
    tp = datetime.datetime.now()
    uni = build_universe(args.data_folder, args.ncores, [args.underlying] if args.underlying else [])
    
    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "build_universe done")
    #uni = add_underlying_ema_stats(uni)
    #uni = add_global_ema_stats(uni)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)

    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "uni done")

    # === Sim run for this single zparam config ===
    zp = RelZParams(entry=args.entry, tp_off=args.tp_off, stop_off=args.stop_off)

    groups = [(under, g) for under, g in uni.groupby('underlying')]

    results = []
    with ProcessPoolExecutor(max_workers=args.ncores) as ex:
        futures = {ex.submit(simulate_mean_reversion, group=g, zparams=zp, lot_notional_fixed=args.lot_notional, max_lots=args.max_lots): g for g in groups}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Simulation Per Underlying {zp}"):
            results.append(fut.result())

    # --- concatenate outputs ---
    if results:
        perf   = pd.concat([r[0] for r in results if r is not None and len(r) >= 1], ignore_index=True)
        summary= pd.concat([r[1] for r in results if r is not None and len(r) >= 2], ignore_index=True)
        trades = pd.concat([r[2] for r in results if r is not None and len(r) >= 3], ignore_index=True)
        summary = summary.sort_values('final_net_pnl', ascending=False)

        # --- add TOTAL row ---
        total_row = {
            'UNDERLYING': 'TOTAL',
            'start': summary['start'].min(),
            'end': summary['end'].max(),
            'final_realized': summary['final_realized'].sum(),
            'final_unrealized': summary['final_unrealized'].sum(),
            'final_equity': summary['final_equity'].sum(),
            'final_gross_pnl': summary['final_gross_pnl'].sum(),
            'final_net_pnl': summary['final_net_pnl'].sum(),
            'final_cost_pnl': summary['final_cost_pnl'].sum(),
            'final_spr_slpg1': summary['final_spr_slpg1'].sum(),
            'final_spr_slpg2': summary['final_spr_slpg2'].sum(),
            'final_own_amount': summary['final_own_amount'].sum(),
            'max_drawdown': summary['max_drawdown'].max(),
            'total_contracts_traded': summary['total_contracts_traded'].sum(),
            'total_shares_traded': summary['total_shares_traded'].sum(),
            'max_delta_lots': summary['max_delta_lots'].max(),
            'min_delta_lots': summary['min_delta_lots'].min()
        }
        summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    else:
        perf = pd.DataFrame()
        summary = pd.DataFrame()
        trades = pd.DataFrame()

    #perf, summary, trades = simulate_mean_reversion(
    #    uni, zparams=zp, lot_notional_fixed=args.lot_notional, max_lots=args.max_lots
    #)

    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "simulation done")
    # stamp parameters (redundant if you already stamp inside simulate)
    for df in (perf, summary, trades):
        df["entry_E"] = args.entry
        df["tp_off"] = args.tp_off
        df["stop_off"] = args.stop_off

    # Add TOTAL row (you already built this earlier; if not, do it here)
    # -- if you already add TOTAL row in summary upstream, skip this block --
    if "UNDERLYING" in summary.columns and "TOTAL" not in summary["UNDERLYING"].values:
        total_row = {
            'UNDERLYING': 'TOTAL',
            'start': summary['start'].min() if 'start' in summary else pd.NaT,
            'end': summary['end'].max() if 'end' in summary else pd.NaT,
            'final_realized': summary.get('final_realized', pd.Series([0])).sum(),
            'final_unrealized': summary.get('final_unrealized', pd.Series([0])).sum(),
            'final_equity': summary.get('final_equity', pd.Series([0])).sum(),
            'max_drawdown': summary.get('max_drawdown', pd.Series([0])).max(),
            'total_contracts_traded': summary.get('total_contracts_traded', pd.Series([0])).sum(),
            'total_shares_traded': summary.get('total_shares_traded', pd.Series([0])).sum(),
            'max_delta_lots': summary.get('max_delta_lots', pd.Series([0])).max(),
            'min_delta_lots': summary.get('min_delta_lots', pd.Series([0])).min(),
            'entry_E': args.entry, 'tp_off': args.tp_off, 'stop_off': args.stop_off
        }
        summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    # === Save with parameterized filenames ===
    tag = f"E{args.entry}_TP{args.tp_off}_SL{args.stop_off}"
    perf.to_csv(os.path.join(args.result_folder, f"{args.prefix}_perminute_{tag}.csv"), index=False)
    trades.to_csv(os.path.join(args.result_folder, f"{args.prefix}_trades_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.result_folder, f"{args.prefix}_summary_{tag}.csv"), index=False)

    print(f"[OK] Saved results for {tag} in {args.result_folder}")

