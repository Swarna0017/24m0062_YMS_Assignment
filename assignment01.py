import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from datetime import timedelta, datetime

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
DATA_DIR = './data/'  # Folder containing your 140 CSV files
OUTPUT_DIR = './output/'
DEBUG = True  # Set to False to silence detailed logs


os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# MODULE 1: UTILITIES (Helper Functions)
# =============================================================================
def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def get_expiry_date(date_obj):
    """
    Returns the last Thursday of the month for the given date's month.
    """
    # Go to the last day of the current month
    next_month = date_obj.replace(day=28) + timedelta(days=4)
    last_day_of_month = next_month - timedelta(days=next_month.day)
    
    # Calculate offset to last Thursday (Thursday is weekday 3)
    day_of_week = last_day_of_month.weekday()
    days_to_subtract = (day_of_week - 3) % 7
    return last_day_of_month - timedelta(days=days_to_subtract)

def parse_future_name(name):
    """
    Extracts (Root_Symbol, Expiry_Month_Year) from 'GAIL25FEBFUT'
    Returns: (GAIL, datetime_object_approx)
    """
    # Regex to capture Symbol, Year, Month. Example: GAIL 25 FEB FUT
    # This is a heuristic. Adjust based on exact naming convention if needed.
    match = re.match(r"([A-Z0-9]+)(\d{2})([A-Z]{3})FUT", name)
    if not match:
        return None, None
    
    symbol, year_str, month_str = match.groups()
    
    # Convert 'FEB' to month number
    try:
        month_dt = datetime.strptime(month_str, '%b')
        month_num = month_dt.month
        year_full = 2000 + int(year_str)
        
        # set expiry roughly to end of month for sorting purposes
        # The exact expiry calculation happens later using the logic rules
        approx_expiry = datetime(year_full, month_num, 1) + timedelta(days=25)
        return symbol, approx_expiry
    except Exception as e:
        return None, None

# =============================================================================
# MODULE 2: DATA INGESTION & PROCESSING
# =============================================================================
class DataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*.csv'))
        log(f"Found {len(self.files)} files in {data_dir}")

    def load_and_clean_file(self, filepath):
        """
        Reads a single CSV, handles bad lines, and standardizes columns.
        """
        try:
            # Load CSV. error_bad_lines=False (or on_bad_lines='skip') handles junk rows
            df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
            
            # Basic Column Check
            required_cols = ['time', 'exchange', 'name', 'ltp', 'total_trade_qty']
            if not all(col in df.columns for col in required_cols):
                log(f"Skipping {filepath}: Missing columns")
                return None

            # 1. Date Handling: Extract date from filename if not in row, or use logic
            # Assuming filename format YYYYMMDD.data.csv
            filename = os.path.basename(filepath)
            date_str = filename.split('.')[0] # '20250217'
            
            # 2. Add proper timestamp
            df['date_str'] = date_str
            df['timestamp'] = pd.to_datetime(df['date_str'] + ' ' + df['time'])

            # 3. Clean Numerics (Force Coerce)
            cols_to_numeric = ['ltp', 'total_trade_qty']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN in critical columns
            df.dropna(subset=['ltp'], inplace=True)
            
            return df
        
        except Exception as e:
            log(f"Error reading {filepath}: {e}")
            return None

    def pivot_data(self, df):
        """
        Transforms Long data to Wide data: 
        Row: [Time, CM_Price, FUT1_Price, FUT2_Price, FUT1_Vol, FUT2_Vol]
        """
        # Separate CM and FO
        df_cm = df[df['exchange'] == 'NSECM'].copy()
        df_fo = df[df['exchange'] == 'NSEFO'].copy()
        
        if df_cm.empty or df_fo.empty:
            return None

        # Process Futures to identify Near (FUT1) vs Far (FUT2)
        # We need to parse the 'name' column to get Expiry
        # Adding 'root_symbol' and 'expiry_sort_key'
        
        parsed_data = df_fo['name'].apply(parse_future_name)
        df_fo['root_symbol'] = [x[0] for x in parsed_data]
        df_fo['expiry_approx'] = [x[1] for x in parsed_data]
        
        # Filter out rows where parsing failed
        df_fo = df_fo.dropna(subset=['root_symbol', 'expiry_approx'])
        
        # Rank Futures by Expiry for each Symbol + Timestamp
        # Rank 1 = Near Month (FUT1), Rank 2 = Next Month (FUT2)
        df_fo['expiry_rank'] = df_fo.groupby(['timestamp', 'root_symbol'])['expiry_approx'].rank(method='dense')
        
        # Select only Rank 1 and Rank 2
        fut1 = df_fo[df_fo['expiry_rank'] == 1].rename(columns={'ltp': 'fut1_price', 'total_trade_qty': 'fut1_vol'})
        fut2 = df_fo[df_fo['expiry_rank'] == 2].rename(columns={'ltp': 'fut2_price', 'total_trade_qty': 'fut2_vol'})
        
        # Prepare CM data for merge
        df_cm = df_cm.rename(columns={'name': 'root_symbol', 'ltp': 'cm_price', 'total_trade_qty': 'cm_vol'})
        
        # MERGE: CM + FUT1
        merged = pd.merge(df_cm, fut1[['timestamp', 'root_symbol', 'fut1_price', 'fut1_vol', 'expiry_approx']], 
                          on=['timestamp', 'root_symbol'], how='inner')
        
        # MERGE: + FUT2
        merged = pd.merge(merged, fut2[['timestamp', 'root_symbol', 'fut2_price', 'fut2_vol']], 
                          on=['timestamp', 'root_symbol'], how='inner')
        
        return merged

    def run(self):
        all_data = []
        # Process first 5 files for testing (Remove [:5] to run all)
        files_to_process = self.files 
        
        for i, f in enumerate(files_to_process):
            if i % 10 == 0: log(f"Processing file {i}/{len(files_to_process)}...")
            
            raw_df = self.load_and_clean_file(f)
            if raw_df is not None:
                pivoted_df = self.pivot_data(raw_df)
                if pivoted_df is not None:
                    all_data.append(pivoted_df)
        
        if not all_data:
            log("No valid data found.")
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

# =============================================================================
# MODULE 3: ANALYSIS & PLOTTING
# =============================================================================
def analyze_and_plot(full_df):
    if full_df.empty:
        print("Dataframe is empty. Cannot plot.")
        return

    log("Calculating Metrics...")
    
    # 1. Calculate Expiry Date (Using the FUT1 expiry month logic)
    # The 'expiry_approx' column in FUT1 helps us know which month it is.
    # We apply the 'Last Thursday' rule to that month.
    full_df['expiry_date'] = full_df['expiry_approx'].apply(get_expiry_date)
    
    # 2. Days To Expiry (DTE)
    full_df['dte'] = (full_df['expiry_date'] - full_df['timestamp']).dt.days
    
    # Filter weird DTEs (negative or too far)
    full_df = full_df[(full_df['dte'] >= 0) & (full_df['dte'] < 40)]

    # 3. Calculate Spreads
    full_df['spread_cm_fut1'] = full_df['cm_price'] - full_df['fut1_price']
    full_df['spread_fut1_fut2'] = full_df['fut1_price'] - full_df['fut2_price']
    
    # 4. Calculate Volume Ratios (Avoid divide by zero)
    full_df['vol_ratio_cm_fut1'] = full_df['cm_vol'] / full_df['fut1_vol'].replace(0, np.nan)
    full_df['vol_ratio_fut1_fut2'] = full_df['fut1_vol'] / full_df['fut2_vol'].replace(0, np.nan)

    # 5. AGGREGATION (Crucial for clean plots)
    # We group by DTE and take the median spread across all stocks/times
    daily_stats = full_df.groupby('dte')[['spread_cm_fut1', 'spread_fut1_fut2', 
                                          'vol_ratio_cm_fut1', 'vol_ratio_fut1_fut2']].median().reset_index()

    log("Generating Plots...")
    
    # --- PLOT A: SPREADS ---
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 1, 1)
    sns.scatterplot(data=full_df.sample(frac=0.01), x='dte', y='spread_cm_fut1', alpha=0.1, color='gray', label='Raw Data') # Sample for background noise
    sns.lineplot(data=daily_stats, x='dte', y='spread_cm_fut1', color='red', linewidth=2, label='Median Trend')
    plt.title('Convergence: CM - FUT1 Spread')
    plt.gca().invert_xaxis()
    plt.axhline(0, color='black', linestyle='--')
    
    plt.subplot(2, 1, 2)
    sns.lineplot(data=daily_stats, x='dte', y='spread_fut1_fut2', color='blue', linewidth=2)
    plt.title('Roll Cost: FUT1 - FUT2 Spread')
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Problem1_A.pdf")
    log("Saved Problem1_A.pdf")
    
    # --- PLOT B: VOLUME RATIOS ---
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 1, 1)
    sns.lineplot(data=daily_stats, x='dte', y='vol_ratio_cm_fut1', color='green', linewidth=2)
    plt.title('Liquidity Ratio: CM / FUT1 Volume')
    plt.gca().invert_xaxis()
    
    plt.subplot(2, 1, 2)
    sns.lineplot(data=daily_stats, x='dte', y='vol_ratio_fut1_fut2', color='purple', linewidth=2)
    plt.title('Rollover Activity: FUT1 / FUT2 Volume')
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Problem1_B.pdf")
    log("Saved Problem1_B.pdf")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    log("Starting Processing Pipeline...")
    
    pipeline = DataPipeline(DATA_DIR)
    master_df = pipeline.run()
    
    if not master_df.empty:
        log(f"Processing Complete. Total Rows: {len(master_df)}")
        analyze_and_plot(master_df)
        log("Done.")
    else:
        log("Pipeline failed to produce data.")