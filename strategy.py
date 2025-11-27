import pandas as pd
import numpy as np

# =============================================================================
# STRATEGY SETTINGS
# =============================================================================
ROLLING_WINDOW = 60   # 1 Hour moving average
Z_ENTRY_THRESHOLD = 2 # Enter trade when Z > 2 or Z < -2
Z_EXIT_THRESHOLD = 0  # Exit when Z returns to 0
TRANSACTION_COST = 0.5 # transaction cost per lot/trade (assumed)

def run_debug_strategy(df):
    print("\n" + "="*50)
    print("DEBUG STRATEGY: INITIALIZATION")
    print("="*50)
    
    # 1. CHECK INPUT DATA
    if df is None or df.empty:
        print("[ERROR] Input Dataframe is EMPTY! Check your data loading step.")
        return pd.DataFrame()
    
    print(f"[INFO] Total Rows in Dataframe: {len(df)}")
    print(f"[INFO] Columns present: {list(df.columns)}")
    
    # Check if we have the right column for the spread
    if 'spread_cm_fut1' not in df.columns:
        print("[ERROR] Column 'spread_cm_fut1' NOT FOUND. Did you run Part 1?")
        return pd.DataFrame()

    # Sort data (Critical for rolling window)
    print("[INFO] Sorting data by Symbol and Time...")
    df = df.sort_values(['root_symbol', 'timestamp']).reset_index(drop=True)

    results = []
    unique_symbols = df['root_symbol'].unique()
    print(f"[INFO] Found {len(unique_symbols)} unique stocks to trade.")

    # 2. LOOP THROUGH EACH STOCK
    for i, symbol in enumerate(unique_symbols):
        # Limit debug prints to the first 2 stocks so we don't spam the console
        show_details = (i < 2) 
        
        if show_details:
            print(f"\n--- Processing Stock {i+1}: {symbol} ---")
        
        # Get data for this stock
        strat_df = df[df['root_symbol'] == symbol].copy()
        
        # 3. CALCULATE INDICATORS
        # We need enough data for the rolling window
        if len(strat_df) < ROLLING_WINDOW:
            if show_details: print(f"[WARN] Not enough data for {symbol} (Rows: {len(strat_df)} < Window: {ROLLING_WINDOW})")
            continue

        strat_df['rolling_mean'] = strat_df['spread_cm_fut1'].rolling(window=ROLLING_WINDOW).mean()
        strat_df['rolling_std'] = strat_df['spread_cm_fut1'].rolling(window=ROLLING_WINDOW).std()
        
        # Avoid division by zero
        strat_df['z_score'] = (strat_df['spread_cm_fut1'] - strat_df['rolling_mean']) / strat_df['rolling_std'].replace(0, np.nan)
        
        # 4. GENERATE SIGNALS
        # Buy (Long Spread) logic: Spread is too low -> Expect it to rise
        strat_df['long_signal'] = np.where(strat_df['z_score'] < -Z_ENTRY_THRESHOLD, 1, 0)
        strat_df['long_exit']   = np.where(strat_df['z_score'] > Z_EXIT_THRESHOLD, 1, 0)
        
        # Sell (Short Spread) logic: Spread is too high -> Expect it to fall
        strat_df['short_signal'] = np.where(strat_df['z_score'] > Z_ENTRY_THRESHOLD, 1, 0)
        strat_df['short_exit']   = np.where(strat_df['z_score'] < -Z_EXIT_THRESHOLD, 1, 0)
        
        if show_details:
            buy_sigs = strat_df['long_signal'].sum()
            sell_sigs = strat_df['short_signal'].sum()
            print(f"[DEBUG] Signals Generated -> Buys: {buy_sigs}, Sells: {sell_sigs}")

        # 5. SIMULATE TRADES (Iterate row by row)
        position = 0 # 0=Flat, 1=Long, -1=Short
        entry_price = 0.0
        pnl_log = []
        trade_count = 0
        
        # Convert to records for faster iteration than iterrows
        records = strat_df.to_dict('records')
        
        for row in records:
            current_price = row['spread_cm_fut1']
            current_pnl = 0
            
            # Skip if Z-score is NaN (first 60 mins)
            if pd.isna(row['z_score']):
                pnl_log.append(0)
                continue

            # --- EXIT LOGIC ---
            if position == 1 and row['long_exit'] == 1:
                # Close Long
                gross_pnl = current_price - entry_price
                current_pnl = gross_pnl - TRANSACTION_COST
                if show_details and trade_count < 3:
                    print(f"   [TRADE] CLOSE LONG at {current_price:.2f} | PnL: {current_pnl:.2f}")
                position = 0
                trade_count += 1
                
            elif position == -1 and row['short_exit'] == 1:
                # Close Short
                gross_pnl = entry_price - current_price
                current_pnl = gross_pnl - TRANSACTION_COST
                if show_details and trade_count < 3:
                    print(f"   [TRADE] CLOSE SHORT at {current_price:.2f} | PnL: {current_pnl:.2f}")
                position = 0
                trade_count += 1
                
            # --- ENTRY LOGIC ---
            if position == 0:
                if row['long_signal'] == 1:
                    position = 1
                    entry_price = current_price
                    if show_details and trade_count < 3:
                        print(f"   [TRADE] OPEN LONG at {current_price:.2f} (Z: {row['z_score']:.2f})")
                        
                elif row['short_signal'] == 1:
                    position = -1
                    entry_price = current_price
                    if show_details and trade_count < 3:
                        print(f"   [TRADE] OPEN SHORT at {current_price:.2f} (Z: {row['z_score']:.2f})")

            pnl_log.append(current_pnl)
            
        # End of Loop for this stock
        total_pnl = sum(pnl_log)
        results.append({
            'stock_name': symbol,
            'net_pnl': total_pnl,
            'trades': trade_count,
            'status': 'Active' if trade_count > 0 else 'No Trades'
        })
        
        if show_details:
            print(f"[RESULT] {symbol} Finished. Total PnL: {total_pnl:.2f}, Trades: {trade_count}")

    # 6. FINAL SUMMARY
    print("\n" + "="*50)
    print("STRATEGY COMPLETED")
    print("="*50)
    
    if not results:
        print("[ERROR] No results generated.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_pnl', ascending=False)
    
    print(results_df.head())
    print(f"\n[INFO] Saving results to 'Problem2_Results.csv'...")
    results_df.to_csv('Problem2_Results.csv', index=False)
    
    return results_df

# =============================================================================
# EXECUTION TRIGGER
# =============================================================================
# UNCOMMENT THE LINES BELOW TO RUN IT IMMEDIATELY
# Assuming your previous data is stored in 'master_df'

if 'master_df' in locals():
     final_res = run_debug_strategy(master_df)
else:
     print("[ERROR] 'master_df' is not found. Please run Step 1 (Data Processing) first.")