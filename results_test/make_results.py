import pandas as pd

s = pd.read_csv("results_test/mr_spread_rel_summary_E2.0_TP0.0_SL4.0.csv")
t = pd.read_csv("results_test/mr_spread_rel_trades_E2.0_TP0.0_SL4.0.csv")
p = pd.read_csv("results_test/mr_spread_rel_perminute_E2.0_TP0.0_SL4.0.csv")


n_days = p.groupby('UNDERLYING')['date'].nunique().rename('n_traded_days')

if 'shares_per_lot' in t.columns:
    t['abs_shares'] = t['delta_lots'].abs() * t['shares_per_lot']
else:
    t['abs_shares'] = t['delta_lots'].abs()
n_days = p.groupby('UNDERLYING')['date'].nunique().rename('n_traded_days')

if 'shares_per_lot' in t.columns:
    t['abs_shares'] = t['delta_lots'].abs() * t['shares_per_lot']
else:
    t['abs_shares'] = t['delta_lots'].abs()

max_delta = t.groupby('UNDERLYING')['abs_shares'].max().rename('max_delta_qty')
total_vol = t.groupby('UNDERLYING')['abs_shares'].sum().rename('total_volume')

# FIXED VERSION
lots = t.groupby('UNDERLYING')['delta_lots'].apply(lambda x: abs(x).sum()).rename('total_lots_traded')

out = s.set_index('UNDERLYING').copy()
out = out.join(n_days).join(max_delta).join(total_vol).join(lots)

out['stock_name'] = out.index
out['net_pnl'] = out['final_net_pnl']
out['gross_pnl'] = out['final_gross_pnl']
out['cost_pnl'] = out['final_cost_pnl']
out['slippage_fut1'] = out.get('final_spr_slpg1', 0)
out['slippage_fut2'] = out.get('final_spr_slpg2', 0)
out['max_gross_qty'] = out['total_volume']
out['drawdown'] = out['max_drawdown']

total_lots_market = out['total_lots_traded'].sum()
out['market_perc'] = out['total_lots_traded'] / max(1, total_lots_market)

cols = [
    'stock_name','n_traded_days','net_pnl','gross_pnl','cost_pnl',
    'slippage_fut1','slippage_fut2','total_lots_traded','total_volume',
    'max_delta_qty','max_gross_qty','drawdown','market_perc'
]

out[cols].sort_values('net_pnl', ascending=False).to_csv(
    "Problem2_results.csv", index=False
)
