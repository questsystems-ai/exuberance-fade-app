# compare_vs_spy.py
"""
Usage:
  python compare_vs_spy.py combined_monthly_pnl.csv spy_monthly.csv 30000
Where:
  combined_monthly_pnl.csv has columns: month, pnl  (month like 2024-08)
  spy_monthly.csv has columns: month, pct (e.g., 0.015 for +1.5%)
  30000 is your bankroll to translate % to dollars.
"""
import sys, pandas as pd

cm = pd.read_csv(sys.argv[1])
spy = pd.read_csv(sys.argv[2])
bank = float(sys.argv[3])

# normalize
cm.columns = [c.lower() for c in cm.columns]
spy.columns = [c.lower() for c in spy.columns]
mcol = "month"; pcol = "pnl"
df = cm.merge(spy, on=mcol, how="outer", suffixes=("_strat", "_spy")).sort_values(mcol)

# compute dollars for SPY given bankroll
df["spy_dollars"] = (df["pct"].fillna(0.0) * bank).round(2)
df["strat_dollars"] = df[pcol].fillna(0.0) * (bank/100000.0) * (0.03/0.015)  # adjust if needed
# NOTE: if your strat pnl in combined_monthly is already absolute dollars on $100k @1.5% stake, the last term (0.03/0.015) scales to 3% stake; adjust to 1.0 if you want baseline.

df.to_csv("compare_vs_spy_out.csv", index=False)
print(df.tail(14))
print("\nSaved: compare_vs_spy_out.csv")
