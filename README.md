# Exuberance Fade Backtester (v2, patched)

Patched to fix `Series has no attribute 'hour'` (use `.dt` accessors) and clean up
timezone/period and deprecation warnings. All intraday logic is evaluated in **US/Eastern**.

## Quick start
```bash
pip install -r requirements.txt

# sanity check with synthetic data
python run_backtest.py --demo
```
