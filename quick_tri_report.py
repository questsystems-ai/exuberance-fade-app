import sys, os, json, pandas as pd
from pathlib import Path

def load_metrics(folder, label):
    f = Path(folder)
    summ = pd.read_csv(f/"summary.csv")
    # try to get symbol count from trades.csv if present
    symbols = None
    tr = f/"trades.csv"
    if tr.exists():
        try:
            tdf = pd.read_csv(tr, usecols=["symbol"])
            symbols = int(tdf["symbol"].nunique())
        except Exception:
            pass
    # fallbacks for columns, depending on your backtester export
    col = lambda c: c if c in summ.columns else None
    out = {
        "run": label,
        "symbols": symbols if symbols is not None else None,
        "mean_cagr": summ[col("cagr")].mean() if col("cagr") else None,
        "mean_sharpe": summ[col("sharpe")].mean() if col("sharpe") else None,
        "mean_maxdd": summ[col("max_dd")].mean() if col("max_dd") else None,
        "mean_calmar": summ[col("calmar")].mean() if col("calmar") else None,
        "mean_win_daily": summ[col("daily_win_rate")].mean() if col("daily_win_rate") else None,
        "mean_win_weekly": summ[col("weekly_win_rate")].mean() if col("weekly_win_rate") else None,
    }
    return out

def html_table(rows):
    # simple HTML table encoder
    cols = ["run","symbols","mean_cagr","mean_sharpe","mean_maxdd","mean_calmar","mean_win_daily","mean_win_weekly"]
    df = pd.DataFrame(rows, columns=cols)
    # nice numeric formatting
    num_cols = [c for c in cols if c not in ("run","symbols")]
    for c in num_cols:
        if c in df and df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    if "symbols" in df:
        df["symbols"] = df["symbols"].fillna("").astype(str)
    return df.to_html(border=1, index=False)

def main():
    if len(sys.argv) < 8:
        print("usage: python quick_tri_report.py <infra_dir> <infra_label> <supply_dir> <supply_label> <hvol_dir> <hvol_label> <out_dir>")
        sys.exit(1)
    infra_dir, infra_lbl, supply_dir, supply_lbl, hvol_dir, hvol_lbl, out_dir = sys.argv[1:]
    rows = []
    for d,lbl in [(infra_dir, infra_lbl), (supply_dir, supply_lbl), (hvol_dir, hvol_lbl)]:
        if d and os.path.isdir(d) and os.path.exists(os.path.join(d,"summary.csv")):
            rows.append(load_metrics(d, lbl))
        else:
            rows.append({"run": lbl, "symbols": "", "mean_cagr":"","mean_sharpe":"","mean_maxdd":"","mean_calmar":"","mean_win_daily":"","mean_win_weekly":""})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    html = "<h1>Options Bottom Line — Summary</h1>\n<h2>Runs</h2><ul>" + "".join(f"<li>{r['run']}</li>" for r in rows) + "</ul>\n<h2>Aggregate Metrics</h2>\n" + html_table(rows)
    with open(Path(out_dir)/"bottom_line_summary.html", "w") as f:
        f.write(html)
    print(f"Wrote {Path(out_dir)/'bottom_line_summary.html'}")

if __name__ == "__main__":
    main()
