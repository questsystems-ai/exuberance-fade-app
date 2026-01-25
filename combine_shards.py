# combine_shards.py
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import zipfile

def extract_zip(zpath: Path) -> Path:
    out = zpath.with_name("_" + zpath.stem)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(out)
    # locate *_ALL folder
    for p in out.rglob("*_ALL"):
        return p
    return out

def find1(root: Path, relname: str):
    for p in root.rglob(relname):
        return p
    return None

def run(zip_path: str):
    z = Path(zip_path)
    root = extract_zip(z)
    shards = sorted([p for p in root.iterdir() if p.is_dir() and "_s" in p.name])

    rows = []
    monthly_list, trades_list, params_rows = [], [], []
    for sd in shards:
        metrics = find1(sd, "metrics_summary.csv")
        monthly = find1(sd, "monthly_pnl.csv")
        trades = find1(sd, "trades.csv")
        best_params = find1(sd, "best_params_applied.json") or find1(sd, "best_params.json")

        m = {}
        if metrics and metrics.exists():
            dfm = pd.read_csv(metrics)
            if {"metric","value"}.issubset(dfm.columns):
                m = dict(zip(dfm["metric"], dfm["value"]))

        if monthly and monthly.exists():
            m2 = pd.read_csv(monthly).copy()
            m2["shard"] = sd.name
            monthly_list.append(m2)

        if trades and trades.exists():
            t = pd.read_csv(trades).copy()
            t["shard"] = sd.name
            trades_list.append(t)

        bp = {}
        if best_params and best_params.exists():
            try: bp = json.loads(best_params.read_text())
            except: pass
        params_rows.append({"shard": sd.name, **bp})

        rows.append({
            "shard": sd.name,
            "total_pnl": m.get("total_pnl"),
            "hit_rate": m.get("hit_rate"),
            "win_loss_ratio": m.get("win_loss_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe": m.get("sharpe")
        })

    outdir = root
    # per-shard metrics
    per_shard = pd.DataFrame(rows)
    per_shard.to_csv(outdir / "per_shard_metrics.csv", index=False)

    # combined monthly pnl
    if monthly_list:
        mcat = []
        for df in monthly_list:
            d = df.copy()
            d.columns = [c.lower() for c in d.columns]
            mcol = "month" if "month" in d.columns else d.columns[0]
            if "pnl" in d.columns: pcol = "pnl"
            elif "total_pnl" in d.columns: pcol = "total_pnl"
            else:
                num = d.select_dtypes(include="number").columns
                if len(num)==0: continue
                pcol = num[-1]
            mcat.append(d[[mcol,pcol]].rename(columns={mcol:"month", pcol:"pnl"}))
        if mcat:
            comb = pd.concat(mcat, ignore_index=True).groupby("month", as_index=False)["pnl"].sum().sort_values("month")
            comb.to_csv(outdir / "combined_monthly_pnl.csv", index=False)

    # combined trades & portfolio stats
    if trades_list:
        T = pd.concat(trades_list, ignore_index=True)
        pnl_col = None
        for c in ["pnl","profit","net_pnl","pnl_usd"]:
            if c in T.columns: pnl_col = c; break
        if pnl_col is None:
            num = T.select_dtypes(include="number").columns
            pnl_col = num[-1] if len(num) else None
        if pnl_col:
            vals = pd.to_numeric(T[pnl_col], errors="coerce").fillna(0.0)
            total_pnl = float(vals.sum())
            wins, losses = vals[vals > 0], vals[vals < 0]
            hit_rate = float((wins.count()) / max(1, (wins.count()+losses.count())))
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            wlr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
            pd.DataFrame([{
                "total_pnl": total_pnl,
                "hit_rate": hit_rate,
                "win_loss_ratio": wlr,
                "num_trades": int(len(T))
            }]).to_csv(outdir / "portfolio_stats.csv", index=False)
            T.to_csv(outdir / "trades_combined.csv", index=False)

    # parameter modes
    params = pd.DataFrame(params_rows)
    if not params.empty:
        modes = {}
        for c in params.columns:
            if c == "shard": continue
            col = params[c].dropna()
            if len(col):
                try:
                    mv = col.mode(dropna=True)
                    if len(mv): modes[c] = mv.iloc[0]
                except: pass
        if modes:
            pd.DataFrame([modes]).to_csv(outdir / "best_params_mode.csv", index=False)

    print(f"Combined outputs written to: {outdir}")

if __name__ == "__main__":
    assert len(sys.argv)==2, "Usage: python combine_shards.py path/to/*_ALL.zip"
    run(sys.argv[1])
