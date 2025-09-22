#!/usr/bin/env python3
import argparse, os, glob, json
from pathlib import Path
import pandas as pd

def month_key(ts):
    return pd.to_datetime(ts, errors="coerce", utc=True).strftime("%Y-%m")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    patt = f"reports/*_{args.run_tag}_S*of*"
    run_dirs = sorted(glob.glob(patt))
    if not run_dirs:
        print(f"No runs matched {patt}"); return

    outdir = Path(args.outdir or f"reports/{args.run_tag}_combined")
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for d in run_dirs:
        meta_p = Path(d) / "run_meta.json"
        tr_p   = Path(d) / "trades.csv"
        if not meta_p.exists() or not tr_p.exists(): continue
        meta = json.loads(meta_p.read_text())
        start = meta.get("start") or meta.get("args",{}).get("start")
        end   = meta.get("end")   or meta.get("args",{}).get("end")
        test_month = (end or "")[:7]  # last month of the window

        df = pd.read_csv(tr_p)
        # Normalize timestamps
        for c in ["entry_ts","exit_ts","entry","exit"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

        # Prefer 'entry_ts', else 'entry'
        ent = "entry_ts" if "entry_ts" in df.columns else ("entry" if "entry" in df.columns else None)
        if ent is None or df.empty: 
            continue

        # Keep only trades whose entry month == test month
        keep = df[ent].dt.tz_convert("UTC").dt.strftime("%Y-%m") == test_month
        kept = df.loc[keep].copy()
        if kept.empty: 
            continue

        kept["_run_dir"] = d
        kept["_window"]  = f"{(start or '')[:7]}→{test_month}"
        all_rows.append(kept)

    if not all_rows:
        print("No test-month trades found."); return

    out = pd.concat(all_rows, ignore_index=True)

    # Dedupe by (symbol, entry_ts, exit_ts) robustly
    sym = next((c for c in ["symbol","ticker","sym"] if c in out.columns), None)
    ent = "entry_ts" if "entry_ts" in out.columns else "entry"
    ext = "exit_ts"  if "exit_ts"  in out.columns else ("exit" if "exit" in out.columns else None)
    if sym and ent and ext:
        out = out.sort_values([sym, ent, ext, "_run_dir"]).drop_duplicates([sym, ent, ext], keep="first")

    out_fp = outdir / "trades_test_only_dedup.csv"
    out.to_csv(out_fp, index=False)

    # Per-window counts for sanity
    cnt = out.groupby("_window").size().reset_index(name="trades_in_test_month")
    cnt.to_csv(outdir / "window_trade_counts.csv", index=False)

    print("Wrote:", out_fp)
    print("Wrote:", outdir / "window_trade_counts.csv")

if __name__ == "__main__":
    main()
