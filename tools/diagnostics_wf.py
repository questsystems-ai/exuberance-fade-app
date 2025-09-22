#!/usr/bin/env python3
import argparse, os, json, glob
from pathlib import Path
import pandas as pd
import numpy as np

def month_key(iso):
    return (iso or "")[:7]

def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def safe_to_datetime(s):
    # Quiet, consistent parse for ISO8601-like strings (incl. TZ)
    return pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")

def safe_trades_read(fp):
    if not os.path.exists(fp): return None
    try:
        df = pd.read_csv(fp)
        for c in ["entry_ts","exit_ts","entry","exit"]:
            if c in df.columns:
                df[c] = safe_to_datetime(df[c])
        return df
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Diagnose WF runs by run-tag")
    ap.add_argument("--run-tag", required=True, help="e.g., wf_2y_rth")
    ap.add_argument("--expected-windows", type=int, default=None)
    ap.add_argument("--expected-shards", type=int, default=8)
    args = ap.parse_args()

    patt = f"reports/*_{args.run_tag}_S*of*"
    run_dirs = sorted(glob.glob(patt))
    if not run_dirs:
        print(f"No runs matched pattern: {patt}")
        return

    rows = []
    all_trades = []
    win_acc = {}  # window -> accumulators for audit

    for d in run_dirs:
        meta = read_json(os.path.join(d, "run_meta.json"))
        argz = meta.get("args", {})
        start = argz.get("start", meta.get("start"))
        end   = argz.get("end",   meta.get("end"))
        shard_idx = argz.get("shard_index")
        shard_cnt = argz.get("shard_count", args.expected_shards)
        kept = meta.get("kept_symbols", meta.get("symbols", [])) or []
        kept = [str(s) for s in kept]
        win_key = f"{month_key(start)}→{month_key(end)}"
        test_month = month_key(end)

        trades_fp  = os.path.join(d, "trades.csv")
        summary_fp = os.path.join(d, "summary.csv")
        report_pdf = os.path.join(d, "quick_report.pdf")

        tdf = safe_trades_read(trades_fp)
        trades_rows = int(len(tdf)) if tdf is not None else None

        # detect PnL column & compute both (a) all-trade PnL and (b) test-month-only PnL
        pnl_sum = None; pnl_mean = None; has_nan = False
        trades_rows_test = None; pnl_sum_test = None

        if tdf is not None and not tdf.empty:
            pnl_col = next((c for c in ["pnl","pnl_net","pnl_$","PNL","profit"] if c in tdf.columns), None)
            if pnl_col:
                pn = pd.to_numeric(tdf[pnl_col], errors="coerce")
                pnl_sum = float(np.nansum(pn.values))
                pnl_mean = float(np.nanmean(pn.values)) if len(pn) else None
                has_nan = bool(pd.isna(pn).any())

                # test-month filter based on entry timestamp
                ent = "entry_ts" if "entry_ts" in tdf.columns else ("entry" if "entry" in tdf.columns else None)
                if ent:
                    ent_m = tdf[ent].dt.tz_convert("UTC").dt.strftime("%Y-%m")
                    mask = (ent_m == test_month)
                    rows_test = tdf.loc[mask]
                    trades_rows_test = int(len(rows_test))
                    pnl_sum_test = float(np.nansum(pd.to_numeric(rows_test[pnl_col], errors="coerce").values))
            all_trades.append(tdf.assign(_run_dir=d, _window=win_key, _shard=shard_idx))

        rows.append({
            "run_dir": d,
            "window": win_key,
            "start": start, "end": end, "test_month": test_month,
            "shard_index": shard_idx, "shard_count": shard_cnt,
            "kept_symbols_count": len(kept), "kept_symbols": ",".join(kept),
            "trades_rows": trades_rows,
            "trades_rows_test": trades_rows_test,
            "trades_csv": os.path.exists(trades_fp),
            "summary_csv": os.path.exists(summary_fp),
            "report_pdf": os.path.exists(report_pdf),
            "pnl_sum": pnl_sum, "pnl_mean": pnl_mean, "pnl_has_nan": has_nan,
            "pnl_sum_test": pnl_sum_test,
        })

        # ---- Window accumulator (for audit) ----
        acc = win_acc.setdefault(win_key, {
            "start": start, "end": end, "test_month": test_month,
            "present_shards": set(), "symbols": set(),
            "trades_files": 0, "trades_rows_sum": 0,
            "trades_rows_sum_test": 0,
            "trades_empty_or_unreadable": 0, "summaries": 0,
            "expected_shards": int(shard_cnt) if shard_cnt else args.expected_shards,
        })
        if shard_idx is not None:
            try: acc["present_shards"].add(int(shard_idx))
            except: pass
        acc["symbols"].update(kept)
        if os.path.exists(trades_fp): acc["trades_files"] += 1
        if trades_rows is None or trades_rows == 0:
            acc["trades_empty_or_unreadable"] += 1
        else:
            acc["trades_rows_sum"] += int(trades_rows or 0)
        if trades_rows_test:  # may be None
            acc["trades_rows_sum_test"] += int(trades_rows_test)
        if os.path.exists(summary_fp): acc["summaries"] += 1

    df = pd.DataFrame(rows)
    outdir = Path(f"reports/{args.run_tag}_combined")
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-window shard completeness
    missing = []
    for w, g in df.groupby("window"):
        present = set(g["shard_index"].dropna().astype(int).tolist())
        exp_n = int(g["shard_count"].iloc[0] or args.expected_shards)
        exp = set(range(exp_n))
        miss = sorted(list(exp - present))
        if miss:
            missing.append({"window": w, "missing_shards": ",".join(map(str, miss))})
    pd.DataFrame(missing).to_csv(outdir / "missing_shards.csv", index=False)

    # Empty/missing files
    df[(df["trades_csv"] == True) & ((df["trades_rows"].isna()) | (df["trades_rows"] == 0))] \
        .to_csv(outdir / "trades_empty_or_unreadable.csv", index=False)
    df[df["trades_csv"] == False].to_csv(outdir / "trades_missing.csv", index=False)
    df[df["summary_csv"] == False].to_csv(outdir / "summary_missing.csv", index=False)

    # Duplicate trades (symbol, entry_ts, exit_ts)
    dupes_csv = outdir / "trades_duplicates.csv"
    if all_trades:
        at = pd.concat(all_trades, ignore_index=True)
        sym_col = next((c for c in ["symbol","ticker","sym"] if c in at.columns), None)
        e_col = "entry_ts" if "entry_ts" in at.columns else ("entry" if "entry" in at.columns else None)
        x_col = "exit_ts"  if "exit_ts"  in at.columns else ("exit"  if "exit"  in at.columns else None)
        if sym_col and e_col and x_col:
            k = (at[sym_col].astype(str) + "|" + at[e_col].astype(str) + "|" + at[x_col].astype(str))
            at["_key"] = k
            vc = at["_key"].value_counts()
            dup_keys = vc[vc > 1].index
            at[at["_key"].isin(dup_keys)].sort_values(["_key","_run_dir"]).to_csv(dupes_csv, index=False)
        else:
            Path(dupes_csv).write_text("insufficient columns to test duplicates\n")

    # PnL outliers on shard-level pnl_sum
    df_ps = df.dropna(subset=["pnl_sum"]).copy()
    if len(df_ps) >= 5:
        mu, sd = df_ps["pnl_sum"].mean(), df_ps["pnl_sum"].std(ddof=0) or 1.0
        df_ps["pnl_z"] = (df_ps["pnl_sum"] - mu) / sd
        df_ps[(df_ps["pnl_z"].abs() >= 4)].to_csv(outdir / "pnl_outliers.csv", index=False)

    # Shard-level summary
    df.sort_values(["window","shard_index"]).to_csv(outdir / "diagnostics_summary.csv", index=False)

    # ---- Window audit (as before) ----
    audit_rows = []
    for w, acc in win_acc.items():
        exp_n = int(acc["expected_shards"])
        present_n = len(acc["present_shards"])
        miss = sorted(list(set(range(exp_n)) - acc["present_shards"]))
        issues = []
        if miss: issues.append(f"missing_shards={','.join(map(str,miss))}")
        if acc["trades_empty_or_unreadable"] > 0:
            issues.append(f"empty_or_bad_trades={acc['trades_empty_or_unreadable']}")
        if acc["trades_files"] < present_n:
            issues.append(f"missing_trades_files={present_n-acc['trades_files']}")
        if acc["summaries"] < present_n:
            issues.append(f"missing_summaries={present_n-acc['summaries']}")
        status = "OK" if not issues else ";".join(issues)

        audit_rows.append({
            "window": w, "start": acc["start"], "end": acc["end"],
            "shards_present": present_n, "shards_expected": exp_n,
            "missing_shards": ",".join(map(str, miss)) if miss else "",
            "unique_symbols_count": len(acc["symbols"]),
            "symbols": ",".join(sorted(acc["symbols"])),
            "trades_files_present": acc["trades_files"],
            "trades_empty_or_unreadable": acc["trades_empty_or_unreadable"],
            "trades_total_rows": acc["trades_rows_sum"],
            "trades_total_rows_test": acc["trades_rows_sum_test"],
            "summaries_present": acc["summaries"],
            "status": status,
        })
    pd.DataFrame(audit_rows).sort_values("window").to_csv(outdir / "window_audit.csv", index=False)

    # ---- NEW: per-window aggregate PnL (TEST-MONTH ONLY) ----
    # Sum shard-level 'pnl_sum_test' per window to avoid cross-window duplicates
    wsum = (df.groupby(["window","start","end"], dropna=False)[["pnl_sum_test","trades_rows_test"]]
              .sum(min_count=1)
              .reset_index())
    wsum.columns = ["window","start","end","pnl_sum_test_total","trades_in_test_month_total"]
    wsum["pnl_per_trade_test"] = np.where(
        wsum["trades_in_test_month_total"] > 0,
        wsum["pnl_sum_test_total"] / wsum["trades_in_test_month_total"],
        np.nan,
    )
    wsum.sort_values("window").to_csv(outdir / "window_pnl.csv", index=False)
    # pandas warning guard for chained sum
    wsum.columns = ["window","start","end","pnl_sum_test_total","trades_in_test_month_total"]
    wsum["pnl_per_trade_test"] = np.where(wsum["trades_in_test_month_total"]>0,
                                          wsum["pnl_sum_test_total"]/wsum["trades_in_test_month_total"],
                                          np.nan)
    wsum.sort_values("window").to_csv(outdir / "window_pnl.csv", index=False)

    # Console summary
    windows = sorted(df["window"].unique().tolist())
    completed = len(windows)
    expected = args.expected_windows or completed
    print(f"Found {len(df)} shard runs across {completed} windows (expected={expected}).")
    if len(missing) > 0:
        print(f"Windows with missing shards: {len(missing)}  -> {outdir/'missing_shards.csv'}")
    print(f"Missing trades files: {len(df[df['trades_csv']==False])}  -> {outdir/'trades_missing.csv'}")
    print(f"Empty/unreadable trades: {len(df[(df['trades_csv']==True)&((df['trades_rows'].isna())|(df['trades_rows']==0))])}  -> {outdir/'trades_empty_or_unreadable.csv'}")
    print(f"Missing summary.csv: {len(df[df['summary_csv']==False])}  -> {outdir/'summary_missing.csv'}")
    if all_trades:
        print(f"Combined trades for duplicate check written (if columns present): {outdir/'trades_duplicates.csv'}")
    print(f"Shard diagnostics summary: {outdir/'diagnostics_summary.csv'}")
    print(f"Window audit: {outdir/'window_audit.csv'}")
    print(f"Window PnL (test-month only): {outdir/'window_pnl.csv'}")

if __name__ == "__main__":
    main()