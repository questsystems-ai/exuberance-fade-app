
#!/usr/bin/env python3
"""
tools/systems_check.py

End-to-end systems check for simplified runs.
- Finds the latest run dirs for baseline & optimizer by tag
- Synthesizes missing CSVs (opt_window_pnl.csv / window_pnl.csv) from summary.csv
- Verifies file presence & non-emptiness
- Checks min-trade floors (train/test)
- Compares PnL, trades, and drawdown between runs
- Writes a report text file alongside console output
"""
import argparse, os, json, glob, sys
from pathlib import Path
import pandas as pd

REQUIRED_FILES = ["summary.csv", "bot_summary.json", "run_meta.json", "trades.csv"]

def find_latest_run(tag: str) -> str | None:
    matches = sorted(glob.glob(f"reports/*_{tag}"))
    return matches[-1] if matches else None

def _nonempty(path: str) -> bool:
    return path and os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0

def synthesize_outputs(run_dir: str, is_optimizer: bool) -> list[str]:
    created = []
    summ = os.path.join(run_dir, "summary.csv")
    if not _nonempty(summ):
        return created
    try:
        df = pd.read_csv(summ)
    except Exception as e:
        print(f"[WARN] failed to read summary.csv in {run_dir}: {e}")
        return created

    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    if "window_start" in cols and "window_end" in cols:
        out["window_start"] = df[cols["window_start"]]
        out["window_end"]   = df[cols["window_end"]]
    elif "window" in cols:
        out["window"] = df[cols["window"]]
    if "phase" in cols:
        out["phase"] = df[cols["phase"]]

    pnl_col = cols.get("opt_test_pnl") or cols.get("pnl")
    trd_col = cols.get("opt_test_trades") or cols.get("test_trades") or cols.get("trades")
    dd_col  = cols.get("test_drawdown") or cols.get("max_drawdown") or cols.get("drawdown")

    if pnl_col:
        out["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    if trd_col:
        out["trades"] = pd.to_numeric(df[trd_col], errors="coerce").fillna(0).astype(int)
    if dd_col:
        out["drawdown"] = pd.to_numeric(df[dd_col], errors="coerce").fillna(0.0)

    target = os.path.join(run_dir, "opt_window_pnl.csv" if is_optimizer else "window_pnl.csv")
    try:
        out.to_csv(target, index=False)
        created.append(target)
        print(f"[INFO] wrote {target} from summary.csv")
    except Exception as e:
        print(f"[WARN] failed to write synthesized CSV in {run_dir}: {e}")
    return created

def presence_check(run_dir: str, is_optimizer: bool) -> dict:
    files = REQUIRED_FILES.copy()
    files += ["opt_window_pnl.csv"] if is_optimizer else ["window_pnl.csv"]
    status = {}
    for f in files:
        p = os.path.join(run_dir, f)
        status[f] = "ok" if _nonempty(p) else "missing/empty"
    el = os.path.join(run_dir, "error.log")
    status["error.log"] = "present" if _nonempty(el) else "absent"
    return status

def floors_check(run_dir: str, is_optimizer: bool, min_train: int, min_test: int, min_test_per_week: float) -> dict:
    csv_name = "opt_window_pnl.csv" if is_optimizer else "window_pnl.csv"
    df = None
    try:
        df = pd.read_csv(os.path.join(run_dir, csv_name))
    except Exception:
        return {"status": "no_data"}

    cols = {c.lower(): c for c in df.columns}
    res = {"status": "ok", "breaches": []}

    # try to merge train_trades from summary
    try:
        summ = pd.read_csv(os.path.join(run_dir, "summary.csv"))
        sc = {c.lower(): c for c in summ.columns}
        if "train_trades" in sc:
            if "window_start" in sc and "window_end" in sc and "window_start" in cols and "window_end" in cols:
                df = df.merge(
                    summ[[sc["window_start"], sc["window_end"], sc["train_trades"]]].rename(
                        columns={sc["window_start"]: "window_start", sc["window_end"]: "window_end", sc["train_trades"]: "train_trades"}
                    ),
                    on=["window_start", "window_end"],
                    how="left"
                )
            elif "window" in sc and "window" in cols:
                df = df.merge(
                    summ[[sc["window"], sc["train_trades"]]].rename(
                        columns={sc["window"]: "window", sc["train_trades"]: "train_trades"}
                    ),
                    on=["window"],
                    how="left"
                )
    except Exception:
        pass

    c = {c.lower(): c for c in df.columns}
    if "train_trades" in c:
        bad = df[c["train_trades"]].fillna(0).astype(int) < int(min_train)
        if bad.any():
            res["status"] = "breach"
            res["breaches"].append(f"train_trades<{min_train}: {int(bad.sum())} rows")

    test_trades_col = c.get("opt_test_trades") or c.get("test_trades") or c.get("trades")
    if test_trades_col:
        bad = df[test_trades_col].fillna(0).astype(int) < int(min_test)
        if bad.any():
            res["status"] = "breach"
            res["breaches"].append(f"test_trades<{min_test}: {int(bad.sum())} rows")

    if min_test_per_week > 0 and test_trades_col:
        weeks = 4.0
        bad = (df[test_trades_col].fillna(0).astype(float) / weeks) < float(min_test_per_week)
        if bad.any():
            res["status"] = "breach"
            res["breaches"].append(f"test_trades/week<{min_test_per_week}: {int(bad.sum())} rows")

    return res

def totals(df: pd.DataFrame) -> dict:
    c = {x.lower(): x for x in df.columns}
    out = {}
    if "pnl" in c:
        out["pnl"] = float(pd.to_numeric(df[c["pnl"]], errors="coerce").fillna(0).sum())
    else:
        pnl_cols = [x for x in df.columns if "pnl" in x.lower()]
        out["pnl"] = float(pd.to_numeric(df[pnl_cols], errors="coerce").fillna(0).sum().sum()) if pnl_cols else 0.0

    for tcol in ["opt_test_trades","test_trades","trades"]:
        if tcol in c:
            out["trades"] = int(pd.to_numeric(df[c[tcol]], errors="coerce").fillna(0).sum())
            break

    for dcol in ["drawdown","test_drawdown","max_drawdown"]:
        if dcol in c:
            out["min_drawdown"] = float(pd.to_numeric(df[c[dcol]], errors="coerce").min())
            break

    return out

def compare_runs(baseline_dir: str, opt_dir: str) -> dict:
    import pandas as pd
    b = pd.read_csv(os.path.join(baseline_dir, "window_pnl.csv"))
    o = pd.read_csv(os.path.join(opt_dir, "opt_window_pnl.csv"))
    for df in (b, o):
        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.contains("test", case=False)]
    tb = totals(b); to = totals(o)
    out = {"baseline": tb, "optimizer": to, "lift_pnl": to.get("pnl",0.0) - tb.get("pnl",0.0)}
    return out

def run_report(baseline_tag: str, opt_tag: str, min_train: int, min_test: int, min_test_per_week: float) -> int:
    bdir = find_latest_run(baseline_tag) if baseline_tag else None
    odir = find_latest_run(opt_tag) if opt_tag else None
    if not bdir:
        print(f"[ERROR] baseline tag '{baseline_tag}' not found."); return 2
    if not odir:
        print(f"[ERROR] optimizer tag '{opt_tag}' not found."); return 2

    print(f"[INFO] baseline dir:  {bdir}")
    print(f"[INFO] optimizer dir: {odir}")

    synthesize_outputs(bdir, is_optimizer=False)
    synthesize_outputs(odir, is_optimizer=True)

    bp = presence_check(bdir, is_optimizer=False)
    op = presence_check(odir, is_optimizer=True)
    print("\n[Presence] baseline:")
    for k,v in bp.items(): print(f"  {k:20} {v}")
    print("[Presence] optimizer:")
    for k,v in op.items(): print(f"  {k:20} {v}")

    bf = floors_check(bdir, False, min_train, min_test, min_test_per_week)
    of = floors_check(odir, True,  min_train, min_test, min_test_per_week)
    print("\n[Floors] baseline:", bf)
    print("[Floors] optimizer:", of)

    try:
        cmpo = compare_runs(bdir, odir)
        print("\n[Compare] totals:", json.dumps(cmpo, indent=2))
    except Exception as e:
        print(f"[WARN] compare failed: {e}")

    rep_path = os.path.join(os.getcwd(), "systems_check_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("Baseline dir:  " + bdir + "\n")
        f.write("Optimizer dir: " + odir + "\n\n")
        f.write("[Presence] baseline:\n")
        for k,v in bp.items(): f.write(f"  {k:20} {v}\n")
        f.write("[Presence] optimizer:\n")
        for k,v in op.items(): f.write(f"  {k:20} {v}\n")
        f.write("\n[Floors] baseline: " + json.dumps(bf) + "\n")
        f.write("[Floors] optimizer: " + json.dumps(of) + "\n")
    print(f"\n[INFO] wrote {rep_path}")
    return 0

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tag", required=True)
    ap.add_argument("--opt-tag", required=True)
    ap.add_argument("--min-train-trades", type=int, default=int(os.getenv("OPT_MIN_TRAIN_TRADES","5")))
    ap.add_argument("--min-test-trades",  type=int, default=int(os.getenv("OPT_MIN_TEST_TRADES","3")))
    ap.add_argument("--min-test-trades-per-week", type=float, default=float(os.getenv("OPT_MIN_TEST_TRADES_PER_WEEK","0")))
    args = ap.parse_args()
    sys.exit(run_report(args.baseline_tag, args.opt_tag, args.min_train_trades, args.min_test_trades, args.min_test_trades_per_week))

if __name__ == "__main__":
    main()
