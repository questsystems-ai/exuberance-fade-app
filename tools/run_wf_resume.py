#!/usr/bin/env python3
import argparse, os, subprocess, glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

def month_floor(dt):
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

def add_months(dt, n):
    y = dt.year + (dt.month - 1 + n) // 12
    m = (dt.month - 1 + n) % 12 + 1
    return dt.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)

def build_windows(start, end, train_m, test_m):
    s = month_floor(datetime.fromisoformat(start).replace(tzinfo=timezone.utc))
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    ts = add_months(s, train_m)  # first test start
    out = []
    while True:
        te = add_months(ts, test_m) - timedelta(seconds=1)
        if te > e:
            break
        # pass naive YYYY-MM-DD to run_backtest.py
        tr0 = add_months(ts, -train_m)
        out.append((tr0.date().isoformat(), te.date().isoformat()))
        ts = add_months(ts, test_m)
    return out

def shard_done(run_tag, wi, i):
    paths = sorted(glob.glob(f"reports/*_{run_tag}_w{wi:02d}_s{i}"))
    if not paths:
        return False
    t = Path(paths[-1]) / "trades.csv"
    if not t.exists():
        return False
    try:
        return (sum(1 for _ in open(t)) - 1) > 0  # has data rows
    except Exception:
        return False

def window_complete(run_tag, wi, shards):
    for i in range(shards):
        if not shard_done(run_tag, wi, i):
            return False
    return True
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-months", type=int, default=2)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--max-combos", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rth-only", action="store_true", default=True)
    args = ap.parse_args()
    if not args.symbols or not args.symbols.strip():
        raise SystemExit('Error: --symbols is empty; pass a space- or comma-separated list')


    wins = build_windows(args.start, args.end, args.train_months, args.test_months)
    print("Total windows:", len(wins))

    per = max(1, (os.cpu_count() or 8) // max(1, args.shards))
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[k] = str(per)
    py = os.getenv("PYTHON", "python")

    for wi, (wstart, wend) in enumerate(wins):
        if window_complete(args.run_tag, wi, args.shards):
            print(f"== Window {wi:02d} {wstart}->{wend}: complete, skipping")
            continue
        print(f"== Window {wi:02d} {wstart}->{wend}: launching missing shards")
        procs = []
        for i in range(args.shards):
            if shard_done(args.run_tag, wi, i):
                print(f"   - shard {i}: ok, skip")
                continue
            cmd = [
                py, "run_backtest.py",
                "--source", "local",
                "--symbols", args.symbols,
                "--start", wstart, "--end", wend,
                "--train-months", str(args.train_months),
                "--test-months",  str(args.test_months),
                "--max-combos",   str(args.max_combos),
                "--run-tag", f"{args.run_tag}_w{wi:02d}_s{i}",
                "--shard-index", str(i),
                "--shard-count", str(args.shards),
                "--seed", str(args.seed),
            ]
            if args.rth_only:
                cmd.append("--rth-only")
            print("   launch:", " ".join(cmd))
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            p.wait()
    print("✅ Resume pass complete.")

if __name__ == "__main__":
    main()
