#!/usr/bin/env python3
import argparse, os, subprocess
from datetime import datetime, timezone, timedelta

def month_floor(dt): return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
def add_months(dt, n):
    y = dt.year + (dt.month-1 + n)//12
    m = (dt.month-1 + n)%12 + 1
    return dt.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)

def build_windows(start, end, train_m, test_m):
    s = month_floor(datetime.fromisoformat(start).replace(tzinfo=timezone.utc))
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    ts = add_months(s, train_m)  # first test_start = start + train months
    out=[]
    while True:
        te = add_months(ts, test_m) - timedelta(seconds=1)
        if te > e: break
        tr0 = add_months(ts, -train_m)
        out.append((tr0.date().isoformat(), te.date().isoformat()))
        ts = add_months(ts, test_m)
    return out

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
    ap.add_argument("--rth-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    wins = build_windows(args.start, args.end, args.train_months, args.test_months)
    print(f"Total WF windows: {len(wins)}")

    cores = os.cpu_count() or 8
    per = max(1, cores // max(1,args.shards))
    base_env = os.environ.copy()
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        base_env[k] = str(per)

    py = os.getenv("PYTHON", "python")
    for wi, (wstart, wend) in enumerate(wins):
        tag_w = f"{args.run_tag}_w{wi:02d}"
        procs=[]
        print(f"\n=== Window {wi:02d}: {wstart} → {wend} (naive dates) ===")
        for i in range(args.shards):
            cmd = [
                py, "run_backtest.py",
                "--source","local",
                "--symbols", args.symbols,
                "--start", wstart, "--end", wend,
                "--train-months", str(args.train_months),
                "--test-months",  str(args.test_months),
                "--max-combos",   str(args.max_combos),
                "--run-tag", f"{tag_w}_s{i}",
                "--shard-index", str(i),
                "--shard-count", str(args.shards),
                "--seed", str(args.seed)
            ]
            if args.rth_only: cmd.append("--rth-only")
            print("launch:", " ".join(cmd))
            procs.append(subprocess.Popen(cmd, env=base_env))
        # wait for shards of this window
        rc = 0
        for p in procs:
            p.wait()
            rc |= p.returncode or 0
        if rc != 0:
            raise SystemExit(f"Window {wi:02d} failed with code {rc}")
    print("\n✅ All windows complete.")
if __name__ == "__main__":
    main()
