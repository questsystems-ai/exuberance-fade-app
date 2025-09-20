#!/usr/bin/env python3
import argparse, os, subprocess, glob
from datetime import datetime, timezone, timedelta

def month_floor(dt): return dt.replace(day=1,hour=0,minute=0,second=0,microsecond=0,tzinfo=timezone.utc)
def add_months(dt, n):
    y = dt.year + (dt.month-1 + n)//12
    m = (dt.month-1 + n)%12 + 1
    return dt.replace(year=y,month=m,day=1,hour=0,minute=0,second=0,microsecond=0)

def build_windows(start, end, train_m, test_m):
    s = month_floor(datetime.fromisoformat(start).replace(tzinfo=timezone.utc))
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    ts = add_months(s, train_m)  # test_start
    out=[]
    while True:
        te = add_months(ts, test_m) - timedelta(seconds=1)  # test_end
        if te > e: break
        tr0 = add_months(ts, -train_m)                      # TRAIN START
        out.append((tr0.date().isoformat(), te.date().isoformat()))
        ts = add_months(ts, test_m)
    return out
def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)  # space- or comma-separated
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-months", type=int, default=2)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--max-combos", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rth-only", action="store_true", default=False)  # DEFAULT FALSE
    ap.add_argument("--win-from", type=int, default=None)
    ap.add_argument("--win-to", type=int, default=None)
    args = ap.parse_args()

    wins = build_windows(args.start, args.end, args.train_months, args.test_months)
    a = args.win_from or 0
    b = (args.win_to+1) if args.win_to is not None else len(wins)
    wins = wins[a:b]
    base_idx = a
    print(f"Total WF windows to run: {len(wins)}")

    per = max(1, (os.cpu_count() or 8)//max(1,args.shards))
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        env[k] = str(per)
    py = os.getenv("PYTHON","python")

    for off,(tr_start, te_end) in enumerate(wins):
        wi = base_idx + off
        print(f"== Window {wi:02d} {tr_start}->{te_end}: launching shards")
        procs=[]
        for i in range(args.shards):
            cmd=[py,"run_backtest.py",
                 "--source","local",
                 "--symbols", args.symbols,
                 "--start", tr_start, "--end", te_end,
                 "--train-months", str(args.train_months),
                 "--test-months",  str(args.test_months),
                 "--max-combos",   str(args.max_combos),
                 "--run-tag", f"{args.run_tag}_w{wi:02d}_s{i}",
                 "--shard-index", str(i),
                 "--shard-count", str(args.shards),
                 "--seed", str(args.seed)]
            if args.rth_only:
                cmd.append("--rth-only")
            print("   launch:", " ".join(cmd))
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs: p.wait()
    print("✅ All requested windows finished.")

if __name__ == "__main__":
    main()
