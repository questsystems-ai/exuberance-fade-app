#!/usr/bin/env python3
"""
wf_driver_multishard.py
Launches walk-forward windows across multiple shards in parallel.

- Clean --seed handling (optionally offset per shard)
- Does NOT force --rth-only (pass it via --extra-args if you want)
- Concurrency control via --max-procs
"""

import argparse, subprocess, sys, time, os
from datetime import datetime
import shlex
import pandas as pd

def month_iter(start: str, end: str):
    """Yield YYYY-MM strings from start to end inclusive."""
    d1 = pd.to_datetime(start).to_period("M")
    d2 = pd.to_datetime(end).to_period("M")
    cur = d1
    while cur <= d2:
        yield str(cur)
        cur += 1

def build_cmd(py_exec: str,
              run_backtest: str,
              symbols: str,
              source: str,
              train_m: int,
              test_m: int,
              train_start_month: str,
              test_end_month: str,
              shard_index: int,
              shard_count: int,
              max_combos: int,
              seed: int,
              run_tag: str,
              extra_args: list[str] | None):
    cmd = [
        py_exec, run_backtest,
        "--symbols", symbols,
        "--source", source,
        "--start", f"{train_start_month}-01",
        "--end",   f"{test_end_month}-28",  # safe last day
        "--train-months", str(train_m),
        "--test-months",  str(test_m),
        "--max-combos",   str(max_combos),
        "--seed",         str(seed),
        "--run-tag",      f"{run_tag}_S{shard_index}of{shard_count}",
        "--shard-index",  str(shard_index),
        "--shard-count",  str(shard_count),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd

def main():
    ap = argparse.ArgumentParser(description="Multi-shard WF driver")
    ap.add_argument("--run-backtester", default="run_backtest.py",
                    help="Path to run_backtest.py (default: run_backtest.py in CWD)")
    ap.add_argument("--symbols", required=True,
                    help="Symbols string (quoted, space or comma separated)")
    ap.add_argument("--source", default="local", choices=["local","alpaca","polygon"])
    ap.add_argument("--train-months", type=int, default=2)
    ap.add_argument("--test-months",  type=int, default=1)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end",   default="2025-12-31")
    ap.add_argument("--max-combos", type=int, default=50)

    ap.add_argument("--shard-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337,
                    help="Base seed for all shards")
    ap.add_argument("--offset-seed-per-shard", action="store_true",
                    help="If set, seed becomes seed + shard_index")
    ap.add_argument("--max-procs", type=int, default=8,
                    help="Max concurrent shard processes")

    ap.add_argument("--run-tag", default="wfms")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable to use")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER,
                    help="Extra args to forward to run_backtest.py (e.g., --rth-only --no-report)")
    args = ap.parse_args()

    months = list(month_iter(args.start, args.end))
    windows = len(months) - args.train_months - args.test_months + 1
    if windows <= 0:
        print("No windows in this range.")
        sys.exit(0)

    # Clean extra args (argparse.REMAINDER may include a leading --)
    extra = []
    if args.extra_args:
        # On some shells the first element might be '--', drop it.
        extra = [a for a in args.extra_args if a != "--"]

    procs: list[tuple[subprocess.Popen, str]] = []  # (proc, pretty_label)

    for w in range(windows):
        train_start = months[w]
        test_end    = months[w + args.train_months + args.test_months - 1]
        print(f"\n=== Window {w+1}/{windows}: train_start={train_start}, test_end={test_end} ===")

        for shard_idx in range(args.shard_count):
            # Concurrency gate
            while len(procs) >= max(1, args.max_procs):
                # Poll & reap finished
                still: list[tuple[subprocess.Popen, str]] = []
                for p, label in procs:
                    ret = p.poll()
                    if ret is None:
                        still.append((p, label))
                    else:
                        print(f"[done {ret:>3}] {label}")
                procs = still
                if len(procs) >= args.max_procs:
                    time.sleep(0.25)

            seed = args.seed + shard_idx if args.offset_seed_per_shard else args.seed

            cmd = build_cmd(
                py_exec=args.python,
                run_backtest=args.run_backtester,
                symbols=args.symbols,
                source=args.source,
                train_m=args.train_months,
                test_m=args.test_months,
                train_start_month=train_start,
                test_end_month=test_end,
                shard_index=shard_idx,
                shard_count=args.shard_count,
                max_combos=args.max_combos,
                seed=seed,
                run_tag=args.run_tag,
                extra_args=extra,
            )

            label = " ".join(shlex.quote(c) for c in cmd)
            print(f"[spawn] {label}")
            # Note: shell=False; works on Windows/PowerShell too.
            p = subprocess.Popen(cmd)
            procs.append((p, label))

        # Optional: wait for all shards in this window before moving on
        # Comment out if you want a rolling pipeline across windows
        while procs:
            still: list[tuple[subprocess.Popen, str]] = []
            for p, label in procs:
                ret = p.poll()
                if ret is None:
                    still.append((p, label))
                else:
                    print(f"[done {ret:>3}] {label}")
            procs = still
            if procs:
                time.sleep(0.25)

    print("\nAll windows complete.")

if __name__ == "__main__":
    main()
