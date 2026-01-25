#!/usr/bin/env python3
"""
analyze_reports.py — Headless bottom-line summarizer for your options runs.

Usage (equal-weight blend across runs):
  python analyze_reports.py \
    --inputs /path/to/run1.zip /path/to/run2.zip \
    --out /path/to/outdir

Usage (custom weights that sum to 1, same order as --inputs):
  python analyze_reports.py \
    --inputs run1.zip run2.zip run3.zip \
    --weights 0.5 0.3 0.2 \
    --out outdir
"""
import os
import sys
import argparse
import zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iter_reports(paths):
    for p in paths:
        p = str(p)
        label = os.path.basename(p).replace('.zip', '')
        if os.path.isdir(p):
            files = {
                'summary': os.path.join(p, 'summary.csv'),
                'trades': os.path.join(p, 'trades.csv'),
                'weekly_states': os.path.join(p, 'weekly_states.csv'),
                'per_symbol_contrib': os.path.join(p, 'per_symbol_contrib.csv'),
                'lifespans': os.path.join(p, 'lifespans.csv'),
            }
            yield label, ('dir', files)
        elif p.endswith('.zip'):
            z = zipfile.ZipFile(p, 'r')
            names = z.namelist()
            def pick(name):
                return next((n for n in names if n.endswith(name)), None)
            files = {k: pick(f'{k}.csv') for k in ['summary','trades','weekly_states','per_symbol_contrib','lifespans']}
            yield label, ('zip', (z, files))
        else:
            print('Skipping (not found/folder/zip):', p)


def _read_csv_from(source, key):
    kind, payload = source
    if kind == 'dir':
        path = payload.get(key)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return None
    else:  # zip
        z, files = payload
        name = files.get(key)
        if name:
            with z.open(name) as f:
                return pd.read_csv(f)
        return None


def _blended_headline(weekly_equs, out_dir, weights=None):
    """
    weekly_equs: list of (label, pd.Series daily_equity by date)
    weights: None=equal, else list of floats matching weekly_equs length
    """
    if not weekly_equs:
        return "", None

    df_eq = pd.concat([ser.rename(lbl) for lbl, ser in weekly_equs], axis=1)
    # require overlap across all series
    df_eq = df_eq.dropna()
    if df_eq.empty:
        return "<p>No overlapping dates across runs for blending.</p>", None

    labels = [lbl for lbl, _ in weekly_equs]
    if weights is not None:
        if len(weights) != len(weekly_equs):
            raise SystemExit("--weights length must match --inputs")
        w = pd.Series(weights, index=labels, dtype=float)
        w = w / w.sum()
    else:
        w = pd.Series(1.0/len(weekly_equs), index=labels)

    # normalize to 1.0 at first common date ⇒ blended index
    df_norm = df_eq / df_eq.iloc[0]
    blended = (df_norm * w).sum(axis=1)

    # metrics
    ret = blended.pct_change().dropna()
    cumret = float(blended.iloc[-1] - 1.0)
    days = (blended.index[-1] - blended.index[0]).days or 1
    years = days / 365.25
    cagr = float(blended.iloc[-1] ** (1/years) - 1.0) if years > 0 else float('nan')
    ann_vol = float(ret.std(ddof=1) * (252 ** 0.5)) if len(ret) > 1 else float('nan')
    sharpe = float(cagr / ann_vol) if ann_vol and ann_vol > 0 else float('nan')
    roll_max = blended.cummax()
    dd = float((blended / roll_max - 1.0).min())
    calmar = float(cagr / abs(dd)) if dd < 0 else float('inf')

    # chart
    fig = plt.figure()
    (blended * 100).plot(title='Blended Equity Index (100 = start)')
    fig.savefig(os.path.join(out_dir, 'blended_equity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    html = (
        f"<h2>Blended Headline (weights: {w.round(3).to_dict()})</h2>"
        f"<p><b>CAGR:</b> {cagr:.2%} &nbsp; <b>Sharpe:</b> {sharpe:.2f} &nbsp; "
        f"<b>MaxDD:</b> {dd:.2%} &nbsp; <b>Calmar:</b> {calmar:.2f}</p>"
        f"<div><img src='blended_equity.png' style='max-width:900px'></div>"
    )
    return html, {"cagr": cagr, "sharpe": sharpe, "max_dd": dd, "calmar": calmar, "cum_return": cumret}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='Report folders or .zip files.')
    ap.add_argument('--out', required=True, help='Output directory for HTML and charts.')
    ap.add_argument('--weights', nargs='*', type=float, default=None, help='Optional weights (must sum to 1).')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs = []
    for label, src in _iter_reports(args.inputs):
        runs.append({'label': label, 'source': src})

    all_summaries, all_contrib, weekly_equs = [], [], []

    for run in runs:
        sm = _read_csv_from(run['source'], 'summary')
        if sm is not None:
            sm['run'] = run['label']
            all_summaries.append(sm)

        ws = _read_csv_from(run['source'], 'weekly_states')
        if ws is not None and not ws.empty and 'date' in ws.columns and 'equity' in ws.columns:
            try:
                ws['date'] = pd.to_datetime(ws['date'])
                daily = ws.groupby('date')['equity'].last().sort_index()
                if not daily.empty:
                    weekly_equs.append((run['label'], daily))
            except Exception:
                pass

        cs = _read_csv_from(run['source'], 'per_symbol_contrib')
        if cs is not None:
            cs['run'] = run['label']
            all_contrib.append(cs)

    if not all_summaries:
        print('No summaries found. Exiting.')
        sys.exit(0)

    df_sum = pd.concat(all_summaries, ignore_index=True)

    # Back-compat: ensure metric columns exist even for older runs
    for col in ['cagr','sharpe','max_dd','calmar','daily_win_rate','weekly_win_rate']:
        if col not in df_sum.columns:
            df_sum[col] = pd.NA

    df_contrib = pd.concat(all_contrib, ignore_index=True) if all_contrib else None

    # Chart 1: Mean Sharpe by run (if present)
    fig1 = plt.figure()
    if 'sharpe' in df_sum.columns and df_sum['sharpe'].notna().any():
        df_sum.groupby('run')['sharpe'].mean().dropna().plot(kind='bar', title='Mean Sharpe by Run')
        fig1.savefig(os.path.join(args.out, 'mean_sharpe_by_run.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Chart 2: Mean Test CAGR by run (optimizer contrib only)
    if df_contrib is not None and not df_contrib.empty and 'mean_test_cagr' in df_contrib.columns:
        fig2 = plt.figure()
        df_contrib.groupby('run')['mean_test_cagr'].mean().dropna().plot(kind='bar', title='Mean Test CAGR by Run')
        fig2.savefig(os.path.join(args.out, 'mean_test_cagr_by_run.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)

    # Chart 3: Max DD histogram for first run (if present)
    fig3 = plt.figure()
    r0 = df_sum['run'].iloc[0]
    if 'max_dd' in df_sum.columns and df_sum['max_dd'].notna().any():
        (df_sum[df_sum['run'] == r0]['max_dd']).plot(kind='hist', bins=20, title=f'Max DD Distribution — {r0}')
        fig3.savefig(os.path.join(args.out, f'maxdd_hist_{r0}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # Aggregate per-run table
    # Build df_contrib if we have any per_symbol_contrib inputs
    df_contrib = pd.concat(all_contrib, ignore_index=True) if all_contrib else None

    # Core metrics come from df_sum (summary.csv from either metrics runs or WF runs)
    agg_metrics = df_sum.groupby('run').agg(
        mean_cagr=('cagr','mean') if 'cagr' in df_sum.columns else ('cagr','mean'),
        mean_sharpe=('sharpe','mean') if 'sharpe' in df_sum.columns else ('sharpe','mean'),
        mean_maxdd=('max_dd','mean') if 'max_dd' in df_sum.columns else ('max_dd','mean'),
        mean_calmar=('calmar','mean') if 'calmar' in df_sum.columns else ('calmar','mean'),
        mean_win_daily=('daily_win_rate','mean') if 'daily_win_rate' in df_sum.columns else ('daily_win_rate','mean'),
        mean_win_weekly=('weekly_win_rate','mean') if 'weekly_win_rate' in df_sum.columns else ('weekly_win_rate','mean'),
    ).reset_index()

    # Symbol counts: prefer per_symbol_contrib when available; else fall back if df_sum has a symbol column.
    sym_counts = None
    if df_contrib is not None and not df_contrib.empty and 'symbol' in df_contrib.columns:
        sym_counts = df_contrib.groupby('run')['symbol'].nunique().rename('symbols').reset_index()
    elif 'symbol' in df_sum.columns:
        sym_counts = df_sum.groupby('run')['symbol'].nunique().rename('symbols').reset_index()

    if sym_counts is not None:
        agg = agg_metrics.merge(sym_counts, on='run', how='left')
    else:
        agg = agg_metrics.copy()
        agg['symbols'] = pd.NA
    agg = df_sum.groupby('run').agg(
        symbols=('symbol','nunique'),
        mean_cagr=('cagr','mean'),
        mean_sharpe=('sharpe','mean'),
        mean_maxdd=('max_dd','mean'),
        mean_calmar=('calmar','mean'),
        mean_win_daily=('daily_win_rate','mean'),
        mean_win_weekly=('weekly_win_rate','mean'),
    ).reset_index()

    # Blended headline (equal weights by default)
    headline_html, _ = _blended_headline(weekly_equs, args.out, args.weights)

    # HTML
    parts = []
    parts.append('<h1>Options Bottom Line — Summary</h1>')
    parts.append('<h2>Runs</h2><ul>' + ''.join(f'<li>{r["label"]}</li>' for r in runs) + '</ul>')
    parts.append('<h2>Aggregate Metrics</h2>')
    parts.append(agg.to_html(index=False))
    if headline_html:
        parts.append(headline_html)

    for fn in ['mean_sharpe_by_run.png','mean_test_cagr_by_run.png', f'maxdd_hist_{r0}.png']:
        fpath = os.path.join(args.out, fn)
        if os.path.exists(fpath):
            parts.append(f'<div><img src="{fn}" style="max-width:900px"></div>')

    out_html = os.path.join(args.out, 'bottom_line_summary.html')
    with open(out_html,'w') as f:
        f.write('\n'.join(parts))
    print('Wrote', out_html)


if __name__ == "__main__":
    main()
