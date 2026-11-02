"""Read-only diagnostics of immutable baseline trades. Never promotes a strategy.

Chronological date split and date-block bootstrap are descriptive: these trades
were already observed during strategy development. They are NOT untouched OOS.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def quantile(values, p):
    s = sorted(values)
    return s[round((len(s)-1)*p)] if s else None


def describe(rows, cost_r=0.05):
    resolved = [r for r in rows if r.get('trade_result_1r') in ('WIN_1R', 'STOP')]
    values = [float(r['realized_r_1r'])-cost_r for r in resolved]
    blocks = defaultdict(list)
    for r, value in zip(resolved, values):
        blocks[r['signal_date']].append(value)
    rng = random.Random(2026)
    groups = list(blocks.values())
    boot = []
    if len(groups) >= 5:
        for _ in range(2000):
            sampled = [v for _ in groups for v in rng.choice(groups)]
            boot.append(sum(sampled)/len(sampled))
    positive = sum(v for v in values if v > 0)
    negative = -sum(v for v in values if v < 0)
    return {'signals': len(rows), 'resolved': len(values), 'unresolved_or_ambiguous':len(rows)-len(values),
            'dates':len(groups), 'expectancy_after_assumed_cost_r':sum(values)/len(values) if values else None,
            'median_after_assumed_cost_r':quantile(values,.5),
            'profit_factor_after_assumed_cost':positive/negative if negative else None,
            'date_block_bootstrap_95_interval_r':[quantile(boot,.025),quantile(boot,.975)] if boot else None}


def diagnostics(rows):
    dates = sorted({r['signal_date'] for r in rows})
    if len(dates)<2:
        raise ValueError('Need at least two signal dates')
    split = dates[max(1,int(len(dates)*.7))]
    strategies = {}
    for version in sorted({r['strategy'] for r in rows}):
        r = [x for x in rows if x['strategy']==version]
        strategies[version] = {'all':describe(r),
             'earlier_70pct_dates':describe([x for x in r if x['signal_date']<split]),
             'later_30pct_dates':describe([x for x in r if x['signal_date']>=split]),
             'exploratory_volume_tod_ge_1_5':describe([x for x in r if x.get('volume_ratio_tod') and float(x['volume_ratio_tod'])>=1.5])}
    return {'schema_version':1, 'source':'data/backtest_trades.csv', 'split_date':split,
            'cost_assumption_r':.05, 'bootstrap':'2000 resamples of complete signal-date blocks, fixed seed 2026',
            'status':'descriptive_only_not_untouched_out_of_sample',
            'limitations':['Versions were developed using observed data; this retrospective date split is not prospective validation.',
              'Overlapping multi-day positions induce dependence beyond date blocks.',
              'Execution, exposure, benchmark returns and currency P/L cannot be reconstructed from per-signal R alone.',
              'A current configured universe may introduce survivorship/selection bias.',
              'Unresolved and ambiguous records excluded from resolved metrics; coverage reported explicitly.',
              'Volume threshold is an exploratory hypothesis, not a new production strategy or evidence of improvement.'],
            'strategies':strategies}

if __name__ == '__main__':
    rows = list(csv.DictReader(Path('data/backtest_trades.csv').open()))
    Path('docs/data/research-diagnostics.json').write_text(json.dumps(diagnostics(rows),indent=2,allow_nan=False)+'\n')
