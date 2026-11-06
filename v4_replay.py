"""Reproducible retrospective V1--V4 replay; never writes baseline histories.

    python v4_replay.py --manifest research/v4_inputs/capture-20261105/manifest.json
    python v4_replay.py --capture research/v4_inputs/NEW_IMMUTABLE_CAPTURE

The chronological tail is ALREADY OBSERVED research, not an untouched test.
No parameter winner is selected, and alternative exit models run on development
sessions only. A genuinely untouched test begins with future frozen signals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import math
import concurrent.futures

import numpy as np
import pandas as pd

from quant_features import BAR_DURATION, NY, PreparedIntraday, daily_features, intraday_features, regular_bars
from quant_execution import ExecutionConfig, simulate_portfolio
from strategy_v2 import score_v2
from strategy_v3 import classify_regime, score_v3
from strategy_v4 import CONFIG_HASH, evaluate_v4

DEFAULT_MANIFEST = Path('research/v4_inputs/capture-20261105/enriched-manifest.json')
OUTPUT = Path('docs/data/v4-replay.json')
CHECKPOINT_MINUTES = tuple(range(600, 931, 30))  # completed cutoffs 10:00--15:30 NY
PRIMARY = ExecutionConfig(max_total_risk_fraction=.025,max_sector_fraction=None)
MODELS = {
    'signal-1R': {}, 'signal-1.5R': {'target_r':1.5}, 'signal-2R': {'target_r':2.},
    'atr-half-1R': {'stop_model':'atr','atr_multiple':.5},
    'fixed-2pct-1R': {'stop_model':'fixed','fixed_stop_pct':.02},
    'trailing-1pct': {'target_r':None,'trailing_pct':.01},
    'time-five-sessions': {'target_r':None},
    'time-one-session': {'target_r':None,'max_holding_sessions':1},
    'time-three-sessions': {'target_r':None,'max_holding_sessions':3},
    'trend-ma20': {'target_r':None,'trend_exit_column':'trend_exit'},
    'partial-half-1R-final-2R': {'target_r':2.,'partial_fraction':.5,'partial_target_r':1.},
    'sizing-notional-cap': {'risk_fraction':1.},
    'sizing-quarter-percent-risk': {'risk_fraction':.0025},
}


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    return value


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(destination):
    """Download once into a NEW directory; a manifest is a sealed input vintage."""
    import yfinance as yf
    root=Path(destination)
    if root.exists() and any(root.iterdir()):raise ValueError('Capture directory must be new/empty; existing vintages are immutable')
    root.mkdir(parents=True,exist_ok=True)
    symbols=sorted(set(json.loads(Path('skew_config.json').read_text())['core_universe'])|{'SPY','QQQ'})
    started=datetime.now(timezone.utc).isoformat()
    def fetch(item):
        symbol,interval,period=item
        frame=yf.Ticker(symbol).history(period=period,interval=interval,auto_adjust=False,prepost=False,actions=True)
        if frame.empty:raise ValueError('Empty Yahoo response')
        raw=frame.to_csv(index_label='Datetime').encode();archive=gzip.compress(raw,mtime=0)
        path=root/f'{symbol}-{interval}.csv.gz';path.write_bytes(archive)
        return {'symbol':symbol,'interval':interval,'period':period,'path':str(path),'sha256':hashlib.sha256(archive).hexdigest(),
                'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(frame),
                'first':frame.index.min().isoformat(),'last':frame.index.max().isoformat()}
    records=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch,(s,i,p)):(s,i,p) for s in symbols for i,p in [('5m','60d'),('1d','1y')]}
        for future in concurrent.futures.as_completed(futures):
            try:records.append(future.result())
            except Exception as exc:errors.append({'request':futures[future],'error':str(exc)})
    result={'schema_version':1,'capture_started_at':started,'captured_at':datetime.now(timezone.utc).isoformat(),
            'source':'Yahoo Finance via yfinance','yfinance_version':yf.__version__,
            'adjustment':'auto_adjust=False; raw OHLC; dividends and splits archived when returned',
            'universe':symbols,'universe_provenance':'Current core_universe; not point-in-time constituents; survivorship/selection bias remains',
            'records':sorted(records,key=lambda x:(x['symbol'],x['interval'])),'errors':errors}
    path=root/'manifest.json';path.write_text(json.dumps(result,indent=2)+'\n');return path


def load_archive(manifest_path):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());frames={}
    for record in manifest['records']:
        path=Path(record['path'])
        if not path.exists():path=manifest_path.parent/path.name
        if digest_file(path)!=record['sha256']:raise ValueError(f'Archive SHA256 mismatch: {path}')
        raw=gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest()!=record['uncompressed_sha256']:raise ValueError('Uncompressed digest mismatch')
        from io import BytesIO
        frame=pd.read_csv(BytesIO(raw),index_col=0)
        frame.index=pd.to_datetime(frame.index,utc=True)
        if len(frame)!=record['rows']:raise ValueError('Manifest row count mismatch')
        frames[(record['symbol'],record['interval'])]=frame
    if not {('SPY','1d'),('SPY','5m'),('QQQ','5m')}.issubset(frames):raise ValueError('Missing benchmark observations')
    return manifest,frames


def chronological_windows(sessions,purge_sessions=5):
    """Two explicit five-session no-evaluation gaps; outcomes never cross windows."""
    dates=sorted(set(sessions));n=len(dates)
    if n<30:raise ValueError('At least 30 sessions required for purged chronological windows')
    train_end=n//2
    validation_start=train_end+purge_sessions
    test_start=(n*3)//4+purge_sessions
    windows={'development':dates[:train_end],
             'validation':dates[validation_start:(n*3)//4],
             'retrospective_tail':dates[test_start:]}
    gaps=dates[train_end:validation_start]+dates[(n*3)//4:test_start]
    return windows,gaps


def collect_signals(manifest,frames):
    # Import the unchanged baseline V1 scorer only; never run any publisher main.
    from day_opportunities import score_row
    captured=pd.Timestamp(manifest['captured_at']);local_capture=captured.tz_convert(NY)
    symbols=manifest['universe'];all_dates=set();signals=defaultdict(list);reasons=Counter();packets=0
    daily_returns = pd.concat({symbol:pd.Series(frame.Close.to_numpy(),index=pd.DatetimeIndex(frame.index).tz_convert(NY).date) for (symbol,interval),frame in frames.items() if interval=='1d'},axis=1).pct_change(fill_method=None)
    correlation_cache={};daily_cache={};market_cache={}
    for symbol in symbols:
        if (symbol,'5m') not in frames or (symbol,'1d') not in frames:continue
        bars=regular_bars(frames[(symbol,'5m')]);prepared=PreparedIntraday(bars);dates=bars.index.tz_convert(NY).date
        # Use complete market sessions only; today's unfinished session is not a full test day.
        sessions=[d for d in sorted(set(dates)) if d<local_capture.date() or (d==local_capture.date() and local_capture.hour>=16)]
        for date in sessions:
            all_dates.add(date);seen=set()
            for minute in CHECKPOINT_MINUTES:
                at=pd.Timestamp(date,tz=NY)+pd.Timedelta(minutes=minute)
                if at>captured:continue
                if (symbol,date) not in daily_cache:
                    daily_cache[(symbol,date)]=daily_features(frames[(symbol,'1d')],at)
                if date not in market_cache:
                    market_cache[date]=daily_features(frames[('SPY','1d')],at)
                daily,daily_asof=daily_cache[(symbol,date)]
                daily=dict(daily)
                market,market_asof=market_cache[date]
                intra,intra_asof=intraday_features(bars,daily,at,prepared=prepared)
                if not daily or not market or not intra:continue
                daily['relative_strength_20d']=daily['return_20d']-market['return_20d']
                context={'signal_at':at.isoformat(),'daily_asof':daily_asof,'market_asof':market_asof,'intraday_asof':intra_asof}
                evaluated=evaluate_v4(daily,intra,market,context=context);packets+=1
                for check in evaluated['checks']:
                    if not check['passed']:reasons[check['name']]+=1
                if symbol in ('SPY','QQQ'):
                    # Benchmarks participate in scanner universe exactly as current public config.
                    pass
                legacy_intra={**intra,'momentum_15m':intra['legacy_momentum_10m']}
                candidates={1:score_row(daily,legacy_intra),2:score_v2(daily,legacy_intra),3:score_v3(daily,legacy_intra,market)}
                candidates[4]=(evaluated['state'],evaluated['score'],'; '.join(evaluated.get('rejections') or evaluated.get('reasons',[])),evaluated['stop'],evaluated['target_1r'],evaluated['target_2r'])
                baseline=bool(daily['prev_close']>daily['ma50']>daily.get('ma200',np.inf) and market['price']>market.get('ma200',np.inf))
                candidates['trend']=(('ENTRY TRIGGERED' if baseline else 'NO TRADE'),50.,'prior daily close > MA50 > MA200; SPY > MA200',intra['price']*(1-.5*daily['atr_pct']),None,None)
                for strategy,result in candidates.items():
                    if strategy in seen or result[0]!='ENTRY TRIGGERED':continue
                    seen.add(strategy);version=f'V{strategy}' if strategy!='trend' else 'TREND'
                    if date not in correlation_cache:
                        prior_returns=daily_returns[daily_returns.index<date].tail(60)
                        correlation_cache[date]=prior_returns.corr(min_periods=40)
                    correlations=correlation_cache[date]
                    correlation_map={other:float(correlations.loc[symbol,other]) for other in symbols if other!=symbol and symbol in correlations.index and other in correlations.columns and np.isfinite(correlations.loc[symbol,other])}
                    row={'signal_id':f'replay|{version}|{symbol}|{date}','symbol':symbol,'strategy':version,
                         'signal_time':at.tz_convert('UTC').isoformat(),'stop_price':result[3],
                         'atr':daily['atr_pct']*daily['prev_close'],'score':result[1],'rationale':result[2],
                         'market_regime':classify_regime(market),'context':context,
                         'features':{'daily':daily,'intra':legacy_intra if strategy in (1,2,3) else intra,'market':market},'correlations':correlation_map,
                         'config_hash':CONFIG_HASH if strategy==4 else None,'retrospective':True}
                    if strategy==4:row['checks']=evaluated['checks']
                    signals[version].append(clean(row))
        print('Features',symbol,len(signals.get('V4',[])),flush=True)
    return dict(signals),sorted(all_dates),{'feature_packets':packets,'v4_failed_check_counts':dict(reasons)}


def block_interval(trades,sessions,seed=4319,resamples=2000):
    closed=[t for t in trades if t['status']=='CLOSED' and t.get('net_r') is not None]
    dates=sorted(sessions);byday=defaultdict(list)
    for t in closed:byday[pd.Timestamp(t['entry_time']).tz_convert(NY).date()].append(t['net_r'])
    base={'method':'moving five-session block bootstrap; entry-date clusters; descriptive, not proof of alpha',
          'block_sessions':5,'session_count':len(dates),'closed_trades':len(closed),'resamples':resamples,'seed':seed}
    if len(closed)<10 or len(dates)<10:
        return {**base,'status':'insufficient observations','lower':None,'upper':None}
    block_values=[sum((byday[d] for d in dates[start:start+5]),[]) for start in range(len(dates)-4)]
    rng=np.random.default_rng(seed);means=[]
    for _ in range(resamples):
        blocks=rng.integers(0,len(block_values),size=math.ceil(len(dates)/5))
        values=sum((block_values[b] for b in blocks),[])
        if values:means.append(float(np.mean(values)))
    return {**base,'status':'descriptive; few independent blocks' if len(dates)<50 else 'descriptive',
            'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975))}


def window_bars(frames,sessions):
    selected=set(sessions);result={}
    for (symbol,interval),bars in frames.items():
        if interval!='5m':continue
        d=regular_bars(bars);d=d[np.isin(d.index.tz_convert(NY).date,list(selected))].copy()
        # Known-at-close trend stop; the engine executes this on the following open.
        daily=frames[(symbol,'1d')]
        averages={day:daily_features(daily,pd.Timestamp(day,tz=NY)+pd.Timedelta(hours=10))[0].get('ma20',np.nan) for day in selected}
        d['trend_exit']=[float(close)<averages[day] for close,day in zip(d.Close,d.index.tz_convert(NY).date)]
        result[symbol]=d
    return result


def select_signals(signals,sessions):
    dates=set(sessions)
    return [s for s in signals if pd.Timestamp(s['signal_time']).tz_convert(NY).date() in dates]


def benchmark(symbol,bars,sessions):
    d=bars[symbol];friction=(PRIMARY.slippage_bps+PRIMARY.spread_bps/2)/10000
    raw_entry=float(d.Open.iloc[0]);entry=raw_entry*(1+friction)
    quantity=PRIMARY.initial_capital/entry;cash=0.;equity=[]
    for date,group in d.groupby(d.index.tz_convert(NY).date):
        value=quantity*float(group.Close.iloc[-1]);equity.append({'time':(group.index[-1]+BAR_DURATION).isoformat(),'equity':value})
    # Liquidate final benchmark at close with the same adverse round-trip friction.
    equity[-1]['equity']*=1-friction
    values=np.array([PRIMARY.initial_capital]+[p['equity'] for p in equity]);dd=1-values/np.maximum.accumulate(values)
    return {'strategy':symbol,'metrics':{'starting_capital':PRIMARY.initial_capital,'ending_equity':values[-1],
            'total_return':values[-1]/PRIMARY.initial_capital-1,'max_drawdown':float(dd.max()),
            'exposure':1.,'trades':1,'annualized_return':None},'equity':equity,
            'assumptions':'Fractional fully invested buy/hold, first session open to final close, 7bps adverse slippage + 1bp half-spread per side; dividend cash excluded (price return); exposure differs from tactical strategies'}


def feature_diagnostics(signals,trades):
    lookup={s['signal_id']:s for s in signals};groups=defaultdict(list)
    for t in trades:
        if t['status']!='CLOSED':continue
        s=lookup[t['signal_id']];i=s['features']['intra'];d=s['features']['daily']
        for name,value,edges in [('rsi14',d.get('rsi14'),[40,60,70]),('relative_strength_20d',d.get('relative_strength_20d'),[0,.05,.15]),
                                 ('vwap_dist',i.get('vwap_dist'),[0,.005,.015]),('volume_ratio_tod',i.get('volume_ratio_tod'),[1.4,2.5,4]),
                                 ('entry_hour_ny',pd.Timestamp(s['signal_time']).tz_convert(NY).hour,[11,13,15])]:
            if value is None:continue
            bucket=int(np.searchsorted(edges,value,side='right'));groups[(name,bucket)].append(t['net_r'])
        groups[('market_regime',s['market_regime'])].append(t['net_r'])
    return [{'feature':name,'bucket':bucket,'closed_trades':len(values),'mean_net_r':float(np.mean(values)),
             'interpretation':'descriptive association; correlated trades, multiple comparisons and no causal identification'}
            for (name,bucket),values in sorted(groups.items(),key=lambda x:str(x[0]))]


def run(manifest_path=DEFAULT_MANIFEST):
    manifest,frames=load_archive(manifest_path);signals,sessions,coverage=collect_signals(manifest,frames)
    windows,gaps=chronological_windows(sessions);results=[];benchmarks=[];experiments=[]
    for name,dates in windows.items():
        bars=window_bars(frames,dates)
        for version in ('V1','V2','V3','V4','TREND'):
            source=select_signals(signals.get(version,[]),dates)
            result=simulate_portfolio(source,bars,config=PRIMARY)
            result.update({'strategy':version,'window':name,'signals':source,'confidence_interval':block_interval(result['trades'],dates),
                           'feature_diagnostics':feature_diagnostics(source,result['trades'])})
            results.append(result)
            print(name,version,result['metrics'].get('closed_trades'),result['metrics'].get('net_expectancy_r'),flush=True)
        for symbol in ('SPY','QQQ'):benchmarks.append({'window':name,**benchmark(symbol,bars,dates)})
        elapsed=(dates[-1]-dates[0]).days
        cash_ending=PRIMARY.initial_capital*(1+PRIMARY.cash_annual_rate)**(elapsed/365.25)
        benchmarks.append({'window':name,'strategy':'CASH','metrics':{'starting_capital':PRIMARY.initial_capital,'ending_equity':cash_ending,'total_return':cash_ending/PRIMARY.initial_capital-1,'max_drawdown':0},
                           'assumptions':'Nominal uninvested USD cash; 0% interest and no fees; not a Treasury total-return benchmark'})
        if name=='development':
            source=select_signals(signals.get('V4',[]),dates)
            for model,changes in MODELS.items():
                cfg=replace(PRIMARY,**changes);trial=simulate_portfolio(source,bars,config=cfg)
                experiments.append({'id':model,'strategy':'V4','window':'development only','status':'exploratory; no winner selected',
                                    'metrics':trial['metrics'],'config':asdict(cfg),'confidence_interval':block_interval(trial['trades'],dates)})
    experiment_config={'checkpoints_minutes_ny':CHECKPOINT_MINUTES,'primary_execution':asdict(PRIMARY),'development_models':MODELS,'window_fractions':[.5,.25,.25],'purge_embargo_sessions':5,'historical_status':'already observed retrospective'}
    experiment_config_hash=hashlib.sha256(json.dumps(experiment_config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    sources={name:digest_file(name) for name in ['quant_features.py','v4_replay.py','quant_execution.py','market_sessions.py','strategy_v4.py','research/v4_config.json','strategy_v2.py','strategy_v3.py','day_opportunities.py']}
    return clean({'schema_version':1,'generated_at':manifest['captured_at'],'market_data_at':(pd.Timestamp(sessions[-1],tz=NY)+pd.Timedelta(hours=16)).isoformat(),'status':'research-only; retrospective replay, no independent profitability claim',
                  'config_hash':CONFIG_HASH,'experiment_config':experiment_config,'experiment_config_hash':experiment_config_hash,'manifest_path':str(manifest_path),'manifest_sha256':digest_file(manifest_path),'source_sha256':sources,
                  'coverage':{**coverage,'sessions':len(sessions),'symbols':len(manifest['universe']),'archive_errors':manifest['errors']},
                  'validation':{'status':'inconclusive; historical period and universe already observed','untouched_test_status':'future only; zero observations',
                                'chronological_windows':{k:[str(d) for d in v] for k,v in windows.items()},'purge_embargo_sessions':[str(d) for d in gaps],
                                'purge_session_count_per_boundary':5,'walk_forward':'Fixed rules evaluated on successive development, validation and retrospective-tail windows; no fitting or parameter selection',
                                'limitations':['All historical windows already inspected; retrospective tail is not untouched out-of-sample evidence',
                                  'Current universe, not historical constituents: survivorship and selection bias remain',
                                  'Yahoo current snapshot is revised/split-adjusted, not a point-in-time vendor vintage',
                                  'Only about 60 intraday sessions; few independent date blocks and regimes',
                                  'No archived intraday quotes, earnings calendar, options chains or sector history; correlations use at least40 prior daily returns in a60-session window',
                                  'Fixed 2bps spread is an assumption, not observed bid/ask; penny rounding and market impact unavailable',
                                  'OHLC stop/target collisions pessimistically resolve stop first; actual intrabar sequencing unknown',
                                  'Unchanged V1-V3 scoring uses their legacy10-minute momentum feature despite its15m name; V4 uses a true15-minute close-to-close move',
                                  'Window-end open positions are right-censored and marked, never resolved with later-window prices',
                                  'Cash uses0% nominal (historical risk-free yield series unavailable); SPY/QQQ are price-return benchmarks',
                                  'Feature bucket analyses and exit comparisons are exploratory multiple comparisons, not causal validation']},
                  'primary_execution_config':asdict(PRIMARY),'comparisons':results,'benchmarks':benchmarks,'experiments':experiments,
                  'prospective':{'status':'reserved future evaluation; see v4-forward.json','signals':[],'trades':[]}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);parser.add_argument('--capture',type=Path);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.capture:print(capture(args.capture))
    else:
        result=run(args.manifest);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(args.output)
