"""Isolated event-sourced prospective V1–V4 local paper experiment. No broker API.

Registration is not evidence. Only decisions frozen at actual collection time can
enter future completed bars. Replays use immutable first-seen observations; any
revision of a previously reported execution pauses publication instead of
rewriting history. Legacy paper state/CSV files are never opened for writing.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import pandas as pd
from day_opportunities import score_row
from strategy_v2 import score_v2
from strategy_v3 import score_v3,classify_regime
from strategy_v4 import evaluate_v4,CONFIG_HASH,STRATEGY_VERSION
from quant_features import feature_packet,regular_bars,utc_timestamp
from quant_execution import ExecutionConfig,simulate_portfolio
from market_sessions import session_bounds,completed_bar

VERSIONS=('V1','V2','V3','V4')
CODE_FILES=('strategy_v4.py','research/v4_config.json','research/v4_constraints.txt','requirements.txt','skew_config.json','quant_features.py','quant_execution.py','quant_forward.py','market_sessions.py','strategy_v2.py','strategy_v3.py','day_opportunities.py')

def runtime_environment():
    return {'python':sys.version.split()[0],'packages':{name:metadata.version(name) for name in ('pandas','numpy','yfinance','exchange_calendars')}}

def timestamp():return pd.Timestamp.now(tz='UTC')
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if hasattr(v,'item'):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,(pd.Timestamp,datetime)):return v.isoformat()
    return v

def digest(v):return hashlib.sha256(json.dumps(clean(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read_events(path):
    if not path.exists():return []
    events=[]
    def invalid_constant(value):raise ValueError(f'Invalid non-finite JSON value: {value}')
    for row,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:event=json.loads(line,parse_constant=invalid_constant)
        except ValueError as exc:raise ValueError(f'Malformed immutable ledger {path.name}, row {row}: {exc}') from exc
        if not isinstance(event,dict):raise ValueError(f'Malformed immutable ledger {path.name}, row {row}: expected an object')
        events.append(event)
    return events

def validate_experiment_events(events):
    """Reject incomplete/corrupt state instead of silently starting a new cohort."""
    identities=set();collections=0
    for event in events:
        kind=event.get('kind')
        if kind not in {'registration','collection_started','decision','execution_observation'}:raise ValueError('Unknown or missing immutable experiment event kind')
        utc_timestamp(event['recorded_at'])
        if kind=='collection_started':collections+=1
        if kind=='decision':
            identity=event['decision_id']
            if identity in identities or identity!=event['signal_id']:raise ValueError('Duplicate or inconsistent immutable decision identity')
            identities.add(identity)
            if event['strategy'] not in VERSIONS or not isinstance(event['accepted'],bool) or not event['symbol']:raise ValueError('Invalid immutable decision fields')
            if utc_timestamp(event['signal_time'])>utc_timestamp(event['frozen_at']):raise ValueError('Decision availability precedes its source cutoff')
        if kind=='execution_observation' and event['observation_id']!=digest({'strategy':event['strategy'],'trade':event['trade']}):raise ValueError('Immutable execution observation checksum failed')
    if collections>1:raise ValueError('Multiple collection starts would reset the prospective cohort')
def append_events(path,events):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as out:
        for event in events:out.write(json.dumps(clean(event),sort_keys=True,allow_nan=False)+'\n')
        out.flush()

def execution_config():
    return ExecutionConfig(initial_capital=10000,risk_fraction=.005,max_position_dollars=1000,max_open_positions=5,max_total_risk_fraction=.025,max_sector_fraction=None,slippage_bps=7,spread_bps=2,cash_annual_rate=0,max_holding_sessions=5)

def register(root=Path('.'),at=None):
    at=utc_timestamp(at or timestamp());directory=root/'data/quant_forward';path=directory/'manifest.json'
    code={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in CODE_FILES}
    runtime=runtime_environment()
    if path.exists():
        result=json.loads(path.read_text())
        if result['config_hash']!=CONFIG_HASH or result['code_hashes']!=code:raise ValueError('Frozen prospective experiment code/config changed. Register a separate experiment; do not reset this cohort.')
        if result.get('runtime')!=runtime:raise ValueError('Frozen prospective runtime changed. Restore the registered Python/dependency versions or register a separate experiment; do not reset this cohort.')
        events=read_events(directory/'events.jsonl');validate_experiment_events(events)
        registrations=[e for e in events if e['kind']=='registration']
        if len(registrations)!=1 or registrations[0].get('manifest_hash')!=digest(result):raise ValueError('Immutable experiment registration is missing or inconsistent')
        return result
    if any((directory/name).exists() for name in ('events.jsonl','bars.jsonl')):raise ValueError('Experiment state exists without its manifest; refusing to reset the prospective cohort')
    result={'schema_version':1,'experiment_id':'v4-prospective-003','registered_at':at.isoformat(),'strategy_version':STRATEGY_VERSION,'config_hash':CONFIG_HASH,'code_hashes':code,'runtime':runtime,'execution':asdict(execution_config()),'versions':list(VERSIONS),'status':'research_only','starting_cash':10000,'note':'Collection begins on first scheduled post-registration run. No elapsed unobserved period is backfilled.'}
    write_json(path,result);append_events(directory/'events.jsonl',[{'kind':'registration','recorded_at':at.isoformat(),'manifest_hash':digest(result)}]);return result

class YahooForwardProvider:
    def daily(self,symbol):
        import yfinance as yf
        return yf.Ticker(symbol).history(period='2y',interval='1d',auto_adjust=False)
    def intraday(self,symbol):
        import yfinance as yf
        return yf.Ticker(symbol).history(period='1mo',interval='5m',prepost=False,auto_adjust=False)

def frozen_input_archive(directory,decision_id,symbol_bars,daily,market):
    def rows(frame):
        return [{'time':pd.Timestamp(t).isoformat(),**{str(k):clean(v) for k,v in r.items()}} for t,r in frame.iterrows()]
    payload={'symbol_bars':rows(symbol_bars),'daily':rows(daily),'market_daily':rows(market)}
    body=json.dumps(clean(payload),sort_keys=True,separators=(',',':'),allow_nan=False).encode();sha=hashlib.sha256(body).hexdigest();path=directory/'inputs'/f'{decision_id}-{sha[:16]}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(gzip.compress(body,mtime=0))
    return str(path.name),sha

def correlation_map(daily_by_symbol,symbol,cutoff):
    known={};date=utc_timestamp(cutoff).tz_convert('America/New_York').date()
    def closes(d):
        if d is None or d.empty:return pd.Series(dtype=float)
        x=d[pd.DatetimeIndex(d.index).tz_convert('America/New_York').date<date].Close
        x=x.copy();x.index=pd.DatetimeIndex(x.index).date
        return x.pct_change(fill_method=None).dropna().tail(60)
    a=closes(daily_by_symbol.get(symbol))
    for other,df in daily_by_symbol.items():
        if other==symbol:continue
        joined=pd.concat([a,closes(df)],axis=1).dropna()
        if len(joined)>=40:
            value=joined.iloc[:,0].corr(joined.iloc[:,1]);
            if pd.notna(value):known[other]=float(value)
    return known

def freeze_decisions(manifest,events,frames,dailies,market,at,directory):
    at=utc_timestamp(at);frozen=[];ids={e.get('decision_id') for e in events if e['kind']=='decision'}
    day=at.tz_convert('America/New_York').date().isoformat()
    accepted={(e['strategy'],e['symbol'],e['signal_day']) for e in events if e['kind']=='decision' and e.get('accepted')}
    bounds=session_bounds(at)
    if bounds is None or not bounds[0]<=at<bounds[1]:return []
    for symbol,bars in sorted(frames.items()):
        packet=feature_packet(bars,dailies.get(symbol),market,at)
        dm,im,mm,ctx=(packet[k] for k in ('daily','intra','market','context'))
        if not dm or not im or not mm:continue
        cutoff=ctx.get('intraday_asof')
        if not cutoff or not 0<=(at-utc_timestamp(cutoff)).total_seconds()<=600:continue
        # Source timestamps must be completed; never treat a scan label as availability.
        correlations=correlation_map(dailies,symbol,at)
        for version in VERSIONS:
            identity=hashlib.sha256(f"{manifest['experiment_id']}|{version}|{symbol}|{cutoff}".encode()).hexdigest()[:24]
            if identity in ids:continue
            if version=='V4':r=evaluate_v4(dm,im,mm,context=ctx);state=r['state'];score=r['score'];reason='; '.join(r.get('reasons',[])+r.get('rejections',[]));stop=r.get('stop');target=r.get('target_1r')
            else:
                fn={'V1':score_row,'V2':score_v2,'V3':score_v3}[version]
                legacy_intra={**im,'momentum_15m':im.get('legacy_momentum_10m',im.get('momentum_15m'))}
                state,score,reason,stop,target,_=fn(dm,legacy_intra,mm) if version=='V3' else fn(dm,legacy_intra)
                r={'checks':{},'unavailable':['Historical earnings, sector and options provenance unavailable']}
            allow=state=='ENTRY TRIGGERED' and (version,symbol,day) not in accepted
            event={'kind':'decision','decision_id':identity,'signal_id':identity,'strategy':version,'symbol':symbol,'signal_day':day,'signal_time':cutoff,'frozen_at':at.isoformat(),'recorded_at':at.isoformat(),'accepted':allow,'state':state,'score':score,'score_kind':'uncalibrated research score, not probability','reason':reason,'stop_price':stop,'target_1r':target,'atr':dm.get('atr_pct',0)*im['price'],'market_regime':classify_regime(mm),'features':packet,'checks':r.get('checks',{}),'unavailable':r.get('unavailable',[]),'correlations':correlations,'config_hash':CONFIG_HASH,'feature_method':'completed_5m_prior_daily_v1'}
            if allow:
                input_file,input_sha=frozen_input_archive(directory,identity,bars,dailies[symbol],market);event.update(input_file=input_file,input_sha256=input_sha);accepted.add((version,symbol,day))
            frozen.append(clean(event));ids.add(identity)
    return frozen

def archive_completed_bars(directory,frames,events,at,collection_start):
    prior=read_events(directory/'bars.jsonl');seen={(r['symbol'],r['time']) for r in prior};watermarks={}
    for r in prior:watermarks[r['symbol']]=max(watermarks.get(r['symbol'],r['time']),r['time'])
    new=[];warnings=[]
    for symbol,df in sorted(frames.items()):
        regular=regular_bars(df)
        for ts,row in regular.iterrows():
            if ts<collection_start or not completed_bar(ts,at):continue
            key=(symbol,ts.isoformat())
            if key in seen:continue
            if symbol in watermarks and ts.isoformat()<=watermarks[symbol]:warnings.append(f'{symbol}: late historical candle excluded to preserve first-seen execution order');continue
            values={k:float(row[k]) for k in ('Open','High','Low','Close','Volume')}
            if values['Volume']<0 or not all(math.isfinite(v) for v in values.values()) or min(values[k] for k in ('Open','High','Low','Close'))<=0 or values['Low']>min(values['Open'],values['Close']) or values['High']<max(values['Open'],values['Close']):warnings.append(f'{symbol}: invalid source candle excluded');continue
            new.append({'symbol':symbol,'time':ts.isoformat(),'first_seen_at':at.isoformat(),'session_end':ts+pd.Timedelta(minutes=5)==session_bounds(ts)[1],**values});seen.add(key)
    append_events(directory/'bars.jsonl',new)
    all_rows=prior+new;out={}
    for symbol in sorted(set(frames)|{r['symbol'] for r in all_rows}):
        rows=[r for r in all_rows if r['symbol']==symbol]
        if rows:
            frame=pd.DataFrame(rows).set_index(pd.DatetimeIndex([r['time'] for r in rows]));frame['session_end']=frame.get('session_end',False)
            frame['session_end']=frame['session_end'].fillna(False).astype(bool)
            out[symbol]=frame[['Open','High','Low','Close','Volume','session_end']].sort_index()
    return out,sorted(set(warnings))

def execution_identity(t):return {k:t.get(k) for k in ('signal_id','entry_time','entry_price','quantity')}
def benchmark(frames,start,at):
    result={'CASH':{'equity':10000,'return':0,'assumption':'0% nominal interest; not a measured Treasury return'}}
    cfg=execution_config();friction=(cfg.slippage_bps+cfg.spread_bps/2)/10000
    for symbol in ('SPY','QQQ'):
        df=frames.get(symbol)
        if df is None or df.empty:result[symbol]={'equity':None,'return':None,'status':'awaiting prospective completed bars'};continue
        df=df[df.index>=start]
        if df.empty:continue
        entry=float(df.Open.iloc[0])*(1+friction);qty=math.floor(10000/entry);value=10000-qty*entry+qty*float(df.Close.iloc[-1]);result[symbol]={'equity':value,'return':value/10000-1,'entry_time':df.index[0].isoformat(),'asof':(df.index[-1]+pd.Timedelta(minutes=5)).isoformat(),'assumption':'Fully invested integer-share buy-and-hold; adverse entry costs, unsold shares marked, exposure differs from capped strategies'}
    return result

def publish_forward(root,manifest,events,frames,at,warnings=None):
    directory=root/'data/quant_forward';old_path=root/'docs/data/v4-forward.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else {}
    strategies={};observations=[]
    for version in VERSIONS:
        signals=[e for e in events if e['kind']=='decision' and e.get('accepted') and e['strategy']==version]
        run=simulate_portfolio(signals,frames,ExecutionConfig(**manifest['execution']),end_time=at)
        former=old.get('strategies',{}).get(version,{})
        current={t['signal_id']:t for t in run['trades']}
        for t in former.get('trades',[]):
            replacement=current.get(t['signal_id'])
            if not replacement or execution_identity(t)!=execution_identity(replacement):raise ValueError('Previously reported prospective entry would change; execution paused, no retrospective rewrite')
            if t['status']=='CLOSED' and digest(t)!=digest(replacement):raise ValueError('Previously closed prospective outcome would change; refusing rewrite')
        for event in events:
            if event.get('kind')=='execution_observation' and event['strategy']==version:
                t=event['trade'];replacement=current.get(t['signal_id'])
                if replacement is None or digest(t)!=digest(replacement):raise ValueError('Immutable prospective outcome would change; refusing rewrite')
        strategies[version]=run
        known_ids={e['observation_id'] for e in events if e['kind']=='execution_observation'}
        for t in run['trades']:
            if t['status']=='CLOSED':
                oid=digest({'strategy':version,'trade':t})
                if oid not in known_ids:observations.append({'kind':'execution_observation','observation_id':oid,'strategy':version,'recorded_at':at.isoformat(),'trade':t})
    append_events(directory/'events.jsonl',observations)
    collection=next((e['recorded_at'] for e in events if e['kind']=='collection_started'),None)
    payload={'schema_version':1,'status':'research_only_inconclusive','generated_at':at.isoformat(),'data_cutoff':max(((df.index[-1]+pd.Timedelta(minutes=5)).isoformat() for df in frames.values() if not df.empty),default=None),'created_at':manifest['registered_at'],'collection_started_at':collection,'config_hash':CONFIG_HASH,'starting_cash':10000,'strategies':strategies,'signals':[e for e in events if e['kind']=='decision'],'benchmarks':benchmark(frames,utc_timestamp(collection or manifest['registered_at']),at),'warnings':warnings or [],'note':'Independent prospective V1–V4 cohort. Legacy forward portfolios keep their own history and method. No historical signals or pre-collection benchmark marks are inserted.'}
    write_json(old_path,payload);return payload

def collect(root=Path('.'),provider=None,clock=None):
    clock=clock or timestamp;provider=provider or YahooForwardProvider();at=utc_timestamp(clock());manifest=register(root,at);directory=root/'data/quant_forward';events=read_events(directory/'events.jsonl')
    if not any(e['kind']=='collection_started' for e in events):append_events(directory/'events.jsonl',[{'kind':'collection_started','recorded_at':at.isoformat()}]);events=read_events(directory/'events.jsonl')
    collection_start=utc_timestamp(next(e['recorded_at'] for e in events if e['kind']=='collection_started'))
    cfg=json.loads((root/'skew_config.json').read_text());symbols=sorted(set(cfg.get('core_universe',[]))|{'SPY','QQQ'})
    frames={};dailies={};warnings=[]
    for symbol in symbols:
        try:
            raw=provider.intraday(symbol)
            # A later symbol's download must never "complete" an earlier partial
            # snapshot. Capture the actual observation clock before any other I/O.
            observed_at=utc_timestamp(clock())
            frame=regular_bars(raw)
            frames[symbol]=frame.loc[[completed_bar(t,observed_at) for t in frame.index]]
            dailies[symbol]=provider.daily(symbol)
        except Exception as exc:warnings.append(f'{symbol}: source unavailable ({type(exc).__name__})')
    # Actual freeze time is AFTER download, not a stale workflow start timestamp.
    at=utc_timestamp(clock());new=freeze_decisions(manifest,events,frames,dailies,dailies.get('SPY',pd.DataFrame()),at,directory);append_events(directory/'events.jsonl',new);events+=new
    execution_frames,problems=archive_completed_bars(directory,frames,events,at,collection_start)
    try:
        split_symbols=[]
        for symbol,df in dailies.items():
            if 'Stock Splits' in df:
                actions=df[(pd.DatetimeIndex(df.index).tz_convert('UTC')>=collection_start)&(pd.to_numeric(df['Stock Splits'],errors='coerce').fillna(0)!=0)]
                if not actions.empty:split_symbols.append(symbol)
        if split_symbols:raise ValueError('Corporate-action share adjustment requires reconciliation: '+', '.join(split_symbols))
        return publish_forward(root,manifest,events,execution_frames,at,warnings+problems)
    except ValueError as exc:
        path=root/'docs/data/v4-forward.json'
        prior=json.loads(path.read_text()) if path.exists() else {'created_at':manifest['registered_at'],'starting_cash':10000,'strategies':{},'signals':[]}
        prior.update(status='paused_reconciliation_required',status_at=at.isoformat(),warnings=warnings+problems+[str(exc)])
        write_json(path,prior);return prior

def pause_forward(root,exc,at=None):
    """Preserve the last valid financial view while making a failure explicit."""
    path=root/'docs/data/v4-forward.json'
    try:prior=json.loads(path.read_text()) if path.exists() else {}
    except (OSError,ValueError):prior={}
    if not isinstance(prior,dict):prior={}
    prior.update(status='paused_reconciliation_required',status_at=utc_timestamp(at or timestamp()).isoformat(),warnings=[f'{type(exc).__name__}: {exc}'])
    write_json(path,prior)
    return prior

def main(argv=None,root=Path('.')):
    parser=argparse.ArgumentParser();parser.add_argument('--initialize-only',action='store_true');args=parser.parse_args(argv)
    directory=root/'data/quant_forward';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            if args.initialize_only:
                m=register(root);result=publish_forward(root,m,read_events(directory/'events.jsonl'),{},timestamp())
            else:result=collect(root)
            return 1 if result['status'].startswith('paused_') else 0
        except Exception as exc:
            pause_forward(root,exc)
            print(f'Prospective experiment paused: {type(exc).__name__}: {exc}')
            return 1

if __name__=='__main__':raise SystemExit(main())
