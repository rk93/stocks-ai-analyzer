"""Publish public daily closes and USD-based FX history; never reads private portfolios.

Latest-session observations are not live prices. Current-session bars are excluded
unless provider regular-session-end metadata confirms the session has ended.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

FX_PAIRS={c:f'{c}USD=X' for c in ('GBP','EUR','INR','CAD','AUD','CHF','JPY','HKD','SGD')}
EXTRA_PUBLIC_SYMBOLS=('RR.L','RELIANCE.NS')


def finite(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None


def normal_currency(value):
    v=str(value or '').strip()
    return 'GBX' if v=='GBp' else v.upper()


def public_universe(root=Path('.')):
    cfg=json.loads((root/'skew_config.json').read_text())
    holdings=json.loads((root/'portfolio_config.json').read_text()).get('holdings',[])
    return sorted(set(cfg.get('core_universe',[]))|{h['symbol'] for h in holdings}|set(EXTRA_PUBLIC_SYMBOLS)|{'SPY','QQQ'})


def instrument_payload(symbol,frame,metadata,now,force_prior_day=False):
    if frame.empty:raise ValueError('No historical bars returned')
    ccy=normal_currency(metadata.get('currency'))
    if ccy not in {'USD','GBP','GBX','EUR','INR','CAD','AUD','CHF','JPY','HKD','SGD'}:
        raise ValueError('Provider quote currency is missing or unsupported')
    provider_symbol=metadata.get('symbol')
    if provider_symbol and str(provider_symbol).upper()!=symbol.upper():
        raise ValueError('Provider symbol does not match requested instrument')
    tzname=metadata.get('exchangeTimezoneName') or str(frame.index.tz or '')
    try:zone=ZoneInfo(tzname)
    except (ValueError,KeyError):raise ValueError('Provider exchange timezone is unavailable')
    local_today=now.astimezone(zone).date()
    end=finite(metadata.get('currentTradingPeriod',{}).get('regular',{}).get('end'))
    confirmed_end=datetime.fromtimestamp(end,timezone.utc) if end else None
    history=[]
    for ts,row in frame.sort_index().iterrows():
        day=ts.date()
        if day>local_today:continue
        if day==local_today and force_prior_day:continue
        if day==local_today and not (confirmed_end and confirmed_end.astimezone(zone).date()==day and now>=confirmed_end+timedelta(minutes=15)):
            continue
        close=finite(row.get('Close'));op=finite(row.get('Open'));high=finite(row.get('High'));low=finite(row.get('Low'))
        adj=finite(row.get('Adj Close'));volume=finite(row.get('Volume'))
        if not all(v is not None and v>0 for v in (close,op,high,low)) or high<max(op,close,low) or low>min(op,close,high):continue
        history.append({'date':day.isoformat(),'open':round(op,8),'high':round(high,8),'low':round(low,8),'close':round(close,8),
                        'adjusted_close':round(adj,8) if adj and adj>0 else None,'volume':volume if volume is not None and volume>=0 else None})
    if not history:raise ValueError('No valid completed-session bars')
    if len({r['date'] for r in history})!=len(history):raise ValueError('Duplicate session dates')
    return {'symbol':symbol,'currency':ccy,'exchange':metadata.get('exchangeName') or metadata.get('exchange') or None,
            'timezone':tzname,'instrument_type':metadata.get('instrumentType') or None,
            'status':'ok','fetched_at':now.isoformat(),'last_session':history[-1]['date'],
            'history':history,'source':'Yahoo Finance via yfinance; history metadata currency; auto_adjust=False',
            'price_method':'provider daily OHLC; current session excluded unless regular end confirmed; adjusted_close is provider-adjusted, not point-in-time archived'}


class YahooProvider:
    def fetch(self,symbol):
        import yfinance as yf
        ticker=yf.Ticker(symbol)
        frame=ticker.history(period='1y',interval='1d',auto_adjust=False,timeout=15)
        return frame,ticker.history_metadata


def build_dataset(symbols,provider,now=None,previous=None):
    now=now or datetime.now(timezone.utc);previous=previous or {}
    instruments={};fx={};errors=[]
    for symbol in list(symbols)+list(FX_PAIRS.values()):
        is_fx=symbol in FX_PAIRS.values();key=next((c for c,s in FX_PAIRS.items() if s==symbol),symbol)
        target=fx if is_fx else instruments;prior=previous.get('fx' if is_fx else 'instruments',{}).get(key)
        try:
            frame,meta=provider.fetch(symbol);r=instrument_payload(symbol,frame,meta,now,force_prior_day=is_fx)
            if is_fx:
                if r['currency']!='USD':raise ValueError('FX quote must be USD per configured currency unit')
                r={**r,'unit_currency':key,'history':[{'date':v['date'],'usd_per_unit':v['close']} for v in r['history']]}
            target[key]=r
        except Exception as exc:
            message=f'{type(exc).__name__}: {exc}';errors.append({'symbol':symbol,'error':message[:350]})
            if prior and prior.get('history'):
                target[key]={**prior,'status':'retained_after_error','last_attempt_at':now.isoformat(),'error':message[:350]}
            else:target[key]={'symbol':symbol,'status':'unavailable','history':[],'error':message[:350]}
        print(symbol,target[key]['status'],len(target[key]['history']),flush=True)
    available=sum(bool(r['history']) for r in instruments.values())
    return {'schema_version':1,'generated_at':now.isoformat(),'pipeline_status':('unavailable' if not available else 'partial' if errors else 'ok'),
            'source':'Yahoo Finance via yfinance','universe':list(symbols),
            'coverage':{'requested':len(symbols),'available':sum(bool(r['history']) for r in instruments.values())},
            'instruments':instruments,'fx':fx,'errors':errors,
            'assumptions':['Public fixed universe only; no private holdings are used or transmitted.',
              'Session dates are observations, not live quotes. Publication time is not market-data time.',
              'FX pairs are explicitly configured USD per unit; USD is a mathematical identity, GBX is one hundredth of GBP.',
              'Adjusted-close history can be revised by the provider; it is not point-in-time evidence for strategy validation.']}


def write_dataset(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
    os.replace(temporary,path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--symbols',nargs='+');args=parser.parse_args()
    path=Path('docs/data/portfolio-market.json')
    try:previous=json.loads(path.read_text())
    except (OSError,ValueError):previous={}
    write_dataset(path,build_dataset(args.symbols or public_universe(),YahooProvider(),previous=previous))
