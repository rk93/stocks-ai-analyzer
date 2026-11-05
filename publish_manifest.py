"""Add a separate provenance contract without rewriting historical datasets."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {'skew':'data/skew_latest.csv','trending':'data/trending_latest.csv',
 'day-opportunities':'data/day_opportunities_latest.csv','backtest':'data/backtest_trades.csv',
 'prediction-tracker':'data/prediction_tracker.csv','prediction-research':'data/predictions.csv','paper-trading':'data/paper_equity.csv'}


def build_manifest(root=Path('.')):
    datasets={}
    now=datetime.now(timezone.utc)
    for name,source in SOURCES.items():
        path=root/'docs/data'/f'{name}.json'
        try:
            raw=path.read_bytes(); payload=json.loads(raw)
            generated=payload.get('generated_at')
            age=(now-datetime.fromisoformat(generated.replace('Z','+00:00'))).total_seconds() if generated else None
            records=list(csv.DictReader((root/source).open()))
            timestamps=sorted({r[k] for r in records for k in ('run_date','timestamp','signal_time') if r.get(k)})
            datasets[name]={'schema_version':1,'generated_at':generated,
              'market_data_timestamp':None,'latest_record_timestamp':timestamps[-1] if timestamps else None,
              'source':source,'provenance':'Existing Python pipeline; Yahoo/yfinance market inputs; see engine execution assumptions.',
              'pipeline_status':'published_snapshot','validity':'unknown_quote_freshness',
              'publication_age_seconds':age,'stale_publication':age is None or age>36*3600,
              'coverage':{'source_rows':len(records),'market_data_cutoff_verified':False},
              'sha256':hashlib.sha256(raw).hexdigest()}
        except (OSError,ValueError,TypeError) as exc:
            datasets[name]={'schema_version':1,'pipeline_status':'unavailable','error':str(exc)}
    market_path=root/'docs/data/portfolio-market.json'
    if market_path.exists():
        raw=market_path.read_bytes()
        market=json.loads(raw)
        sessions=sorted(r['last_session'] for r in list(market.get('instruments',{}).values())+list(market.get('fx',{}).values()) if r.get('last_session'))
        datasets['portfolio-market']={'schema_version':market.get('schema_version'),'generated_at':market.get('generated_at'),
          'market_data_timestamp':None,'market_data_session_range':{'earliest_last_session':sessions[0] if sessions else None,'latest_last_session':sessions[-1] if sessions else None},
          'source':'portfolio_market_data.py: fixed public universe; Yahoo Finance via yfinance',
          'validity':'dated_completed_session_observations','pipeline_status':market.get('pipeline_status','unavailable'),
          'coverage':market.get('coverage',{}),'sha256':hashlib.sha256(raw).hexdigest(),
          'note':'Clients check individual close and FX dates at runtime. Generated time does not establish quote freshness.'}
    return {'schema_version':1,'generated_at':now.isoformat(),'datasets':datasets,
     'note':'Publication time is not quote time. Staleness threshold is a conservative 36-hour publication flag, not exchange-session validation.'}

if __name__=='__main__':
    Path('docs/data/manifest.json').write_text(json.dumps(build_manifest(),indent=2,allow_nan=False)+'\n')
