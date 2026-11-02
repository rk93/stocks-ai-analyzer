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
    return {'schema_version':1,'generated_at':now.isoformat(),'datasets':datasets,
     'note':'Publication time is not quote time. Staleness threshold is a conservative 36-hour publication flag, not exchange-session validation.'}

if __name__=='__main__':
    Path('docs/data/manifest.json').write_text(json.dumps(build_manifest(),indent=2,allow_nan=False)+'\n')
