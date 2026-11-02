"""Publish frozen strategy-specific rationale and append-only close-return observations.

Legacy daily OHLC hit/miss classifications are deliberately not carried forward:
the old tracker used generic V1 levels for some V2/V3 records. Close returns use
the common original signal price and are explicitly NOT executable trade P/L.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def finite(value):
    import math
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):
        return None


def original(row):
    version=row.get('strategy_version') or 'V1 (legacy)'
    prefix={'V2':'v2_','V3':'v3_'}.get(version,'')
    def frozen(field, fallback):
        return row.get('prediction_'+field) or row.get(prefix+fallback) or None
    return {'prediction_id':row['prediction_id'],'symbol':row['symbol'],
      'strategy_version':version,'signal_time':row.get('generated_at'),
      'frozen_at':row.get('frozen_at'),'data_availability_cutoff':None,
      'market_regime':row.get('market_regime') or 'Unknown',
      'prediction':frozen('state','state'),'score':finite(frozen('score','score')),
      'signal_price':finite(row.get('price')),'stop':finite(frozen('stop','stop')),
      'target_1r':finite(frozen('target_1r','target_1r')),
      'target_2r':finite(frozen('target_2r','target_2r')),
      'original_reason':frozen('reason','reason')}


def observations(tracker):
    for row in tracker:
        for h in (1,5,15,30):
            value=finite(row.get(f'return_{h}d'))
            if value is None:continue
            obs={'prediction_id':row['prediction_id'],'horizon_sessions':h,
              'evaluation_date':row.get(f'evaluation_date_{h}d'),
              'close_return':value,'methodology':'legacy_daily_close_return_v1',
              'source':'data/prediction_tracker.csv','execution_pnl':False}
            obs['observation_id']=hashlib.sha256(json.dumps(obs,sort_keys=True).encode()).hexdigest()[:24]
            yield obs


def publish(root=Path('.')):
    ledger=list(csv.DictReader((root/'data/predictions.csv').open()))
    tracker=list(csv.DictReader((root/'data/prediction_tracker.csv').open()))
    archive=root/'data/prediction_observations_v1.jsonl'
    prior=[json.loads(line) for line in archive.read_text().splitlines()] if archive.exists() else []
    seen={r['observation_id'] for r in prior}
    now=datetime.now(timezone.utc).isoformat()
    with archive.open('a') as out:
        for obs in observations(tracker):
            if obs['observation_id'] in seen:continue
            obs['recorded_at']=now
            out.write(json.dumps(obs,allow_nan=False)+'\n');prior.append(obs);seen.add(obs['observation_id'])
    latest={}
    for obs in prior:latest[(obs['prediction_id'],obs['horizon_sessions'])]=obs
    records=[]
    for row in ledger:
        r=original(row)
        r['observations']=[latest[(r['prediction_id'],h)] for h in (1,5,15,30) if (r['prediction_id'],h) in latest]
        records.append(r)
    payload={'schema_version':1,'generated_at':now,'predictions':list(reversed(records)),
       'source':'data/predictions.csv','observations':'data/prediction_observations_v1.jsonl',
       'limitations':['Data availability cutoff is not recorded in the original ledger; unknown, not inferred.',
         'Close-return observations are derived from the legacy tracker and may revise as source prices revise; prior versions remain appended.',
         'Target/stop sequence and legacy correctness labels are not used as executable P/L.']}
    (root/'docs/data/prediction-research.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n')

if __name__=='__main__':publish()
