"""Retrospective explanation examples; never emits prospective predictions.

The grid is chosen by dates/symbol names/clock times, not realised outcomes.
These examples cannot estimate acceptance rates or demonstrate profitability.
"""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import pandas as pd
from quant_features import feature_packet,NY
from strategy_v4 import evaluate_v4,CONFIG_HASH
from v4_replay import load_archive,clean,digest_file

MANIFEST=Path('research/v4_inputs/capture-20261105/enriched-manifest.json')

def build(manifest_path=MANIFEST,replay_path=Path('docs/data/v4-replay.json')):
    manifest,frames=load_archive(manifest_path);market=frames[('SPY','1d')]
    sessions=sorted(set(frames[('SPY','5m')].index.tz_convert(NY).date))
    capture=pd.Timestamp(manifest['captured_at']).tz_convert(NY).date();sessions=[d for d in sessions if d<capture]
    decisions=[]
    for day in sessions[10::10]:
        for symbol in sorted(manifest['universe'])[:4]:
            at=pd.Timestamp(day,tz=NY)+pd.Timedelta(hours=13,minutes=30)
            p=feature_packet(frames[(symbol,'5m')],frames[(symbol,'1d')],market,at)
            e=evaluate_v4(p['daily'],p['intra'],p['market'],context=p['context'])
            decisions.append({'strategy':'V4','symbol':symbol,'signal_time':at.isoformat(),'accepted':e['accepted'],'score':e['score'],'score_kind':e['score_kind'],'market_regime':e['regime'],'reason':'; '.join(e['rejections'] or e['reasons']),'supports':[c.get('name') for c in e['checks'] if c.get('passed')],'contradictions':e['rejections'],'checks':e['checks'],'context':p['context'],'source':'retrospective_explanation_sample','retrospective':True})
    if replay_path.exists():
        replay=json.loads(replay_path.read_text());accepted=[]
        for comparison in replay.get('comparisons',[]):
            if comparison['strategy']=='V4':accepted+=comparison.get('signals',[])
        for signal in sorted(accepted,key=lambda x:(x['signal_time'],x['symbol']))[:5]:
            decisions.append({**signal,'accepted':True,'source':'chronological_accepted_replay_examples_not_outcome_selected','score_kind':'uncalibrated research evidence, not probability','reason':signal.get('rationale'),'supports':[c.get('name') for c in signal.get('checks',[]) if c.get('passed')],'contradictions':[]})
    return clean({'schema_version':1,'config_hash':CONFIG_HASH,'generated_at':manifest['captured_at'],'built_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest_file(manifest_path),'status':'retrospective_examples_only','sampling':'Every tenth completed captured date after a ten-session warmup; first four alphabetical public symbols;13:30NY; plus first five acceptedV4signals by timestamp, never by outcome. Not exhaustive or an acceptance-rate estimate.','decisions':decisions})

if __name__=='__main__':Path('docs/data/v4-candidates.json').write_text(json.dumps(build(),indent=2,allow_nan=False)+'\n')
