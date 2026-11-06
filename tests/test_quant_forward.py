from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import pandas as pd
import pytest
import quant_forward as q

@pytest.fixture
def root(tmp_path):
    for name in q.CODE_FILES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy(Path(name),target)
    (tmp_path/'skew_config.json').write_text(json.dumps({'core_universe':['TEST']}))
    return tmp_path

AT=pd.Timestamp('2026-11-06T17:00:00Z')

def test_registration_no_retrospective_signals_and_code_freeze(root):
    manifest=q.register(root,AT);events=q.read_events(root/'data/quant_forward/events.jsonl')
    assert len(events)==1 and events[0]['kind']=='registration'
    assert manifest['starting_cash']==10000 and manifest['versions']==['V1','V2','V3','V4']
    assert q.register(root,AT)==manifest
    (root/'strategy_v4.py').write_text('changed')
    with pytest.raises(ValueError,match='Frozen'):q.register(root,AT)

def test_registration_freezes_the_candidate_universe(root):
    q.register(root,AT)
    (root/'skew_config.json').write_text(json.dumps({'core_universe':['TEST','EXTRA']}))
    with pytest.raises(ValueError,match='Frozen prospective experiment code/config changed'):q.register(root,AT)

def test_registration_freezes_actual_runtime_versions(root,monkeypatch):
    manifest=q.register(root,AT)
    assert manifest['runtime']['python'] and manifest['runtime']['packages']['exchange_calendars']
    changed={**manifest['runtime'],'packages':{**manifest['runtime']['packages'],'exchange_calendars':'999.0'}}
    monkeypatch.setattr(q,'runtime_environment',lambda:changed)
    with pytest.raises(ValueError,match='Frozen prospective runtime changed'):q.register(root,AT)

def test_initial_portfolios_are_equal_unobserved_cash_not_backfilled(root):
    m=q.register(root,AT);view=q.publish_forward(root,m,q.read_events(root/'data/quant_forward/events.jsonl'),{},AT)
    assert all(x['metrics']['ending_equity']==10000 for x in view['strategies'].values())
    assert all(not x['trades'] for x in view['strategies'].values())
    assert view['collection_started_at'] is None
    assert view['benchmarks']['SPY']['equity'] is None

def bars(times):
    return pd.DataFrame({'Open':[100]*len(times),'High':[100.5]*len(times),'Low':[99.5]*len(times),'Close':[100.0]*len(times),'Volume':[1000]*len(times)},index=pd.DatetimeIndex(times))

def test_only_first_seen_post_collection_completed_bars_are_archived(root):
    directory=root/'data/quant_forward';directory.mkdir(parents=True)
    frame=bars(['2026-11-06T16:55:00Z','2026-11-06T17:00:00Z','2026-11-06T17:05:00Z'])
    archive,w=q.archive_completed_bars(directory,{'TEST':frame},[],AT+pd.Timedelta(minutes=7),AT)
    assert list(archive['TEST'].index)==[AT]
    changed=frame.copy();changed.loc[AT,'Close']=99.8
    archive,_=q.archive_completed_bars(directory,{'TEST':changed},[],AT+pd.Timedelta(minutes=12),AT)
    assert archive['TEST'].loc[AT,'Close']==100
    assert len(q.read_events(directory/'bars.jsonl'))==2

def test_no_prospective_fill_before_publication_or_on_unfinished_bar(root):
    m=q.register(root,AT);events=[{'kind':'decision','accepted':True,'strategy':'V4','symbol':'TEST','signal_id':'a','decision_id':'a','signal_time':AT.isoformat(),'frozen_at':(AT+pd.Timedelta(minutes=2)).isoformat(),'stop_price':99,'score':80,'market_regime':'BULL_TREND'}]
    frame=bars(['2026-11-06T17:00:00Z','2026-11-06T17:05:00Z'])
    v=q.publish_forward(root,m,events,{'TEST':frame},AT+pd.Timedelta(minutes=7))
    assert not v['strategies']['V4']['trades']
    v=q.publish_forward(root,m,events,{'TEST':frame},AT+pd.Timedelta(minutes=10))
    trade=v['strategies']['V4']['trades'][0]
    assert pd.Timestamp(trade['entry_time'])==AT+pd.Timedelta(minutes=5)
    assert trade['status']=='OPEN'

def test_revised_reported_execution_fails_closed(root):
    m=q.register(root,AT);events=[{'kind':'decision','accepted':True,'strategy':'V1','symbol':'TEST','signal_id':'a','decision_id':'a','signal_time':AT.isoformat(),'frozen_at':AT.isoformat(),'stop_price':99,'score':80,'market_regime':'BULL_TREND'}]
    frame=bars(['2026-11-06T17:00:00Z'])
    q.publish_forward(root,m,events,{'TEST':frame},AT+pd.Timedelta(minutes=5))
    before=(root/'docs/data/v4-forward.json').read_bytes()
    frame['Open']=100.1
    with pytest.raises(ValueError,match='entry would change'):q.publish_forward(root,m,events,{'TEST':frame},AT+pd.Timedelta(minutes=5))
    assert (root/'docs/data/v4-forward.json').read_bytes()==before

def test_offhours_decisions_never_inserted(root):
    m=q.register(root,AT)
    assert q.freeze_decisions(m,[],{'TEST':bars(['2026-11-06T21:00:00Z'])},{},pd.DataFrame(),pd.Timestamp('2026-11-06T23:00:00Z'),root/'data/quant_forward')==[]

def test_forward_contains_no_broker_order_or_legacy_write_paths():
    source=Path('quant_forward.py').read_text()
    assert 'requests.post' not in source and 'submit_alpaca' not in source
    for old in ['data/paper_state.json','data/paper_trades.csv','data/predictions.csv']:assert old not in source

def test_fetch_snapshot_cannot_complete_while_other_symbols_download(root,monkeypatch):
    """A 17:00 partial fetched at 17:04:59 stays partial after the scan passes 17:05."""
    calls=[]
    moments=iter(pd.Timestamp(t) for t in ['2026-11-06T16:59:59Z','2026-11-06T17:04:59Z','2026-11-06T17:05:01Z','2026-11-06T17:05:02Z','2026-11-06T17:05:03Z'])
    def clock():
        value=next(moments);calls.append(('clock',value));return value
    class Provider:
        def intraday(self,symbol):
            calls.append(('intraday',symbol))
            return bars(['2026-11-06T16:55:00Z','2026-11-06T17:00:00Z'])
        def daily(self,symbol):calls.append(('daily',symbol));return pd.DataFrame()
    captured={}
    def freeze(manifest,events,frames,dailies,market,at,directory):
        captured.update(frames=frames,at=at);return []
    monkeypatch.setattr(q,'freeze_decisions',freeze)
    view=q.collect(root,Provider(),clock)
    assert list(captured['frames']['QQQ'].index)==[pd.Timestamp('2026-11-06T16:55:00Z')]
    assert list(captured['frames']['TEST'].index)==[pd.Timestamp('2026-11-06T16:55:00Z'),AT]
    assert captured['at']==pd.Timestamp('2026-11-06T17:05:03Z')
    for symbol in ['QQQ','SPY','TEST']:
        i=calls.index(('intraday',symbol))
        assert calls[i+1][0]=='clock' and calls[i+2]==('daily',symbol)
    archived=q.read_events(root/'data/quant_forward/bars.jsonl')
    assert {(r['symbol'],r['time']) for r in archived}=={('SPY',AT.isoformat()),('TEST',AT.isoformat())}
    assert view['collection_started_at']=='2026-11-06T16:59:59+00:00'

def test_late_candles_do_not_change_first_seen_execution_order(root):
    directory=root/'data/quant_forward';directory.mkdir(parents=True)
    q.archive_completed_bars(directory,{'TEST':bars(['2026-11-06T17:05:00Z'])},[],AT+pd.Timedelta(minutes=10),AT)
    archive,warnings=q.archive_completed_bars(directory,{'TEST':bars(['2026-11-06T17:00:00Z','2026-11-06T17:05:00Z'])},[],AT+pd.Timedelta(minutes=15),AT)
    assert list(archive['TEST'].index)==[AT+pd.Timedelta(minutes=5)]
    assert any('late historical candle excluded' in w for w in warnings)

def test_missing_manifest_never_resets_existing_experiment(root):
    directory=root/'data/quant_forward';directory.mkdir(parents=True)
    (directory/'events.jsonl').write_text('{"kind":"registration"}\n')
    with pytest.raises(ValueError,match='without its manifest'):q.register(root,AT)
    assert not (directory/'manifest.json').exists()

def test_registration_checksum_freezes_execution_settings(root):
    q.register(root,AT)
    path=root/'data/quant_forward/manifest.json';manifest=json.loads(path.read_text())
    manifest['execution']['initial_capital']=20000;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='registration is missing or inconsistent'):q.register(root,AT)

@pytest.mark.parametrize('corrupt',['{bad json}\n','[]\n','{"kind":"unknown","recorded_at":"2026-11-06T17:00:00Z"}\n'])
def test_malformed_immutable_state_fails_closed(root,corrupt):
    q.register(root,AT)
    path=root/'data/quant_forward/events.jsonl';path.write_text(path.read_text()+corrupt);before=path.read_bytes()
    with pytest.raises((ValueError,KeyError)):q.collect(root,clock=lambda:AT)
    assert path.read_bytes()==before

def test_cli_pauses_on_frozen_code_change_without_resetting_financials(root):
    m=q.register(root,AT);q.publish_forward(root,m,q.read_events(root/'data/quant_forward/events.jsonl'),{},AT)
    path=root/'data/quant_forward/events.jsonl';events_before=path.read_bytes()
    manifest_before=(root/'data/quant_forward/manifest.json').read_bytes()
    (root/'strategy_v4.py').write_text('changed')
    assert q.main(['--initialize-only'],root)==1
    view=json.loads((root/'docs/data/v4-forward.json').read_text())
    assert view['status']=='paused_reconciliation_required'
    assert all(x['metrics']['ending_equity']==10000 for x in view['strategies'].values())
    assert 'Frozen prospective experiment code/config changed' in view['warnings'][0]
    assert path.read_bytes()==events_before
    assert (root/'data/quant_forward/manifest.json').read_bytes()==manifest_before

def test_closed_execution_ledger_guards_outcomes_even_without_derived_report(root):
    m=q.register(root,AT)
    events=[{'kind':'decision','accepted':True,'strategy':'V1','symbol':'TEST','signal_id':'a','decision_id':'a','signal_time':AT.isoformat(),'frozen_at':AT.isoformat(),'stop_price':99,'score':80,'market_regime':'BULL_TREND'}]
    frame=bars(['2026-11-06T17:00:00Z']);frame['Low']=98
    view=q.publish_forward(root,m,events,{'TEST':frame},AT+pd.Timedelta(minutes=5))
    assert view['strategies']['V1']['trades'][0]['status']=='CLOSED'
    observations=[e for e in q.read_events(root/'data/quant_forward/events.jsonl') if e['kind']=='execution_observation']
    (root/'docs/data/v4-forward.json').unlink()
    frame['Low']=99.5
    with pytest.raises(ValueError,match='Immutable prospective outcome'):q.publish_forward(root,m,events+observations,{'TEST':frame},AT+pd.Timedelta(minutes=5))

def test_holiday_rejects_decisions_and_archived_market_bars(root):
    holiday=pd.Timestamp('2026-11-26T17:00:00Z');m=q.register(root,AT)
    assert q.freeze_decisions(m,[],{'TEST':bars([holiday])},{},pd.DataFrame(),holiday,root/'data/quant_forward')==[]
    archive,_=q.archive_completed_bars(root/'data/quant_forward',{'TEST':bars([holiday])},[],holiday+pd.Timedelta(minutes=5),AT)
    assert not archive

def test_half_day_close_is_an_explicit_session_end(root):
    close=pd.Timestamp('2026-11-27T18:00:00Z')
    directory=root/'data/quant_forward';directory.mkdir(parents=True)
    archive,_=q.archive_completed_bars(directory,{'TEST':bars([close-pd.Timedelta(minutes=5),close])},[],close+pd.Timedelta(minutes=5),AT)
    assert list(archive['TEST'].index)==[close-pd.Timedelta(minutes=5)]
    assert bool(archive['TEST']['session_end'].iloc[-1])
