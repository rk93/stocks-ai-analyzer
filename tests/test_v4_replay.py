import gzip
import hashlib
import json
import pandas as pd
import pytest
from v4_replay import block_interval, chronological_windows, load_archive, select_signals, window_bars


def test_chronological_windows_purge_five_sessions_and_no_overlap():
    dates=list(pd.bdate_range('2026-01-01',periods=60).date)
    windows,gaps=chronological_windows(dates)
    assert [len(windows[k]) for k in ('development','validation','retrospective_tail')]==[30,10,10]
    assert len(gaps)==10
    assert set().union(*map(set,windows.values())).isdisjoint(gaps)
    for left,right in [('development','validation'),('validation','retrospective_tail')]:
        assert dates.index(windows[right][0])-dates.index(windows[left][-1])==6


def test_window_paths_never_include_later_split_and_signal_selection_is_local():
    idx=pd.date_range('2026-01-05 09:30',periods=3,freq='5min',tz='America/New_York')
    bars=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':100.},index=idx)
    later=bars.copy();later.index+=pd.Timedelta(days=1);later.Close=1000.
    daily_idx=pd.bdate_range('2025-01-01',periods=260,tz='America/New_York')
    daily=pd.DataFrame({'High':101.,'Low':99.,'Close':100.},index=daily_idx)
    result=window_bars({('A','5m'):pd.concat([bars,later]),('A','1d'):daily},[idx[0].date()])
    assert len(result['A'])==3
    assert result['A'].Close.max()==100
    signals=[{'signal_time':idx[0].isoformat()},{'signal_time':later.index[0].isoformat()}]
    assert select_signals(signals,[idx[0].date()])==signals[:1]


def test_archive_digest_checked_before_parse(tmp_path):
    records=[]
    raw=b'Datetime,Open,High,Low,Close,Volume\n2026-01-05T14:30:00Z,100,101,99,100,10\n'
    for symbol,interval in [('SPY','1d'),('SPY','5m'),('QQQ','5m')]:
        path=tmp_path/f'{symbol}-{interval}.csv.gz';archive=gzip.compress(raw,mtime=0);path.write_bytes(archive)
        records.append({'symbol':symbol,'interval':interval,'path':str(path),'rows':1,'sha256':hashlib.sha256(archive).hexdigest(),'uncompressed_sha256':hashlib.sha256(raw).hexdigest()})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'records':records}))
    _,frames=load_archive(manifest)
    assert len(frames)==3
    (tmp_path/'SPY-5m.csv.gz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA256 mismatch'):load_archive(manifest)


def test_block_bootstrap_is_deterministic_and_does_not_invent_small_sample_ci():
    dates=list(pd.bdate_range('2026-01-01',periods=30).date)
    trades=[{'status':'CLOSED','entry_time':pd.Timestamp(day,tz='America/New_York').isoformat(),'net_r':(-1 if i%2 else .8)} for i,day in enumerate(dates)]
    a=block_interval(trades,dates,resamples=200)
    assert a==block_interval(trades,dates,resamples=200)
    assert a['lower']<=a['upper']
    assert block_interval(trades[:3],dates)['lower'] is None


def test_declared_capture_is_complete_and_immutable():
    manifest,frames=load_archive('research/v4_inputs/capture-20261105/manifest.json')
    assert not manifest['errors']
    assert len(manifest['records'])==2*len(manifest['universe'])
    assert {'SPY','QQQ'}.issubset(manifest['universe'])
    assert all(frame.index.tz is not None for frame in frames.values())
