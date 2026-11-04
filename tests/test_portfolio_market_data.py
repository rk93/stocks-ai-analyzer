from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile
import unittest
import pandas as pd
from portfolio_market_data import instrument_payload, build_dataset, public_universe, write_dataset

NOW=datetime(2026,11,4,23,tzinfo=timezone.utc)

def frame():
    index=pd.date_range('2026-11-02',periods=4,tz='UTC')
    return pd.DataFrame({'Open':[100]*4,'High':[102]*4,'Low':[98]*4,'Close':[101]*4,'Adj Close':[100.5]*4,'Volume':[1000]*4},index=index)

def metadata(symbol='TEST'):
    return {'symbol':symbol,'currency':'USD','exchangeTimezoneName':'UTC','exchangeName':'NMS','instrumentType':'EQUITY',
            'currentTradingPeriod':{'regular':{'end':int(datetime(2026,11,4,21,tzinfo=timezone.utc).timestamp())}}}

class MarketDataTests(unittest.TestCase):
    def test_only_completed_sessions_and_buffer(self):
        before=datetime(2026,11,4,21,5,tzinfo=timezone.utc)
        r=instrument_payload('TEST',frame(),metadata(),before)
        self.assertEqual(r['last_session'],'2026-11-03')
        r=instrument_payload('TEST',frame(),metadata(),NOW)
        self.assertEqual(r['last_session'],'2026-11-04')
        self.assertEqual(len(r['history']),3)
        self.assertEqual(r['history'][0]['adjusted_close'],100.5)
        self.assertEqual(instrument_payload('TEST',frame(),metadata(),NOW,True)['last_session'],'2026-11-03')
    def test_metadata_missing_currency_or_ambiguous_symbol_is_not_guessed(self):
        for changes in [{'currency':None},{'symbol':'OTHER'},{'exchangeTimezoneName':'Not/AZone'}]:
            with self.assertRaises(ValueError):instrument_payload('TEST',frame(),{**metadata(),**changes},NOW)
        self.assertEqual(instrument_payload('TEST',frame(),{**metadata(),'currency':'GBp'},NOW)['currency'],'GBX')
    def test_unknown_current_close_and_invalid_ohlc_excluded(self):
        r=instrument_payload('TEST',frame(),{**metadata(),'currentTradingPeriod':{}},NOW)
        self.assertEqual(r['last_session'],'2026-11-03')
        bad=frame();bad['Low']=200
        with self.assertRaises(ValueError):instrument_payload('TEST',bad,metadata(),NOW)
    def test_partial_errors_retain_original_dates_not_new_quote_times(self):
        old=instrument_payload('TEST',frame(),metadata(),NOW)
        class Broken:
            def fetch(self,symbol):raise RuntimeError('source offline')
        result=build_dataset(['TEST'],Broken(),now=NOW,previous={'instruments':{'TEST':old}})
        retained=result['instruments']['TEST']
        self.assertEqual(retained['status'],'retained_after_error')
        self.assertEqual(retained['last_session'],old['last_session'])
        self.assertEqual(retained['fetched_at'],old['fetched_at'])
        self.assertEqual(result['pipeline_status'],'partial')
    def test_fixed_public_universe_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'skew_config.json').write_text(json.dumps({'core_universe':['AAPL']}))
            (root/'portfolio_config.json').write_text(json.dumps({'holdings':[{'symbol':'CNX1.L'}]}))
            self.assertIn('CNX1.L',public_universe(root));self.assertIn('SPY',public_universe(root))
            path=root/'published.json';write_dataset(path,{'schema_version':1})
            self.assertEqual(json.loads(path.read_text())['schema_version'],1)
            self.assertFalse(path.with_suffix('.tmp').exists())

if __name__=='__main__':unittest.main()
