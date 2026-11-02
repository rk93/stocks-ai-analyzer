import unittest
from research_diagnostics import describe, diagnostics

class DiagnosticsTests(unittest.TestCase):
    def test_costs_and_unresolved_coverage(self):
        rows=[{'strategy':'V1','signal_date':'2026-01-01','trade_result_1r':'WIN_1R','realized_r_1r':'1'},
              {'strategy':'V1','signal_date':'2026-01-02','trade_result_1r':'STOP','realized_r_1r':'-1'},
              {'strategy':'V1','signal_date':'2026-01-03','trade_result_1r':'AMBIGUOUS','realized_r_1r':''}]
        result=describe(rows)
        self.assertAlmostEqual(result['expectancy_after_assumed_cost_r'],-.05)
        self.assertEqual(result['unresolved_or_ambiguous'],1)
        self.assertIsNone(result['date_block_bootstrap_95_interval_r'])
        d=diagnostics(rows)
        self.assertEqual(d['status'],'descriptive_only_not_untouched_out_of_sample')
        self.assertEqual(sum(d['strategies']['V1'][k]['signals'] for k in ['earlier_70pct_dates','later_30pct_dates']),3)
    def test_bootstrap_reproducible(self):
        rows=[{'signal_date':str(i),'trade_result_1r':'WIN_1R','realized_r_1r':'1'} for i in range(10)]
        self.assertEqual(describe(rows),describe(rows))

if __name__=='__main__': unittest.main()
