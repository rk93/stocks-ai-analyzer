import unittest
from prediction_research import original, observations

class PredictionResearchTests(unittest.TestCase):
    def test_strategy_specific_frozen_fields(self):
        r=original({'prediction_id':'a','symbol':'A','strategy_version':'V3','reason':'V1 reason','prediction_reason':'frozen V3 reason','prediction_score':'70','prediction_stop':'99','stop':'50','price':'100'})
        self.assertEqual(r['original_reason'],'frozen V3 reason')
        self.assertEqual(r['stop'],99)
        self.assertIsNone(r['data_availability_cutoff'])
    def test_observation_idempotency_and_no_hit_labels(self):
        r={'prediction_id':'a','return_1d':'.01','result_1d':'CORRECT'}
        a=list(observations([r]));b=list(observations([r]))
        self.assertEqual(a,b)
        self.assertEqual(len(a),1)
        self.assertNotIn('result_1d',a[0])
        self.assertFalse(a[0]['execution_pnl'])

if __name__=='__main__':unittest.main()
