"""Hand-checkable train-only geometry distributions, not inference or power analysis."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from boxroute.data import shape_statistics

class ShapeStatisticsChecks(unittest.TestCase):
    def test_no_train_GT_is_blocked_not_a_pilot(self):
        stats=shape_statistics(dict(names=['empty'],records=dict(train=[dict(width=100,height=100,boxes=[])])))
        self.assertEqual(stats['status'],'blocked_shape_support');self.assertEqual(stats['informative_boxes'],0);self.assertIsNone(stats['q_distribution'])
    def test_geometry_histograms_quantiles_and_coverage(self):
        manifest=dict(names=['square','horizontal','vertical'],records=dict(train=[dict(width=100,height=100,boxes=[[0,.5,.5,.2,.2],[1,.5,.5,.7,.1],[2,.5,.5,.1,.7]])]))
        stats=shape_statistics(manifest)
        self.assertEqual(stats['informative_boxes'],2);self.assertEqual(stats['status'],'blocked_shape_support')
        self.assertEqual(stats['mean_q'],[.5,.25,.25])
        for group in stats['q_distribution'].values():self.assertEqual(sum(group['histogram_counts']),3)
        self.assertEqual(stats['q_distribution']['I']['histogram_counts'][2],2)
        self.assertEqual(stats['q_distribution']['I']['histogram_counts'][9],1)
        self.assertEqual(stats['q_distribution']['H']['quantiles'][-1],.75)
        self.assertEqual(stats['class_coverage']['horizontal']['informative_boxes'],1)
        self.assertEqual(stats['class_coverage']['square']['informative_boxes'],0)

if __name__=='__main__':unittest.main(verbosity=2)
