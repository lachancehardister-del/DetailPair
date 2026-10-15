"""Offline metric definitions and comparison regression; no model execution."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from detailpair.reporting import summarize_counts,compare_fixed_summaries
from matched_checkpoint_eval import comparison


def summary(rows):
    return summarize_counts([[dict(class_id=i,name=str(i),TP=tp,FP=fp,FN=gt-tp,GT=gt) for i,(tp,fp,gt) in enumerate(rows)]],[str(i) for i in range(len(rows))])

class ReportingChecks(unittest.TestCase):
    def test_no_predictions_with_GT_keeps_null(self):
        s=summary([(0,0,3),(1,1,2)])
        self.assertIsNone(s['classes'][0]['Precision']);self.assertEqual(s['classes'][0]['Recall'],0.)
        self.assertEqual(s['macro']['Precision_class_ids'],[1]);self.assertEqual(s['macro']['Precision_classes_with_denominator'],1)
        self.assertEqual(s['macro']['Recall_class_ids'],[0,1]);self.assertEqual(s['macro']['Recall_classes_with_denominator'],2)
        self.assertEqual(s['macro']['Precision'],.5);self.assertEqual(s['macro_fixed_class_zero_no_predictions']['Precision'],.25)
        self.assertEqual(s['micro']['Precision'],.5)
    def test_FP_without_TP_is_defined_zero(self):
        s=summary([(0,2,3),(1,1,2)])
        self.assertEqual(s['classes'][0]['Precision'],0.);self.assertEqual(s['macro']['Precision_class_ids'],[0,1])
        self.assertEqual(s['macro']['Precision'],.25);self.assertEqual(s['micro']['Precision'],.25)
        self.assertEqual(s['classes'][0]['FP'],2);self.assertEqual(s['classes'][0]['FN'],3)
    def test_all_classes_have_predictions_macro_differs_from_micro(self):
        s=summary([(1,0,1),(1,3,2)])
        self.assertEqual(s['macro']['Precision'],.625);self.assertEqual(s['macro_fixed_class_zero_no_predictions']['Precision'],.625)
        self.assertEqual(s['micro']['Precision'],.4);self.assertEqual(s['macro']['Precision_classes_with_denominator'],2)
    def test_same_support_count_different_class_sets_not_comparable(self):
        a=summary([(1,0,1),(0,0,1)]);b=summary([(0,0,1),(1,0,1)])
        r={r['metric']:r for r in compare_fixed_summaries(a,b)}
        self.assertEqual(r['macro_Precision']['B5_support_count'],r['macro_Precision']['B6_support_count'])
        self.assertFalse(r['macro_Precision']['comparable']);self.assertIsNone(r['macro_Precision']['B5_minus_B6'])
        self.assertIn('different_class_support',r['macro_Precision']['reason'])
        self.assertTrue(r['macro_Precision_zero_no_predictions']['comparable'])
        self.assertTrue(r['macro_Recall']['comparable'])
        # Recall must use its actual support too, even if the precision supports match.
        c=summary([(1,0,1),(0,1,0)]);d=summary([(1,0,1),(0,1,2)])
        recall=next(r for r in compare_fixed_summaries(c,d) if r['metric']=='macro_Recall')
        self.assertIsNone(recall['B5_minus_B6']);self.assertFalse(recall['comparable'])
    def test_empty_denominators_and_mapping_guard(self):
        s=summary([(0,0,0),(0,0,0)])
        self.assertIsNone(s['macro']['Precision']);self.assertIsNone(s['micro']['Precision']);self.assertIsNone(s['micro']['Recall'])
        self.assertEqual(s['macro_fixed_class_zero_no_predictions']['Precision'],0.)
        self.assertIsNone(compare_fixed_summaries(s,s)[0]['B5_minus_B6'])
        bad=copy.deepcopy(s);bad['classes'][0]['name']='other'
        with self.assertRaises(ValueError):compare_fixed_summaries(s,bad)
    def test_comparison_preserves_AP_counts_legacy_values_and_inputs(self):
        a=summary([(0,1,1),(2,1,3)]);b=summary([(0,0,1),(1,0,3)])
        # Legacy summaries without new support keys are still safe to compare.
        for s in [a,b]:
            s['FPPI']=sum(r['FP'] for r in s['classes'])/2
            s['macro'].pop('Precision_class_ids');s['macro'].pop('Recall_class_ids')
        def entry(s,ap):
            return dict(ap=dict(metrics={'metrics/mAP50-95(B)':ap,'metrics/mAP50(B)':ap+.1},classes=[dict(class_id=i,name=str(i),AP50=ap+.1,AP50_95=ap) for i in range(2)]),fixed=s)
        results={('B5','clean'):entry(a,.2),('B6','clean'):entry(b,.3)};before=copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as d:
            comparison(results,Path(d));out=json.loads((Path(d)/'comparison.json').read_text());rows={r['metric']:r for r in out['rows']}
            self.assertAlmostEqual(rows['metrics/mAP50-95(B)']['B5_minus_B6'],-.1)
            self.assertIsNone(rows['macro_Precision']['B5_minus_B6']);self.assertEqual(rows['macro_Precision']['B5_support_count'],2)
            self.assertEqual(rows['macro_Precision']['B6_support_count'],1)
            self.assertTrue(rows['macro_Precision_zero_no_predictions']['comparable'])
            self.assertEqual([r['B5'] for r in out['classes'] if r['metric']=='TP'],[0,2])
            self.assertIn('different_class_support',(Path(d)/'comparison_B5_minus_B6.csv').read_text())
        self.assertEqual(results,before)

if __name__=='__main__':unittest.main(verbosity=2)
