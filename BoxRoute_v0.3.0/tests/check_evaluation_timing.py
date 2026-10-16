"""Small CPU checks for new-condition standalone evaluator, balanced accounting and report gates."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'));os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
import torch
from ultralytics import YOLO
from boxroute.data import prepare_private
from boxroute.evaluate import native_records,prepare_dataset,official_ap,fixed_evaluate
from boxroute.model import make_common,build_conditions
from boxroute.benchmark import HostDetector
from boxroute.reporting import balanced_stats,dump,compare_fixed_summaries
from check_data_pipeline import fixture
from run_boxroute import eval_config,approve_G1,balanced_timing,stamp

torch.set_num_threads(2)

class EvaluationTimingChecks(unittest.TestCase):
    def test_standalone_AP_fixed_GT_and_square_no_training(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=fixture(d/'source');private,manifest=prepare_private(source,d/'private',(4,2))
            data,images,records=prepare_dataset(native_records(manifest),manifest['names'],d/'derived','val')
            native,identity=make_common(ROOT/'sources/official_yolo11n_v830.pt',manifest['names']);path=d/'public_initialized_nc6_fixture.pt'
            torch.save(dict(model=native,ema=None,train_args={}),path);dest=d/'evaluation';dest.mkdir()
            cfg=eval_config('cpu');result=official_ap(YOLO(str(path)),data,manifest['names'],cfg,dest)
            self.assertIn('metrics/mAP75(B)',result['metrics']);self.assertEqual(len((dest/'official/ap_predictions.jsonl').read_text().splitlines()),2)
            self.assertTrue(all(x['shape']==[1,3,320,320] for x in [json.loads(x) for x in (dest/'official/actual_inputs.jsonl').read_text().splitlines()]))
            detector=HostDetector(copy.deepcopy(native).float().eval(),torch.device('cpu'),cfg)
            fixed=fixed_evaluate(detector,images,manifest['names'],cfg,dest)
            self.assertEqual(fixed['images'],2);self.assertEqual(sum(r['GT'] for r in fixed['classes']),2)
            self.assertIn('micro',fixed);self.assertIn('macro_fixed_class_zero_no_predictions',fixed)
    def test_balanced_screen_and_approval_refusal(self):
        blocks=[]
        for i,v in enumerate(['N','G','G','N','G','N','N','G']):blocks.append(dict(block_id=i,variant=v,rows=[dict(latency_ms=10. if v=='N' else 10.2) for _ in range(6)]))
        stats=balanced_stats(blocks,6);self.assertEqual(stats['actual_samples'],48);self.assertTrue(stats['comparison_credible'])
        blocks[0]['rows'][0]['latency_ms']=30.;self.assertFalse(balanced_stats(blocks,6)['comparison_credible'])
        with self.assertRaises(ValueError):approve_G1(None)
    def test_G1_threshold_units_and_no_auto_extension(self):
        from boxroute.decisions import G1_report
        from boxroute.reporting import summarize_counts
        results={}
        for scenario in ['clean','gaussian_0.03']:
            for v,ap in [('N',.4),('S',.419),('U',.41),('G',.42),('W',.41)]:
                fixed=summarize_counts([[dict(class_id=0,name='a',TP=1,FP=0,FN=0,GT=1)]],['a']);fixed['FPPI']=0.
                results[v,scenario]=dict(ap=dict(metrics={'metrics/mAP50-95(B)':ap,'metrics/mAP50(B)':.6,'metrics/mAP75(B)':.5},classes=[dict(class_id=0,name='a',AP50_95=ap)]),fixed=fixed)
        timing=dict(comparison_credible=True,variants={v:dict(overall=dict(mean_ms=10.,p95_ms=11.)) for v in ['N','G']})
        cost=dict(parameter_ratio_G_to_N=1.01,conv_MAC_ratio_G_to_N=1.01)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(G1_report(results,cost,timing,Path(d)),'screen_passed_return_to_Work')
            cost['parameter_ratio_G_to_N']=1.06
            self.assertEqual(G1_report(results,cost,timing,Path(d)),'screen_failed_return_to_Work')
            self.assertFalse(json.loads((Path(d)/'G1_decision.json').read_text())['automatic_next_experiment'])
    def test_CPU_initial_balanced_route_execution_bookkeeping(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=fixture(d/'source',n_val=6);_,manifest=prepare_private(source,d/'private',(4,6))
            native,_=make_common(ROOT/'sources/official_yolo11n_v830.pt',manifest['names']);models=build_conditions(native)
            result=balanced_timing(models,manifest,6,ROOT/('verification/timing_cpu_synthetic_'+stamp()),torch.device('cpu'),synthetic=True,require_equal=True)
            self.assertEqual(result['actual_samples'],48)
            self.assertTrue(all(row['variant'] in ['N','G'] for row in result['blocks']))
            dump(ROOT/'verification/timing_cpu_scope.json',dict(status='bookkeeping_passed',scope='CPU synthetic6images x8blocks, reduced warmup explicitly; not actual GPU speed or G0 384-sample timing',actual_samples=48))

if __name__=='__main__':unittest.main(verbosity=2)
