"""Hand-checkable metric tests and explicit random-checkpoint CPU integration."""
import copy, hashlib, json, os, subprocess, sys, tempfile, types, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/detailpair-matplotlib')
from detailpair.reporting import match_image,summarize_counts,latency_stats
from realtime_eval import validate_config


class MetricChecks(unittest.TestCase):
    def test_duplicate_wrong_class_and_empty(self):
        names=['a','b'];gt=[dict(class_id=0,xyxy=[0,0,10,10])]
        preds=[dict(class_id=k,xyxy=[0,0,10,10],confidence=s) for k,s in [(0,.9),(0,.8),(1,.7)]]
        rows,matches=match_image(preds,gt,names)
        self.assertEqual((rows[0]['TP'],rows[0]['FP'],rows[0]['FN']),(1,1,0))
        self.assertEqual(rows[1]['FP'],1)
        missing,_=match_image([],gt,names);self.assertEqual(missing[0]['FN'],1)
        background,_=match_image(preds,[],names);self.assertEqual(sum(r['FP'] for r in background),3)
        empty,_=match_image([],[],names)
        summary=summarize_counts([empty],names)
        self.assertIsNone(summary['classes'][0]['Precision']);self.assertIsNone(summary['classes'][0]['Recall'])
        self.assertIsNone(summary['macro']['Recall'])

    def test_score_order_and_threshold(self):
        preds=[dict(class_id=0,xyxy=[0,0,10,10],confidence=s) for s in [.3,.9]]
        rows,m=match_image(preds,[dict(class_id=0,xyxy=[0,0,10,10])],['a'])
        self.assertEqual(m[0]['prediction_index'],1);self.assertEqual(m[0]['status'],'TP');self.assertEqual(m[1]['status'],'FP')
        miss,_=match_image([dict(class_id=0,xyxy=[20,20,30,30],confidence=.9)],[dict(class_id=0,xyxy=[0,0,10,10])],['a'])
        self.assertEqual((miss[0]['FP'],miss[0]['FN']),(1,1))

    def test_statistics_recompute(self):
        stats=latency_stats([dict(latency_ms=n) for n in [1,2,3,4]])
        self.assertEqual(stats['n'],4);self.assertEqual(stats['mean_ms'],2.5)
        self.assertEqual(stats['p50_ms'],2.5);self.assertAlmostEqual(stats['p95_ms'],3.85)
        self.assertEqual(stats['serial_fps'],400.)
        with self.assertRaises(ValueError):latency_stats([])

    def test_host_pipeline_and_real_nms(self):
        import torch, numpy as np
        from ultralytics.data.augment import LetterBox
        from detailpair.benchmark import HostDetector
        detector=HostDetector.__new__(HostDetector)
        detector.device=torch.device('cpu');detector.nc=2
        detector.config=dict(imgsz=320,conf=.25,nms_iou=.7,max_det=300)
        detector.letterbox=LetterBox((320,320),auto=False);detector.shape=None
        capture=[]
        # Three known boxes: class0 duplicate suppressed, class1 overlap retained.
        raw=torch.tensor([[[160.,160.,160.],[160.,160.,160.],[80.,80.,80.],[80.,80.,80.],
                           [.9,.8,.01],[.01,.01,.85]]])
        def backend(tensor):capture.append(tensor);return raw.clone()
        detector.backend=backend
        image=np.zeros((160,160,3),dtype=np.uint8);image[:]=[10,20,30]
        pred,elapsed=detector.timed(image)
        self.assertEqual(pred.shape,(2,6));self.assertGreater(elapsed,0)
        self.assertEqual(detector.shape,[1,3,320,320])
        np.testing.assert_allclose(pred[0,:4],[60,60,100,100])
        self.assertEqual(set(pred[:,5]),{0.,1.})
        torch.testing.assert_close(capture[0][0,:,0,0],torch.tensor([30,20,10])/255.)

    def test_child_exit_and_gpu_conflict(self):
        from compare_long import run_child
        child=run_child([sys.executable,'-c','import sys;sys.exit(7)'],os.environ.copy(),ROOT)
        self.assertEqual(child.returncode,7)
        from detailpair.benchmark import gpu_guard
        with tempfile.TemporaryDirectory() as d:
            fake=types.SimpleNamespace(returncode=0,stdout='GPU-x, 999999, prunecheck.exe, 123\n',stderr='')
            with patch('detailpair.benchmark.subprocess.run',return_value=fake):
                with self.assertRaisesRegex(RuntimeError,'prunecheck'):gpu_guard(types.SimpleNamespace(type='cuda'),Path(d)/'gpu.json')
            self.assertTrue((Path(d)/'gpu.json').exists())


def integration():
    import torch,yaml,numpy as np
    from detailpair.data import synthetic,digest,load_spec,label_for
    from detailpair.model import PairDetectionModel
    from detailpair.evaluation import read_records,prepare_dataset,validate_checkpoint,verify_selection
    with tempfile.TemporaryDirectory(prefix='DetailPair CPU check ') as d:
        d=Path(d);data=synthetic(d/'input data');names,splits=load_spec(data)
        selected={split:[dict(image=str(p),label=str(label_for(p)),image_sha256=digest(p),label_sha256=digest(label_for(p))) for p in splits[split]] for split in ['train','val']}
        selection=d/'selection.json';selection.write_text(json.dumps(selected))
        config=yaml.safe_load((ROOT/'configs/realtime_comparison.yaml').read_text())
        config.update(data=str(data),device='cpu',synthetic_check=True,expected_images=4,expected_gt=4,warmup=2)
        originals={str(f):digest(f) for f in (d/'input data').rglob('*') if f.is_file()}
        for entry in config['variants']:
            v=entry['variant'];model=PairDetectionModel(nc=1,variant=v);model.names={0:names[0]}
            path=d/(v+' random check.pt')
            entry.update(weights=str(path),selection=str(selection),run_name='synthetic_fixture',prior_metrics=str(d/'absent_old_metrics.json'))
            torch.save(dict(model=model.half(),ema=None,train_args=dict(seed=0,project='synthetic_fixture/'+v+'_s0',name='training'),epoch=-1,optimizer=None),path)
            originals[str(path)]=digest(path)
            loaded,identity=validate_checkpoint(entry,config,names)
            for changed in [dict(entry,seed=1),dict(entry,run_name='other_run')]:
                try:validate_checkpoint(changed,config,names)
                except ValueError:pass
                else:raise AssertionError('Wrong seed/run accepted')
            try:validate_checkpoint(dict(entry,weights=str(d/'missing.pt')),config,names)
            except FileNotFoundError:pass
            else:raise AssertionError('Missing checkpoint accepted')
            self_entry=dict(entry,variant='B6' if v=='B2' else 'B2')
            try:validate_checkpoint(self_entry,config,names)
            except ValueError:pass
            else:raise AssertionError('Wrong variant accepted')
        validate_config(config,True)
        bad=copy.deepcopy(config);bad['repeats']=1
        try:validate_config(bad,True)
        except ValueError:pass
        else:raise AssertionError('Changed measurement repeats accepted')
        # Failure preflight detects a hash change, restores only this test-owned fixture.
        label=label_for(splits['val'][0]);original=label.read_bytes();label.write_bytes(original+b'\n')
        try:verify_selection(config['variants'][0],data)
        except ValueError:pass
        else:raise AssertionError('Modified source label accepted')
        label.write_bytes(original)
        # Shared corruption generation must be byte-reproducible and never modify source.
        _,rows=read_records(data,'val')
        a=prepare_dataset(rows,names,d/'noise a','val',dict(kind='gaussian',value=.03))[2]
        b=prepare_dataset(rows,names,d/'noise b','val',dict(kind='gaussian',value=.03))[2]
        assert [r['derived_image_sha256'] for r in a]==[r['derived_image_sha256'] for r in b]
        cfg=d/'CPU config.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
        before=set((ROOT/'runs').glob('realtime_eval_*'))
        result=subprocess.run([sys.executable,str(ROOT/'realtime_eval.py'),'--config',str(cfg),'--synthetic-check'],cwd=ROOT,check=False)
        assert result.returncode==0,'Full CPU integration failed'
        created=set((ROOT/'runs').glob('realtime_eval_*'))-before;assert len(created)==1
        out=created.pop();status=json.loads((out/'status.json').read_text())
        assert status['status']=='success' and status['successful_evaluations']==10
        assert status['actual_latency_samples']==24
        for v in ['B2','B6']:
            timing=[json.loads(line) for line in (out/v/'timing/latency.jsonl').read_text().splitlines()]
            assert len(timing)==12 and all(r['actual_shape']==[1,3,320,320] for r in timing)
            summary=json.loads((out/v/'timing/latency_summary.json').read_text())
            assert abs(summary['overall']['mean_ms']-np.mean([r['latency_ms'] for r in timing]))<1e-10
            for scenario in ['clean','gaussian_0.01','gaussian_0.03','gaussian_0.06','impulse_0.01']:
                dest=out/v/scenario
                fixed=json.loads((dest/'fixed_metrics.json').read_text());assert fixed['classes'][0]['GT']==4
                assert len((dest/'official/ap_predictions.jsonl').read_text().splitlines())==4
                assert len((dest/'fixed_predictions_matches.jsonl').read_text().splitlines())==4
        assert all(digest(p)==h for p,h in originals.items()),'Original images/labels/weights mutated'
        # Actual CLI failure path: missing config saves failure + ZIP, returns nonzero.
        before=set((ROOT/'runs').glob('realtime_eval_*'))
        failed=subprocess.run([sys.executable,str(ROOT/'realtime_eval.py'),'--config',str(d/'missing.yaml')],cwd=ROOT,check=False)
        assert failed.returncode==1
        failure=(set((ROOT/'runs').glob('realtime_eval_*'))-before).pop()
        assert json.loads((failure/'status.json').read_text())['status']=='failed' and (failure/'failure.txt').is_file()
        from realtime_eval import pack_results
        archive=pack_results()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert not any('/datasets/' in n or n.endswith(('.pt','.cache')) for n in z.namelist())
            assert any('failure.txt' in n for n in z.namelist())
        print(json.dumps(dict(status='passed',scope='random checkpoint CPU synthetic plumbing only',run=str(out),failed_run=str(failure),
            models=2,scenarios=5,latency_samples=24,ap_prediction_records=40,weights_and_sources_unchanged=True)))


if __name__=='__main__':
    if '--integration' in sys.argv:
        sys.argv.remove('--integration')
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricChecks))
        if not result.wasSuccessful():sys.exit(1)
        integration()
    else:unittest.main()
