"""Balanced schedule/statistics/failure checks and explicit random CPU integration."""
import copy,json,os,sys,tempfile,subprocess,unittest,contextlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(exist_ok=True,parents=True)
from detailpair.reporting import balanced_stats

ORDER=['B2','B6','B6','B2','B6','B2','B2','B6']

class RecheckTests(unittest.TestCase):
    def blocks(self):
        return [dict(block_id=i,variant=v,rows=[dict(latency_ms=2. if v=='B2' else 3.) for _ in range(6)]) for i,v in enumerate(ORDER)]
    def test_balanced_windows_and_positions(self):
        b=self.blocks();s=balanced_stats(b,6)
        self.assertTrue(s['complete']);self.assertTrue(s['comparison_credible'])
        self.assertEqual(s['actual_samples'],48);self.assertEqual(len(s['cycles']),2)
        self.assertEqual([p['first_variant'] for p in s['adjacent_pairs']],['B2','B6','B6','B2'])
        self.assertAlmostEqual(s['variants']['B2']['overall']['serial_fps'],500.)
    def test_drift_and_incomplete_preserve_samples(self):
        b=self.blocks();b[0]['rows'][0]['latency_ms']=10.
        s=balanced_stats(b,6);self.assertEqual(s['timing_status'],'timing_unstable')
        self.assertFalse(s['comparison_credible']);self.assertEqual(s['actual_samples'],48)
        s=balanced_stats(b[:-1],6);self.assertFalse(s['complete']);self.assertEqual(s['actual_samples'],42)
    def test_buffered_schedule_and_partial(self):
        import torch,numpy as np
        from detailpair.benchmark import balanced_measure
        class FakePool:
            device=torch.device('cpu')
            def __init__(self,fail_at=None):self.calls=0;self.active=None;self.fail_at=fail_at
            @contextlib.contextmanager
            def lease(self,v):
                assert self.active is None;self.active=v
                pool=self
                class Detector:
                    shape=[1,3,320,320];metadata={'fixture':True}
                    def timed(self,image):
                        pool.calls+=1
                        if pool.calls==pool.fail_at:raise RuntimeError('simulated block failure')
                        return np.empty((0,6)),2. if v=='B2' else 3.
                try:yield Detector()
                finally:self.active=None
        c=dict(block_order=ORDER,common_warmup_seconds=1e-9,block_warmup_iterations=2,drift_threshold=1.10,hardware_notes={})
        ids={v:dict(variant=v,seed=0,run_name='fixture',weights_sha256=v) for v in ['B2','B6']}
        images=[dict(image_id=str(i),bgr=np.zeros((4,4,3),dtype=np.uint8)) for i in range(6)]
        with tempfile.TemporaryDirectory() as d,patch('detailpair.benchmark.device_snapshot'):
            out=Path(d)/'good';s=balanced_measure(FakePool(),images,ids,c,out)
            self.assertEqual(s['actual_samples'],48)
            rows=[json.loads(x) for x in (out/'warmup.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),20)
            self.assertEqual(len((out/'latency_all.jsonl').read_text().splitlines()),48)
            bad=Path(d)/'bad'
            with self.assertRaisesRegex(RuntimeError,'simulated'):balanced_measure(FakePool(fail_at=9),images,ids,c,bad)
            self.assertEqual(len((bad/'block_00_B2/latency_partial.jsonl').read_text().splitlines()),2)
            self.assertEqual(json.loads((bad/'balanced_summary_partial.json').read_text())['actual_samples'],2)
    def test_windows_style_gpu_unknown_fields(self):
        import torch,types
        from detailpair.benchmark import device_snapshot
        def response(command,**kwargs):
            text='name, power.draw [W], temperature.gpu\nRTX 4060 Laptop GPU, [N/A], 55\n' if command[1].startswith('--query-gpu=') else f'GPU-fixture, {os.getpid()}, python.exe, [N/A]\n'
            return types.SimpleNamespace(returncode=0,stdout=text,stderr='')
        with tempfile.TemporaryDirectory() as d,patch('detailpair.benchmark.subprocess.run',side_effect=response):
            snapshot=device_snapshot(torch.device('cuda:0'),Path(d)/'state.json',dict(plugged_in='unknown'))
            self.assertEqual(snapshot['gpu_fields'][0]['power.draw [W]'],'unknown')
            self.assertEqual(snapshot['gpu_fields'][0]['temperature.gpu'],'55')
            self.assertEqual(snapshot['plugged_in'],'unknown')

    def test_actual_pool_serial_device_lease(self):
        import torch,types
        from detailpair.benchmark import DetectorPool
        class Backend:
            def __init__(self):self.moves=[]
            def to(self,device):self.moves.append(str(device));return self
        detectors=[]
        def host(*args):
            d=types.SimpleNamespace(backend=Backend(),device=torch.device('cpu'));detectors.append(d);return d
        models={v:types.SimpleNamespace(model=torch.nn.Linear(2,2)) for v in ['B2','B6']}
        with patch('detailpair.benchmark.HostDetector',side_effect=host),patch('torch.cuda.empty_cache'):
            pool=DetectorPool(models,torch.device('cuda:0'),{})
            with pool.lease('B2') as active:
                self.assertEqual(active.device.type,'cuda')
                with self.assertRaises(RuntimeError):
                    with pool.lease('B6'):pass
            self.assertIsNone(pool.active);self.assertEqual(detectors[0].backend.moves,['cuda:0','cpu'])
            self.assertEqual(detectors[1].backend.moves,[])

    def test_torchvision_bootstrap_geometry(self):
        import torch
        from detailpair.benchmark import initialize_torchvision_nms
        result=initialize_torchvision_nms(torch.device('cpu'))
        self.assertEqual(result['indices'],[0]);self.assertEqual(result['iou_threshold'],.7)

    def test_new_protocol_does_not_relax_old(self):
        import yaml
        from timing_recheck import validate_recheck_config
        c=yaml.safe_load((ROOT/'configs/balanced_recheck.yaml').read_text())
        validate_recheck_config(c,False)
        for field,value in [('device','cpu'),('block_order',ORDER[::-1]),('drift_threshold',1.5),('split','test')]:
            b=copy.deepcopy(c);b[field]=value
            with self.assertRaises(ValueError):validate_recheck_config(b,False)
        b=copy.deepcopy(c);b['variants'][0]['expected_sha256']='0'*64
        with self.assertRaises(ValueError):validate_recheck_config(b,False)
        from realtime_eval import validate_config
        old=yaml.safe_load((ROOT/'configs/realtime_comparison.yaml').read_text());validate_config(old,False)
        with self.assertRaises(ValueError):validate_config(c,False)


def integration():
    import torch,yaml,numpy as np,zipfile,hashlib
    from PIL import Image
    from detailpair.data import load_spec,label_for,digest
    from detailpair.model import PairDetectionModel
    with tempfile.TemporaryDirectory(prefix='DetailPair v026 CPU ') as d:
        d=Path(d);data_root=d/'original inputs';names=['synthetic_rectangle']
        for split,n in [('train',3),('val',6),('test',1)]:
            images=data_root/'images'/split;labels=data_root/'labels'/split;images.mkdir(parents=True);labels.mkdir(parents=True)
            for i in range(n):
                rng=np.random.default_rng(i+{'train':0,'val':100,'test':200}[split])
                a=rng.integers(0,255,(60+i%2*8,80,3),dtype=np.uint8)
                Image.fromarray(a).save(images/f'{i}.jpg',quality=90)
                (labels/f'{i}.txt').write_text('0 0.5 0.5 0.3 0.4\n')
        data=data_root/'data.yaml';data.write_text(yaml.safe_dump(dict(path=str(data_root),names=names,train='images/train',val='images/val',test='images/test')))
        _,splits=load_spec(data)
        selected={split:[dict(image=str(p),label=str(label_for(p)),image_sha256=digest(p),label_sha256=digest(label_for(p)),
            working_image=str(d/'old working'/f'{i:06d}_old.jpg')) for i,p in enumerate(reversed(splits[split]))] for split in ['train','val']}
        selection=d/'selection.json';selection.write_text(json.dumps(selected))
        c=yaml.safe_load((ROOT/'configs/balanced_recheck.yaml').read_text());c.update(synthetic_check=True,device='cpu',data=str(data),expected_images=6,expected_gt=6,common_warmup_seconds=.001,block_warmup_iterations=2)
        sources={str(p):digest(p) for p in data_root.rglob('*') if p.is_file()}
        for e in c['variants']:
            v=e['variant'];m=PairDetectionModel(nc=1,variant=v);m.names={0:names[0]}
            weight=d/(v+' random.pt');torch.save(dict(model=m.half(),ema=None,epoch=-1,optimizer=None,
                train_args=dict(seed=0,project='fixture_run/'+v+'_s0',name='training')),weight)
            e.update(weights=str(weight),expected_sha256=digest(weight),selection=str(selection),run_name='fixture_run',prior_metrics=str(d/'absent_old_metrics.json'))
            sources[str(weight)]=digest(weight)
        cfg=d/'CPU synthetic.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False))
        before=set((ROOT/'runs').glob('recheck_*'))
        r=subprocess.run([sys.executable,str(ROOT/'timing_recheck.py'),'--config',str(cfg),'--synthetic-check'],cwd=ROOT)
        new=set((ROOT/'runs').glob('recheck_*'))-before;assert len(new)==1;out=new.pop()
        assert r.returncode==0,'CPU integration failed: '+str(out)
        status=json.loads((out/'status.json').read_text());assert status['status']=='success' and status['timing_data_complete']
        summary=json.loads((out/'timing/balanced_summary.json').read_text());assert summary['actual_samples']==48
        assert [b['variant'] for b in summary['blocks']]==ORDER
        assert all(len(b['windows'])==3 for b in summary['blocks'])
        for v in ['B2','B6']:
            dest=out/'clean_diagnostic'/v
            for arm in ['legacy_jpeg','new_png']:
                for name in ['decoded_pixels.jsonl','input_tensors.jsonl','predictions_official_matches.jsonl']:
                    assert len((dest/arm/name).read_text().splitlines())==6
            effective=json.loads((dest/'legacy_jpeg/effective_validator.json').read_text())
            assert [r['image_id'] for r in effective['loader_order']]!=[f'{i:06d}' for i in range(6)],'Original selection order was not restored'
            assert (dest/'comparison.json').exists()
        # Exercise the predeclared rect-only implementation even if no extra factor is needed by the fixture's automatic criterion.
        from detailpair.evaluation import diagnostic_ap
        png_manifest=json.loads((out/'clean_png_manifest.json').read_text())['records']
        mapping={str(Path(row['working_image']).resolve()):row['image_id'] for row in png_manifest}
        controlled=diagnostic_ap(c['variants'][0],c,names,out/'datasets/new_png/data.yaml',mapping,'png_rect_only',out/'rect_control_fixture_only')
        assert controlled[1]['validator_args']['rect'] is True and len(controlled[2])==6
        assert all(digest(path)==h for path,h in sources.items()),'Original data/weights changed'
        from timing_recheck import pack_recheck
        archive=pack_recheck()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert not any('/datasets/' in n or n.endswith(('.pt','.jpg','.png','.cache')) for n in z.namelist())
        before=set((ROOT/'runs').glob('recheck_*'))
        r=subprocess.run([sys.executable,str(ROOT/'timing_recheck.py'),'--config',str(d/'missing.yaml')],cwd=ROOT)
        assert r.returncode==1
        failed=(set((ROOT/'runs').glob('recheck_*'))-before).pop();assert (failed/'failure.txt').exists()
        print(json.dumps(dict(status='passed',scope='CPU random checkpoints/synthetic JPEG engineering only',run=str(out),
            expected_failure=str(failed),timing_samples=48,balanced_blocks=8,diagnostic_images_per_arm=6,
            read_only_sources=True,Windows_GPU='pending')))

if __name__=='__main__':
    if '--integration' in sys.argv:
        sys.argv.remove('--integration');result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecheckTests))
        if not result.wasSuccessful():sys.exit(1)
        integration()
    else:unittest.main()
