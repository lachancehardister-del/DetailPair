"""Hand-checkable local gradients, original-loss equality and read-only CPU integration."""
import contextlib,copy,hashlib,json,os,subprocess,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'));(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from detailpair.model import PairDetectionModel,region_loss,amplitude_loss
from detailpair.mechanism import prepare_loss_mode,analyze_batch,decomposed_loss,region_components,gradients,gradient_groups,state_identity
from ultralytics.cfg import DEFAULT_CFG_DICT

torch.set_num_threads(2)


def fixture_model(variant):
    torch.manual_seed(7);m=PairDetectionModel(nc=1,variant=variant);m.args=copy.deepcopy(DEFAULT_CFG_DICT)
    for p in m.parameters():p.requires_grad_(False)
    return m


def fixture_batch(n=2,size=64):
    gen=torch.Generator().manual_seed(19)
    return dict(img=torch.rand(n,3,size,size,generator=gen),batch_idx=torch.arange(n,dtype=torch.float32),
        cls=torch.zeros(n,1),bboxes=torch.tensor([[.5,.5,.3,.4]]).repeat(n,1))


class MechanismTests(unittest.TestCase):
    def test_scalar_gate_derivative_and_stop_gradient(self):
        for A in [.2,1.,3.]:
            for gate in [.2,.5,.8]:
                g=torch.tensor(gate,dtype=torch.float64,requires_grad=True)
                target=torch.tensor([[[[A]]]],dtype=torch.float64,requires_grad=True)
                loss=amplitude_loss((g*A).reshape(1,1,1,1),target,torch.ones_like(target))
                dg,dt=torch.autograd.grad(loss,[g,target],allow_unused=True)
                expected=-A/(A+.001);self.assertAlmostEqual(float(dg),expected,places=12);self.assertIsNone(dt)
                f=lambda x:abs(x*A-A)/(A+.001)
                self.assertAlmostEqual((f(gate+1e-6)-f(gate-1e-6))/2e-6,expected,places=8)
    def test_region_empty_and_zero_amplitude(self):
        for fill in [0.,1.]:
            p=torch.rand(2,3,4,4,requires_grad=True);target=torch.rand_like(p,requires_grad=True);mask=torch.full((2,1,4,4),fill)
            fg,bg=region_components(p,target,mask)
            torch.testing.assert_close(fg+bg,region_loss(p,target,mask))
            self.assertEqual(float(fg if fill==0 else bg),0.)
            gp,gt=torch.autograd.grad(fg+bg,[p,target],allow_unused=True);self.assertTrue(torch.isfinite(gp).all());self.assertIsNone(gt)
        gate=torch.tensor(.5,requires_grad=True);zero=(gate*0).reshape(1,1,1,1)
        loss=amplitude_loss(zero,torch.zeros_like(zero),torch.ones_like(zero));self.assertEqual(float(loss),0.)
        self.assertEqual(float(torch.autograd.grad(loss,gate)[0]),0.)
    def test_original_total_gradients_batch_scaling_and_state(self):
        for variant in ['B2','B6']:
            m=fixture_model(variant);named,groups,mode=prepare_loss_mode(m);before=state_identity(m)
            self.assertTrue(all(not module.training for module in m.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)))
            for n in [1,2]:
                row=analyze_batch(m,fixture_batch(n),2,named,groups)
                self.assertEqual(row['consistency']['status'],'passed')
                self.assertAlmostEqual(row['weighted']['C'],n*.05*row['unweighted']['C'],places=5)
                if variant=='B6':
                    self.assertAlmostEqual(row['weighted']['A'],n*.02*row['unweighted']['A'],places=5)
                    self.assertEqual(row['gradient_groups']['projections']['losses']['A']['status'],'disconnected')
                    self.assertAlmostEqual(sum(region['amplitude_full_weighted_contribution'] for band in row['gate_subbands'] for region in band['regions'].values()),row['weighted']['A'],places=5)
                    self.assertIsNone(row['gradient_groups']['projections']['comparisons']['D_vs_A']['cos'])
                else:self.assertEqual(row['gradient_groups']['whole_layer']['losses']['A']['status'],'not_applicable')
            self.assertEqual(before,state_identity(m))
    def test_checkpoint_args_restore_without_guessed_coefficients(self):
        m=fixture_model('B2');m.args={'imgsz':320,'task':'detect'}
        with self.assertRaisesRegex(ValueError,'no guessed'):prepare_loss_mode(m,dict(box=7.5,cls=.5))
        named,groups,mode=prepare_loss_mode(m,dict(DEFAULT_CFG_DICT,box=7.5,cls=.5,dfl=1.5))
        self.assertEqual(mode['loss_hyperparameter_source'],'checkpoint train_args')
        self.assertEqual(m.args.dfl,1.5)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError,'Autograd'):prepare_loss_mode(m,DEFAULT_CFG_DICT)
        next(module for module in m.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)).train()
        with self.assertRaisesRegex(RuntimeError,'Frozen BN'):analyze_batch(m,fixture_batch(),0,named,groups)

    def test_hook_cleanup_on_exception(self):
        m=fixture_model('B6');prepare_loss_mode(m);layer=m.model[1]
        def fail(*args):raise RuntimeError('injected criterion failure')
        m.criterion=fail
        before=len(layer._forward_hooks)
        with self.assertRaisesRegex(RuntimeError,'injected'):decomposed_loss(m,fixture_batch(),0)
        self.assertFalse(layer.capture);self.assertIsNone(layer.cache);self.assertEqual(len(layer._forward_hooks),before)
    def test_zero_norm_and_disconnected_are_distinct(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));named=[('p',p)];groups={'all':[0]}
        g=gradient_groups({'D':(torch.ones(2),),'C':(torch.zeros(2),),'A':(None,), 'C_plus_A':(None,)},named,groups)['all']
        self.assertEqual(g['losses']['C']['status'],'zero_norm');self.assertEqual(g['losses']['A']['status'],'disconnected')
        self.assertEqual(g['losses']['C']['L2_norm'],0.);self.assertIsNone(g['losses']['A']['L2_norm'])
        self.assertIsNone(g['comparisons']['D_vs_C']['cos']);self.assertIsNone(g['comparisons']['D_vs_A']['cos'])
    def test_letterbox_targets_train_only_and_seed_selection(self):
        import numpy as np,yaml
        from PIL import Image
        from detailpair.data import digest
        from detailpair.mechanism_data import choose_training,load_batch
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);images=d/'images/train';labels=d/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
            for i in range(4):
                Image.fromarray(np.zeros((100,200,3),dtype=np.uint8)).save(images/f'{i}.jpg');(labels/f'{i}.txt').write_text('0 .5 .5 .5 .5\n')
            data=d/'data.yaml';data.write_text(yaml.safe_dump(dict(path=str(d),train='images/train',names=['fixture'],val='nonexistent-val',test='nonexistent-test')))
            selection=d/'selection.json';selection.write_text(json.dumps(dict(train=[dict(image=str(p),label=str(labels/(p.stem+'.txt')),image_sha256=digest(p),label_sha256=digest(labels/(p.stem+'.txt'))) for p in images.glob('*.jpg')])))
            entries=[dict(variant=v,selection=str(selection)) for v in ['B2','B6']]
            names,a,meta=choose_training(data,entries,2,178,4);_,b,_=choose_training(data,entries,2,178,4)
            self.assertEqual(a,b)
            batch,info=load_batch(a,torch.device('cpu'))
            torch.testing.assert_close(batch['bboxes'],torch.tensor([[.5,.5,.5,.25],[.5,.5,.5,.25]]))
            self.assertEqual(list(batch['img'].shape),[2,3,320,320]);self.assertEqual(meta['selected_images'],2)
    def test_frozen_protocol_and_nonindependent_summary(self):
        import yaml
        from mechanism_diagnose import validate_config,summarize_batches
        c=yaml.safe_load((ROOT/'configs/mechanism_diagnosis.yaml').read_text());validate_config(c,False)
        for key,value in [('device','cpu'),('selected_images',128),('amplitude_weight',.01),('batch',1)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad,False)
        records=[dict(status='failed',batch_index=0),dict(status='not_run',batch_index=1)]
        s=summarize_batches(records,2);self.assertEqual(s['failed_batches'],1);self.assertEqual(s['not_run_batches'],1)
        self.assertEqual(s['successful_batches'],0)


def integration():
    import yaml,numpy as np
    from PIL import Image
    from detailpair.data import digest
    with tempfile.TemporaryDirectory(prefix='DetailPair mechanism CPU ') as d:
        d=Path(d);images=d/'images/train';labels=d/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
        for i in range(6):
            a=np.random.default_rng(i).integers(0,255,(60,80,3),dtype=np.uint8);Image.fromarray(a).save(images/f'{i}.jpg')
            (labels/f'{i}.txt').write_text('0 .5 .5 .3 .4\n')
        data=d/'data.yaml';data.write_text(yaml.safe_dump(dict(path=str(d),train='images/train',names=['synthetic_rectangle'],val='absent',test='absent')))
        selected=[dict(image=str(p),label=str(labels/(p.stem+'.txt')),image_sha256=digest(p),label_sha256=digest(labels/(p.stem+'.txt'))) for p in sorted(images.glob('*.jpg'))]
        selection=d/'selection.json';selection.write_text(json.dumps(dict(train=selected)))
        sources={str(p):digest(p) for p in d.rglob('*') if p.is_file()}
        c=yaml.safe_load((ROOT/'configs/mechanism_diagnosis.yaml').read_text());c.update(synthetic_check=True,device='cpu',data=str(data),expected_train_images=6,selected_images=4)
        for e in c['variants']:
            model=fixture_model(e['variant']);model.names={0:'synthetic_rectangle'}
            weight=d/(e['variant']+' random.pt')
            torch.save(dict(model=model.half(),ema=None,train_args=dict(DEFAULT_CFG_DICT,seed=0,imgsz=320,batch=2,project='fixture/'+e['variant']+'_s0',name='training'),epoch=-1,optimizer=None),weight)
            e.update(weights=str(weight),selection=str(selection),expected_sha256=digest(weight),run_name='fixture')
            sources[str(weight)]=digest(weight)
        cfg=d/'CPU mechanism.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False))
        before=set((ROOT/'runs').glob('mechanism_*'))
        r=subprocess.run([sys.executable,str(ROOT/'mechanism_diagnose.py'),'--config',str(cfg),'--synthetic-check'],cwd=ROOT)
        run=(set((ROOT/'runs').glob('mechanism_*'))-before).pop();assert r.returncode==0,'Integration failed: '+str(run)
        assert json.loads((run/'status.json').read_text())['status']=='success'
        replay=json.loads((run/'paired_input_noise_replay.json').read_text());assert all(r['status']=='passed' for r in replay)
        for v in ['B2','B6']:
            rows=[json.loads(x) for x in (run/v/'batches.jsonl').read_text().splitlines()]
            assert len(rows)==2 and all(r['status']=='success' and r['state']['unchanged'] for r in rows)
            assert all(r['input']['input']['shape']==[2,3,320,320] for r in rows)
            summary=json.loads((run/v/'summary.json').read_text());assert summary['successful_batches']==2
        assert all(digest(p)==h for p,h in sources.items())
        before=set((ROOT/'runs').glob('mechanism_*'))
        failure=subprocess.run([sys.executable,str(ROOT/'mechanism_diagnose.py'),'--config',str(d/'absent.yaml')],cwd=ROOT)
        bad=(set((ROOT/'runs').glob('mechanism_*'))-before).pop();assert failure.returncode==1 and (bad/'failure.txt').exists()
        from mechanism_diagnose import pack_results
        archive=pack_results()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and any('failure.txt' in n for n in z.namelist())
            assert not any(n.endswith(('.pt','.jpg','.png','.npy','.npz')) or '/labels/' in n or '/images/' in n for n in z.namelist())
        print(json.dumps(dict(status='passed',scope='CPU random checkpoint synthetic train-only engineering, not NEU/GPU',run=str(run),failure_run=str(bad),
            images=4,batches_per_variant=2,original_total_and_gradients='matched',parameters_and_BN_unchanged=True,same_paired_noise=True)))

if __name__=='__main__':
    if '--integration' in sys.argv:
        sys.argv.remove('--integration');result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MechanismTests))
        if not result.wasSuccessful():sys.exit(1)
        integration()
    else:unittest.main()
