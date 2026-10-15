"""Old40 identity refusals and a complete four-cell CPU random-checkpoint fixture."""
import copy,json,os,subprocess,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
import torch,yaml
from ultralytics.cfg import DEFAULT_CFG_DICT
from detailpair.model import PairDetectionModel
from detailpair.data import synthetic,digest
from detailpair.matched_checkpoints import EXPECTED,RUN,inspect_checkpoint,completed_csv,select_pair,epoch_semantics,inventory
from matched_checkpoint_eval import validate_config

torch.set_num_threads(2)

def make_checkpoint(path,variant,epoch=37,**overrides):
    torch.manual_seed(9)
    m=PairDetectionModel(nc=1,variant=variant);m.names={0:'synthetic_rectangle'};m.args=copy.deepcopy(DEFAULT_CFG_DICT)
    args=dict(DEFAULT_CFG_DICT,**EXPECTED);args.update(project='D:/old/'+RUN+'/'+variant+'_s0',name='training',save_dir='D:/old/'+RUN+'/'+variant+'_s0/training')
    args.update(overrides)
    torch.save(dict(model=None,ema=m.half(),train_args=args,epoch=epoch,version='8.3.221',optimizer=None),path)

def make_csv(path,count=40):
    path.write_text('epoch,time,metrics/mAP50(B),metrics/mAP50-95(B)\n'+''.join(f'{i},1,.3,.2\n' for i in range(1,count+1)))

class MatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.csv=cls.root/'results.csv';make_csv(cls.csv)
        cls.rows=completed_csv(cls.csv)[0]
        for v in ['B5','B6']:make_checkpoint(cls.root/(v+'.pt'),v)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def mutate(self,variant='B5',epoch=37,**args):
        p=self.root/'altered.pt';d=torch.load(self.root/(variant+'.pt'),weights_only=False);d['epoch']=epoch;d['train_args'].update(args);torch.save(d,p);return p
    def test_real_epoch_semantics_and_no_filename_inference(self):
        self.assertTrue(epoch_semantics(self.root)['checkpoint_zero_based'])
        row=inspect_checkpoint(self.root/'B5.pt','B5',self.rows)
        self.assertEqual(row['completed_epoch'],38)
        p=self.root/'epoch37.pt';d=torch.load(self.root/'B5.pt',weights_only=False);d['epoch']=-1;torch.save(d,p)
        with self.assertRaisesRegex(ValueError,'unconfirmed'):inspect_checkpoint(p,'B5',self.rows)
        d['epoch']=36;torch.save(d,p)
        with self.assertRaisesRegex(ValueError,'contradicts'):inspect_checkpoint(p,'B5',self.rows)
    def test_wrong_variant_and_names_refused(self):
        p=self.mutate(save_dir='D:/old/'+RUN+'/B6_s0/training')
        with self.assertRaisesRegex(ValueError,'variant'):inspect_checkpoint(p,'B6',self.rows)
        with self.assertRaisesRegex(ValueError,'names'):inspect_checkpoint(self.root/'B5.pt','B5',self.rows,['wrong'])
    def test_wrong_run_and_budget_refused(self):
        p=self.mutate(save_dir='D:/wrong/B5_s0/training')
        with self.assertRaisesRegex(ValueError,'run'):inspect_checkpoint(p,'B5',self.rows)
        p=self.mutate(epochs=200)
        with self.assertRaisesRegex(ValueError,'epochs'):inspect_checkpoint(p,'B5',self.rows)
    def test_different_epochs_block_and_latest_not_AP(self):
        a=inspect_checkpoint(self.root/'B5.pt','B5',self.rows);b=inspect_checkpoint(self.root/'B6.pt','B6',self.rows)
        earlier=copy.deepcopy(b);earlier['completed_epoch']=37
        self.assertEqual(select_pair(dict(B5=[a],B6=[earlier]))['status'],'blocked_missing_matched_checkpoint')
        pair=select_pair(dict(B5=[a],B6=[earlier,b]));self.assertEqual(pair['completed_epoch'],38)
        self.assertEqual(pair['selected']['B5']['selected_role'],'ema')
    def test_training_policy_mismatch_and_role_mismatch(self):
        a=inspect_checkpoint(self.root/'B5.pt','B5',self.rows);b=inspect_checkpoint(self.root/'B6.pt','B6',self.rows)
        b['train_args']['weight_decay']=.9
        with self.assertRaisesRegex(ValueError,'arguments'):select_pair(dict(B5=[a],B6=[b]))
        b['selected_role']='model'
        with self.assertRaisesRegex(ValueError,'role'):select_pair(dict(B5=[a],B6=[b]))
    def test_incomplete_csv_and_missing_round_rejected(self):
        p=self.root/'partial.csv';make_csv(p,37);p.write_text(p.read_text()+'38,1,.3,\n')
        rows,meta=completed_csv(p);self.assertEqual(max(rows),37);self.assertTrue(meta['issues'])
        with self.assertRaisesRegex(ValueError,'CSV'):inspect_checkpoint(self.root/'B5.pt','B5',rows)
        p=self.mutate(epoch=38)
        with self.assertRaisesRegex(ValueError,'38'):inspect_checkpoint(p,'B5',self.rows)
    def test_corrupt_inventory_recorded_and_no_recursive_search(self):
        folder=self.root/'inventory';folder.mkdir(exist_ok=True);(folder/'epoch37.pt').write_bytes(b'corrupt')
        nested=folder/'not_scanned';nested.mkdir(exist_ok=True);(nested/'epoch36.pt').write_bytes(b'also corrupt')
        entries=[dict(variant=v,weight_dirs=[str(folder)],csv=str(self.csv)) for v in ['B5','B6']]
        data,_,_=inventory(entries,self.root)
        self.assertEqual(len(data['B5']),1);self.assertEqual(data['B5'][0]['status'],'rejected_or_corrupt')
        self.assertEqual(select_pair(data)['status'],'blocked_missing_matched_checkpoint')
    def test_protocol_frozen(self):
        c=yaml.safe_load((ROOT/'configs/matched_checkpoint_eval.yaml').read_text());validate_config(c)
        for k,v in [('device','cpu'),('corruptions',[]),('split','test'),('imgsz',352)]:
            bad=copy.deepcopy(c);bad[k]=v
            with self.assertRaises(ValueError):validate_config(bad)
    def test_nonfinite_and_wrong_objective_refused(self):
        p=self.mutate();d=torch.load(p,weights_only=False);d['ema'].pair_config['energy']=.02;torch.save(d,p)
        with self.assertRaisesRegex(ValueError,'pair_config'):inspect_checkpoint(p,'B5',self.rows)
        p=self.mutate();d=torch.load(p,weights_only=False)
        with torch.no_grad():next(d['ema'].parameters()).flatten()[0]=float('nan')
        torch.save(d,p)
        with self.assertRaisesRegex(ValueError,'nonfinite'):inspect_checkpoint(p,'B5',self.rows)
    def test_earlier_common_epoch_fallback(self):
        a=inspect_checkpoint(self.mutate(epoch=36),'B5',self.rows);b=inspect_checkpoint(self.root/'B6.pt','B6',self.rows)
        b['completed_epoch']=37;b['checkpoint_epoch']=36
        pair=select_pair(dict(B5=[a],B6=[b]));self.assertEqual(pair['completed_epoch'],37);self.assertIsNotNone(pair['fallback_reason'])
    def test_hand_check_prediction_matching(self):
        from detailpair.reporting import match_image,summarize_counts
        gt=[dict(class_id=0,xyxy=[0,0,10,10])]
        predictions=[dict(class_id=0,xyxy=[0,0,10,10],confidence=.9),dict(class_id=0,xyxy=[0,0,10,10],confidence=.8)]
        counts,matches=match_image(predictions,gt,['a']);s=summarize_counts([counts],['a'])
        self.assertEqual([m['status'] for m in matches],['TP','FP']);self.assertEqual(s['classes'][0]['Precision'],.5)

def integration():
    from matched_checkpoint_eval import pack_results
    with tempfile.TemporaryDirectory(prefix='DetailPair old40 CPU ') as temp:
        temp=Path(temp);data=synthetic(temp/'synthetic_data');selection={}
        for split in ['train','val']:
            selection[split]=[dict(image=str(p),label=str(temp/'synthetic_data/labels'/split/(p.stem+'.txt')),image_sha256=digest(p),label_sha256=digest(temp/'synthetic_data/labels'/split/(p.stem+'.txt'))) for p in sorted((temp/'synthetic_data/images'/split).glob('*.png'))]
        sel=temp/'selection.json';sel.write_text(json.dumps(selection))
        c=yaml.safe_load((ROOT/'configs/matched_checkpoint_eval.yaml').read_text());c.update(synthetic_check=True,device='cpu',data=str(data),expected_images=4,expected_gt=4,expected_train_images=8)
        for e in c['variants']:
            folder=temp/e['variant'];folder.mkdir();make_checkpoint(folder/'epoch37.pt',e['variant']);make_csv(folder/'results.csv')
            e.update(weight_dirs=[str(folder)],csv=str(folder/'results.csv'),selection=str(sel))
        cfg=temp/'config.yaml';cfg.write_text(yaml.safe_dump(c));before_files={str(p):digest(p) for p in temp.rglob('*') if p.is_file()}
        def execute(args,expected):
            before=set((ROOT/'runs').glob('matched_*'))
            result=subprocess.run([sys.executable,str(ROOT/'matched_checkpoint_eval.py'),*args],cwd=ROOT)
            run=(set((ROOT/'runs').glob('matched_*'))-before).pop();assert result.returncode==expected,str(run)
            return run
        run=execute(['--config',str(cfg),'--synthetic-check'],0)
        status=json.loads((run/'status.json').read_text());assert status['successful_evaluations']==4 and status['weights_and_sources_verified_unchanged']
        for scenario in ['clean','gaussian_0.03']:
            assert json.loads((run/(scenario+'_paired_inputs.json')).read_text())['equal']
            assert json.loads((run/(scenario+'_paired_fixed_inputs.json')).read_text())['equal']
            for v in ['B5','B6']:
                dest=run/v/scenario
                assert len((dest/'fixed_predictions_matches.jsonl').read_text().splitlines())==4
                assert all(r['actual_shape']==[1,3,320,320] for r in json.loads((dest/'official_actual_inputs.json').read_text()))
        assert all(digest(p)==h for p,h in before_files.items())
        missing=execute(['--config',str(temp/'absent.yaml')],1)
        blocked=execute([],2);assert json.loads((blocked/'status.json').read_text())['status']=='blocked_missing_matched_checkpoint'
        archive=pack_results()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert any('failure.txt' in n for n in z.namelist()) and any('pairing.json' in n for n in z.namelist())
            assert not any(n.endswith(('.pt','.png','.jpg','.npy','.npz')) or '/images/' in n or '/labels/' in n or '/datasets/' in n for n in z.namelist())
        print(json.dumps(dict(status='passed',scope='CPU synthetic random checkpoints, not old trained GPU weights',evaluations=4,run=str(run),blocked_run=str(blocked),failure_run=str(missing),sources_and_weights_unchanged=True)))

if __name__=='__main__':
    integration_requested='--integration' in sys.argv
    if integration_requested:sys.argv.remove('--integration')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MatchedTests))
    if not result.wasSuccessful():sys.exit(1)
    if integration_requested:integration()
