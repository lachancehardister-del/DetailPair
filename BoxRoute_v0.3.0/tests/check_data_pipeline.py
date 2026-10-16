"""Byte-copy/cache isolation and actual square tensors on tiny procedural source images."""
import os,sys,json,tempfile,unittest,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
import numpy as np,torch,yaml
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from torch.utils.data import DataLoader
from boxroute.data import prepare_private,verify_sources,choose
from boxroute.identity import file_sha256,dump
from boxroute.trainer import SquareTrainer,SquareValidator


def fixture(root,n_train=4,n_val=2,names=None):
    names=names or ['a','b','c','d','e','f'];root=Path(root);root.mkdir()
    for split,n in [('train',n_train),('val',n_val)]:
        (root/'images'/split).mkdir(parents=True);(root/'labels'/split).mkdir(parents=True)
        for i in range(n):
            a=np.random.default_rng(i+(0 if split=='train' else 100)).integers(0,255,(80,120,3),dtype=np.uint8)
            Image.fromarray(a).save(root/'images'/split/f'{i}.png');(root/'labels'/split/f'{i}.txt').write_text(f'{i%len(names)} .5 .5 .6 .3\n')
    p=root/'data.yaml';p.write_text(yaml.safe_dump(dict(path=str(root.resolve()),names=names,train='images/train',val='images/val',test='DO_NOT_ENUMERATE_TEST')));return p


class DataChecks(unittest.TestCase):
    def test_private_copies_square_train_val_standalone_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);data=fixture(d/'original');source_before={str(p):file_sha256(p) for p in (d/'original').rglob('*') if p.is_file()}
            private,manifest=prepare_private(data,d/'private',expected=(4,2),selections=[d/'missing_selection.json'])
            self.assertFalse(manifest['test_enumerated']);self.assertNotIn('test',yaml.safe_load(private.read_text()))
            self.assertEqual(manifest['historical_selection_checks'][0]['status'],'missing_historical_selection')
            a,_=choose(manifest,'train',2);b,_=choose(manifest,'train',2);self.assertEqual(a,b)
            trainer=SquareTrainer.__new__(SquareTrainer);trainer.args=get_cfg();trainer.args.rect=False;trainer.args.imgsz=320;trainer.args.amp=False
            trainer.args.mosaic=0.;trainer.args.mixup=0.;trainer.args.copy_paste=0.;trainer.args.hsv_h=0.;trainer.args.hsv_s=0.;trainer.args.hsv_v=0.
            trainer.model=DetectionModel('yolo11n.yaml',nc=6,verbose=False);trainer.device=torch.device('cpu');trainer.save_dir=d/'logs';trainer.save_dir.mkdir()
            trainer.data=dict(nc=6,names=dict(enumerate(manifest['names'])),channels=3)
            shapes=[]
            for mode in ['train','val']:
                ds=trainer.build_dataset(str(d/'private/images'/mode),mode,2);self.assertFalse(ds.rect)
                loader=DataLoader(ds,batch_size=2,collate_fn=ds.collate_fn,num_workers=0)
                batch=next(iter(loader));self.assertEqual(list(batch['img'].shape),[2,3,320,320])
                if mode=='train':prepared=trainer.preprocess_batch(batch)
                else:
                    v=SquareValidator.__new__(SquareValidator);v.args=copy.copy(trainer.args);v.device=torch.device('cpu');v.save_dir=d/'standalone';v.save_dir.mkdir()
                    prepared=v.preprocess(batch)
                    v.data=trainer.data;v.stride=32
                    standalone=v.build_dataset(str(d/'private/images/val'),'val',1);self.assertFalse(standalone.rect)
                self.assertEqual(prepared['img'].dtype,torch.float32);shapes.append(list(prepared['img'].shape))
            broken=dict(batch);broken['img']=torch.zeros(1,3,352,352,dtype=torch.uint8)
            with self.assertRaisesRegex(RuntimeError,'320x320'):v.preprocess(broken)
            self.assertEqual(verify_sources(manifest)['status'],'verified_unchanged')
            after={str(p):file_sha256(p) for p in (d/'original').rglob('*') if p.is_file()};self.assertEqual(source_before,after)
            self.assertTrue(list((d/'private').rglob('*.cache')))
            dump(ROOT/'verification/data_pipeline_cpu.json',dict(status='passed',scope='Synthetic source images, not NEU audit',actual_shapes=shapes,test_enumerated=False,original_tree_unchanged=True,label_caches_private=True,shape352_rejected=True))

if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
