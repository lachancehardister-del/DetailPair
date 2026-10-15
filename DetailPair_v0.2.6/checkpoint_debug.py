"""Read-only checkpoint inspection and bounded CPU/CUDA backward checks.

Only load this project's own local checkpoints. No optimizer step or checkpoint write.
This is NOT resume and NOT a replay of the original augmented failing batch.
"""
import argparse, copy, json, math, os, random, sys, traceback
from pathlib import Path
from run import ROOT, dump, environment, stamp
from detailpair.data import digest

def select_rows(pilot, images=32):
    if images < 2 or images % 2: raise ValueError('images must be even and >=2')
    selection=json.loads((pilot/'subset_s0/selection.json').read_text(encoding='utf-8'))
    rows=list(selection['train'])
    random.Random(20261011).shuffle(rows)
    return rows[:images]

def labels(path, nc):
    result=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        values=[float(v) for v in line.split()]
        if len(values)!=5 or not all(math.isfinite(v) for v in values): raise ValueError('Invalid label: '+str(path))
        c,x,y,w,h=values
        if c!=int(c) or not 0<=c<nc or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1): raise ValueError('Class/coordinate outside range: '+str(path))
        if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6: raise ValueError('Box outside image: '+str(path))
        result.append(values)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['inspect','replay'])
    p.add_argument('--pilot',required=True)
    p.add_argument('--device',choices=['cpu','0'],default='cpu')
    p.add_argument('--images',type=int,default=32)
    args=p.parse_args()
    out=ROOT/'runs'/('checkpoint_'+args.mode+'_'+args.device+'_'+stamp())
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'request.json',vars(args))
    dump(out/'environment.json',environment())
    phase={'stage':'import'}
    try:
        import torch, numpy as np
        from PIL import Image
        import detailpair.model
        from detailpair.model import PairDetectionModel
        from ultralytics.cfg import DEFAULT_CFG
        pilot=Path(args.pilot).resolve()
        if not pilot.is_dir(): raise FileNotFoundError(pilot)
        weights=pilot/'B5_s0/training/weights'
        listing=[dict(name=f.name,size_bytes=f.stat().st_size) for f in sorted(weights.glob('*.pt'))]
        dump(out/'checkpoint_listing.json',listing)
        if not listing: raise FileNotFoundError('No B5 local checkpoint files; old weights must remain in the old run directory')
        if args.mode=='inspect':
            reports=[]
            for name in ['last.pt','best.pt','epoch37.pt']:
                path=weights/name
                if not path.exists():
                    reports.append(dict(name=name,status='missing'));continue
                phase={'stage':'checkpoint_inspect','path':str(path)};dump(out/'last_stage.json',phase)
                before=digest(path)
                ckpt=torch.load(path,map_location='cpu',weights_only=False)
                model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
                finite=model is not None and all(not t.is_floating_point() or torch.isfinite(t).all().item() for t in model.state_dict().values())
                reports.append(dict(name=name,sha256=before,epoch_zero_based=ckpt.get('epoch'),optimizer_present=ckpt.get('optimizer') is not None,ema_present=ckpt.get('ema') is not None,model_type=type(model).__name__,pair_config=getattr(model,'pair_config',None),floating_state_finite=bool(finite),scope='Metadata only; optimizer presence does not prove exact resumability'))
                assert before==digest(path),'Checkpoint unexpectedly changed'
                dump(out/'checkpoint_inspection.json',reports)
                del ckpt,model
        else:
            device=torch.device('cuda:0' if args.device=='0' else 'cpu')
            if device.type=='cuda':
                if os.environ.get('CUDA_LAUNCH_BLOCKING')!='1': raise RuntimeError('Set CUDA_LAUNCH_BLOCKING=1 before starting this fresh Python process')
                if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True,warn_only=True)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            path=weights/'last.pt'
            if not path.exists(): raise FileNotFoundError(path)
            before=digest(path)
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
            if not isinstance(model,PairDetectionModel): raise TypeError('Expected this project PairDetectionModel')
            if model.pair_config.get('module')!='gate' or model.pair_config.get('energy')!=0: raise ValueError('Checkpoint is not B5')
            model=model.float().to(device)
            model.args=copy.deepcopy(DEFAULT_CFG)
            for key in ['box','cls','dfl']:
                if key in ckpt.get('train_args',{}):setattr(model.args,key,ckpt['train_args'][key])
            model.criterion=None
            model.perturbation_seed=100003
            model.perturbation_step=0
            model.train()
            for name,param in model.named_parameters():param.requires_grad_('.dfl.' not in name)
            nc=model.model[-1].nc
            rows=select_rows(pilot,args.images)
            if len(rows)!=args.images:raise ValueError('Not enough train images')
            for row in rows:
                for kind in ['image','label']:
                    if digest(row[kind])!=row[kind+'_sha256']:raise ValueError('Original source changed: '+row[kind])
            dump(out/'selected_train.json',rows)
            dump(out/'scope.json',dict(checkpoint=str(path),sha256=before,model_source='checkpoint EMA if present, else model; not original live optimizer weights',input='32 fixed train images by default, RGB resize 320 square, no geometric augmentation',optimizer_updates=0,checkpoint_writes=0,perturbation_step_reset=True,failed_batch_reproduced=False,limits='CPU/GPU random draws need not be identical; no metric comparison or exact resume claim'))
            results=[]
            for index in range(0,len(rows),2):
                pair=rows[index:index+2]
                phase=dict(batch=index//2,images=[r['image'] for r in pair],stage='decode_cpu');dump(out/'last_stage.json',phase)
                arrays=[];idx=[];cls=[];boxes=[]
                for j,row in enumerate(pair):
                    with Image.open(row['image']) as im:arrays.append(np.asarray(im.convert('RGB').resize((320,320)),dtype=np.float32).transpose(2,0,1)/255)
                    for label in labels(row['label'],nc):idx.append(float(j));cls.append([label[0]]);boxes.append(label[1:])
                phase['stage']='transfer';dump(out/'last_stage.json',phase)
                batch=dict(img=torch.tensor(np.stack(arrays),device=device),batch_idx=torch.tensor(idx,device=device),cls=torch.tensor(cls,device=device).reshape(-1,1),bboxes=torch.tensor(boxes,device=device).reshape(-1,4))
                if device.type=='cuda':torch.cuda.synchronize()
                model.zero_grad(set_to_none=True)
                phase['stage']='paired_forward_and_loss';dump(out/'last_stage.json',phase)
                loss,_=model(batch);loss=loss.sum()
                if device.type=='cuda':torch.cuda.synchronize()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                phase['stage']='backward';dump(out/'last_stage.json',phase)
                loss.backward()
                if device.type=='cuda':torch.cuda.synchronize()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Non-finite gradients')
                results.append(dict(batch=index//2,loss=float(loss.detach()),aux=model.latest_aux,device=str(device),forward_backward=True))
                dump(out/'batches.json',results)
                print('PASS',args.device,index//2,flush=True)
            assert before==digest(path),'Checkpoint unexpectedly changed'
        dump(out/'status.json',dict(status='success',mode=args.mode,scope='Checkpoint diagnostics only; not training recovery or efficacy evidence'))
        return 0
    except Exception:
        dump(out/'last_stage.json',phase)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        dump(out/'status.json',dict(status='failed'))
        traceback.print_exc()
        return 1

if __name__=='__main__':sys.exit(main())
