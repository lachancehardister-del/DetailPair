"""All actions preserve source data and create timestamped output directories."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,traceback,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
os.environ.setdefault('WANDB_DISABLED','true')

def stamp():return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def environment():
    import importlib.metadata as md
    r=dict(time=datetime.datetime.now().astimezone().isoformat(),python=sys.version,platform=platform.platform(),executable=sys.executable)
    r['packages']={}
    for m in ['torch','torchvision','ultralytics','numpy','pillow','pyyaml']:
        try:r['packages'][m]=md.version(m)
        except md.PackageNotFoundError:r['packages'][m]='NOT_INSTALLED'
    try:
        import torch
        r['cuda_available']=torch.cuda.is_available();r['torch_cuda']=torch.version.cuda
        r['gpus']=[dict(name=torch.cuda.get_device_name(i),memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
    except Exception as e:r['torch_error']=str(e)
    try:r['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=15).stdout
    except Exception as e:r['nvidia_smi_error']=str(e)
    r['pip_freeze']=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout
    return r

def pack():
    from realtime_eval import pack_results
    return pack_results()


def evaluate(weights,data,device,imgsz,out,split='val'):
    import detailpair.model  # register custom checkpoint classes
    from ultralytics import YOLO
    m=YOLO(str(Path(weights).resolve()))
    results=m.val(data=str(data),split=split,imgsz=imgsz,batch=1,device=device,workers=0,plots=True,project=str(out),name='evaluation',exist_ok=True)
    dump(out/'metrics.json',{'split':split,**results.results_dict})
    m.predict(source=str(_first_images(data,split)[0]),imgsz=imgsz,device=device,save=True,save_txt=True,save_conf=True,project=str(out),name='predictions',exist_ok=True)
    return m

def _first_images(data,split):
    from detailpair.data import load_spec
    return load_spec(data)[1][split]

def benchmark(weights,device,imgsz,out):
    import torch,time,numpy as np
    import detailpair.model
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    dev=select_device(device)
    if dev.type=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 1))
    m=YOLO(str(Path(weights).resolve())).model.float().eval().to(dev)
    x=torch.zeros(1,3,imgsz,imgsz,device=dev)
    cuda=x.is_cuda
    times=[]
    with torch.inference_mode():
        for _ in range(20):m(x)
        if cuda:torch.cuda.synchronize()
        for _ in range(100):
            t=time.perf_counter();m(x)
            if cuda:torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
    dump(out/'latency.json',dict(scope='FP32 unfused model forward only; excludes preprocess and NMS; synthetic zeros; not end-to-end production speed',device=str(device),imgsz=imgsz,batch=1,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),samples_ms=times,parameters=sum(p.numel() for p in m.parameters())))

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('env');sub.add_parser('smoke');sub.add_parser('pack');sub.add_parser('collect')
    q=sub.add_parser('pilot');q.add_argument('--data',required=True);q.add_argument('--device',default='cpu');q.add_argument('--config',default='configs/pilot.yaml')
    q.add_argument('--accept-near-duplicates-for-pilot',action='store_true')
    a=sub.add_parser('audit');a.add_argument('--data',required=True)
    t=sub.add_parser('train');t.add_argument('--data',required=True);t.add_argument('--manifest',required=True)
    VARIANTS=['B0','B1','B2','B3','B4','B5','B6','B7','E_UNIFORM','E_SHIFT','E_SIGNED','E_FFT','SPD']
    t.add_argument('--variant',choices=VARIANTS,default='B6')
    t.add_argument('--epochs',type=int,default=3);t.add_argument('--batch',type=int,default=2);t.add_argument('--imgsz',type=int,default=320)
    t.add_argument('--device',default='cpu');t.add_argument('--seed',type=int,default=0);t.add_argument('--weights')
    t.add_argument('--train-limit',type=int,default=64);t.add_argument('--val-limit',type=int,default=32)
    t.add_argument('--accept-near-duplicates-for-pilot',action='store_true')
    s=sub.add_parser('synthetic');s.add_argument('--device',default='cpu');s.add_argument('--variant',choices=VARIANTS,default='B6')
    for name in ['evaluate','benchmark','export','robustness']:
        q=sub.add_parser(name);q.add_argument('--weights',required=True);q.add_argument('--device',default='cpu');q.add_argument('--imgsz',type=int,default=320)
        if name in ('evaluate','robustness'):q.add_argument('--data',required=True);q.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args()
    if args.command=='pack':pack();return
    run=ROOT/'runs'/f'{args.command}_{getattr(args,"variant","")}_{stamp()}'
    run.mkdir(parents=True,exist_ok=False)
    dump(run/'request.json',vars(args));dump(run/'environment.json',environment())
    try:
        if args.command not in ('env','collect'):
            from ultralytics import settings
            settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb')})
        if args.command=='env':print(json.dumps(environment(),ensure_ascii=False,indent=2))
        elif args.command=='audit':
            from detailpair.data import audit
            r=audit(args.data,run/'audit.json');print(f'AUDIT OK: {run / "audit.json"}; warnings={len(r["warnings"])}')
        elif args.command=='smoke':
            from detailpair.engine import smoke
            print(smoke(run/'smoke.json'))
        elif args.command=='pilot':
            import yaml,torch,random
            from detailpair.data import audit,subset,check_manifest
            from detailpair.engine import train
            c=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
            torch.set_num_threads(min(4,os.cpu_count() or 1))
            r=audit(args.data,run/'audit.json')
            if r['warnings'] and not args.accept_near_duplicates_for_pilot:raise ValueError('Review audit dHash warnings before pilot override')
            from detailpair.model import VARIANTS
            if len(c['variants'])!=len(set(c['variants'])):raise ValueError('Duplicate conditions in schedule')
            for v in c['variants']:
                if v not in VARIANTS:raise ValueError('Unknown condition: '+v)
            dump(run/'effective_config.json',c)
            dump(run/'implemented_objectives.json',{v:VARIANTS[v] for v in c['variants']})
            for seed in c['seeds']:
                data=subset(run/'audit.json',run/f'subset_s{seed}',c['train_limit'],c['val_limit'],seed)
                order=list(c['variants']);random.Random(seed+127).shuffle(order)
                dump(run/f'order_s{seed}.json',order)
                for variant in order:
                    check_manifest(run/'audit.json')
                    dest=run/f'{variant}_s{seed}'
                    dest.mkdir()
                    try:
                        research=dict(variant=variant,noise_std=c['noise_std'],brightness=c['brightness_offset'])
                        result=train(research,data,dest/'training',c['epochs'],c['batch'],c['imgsz'],args.device,seed)
                        evaluate(result/'weights'/'best.pt',data,args.device,c['imgsz'],dest)
                        dump(dest/'status.json',dict(status='success',kind='EXPLORATORY_PILOT'))
                    except Exception:
                        (dest/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                        raise
            dump(run/'completed.json',dict(status='completed',kind='SYNTHETIC_PLUMBING_ONLY' if 'synthetic' in str(args.data).lower() else 'EXPLORATORY_PILOT_NOT_FORMAL_EXPERIMENT'))
        elif args.command in ['train','synthetic']:
            import torch
            from detailpair.engine import train
            from detailpair.data import synthetic,audit,subset,check_manifest
            torch.set_num_threads(min(4,os.cpu_count() or 1))
            if args.command=='synthetic':
                data=synthetic(run/'synthetic_data');manifest=run/'audit.json';audit(data,manifest)
                research={'variant':args.variant};epochs,batch,imgsz,seed,weights=1,2,64,0,None
            else:
                r=check_manifest(args.manifest)
                if str(Path(args.data).resolve())!=r['source_yaml']:raise ValueError('--data must match audited source YAML')
                if r['warnings'] and not args.accept_near_duplicates_for_pilot:raise ValueError('Review dHash warnings before training; pilot override is explicit')
                data=subset(args.manifest,run/'subset',args.train_limit,args.val_limit,args.seed)
                research={'variant':args.variant};epochs,batch,imgsz,seed,weights=args.epochs,args.batch,args.imgsz,args.seed,args.weights
            dump(run/'research.json',research)
            result=train(research,data,run/'training',epochs,batch,imgsz,args.device,seed,weights)
            best=result/'weights'/'best.pt'
            if not best.exists():raise FileNotFoundError(best)
            evaluate(best,data,args.device,imgsz,run)
            dump(run/'completed.json',dict(status='completed',kind='SYNTHETIC_PLUMBING_ONLY' if args.command=='synthetic' else 'REAL_DATA_EXPLORATORY_PILOT',weights=str(best),data=str(data)))
        elif args.command=='collect':
            import csv,statistics
            rows=[]
            for f in (ROOT/'runs').glob('pilot_*/*/metrics.json'):
                variant,seed=f.parent.name.rsplit('_s',1)
                rows.append(dict(pilot=f.parent.parent.name,variant=variant,seed=int(seed),**json.loads(f.read_text())))
            dump(run/'collected_metrics.json',rows)
            groups={}
            for r in rows:groups.setdefault((r['pilot'],r['variant']),[]).append(r)
            summary=[]
            key='metrics/mAP50-95(B)'
            for (pilot,v),rs in groups.items():
                values=[r[key] for r in rs]
                summary.append(dict(pilot=pilot,variant=v,n=len(values),AP5095_mean=statistics.mean(values),AP5095_sample_std=statistics.stdev(values) if len(values)>1 else None,seeds=[r['seed'] for r in rs]))
            dump(run/'summary.json',summary)
        elif args.command=='evaluate':evaluate(args.weights,args.data,args.device,args.imgsz,run,args.split)
        elif args.command=='robustness':
            import yaml,numpy as np,shutil
            from PIL import Image
            from detailpair.data import load_spec,label_for
            names,splits=load_spec(args.data)
            # Default val only. Final test evaluation is an explicit command after model freeze.
            reports=[]
            for kind,severity in [('gaussian',0.01),('gaussian',0.03),('gaussian',0.06),('impulse',0.01)]:
                root=run/f'{kind}_{severity}';imdir=root/'images'/args.split;labdir=root/'labels'/args.split
                imdir.mkdir(parents=True);labdir.mkdir(parents=True)
                rng=np.random.default_rng(178);snr=[]
                for i,src in enumerate(splits[args.split]):
                    a=np.asarray(Image.open(src).convert('RGB'),dtype=np.float32)/255
                    if kind=='gaussian':b=np.clip(a+rng.normal(0,severity,a.shape),0,1)
                    else:
                        b=a.copy();mask=rng.random(a.shape[:2]);b[mask<severity/2]=0;b[mask>1-severity/2]=1
                    clipped=np.asarray(np.round(b*255),dtype=np.uint8)
                    snr.append(float(10*np.log10((np.mean(a*a)+1e-12)/(np.mean((clipped/255-a)**2)+1e-12))))
                    Image.fromarray(clipped).save(imdir/f'{i:06d}.png')
                    shutil.copyfile(label_for(src),labdir/f'{i:06d}.txt')
                y=root/'data.yaml';y.write_text(yaml.safe_dump(dict(path=str(root.resolve()),train=f'images/{args.split}',val=f'images/{args.split}',test=f'images/{args.split}',names=names),allow_unicode=True))
                dump(root/'corruption.json',dict(kind=kind,severity=severity,seed=178,actual_snr_db=snr,scope='Synthetic perturbations on selected split; not real acquisition noise'))
                evaluate(args.weights,y,args.device,args.imgsz,root,args.split)
                reports.append(dict(kind=kind,severity=severity,**json.loads((root/'metrics.json').read_text())))
            dump(run/'robustness.json',reports)
        elif args.command=='benchmark':benchmark(args.weights,args.device,args.imgsz,run)
        elif args.command=='export':
            import detailpair.model
            from ultralytics import YOLO
            # ONNX dependency installed separately; not represented as verified if not run.
            y=YOLO(str(Path(args.weights).resolve()))
            file=y.export(format='onnx',imgsz=args.imgsz,device=args.device,simplify=False,opset=17)
            dump(run/'export.json',dict(path=str(file),runtime_validation=False))
        dump(run/'status.json',dict(status='success',finished=datetime.datetime.now().astimezone().isoformat()))
    except Exception:
        (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        dump(run/'status.json',dict(status='failed'))
        traceback.print_exc();sys.exit(1)

if __name__=='__main__':main()
