"""Narrow read-only validator for the original 40-epoch B5/B6 pilot only."""
import csv, hashlib, inspect, json, math, re
from pathlib import Path
import torch
from ultralytics.engine.trainer import BaseTrainer
from .data import digest
from .model import PairDetectionModel, DetailDown, VARIANTS
from .reporting import dump

RUN='pilot__20261011_112235_644745'
EXPECTED=dict(seed=0,imgsz=320,batch=2,epochs=40,patience=50,amp=False,pretrained=False,resume=False,
    optimizer='AdamW',lr0=.001,lrf=.01,deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
    degrees=0.,translate=.1,scale=.1,fliplr=.5,flipud=0.,close_mosaic=0,save_period=1)


def epoch_semantics(out):
    save=inspect.getsource(BaseTrainer.save_model);metrics=inspect.getsource(BaseTrainer.save_metrics)
    (Path(out)/'installed_save_model.py.txt').write_text(save,encoding='utf-8')
    (Path(out)/'installed_save_metrics.py.txt').write_text(metrics,encoding='utf-8')
    if not re.search(r'[\'"]epoch[\'"]\s*:\s*self\.epoch\s*,',save) or 'self.epoch + 1' not in metrics:
        raise RuntimeError('Installed epoch semantics unconfirmed; refuse filename inference')
    return dict(checkpoint_zero_based=True,csv_epoch='checkpoint epoch + 1',filename_is_not_evidence=True,
        save_source_sha256=hashlib.sha256(save.encode()).hexdigest(),csv_source_sha256=hashlib.sha256(metrics.encode()).hexdigest())


def completed_csv(path):
    path=Path(path);rows={};issues=[]
    if not path.is_file():return rows,dict(path=str(path),status='missing',issues=[])
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        for line,row in enumerate(reader,2):
            try:
                normalized={k.strip():v for k,v in row.items() if k is not None}
                if None in row or not normalized or any(v is None or not v.strip() for v in normalized.values()):raise ValueError('incomplete row')
                numeric={k:float(v) for k,v in normalized.items()}
                if not all(math.isfinite(v) for v in numeric.values()):raise ValueError('nonfinite CSV')
                ep=numeric['epoch']
                if ep!=int(ep) or ep!=len(rows)+1:raise ValueError('duplicate, missing or unordered completed epoch')
                if not {'metrics/mAP50(B)','metrics/mAP50-95(B)'}<=numeric.keys():raise ValueError('missing AP columns')
                rows[int(ep)]=numeric
            except Exception as e:
                issues.append(dict(line=line,error=str(e)));break  # only a contiguous, fully written prefix is evidence
    return rows,dict(path=str(path.resolve()),sha256=digest(path),complete_epochs=list(rows),issues=issues,status='partial' if issues else 'read')


def model_identity(model,variant,names=None):
    if not isinstance(model,PairDetectionModel) or model.variant!=variant:raise ValueError('wrong custom model/variant')
    if model.pair_config!=VARIANTS[variant]:raise ValueError('wrong pair_config')
    if model.research_layer!=1:raise ValueError('wrong research layer')
    layer=model.model[1]
    if not isinstance(layer,DetailDown) or not layer.gated or layer.floor!=.1:raise ValueError('both B5/B6 require gated DetailDown floor .1')
    for key,value in dict(noise_std=.03,brightness=.05,perturbation_seed=100003).items():
        if getattr(model,key,None)!=value:raise ValueError('wrong perturbation '+key)
    mapping=model.names
    mapping=[mapping[k] for k in sorted(mapping)] if isinstance(mapping,dict) else list(mapping)
    if names is not None and mapping!=list(names):raise ValueError('wrong checkpoint class names')
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite model parameters/buffers')
    return dict(type=type(model).__module__+'.'+type(model).__name__,variant=variant,pair_config=model.pair_config,
        names=mapping,parameters=sum(p.numel() for p in model.parameters()),
        parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},parameters_and_buffers_finite=True)


def inspect_checkpoint(path,variant,csv_rows,names=None,report=None):
    path=Path(path);before=digest(path)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)  # configured, trusted local old files only
    if not isinstance(ckpt,dict):raise ValueError('checkpoint must be a metadata dictionary')
    if report is not None:
        report.update(checkpoint_epoch=ckpt.get('epoch'),checkpoint_version=ckpt.get('version'),train_args=ckpt.get('train_args',{}),model_and_ema={})
        for key in ['model','ema']:
            m=ckpt.get(key)
            report['model_and_ema'][key]=None if m is None else dict(type=type(m).__module__+'.'+type(m).__name__,variant=getattr(m,'variant',None),pair_config=getattr(m,'pair_config',None),parameters_and_buffers_finite=bool(all(torch.isfinite(v).all() for v in m.state_dict().values())) if isinstance(m,torch.nn.Module) else None)
    if ckpt.get('version')!='8.3.221':raise ValueError('historical checkpoint version must confirm installed 8.3.221 semantics')
    args=ckpt.get('train_args',{})
    for k,v in EXPECTED.items():
        if args.get(k)!=v:raise ValueError(f'old40 train_args {k} expected {v!r}, got {args.get(k)!r}')
    saved=str(args.get('save_dir') or str(args.get('project',''))+'/'+str(args.get('name',''))).replace('\\','/').rstrip('/')
    parts=saved.split('/')
    if RUN not in parts or variant+'_s0' not in parts or parts[-1]!='training':raise ValueError('wrong saved original run/condition')
    identities={}
    for key in ['model','ema']:
        if ckpt.get(key) is not None:identities[key]=model_identity(ckpt[key],variant,names)
        else:identities[key]=None
    role='ema' if ckpt.get('ema') is not None else 'model'
    if identities[role] is None:raise ValueError('no model or EMA')
    epoch=ckpt.get('epoch')
    if type(epoch) is not int or epoch<0:raise ValueError('saved epoch unconfirmed (including stripped epoch=-1); never infer from filename/best AP')
    completed=epoch+1
    if completed>38 or completed not in csv_rows:raise ValueError('epoch outside fully completed CSV prefix <=38')
    # Periodic files are metadata-led; a conflicting filename is evidence of ambiguity.
    match=re.fullmatch(r'epoch(\d+)\.pt',path.name)
    if match and int(match[1])!=epoch:raise ValueError('filename contradicts saved epoch')
    if digest(path)!=before:raise RuntimeError('checkpoint changed while inspecting')
    identity=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=before,status='eligible',variant=variant,
        selected_role=role,model_and_ema=identities,train_args=args,saved_original_run=saved,
        checkpoint_epoch=epoch,completed_epoch=completed,original_budget=40,csv_row=csv_rows[completed],
        checkpoint_version=ckpt.get('version'),optimizer_present=ckpt.get('optimizer') is not None,
        provenance_limits=['historical initialization tensor hash unavailable; equal seed does not prove identical trajectory'])
    del ckpt
    return identity


def inventory(entries,out):
    inventories={};csvs={};directories={}
    for entry in entries:
        variant=entry['variant'];rows,csv_info=completed_csv(entry['csv']);csvs[variant]=csv_info;found=[];dirs=[]
        for directory in entry['weight_dirs']:
            root=Path(directory);dirs.append(dict(path=str(root),exists=root.is_dir()))
            if root.is_dir():found.extend(p for p in root.iterdir() if p.is_file() and (re.fullmatch(r'epoch\d+\.pt',p.name) or p.name in {'last.pt','best.pt'}))
        result=[]
        for path in sorted(set(p.resolve() for p in found),key=str):
            item=dict(path=str(path))
            try:
                item.update(bytes=path.stat().st_size,sha256=digest(path))
                item=inspect_checkpoint(path,variant,rows,report=item)
            except Exception as e:item.update(status='rejected_or_corrupt',error=type(e).__name__+': '+str(e))
            result.append(item);dump(Path(out)/(variant+'_inventory.json'),dict(directories=dirs,csv=csv_info,checkpoints=result))
        inventories[variant]=result;directories[variant]=dirs
        dump(Path(out)/(variant+'_inventory.json'),dict(directories=dirs,csv=csv_info,checkpoints=result))
    return inventories,csvs,directories


def select_pair(inventories):
    eligible={v:[r for r in inventories[v] if r['status']=='eligible'] for v in ['B5','B6']}
    common=sorted({r['completed_epoch'] for r in eligible['B5']}&{r['completed_epoch'] for r in eligible['B6']})
    if not common:return dict(status='blocked_missing_matched_checkpoint',reason='No metadata/CSV-confirmed common completed epoch <=38. No training authorized.',selected={})
    epoch=common[-1]
    def rank(r):
        name=Path(r['path']).name
        return (0 if name.startswith('epoch') else 1 if name=='last.pt' else 2,r['path'])
    selected={v:sorted([r for r in eligible[v] if r['completed_epoch']==epoch],key=rank)[0] for v in eligible}
    a,b=selected['B5'],selected['B6']
    if a['selected_role']!=b['selected_role']:raise ValueError('EMA/model selection role differs; refuse heterogeneous pair')
    excluded={'project','name','save_dir'}
    aa={k:v for k,v in a['train_args'].items() if k not in excluded};bb={k:v for k,v in b['train_args'].items() if k not in excluded}
    if aa!=bb:raise ValueError('training/augmentation/optimizer arguments differ: '+str([k for k in aa.keys()|bb.keys() if aa.get(k)!=bb.get(k)]))
    ia=a['model_and_ema'][a['selected_role']];ib=b['model_and_ema'][b['selected_role']]
    if any(ia[k]!=ib[k] for k in ['names','parameters','parameter_shapes']):raise ValueError('architecture/parameter count/names differ')
    return dict(status='matched',completed_epoch=epoch,checkpoint_epoch=epoch-1,original_budget=40,common_completed_epochs=common,
        selection_rule='latest common metadata+CSV-confirmed epoch <=38; never AP-selected; periodic preferred then last then best at same confirmed epoch',
        fallback_reason=None if epoch==38 else '38 not jointly verifiable; latest earlier common epoch chosen without AP',
        train_args_equal_except_output_paths=True,selected=selected,
        provenance_limits=a['provenance_limits'])


def load_selected(identity,names):
    from ultralytics import YOLO
    rows={identity['completed_epoch']:identity['csv_row']}
    current=inspect_checkpoint(identity['path'],identity['variant'],rows,names)
    if current['sha256']!=identity['sha256'] or current['completed_epoch']!=identity['completed_epoch']:raise ValueError('selected checkpoint changed')
    yolo=YOLO(identity['path']);model_identity(yolo.model,identity['variant'],names)
    return yolo
