"""Read-only checkpoint/data identity, shared corruptions and official AP export."""
import copy, hashlib, json, math, shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from .data import digest, label_for, load_spec
from .model import PairDetectionModel, DetailDown, VARIANTS
from .reporting import dump, match_image, summarize_counts, write_csv


def read_records(data, split):
    names, splits = load_spec(data)
    if split not in splits: raise ValueError('Missing requested split: '+split)
    rows = []
    for i, image in enumerate(splits[split]):
        label = label_for(image)
        if not image.is_file() or not label.is_file(): raise FileNotFoundError(str(image)+' or '+str(label))
        with Image.open(image) as im: width, height = im.size; im.verify()
        gt, normalized = [], []
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            values = list(map(float,line.split()))
            if len(values)!=5 or not all(math.isfinite(v) for v in values): raise ValueError('Invalid label: '+str(label))
            k,x,y,w,h = values
            if k!=int(k) or not 0<=k<len(names) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-5 or max(x+w/2,y+h/2)>1+1e-5:
                raise ValueError('Invalid class/box: '+str(label))
            if values in normalized: raise ValueError('Duplicate GT label: '+str(label))
            normalized.append(values)
            gt.append(dict(class_id=int(k),xyxy=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
        rows.append(dict(image_id=f'{i:06d}', source_image=str(image),source_label=str(label),
                         image_sha256=digest(image),label_sha256=digest(label),width=width,height=height,gt=gt))
    return names, rows


def verify_selection(entry, data):
    path = Path(entry['selection'])
    if not path.is_file(): raise FileNotFoundError('Original selection manifest required: '+str(path))
    selection = json.loads(path.read_text(encoding='utf-8'))
    names, splits = load_spec(data)
    identities = {}
    for split in ('train','val'):
        records = selection.get(split)
        if not isinstance(records,list): raise ValueError('Selection missing '+split)
        actual = {str(p.resolve()):p for p in splits[split]}
        if entry.get('expected_train_images') is not None and split=='train' and len(records)!=entry['expected_train_images']:
            raise ValueError('Unexpected training split size')
        if len(records)!=len(actual): raise ValueError('Selection count differs from source '+split)
        found = set(); values=[]
        for row in records:
            p=Path(row['image']).resolve(); key=str(p)
            if key not in actual or key in found: raise ValueError('Selection identity mismatch: '+key)
            found.add(key); lab=label_for(p)
            if digest(p)!=row['image_sha256'] or digest(lab)!=row['label_sha256']:
                raise ValueError('Source data changed since training: '+key)
            if row.get('label') and Path(row['label']).resolve()!=lab.resolve(): raise ValueError('Selection label mismatch: '+key)
            values.append(dict(image=key,image_sha256=row['image_sha256'],label_sha256=row['label_sha256']))
        identities[split]=sorted(values,key=lambda x:x['image'])
    return dict(selection_path=str(path.resolve()),selection_sha256=digest(path),source_identity=identities)


def validate_checkpoint(entry, config, names):
    path=Path(entry['weights'])
    if not path.is_file(): raise FileNotFoundError('Required local trained checkpoint missing: '+str(path))
    before=digest(path)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
    if not isinstance(model,PairDetectionModel) or model.variant!=entry['variant']:
        raise ValueError('Checkpoint is not the required custom '+entry['variant']+' model')
    if model.pair_config!=VARIANTS[entry['variant']]: raise ValueError('Checkpoint training objective identity differs')
    if model.research_layer!=1:raise ValueError('Checkpoint research layer must remain index1')
    layer=model.model[model.research_layer]
    if (isinstance(layer,DetailDown) and layer.gated)!=(entry['variant']=='B6'):
        raise ValueError('Checkpoint architecture/variant mismatch')
    args=ckpt.get('train_args',{})
    for key,expected in {'seed':entry['seed'],'imgsz':320,'batch':2,'epochs':200,'patience':50,'amp':False,'pretrained':False,'resume':False,'optimizer':'AdamW','lr0':.001,'lrf':.01}.items():
        if not config.get('synthetic_check') and args.get(key)!=expected:
            raise ValueError(f'Checkpoint train_args {key}: expected {expected!r}, found {args.get(key)!r}')
    if args.get('seed')!=entry['seed']: raise ValueError('Checkpoint seed mismatch')
    # Verify recorded run and condition, independent of relocation of the weights file.
    saved=str(args.get('save_dir') or str(args.get('project',''))+'/'+str(args.get('name',''))).replace('\\','/').rstrip('/')
    if entry['run_name'] not in saved.split('/') or entry['variant']+'_s'+str(entry['seed']) not in saved.split('/'):
        raise ValueError('Checkpoint saved run/variant identity does not match explicit binding: '+saved)
    checkpoint_names=model.names
    if isinstance(checkpoint_names,dict): checkpoint_names=[checkpoint_names[k] for k in sorted(checkpoint_names)]
    if list(checkpoint_names)!=list(names): raise ValueError('Checkpoint class mapping differs from data.yaml')
    yolo=YOLO(str(path.resolve()))
    if not isinstance(yolo.model,PairDetectionModel) or yolo.model.variant!=entry['variant']: raise ValueError('Reload lost custom model identity')
    if digest(path)!=before: raise RuntimeError('Checkpoint changed while loading')
    return yolo, dict(variant=entry['variant'],seed=entry['seed'],run_name=entry['run_name'],weights=str(path.resolve()),
        weights_sha256=before,checkpoint_epoch=ckpt.get('epoch'),optimizer_present=ckpt.get('optimizer') is not None,
        train_args=args,read_only=True,loaded_parameters=sum(p.numel() for p in yolo.model.parameters()))


def prepare_dataset(rows, names, root, split, corruption=None, seed=178):
    """One shared lossless derived dataset, never writes caches into original data."""
    root=Path(root);images=root/'images'/split;labels=root/'labels'/split
    images.mkdir(parents=True,exist_ok=False);labels.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed)  # reset for every severity; Gaussian fields shared across severity levels
    manifest=[];decoded=[]
    for row in rows:
        if digest(row['source_image'])!=row['image_sha256'] or digest(row['source_label'])!=row['label_sha256']:
            raise ValueError('Source changed before copy: '+row['image_id'])
        rgb=np.asarray(Image.open(row['source_image']).convert('RGB'),dtype=np.float64)/255.
        noisy=rgb.copy()
        if corruption:
            kind,value=corruption['kind'],corruption['value']
            if kind=='gaussian': noisy=np.clip(rgb+rng.normal(0,value,rgb.shape),0,1)
            elif kind=='impulse':
                mask=rng.random(rgb.shape[:2]);noisy[mask<value/2]=0.;noisy[mask>1-value/2]=1.
            else: raise ValueError('Unknown corruption')
        quantized=np.round(noisy*255).astype(np.uint8)
        image=images/(row['image_id']+'.png');label=labels/(row['image_id']+'.txt')
        Image.fromarray(quantized).save(image);shutil.copyfile(row['source_label'],label)
        mse=float(np.mean((quantized.astype(float)/255-rgb)**2));signal=float(np.mean(rgb**2))
        snr=float(10*np.log10((signal+1e-12)/(mse+1e-12)))
        item=dict(row,working_image=str(image.resolve()),derived_image_sha256=digest(image),derived_label_sha256=digest(label),snr_db=snr,mse=mse)
        manifest.append(item)
        decoded.append(dict(image_id=row['image_id'],bgr=np.ascontiguousarray(quantized[:,:,::-1]),gt=row['gt']))
    spec=dict(path=str(root.resolve()),train=f'images/{split}',val=f'images/{split}',test=f'images/{split}',names=names)
    data=root/'data.yaml';data.write_text(yaml.safe_dump(spec,allow_unicode=True),encoding='utf-8')
    dump(root/'manifest.json',dict(split=split,seed=seed,rng='numpy.default_rng; reset each severity; Gaussian base samples shared across severities; impulse uses its own uniform fields',
        quantization='RGB float64 [0,1] -> clip -> round(255*x) uint8 PNG; impulse mask shared by RGB channels',
        corruption=corruption,records=manifest,scope='synthetic perturbation, not acquisition noise; exploratory after clean training results'))
    return data, decoded, manifest


class RecordingValidator(DetectionValidator):
    def update_metrics(self,preds,batch):
        super().update_metrics(preds,batch)
        with (self.save_dir/'ap_predictions.jsonl').open('a',encoding='utf-8') as f:
            for si,pred in enumerate(preds):
                pbatch=self._prepare_batch(si,batch)
                native=self.scale_preds(self._prepare_pred(pred),pbatch)
                f.write(json.dumps(dict(image_id=Path(pbatch['im_file']).stem,
                    boxes=native['bboxes'].detach().cpu().tolist(),confidence=native['conf'].detach().cpu().tolist(),
                    classes=native['cls'].detach().cpu().tolist(),scope='official AP low-conf multi-label NMS; not fixed operating point'))+'\n')


def official_ap(yolo,data,names,config,out):
    result=yolo.val(validator=RecordingValidator,data=str(data),split=config['split'],device=config['device'],
        imgsz=config['imgsz'],batch=1,workers=0,half=False,rect=False,conf=config['ap_conf'],iou=config['nms_iou'],
        max_det=config['max_det'],plots=False,save_json=False,save_txt=False,project=str(out),name='official',exist_ok=False)
    by_id={int(k):i for i,k in enumerate(result.box.ap_class_index)}
    classes=[]
    for k,name in enumerate(names):
        if k in by_id:
            i=by_id[k];p,r,ap50,ap=result.box.class_result(i)
            row=dict(class_id=k,name=name,AP50=float(ap50),AP50_95=float(ap),official_best_F1_P=float(p),official_best_F1_R=float(r))
        else: row=dict(class_id=k,name=name,AP50=None,AP50_95=None,official_best_F1_P=None,official_best_F1_R=None)
        classes.append(row)
    payload=dict(metrics={k:float(v) for k,v in result.results_dict.items()},classes=classes,
        conf=config['ap_conf'],nms_iou=config['nms_iou'],max_det=config['max_det'],multi_label=True,
        match='Ultralytics official validator IoU 0.50:0.05:0.95',split=config['split'],
        warning='official best-F1 P/R are not fixed conf=.25 P/R')
    dump(out/'official_ap.json',payload);write_csv(out/'official_per_class.csv',classes)
    return payload


def fixed_evaluate(detector,images,names,config,out):
    counts=[]
    with (out/'fixed_predictions_matches.jsonl').open('w',encoding='utf-8') as f:
        for image in images:
            pred=detector.predict(image['bgr'])
            predictions=[dict(xyxy=row[:4].tolist(),confidence=float(row[4]),class_id=int(row[5])) for row in pred]
            rows,matches=match_image(predictions,image['gt'],names,config['match_iou']);counts.append(rows)
            f.write(json.dumps(dict(image_id=image['image_id'],predictions=predictions,gt=image['gt'],matches=matches))+'\n');f.flush()
    summary=summarize_counts(counts,names)
    summary.update(conf=config['conf'],nms_iou=config['nms_iou'],match_iou=config['match_iou'],max_det=config['max_det'],
        multi_label=False,matching='per class score-descending greedy highest IoU unmatched GT, stable index tie-break, one-to-one; duplicate and wrong-class predictions FP')
    dump(out/'fixed_metrics.json',summary);write_csv(out/'fixed_per_class.csv',summary['classes'])
    return summary
