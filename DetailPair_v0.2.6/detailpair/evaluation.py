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


def array_identity(array):
    array=np.ascontiguousarray(array)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def prepare_legacy_copy(records, names, entry, root):
    """Byte-identical private source copies, preserving old selection working-name order."""
    from pathlib import PureWindowsPath
    root=Path(root);images=root/'images/val';labels=root/'labels/val'
    images.mkdir(parents=True);labels.mkdir(parents=True)
    source={str(Path(r['source_image']).resolve()):r for r in records}
    selection=json.loads(Path(entry['selection']).read_text())['val']
    mapping={};manifest=[];ordered=[]
    for i,item in enumerate(selection):
        row=source[str(Path(item['image']).resolve())]
        source_path=Path(row['source_image']);working=item.get('working_image')
        filename=PureWindowsPath(working).name if working else f'{i:06d}_'+source_path.name
        if Path(filename).suffix.lower()!=source_path.suffix.lower():raise ValueError('Legacy source-format mismatch')
        dest=images/filename;label=labels/Path(filename).with_suffix('.txt')
        if dest.exists():raise ValueError('Duplicate legacy working filename')
        shutil.copyfile(source_path,dest);shutil.copyfile(row['source_label'],label)
        if digest(dest)!=row['image_sha256'] or digest(label)!=row['label_sha256']:raise RuntimeError('Legacy byte copy differs')
        mapping[str(dest.resolve())]=row['image_id'];ordered.append(filename)
        manifest.append(dict(row,working_image=str(dest.resolve()),copy_sha256=digest(dest),copy_label_sha256=digest(label)))
    spec=dict(path=str(root.resolve()),train='images/val',val='images/val',names=names)
    data=root/'data.yaml';data.write_text(yaml.safe_dump(spec,allow_unicode=True),encoding='utf-8')
    dump(root/'manifest.json',dict(records=manifest,selection_order=ordered,source_format_preserved=True,
        note='Official rect loader may sort by aspect ratio; actual order is recorded by the observing validator'))
    return data,mapping,manifest


def diagnostic_ap(entry, config, names, data, mapping, arm, out):
    """Observe pinned official validator without changing its parsing or tensors."""
    import inspect
    from unittest.mock import patch
    import ultralytics.data.base as base_module
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    decoded=[];inputs=[];matches=[];effective={}
    original_imread=base_module.imread
    def observe_imread(filename,*args,**kwargs):
        image=original_imread(filename,*args,**kwargs)
        if image is not None and str(Path(filename).resolve()) in mapping:
            decoded.append(dict(image_id=mapping[str(Path(filename).resolve())],file=str(filename),
                loader='ultralytics.data.base.imread',BGR=array_identity(image),RGB=array_identity(image[:,:,::-1])))
        return image
    def transforms_info(value):
        info=[]
        for transform in getattr(value,'transforms',[]):
            attrs={k:getattr(transform,k) for k in ['new_shape','auto','scale_fill','scaleup','center','stride','padding_value','interpolation','normalize','bbox_format'] if hasattr(transform,k)}
            info.append(dict(type=type(transform).__name__,attributes=attrs))
        return info
    class ObservedValidator(RecordingValidator):
        def init_metrics(self,model):
            super().init_metrics(model)
            dataset=self.dataloader.dataset
            effective.update(validator_args=vars(self.args).copy(),device=str(self.device),
                loader_order=[dict(file=str(f),image_id=mapping[str(Path(f).resolve())]) for f in dataset.im_files],
                dataset=dict(imgsz=dataset.imgsz,rect=dataset.rect,channels=getattr(dataset,'channels','unknown'),
                    batch_shapes=getattr(dataset,'batch_shapes',None).tolist() if getattr(dataset,'batch_shapes',None) is not None else None,
                    transforms=transforms_info(dataset.transforms)),
                backend=dict(type=type(model).__name__,fp16=getattr(model,'fp16','unknown'),
                    parameters=sum(p.numel() for p in model.parameters())))
            dump(out/'effective_validator.json',effective)
        def preprocess(self,batch):
            before=batch['img'].detach().cpu().numpy()
            prepared=super().preprocess(batch)
            after=prepared['img'].detach().cpu().numpy()
            for i,file in enumerate(prepared['im_file']):
                inputs.append(dict(image_id=mapping[str(Path(file).resolve())],file=str(file),batch_shape=list(prepared['img'].shape),
                    pre_normalization=array_identity(before[i]),normalized=array_identity(after[i]),
                    ori_shape=list(prepared['ori_shape'][i]),ratio_pad=prepared['ratio_pad'][i],
                    normalized_min=float(after[i].min()),normalized_max=float(after[i].max())))
            return prepared
        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch)
            for si,pred in enumerate(preds):
                target=self._prepare_batch(si,batch);prepared=self._prepare_pred(pred)
                native=self.scale_preds(prepared,target)
                tp=self._process_batch(prepared,target)['tp']
                gt=ops.scale_boxes(target['imgsz'],target['bboxes'].clone(),target['ori_shape'],ratio_pad=target['ratio_pad'])
                matches.append(dict(image_id=mapping[str(Path(target['im_file']).resolve())],
                    boxes=native['bboxes'].detach().cpu().tolist(),confidence=native['conf'].detach().cpu().tolist(),
                    classes=native['cls'].detach().cpu().tolist(),GT_classes=target['cls'].detach().cpu().tolist(),
                    GT_xyxy=gt.detach().cpu().tolist(),TP_by_official_IoU_50_to_95=tp.tolist(),
                    warning='Official multi-label AP matches, not fixed-conf single-label P/R'))
    out.mkdir(parents=True,exist_ok=True)
    (out/'installed_loader_source.py').write_text(inspect.getsource(BaseDataset.load_image),encoding='utf-8')
    (out/'installed_letterbox_source.py').write_text(inspect.getsource(LetterBox),encoding='utf-8')
    yolo,identity=validate_checkpoint(entry,config,names)
    # Old evaluate signature is copied from v0.2.4, with output path/class observation only.
    kwargs=dict(data=str(data),split='val',imgsz=config['imgsz'],batch=1,device=config['device'],workers=0,
        plots=True,project=str(out),name='official',exist_ok=True)
    if arm!='legacy_jpeg':
        kwargs.update(half=False,rect=arm=='png_rect_only',conf=.001,iou=.7,max_det=300,plots=False,save_json=False,save_txt=False)
    dump(out/'call_kwargs.json',kwargs)
    dump(out/'model_overrides_before_val.json',yolo.overrides)
    try:
        with patch.object(base_module,'imread',side_effect=observe_imread):
            result=yolo.val(validator=ObservedValidator,**kwargs)
        classes=[];by_id={int(k):i for i,k in enumerate(result.box.ap_class_index)}
        for k,name in enumerate(names):
            values=result.box.class_result(by_id[k]) if k in by_id else [None]*4
            classes.append(dict(class_id=k,name=name,official_best_F1_P=float(values[0]) if values[0] is not None else None,
                official_best_F1_R=float(values[1]) if values[1] is not None else None,
                AP50=float(values[2]) if values[2] is not None else None,AP50_95=float(values[3]) if values[3] is not None else None))
        payload=dict(arm=arm,identity=identity,metrics={k:float(v) for k,v in result.results_dict.items()},classes=classes)
        dump(out/'official_ap.json',payload);write_csv(out/'official_per_class.csv',classes)
        if len(inputs)!=len(mapping) or len(decoded)!=len(mapping) or len(matches)!=len(mapping):
            raise RuntimeError('Incomplete observed loader/input/prediction records')
        dump(out/'status.json',dict(status='success',images=len(inputs)))
        return payload,effective,inputs,decoded,matches
    finally:
        for filename,rows in [('decoded_pixels.jsonl',decoded),('input_tensors.jsonl',inputs),('predictions_official_matches.jsonl',matches)]:
            with (out/filename).open('w',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,default=str)+'\n')
        yolo.model.cpu()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def compare_diagnostic_arms(legacy,new):
    def keyed(rows):return {r['image_id']:r for r in rows}
    li,ni=keyed(legacy[2]),keyed(new[2]);ld,nd=keyed(legacy[3]),keyed(new[3])
    lp,npred=keyed(legacy[4]),keyed(new[4])
    rows=[]
    for image_id in sorted(set(li)|set(ni)):
        a,b=li[image_id],ni[image_id]
        pa,pb=lp[image_id],npred[image_id]
        rows.append(dict(image_id=image_id,legacy_shape=a['batch_shape'],new_shape=b['batch_shape'],
            shape_equal=a['batch_shape']==b['batch_shape'],dtype_equal=a['normalized']['dtype']==b['normalized']['dtype'],
            normalized_tensor_equal=a['normalized']==b['normalized'],
            loader_RGB_equal=ld[image_id]['RGB']==nd[image_id]['RGB'],
            legacy_predictions=len(pa['classes']),new_predictions=len(pb['classes']),predictions_exactly_equal=pa==pb))
    args_a,args_b=legacy[1]['validator_args'],new[1]['validator_args']
    excluded={'data','project','name','save_dir','model'}
    diff={k:dict(legacy=args_a.get(k),new=args_b.get(k)) for k in set(args_a)|set(args_b) if k not in excluded and args_a.get(k)!=args_b.get(k)}
    class_differences=[]
    for a,b in zip(legacy[0]['classes'],new[0]['classes']):
        subtract=lambda x,y:x-y if x is not None and y is not None else None
        class_differences.append(dict(class_id=a['class_id'],name=a['name'],legacy_AP50=a['AP50'],new_AP50=b['AP50'],
            legacy_AP50_95=a['AP50_95'],new_AP50_95=b['AP50_95'],new_minus_legacy_AP50=subtract(b['AP50'],a['AP50']),
            new_minus_legacy_AP50_95=subtract(b['AP50_95'],a['AP50_95'])))
    return dict(images=rows,classes=class_differences,effective_argument_differences=diff,
        decoded_RGB_changed=sum(not r['loader_RGB_equal'] for r in rows),
        input_tensor_changed=sum(not r['normalized_tensor_equal'] for r in rows),
        shape_changed=sum(not r['shape_equal'] for r in rows),
        dtype_changed=sum(not r['dtype_equal'] for r in rows),
        predictions_changed=sum(not r['predictions_exactly_equal'] for r in rows),
        new_minus_legacy={k:new[0]['metrics'][k]-legacy[0]['metrics'][k] for k in new[0]['metrics']},
        interpretation='Observed differences only. Main old/new arms are confounded if multiple settings/pixels differ; no pre-assigned JPEG, FP32 or model blame.')
