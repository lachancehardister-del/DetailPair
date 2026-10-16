"""New-condition evaluation; no old B2/B6 or old40 checkpoint validator is used."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from .identity import file_sha256 as digest
from .trainer import SquareValidator
from .reporting import dump,write_csv,match_image,summarize_counts

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

class RecordingValidator(SquareValidator):
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
            row=dict(class_id=k,name=name,AP50=float(ap50),AP50_95=float(ap),AP75=float(result.box.all_ap[i,5]),official_best_F1_P=float(p),official_best_F1_R=float(r))
        else: row=dict(class_id=k,name=name,AP50=None,AP50_95=None,AP75=None,official_best_F1_P=None,official_best_F1_R=None)
        classes.append(row)
    metrics={k:float(v) for k,v in result.results_dict.items()};all_ap=np.asarray(result.box.all_ap);metrics['metrics/mAP75(B)']=float(all_ap[:,5].mean()) if all_ap.ndim==2 and len(all_ap) else 0.
    payload=dict(metrics=metrics,classes=classes,
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
        images=len(images),FPPI=sum(r['FP'] for r in summary['classes'])/len(images) if images else None,
        multi_label=False,matching='per class score-descending greedy highest IoU unmatched GT, stable index tie-break, one-to-one; duplicate and wrong-class predictions FP')
    dump(out/'fixed_metrics.json',summary);write_csv(out/'fixed_per_class.csv',summary['classes'])
    return summary

def array_identity(array):
    array=np.ascontiguousarray(array)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def native_records(manifest):
    rows=[]
    for source in manifest['records']['val']:
        gt=[]
        for k,x,y,w,h in source['boxes']:
            gt.append(dict(class_id=int(k),xyxy=[(x-w/2)*source['width'],(y-h/2)*source['height'],(x+w/2)*source['width'],(y+h/2)*source['height']]))
        rows.append(dict(image_id=source['image_id'],source_image=source['source_image'],source_label=source['source_label'],
            image_sha256=source['image_sha256'],label_sha256=source['label_sha256'],width=source['width'],height=source['height'],gt=gt))
    return rows


def validate_endpoint(path,variant,epochs,names):
    from .model import BoxRouteModel
    from ultralytics.nn.tasks import DetectionModel
    path=Path(path);sha=digest(path);checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint.get('model')
    if checkpoint.get('epoch')!=epochs-1 or checkpoint.get('ema') is None:raise ValueError('Unstripped exact completed-epoch EMA required')
    if variant=='N':
        if type(model)!=DetectionModel:raise ValueError('Native endpoint identity differs')
    elif not isinstance(model,BoxRouteModel) or model.variant!=variant:raise ValueError('Candidate endpoint identity differs')
    expected=dict(epochs=epochs,batch=2,imgsz=320,amp=False,optimizer='AdamW',lr0=.001,lrf=.01,nbs=64,seed=0,resume=False,rect=False,save_period=-1,pretrained=True,patience=0,deterministic=True,compile=False)
    for k,value in expected.items():
        if checkpoint['train_args'].get(k)!=value:raise ValueError('Endpoint training identity '+k)
    mapping=model.names;mapping=[mapping[k] for k in sorted(mapping)] if isinstance(mapping,dict) else list(mapping)
    if mapping!=names or model.model[-1].f!=[16,19,22]:raise ValueError('Class mapping or graph changed')
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Nonfinite endpoint')
    import csv
    csv_path=path.parents[1]/'results.csv'
    with csv_path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=epochs or [int(float(r['epoch'])) for r in rows]!=list(range(1,epochs+1)):raise ValueError('Incomplete or mismatched endpoint CSV')
    if digest(path)!=sha:raise RuntimeError('Endpoint file changed during validation')
    return dict(variant=variant,sha256=sha,completed_epoch=epochs,checkpoint_epoch=epochs-1,selected_role='ema',
        read_only=True,CSV_sha256=digest(csv_path),train_args=checkpoint['train_args'])


def grouped_GT_diagnostics(dest):
    """GT groups only for recall/IoU; FP kept overall/per class, never shape-attributed."""
    groups={};records=[json.loads(x) for x in (dest/'fixed_predictions_matches.jsonl').read_text().splitlines()]
    for row in records:
        matched={m['gt_index']:m['matched_iou'] for m in row['matches'] if m['status']=='TP'}
        for i,target in enumerate(row['gt']):
            x0,y0,x1,y1=target['xyxy'];w,h=x1-x0,y1-y0;ratio=w/h;area=w*h
            labels=['aspect_horizontal' if ratio>=1.5 else 'aspect_vertical' if ratio<=1/1.5 else 'aspect_balanced',
                'area_small' if area<32**2 else 'area_medium' if area<96**2 else 'area_large']
            for group in labels:
                g=groups.setdefault(group,dict(GT=0,TP=0,matched_IoU=[]));g['GT']+=1
                if i in matched:g['TP']+=1;g['matched_IoU'].append(matched[i])
    for group in groups.values():
        group['Recall']=group['TP']/group['GT'];values=group.pop('matched_IoU');group['matched_IoU_mean']=sum(values)/len(values) if values else None
    dump(dest/'GT_group_diagnostics.json',dict(groups=groups,definition='Native-source GT pixel area (<32^2,32^2..96^2,>=96^2), aspect ratio1.5 and reciprocal; recall and matched-IoU only, no grouped AP or FP assignment'))
    return groups


def evaluate_suite(endpoints,manifest,config,out,epochs=60):
    import copy
    from ultralytics import YOLO
    from .benchmark import HostDetector
    out=Path(out);out.mkdir(parents=True,exist_ok=False);names=manifest['names'];rows=native_records(manifest);results={}
    for scenario,noise in [('clean',None),('gaussian_0.03',dict(kind='gaussian',value=.03))]:
        data,images,records=prepare_dataset(rows,names,out/'datasets'/scenario,'val',noise,178)
        dump(out/(scenario+'_manifest.json'),dict(shared_by=list(endpoints),records=records,seed=178,corruption=noise))
        for variant,path in endpoints.items():
            dest=out/variant/scenario;dest.mkdir(parents=True);identity=validate_endpoint(path,variant,epochs,names)
            try:
                yolo=YOLO(str(path));ap=official_ap(yolo,data,names,config,dest)
                # Fresh reload; do not reuse any validator fusion state.
                detector=HostDetector(YOLO(str(path)).model.float().eval(),torch.device('cuda:0' if config['device']=='0' else 'cpu'),config)
                fixed=fixed_evaluate(detector,images,names,config,dest)
                if detector.shape!=[1,3,320,320]:raise RuntimeError('Fixed input shape differs')
                grouped_GT_diagnostics(dest);results[variant,scenario]=dict(ap=ap,fixed=fixed,identity=identity)
                dump(dest/'status.json',dict(status='success',identity=identity))
            except Exception:
                import traceback
                (dest/'failure.txt').write_text(traceback.format_exc());dump(dest/'status.json',dict(status='failed',identity=identity));raise
            finally:
                if digest(path)!=identity['sha256']:raise RuntimeError('Frozen endpoint file changed')
                if 'detector' in locals():del detector
                if 'yolo' in locals():del yolo
                if torch.cuda.is_available():torch.cuda.empty_cache()
    dump(out/'all_results.json',{v+'/'+s:r for (v,s),r in results.items()});return results
