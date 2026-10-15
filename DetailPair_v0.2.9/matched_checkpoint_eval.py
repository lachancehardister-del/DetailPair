"""Inventory old B5/B6 checkpoints, then evaluate only a confirmed common epoch."""
import argparse,copy,hashlib,json,os,sys,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('WANDB_DISABLED','true')
from run import stamp,environment
from detailpair.reporting import dump,write_csv,comparison_row,compare_fixed_summaries


def validate_config(c,synthetic=False):
    expected=dict(protocol_version='DetailPair-old40-matched-1.0',device='0',split='val',imgsz=320,batch=1,precision='FP32',fuse=True,
        conf=.25,ap_conf=.001,nms_iou=.7,match_iou=.5,max_det=300,seed=178,cpu_threads=2,expected_images=180,expected_gt=429,
        expected_train_images=1438,corruptions=[dict(kind='gaussian',value=.03)])
    if bool(c.get('synthetic_check'))!=synthetic:raise ValueError('CPU fixture needs explicit --synthetic-check')
    if synthetic:expected.update(device='cpu',expected_images=c['expected_images'],expected_gt=c['expected_gt'],expected_train_images=c['expected_train_images'])
    for k,v in expected.items():
        if c.get(k)!=v:raise ValueError(f'Frozen {k} requires {v!r}')
    if [v['variant'] for v in c['variants']]!=['B5','B6']:raise ValueError('Only original B5/B6 pair supported')
    for e in c['variants']:
        if not isinstance(e['weight_dirs'],list) or not e['weight_dirs']:raise ValueError('Explicit bounded trusted weight directories required')


def pack_results():
    exports=ROOT/'exports';exports.mkdir(exist_ok=True);target=exports/('DetailPair_matched_results_'+stamp()+'.zip')
    excluded={'.pt','.pth','.npy','.npz','.cache','.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.pyc'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['runs','verification','detailpair','configs','windows','tests','archive']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix.lower() not in excluded and not {'datasets','images','labels','__pycache__'}&set(p.relative_to(ROOT).parts):z.write(p,p.relative_to(ROOT))
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in {'.py','.md','.json','.txt'} and p.name!='ARTIFACT_MANIFEST.json':z.write(p,p.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Result ZIP CRC failed')
    print('Result ZIP:',target,flush=True);return target


def preflight(c,synthetic,out,require_gpu=True):
    import torch,torchvision,ultralytics
    from ultralytics import settings
    info=dict(torch=torch.__version__,torchvision=torchvision.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),synthetic_check=synthetic,evaluation_device_requested=c['device'],gpu_required_now=require_gpu and not synthetic)
    dump(out/'dependency_device.json',info)
    for actual,expected in [(torch.__version__,'2.5.1'),(torchvision.__version__,'0.20.1'),(ultralytics.__version__,'8.3.221')]:
        if actual.split('+')[0]!=expected:raise RuntimeError('Pinned versions required; no automatic install/upgrade')
    settings.update({k:False for k in ['sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb']})
    if require_gpu and not synthetic and (not torch.cuda.is_available() or torch.version.cuda!='12.1'):raise RuntimeError('CUDA12.1 GPU0 required; no CPU fallback')
    device=torch.device('cpu' if synthetic or not require_gpu else 'cuda:0')
    if not synthetic and torch.cuda.is_available():
        info['gpu_name']=torch.cuda.get_device_name(0);dump(out/'dependency_device.json',info)
    torch.set_num_threads(c['cpu_threads']);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return device


def private_spec(data,out):
    """Keep source train/val paths, never enumerate or load test entries."""
    import yaml
    data=Path(data).resolve();spec=yaml.safe_load(data.read_text(encoding='utf-8'))
    base=Path(spec.get('path',data.parent));base=base if base.is_absolute() else data.parent/base
    narrowed=dict(path=str(base.resolve()),names=spec['names'],train=spec['train'],val=spec['val'])
    dest=out/'train_val_only.yaml';dest.write_text(yaml.safe_dump(narrowed,allow_unicode=True),encoding='utf-8');return dest


def matched_ap(yolo,data,names,c,dest):
    """Reuse AP reporting, observing every actual input after pinned preprocessing."""
    from unittest.mock import patch
    from detailpair import evaluation
    from detailpair.evaluation import array_identity
    inputs=[]
    class Observed(evaluation.RecordingValidator):
        def init_metrics(self,model):
            super().init_metrics(model)
            expected=dict(rect=False,half=False,augment=False,single_cls=False,agnostic_nms=False,classes=None,
                conf=.001,iou=.7,max_det=300,batch=1,imgsz=320)
            actual={k:getattr(self.args,k,None) for k in expected}
            dump(dest/'effective_official_settings.json',dict(arguments=actual,device=str(self.device),backend_fp16=getattr(model,'fp16',None)))
            if actual!=expected:raise RuntimeError('Official effective settings differ from frozen protocol')
        def preprocess(self,batch):
            batch=super().preprocess(batch)
            shape=list(batch['img'].shape)
            if shape!=[1,3,320,320] or str(batch['img'].dtype)!='torch.float32':raise RuntimeError('Official actual shape/dtype changed')
            if str(batch['img'].device)!=('cpu' if c['device']=='cpu' else 'cuda:0'):raise RuntimeError('Official actual device changed')
            for i,file in enumerate(batch['im_file']):inputs.append(dict(image_id=Path(file).stem,actual_shape=shape,actual_device=str(batch['img'].device),dtype=str(batch['img'].dtype),input=array_identity(batch['img'][i].cpu().numpy())))
            return batch
    try:
        with patch.object(evaluation,'RecordingValidator',Observed):result=evaluation.official_ap(yolo,data,names,c,dest)
    finally:dump(dest/'official_actual_inputs.json',inputs)
    predictions=dest/'official/ap_predictions.jsonl'
    if len(inputs)!=c['expected_images'] or not predictions.is_file() or len(predictions.read_text().splitlines())!=len(inputs):raise RuntimeError('Official predictions/input records incomplete')
    return result,inputs


def comparison(results,out):
    rows=[];classes=[]
    for scenario in ['clean','gaussian_0.03']:
        if not all((v,scenario) in results for v in ['B5','B6']):continue
        a,b=[results[v,scenario] for v in ['B5','B6']]
        for key in ['metrics/mAP50-95(B)','metrics/mAP50(B)']:
            x,y=a['ap']['metrics'][key],b['ap']['metrics'][key]
            rows.append(dict(scenario=scenario,**comparison_row(key,x,y,definition='Unchanged official AP output')))
        rows.extend(dict(scenario=scenario,**row) for row in compare_fixed_summaries(a['fixed'],b['fixed']))
        x,y=a['fixed']['FPPI'],b['fixed']['FPPI']
        rows.append(dict(scenario=scenario,**comparison_row('FPPI',x,y,definition='Total FP / evaluated images')))
        for k in range(len(a['ap']['classes'])):
            for key in ['AP50','AP50_95','TP','FP','FN','GT','Precision','Recall']:
                source='ap' if key.startswith('AP') else 'fixed';x,y=a[source]['classes'][k][key],b[source]['classes'][k][key]
                classes.append(dict(scenario=scenario,class_id=k,name=a[source]['classes'][k]['name'],metric=key,B5=x,B6=y,B5_minus_B6=None if x is None or y is None else x-y))
    write_csv(out/'comparison_B5_minus_B6.csv',rows);write_csv(out/'class_comparison_B5_minus_B6.csv',classes)
    dump(out/'comparison.json',dict(rows=rows,classes=classes,reporting_schema='DetailPair-fixed-operating-point-2.0',
        scope='Existing short-budget single seed only; available-class macro differences require equal actual support sets. No causal proof, speed claim, significance, automatic training or final method claim'))


def run(c,out,synthetic=False,inventory_only=False):
    import torch
    from detailpair.matched_checkpoints import epoch_semantics,inventory,select_pair,load_selected
    from detailpair.evaluation import read_records,verify_selection,prepare_dataset,fixed_evaluate,array_identity
    from detailpair.benchmark import HostDetector
    from detailpair.data import digest
    validate_config(c,synthetic)
    preflight(c,synthetic,out,require_gpu=False)  # pinned imports first, inventory remains possible without GPU
    dump(out/'epoch_semantics.json',epoch_semantics(out))
    inventories,csvs,dirs=inventory(c['variants'],out)
    pair=select_pair(inventories);dump(out/'pairing.json',pair)
    base=dict(no_training=True,no_optimizer=True,no_resume=True,timing_performed=False,test_evaluated=False,synthetic_check=synthetic,
        successful_evaluations=0,expected_evaluations=4,weights_and_sources_verified_unchanged=None)
    if pair['status']!='matched':
        dump(out/'status.json',dict(base,status=pair['status']));return 2
    if inventory_only:
        dump(out/'status.json',dict(base,status='inventory_only_matched_not_evaluated'));return 0
    device=preflight(c,synthetic,out)
    protected={};results={};failures=[];immutability=None
    try:
        protected[str(Path(c['data']).resolve())]=digest(c['data'])
        data=private_spec(c['data'],out);names,records=read_records(data,'val')
        if len(records)!=c['expected_images'] or sum(len(r['gt']) for r in records)!=c['expected_gt']:raise ValueError('Wrong val180/429GT cardinality')
        dump(out/'source_manifest.json',dict(names=names,records=records,split='val',test_read=False))
        selections={}
        for e in c['variants']:
            selection=verify_selection(dict(e,expected_train_images=c['expected_train_images']),data)
            selections[e['variant']]=selection['source_identity'];dump(out/(e['variant']+'_selection_identity.json'),selection)
            protected[str(Path(e['selection']).resolve())]=digest(e['selection'])
            protected[str(Path(e['csv']).resolve())]=digest(e['csv'])
            identity=pair['selected'][e['variant']];protected[identity['path']]=identity['sha256']
        if selections['B5']!=selections['B6']:raise ValueError('Training/validation source selections differ')
        for split in ['train','val']:
            for row in selections['B5'][split]:
                protected[row['image']]=row['image_sha256']
                from detailpair.data import label_for
                protected[str(label_for(Path(row['image'])))]=row['label_sha256']
        dump(out/'training_selection_pair.json',dict(equal=True,source_identity_sha256=hashlib.sha256(json.dumps(selections['B5'],sort_keys=True).encode()).hexdigest(),
            original_initialization_tensor_hash='unavailable; equal seed is not historical trajectory proof'))
        datasets={}
        for scenario,noise in [('clean',None),('gaussian_0.03',dict(kind='gaussian',value=.03))]:
            ds,images,manifest=prepare_dataset(records,names,out/'datasets'/scenario,'val',noise,c['seed'])
            dump(out/(scenario+'_derived_manifest.json'),dict(records=manifest,shared_by=['B5','B6'],corruption=noise,seed=178,
                decoding='PIL RGB float64 /255; clip + numpy.round *255 -> shared uint8 lossless PNG; native GT coordinates unchanged'))
            datasets[scenario]=(ds,images,manifest)
        for scenario,(ds,images,manifest) in datasets.items():
            observed={};fixed_observed={}
            for variant in ['B5','B6']:
                dest=out/variant/scenario;dest.mkdir(parents=True)
                identity=pair['selected'][variant]
                dump(dest/'evaluation_identity.json',dict(checkpoint=identity,completed_epoch=pair['completed_epoch'],original_budget=40,scenario=scenario,
                    data_config_sha256=protected[str(Path(c['data']).resolve())],derived_manifest_sha256=digest(out/(scenario+'_derived_manifest.json')),settings=c))
                try:
                    if any(digest(r['working_image'])!=r['derived_image_sha256'] or digest(Path(r['working_image']).parents[2]/'labels/val'/(r['image_id']+'.txt'))!=r['derived_label_sha256'] for r in manifest):raise ValueError('Shared image/label changed')
                    yolo=load_selected(identity,names)
                    ap,inputs=matched_ap(yolo,ds,names,c,dest);observed[variant]=inputs
                    # Use a separate reload so official validation fusion cannot affect fixed evaluation.
                    del yolo;yolo=load_selected(identity,names)
                    detector=HostDetector(copy.deepcopy(yolo.model).float().eval(),device,c);actual=[]
                    def capture(module,args):
                        tensor=args[0]
                        if list(tensor.shape)!=[1,3,320,320] or tensor.dtype!=torch.float32:raise RuntimeError('Fixed input not FP32 square320')
                        if str(tensor.device)!=str(device):raise RuntimeError('Fixed actual device changed')
                        actual.append(dict(image_id=images[len(actual)]['image_id'],actual_shape=list(tensor.shape),actual_device=str(tensor.device),dtype=str(tensor.dtype),input=array_identity(tensor.cpu().numpy())))
                    handle=detector.backend.model.register_forward_pre_hook(capture)
                    try:fixed=fixed_evaluate(detector,images,names,c,dest)
                    finally:handle.remove();dump(dest/'fixed_actual_inputs.json',actual)
                    if len(actual)!=len(records):raise RuntimeError('Fixed prediction count mismatch')
                    fixed_observed[variant]=actual
                    fixed.update(FPPI=sum(r['FP'] for r in fixed['classes'])/len(records),images=len(records));dump(dest/'fixed_metrics.json',fixed)
                    results[variant,scenario]=dict(ap=ap,fixed=fixed)
                    dump(dest/'status.json',dict(status='success',checkpoint_sha256=identity['sha256'],completed_epoch=pair['completed_epoch'],actual_images=len(inputs)))
                except Exception:
                    failures.append(variant+'/'+scenario);(dest/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
                    dump(dest/'status.json',dict(status='failed',checkpoint_sha256=identity['sha256']))
                finally:
                    if 'detector' in locals():del detector
                    if 'yolo' in locals():del yolo
                    if device.type=='cuda':torch.cuda.empty_cache()
            if len(observed)==2:
                same=observed['B5']==observed['B6'];dump(out/(scenario+'_paired_inputs.json'),dict(equal=same))
                if not same:raise RuntimeError('Models did not see identical official inputs')
            if len(fixed_observed)==2:
                same=fixed_observed['B5']==fixed_observed['B6'];dump(out/(scenario+'_paired_fixed_inputs.json'),dict(equal=same))
                if not same:raise RuntimeError('Models did not see identical fixed inputs')
        comparison(results,out)
    finally:
        changes=[]
        for path,sha in protected.items():
            try:
                if digest(path)!=sha:changes.append(path)
            except Exception:changes.append(path)
        immutability=not changes
        dump(out/'read_only_verification.json',dict(verified_unchanged=immutability,files_checked=len(protected),changed_or_missing=changes))
        if changes:raise RuntimeError('Protected source/weight changed: '+str(changes))
    dump(out/'status.json',dict(base,status='partial_or_failed' if failures else 'success',failures=failures,successful_evaluations=len(results),
        weights_and_sources_verified_unchanged=immutability,completed_epoch=pair['completed_epoch']))
    return 1 if failures else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'configs/matched_checkpoint_eval.yaml'))
    p.add_argument('--pack-only',action='store_true');p.add_argument('--inventory-only',action='store_true');p.add_argument('--synthetic-check',action='store_true')
    args=p.parse_args()
    if args.pack_only:pack_results();return 0
    out=ROOT/'runs'/('matched_'+stamp());out.mkdir(parents=True)
    dump(out/'request.json',vars(args));dump(out/'environment.json',environment());code=1
    try:
        import yaml
        raw=Path(args.config).read_bytes();c=yaml.safe_load(raw);(out/'frozen_config.yaml').write_bytes(raw)
        dump(out/'config_identity.json',dict(path=args.config,sha256=hashlib.sha256(raw).hexdigest()))
        code=run(c,out,args.synthetic_check,args.inventory_only)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
        dump(out/'status.json',dict(status='failed',no_training=True,test_evaluated=False,timing_performed=False))
    finally:
        try:pack_results()
        except Exception:
            (out/'pack_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc();code=1
    print('Evaluation directory:',out,flush=True);print('Exit code:',code,flush=True);return code

if __name__=='__main__':sys.exit(main())
