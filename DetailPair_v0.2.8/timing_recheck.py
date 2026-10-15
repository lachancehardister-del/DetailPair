"""Bounded balanced GPU timing and controlled clean AP diagnostics; no training/test."""
import argparse,datetime,hashlib,json,os,sys,traceback,zipfile
from pathlib import Path
# Create the configuration path before importing Ultralytics.
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('WANDB_DISABLED','true')
from run import stamp,environment
from detailpair.reporting import dump,write_csv

EXPECTED_HASHES={
 'B2':'c234ae47e6d8257543a038abc3ea0714d0f081abf2ce5e9dbe0dcd9c73a2c21b',
 'B6':'2a5f5dd7152b28a3a3952f96fbeca409e231e07de2a06a6b5686d7a01c6bc324'}


def validate_recheck_config(c,synthetic):
    if c.get('protocol_version')!='DetailPair-timing-recheck-2.0':raise ValueError('Separate 2.0 recheck protocol required; legacy1.0 unchanged')
    if bool(c.get('synthetic_check'))!=synthetic:raise ValueError('CPU synthetic configuration requires explicit --synthetic-check')
    expected=dict(split='val',imgsz=320,batch=1,precision='FP32',fuse=True,conf=.25,ap_conf=.001,nms_iou=.7,max_det=300,
        block_order=['B2','B6','B6','B2','B6','B2','B2','B6'],drift_threshold=1.10,single_factor='rect_if_multiple_differences')
    if not synthetic:expected.update(device='0',common_warmup_seconds=30.,block_warmup_iterations=200,expected_images=180,expected_gt=429,expected_gpu_substring='RTX 4060 Laptop GPU')
    else:
        if c['device']!='cpu' or c['common_warmup_seconds']<=0 or c['block_warmup_iterations']<1:raise ValueError('Explicit short CPU plumbing configuration required')
    for k,v in expected.items():
        if c.get(k)!=v:raise ValueError(f'Frozen {k} must be {v!r}, got {c.get(k)!r}')
    if c['expected_images']<3 or c['expected_images']%3 or c['cpu_threads']<1:raise ValueError('Three equal positive windows and CPU threads required')
    if [e['variant'] for e in c['variants']]!=['B2','B6'] or any(e['seed']!=0 for e in c['variants']):raise ValueError('Explicit B2/B6 seed0 identities required')
    if not synthetic:
        for e in c['variants']:
            if e['expected_sha256']!=EXPECTED_HASHES[e['variant']]:raise ValueError('Frozen real checkpoint SHA256 changed')


def pack_recheck():
    exports=ROOT/'exports';exports.mkdir(exist_ok=True)
    target=exports/('DetailPair_recheck_results_'+stamp()+'.zip')
    images={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['runs','verification','detailpair','configs','windows','tests']:
            base=ROOT/folder
            if not base.exists():continue
            for f in sorted(base.rglob('*')):
                if not f.is_file():continue
                rel=f.relative_to(ROOT)
                if {'datasets','__pycache__','images','labels','synthetic_data'}&set(rel.parts) or any(p.startswith('subset') for p in rel.parts):continue
                if f.suffix.lower() in images|{'.pt','.pyc','.cache','.npy','.onnx','.engine'}:continue
                z.write(f,rel)
        for f in ROOT.iterdir():
            if f.is_file() and f.suffix in {'.py','.md','.txt','.json'} and f.name not in {'ARTIFACT_MANIFEST.json'}:z.write(f,f.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Result ZIP CRC failure')
    print('Result ZIP:',target,flush=True);return target


def preflight(c,out,args):
    import torch,torchvision,ultralytics
    from ultralytics import settings
    from detailpair.data import digest
    from detailpair.evaluation import read_records,verify_selection,validate_checkpoint
    from detailpair.benchmark import gpu_guard
    validate_recheck_config(c,args.synthetic_check)
    for name,version,expected in [('torch',torch.__version__,'2.5.1'),('torchvision',torchvision.__version__,'0.20.1'),('ultralytics',ultralytics.__version__,'8.3.221')]:
        if version.split('+')[0]!=expected:raise RuntimeError(f'{name} requires {expected}; found {version}; no auto installation')
    settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb')})
    if args.synthetic_check:device=torch.device('cpu')
    else:
        if not torch.cuda.is_available() or torch.version.cuda!='12.1':raise RuntimeError('Working CUDA12.1 required; no CPU fallback')
        device=torch.device('cuda:0')
        if c['expected_gpu_substring'] not in torch.cuda.get_device_name(device):raise RuntimeError('GPU identity differs from frozen RTX4060 laptop protocol')
    torch.set_num_threads(c['cpu_threads']);gpu_guard(device,out/'gpu_processes_start.json')
    names,records=read_records(c['data'],'val')
    if len(records)!=c['expected_images'] or sum(len(r['gt']) for r in records)!=c['expected_gt']:raise ValueError('Source val image/GT counts changed')
    if not args.synthetic_check and any(Path(r['source_image']).suffix.lower() not in {'.jpg','.jpeg'} for r in records):raise ValueError('Legacy source protocol expects original JPEGs')
    dump(out/'source_manifest.json',dict(names=names,records=records,scope='val only; test sealed'))
    models={};identities={};selections={}
    for entry in c['variants']:
        v=entry['variant'];p=Path(entry['weights'])
        if not p.is_file():raise FileNotFoundError('Required local checkpoint: '+str(p))
        if digest(p)!=entry['expected_sha256']:raise ValueError(v+' checkpoint SHA256 differs from Work binding')
        selection=verify_selection(dict(entry,**({} if args.synthetic_check else {'expected_train_images':1438})),c['data'])
        selections[v]=selection['source_identity'];dump(out/(v+'_selection_identity.json'),selection)
        models[v],identities[v]=validate_checkpoint(entry,c,names);dump(out/(v+'_checkpoint_identity.json'),identities[v])
    if selections['B2']!=selections['B6']:raise ValueError('B2/B6 training selections differ')
    return device,names,records,models,identities


def clean_diagnostic(c,names,records,png_data,png_manifest,out,device):
    from detailpair.evaluation import prepare_legacy_copy,diagnostic_ap,compare_diagnostic_arms
    from detailpair.benchmark import device_snapshot
    from detailpair.data import digest
    failures=[];reports=[];class_rows=[]
    png_mapping={str(Path(r['working_image']).resolve()):r['image_id'] for r in png_manifest}
    for entry in c['variants']:
        v=entry['variant'];dest=out/v;dest.mkdir(parents=True)
        try:
            device_snapshot(device,dest/'device_before.json',c['hardware_notes'])
            source,mapping,manifest=prepare_legacy_copy(records,names,entry,out.parent/'datasets'/('legacy_'+v))
            dump(dest/'legacy_copy_manifest.json',dict(records=manifest,shared_weights=entry['expected_sha256']))
            old=diagnostic_ap(entry,c,names,source,mapping,'legacy_jpeg',dest/'legacy_jpeg')
            new=diagnostic_ap(entry,c,names,png_data,png_mapping,'new_png',dest/'new_png')
            comparison=compare_diagnostic_arms(old,new)
            third=None
            rect_diff=comparison['effective_argument_differences'].get('rect')
            # Predeclared single-factor arm only; no tuning loop or selection by AP direction.
            additional=comparison['decoded_RGB_changed']>0 or any(k not in {'rect','plots'} for k in comparison['effective_argument_differences'])
            if rect_diff and rect_diff['legacy'] is True and rect_diff['new'] is False and additional:
                third=diagnostic_ap(entry,c,names,png_data,png_mapping,'png_rect_only',dest/'png_rect_only')
                single=compare_diagnostic_arms(third,new)
                comparison['single_factor']=dict(executed=True,factor='rect false -> true on unchanged new PNGs and all other new args',
                    rect_true_minus_rect_false={k:third[0]['metrics'][k]-new[0]['metrics'][k] for k in new[0]['metrics']},
                    remaining_legacy_minus_control={k:old[0]['metrics'][k]-third[0]['metrics'][k] for k in old[0]['metrics']},
                    controlled_observations=single,interpretation='Within this PNG dataset, isolates rect setting including resulting shape/padding. Other residual causes remain undetermined.')
            else:comparison['single_factor']=dict(executed=False,reason='Predeclared rect plus additional measured-difference criterion not met; no alternate factor search')
            prior=Path(entry['prior_metrics']);prior_data=json.loads(prior.read_text()) if prior.is_file() else None
            comparison['historical_metrics']=dict(path=str(prior),sha256=digest(prior) if prior.is_file() else None,old=prior_data,
                status='present' if prior_data else 'missing',new_legacy=old[0]['metrics'],new_explicit=new[0]['metrics'],
                new_legacy_minus_historical={k:old[0]['metrics'][k]-prior_data[k] for k in old[0]['metrics'] if prior_data and k in prior_data})
            dump(dest/'comparison.json',comparison)
            reports.append(dict(variant=v,**{k:comparison[k] for k in ['decoded_RGB_changed','input_tensor_changed','shape_changed','dtype_changed','predictions_changed']}))
            for arm,result in [('legacy_jpeg',old),('new_png',new)]+([('png_rect_only',third)] if third else []):
                for row in result[0]['classes']:class_rows.append(dict(variant=v,arm=arm,**row))
            device_snapshot(device,dest/'device_after.json',c['hardware_notes'])
            dump(dest/'status.json',dict(status='success',arms=3 if third else 2))
        except Exception:
            message=traceback.format_exc();(dest/'failure.txt').write_text(message,encoding='utf-8');print(message,file=sys.stderr)
            dump(dest/'status.json',dict(status='failed'));failures.append(v)
    write_csv(out/'observed_differences.csv',reports);write_csv(out/'all_class_AP.csv',class_rows)
    dump(out/'status.json',dict(status='partial_or_failed' if failures else 'success',failures=failures,
        no_fixed_conf_re_evaluation=True,no_corruptions=True,no_training=True,test_evaluated=False))
    return failures


def run(c,out,args):
    from detailpair.data import digest
    from detailpair.evaluation import prepare_dataset
    from detailpair.benchmark import DetectorPool,balanced_measure
    device,names,records,models,identities=preflight(c,out,args)
    data,images,manifest=prepare_dataset(records,names,out/'datasets/new_png','val')
    from detailpair.evaluation import array_identity
    from PIL import Image
    import numpy as np
    for row in manifest:
        row['PIL_source_RGB_pixels']=array_identity(np.asarray(Image.open(row['source_image']).convert('RGB')))
        row['PIL_derived_PNG_RGB_pixels']=array_identity(np.asarray(Image.open(row['working_image']).convert('RGB')))
    dump(out/'clean_png_manifest.json',dict(records=manifest,derivation='v0.2.5 PIL RGB -> quantized lossless PNG, clean only'))
    failures=[];timing=None
    if args.mode in {'all','timing-only'}:
        (out/'timing').mkdir(parents=True,exist_ok=True)
        try:
            pool=DetectorPool(models,device,c)
            timing=balanced_measure(pool,images,identities,c,out/'timing')
            dump(out/'timing/status.json',dict(status='success',data_complete=timing['complete'],
                timing_status=timing['timing_status'],comparison_credible=timing['comparison_credible']))
        except Exception:
            message=traceback.format_exc();(out/'timing/failure.txt').write_text(message,encoding='utf-8');print(message,file=sys.stderr)
            dump(out/'timing/status.json',dict(status='failed'));failures.append('timing')
        finally:
            if 'pool' in locals():del pool
    if args.mode in {'all','clean-diagnostic'}:
        diag=out/'clean_diagnostic';diag.mkdir()
        failures+=['clean/'+v for v in clean_diagnostic(c,names,records,data,manifest,diag,device)]
    for e in c['variants']:
        if digest(e['weights'])!=identities[e['variant']]['weights_sha256']:raise RuntimeError('Input weight was modified during run')
    for row in records:
        if digest(row['source_image'])!=row['image_sha256'] or digest(row['source_label'])!=row['label_sha256']:raise RuntimeError('Original source data changed during run')
    dump(out/'status.json',dict(status='partial_or_failed' if failures else 'success',mode=args.mode,failures=failures,
        synthetic_check=args.synthetic_check,weights_read_only=True,test_evaluated=False,no_training=True,
        timing_data_complete=timing['complete'] if timing else None,timing_status=timing['timing_status'] if timing else 'not_completed_or_not_requested',
        comparison_credible=timing['comparison_credible'] if timing else None,
        distinction='Execution success, data completeness and speed credibility are separate. Unstable timing is retained, not automatically retried.'))
    return 1 if failures else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/balanced_recheck.yaml'))
    p.add_argument('--mode',choices=['all','timing-only','clean-diagnostic'],default='all')
    p.add_argument('--synthetic-check',action='store_true')
    p.add_argument('--pack-only',action='store_true')
    args=p.parse_args()
    if args.pack_only:pack_recheck();return 0
    out=ROOT/'runs'/('recheck_'+stamp());out.mkdir(parents=True,exist_ok=False)
    dump(out/'request.json',vars(args));dump(out/'environment.json',environment())
    code=1
    try:
        import yaml
        raw=Path(args.config).read_bytes();c=yaml.safe_load(raw);(out/'frozen_config.yaml').write_bytes(raw)
        dump(out/'config_identity.json',dict(path=args.config,sha256=hashlib.sha256(raw).hexdigest()))
        dump(out/'effective_config.json',c);code=run(c,out,args)
    except BaseException:
        message=traceback.format_exc();(out/'failure.txt').write_text(message,encoding='utf-8');print(message,file=sys.stderr)
        dump(out/'status.json',dict(status='failed',no_training=True,test_evaluated=False))
    finally:
        try:pack_recheck()
        except Exception:
            (out/'pack_failure.txt').write_text(traceback.format_exc());traceback.print_exc();code=1
    print('Recheck directory:',out,flush=True);return code

if __name__=='__main__':sys.exit(main())
