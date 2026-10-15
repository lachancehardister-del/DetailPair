"""Fixed train-only endpoint gradients; no parameter updates, epochs or test evaluation."""
import argparse,datetime,hashlib,json,math,os,sys,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('WANDB_DISABLED','true')
from run import environment,stamp
from detailpair.reporting import dump,write_csv

EXPECTED_HASHES={'B2':'c234ae47e6d8257543a038abc3ea0714d0f081abf2ce5e9dbe0dcd9c73a2c21b',
                'B6':'2a5f5dd7152b28a3a3952f96fbeca409e231e07de2a06a6b5686d7a01c6bc324'}


def validate_config(c,synthetic):
    expected=dict(protocol_version='DetailPair-mechanism-local-1.0',device='0',expected_train_images=1438,selected_images=64,
        selection_seed=178,batch=2,imgsz=320,precision='FP32',cpu_threads=2,perturbation_seed=100003,noise_std=.03,brightness=.05,
        consistency_weight=.05,amplitude_weight=.02,background_weight=.1,consistency_rtol=1e-4,consistency_atol=1e-5)
    if bool(c.get('synthetic_check'))!=synthetic:raise ValueError('Synthetic CPU configuration requires explicit --synthetic-check')
    if synthetic:
        expected.update(device='cpu',expected_train_images=c['expected_train_images'],selected_images=c['selected_images'])
        if c['selected_images']>c['expected_train_images'] or c['selected_images']%2 or c['selected_images']<2:raise ValueError('Positive complete batch2 CPU fixture required')
    for k,v in expected.items():
        if c.get(k)!=v:raise ValueError(f'Frozen {k} must be {v!r}; got {c.get(k)!r}')
    if [e['variant'] for e in c['variants']]!=['B2','B6'] or any(e['seed']!=0 for e in c['variants']):raise ValueError('B2/B6 seed0 identity bindings required')
    if not synthetic:
        for e in c['variants']:
            if e['expected_sha256']!=EXPECTED_HASHES[e['variant']]:raise ValueError('Frozen checkpoint identity changed')


def pack_results():
    exports=ROOT/'exports';exports.mkdir(exist_ok=True)
    target=exports/('DetailPair_mechanism_results_'+stamp()+'.zip')
    excluded={'.pt','.npy','.npz','.pth','.cache','.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.pyc'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['runs','verification','detailpair','configs','windows','tests']:
            base=ROOT/folder
            if not base.exists():continue
            for p in sorted(base.rglob('*')):
                if not p.is_file():continue
                rel=p.relative_to(ROOT)
                if p.suffix.lower() in excluded or {'datasets','images','labels','__pycache__'}&set(rel.parts):continue
                z.write(p,rel)
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in {'.md','.py','.json','.txt'} and p.name!='ARTIFACT_MANIFEST.json':z.write(p,p.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Result ZIP CRC failure')
    print('Result ZIP:',target,flush=True);return target


def summarize_batches(records,expected):
    import numpy as np
    numeric={};states={};nulls={};cosines={}
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path+'.'+k if path else k)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,path+f'[{i}]')
        elif value is None:nulls[path]=nulls.get(path,0)+1
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            if not math.isfinite(value):raise ValueError('Nonfinite raw statistic '+path)
            numeric.setdefault(path,[]).append(value)
            if path.endswith('.cos'):cosines.setdefault(path,[]).append(value)
        elif isinstance(value,str) and path.endswith('.status'):states.setdefault(path,{})[value]=states.setdefault(path,{}).get(value,0)+1
    good=[r for r in records if r['status']=='success']
    for row in good:
        for section in ['unweighted','weighted','gradient_groups','gate_subbands','proxy_regions']:
            walk(row[section],section)
    metrics={k:dict(valid_batches=len(v),mean=float(np.mean(v)),median=float(np.median(v)),min=float(min(v)),max=float(max(v)),
                   p05=float(np.percentile(v,5)),p95=float(np.percentile(v,95))) for k,v in numeric.items()}
    cos={k:dict(valid_batches=len(v),negative_batches=sum(x<0 for x in v),negative_fraction=sum(x<0 for x in v)/len(v)) for k,v in cosines.items()}
    return dict(expected_batches=expected,recorded_batches=len(records),successful_batches=len(good),failed_batches=sum(r['status']=='failed' for r in records),
        not_run_batches=sum(r['status']=='not_run' for r in records),distributions=metrics,gradient_and_region_status_counts=states,
        undefined_counts=nulls,negative_cosine_fractions=cos,
        empty_foreground_images=sum(sum(x==0 for x in r['proxy_regions']['fg_pixels_per_image']) for r in good),
        empty_background_images=sum(sum(x==0 for x in r['proxy_regions']['bg_pixels_per_image']) for r in good),
        scope='All planned batches retained. Correlated local observations, not independent training seeds or a significance test.')


def run(c,out,args):
    import torch,torchvision,ultralytics
    from ultralytics import settings
    from detailpair.data import digest
    from detailpair.evaluation import validate_checkpoint
    from detailpair.mechanism_data import choose_training,load_batch
    from detailpair.mechanism import prepare_loss_mode,analyze_batch,state_identity
    validate_config(c,args.synthetic_check)
    for name,version,expected in [('torch',torch.__version__,'2.5.1'),('torchvision',torchvision.__version__,'0.20.1'),('ultralytics',ultralytics.__version__,'8.3.221')]:
        if version.split('+')[0]!=expected:raise RuntimeError(f'{name} requires {expected}, got {version}; no installation/upgrades')
    settings.update({k:False for k in ['sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb']})
    if args.synthetic_check:device=torch.device('cpu')
    else:
        if not torch.cuda.is_available() or torch.version.cuda!='12.1':raise RuntimeError('Working CUDA12.1 GPU required; no CPU fallback')
        device=torch.device('cuda:0')
        if c['expected_gpu_substring'] not in torch.cuda.get_device_name(device):raise RuntimeError('Unexpected GPU identity')
    torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    names,selected,selection=choose_training(c['data'],c['variants'],c['selected_images'],c['selection_seed'],c['expected_train_images'])
    dump(out/'selection.json',dict(**selection,records=[{k:v for k,v in r.items() if k!='annotations'} for r in selected]))
    dump(out/'execution_policy.json',dict(device=str(device),precision='FP32',deterministic_algorithms=True,TF32=False,
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],graph_policy='One batch graph at a time; autograd.grad, no .backward or optimizer; original and rebuilt routes compare each batch',
        perturbation_policy='seed100003+batch_index, both models same images and exact local generator replay; not historical batch replay',
        state_policy='Unfused endpoint weights, training loss path, all BN/random modules eval, layer1 only requires_grad'))
    batches=len(selected)//c['batch'];results={};failures=[];model_summaries={}
    for entry in c['variants']:
        v=entry['variant'];dest=out/v;dest.mkdir();records=[];initial=None;model=None
        try:
            path=Path(entry['weights'])
            if not path.is_file():raise FileNotFoundError('Missing local weight '+str(path))
            if digest(path)!=entry['expected_sha256']:raise ValueError(v+' weight SHA256 differs')
            yolo,identity=validate_checkpoint(entry,c,names)
            model=yolo.model.float().to(device)
            named,groups,mode=prepare_loss_mode(model,identity['train_args']);initial=state_identity(model)
            dump(dest/'checkpoint_identity.json',identity);dump(dest/'loss_mode.json',mode);dump(dest/'state_before.json',initial)
            with (dest/'batches.jsonl').open('w',encoding='utf-8') as stream:
                aborted=False
                for step in range(batches):
                    chosen=selected[step*c['batch']:(step+1)*c['batch']]
                    row=dict(status='not_run',batch_index=step,variant=v,image_ids=[r['image_id'] for r in chosen],intended_perturbation_seed=c['perturbation_seed']+step)
                    if aborted:row['reason']='Earlier condition failure; no automatic retry/reselection'
                    else:
                        print(v,'local diagnostic batch',step+1,'/',batches,flush=True)
                        try:
                            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                            batch,inputs=load_batch(chosen,device,c['imgsz']);row['input']=inputs
                            row.update(analyze_batch(model,batch,step,named,groups,c['consistency_rtol'],c['consistency_atol']))
                            row.update(status='success',input=inputs,memory=dict(allocated_bytes=torch.cuda.memory_allocated(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else dict(status='CPU fixture'))
                            del batch
                        except Exception:
                            row.update(status='failed',failure=traceback.format_exc(),state_unchanged='failed_or_unknown')
                            (dest/f'failure_batch_{step:02d}.txt').write_text(row['failure'],encoding='utf-8');print(row['failure'],file=sys.stderr)
                            aborted=True
                    records.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            after=state_identity(model);dump(dest/'state_after.json',after)
            if initial!=after:raise AssertionError('Final parameter/BN identity changed')
            if digest(entry['weights'])!=entry['expected_sha256']:raise AssertionError('Disk weight changed')
            summary=summarize_batches(records,batches);dump(dest/'summary.json',summary)
            state='success' if summary['successful_batches']==batches else 'partial_or_failed'
            dump(dest/'status.json',dict(status=state,parameters_and_BN_unchanged=True,disk_weight_unchanged=True,
                batches_success=summary['successful_batches'],batches_expected=batches,scope='Frozen-BN endpoint local gradients, not B5 ablation or historical training'))
            if state!='success':failures.append(v)
            results[v]=records;model_summaries[v]=summary
        except Exception:
            text=traceback.format_exc();(dest/'failure.txt').write_text(text,encoding='utf-8');print(text,file=sys.stderr)
            for step in range(len(records),batches):records.append(dict(status='not_run',batch_index=step,variant=v,reason='Condition setup/state failure'))
            dump(dest/'incomplete_batches.json',records);dump(dest/'summary.json',summarize_batches(records,batches))
            dump(dest/'status.json',dict(status='failed',state_unchanged='unknown_or_failed',batches_expected=batches))
            results[v]=records;failures.append(v)
        finally:
            if model is not None:del model
            if 'yolo' in locals():del yolo
            if device.type=='cuda':
                try:torch.cuda.empty_cache()
                except Exception:(dest/'cuda_cleanup_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    replay=[]
    for step in range(batches):
        a,b=results['B2'][step],results['B6'][step]
        if a['status']=='success' and b['status']=='success':
            shared=a['perturbation']==b['perturbation'] and a['input']==b['input']
            replay.append(dict(batch_index=step,status='passed' if shared else 'failed'))
            if not shared:failures.append('paired_replay_'+str(step))
        else:replay.append(dict(batch_index=step,status='not_verified_due_to_failure'))
    dump(out/'paired_input_noise_replay.json',replay)
    for row in selected:
        if digest(row['source_image'])!=row['image_sha256'] or digest(row['source_label'])!=row['label_sha256']:raise AssertionError('Original selected data changed')
    verified_states={v:json.loads((out/v/'status.json').read_text()) for v in ['B2','B6']}
    dump(out/'status.json',dict(status='partial_or_failed' if failures else 'success',failures=failures,synthetic_check=args.synthetic_check,
        expected_batches_per_model=batches,selected_train_images=len(selected),no_optimizer=True,no_training=True,validation_evaluated=False,timing_performed=False,parameter_updates_requested=False,
        parameters_and_BN_verified_unchanged=all(r.get('parameters_and_BN_unchanged') is True for r in verified_states.values()),
        disk_weights_verified_unchanged=all(r.get('disk_weight_unchanged') is True for r in verified_states.values()),test_evaluated=False,
        interpretation='Local endpoint/frozen-BN evidence only; no causal AP attribution, no automatic training or new mechanism'))
    return 1 if failures else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/mechanism_diagnosis.yaml'))
    p.add_argument('--pack-only',action='store_true');p.add_argument('--synthetic-check',action='store_true')
    args=p.parse_args()
    if args.pack_only:pack_results();return 0
    out=ROOT/'runs'/('mechanism_'+stamp());out.mkdir(parents=True,exist_ok=False)
    dump(out/'request.json',vars(args));dump(out/'environment.json',environment());code=1
    try:
        import yaml
        raw=Path(args.config).read_bytes();(out/'frozen_config.yaml').write_bytes(raw);c=yaml.safe_load(raw)
        dump(out/'config_identity.json',dict(path=args.config,sha256=hashlib.sha256(raw).hexdigest()))
        code=run(c,out,args)
    except BaseException:
        text=traceback.format_exc();(out/'failure.txt').write_text(text,encoding='utf-8');print(text,file=sys.stderr)
        dump(out/'status.json',dict(status='failed',test_evaluated=False,no_optimizer=True,no_training=True))
    finally:
        try:pack_results()
        except Exception:(out/'pack_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc();code=1
    print('Mechanism directory:',out,flush=True);return code

if __name__=='__main__':sys.exit(main())
