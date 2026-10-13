"""Frozen-weight evaluation only. No training, resume or optimizer operations."""
import argparse, copy, datetime, hashlib, json, os, platform, subprocess, sys, traceback, zipfile
from pathlib import Path
from run import ROOT, stamp, environment
from detailpair.reporting import dump, write_csv


def pack_results():
    exports=ROOT/'exports';exports.mkdir(exist_ok=True)
    target=exports/('DetailPair_eval_results_'+stamp()+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for base in (ROOT/'runs',ROOT/'verification'):
            if not base.exists(): continue
            for f in sorted(base.rglob('*')):
                if not f.is_file():continue
                rel=f.relative_to(ROOT)
                is_subset = any(part.startswith('subset') for part in rel.parts)
                if (is_subset and f.name not in {'selection.json','data.yaml'}) or 'synthetic_data' in rel.parts or 'images' in rel.parts or ('labels' in rel.parts and 'predictions' not in rel.parts):continue
                if 'datasets' in rel.parts or '__pycache__' in rel.parts or f.suffix in {'.pt','.cache','.onnx','.engine','.pyc'}:continue
                z.write(f,rel)
        for folder in ('detailpair','configs','windows','tests'):
            for f in sorted((ROOT/folder).rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':z.write(f,f.relative_to(ROOT))
        for f in ROOT.iterdir():
            if f.is_file() and f.suffix in {'.py','.md','.json','.txt'}:z.write(f,f.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Result ZIP integrity check failed')
    print('Result ZIP:',target,flush=True)
    return target


def validate_config(config, synthetic=False):
    if config.get('protocol_version')!='DetailPair-reuse-eval-1.0':raise ValueError('Unknown protocol')
    if bool(config.get('synthetic_check'))!=synthetic:raise ValueError('Synthetic config requires explicit --synthetic-check; never use it for real results')
    if config['split']!='val':raise ValueError('Default config must be val. Final test requires --final-test and freeze record')
    expected=dict(imgsz=320,batch=1,precision='FP32',conf=.25,ap_conf=.001,nms_iou=.7,match_iou=.5,max_det=300,repeats=3,seed=178)
    if not synthetic:expected.update(warmup=30,expected_images=180,expected_gt=429,device='0')
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError(f'Frozen {k} requires {v!r}, found {config.get(k)!r}')
    if type(config['fuse']) is not bool:raise ValueError('fuse must be Boolean')
    if not synthetic and not config['fuse']:raise ValueError('Frozen production protocol uses fuse=true for both conditions')
    if config['warmup']<1 or config['cpu_threads']<1:raise ValueError('Positive warmup and CPU threads required')
    if [v['variant'] for v in config['variants']]!=['B2','B6'] or any(v['seed']!=0 for v in config['variants']):raise ValueError('Explicit B2/B6 seed0 bindings required')
    if config['corruptions']!=[dict(kind='gaussian',value=v) for v in [.01,.03,.06]]+[dict(kind='impulse',value=.01)]:raise ValueError('All four frozen corruption levels required')


def comparison(results, names, out):
    rows=[];class_rows=[]
    for scenario in ['clean','gaussian_0.01','gaussian_0.03','gaussian_0.06','impulse_0.01']:
        a=results.get(('B2',scenario));b=results.get(('B6',scenario))
        if not a or not b:continue
        for variant,r in [('B2',a),('B6',b)]:
            clean=results.get((variant,'clean'))
            metrics=r['ap']['metrics'];base=clean['ap']['metrics'] if clean else {}
            rows.append(dict(scenario=scenario,variant=variant,AP50=metrics['metrics/mAP50(B)'],AP50_95=metrics['metrics/mAP50-95(B)'],
                AP50_drop_from_own_clean=base.get('metrics/mAP50(B)',0)-metrics['metrics/mAP50(B)'] if clean else None,
                AP50_95_drop_from_own_clean=base.get('metrics/mAP50-95(B)',0)-metrics['metrics/mAP50-95(B)'] if clean else None,
                B6_minus_B2_AP50=b['ap']['metrics']['metrics/mAP50(B)']-a['ap']['metrics']['metrics/mAP50(B)'],
                B6_minus_B2_AP50_95=b['ap']['metrics']['metrics/mAP50-95(B)']-a['ap']['metrics']['metrics/mAP50-95(B)']))
            for k,name in enumerate(names):
                rr=r['fixed']['classes'][k];aa=a['fixed']['classes'][k];bb=b['fixed']['classes'][k]
                ap=r['ap']['classes'][k];ca=clean['ap']['classes'][k] if clean else None
                cr=clean['fixed']['classes'][k] if clean else None
                subtract=lambda x,y:x-y if x is not None and y is not None else None
                class_rows.append(dict(scenario=scenario,variant=variant,class_id=k,name=name,TP=rr['TP'],FP=rr['FP'],FN=rr['FN'],GT=rr['GT'],
                    Precision=rr['Precision'],Recall=rr['Recall'],AP50=ap['AP50'],AP50_95=ap['AP50_95'],
                    Recall_drop_from_own_clean=subtract(cr['Recall'],rr['Recall']) if cr else None,
                    Precision_drop_from_own_clean=subtract(cr['Precision'],rr['Precision']) if cr else None,
                    AP50_drop_from_own_clean=subtract(ca['AP50'],ap['AP50']) if ca else None,
                    AP50_95_drop_from_own_clean=subtract(ca['AP50_95'],ap['AP50_95']) if ca else None,
                    B6_minus_B2_Recall=subtract(bb['Recall'],aa['Recall']),B6_minus_B2_Precision=subtract(bb['Precision'],aa['Precision']),
                    B6_minus_B2_AP50=subtract(b['ap']['classes'][k]['AP50'],a['ap']['classes'][k]['AP50']),
                    B6_minus_B2_AP50_95=subtract(b['ap']['classes'][k]['AP50_95'],a['ap']['classes'][k]['AP50_95'])))
    write_csv(out/'comparison.csv',rows);write_csv(out/'class_comparison.csv',class_rows)
    dump(out/'comparison.json',dict(rows=rows,classes=class_rows,scope='single-seed exploratory comparison; no significance or novelty claim; all severities reported'))


def run(config, out, config_sha, args):
    import yaml,torch
    import ultralytics
    from ultralytics import settings
    from detailpair.data import digest
    from detailpair.evaluation import read_records,verify_selection,validate_checkpoint,prepare_dataset,official_ap,fixed_evaluate
    from detailpair.benchmark import HostDetector,gpu_guard,measure
    settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb')})
    validate_config(config,args.synthetic_check)
    if args.final_test:
        if args.synthetic_check or not args.freeze_record:raise ValueError('Final test needs real frozen-method record')
        config['split']='test'
    elif args.freeze_record:raise ValueError('--freeze-record only applies to explicit --final-test')
    if torch.__version__.split('+')[0]!='2.5.1' or ultralytics.__version__!='8.3.221':raise RuntimeError('Pinned Torch2.5.1/Ultralytics8.3.221 required; no automatic upgrade')
    import torchvision
    if torchvision.__version__.split('+')[0]!='0.20.1':raise RuntimeError('Pinned torchvision0.20.1 required')
    if args.synthetic_check:
        if config['device']!='cpu':raise ValueError('Synthetic acceptance workflow uses explicit CPU')
        device=torch.device('cpu')
    else:
        if not torch.cuda.is_available() or torch.version.cuda!='12.1':raise RuntimeError('Working CUDA12.1 stack required; no silent CPU fallback')
        device=torch.device('cuda:0')
    torch.set_num_threads(config['cpu_threads'])
    dump(out/'effective_config.json',config)
    gpu_guard(device,out/'gpu_processes_start.json')
    identities={};models={};selections={}
    names,records=read_records(config['data'],config['split'])
    gt_count=sum(len(r['gt']) for r in records)
    if len(records)!=config['expected_images'] or (not args.final_test and gt_count!=config['expected_gt']):raise ValueError(f'Expected split cardinality violated: images={len(records)}, GT={gt_count}')
    dump(out/'source_manifest.json',dict(names=names,split=config['split'],images=len(records),GT=gt_count,records=records))
    for entry in config['variants']:
        v=entry['variant']
        selection=verify_selection(dict(entry,**({} if args.synthetic_check else {'expected_train_images':1438})),config['data']);selections[v]=selection
        dump(out/(v+'_selection_identity.json'),selection)
        models[v],identities[v]=validate_checkpoint(entry,config,names)
        dump(out/(v+'_checkpoint_identity.json'),identities[v])
    if selections['B2']['source_identity']!=selections['B6']['source_identity']:raise ValueError('Conditions were trained with different source selection')
    if args.final_test:
        freeze=json.loads(Path(args.freeze_record).read_text(encoding='utf-8'))
        if freeze.get('config_sha256')!=config_sha or freeze.get('split')!='test' or freeze.get('method_and_thresholds_frozen') is not True:
            raise ValueError('Final test freeze record does not bind this frozen protocol')
        if freeze.get('weights_sha256')!={v:i['weights_sha256'] for v,i in identities.items()}:
            raise ValueError('Final test weights differ from freeze record')
        # Canonical source records include actual held-out image/label hashes and GT.
        data_sha=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
        if freeze.get('test_records_sha256')!=data_sha:raise ValueError('Final test data differ from freeze record; canonical hash='+data_sha)
        dump(out/'final_test_freeze.json',freeze)
    scenarios=[('clean',None)]+[(f"{c['kind']}_{c['value']}",c) for c in config['corruptions']]
    data_sets={}
    for scenario,corruption in scenarios:
        data,images,manifest=prepare_dataset(records,names,out/'datasets'/scenario,config['split'],corruption,config['seed'])
        dump(out/(scenario+'_derived_manifest.json'),dict(records=manifest,shared_by=['B2','B6'],seed=config['seed'],
            corruption=corruption,rng_reset_each_severity=True,quantization='RGB [0,1], clip, numpy.round -> uint8 PNG'))
        data_sets[scenario]=(data,images,manifest)
    failures=[];results={};latencies={}
    # Measure one model at a time; original registered checkpoints stay on CPU here.
    for entry in config['variants']:
        v=entry['variant'];dest=out/v/'timing';dest.mkdir(parents=True)
        try:
            detector=HostDetector(copy.deepcopy(models[v].model).float().eval(),device,config)
            latency=measure(detector,data_sets['clean'][1],identities[v],config,dest);latencies[v]=latency
            dump(dest/'status.json',dict(status='success'))
        except Exception:
            text=traceback.format_exc();(dest/'failure.txt').write_text(text);print(text,file=sys.stderr)
            failures.append(v+'/timing');dump(dest/'status.json',dict(status='failed'))
        finally:
            if 'detector' in locals():del detector
            if device.type=='cuda':torch.cuda.empty_cache()
    for scenario,_ in scenarios:
        data,images,manifest=data_sets[scenario]
        expected=[(r['derived_image_sha256'],r['derived_label_sha256']) for r in manifest]
        for entry in config['variants']:
            v=entry['variant'];dest=out/v/scenario;dest.mkdir(parents=True)
            try:
                gpu_guard(device,dest/'gpu_processes_before_evaluation.json')
                if expected!=[(digest(r['working_image']),digest(Path(r['working_image']).parents[2]/'labels'/config['split']/(r['image_id']+'.txt'))) for r in manifest]:
                    raise ValueError('Shared derived dataset changed')
                # Independent checkpoint reload for official AP, never substitute old metrics.
                yolo,identity=validate_checkpoint(entry,config,names)
                ap=official_ap(yolo,data,names,config,dest)
                ap_predictions=dest/'official/ap_predictions.jsonl'
                if not ap_predictions.is_file() or len(ap_predictions.read_text().splitlines())!=len(images):
                    raise RuntimeError('Official AP prediction records incomplete')
                if set(ap['metrics']) == set():raise RuntimeError('Official AP evaluation produced no metrics')
                detector=HostDetector(copy.deepcopy(models[v].model).float().eval(),device,config)
                fixed=fixed_evaluate(detector,images,names,config,dest)
                results[v,scenario]=dict(ap=ap,fixed=fixed)
                if scenario=='clean':
                    old=Path(entry['prior_metrics'])
                    prior=json.loads(old.read_text()) if old.is_file() else None
                    delta={k:float(ap['metrics'][k])-float(prior[k]) for k in ap['metrics'] if prior and k in prior and isinstance(prior[k],(int,float))}
                    dump(dest/'prior_metric_difference.json',dict(path=str(old),status='compared' if prior else 'missing_original_metrics',
                        old=prior,new=ap['metrics'],new_minus_old=delta,warning='AP uses frozen new validator settings; differences retained without replacing results'))
                dump(dest/'status.json',dict(status='success',shared_manifest_sha256=digest(out/(scenario+'_derived_manifest.json'))))
            except Exception:
                text=traceback.format_exc();(dest/'failure.txt').write_text(text);print(text,file=sys.stderr)
                failures.append(v+'/'+scenario);dump(dest/'status.json',dict(status='failed'))
            finally:
                if 'detector' in locals():del detector
                if 'yolo' in locals():del yolo
                if device.type=='cuda':torch.cuda.empty_cache()
    comparison(results,names,out)
    if len(latencies)==2 and all((v,'clean') in results for v in ('B2','B6')):
        table=[]
        for v in ('B2','B6'):
            table.append(dict(variant=v,AP50=results[v,'clean']['ap']['metrics']['metrics/mAP50(B)'],
                AP50_95=results[v,'clean']['ap']['metrics']['metrics/mAP50-95(B)'],**latencies[v]['overall']))
        write_csv(out/'accuracy_latency.csv',table)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for r in table:ax.scatter(r['mean_ms'],100*r['AP50_95']);ax.annotate(r['variant'],(r['mean_ms'],100*r['AP50_95']))
        ax.set(xlabel='Decoded host image -> host detections mean latency (ms)',ylabel='Official AP50:95 (%)',
            title=f"{torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or 'CPU'}; {config['split']}; 320x320; batch1 FP32\nTwo-model comparison, not a Pareto frontier")
        fig.tight_layout();fig.savefig(out/'accuracy_latency.png');plt.close(fig)
    for entry in config['variants']:
        if digest(entry['weights'])!=identities[entry['variant']]['weights_sha256']:raise RuntimeError('Input weight changed during evaluation')
    for row in records:
        if digest(row['source_image'])!=row['image_sha256'] or digest(row['source_label'])!=row['label_sha256']:raise RuntimeError('Source data changed during evaluation')
    dump(out/'status.json',dict(status='partial_or_failed' if failures else 'success',failures=failures,
        expected_latency_samples=len(records)*config['repeats']*2,
        actual_latency_samples=sum(s['overall']['n'] for s in latencies.values()),
        successful_evaluations=len(results),expected_evaluations=10,synthetic_check=args.synthetic_check,
        weights_read_only=True,no_training=True))
    return 1 if failures else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/realtime_comparison.yaml'))
    parser.add_argument('--pack-only',action='store_true')
    parser.add_argument('--synthetic-check',action='store_true',help='explicit CPU plumbing tests only; never use on real experiments')
    parser.add_argument('--final-test',action='store_true')
    parser.add_argument('--freeze-record')
    args=parser.parse_args()
    if args.pack_only:pack_results();return 0
    out=ROOT/'runs'/('realtime_eval_'+stamp());out.mkdir(parents=True,exist_ok=False)
    dump(out/'request.json',vars(args));dump(out/'environment.json',environment())
    code=1
    try:
        import yaml
        raw=Path(args.config).read_bytes();config=yaml.safe_load(raw)
        (out/'frozen_config.yaml').write_bytes(raw)
        config_sha=hashlib.sha256(raw).hexdigest();dump(out/'config_identity.json',dict(path=args.config,sha256=config_sha))
        code=run(config,out,config_sha,args)
    except Exception:
        message=traceback.format_exc();(out/'failure.txt').write_text(message,encoding='utf-8');print(message,file=sys.stderr)
        dump(out/'status.json',dict(status='failed',no_training=True))
    finally:
        try:pack_results()
        except Exception:
            (out/'pack_failure.txt').write_text(traceback.format_exc());traceback.print_exc();code=1
    print('Evaluation directory:',out,flush=True)
    return code

if __name__=='__main__':sys.exit(main())
