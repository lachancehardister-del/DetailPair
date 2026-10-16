"""BoxRoute G0 engineering by default; G1 is a separate review-gated entry, never chained."""
import argparse,copy,datetime,hashlib,json,os,subprocess,sys,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True);os.environ.setdefault('WANDB_DISABLED','true')
from boxroute.identity import dump,file_sha256,state_sha256


def stamp():return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def pack_results():
    directory=ROOT/'exports';directory.mkdir(exist_ok=True);target=directory/('BoxRoute_results_'+stamp()+'.zip')
    skip_ext={'.pt','.pth','.npy','.npz','.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff','.pyc','.cache'}
    skip_dirs={'datasets','images','labels','private_data','initial_models','__pycache__','local_settings'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['runs','verification','boxroute','legacy_detailpair','configs','tests','windows','sources']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix.lower() not in skip_ext and not skip_dirs&set(p.relative_to(ROOT).parts):z.write(p,p.relative_to(ROOT))
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in {'.md','.yaml','.json','.py','.txt'} and p.name!='ARTIFACT_MANIFEST.json':z.write(p,p.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Result archive CRC failed')
    print('Result ZIP:',target,flush=True);return target


def local_paths(filename=None):
    import yaml
    path=Path(filename) if filename else ROOT/'configs/local_paths.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8-sig'))


def eval_config(device='0'):
    return dict(device=device,split='val',imgsz=320,batch=1,fuse=True,conf=.25,ap_conf=.001,nms_iou=.7,match_iou=.5,max_det=300)


def frozen_train_args(protocol,data,output,epochs,device):
    c=protocol['G1'];args={k:c[k] for k in ['patience','resume','save_period','optimizer','lr0','lrf','nbs','batch','imgsz','rect','amp','workers','deterministic','cache','compile']}
    args.update(c['augment']);args.update(model='yolo11n.yaml',data=str(data),project=str(output),name='training',epochs=epochs,device=device,
        seed=0,pretrained=True,plots=False,save=True,exist_ok=False,multi_scale=False,val=True)
    return args


def worker(args):
    import torch
    from boxroute.trainer import SquareTrainer
    request=json.loads(Path(args.worker_request).read_text());out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
    from boxroute.config import preflight,load_protocol
    try:
        preflight(out,args.synthetic_check);protocol=load_protocol()
        source=Path(request['initial']);model=torch.load(source,map_location='cpu',weights_only=False)['model']
        if file_sha256(source)!=request['initial_sha256']:raise ValueError('Initial condition file changed')
        trainer=SquareTrainer(initial_model=model,overrides=frozen_train_args(protocol,request['data'],out,request['epochs'],'cpu' if args.synthetic_check else '0'))
        trainer.train()
        endpoint=out/'training/weights'/('matched_epoch'+str(request['epochs'])+'_unstripped.pt')
        from boxroute.evaluate import validate_endpoint
        identity=validate_endpoint(endpoint,request['variant'],request['epochs'],request['names'])
        dump(out/'worker_status.json',dict(status='success',endpoint=identity,optimizer_steps=trainer.actual_optimizer_steps,synthetic_check=args.synthetic_check))
        return 0
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');dump(out/'worker_status.json',dict(status='failed',synthetic_check=args.synthetic_check));traceback.print_exc();return 1


def child_train(variant,model_path,data,out,epochs,names,synthetic):
    request=dict(variant=variant,initial=str(model_path),initial_sha256=file_sha256(model_path),data=str(data),output=str(out),epochs=epochs,names=names)
    request_path=out.parent/(variant+'_request.json');dump(request_path,request)
    cmd=[sys.executable,str(ROOT/'run_boxroute.py'),'worker','--worker-request',str(request_path)]
    if synthetic:cmd.append('--synthetic-check')
    # Terminal inherited unchanged. Never PIPE/tee native training progress.
    result=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
    dump(out.parent/(variant+'_exit.json'),dict(command=cmd,returncode=result.returncode))
    if result.returncode:raise RuntimeError(variant+' child failed; preserve partial and stop suite, exit '+str(result.returncode))
    return out/'training/weights'/('matched_epoch'+str(epochs)+'_unstripped.pt')


def balanced_timing(models,manifest,count,out,device,synthetic=False,require_equal=False,checkpoint_paths=None):
    import cv2,numpy as np,torch
    from boxroute.data import choose
    from boxroute.benchmark import DetectorPool,balanced_measure
    rows,selection=choose(manifest,'val',count);images=[]
    for r in rows:
        image=cv2.imread(r['working_image'])
        if image is None:raise ValueError('Cannot decode timing source copy')
        images.append(dict(image_id=r['image_id'],bgr=image))
    cfg=eval_config('cpu' if synthetic else '0');cfg.update(block_order=['N','G','G','N','G','N','N','G'],common_warmup_seconds=30.,block_warmup_iterations=200,drift_threshold=1.1,
        hardware_notes=dict(machine='user machine',power_mode='unknown',plugged_in='unknown'),expected_images=count)
    if synthetic:cfg.update(common_warmup_seconds=0.02,block_warmup_iterations=1)
    pool=DetectorPool({v:type('Holder',(),dict(model=models[v]))() for v in ['N','G']},device,cfg)
    if require_equal:
        predictions={}
        for v in ['N','G']:
            with pool.lease(v) as detector:predictions[v]=[detector.predict(row['bgr']) for row in images]
        if any(a.shape!=b.shape or not np.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(predictions['N'],predictions['G'])):raise RuntimeError('Initial gamma0 NMS outputs differ')
    identities={}
    for v in ['N','G']:
        file=Path(checkpoint_paths[v]) if checkpoint_paths else out.parent/'initial_models'/(v+'.pt')
        kind='checkpoint_file_sha256' if file.is_file() else 'model_state_sha256_synthetic_only'
        identities[v]=dict(variant=v,seed=0,run_name='initial_gamma0' if require_equal else 'completed60_EMA',
            weights_sha256=file_sha256(file) if file.is_file() else state_sha256(models[v].state_dict()),
            weights_identity_kind=kind,model_state_sha256=state_sha256(models[v].state_dict()))
    dump(out.parent/(out.name+'_weight_identities.json'),identities)
    dump(out.parent/(out.name+'_selection.json'),selection)
    result=balanced_measure(pool,images,identities,cfg,out);dump(out/'context.json',dict(synthetic_check=synthetic,initial_gamma_zero=require_equal,final_trained_speed=not require_equal and not synthetic,selection=selection))
    return result


def code_identity():
    files=[ROOT/'run_boxroute.py',ROOT/'legacy_detailpair/model.py',*sorted((ROOT/'boxroute').glob('*.py'))]
    identities={p.relative_to(ROOT).as_posix():file_sha256(p) for p in files}
    return dict(files=identities,sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest())


def approve_G1(path):
    if not path:raise ValueError('G1 requires an explicit Work-reviewed G0 record; default stops after G0')
    record=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if record.get('approved_by_Work') is not True or record.get('stage')!='G1':raise ValueError('Work approval record does not authorize G1')
    g0=Path(record['G0_run']);status=json.loads((g0/'status.json').read_text());g0_protocol=json.loads((g0/'protocol_identity.json').read_text())
    if status.get('status')!='G0_passed_return_to_Work' or status.get('synthetic_check') or record.get('G0_status_sha256')!=file_sha256(g0/'status.json'):raise ValueError('Approval is not bound to a passed real G0')
    if g0_protocol['sha256']!=file_sha256(ROOT/'protocol.yaml'):raise ValueError('G0 protocol differs from frozen G1')
    if json.loads((g0/'code_identity.json').read_text())['sha256']!=code_identity()['sha256']:raise ValueError('G0 and G1 implementation identities differ')
    return record


def run_stage(args,out):
    import torch
    from boxroute.config import load_protocol,preflight
    from boxroute.data import prepare_private,shape_statistics,subset_spec,verify_sources
    from boxroute.model import make_common,build_conditions
    from boxroute.checks import validate_models,verify_PKS_and_geometry
    from boxroute.native_equivalence import run_native_equivalence
    protocol=load_protocol();dump(out/'code_identity.json',code_identity());dump(out/'protocol_identity.json',dict(sha256=file_sha256(ROOT/'protocol.yaml'),frozen_protocol=protocol))
    if args.stage=='g1':
        review=approve_G1(args.Work_review);dump(out/'Work_review.json',review)
        if args.synthetic_check:raise ValueError('No CPU G1 execution; engineering integration only runs G0')
    device=preflight(out,args.synthetic_check)
    if args.stage=='env':dump(out/'status.json',dict(status='environment_checked',synthetic_check=args.synthetic_check));return 0
    paths=local_paths(args.paths);manifest=None
    phase_names=['data','engineering','flow_S','flow_W','flow_U','flow_G','flow_N','timing']
    if args.stage=='g1':phase_names+=['final_evaluation','decision']
    phases={name:'not_run' for name in phase_names};dump(out/'phases.json',phases)
    def phase(name,status):phases[name]=status;dump(out/'phases.json',phases)
    try:
        phase('data','running');expected=(paths.get('synthetic_train_images',4),paths.get('synthetic_val_images',2)) if args.synthetic_check else (1438,180)
        private,manifest=prepare_private(paths['data'],out/'private_data',expected=expected,selections=paths.get('historical_selections',[]))
        dump(out/'source_manifest.json',manifest);stats=shape_statistics(manifest);dump(out/'train_shape_statistics.json',stats)
        if not args.synthetic_check and sum(len(r['boxes']) for r in manifest['records']['val'])!=429:raise ValueError('Frozen val429GT differs')
        if args.stage=='g1':
            old=json.loads((Path(review['G0_run'])/'source_manifest.json').read_text())
            fingerprint=lambda m:{split:[(r['image_sha256'],r['label_sha256']) for r in m['records'][split]] for split in ['train','val']}
            if manifest['names']!=old['names'] or fingerprint(manifest)!=fingerprint(old):raise ValueError('G0/G1 frozen source data differ')
        phase('data','success')
        if stats['status']=='blocked_shape_support' and not args.synthetic_check:
            dump(out/'status.json',dict(status='blocked_shape_support',synthetic_check=False,automatic_G1=False));return 2
        native,initial=make_common(ROOT/'sources/official_yolo11n_v830.pt',manifest['names']);models=build_conditions(native);dump(out/'initialization.json',initial)
        if args.stage=='g1' and initial['common_nc6_state_sha256']!=json.loads((Path(review['G0_run'])/'initialization.json').read_text())['common_nc6_state_sha256']:raise ValueError('Common initialization differs from reviewed G0')
        init_dir=out/'initial_models';init_dir.mkdir()
        for v,m in models.items():torch.save(dict(model=m),init_dir/(v+'.pt'))
        phase('engineering','running')
        if args.stage=='g0':
            dump(out/'native_B0_equivalence.json',run_native_equivalence(str(device)))
        dump(out/'PKS_geometry_checks.json',verify_PKS_and_geometry(device))
        checks=validate_models(models,device,out);phase('engineering','success')
        if args.stage=='g0':
            train_n=paths.get('synthetic_train_images',4) if args.synthetic_check else 64;val_n=paths.get('synthetic_val_images',2) if args.synthetic_check else 16
            flow_data,selection=subset_spec(manifest,out/'flow_selection',train_n,val_n);dump(out/'G0_selection.json',selection);epochs=2
        else:flow_data=private;epochs=60
        endpoints={}
        if args.stage=='g1' or args.flow_check:
            for variant in ['S','W','U','G','N']:
                if device.type=='cuda':torch.cuda.empty_cache()
                phase('flow_'+variant,'running')
                endpoints[variant]=child_train(variant,init_dir/(variant+'.pt'),flow_data,out/variant,epochs,manifest['names'],args.synthetic_check)
                phase('flow_'+variant,'success')
        timing_models=models
        if args.stage=='g1':
            from ultralytics import YOLO
            timing_models={v:YOLO(str(endpoints[v])).model.float().eval() for v in ['N','G']}
        phase('timing','running');count=paths.get('synthetic_val_images',2) if args.synthetic_check else (48 if args.stage=='g0' else 180)
        if args.synthetic_check and count%3:count=0 # Synthetic timing uses repeated procedural rows only when explicitly requested below.
        if args.synthetic_check:
            dump(out/'timing_skipped.json',dict(status='not_run_synthetic_fixture',reason='No GPU speed claim; separate balanced-statistics checks cover timing bookkeeping'));timing=None;phase('timing','not_run_synthetic_fixture')
        else:
            timing=balanced_timing(timing_models,manifest,count,out/'timing',device,require_equal=args.stage=='g0',checkpoint_paths=endpoints if args.stage=='g1' else None);phase('timing','success')
        if args.stage=='g1':
            from boxroute.evaluate import evaluate_suite
            from boxroute.decisions import G1_report
            phase('final_evaluation','running');results=evaluate_suite(endpoints,manifest,eval_config(),out/'evaluation',60);phase('final_evaluation','success')
            phase('decision','running');status=G1_report(results,checks['overhead'],timing,out);phase('decision','success')
        else:
            if not args.flow_check:status='G0_engineering_only_flow_not_run'
            elif args.synthetic_check:status='synthetic_G0_flow_passed_no_real_screen'
            else:
                costs=checks['overhead'];n=timing['variants']['N']['overall'];g=timing['variants']['G']['overall']
                gates=dict(stable=timing['comparison_credible'],parameters=costs['parameter_ratio_G_to_N']<=1.05,conv_MAC=costs['conv_MAC_ratio_G_to_N']<=1.05,mean_latency=g['mean_ms']/n['mean_ms']<=1.05,P95_latency=g['p95_ms']/n['p95_ms']<=1.1)
                dump(out/'G0_screen.json',dict(gates=gates,mean_ratio=g['mean_ms']/n['mean_ms'],P95_ratio=g['p95_ms']/n['p95_ms'],scope='Initial gamma0 overhead engineering screen, not trained-model speed or detection evidence'))
                passed=all(gates.values())
                status='G0_passed_return_to_Work' if passed else 'G0_overhead_or_stability_screen_failed_return_to_Work'
        dump(out/'status.json',dict(status=status,stage=args.stage,synthetic_check=args.synthetic_check,automatic_G1=False,test_evaluated=False,
            G0_flow_not_effectiveness_experiment=args.stage=='g0',successful_conditions=list(endpoints)))
        return 0 if status in ['G0_passed_return_to_Work','synthetic_G0_flow_passed_no_real_screen','screen_passed_return_to_Work'] else 2
    except BaseException:
        for name,value in phases.items():
            if value=='running':phases[name]='failed'
        dump(out/'phases.json',phases);raise
    finally:
        if manifest is not None:dump(out/'source_read_only_verification.json',verify_sources(manifest))
        if code_identity()!=json.loads((out/'code_identity.json').read_text()):raise RuntimeError('Source implementation changed during suite')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',default='g0',choices=['env','g0','g1','worker','pack'])
    p.add_argument('--paths');p.add_argument('--flow-check',action='store_true',help='Explicitly run G0 two-epoch engineering flow, never G1')
    p.add_argument('--Work-review',dest='Work_review');p.add_argument('--worker-request');p.add_argument('--synthetic-check',action='store_true')
    args=p.parse_args()
    if args.stage=='pack':pack_results();return 0
    if args.stage=='worker':return worker(args)
    out=ROOT/'runs'/(args.stage+'_'+stamp());out.mkdir(parents=True);dump(out/'request.json',vars(args));code=1
    try:code=run_stage(args,out)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
        dump(out/'status.json',dict(status='failed',stage=args.stage,synthetic_check=args.synthetic_check,test_evaluated=False,automatic_G1=False))
    finally:
        try:pack_results()
        except Exception:
            (out/'pack_failure.txt').write_text(traceback.format_exc());traceback.print_exc();code=1
    print('Run directory:',out,flush=True);print('Exit code:',code,flush=True);return code

if __name__=='__main__':sys.exit(main())
