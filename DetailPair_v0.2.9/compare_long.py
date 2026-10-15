"""B2/B6 long-budget comparison: separate processes, failures retained."""
import argparse, csv, inspect, json, os, random, subprocess, sys, traceback
from pathlib import Path
from run import ROOT, stamp, dump, environment, pack

def schedule(config):
    if config['variants']!=['B2','B6'] or config['seeds']!=[0]: raise ValueError('Frozen comparison requires B2/B6, seed 0')
    if config['epochs']!=200 or config['patience']!=50 or config['train_limit']!=0 or config['val_limit']!=0: raise ValueError('Expected 200 maximum epochs, patience 50 and full train/val')
    if config['initialization']!='scratch' or config['batch']!=2 or config['imgsz']!=320: raise ValueError('Frozen initialization/input/batch changed')
    order=list(config['variants']);random.Random(127).shuffle(order)
    return order

def run_child(command, env, cwd):
    """Child inherits terminal stdout/stderr; no PIPE, tee or line forwarding."""
    return subprocess.run(command, cwd=cwd, env=env, check=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=r'D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml')
    p.add_argument('--config',default=str(ROOT/'configs/long_comparison.yaml'))
    p.add_argument('--accept-reviewed-near-duplicates',action='store_true')
    args=p.parse_args()
    parent=ROOT/'runs'/('long_comparison_'+stamp());parent.mkdir(parents=True,exist_ok=False)
    dump(parent/'request.json',vars(args));dump(parent/'environment.json',environment())
    failed=False
    try:
        import torch, yaml
        from ultralytics.utils.torch_utils import EarlyStopping
        from detailpair.data import audit, check_manifest
        if not torch.cuda.is_available():raise RuntimeError('Working CUDA required; CPU fallback forbidden')
        c=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        order=schedule(c);dump(parent/'protocol.json',c);dump(parent/'order.json',order)
        # Save the actual pinned installed early-stopping implementation, not a guessed version.
        (parent/'EarlyStopping_source.py').write_text(inspect.getsource(EarlyStopping),encoding='utf-8')
        stopper=EarlyStopping(patience=50)
        assert stopper.patience==50
        audited=audit(args.data,parent/'audit.json')
        if audited['warnings'] and not args.accept_reviewed_near_duplicates:raise ValueError('Previously reviewed dHash override must be explicit')
        children=[]
        for variant in order:
            check_manifest(parent/'audit.json')
            child_config=dict(c,variants=[variant]);cfg=parent/(variant+'_config.yaml')
            cfg.write_text(yaml.safe_dump(child_config,sort_keys=False),encoding='utf-8')
            before={f.name for f in (ROOT/'runs').glob('pilot_*')}
            cmd=[sys.executable,str(ROOT/'run.py'),'pilot','--data',str(Path(args.data).resolve()),'--device','0','--config',str(cfg)]
            if args.accept_reviewed_near_duplicates:cmd.append('--accept-near-duplicates-for-pilot')
            child_env=os.environ.copy()
            # Full training is asynchronous. This is not the synchronous debugging entry.
            child_env.pop('CUDA_LAUNCH_BLOCKING',None)
            child_env['PYTHONUNBUFFERED']='1'
            child_env['PYTHONIOENCODING']='utf-8'
            dump(parent/(variant+'_command.json'),dict(command=cmd,CUDA_LAUNCH_BLOCKING='unset'))
            print('Starting',variant,'max epochs=200, patience=50; native terminal output',flush=True)
            result=run_child(cmd,child_env,ROOT)
            new=sorted(f for f in (ROOT/'runs').glob('pilot_*') if f.name not in before)
            child=dict(variant=variant,exit_code=result.returncode,pilot_paths=[str(f) for f in new],actual_epochs=None)
            if len(new)==1:
                csvfile=new[0]/(variant+'_s0/training/results.csv')
                if csvfile.exists():
                    with csvfile.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                    child['actual_epochs']=len(rows)
                condition=new[0]/(variant+'_s0/status.json')
                child['condition_status']=json.loads(condition.read_text()) if condition.exists() else None
                child['successful']=result.returncode==0 and child['condition_status'] is not None and child['condition_status'].get('status')=='success'
            else:child['successful']=False
            children.append(child);dump(parent/'children.json',children)
            failed=failed or not child['successful']
            # Always use a fresh process for the other condition, even if the first fails.
            print('Finished',variant,'exit',result.returncode,'actual epochs',child['actual_epochs'],flush=True)
        dump(parent/'status.json',dict(status='partial_or_failed' if failed else 'success',kind='SINGLE_SEED_LONG_BUDGET_COMPARISON_NOT_PUBLICATION_EVIDENCE',children=children))
    except Exception:
        failed=True
        (parent/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        dump(parent/'status.json',dict(status='failed'));traceback.print_exc()
    finally:
        pack()
    return 1 if failed else 0

if __name__=='__main__':sys.exit(main())
