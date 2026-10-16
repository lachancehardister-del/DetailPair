"""Full G0 CLI with tiny procedural data on explicit CPU; never a real experiment."""
import os,sys,tempfile,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'));os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
import yaml
from check_data_pipeline import fixture
from boxroute.identity import file_sha256,dump


def main():
    with tempfile.TemporaryDirectory(prefix='BoxRoute G0 fixture ') as temp:
        temp=Path(temp);data=fixture(temp/'synthetic_source');source_files={str(p):file_sha256(p) for p in (temp/'synthetic_source').rglob('*') if p.is_file()}
        paths=temp/'paths.yaml';paths.write_text(yaml.safe_dump(dict(data=str(data),historical_selections=[],synthetic_train_images=4,synthetic_val_images=2)))
        before=set((ROOT/'runs').glob('g0_*'))
        result=subprocess.run([sys.executable,str(ROOT/'run_boxroute.py'),'g0','--paths',str(paths),'--flow-check','--synthetic-check'],cwd=ROOT)
        run=(set((ROOT/'runs').glob('g0_*'))-before).pop()
        assert result.returncode==0,'Full synthetic G0 failed: '+str(run)
        status=json.loads((run/'status.json').read_text());assert status['status']=='synthetic_G0_flow_passed_no_real_screen' and not status['automatic_G1']
        assert status['successful_conditions']==['S','W','U','G','N']
        augmentation={}
        for variant in ['N','S','U','G','W']:
            path=run/variant/'training';state=json.loads((run/variant/'worker_status.json').read_text())
            assert state['status']=='success' and state['endpoint']['checkpoint_epoch']==1 and state['endpoint']['completed_epoch']==2
            shapes=[json.loads(x) for x in (path/'actual_inputs.jsonl').read_text().splitlines()]
            assert {x['stage'] for x in shapes}=={'train','val'} and all(x['shape'][-2:]==[320,320] and x['dtype']=='torch.float32' for x in shapes)
            assert any(x['stage']=='train' and x.get('augmented_image_and_target_sha256') for x in shapes)
            assert state['optimizer_steps']>0
            augmentation[variant]=[x['augmented_image_and_target_sha256'] for x in shapes if x['stage']=='train']
        assert all(file_sha256(p)==sha for p,sha in source_files.items())
        assert all(value==augmentation['N'] for value in augmentation.values())
        before=set((ROOT/'runs').glob('g1_*'))
        fail=subprocess.run([sys.executable,str(ROOT/'run_boxroute.py'),'g1','--synthetic-check'],cwd=ROOT)
        blocked=(set((ROOT/'runs').glob('g1_*'))-before).pop();assert fail.returncode==1 and (blocked/'failure.txt').is_file()
        from run_boxroute import pack_results
        archive=pack_results()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and any('worker_status.json' in n for n in z.namelist())
            assert not any(n.endswith(('.pt','.pth','.png','.jpg','.npy','.npz')) or '/private_data/' in n or '/datasets/' in n or '/images/' in n or '/labels/' in n for n in z.namelist())
        report=dict(status='passed',scope='Explicit CPU procedural4train/2val,2epochs five conditions; not NEU/GPU/results',G0_run=str(run),G1_blocked_run=str(blocked),
            train_val_final_eval_actual_shape='320 square FP32',unstripped_epoch2_verified=True,source_bytes_unchanged=True,no_automatic_G1=True,timing='not_run_CPU_fixture_no_speed_claim')
        dump(ROOT/'verification/G0_cpu_flow.json',report);print(json.dumps(report))

if __name__=='__main__':main()
