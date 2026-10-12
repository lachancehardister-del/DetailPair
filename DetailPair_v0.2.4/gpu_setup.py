"""Windows-only independent CUDA environment clone; retains every failed attempt."""
import argparse, datetime, json, os, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=r'D:\ModelImprovement\envs\detailpair_v02')
    p.add_argument('--target', default=r'D:\ModelImprovement\envs\detailpair_cuda_v021')
    args = p.parse_args()
    folder = ROOT / 'install_logs' / ('cuda_setup_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(source=args.source, target=args.target, platform=sys.platform, commands=[])
    try:
        if sys.platform != 'win32': raise RuntimeError('Run this setup in Windows Anaconda Prompt only')
        source, target = Path(args.source), Path(args.target)
        if not (source / 'python.exe').is_file(): raise FileNotFoundError(source / 'python.exe')
        if target.exists(): raise FileExistsError('Target exists; not modified. Activate it for checks, or choose a fresh --target.')
        conda = os.environ.get('CONDA_EXE', '')
        if not conda or not Path(conda).is_file(): raise RuntimeError('CONDA_EXE unavailable. Open Anaconda Prompt and activate the existing CPU environment first.')
        commands = [
            [conda, 'create', '--yes', '--copy', '--prefix', str(target), '--clone', str(source)],
            [str(target / 'python.exe'), '-m', 'pip', 'install', '--force-reinstall', '--no-deps', 'torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121'],
            [str(target / 'python.exe'), '-m', 'pip', 'check'],
        ]
        for i, cmd in enumerate(commands):
            print('Executing:', subprocess.list2cmdline(cmd), flush=True)
            record['commands'].append(cmd)
            with (folder / f'step_{i}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Step {i} failed with exit {result.returncode}; see {folder}')
        record['status'] = 'success'
        print('Environment created. CUDA NOT YET VERIFIED. Run windows\\11_check_gpu.cmd next.')
    except Exception:
        record['status'] = 'failed'
        (folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        (folder / 'setup.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if record['status'] == 'success' else 1

if __name__ == '__main__': sys.exit(main())
