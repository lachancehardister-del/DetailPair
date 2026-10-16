"""Run the requested synthetic native/B0 numerical comparison, no training suite."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
from boxroute.native_equivalence import run_native_equivalence
from boxroute.identity import dump

if __name__=='__main__':
    result=run_native_equivalence('cpu');dump(ROOT/'verification/native_equivalence_cpu.json',result)
    print('PASSED',len(result['records']),'synthetic size/batch/GT cases; CPU only, no NEU performance claim')
