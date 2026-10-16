"""Frozen research protocol and explicit real-GPU versus synthetic CPU preflight."""
import hashlib,importlib,inspect,platform,subprocess,sys,json
from pathlib import Path
import yaml
from .identity import dump,file_sha256
ROOT=Path(__file__).resolve().parents[1]


def load_protocol():
    data=yaml.safe_load((ROOT/'protocol.yaml').read_text(encoding='utf-8-sig'))
    if data['version']!='BoxRoute-v0.3.0-proposed-1.0':raise ValueError('Unknown frozen protocol')
    # The source manifest is checked independently; edits to the frozen protocol are rejected.
    import json
    manifest=json.loads((ROOT/'sources_manifest.json').read_text())
    if file_sha256(ROOT/'protocol.yaml')!=manifest['protocol_sha256']:raise ValueError('Frozen protocol modified')
    return data


def preflight(out,synthetic=False):
    import torch,torchvision,ultralytics
    from ultralytics import settings
    versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,ultralytics=ultralytics.__version__)
    dump(Path(out)/'environment.json',dict(versions=versions,python=sys.version,platform=platform.platform(),cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),synthetic_check=synthetic))
    required={'torch':'2.5.1','torchvision':'0.20.1','ultralytics':'8.3.221'}
    if any(versions[k].split('+')[0]!=v for k,v in required.items()):raise RuntimeError('Pinned dependencies required; no installation or upgrades')
    if not synthetic:
        if versions['torch']!='2.5.1+cu121' or versions['torchvision']!='0.20.1+cu121' or torch.version.cuda!='12.1' or not torch.cuda.is_available():raise RuntimeError('Real G0/G1 requires the existing CUDA12.1 GPU0 stack; no CPU fallback')
        result=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=15)
        dump(Path(out)/'nvidia_smi.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,gpu_name=torch.cuda.get_device_name(0)))
        if result.returncode:raise RuntimeError('GPU driver query failed')
    manifest=json.loads((ROOT/'sources_manifest.json').read_text())
    for relative,expected_sha in manifest['source_files'].items():
        if file_sha256(ROOT/relative)!=expected_sha:raise ValueError('Supplied fixed source/resource changed: '+relative)
    dump(Path(out)/'resource_identities.json',dict(fixed_source_files_verified=len(manifest['source_files']),
        public_initialization_sha256=file_sha256(ROOT/'sources/official_yolo11n_v830.pt'),
        public_initialization_bytes=(ROOT/'sources/official_yolo11n_v830.pt').stat().st_size))
    settings.update({k:False for k in ['sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb']})
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_modules=['ultralytics.nn.tasks','ultralytics.utils.loss','ultralytics.utils.tal','ultralytics.engine.trainer','ultralytics.models.yolo.detect.train','ultralytics.data.build','ultralytics.models.yolo.detect.val']
    reference={'ultralytics.nn.tasks':'ultralytics_v83221_ultralytics_nn_tasks.py','ultralytics.utils.loss':'ultralytics_v83221_ultralytics_utils_loss.py','ultralytics.utils.tal':'ultralytics_v83221_ultralytics_utils_tal.py','ultralytics.engine.trainer':'ultralytics_v83221_ultralytics_engine_trainer.py','ultralytics.models.yolo.detect.train':'ultralytics_v83221_detect_train.py','ultralytics.data.build':'ultralytics_fixed_ultralytics_data_build.py','ultralytics.models.yolo.detect.val':'ultralytics_fixed_ultralytics_models_yolo_detect_val.py'}
    identities={}
    for name in source_modules:
        installed=Path(importlib.import_module(name).__file__);expected=ROOT/'sources'/reference[name]
        equal=installed.read_text(encoding='utf-8-sig')==expected.read_text(encoding='utf-8-sig')
        identities[name]=dict(path=str(installed),sha256=file_sha256(installed),reference=str(expected),equal_after_universal_newlines=equal)

    dump(Path(out)/'installed_source_identities.json',identities)
    if not all(row['equal_after_universal_newlines'] for row in identities.values()):raise RuntimeError('Installed fixed-version source differs from handoff; see source identities')
    return torch.device('cpu' if synthetic else 'cuda:0')
