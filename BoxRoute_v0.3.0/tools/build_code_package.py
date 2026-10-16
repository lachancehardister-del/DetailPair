"""Complete independent source/resources code package; user/runtime images and weights excluded."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'runs','exports','releases','work_inputs','local_settings','__pycache__'}
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if EXCLUDED&set(rel.parts) or p.name in {'ARTIFACT_MANIFEST.json','verification_manifest.json','WORK_LOG.md'} or p.suffix in {'.pyc','.cache'}:continue
    if p.suffix in {'.pt','.pth','.npy','.npz'} and rel.as_posix()!='sources/official_yolo11n_v830.pt':continue
    files.append(p)
manifest=dict(version='BoxRoute_v0.3.0',scope='Full independent source/config/Windows/protocol/licenses/actual engineering checks, plus pinned public initialization; excludes original user data/weights/runs',
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
path=ROOT/'verification_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');files.append(path)
dest=ROOT/'releases';dest.mkdir(exist_ok=True);archive=dest/'BoxRoute_v0.3.0_Code.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,sha in manifest['files'].items():assert hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()==sha,name
sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha,entries=len(files))))
