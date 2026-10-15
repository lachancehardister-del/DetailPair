"""Build a source/verification-only release ZIP; runtime results stay separate."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'runs','exports','local_settings','__pycache__','releases','work_inputs','onboarding'}
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not (EXCLUDED&set(p.relative_to(ROOT).parts)) and p.suffix not in {'.pyc','.pt','.cache'} and p.name not in {'verification_manifest.json','ARTIFACT_MANIFEST.json','WORK_LOG.md','AGENTS.md'}]
manifest=dict(version='0.2.7',scope='source, protocol, dependencies, Windows entries, actual CPU verification; no original data or weights',
 files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
m=ROOT/'verification_manifest.json';m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files.append(m)
out=ROOT/'releases';out.mkdir(exist_ok=True)
zip_path=out/'DetailPair_v0.2.7_Experiment_Package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 for name,expected in manifest['files'].items():assert hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()==expected
 assert not any('/runs/' in n or n.endswith('.pt') for n in z.namelist())
sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
zip_path.with_suffix('.zip.sha256').write_text(sha+'  '+zip_path.name+'\n')
print(json.dumps(dict(path=str(zip_path),SHA256=sha,bytes=zip_path.stat().st_size,files=len(files))))
