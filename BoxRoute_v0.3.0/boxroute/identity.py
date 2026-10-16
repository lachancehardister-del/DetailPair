"""Streaming identities for read-only inputs and shared initialization checks."""
import hashlib,json
from pathlib import Path


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def state_sha256(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
