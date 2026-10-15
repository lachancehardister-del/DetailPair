from pathlib import Path
import hashlib,json,math,random,shutil
import numpy as np
from PIL import Image
import yaml

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def label_for(p):
    parts=list(p.parts)
    indices=[i for i,v in enumerate(parts) if v=='images']
    if not indices: raise ValueError(f'Image path must contain an images folder: {p}')
    parts[indices[-1]]='labels'
    return Path(*parts).with_suffix('.txt')

def load_spec(filename):
    filename=Path(filename).resolve()
    spec=yaml.safe_load(filename.read_text(encoding='utf-8'))
    root=Path(spec.get('path',filename.parent))
    if not root.is_absolute(): root=filename.parent/root
    names=spec['names']
    if isinstance(names,dict):
        names={int(k):v for k,v in names.items()}
        if sorted(names)!=list(range(len(names))):raise ValueError('names indices must be contiguous from 0')
        names=[names[i] for i in range(len(names))]
    splits={}
    for split in ('train','val','test'):
        if split not in spec: continue
        sources=spec[split] if isinstance(spec[split],list) else [spec[split]]
        files=[]
        for source in sources:
            p=Path(source)
            if not p.is_absolute():p=root/p
            if p.is_dir():files+=sorted(p.rglob('*'))
            elif p.suffix.lower()=='.txt':
                for line in p.read_text(encoding='utf-8-sig').splitlines():
                    if not line.strip():continue
                    q=Path(line.strip())
                    if not q.is_absolute():q=p.parent/q
                    files.append(q)
            else:files.append(p)
        splits[split]=sorted({p.resolve() for p in files if p.suffix.lower() in EXT})
        if not splits[split]:raise ValueError(f'Empty {split} split')
    if not {'train','val'}<=splits.keys():raise ValueError('train and val splits are required')
    return names,splits

def audit(filename,out):
    names,splits=load_spec(filename)
    records=[];errors=[];hashes={};pixels={};near={};warnings=[]
    for split,files in splits.items():
        for p in files:
            try:
                with Image.open(p) as im:
                    im.load();w,h=im.size
                    a=np.asarray(im.convert('RGB'))
                    gray=np.asarray(im.convert('L').resize((9,8)),dtype=np.float32)
                ph=hashlib.sha256(a.tobytes()+str(a.shape).encode()).hexdigest()
                nh=np.packbits(gray[:,1:]>gray[:,:-1]).tobytes().hex()
                if ph in pixels:
                    if pixels[ph][0]!=split:errors.append(f'Identical decoded image across splits: {p} / {pixels[ph]}')
                    else:warnings.append(f'Identical decoded image within {split}: {p} / {pixels[ph]}')
                pixels.setdefault(ph,(split,str(p)))
                if nh in near and near[nh][0]!=split and near[nh][1]!=ph:
                    warnings.append(f'Identical dHash candidate across splits; inspect manually: {p} / {near[nh][2]}')
                near.setdefault(nh,(split,ph,str(p)))
                sha=digest(p)
                if sha in hashes and hashes[sha][0]!=split:errors.append(f'File duplicate across splits: {p}')
                hashes.setdefault(sha,(split,str(p)))
                lab=label_for(p)
                if not lab.exists():raise ValueError(f'Missing label {lab}; explicitly create empty txt only for confirmed background')
                boxes=[]
                for line in lab.read_text(encoding='utf-8-sig').splitlines():
                    if not line.strip():continue
                    row=list(map(float,line.split()))
                    if len(row)!=5:raise ValueError('Detection requires exactly 5 YOLO columns')
                    k,x,y,bw,bh=row
                    if not all(math.isfinite(t) for t in row):raise ValueError('Nonfinite label')
                    if k!=int(k) or not 0<=k<len(names):raise ValueError('Invalid class')
                    if bw<=0 or bh<=0 or min(x-bw/2,y-bh/2)<-1e-5 or max(x+bw/2,y+bh/2)>1+1e-5:
                        raise ValueError('Invalid normalized box or box outside image')
                    boxes.append(row)
                if len({tuple(x) for x in boxes})!=len(boxes):raise ValueError('Duplicate boxes in one label')
                records.append(dict(split=split,image=str(p),label=str(lab),image_sha256=sha,label_sha256=digest(lab),width=w,height=h,boxes=boxes))
            except Exception as e:errors.append(f'{p}: {type(e).__name__}: {e}')
    coverage={s:[sum(int(b[0])==k for row in records if row['split']==s for b in row['boxes']) for k in range(len(names))] for s in splits}
    for k,n in enumerate(coverage.get('train',[])):
        if not n:errors.append(f'Class {k} ({names[k]}) has no training boxes')
    for k,n in enumerate(coverage.get('val',[])):
        if not n:warnings.append(f'Class {k} ({names[k]}) has no validation boxes')
    result=dict(schema_version=1,source_yaml=str(Path(filename).resolve()),names=names,class_box_counts=coverage,counts={s:len(fs) for s,fs in splits.items()},errors=errors,warnings=warnings,
                scope='YOLO detection labels, exact byte/decoded duplicates and identical dHash candidates. Not exhaustive near-duplicate/group leakage or source/license audit.',records=records)
    Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise ValueError(f'Audit failed with {len(errors)} error(s); see {out}')
    return result

def subset(manifest,out,train_limit=64,val_limit=32,seed=0):
    r=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if r['errors']:raise ValueError('Cannot use failed audit')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(seed)
    d={'names':r['names'],'path':str(out.resolve())}
    selected={}
    for split,limit in [('train',train_limit),('val',val_limit)]:
        rs=[x for x in r['records'] if x['split']==split]
        rng.shuffle(rs);rs=rs[:limit] if limit else rs
        selected[split]=rs
        # Private working copy also keeps Ultralytics label caches away from originals.
        images=out/'images'/split;labels=out/'labels'/split
        images.mkdir(parents=True);labels.mkdir(parents=True)
        for i,row in enumerate(rs):
            stem=f'{i:06d}_{row["image_sha256"][:12]}'
            target=images/(stem+Path(row['image']).suffix)
            shutil.copyfile(row['image'],target)
            shutil.copyfile(row['label'],labels/(stem+'.txt'))
            row['working_image']=str(target.resolve())
        d[split]=f'images/{split}'
    (out/'data.yaml').write_text(yaml.safe_dump(d,allow_unicode=True),encoding='utf-8')
    (out/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    return out/'data.yaml'

def check_manifest(manifest):
    r=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if r['errors']:raise ValueError('Failed audit')
    for row in r['records']:
        if digest(row['image'])!=row['image_sha256'] or digest(row['label'])!=row['label_sha256']:
            raise ValueError(f'Data changed after audit: {row["image"]}')
    return r

def synthetic(root):
    # Procedural rectangle images for plumbing only. No defect performance meaning.
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for split,n in [('train',8),('val',4),('test',4)]:
        (root/'images'/split).mkdir(parents=True);(root/'labels'/split).mkdir(parents=True)
        for i in range(n):
            rng=np.random.default_rng(i+{'train':0,'val':100,'test':200}[split])
            a=rng.integers(0,35,(64,64,3),dtype=np.uint8);a[20:44,16:40]=220
            Image.fromarray(a).save(root/'images'/split/f'{i}.png')
            (root/'labels'/split/f'{i}.txt').write_text('0 0.4375 0.5 0.375 0.375\n')
    y=root/'data.yaml';y.write_text(yaml.safe_dump(dict(path=str(root.resolve()),train='images/train',val='images/val',test='images/test',names=['synthetic_rectangle'])))
    return y
