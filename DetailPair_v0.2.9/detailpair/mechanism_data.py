"""Train-only selection and installed LetterBox label transform, no geometric randomness."""
import random,hashlib,json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances
from ultralytics.utils.patches import imread
from .data import label_for,digest,EXT
from .mechanism import tensor_identity


def normalized_path(path):return Path(path).resolve().as_posix().casefold()


def training_paths(data):
    data=Path(data).resolve();spec=yaml.safe_load(data.read_text(encoding='utf-8'))
    root=Path(spec.get('path',data.parent));root=root if root.is_absolute() else data.parent/root
    names=spec['names']
    if isinstance(names,dict):
        names={int(k):v for k,v in names.items()}
        if sorted(names)!=list(range(len(names))):raise ValueError('Class indices must be contiguous')
        names=[names[k] for k in sorted(names)]
    sources=spec['train'] if isinstance(spec['train'],list) else [spec['train']]
    files=[]
    for source in sources:
        path=Path(source);path=path if path.is_absolute() else root/path
        if path.is_dir():files.extend(p for p in path.rglob('*') if p.suffix.lower() in EXT and p.is_file())
        elif path.suffix.lower()=='.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():continue
                p=Path(line.strip());files.append(p if p.is_absolute() else path.parent/p)
        else:files.append(path)
    files=sorted({p.resolve() for p in files},key=lambda p:(normalized_path(p),p.as_posix()))
    if not files or any(not p.is_file() or p.suffix.lower() not in EXT for p in files):raise ValueError('Invalid train-only source list')
    if len({normalized_path(p) for p in files})!=len(files):raise ValueError('Ambiguous case-normalized source paths')
    return names,files


def choose_training(data,entries,count=64,seed=178,expected_train=1438):
    names,files=training_paths(data)
    if len(files)!=expected_train:raise ValueError(f'Expected {expected_train} train images; found {len(files)}')
    if count>len(files) or count<2:raise ValueError('Invalid fixed selection size')
    canonical={normalized_path(p):p for p in files};frozen=[];selection_identities=[]
    for entry in entries:
        path=Path(entry['selection']);raw=json.loads(path.read_text(encoding='utf-8'));rows=raw['train']
        table={normalized_path(row['image']):row for row in rows}
        if len(rows)!=len(table) or set(table)!=set(canonical):raise ValueError('Frozen train membership differs for '+entry['variant'])
        frozen.append(table);selection_identities.append(dict(variant=entry['variant'],path=str(path.resolve()),sha256=digest(path)))
    shuffled=list(files);random.Random(seed).shuffle(shuffled);chosen=shuffled[:count]
    records=[];coverage=[0 for _ in names]
    for i,path in enumerate(chosen):
        label=label_for(path);image_sha=digest(path);label_sha=digest(label)
        for table in frozen:
            row=table[normalized_path(path)]
            if image_sha!=row['image_sha256'] or label_sha!=row['label_sha256'] or normalized_path(row['label'])!=normalized_path(label):
                raise ValueError('Selected train image/label differs from frozen source: '+str(path))
        with Image.open(path) as image:width,height=image.size;image.verify()
        annotations=[]
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()))
            if len(values)!=5 or not np.isfinite(values).all():raise ValueError('Invalid label '+str(label))
            k,x,y,w,h=values
            if k!=int(k) or not 0<=k<len(names) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-5 or max(x+w/2,y+h/2)>1+1e-5:raise ValueError('Invalid class/box '+str(label))
            if values in annotations:raise ValueError('Duplicate selected label '+str(label))
            annotations.append(values);coverage[int(k)]+=1
        records.append(dict(image_id=f'train_{i:04d}',source_image=str(path),source_label=str(label),image_sha256=image_sha,label_sha256=label_sha,
            width=width,height=height,annotations=annotations))
    metadata=dict(seed=seed,rule=f'sort normalized resolved source path (casefold, POSIX separators); independent random.Random({seed}).shuffle then first{count}, no re-draw',
        train_membership=len(files),selected_images=len(records),selection_identities=selection_identities,
        class_box_counts={str(i):dict(name=n,boxes=coverage[i]) for i,n in enumerate(names)},
        missing_classes=[names[i] for i,n in enumerate(coverage) if n==0],
        content_verification='Selected images/labels checked against both historical selections; unselected train membership only. No val/test paths or labels loaded.')
    return names,records,metadata


def load_batch(records,device,imgsz=320):
    images=[];boxes=[];classes=[];indices=[];transforms=[]
    letterbox=LetterBox((imgsz,imgsz),auto=False,scale_fill=False,scaleup=True,center=True,stride=32)
    for batch_idx,row in enumerate(records):
        if digest(row['source_image'])!=row['image_sha256'] or digest(row['source_label'])!=row['label_sha256']:raise ValueError('Selected source changed after audit')
        image=imread(row['source_image'])
        if image is None:raise ValueError('Cannot decode selected image')
        if list(image.shape[:2])!=[row['height'],row['width']]:raise ValueError('PIL/official decoder dimension mismatch')
        a=np.asarray(row['annotations'],dtype=np.float32).reshape(-1,5)
        labels=dict(img=image,cls=a[:,:1].copy(),instances=Instances(a[:,1:].copy(),segments=np.zeros((0,0,2),dtype=np.float32),bbox_format='xywh',normalized=True))
        result=letterbox(labels)
        result['instances'].convert_bbox('xywh');result['instances'].normalize(imgsz,imgsz)
        transformed=result['instances'].bboxes
        boxes.append(torch.from_numpy(transformed));classes.append(torch.from_numpy(a[:,:1]));indices.append(torch.full((len(a),),float(batch_idx)))
        rgb=np.ascontiguousarray(result['img'][:,:,::-1].transpose(2,0,1));images.append(torch.from_numpy(rgb))
        transforms.append(dict(image_id=row['image_id'],source_height_width=[row['height'],row['width']],actual_height_width=list(result['img'].shape[:2]),
            normalized_xywh=tensor_identity(torch.from_numpy(transformed)),classes=tensor_identity(torch.from_numpy(a[:,:1])),
            letterbox='installed LetterBox auto=False scale_fill=False scaleup=True center=True stride32; no flips/geometry randomness'))
    batch=dict(img=torch.stack(images).to(device).float()/255.,bboxes=torch.cat(boxes).to(device),cls=torch.cat(classes).to(device),batch_idx=torch.cat(indices).to(device))
    if list(batch['img'].shape)!=[len(records),3,imgsz,imgsz]:raise RuntimeError('Actual input shape differs from fixed square protocol')
    return batch,dict(image_ids=[r['image_id'] for r in records],input=tensor_identity(batch['img']),transforms=transforms,
        target_count=int(batch['bboxes'].shape[0]),bboxes=tensor_identity(batch['bboxes']),classes=tensor_identity(batch['cls']),batch_idx=tensor_identity(batch['batch_idx']))
