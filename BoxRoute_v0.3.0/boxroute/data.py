"""Read train/val only, audit bytes, then private independent source copies for all caches."""
import json,math,random,shutil
from pathlib import Path
import yaml
from PIL import Image
from .identity import file_sha256,dump
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def label_for(p):
    parts=list(Path(p).parts);positions=[i for i,x in enumerate(parts) if x=='images']
    if not positions:raise ValueError('Frozen YOLO image path must contain images directory')
    parts[positions[-1]]='labels';return Path(*parts).with_suffix('.txt')


def paths_for(spec,base,split):
    value=spec[split];sources=value if isinstance(value,list) else [value];files=[]
    for source in sources:
        p=Path(source);p=p if p.is_absolute() else base/p
        if p.is_dir():files.extend(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in EXT)
        elif p.suffix.lower()=='.txt':
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    item=Path(line.strip());files.append(item if item.is_absolute() else p.parent/item)
        else:files.append(p)
    canonical=sorted([p.resolve() for p in files],key=lambda p:(p.as_posix().casefold(),p.as_posix()))
    if not canonical or len({str(p).casefold() for p in canonical})!=len(canonical):raise ValueError('Empty/duplicate/ambiguous split')
    return canonical


def prepare_private(data,destination,expected=(1438,180),classes=6,selections=()):
    data=Path(data).resolve();spec=yaml.safe_load(data.read_text(encoding='utf-8-sig'));base=Path(spec.get('path',data.parent));base=base if base.is_absolute() else data.parent/base
    names=spec['names'];names=[names[k] for k in sorted(names)] if isinstance(names,dict) else list(names)
    if len(names)!=classes:raise ValueError('Unexpected class count')
    source={split:paths_for(spec,base,split) for split in ['train','val']} # never access spec.test
    if [len(source[k]) for k in ['train','val']]!=list(expected):raise ValueError('Frozen train/val cardinality differs')
    if set(source['train'])&set(source['val']):raise ValueError('Train/val file overlap')
    records={};seen_hash={}
    for split in ['train','val']:
        rows=[]
        for i,image in enumerate(source[split]):
            label=label_for(image)
            if not image.is_file() or not label.is_file():raise FileNotFoundError(str(image)+' or '+str(label))
            with Image.open(image) as im:width,height=im.size;im.verify()
            boxes=[]
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():continue
                values=list(map(float,line.split()))
                if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('Invalid label')
                k,x,y,w,h=values
                if k!=int(k) or not 0<=k<len(names) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:raise ValueError('Invalid normalized class/box')
                if values in boxes:raise ValueError('Duplicate label')
                boxes.append(values)
            sha=file_sha256(image)
            if sha in seen_hash and seen_hash[sha]!=split:raise ValueError('Duplicate source bytes across train/val')
            seen_hash[sha]=split
            rows.append(dict(image_id=f'{i:06d}',source_image=str(image),source_label=str(label),image_sha256=sha,label_sha256=file_sha256(label),width=width,height=height,boxes=boxes))
        records[split]=rows
    history=[]
    for filename in selections:
        filename=Path(filename)
        if not filename.is_file():history.append(dict(path=str(filename),status='missing_historical_selection',impact='Current split can be checked, historical selection membership cannot be tied to this absent file'));continue
        old=json.loads(filename.read_text(encoding='utf-8-sig'))
        for split in ['train','val']:
            prior={str(Path(r['image']).resolve()).casefold():(r['image_sha256'],r['label_sha256']) for r in old[split]}
            current={r['source_image'].casefold():(r['image_sha256'],r['label_sha256']) for r in records[split]}
            if prior!=current:raise ValueError('Frozen historical membership/content differs: '+str(filename))
        history.append(dict(path=str(filename),status='verified',sha256=file_sha256(filename)))
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    for split,rows in records.items():
        images=root/'images'/split;labels=root/'labels'/split;images.mkdir(parents=True);labels.mkdir(parents=True)
        for row in rows:
            image=Path(row['source_image']);label=Path(row['source_label']);stem=row['image_id']+'_'+row['image_sha256'][:12]
            working=images/(stem+image.suffix);working_label=labels/(stem+'.txt')
            shutil.copyfile(image,working);shutil.copyfile(label,working_label)
            if file_sha256(working)!=row['image_sha256'] or file_sha256(working_label)!=row['label_sha256']:raise RuntimeError('Byte copy differs')
            row.update(working_image=str(working.resolve()),working_label=str(working_label.resolve()))
    private=root/'data.yaml';private.write_text(yaml.safe_dump(dict(path=str(root.resolve()),train='images/train',val='images/val',names=names)),encoding='utf-8')
    manifest=dict(source_data_yaml=str(data),source_yaml_sha256=file_sha256(data),names=names,records=records,historical_selection_checks=history,
        test_enumerated=False,test_images=180,test_count_scope='Historical count only, no test paths/labels read',private_copies_byte_identical=True)
    dump(root/'manifest.json',manifest);return private,manifest


def verify_sources(manifest):
    changed=[]
    if file_sha256(manifest['source_data_yaml'])!=manifest['source_yaml_sha256']:changed.append(manifest['source_data_yaml'])
    for rows in manifest['records'].values():
        for row in rows:
            for key,sha in [('source_image','image_sha256'),('source_label','label_sha256'),('working_image','image_sha256'),('working_label','label_sha256')]:
                if not Path(row[key]).is_file() or file_sha256(row[key])!=row[sha]:changed.append(row[key])
    if changed:raise RuntimeError('Read-only source changed: '+str(changed))
    return dict(status='verified_unchanged',source_files=1+2*sum(len(rows) for rows in manifest['records'].values()),working_copies_also_verified=True,test_read=False)


def choose(manifest,split,n,seed=178):
    rows=list(manifest['records'][split]);random.Random(seed).shuffle(rows)
    if n>len(rows):raise ValueError('Not enough fixed split images')
    selected=rows[:n];coverage=[sum(int(box[0])==k for r in selected for box in r['boxes']) for k in range(len(manifest['names']))]
    return selected,dict(seed=seed,rule='canonical resolved source path order, independent Random178 shuffle, no coverage redraw',count=n,class_box_counts=coverage,
        source_image_sha256=[r['image_sha256'] for r in selected])


def subset_spec(manifest,root,train_n=64,val_n=16):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);selection={};spec=dict(names=manifest['names'])
    for split,count in [('train',train_n),('val',val_n)]:
        rows,metadata=choose(manifest,split,count);selection[split]=dict(records=rows,selection=metadata)
        listing=root/(split+'.txt');listing.write_text(''.join(r['working_image']+'\n' for r in rows),encoding='utf-8');spec[split]=str(listing.resolve())
    path=root/'data.yaml';path.write_text(yaml.safe_dump(spec),encoding='utf-8');dump(root/'selection.json',selection);return path,selection


def shape_statistics(manifest):
    import torch
    from .geometry import geometry_targets
    boxes=[(row,b) for row in manifest['records']['train'] for b in row['boxes']]
    if not boxes:
        return dict(train_images=len(manifest['records']['train']),train_boxes=0,informative_boxes=0,
            status='blocked_shape_support',threshold=30,horizontal_boxes=0,vertical_boxes=0,
            aspect_ratio=None,pixel_area=None,mean_q=None,q_distribution=None,
            class_coverage={name:dict(boxes=0,informative_boxes=0) for name in manifest['names']},
            scope='No train boxes: no geometry proxy distribution; engineering support screen only')
    wh=torch.tensor([[b[3]*row['width'],b[4]*row['height']] for row,b in boxes],dtype=torch.float64)
    q=geometry_targets(wh);ratio=wh[:,0]/wh[:,1];area=wh.prod(1)
    summary=lambda x:dict(min=float(x.min()),median=float(x.median()),mean=float(x.mean()),max=float(x.max()))
    informative=int(((q[:,1]-q[:,2]).abs()>=.2).sum())
    quantiles=torch.tensor([0.,.05,.25,.5,.75,.95,1.],dtype=wh.dtype)
    distribution=lambda x:dict(quantile_probabilities=quantiles.tolist(),quantiles=torch.quantile(x,quantiles).tolist())
    q_distribution={name:dict(**distribution(q[:,k]),histogram_edges=[i/10 for i in range(11)],
        histogram_counts=torch.histc(q[:,k],bins=10,min=0,max=1).long().tolist()) for k,name in enumerate(['I','H','V'])}
    per_class={name:dict(boxes=sum(int(b[0])==k for _,b in boxes),
        informative_boxes=sum(int(b[0])==k and bool(abs(q[j,1]-q[j,2])>=.2) for j,(_,b) in enumerate(boxes)))
        for k,name in enumerate(manifest['names'])}
    return dict(train_images=len(manifest['records']['train']),train_boxes=len(boxes),aspect_ratio=summary(ratio),pixel_area=summary(area),mean_q=q.mean(0).tolist(),
        aspect_ratio_distribution=distribution(ratio),pixel_area_distribution=distribution(area),q_distribution=q_distribution,
        class_coverage=per_class,
        horizontal_boxes=int((q[:,1]>0).sum()),vertical_boxes=int((q[:,2]>0).sum()),informative_boxes=informative,
        status='shape_support_passed' if informative>=30 else 'blocked_shape_support',threshold=30,
        scope='Engineering proxy-support screen, not sample-size power or true orientation labels')
