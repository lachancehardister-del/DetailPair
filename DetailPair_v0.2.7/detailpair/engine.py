import json,time,copy,hashlib
from pathlib import Path
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from .model import PairDetectionModel,VARIANTS


class PairTrainer(DetectionTrainer):
    def __init__(self,research,**kwargs):
        self.research=copy.deepcopy(research)
        super().__init__(**kwargs)
        self.add_callback('on_train_batch_end',self.log_aux)
    def get_model(self,cfg=None,weights=None,verbose=True):
        m=PairDetectionModel(cfg or 'yolo11n.yaml',ch=self.data.get('channels',3),nc=self.data['nc'],verbose=verbose,variant=self.research['variant'])
        m.noise_std=self.research.get('noise_std',0.03)
        m.brightness=self.research.get('brightness',0.05)
        m.perturbation_seed=100003+int(self.args.seed)*1000000
        if weights:m.load(weights)
        return m
    @staticmethod
    def log_aux(trainer):
        m=trainer.model.module if hasattr(trainer.model,'module') else trainer.model
        row={'epoch':trainer.epoch,'batch':int(getattr(trainer,'batch_i',-1)),**getattr(m,'latest_aux',{})}
        with open(Path(trainer.save_dir)/'auxiliary.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')


def train(research,data,run_dir,epochs,batch,imgsz,device,seed,weights=None):
    from ultralytics import settings
    settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb')})
    # No automatic pretrained weight download. Optional weights must be a trusted local file.
    if weights and not Path(weights).is_file():raise ValueError('weights must be an existing local path')
    model=str(Path(weights).resolve()) if weights else 'yolo11n.yaml'
    overrides=dict(model=model,data=str(data),epochs=epochs,batch=batch,imgsz=imgsz,device=device,seed=seed,workers=0,
        project=str(Path(run_dir).parent),name=Path(run_dir).name,exist_ok=True,optimizer='AdamW',lr0=0.001,lrf=0.01,
        amp=False,pretrained=bool(weights),deterministic=True,plots=True,save=True,val=True,patience=50,
        mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=0.1,scale=0.1,
        fliplr=0.5,flipud=0.,close_mosaic=0,cache=False,save_period=1)
    trainer=PairTrainer(research=research,overrides=overrides)
    trainer.train()
    return Path(trainer.save_dir)


def smoke(out):
    from ultralytics.cfg import DEFAULT_CFG
    torch.set_num_threads(2)
    rows=[]
    for variant in VARIANTS:
        torch.manual_seed(0)
        m=PairDetectionModel(nc=2,variant=variant)
        m.args=copy.deepcopy(DEFAULT_CFG)
        m.train()
        x=torch.rand(2,3,64,64)
        b=dict(img=x,batch_idx=torch.tensor([0.,1.]),cls=torch.tensor([[0.],[1.]]),bboxes=torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]]))
        l,it=m(b)
        total=l.sum();assert torch.isfinite(total)
        total.backward()
        params=[p for p in m.parameters() if p.requires_grad]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        if variant=='B6':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.model[1].gates.parameters())
        # Model must be deepcopy-able after training forward (EMA/checkpoint compatibility).
        c=copy.deepcopy(m).eval()
        with torch.no_grad():p=c(x)
        rows.append(dict(variant=variant,loss=float(total.detach()),parameters=sum(p.numel() for p in m.parameters()),aux=m.latest_aux,layer_match=getattr(m.model[1],'match',None),forward_backward=True,ema_deepcopy=True))
    Path(out).write_text(json.dumps(rows,indent=2))
    return rows
