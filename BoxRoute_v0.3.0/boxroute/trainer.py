"""All conditions share a square data pipeline; official detector/loss/metric logic retained."""
import copy,json,hashlib,shutil
from pathlib import Path
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.torch_utils import unwrap_model
from .identity import dump,file_sha256


def observe(batch,path,stage):
    image=batch['img'];record=dict(stage=stage,shape=list(image.shape),dtype=str(image.dtype),device=str(image.device))
    record['files']=list(batch.get('im_file',[]))
    valid=list(image.shape[-2:])==[320,320] and image.dtype==torch.float32
    record['status']='passed' if valid else 'failed_shape_or_dtype'
    if stage=='train':
        h=hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes())
        for key in ['batch_idx','cls','bboxes']:h.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
        record['augmented_image_and_target_sha256']=h.hexdigest()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    if not valid:raise RuntimeError('Actual '+stage+' input is not FP32 320x320; preserving failed record')


class SquareValidator(DetectionValidator):
    def build_dataset(self,img_path,mode='val',batch=None):
        args=copy.copy(self.args);args.rect=False
        dataset=build_yolo_dataset(args,img_path,batch,self.data,mode=mode,rect=False,stride=self.stride)
        if dataset.rect:raise RuntimeError('Validator rectangle override failed')
        return dataset
    def preprocess(self,batch):
        batch=super().preprocess(batch);observe(batch,self.save_dir/'actual_inputs.jsonl','val');return batch


class SquareTrainer(DetectionTrainer):
    def __init__(self,initial_model=None,**kwargs):
        self.initial_model=initial_model;self.actual_optimizer_steps=0
        super().__init__(**kwargs)
        if self.args.imgsz!=320 or self.args.multi_scale or self.args.rect or self.args.amp:raise ValueError('Frozen square FP32320 trainer only')
        self.add_callback('on_train_epoch_start',self.route_epoch)
        self.add_callback('on_train_batch_end',self.route_log)
        self.add_callback('on_model_save',self.copy_endpoint)
    def get_model(self,cfg=None,weights=None,verbose=True):
        if self.initial_model is None:raise RuntimeError('Explicit common initial model required; no automatic independent head initialization')
        return copy.deepcopy(self.initial_model)
    def build_dataset(self,img_path,mode='train',batch=None):
        args=copy.copy(self.args);args.rect=False
        stride=max(int(unwrap_model(self.model).stride.max()) if self.model else 32,32)
        dataset=build_yolo_dataset(args,img_path,batch,self.data,mode=mode,rect=False,stride=stride)
        if dataset.rect:raise RuntimeError('Trainer rectangle override failed')
        return dataset
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch);observe(batch,self.save_dir/'actual_inputs.jsonl','train');return batch
    def get_validator(self):
        self.loss_names=('box_loss','cls_loss','dfl_loss')
        return SquareValidator(self.test_loader,save_dir=self.save_dir,args=copy.copy(self.args),_callbacks=self.callbacks)
    @staticmethod
    def route_epoch(trainer):
        model=unwrap_model(trainer.model)
        if hasattr(model,'route_epoch'):model.route_epoch=trainer.epoch
    @staticmethod
    def route_log(trainer):
        model=unwrap_model(trainer.model);row=dict(epoch=trainer.epoch,optimizer_steps=trainer.actual_optimizer_steps,
            accumulation=trainer.accumulate,lr=[g['lr'] for g in trainer.optimizer.param_groups],
            EMA_updates=trainer.ema.updates if trainer.ema else None,**getattr(model,'latest_route',{}))
        if hasattr(model.model[16],'adapter'):row['gamma']=float(model.model[16].adapter.gamma.detach())
        with (trainer.save_dir/'route_optimizer.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def optimizer_step(self):
        super().optimizer_step();self.actual_optimizer_steps+=1
    @staticmethod
    def copy_endpoint(trainer):
        if trainer.epoch+1==trainer.args.epochs:
            destination=trainer.save_dir/'weights'/('matched_epoch'+str(trainer.args.epochs)+'_unstripped.pt')
            shutil.copyfile(trainer.last,destination)
            dump(trainer.save_dir/'endpoint_identity.json',dict(completed_epoch=trainer.epoch+1,checkpoint_epoch=trainer.epoch,
                path=str(destination),sha256=file_sha256(destination),source_last_sha256=file_sha256(trainer.last),
                byte_copy_before_strip=True,optimizer_steps=trainer.actual_optimizer_steps,EMA_updates=trainer.ema.updates))
