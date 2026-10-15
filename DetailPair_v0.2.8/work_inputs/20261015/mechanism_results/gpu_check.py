"""Actual CUDA operator/model checks, without real data or downloaded weights."""
import copy, datetime, json, sys, traceback
from pathlib import Path
from run import environment, dump, ROOT, stamp

def main():
    out = ROOT / 'runs' / ('gpu_check_' + stamp())
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'environment.json', environment())
    rows = []
    try:
        import torch, torchvision, ultralytics
        from ultralytics import settings
        settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb')})
        from ultralytics.cfg import DEFAULT_CFG
        from detailpair.model import PairDetectionModel
        if torch.__version__.split('+')[0] != '2.5.1' or torchvision.__version__.split('+')[0] != '0.20.1' or ultralytics.__version__ != '8.3.221':
            raise RuntimeError('Requires Torch 2.5.1, torchvision 0.20.1 and Ultralytics 8.3.221')
        if torch.version.cuda != '12.1' or not torch.cuda.is_available():
            raise RuntimeError('CUDA 12.1 wheel with a working GPU required; CPU fallback forbidden in this check')
        dev = torch.device('cuda:0')
        boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device=dev)
        kept = torchvision.ops.nms(boxes, torch.tensor([.9,.8], device=dev), .5)
        torch.cuda.synchronize()
        if kept.tolist() != [0]: raise RuntimeError('Unexpected CUDA NMS output')
        dump(out / 'cuda_nms.json', dict(device=str(kept.device), indices=kept.tolist()))
        for variant in ['B1','B2','B5','B6']:
            torch.manual_seed(0)
            m = PairDetectionModel(nc=2, variant=variant).to(dev)
            m.args = copy.deepcopy(DEFAULT_CFG)
            m.train()
            x = torch.rand(2,3,64,64,device=dev)
            batch = dict(img=x,batch_idx=torch.tensor([0.,1.],device=dev),cls=torch.tensor([[0.],[1.]],device=dev),bboxes=torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]],device=dev))
            loss, _ = m(batch)
            total = loss.sum()
            if not torch.isfinite(total): raise RuntimeError('Non-finite loss')
            total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()): raise RuntimeError('Non-finite gradients')
            if variant == 'B6' and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.model[1].gates.parameters()): raise RuntimeError('B6 gate gradient missing')
            c = copy.deepcopy(m).eval()
            with torch.no_grad(): c(x)
            torch.cuda.synchronize()
            rows.append(dict(variant=variant,device=str(next(m.parameters()).device),loss=float(total.detach()),aux=m.latest_aux,forward_backward=True,ema_deepcopy=True))
            dump(out / 'model_checks.json', rows)
            del c, m, batch, loss, total, x
            torch.cuda.empty_cache()
        dump(out / 'status.json',dict(status='success',finished=datetime.datetime.now().astimezone().isoformat(),scope='CUDA NMS and four synthetic single-batch model checks only; no real-data performance evidence'))
        print('CUDA checks passed:', out)
        return 0
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        dump(out / 'status.json',dict(status='failed'))
        traceback.print_exc()
        return 1

if __name__ == '__main__': sys.exit(main())
