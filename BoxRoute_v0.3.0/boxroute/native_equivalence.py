"""Numerical old-B0/native equivalence on synthetic tensors, never real baseline evidence."""
import copy,inspect
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from legacy_detailpair.model import PairDetectionModel
from .identity import state_sha256


def compare_tensors(a,b,atol=1e-6,rtol=1e-5):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Tensor shape/dtype differs')
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol,equal_nan=False)
    aa,bb=a.detach().double(),b.detach().double();delta=(aa-bb).abs()
    return dict(max_abs=float(delta.max()) if delta.numel() else 0.,
        max_rel=float((delta/torch.maximum(aa.abs(),bb.abs()).clamp_min(atol)).max()) if delta.numel() else 0.)


def flattened(value):
    if isinstance(value,torch.Tensor):return [value]
    if isinstance(value,(tuple,list)):return [x for v in value for x in flattened(v)]
    if isinstance(value,dict):return [x for key in sorted(value) for x in flattened(value[key])]
    raise TypeError('Unexpected native output '+str(type(value)))


def compare_named(a,b):
    if a.keys()!=b.keys():raise AssertionError('Common parameter or buffer names differ')
    result=dict(max_abs=0.,max_rel=0.,tensors=len(a),both_none=0)
    for name in a:
        x,y=a[name],b[name]
        if x is None or y is None:
            if x is not None or y is not None:raise AssertionError('Gradient connectivity differs: '+name)
            result['both_none']+=1;continue
        error=compare_tensors(x,y)
        for key in ['max_abs','max_rel']:result[key]=max(result[key],error[key])
    return result


def run_native_equivalence(device='cpu'):
    """8 size/batch/GT cases. A single AdamW step per case is synthetic engineering only."""
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; no fallback')
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(0);native_initial=DetectionModel('yolo11n.yaml',nc=6,verbose=False)
        legacy_initial=PairDetectionModel(nc=6,variant='B0')
    if native_initial.state_dict().keys()!=legacy_initial.state_dict().keys():raise AssertionError('B0/native state keys differ')
    common=copy.deepcopy(native_initial.state_dict());legacy_initial.load_state_dict(common,strict=True)
    if state_sha256(common)!=state_sha256(legacy_initial.state_dict()):raise AssertionError('Initial state differs')
    records=[]
    for size in [64,320]:
        for n in [1,2]:
            for has_GT in [False,True]:
                native=copy.deepcopy(native_initial).to(device).float();legacy=copy.deepcopy(legacy_initial).to(device).float()
                native.args=copy.deepcopy(get_cfg());legacy.args=copy.deepcopy(native.args)
                gen=torch.Generator(device=device).manual_seed(178)
                image=torch.rand(n,3,size,size,device=device,generator=gen)
                if has_GT:
                    batch=dict(img=image,batch_idx=torch.arange(n,device=device,dtype=torch.float32),
                        cls=torch.arange(n,device=device,dtype=torch.float32).remainder(6).view(n,1),
                        bboxes=torch.tensor([[.5,.5,.4,.3]],device=device).repeat(n,1))
                else:
                    batch=dict(img=image,batch_idx=torch.empty(0,device=device),cls=torch.empty(0,1,device=device),bboxes=torch.empty(0,4,device=device))
                native.eval();legacy.eval()
                with torch.no_grad():
                    a,b=flattened(native(image)),flattened(legacy(image))
                    if len(a)!=len(b):raise AssertionError('Output tree differs')
                    errors=[compare_tensors(x,y) for x,y in zip(a,b)]
                native.train();legacy.train()
                na,ni=native.loss(batch);la,li=legacy.loss(batch)
                if ni.shape!=torch.Size([3]) or li.shape!=torch.Size([3]):raise AssertionError('Official box/cls/dfl components missing')
                total_error=compare_tensors(na.sum(),la.sum());component_error=compare_tensors(ni,li)
                criterion=type(native.criterion).__module__+'.'+type(native.criterion).__name__
                if type(native.criterion)!=type(legacy.criterion) or type(native.criterion.assigner)!=type(legacy.criterion.assigner):raise AssertionError('Criterion/TAL differs')
                na.sum().backward();la.sum().backward()
                grad=compare_named({k:p.grad for k,p in native.named_parameters()},{k:p.grad for k,p in legacy.named_parameters()})
                buffers=compare_named(dict(native.named_buffers()),dict(legacy.named_buffers()))
                optimizer=dict(lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0005,foreach=False)
                op_n=torch.optim.AdamW(native.parameters(),**optimizer);op_b=torch.optim.AdamW(legacy.parameters(),**optimizer)
                op_n.step();op_b.step()
                step=compare_named(dict(native.named_parameters()),dict(legacy.named_parameters()))
                records.append(dict(size=size,batch=n,GT='nonempty' if has_GT else 'empty',
                    eval=max(errors,key=lambda x:x['max_abs']),detection_total=total_error,box_cls_dfl=component_error,
                    gradients=grad,BN_and_other_buffers=buffers,after_one_step=step,criterion=criterion,
                    TAL=type(native.criterion.assigner).__module__+'.'+type(native.criterion.assigner).__name__,
                    native_loss_vector_shape=list(na.shape),optimizer=optimizer,status='passed'))
                del op_n,op_b,native,legacy,na,la,ni,li
    return dict(status='passed',scope='Synthetic engineering equality only; no real B0/native training result',
        device=str(device),atol=1e-6,rtol=1e-5,tolerance_basis='Task-suggested FP32 tolerances; no relaxation; raw maxima retained',
        shared_state_sha256=state_sha256(common),state_keys=len(common),records=records,
        init_criterion_source=inspect.getsource(DetectionModel.init_criterion),
        official_loss_source=inspect.getsource(DetectionModel.loss),
        scaling='Both native and B0 use official criterion without auxiliary terms; total is returned native vector.sum()')
