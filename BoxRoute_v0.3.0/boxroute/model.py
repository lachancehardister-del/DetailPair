"""Native YOLO11n plus one post-layer16 adapter; no image-pair or legacy auxiliary losses."""
import copy
from pathlib import Path
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .geometry import route_loss,lambda_at
from .identity import file_sha256,state_sha256
from .pks import PKSModule
PUBLIC_SHA='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


class RouteBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.experts=nn.ModuleList(nn.Sequential(nn.Conv2d(c,c,k,padding=(k[0]//2,k[1]//2),groups=c,bias=False),nn.BatchNorm2d(c),nn.SiLU()) for k in [(3,3),(1,7),(7,1)])
        self.router=nn.Sequential(nn.Conv2d(c,3,1,bias=True),nn.Softmax(dim=1));nn.init.zeros_(self.router[0].weight);nn.init.zeros_(self.router[0].bias)
        self.project=nn.Sequential(nn.Conv2d(c,c,1,bias=False),nn.BatchNorm2d(c));self.gamma=nn.Parameter(torch.zeros(()));self.diagnostics={}
    def forward(self,x):
        alpha=self.router(x);features=[expert(x) for expert in self.experts]
        weighted=[feature*alpha[:,i:i+1] for i,feature in enumerate(features)]
        if self.training:
            self.diagnostics=dict(gamma=float(self.gamma.detach()),alpha_mean=alpha.detach().mean((0,2,3)).tolist(),
                expert_L2=[float(t.detach().norm()) for t in features],weighted_expert_L2=[float(t.detach().norm()) for t in weighted])
        return x+self.gamma*self.project(sum(weighted))  # no gamma-zero shortcut
    def fuse(self):
        if self.training:raise RuntimeError('Fusion requires eval')
        for seq in [*self.experts,self.project]:
            if len(seq)>1 and isinstance(seq[1],nn.BatchNorm2d):seq[0]=fuse_conv_and_bn(seq[0],seq[1]);seq[1]=nn.Identity()


class PKSAdapter(nn.Module):
    def __init__(self,c):
        super().__init__();self.pks=PKSModule(c,branch_scale=1.);self.gamma=nn.Parameter(torch.zeros(()))
    def forward(self,x):return x+self.gamma*self.pks(x)


class PostLayer16(nn.Module):
    def __init__(self,original,adapter):
        super().__init__();self.original=original;self.adapter=adapter
        self.original_metadata={k:getattr(original,k) for k in ['i','f','type','np']}
        self.i=original.i;self.f=original.f;self.type='BoxRoutePost('+original.type+')';self.np=sum(p.numel() for p in self.parameters())
    def forward(self,x):return self.adapter(self.original(x))


def probe_layer16(model):
    outputs=[];mode=model.training;h=model.model[16].register_forward_hook(lambda m,a,o:outputs.append(list(o.shape)))
    try:
        model.eval()
        with torch.no_grad():model(torch.zeros(1,3,320,320,device=next(model.parameters()).device))
    finally:h.remove();model.train(mode)
    if len(outputs)!=1 or outputs[0][2:]!=[40,40] or model.model[-1].f!=[16,19,22]:raise RuntimeError('Unexpected fixed YOLO11 P3/Detect topology')
    return outputs[0]


class BoxRouteModel(DetectionModel):
    def __init__(self,variant='G',nc=6):
        if variant not in ['S','U','G','W']:raise ValueError('Unknown candidate condition')
        super().__init__('yolo11n.yaml',nc=nc,verbose=False);self.variant=variant;self.route_epoch=0;self.latest_route={}
        self.probed_layer16=probe_layer16(self);c=self.probed_layer16[1]
        self.model[16]=PostLayer16(self.model[16],PKSAdapter(c) if variant=='S' else RouteBlock(c))
    def loss(self,batch,preds=None):
        if not self.training or self.variant=='S':return super().loss(batch,preds)
        if preds is not None:raise ValueError('Route training requires one fresh native forward')
        block=self.model[16].adapter;captured=[]
        handle=block.router.register_forward_hook(lambda m,a,o:captured.append(o))
        try:
            native,items=super().loss(batch)
            self.latest_route=dict(block.diagnostics)
            aux=native.new_zeros(());coefficient=0.
            if self.variant in ['G','W']:
                if len(captured)!=1:raise RuntimeError('Route captured more than one forward')
                aux,metadata=route_loss(captured[0],batch['bboxes'],batch['batch_idx'],batch['img'].shape[-2:],swap=self.variant=='W')
                coefficient=lambda_at(self.route_epoch);self.latest_route.update(metadata)
            weighted=batch['img'].shape[0]*coefficient*aux
            self.latest_route.update(route_loss=float(aux.detach()) if self.variant in ['G','W'] else None,
                lambda_value=coefficient,batch=batch['img'].shape[0],weighted_aux=float(weighted.detach()),native_total=float(native.sum().detach()))
            return native.sum()+weighted,items
        finally:handle.remove();captured.clear()
    def fuse(self,verbose=True):
        super().fuse(verbose)
        for module in self.modules():
            if isinstance(module,RouteBlock):module.fuse()
            if isinstance(module,PKSAdapter):module.pks.switch_to_deploy()
        return self


def make_common(public_path,names):
    path=Path(public_path)
    if path.stat().st_size!=5613764 or file_sha256(path)!=PUBLIC_SHA:raise ValueError('Official initialization identity differs')
    ckpt=torch.load(path,map_location='cpu',weights_only=False);source=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
    if not isinstance(source,DetectionModel):raise ValueError('Public checkpoint is not a native DetectionModel')
    with torch.random.fork_rng():
        torch.manual_seed(0);native=DetectionModel('yolo11n.yaml',nc=len(names),verbose=False)
    original=source.float().state_dict();target=native.state_dict();transferred={k:v for k,v in original.items() if k in target and v.shape==target[k].shape}
    native.load_state_dict(transferred,strict=False);native.names=dict(enumerate(names))
    return native,dict(public_sha256=PUBLIC_SHA,transferred_keys=sorted(transferred),
        untransferred_nc6_keys=sorted(set(target)-set(transferred)),source_unmatched_keys=sorted(set(original)-set(transferred)),
        common_nc6_state_sha256=state_sha256(native.state_dict()),seed=0,probed_layer16=probe_layer16(native))


def build_conditions(native):
    models={'N':copy.deepcopy(native)};base=native.state_dict()
    for variant in ['U','G','W','S']:
        with torch.random.fork_rng():
            torch.manual_seed(1);model=BoxRouteModel(variant,nc=len(native.names))
        mapping={('model.16.original.'+k[len('model.16.'):] if k.startswith('model.16.') else k):v for k,v in base.items()}
        missing=model.load_state_dict(mapping,strict=False).missing_keys
        if any(not k.startswith('model.16.adapter.') for k in missing):raise RuntimeError('Common tensor mapping incomplete')
        model.names=copy.deepcopy(native.names)
        if variant in ['G','W']:model.load_state_dict(models['U'].state_dict(),strict=True)
        for k,value in mapping.items():
            if not torch.equal(model.state_dict()[k],value):raise RuntimeError('A common native tensor changed')
        models[variant]=model
    if len({state_sha256(models[v].state_dict()) for v in ['U','G','W']})!=1:raise RuntimeError('U/G/W initial states differ')
    return models
