import copy
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv


def haar(x):
    # Orthogonal 2x2 Haar basis; odd borders use replication, not reversible there.
    x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode='replicate')
    a,b,c,d = x[...,0::2,0::2],x[...,0::2,1::2],x[...,1::2,0::2],x[...,1::2,1::2]
    return (a+b+c+d)/2, torch.stack(((a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),dim=1)


class DetailDown(nn.Module):
    def __init__(self, c1, c2, gated=True, floor=0.1):
        super().__init__()
        self.floor = floor
        self.gated = gated
        self.gates = nn.ModuleList(nn.Conv2d(2*c1,c1,1) for _ in range(3)) if gated else nn.ModuleList()
        self.low = nn.Conv2d(c1,c2,1,bias=False)
        self.detail = nn.Conv2d(3*c1,c2,1,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.capture = False
        self.cache = None

    def forward(self,x):
        low,raw = haar(x)
        if self.gated:
            gate = torch.stack([self.floor+(1-self.floor)*g(torch.cat((low,raw[:,i].abs()),1)).sigmoid()
                                for i,g in enumerate(self.gates)],1)
            detail = raw*gate
        else:
            detail = raw
        if self.capture:
            self.cache = (raw,detail,gate if self.gated else torch.ones_like(raw))
        return self.act(self.bn(self.low(low)+self.detail(detail.flatten(1,2))))


class SPDDown(nn.Module):
    """Space-to-depth plus convolution comparator; not full SPD-Conv paper reproduction."""
    def __init__(self,c1,c2):
        super().__init__()
        self.project=Conv(4*c1,c2,3,1)
    def forward(self,x):
        x=F.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2),mode='replicate')
        return self.project(torch.cat([x[...,i::2,j::2] for i in range(2) for j in range(2)],1))


class ParameterMatchedDown(nn.Module):
    """Ordinary convolution comparator matched by layer parameter count, not latency."""
    def __init__(self,c1,c2,target):
        super().__init__()
        def count(mid): return 9*c1*mid+2*mid+mid*c2+2*c2
        mid=min(range(1,4*c2+1),key=lambda m:abs(count(m)-target))
        self.project=nn.Sequential(Conv(c1,mid,3,2),Conv(mid,c2,1,1))
        self.match=dict(target_parameters=target,actual_parameters=count(mid),mid_channels=mid,
                        scope='layer parameters only; measure actual latency separately')
    def forward(self,x): return self.project(x)


def amplitude_loss(pred,target,mask,mode='region'):
    """Mean absolute subband amplitude; not squared physical energy."""
    if mode=='uniform':
        return region_loss(pred,target,torch.ones_like(mask),bg=0.)
    if mode=='shift':
        # Negative control keeps the same image/band amplitude distribution,
        # but breaks location correspondence; no additional random draws.
        target=torch.roll(target,shifts=(max(1,target.shape[-2]//2),max(1,target.shape[-1]//2)),dims=(-2,-1))
    return region_loss(pred,target,mask)


def fft_amplitude_loss(pred,target):
    """PCW-inspired objective comparator, not a reproduction of PCW-UDM."""
    a=torch.log1p(torch.fft.rfft2(pred.float(),norm='ortho').abs())
    b=torch.log1p(torch.fft.rfft2(target.detach().float(),norm='ortho').abs())
    return (a-b).abs().mean()


def gate_summary(raw,gated,gate,mask):
    result={}
    g=gate.detach().flatten(1,2)
    for name,m in [('fg',mask),('bg',1-mask)]:
        m=m.expand_as(g);count=m.sum()
        if count.item()==0:
            result['gate_'+name+'_mean']=None;result['gate_'+name+'_near_one']=None
        else:
            result['gate_'+name+'_mean']=float((g*m).sum()/count)
            result['gate_'+name+'_near_one']=float(((g>0.99)*m).sum()/count)
    result['amplitude_retention_ratio']=float(gated.detach().abs().sum()/raw.detach().abs().sum().clamp_min(1e-9))
    return result


def box_mask(batch, h, w, n, device):
    mask = torch.zeros(n,1,h,w,device=device)
    # Ultralytics training boxes after geometric augmentation: normalized xywh.
    for idx,box in zip(batch['batch_idx'].long().flatten(),batch['bboxes']):
        cx,cy,bw,bh=box.detach().tolist()
        x0=max(0,min(w,int((cx-bw/2)*w)))
        y0=max(0,min(h,int((cy-bh/2)*h)))
        x1=max(0,min(w,int(__import__('math').ceil((cx+bw/2)*w))))
        y1=max(0,min(h,int(__import__('math').ceil((cy+bh/2)*h))))
        mask[int(idx),:,y0:y1,x0:x1]=1
    return mask


def region_loss(pred,target,mask,bg=0.1):
    # Reference scaling is detached; absent regions contribute zero, not NaN.
    target=target.detach()
    err=(pred-target).abs().mean(1,keepdim=True)
    scale=target.abs().mean(1,keepdim=True)
    values=[]
    for m,weight in ((mask,1.0),(1-mask,bg)):
        count=m.sum((1,2,3))
        denominator=(scale*m).sum((1,2,3))/count.clamp_min(1)+1e-3
        term=(err*m).sum((1,2,3))/count.clamp_min(1)/denominator
        values.append(weight*(term*(count>0)).mean())
    return sum(values)


VARIANTS={
    'B0':dict(module='base',pair=False,cons=0.,energy=0.),
    'B1':dict(module='base',pair=True,cons=0.,energy=0.),
    'B2':dict(module='base',pair=True,cons=0.05,energy=0.),
    'B3':dict(module='haar',pair=True,cons=0.,energy=0.),
    'B4':dict(module='gate',pair=True,cons=0.,energy=0.),
    'B5':dict(module='gate',pair=True,cons=0.05,energy=0.),
    'B6':dict(module='gate',pair=True,cons=0.05,energy=0.02),
    'SPD':dict(module='spd',pair=True,cons=0.,energy=0.),
    'B7':dict(module='matched',pair=True,cons=0.05,energy=0.),
    'E_UNIFORM':dict(module='gate',pair=True,cons=0.05,energy=0.02,amplitude_mode='uniform'),
    'E_SHIFT':dict(module='gate',pair=True,cons=0.05,energy=0.02,amplitude_mode='shift'),
    'E_SIGNED':dict(module='gate',pair=True,cons=0.05,energy=0.02,amplitude_mode='signed'),
    'E_FFT':dict(module='gate',pair=True,cons=0.05,energy=0.,fft_weight=0.02),
}


class PairDetectionModel(DetectionModel):
    def __init__(self,cfg='yolo11n.yaml',ch=3,nc=6,verbose=False,variant='B6'):
        super().__init__(cfg,ch,nc,verbose)
        self.variant=variant
        self.pair_config=copy.deepcopy(VARIANTS[variant])
        self.noise_std=0.03
        self.brightness=0.05
        self.latest_aux={}
        self.research_layer=1  # P2/4 downsample; channels read from the pinned model.
        old=self.model[self.research_layer]
        c1,c2=old.conv.in_channels,old.conv.out_channels
        kind=self.pair_config['module']
        if kind in ('haar','gate'):
            new=DetailDown(c1,c2,gated=kind=='gate')
        elif kind=='matched':
            target=sum(p.numel() for p in DetailDown(c1,c2).parameters())
            new=ParameterMatchedDown(c1,c2,target)
        elif kind=='spd':
            new=SPDDown(c1,c2)
        else:
            new=None
        if new is not None:
            for attr in ('i','f','type','np'):
                setattr(new,attr,getattr(old,attr))
            new.type=type(new).__name__
            new.np=sum(p.numel() for p in new.parameters())
            self.model[self.research_layer]=new
        self.perturbation_step=0
        self.perturbation_seed=100003
        self.perturbation_generator=None

    def loss(self,batch,preds=None):
        if not self.training or not self.pair_config['pair']:
            return super().loss(batch,preds)
        if preds is not None:
            raise ValueError('Paired training requires fresh two-view forwards.')
        if getattr(self,'criterion',None) is None:
            self.criterion=self.init_criterion()
        layer=self.model[self.research_layer]
        cache=[]
        handle=layer.register_forward_hook(lambda module,inputs,out: cache.append(out))
        gated=isinstance(layer,DetailDown)
        if gated: layer.capture=True
        try:
            clean=self.predict(batch['img'])
            clean_loss,items_c=self.criterion(clean,batch)
            feat_c=cache[-1]
            raw_c=layer.cache[0] if gated else None
            noisy=dict(batch)
            # Per-step independent RNG avoids architecture initialization consuming
            # different perturbation draws; record step/seed for replay.
            gen=torch.Generator(device=batch['img'].device)
            gen.manual_seed(int(self.perturbation_seed)+int(self.perturbation_step))
            self.perturbation_step+=1
            offset=(torch.rand(batch['img'].shape[0],1,1,1,device=batch['img'].device,generator=gen)*2-1)*self.brightness
            noise=torch.randn(batch['img'].shape,device=batch['img'].device,dtype=batch['img'].dtype,generator=gen)
            noisy['img']=(batch['img']+noise*self.noise_std+offset).clamp(0,1)
            preds_n=self.predict(noisy['img'])
            noisy_loss,items_n=self.criterion(preds_n,noisy)
            feat_n=cache[-1]
            mask=box_mask(batch,*feat_n.shape[-2:],feat_n.shape[0],feat_n.device)
            cons=region_loss(feat_n,feat_c,mask) if self.pair_config['cons'] else feat_n.new_zeros(())
            energy=feat_n.new_zeros(())
            if self.pair_config['energy']:
                gated_n=layer.cache[1]
                # 3 separate subband energies, channel mean; no sign/detail recovery guarantee.
                mode=self.pair_config.get('amplitude_mode','region')
                if mode=='signed':
                    energy=region_loss(gated_n.flatten(1,2),raw_c.flatten(1,2),mask)
                else:
                    energy=amplitude_loss(gated_n.abs().mean(2),raw_c.abs().mean(2),mask,mode)
            fft=fft_amplitude_loss(feat_n,feat_c) if self.pair_config.get('fft_weight',0.) else feat_n.new_zeros(())
            aux=self.pair_config['cons']*cons+self.pair_config['energy']*energy+self.pair_config.get('fft_weight',0.)*fft
            self.latest_aux={'consistency':float(cons.detach()),'detail_amplitude':float(energy.detach()),'fft_amplitude':float(fft.detach()),'perturbation_step':self.perturbation_step,'weighted_aux':float(aux.detach())}
            if gated:
                self.latest_aux.update(gate_summary(*layer.cache,mask))
            # Standard Ultralytics detection loss already scales by batch size.
            return (clean_loss.sum()+noisy_loss.sum())/2+batch['img'].shape[0]*aux,(items_c+items_n)/2
        finally:
            handle.remove()
            if gated:
                layer.capture=False
                layer.cache=None
