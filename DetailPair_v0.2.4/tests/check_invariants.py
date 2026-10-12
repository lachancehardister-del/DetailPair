"""Numerical checks; these do not measure defect performance."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from detailpair.model import haar,DetailDown,region_loss,box_mask,PairDetectionModel

torch.set_num_threads(2)
torch.manual_seed(123)
x=torch.randn(2,4,16,18)
low,details=haar(x)
torch.testing.assert_close(x.square().sum(),low.square().sum()+details.square().sum())
a,b,c=details[:,0],details[:,1],details[:,2]
r=torch.empty_like(x)
r[...,0::2,0::2]=(low+a+b+c)/2
r[...,0::2,1::2]=(low-a+b-c)/2
r[...,1::2,0::2]=(low+a-b-c)/2
r[...,1::2,1::2]=(low-a-b+c)/2
torch.testing.assert_close(x,r)
m=DetailDown(4,8);m.capture=True;y=m(x);raw,gated,gate=m.cache
assert (gated.abs()<=raw.abs()+1e-6).all()
assert (gated.abs()+1e-6>=0.1*raw.abs()).all()
empty=torch.zeros(2,1,*y.shape[-2:]);p=y.clone().requires_grad_()
target=y.detach().clone().requires_grad_()
l=region_loss(p,target,empty);assert torch.isfinite(l);l.backward()
assert target.grad is None
torch.testing.assert_close(l,torch.zeros_like(l))
batch=dict(batch_idx=torch.tensor([0]),bboxes=torch.tensor([[0.5,0.5,1.,1.]]))
mask=box_mask(batch,8,9,2,'cpu');assert mask[0].sum()==72 and mask[1].sum()==0
torch.manual_seed(8);base=PairDetectionModel(nc=2,variant='B2').state_dict()
torch.manual_seed(8);candidate=PairDetectionModel(nc=2,variant='B6').state_dict()
for key,value in base.items():
    if not key.startswith('model.1.'):
        torch.testing.assert_close(value,candidate[key],rtol=0,atol=0)
print(json.dumps(dict(haar_energy=True,haar_inverse=True,bounded_gate=True,empty_regions_finite=True,reference_stop_gradient=True,box_mask=True,shared_initialization_identical=True)))
