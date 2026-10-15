"""Mechanism counterexamples and implementation checks, not data evidence."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from ultralytics.cfg import DEFAULT_CFG
from detailpair.model import DetailDown,PairDetectionModel,region_loss,amplitude_loss,fft_amplitude_loss

torch.set_num_threads(2);torch.manual_seed(241)
x=torch.randn(2,4,16,18)
m=DetailDown(4,8,gated=False).eval()
l=m.low.weight[:,:,0,0];d=m.detail.weight[:,:,0,0].reshape(8,3,4)
w=torch.stack([l+d[:,0]+d[:,1]+d[:,2],l-d[:,0]+d[:,1]-d[:,2],l+d[:,0]-d[:,1]-d[:,2],l-d[:,0]-d[:,1]+d[:,2]],-1).reshape(8,4,2,2)/2
expected=m.act(m.bn(F.conv2d(x,w,stride=2)))
torch.testing.assert_close(m(x),expected,rtol=1e-5,atol=1e-6)
# Same raw amplitudes: all-pass gating strictly reduces auxiliary loss.
a=torch.rand(2,3,8,9)+0.1;mask=torch.ones(2,1,8,9)
losses=[float(amplitude_loss(a*g,a,mask)) for g in [0.1,0.55,0.99,1.0]]
assert all(losses[i]>losses[i+1] for i in range(3))
# Amplitude cannot distinguish sign: exact counterexample.
torch.testing.assert_close(amplitude_loss(a.abs(),(-a).abs(),mask),torch.tensor(0.))
assert region_loss(a,-a,mask)>0
# Foreground/background reweighting and deliberately misplaced references.
mask[:,:,0:3,0:4]=0
assert amplitude_loss(a,a,mask,'shift')>0
p=a.clone().requires_grad_();ref=a.clone().requires_grad_()
l=fft_amplitude_loss(p,ref);assert torch.isfinite(l);l.backward();assert ref.grad is None
# Perturbation replay does not depend on model-specific RNG consumption.
b=dict(img=torch.rand(2,3,64,64),batch_idx=torch.tensor([0.,1.]),cls=torch.tensor([[0.],[1.]]),bboxes=torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]]))
inputs=[]
for v in ['B2','B6','E_FFT']:
 torch.manual_seed(3);n=PairDetectionModel(nc=2,variant=v);n.args=copy.deepcopy(DEFAULT_CFG);n.train()
 capture=[];h=n.model[0].register_forward_pre_hook(lambda mod,args:capture.append(args[0].detach().clone()))
 loss,_=n(b);loss.sum().backward();h.remove();inputs.append(capture[-1])
for t in inputs[1:]:torch.testing.assert_close(t,inputs[0],rtol=0,atol=0)
n=PairDetectionModel(nc=2,variant='B7');info=n.model[1].match
assert abs(info['actual_parameters']/info['target_parameters']-1)<0.05
print(json.dumps(dict(haar_projection_equals_stride2_kernel2=True,allpass_counterexample_losses=losses,sign_ambiguity=True,shift_control=True,fft_stop_gradient=True,paired_perturbation_replay=True,parameter_match=info)))
