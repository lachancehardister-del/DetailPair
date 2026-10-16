"""Executed design checks only: no Torch, model training, or real-data evidence."""
from pathlib import Path
import ast,hashlib,importlib.util,json,platform,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'sources'
OUT=ROOT/'checks';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shape_target(w,h):
    w,h=np.broadcast_arrays(np.asarray(w,dtype=float),np.asarray(h,dtype=float))
    if not (np.isfinite(w).all() and np.isfinite(h).all() and (w>0).all() and (h>0).all()):raise ValueError('finite positive widths/heights required')
    return np.stack([2*np.minimum(w,h),np.maximum(w-h,0),np.maximum(h-w,0)],axis=-1)/(w+h)[...,None]
def loss_and_grad(logits,q,eps=1e-6):
    a=np.exp(logits-np.max(logits));a/=a.sum()
    r=(a+eps)/(1+3*eps)
    keep=q>0
    loss=np.sum(q[keep]*(np.log(q[keep])-np.log(r[keep])))
    da=-q/(a+eps)
    grad=a*(da-np.dot(da,a))
    return float(loss),grad

rng=np.random.default_rng(178)
w,h=np.exp(rng.uniform(-5,5,(2,10000)));q=shape_target(w,h)
assert np.all(q>=0) and np.allclose(q.sum(-1),1)
assert np.allclose(shape_target(7*w,7*h),q)
assert np.allclose(shape_target(h,w),q[:,[0,2,1]])
assert np.allclose(shape_target(1,1),[1,0,0])
assert np.allclose(shape_target(7,1),[.25,.75,0])
assert np.allclose(shape_target(1,7),[.25,0,.75])
invalid=0
for a,b in [(0,1),(-1,2),(np.nan,1),(1,np.inf)]:
    try:shape_target(a,b)
    except ValueError:invalid+=1
assert invalid==4
errors=[];min_kl=1e9
for _ in range(128):
    z=rng.normal(size=3);target=shape_target(*np.exp(rng.normal(size=2)))
    loss,grad=loss_and_grad(z,target);min_kl=min(min_kl,loss)
    assert loss>=-1e-12
    numeric=[]
    for j in range(3):
        d=np.zeros(3);d[j]=1e-5
        numeric.append((loss_and_grad(z+d,target)[0]-loss_and_grad(z-d,target)[0])/(2e-5))
    errors.append(float(np.max(np.abs(grad-numeric))))
assert max(errors)<1e-8
shape_pairs=[(1,1),(7,1),(1,7),(2,1),(1,2)]
# Single input and common gamma=0 preserve the identity algebraically only.
# This is not the unexecuted PyTorch feature/loss/gradient parity test.
x=rng.normal(size=(2,4,7,7));t=rng.normal(size=x.shape)
assert np.array_equal(x+0*t,x)

project=ROOT.parent/'received/v029_archive_20261016/DetailPair_v0.2.9'
tree=ast.parse((project/'detailpair/model.py').read_text())
variants=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VARIANTS' for t in n.targets))
assert variants['B0']==dict(module='base',pair=False,cons=0.,energy=0.)
pair=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PairDetectionModel')
assert [n.name for n in pair.body if isinstance(n,ast.FunctionDef)]==['__init__','loss']
loss=next(n for n in pair.body if isinstance(n,ast.FunctionDef) and n.name=='loss')
branch=loss.body[0]
assert isinstance(branch,ast.If) and ast.unparse(branch.test)=='not self.training or not self.pair_config[\'pair\']'
assert ast.unparse(branch.body[0])=='return super().loss(batch, preds)'
official=(SOURCE/'ultralytics_v83221_ultralytics_utils_loss.py').read_text()
assert 'return loss * batch_size, loss.detach()' in official
assert 'alpha=0.5, beta=6.0' in official and 'self.bce(pred_scores, target_scores.to(dtype))' in official
yaml=(SOURCE/'ultralytics_v83221_ultralytics_cfg_models_11_yolo11.yaml').read_text()
assert '# 16 (P3/8-small)' in yaml and '[[16, 19, 22], 1, Detect' in yaml

C=64;HW=40*40
params=(23*C+6*C)+(3*C+3)+(C*C+2*C)+1
macs=(23*C+3*C+C*C)*HW
assert params==6276 and macs==9216000
result=dict(scope='Analytic identities and NumPy finite differences; not neural-network or real-data validation',
    python=sys.version,numpy=np.__version__,platform=platform.platform(),
    deps={n:importlib.util.find_spec(n) is not None for n in ['numpy','pandas','pydoe','requests','torch','ultralytics']},
    simplex_points=10000,scale_and_swap_checks=True,invalid_cases_rejected=invalid,
    KL_finite_difference_points=128,max_absolute_gradient_error=max(errors),minimum_KL_in_sample=min_kl,
    B0_parent_loss_branch_ast_verified=True,official_fixed_version_loss_scaling_verified=True,
    actual_model_parity_executed=False,Windows_CUDA_executed=False,actual_data_read=False,
    proposed_module_trainable_parameters_at_C64=params,convolution_only_MAC_at_40x40=macs,
    cost_excludes='BN, activations, softmax, weighted sums, memory traffic; no latency/FPS claim',
    examples=[dict(width=w,height=h,target=shape_target(w,h).tolist()) for w,h in shape_pairs])
(OUT/'design_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
