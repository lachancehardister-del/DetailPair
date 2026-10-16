"""Executable full-model checks, always labelled engineering rather than detection evidence."""
import copy,tempfile
from pathlib import Path
import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA
from .identity import dump,state_sha256


def conv_cost(model):
    hooks=[];mac=[0];original=model.training;model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.Conv2d):
            def count(m,a,o):mac[0]+=o.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            hooks.append(module.register_forward_hook(count))
    try:
        with torch.no_grad():model(torch.zeros(1,3,320,320,device=next(model.parameters()).device))
    finally:
        for h in hooks:h.remove()
        model.train(original)
    return dict(parameters=sum(p.numel() for p in model.parameters()),conv_MAC=mac[0],scope='Conv2d multiply-accumulates at batch1 320; excludes BN/softmax/activation/transport/kernel launch')


def validate_models(models,device,out):
    device=torch.device(device);generator=torch.Generator(device=device).manual_seed(178);image=torch.rand(2,3,320,320,device=device,generator=generator)
    batch=dict(img=image,batch_idx=torch.tensor([0.,1.],device=device),cls=torch.tensor([[0.],[1.]],device=device),bboxes=torch.tensor([[.5,.5,.7,.1],[.5,.5,.1,.7]],device=device))
    native=copy.deepcopy(models['N']).to(device).float().eval()
    with torch.no_grad():expected=native(image)[0]
    report={}
    for variant,source in models.items():
        model=copy.deepcopy(source).to(device).float().eval();cost=conv_cost(model)
        with torch.no_grad():actual=model(image)[0]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        initial_error=float((actual-expected).abs().max());model.args=copy.deepcopy(get_cfg());model.train();total,items=model.loss(batch)
        if not torch.isfinite(total).all() or items.shape!=torch.Size([3]):raise RuntimeError('Loss invalid: '+variant)
        total.sum().backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradients')
        if variant in ['U','G','W'] and model.model[16].adapter.router._forward_hooks:raise RuntimeError('Leaked route hook')
        ema=ModelEMA(model);ema.update(model);copy.deepcopy(model)
        with tempfile.TemporaryDirectory(prefix='BoxRoute engineering ') as temp:
            path=Path(temp)/'roundtrip.pt';torch.save(dict(model=ema.ema),path);reloaded=torch.load(path,map_location=device,weights_only=False)['model'].float().eval()
            with torch.no_grad():before=reloaded(image)[0];reloaded.fuse(verbose=False);after=reloaded(image)[0]
            torch.testing.assert_close(before,after,atol=1e-4,rtol=1e-4)
            fusion_error=float((before-after).abs().max())
        report[variant]=dict(status='passed',cost=cost,initial_prediction_max_abs=initial_error,
            finite_loss_gradients=True,EMA_deepcopy_reload=True,fusion_max_abs=fusion_error,fusion_atol=1e-4,fusion_rtol=1e-4,
            scope='Synthetic engineering, not NEU performance',diagnostics=getattr(model,'latest_route',{}))
        del model,ema,reloaded,total,items
    report['overhead']=dict(parameter_ratio_G_to_N=report['G']['cost']['parameters']/report['N']['cost']['parameters'],
        conv_MAC_ratio_G_to_N=report['G']['cost']['conv_MAC']/report['N']['cost']['conv_MAC'])
    dump(Path(out)/'five_condition_checks.json',report);return report


def verify_PKS_and_geometry(device):
    import ast
    from torch import nn
    from .pks import PKSModule
    from .geometry import route_loss,geometry_targets
    source=(Path(__file__).resolve().parents[1]/'sources/pkinet_v2_fixed.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['_fuse_bn_tensor','PKSModule']]
    namespace=dict(torch=torch,nn=nn,build_norm_layer=lambda cfg,c:('BN',nn.BatchNorm2d(c)))
    exec(compile(ast.Module(body=selected,type_ignores=[]),'audited fixed PKS-only definitions','exec'),namespace)
    original=namespace['PKSModule'](4).to(device);port=PKSModule(4).to(device);port.load_state_dict(original.state_dict(),strict=True)
    x=torch.randn(2,4,23,21,device=device,requires_grad=True);xx=x.detach().clone().requires_grad_()
    a,b=original(x),port(xx);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    a.square().mean().backward();b.square().mean().backward();torch.testing.assert_close(x.grad,xx.grad,atol=1e-6,rtol=1e-5)
    for p,q in zip(original.parameters(),port.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    errors=[]
    for m in [original,port]:
        m.eval()
        with torch.no_grad():before=m(x);m.switch_to_deploy();after=m(x)
        torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-5);errors.append(float((before-after).abs().max()))
    q=geometry_targets(torch.tensor([[1.,1.],[7.,1.],[1.,7.]],device=device))
    torch.testing.assert_close(q,torch.tensor([[1.,0.,0.],[.25,.75,0.],[.25,0.,.75]],device=device))
    logits=torch.randn(2,3,4,4,device=device,requires_grad=True)
    boxes=torch.tensor([[.5,.5,.001,.001],[.5,.5,.001,.001]],device=device)
    loss,info=route_loss(logits.softmax(1),boxes,torch.zeros(2,device=device),(320,320));loss.backward()
    if not torch.isfinite(logits.grad).all() or info['fallback_instances']!=2 or info['overlap_instances']!=2 or info['empty_images']!=1:raise RuntimeError('Geometry fallback/overlap/empty/gradient check failed')
    return dict(status='passed',device=str(device),PKS_original_port_output_and_gradients='passed',PKS_deploy_max_abs=errors,all_edges_included=True,geometry=info)
