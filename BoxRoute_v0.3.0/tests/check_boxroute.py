"""Real PyTorch numerical/gradient and serialization checks, all synthetic except public initialization."""
import ast,copy,os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'local_settings'))
import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA
from boxroute.geometry import geometry_targets,route_loss,lambda_at
from boxroute.model import make_common,build_conditions,RouteBlock
from boxroute.pks import PKSModule
from boxroute.identity import dump,state_sha256

torch.set_num_threads(2)


def reference_PKS():
    source=(ROOT/'sources/pkinet_v2_fixed.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['_fuse_bn_tensor','PKSModule']]
    namespace=dict(torch=torch,nn=nn,build_norm_layer=lambda cfg,c:('BN',nn.BatchNorm2d(c)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'fixed reviewed PKSModule','exec'),namespace)
    return namespace['PKSModule']


class BoxRouteChecks(unittest.TestCase):
    def test_q_simplex_scale_swap_and_invalid(self):
        wh=torch.tensor([[1.,1.],[7.,1.],[1.,7.]],requires_grad=True);q=geometry_targets(wh)
        torch.testing.assert_close(q,torch.tensor([[1.,0.,0.],[.25,.75,0.],[.25,0.,.75]]));self.assertFalse(q.requires_grad)
        torch.testing.assert_close(q,geometry_targets(wh*12));torch.testing.assert_close(geometry_targets(wh[:,[1,0]]),geometry_targets(wh,True))
        for pair in [[0.,1.],[-1.,1.],[float('nan'),1.],[float('inf'),1.]]:
            with self.assertRaises(ValueError):geometry_targets(torch.tensor([pair]))
    def test_KL_ROI_empty_overlap_fallback_and_gradient(self):
        logits=torch.randn(2,3,4,4,dtype=torch.float64,requires_grad=True);boxes=torch.tensor([[.5,.5,.8,.1],[.5,.5,.8,.1],[.501,.501,.001,.001]],dtype=torch.float64)
        loss,info=route_loss(logits.softmax(1),boxes,torch.zeros(3),(320,320));self.assertEqual(info['empty_images'],1);self.assertGreater(info['fallback_instances'],0);self.assertGreater(info['overlap_instances'],0)
        self.assertTrue(torch.autograd.gradcheck(lambda x:route_loss(x.softmax(1),boxes,torch.zeros(3),(320,320))[0],(logits,),eps=1e-6,atol=1e-5))
        empty=route_loss(logits.softmax(1),torch.empty(0,4),torch.empty(0),(320,320))[0];self.assertEqual(float(empty),0.)
        # One box includes only left/top grid centre; right/bottom boundaries excluded.
        a=torch.tensor([[[[.8,.1],[.1,.1]],[[.1,.8],[.1,.1]],[[.1,.1],[.8,.8]]]],dtype=torch.float64,requires_grad=True)
        b=torch.tensor([[.375,.375,.25,.25]],dtype=torch.float64) # [0.25,0.5) covers centre(0.25,0.25)
        value,_=route_loss(a,b,torch.tensor([0]),(320,320));self.assertAlmostEqual(float(value),-__import__('math').log((.8+1e-6)/(1+3e-6)),places=10)
    def test_lambda_and_no_broadcast(self):

        for actual,expected in zip([lambda_at(x) for x in [0,4,5]],[.01,.05,.05]):self.assertAlmostEqual(actual,expected,places=14)
        for n in [1,2]:
            base=torch.tensor([1.,2.,3.]);aux=torch.tensor(2.)
            torch.testing.assert_close(base.sum()+n*lambda_at(4)*aux,torch.tensor(6+n*.1),atol=1e-6,rtol=1e-6)
    def test_PKS_original_port_gradients_deploy_all_edges(self):
        Reference=reference_PKS();a=Reference(4);b=PKSModule(4);b.load_state_dict(a.state_dict(),strict=True)
        xa=torch.randn(2,4,23,21,requires_grad=True);xb=xa.detach().clone().requires_grad_();ya=a(xa);yb=b(xb)
        torch.testing.assert_close(ya,yb,atol=1e-6,rtol=1e-5);ya.square().mean().backward();yb.square().mean().backward()
        torch.testing.assert_close(xa.grad,xb.grad,atol=1e-6,rtol=1e-5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        for model in [a,b]:
            model.eval();x=torch.randn(2,4,23,21)
            with torch.no_grad():before=model(x);model.switch_to_deploy();after=model(x)
            torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-5)
    def test_common_initialization_full_models_loss_ema_reload_fuse(self):
        native,identity=make_common(ROOT/'sources/official_yolo11n_v830.pt',['a','b','c','d','e','f']);models=build_conditions(native)
        self.assertEqual(identity['probed_layer16'],[1,64,40,40]);self.assertEqual(sum(p.numel() for p in models['G'].model[16].adapter.parameters()),6276)
        image=torch.rand(2,3,320,320);batch=dict(img=image,batch_idx=torch.tensor([0.,1.]),cls=torch.tensor([[0.],[1.]]),bboxes=torch.tensor([[.5,.5,.7,.1],[.5,.5,.1,.7]]))
        native.eval()
        with torch.no_grad():expected=native(image)[0]
        statuses={}
        for variant,model in models.items():
            model.eval()
            with torch.no_grad():torch.testing.assert_close(model(image)[0],expected,atol=1e-6,rtol=1e-5)
            model.args=copy.deepcopy(get_cfg());model.train();total,items=model.loss(batch);total.sum().backward()
            self.assertEqual(list(items.shape),[3]);self.assertTrue(torch.isfinite(total).all())
            if variant in ['U','G','W']:
                self.assertEqual(len(model.model[16].adapter.router._forward_hooks),0)
                if variant in ['G','W']:
                    self.assertAlmostEqual(float(total),model.latest_route['native_total']+model.latest_route['weighted_aux'],places=4)
                    self.assertGreater(float(model.model[16].adapter.router[0].weight.grad.norm()),0.)
            ema=ModelEMA(model);ema.update(model);copy.deepcopy(model)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'fixture.pt';torch.save(dict(model=ema.ema),path);loaded=torch.load(path,weights_only=False)['model'].float().eval()
                with torch.no_grad():before=loaded(image)[0];loaded.fuse(verbose=False);after=loaded(image)[0]
                torch.testing.assert_close(before,after,atol=1e-4,rtol=1e-4)
            statuses[variant]=dict(single_batch_loss_backward=True,EMA_deepcopy_reload=True,fusion_close=True)
        dump(ROOT/'verification/five_condition_cpu.json',dict(status='passed',scope='Synthetic full models, no NEU result',initialization=identity,conditions=statuses,UGW_state_identical=True))
    def test_hook_cleanup_injected_failure(self):
        from boxroute.model import BoxRouteModel
        model=BoxRouteModel('G');model.args=copy.deepcopy(get_cfg());model.train()
        def fail(*args):raise RuntimeError('injected criterion failure')
        model.criterion=fail
        with self.assertRaises(RuntimeError):model.loss(dict(img=torch.rand(1,3,64,64)))
        self.assertEqual(len(model.model[16].adapter.router._forward_hooks),0);copy.deepcopy(model)

if __name__=='__main__':unittest.main(verbosity=2)
