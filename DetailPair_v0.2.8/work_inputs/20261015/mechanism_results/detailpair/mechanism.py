"""Frozen endpoint diagnostics built from the unchanged model loss primitives."""
import hashlib,json,math,copy
import numpy as np
import torch
from torch import nn
from .model import DetailDown,region_loss,amplitude_loss,box_mask


def tensor_identity(t):
    a=t.detach().cpu().contiguous()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())


def state_identity(model):
    def hash_named(items):
        h=hashlib.sha256();count=0
        for name,t in items:
            info=tensor_identity(t);h.update(json.dumps([name,info],sort_keys=True).encode());count+=t.numel()
        return dict(sha256=h.hexdigest(),elements=count)
    params=hash_named(model.named_parameters())
    bn=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            for key,t in module.named_buffers(recurse=False):bn.append((name+'.'+key,t))
    return dict(parameters=params,batchnorm_buffers=hash_named(bn))


def prepare_loss_mode(model,training_args=None):
    """Training loss routes, fixed BN/random layers, gradients only at research layer1."""
    from types import SimpleNamespace
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise RuntimeError('Autograd must be enabled for local diagnosis')
    if training_args is not None:
        if not all(k in training_args for k in ['box','cls','dfl']):raise ValueError('Checkpoint train_args missing original box/cls/dfl; no guessed defaults')
        model.args=SimpleNamespace(**copy.deepcopy(training_args))
    elif isinstance(model.args,dict):model.args=SimpleNamespace(**model.args)
    for key in ['box','cls','dfl']:
        if not hasattr(model.args,key):raise ValueError('Missing resolved loss hyperparameter '+key)
    model.train()
    disabled=[]
    for name,m in model.named_modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm) or isinstance(m,nn.modules.dropout._DropoutNd) or type(m).__name__.lower() in {'droppath','stochasticdepth'}:
            m.eval();disabled.append(dict(name=name,type=type(m).__name__,training=m.training))
    if model.research_layer!=1:raise ValueError('Expected layer1')
    layer=model.model[1]
    if not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in layer.modules()):raise ValueError('Unfused research layer with BN required')
    original_requires_grad={name:p.requires_grad for name,p in model.named_parameters()}
    for p in model.parameters():p.requires_grad_(False)
    for p in layer.parameters():p.requires_grad_(True)
    # A stale checkpoint criterion may reference an old device/hyperparameter object.
    model.criterion=None
    model.noise_std=.03;model.brightness=.05;model.perturbation_seed=100003;model.perturbation_step=0
    if isinstance(layer,DetailDown) and (not layer.gated or layer.floor!=.1):raise ValueError('Original B6 bounded gate required')
    named=list(layer.named_parameters())
    groups={'whole_layer':list(range(len(named)))}
    if model.variant=='B6':
        groups={'gates':[i for i,(n,p) in enumerate(named) if n.startswith('gates.')],
                'projections':[i for i,(n,p) in enumerate(named) if n.startswith(('low.','detail.'))],
                'whole_layer':list(range(len(named)))}
    return named,groups,dict(disabled_or_fixed_modules=disabled,original_requires_grad=original_requires_grad,
        enabled_parameters=['model.1.'+n for n,p in named],loss_hyperparameters={k:getattr(model.args,k) for k in ['box','cls','dfl']},
        loss_hyperparameter_source='checkpoint train_args' if training_args is not None else 'explicit in-memory fixture args',
        mode='endpoint weights; unfused training loss route; BN eval; random layers eval; no parameter or BN updates')


def paired_perturbation(model,image,step):
    gen=torch.Generator(device=image.device);seed=int(model.perturbation_seed)+int(step);gen.manual_seed(seed)
    offset=(torch.rand(image.shape[0],1,1,1,device=image.device,generator=gen)*2-1)*model.brightness
    noise=torch.randn(image.shape,device=image.device,dtype=image.dtype,generator=gen)
    noisy=(image+noise*model.noise_std+offset).clamp(0,1)
    return noisy,dict(seed=seed,step=step,offsets=offset.detach().cpu().flatten().tolist(),
        standard_noise=tensor_identity(noise),brightness_offsets=tensor_identity(offset),noisy_image=tensor_identity(noisy),
        noise_std=model.noise_std,brightness_range=[-model.brightness,model.brightness])


def region_components(pred,target,mask,bg=.1):
    """Exact FG/BG algebra of unchanged region_loss, including absent-image and batch mean."""
    target=target.detach();err=(pred-target).abs().mean(1,keepdim=True);scale=target.abs().mean(1,keepdim=True)
    values=[]
    for m,weight in [(mask,1.),(1-mask,bg)]:
        count=m.sum((1,2,3));denominator=(scale*m).sum((1,2,3))/count.clamp_min(1)+1e-3
        term=(err*m).sum((1,2,3))/count.clamp_min(1)/denominator
        values.append(weight*(term*(count>0)).mean())
    return values[0],values[1]


def distribution(values,gate=False,floor=.1):
    values=values.detach().double().cpu().flatten()
    if not values.numel():return dict(status='empty_region',n=0,mean=None,median=None,quantiles=None,near_one=None,near_floor=None)
    if not torch.isfinite(values).all():raise ValueError('Nonfinite gate/amplitude values')
    q=[.01,.05,.25,.5,.75,.95,.99];quantiles=torch.quantile(values,torch.tensor(q,dtype=torch.float64)).tolist()
    result=dict(status='ok',n=values.numel(),mean=float(values.mean()),median=quantiles[3],
        min=float(values.min()),max=float(values.max()),quantiles={str(k):v for k,v in zip(q,quantiles)})
    if gate:result.update(near_one=float((values>.99).double().mean()),near_floor=float((values<=floor+.01).double().mean()),
        near_one_definition='g>0.99',near_floor_definition='g<=floor+0.01',floor=floor)
    return result


def band_statistics(raw_clean,raw_noisy,gated_noisy,gate_clean,gate_noisy,mask,floor):
    target=raw_clean.detach().abs().mean(2);pred=gated_noisy.detach().abs().mean(2)
    scale=target.abs().mean(1,keepdim=True);error=(pred-target).abs()
    rows=[]
    for band in range(3):
        row=dict(subband=band,regions={})
        for label,m,weight in [('fg_box_union_proxy',mask,1.),('bg_box_complement_proxy',1-mask,.1)]:
            selected=m.bool().expand_as(gate_noisy[:,band])
            count=m.sum((1,2,3));denominator=(scale*m).sum((1,2,3))/count.clamp_min(1)+1e-3
            # Share original three-channel denominator; /3 apportions the original channel mean.
            term=(error[:,band:band+1]*m).sum((1,2,3))/3/count.clamp_min(1)/denominator
            contribution=weight*(term*(count>0)).mean()
            row['regions'][label]=dict(gate=distribution(gate_noisy[:,band][selected],True,floor),
                clean_gate=distribution(gate_clean[:,band][selected],True,floor),
                raw_noisy_abs=distribution(raw_noisy[:,band].detach().abs()[selected]),
                gated_noisy_abs=distribution(gated_noisy[:,band].detach().abs()[selected]),
                clean_reference_abs=distribution(raw_clean[:,band].detach().abs()[selected]),
                original_region_weight=weight,amplitude_loss_unweighted_contribution=float(contribution))
        rows.append(row)
    return rows


def decomposed_loss(model,batch,step):
    """Fresh exact graph, original primitives/stop-grad/batch scaling; always clears capture."""
    if model.variant not in {'B2','B6'}:raise ValueError('Only frozen B2/B6 objectives supported')
    if not model.training:raise ValueError('Training loss route required, with BN separately fixed')
    if getattr(model,'criterion',None) is None:model.criterion=model.init_criterion()
    model.perturbation_step=step;layer=model.model[1];cache=[]
    handle=layer.register_forward_hook(lambda module,args,out:cache.append(out))
    gated=isinstance(layer,DetailDown)
    if gated:layer.capture=True
    try:
        clean=model.predict(batch['img']);clean_loss,_=model.criterion(clean,batch)
        feat_c=cache[-1];raw_c=layer.cache[0] if gated else None;gate_c=layer.cache[2] if gated else None
        noisy=dict(batch);noisy['img'],replay=paired_perturbation(model,batch['img'],step);model.perturbation_step=step+1
        output=model.predict(noisy['img']);noisy_loss,_=model.criterion(output,noisy);feat_n=cache[-1]
        mask=box_mask(batch,*feat_n.shape[-2:],feat_n.shape[0],feat_n.device)
        n=batch['img'].shape[0]
        D=(clean_loss.sum()+noisy_loss.sum())/2
        c=region_loss(feat_n,feat_c,mask);c_fg,c_bg=region_components(feat_n,feat_c,mask)
        terms={'D':D,'C':n*(model.pair_config['cons']*c),
               'C_fg':n*(model.pair_config['cons']*c_fg),'C_bg':n*(model.pair_config['cons']*c_bg)}
        raw=dict(D=float(D.detach()),clean_detection_sum=float(clean_loss.sum().detach()),noisy_detection_sum=float(noisy_loss.sum().detach()),
            C=float(c.detach()),C_fg=float(c_fg.detach()),C_bg=float(c_bg.detach()),A='not_applicable',A_fg='not_applicable',A_bg='not_applicable')
        gate_info='not_applicable'
        if model.variant=='B6':
            raw_n,gated_n,gate_n=layer.cache
            pred=gated_n.abs().mean(2);target=raw_c.abs().mean(2)
            a=amplitude_loss(pred,target,mask,model.pair_config.get('amplitude_mode','region'))
            a_fg,a_bg=region_components(pred,target,mask)
            terms.update(A=n*(model.pair_config['energy']*a),A_fg=n*(model.pair_config['energy']*a_fg),A_bg=n*(model.pair_config['energy']*a_bg))
            terms['C_plus_A']=terms['C']+terms['A']
            gate_info=band_statistics(raw_c,raw_n,gated_n,gate_c,gate_n,mask,layer.floor)
            for band in gate_info:
                for region in band['regions'].values():
                    region['amplitude_full_weighted_contribution']=n*model.pair_config['energy']*region['amplitude_loss_unweighted_contribution']
            raw.update(A=float(a.detach()),A_fg=float(a_fg.detach()),A_bg=float(a_bg.detach()))
            total=D+(terms['C']+terms['A'])
        else:total=D+terms['C']
        terms['total']=total
        return terms,dict(unweighted=raw,weighted={k:float(v.detach()) for k,v in terms.items()},batch_factor=n,
            weights=dict(C=model.pair_config['cons'],A=model.pair_config['energy'] if model.variant=='B6' else 'not_applicable',background=.1),
            perturbation=replay,gate_subbands=gate_info,
            proxy_regions=dict(fg_pixels_per_image=mask.sum((1,2,3)).detach().cpu().tolist(),
                bg_pixels_per_image=(1-mask).sum((1,2,3)).detach().cpu().tolist(),mask_shape=list(mask.shape),
                limitation='GT box unions include background; complement may include unannotated defects; not segmentation truth'))
    finally:
        handle.remove()
        if gated:layer.capture=False;layer.cache=None


def gradients(loss,params,retain_graph=True):
    if not loss.requires_grad:return tuple(None for _ in params)
    values=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain_graph,create_graph=False)
    for value in values:
        if value is not None and not torch.isfinite(value).all():raise ValueError('Nonfinite gradient')
    return tuple(None if v is None else v.detach().cpu().clone() for v in values)


def assert_gradient_match(a,b,rtol,atol):
    if len(a)!=len(b):raise AssertionError('Gradient length mismatch')
    max_error=0.
    for left,right in zip(a,b):
        if (left is None)!=(right is None):raise AssertionError('Gradient connectivity mismatch')
        if left is not None:
            torch.testing.assert_close(left,right,rtol=rtol,atol=atol)
            max_error=max(max_error,float((left-right).abs().max()))
    return max_error


def gradient_groups(gradients_by_term,named,groups):
    result={}
    for group,indices in groups.items():
        vectors={};stats={}
        for term,values in gradients_by_term.items():
            connected=sum(values[i] is not None for i in indices)
            vector=torch.cat([(values[i].double().flatten() if values[i] is not None else torch.zeros(named[i][1].numel(),dtype=torch.float64)) for i in indices])
            norm=float(torch.linalg.vector_norm(vector))
            vectors[term]=vector;stats[term]=dict(status='disconnected' if not connected else 'zero_norm' if norm==0 else 'ok',
                connected_tensors=connected,disconnected_tensors=len(indices)-connected,L2_norm=norm if connected else None)
        D=stats['D']
        for term,stat in stats.items():
            stat['relative_to_detection_norm']=stat['L2_norm']/D['L2_norm'] if D['status']=='ok' and stat['status']!='disconnected' else None
            stat['relative_ratio_status']='disconnected' if stat['status']=='disconnected' else 'ok' if D['status']=='ok' else 'detection_'+D['status']
        comparisons={}
        for other in ['C','A','C_plus_A','C_fg','C_bg','A_fg','A_bg']:
            if other not in stats:
                comparisons['D_vs_'+other]=dict(status='not_applicable',cos=None,dot=None);continue
            b=stats[other];dot=float(torch.dot(vectors['D'],vectors[other])) if D['status']!='disconnected' and b['status']!='disconnected' else None
            status='ok' if D['status']=='ok' and b['status']=='ok' else 'left_'+D['status'] if D['status']!='ok' else 'right_'+b['status']
            comparisons['D_vs_'+other]=dict(status=status,cos=max(-1.,min(1.,dot/(D['L2_norm']*b['L2_norm']))) if status=='ok' else None,dot=dot)
        if 'A' not in stats:
            for term in ['A','A_fg','A_bg','C_plus_A']:stats[term]=dict(status='not_applicable',L2_norm=None,relative_to_detection_norm=None)
        result[group]=dict(parameter_names=[named[i][0] for i in indices],losses=stats,comparisons=comparisons,
            scope='Only directions within this model and parameter group are compared; no cross-model dimension concatenation')
    return result


def analyze_batch(model,batch,step,named,groups,rtol=1e-4,atol=1e-5):
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise RuntimeError('Autograd must be enabled')
    if any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)):raise RuntimeError('Frozen BN eval is required')
    params=tuple(p for name,p in named);layer=model.model[1];initial_hooks=len(layer._forward_hooks)
    state_before=state_identity(model);inputs=[];first=model.model[0]
    handle=first.register_forward_pre_hook(lambda module,args:inputs.append(tensor_identity(args[0])))
    try:
        model.perturbation_step=step
        original,items=model.loss(batch)
        original_scalar=float(original.detach())
        reference=gradients(original,params,retain_graph=False)
        del original,items
    finally:handle.remove()
    terms,record=decomposed_loss(model,batch,step)
    torch.testing.assert_close(terms['total'].detach(),terms['total'].new_tensor(original_scalar),rtol=rtol,atol=atol)
    if len(inputs)!=2 or inputs[0]!=tensor_identity(batch['img']) or inputs[1]!=record['perturbation']['noisy_image']:
        raise AssertionError('Original and rebuilt paired inputs differ')
    torch.testing.assert_close(terms['C_fg']+terms['C_bg'],terms['C'],rtol=rtol,atol=atol)
    values={k:gradients(t,params) for k,t in terms.items() if k!='total'}
    c_rebuilt=gradients(terms['C_fg']+terms['C_bg'],params)
    assert_gradient_match(c_rebuilt,values['C'],rtol,atol)
    if model.variant=='B6':
        torch.testing.assert_close(terms['A_fg']+terms['A_bg'],terms['A'],rtol=rtol,atol=atol)
        rebuilt=gradients(terms['A_fg']+terms['A_bg'],params)
        assert_gradient_match(rebuilt,values['A'],rtol,atol)
        contributions=sum(r['regions'][region]['amplitude_loss_unweighted_contribution'] for r in record['gate_subbands'] for region in ['fg_box_union_proxy','bg_box_complement_proxy'])
        if not math.isclose(contributions,record['unweighted']['A'],rel_tol=rtol,abs_tol=atol):raise AssertionError('Subband errors do not reconstruct amplitude loss')
    actual=gradients(terms['total'],params,retain_graph=False)
    error=assert_gradient_match(actual,reference,rtol,atol)
    record['gradient_groups']=gradient_groups(values,named,groups)
    del terms,values,actual,reference
    state_after=state_identity(model)
    if state_before!=state_after:raise AssertionError('Parameters or BN buffers changed')
    if len(layer._forward_hooks)!=initial_hooks or (isinstance(layer,DetailDown) and (layer.capture or layer.cache is not None)):
        raise AssertionError('Temporary forward hook/capture not cleaned')
    if any(p.grad is not None for p in model.parameters()):raise AssertionError('Unexpected accumulated .grad')
    record.update(original_total=original_scalar,total_difference=record['weighted']['total']-original_scalar,
        consistency=dict(status='passed',gradient_max_abs_error=error,rtol=rtol,atol=atol,paired_input_hashes_equal=True),
        state=dict(before=state_before,after=state_after,unchanged=True,parameter_grads_not_accumulated=True,temporary_hooks_clean=True))
    return record
