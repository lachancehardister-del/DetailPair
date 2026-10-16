"""Differentiable instance-ROI box-geometry KL from the frozen BoxRoute proposal."""
import torch


def geometry_targets(wh,swap=False):
    wh=wh.detach()
    if wh.ndim!=2 or wh.shape[1]!=2 or not torch.isfinite(wh).all() or (wh<=0).any():raise ValueError('Finite positive Nx2 widths/heights required')
    w,h=wh.unbind(1);q=torch.stack([2*torch.minimum(w,h),(w-h).clamp_min(0),(h-w).clamp_min(0)],1)/(w+h)[:,None]
    return q[:,[0,2,1]] if swap else q


def route_loss(alpha,boxes,batch_idx,input_hw,swap=False,epsilon=1e-6):
    if alpha.ndim!=4 or alpha.shape[1]!=3 or not torch.isfinite(alpha).all() or (alpha<0).any():raise ValueError('Finite Bx3xHxW route probabilities required')
    if not torch.allclose(alpha.sum(1),torch.ones_like(alpha[:,0]),atol=2e-6,rtol=2e-6):raise ValueError('Routes must sum to one')
    if boxes.ndim!=2 or boxes.shape[1]!=4 or batch_idx.numel()!=len(boxes):raise ValueError('Box/index shapes differ')
    boxes=boxes.detach().to(alpha.device);indices=batch_idx.detach().flatten().to(alpha.device)
    if not torch.isfinite(boxes).all() or not torch.isfinite(indices).all():raise ValueError('Nonfinite labels')
    B,_,H,W=alpha.shape
    if (indices!=indices.long()).any() or (indices<0).any() or (indices>=B).any():raise ValueError('Invalid image index')
    if len(boxes) and ((boxes[:,2:]<=0).any() or (boxes[:,:2]-boxes[:,2:]/2 < -1e-6).any() or (boxes[:,:2]+boxes[:,2:]/2>1+1e-6).any()):raise ValueError('Invalid normalized box')
    dtype=torch.float64 if alpha.dtype==torch.float64 else torch.float32;alpha=alpha.to(dtype);boxes=boxes.to(dtype)
    q=geometry_targets(boxes[:,2:]*boxes.new_tensor([input_hw[1],input_hw[0]]),swap)
    gx=(torch.arange(W,device=alpha.device,dtype=dtype)+.5)/W;gy=(torch.arange(H,device=alpha.device,dtype=dtype)+.5)/H
    by_image=[];fallback=0;overlap_instances=0;overlap_cells=0
    for image in range(B):
        losses=[];masks=[]
        for j in torch.nonzero(indices.long()==image,as_tuple=False).flatten():
            cx,cy,w,h=boxes[j];mask=((gy[:,None]>=cy-h/2)&(gy[:,None]<cy+h/2)&(gx[None,:]>=cx-w/2)&(gx[None,:]<cx+w/2))
            if not mask.any():
                # argmin ties choose the first/lower legal grid centre.
                mask=torch.zeros(H,W,device=alpha.device,dtype=torch.bool)
                mask[torch.argmin((gy-cy).square()),torch.argmin((gx-cx).square())]=True;fallback+=1
            mean=alpha[image,:,mask].mean(1);p=(mean+epsilon)/(1+3*epsilon)
            losses.append((torch.xlogy(q[j],q[j])-q[j]*p.log()).sum());masks.append(mask)
        if masks:
            occupancy=torch.stack(masks).sum(0);overlap_cells+=int((occupancy>1).sum())
            overlap_instances+=sum(bool((mask&(occupancy>1)).any()) for mask in masks)
        by_image.append(torch.stack(losses).mean() if losses else alpha[image].sum()*0.)
    loss=torch.stack(by_image).mean()
    return loss,dict(instances=len(boxes),empty_images=sum(not bool((indices.long()==i).any()) for i in range(B)),
        fallback_instances=fallback,overlap_instances=overlap_instances,overlap_grid_cells=overlap_cells,
        fallback_ratio=fallback/len(boxes) if len(boxes) else 0.,overlap_instance_ratio=overlap_instances/len(boxes) if len(boxes) else 0.,
        mean_q=q.mean(0).tolist() if len(q) else None,ROI='grid centres left/top inclusive, right/bottom exclusive; nearest centre fallback, lower-index ties',swap_HV=swap)


def lambda_at(epoch):
    if int(epoch)!=epoch or epoch<0:raise ValueError('Nonnegative zero-based epoch required')
    return .05*min(1.,(epoch+1)/5)
