"""Offline audit of balanced timing and decoded/input identity, not inference."""
from pathlib import Path
import json,math,csv,hashlib
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'received/DetailPair_recheck_results_20261015_133501_258306'
RUN=SOURCE/'runs/recheck_20261015_133242_867676'
OUT=ROOT/'recheck_review_20261015';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def stats(rows):
    x=np.asarray([r['latency_ms'] for r in rows]);assert np.isfinite(x).all() and (x>0).all()
    return dict(n=len(x),mean_ms=float(x.mean()),p50_ms=float(np.percentile(x,50)),p95_ms=float(np.percentile(x,95)),serial_fps=float(1000*len(x)/x.sum()),total_ms=float(x.sum()))
def same(a,b):
    for k,v in a.items():assert math.isclose(v,b[k],abs_tol=1e-8),(k,v,b[k])
config=read(RUN/'effective_config.json');summary=read(RUN/'timing/balanced_summary.json')
rows=lines(RUN/'timing/latency_all.jsonl');assert len(rows)==1440
order=['B2','B6','B6','B2','B6','B2','B2','B6'];assert config['block_order']==order
ids={r['image_id'] for r in read(RUN/'source_manifest.json')['records']};assert len(ids)==180
block_stats=[]
for i,v in enumerate(order):
    b=[r for r in rows if r['block_id']==i];assert len(b)==180 and {r['image_id'] for r in b}==ids
    identity=read(RUN/(v+'_checkpoint_identity.json'))
    assert all(r['variant']==v and r['weights_sha256']==identity['weights_sha256'] and r['actual_shape']==[1,3,320,320] for r in b)
    saved=lines(RUN/f'timing/block_{i:02d}_{v}/latency.jsonl');assert b==saved
    s=stats(b);same(s,summary['blocks'][i])
    w=[stats(b[j:j+60]) for j in range(0,180,60)]
    for a,z in zip(w,summary['blocks'][i]['windows']):same(a,z)
    ratio=max(x['mean_ms'] for x in w)/min(x['mean_ms'] for x in w)
    assert math.isclose(ratio,summary['blocks'][i]['window_max_min_ratio'],abs_tol=1e-10)
    block_stats.append(dict(block_id=i,variant=v,**s,window_max_min_ratio=ratio))
overall={v:stats([r for r in rows if r['variant']==v]) for v in ['B2','B6']}
for v,s in overall.items():same(s,summary['variants'][v]['overall'])
    
warm=lines(RUN/'timing/warmup.jsonl');counts=Counter((r['phase'],r['variant']) for r in warm)
assert counts[('common','B2')]==counts[('common','B6')]==607
assert counts[('block','B2')]==counts[('block','B6')]==800
assert counts[('bootstrap','B2')]==counts[('bootstrap','B6')]==1
diagnostics={}
previous=ROOT/'received/DetailPair_eval_results_20261015_055914_339322/runs/realtime_eval_20261015_055635_628010'
for v in ['B2','B6']:
    q=RUN/f'clean_diagnostic/{v}'
    a={r['image_id']:r for r in lines(q/'legacy_jpeg/decoded_pixels.jsonl')}
    b={r['image_id']:r for r in lines(q/'new_png/decoded_pixels.jsonl')}
    ta={r['image_id']:r for r in lines(q/'legacy_jpeg/input_tensors.jsonl')}
    tb={r['image_id']:r for r in lines(q/'new_png/input_tensors.jsonl')}
    assert set(a)==set(b)==set(ta)==set(tb)==ids
    assert all(a[k]['RGB']==b[k]['RGB'] for k in ids)
    assert all(ta[k]['batch_shape']==[1,3,352,352] and tb[k]['batch_shape']==[1,3,320,320] for k in ids)
    assert all(ta[k]['normalized']['dtype']==tb[k]['normalized']['dtype']=='float32' for k in ids)
    old=read(q/'legacy_jpeg/official_ap.json')['metrics'];new=read(q/'new_png/official_ap.json')['metrics']
    comp=read(q/'comparison.json')
    # Historical JSON also carries split='val'; compare all numerical metric keys.
    assert old=={k:comp['historical_metrics']['old'][k] for k in old}
    assert new==read(previous/v/'clean/official_ap.json')['metrics']
    diagnostics[v]=dict(images=180,decoded_RGB_equal=True,legacy_shape=[1,3,352,352],new_shape=[1,3,320,320],legacy_metrics=old,new_metrics=new,effective_differences=comp['effective_argument_differences'],single_factor=comp['single_factor'])
    for arm in ['legacy_jpeg','new_png']:
        ap=read(q/arm/'official_ap.json');assert math.isclose(sum(c['AP50_95'] for c in ap['classes'])/6,ap['metrics']['metrics/mAP50-95(B)'],abs_tol=1e-12)

# A conditional scalar illustration of the existing amplitude objective.
# If clean/noisy raw amplitude A is identical and upstream features are fixed,
# L(g)=|g*A-A|/(A+eps), g<1, then dL/dg=-A/(A+eps).
toy=[]
for A in [.2,1.,3.]:
    for g in [.2,.5,.8]:
        f=lambda gate:abs(gate*A-A)/(A+.001)
        numerical=(f(g+1e-6)-f(g-1e-6))/(2e-6);analytic=-A/(A+.001)
        assert math.isclose(numerical,analytic,rel_tol=1e-8)
        toy.append(dict(A=A,g=g,loss=f(g),analytic_gradient=analytic,finite_difference=numerical))
result=dict(zip_sha256=hashlib.sha256((ROOT.parent/'upload/DetailPair_recheck_results_20261015_133501_258306.zip').read_bytes()).hexdigest(),timing=overall,blocks=block_stats,diagnostics=diagnostics,
    latency_B6_increase_percent=(overall['B6']['mean_ms']/overall['B2']['mean_ms']-1)*100,
    fixed_screen_passed=all(r['window_max_min_ratio']<=1.1 for r in block_stats) and all(max(r['mean_ms'] for r in block_stats if r['variant']==v)/min(r['mean_ms'] for r in block_stats if r['variant']==v)<=1.1 for v in ['B2','B6']),
    conditional_toy=toy,scope='Returned Windows records audited offline. No GPU/model execution, no independent AP recomputation; toy is conditional numerical illustration, not empirical cause of false positives.')
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
with (OUT/'block_statistics_recomputed.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=block_stats[0].keys());w.writeheader();w.writerows(block_stats)
print(json.dumps({'timing':overall,'B6_latency_increase_percent':result['latency_B6_increase_percent'],'fixed_screen_passed':result['fixed_screen_passed'],'decoded_and_shape_checks':'passed','historical_and_previous_AP_exact':'passed','warmup_counts':{str(k):v for k,v in counts.items()},'toy_checks':len(toy)},ensure_ascii=False))
