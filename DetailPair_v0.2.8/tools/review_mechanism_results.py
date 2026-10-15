"""Recompute supplied v0.2.7 raw batches; this is a Codex audit, not a Work review."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mechanism_diagnose import summarize_batches
source=ROOT/'work_inputs/20261015/mechanism_results';run=source/'runs/mechanism_20261015_141404_504463'
def equal(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b
models={};records={}
for v in ['B2','B6']:
    rows=[json.loads(l) for l in (run/v/'batches.jsonl').read_text().splitlines()];records[v]=rows
    assert len(rows)==32 and all(r['status']=='success' and r['state']['unchanged'] and r['state']['before']==r['state']['after'] and r['consistency']['status']=='passed' for r in rows)
    assert len({x for r in rows for x in r['image_ids']})==64
    summary=summarize_batches(rows,32);assert equal(summary,json.loads((run/v/'summary.json').read_text()))
    models[v]=dict(raw_batches=32,unique_images=64,recomputed_summary_matches=True,
        largest_total_difference=max(abs(r['total_difference']) for r in rows),parameters_and_BN_equal_each_batch=True)
assert all(a['image_ids']==b['image_ids'] and a['input']==b['input'] and a['perturbation']==b['perturbation'] for a,b in zip(records['B2'],records['B6']))
rows=records['B6'];cos=[r['gradient_groups']['gates']['comparisons']['D_vs_A']['cos'] for r in rows];ratio=[r['gradient_groups']['gates']['losses']['A']['relative_to_detection_norm'] for r in rows]
manifest=json.loads((source/'verification_manifest.json').read_text());present=[];missing=[];different=[]
for name,sha in manifest['files'].items():
    path=source/name
    if not path.is_file():missing.append(name)
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=sha:different.append(name)
    else:present.append(name)
audit=dict(scope='Codex recomputation of uploaded raw scalar records; no original weights/data provided, no GPU rerun',models=models,
    paired_input_and_perturbation_records_equal=True,B6_gate_D_vs_A_negative_batches=sum(v<0 for v in cos),B6_gate_batches=len(cos),
    B6_gate_A_to_D_norm_mean=sum(ratio)/len(ratio),B6_gate_D_vs_A_cos_mean=sum(cos)/len(cos),
    supplied_source_manifest=dict(present_matching=len(present),absent=missing,different=different),
    missing_requested_document='MECHANISM_RESULT_REVIEW_20261015.md was not attached; task document supplies the current boundary',
    interpretation='Endpoint local conflict in 21/32 batches; A norm averages 2.03% of D. Does not prove removing A improves AP. No new training authorized.')
(ROOT/'verification/mechanism_result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
