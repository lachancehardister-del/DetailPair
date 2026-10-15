"""Independent offline review of returned B5/B6 records; no model execution.
Run in the supplied project tree using Python 3.10+ (standard library only).
Original ZIP/checkpoints/data are never modified. This is an audit script,
not the full experimental engineering package needed for a new local run.
"""
from pathlib import Path, PurePosixPath
import csv, hashlib, importlib.util, json, math, traceback, zipfile

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'received/DetailPair_matched_results_20261015_152500_121924'
RUN = SRC / 'runs/matched_20261015_152226_359125'
OUT = ROOT / 'matched_review_20261015'
OUT.mkdir(exist_ok=True)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a, b):
    if a is None or b is None: assert a is b
    else: assert math.isclose(a, b, abs_tol=1e-11), (a, b)
def iou(a, b):
    inter = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    area = lambda x: max(0.,x[2]-x[0])*max(0.,x[3]-x[1])
    union = area(a)+area(b)-inter
    return inter/union if union else 0.
def rematch(pred, gt):
    counts = [dict(TP=0,FP=0,FN=0,GT=0) for _ in range(6)]
    used = set()
    for g in gt: counts[g['class_id']]['GT'] += 1
    for j in sorted(range(len(pred)), key=lambda j:(-pred[j]['confidence'],j)):
        p = pred[j]; c = p['class_id']
        candidates = [(iou(p['xyxy'],g['xyxy']),k) for k,g in enumerate(gt) if k not in used and g['class_id']==c]
        overlap,k = max(candidates,key=lambda z:(z[0],-z[1]),default=(0.,-1))
        hit = k>=0 and overlap>=.5
        counts[c]['TP' if hit else 'FP'] += 1
        if hit: used.add(k)
    for k,g in enumerate(gt):
        if k not in used: counts[g['class_id']]['FN'] += 1
    return counts

def main():
    archive = ROOT.parent/'upload/DetailPair_matched_results_20261015_152500_121924.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(not PurePosixPath(n.replace('\\','/')).is_absolute() and '..' not in PurePosixPath(n.replace('\\','/')).parts for n in z.namelist())
        nzip = len(z.infolist())
    assert sha(archive)=='9b050970447a1f997d48011840d57246c5c825af1a72bf0b5b76a9a597574c6f'
    state=read(RUN/'status.json'); assert state['status']=='success' and state['successful_evaluations']==4
    assert all(state[k] for k in ['no_training','no_optimizer','no_resume','weights_and_sources_verified_unchanged'])
    assert not any(state[k] for k in ['timing_performed','test_evaluated','synthetic_check'])
    pair=read(RUN/'pairing.json'); assert pair['completed_epoch']==38 and pair['checkpoint_epoch']==37
    eligible={v:[r for r in read(RUN/(v+'_inventory.json'))['checkpoints'] if r['status']=='eligible'] for v in ['B5','B6']}
    common=sorted(set(r['completed_epoch'] for r in eligible['B5']) & set(r['completed_epoch'] for r in eligible['B6']))
    assert common==pair['common_completed_epochs'] and max(common)==38
    historical=ROOT/'received/DetailPair_results_20261011_164903_059647/runs/pilot__20261011_112235_644745'
    for v in ['B5','B6']:
        selected=pair['selected'][v]; assert selected in eligible[v]
        assert selected['selected_role']=='ema' and selected['completed_epoch']==38
        model=selected['model_and_ema']['ema']; assert model['parameters_and_buffers_finite'] and model['parameters']==2590034
        assert model['pair_config']==dict(module='gate',pair=True,cons=.05,energy=0. if v=='B5' else .02)
        inv=read(RUN/(v+'_inventory.json'))
        original=historical/v.__add__('_s0')/'training/results.csv'
        assert sha(original)==inv['csv']['sha256']
        with original.open(encoding='utf-8-sig',newline='') as f:
            csvrows=[{k.strip():float(x) for k,x in row.items()} for row in csv.DictReader(f)]
        assert csvrows[37]==selected['csv_row']
    aa,bb=[pair['selected'][v]['train_args'] for v in ['B5','B6']]
    ignore={'name','project','save_dir'}
    assert {k:v for k,v in aa.items() if k not in ignore}=={k:v for k,v in bb.items() if k not in ignore}
    semantics=read(RUN/'epoch_semantics.json')
    assert sha(RUN/'installed_save_model.py.txt')==semantics['save_source_sha256']
    assert sha(RUN/'installed_save_metrics.py.txt')==semantics['csv_source_sha256']
    assert 'self.epoch + 1' in (RUN/'installed_save_metrics.py.txt').read_text()
    identities=[read(RUN/(v+'_selection_identity.json'))['source_identity'] for v in ['B5','B6']]
    assert identities[0]==identities[1] and len(identities[0]['train'])==1438 and len(identities[0]['val'])==180
    manifest=read(RUN/'source_manifest.json'); records={r['image_id']:r for r in manifest['records']}
    assert len(records)==180 and sum(len(r['gt']) for r in records.values())==429
    previous=read(ROOT/'received/DetailPair_eval_results_20261015_055914_339322/runs/realtime_eval_20261015_055635_628010/source_manifest.json')
    assert manifest['records']==previous['records']
    readonly=read(RUN/'read_only_verification.json'); assert readonly['verified_unchanged'] and not readonly['changed_or_missing']
    source_equal={n:(SRC/'detailpair'/n).read_bytes()==(ROOT/'DetailPair_v0.2.4/detailpair'/n).read_bytes() for n in ['model.py','engine.py','data.py']}
    assert all(source_equal.values())
    allrows=[]; classes=[]; output={}
    for scenario in ['clean','gaussian_0.03']:
        shared=read(RUN/(scenario+'_derived_manifest.json')); assert shared['shared_by']==['B5','B6'] and shared['seed']==178
        assert len(shared['records'])==180 and {r['image_id'] for r in shared['records']}==set(records)
        if scenario=='clean': assert all(r['mse']==0 for r in shared['records'])
        for v in ['B5','B6']:
            dest=RUN/v/scenario
            ident=read(dest/'evaluation_identity.json')
            assert ident['checkpoint']==pair['selected'][v] and ident['derived_manifest_sha256']==sha(RUN/(scenario+'_derived_manifest.json'))
            assert read(dest/'status.json')['status']=='success'
            a=read(dest/'official_actual_inputs.json'); b=read(dest/'fixed_actual_inputs.json')
            assert len(a)==len(b)==180 and [r['image_id'] for r in a]==[r['image_id'] for r in b]
            for x,y in zip(a,b):
                assert x['actual_shape']==y['actual_shape']==[1,3,320,320]
                assert x['actual_device']==y['actual_device']=='cuda:0'
                assert x['dtype']==y['dtype']=='torch.float32' and x['input']['sha256']==y['input']['sha256']
            if v=='B5': base_a,base_b=a,b
            else: assert a==base_a and b==base_b
            pred=lines(dest/'fixed_predictions_matches.jsonl'); appred=lines(dest/'official/ap_predictions.jsonl')
            assert len(pred)==len(appred)==180
            assert {r['image_id'] for r in pred}=={r['image_id'] for r in appred}==set(records)
            totals=[dict(TP=0,FP=0,FN=0,GT=0) for _ in range(6)]
            for image in pred:
                assert image['gt']==records[image['image_id']]['gt']
                assert all(p['confidence']>=.25 and all(math.isfinite(t) for t in p['xyxy']) for p in image['predictions'])
                for c,row in zip(totals,rematch(image['predictions'],image['gt'])):
                    for k,n in row.items():c[k]+=n
            saved=read(dest/'fixed_metrics.json'); ap=read(dest/'official_ap.json')
            for c,s in zip(totals,saved['classes']):
                for k,n in c.items(): assert n==s[k]
                c['Precision']=c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else None
                c['Recall']=c['TP']/c['GT']
                same(c['Precision'],s['Precision']);same(c['Recall'],s['Recall'])
            for key in ['Precision','Recall']:
                valid=[c[key] for c in totals if c[key] is not None]
                same(sum(valid)/len(valid),saved['macro'][key]);assert len(valid)==saved['macro'][key+'_classes_with_denominator']
            for key,metric in [('AP50','metrics/mAP50(B)'),('AP50_95','metrics/mAP50-95(B)')]:
                same(sum(c[key] for c in ap['classes'])/6,ap['metrics'][metric])
            row=dict(scenario=scenario,variant=v,AP50=ap['metrics']['metrics/mAP50(B)'],AP50_95=ap['metrics']['metrics/mAP50-95(B)'],
                macro_recall=saved['macro']['Recall'],macro_precision_available=saved['macro']['Precision'],precision_support=saved['macro']['Precision_classes_with_denominator'],
                macro_precision_six_zero=sum(c['Precision'] or 0. for c in totals)/6,**{k:sum(c[k] for c in totals) for k in ['TP','FP','FN','GT']})
            row['micro_precision']=row['TP']/(row['TP']+row['FP']);row['FPPI']=row['FP']/180
            same(row['FPPI'],saved['FPPI']);allrows.append(row);output[v,scenario]=(ap,saved)
            for c,s,t in zip(totals,saved['classes'],ap['classes']): classes.append(dict(scenario=scenario,variant=v,**s,AP50=t['AP50'],AP50_95=t['AP50_95']))
    comparisons=read(RUN/'comparison.json')
    for r in comparisons['rows']:
        def get(v):
            ap,fixed=output[v,r['scenario']];k=r['metric']
            return ap['metrics'][k] if k.startswith('metrics/') else fixed['macro'][k.removeprefix('macro_')] if k.startswith('macro_') else fixed[k]
        a,b=get('B5'),get('B6');same(a,r['B5']);same(b,r['B6']);same(a-b,r['B5_minus_B6'])
    audit=dict(zip_sha256=sha(archive),zip_entries=nzip,run=RUN.name,completed_epoch=38,checkpoint_epoch=37,
        weights={v:pair['selected'][v]['sha256'] for v in ['B5','B6']},source_code_equal_v024=source_equal,
        deps={k:importlib.util.find_spec(k) is not None for k in ['numpy','yaml','torch']},scenarios=allrows,
        checks=dict(all_assertions_passed=True,fixed_rows_independently_rematched=720,official_prediction_rows_cardinality_checked=720,
            actual_input_records_checked=1440,read_only_files_reported=readonly['files_checked'],old_csv_hash_and_epoch38_rechecked=True,
            official_AP_recomputed_from_predictions=False,model_or_gradients_rerun=False,windows_cuda_executed_here=False),
        issues=['Original available-class macro precision compares B5 six classes to B6 five; retain original, disclose support, add explicit zero-convention six-class and micro precision.',
                'Checkpoint/data bytes stay in Windows; audit verifies returned evidence, not fresh on-disk hashes.',
                'Historical initialization tensor hashes unavailable; matching seed does not establish identical trajectories.'])
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename,rows in [('scenarios_recomputed.csv',allrows),('classes_recomputed.csv',classes)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except BaseException:
        import datetime
        (OUT/('failure_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        raise
