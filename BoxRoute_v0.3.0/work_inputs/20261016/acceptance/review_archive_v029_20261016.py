"""Audit an archive and its aggregate tables. No model/data/GPU execution."""
from pathlib import Path
import ast,csv,hashlib,importlib.util,json,math

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'received/v029_archive_20261016/DetailPair_v0.2.9'
OUT=ROOT/'archive_review_20261016';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,abs_tol=1e-12),(a,b)

manifest=read(SRC/'verification_manifest.json')['files']
for name,h in manifest.items():assert sha(SRC/name)==h,name
python_files=list(SRC.rglob('*.py'))
for p in python_files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
with (SRC/'archive/work_review/classes_recomputed.csv').open(encoding='utf-8-sig',newline='') as f:classes=list(csv.DictReader(f))
with (SRC/'archive/work_review/scenarios_recomputed.csv').open(encoding='utf-8-sig',newline='') as f:scenarios=list(csv.DictReader(f))
assert len(classes)==24 and len(scenarios)==4
comparison=read(SRC/'archive/reporting_correction/comparison.json')
summary=read(SRC/'archive/reporting_correction/fixed_summary_from_Work_class_tables.json')
summaries={(r['variant'],r['scenario']):r for r in summary['conditions']}
recomputed={};rows_out=[]
for row in scenarios:
    key=row['variant'],row['scenario'];group=sorted([r for r in classes if (r['variant'],r['scenario'])==key],key=lambda r:int(r['class_id']))
    assert [int(r['class_id']) for r in group]==list(range(6))
    tp=sum(int(r['TP']) for r in group);fp=sum(int(r['FP']) for r in group);fn=sum(int(r['FN']) for r in group);gt=sum(int(r['GT']) for r in group)
    assert gt==429 and tp+fn==gt
    for k,n in [('TP',tp),('FP',fp),('FN',fn),('GT',gt)]:assert n==int(row[k])
    precision=[];recall=[];support=[];original=summaries[key]
    for r,saved in zip(group,original['classes']):
        a,b,g=int(r['TP']),int(r['FP']),int(r['GT']);assert int(r['FN'])+a==g
        for k in ['TP','FP','FN','GT']:assert saved[k]==int(r[k])
        assert saved['class_id']==int(r['class_id']) and saved['name']==r['name']
        if a+b:
            p=a/(a+b);close(p,float(r['Precision']));close(p,saved['Precision']);support.append(int(r['class_id']));precision.append(p)
        else:assert r['Precision']=='' and saved['Precision'] is None;precision.append(None)
        recall.append(a/g);close(a/g,float(r['Recall']));close(a/g,saved['Recall'])
    valid=[x for x in precision if x is not None]
    available=sum(valid)/len(valid);zero=sum(x or 0. for x in precision)/6;micro=tp/(tp+fp);macro_r=sum(recall)/6
    for x,k in [(available,'macro_precision_available'),(zero,'macro_precision_six_zero'),(micro,'micro_precision'),(macro_r,'macro_recall')]:close(x,float(row[k]))
    close(available,original['macro']['Precision']);close(zero,original['macro_fixed_class_zero_no_predictions']['Precision']);close(micro,original['micro']['Precision'])
    assert support==original['macro']['Precision_class_ids']
    assert support==([0,1,2,3,4,5] if row['variant']=='B5' else [1,2,3,4,5])
    # Class AP arithmetic only: not official AP curve recomputation.
    for k,col in [('AP50','AP50'),('AP50_95','AP50_95')]:close(sum(float(r[k]) for r in group)/6,float(row[col]))
    rec=dict(scenario=row['scenario'],variant=row['variant'],TP=tp,FP=fp,FN=fn,GT=gt,AP50=float(row['AP50']),AP50_95=float(row['AP50_95']),
        precision_support=support,macro_precision_available=available,macro_precision_zero=zero,micro_precision=micro,macro_recall=macro_r,FPPI=fp/180)
    recomputed[key]=rec;rows_out.append(rec)
metrics={'metrics/mAP50-95(B)':'AP50_95','metrics/mAP50(B)':'AP50','macro_Precision':'macro_precision_available',
    'macro_Recall':'macro_recall','micro_Precision':'micro_precision','micro_Recall':None,
    'macro_Precision_zero_no_predictions':'macro_precision_zero','FPPI':'FPPI'}
assert len(comparison['rows'])==16 and len(comparison['classes'])==96
for r in comparison['rows']:
    a,b=[recomputed[v,r['scenario']] for v in ['B5','B6']];k=metrics[r['metric']]
    x,y=(a[k],b[k]) if k else (a['TP']/a['GT'],b['TP']/b['GT'])
    close(x,r['B5']);close(y,r['B6'])
    if r['metric']=='macro_Precision':
        assert not r['comparable'] and r['B5_minus_B6'] is None and r['B5_support_count']==6 and r['B6_support_count']==5
    else:assert r['comparable'];close(x-y,r['B5_minus_B6'])
for r in comparison['classes']:
    left,right=[next(c for c in classes if c['variant']==v and c['scenario']==r['scenario'] and int(c['class_id'])==r['class_id']) for v in ['B5','B6']]
    def value(c):return float(c[r['metric']]) if c[r['metric']]!='' else None
    x,y=value(left),value(right)
    for actual,expected in [(r['B5'],x),(r['B6'],y)]:
        if expected is None:assert actual is None
        else:close(actual,expected)
    if x is None or y is None:assert r['B5_minus_B6'] is None
    else:close(x-y,r['B5_minus_B6'])
branch=read(SRC/'archive/branch_status.json')
assert branch['status']=='archived_no_consistent_benefit' and not branch['user_local_run_required'] and not branch['new_experiments_authorized']
identity=read(SRC/'verification/source_identity.json')
for n,h in identity['core_unchanged_sha256'].items():assert sha(SRC/'detailpair'/n)==h
assert not list(SRC.rglob('*.pt')) and not (SRC/'runs').exists()
test_log=(OUT/'reporting_tests_actual.log').read_text();assert 'Ran 6 tests' in test_log and '\nOK\n' in test_log
audit=dict(status='passed_archive_and_reporting_review',archive_sha256=sha(ROOT.parent/'upload/DetailPair_v0.2.9_Archive_Package.zip'),
    manifest_files_checked=len(manifest),python_files_ast_checked=len(python_files),reporting_tests_rerun=6,
    class_rows_checked=24,scenario_rows_checked=4,comparison_rows_checked=16,per_class_comparison_rows_checked=96,
    original_count_AP_values_preserved=True,macro_precision_support_mismatch_handled=True,branch_status=branch,
    core_files_match_packaged_identity=True,independent_previous_v028_byte_comparison=False,
    deps={n:importlib.util.find_spec(n) is not None for n in ['numpy','yaml','torch','ultralytics']},
    model_execution=False,windows_cuda_execution=False,raw_predictions_rematched=False,official_AP_curve_recomputed=False,
    test_dataset_accessed=False,limitations=['Aggregate arithmetic only; no new model evidence.',
        'Old raw results and v0.2.8 bytes were not re-materialized; previous experimental findings remain earlier Work evidence.',
        'Archive retains runnable historical entries; archival state is documentation, not a runtime prohibition.'],scenarios=rows_out)
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='scenarios'},ensure_ascii=False,indent=2))
