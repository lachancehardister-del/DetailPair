"""Check Work-derived aggregate tables only; no predictions, AP curves or models are run."""
import csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from detailpair.reporting import summarize_counts,dump,write_csv
from matched_checkpoint_eval import comparison


def main():
    source=ROOT/'archive/work_review';review=ROOT/'work_inputs/20261015/review'
    provenance=json.loads((ROOT/'verification/input_provenance.json').read_text())
    manifest_checked=0
    if review.is_dir():
        manifest=json.loads((review/'matched_review_20261015/ARTIFACT_MANIFEST.json').read_text())
        for item in manifest['files']:
            raw=(review/item['path']).read_bytes()
            assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
            manifest_checked+=1
    def read_csv(name):
        with (source/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    classes=read_csv('classes_recomputed.csv');scenarios=read_csv('scenarios_recomputed.csv')
    results={};summaries=[]
    for row in scenarios:
        key=row['variant'],row['scenario']
        group=sorted([x for x in classes if (x['variant'],x['scenario'])==key],key=lambda x:int(x['class_id']))
        names=[x['name'] for x in group]
        counts=[dict(class_id=int(x['class_id']),name=x['name'],**{k:int(x[k]) for k in ['TP','FP','FN','GT']}) for x in group]
        fixed=summarize_counts([counts],names)
        fixed['FPPI']=int(row['FP'])/180
        def close(x,y):assert math.isclose(x,float(y),rel_tol=1e-12,abs_tol=1e-12),(x,y)
        for k in ['TP','FP','FN','GT']:assert fixed['micro'][k]==int(row[k])
        close(fixed['macro']['Precision'],row['macro_precision_available'])
        close(fixed['macro']['Recall'],row['macro_recall'])
        assert fixed['macro']['Precision_classes_with_denominator']==int(row['precision_support'])
        close(fixed['macro_fixed_class_zero_no_predictions']['Precision'],row['macro_precision_six_zero'])
        close(fixed['micro']['Precision'],row['micro_precision']);close(fixed['FPPI'],row['FPPI'])
        for k,original in enumerate(group):
            for metric in ['Precision','Recall']:
                value=fixed['classes'][k][metric]
                if original[metric]=='':assert value is None
                else:close(value,original[metric])
        # Keep Work-reported official AP values; no prediction integration or AP recomputation.
        ap=dict(metrics={'metrics/mAP50-95(B)':float(row['AP50_95']),'metrics/mAP50(B)':float(row['AP50'])},
            classes=[dict(class_id=int(x['class_id']),name=x['name'],AP50=float(x['AP50']),AP50_95=float(x['AP50_95'])) for x in group])
        results[key]=dict(ap=ap,fixed=fixed)
        summaries.append(dict(scenario=row['scenario'],variant=row['variant'],**fixed))
    out=ROOT/'archive/reporting_correction';out.mkdir(exist_ok=True)
    comparison(results,out)
    dump(out/'fixed_summary_from_Work_class_tables.json',dict(source='Work-derived classes_recomputed.csv; aggregate arithmetic only',conditions=summaries))
    for scenario in ['clean','gaussian_0.03']:
        fixed_a,fixed_b=[results[v,scenario]['fixed'] for v in ['B5','B6']]
        assert fixed_a['macro']['Precision_class_ids']==[0,1,2,3,4,5]
        assert fixed_b['macro']['Precision_class_ids']==[1,2,3,4,5]
    corrected=json.loads((out/'comparison.json').read_text())
    corrected['provenance']=dict(source='Work-derived class and scenario CSV tables',
        raw_matched_result_zip_available=False,independent_prediction_rematch=False,
        official_AP_recomputed=False,new_experiment=False)
    dump(out/'comparison.json',corrected)
    for row in corrected['rows']:
        if row['metric']=='macro_Precision':assert row['B5_minus_B6'] is None and not row['comparable']
    audit=dict(status='passed_review_table_arithmetic_only',branch_status='archived_no_consistent_benefit',
        source=provenance,Work_manifest_files_verified=manifest_checked,
        raw_matched_results_available=False,independent_prediction_rematch=False,official_AP_recomputed=False,
        model_execution=False,GPU_evaluation=False,training=False,test_accessed=False,
        conditions_checked=4,class_count_rows_checked=len(classes),AP_and_TP_FP_FN_GT='preserved from Work-derived tables',
        limitation='Raw 154-entry matched ZIP and 720 image predictions are absent. Work reported their verification; this process only verifies supplied review bytes and aggregate table arithmetic. No new experimental evidence.')
    dump(ROOT/'verification/offline_archive_checks.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
