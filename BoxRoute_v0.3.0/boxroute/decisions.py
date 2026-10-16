"""Predeclared complete-table reporting and engineering gates, not statistical tests."""
from .reporting import dump,write_csv,compare_fixed_summaries,comparison_row


def G1_report(results,costs,timing,out):
    rows=[]
    for scenario in ['clean','gaussian_0.03']:
        for other in ['N','S','U','W']:
            a,b=results['G',scenario],results[other,scenario]
            for metric in ['metrics/mAP50-95(B)','metrics/mAP50(B)','metrics/mAP75(B)']:
                rows.append(dict(scenario=scenario,**comparison_row(metric,a['ap']['metrics'][metric],b['ap']['metrics'][metric],left='G',right=other)))
            rows.extend(dict(scenario=scenario,**row) for row in compare_fixed_summaries(a['fixed'],b['fixed'],left='G',right=other))
    # Comparisons have dynamic right-condition keys, so emit separate consistent tables.
    dump(out/'comparison_all.json',dict(rows=rows,scope='Single-seed validation engineering screen; all conditions retained, no significance or novelty claim'))
    for other in ['N','S','U','W']:
        write_csv(out/('G_minus_'+other+'.csv'),[r for r in rows if 'G_minus_'+other in r])
    ap=lambda v,key:results[v,'clean']['ap']['metrics'][key]
    key='metrics/mAP50-95(B)';class_deltas=[g['AP50_95']-n['AP50_95'] for g,n in zip(results['G','clean']['ap']['classes'],results['N','clean']['ap']['classes']) if g['AP50_95'] is not None and n['AP50_95'] is not None]
    all_classes=len(class_deltas)==len(results['N','clean']['ap']['classes'])
    gates=dict(G_minus_N_pp_min1=ap('G',key)-ap('N',key)>=.01,G_minus_U_pp_min03=ap('G',key)-ap('U',key)>=.003,
        G_minus_W_pp_min03=ap('G',key)-ap('W',key)>=.003,G_not_below_S=ap('G',key)>=ap('S',key),
        AP75_G_not_below_N=ap('G','metrics/mAP75(B)')>=ap('N','metrics/mAP75(B)'),
        worst_class_drop_max3pp=all_classes and min(class_deltas)>=-.03,
        parameter_increase_max5pct=costs['parameter_ratio_G_to_N']<=1.05,conv_MAC_increase_max5pct=costs['conv_MAC_ratio_G_to_N']<=1.05,
        stable_timing=timing['comparison_credible'],
        mean_latency_ratio_max105=timing['variants']['G']['overall']['mean_ms']/timing['variants']['N']['overall']['mean_ms']<=1.05,
        P95_latency_ratio_max110=timing['variants']['G']['overall']['p95_ms']/timing['variants']['N']['overall']['p95_ms']<=1.10)
    status='screen_passed_return_to_Work' if all(gates.values()) else 'screen_failed_return_to_Work'
    dump(out/'G1_decision.json',dict(status=status,gates=gates,all_required_classes_available=all_classes,
        automatic_next_experiment=False,test_evaluated=False,scope='Prespecified engineering screen, not significance, journal acceptance or automatic200epoch authorization'))
    return status
