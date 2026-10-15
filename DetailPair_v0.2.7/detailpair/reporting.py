"""Auditable fixed operating-point matching and serial latency statistics."""
import csv, json
from pathlib import Path
import numpy as np


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8'); return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., right-left)*max(0., bottom-top)
    area = lambda x: max(0., x[2]-x[0])*max(0., x[3]-x[1])
    union = area(a)+area(b)-inter
    return inter/union if union > 0 else 0.


def match_image(predictions, gt, names, threshold=.5):
    """Class-aware score-descending greedy matching; each GT used at most once."""
    used = set(); matches = []
    counts = [dict(class_id=k, name=n, TP=0, FP=0, FN=0, GT=0) for k, n in enumerate(names)]
    for target in gt: counts[target['class_id']]['GT'] += 1
    order = sorted(range(len(predictions)), key=lambda k: (-predictions[k]['confidence'], k))
    for j in order:
        pred = predictions[j]; k = pred['class_id']
        if not 0 <= k < len(names): raise ValueError('Prediction class outside data.yaml mapping')
        candidates = [(iou(pred['xyxy'], t['xyxy']), i) for i, t in enumerate(gt)
                      if i not in used and t['class_id'] == k]
        score, target = max(candidates, default=(0., -1), key=lambda x: (x[0], -x[1]))
        ok = target >= 0 and score >= threshold
        counts[k]['TP' if ok else 'FP'] += 1
        if ok: used.add(target)
        matches.append(dict(prediction_index=j, class_id=k, confidence=pred['confidence'],
                            status='TP' if ok else 'FP', gt_index=target if ok else None,
                            matched_iou=score if ok else None))
    for i, target in enumerate(gt):
        if i not in used:
            counts[target['class_id']]['FN'] += 1
            matches.append(dict(status='FN', gt_index=i, class_id=target['class_id']))
    return counts, matches


def summarize_counts(rows, names):
    total = [dict(class_id=k, name=n, TP=0, FP=0, FN=0, GT=0) for k, n in enumerate(names)]
    for image_rows in rows:
        for target, source in zip(total, image_rows):
            for key in ('TP', 'FP', 'FN', 'GT'): target[key] += source[key]
    for r in total:
        r['Precision'] = r['TP']/(r['TP']+r['FP']) if r['TP']+r['FP'] else None
        r['Recall'] = r['TP']/r['GT'] if r['GT'] else None
    macro = {}
    for key in ('Precision', 'Recall'):
        values = [r[key] for r in total if r[key] is not None]
        macro[key] = sum(values)/len(values) if values else None
        macro[key+'_classes_with_denominator'] = len(values)
    return dict(classes=total, macro=macro, undefined='null when denominator is zero; macro excludes undefined classes')


def latency_stats(rows):
    values = np.asarray([r['latency_ms'] for r in rows], dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Empty, nonfinite or nonpositive latency samples')
    return dict(n=len(values), mean_ms=float(values.mean()), p50_ms=float(np.percentile(values,50)),
                p95_ms=float(np.percentile(values,95)), serial_fps=float(1000*len(values)/values.sum()),
                total_ms=float(values.sum()))


def balanced_stats(blocks, expected_images, threshold=1.10):
    """Predefined engineering drift screen; no samples are removed."""
    order=['B2','B6','B6','B2','B6','B2','B2','B6']
    table=[];reasons=[]
    for block in blocks:
        rows=block['rows'];n=len(rows)
        if n!=expected_images:
            reasons.append(f"block{block['block_id']}:incomplete");continue
        width=expected_images//3
        if expected_images%3:raise ValueError('Three equal windows required')
        overall=latency_stats(rows)
        windows=[latency_stats(rows[i*width:(i+1)*width]) for i in range(3)]
        ratio=max(w['mean_ms'] for w in windows)/min(w['mean_ms'] for w in windows)
        table.append(dict(block_id=block['block_id'],variant=block['variant'],**overall,
            windows=windows,window_max_min_ratio=ratio))
        if ratio>threshold:reasons.append(f"block{block['block_id']}:window_ratio>{threshold}")
    complete=len(table)==8 and [b['variant'] for b in sorted(table,key=lambda x:x['block_id'])]==order
    if not complete:reasons.append('incomplete_balanced_design')
    variants={}
    for v in ['B2','B6']:
        items=[b for b in table if b['variant']==v]
        if not items:continue
        means=[b['mean_ms'] for b in items];ratio=max(means)/min(means)
        all_rows=[r for b in blocks if b['variant']==v for r in b['rows']]
        variants[v]=dict(overall=latency_stats(all_rows),block_means_ms=means,block_max_min_ratio=ratio)
        if ratio>threshold:reasons.append(v+':between_block_ratio>'+str(threshold))
    pairs=[];cycles=[];lookup={r['block_id']:r for r in table}
    if complete:
        for start in [0,2,4,6]:
            group=[lookup[start],lookup[start+1]];a=next(r for r in group if r['variant']=='B2');b=next(r for r in group if r['variant']=='B6')
            pairs.append(dict(block_ids=[start,start+1],first_variant=group[0]['variant'],B2_mean_ms=a['mean_ms'],B6_mean_ms=b['mean_ms'],
                B6_minus_B2_ms=b['mean_ms']-a['mean_ms'],faster_variant='B2' if a['mean_ms']<b['mean_ms'] else 'B6' if b['mean_ms']<a['mean_ms'] else 'tie'))
        for start,label in [(0,'ABBA'),(4,'BAAB')]:
            means={v:sum(lookup[i]['mean_ms'] for i in range(start,start+4) if lookup[i]['variant']==v)/2 for v in ['B2','B6']}
            cycles.append(dict(cycle=label,**means,B6_minus_B2_ms=means['B6']-means['B2']))
    position_changes=len({p['faster_variant'] for p in pairs})>1
    credible=complete and not reasons and not position_changes
    return dict(blocks=table,variants=variants,adjacent_pairs=pairs,cycles=cycles,complete=complete,
        expected_samples=8*expected_images,actual_samples=sum(len(b['rows']) for b in blocks),drift_threshold=threshold,
        drift_reasons=reasons,timing_status='incomplete' if not complete else 'timing_unstable' if reasons else 'screen_passed',
        position_ordering_changes=position_changes,comparison_credible=credible,
        speed_conclusion='pending_Work_review' if not credible else 'passed_engineering_screen_not_proof_of_unconfounded_speed',
        no_automatic_retries=True,scope='All samples retained; window/condition ratios are engineering screens, not statistical significance')
