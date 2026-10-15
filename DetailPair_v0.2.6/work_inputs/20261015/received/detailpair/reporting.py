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
