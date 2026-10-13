"""Recompute a descriptive review from the received ZIP extraction. No training."""
from pathlib import Path
import csv, json, math, re, hashlib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'received/DetailPair_results_20261013_054850_726320'
OUT = ROOT / 'long_review_20261013'
OUT.mkdir(exist_ok=True)
parent = next((SOURCE / 'runs').glob('long_*'))
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
report = {'source': str(SOURCE), 'zip_sha256': hashlib.sha256((ROOT.parent / 'upload/DetailPair_results_20261013_054850_726320.zip').read_bytes()).hexdigest(), 'conditions': {}, 'classes': [], 'scope': 'Returned Windows real-data validation; descriptive single training seed, not a significance test'}
selections = []
for pilot in sorted((SOURCE / 'runs').glob('pilot_*')):
    condition = next(pilot.glob('B*_s0'))
    variant = condition.name.split('_')[0]
    metrics = read(condition / 'metrics.json')
    with (condition / 'training/results.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
    selections.append({k: [{q: v for q, v in row.items() if q != 'working_image'} for row in records] for k, records in read(pilot / 'subset_s0/selection.json').items()})
    aux_count = nonfinite = 0
    for line in (condition / 'training/auxiliary.jsonl').open(encoding='utf-8'):
        v = json.loads(line)
        aux_count += 1
        nonfinite += any(isinstance(x, (int, float)) and not math.isfinite(x) for x in v.values())
    text = (parent / f'{variant}_console.log').read_text(encoding='utf-8')
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # Final independent validation table, rather than the earlier trainer table.
    tail = clean[clean.rfind('                   all'):]
    for line in tail.splitlines():
        m = re.fullmatch(r'\s*(crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*', line)
        if m:
            report['classes'].append(dict(variant=variant, name=m[1], images=int(m[2]), gt=int(m[3]), precision=float(m[4]), recall=float(m[5]), ap50=float(m[6]), ap5095=float(m[7]), source='Final console table; rounded, official validator working point, not fixed conf=0.25'))
    speed = re.search(r'Speed: ([\d.]+)ms preprocess, ([\d.]+)ms inference, ([\d.]+)ms loss, ([\d.]+)ms postprocess per image', tail)
    report['conditions'][variant] = dict(pilot=pilot.name, metrics=metrics, actual_epochs=len(rows), csv_best_epoch=int(best['epoch']), csv_best_ap5095=float(best['metrics/mAP50-95(B)']), csv_elapsed_hours=float(rows[-1]['time'])/3600, aux_count=aux_count, aux_nonfinite_rows=nonfinite, console_speed_ms=list(map(float,speed.groups())) if speed else None)
report['same_source_selection_and_order'] = selections[0] == selections[1]
report['counts'] = {k: len(v) for k, v in selections[0].items()}
report['deltas_pp_B6_minus_B2'] = {k: (report['conditions']['B6']['metrics'][k]-report['conditions']['B2']['metrics'][k])*100 for k in report['conditions']['B2']['metrics'] if k.startswith('metrics/')}
(OUT / 'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT / 'class_metrics_from_console.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=report['classes'][0].keys());writer.writeheader();writer.writerows(report['classes'])
with (OUT / 'training_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['variant','epoch','elapsed_seconds','val_mAP50','val_mAP50_95'])
    for pilot in sorted((SOURCE / 'runs').glob('pilot_*')):
        c=next(pilot.glob('B*_s0'))
        with (c/'training/results.csv').open() as g:
            for row in csv.DictReader(g):writer.writerow([c.name.split('_')[0],row['epoch'],row['time'],row['metrics/mAP50(B)'],row['metrics/mAP50-95(B)']])
print(json.dumps(report,ensure_ascii=False,indent=2))
