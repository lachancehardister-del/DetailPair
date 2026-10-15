"""Decoded host image -> host detections, fixed FP32 square input, batch one."""
import datetime, os, subprocess, time
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from .reporting import dump, latency_stats


def gpu_guard(device, destination):
    if device.type != 'cuda': return {'scope':'CPU; GPU exclusivity not applicable'}
    command = ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    record = dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    dump(destination, record)
    if result.returncode: raise RuntimeError('Cannot establish GPU process state; see '+str(destination))
    other = []
    for line in result.stdout.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) < 4 or not fields[1].isdigit():
            raise RuntimeError('Unrecognized nvidia-smi process response: '+line)
        if int(fields[1]) != os.getpid(): other.append(line)
    if other: raise RuntimeError('Other GPU compute processes found; stop them yourself before benchmarking: '+'; '.join(other))
    return record


class HostDetector:
    def __init__(self, model, device, config):
        self.device, self.config = device, config
        self.backend = AutoBackend(model=model, device=device, fp16=False, fuse=config['fuse'], verbose=False)
        self.backend.eval()
        self.letterbox = LetterBox((config['imgsz'],config['imgsz']), auto=False, stride=32)
        self.nc = int(model.model[-1].nc)
        self.shape = None
        self.metadata = dict(parameters=sum(p.numel() for p in self.backend.model.parameters()),
            requested_fuse=config['fuse'], batchnorm_modules=sum(isinstance(m,torch.nn.BatchNorm2d) for m in self.backend.model.modules()),
            dtype=str(next(self.backend.model.parameters()).dtype), flops=None,
            flops_status='not measured; pre-replacement YOLO summary is not used')

    @torch.inference_mode()
    def predict(self, bgr):
        im = self.letterbox(image=bgr)
        im = np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))
        tensor = torch.from_numpy(im).unsqueeze(0).to(self.device).float()/255.
        self.shape = list(tensor.shape)
        if self.shape != [1,3,self.config['imgsz'],self.config['imgsz']]: raise RuntimeError('Actual input shape changed')
        outputs = self.backend(tensor)
        pred = non_max_suppression(outputs, conf_thres=self.config['conf'], iou_thres=self.config['nms_iou'],
            max_det=self.config['max_det'], multi_label=False, nc=self.nc, max_time_img=10.)[0]
        pred[:,:4] = ops.scale_boxes(tensor.shape[2:], pred[:,:4], bgr.shape[:2])
        host = pred[:,:6].detach().cpu().numpy()
        if not np.isfinite(host).all(): raise RuntimeError('Nonfinite host detections')
        return host

    def timed(self, bgr):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        begin = time.perf_counter_ns()
        prediction = self.predict(bgr)
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter_ns()-begin)/1e6
        return prediction, elapsed


def measure(detector, images, identity, config, out):
    """Images are already decoded. JSON conversion and disk writes stay outside timer."""
    if not images: raise ValueError('No decoded images')
    gpu_guard(detector.device, out/'gpu_processes_before_warmup.json')
    for i in range(config['warmup']): detector.predict(images[i%len(images)]['bgr'])
    if detector.device.type == 'cuda': torch.cuda.synchronize(detector.device)
    rows = []
    # Durable partial records remain available after an interrupted/failed measurement.
    with (out/'latency.jsonl').open('w', encoding='utf-8') as f:
        import json
        for repeat in range(config['repeats']):
            gpu_guard(detector.device, out/f'gpu_processes_round_{repeat}_before.json')
            for image in images:
                pred, elapsed = detector.timed(image['bgr'])
                row = dict(**identity, repeat=repeat, image_id=image['image_id'], latency_ms=elapsed,
                           predictions=len(pred), actual_shape=detector.shape,
                           timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
                rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
            gpu_guard(detector.device, out/f'gpu_processes_round_{repeat}_after.json')
    expected = len(images)*config['repeats']
    if len(rows) != expected: raise RuntimeError('Incomplete latency measurement')
    summary = dict(scope='decoded host BGR -> host xyxy/conf/class; letterbox, normalize, H2D, FP32 inference, class-aware NMS, D2H; excludes disk/decode/load/warmup/plot/write',
        warmup_excluded=config['warmup'], expected_samples=expected, actual_shape=detector.shape,
        overall=latency_stats(rows), rounds={str(r):latency_stats([v for v in rows if v['repeat']==r]) for r in range(config['repeats'])},
        model=detector.metadata, repeated_images_are_not_training_seeds=True)
    dump(out/'latency_summary.json', summary)
    return summary


class DetectorPool:
    """Retain fused CPU models; only the active model's weights reside on GPU."""
    def __init__(self, models, device, config):
        import copy
        self.device=device;self.active=None
        self.detectors={v:HostDetector(copy.deepcopy(y.model).float().eval(),torch.device('cpu'),config) for v,y in models.items()}

    def lease(self, variant):
        from contextlib import contextmanager
        @contextmanager
        def active():
            if self.active is not None:raise RuntimeError('Concurrent model lease forbidden')
            self.active=variant;det=self.detectors[variant]
            try:
                det.backend.to(self.device);det.backend.device=self.device;det.device=self.device
                yield det
            finally:
                det.backend.to('cpu');det.backend.device=torch.device('cpu');det.device=torch.device('cpu')
                self.active=None
                if self.device.type=='cuda':torch.cuda.empty_cache()
        return active()


def device_snapshot(device, destination, notes):
    """Low-frequency snapshots outside measurement blocks; unknowns stay unknown."""
    import platform
    record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),device=str(device),
        CPU=dict(processor=platform.processor() or 'unknown',machine=platform.machine(),threads=torch.get_num_threads()),
        power_mode=notes.get('power_mode','unknown'),plugged_in=notes.get('plugged_in','unknown'),hardware_notes=notes)
    def command(args):
        try:
            r=subprocess.run(args,capture_output=True,text=True,timeout=15)
            return dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as e:return dict(command=args,status='unknown',error=str(e))
    if device.type=='cuda':
        fields='index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.used,memory.total'
        record['gpu_state']=command(['nvidia-smi','--query-gpu='+fields,'--format=csv'])
        import csv,io
        raw=record['gpu_state']
        if raw.get('exit_code')==0:
            parsed=list(csv.DictReader(io.StringIO(raw['stdout'])))
            record['gpu_fields']=[{k.strip():('unknown' if not isinstance(v,str) or v.strip() in {'N/A','[N/A]','Not Supported','[Not Supported]',''} else v.strip()) for k,v in row.items() if k is not None} for row in parsed] if parsed else 'unknown'
        else:record['gpu_fields']='unknown'
        record['gpu_processes']=gpu_guard(device,destination.with_name(destination.stem+'_processes.json'))
        record['unavailable_fields']='N/A, failed/unsupported queries are unknown; no thermal throttling inference'
    else:record['gpu_state']='not applicable: synthetic CPU check'
    if os.name=='nt':
        record['powercfg']=command(['powercfg','/getactivescheme'])
        record['battery_status']=command(['powershell','-NoProfile','-NonInteractive','-Command',
            'Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json -Compress'])
    dump(destination,record);return record


def initialize_torchvision_nms(device):
    import torchvision
    boxes=torch.tensor([[0.,0.,10.,10.],[.5,.5,9.5,9.5]],device=device)
    kept=torchvision.ops.nms(boxes,torch.tensor([.9,.8],device=device),.7).cpu().tolist()
    if kept!=[0]:raise RuntimeError('torchvision NMS bootstrap returned unexpected indices')
    return dict(device=str(device),indices=kept,iou_threshold=.7,scope='kernel initialization; excluded from timing')


def balanced_measure(pool, images, identities, config, out):
    """One ABBA+BAAB pass, bounded common warmup, no per-image writes/polls."""
    import json
    from .reporting import balanced_stats,write_csv
    out.mkdir(parents=True,exist_ok=True)
    order=['B2','B6','B6','B2','B6','B2','B2','B6']
    if config['block_order']!=order:raise ValueError('Balanced order changed')
    if not images:raise ValueError('No clean images')
    warm=[];blocks=[];counts={'B2':0,'B6':0}
    def save_rows(path,rows):
        with path.open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    def warm_call(det,variant,index,phase,block_id=None):
        pred,elapsed=det.timed(images[index%len(images)]['bgr'])
        warm.append(dict(phase=phase,variant=variant,index=index,block_id=block_id,duration_ms=elapsed,
            actual_shape=det.shape,detections=len(pred),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    try:
        device_snapshot(pool.device,out/'device_initial.json',config['hardware_notes'])
        if pool.device.type=='cuda':dump(out/'torchvision_initialization.json',initialize_torchvision_nms(pool.device))
        # Initialize both actual model/NMS routes before the timed common warm phase.
        for v in ['B2','B6']:
            with pool.lease(v) as det:warm_call(det,v,0,'bootstrap')
        print('Common paired warmup target seconds:',config['common_warmup_seconds'],flush=True)
        begin=time.perf_counter()
        while time.perf_counter()-begin<config['common_warmup_seconds'] or counts['B2']==0:
            # Complete equal pairs; final pair may overrun target by one pair duration.
            for v in ['B2','B6']:
                with pool.lease(v) as det:warm_call(det,v,counts[v],'common')
                counts[v]+=1
        dump(out/'common_warmup_summary.json',dict(target_seconds=config['common_warmup_seconds'],
            actual_seconds=time.perf_counter()-begin,counts=counts,equal_counts=counts['B2']==counts['B6'],
            rule='paired alternating B2,B6; includes transport gaps; all warmup excluded from measurement'))
        save_rows(out/'warmup.jsonl',warm)
        print('Common warmup complete; counts:',counts,flush=True)
        for block_id,variant in enumerate(order):
            print('Starting block',block_id+1,'of 8:',variant,'; excluded warmup',config['block_warmup_iterations'],flush=True)
            rows=[];block=dict(block_id=block_id,variant=variant,rows=rows);blocks.append(block)
            dest=out/f'block_{block_id:02d}_{variant}';dest.mkdir()
            try:
                device_snapshot(pool.device,dest/'before.json',config['hardware_notes'])
                with pool.lease(variant) as det:
                    for i in range(config['block_warmup_iterations']):warm_call(det,variant,i,'block',block_id)
                    identity={k:identities[variant][k] for k in ['variant','seed','run_name','weights_sha256']}
                    for image in images:
                        pred,elapsed=det.timed(image['bgr'])
                        rows.append(dict(**identity,block_id=block_id,image_id=image['image_id'],latency_ms=elapsed,
                            actual_shape=det.shape,detections=len(pred),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                    dump(dest/'model.json',det.metadata)
                save_rows(dest/'latency.jsonl',rows)
                device_snapshot(pool.device,dest/'after.json',config['hardware_notes'])
                dump(dest/'status.json',dict(status='success',samples=len(rows),warmup_excluded=config['block_warmup_iterations']))
                print('Completed block',block_id+1,'samples',len(rows),flush=True)
            except BaseException:
                import traceback
                save_rows(dest/'latency_partial.jsonl',rows)
                (dest/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                dump(dest/'status.json',dict(status='failed',completed_samples=len(rows)))
                raise
            finally:save_rows(out/'warmup.jsonl',warm)
        summary=balanced_stats(blocks,len(images),config['drift_threshold'])
        dump(out/'balanced_summary.json',summary)
        write_csv(out/'block_statistics.csv',[{k:v for k,v in b.items() if k!='windows'} for b in summary['blocks']])
        all_rows=[r for b in blocks for r in b['rows']];save_rows(out/'latency_all.jsonl',all_rows)
        return summary
    except BaseException:
        save_rows(out/'warmup_partial.jsonl',warm)
        save_rows(out/'latency_all_partial.jsonl',[r for b in blocks for r in b['rows']])
        dump(out/'balanced_summary_partial.json',balanced_stats(blocks,len(images),config['drift_threshold']))
        raise
