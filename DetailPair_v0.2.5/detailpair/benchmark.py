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
