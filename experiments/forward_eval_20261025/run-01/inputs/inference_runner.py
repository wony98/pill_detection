"""Run the frozen B-stage input once. No labels, training, downloads or scoring."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def checked_inputs(root):
    manifest = json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'healtheat-forward-input-v1':
        raise ValueError('INVALID_INPUT_SCHEMA')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('INPUT_HASH_MISMATCH')
    observed = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and p.name != 'inputs.json'}
    if observed != set(manifest['files']):
        raise ValueError('UNDECLARED_OR_MISSING_INPUT')
    if sha(Path(__file__)) != manifest['files']['inference_runner.py']:
        raise ValueError('RUNNER_VERSION_MISMATCH')
    return manifest


def inference_config(cfg, names):
    """Keep the v4 image transforms; explicitly remove annotation and remote-init paths."""
    cfg = cfg.copy()
    pipeline = cfg.test_dataloader.dataset.pipeline
    if [x['type'] for x in pipeline] != ['LoadImageFromFile','Resize','Pad','LoadAnnotations','PackDetInputs']:
        raise ValueError('UNREVIEWED_PIPELINE')
    if tuple(pipeline[1]['scale']) != (1536,1536) or tuple(pipeline[2]['size']) != (1536,1536):
        raise ValueError('UNEXPECTED_V4_IMAGE_SIZE')
    if cfg.model.bbox_head.num_classes != 118 or list(cfg.metainfo.classes) != names:
        raise ValueError('CONFIG_CLASS_MAPPING_MISMATCH')
    cfg.model.backbone.init_cfg = None
    cfg.load_from = None
    pipeline = [x for x in pipeline if x['type'] != 'LoadAnnotations']
    cfg.test_dataloader.dataset.pipeline = pipeline
    cfg.test_dataloader.dataset.ann_file = ''
    return cfg, pipeline


def validate_raw(row, nclasses=118):
    import numpy as np
    boxes = np.asarray(row['bboxes'], dtype=np.float32).reshape(-1,4)
    scores = np.asarray(row['scores'], dtype=np.float32)
    labels = np.asarray(row['labels'])
    if (scores.ndim != 1 or labels.ndim != 1 or len(boxes) != len(scores) or len(scores) != len(labels)
            or len(scores) > 300 or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (boxes[:,2:] < boxes[:,:2]).any() or (scores < 0).any() or (scores > 1).any()
            or any(type(v) is not int or not 0 <= v < nclasses for v in row['labels'])):
        raise ValueError('INVALID_RAW_PREDICTIONS')
    return boxes, scores, labels.astype(np.int64)


def run(root, output, device, smoke):
    root = Path(root).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('OUTPUT_EXISTS_OR_INSIDE_INPUT')
    manifest = checked_inputs(root)
    input_sha = sha(root/'inputs.json')
    output.mkdir(parents=True)
    started = time.time()
    try:
        import numpy as np
        import torch
        import yaml
        from mmengine.config import Config
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector, inference_detector

        torch.set_num_threads(4)
        torch.manual_seed(100)
        np.random.seed(100)
        if device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('REQUESTED_CUDA_UNAVAILABLE')
        names_map = yaml.safe_load((root/'pill.yaml').read_text(encoding='utf-8'))['names']
        names = [names_map[i] for i in range(118)]
        cfg, pipeline = inference_config(Config.fromfile(str(root/'config.py')), names)
        cfg.dump(str(output/'effective_config.py'))
        env = {'python':sys.version, 'platform':platform.platform(), 'device':device,
               'packages':{n:importlib.metadata.version(n) for n in
                           ['torch','torchvision','mmcv','mmengine','mmdet','numpy','PyYAML']},
               'torch_threads':4, 'cuda':torch.version.cuda,
               'gpu':torch.cuda.get_device_name(0) if device != 'cpu' else None}
        write(output/'environment.json',env)
        # Explicit palette avoids dataset construction / annotation-file access in init_detector.
        model = init_detector(cfg, str(root/'model.pth'), device=device, palette='coco')
        if list(model.dataset_meta['classes']) != names:
            raise ValueError('CHECKPOINT_CLASS_MAPPING_MISMATCH')
        selected = manifest['images'][:1] if smoke else manifest['images']
        rows = []
        for name in selected:
            if device != 'cpu':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred = inference_detector(model, str(root/'images'/name), test_pipeline=Compose(pipeline)).pred_instances
            if device != 'cpu':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter()-t0)*1000
            if pred.bboxes.dtype != torch.float32 or pred.scores.dtype != torch.float32:
                raise ValueError('UNEXPECTED_RAW_PRECISION')
            row = {'filename':name, 'bboxes':pred.bboxes.cpu().tolist(),
                   'scores':pred.scores.cpu().tolist(), 'labels':pred.labels.cpu().tolist(),
                   'elapsed_ms':elapsed, 'float_dtype':'float32'}
            validate_raw(row)
            rows.append(row)
            write(output/'partial_predictions.json', {'rows':rows, 'complete':False})
            print(json.dumps({'image':len(rows), 'of':len(selected), 'boxes':len(row['scores']),
                              'elapsed_ms':round(elapsed,1)}), flush=True)
        checked_inputs(root)
        if sha(root/'inputs.json') != input_sha:
            raise ValueError('MANIFEST_CHANGED_DURING_RUN')
        write(output/'raw_predictions.json', {'schema':'healtheat-raw-v1', 'rows':rows,
              'output_stage':'MMDetection pred_instances after model NMS, before Top-3 grouping',
              'input_manifest_sha256':input_sha})
        import resource
        receipt = {'status':'PASS', 'model_executed':True, 'training':False,
                   'scope':'smoke' if smoke else 'full', 'image_count':len(rows),
                   'input_manifest_sha256':input_sha, 'started_unix':started, 'finished_unix':time.time(),
                   'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device != 'cpu' else None,
                   'artifacts':{n:sha(output/n) for n in
                                ['raw_predictions.json','environment.json','effective_config.py']},
                   'overrides':['backbone.init_cfg=None','load_from=None','ann_file empty',
                                'LoadAnnotations removed','explicit palette=coco; no dataset construction']}
        write(output/'run.json',receipt)
        return receipt
    except BaseException as exc:
        write(output/'failure.json', {'status':'FAILED','error_type':type(exc).__name__,
              'input_manifest_sha256':input_sha,'started_unix':started,'finished_unix':time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.inputs,args.output,args.device,args.smoke),ensure_ascii=False))
