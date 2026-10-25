"""Import historical HealthEat records into a local MLflow store. No model execution."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlparse
from urllib.request import url2pathname
import zipfile

RECIPE = 'healtheat-backfill-v1'
EVAL_PARAMS = ('exp_name', 'model_framework', 'model_params_M', 'batch_size_used',
               'seed', 'conf_threshold', 'nms_iou', 'best_epoch', 'eval_version')
CONFIG_PATHS = {'seed': ('randomness', 'seed'), 'max_epochs': ('max_epochs',),
                'batch_size': ('train_dataloader', 'batch_size'),
                'num_classes': ('model', 'bbox_head', 'num_classes'),
                'learning_rate': ('optim_wrapper', 'optimizer', 'lr')}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bbox_ok(box):
    return (isinstance(box, list) and len(box) == 4 and all(finite(x) for x in box)
            and box[2] >= box[0] and box[3] >= box[1])


def validate_evaluation(data):
    rows = data['results']
    if not rows or len(rows) != data['num_images']:
        raise ValueError('EVAL_IMAGE_COUNT_MISMATCH')
    names = [unicodedata.normalize('NFC', row['filename']) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('EVAL_DUPLICATE_IMAGE')
    if sum(len(row['gt_bboxes']) for row in rows) != data['num_gt_bboxes'] or data['num_gt_bboxes'] <= 0:
        raise ValueError('EVAL_GT_COUNT_MISMATCH')
    for row in rows:
        for gt in row['gt_bboxes']:
            if not bbox_ok(gt['bbox']) or not isinstance(gt['dl_name'], str):
                raise ValueError('EVAL_BAD_GT')
        for pred in row['predictions']:
            classes, scores = pred['top3_class_name'], pred['top3_confidence']
            if (not bbox_ok(pred['bbox']) or not isinstance(classes, list)
                    or len(classes) != len(scores) or len(classes) > 3
                    or not all(isinstance(c, str) for c in classes)
                    or not all(finite(s) and 0 <= s <= 1 for s in scores)):
                raise ValueError('EVAL_BAD_PREDICTION')
    if not finite(data['ms_per_image']) or data['ms_per_image'] < 0:
        raise ValueError('EVAL_BAD_REPORTED_LATENCY')


def iou_xyxy(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection/(area_a+area_b-intersection+1e-9)


def score_evaluation(data):
    """Preserve the notebook's GT-order greedy IoU >= 0.5, one-use prediction rule."""
    validate_evaluation(data)
    matched = top1 = top3 = false_pos = full1 = full3 = 0
    for row in data['results']:
        used = set()
        all1 = all3 = bool(row['gt_bboxes'])
        for gt in row['gt_bboxes']:
            best_iou, best_index = 0.0, -1
            for j, pred in enumerate(row['predictions']):
                if j in used:
                    continue
                overlap = iou_xyxy(gt['bbox'], pred['bbox'])
                if overlap > best_iou:
                    best_iou, best_index = overlap, j
            if best_iou >= 0.5 and best_index >= 0:
                used.add(best_index)
                matched += 1
                classes = [unicodedata.normalize('NFC', c) for c in row['predictions'][best_index]['top3_class_name']]
                name = unicodedata.normalize('NFC', gt['dl_name'])
                hit1, hit3 = bool(classes) and classes[0] == name, name in classes
                top1 += hit1
                top3 += hit3
                all1, all3 = all1 and hit1, all3 and hit3
            else:
                all1 = all3 = False
        full1 += all1
        full3 += all3
        false_pos += len(row['predictions'])-len(used)
    n, g = len(data['results']), data['num_gt_bboxes']
    return {'detection_rate': matched/g, 'top1_accuracy': top1/g, 'top3_accuracy': top3/g,
            'image_top1': full1/n, 'image_top3': full3/n, 'n_top1': top1, 'n_top3': top3,
            'n_match': matched, 'n_false_pos': false_pos, 'total_gt': g, 'n_imgs': n,
            'reported_ms_per_image': data['ms_per_image']}


def gt_digest(data):
    # Order inside an image affects greedy matching, so only sort images, never GT boxes.
    return digest(canonical(sorted([{'filename': unicodedata.normalize('NFC', r['filename']),
                                    'gt_bboxes': r['gt_bboxes']} for r in data['results']], key=lambda r: r['filename'])))


def config_params(text):
    """Read literal settings without importing or executing the historical Python config."""
    assignments = {n.targets[0].id: n.value for n in ast.parse(text).body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    result = {}
    for key, path in CONFIG_PATHS.items():
        node = assignments.get(path[0])
        for segment in path[1:]:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
                node = next((k.value for k in node.keywords if k.arg == segment), None)
            elif isinstance(node, ast.Dict):
                node = next((v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value == segment), None)
            else:
                node = None
        if isinstance(node, ast.Constant) and finite(node.value):
            result[key] = node.value
    return result


def scalar_metrics(raw):
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    points = []
    seen = set()
    for row in rows:
        step = row.get('step')
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError('SCALAR_INVALID_STEP')
        for key,value in row.items():
            if key == 'step':
                continue
            if not finite(value) or not re.fullmatch(r'[\w./ -]+', key):
                raise ValueError('SCALAR_INVALID_METRIC')
            if (key,step) in seen:
                raise ValueError('SCALAR_DUPLICATE_KEY_STEP')
            seen.add((key,step))
            points.append((key,float(value),step))
    if not points:
        raise ValueError('SCALAR_EMPTY')
    return rows, points


def prepare(manifest):
    """Validate every input before opening a tracking store; paths stay in private provenance."""
    records, checkpoints, provenance, snapshots = [], [], [], []
    labels = [item['label'] for group in ('archives','evaluations','checkpoints') for item in manifest[group]]
    if len(labels) != len(set(labels)) or not all(re.fullmatch(r'[a-zA-Z0-9_-]+', s) for s in labels):
        raise ValueError('SOURCE_LABELS_MUST_BE_UNIQUE_SAFE_IDS')
    def source(item):
        path = Path(item['path']).resolve(strict=True)
        sha = file_digest(path)
        snapshots.append((path,sha))
        provenance.append({'label':item['label'], 'path':str(path), 'sha256':sha,
                           'mtime_ns':path.stat().st_mtime_ns, 'mtime_is_not_experiment_time':True})
        return path,sha
    for item in manifest['archives']:
        path,sha = source(item)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('ARCHIVE_DUPLICATE_MEMBER')
            scalar_names = [n for n in names if n.endswith('/vis_data/scalars.json')]
            config_names = [n for n in names if n.endswith('/vis_data/config.py')]
            weight_names = [n for n in names if re.search(r'(^|/)best_coco_bbox_mAP_epoch_\d+\.pth$', n)]
            if len(scalar_names) != 1 or len(config_names) != 1 or len(weight_names) != 1:
                raise ValueError('ARCHIVE_AMBIGUOUS_CONTENTS')
            scalar_name,config_name,weight_name = scalar_names[0],config_names[0],weight_names[0]
            if scalar_name.rsplit('/',1)[0] != config_name.rsplit('/',1)[0]:
                raise ValueError('ARCHIVE_CONFIG_SCALAR_DIRECTORY_MISMATCH')
            if any(z.getinfo(n).file_size > 10_000_000 for n in (scalar_name,config_name)):
                raise ValueError('ARCHIVE_TEXT_TOO_LARGE')
            raw_scalar,raw_config = z.read(scalar_name),z.read(config_name)
            rows,points = scalar_metrics(raw_scalar)
            with z.open(weight_name) as stream:
                weight_sha = stream_digest(stream)
            checkpoints.append({'source':item['label'], 'member':weight_name, 'sha256':weight_sha,
                                'bytes':z.getinfo(weight_name).file_size, 'evaluation_link':'UNRESOLVED',
                                'evidence':'co-packaged in this archive; no checkpoint load or evaluation binding'})
            evidence = {'archive_sha256':sha, 'config_member':config_name, 'config_sha256':digest(raw_config),
                        'scalar_member':scalar_name,'scalar_sha256':digest(raw_scalar),'scalar_rows':len(rows),
                        'checkpoint_member':weight_name,'checkpoint_sha256':weight_sha,
                        'historical_time_label':scalar_name.split('/')[0], 'historical_timezone':'UNKNOWN',
                        'bundle_link':'same archive and log directory; not proof of evaluation identity',
                        'evaluation_link':'UNRESOLVED'}
            records.append({'label':item['label'],'kind':'training_history','source_sha256':sha,
                            'params':config_params(raw_config.decode('utf-8')), 'points':points, 'evidence':evidence,
                            'artifact_bytes':{'config.py.txt':raw_config,'scalars.jsonl':raw_scalar}})
    for item in manifest['evaluations']:
        path,sha = source(item)
        raw = path.read_bytes()
        if digest(raw) != sha:
            raise ValueError('INPUT_CHANGED_DURING_READ')
        data = json.loads(raw.decode('utf-8-sig'))
        metrics = score_evaluation(data)
        evidence = {'evaluation_sha256':sha,'gt_set_sha256':gt_digest(data),
                    'reported_created_at':data.get('created_at','UNKNOWN'),'historical_timezone':'UNKNOWN',
                    'checkpoint_link':'UNRESOLVED','config_link':'UNRESOLVED',
                    'reason':'evaluation JSON has no checkpoint/config hash or path',
                    'metric_rule':'existing notebook GT-order greedy IoU>=0.5; stored predictions; no new inference'}
        records.append({'label':item['label'],'kind':'evaluation_history','source_sha256':sha,
                        'params':{k:data[k] for k in EVAL_PARAMS},
                        'points':[(k,float(v),0) for k,v in metrics.items()], 'evidence':evidence,
                        'artifact_bytes':{'evaluation.json':raw}})
    gt_hashes = {r['evidence']['gt_set_sha256'] for r in records if r['kind']=='evaluation_history'}
    if len(gt_hashes) != 1:
        raise ValueError('EVALUATION_GT_SETS_DIFFER')
    for item in manifest['checkpoints']:
        path,sha = source(item)
        checkpoints.append({'source':item['label'],'member':path.name,'sha256':sha,'bytes':path.stat().st_size,
                            'hub_url':item.get('hub_url',''), 'hub_status':'documented reference; remote identity unverified',
                            'evaluation_link':'UNRESOLVED','evidence':'file bytes only; weights never loaded'})
    notebook, notebook_sha = source({'label':'evaluation_notebook','path':manifest['evaluation_notebook']})
    assets = []
    for group,pattern in [('images','*.png'),('labels','*.txt')]:
        paths = sorted(Path(manifest['evaluation_assets'][group]).glob(pattern))
        if not paths:
            raise ValueError('EVALUATION_ASSETS_MISSING')
        entries = []
        for path in paths:
            h = file_digest(path)
            snapshots.append((path,h))
            entries.append({'name':path.name,'sha256':h})
        assets.append({'group':group,'count':len(entries),'sha256':digest(canonical(entries)),'entries':entries})
    expected_names = {r['filename'] for r in json.loads(next(r['artifact_bytes']['evaluation.json'] for r in records if r['kind']=='evaluation_history'))['results']}
    if expected_names != {e['name'] for e in assets[0]['entries']}:
        raise ValueError('EVALUATION_IMAGE_NAMES_DIFFER')
    if {Path(n).stem for n in expected_names} != {Path(e['name']).stem for e in assets[1]['entries']}:
        raise ValueError('EVALUATION_LABEL_NAMES_DIFFER')
    for record in records:
        record['evidence']['evaluation_notebook_sha256'] = notebook_sha
        record['evidence']['current_assets_sha256'] = digest(canonical(assets))
        record['evidence']['historical_asset_identity'] = 'UNRESOLVED: hashes describe present files, not files at evaluation time'
    return records,checkpoints,provenance,snapshots,assets


def verify_run(client, run_id, record, tags, artifact_bytes, store, expected_status):
    observed = client.get_run(run_id)
    if (observed.info.status != expected_status
            or observed.data.params != {k:str(v) for k,v in record['params'].items()}
            or any(observed.data.tags.get(k) != v for k,v in tags.items())):
        raise ValueError('RUN_READBACK_MISMATCH')
    for key in {p[0] for p in record['points']}:
        expected = sorted((step,value) for k,value,step in record['points'] if k == key)
        actual = sorted((m.step,m.value) for m in client.get_metric_history(run_id,key))
        if expected != actual:
            raise ValueError('METRIC_HISTORY_READBACK_MISMATCH')
    if set(observed.data.metrics) != {p[0] for p in record['points']}:
        raise ValueError('UNEXPECTED_RUN_METRIC')
    uri = urlparse(observed.info.artifact_uri)
    if uri.scheme != 'file' or uri.netloc:
        raise ValueError('NONLOCAL_ARTIFACT_URI')
    root = Path(url2pathname(uri.path)).resolve(strict=True)
    if root != (store/'artifacts'/run_id/'artifacts').resolve():
        raise ValueError('ARTIFACT_ROOT_MISMATCH')
    actual_paths = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual_paths != set(artifact_bytes):
        raise ValueError('ARTIFACT_SET_MISMATCH')
    for name, raw in artifact_bytes.items():
        p = root/name
        if not p.resolve().is_relative_to(root) or file_digest(p) != digest(raw):
            raise ValueError('ARTIFACT_HASH_MISMATCH')


def ingest(records, store, out, provenance, checkpoints, snapshots, assets):
    if out.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS: choose a new output folder; do not overwrite evidence')
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param
    code_path = Path(__file__).resolve()
    code_bytes = code_path.read_bytes()
    code_sha = digest(code_bytes)
    requirement_path = code_path.with_name('requirements.txt')
    requirement_bytes = requirement_path.read_bytes()
    requirement_sha = digest(requirement_bytes)
    if importlib.metadata.version('mlflow') != '3.17.2':
        raise ValueError('MLFLOW_VERSION_MISMATCH')
    snapshots = [*snapshots, (code_path,code_sha), (requirement_path,requirement_sha)]
    def check_sources():
        if any(file_digest(p) != h for p,h in snapshots):
            raise ValueError('SOURCE_CHANGED_DURING_BACKFILL')
    check_sources()
    environment = {'code_sha256':code_sha, 'requirements_sha256':requirement_sha,
                   'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    store.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    uri = 'sqlite:///'+(store/'tracking.db').as_posix()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    name = 'HealthEat historical backfill'
    experiment = client.get_experiment_by_name(name)
    if experiment and (experiment.artifact_location != (store/'artifacts').as_uri() or experiment.lifecycle_stage != 'active'):
        raise ValueError('EXISTING_EXPERIMENT_LOCATION_OR_STATE_MISMATCH')
    exp_id = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(store/'artifacts').as_uri())
    now = int(time.time()*1000)
    exports, new_runs = [], []
    try:
        for record in records:
            identity = {k:record[k] for k in ('label','kind','source_sha256','params','points','evidence')}
            artifact_bytes = {'historical_source/'+k:v for k,v in record['artifact_bytes'].items()}
            artifact_bytes.update({'lineage.json':canonical(record['evidence']),
                                   'checkpoint_inventory.json':canonical(checkpoints),
                                   'backfill_environment.json':canonical(environment),
                                   'backfill_code/backfill_runs.py':code_bytes,
                                   'backfill_code/requirements.txt':requirement_bytes})
            artifact_hashes = {k:digest(v) for k,v in artifact_bytes.items()}
            fingerprint = digest(canonical({'recipe':RECIPE, 'record':identity, 'artifacts':artifact_hashes}))
            tags = {'mlflow.runName':record['label'],'record_kind':record['kind'],
                    'backfill_fingerprint':fingerprint,'recipe':RECIPE,'capture_mode':'historical_backfill',
                    'checkpoint_evaluation_link':'UNRESOLVED','code_sha256':code_sha,
                    'metric_timestamp_semantics':'backfill time, not historical execution time'}
            existing = client.search_runs([exp_id], filter_string=f"tags.backfill_fingerprint = '{fingerprint}'", max_results=2)
            if len(existing)>1 or (existing and existing[0].info.status != 'FINISHED'):
                raise ValueError('EXISTING_RUN_NOT_UNIQUELY_FINISHED: preserve it; inspect before retry')
            if existing:
                run_id, reused = existing[0].info.run_id, True
            else:
                run = client.create_run(exp_id, tags=tags)
                run_id, reused = run.info.run_id, False
                new_runs.append(run_id)
                client.log_batch(run_id, params=[Param(k,str(v)) for k,v in record['params'].items()])
                points = [Metric(k,v,now,step) for k,v,step in record['points']]
                for start in range(0,len(points),500):
                    client.log_batch(run_id, metrics=points[start:start+500])
                for filename,raw in artifact_bytes.items():
                    p=out/record['label']/filename
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(raw)
                    parent=Path(filename).parent.as_posix()
                    client.log_artifact(run_id,str(p),None if parent=='.' else parent)
            verify_run(client,run_id,record,tags,artifact_bytes,store,'FINISHED' if reused else 'RUNNING')
            row={'run_id':run_id,'label':record['label'],'kind':record['kind'],'source_sha256':record['source_sha256'],
                 'fingerprint':fingerprint,'reused':reused,'metric_points':len(record['points']),'checkpoint_link':'UNRESOLVED'}
            if record['kind']=='evaluation_history':
                row.update({k:v for k,v,_ in record['points']})
            exports.append(row)
        check_sources()
        # New runs remain RUNNING until the entire batch and source hashes are verified.
        for run_id in new_runs:
            client.set_terminated(run_id,'FINISHED')
    except Exception as exc:
        status_errors=[]
        for run_id in new_runs:
            try:
                client.set_terminated(run_id,'FAILED')
            except Exception as status_exc:
                status_errors.append({'run_id':run_id,'error_type':type(status_exc).__name__})
        write_json(out/'failure.json',{'status':'FAILED','error_type':type(exc).__name__,
                                     'new_runs':new_runs,'status_update_errors':status_errors})
        raise
    write_json(out/'local_provenance.json',provenance)
    write_json(out/'checkpoint_inventory.json',checkpoints)
    write_json(out/'current_evaluation_assets.json',assets)
    write_json(out/'records.json',[{k:v for k,v in r.items() if k not in ('artifact_bytes','points')} for r in records])
    fields=list(dict.fromkeys(k for row in exports for k in row))
    with (out/'runs_export.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(exports)
    receipt={'status':'PASS','mode':'historical_backfill_no_training_no_inference','mlflow_version':'3.17.2',
             'code_sha256':code_sha,'source_files_unchanged':len(snapshots)-2,'runs':len(exports),
             'new_runs':sum(not r['reused'] for r in exports),'reused_runs':sum(r['reused'] for r in exports),
             'all_metric_histories_verified':True,'all_run_artifacts_verified':True,'checkpoint_files':len(checkpoints),
             'unique_checkpoint_hashes':len({c['sha256'] for c in checkpoints}),
             'evaluation_checkpoint_links':'UNRESOLVED','model_registry_used':False}
    write_json(out/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--store-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    records,checkpoints,provenance,snapshots,assets=prepare(json.loads(args.manifest.read_text(encoding='utf-8-sig')))
    receipt=ingest(records,args.store_dir.resolve(),args.output_dir.resolve(),provenance,checkpoints,snapshots,assets)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
