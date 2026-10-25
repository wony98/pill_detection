"""A-stage HealthEat checks: stored predictions only; no model imports or inference."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mlflow'))
import backfill_runs as bf

VERSION = 'healtheat-checkpoint-eval-a1'
KINDS = {'historical', 'synthetic'}
CORE_FILES = ('predictions.json', 'ground_truth.json', 'metrics.json', 'environment.json',
              'evaluation.py.txt', 'backfill_runs.py.txt')


def nfc(value):
    return unicodedata.normalize('NFC', value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate(data):
    """Keep the existing scoring contract and reject ambiguous names/conditions."""
    for key in ('num_images', 'num_gt_bboxes'):
        if type(data.get(key)) is not int or data[key] <= 0:
            raise ValueError('INVALID_COUNTS')
    bf.validate_evaluation(data)
    for row in data['results']:
        name = row['filename']
        if not isinstance(name, str) or not name or name in {'.', '..'} or any(c in name for c in '/\\:\x00'):
            raise ValueError('INVALID_IMAGE_NAME')
    for key in ('conf_threshold', 'nms_iou'):
        value = data.get(key)
        if not bf.finite(value) or not 0 <= value <= 1:
            raise ValueError('MISSING_OR_INVALID_POSTPROCESS_CONDITION')


def named_files(directory, suffix):
    result = {}
    for path in sorted(Path(directory).glob('*' + suffix)):
        key = nfc(path.name)
        if key in result:
            raise ValueError('DUPLICATE_NORMALIZED_FILENAME')
        result[key] = path
    if not result:
        raise ValueError('EMPTY_INPUT_DIRECTORY')
    return result


def reconstruct_gt(root, *, labels_directory=None):
    """Rebuild xyxy GT in label order using the notebook's arithmetic, not its outputs."""
    from PIL import Image
    import yaml

    root = Path(root).resolve(strict=True)
    names_path = root / 'demo/v4/pill.yaml'
    answer_path = root / 'eval/answer_key.json'
    names = yaml.safe_load(names_path.read_text(encoding='utf-8'))['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if set(names) != set(range(len(names))) or not all(isinstance(v, str) and v for v in names.values()):
        raise ValueError('INVALID_CLASS_MAPPING')
    answer = read_json(answer_path)
    answer_normalized = {nfc(k): v for k, v in answer.items()}
    if len(answer_normalized) != len(answer):
        raise ValueError('DUPLICATE_ANSWER_KEY')
    images = named_files(root / 'eval/test_images', '.png')
    labels_path = Path(labels_directory).resolve(strict=True) if labels_directory is not None else root/'eval/test_labels'
    if not labels_path.is_relative_to(root):
        raise ValueError('LABEL_DIRECTORY_OUTSIDE_PROJECT')
    labels = named_files(labels_path, '.txt')
    if {Path(n).stem for n in images} != {Path(n).stem for n in labels} or set(answer_normalized) != set(images):
        raise ValueError('IMAGE_LABEL_ANSWER_SET_MISMATCH')
    source_paths = [names_path, answer_path]
    result, warnings, conflicts = [], [], []
    for name, path in images.items():
        label = labels[Path(name).stem + '.txt']
        source_paths.extend((path, label))
        with Image.open(path) as image:
            width, height = image.size
        boxes = []
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5 or not parts[0].isdigit() or int(parts[0]) not in names:
                raise ValueError('INVALID_YOLO_LABEL')
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0):
                raise ValueError('INVALID_YOLO_COORDINATES')
            boxes.append({'bbox': [(cx-w/2)*width, (cy-h/2)*height,
                                   (cx+w/2)*width, (cy+h/2)*height], 'dl_name': nfc(names[cls])})
        # The historical notebook uses YOLO IDs + pill.yaml, not answer_key.class_idx.
        # Keep the auxiliary table untouched and expose differences instead of remapping GT.
        key = answer_normalized[name]
        if key.get('class_idx') not in names or nfc(key['dl_name']) != nfc(names[key['class_idx']]):
            warnings.append({'filename': name, 'code': 'AUXILIARY_INDEX_NAMESPACE_DIFFERS',
                             'answer_class_idx': key.get('class_idx')})
        if not boxes or any(box['dl_name'] != nfc(key['dl_name']) for box in boxes):
            conflicts.append({'filename': name, 'code': 'ANSWER_LABEL_CLASS_MISMATCH',
                              'answer_name': key['dl_name'], 'label_names': [b['dl_name'] for b in boxes]})
        result.append({'filename': name, 'gt_bboxes': boxes, 'predictions': []})
    return {'num_images': len(result), 'num_gt_bboxes': sum(len(r['gt_bboxes']) for r in result),
            'ms_per_image': 0.0, 'results': result}, source_paths, {'warnings': warnings, 'conflicts': conflicts}


def input_inventory(root, paths):
    root = Path(root).resolve(strict=True)
    entries = []
    for path in sorted(set(Path(p).resolve(strict=True) for p in paths)):
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('INPUT_OUTSIDE_PROJECT')
        entries.append({'file': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                        'sha256': bf.file_digest(path)})
    return entries


def check_sources(root, inventory):
    root = Path(root).resolve(strict=True)
    for entry in inventory:
        path = (root / entry['file']).resolve(strict=True)
        if not path.is_relative_to(root) or bf.file_digest(path) != entry['sha256']:
            raise ValueError('SOURCE_HASH_MISMATCH')
    for group, suffix in [('eval/test_images', '.png'), ('eval/test_labels', '.txt')]:
        expected = {e['file'] for e in inventory if e['file'].startswith(group+'/')}
        observed = {p.relative_to(root).as_posix() for p in (root/group).glob('*'+suffix)}
        if not expected or expected != observed:
            raise ValueError('SOURCE_FILE_SET_CHANGED')


def audit_gt(root, output):
    root = Path(root).resolve(strict=True)
    output = unused_directory(output, [root])
    reference_paths = sorted((root/'eval').glob('*_results.json'))
    inventory = input_inventory(root, [root/'demo/v4/pill.yaml', root/'eval/answer_key.json',
                                      *named_files(root/'eval/test_images', '.png').values(),
                                      *named_files(root/'eval/test_labels', '.txt').values(), *reference_paths])
    gt, paths, audit = reconstruct_gt(root)
    references = []
    for p in reference_paths:
        d = read_json(p)
        validate(d)
        references.append({'file': p.name, 'sha256': bf.file_digest(p),
                           'gt_sha256': bf.gt_digest(d), 'matches_reconstructed_gt': bf.gt_digest(d) == bf.gt_digest(gt),
                           'legacy_scores_not_revalidated_labels': bf.score_evaluation(d)})
    if not references:
        raise ValueError('NO_HISTORICAL_REFERENCES')
    report = {'status': 'NEEDS_LABEL_REVIEW' if audit['conflicts'] else
              ('PASS' if all(r['matches_reconstructed_gt'] for r in references) else 'GT_MISMATCH'),
              'model_executed': False, 'gt_source': 'YOLO labels + pill.yaml, matching historical notebook arithmetic',
              'gt_sha256': bf.gt_digest(gt), 'images': gt['num_images'], 'boxes': gt['num_gt_bboxes'],
              'reference_results': references, **audit, 'source_files': inventory,
              'warning': 'Matching historical GT proves reconstruction, not semantic correctness of the labels'}
    check_sources(root, inventory)
    if sorted((root/'eval').glob('*_results.json')) != reference_paths:
        raise ValueError('REFERENCE_FILE_SET_CHANGED')
    output.mkdir(parents=True, exist_ok=False)
    bf.write_json(output/'gt_audit.json', report)
    bf.write_json(output/'reconstructed_gt.json', gt)
    return report


def unused_directory(path, forbidden):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    if any(path.is_relative_to(Path(p).resolve()) for p in forbidden):
        raise ValueError('OUTPUT_INSIDE_PROTECTED_INPUT')
    return path


def prepare(root, prediction, kind, output):
    if kind not in KINDS:
        raise ValueError('A_STAGE_NO_FORWARD_RUN')
    root = Path(root).resolve(strict=True)
    prediction = Path(prediction).resolve(strict=True)
    output = unused_directory(output, [root])
    # Fingerprint before parsing so an input changed during preparation cannot pass.
    paths = [root/'demo/v4/pill.yaml', root/'eval/answer_key.json', prediction,
             root/'demo/v4/v4.pth', root/'demo/v4/rtmdet_tiny_pill.py',
             *named_files(root/'eval/test_images', '.png').values(),
             *named_files(root/'eval/test_labels', '.txt').values()]
    inventory = input_inventory(root, paths)
    prediction_bytes = prediction.read_bytes()
    data = json.loads(prediction_bytes.decode('utf-8-sig'))
    validate(data)
    ground_truth, _, gt_audit = reconstruct_gt(root)
    if gt_audit['conflicts']:
        raise ValueError('GT_REFERENCE_CONFLICT: inspect audit-gt output; no automatic relabeling')
    if bf.gt_digest(data) != bf.gt_digest(ground_truth):
        raise ValueError('GROUND_TRUTH_MISMATCH')
    metrics = bf.score_evaluation(data)
    environment = {'python': sys.version.split()[0], 'platform': sys.platform,
                   'capture': 'current_validation_environment_not_historical_inference',
                   'model_imported': False,
                   'packages': {n: importlib.metadata.version(n) for n in ('Pillow', 'PyYAML')}}
    blobs = {'predictions.json': prediction_bytes, 'ground_truth.json': bf.canonical(ground_truth),
             'metrics.json': bf.canonical(metrics), 'environment.json': bf.canonical(environment),
             'evaluation.py.txt': Path(__file__).read_bytes(), 'backfill_runs.py.txt': Path(bf.__file__).read_bytes()}
    manifest = {'schema_version': 1, 'recipe': VERSION, 'kind': kind, 'model_executed': False,
                'checkpoint_evaluation_link': 'UNRESOLVED' if kind == 'historical' else 'SYNTHETIC_ONLY',
                'checkpoint_and_config_hashes_are': 'present_file_context_not_proof_of_prediction_origin',
                'seed': data.get('seed'), 'postprocess': {k: data[k] for k in ('conf_threshold', 'nms_iou')},
                'gt_sha256': bf.gt_digest(data), 'image_count': data['num_images'], 'gt_count': data['num_gt_bboxes'],
                'prediction_source': prediction.relative_to(root).as_posix(), 'gt_audit': gt_audit,
                'source_files': inventory,
                'artifacts': {name: bf.digest(raw) for name, raw in blobs.items()}}
    check_sources(root, inventory)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw in blobs.items():
            (output/name).write_bytes(raw)
        bf.write_json(output/'manifest.json', manifest)
        # Never log this private file to MLflow or publish it.
        bf.write_json(output/'local_sources.json', {'root': str(root)})
        check_sources(root, inventory)
        load_bundle(output)
    except Exception as exc:
        bf.write_json(output/'failure.json', {'status': 'FAILED', 'error_type': type(exc).__name__})
        raise
    return manifest


def load_bundle(bundle):
    bundle = Path(bundle).resolve(strict=True)
    if (bundle/'failure.json').exists():
        raise ValueError('FAILED_BUNDLE_PRESERVED')
    manifest = read_json(bundle/'manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('recipe') != VERSION
            or manifest.get('kind') not in KINDS or manifest.get('model_executed') is not False):
        raise ValueError('INVALID_A_STAGE_MANIFEST')
    expected_link = 'UNRESOLVED' if manifest['kind'] == 'historical' else 'SYNTHETIC_ONLY'
    if manifest.get('checkpoint_evaluation_link') != expected_link or set(manifest['artifacts']) != set(CORE_FILES):
        raise ValueError('INVALID_ARTIFACT_CONTRACT')
    for name in CORE_FILES:
        if bf.file_digest(bundle/name) != manifest['artifacts'][name]:
            raise ValueError('BUNDLE_ARTIFACT_HASH_MISMATCH')
    root = Path(read_json(bundle/'local_sources.json')['root'])
    check_sources(root, manifest['source_files'])
    source_prediction = (root / manifest['prediction_source']).resolve(strict=True)
    if not source_prediction.is_relative_to(root.resolve()) or bf.file_digest(source_prediction) != manifest['artifacts']['predictions.json']:
        raise ValueError('PREDICTION_ORIGIN_MISMATCH')
    if (bf.file_digest(Path(__file__)) != manifest['artifacts']['evaluation.py.txt']
            or bf.file_digest(Path(bf.__file__)) != manifest['artifacts']['backfill_runs.py.txt']):
        raise ValueError('SCORING_CODE_VERSION_MISMATCH')
    data = read_json(bundle/'predictions.json')
    validate(data)
    if (bf.gt_digest(data) != manifest['gt_sha256']
            or bf.gt_digest(read_json(bundle/'ground_truth.json')) != manifest['gt_sha256']
            or bf.score_evaluation(data) != read_json(bundle/'metrics.json')
            or manifest['postprocess'] != {k: data[k] for k in ('conf_threshold', 'nms_iou')}
            or manifest['image_count'] != data['num_images'] or manifest['gt_count'] != data['num_gt_bboxes']):
        raise ValueError('BUNDLE_CONTENT_MISMATCH')
    for name in CORE_FILES:
        if bf.file_digest(bundle/name) != manifest['artifacts'][name]:
            raise ValueError('BUNDLE_CHANGED_DURING_READ')
    return manifest, data


def compare(reference, candidate):
    validate(reference)
    validate(candidate)
    refs = {nfc(r['filename']): r for r in reference['results']}
    news = {nfc(r['filename']): r for r in candidate['results']}
    shared = sorted(refs.keys() & news.keys())
    equal_gt = bf.gt_digest(reference) == bf.gt_digest(candidate)
    conditions = {name: {key: data[key] for key in ('conf_threshold', 'nms_iou')}
                  for name, data in [('reference', reference), ('candidate', candidate)]}
    equal_conditions = conditions['reference'] == conditions['candidate']
    rows = []
    for name in shared:
        old, new = refs[name]['predictions'], news[name]['predictions']
        used, pairs = set(), []
        for i, pred in enumerate(old):
            best, best_iou = None, 0.0
            for j, other in enumerate(new):
                overlap = bf.iou_xyxy(pred['bbox'], other['bbox'])
                if j not in used and overlap > best_iou:
                    best, best_iou = j, overlap
            if best is not None and best_iou >= 0.5:
                used.add(best)
                oc, nc = pred['top3_class_name'], new[best]['top3_class_name']
                oscores, nscores = pred['top3_confidence'], new[best]['top3_confidence']
                pairs.append({'reference_index': i, 'candidate_index': best, 'iou': best_iou,
                              'top1_agrees': bool(oc and nc) and nfc(oc[0]) == nfc(nc[0]),
                              'top1_confidence_delta': nscores[0]-oscores[0] if oscores and nscores else None})
        rows.append({'filename': name, 'reference_boxes': len(old), 'candidate_boxes': len(new),
                     'reference_only_boxes': len(old)-len(pairs), 'candidate_only_boxes': len(new)-len(pairs),
                     'matched': len(pairs), 'top1_agreements': sum(p['top1_agrees'] for p in pairs), 'pairs': pairs})
    scores = {name: bf.score_evaluation(data) for name, data in [('reference',reference), ('candidate',candidate)]}
    # Historical latency is a source-reported value, not a new benchmark.
    delta = {k: scores['candidate'][k]-v for k,v in scores['reference'].items() if k != 'reported_ms_per_image'} if equal_gt and equal_conditions else None
    return {'rule': 'reference order; highest IoU unused candidate >=0.5; ties by candidate order',
            'reference_is_not_ground_truth': True, 'same_gt': equal_gt, 'same_postprocess': equal_conditions,
            'postprocess': conditions, 'reference_only_images': sorted(refs.keys()-news.keys()),
            'candidate_only_images': sorted(news.keys()-refs.keys()), 'images': rows,
            'metrics': scores, 'metric_delta_candidate_minus_reference': delta,
            'meaning': 'Stored predictions compared; no new inference, checkpoint provenance proof or generalization claim'}


def compare_bundles(reference, candidate, output):
    roots = [read_json(Path(p)/'local_sources.json')['root'] for p in (reference,candidate)]
    output = unused_directory(output, [reference, candidate, *roots])
    rm, rd = load_bundle(reference)
    cm, cd = load_bundle(candidate)
    report = compare(rd, cd)
    report['input_manifest_sha256'] = {'reference': bf.file_digest(Path(reference)/'manifest.json'),
                                      'candidate': bf.file_digest(Path(candidate)/'manifest.json')}
    output.mkdir(parents=True, exist_ok=False)
    bf.write_json(output/'comparison.json', report)
    text = ['# 저장 예측 비교', '', '모델 실행 없이 저장된 예측을 비교했습니다. 과거 예측은 정답이 아닙니다.',
            f"동일 GT: {report['same_gt']} / 동일 후처리 메타: {report['same_postprocess']}",
            '데이터 또는 후처리 조건이 다르면 전체 지표 차이는 표시하지 않습니다.', '',
            '| 입력 | conf | Top-1 정답 | 정답 박스 | 미매칭 예측 박스 |', '|---|---:|---:|---:|---:|']
    for label in ('reference','candidate'):
        m = report['metrics'][label]
        text.append(f"| {label} | {report['postprocess'][label]['conf_threshold']} | {m['n_top1']} | {m['total_gt']} | {m['n_false_pos']} |")
    text += ['', f"과거에만 있는 이미지: {len(report['reference_only_images'])}; 새 파일에만 있는 이미지: {len(report['candidate_only_images'])}",
             f"공통 이미지에서 과거에만 있는 박스: {sum(r['reference_only_boxes'] for r in report['images'])}; 새 파일에만 있는 박스: {sum(r['candidate_only_boxes'] for r in report['images'])}",
             '', '박스 대응과 클래스·점수 차이를 관찰합니다. 과거 사용 체크포인트를 확정하거나 새 데이터 성능을 증명하지 않습니다.']
    (output/'comparison.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


def record_synthetic(bundle, workspace):
    """One new isolated SQLite store per validation. Never open the historical store."""
    workspace = unused_directory(workspace, [bundle])
    manifest, data = load_bundle(bundle)
    if manifest['kind'] != 'synthetic':
        raise ValueError('TRACKING_ACCEPTS_SYNTHETIC_ONLY_IN_STAGE_A')
    project = Path(read_json(Path(bundle)/'local_sources.json')['root'])
    if workspace.is_relative_to(project.resolve()):
        raise ValueError('TRACKING_INSIDE_PROTECTED_INPUT')
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
    if importlib.metadata.version('mlflow') != '3.17.2':
        raise ValueError('MLFLOW_VERSION_MISMATCH')
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    workspace.mkdir(parents=True, exist_ok=False)
    store = workspace/'tracking'
    store.mkdir()
    bundle = Path(bundle).resolve()
    blobs = {name: (bundle/name).read_bytes() for name in CORE_FILES}
    blobs['manifest.json'] = (bundle/'manifest.json').read_bytes()
    manifest_sha = bf.digest(blobs['manifest.json'])
    scores = bf.score_evaluation(data)
    record = {'params': {'bundle_sha256': manifest_sha, 'gt_sha256': manifest['gt_sha256'], **manifest['postprocess']},
              'points': [(k, float(v), 0) for k,v in scores.items()]}
    tags = {'mlflow.runName': 'A-stage synthetic validation', 'record_kind': 'synthetic_validation',
            'model_executed': 'false', 'checkpoint_evaluation_link': 'SYNTHETIC_ONLY', 'recipe': VERSION}
    uri = 'sqlite:///' + (store/'tracking.db').as_posix()
    run_id = None
    try:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        exp = client.create_experiment('HealthEat A-stage synthetic validation', artifact_location=(store/'artifacts').as_uri())
        run_id = client.create_run(exp, tags=tags).info.run_id
        client.log_batch(run_id, params=[Param(k,str(v)) for k,v in record['params'].items()],
                         metrics=[Metric(k,v,int(time.time()*1000),step) for k,v,step in record['points']])
        upload = workspace/'recorded_files'
        upload.mkdir()
        for name, raw in blobs.items():
            (upload/name).write_bytes(raw)
            client.log_artifact(run_id, str(upload/name))
        bf.verify_run(client, run_id, record, tags, blobs, store, 'RUNNING')
        load_bundle(bundle)
        if bf.file_digest(bundle/'manifest.json') != manifest_sha:
            raise ValueError('MANIFEST_CHANGED_DURING_RECORD')
        client.set_terminated(run_id, 'FINISHED')
        bf.verify_run(client, run_id, record, tags, blobs, store, 'FINISHED')
        receipt = {'status': 'PASS', 'synthetic_only': True, 'model_executed': False, 'run_id': run_id,
                   'record': record, 'tags': tags, 'artifacts': {n:bf.digest(b) for n,b in blobs.items()}}
        bf.write_json(workspace/'receipt.json', receipt)
        return receipt
    except Exception as exc:
        status_error = None
        if run_id:
            try:
                client.set_terminated(run_id, 'FAILED')
            except Exception as status_exc:
                status_error = type(status_exc).__name__
        bf.write_json(workspace/'failure.json', {'status': 'FAILED', 'error_type': type(exc).__name__,
                                              'run_id': run_id, 'status_update_error': status_error})
        raise


def verify_record(workspace):
    workspace = Path(workspace).resolve(strict=True)
    if (workspace/'failure.json').exists():
        raise ValueError('FAILED_WORKSPACE_PRESERVED')
    receipt = read_json(workspace/'receipt.json')
    names = {*CORE_FILES, 'manifest.json'}
    if set(receipt['artifacts']) != names or receipt.get('synthetic_only') is not True or receipt.get('model_executed') is not False:
        raise ValueError('INVALID_RECORD_RECEIPT')
    blobs = {n:(workspace/'recorded_files'/n).read_bytes() for n in names}
    if any(bf.digest(raw) != receipt['artifacts'][n] for n,raw in blobs.items()):
        raise ValueError('RECORDED_FILE_HASH_MISMATCH')
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
    from mlflow import MlflowClient
    store = workspace/'tracking'
    if not (store/'tracking.db').is_file():
        raise ValueError('TRACKING_DB_MISSING')
    uri = 'sqlite:///' + (store/'tracking.db').as_posix()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    bf.verify_run(client, receipt['run_id'], receipt['record'], receipt['tags'], blobs, store, 'FINISHED')
    return {'status':'PASS', 'synthetic_only':True, 'run_id':receipt['run_id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('audit-gt')
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--kind', choices=sorted(KINDS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('compare')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('record-synthetic')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p = sub.add_parser('verify-record')
    p.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit-gt':
        report = audit_gt(args.project_root,args.output)
        result = {k:report[k] for k in ('status','gt_sha256','images','boxes')}
        result.update({'warnings':len(report['warnings']), 'conflicts':len(report['conflicts'])})
    elif args.command == 'prepare':
        result = prepare(args.project_root,args.predictions,args.kind,args.output)
        result = {k:result[k] for k in ('kind','model_executed','gt_sha256','image_count','gt_count')}
    elif args.command == 'compare':
        result = compare_bundles(args.reference,args.candidate,args.output)
        result = {k:result[k] for k in ('same_gt','same_postprocess','metric_delta_candidate_minus_reference')}
    elif args.command == 'record-synthetic':
        receipt = record_synthetic(args.bundle,args.workspace)
        result = {k:receipt[k] for k in ('status','synthetic_only','model_executed','run_id')}
    else:
        result = verify_record(args.workspace)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'audit-gt' and report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
