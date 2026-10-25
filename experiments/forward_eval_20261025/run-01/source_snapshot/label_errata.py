"""Create the owner-approved GT v2 and a local errata, without touching v1 or running models."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import re

import evaluation as ev

TARGET_LABEL = '프로스카정.txt'
OLD_CLASS, NEW_CLASS = 47, 16
NEW_NAME = '프로스카정'
OLD_LABEL_SHA = 'a9c51c609dca9b71d4b67be6a7893047be7cd7c412c6f2f1bf579d53dbe9c4d5'
OLD_GT_SHA = '20d8da2c0019248e3e1c8c407135975278db2da46f4e3885aa34e690338c785d'
THRESHOLDS = (0.05, 0.25)
VERSION = 'healtheat-label-v2-errata-1'


def corrected_bytes(raw):
    if ev.bf.digest(raw) != OLD_LABEL_SHA:
        raise ValueError('UNREVIEWED_SOURCE_LABEL')
    fixed, count = re.subn(rb'(?m)^47(?=[ \t])', b'16', raw)
    if count != 2 or len(raw.splitlines()) != 2:
        raise ValueError('EXPECTED_TWO_CLASS_TOKENS')
    # Byte replacement preserves coordinates, spacing and line endings.
    return fixed


def reference_files(root):
    files = sorted((root/'eval').glob('*_results.json'))
    if len(files) != 4:
        raise ValueError('EXPECTED_FOUR_HISTORICAL_JSONS')
    return files


def confirm_mapping(root):
    path = root/'eval/reference/class_mapping.json'
    evidence = ev.read_json(path)
    columns = ['dl_idx','final_class_idx_merged','dl_name']
    rows = evidence.get('rows', [])
    if (evidence.get('schema_version') != 1 or evidence.get('columns') != columns
            or evidence.get('source_required_at_runtime') is not False
            or not re.fullmatch(r'[0-9a-f]{64}', evidence.get('source_sha256',''))
            or not rows or evidence.get('row_count') != len(rows)):
        raise ValueError('INVALID_MAPPING_EVIDENCE')
    if ev.bf.digest(ev.bf.canonical(rows)) != evidence.get('rows_sha256'):
        raise ValueError('MAPPING_ROWS_HASH_MISMATCH')
    for row in rows:
        if (set(row) != set(columns) or not isinstance(row['dl_idx'],str)
                or (row['dl_idx'] and not row['dl_idx'].isdigit())
                or type(row['final_class_idx_merged']) is not int or row['final_class_idx_merged'] < 0
                or not isinstance(row['dl_name'],str) or not row['dl_name']):
            raise ValueError('INVALID_MAPPING_ROW')
    ids = [r['dl_idx'] for r in rows if r['dl_idx']]
    if (len(set(ids)) != len(ids) or len({r['final_class_idx_merged'] for r in rows}) != len(rows)
            or evidence.get('empty_source_dl_idx_preserved') != len(rows)-len(ids)):
        raise ValueError('AMBIGUOUS_MAPPING_ROWS')
    match = [r for r in rows if r['dl_idx'] == '5390']
    if len(match) != 1 or int(match[0]['final_class_idx_merged']) != NEW_CLASS or ev.nfc(match[0]['dl_name']) != NEW_NAME:
        raise ValueError('CORRECTION_MAPPING_MISMATCH')
    return path


def mapping_evidence(root, path):
    return {'file':path.relative_to(root).as_posix(),'sha256':ev.bf.file_digest(path),
            'source_sha256':ev.read_json(path)['source_sha256'],'original_source_required':False}


def gt_only_changes_names(before, after):
    a = {ev.nfc(r['filename']): r for r in before['results']}
    b = {ev.nfc(r['filename']): r for r in after['results']}
    if a.keys() != b.keys():
        raise ValueError('GT_IMAGE_SET_CHANGED')
    changes = []
    for name, old in a.items():
        new = b[name]
        if len(old['gt_bboxes']) != len(new['gt_bboxes']):
            raise ValueError('GT_BOX_COUNT_CHANGED')
        for i, (x, y) in enumerate(zip(old['gt_bboxes'], new['gt_bboxes'])):
            if x['bbox'] != y['bbox']:
                raise ValueError('GT_COORDINATES_CHANGED')
            if x['dl_name'] != y['dl_name']:
                if name != Path(TARGET_LABEL).with_suffix('.png').name or y['dl_name'] != NEW_NAME:
                    raise ValueError('UNAPPROVED_GT_CHANGE')
                changes.append({'filename': name, 'box_index': i, 'old_name': x['dl_name'], 'new_name': y['dl_name']})
    if len(changes) != 2:
        raise ValueError('EXPECTED_TWO_GT_NAME_CHANGES')
    return changes


def create_v2(root):
    root = Path(root).resolve(strict=True)
    output = root/'eval/label_versions/v2'
    if output.exists():
        raise FileExistsError('LABEL_V2_ALREADY_EXISTS')
    mapping = confirm_mapping(root)
    v1, inputs, _ = ev.reconstruct_gt(root)
    if ev.bf.gt_digest(v1) != OLD_GT_SHA:
        raise ValueError('UNREVIEWED_GT_V1')
    refs = reference_files(root)
    inventory = ev.input_inventory(root, [*inputs, mapping, *refs])
    for p in refs:
        data = ev.read_json(p)
        ev.validate(data)
        if ev.bf.gt_digest(data) != OLD_GT_SHA:
            raise ValueError('HISTORICAL_GT_NOT_V1')
    labels = ev.named_files(root/'eval/test_labels', '.txt')
    if {p.name for p in labels.values()} != set(labels):
        raise ValueError('LABEL_FILENAME_NORMALIZATION_REQUIRES_REVIEW')
    originals = {name:p.read_bytes() for name,p in labels.items()}
    revised = {**originals, TARGET_LABEL:corrected_bytes(originals[TARGET_LABEL])}
    ev.check_sources(root, inventory)
    output.mkdir(parents=True, exist_ok=False)
    try:
        destination = output/'labels'
        destination.mkdir()
        for name, raw in revised.items():
            (destination/name).write_bytes(raw)
        v2, _, audit = ev.reconstruct_gt(root, labels_directory=destination)
        if audit['conflicts']:
            raise ValueError('V2_REFERENCE_CONFLICT')
        changes = gt_only_changes_names(v1,v2)
        ev.bf.write_json(output/'ground_truth.json',v2)
        manifest = {
            'recipe':VERSION, 'gt_version':'v2', 'base_gt_version':'v1',
            'mapping_evidence':mapping_evidence(root,mapping),
            'v1_gt_sha256':OLD_GT_SHA, 'v2_gt_sha256':ev.bf.gt_digest(v2),
            'images':v2['num_images'], 'gt_boxes':v2['num_gt_bboxes'],
            'correction':{'label':TARGET_LABEL,'old_class':OLD_CLASS,'new_class':NEW_CLASS,
                          'old_label_sha256':ev.bf.digest(originals[TARGET_LABEL]),
                          'new_label_sha256':ev.bf.digest(revised[TARGET_LABEL]),'coordinate_bytes_preserved':True},
            'name_changes':changes,
            'labels':{n:ev.bf.digest(b) for n,b in revised.items()},
            'source_files':inventory,
            'approval':{'date_kst':'2026-10-24','scope':'Separate label v2 and errata only; originals and public metrics unchanged',
                        'evidence':'Owner visual confirmation relayed in thread #504; direct approval in Codex chat after #505'},
            'auxiliary_index_warnings_preserved':audit['warnings'],
            'model_executed':False, 'creator_code_sha256':ev.bf.file_digest(Path(__file__)),
            'ground_truth_file_sha256':ev.bf.file_digest(output/'ground_truth.json')}
        ev.bf.write_json(output/'manifest.json',manifest)
        ev.check_sources(root,inventory)
        verify_v2(root)
    except Exception as exc:
        ev.bf.write_json(output/'failure.json',{'status':'FAILED','error_type':type(exc).__name__})
        raise
    return manifest


def verify_v2(root):
    root = Path(root).resolve(strict=True)
    directory = root/'eval/label_versions/v2'
    if (directory/'failure.json').exists():
        raise ValueError('FAILED_V2_PRESERVED')
    manifest = ev.read_json(directory/'manifest.json')
    if manifest.get('recipe') != VERSION or manifest.get('gt_version') != 'v2' or manifest.get('model_executed') is not False:
        raise ValueError('INVALID_V2_MANIFEST')
    mapping = confirm_mapping(root)
    if manifest.get('mapping_evidence') != mapping_evidence(root,mapping):
        raise ValueError('MAPPING_PROVENANCE_MISMATCH')
    mapping_inputs = [r for r in manifest['source_files'] if r['file']==mapping.relative_to(root).as_posix()]
    if len(mapping_inputs)!=1 or mapping_inputs[0]['sha256']!=ev.bf.file_digest(mapping):
        raise ValueError('MAPPING_INPUT_MISSING_OR_CHANGED')
    if any(r['file']=='data/final_mapping_after_category_merge.csv' for r in manifest['source_files']):
        raise ValueError('LEGACY_PRIVATE_MAPPING_DEPENDENCY')
    ev.check_sources(root,manifest['source_files'])
    current_labels = ev.named_files(directory/'labels','.txt')
    original_labels = ev.named_files(root/'eval/test_labels','.txt')
    if set(current_labels) != set(original_labels) or set(current_labels) != set(manifest['labels']):
        raise ValueError('V2_LABEL_SET_CHANGED')
    for name,p in current_labels.items():
        raw=original_labels[name].read_bytes()
        expected=corrected_bytes(raw) if name==TARGET_LABEL else raw
        if p.read_bytes()!=expected or ev.bf.file_digest(p)!=manifest['labels'][name]:
            raise ValueError('UNAPPROVED_V2_LABEL_BYTES')
    v1,_,_=ev.reconstruct_gt(root)
    v2,_,audit=ev.reconstruct_gt(root,labels_directory=directory/'labels')
    if (ev.bf.gt_digest(v1)!=OLD_GT_SHA or manifest['v1_gt_sha256']!=OLD_GT_SHA
            or ev.bf.gt_digest(v2)!=manifest['v2_gt_sha256'] or audit['conflicts']):
        raise ValueError('V2_GT_MISMATCH')
    if (ev.bf.file_digest(directory/'ground_truth.json')!=manifest['ground_truth_file_sha256']
            or ev.read_json(directory/'ground_truth.json')!=v2
            or gt_only_changes_names(v1,v2)!=manifest['name_changes']):
        raise ValueError('V2_GROUND_TRUTH_ARTIFACT_CHANGED')
    return manifest, v1, v2


def posthoc_filter(data, threshold):
    """The old notebook's stored Top-1 score filter; not fresh inference postprocessing."""
    ev.validate(data)
    if threshold not in THRESHOLDS or data['conf_threshold'] != 0.05:
        raise ValueError('UNSUPPORTED_STORED_PREDICTION_THRESHOLD')
    out = copy.deepcopy(data)
    for row in out['results']:
        if any(not p['top3_confidence'] for p in row['predictions']):
            raise ValueError('MISSING_TOP1_CONFIDENCE')
        row['predictions'] = [p for p in row['predictions'] if p['top3_confidence'][0] >= threshold]
    out['source_conf_threshold'] = data['conf_threshold']
    out['conf_threshold'] = threshold
    out['filter_semantics'] = 'stored top3_confidence[0] >= threshold; groups and scores unchanged'
    return out


def rescore_gt(data, gt, version, threshold, source_sha):
    if ev.bf.gt_digest(data) != OLD_GT_SHA or version not in ('v1','v2'):
        raise ValueError('EXPLICIT_V1_SOURCE_REQUIRED')
    if version == 'v1' and ev.bf.gt_digest(gt) != OLD_GT_SHA:
        raise ValueError('V1_LABEL_CANNOT_HIDE_GT_SUBSTITUTION')
    if version == 'v2':
        gt_only_changes_names(data,gt)
    out = posthoc_filter(data,threshold)
    target = {ev.nfc(r['filename']):r['gt_bboxes'] for r in gt['results']}
    if {ev.nfc(r['filename']) for r in out['results']} != set(target):
        raise ValueError('GT_SUBSTITUTION_IMAGE_SET_MISMATCH')
    for row in out['results']:
        row['gt_bboxes'] = copy.deepcopy(target[ev.nfc(row['filename'])])
    out.update({'gt_version':version,'historical_json_gt_version':'v1',
                'gt_substitution_explicit':version=='v2','historical_source_sha256':source_sha,
                'model_executed':False,'checkpoint_evaluation_link':'UNRESOLVED',
                'rescore_semantics':'same stored predictions; label-version comparison, not model improvement'})
    ev.validate(out)
    return out, ev.bf.score_evaluation(out)


def write_errata(root, output):
    root=Path(root).resolve(strict=True)
    output=ev.unused_directory(output,[root])
    manifest,v1,v2=verify_v2(root)
    version_dir=root/'eval/label_versions/v2'
    version_sha=ev.bf.file_digest(version_dir/'manifest.json')
    refs=reference_files(root)
    snapshots={p:ev.bf.file_digest(p) for p in refs}
    code_files=[Path(__file__),Path(ev.__file__),Path(ev.bf.__file__)]
    code_bytes={p.name:p.read_bytes() for p in code_files}
    rows,derived={},{}
    output.mkdir(parents=True,exist_ok=False)
    try:
        for path in refs:
            source=ev.read_json(path)
            for threshold in THRESHOLDS:
                pair=[]
                for version,gt in [('v1',v1),('v2',v2)]:
                    data,scores=rescore_gt(source,gt,version,threshold,snapshots[path])
                    filename=f'{path.stem}__gt-{version}__posthoc-{str(threshold).replace(".","p")}.json'
                    target=output/'derived'/filename
                    target.parent.mkdir(exist_ok=True)
                    ev.bf.write_json(target,data)
                    derived[filename]=ev.bf.file_digest(target)
                    pair.append({'model_record':path.stem,'historical_source_sha256':snapshots[path],
                                 'gt_version':version,'gt_sha256':ev.bf.gt_digest(data),'posthoc_threshold':threshold,
                                 'derived_file':'derived/'+filename,**scores})
                pred_digests=[ev.bf.digest(ev.bf.canonical([r['predictions'] for r in ev.read_json(output/r['derived_file'])['results']])) for r in pair]
                if len(set(pred_digests))!=1:
                    raise ValueError('PREDICTIONS_CHANGED_BETWEEN_GT_VERSIONS')
                rows[(path.name,threshold)]=pair
        flat=[r for pair in rows.values() for r in pair]
        report={'recipe':VERSION,'status':'PASS_LOCAL_PENDING_CROSS_REVIEW','model_executed':False,
                'public_metrics_changed':False,'source_json_gt_version':'v1','v1_gt_sha256':OLD_GT_SHA,
                'v2_gt_sha256':manifest['v2_gt_sha256'],'label_manifest_sha256':version_sha,
                'semantics':'label correction only; predictions not rerun; 0.25 is historical stored-score filtering',
                'rows':flat,'derived_files':derived,'code_sha256':{n:ev.bf.digest(b) for n,b in code_bytes.items()}}
        ev.bf.write_json(output/'errata.json',report)
        with (output/'errata.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        lines=['# HealthEat 정답 라벨 v2 재채점 정오표 (로컬 검토용)','',
               '모델이나 예측을 바꾸지 않고, 승인된 라벨의 클래스 ID 두 곳만 47→16으로 정정했습니다.',
               '점수 변화는 모델 성능 향상이 아니라 정답 정정의 영향입니다. 원본 v1·과거 JSON·공개 수치는 그대로입니다.',
               '0.25는 과거 노트북의 저장된 Top-1 점수 필터를 적용한 결과이며 새 모델 출력의 재후처리가 아닙니다.','',
               '| 모델 기록 | 저장 점수 필터 | Top-1 v1 | Top-1 v2 | Top-3 v1 | Top-3 v2 | 매칭 v1→v2 | 미매칭 예측 v1→v2 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for (_,threshold),(a,b) in rows.items():
            lines.append(f"| {a['model_record']} | {threshold} | {a['n_top1']}/{a['total_gt']} ({a['top1_accuracy']:.1%}) | {b['n_top1']}/{b['total_gt']} ({b['top1_accuracy']:.1%}) | {a['n_top3']}/{a['total_gt']} ({a['top3_accuracy']:.1%}) | {b['n_top3']}/{b['total_gt']} ({b['top3_accuracy']:.1%}) | {a['n_match']}→{b['n_match']} | {a['n_false_pos']}→{b['n_false_pos']} |")
        lines += ['',f'GT v1 SHA-256: `{OLD_GT_SHA}`',f"GT v2 SHA-256: `{manifest['v2_gt_sha256']}`",'',
                  '라벨 버전 v2는 모델 실험 v2와 다른 의미입니다. 원본 JSON의 GT는 v1이며 파생 파일에만 GT 치환을 명시했습니다.',
                  '과거 평가와 특정 체크포인트의 연결은 여전히 미확정입니다. 임계값 최적화·새 데이터 일반화·의료 사용 가능성을 검증한 결과가 아닙니다.',
                  '클로드 교차검토와 별도 사용자 승인 전 이력서·포트폴리오·공개 보고서에는 반영하지 않습니다.']
        (output/'정오표.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (output/'code_snapshot').mkdir()
        for name,raw in code_bytes.items():(output/'code_snapshot'/name).write_bytes(raw)
        verify_v2(root)
        if version_sha!=ev.bf.file_digest(version_dir/'manifest.json') or any(ev.bf.file_digest(p)!=h for p,h in snapshots.items()):
            raise ValueError('INPUT_CHANGED_DURING_RESCORE')
        if any(p.read_bytes()!=code_bytes[p.name] for p in code_files):
            raise ValueError('SCORER_CODE_CHANGED_DURING_RESCORE')
        return report
    except Exception as exc:
        ev.bf.write_json(output/'failure.json',{'status':'FAILED','error_type':type(exc).__name__})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['create-v2','verify-v2','rescore-gt-v2'])
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='create-v2':
        m=create_v2(a.project_root);result={'gt_version':m['gt_version'],'v1_gt_sha256':m['v1_gt_sha256'],'v2_gt_sha256':m['v2_gt_sha256']}
    elif a.command=='verify-v2':
        m,_,_=verify_v2(a.project_root);result={'status':'PASS','v2_gt_sha256':m['v2_gt_sha256']}
    else:
        if a.output is None:p.error('rescore-gt-v2 requires --output')
        r=write_errata(a.project_root,a.output);result={'status':r['status'],'rows':len(r['rows']),'model_executed':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
