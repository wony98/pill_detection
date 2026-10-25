"""B-stage preparation, GT-v2 scoring and isolated MLflow recording; no training."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import evaluation as ev
import label_errata as le
from inference_runner import validate_raw

bf = ev.bf
NOTEBOOK = 'training/notebooks/rtmdet_tiny_synth_train_v4.ipynb'


def postprocess_source(notebook):
    cells = ev.read_json(notebook)['cells']
    candidates = [''.join(c['source']) for c in cells if 'def extract_top3(' in ''.join(c['source'])]
    if len(candidates) != 1:
        raise ValueError('POSTPROCESS_SOURCE_AMBIGUOUS')
    tree = ast.parse(candidates[0])
    funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'iou_xyxy','extract_top3'}]
    if [n.name for n in funcs] != ['iou_xyxy','extract_top3']:
        raise ValueError('POSTPROCESS_FUNCTIONS_MISSING')
    return 'import numpy as np\n\n'+'\n\n'.join(ast.get_source_segment(candidates[0],n) for n in funcs)+'\n'


def module(path):
    spec=importlib.util.spec_from_file_location('frozen_top3',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def protected_files(root):
    paths=set()
    for directory in ['demo/v4','eval/test_images','eval/test_labels','eval/label_versions/v2',
                      'eval/reference','.mlflow-backfill']:
        paths.update(p for p in (root/directory).rglob('*') if p.is_file())
    paths.update((root/'eval').glob('*_results.json'))
    paths.update([root/'eval/answer_key.json',root/NOTEBOOK,
                  Path(ev.__file__),Path(le.__file__),Path(bf.__file__)])
    return {p.relative_to(root).as_posix():bf.file_digest(p) for p in sorted(paths)}


def prepare(root, workspace):
    root=Path(root).resolve(strict=True)
    workspace=ev.unused_directory(workspace,[root])
    label_manifest,_,gt=le.verify_v2(root)
    before=protected_files(root)
    images=ev.named_files(root/'eval/test_images','.png')
    if len(images)!=33 or gt['num_gt_bboxes']!=47:
        raise ValueError('UNEXPECTED_APPROVED_DATASET')
    workspace.mkdir(parents=True)
    try:
        inputs=workspace/'inputs';inputs.mkdir();(inputs/'images').mkdir()
        for name,path in images.items():shutil.copyfile(path,inputs/'images'/name)
        for source,target in [('demo/v4/v4.pth','model.pth'),('demo/v4/rtmdet_tiny_pill.py','config.py'),
                              ('demo/v4/pill.yaml','pill.yaml'),
                              ('experiments/checkpoint_eval/inference_runner.py','inference_runner.py')]:
            shutil.copyfile(root/source,inputs/target)
        (inputs/'postprocess.py').write_text(postprocess_source(root/NOTEBOOK),encoding='utf-8')
        input_manifest={'schema':'healtheat-forward-input-v1','images':sorted(images),
                        'files':{p.relative_to(inputs).as_posix():bf.file_digest(p)
                                 for p in sorted(inputs.rglob('*')) if p.is_file()}}
        bf.write_json(inputs/'inputs.json',input_manifest)
        bf.write_json(workspace/'ground_truth_v2.json',gt)
        shutil.copyfile(root/'eval/label_versions/v2/manifest.json',workspace/'label_v2_manifest.json')
        bf.write_json(workspace/'local_sources.json',{'project_root':str(root),'protected':before})
        manifest={'schema':'healtheat-forward-v1','gt_version':'v2','gt_sha256':bf.gt_digest(gt),
                  'ground_truth_file_sha256':bf.file_digest(workspace/'ground_truth_v2.json'),
                  'label_manifest_sha256':bf.file_digest(workspace/'label_v2_manifest.json'),
                  'input_manifest_sha256':bf.file_digest(inputs/'inputs.json'),
                  'postprocess_source':NOTEBOOK,'postprocess_source_sha256':before[NOTEBOOK],
                  'forward_code_sha256':bf.file_digest(Path(__file__)),
                  'thresholds':[0.05,0.25],'grouping_iou':0.5,'topk':3,
                  'historical_checkpoint_lineage':'UNRESOLVED','reference_mode':'same reused image set; GT-v2 rescoring'}
        bf.write_json(workspace/'manifest.json',manifest)
        with zipfile.ZipFile(workspace/'inputs.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(inputs.rglob('*')):
                if p.is_file():archive.write(p,p.relative_to(inputs).as_posix())
        verify_inputs(workspace)
        return manifest
    except BaseException as exc:
        bf.write_json(workspace/'failure.json',{'status':'FAILED','error_type':type(exc).__name__})
        raise


def verify_inputs(workspace):
    workspace=Path(workspace).resolve(strict=True)
    if (workspace/'failure.json').exists():raise ValueError('FAILED_WORKSPACE_PRESERVED')
    manifest=ev.read_json(workspace/'manifest.json')
    if manifest['schema']!='healtheat-forward-v1' or manifest['gt_version']!='v2':
        raise ValueError('INVALID_FORWARD_MANIFEST')
    if bf.file_digest(Path(__file__))!=manifest['forward_code_sha256']:
        raise ValueError('FORWARD_CODE_CHANGED')
    # Check the payload directly; checked_inputs also binds the running runner copy.
    inputs=workspace/'inputs';im=ev.read_json(inputs/'inputs.json')
    if bf.file_digest(inputs/'inputs.json')!=manifest['input_manifest_sha256']:
        raise ValueError('INPUT_MANIFEST_CHANGED')
    for name,sha in im['files'].items():
        p=(inputs/name).resolve(strict=True)
        if not p.is_relative_to(inputs) or bf.file_digest(p)!=sha:raise ValueError('INPUT_HASH_MISMATCH')
    observed={p.relative_to(inputs).as_posix() for p in inputs.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name!='inputs.json'}
    if observed!=set(im['files']):raise ValueError('INPUT_FILE_SET_CHANGED')
    gt=ev.read_json(workspace/'ground_truth_v2.json')
    if bf.file_digest(workspace/'ground_truth_v2.json')!=manifest['ground_truth_file_sha256'] or bf.gt_digest(gt)!=manifest['gt_sha256']:
        raise ValueError('GT_CHANGED')
    if bf.file_digest(workspace/'label_v2_manifest.json')!=manifest['label_manifest_sha256']:
        raise ValueError('GT_MANIFEST_CHANGED')
    sources=ev.read_json(workspace/'local_sources.json');root=Path(sources['project_root'])
    if protected_files(root)!=sources['protected']:raise ValueError('PROTECTED_ORIGINAL_CHANGED')
    le.verify_v2(root)
    return manifest,gt,root,im


def read_run(directory, expected_sha, expected_names, scope):
    directory=Path(directory)
    if (directory/'failure.json').exists():raise ValueError('FAILED_RUN_PRESERVED')
    receipt=ev.read_json(directory/'run.json')
    if (receipt.get('status')!='PASS' or receipt.get('model_executed') is not True
            or receipt.get('training') is not False or receipt.get('scope')!=scope
            or receipt.get('input_manifest_sha256')!=expected_sha
            or receipt.get('image_count')!=len(expected_names)):
        raise ValueError('INVALID_RUN_RECEIPT')
    if set(receipt['artifacts'])!={'raw_predictions.json','environment.json','effective_config.py'}:
        raise ValueError('INVALID_RUN_ARTIFACT_SET')
    for name,sha in receipt['artifacts'].items():
        if bf.file_digest(directory/name)!=sha:raise ValueError('RUN_ARTIFACT_CHANGED')
    raw=ev.read_json(directory/'raw_predictions.json')
    if raw.get('schema')!='healtheat-raw-v1' or raw.get('input_manifest_sha256')!=expected_sha:
        raise ValueError('RAW_INPUT_MISMATCH')
    if [r['filename'] for r in raw['rows']]!=expected_names:raise ValueError('RAW_IMAGE_SET_MISMATCH')
    for row in raw['rows']:
        validate_raw(row)
        if row.get('float_dtype')!='float32' or not bf.finite(row['elapsed_ms']) or row['elapsed_ms']<0:
            raise ValueError('RAW_METADATA_INVALID')
    return receipt,raw


def score(workspace):
    import yaml
    workspace=Path(workspace).resolve(strict=True)
    manifest,gt,root,im=verify_inputs(workspace)
    output=ev.unused_directory(workspace/'evaluation',[])
    smoke,_=read_run(workspace/'smoke',manifest['input_manifest_sha256'],im['images'][:1],'smoke')
    run,raw=read_run(workspace/'full',manifest['input_manifest_sha256'],im['images'],'full')
    if smoke['finished_unix']>run['started_unix']:raise ValueError('FULL_RUN_PRECEDED_SMOKE_GATE')
    for key in ['environment.json','effective_config.py']:
        if smoke['artifacts'][key]!=run['artifacts'][key]:raise ValueError('SMOKE_FULL_ENVIRONMENT_MISMATCH')
    names=yaml.safe_load((workspace/'inputs/pill.yaml').read_text(encoding='utf-8'))['names']
    extractor=module(workspace/'inputs/postprocess.py').extract_top3
    output.mkdir()
    try:
        summaries={};rows_by_name={r['filename']:r for r in raw['rows']}
        for threshold in manifest['thresholds']:
            data=copy.deepcopy(gt)
            data.update(conf_threshold=threshold,nms_iou=manifest['grouping_iou'],seed=100,
                        eval_version='B-forward-gt-v2',ms_per_image=sum(r['elapsed_ms'] for r in raw['rows'])/len(raw['rows']))
            for row in data['results']:
                b,s,l=validate_raw(rows_by_name[row['filename']])
                row['predictions']=extractor(b,s,l,names,conf_thr=threshold,iou_thr=0.5,topk=3)
            ev.validate(data)
            suffix=f'{threshold:.2f}'
            bf.write_json(output/f'predictions-{suffix}.json',data)
            metrics=bf.score_evaluation(data);summaries[suffix]=metrics
            bf.write_json(output/f'metrics-{suffix}.json',metrics)
            comparisons=[]
            for path in sorted((root/'eval').glob('*_results.json')):
                historical,_=le.rescore_gt(ev.read_json(path),gt,'v2',threshold,bf.file_digest(path))
                report=ev.compare(historical,data)
                # A stored Top-3 post-filter is not the same contract as filtering raw predictions.
                report['metric_delta_candidate_minus_reference']=None
                report['same_numeric_conditions']=report['same_postprocess']
                report['same_postprocess']=False
                report['historical_postprocess_lineage']='UNRESOLVED; post-hoc stored Top-3 filtering'
                report['forward_postprocess']='float32 pred_instances -> independent threshold -> notebook grouping'
                report['meaning']='Descriptive stored/new comparison only; not a controlled model-improvement claim'
                report['reference_file']=path.name
                comparisons.append(report)
            bf.write_json(output/f'comparisons-{suffix}.json',comparisons)
        verify_inputs(workspace)
        artifacts={p.name:bf.file_digest(p) for p in output.iterdir() if p.is_file()}
        receipt={'status':'PASS','model_executed':True,'gt_version':'v2','metrics':summaries,
                 'artifacts':artifacts,'smoke_run_sha256':bf.file_digest(workspace/'smoke/run.json'),
                 'full_run_sha256':bf.file_digest(workspace/'full/run.json'),
                 'manifest_sha256':bf.file_digest(workspace/'manifest.json'),
                 'limitations':['Reused evaluation set','Historical checkpoint lineage unresolved',
                                'GT correction is not model improvement','No generalization or medical-use validation']}
        bf.write_json(output/'receipt.json',receipt)
        return receipt
    except BaseException as exc:
        bf.write_json(output/'failure.json',{'status':'FAILED','error_type':type(exc).__name__})
        raise


def record(workspace, tracking):
    workspace=Path(workspace).resolve(strict=True)
    manifest,gt,root,im=verify_inputs(workspace)
    report=ev.read_json(workspace/'evaluation/receipt.json')
    if (workspace/'evaluation/failure.json').exists() or report['status']!='PASS':
        raise ValueError('EVALUATION_NOT_PASSED')
    for name,sha in report['artifacts'].items():
        if bf.file_digest(workspace/'evaluation'/name)!=sha:raise ValueError('EVALUATION_CHANGED')
    for scope in ['smoke','full']:
        read_run(workspace/scope,manifest['input_manifest_sha256'],im['images'][:1] if scope=='smoke' else im['images'],scope)
        if bf.file_digest(workspace/scope/'run.json')!=report[f'{scope}_run_sha256']:raise ValueError('RUN_RECEIPT_CHANGED')
    if bf.file_digest(workspace/'manifest.json')!=report['manifest_sha256']:raise ValueError('MANIFEST_CHANGED')
    for threshold,metrics in report['metrics'].items():
        data=ev.read_json(workspace/f'evaluation/predictions-{threshold}.json')
        if bf.score_evaluation(data)!=metrics or bf.gt_digest(data)!=manifest['gt_sha256']:
            raise ValueError('SCORE_RECHECK_FAILED')
    tracking=ev.unused_directory(tracking,[root,workspace]);tracking.mkdir(parents=True)
    store=tracking/'tracking';store.mkdir()
    os.environ['MLFLOW_DISABLE_TELEMETRY']='true';os.environ['MLFLOW_DISABLE_AGENT_HINT']='true'
    from mlflow import MlflowClient
    from mlflow.entities import Param,Metric
    uri='sqlite:///'+(store/'tracking.db').as_posix();client=MlflowClient(tracking_uri=uri,registry_uri=uri)
    tags={'mlflow.runName':'HealthEat v4 forward evaluation / GT v2','record_kind':'new_inference',
          'model_executed':'true','training':'false','gt_version':'v2','historical_checkpoint_lineage':'UNRESOLVED'}
    values={'params':{'checkpoint_sha256':im['files']['model.pth'],'config_sha256':im['files']['config.py'],
                     'gt_sha256':manifest['gt_sha256'],'input_manifest_sha256':manifest['input_manifest_sha256']},
            'points':[(f'conf{t.replace(".","")}_{k}',float(v),0) for t,m in report['metrics'].items() for k,v in m.items()]}
    names=['manifest.json','ground_truth_v2.json','label_v2_manifest.json','inputs/inputs.json',
           'inputs/inference_runner.py','inputs/postprocess.py','inputs/config.py','inputs/pill.yaml',
           'full/run.json','full/raw_predictions.json','full/environment.json','full/effective_config.py',
           'smoke/run.json','smoke/raw_predictions.json','smoke/environment.json','smoke/effective_config.py',
           *[f'evaluation/{n}' for n in report['artifacts']],'evaluation/receipt.json']
    blobs={n.replace('/','__'):(workspace/n).read_bytes() for n in names}
    blobs['forward.py.txt']=Path(__file__).read_bytes();blobs['backfill_runs.py.txt']=Path(bf.__file__).read_bytes()
    run_id=None
    try:
        exp=client.create_experiment('HealthEat B-stage forward',artifact_location=(store/'artifacts').as_uri())
        run_id=client.create_run(exp,tags=tags).info.run_id
        client.log_batch(run_id,params=[Param(k,str(v)) for k,v in values['params'].items()],
                         metrics=[Metric(k,v,int(time.time()*1000),step) for k,v,step in values['points']])
        upload=tracking/'recorded_files';upload.mkdir()
        for name,raw in blobs.items():
            (upload/name).write_bytes(raw);client.log_artifact(run_id,str(upload/name))
        bf.verify_run(client,run_id,values,tags,blobs,store,'RUNNING')
        verify_inputs(workspace)
        client.set_terminated(run_id,'FINISHED')
        bf.verify_run(client,run_id,values,tags,blobs,store,'FINISHED')
        receipt={'status':'PASS','run_id':run_id,'record':values,'tags':tags,
                 'artifacts':{n:bf.digest(raw) for n,raw in blobs.items()}}
        bf.write_json(tracking/'receipt.json',receipt)
        return {'status':'PASS','run_id':run_id}
    except BaseException as exc:
        if run_id:client.set_terminated(run_id,'FAILED')
        bf.write_json(tracking/'failure.json',{'status':'FAILED','run_id':run_id,'error_type':type(exc).__name__})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project-root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
    for name in ['verify','score','record']:
        p=sub.add_parser(name);p.add_argument('--workspace',type=Path,required=True)
        if name=='record':p.add_argument('--tracking',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':r=prepare(args.project_root,args.workspace)
    elif args.command=='verify':r={'status':'PASS','gt_sha256':verify_inputs(args.workspace)[0]['gt_sha256']}
    elif args.command=='score':r=score(args.workspace)
    else:r=record(args.workspace,args.tracking)
    print(json.dumps(r,ensure_ascii=False,indent=2))
