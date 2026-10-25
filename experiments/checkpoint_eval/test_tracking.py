"""Retain isolated synthetic MLflow stores, including expected failure cases."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import url2pathname

os.environ['MLFLOW_DISABLE_TELEMETRY']='true'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='true'
from mlflow import MlflowClient
import evaluation as ev
from test_evaluation import fixture


def expect_error(token, call, cls=ValueError):
    try:
        call()
    except cls as exc:
        if token not in str(exc):
            raise
    else:
        raise AssertionError('Expected ' + token)


def run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    project=root/'project'
    fixture(project)
    bundle=root/'bundle'
    ev.prepare(project,project/'eval/synthetic_results.json','synthetic',bundle)
    workspace=root/'success'
    receipt=ev.record_synthetic(bundle,workspace)
    assert ev.verify_record(workspace)['status']=='PASS'
    assert receipt['synthetic_only'] and not receipt['model_executed']
    assert not (workspace/'recorded_files/local_sources.json').exists()
    expect_error('OUTPUT_ALREADY_EXISTS',lambda:ev.record_synthetic(bundle,workspace),FileExistsError)
    uri='sqlite:///'+(workspace/'tracking/tracking.db').as_posix()
    client=MlflowClient(tracking_uri=uri,registry_uri=uri)
    run_id=receipt['run_id']
    assert client.get_run(run_id).info.status=='FINISHED'

    # These are deliberately corrupted synthetic outputs, not production artifacts.
    artifact=Path(url2pathname(urlparse(client.get_run(run_id).info.artifact_uri).path))/'metrics.json'
    original=artifact.read_bytes()
    artifact.write_bytes(b'{}')
    expect_error('ARTIFACT_HASH_MISMATCH',lambda:ev.verify_record(workspace))
    artifact.write_bytes(original)
    client.set_tag(run_id,'model_executed','true')
    expect_error('RUN_READBACK_MISMATCH',lambda:ev.verify_record(workspace))
    client.set_tag(run_id,'model_executed','false')
    assert ev.verify_record(workspace)['status']=='PASS'

    failing_project=root/'failing_project'
    fixture(failing_project)
    failing_bundle=root/'failing_bundle'
    ev.prepare(failing_project,failing_project/'eval/synthetic_results.json','synthetic',failing_bundle)
    original_verify=ev.bf.verify_run
    def change_after_readback(*args,**kwargs):
        original_verify(*args,**kwargs)
        (failing_project/'eval/test_labels/one.txt').write_text('0 0.5 0.5 0.3 0.3')
    failed_space=root/'source-changed'
    with patch.object(ev.bf,'verify_run',side_effect=change_after_readback):
        expect_error('SOURCE_HASH_MISMATCH',lambda:ev.record_synthetic(failing_bundle,failed_space))
    failure=ev.read_json(failed_space/'failure.json')
    uri2='sqlite:///'+(failed_space/'tracking/tracking.db').as_posix()
    failed_client=MlflowClient(tracking_uri=uri2,registry_uri=uri2)
    assert failed_client.get_run(failure['run_id']).info.status=='FAILED'
    expect_error('OUTPUT_ALREADY_EXISTS',lambda:ev.record_synthetic(failing_bundle,failed_space),FileExistsError)
    expect_error('FAILED_WORKSPACE_PRESERVED',lambda:ev.verify_record(failed_space))
    assert client.get_run(run_id).info.status=='FINISHED'
    assert not any(n.split('.')[0] in {'torch','mmdet','mmcv'} for n in sys.modules)
    checks=['synthetic_run_label','params_metrics_artifacts_readback','private_paths_not_logged',
            'existing_workspace_refused','artifact_tamper_detected','tag_tamper_detected',
            'input_change_marks_failed','failed_workspace_preserved','completed_run_preserved','no_model_imports']
    report={'status':'PASS','checks':checks,'synthetic_only':True,'model_executed':False,
            'finished_run_id':run_id,'failed_run_id':failure['run_id'],
            'note':'Both disposable stores remain for inspection; originals and backfill DB are not opened'}
    ev.bf.write_json(root/'integration_receipt.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    run(parser.parse_args().workspace)
