"""Exercise MLflow failure paths with synthetic records in a new caller-owned directory."""
import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import url2pathname

os.environ['MLFLOW_DISABLE_TELEMETRY']='true'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='true'
from mlflow import MlflowClient
import backfill_runs as bf


def expect_error(token, call):
    try:
        call()
    except ValueError as exc:
        if token not in str(exc):
            raise
    else:
        raise AssertionError('expected '+token)


def main(root):
    root=root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source=root/'synthetic_source.json'
    source.write_text('{}',encoding='utf-8')
    snapshots=[(source,bf.file_digest(source))]
    record={'label':'synthetic_evaluation','kind':'evaluation_history','source_sha256':snapshots[0][1],
            'params':{'seed':100},'points':[('accuracy',0.5,0)],
            'evidence':{'checkpoint_link':'UNRESOLVED'},'artifact_bytes':{'evaluation.json':b'{}'}}
    store=root/'store'
    def run(label,record=record):
        return bf.ingest([record],store,root/label,[],[],snapshots,[])
    assert run('first')['new_runs']==1
    repeated=run('repeat')
    assert repeated['new_runs']==0 and repeated['reused_runs']==1
    client=MlflowClient(tracking_uri='sqlite:///'+(store/'tracking.db').as_posix())
    exp=client.get_experiment_by_name('HealthEat historical backfill')
    runs=client.search_runs([exp.experiment_id])
    assert len(runs)==1
    run_id=runs[0].info.run_id
    uri=urlparse(runs[0].info.artifact_uri)
    artifact_root=Path(url2pathname(uri.path))
    lineage=artifact_root/'lineage.json'
    original=lineage.read_bytes()
    lineage.write_text('{"checkpoint_link":"wrong"}',encoding='utf-8')
    expect_error('ARTIFACT_HASH_MISMATCH',lambda:run('artifact_corruption'))
    lineage.write_bytes(original)  # Restore only this test's synthetic fixture.
    client.set_tag(run_id,'capture_mode','incorrect')
    expect_error('RUN_READBACK_MISMATCH',lambda:run('tag_corruption'))
    client.set_tag(run_id,'capture_mode','historical_backfill')
    changed=copy.deepcopy(record);changed['label']='synthetic_source_change'
    original_verify=bf.verify_run
    def mutate_after_verification(*args,**kwargs):
        original_verify(*args,**kwargs)
        source.write_text('{"changed":true}',encoding='utf-8')
    with patch.object(bf,'verify_run',side_effect=mutate_after_verification):
        expect_error('SOURCE_CHANGED_DURING_BACKFILL',lambda:run('source_changed',changed))
    all_runs=client.search_runs([exp.experiment_id])
    assert len(all_runs)==2
    failed=next(r for r in all_runs if r.data.tags['mlflow.runName']=='synthetic_source_change')
    assert failed.info.status=='FAILED'
    assert client.get_run(run_id).info.status=='FINISHED'
    source.write_text('{}',encoding='utf-8')
    expect_error('EXISTING_RUN_NOT_UNIQUELY_FINISHED',lambda:run('retry_failed',changed))
    assert len(client.search_runs([exp.experiment_id]))==2
    assert run('final_repeat')['reused_runs']==1
    receipt={'status':'PASS','checks':['new_run','idempotent_reuse','artifact_tamper_rejected',
             'tag_tamper_rejected','source_change_marks_failed','failed_run_not_retried','finished_run_preserved'],
             'synthetic_only':True,'runs_remaining':2,'statuses':['FINISHED','FAILED']}
    bf.write_json(root/'integration_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    main(parser.parse_args().workspace)
