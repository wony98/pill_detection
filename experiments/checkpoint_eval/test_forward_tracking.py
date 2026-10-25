"""Isolated synthetic MLflow tests. No detector/model import or existing store access."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import forward as fw


class TrackingChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='healtheat-forward-tracking-test-')
        self.addCleanup(self.tmp.cleanup)
        self.target=Path(self.tmp.name)/'tracking'
        self.values={'params':{'kind':'synthetic'},'points':[('example',.5,0)]}
        self.tags={'mlflow.runName':'B tracking synthetic test','record_kind':'synthetic_validation','model_executed':'false'}
        self.blobs={'fixture.json':b'{"synthetic":true}\n'}

    def create(self,check=lambda:None):
        return fw.write_tracking(self.target,self.values,self.tags,self.blobs,check)

    def test_real_sqlite_roundtrip(self):
        result=self.create()
        self.assertEqual(fw.verify_record(self.target)['run_id'],result['run_id'])
        self.assertFalse((self.target/'failure.json').exists())

    def test_existing_store_never_overwritten(self):
        self.create()
        before=(self.target/'receipt.json').read_bytes()
        with self.assertRaises(FileExistsError):self.create()
        self.assertEqual(before,(self.target/'receipt.json').read_bytes())

    def test_client_initialization_failure_preserved(self):
        with patch.object(fw,'tracking_client',side_effect=ImportError('synthetic init failure')):
            with self.assertRaises(ImportError):self.create()
        failure=fw.ev.read_json(self.target/'failure.json')
        self.assertEqual(failure['error_type'],'ImportError')
        self.assertIsNone(failure['run_id'])
        self.assertFalse((self.target/'receipt.json').exists())
        with self.assertRaisesRegex(ValueError,'FAILED_TRACKING'):fw.verify_record(self.target)

    def test_postcheck_failure_marks_run_failed(self):
        def fail():raise ValueError('synthetic source changed')
        with self.assertRaises(ValueError):self.create(fail)
        receipt=fw.ev.read_json(self.target/'failure.json')
        client=fw.tracking_client(self.target/'tracking')
        self.assertEqual(client.get_run(receipt['run_id']).info.status,'FAILED')
        self.assertFalse((self.target/'receipt.json').exists())

    def test_artifact_tamper_detected(self):
        result=self.create()
        artifact=self.target/'tracking/artifacts'/result['run_id']/'artifacts/fixture.json'
        artifact.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_HASH_MISMATCH'):fw.verify_record(self.target)

    def test_metric_tamper_detected(self):
        result=self.create()
        fw.tracking_client(self.target/'tracking').log_metric(result['run_id'],'example',99)
        with self.assertRaisesRegex(ValueError,'METRIC_HISTORY'):fw.verify_record(self.target)

    def test_local_recorded_copy_tamper_detected(self):
        self.create();(self.target/'recorded_files/fixture.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'RECORDED_ARTIFACT'):fw.verify_record(self.target)

    def test_missing_database_not_recreated(self):
        self.create();(self.target/'tracking/tracking.db').rename(self.target/'tracking/original.db')
        with self.assertRaisesRegex(ValueError,'TRACKING_DB_MISSING'):fw.verify_record(self.target)
        self.assertFalse((self.target/'tracking/tracking.db').exists())


if __name__=='__main__':unittest.main()
