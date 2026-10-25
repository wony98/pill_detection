"""Model-free tests. All writable fixtures are created in temporary directories."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import evaluation as ev


def fixture(root):
    root = Path(root)
    for folder in ('demo/v4','eval/test_images','eval/test_labels'):
        (root/folder).mkdir(parents=True)
    (root/'demo/v4/pill.yaml').write_text('names:\n  0: 약A\n  1: 약B\n', encoding='utf-8')
    (root/'demo/v4/v4.pth').write_bytes(b'synthetic opaque bytes; never deserialize')
    (root/'demo/v4/rtmdet_tiny_pill.py').write_text("raise RuntimeError('must not execute')\n", encoding='utf-8')
    Image.new('RGB',(10,10)).save(root/'eval/test_images/one.png')
    (root/'eval/test_labels/one.txt').write_text('0 0.5 0.5 0.4 0.4\n', encoding='utf-8')
    ev.bf.write_json(root/'eval/answer_key.json', {'one.png':{'class_idx':0,'dl_name':'약A'}})
    data = {'num_images':1, 'num_gt_bboxes':1, 'ms_per_image':0.0, 'seed':100,
            'conf_threshold':0.05,'nms_iou':0.5,
            'results':[{'filename':'one.png','gt_bboxes':[{'bbox':[3.0,3.0,7.0,7.0],'dl_name':'약A'}],
                        'predictions':[{'bbox':[3,3,7,7],'top3_class_name':['약A','약B'], 'top3_confidence':[0.9,0.4]}]}]}
    ev.bf.write_json(root/'eval/synthetic_results.json',data)
    return data


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='healtheat-a-unit-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'project'
        self.data = fixture(self.root)
        self.pred = self.root/'eval/synthetic_results.json'
        self.bundle = self.base/'bundle'

    def prepared(self, kind='synthetic'):
        return ev.prepare(self.root,self.pred,kind,self.bundle)

    def test_gt_coordinates_and_order(self):
        gt, _, audit = ev.reconstruct_gt(self.root)
        self.assertEqual(gt['results'][0]['gt_bboxes'], self.data['results'][0]['gt_bboxes'])
        self.assertEqual(audit, {'warnings':[], 'conflicts':[]})
        self.assertEqual(ev.bf.gt_digest(gt),ev.bf.gt_digest(self.data))

    def test_auxiliary_index_is_not_silently_used_as_label_id(self):
        ev.bf.write_json(self.root/'eval/answer_key.json', {'one.png':{'class_idx':1,'dl_name':'약A'}})
        gt, _, audit = ev.reconstruct_gt(self.root)
        self.assertEqual(len(audit['warnings']),1)
        self.assertFalse(audit['conflicts'])
        self.assertEqual(gt['results'][0]['gt_bboxes'][0]['dl_name'],'약A')

    def test_auxiliary_name_conflict_stops_prepare(self):
        ev.bf.write_json(self.root/'eval/answer_key.json', {'one.png':{'class_idx':1,'dl_name':'약B'}})
        with self.assertRaisesRegex(ValueError,'GT_REFERENCE_CONFLICT'):
            self.prepared()
        self.assertFalse(self.bundle.exists())

    def test_audit_reports_conflict_without_relabeling(self):
        p=self.root/'eval/answer_key.json'
        ev.bf.write_json(p,{'one.png':{'class_idx':1,'dl_name':'약B'}})
        before=p.read_bytes()
        report=ev.audit_gt(self.root,self.base/'audit')
        self.assertEqual(report['status'],'NEEDS_LABEL_REVIEW')
        self.assertTrue(report['reference_results'][0]['matches_reconstructed_gt'])
        self.assertEqual(p.read_bytes(),before)

    def test_audit_input_change_is_not_reported_as_success(self):
        original=ev.reconstruct_gt
        def changed(root):
            result=original(root)
            (self.root/'eval/test_labels/one.txt').write_text('0 0.5 0.5 0.3 0.3')
            return result
        with patch.object(ev,'reconstruct_gt',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):
                ev.audit_gt(self.root,self.base/'audit')
        self.assertFalse((self.base/'audit').exists())

    def test_audit_gt_mismatch_is_explicit(self):
        self.data['results'][0]['gt_bboxes'][0]['bbox'][0]=2
        ev.bf.write_json(self.pred,self.data)
        report=ev.audit_gt(self.root,self.base/'audit')
        self.assertEqual(report['status'],'GT_MISMATCH')

    def test_missing_label_rejected(self):
        (self.root/'eval/test_labels/one.txt').unlink()
        with self.assertRaises(ValueError):self.prepared()

    def test_invalid_yolo_class_rejected(self):
        (self.root/'eval/test_labels/one.txt').write_text('7 0.5 0.5 0.4 0.4')
        with self.assertRaisesRegex(ValueError,'INVALID_YOLO'):self.prepared()

    def test_nonfinite_yolo_rejected(self):
        (self.root/'eval/test_labels/one.txt').write_text('0 nan 0.5 0.4 0.4')
        with self.assertRaisesRegex(ValueError,'COORDINATES'):self.prepared()

    def test_invalid_image_filename_rejected(self):
        self.data['results'][0]['filename']='../one.png'
        with self.assertRaisesRegex(ValueError,'IMAGE_NAME'):ev.validate(self.data)

    def test_duplicate_image_rejected(self):
        self.data['results']*=2; self.data['num_images']=2; self.data['num_gt_bboxes']=2
        with self.assertRaisesRegex(ValueError,'DUPLICATE_IMAGE'):ev.validate(self.data)

    def test_missing_condition_rejected(self):
        del self.data['conf_threshold']
        with self.assertRaisesRegex(ValueError,'POSTPROCESS'):ev.validate(self.data)

    def test_historical_bundle_keeps_lineage_unknown(self):
        manifest=self.prepared('historical')
        self.assertEqual(manifest['checkpoint_evaluation_link'],'UNRESOLVED')
        self.assertFalse(manifest['model_executed'])
        loaded,data=ev.load_bundle(self.bundle)
        self.assertEqual(loaded,manifest)
        self.assertEqual(data,self.data)

    def test_forward_claim_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'NO_FORWARD'):self.prepared('forward')

    def test_gt_difference_rejected(self):
        self.data['results'][0]['gt_bboxes'][0]['bbox'][0]=2
        ev.bf.write_json(self.pred,self.data)
        with self.assertRaisesRegex(ValueError,'GROUND_TRUTH_MISMATCH'):self.prepared()

    def test_existing_output_never_overwritten(self):
        self.bundle.mkdir(); (self.bundle/'sentinel').write_text('keep')
        with self.assertRaisesRegex(FileExistsError,'OUTPUT_ALREADY_EXISTS'):self.prepared()
        self.assertEqual((self.bundle/'sentinel').read_text(),'keep')

    def test_output_under_project_rejected(self):
        with self.assertRaisesRegex(ValueError,'PROTECTED_INPUT'):
            ev.prepare(self.root,self.pred,'synthetic',self.root/'new-output')

    def test_artifact_tamper_rejected(self):
        self.prepared(); (self.bundle/'metrics.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_HASH'):ev.load_bundle(self.bundle)

    def test_input_tamper_rejected(self):
        self.prepared(); (self.root/'eval/test_labels/one.txt').write_text('0 0.5 0.5 0.3 0.3')
        with self.assertRaisesRegex(ValueError,'SOURCE_HASH'):ev.load_bundle(self.bundle)

    def test_new_input_file_rejected(self):
        self.prepared(); Image.new('RGB',(10,10)).save(self.root/'eval/test_images/two.png')
        with self.assertRaisesRegex(ValueError,'FILE_SET_CHANGED'):ev.load_bundle(self.bundle)

    def test_manifest_gt_tamper_rejected(self):
        self.prepared(); path=self.bundle/'manifest.json'; m=ev.read_json(path); m['gt_sha256']='0'*64
        ev.bf.write_json(path,m)
        with self.assertRaisesRegex(ValueError,'CONTENT_MISMATCH'):ev.load_bundle(self.bundle)

    def test_manifest_forward_tamper_rejected(self):
        self.prepared(); p=self.bundle/'manifest.json'; m=ev.read_json(p);m['model_executed']=True;ev.bf.write_json(p,m)
        with self.assertRaisesRegex(ValueError,'INVALID_A_STAGE'):ev.load_bundle(self.bundle)

    def test_identical_predictions_zero_change(self):
        r=ev.compare(self.data,self.data)
        self.assertTrue(r['same_gt']);self.assertTrue(r['same_postprocess'])
        self.assertEqual(r['images'][0]['reference_only_boxes'],0)
        self.assertTrue(all(x==0 for x in r['metric_delta_candidate_minus_reference'].values()))

    def test_box_move_has_both_unmatched_sides(self):
        other=copy.deepcopy(self.data);other['results'][0]['predictions'][0]['bbox']=[20,20,30,30]
        r=ev.compare(self.data,other)['images'][0]
        self.assertEqual((r['reference_only_boxes'],r['candidate_only_boxes']),(1,1))

    def test_class_change_is_not_geometric_miss(self):
        other=copy.deepcopy(self.data);other['results'][0]['predictions'][0]['top3_class_name']=['약B','약A']
        r=ev.compare(self.data,other)
        self.assertEqual(r['images'][0]['matched'],1)
        self.assertEqual(r['images'][0]['top1_agreements'],0)
        self.assertEqual(r['metric_delta_candidate_minus_reference']['n_top1'],-1)

    def test_ties_use_candidate_order_and_predictions_once(self):
        other=copy.deepcopy(self.data);other['results'][0]['predictions']*=2
        r=ev.compare(self.data,other)['images'][0]
        self.assertEqual(r['pairs'][0]['candidate_index'],0)
        self.assertEqual(r['candidate_only_boxes'],1)

    def test_changed_gt_suppresses_global_delta(self):
        other=copy.deepcopy(self.data);other['results'][0]['gt_bboxes'][0]['bbox'][0]=2
        r=ev.compare(self.data,other)
        self.assertFalse(r['same_gt']);self.assertIsNone(r['metric_delta_candidate_minus_reference'])

    def test_changed_threshold_suppresses_global_delta(self):
        other=copy.deepcopy(self.data);other['conf_threshold']=0.25
        r=ev.compare(self.data,other)
        self.assertFalse(r['same_postprocess']);self.assertIsNone(r['metric_delta_candidate_minus_reference'])

    def test_missing_image_is_explicit(self):
        other=copy.deepcopy(self.data);other['results'][0]['filename']='two.png'
        r=ev.compare(self.data,other)
        self.assertEqual(r['reference_only_images'],['one.png']);self.assertEqual(r['candidate_only_images'],['two.png'])
        self.assertIsNone(r['metric_delta_candidate_minus_reference'])

    def test_confidence_delta_keeps_precision(self):
        other=copy.deepcopy(self.data);other['results'][0]['predictions'][0]['top3_confidence'][0]=0.900001
        r=ev.compare(self.data,other)
        self.assertAlmostEqual(r['images'][0]['pairs'][0]['top1_confidence_delta'],0.000001)

    def test_historical_bundle_cannot_create_synthetic_run(self):
        self.prepared('historical')
        with self.assertRaisesRegex(ValueError,'SYNTHETIC_ONLY'):
            ev.record_synthetic(self.bundle,self.base/'tracking')
        self.assertFalse((self.base/'tracking').exists())

    def test_compare_writes_bounded_report_without_source_changes(self):
        self.prepared(); before=self.pred.read_bytes()
        r=ev.compare_bundles(self.bundle,self.bundle,self.base/'comparison')
        self.assertTrue(r['same_gt'])
        self.assertTrue((self.base/'comparison/comparison.md').is_file())
        self.assertEqual(self.pred.read_bytes(),before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
