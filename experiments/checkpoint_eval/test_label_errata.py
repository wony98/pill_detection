"""Synthetic-only tests for the explicitly versioned, two-token GT correction."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import unicodedata

from PIL import Image
import yaml
import evaluation as ev
import label_errata as le


def fixture(root):
    for folder in ('demo/v4','eval/test_images','eval/test_labels','eval/reference'):
        (root/folder).mkdir(parents=True)
    names={i:f'class_{i}' for i in range(48)}
    names[16]='프로스카정';names[47]='콜리네이트연질캡슐 400mg'
    (root/'demo/v4/pill.yaml').write_text(yaml.safe_dump({'names':names},allow_unicode=True),encoding='utf-8')
    (root/'demo/v4/v4.pth').write_bytes(b'opaque synthetic checkpoint')
    (root/'demo/v4/rtmdet_tiny_pill.py').write_text("raise RuntimeError('not executed')")
    Image.new('RGB',(100,100)).save(root/'eval/test_images/프로스카정.png')
    raw=b'47 0.25 0.5 0.2 0.2\n47 0.75 0.5 0.2 0.2\n'
    (root/'eval/test_labels'/le.TARGET_LABEL).write_bytes(raw)
    ev.bf.write_json(root/'eval/answer_key.json',{'프로스카정.png':{'class_idx':16,'dl_name':'프로스카정'}})
    rows=[{'dl_idx':'5390','final_class_idx_merged':16,'dl_name':'프로스카정'}]
    ev.bf.write_json(root/'eval/reference/class_mapping.json',{
        'schema_version':1,'columns':['dl_idx','final_class_idx_merged','dl_name'],
        'row_count':len(rows),'rows':rows,'rows_sha256':ev.bf.digest(ev.bf.canonical(rows)),
        'source_sha256':ev.bf.digest(b'synthetic mapping source'),'source_required_at_runtime':False,
        'empty_source_dl_idx_preserved':0})
    gt,_,_=ev.reconstruct_gt(root)
    data=copy.deepcopy(gt);data.update(conf_threshold=0.05,nms_iou=0.5,seed=100)
    data['results'][0]['predictions']=[{'bbox':x['bbox'],'top3_class_name':['프로스카정'],'top3_confidence':[0.9]} for x in gt['results'][0]['gt_bboxes']]
    data['results'][0]['predictions'].append({'bbox':[0,0,1,1],'top3_class_name':['프로스카정'],'top3_confidence':[0.1]})
    for name in ('e11','e14','e15','e13'):
        ev.bf.write_json(root/f'eval/{name}_results.json',data)
    return raw,gt,data


class VersionedLabels(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='label-v2-unit-');self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'project'
        self.raw,self.gt,self.data=fixture(self.root)
        self.p1=patch.object(le,'OLD_LABEL_SHA',ev.bf.digest(self.raw));self.p1.start();self.addCleanup(self.p1.stop)
        self.p2=patch.object(le,'OLD_GT_SHA',ev.bf.gt_digest(self.gt));self.p2.start();self.addCleanup(self.p2.stop)

    def test_two_tokens_only_coordinates_and_v1_preserved(self):
        m=le.create_v2(self.root)
        old=self.root/'eval/test_labels'/le.TARGET_LABEL
        new=self.root/'eval/label_versions/v2/labels'/le.TARGET_LABEL
        self.assertEqual(old.read_bytes(),self.raw)
        self.assertEqual(new.read_bytes(),self.raw.replace(b'47 ',b'16 '))
        self.assertEqual(len(m['name_changes']),2)
        self.assertNotEqual(m['v1_gt_sha256'],m['v2_gt_sha256'])
        le.verify_v2(self.root)

    def test_existing_v2_not_overwritten(self):
        le.create_v2(self.root)
        with self.assertRaisesRegex(FileExistsError,'ALREADY_EXISTS'):le.create_v2(self.root)

    def test_unreviewed_source_label_refused(self):
        (self.root/'eval/test_labels'/le.TARGET_LABEL).write_bytes(self.raw.replace(b'0.25',b'0.26'))
        with self.assertRaises(ValueError):le.create_v2(self.root)
        self.assertFalse((self.root/'eval/label_versions/v2').exists())

    def test_mapping_change_refused(self):
        path=self.root/'eval/reference/class_mapping.json';d=ev.read_json(path)
        d['rows'][0]['final_class_idx_merged']=17;d['rows_sha256']=ev.bf.digest(ev.bf.canonical(d['rows']))
        ev.bf.write_json(path,d)
        with self.assertRaisesRegex(ValueError,'MAPPING_MISMATCH'):le.create_v2(self.root)

    def test_mapping_row_hash_mismatch_refused(self):
        path=self.root/'eval/reference/class_mapping.json';d=ev.read_json(path)
        d['rows'][0]['dl_name']='changed';ev.bf.write_json(path,d)
        with self.assertRaisesRegex(ValueError,'ROWS_HASH_MISMATCH'):le.create_v2(self.root)

    def test_no_private_data_directory_needed(self):
        self.assertFalse((self.root/'data').exists())
        le.create_v2(self.root);le.verify_v2(self.root)
        r=le.write_errata(self.root,self.base/'portable-report')
        self.assertEqual(len(r['rows']),16)

    def test_missing_mapping_input_in_manifest_refused(self):
        le.create_v2(self.root)
        path=self.root/'eval/label_versions/v2/manifest.json';d=ev.read_json(path)
        d['source_files']=[r for r in d['source_files'] if r['file']!='eval/reference/class_mapping.json']
        ev.bf.write_json(path,d)
        with self.assertRaisesRegex(ValueError,'MAPPING_INPUT_MISSING'):le.verify_v2(self.root)

    def test_legacy_private_dependency_refused(self):
        le.create_v2(self.root)
        path=self.root/'eval/label_versions/v2/manifest.json';d=ev.read_json(path)
        d['source_files'].append({'file':'data/final_mapping_after_category_merge.csv','sha256':'0'*64})
        ev.bf.write_json(path,d)
        with self.assertRaisesRegex(ValueError,'LEGACY_PRIVATE_MAPPING'):le.verify_v2(self.root)

    def test_original_source_hash_is_required_metadata(self):
        path=self.root/'eval/reference/class_mapping.json';d=ev.read_json(path)
        d['source_sha256']='missing';ev.bf.write_json(path,d)
        with self.assertRaisesRegex(ValueError,'INVALID_MAPPING_EVIDENCE'):le.create_v2(self.root)

    def test_coordinate_tamper_cannot_be_hidden_by_rehash(self):
        le.create_v2(self.root)
        p=self.root/'eval/label_versions/v2/labels'/le.TARGET_LABEL
        p.write_bytes(p.read_bytes().replace(b'0.25',b'0.26'))
        mp=self.root/'eval/label_versions/v2/manifest.json';m=ev.read_json(mp)
        m['labels'][le.TARGET_LABEL]=ev.bf.file_digest(p);ev.bf.write_json(mp,m)
        with self.assertRaisesRegex(ValueError,'UNAPPROVED_V2_LABEL_BYTES'):le.verify_v2(self.root)

    def test_extra_label_rejected(self):
        le.create_v2(self.root)
        (self.root/'eval/label_versions/v2/labels/extra.txt').write_bytes(self.raw)
        with self.assertRaisesRegex(ValueError,'LABEL_SET_CHANGED'):le.verify_v2(self.root)

    def test_v1_source_change_rejected(self):
        le.create_v2(self.root)
        (self.root/'eval/e11_results.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):le.verify_v2(self.root)

    def test_ground_truth_artifact_tamper_rejected(self):
        le.create_v2(self.root)
        (self.root/'eval/label_versions/v2/ground_truth.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_CHANGED'):le.verify_v2(self.root)

    def test_prepare_remains_strict_v1(self):
        le.create_v2(self.root)
        with self.assertRaisesRegex(ValueError,'GT_REFERENCE_CONFLICT'):
            ev.prepare(self.root,self.root/'eval/e11_results.json','historical',self.base/'bundle')

    def test_posthoc_filter_preserves_source_and_score_precision(self):
        before=ev.bf.canonical(self.data)
        result=le.posthoc_filter(self.data,0.25)
        self.assertEqual(len(result['results'][0]['predictions']),2)
        self.assertEqual(ev.bf.canonical(self.data),before)
        self.assertEqual(result['results'][0]['predictions'][0],self.data['results'][0]['predictions'][0])

    def test_unsupported_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED'):le.posthoc_filter(self.data,0.1)

    def test_explicit_v2_metadata_and_no_prediction_change(self):
        le.create_v2(self.root);m,_,v2=le.verify_v2(self.root)
        new,metrics=le.rescore_gt(self.data,v2,'v2',0.05,'source-hash')
        self.assertEqual(new['gt_version'],'v2')
        self.assertEqual(new['historical_json_gt_version'],'v1')
        self.assertTrue(new['gt_substitution_explicit'])
        self.assertFalse(new['model_executed'])
        self.assertEqual(new['results'][0]['predictions'],self.data['results'][0]['predictions'])
        self.assertEqual(metrics['n_top1'],2)

    def test_v2_cannot_be_labeled_v1(self):
        le.create_v2(self.root);_,_,v2=le.verify_v2(self.root)
        with self.assertRaisesRegex(ValueError,'CANNOT_HIDE'):
            le.rescore_gt(self.data,v2,'v1',0.05,'source-hash')

    def test_other_gt_change_cannot_hide_in_v2(self):
        le.create_v2(self.root);_,_,v2=le.verify_v2(self.root)
        v2['results'][0]['gt_bboxes'][0]['bbox'][0]+=1
        with self.assertRaisesRegex(ValueError,'COORDINATES_CHANGED'):
            le.rescore_gt(self.data,v2,'v2',0.05,'source-hash')

    def test_errata_has_16_rows_and_originals_unchanged(self):
        le.create_v2(self.root)
        files=le.reference_files(self.root);before={p:p.read_bytes() for p in files}
        r=le.write_errata(self.root,self.base/'errata')
        self.assertEqual(len(r['rows']),16)
        self.assertEqual(len(r['derived_files']),16)
        self.assertEqual(before,{p:p.read_bytes() for p in files})
        self.assertFalse(r['model_executed']);self.assertFalse(r['public_metrics_changed'])

    def test_existing_report_refused(self):
        le.create_v2(self.root);out=self.base/'report';out.mkdir()
        with self.assertRaisesRegex(FileExistsError,'OUTPUT_ALREADY_EXISTS'):le.write_errata(self.root,out)

    def test_no_report_inside_original_project(self):
        le.create_v2(self.root)
        with self.assertRaisesRegex(ValueError,'PROTECTED_INPUT'):le.write_errata(self.root,self.root/'bad-report')

    def test_posthoc_rule_matches_notebook_pure_function(self):
        root=Path(ev.__file__).resolve().parents[2]
        nb=ev.read_json(root/'analysis/03_v4_evaluation_analysis.ipynb')
        source=next(''.join(c['source']) for c in nb['cells'] if 'def evaluate_with_conf_threshold(' in ''.join(c.get('source',[])))
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_with_conf_threshold')
        ns={'iou_xyxy':ev.bf.iou_xyxy,'nfc':lambda s:unicodedata.normalize('NFC',s)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'reviewed_notebook_function','exec'),ns)
        for threshold in le.THRESHOLDS:
            with self.subTest(threshold=threshold):
                expected=ns['evaluate_with_conf_threshold'](self.data,threshold)
                got=ev.bf.score_evaluation(le.posthoc_filter(self.data,threshold))
                self.assertEqual(expected,{'top1':got['top1_accuracy'],'top3':got['top3_accuracy'],'det':got['detection_rate'],'fp':got['n_false_pos']})


if __name__=='__main__':unittest.main(verbosity=2)
