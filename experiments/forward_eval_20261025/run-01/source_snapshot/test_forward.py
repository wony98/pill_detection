"""Model-free B-stage tests; never presents synthetic output as a real model run."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import forward as fw
import inference_runner as ir


class ForwardChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='healtheat-b-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.row={'filename':'a.png','bboxes':[[0,0,10,10]],'scores':[0.9],
                  'labels':[1],'elapsed_ms':10.0,'float_dtype':'float32'}

    def test_raw_preserves_float32(self):
        self.row['scores']=[float(np.float32(0.250004))]
        _,scores,labels=ir.validate_raw(self.row)
        self.assertEqual(scores.dtype,np.float32)
        self.assertEqual(scores[0],np.float32(0.250004))
        self.assertEqual(labels.dtype,np.int64)

    def test_raw_empty_allowed(self):
        self.row.update(bboxes=[],scores=[],labels=[])
        b,s,l=ir.validate_raw(self.row)
        self.assertEqual(b.shape,(0,4));self.assertEqual(len(s),0)

    def test_raw_mismatched_lengths_rejected(self):
        self.row['labels']=[]
        with self.assertRaises(ValueError):ir.validate_raw(self.row)

    def test_raw_nan_rejected(self):
        self.row['scores']=[float('nan')]
        with self.assertRaises(ValueError):ir.validate_raw(self.row)

    def test_raw_invalid_class_rejected(self):
        for label in [118,-1,1.0,True]:
            self.row['labels']=[label]
            with self.assertRaises(ValueError):ir.validate_raw(self.row)

    def test_raw_inverted_box_rejected(self):
        self.row['bboxes']=[[10,0,0,10]]
        with self.assertRaises(ValueError):ir.validate_raw(self.row)

    def test_raw_oversized_output_rejected(self):
        self.row.update(bboxes=[[0,0,1,1]]*301,scores=[0.9]*301,labels=[1]*301)
        with self.assertRaises(ValueError):ir.validate_raw(self.row)

    def test_extracted_function_matches_notebook_ast(self):
        root=Path(__file__).resolve().parents[2]
        text=fw.postprocess_source(root/fw.NOTEBOOK)
        tree=ast.parse(text)
        self.assertEqual([n.name for n in tree.body if isinstance(n,ast.FunctionDef)],['iou_xyxy','extract_top3'])
        self.assertNotIn('print(',text)
        self.assertNotIn('inference_detector',text)

    def extractor(self):
        root=Path(__file__).resolve().parents[2]
        source=fw.postprocess_source(root/fw.NOTEBOOK)
        target=self.base/'extracted.py';target.write_text(source,encoding='utf-8')
        return fw.module(target).extract_top3

    def test_independent_threshold_is_not_posthoc_top3_filter(self):
        fn=self.extractor()
        b=np.array([[0,0,10,10]]*3,dtype=np.float32)
        s=np.array([0.9,0.2,0.1],dtype=np.float32);l=np.array([0,1,2])
        low=fn(b,s,l,['a','b','c'],conf_thr=.05)
        high=fn(b,s,l,['a','b','c'],conf_thr=.25)
        self.assertEqual(low[0]['top3_class_name'],['a','b','c'])
        self.assertEqual(high[0]['top3_class_name'],['a'])
        self.assertEqual(len(low),len(high))

    def test_ties_duplicates_and_score_precision(self):
        fn=self.extractor()
        b=np.array([[0,0,10,10]]*3,dtype=np.float32)
        s=np.array([0.250004,0.250004,0.249999],dtype=np.float32)
        out=fn(b,s,np.array([0,1,0]),['a','b'],conf_thr=.25)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['top3_class_name'],['a','b'])
        self.assertEqual(out[0]['top3_confidence'],[float(s[0]),float(s[1])])
        self.assertNotEqual(out[0]['top3_confidence'][0],round(float(s[0]),4))

    def test_empty_extractor(self):
        fn=self.extractor()
        self.assertEqual(fn(np.empty((0,4)),np.array([]),np.array([],dtype=int),['a']),[])

    def test_run_receipt_refuses_synthetic_model_false(self):
        receipt={'status':'PASS','model_executed':False,'training':False,'scope':'full',
                 'input_manifest_sha256':'x','image_count':1}
        fw.bf.write_json(self.base/'run.json',receipt)
        with self.assertRaisesRegex(ValueError,'INVALID_RUN_RECEIPT'):
            fw.read_run(self.base,'x',['a.png'],'full')

    def test_failed_run_not_reused(self):
        fw.bf.write_json(self.base/'failure.json',{'status':'FAILED'})
        with self.assertRaisesRegex(ValueError,'FAILED_RUN'):
            fw.read_run(self.base,'x',['a.png'],'full')

    def test_existing_output_not_overwritten(self):
        with self.assertRaisesRegex(ValueError,'OUTPUT_EXISTS'):
            ir.run(self.base,self.base,'cpu',True)

    def test_payload_rejects_path_escape(self):
        parent=self.base/'payload';parent.mkdir()
        (self.base/'secret').write_text('fixture')
        fw.bf.write_json(parent/'inputs.json',{'schema':'healtheat-forward-input-v1','files':{'../secret':ir.sha(self.base/'secret')}})
        with self.assertRaisesRegex(ValueError,'INPUT_HASH_MISMATCH'):ir.checked_inputs(parent)

    def test_payload_rejects_extra_annotation_file(self):
        parent=self.base/'payload';parent.mkdir()
        (parent/'inference_runner.py').write_bytes(Path(ir.__file__).read_bytes())
        (parent/'labels.json').write_text('{}')
        fw.bf.write_json(parent/'inputs.json',{'schema':'healtheat-forward-input-v1','files':{'inference_runner.py':ir.sha(Path(ir.__file__))}})
        with self.assertRaisesRegex(ValueError,'UNDECLARED'):ir.checked_inputs(parent)

    def config(self):
        names=[f'class_{i}' for i in range(118)]
        pipeline=[{'type':'LoadImageFromFile'},{'type':'Resize','scale':(1536,1536)},
                  {'type':'Pad','size':(1536,1536)},{'type':'LoadAnnotations'},
                  {'type':'PackDetInputs'}]
        cfg=SimpleNamespace(model=SimpleNamespace(backbone=SimpleNamespace(init_cfg={'url':'fixture'}),
                            bbox_head=SimpleNamespace(num_classes=118)),
                            metainfo=SimpleNamespace(classes=names),load_from='fixture',
                            test_dataloader=SimpleNamespace(dataset=SimpleNamespace(pipeline=pipeline,ann_file='fixture')))
        cfg.copy=lambda:copy.deepcopy(cfg)
        return cfg,names

    def test_config_keeps_resize_removes_annotation_and_remote_init(self):
        cfg,names=self.config();new,pipeline=ir.inference_config(cfg,names)
        self.assertIsNone(new.model.backbone.init_cfg);self.assertIsNone(new.load_from)
        self.assertEqual(new.test_dataloader.dataset.ann_file,'')
        self.assertNotIn('LoadAnnotations',[x['type'] for x in pipeline])
        self.assertEqual(pipeline[1]['scale'],(1536,1536))
        self.assertEqual(len(cfg.test_dataloader.dataset.pipeline),5)

    def test_config_wrong_class_mapping_rejected(self):
        cfg,names=self.config();names=list(reversed(names))
        with self.assertRaisesRegex(ValueError,'CLASS_MAPPING'):ir.inference_config(cfg,names)

    def test_config_wrong_resize_rejected(self):
        cfg,names=self.config();cfg.test_dataloader.dataset.pipeline[1]['scale']=(640,640)
        with self.assertRaisesRegex(ValueError,'IMAGE_SIZE'):ir.inference_config(cfg,names)

    def test_config_unreviewed_transform_rejected(self):
        cfg,names=self.config();cfg.test_dataloader.dataset.pipeline.insert(1,{'type':'CustomTransform'})
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_PIPELINE'):ir.inference_config(cfg,names)

    def receipt_fixture(self):
        raw={'schema':'healtheat-raw-v1','input_manifest_sha256':'x','rows':[self.row]}
        fw.bf.write_json(self.base/'raw_predictions.json',raw)
        fw.bf.write_json(self.base/'environment.json',{'test_only':True})
        (self.base/'effective_config.py').write_text('# synthetic test only')
        receipt={'status':'PASS','model_executed':True,'training':False,'scope':'full',
                 'image_count':1,'input_manifest_sha256':'x','started_unix':1,'finished_unix':2,
                 'artifacts':{n:ir.sha(self.base/n) for n in ['raw_predictions.json','environment.json','effective_config.py']}}
        fw.bf.write_json(self.base/'run.json',receipt)

    def test_run_artifact_tamper_rejected(self):
        self.receipt_fixture();(self.base/'environment.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_CHANGED'):fw.read_run(self.base,'x',['a.png'],'full')

    def test_run_missing_image_rejected(self):
        self.receipt_fixture()
        with self.assertRaisesRegex(ValueError,'IMAGE_SET'):fw.read_run(self.base,'x',['b.png'],'full')

    def test_smoke_cannot_be_claimed_as_full_run(self):
        self.receipt_fixture()
        with self.assertRaisesRegex(ValueError,'INVALID_RUN_RECEIPT'):fw.read_run(self.base,'x',['a.png'],'smoke')

    def test_scoring_uses_raw_thresholds_and_existing_scorer(self):
        # Simulated execution only in a temporary test directory, never registered in MLflow.
        workspace=self.base/'workspace';workspace.mkdir();(workspace/'inputs').mkdir()
        (workspace/'smoke').mkdir();(workspace/'full').mkdir()
        for folder in ['smoke','full']:fw.bf.write_json(workspace/folder/'run.json',{'fixture':True})
        fw.bf.write_json(workspace/'manifest.json',{'fixture':True})
        (workspace/'inputs/pill.yaml').write_text('names:\n  0: a\n  1: b\n',encoding='utf-8')
        source=fw.postprocess_source(Path(__file__).resolve().parents[2]/fw.NOTEBOOK)
        (workspace/'inputs/postprocess.py').write_text(source,encoding='utf-8')
        gt={'num_images':1,'num_gt_bboxes':1,'ms_per_image':0.0,'results':[
            {'filename':'a.png','gt_bboxes':[{'bbox':[0,0,10,10],'dl_name':'a'}],'predictions':[]}]}
        raw={'rows':[{'filename':'a.png','bboxes':[[0,0,10,10],[0,0,10,10]],
                     'scores':[0.9,0.2],'labels':[1,0],'elapsed_ms':10.0,'float_dtype':'float32'}]}
        manifest={'input_manifest_sha256':'x','thresholds':[.05,.25],'grouping_iou':.5}
        def read(directory,*args):
            scope=Path(directory).name
            return {'started_unix':3 if scope=='full' else 1,'finished_unix':4 if scope=='full' else 2,
                    'artifacts':{'environment.json':'same','effective_config.py':'same'}},raw
        with patch.object(fw,'verify_inputs',return_value=(manifest,gt,self.base,{'images':['a.png']})),patch.object(fw,'read_run',side_effect=read):
            report=fw.score(workspace)
        self.assertEqual(report['metrics']['0.05']['n_top3'],1)
        self.assertEqual(report['metrics']['0.25']['n_top3'],0)
        self.assertEqual(report['metrics']['0.05']['n_top1'],0)


if __name__=='__main__':unittest.main()
