"""Synthetic regressions plus a read-only differential check against the existing scorer."""
import ast
import copy
import json
from pathlib import Path
import unicodedata
import unittest

import backfill_runs as bf


def sample():
    return {'num_images':1,'num_gt_bboxes':2,'ms_per_image':1.5,'results':[
        {'filename':'one.png','gt_bboxes':[{'bbox':[0,0,10,10],'dl_name':'약'},
                                        {'bbox':[20,0,30,10],'dl_name':'B'}],
         'predictions':[{'bbox':[0,0,10,10],'top3_class_name':['약'],'top3_confidence':[0.9]},
                        {'bbox':[20,0,30,10],'top3_class_name':['C','B'],'top3_confidence':[0.8,0.7]},
                        {'bbox':[50,0,60,10],'top3_class_name':['D'],'top3_confidence':[0.6]}]}]}


class Regression(unittest.TestCase):
    def test_scoring_counts(self):
        m=bf.score_evaluation(sample())
        self.assertEqual((m['n_top1'],m['n_top3'],m['n_match'],m['n_false_pos']),(1,2,2,1))
        self.assertEqual((m['top1_accuracy'],m['top3_accuracy']),(0.5,1.0))

    def test_prediction_cannot_match_twice(self):
        s=sample();s['results'][0]['gt_bboxes'][1]=copy.deepcopy(s['results'][0]['gt_bboxes'][0])
        self.assertEqual(bf.score_evaluation(s)['n_match'],1)

    def test_name_normalization(self):
        s=sample();s['results'][0]['predictions'][0]['top3_class_name']=[unicodedata.normalize('NFD','약')]
        self.assertEqual(bf.score_evaluation(s)['n_top1'],1)

    def test_metadata_count_rejected(self):
        s=sample();s['num_gt_bboxes']=3
        with self.assertRaisesRegex(ValueError,'GT_COUNT'):bf.score_evaluation(s)

    def test_duplicate_image_rejected(self):
        s=sample();s['num_images']=2;s['results']*=2;s['num_gt_bboxes']=4
        with self.assertRaisesRegex(ValueError,'DUPLICATE_IMAGE'):bf.score_evaluation(s)

    def test_nonfinite_rejected(self):
        s=sample();s['results'][0]['predictions'][0]['bbox'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'BAD_PREDICTION'):bf.score_evaluation(s)

    def test_gt_identity_checks_boxes(self):
        a=sample();b=sample();b['results'][0]['gt_bboxes'][0]['bbox'][0]=1
        self.assertNotEqual(bf.gt_digest(a),bf.gt_digest(b))

    def test_config_not_executed(self):
        text="raise RuntimeError('must not execute')\nrandomness=dict(seed=100)\nmax_epochs=13\ntrain_dataloader=dict(batch_size=16)"
        self.assertEqual(bf.config_params(text),{'seed':100,'max_epochs':13,'batch_size':16})

    def test_nonliteral_config_not_invented(self):
        self.assertEqual(bf.config_params('max_epochs=6+7'),{})

    def test_scalar_steps_preserved(self):
        _,p=bf.scalar_metrics(b'{"step":50,"loss":2.0}\n{"step":100,"loss":1.0}')
        self.assertEqual(p,[('loss',2.0,50),('loss',1.0,100)])

    def test_duplicate_scalar_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE_KEY_STEP'):
            bf.scalar_metrics(b'{"step":1,"loss":2}\n{"step":1,"loss":3}')

    def test_negative_scalar_step_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_STEP'):
            bf.scalar_metrics(b'{"step":-1,"loss":2}')

    def test_existing_output_refused_before_mlflow(self):
        existing=Path(__file__).resolve().parent
        with self.assertRaisesRegex(FileExistsError,'OUTPUT_ALREADY_EXISTS'):
            bf.ingest([],existing/'never-created',existing,[],[],[],[])

    def test_real_json_differential_against_notebook(self):
        repo=Path(__file__).resolve().parents[2]
        nb_path=repo/'analysis/03_v4_evaluation_analysis.ipynb'
        if not nb_path.exists():self.skipTest('historical notebook unavailable')
        nb=json.loads(nb_path.read_text(encoding='utf-8'))
        source=next(''.join(c['source']) for c in nb['cells'] if 'def evaluate_v4_json(' in ''.join(c.get('source',[])))
        # Execute only three inspected pure function definitions, not notebook setup or outputs.
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in {'nfc','iou_xyxy','evaluate_v4_json'}]
        self.assertEqual(len(nodes),3)
        ns={'unicodedata':unicodedata}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(nb_path),'exec'),ns)
        paths=list((repo/'eval').glob('*results.json'))
        self.assertEqual(len(paths),4)
        for p in paths:
            with self.subTest(file=p.name):
                data=json.loads(p.read_text(encoding='utf-8-sig'))
                expected=ns['evaluate_v4_json'](data)
                actual=bf.score_evaluation(data)
                for key,value in expected.items():self.assertEqual(actual[key],value)


if __name__=='__main__':unittest.main(verbosity=2)
