import unittest
from copy import deepcopy

import numpy as np

from relground.relation_reasoning import PairRelationGrounder, PairReasoningConfig
from relground.schemas import GroundingQuery
from relground.clio_relation_improvement import evaluate_method, aggregate


def candidate(name, cls, x, *, quality=.8, extent=1):
    return {'object_id':name,'class_text':cls,'quality':quality,'center':[x,0,0],
            'obb':{'center':[x,0,0],'extent':[extent,extent,extent],'rotation':np.eye(3).tolist()},
            'frame_ids':['f0']}


def query(relation='left_of'):
    return GroundingQuery('q','book',relation,'desk','anchor')


class PairReasoningTests(unittest.TestCase):
    def setUp(self):
        self.anchors={'anchor':np.eye(4)}
        self.candidates=[candidate('wrong','book',2,quality=.95),candidate('right','book',-2,quality=.7),candidate('reference','desk',0,quality=.9)]

    def test_relation_selects_lower_score_target_instead_of_rejecting_top_one(self):
        model=PairRelationGrounder(self.candidates,self.anchors)
        old=model.ground(query(),'check_after_selection')
        new=model.ground(query(),'filter_with_entity_guard')
        self.assertTrue(old.abstain)
        self.assertEqual(old.ranked_ids[0],'wrong')
        self.assertFalse(new.abstain)
        self.assertEqual(new.ranked_ids[0],'right')

    def test_reference_identity_does_not_change_with_requested_direction(self):
        objects=[*self.candidates,candidate('second_reference','desk',5,quality=.6)]
        model=PairRelationGrounder(objects,self.anchors)
        left=model.ground(query('left_of'))
        right=model.ground(query('right_of'))
        self.assertEqual(left.explanation['reference_id'],'reference')
        self.assertEqual(right.explanation['reference_id'],'reference')

    def test_absence_conflict_and_weak_semantic_reference_are_distinct(self):
        model=PairRelationGrounder(self.candidates[:-1],self.anchors)
        self.assertEqual(model.ground(query()).reason,'reference_not_found')
        model=PairRelationGrounder(self.candidates[1:],self.anchors)
        self.assertEqual(model.ground(query('right_of')).reason,'relation_conflict_or_boundary')
        objects=[candidate('book','book',-2),candidate('bottle','red bottle',0)]
        request=GroundingQuery('q','book','left_of','silver water bottle','anchor')
        result=PairRelationGrounder(objects,self.anchors).ground(request)
        self.assertEqual(result.reason,'insufficient_entity_evidence')

    def test_candidate_order_invariance_and_no_input_mutation(self):
        before=deepcopy(self.candidates)
        first=PairRelationGrounder(self.candidates,self.anchors).ground(query()).to_dict()
        second=PairRelationGrounder(list(reversed(self.candidates)),self.anchors).ground(query()).to_dict()
        self.assertEqual(first,second)
        self.assertEqual(before,self.candidates)

    def test_anchor_rotation_changes_direction(self):
        rotated=np.eye(4);rotated[:3,:3]=np.diag([-1.,1.,-1.])
        result=PairRelationGrounder(self.candidates,{'anchor':rotated}).ground(query())
        self.assertEqual(result.ranked_ids[0],'wrong')

    def test_ambiguous_gap_ablation_rejects_but_final_policy_preserves_valid_choices(self):
        objects=[candidate('a','book',-2,quality=.8),candidate('b','book',-4,quality=.79),candidate('r','desk',0)]
        model=PairRelationGrounder(objects,self.anchors)
        self.assertEqual(model.ground(query(),'filter_with_abstention').reason,'ambiguous_target')
        self.assertFalse(model.ground(query(),'filter_with_entity_guard').abstain)

    def test_equivalence_groups_do_not_chain_through_bridge(self):
        objects=[candidate('a','book',0,extent=10),candidate('b','book',.1,extent=10),candidate('c','book',.2,extent=10)]
        model=PairRelationGrounder(objects,self.anchors)
        groups=model._groups(model._matching('book'))
        self.assertEqual([len(g) for g in groups],[2,1])

    def test_coarse_alias_is_explicit_and_shared_with_relation_free_baseline(self):
        objects=deepcopy(self.candidates)
        for item in objects[:2]:
            item['class_text']='textbooks';item['query_aliases']=['book']
        model=PairRelationGrounder(objects,self.anchors)
        self.assertEqual(model.ground(query(),'no_relation').ranked_ids[0],'wrong')
        self.assertEqual(model.ground(query(),'filter_before_selection').ranked_ids[0],'right')

    def test_invalid_scores_and_poses_fail_explicitly(self):
        with self.assertRaises(ValueError): PairReasoningConfig(minimum_entity_score=float('nan'))
        objects=deepcopy(self.candidates);objects[0]['center'][0]=float('nan')
        with self.assertRaises(ValueError): PairRelationGrounder(objects,self.anchors)
        bad=np.eye(4);bad[0,0]=-1
        self.assertEqual(PairRelationGrounder(self.candidates,{'anchor':bad}).ground(query()).reason,'invalid_anchor_pose')


class RelationImprovementEvaluationTests(unittest.TestCase):
    def test_both_roles_and_actual_current_centers_are_required(self):
        objects=[candidate('target__1','book',-2),candidate('reference__1','desk',0)]
        result=PairRelationGrounder(objects,{'anchor':np.eye(4)}).ground(query()).to_dict()
        labels=[{'query_id':'q','answerable':True,'target_task':'target','reference_task':'reference'}]
        gt={name:[{'center':np.array([x,0,0]),'extent':np.ones(3),'rotation':np.eye(3)}] for name,x in [('target',-2),('reference',0)]}
        alignment={'sim3':{'scale':1,'rotation':np.eye(3).tolist(),'translation':[0,0,0]},'error_m':{'rmse':.1}}
        self.assertEqual(evaluate_method([result],objects,labels,gt,alignment)['metrics']['positive_correct_strict'],1)
        moved=deepcopy(objects);moved[1]['center']=[8,0,0]
        self.assertEqual(evaluate_method([result],moved,labels,gt,alignment)['metrics']['positive_correct_strict'],0)
        # GT can alter scoring, never the already fixed prediction.
        self.assertFalse(result['abstain'])
        with self.assertRaises(ValueError):evaluate_method([],objects,labels,gt,alignment)

    def test_zero_answer_risk_and_unbalanced_denominators(self):
        rows=[]
        for positive in [True,False,False,False]:
            row={'answerable':positive,'answered':False}
            for mode in ('strict','padded'):
                row['correct_'+mode]=False;row['grounded_relation_rejection_'+mode]=False
            rows.append(row)
        metrics=aggregate(rows)
        self.assertIsNone(metrics['answer_error_rate_strict'])
        self.assertEqual(metrics['task_accuracy_strict'],.75)
        self.assertEqual(metrics['balanced_task_accuracy_strict'],.5)
        self.assertEqual(metrics['positive_false_rejections'],1)


if __name__=='__main__':
    unittest.main()
