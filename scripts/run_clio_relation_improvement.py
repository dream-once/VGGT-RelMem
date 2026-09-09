"""Run controlled Top-1 vs multi-view relation grounding and rejection experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

from relground.clio_relation_improvement import SCENES, POOLS, read, write, sha256, build_candidates, evaluate_method, _parse_gt_boxes, build_disambiguation_queries, alias_candidates, evaluate_disambiguation
from relground.relation_reasoning import PairReasoningConfig, PairRelationGrounder
from relground.schemas import GroundingQuery


def run(root, output, reuse_candidates=False, protocol_file="configs/clio_relation_improvement_v2.json"):
    protocol_path = root/protocol_file
    protocol = read(protocol_path)
    cfg = protocol['reasoning']
    config = PairReasoningConfig(cfg['relation_margin_reconstruction_units'], cfg['minimum_entity_score'], cfg['ambiguity_score_gap'], cfg['equivalence_center_distance_reconstruction_units'], cfg['equivalence_min_aabb_iou'])
    output.mkdir(parents=True,exist_ok=True)
    write(output/'protocol.json',protocol)
    report = {'experiment_id': protocol['experiment_id'], 'scope': protocol['scope'], 'protocol_sha256': sha256(protocol_path),
              'upstream_native_relation_claim': False, 'calibration_fitted': False,
              'code_sha256': {name:sha256(root/name) for name in ('relground/relation_reasoning.py','relground/clio_relation_improvement.py','scripts/run_clio_relation_improvement.py')}, 'scenes':{}}
    # Predict both scenes before running any of the evaluators.
    pending = {}
    for scene, directory in SCENES.items():
        destination = output/scene
        destination.mkdir(parents=True, exist_ok=True)
        candidates_path = destination/'candidates.json'
        start = time.perf_counter()
        if reuse_candidates:
            inputs = read(candidates_path)
            for name, digest in inputs['sources'].items():
                if sha256(root/name) != digest: raise ValueError(f'candidate source changed: {name}')
        else:
            inputs = build_candidates(root, scene)
            write(candidates_path, inputs)
        bundle = root/'runs'/directory/'relation-benchmark-v2'
        queries = read(bundle/'queries.json')
        anchors = read(bundle/'anchor_poses.json')
        predictions = {}
        for pool in POOLS:
            grounder = PairRelationGrounder(inputs['pools'][pool], anchors, config)
            for mode in protocol['reasoning_ablations']:
                key = pool+'__'+mode
                predictions[key] = [grounder.ground(GroundingQuery.from_dict(q), mode).to_dict() for q in queries['queries']]
        write(destination/'prediction.json', {'scene_id':scene,'config':asdict(config),'gt_read':False,'query_sha256':sha256(bundle/'queries.json'), 'anchor_sha256':sha256(bundle/'anchor_poses.json'),'candidates_sha256':sha256(candidates_path),'methods':predictions})
        pending[scene] = (inputs, predictions, bundle)
        print(f'{scene}: {len(queries["queries"])} queries, {len(predictions)} methods, candidate and prediction materialization {time.perf_counter()-start:.2f}s', flush=True)
    for scene, (inputs,predictions,bundle) in pending.items():
        labels=read(bundle/'labels.json')['labels']
        sources=read(bundle/'evaluation.json')['source']
        alignment=read(root/sources['world_alignment'])
        tasks=sorted({row[role+'_task'] for row in labels for role in ('target','reference')})
        gt={task:_parse_gt_boxes(root/sources['task_gt'],task) for task in tasks}
        evaluated={}
        for key, prediction in predictions.items():
            pool=key.split('__')[0]
            evaluated[key]=evaluate_method(prediction,inputs['pools'][pool],labels,gt,alignment)
        write(output/scene/'evaluation.json', {'scope':report['scope'], 'scene_id':scene,'prediction_sha256':sha256(output/scene/'prediction.json'),
             'labels_sha256':sha256(bundle/'labels.json'),'gt_sha256':sha256(root/sources['task_gt']),'alignment_sha256':sha256(root/sources['world_alignment']),'methods':evaluated})
        report['scenes'][scene]={'metrics':{key:value['metrics'] for key,value in evaluated.items()},'task_counts':inputs['task_counts']}
        for key,result in evaluated.items():
            m=result['metrics']
            print(scene,key,'strict/padded',m['positive_correct_strict'],m['positive_correct_padded'],'answers',m['answered'],'negative false',m['negative_false_answers'],flush=True)
    for scene, (inputs,_,bundle) in pending.items():
        sources=read(bundle/'evaluation.json')['source']
        alignment=read(root/sources['world_alignment'])
        anchors=read(bundle/'anchor_poses.json')
        manifest=read(root/f'configs/clio_{scene}_queries.json')
        queries,labels=build_disambiguation_queries(scene,protocol,manifest,root/sources['task_gt'],alignment,anchors)
        destination=output/scene/'disambiguation'
        write(destination/'queries.json',queries)
        write(destination/'labels.json',labels)
        pools={key:alias_candidates(value,protocol['supplement']['families'][scene]) for key,value in inputs['pools'].items()}
        methods={}
        for pool in POOLS:
            grounder=PairRelationGrounder(pools[pool],anchors,config)
            for mode in protocol['reasoning_ablations']:
                methods[pool+'__'+mode]=[grounder.ground(GroundingQuery.from_dict(q),mode).to_dict() for q in queries['queries']]
        write(destination/'prediction.json',{'gt_read':False,'query_sha256':sha256(destination/'queries.json'),'methods':methods})
        tasks=sorted({t for label in labels['labels'] for t in [label['reference_task'],*label['family_tasks']]})
        gt={task:_parse_gt_boxes(root/sources['task_gt'],task) for task in tasks}
        evaluations={key:evaluate_disambiguation(pred,pools[key.split('__')[0]],labels['labels'],gt,alignment) for key,pred in methods.items()}
        write(destination/'evaluation.json',evaluations)
        report['scenes'][scene]['disambiguation']={'query_count':len(queries['queries']), 'skipped':labels['skipped'], 'metrics':{key:e['metrics'] for key,e in evaluations.items()},'by_family':{key:e['by_family'] for key,e in evaluations.items()}}
        for key,evaluation in evaluations.items():
            m=evaluation['metrics']
            print('DISAMBIG',scene,key,'strict/padded',m['positive_correct_strict'],m['positive_correct_padded'],'positive count',m['positive_count'],'answers',m['answered'],'negative false',m['negative_false_answers'],flush=True)
    write(output/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/clio-relation-improvement-v2'))
    parser.add_argument('--reuse-candidates',action='store_true')
    parser.add_argument('--protocol',default='configs/clio_relation_improvement_v2.json')
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.output.resolve(),args.reuse_candidates,args.protocol)
