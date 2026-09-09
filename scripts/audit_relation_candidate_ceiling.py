"""Evaluator-only oracle ceiling for fixed relation candidate centers.

GT may select candidates here only to measure an upper bound. This script never
changes the grounder's inputs, predictions, or headline evaluation results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from relground.clio_relation_improvement import SCENES, _parse_gt_boxes, d6_path, read, sha256, write
from relground.clio_retrieval_evaluation import slugify_task
from relground.clio_task_evaluation import point_in_obb
from relground.relations import RelationGrounder


def audit(root, run):
    result = {'scope': 'GT-assisted diagnostic upper bound over fixed existing centers; not inference performance or an architectural limit',
              'gt_read': True, 'predictions_modified': False, 'strict_primary': True,
              'sources': {}, 'scenes': {}}
    def recorded(path):
        result['sources'][str(path.relative_to(root))] = sha256(path)
        return read(path)
    recorded(run / 'protocol.json')
    margin = read(run / 'protocol.json')['reasoning']['relation_margin_reconstruction_units']
    for scene, directory in SCENES.items():
        original = root / 'runs' / directory / 'relation-benchmark-v2'
        candidates = recorded(run / scene / 'candidates.json')['pools']
        labels = [l for l in recorded(original / 'labels.json')['labels'] if l['answerable']]
        queries = {q['query_id']: q for q in recorded(original / 'queries.json')['queries']}
        source = recorded(original / 'evaluation.json')['source']
        alignment = recorded(root / source['world_alignment'])
        result['sources'][source['task_gt']] = sha256(root / source['task_gt'])
        anchors = recorded(original / 'anchor_poses.json')
        scale = alignment['sim3']['scale']
        rotation = np.asarray(alignment['sim3']['rotation'])
        translation = np.asarray(alignment['sim3']['translation'])
        tasks = sorted({l[role + '_task'] for l in labels for role in ('target', 'reference')})
        gt = {task: _parse_gt_boxes(root / source['task_gt'], task) for task in tasks}
        candidates['all_observations_in_a2_objects'] = [
            {'object_id': c['object_id'] + '__' + obs['obs_id'], 'center': obs['center']}
            for c in candidates['a2_fused'] for obs in c['observations']]
        candidates['a2_pe_union_top1_pca'] = candidates['a2_pe'] + candidates['top1_pca']
        candidates['all_top5_robust_observations'] = []
        for task in tasks:
            slug = slugify_task(task)
            observations = recorded(d6_path(root / 'runs' / directory, scene, slug) / 'observations.json')['observations']
            candidates['all_top5_robust_observations'].extend(
                {'object_id': slug + '__' + o['obs_id'], 'center': o['center']} for o in observations)
        pools = {}
        for name, rows in candidates.items():
            good = {task: [c for c in rows if c['object_id'].startswith(slugify_task(task) + '__') and any(
                point_in_obb(scale * (rotation @ np.asarray(c['center'])) + translation,
                             center=b['center'], extent=b['extent'], rotation=b['rotation']) for b in gt[task])]
                    for task in tasks}
            possible, compatible = [], []
            for label in labels:
                targets, references = good[label['target_task']], good[label['reference_task']]
                if not targets or not references:
                    continue
                possible.append(label['query_id'])
                query = queries[label['query_id']]
                anchor_rotation = np.asarray(anchors[query['anchor_frame']])[:3, :3]
                if any(RelationGrounder._signed_distance(query['relation'], anchor_rotation.T @ (
                        np.asarray(t['center']) - np.asarray(r['center']))) > margin
                       for t in targets for r in references):
                    compatible.append(label['query_id'])
            pools[name] = {'localized_tasks': [task for task in tasks if good[task]],
                           'positive_count': len(labels), 'strict_pair_oracle': len(possible),
                           'strict_pair_direction_oracle': len(compatible),
                           'strict_pair_query_ids': possible, 'strict_pair_direction_query_ids': compatible}
        result['scenes'][scene] = pools
    result['sources']['scripts/audit_relation_candidate_ceiling.py'] = sha256(root / 'scripts/audit_relation_candidate_ceiling.py')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/clio-relation-improvement-v2'))
    parser.add_argument('--output', type=Path, default=Path('runs/clio-relation-improvement-v2/candidate_ceiling.json'))
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.run.resolve())
    write(args.output, result)
    for scene, pools in result['scenes'].items():
        print(scene, {name: value['strict_pair_direction_oracle'] for name, value in pools.items()})


if __name__ == '__main__':
    main()
