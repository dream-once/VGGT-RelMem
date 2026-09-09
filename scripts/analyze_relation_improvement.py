"""Read-only direction and multi-view error diagnostics for a verified relation run."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from relground.clio_relation_improvement import SCENES, aggregate, read, sha256, write
from relground.relations import RelationGrounder


def directions(queries, evaluated):
    lookup = {q['query_id']: q['relation'] for q in queries['queries']}
    return {relation: aggregate([r for r in evaluated['rows'] if lookup[r['query_id']] == relation])
            for relation in ('left_of', 'right_of', 'front_of', 'behind')}


def observation_votes(predictions, queries, evaluated, candidates, anchors, margin):
    """All observation-pair signs; correlated views are not independent votes."""
    by_id = {c['object_id']: c for c in candidates}
    rows = []
    for pred, query, scored in zip(predictions, queries['queries'], evaluated['rows'], strict=True):
        if not pred['ranked_ids'] or not pred.get('explanation', {}).get('reference_id'):
            continue
        target, reference = by_id[pred['ranked_ids'][0]], by_id[pred['explanation']['reference_id']]
        rotation = np.asarray(anchors[query['anchor_frame']])[:3, :3]
        distances = [RelationGrounder._signed_distance(query['relation'], rotation.T @ (
            np.asarray(t['center']) - np.asarray(r['center'])))
            for t in target['observations'] for r in reference['observations']]
        rows.append({
            'query_id': query['query_id'], 'relation': query['relation'],
            'answerable': scored['answerable'], 'answered': scored['answered'],
            'correct_strict': scored['correct_strict'], 'correct_padded': scored['correct_padded'],
            'target_id': target['object_id'], 'reference_id': reference['object_id'],
            'target_strict': scored['target_strict'], 'reference_strict': scored['reference_strict'],
            'observation_pairs': len(distances),
            'compatible_fraction': sum(d > margin for d in distances) / len(distances),
            'opposite_fraction': sum(d < -margin for d in distances) / len(distances),
            'cross_view_direction_disagreement': min(distances) < -margin and max(distances) > margin,
            'min_signed_distance': min(distances), 'max_signed_distance': max(distances),
        })
    def summarize(subset):
        return {
            'queries': len(subset),
            'direction_disagreement': sum(r['cross_view_direction_disagreement'] for r in subset),
            'all_pairs_compatible': sum(r['compatible_fraction'] == 1 for r in subset),
            'mean_compatible_fraction': float(np.mean([r['compatible_fraction'] for r in subset])) if subset else None,
        }
    positives = [r for r in rows if r['answerable']]
    return {'scope': 'diagnostic only, unweighted Cartesian observation pairs are correlated; no new threshold or learned calibration',
            'positive_selected_pairs': summarize(positives),
            'answered_strict_correct': summarize([r for r in rows if r['correct_strict']]),
            'answered_strict_wrong': summarize([r for r in rows if r['answered'] and not r['correct_strict']]),
            'rows': rows}


def analyze(root, run):
    summary = read(run / 'summary.json')
    validation = read(run / 'validation_report.json')
    if validation['status'] != 'PASS' or validation['summary_sha256'] != sha256(run / 'summary.json'):
        raise ValueError('run must pass source replay before generating diagnostics')
    result = {'scope': 'post-hoc explanatory diagnostics; does not modify predictions or labels',
              'summary_sha256': sha256(run / 'summary.json'), 'scenes': {}}
    for scene, directory in SCENES.items():
        path, bundle = run / scene, root / 'runs' / directory / 'relation-benchmark-v2'
        main = read(path / 'evaluation.json')['methods']
        queries = read(bundle / 'queries.json')
        supplement = read(path / 'disambiguation/evaluation.json')
        supplement_queries = read(path / 'disambiguation/queries.json')
        pools = read(path / 'candidates.json')['pools']
        predictions = read(path / 'prediction.json')['methods']
        anchors = read(bundle / 'anchor_poses.json')
        names = ['top1_pca__filter_with_entity_guard', 'a2_fused__filter_with_entity_guard',
                 'a2_medoid__filter_with_entity_guard', 'a2_pe__filter_with_entity_guard']
        final = main['a2_pe__filter_with_entity_guard']['rows']
        selected_positive = [r for r in final if r['answerable'] and r['target_id'] and r['reference_id']]
        roles = Counter(('both_correct' if r['pair_strict'] else 'both_wrong' if not r['target_strict'] and not r['reference_strict']
                         else 'target_wrong' if not r['target_strict'] else 'reference_wrong') for r in selected_positive)
        result['scenes'][scene] = {
            'main_by_direction': {name: directions(queries, main[name]) for name in names},
            'supplement_by_direction': {name: directions(supplement_queries, supplement[name])
                                       for name in ['a2_fused__check_after_selection', 'a2_fused__filter_before_selection']},
            'final_positive_selected_role_diagnosis': {'queries': len(selected_positive), 'counts': dict(roles)},
            'view_consistency': {pool: observation_votes(predictions[pool + '__filter_with_entity_guard'], queries,
                                                        main[pool + '__filter_with_entity_guard'], pools[pool], anchors,
                                                        read(run / 'protocol.json')['reasoning']['relation_margin_reconstruction_units'])
                                 for pool in ('a2_fused', 'a2_pe')},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/clio-relation-improvement-v2'))
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = analyze(root, args.run.resolve())
    write(args.run / 'diagnostics.json', result)
    if args.evidence:
        write(args.evidence / 'diagnostics.json', result)
    for scene, info in result['scenes'].items():
        print(scene, 'selected positive role errors', info['final_positive_selected_role_diagnosis'])
        for name, groups in info['main_by_direction'].items():
            print(name, {k: (m['positive_correct_strict'], m['positive_count'], m['negative_false_answers']) for k, m in groups.items()})
        print('view consistency', {k: {name: value for name, value in v.items() if name != 'rows'} for k, v in info['view_consistency'].items()})


if __name__ == '__main__':
    main()
