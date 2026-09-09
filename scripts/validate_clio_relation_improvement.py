"""Rebuild relation candidates, replay predictions, and verify saved experiment evidence.

Requires local Clio caches, but no GPU or model forward pass. This checks a saved
run against its sources and current implementation; it is not an independent
implementation of the full vision pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from relground.clio_relation_improvement import (
    POOLS, SCENES, _parse_gt_boxes, alias_candidates, build_candidates,
    build_disambiguation_queries, evaluate_disambiguation, evaluate_method,
    read, sha256, write,
)
from relground.relation_reasoning import PairReasoningConfig, PairRelationGrounder
from relground.schemas import GroundingQuery


def require_equal(actual, expected, name):
    if actual != expected:
        raise ValueError(f'{name} differs from source replay')


def check_counts(evaluated):
    """Count key outcomes directly from per-query rows, separately from aggregate()."""
    rows, metrics = evaluated['rows'], evaluated['metrics']
    positive = [r for r in rows if r['answerable']]
    negative = [r for r in rows if not r['answerable']]
    counts = {
        'queries': len(rows), 'positive_count': len(positive), 'negative_count': len(negative),
        'answered': sum(r['answered'] for r in rows),
        'negative_false_answers': sum(r['answered'] for r in negative),
        'positive_false_rejections': sum(not r['answered'] for r in positive),
    }
    for mode in ('strict', 'padded'):
        counts['positive_correct_' + mode] = sum(
            r['answered'] and r['target_' + mode] and r['reference_' + mode] for r in positive
        )
    for key, value in counts.items():
        require_equal(metrics[key], value, 'direct row count: ' + key)


def replay(queries, pools, anchors, config, modes):
    result = {}
    for pool in POOLS:
        grounder = PairRelationGrounder(pools[pool], anchors, config)
        for mode in modes:
            result[pool + '__' + mode] = [
                grounder.ground(GroundingQuery.from_dict(q), mode).to_dict() for q in queries['queries']
            ]
    return result


def comparison(evaluations, before, after):
    first, second = evaluations[before]['rows'], evaluations[after]['rows']
    require_equal([r['query_id'] for r in first], [r['query_id'] for r in second], 'comparison query IDs')
    wins = [b for a, b in zip(first, second) if b['correct_strict'] and not a['correct_strict']]
    losses = [a for a, b in zip(first, second) if a['correct_strict'] and not b['correct_strict']]
    return {
        'before': before, 'after': after,
        'strict_wins': [r['query_id'] for r in wins], 'strict_losses': [r['query_id'] for r in losses],
        'win_target_tasks': dict(Counter(r['target_task'] for r in wins)),
        'win_reference_tasks': dict(Counter(r['reference_task'] for r in wins)),
    }


def reason_counts(evaluation):
    return {
        polarity: dict(Counter(r['reason'] for r in evaluation['rows'] if r['answerable'] == answerable and not r['answered']))
        for polarity, answerable in [('positive_rejections', True), ('negative_rejections', False)]
    }


def validate(root, output, protocol_path):
    summary, protocol = read(output / 'summary.json'), read(protocol_path)
    require_equal(read(output / 'protocol.json'), protocol, 'protocol snapshot')
    require_equal(summary['protocol_sha256'], sha256(protocol_path), 'protocol hash')
    for name, digest in summary['code_sha256'].items():
        require_equal(sha256(root / name), digest, 'implementation hash: ' + name)
    cfg = protocol['reasoning']
    config = PairReasoningConfig(
        relation_margin=cfg['relation_margin_reconstruction_units'],
        minimum_entity_score=cfg['minimum_entity_score'],
        ambiguity_score_gap=cfg['ambiguity_score_gap'],
        equivalence_center_distance=cfg['equivalence_center_distance_reconstruction_units'],
        equivalence_min_aabb_iou=cfg['equivalence_min_aabb_iou'],
    )
    report = {'status': 'PASS', 'candidate_rebuild': True, 'gpu_required': False,
              'scope': 'local source replay, upstream PCA numerical cross-check, query authoring replay, per-query score replay and direct recount',
              'summary_sha256': sha256(output / 'summary.json'), 'scenes': {}}
    evidence = {'experiment_id': summary['experiment_id'], 'scope': summary['scope'],
                'protocol': protocol, 'protocol_sha256': summary['protocol_sha256'],
                'code_sha256': summary['code_sha256'], 'scenes': {}}
    method = 'filter_with_entity_guard'
    for scene, directory in SCENES.items():
        path, bundle = output / scene, root / 'runs' / directory / 'relation-benchmark-v2'
        inputs = read(path / 'candidates.json')
        require_equal(inputs, build_candidates(root, scene), scene + ' rebuilt candidates')
        anchors, queries = read(bundle / 'anchor_poses.json'), read(bundle / 'queries.json')
        saved_prediction = read(path / 'prediction.json')
        for key, source in [('query', bundle / 'queries.json'), ('anchor', bundle / 'anchor_poses.json'), ('candidates', path / 'candidates.json')]:
            require_equal(saved_prediction[key + '_sha256'], sha256(source), scene + ' ' + key + ' hash')
        require_equal(saved_prediction['config'], asdict(config), scene + ' reasoning config')
        predictions = replay(queries, inputs['pools'], anchors, config, protocol['reasoning_ablations'])
        require_equal(saved_prediction['methods'], predictions, scene + ' label-free predictions')
        source = read(bundle / 'evaluation.json')['source']
        alignment = read(root / source['world_alignment'])
        labels = read(bundle / 'labels.json')['labels']
        tasks = {r[role + '_task'] for r in labels for role in ('target', 'reference')}
        gt = {task: _parse_gt_boxes(root / source['task_gt'], task) for task in tasks}
        saved_evaluation = read(path / 'evaluation.json')
        for key, file in [('prediction', path / 'prediction.json'), ('labels', bundle / 'labels.json'),
                          ('gt', root / source['task_gt']), ('alignment', root / source['world_alignment'])]:
            require_equal(saved_evaluation[key + '_sha256'], sha256(file), scene + ' ' + key + ' hash')
        evaluations = {key: evaluate_method(pred, inputs['pools'][key.split('__')[0]], labels, gt, alignment)
                       for key, pred in predictions.items()}
        require_equal(saved_evaluation['methods'], evaluations, scene + ' scoring')
        require_equal(summary['scenes'][scene]['metrics'], {k: e['metrics'] for k, e in evaluations.items()}, scene + ' main summary')
        require_equal(summary['scenes'][scene]['task_counts'], inputs['task_counts'], scene + ' task counts')
        for evaluated in evaluations.values():
            check_counts(evaluated)

        supplement = path / 'disambiguation'
        authored_queries, authored_labels = build_disambiguation_queries(
            scene, protocol, read(root / f'configs/clio_{scene}_queries.json'), root / source['task_gt'], alignment, anchors)
        require_equal(read(supplement / 'queries.json'), authored_queries, scene + ' GT-only query authoring')
        require_equal(read(supplement / 'labels.json'), authored_labels, scene + ' GT-only label authoring')
        pools = {key: alias_candidates(value, protocol['supplement']['families'][scene]) for key, value in inputs['pools'].items()}
        dis_predictions = replay(authored_queries, pools, anchors, config, protocol['reasoning_ablations'])
        require_equal(read(supplement / 'prediction.json')['methods'], dis_predictions, scene + ' supplement predictions')
        require_equal(read(supplement / 'prediction.json')['query_sha256'], sha256(supplement / 'queries.json'), scene + ' supplement query hash')
        dis_evaluations = {key: evaluate_disambiguation(pred, pools[key.split('__')[0]], authored_labels['labels'], gt, alignment)
                           for key, pred in dis_predictions.items()}
        require_equal(read(supplement / 'evaluation.json'), dis_evaluations, scene + ' supplement scoring')
        expected_dis_summary = {'query_count': len(authored_queries['queries']), 'skipped': authored_labels['skipped'],
                                'metrics': {k: e['metrics'] for k, e in dis_evaluations.items()},
                                'by_family': {k: e['by_family'] for k, e in dis_evaluations.items()}}
        require_equal(summary['scenes'][scene]['disambiguation'], expected_dis_summary, scene + ' supplement summary')
        for evaluated in dis_evaluations.values():
            check_counts(evaluated)

        comparisons = {
            'top1_to_fused': comparison(evaluations, 'top1_pca__' + method, 'a2_fused__' + method),
            'fused_to_pe': comparison(evaluations, 'a2_fused__' + method, 'a2_pe__' + method),
            'relation_only': comparison(dis_evaluations, 'a2_fused__check_after_selection', 'a2_fused__filter_before_selection'),
        }
        case_ids = comparisons['relation_only']['strict_wins']
        cases = []
        for query in authored_queries['queries']:
            if query['query_id'] in case_ids:
                index = next(i for i, q in enumerate(authored_queries['queries']) if q['query_id'] == query['query_id'])
                cases.append({'query': query,
                              'before': dis_predictions['a2_fused__check_after_selection'][index],
                              'after': dis_predictions['a2_fused__filter_before_selection'][index],
                              'evaluation': dis_evaluations['a2_fused__filter_before_selection']['rows'][index]})
        artifacts = [path / name for name in ('candidates.json', 'prediction.json', 'evaluation.json')]
        artifacts += [supplement / name for name in ('queries.json', 'labels.json', 'prediction.json', 'evaluation.json')]
        evidence['scenes'][scene] = {
            **summary['scenes'][scene], 'comparisons': comparisons, 'relation_only_cases': cases,
            'main_final_rejection_reasons': reason_counts(evaluations['a2_pe__' + method]),
            'main_final_negative_false_answer_ids': [r['query_id'] for r in evaluations['a2_pe__' + method]['rows'] if not r['answerable'] and r['answered']],
            'supplement_fused_rejection_reasons': reason_counts(dis_evaluations['a2_fused__' + method]),
            'candidate_counts': {pool: len(rows) for pool, rows in inputs['pools'].items()},
            'source_sha256': inputs['sources'],
            'artifact_sha256': {str(f.relative_to(root)): sha256(f) for f in artifacts},
        }
        report['scenes'][scene] = {
            'upstream_pca_numeric_check': inputs['upstream_pca_numeric_check'],
            'methods_per_suite': len(predictions), 'main_queries': len(labels),
            'supplement_queries': len(authored_queries['queries']),
            'predictions_replayed': len(predictions) * (len(labels) + len(authored_queries['queries'])),
            'source_files_checked': len(inputs['sources']),
        }
        print(f'{scene}: candidate rebuild, prediction replay, GT scoring and counts PASS', flush=True)
    return report, evidence



def validate_summary(directory):
    """Portable consistency checks; cannot establish source correctness without data."""
    data = read(directory / 'benchmark_summary.json')
    validation = read(directory / 'validation_report.json')
    require_equal(validation['status'], 'PASS', 'saved source validation status')
    require_equal(validation['evidence_sha256'], sha256(directory / 'benchmark_summary.json'), 'evidence content hash')
    require_equal(data['protocol']['upstream_baseline']['native_upstream_relation_claim'], False, 'baseline claim boundary')
    require_equal(data['protocol']['calibration_fitted'], False, 'calibration claim boundary')
    require_equal(set(data['scenes']), set(SCENES), 'scene coverage')
    expected_methods = {pool + '__' + mode for pool in POOLS for mode in data['protocol']['reasoning_ablations']}
    for scene, entry in data['scenes'].items():
        for suite in (entry['metrics'], entry['disambiguation']['metrics']):
            require_equal(set(suite), expected_methods, scene + ' complete ablation matrix')
            denominators = {(m['positive_count'], m['negative_count']) for m in suite.values()}
            require_equal(len(denominators), 1, scene + ' common query denominator')
            for name, m in suite.items():
                p, n, answered = m['positive_count'], m['negative_count'], m['answered']
                require_equal(m['queries'], p + n, name + ' query count')
                require_equal(answered, p - m['positive_false_rejections'] + m['negative_false_answers'], name + ' answer count')
                require_equal(m['answer_coverage'], answered / (p + n), name + ' coverage')
                require_equal(m['negative_rejection_rate'], (n - m['negative_false_answers']) / n, name + ' negative rejection')
                for mode in ('strict', 'padded'):
                    correct = m['positive_correct_' + mode]
                    if not 0 <= correct <= p - m['positive_false_rejections']:
                        raise ValueError(name + ' invalid correct count')
                    require_equal(m['positive_accuracy_' + mode], correct / p if p else None, name + ' positive accuracy')
                    require_equal(m['answer_error_rate_' + mode], (answered - correct) / answered if answered else None, name + ' answer risk')
        for name, comp in entry['comparisons'].items():
            suite = entry['disambiguation']['metrics'] if name == 'relation_only' else entry['metrics']
            difference = suite[comp['after']]['positive_correct_strict'] - suite[comp['before']]['positive_correct_strict']
            require_equal(difference, len(comp['strict_wins']) - len(comp['strict_losses']), scene + ' paired improvement')
        for method, families in entry['disambiguation']['by_family'].items():
            for key in ('positive_count', 'negative_count', 'positive_correct_strict', 'positive_correct_padded', 'answered', 'negative_false_answers'):
                require_equal(sum(m[key] for m in families.values()), entry['disambiguation']['metrics'][method][key], scene + ' family subtotal: ' + key)
    return {'status': 'PASS', 'scope': 'portable evidence integrity and arithmetic; no raw-data replay'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/clio-relation-improvement-v2'))
    parser.add_argument('--protocol', type=Path, default=Path('configs/clio_relation_improvement_v2.json'))
    parser.add_argument('--evidence', type=Path, help='Optional output directory for a compact evidence copy')
    parser.add_argument('--summary-only', type=Path, help='Validate portable evidence without Clio data')
    args = parser.parse_args()
    if args.summary_only:
        print(validate_summary(args.summary_only))
        return
    root = Path(__file__).resolve().parents[1]
    report, evidence = validate(root, args.run.resolve(), args.protocol.resolve())
    write(args.run / 'validation_report.json', report)
    if args.evidence:
        write(args.evidence / 'benchmark_summary.json', evidence)
        report['evidence_sha256'] = sha256(args.evidence / 'benchmark_summary.json')
        write(args.evidence / 'validation_report.json', report)
    print('PASS: all saved results reproduced from local sources')


if __name__ == '__main__':
    main()
