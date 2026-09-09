"""Controlled Clio relation experiment inputs and evaluator.

Candidate construction is label-free. Evaluation reads the fixed query labels,
GT boxes and scene Sim(3) only after all predictions have been saved.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.geometry import load_geometry_npz
from relground.clio_retrieval_evaluation import slugify_task
from relground.clio_task_evaluation import _parse_gt_boxes, point_in_obb
from relground.schemas import ObjectObservation, observation_quality
from relground.single_view import official_pca_lift

SCENES = {'apartment': 'clio-apartment-dev-v2-lc', 'cubicle': 'clio-cubicle-heldout-v1'}
POOLS = ('top1_pca', 'top1_robust', 'a2_fused', 'a2_quality', 'a2_medoid', 'a2_pe')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def d6_path(root, scene, slug):
    prefix = 'dev-' if scene == 'apartment' and slug != 'bring-me-a-pillow' else ''
    return root / f'{prefix}d6-{slug}-k5'


def upstream_pca_function(project):
    path = project / 'third_party/VGGT-SLAM/vggt_slam/slam_utils.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_obb_from_points')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compute_obb_from_points']


def _observation_candidate(observation, *, slug, object_id, source):
    obs = ObjectObservation.from_dict(observation)
    return {'object_id': slug + '__' + object_id, 'class_text': obs.class_text,
            'center': obs.center.tolist(), 'obb': obs.obb.to_dict(),
            'quality': observation_quality(obs), 'frame_ids': [obs.frame_id],
            'observations': [observation], 'source': source}


def build_candidates(project: Path, scene: str):
    root = project / 'runs' / SCENES[scene]
    manifest_path = project / f'configs/clio_{scene}_queries.json'
    manifest = read(manifest_path)
    tasks = [row for row in manifest['queries'] if row['split'] == manifest['role']]
    pe_path = project / 'runs/clio-pe-semantic-fusion-v1' / scene / 'prediction.json'
    pe = {t['task']: t for t in read(pe_path)['tasks']}
    pools = {name: [] for name in POOLS}
    sources = {str(manifest_path.relative_to(project)): sha256(manifest_path), str(pe_path.relative_to(project)): sha256(pe_path)}
    def record(path):
        relative = str(path.resolve().relative_to(project))
        sources[relative] = sha256(path)
    record(project / 'relground/single_view.py')
    record(project / 'third_party/VGGT-SLAM/vggt_slam/slam_utils.py')
    geometry_path = root / 'geometry.npz'
    record(geometry_path)
    geometry = load_geometry_npz(geometry_path)
    upstream_pca = upstream_pca_function(project)
    counts = []
    for task in tasks:
        slug = slugify_task(task['task'])
        d6 = d6_path(root, scene, slug)
        for name in ('d6_result.json', 'masks.json', 'observations.json'):
            record(d6 / name)
        result, masks, observations = read(d6/'d6_result.json'), read(d6/'masks.json')['records'], read(d6/'observations.json')['observations']
        if result['mask_resizing_after_sam'] or float(result['sam_threshold']) != .5:
            raise ValueError('single-frame replay requires frozen same-grid SAM inputs')
        top1 = result['selected_frames'][0]['frame_id']
        retrieval_path = root / f'retrieval-{slug}' / 'retrieval.json'
        record(retrieval_path)
        if read(retrieval_path)['upstream_top1']['frame_id'] != top1:
            raise ValueError('D6 first frame differs from upstream PE Top-1')
        frame = geometry.get(top1)
        frame_masks = [m for m in masks if m['frame_id'] == top1]
        rejected = []
        for mask in frame_masks:
            path = d6 / mask['mask_ref']; record(path)
            array = np.load(path, allow_pickle=False).astype(bool)
            try:
                lifted = official_pca_lift(array, frame.point_map, frame.world_from_camera)
            except ValueError as error:
                rejected.append({'observation_id': mask['obs_id'], 'reason': str(error)})
                continue
            center, extent, rotation = upstream_pca(lifted.points)
            if not np.allclose(center, lifted.center, atol=1e-10, rtol=1e-10) or not np.allclose(extent, lifted.obb.extent, atol=1e-10, rtol=1e-10):
                raise ValueError('finite-only lifting disagrees with pinned upstream PCA function')
            obs = ObjectObservation(obs_id=mask['obs_id'], class_text=task['sam_query'], frame_id=top1,
                                    mask_ref=None, points_ref=None, retrieval_score=mask['retrieval_score'],
                                    sam_score=mask['sam_score'], valid_point_ratio=lifted.valid_point_ratio,
                                    center=lifted.center, obb=lifted.obb)
            pools['top1_pca'].append(_observation_candidate(obs.to_dict(), slug=slug, object_id='pca_'+obs.obs_id, source='upstream_top1_pca'))
        for obs in observations:
            if obs['frame_id'] == top1:
                pools['top1_robust'].append(_observation_candidate(obs, slug=slug, object_id='robust_'+obs['obs_id'], source='top1_robust'))
        memory_path = d6.parent / (d6.name.replace('d6-', 'a2-', 1)) / 'prediction/object_memory.json'
        if memory_path.is_file():
            record(memory_path)
            objects = read(memory_path)['objects']
        else:
            objects = []
        pe_rows = {o['observation_id']: o for o in pe[task['task']]['observations']}
        for obj in objects:
            obs_rows = obj['observations']
            # Reconcile PE cache against the current object observation input.
            if any(o['obs_id'] not in pe_rows or not np.allclose(o['center'], pe_rows[o['obs_id']]['center_vggt'], atol=1e-10) for o in obs_rows):
                raise ValueError('PE score cache differs from A2 source observations')
            qualities = {o['obs_id']: observation_quality(ObjectObservation.from_dict(o)) for o in obs_rows}
            quality = max(qualities.values())
            centers = np.asarray([o['center'] for o in obs_rows])
            distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2).sum(axis=1)
            representatives = {
                'a2_quality': min(obs_rows, key=lambda o: (-qualities[o['obs_id']], o['obs_id'])),
                'a2_pe': min(obs_rows, key=lambda o: (-pe_rows[o['obs_id']]['semantic_score'], o['obs_id'])),
                'a2_medoid': obs_rows[min(range(len(obs_rows)), key=lambda i: (distances[i], obs_rows[i]['obs_id']))],
            }
            base = {'object_id': slug+'__'+obj['object_id'], 'class_text': task['sam_query'],
                    'center': obj['fused_center'], 'obb': obj['fused_obb'], 'quality': quality,
                    'frame_ids': sorted({o['frame_id'] for o in obs_rows}), 'observations': obs_rows,
                    'source': 'a2_fused', 'representative_observation_id': None}
            pools['a2_fused'].append(deepcopy(base))
            for name, chosen in representatives.items():
                candidate = deepcopy(base)
                candidate.update(center=chosen['center'], source=name, representative_observation_id=chosen['obs_id'])
                pools[name].append(candidate)
        counts.append({'task': task['task'], 'top1_frame': top1, 'top1_sam_masks': len(frame_masks),
                       'top1_pca_invalid': rejected, 'top1_robust_observations': sum(o['frame_id']==top1 for o in observations),
                       'a2_objects': len(objects), 'sam_calls_top1': 1, 'sam_calls_top5': len(result['processed_frames'])})
    return {'scene_id': scene, 'pools': pools, 'sources': sources, 'task_counts': counts,
            'upstream_pca_numeric_check': 'PASS', 'gt_read': False}


def evaluate_method(predictions, candidates, labels, gt, alignment):
    by_id = {c['object_id']: c for c in candidates}
    scale = float(alignment['sim3']['scale'])
    rotation = np.asarray(alignment['sim3']['rotation']); translation = np.asarray(alignment['sim3']['translation'])
    padding = float(alignment['error_m']['rmse'])
    def matches(object_id, task, margin):
        if object_id not in by_id:
            return False
        candidate = by_id[object_id]
        # Preserve both spatial and semantic-role identity; no nearest-label relabelling.
        if not object_id.startswith(slugify_task(task)+'__'):
            return False
        center = scale * (rotation @ np.asarray(candidate['center'])) + translation
        return any(point_in_obb(center, center=b['center'], extent=b['extent'], rotation=b['rotation'], padding_m=margin) for b in gt[task])
    if len(predictions) != len(labels):
        raise ValueError('prediction/label length mismatch')
    rows = []
    for pred, label in zip(predictions, labels, strict=True):
        if pred['query_id'] != label['query_id']:
            raise ValueError('prediction/label order mismatch')
        target = pred['ranked_ids'][0] if pred['ranked_ids'] else None
        reference = (pred.get('explanation') or {}).get('reference_id')
        row = {'query_id': label['query_id'], 'target_task': label['target_task'], 'reference_task': label['reference_task'],
               'answerable': label['answerable'], 'answered': not pred['abstain'], 'reason': pred['reason'],
               'target_id': target, 'reference_id': reference, 'confidence': pred['confidence']}
        for mode, margin in (('strict', 0.0), ('padded', padding)):
            target_match, ref_match = matches(target, label['target_task'], margin), matches(reference, label['reference_task'], margin)
            row[f'target_{mode}'] = target_match
            row[f'reference_{mode}'] = ref_match
            row[f'pair_{mode}'] = target_match and ref_match
            row[f'correct_{mode}'] = bool(label['answerable'] and row['answered'] and target_match and ref_match)
            row[f'grounded_relation_rejection_{mode}'] = bool(not label['answerable'] and pred['abstain'] and pred['reason']=='relation_conflict_or_boundary' and target_match and ref_match)
        rows.append(row)
    return {'rows': rows, 'metrics': aggregate(rows)}


def aggregate(rows):
    def ratio(n, d): return n / d if d else None
    pos = [r for r in rows if r['answerable']]; neg = [r for r in rows if not r['answerable']]
    answered = sum(r['answered'] for r in rows)
    metrics = {'queries': len(rows), 'positive_count': len(pos), 'negative_count': len(neg), 'answered': answered,
               'answer_coverage': ratio(answered, len(rows)), 'positive_false_rejections': sum(not r['answered'] for r in pos),
               'negative_false_answers': sum(r['answered'] for r in neg),
               'negative_rejection_rate': ratio(sum(not r['answered'] for r in neg), len(neg))}
    for mode in ('strict', 'padded'):
        correct = sum(r[f'correct_{mode}'] for r in rows)
        metrics[f'positive_correct_{mode}'] = correct
        metrics[f'positive_accuracy_{mode}'] = ratio(correct, len(pos))
        metrics[f'answer_error_rate_{mode}'] = ratio(answered-correct, answered)
        metrics[f'grounded_negative_rejections_{mode}'] = sum(r[f'grounded_relation_rejection_{mode}'] for r in neg)
        metrics[f'task_accuracy_{mode}'] = ratio(correct+sum(not r['answered'] for r in neg), len(rows))
        metrics[f'balanced_task_accuracy_{mode}'] = (0.5 * (correct/len(pos) + sum(not r['answered'] for r in neg)/len(neg))) if pos and neg else None
    return metrics


def build_disambiguation_queries(scene, protocol, manifest, task_yaml, alignment, anchors):
    """Evaluator-side query authoring from GT geometry, never prediction outcomes."""
    from .relations import RelationGrounder
    families = protocol['supplement']['families'][scene]
    tasks = [t for t in manifest['queries'] if t['split'] == manifest['role']]
    by_task = {t['task']: t for t in tasks}
    gt = {task: _parse_gt_boxes(task_yaml, task) for task in by_task}
    anchor = next(iter(anchors))
    rotation = np.asarray(alignment['sim3']['rotation']) @ np.asarray(anchors[anchor])[:3, :3]
    margin = protocol['supplement']['minimum_axis_m']
    queries, labels, skipped = [], [], {'boundary': 0, 'multiple_valid_targets': 0}
    for family, members in families.items():
        if any(len(gt[task]) != 1 for task in members):
            raise ValueError('declared family requires single-GT task members')
        for reference in tasks:
            reference_task = reference['task']
            if reference_task in members or len(gt[reference_task]) != 1:
                continue
            for relation in ('left_of', 'right_of', 'front_of', 'behind'):
                distances = {task: RelationGrounder._signed_distance(relation, rotation.T @ (gt[task][0]['center']-gt[reference_task][0]['center'])) for task in members}
                if any(abs(d) < margin for d in distances.values()):
                    skipped['boundary'] += 1
                    continue
                allowed = [task for task in members if distances[task] > margin]
                if len(allowed) > 1:
                    skipped['multiple_valid_targets'] += 1
                    continue
                query_id = f'{family}--{slugify_task(reference_task)}--{relation}'
                queries.append({'query_id':query_id, 'target':family, 'reference':reference['sam_query'], 'relation':relation, 'anchor_frame':anchor})
                labels.append({'query_id':query_id, 'family':family, 'answerable':bool(allowed),
                               'acceptable_target_tasks':allowed, 'family_tasks':members, 'reference_task':reference_task,
                               'gt_signed_distances_m':distances})
    return {'scene_id':scene,'queries':queries}, {'scene_id':scene,'labels':labels,'skipped':skipped}


def alias_candidates(candidates, families):
    """Augment labels using the published taxonomy, independently of GT geometry."""
    result = deepcopy(candidates)
    for candidate in result:
        aliases = [family for family,tasks in families.items() if any(candidate['object_id'].startswith(slugify_task(task)+'__') for task in tasks)]
        candidate['query_aliases'] = aliases
    return result


def evaluate_disambiguation(predictions, candidates, labels, gt, alignment):
    # Use the same spatial evaluator for the single eligible target of positives.
    # For negatives, preserve geometric role matching against any family member.
    converted = [{**lab,'target_task':(lab['acceptable_target_tasks'] or lab['family_tasks'])[0]} for lab in labels]
    evaluated = evaluate_method(predictions,candidates,converted,gt,alignment)
    for row, label in zip(evaluated['rows'],labels,strict=True):
        row['family'] = label['family']
    for i,(row,label) in enumerate(zip(evaluated['rows'],labels,strict=True)):
        if not label['answerable']:
            alternatives=[evaluate_method([predictions[i]],candidates,[{**label,'target_task':task}],gt,alignment)['rows'][0] for task in label['family_tasks']]
            for mode in ('strict','padded'):
                for key in ('target','pair'):
                    row[f'{key}_{mode}']=any(r[f'{key}_{mode}'] for r in alternatives)
                row[f'grounded_relation_rejection_{mode}']=any(r[f'grounded_relation_rejection_{mode}'] for r in alternatives)
        row['target_task'] = label['acceptable_target_tasks'][0] if label['answerable'] else None
    evaluated['metrics']=aggregate(evaluated['rows'])
    evaluated['by_family']={family:aggregate([r for r in evaluated['rows'] if r['family']==family]) for family in sorted({r['family'] for r in evaluated['rows']})}
    return evaluated
