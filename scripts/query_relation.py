"""Run one structured relation query from a candidate cache without evaluator inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_relation_improvement import POOLS, alias_candidates, read
from relground.relation_reasoning import PairReasoningConfig, PairRelationGrounder
from relground.schemas import GroundingQuery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--pool', choices=POOLS, default='a2_pe')
    parser.add_argument('--protocol', type=Path, default=Path('configs/clio_relation_improvement_v2.json'))
    parser.add_argument('--target', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--relation', choices=['left_of', 'right_of', 'front_of', 'behind'], required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--mode', choices=PairRelationGrounder.MODES, default='filter_with_entity_guard')
    parser.add_argument('--coarse-aliases', action='store_true', help='Use the explicit diagnostic taxonomy in the protocol')
    args = parser.parse_args()
    cache, protocol = read(args.candidates), read(args.protocol)
    candidates = cache['pools'][args.pool]
    if args.coarse_aliases:
        candidates = alias_candidates(candidates, protocol['supplement']['families'][cache['scene_id']])
    cfg = protocol['reasoning']
    config = PairReasoningConfig(
        relation_margin=cfg['relation_margin_reconstruction_units'],
        minimum_entity_score=cfg['minimum_entity_score'],
        ambiguity_score_gap=cfg['ambiguity_score_gap'],
        equivalence_center_distance=cfg['equivalence_center_distance_reconstruction_units'],
        equivalence_min_aabb_iou=cfg['equivalence_min_aabb_iou'],
    )
    query = GroundingQuery('interactive', args.target, args.relation, args.reference, args.anchor)
    result = PairRelationGrounder(candidates, read(args.anchors), config).ground(query, args.mode).to_dict()
    by_id = {item['object_id']: item for item in candidates}
    ids = {'target': result['ranked_ids'][0] if result['ranked_ids'] else None,
           'reference': result.get('explanation', {}).get('reference_id')}
    result['selected_candidate_centers'] = {role: by_id[value]['center'] if value else None for role, value in ids.items()}
    result['center_frame'] = 'VGGT reconstruction coordinates'
    result['gt_read'] = False
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
