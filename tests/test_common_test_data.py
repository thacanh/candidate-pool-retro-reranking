"""Check the published cohort against its frozen numerical summaries."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/common_test_3814'
POOLS = {'historical_cap10', 'aizynthfinder_only', 'localretro_only', 'merged'}


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f)


def test_export_matches_recorded_hashes():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['training_performed'] is False
    for name, record in manifest['files'].items():
        data = (DATA / name).read_bytes()
        assert len(data) == record['size_bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']


def test_every_reaction_has_four_covered_ordered_lists():
    reactions = {int(r['reaction_id']): r for r in rows(DATA / 'reactions.csv')}
    assert len(reactions) == 3814
    assert {r['source_split'] for r in reactions.values()} == {'test'}
    assert len({r['canonical_product'] for r in reactions.values()}) == 3814
    seen, candidate_count = set(), 0
    with (DATA / 'candidate_pools.jsonl').open(encoding='utf-8') as f:
        for record in map(json.loads, f):
            rid, pool = record['reaction_id'], record['pool']
            assert (rid, pool) not in seen
            seen.add((rid, pool))
            assert pool in POOLS
            assert record['product_smiles'] == reactions[rid]['canonical_product']
            candidates = record['candidates']
            assert 1 <= len(candidates) <= 10
            assert [c['prior_rank'] for c in candidates] == list(range(1, len(candidates)+1))
            identities = {c['reactants_smiles'] for c in candidates}
            assert len(identities) == len(candidates)
            assert reactions[rid]['canonical_reference_reactants'] in identities
            candidate_count += len(candidates)
    assert seen == {(rid, pool) for rid in reactions for pool in POOLS}
    assert candidate_count == 140386


def test_paired_ranks_reproduce_all_frozen_reaction_effects():
    source = ROOT / 'outputs/transfer_analysis/m2_pool_shift/m2_reaction_pool_descriptors.csv'
    descriptors = {(int(r['reaction_id']), r['pool']): r for r in rows(source)}
    effects = defaultdict(lambda: [0.0, 0.0])
    seeds = defaultdict(set)
    for row in rows(DATA / 'reference_ranks.csv'):
        key = (int(row['reaction_id']), row['pool'])
        seed = int(row['seed'])
        assert seed not in seeds[key]
        seeds[key].add(seed)
        n = int(row['candidate_count'])
        b, a = int(row['baseline_reference_rank']), int(row['augmented_reference_rank'])
        assert 1 <= int(row['prior_reference_rank']) <= n <= 10
        assert 1 <= a <= n and 1 <= b <= n
        assert n == int(descriptors[key]['candidate_count'])
        effects[key][0] += (float(a == 1) - float(b == 1))/20
        effects[key][1] += (1/a - 1/b)/20
    assert set(effects) == set(descriptors)
    for key, effect in effects.items():
        assert seeds[key] == set(range(42, 62))
        assert abs(effect[0] - float(descriptors[key]['mean_20seed_delta_top1'])) < 1e-12
        assert abs(effect[1] - float(descriptors[key]['mean_20seed_delta_mrr'])) < 1e-12
