"""Print the four candidate lists and stored reference ranks for one test reaction."""
import argparse
import csv
import json
from pathlib import Path


def inspect(data: Path, reaction_id: int | None, seed: int) -> dict:
    with (data / 'reactions.csv').open(encoding='utf-8', newline='') as handle:
        reaction = next(
            (
                r
                for r in csv.DictReader(handle)
                if reaction_id is None or int(r['reaction_id']) == reaction_id
            ),
            None,
        )
    if reaction is None:
        raise ValueError('Reaction ID is not in the common test cohort.')
    rid = int(reaction['reaction_id'])
    with (data / 'candidate_pools.jsonl').open(encoding='utf-8') as handle:
        pools = [row for row in map(json.loads, handle) if row['reaction_id'] == rid]
    with (data / 'reference_ranks.csv').open(encoding='utf-8', newline='') as handle:
        ranks = [
            row
            for row in csv.DictReader(handle)
            if int(row['reaction_id']) == rid and int(row['seed']) == seed
        ]
    if len(pools) != 4 or len(ranks) != 4:
        raise ValueError('Expected four paired candidate lists and reference-rank records.')
    return dict(reaction=reaction, candidate_pools=pools, reference_ranks=ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/common_test_3814',
    )
    parser.add_argument(
        '--reaction-id', type=int, help='Default: first reaction in the supplied table.'
    )
    parser.add_argument('--seed', type=int, choices=range(42, 62), default=42)
    args = parser.parse_args()
    print(json.dumps(inspect(args.data_dir, args.reaction_id, args.seed), indent=2))


if __name__ == '__main__':
    main()
