# Common test reactions and candidate pools

This export contains the 3,814 official-test reactions whose recorded reference
reactants occur in all four frozen cap-10 candidate pools. There are 3,814
distinct canonical products. Each reaction has four separate candidate lists;
the number 3,814 does not count individual candidates.

## Files

- `reactions.csv`: reaction ID, split, class, original product/reference SMILES
  and canonical identities. `reaction_id` is the zero-based row index in the
  verified 50,037-reaction Chemformer USPTO-50K table.
- `candidate_pools.jsonl`: 15,256 records, one per reaction and pool, containing
  140,386 candidates in total. Each record contains the product and an ordered
  `candidates` list with `reactants_smiles`, `prior_rank` and `prior` fields.
  Alternative-pool candidates also retain generator membership indicators.
- `reference_ranks.csv`: 305,120 records, one per reaction, pool and paired
  seed (42--61). It reports the reference rank under the candidate prior,
  the prior+2D baseline, and the prior+2D+Uni-Mol scalar-feature ranker.
- `manifest.json`: source fingerprints, export settings and validation counts.

File hashes are covered by the repository's top-level `checksums.sha256`.

The pool names are `historical_cap10`, `aizynthfinder_only`, `localretro_only`
and `merged`. All lists are capped at ten. Alternative pools were originally
constructed at larger caps; this export preserves the exact first ten used in
the comparison. It does not contain the full Top-50 or Top-100 lists.

Candidate order and priors were checked against the frozen source pools.
Historical list identities were also checked against all 40 paired prediction
files. Reference ranks for all four pools and 20 seeds were checked against
the frozen predictions. No model was trained and no candidate was generated.

## Inspect one reaction

This example needs only Python's standard library:

```python
import csv
import json
from pathlib import Path

root = Path("data/common_test_3814")  # run from the repository root
with (root / "reactions.csv").open(encoding="utf-8") as f:
    reaction = next(csv.DictReader(f))
rid = int(reaction["reaction_id"])
print(reaction)

with (root / "candidate_pools.jsonl").open(encoding="utf-8") as f:
    for record in map(json.loads, f):
        if record["reaction_id"] == rid:
            print(record["pool"], record["candidates"])

with (root / "reference_ranks.csv").open(encoding="utf-8") as f:
    for record in csv.DictReader(f):
        if int(record["reaction_id"]) == rid and int(record["seed"]) == 42:
            print(record)
```

Ranks are one-based. Candidate priors are not calibrated probabilities:
historical lists retain their raw stored prior, whereas alternative pools
retain their generator-normalized prior before cap-10 truncation.
`baseline_reference_rank` means the learned prior+2D model; it is different
from `prior_reference_rank`. A reference match identifies the recorded
reactants, not experimental validation of the candidate's chemical feasibility.

This is a coverage-selected test subset for inspection and paired comparisons.
It must not be used as a training/validation set. Training a new embedding
ranker requires the corresponding official train/validation inputs. This
package contains reference ranks, not a complete ordering or scores for every
candidate under each learned model.

## Provenance and use

Reaction structures derive from the Chemformer distribution of USPTO-50K,
itself derived from Lowe's patent-reaction corpus. Source details:
https://github.com/MolecularAI/Chemformer and
https://doi.org/10.6084/m9.figshare.5104873.v1.
Candidate lists are the frozen outputs of this project's generators and pool
construction. The source repository records the exact upstream identities.

The repository's MIT license covers its original software; it does not grant
a blanket license to this dataset. Preserve the upstream attribution and
check the applicable data terms before redistribution.
