# USPTO-50K source audit

- **Audit date:** 2026-08-09
- **Decision:** keep the current 50,037-reaction benchmark
- **Protocol impact:** provenance repair only; no dataset migration and no cache
  invalidation

## Why the current dataset is defensible

The study uses the USPTO-50K variant distributed with Chemformer. The original
Chemformer `uspto_50.pickle` was previously verified reaction by reaction
against `data/uspto_smiles.csv`:

- 50,037 total reactions;
- 40,029 train, 5,004 validation, and 5,004 test reactions;
- 50,037/50,037 canonical products match;
- 50,037/50,037 canonical reactant sets match;
- local CSV SHA-256:
  `688c5b8ea7c3269b53ae15ffca9ec98f51fd29ea3fc25edc8fa66cabe9042d6a`;
- Chemformer pickle SHA-256:
  `b069af6e459e7deead8ef3d820b5984b0379230944a3fe511aed0fc8ff8c07a2`.

This is not an ad hoc local dataset. It is the same 50,037-reaction split size
reported by the peer-reviewed T5Chem paper.

## DOI and publication lineage

Use the following chain in the manuscript instead of citing Chemformer alone:

1. **Raw patent-reaction corpus.** Daniel Lowe, *Chemical reactions from US
   patents (1976-Sep2016)*, Figshare dataset,
   DOI `10.6084/m9.figshare.5104873.v1`, CC0.
2. **Reaction-role assignment and random USPTO subset.** Schneider, Stiefl, and
   Landrum, *What's What: The (Nearly) Definitive Guide to Reaction Role
   Assignment*, DOI `10.1021/acs.jcim.6b00564`.
3. **Retrosynthesis benchmark formulation.** Liu et al., *Retrosynthetic
   Reaction Prediction Using Neural Sequence-to-Sequence Models*,
   DOI `10.1021/acscentsci.7b00303`.
4. **Exact distribution used locally.** Irwin et al., *Chemformer*,
   DOI `10.1088/2632-2153/ac3ffb`, with the archived Chemformer repository and
   its `uspto_50.pickle` distribution.
5. **Independent persistent mirror and count confirmation.** Lu and Zhang,
   *Unified Deep Learning Model for Multitask Reaction Predictions with
   Explanation* (T5Chem), DOI `10.1021/acs.jcim.1c01467`, Zenodo record
   `https://zenodo.org/records/14280768`.

The Lowe corpus is the citable raw-data source; it is not itself a drop-in
replacement for the cleaned USPTO-50K benchmark. The paper and dataset records
must therefore describe both the upstream corpus and the exact processed split.

## Independent archive verification

The T5Chem Zenodo record contains `USPTO_50k.tar.bz2`:

- size: 913,111 bytes;
- MD5: `44a5f3ae08fe55933404c9398be22f5b`;
- archive counts: 40,029 train / 5,004 validation / 5,004 test.

The archive was compared with the local CSV in split order using RDKit
canonical SMILES with stereochemistry enabled:

| Split | Rows | Canonical product matches | Canonical reactant matches |
|---|---:|---:|---:|
| Train | 40,029 | 40,029 | 40,029 |
| Validation | 5,004 | 5,004 | 5,004 |
| Test | 5,004 | 5,004 | 5,004 |
| **Total** | **50,037** | **50,037** | **50,037** |

There are 49,606 byte-identical product/reactant SMILES pairs. The remaining
431 rows use different but canonically equivalent SMILES strings. This makes
the Zenodo archive a semantic mirror for products, reactants, split, and row
order, but not a byte-identical replacement. It does not include Chemformer's
reaction-class labels, so the verified Chemformer metadata remains necessary.

The cross-check can be repeated with:

```powershell
$env:PYTHONPATH = 'src'
python -c "from import_chemformer_metadata import verify_t5chem_archive; print(verify_t5chem_archive('data/uspto_smiles.csv', 'USPTO_50k.tar.bz2'))"
```

## Rejected replacements

Do not silently substitute the GLN/GraphRetro branch of USPTO-50K. The
DOI-backed Figshare archive `10.6084/m9.figshare.25459573.v1` contains 50,016
rows split as 40,008/5,001/5,007, rather than 50,037 split as
40,029/5,004/5,004. It is a different benchmark variant and would require new
candidate generation, leakage analysis, embeddings, model runs, statistics,
and manuscript numbers under a new protocol ID.

The anonymous Zenodo record `10.5281/zenodo.8114657` has a dataset DOI but
insufficient authorship and transformation provenance for preference over the
Chemformer/T5Chem lineage.

## Paper-ready provenance statement

> We used the 50,037-reaction USPTO-50K benchmark variant distributed with
> Chemformer, derived from Lowe's CC0 USPTO patent-reaction corpus. The local
> table was verified against the Chemformer pickle by SHA-256 and by
> reaction-level canonical product and reactant matching. As an independent
> archival cross-check, the 50,037-reaction T5Chem copy in Zenodo record
> 14280768 gave identical train/validation/test membership and row-wise
> canonical products and reactants. Reaction-class identifiers were retained
> from the verified Chemformer distribution.

## Cache decision

Because the DOI-backed archive is canonically identical to the dataset already
used, no current cache becomes invalid and the user's condition for deleting a
cache after adopting a new dataset is not triggered. Large-cache cleanup remains
a separate storage decision. In particular, deleting the 17.643 GiB embedding
pickle still requires full validation of the primary SQLite plus repair store;
it must not be justified as a dataset migration.
