# Prespecified revision analysis plan

> **Publication-target update (2026-08-27):** This protocol was originally
> drafted while a *Journal of Cheminformatics* manuscript was under internal
> development. That draft was never submitted or published and has been
> abandoned. Its frozen experiments and artifacts now form the historical
> anchor within the sole active, self-contained *Digital Discovery* study. This
> note changes the publication target only; it does not revise any scientific
> decision, comparator, gate or numerical result below.

- **Date drafted:** 2026-08-09
- **Review incorporated:** 2026-08-09
- **Original target:** *Journal of Cheminformatics* (abandoned internal draft)
- **Active target:** *Digital Discovery*
- **Status:** APPROVED - EXECUTION LOCKS STILL REQUIRED
- **Run gate:** The scientific design is frozen by the explicit user approval
  recorded below. No scientific experiment may run until this approved plan is
  committed and the applicable platform-specific environment lock is recorded.

## Objective and claim boundary

The revision tests which part of the current small reranking gain is attributable
to Uni-Mol specifically, whether it is stable to conformer generation, and
whether it survives stronger baselines and candidate pools.

The existing evidence supports only a small complementary signal under a fixed
candidate pool and ranker. It does not yet isolate 3D information, conformer
stability, or encoder identity. Null or reversed controls will be reported and
will narrow the permitted attribution and conclusion. The current neutral
Uni-Mol-derived-features title does not require WS-C to remain valid.

An encoder comparison is an attribution control, not a causal isolation of 3D
pretraining. Uni-Mol outperforming alternative encoders may support an
Uni-Mol-specific representation effect; attributing that difference to 3D
pretraining would additionally require a matched architecture or a direct
coordinate/conformer perturbation control.

## Evidence to preserve

The legacy `legacy-cap10-fixed50-v1` result and the following design elements
remain unchanged as a documented sensitivity analysis:

- official USPTO-50K split;
- leakage audit removing 233 overlapping training reactions;
- identical candidate lists within paired comparisons;
- identical seed-specific pair sampling;
- training-pair-only normalization;
- permuted-feature matched control;
- coverage/conditional/end-to-end separation; and
- clustered paired inference with rank-promotion and degradation counts.

No existing result is silently replaced. Revised-primary and legacy-sensitivity
tables must be labeled separately.

## Primary estimands

1. Conditional Top-1 difference between augmented and matched baseline arms.
2. Conditional MRR difference between augmented and matched baseline arms.
3. End-to-end Top-1 and MRR after assigning zero credit to uncovered reactions.
4. Candidate coverage for each pool/cap.
5. Reaction-level promotions, degradations, and net flips.

All comparisons are paired by reaction and seed. Product-clustered uncertainty
retains all seed predictions for a sampled canonical product. If G1 is run, the
20-seed analysis must add seed-marginal uncertainty using a method fixed before
those runs are examined. G1 strengthens the analysis but is not used to imply
that the existing five paired seeds are invalid.

## Experiment matrix and reporting rules

| ID | Workstream | Single intended change | Frozen comparator | Prespecified reporting/decision rule |
|---|---|---|---|---|
| A-CAP10-REPRO | A1/A4 | Regenerate candidate pool with recorded environment | Historical cap-10 pool | Ordered canonical candidate identities, duplicate handling, and priors must match the approved tolerance. Any mismatch stops downstream work. |
| A-CAP50 | A1/A4 | Candidate cap 10 -> 50 | A-CAP10-REPRO | Secondary expanded-pool robustness analysis. Report cap-10 and cap-50 coverage/quality separately; do not relabel or automatically replace the cap-10 result. |
| A-CAP50-ANCHORED | A1/A4 | Append clean-run Top-50 identities beyond an immutable historical cap-10 anchor | Historical cap-10 pool plus the failed clean A-CAP10/A-CAP50 run | Post-gate secondary sensitivity only. Preserve the failed exact-reproduction result. Require every historical canonical candidate to occur in the checksummed clean Top-50, retain historical identity/order/prior for the anchor, append only new clean-run identities, cap at 50, and report this as `cap50-legacy-anchored-v1`, never as exact reproduction. |
| B-C1..C5 | B1 | RDKit conformer seed only | Same molecule, encoder, checkpoint, feature code | For each of three scalars report pairwise absolute differences, Pearson, Spearman, CV, and rigid vs flexible strata. |
| B-25GRID | B2 | Five conformer replicates crossed with seeds 42-46 | Prior+2D baseline uses seeds 42-46 | Call the gain robust only if augmented beats prior+2D in at least 20/25 runs and the conformer-marginal interval excludes zero; otherwise call it conformer-sensitive and reframe. |
| B-AVG10 | B3 | Average the three pair-level scalars over 10 indexed conformer replicates | Single-conformer augmented arm | For each replicate, compute the existing fragment handling and all three scalars first, then take the arithmetic mean of each scalar across replicates. Do not average embeddings before feature computation. Report effect and stability whether improved, unchanged, or worse. |
| C-GROVER | C1a | Replace Uni-Mol source with frozen GROVER-base concatenated 2D atom states | Same three scalar formulas and matched ranker | Concatenate `atom_from_atom` and `atom_from_bond`, the complete official two-view atom representation. Generate deterministic scalar features once using four independent query-scoped GPU workers over contiguous ranges; never mix molecules from different queries in an encoder batch and write no global atom cache. Then fit the matched frozen ranker with seeds 42--46. This replaces the proposed ChemBERTa arm because its BPE token states cannot be mapped defensibly to all SMILES atoms without changing the input or discarding structural tokens. |
| C-MORGAN-ATOM | C1b | Replace learned encoder with deterministic per-atom Morgan vectors | Same three scalar formulas and matched ranker | Use radius-2, 2,048-bit atom-centred Morgan vectors with `fromAtoms=[atom_index]` and `useChirality=False`, cast to float for the same scalar formulas. Record the RDKit version and exact options; do not substitute Mol2vec. |
| C-PROJECTED | C2 | Replace three scalars with projected pooled embeddings | Matched `cap10-tuned-v1` protocol | Label as an upper-bound probe, report train/validation curves, and do not present it as a controlled architecture match. |
| D-TUNE-2D / D-TUNE-AUG | D1 | Validation-only hyperparameter selection for both arms on the approved cap-10 pool | Identical 81-point search budget | Intended revised primary is `cap10-tuned-v1` tuned-vs-tuned. Test data are never used for selection. Legacy fixed-50 remains sensitivity; cap-50 remains secondary pool robustness. |
| D-PRIOR-RAW/LOG/RANK | D2 | Prior transform only | Same baseline protocol | Select only by validation MRR using a prespecified epsilon and tie rule; apply the selected transform consistently in the paired augmented comparison. |
| D-CAPACITY | D3 | Baseline capacity only | 289-parameter augmented model | Verify parameter count or use the approved nearest-count rule; report exact counts. |
| D-LGBM-2D/AUG | D4 | Learner family -> LambdaMART | Same feature sets and query groups | Use the frozen 27-configuration LightGBM budget below for both feature sets. Persistence of the gain is evidence beyond one MLP family. |
| D-EXTERNAL | D5 | Published reranker -> rxn-ebm | Same candidate pool where inputs allow | Attempt the pinned rxn-ebm revision first. If its graph/reaction inputs cannot be constructed without changing the candidate pool, record the exact failure and audit the pinned RetroRanker revision as the documented fallback. |
| E-POOLS | E1/E2 | Candidate generator/pool only | Frozen final ranker protocol | Report AiZynthFinder-only, second-generator-only, and merged-deduplicated coverage plus Top-1/MRR; report whether the gain persists, grows, or vanishes. |
| F-ROUNDTRIP | F1 | Exact match -> forward-model round trip | Frozen final top-1 predictions | Report all three systems and retain exact-match results; explicitly state forward-model limitations. |
| F-CHIRAL | F2 | Morgan chirality false -> true | Same data/model | Report both settings and the number of stereo-only reference differences. |
| F-SALT | F3 | Salt-removal handling only | Current fragmentwise handling | Report sensitivity and diagnose every zero-fallback SMILES by failure cause. |
| G-20SEED | G1 | Seeds 42-46 -> 42-61 | Frozen primary arms | High-priority strengthening analysis when compute is available. Report all point estimates/SD and approved seed-marginal intervals; do not frame 20 seeds as a methodological necessity or absolute submission gate. |
| G-BH | G2 | Multiplicity adjustment only | Ten class-level tests | Report raw p-values and BH q-values for the fixed family of 10 classes; state whether class 3 survives. |
| G-NETFLIP | G3 | Framing/reporting only | Frozen predictions | Put improved, degraded, unchanged, and net-flip counts in Results. They may be discussed and may enter the abstract if materially useful and space permits, but are not an abstract requirement. |
| H-INTERPRET | H1-H3 | Descriptive model over frozen rank shifts | Frozen final predictions | Report descriptor distributions and logistic coefficients with intervals; label the paragraph descriptive and post hoc. |

## Encoder-attribution interpretation

After an equivalence margin is approved:

- Uni-Mol above both matched alternative arms supports an Uni-Mol-specific
  representation effect. It does not by itself isolate 3D pretraining as the
  causal source; that would require a matched architecture or a direct
  coordinate/conformer perturbation control.
- Uni-Mol equivalent to GROVER and above Morgan-atom supports only a learned
  atom-embedding claim. Narrow the conclusion, but do not automatically change
  the current neutral Uni-Mol-derived-features title.
- All three equivalent supports a representation-agnostic atom-level comparison
  claim. Remove uniqueness or causal 3D-pretraining language; change the title
  only if its final wording contains such a stronger claim.

"Equivalent" must be an interval-based prespecified rule, not visual similarity
or overlapping error bars.

The attribution endpoint is the encoder-minus-baseline gain. Equivalence uses
two one-sided tests at alpha 0.05, equivalently a 90% paired
canonical-product-clustered interval. The margins are +/-0.0025 absolute Top-1
and +/-0.0020 MRR. An arm is called equivalent only when both intervals fall
fully inside their respective margins; an inconclusive interval is not called
equivalent.

## Candidate reproduction gate

The historical reference is `outputs/rerank_dataset.jsonl`, SHA-256
`9ec1cf192c49eeb7d74a320dd721287fabdef9863cc06f95d0f13baab8c3ff85`.
The normalized cap-10 comparison uses RDKit canonical isomeric product SMILES
and fragment-order-invariant canonical reactant sets. Canonical duplicates keep
the largest prior; an exact prior tie keeps the first occurrence. Candidates
are stable-sorted by decreasing prior. Product keys, candidate identities,
candidate order, missing/extra products, duplicate winners and tie order must
match exactly. `label` is ignored because generation does not produce it and
the controlled loader reconstructs ground truth independently.

Every prior must be finite and satisfy
`abs(new-reference) <= max(1e-8, 1e-6*abs(reference))`, with no rank inversion.
The report also records exact-float fraction, maximum absolute/relative error
and float32 ULP distance. Any identity/order mismatch fails the gate regardless
of prior tolerance. The expected normalized audit is 396,219 raw records,
49,584 products, 5 invalid records, 3,094 canonical duplicates and 393,120
unique candidates. Cap-50 may proceed only if cap-10 passes; the normalized
cap-10 list must then be an exact prefix of cap-50 for every product. Cap-50 is
written to a new artifact and never overwrites the historical pool.

## Pinned external identities

- Uni-Mol wrapper: `unimol-tools==0.1.3`; PyPI source archive SHA-256
  `d00b86bebd556d7c82c10b36c019599d58f71e1c7d1b71dcfc51cd1ae6f8125e`.
- Chemistry toolkit for the new revision protocols: RDKit `2025.09.6`
  (`rdkit==2025.9.6` in the Python package index).
- Uni-Mol molecular no-hydrogen checkpoint:
  `mol_pre_no_h_220816.pt`, 190,540,187 bytes, SHA-256
  `da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9`,
  downloaded from the official Uni-Mol v0.1 release URL.
- Learned 2D atom encoder: GROVER-base, official repository commit
  `40b6d97098e4508687912f3c05eca369fc2c6213`; official source URL
  `https://drive.google.com/uc?id=1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl`, recorded locally as
  `grover_base.pt`, 193,589,423 bytes, SHA-256
  `47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de`.
- External reranker: `coleygroup/rxn-ebm` commit
  `1919eeccdd31e16ec7a44478b756bcd974c35a3c`; feasibility fallback:
  `catalystforyou/RetroRanker` commit
  `22b5765743f3325b0e3b51b0cb9cf2d908081399`.
- Second generator: `kaist-amsg/LocalRetro` commit
  `eba83e72efabeb854fec86c865e8743c295a8a1e`, trained only on the verified
  current USPTO-50K train partition after approval. The resulting checkpoint
  hash is an output manifest field, not a value to invent in advance.
- Atom mapper required by that second-generator training data: RXNMapper commit
  `a01ecdcd5ac944850e9691739c1df858e005fd39`, MIT license, using the bundled
  publication model `albert_heads_8_uspto_all_1310k`. Its
  `pytorch_model.bin` is 3,212,952 bytes, SHA-256
  `8541f3f500dae71abe678d546bd035ca946e2d1c819f6b2cf41a97faedd7e6a2`;
  `vocab.txt` SHA-256 is
  `99e9ad4949844c56205ed51dc62edaecb69787d63c6c74d607116c93eb2a5738`.
- Forward model: Figshare DOI `10.6084/m9.figshare.23960724.v1`, record
  23960724, version 1, Apache-2.0; file `chemformer_forward.zip`, file ID
  42012708, source URL `https://ndownloader.figshare.com/files/42012708`,
  490,543,340 bytes, published MD5
  `f0bf5b9f49037c31efe90e48ec7ab16e`, verified archive SHA-256
  `1ee180c898d87b770b98d2ee60035cf594cb897990d675230c7e9453e8a4cab4`.
  The archive contains `last.ckpt`, 537,441,893 bytes, SHA-256
  `44203e603e0ed9919213fdd822cb0bff844bd9fbae6f5f5882e1771046f0b287`.

Every downloaded archive/checkpoint, environment lock and local wrapper file is
fingerprinted in the relevant manifest. A repository name or mutable URL alone
does not satisfy the manifest gate.

`environment-revision.yml`, `requirements-revision.txt`, and
`constraints-revision-py310.txt` define the pinned top-level core revision
stack. They are not a transitive platform lock. Resolver checks found that
AiZynthFinder 4.4.1 cannot share the Uni-Mol 0.1.3 core environment because of
the latter's `pandas<2` boundary and the candidate generator's dependency
graph. Candidate generation, LocalRetro, GROVER, rxn-ebm/RetroRanker and
Chemformer therefore require separate platform-specific locks. The remaining
environment gate closes only after the actual Windows/Linux CPU/CUDA targets
have artifact-hash/build-tag locks for every environment used.

## Baseline selection rules

The D1 grid is the Cartesian product specified in the feedback:

- hidden width: 32, 64, 128;
- dropout: 0.0, 0.1, 0.3;
- learning rate: 1e-4, 3e-4, 1e-3; and
- BPR margin: 0.0, 0.1, 0.3.

Both baseline and augmented arms receive the same 81 configurations and the
same early-stopping implementation. Selection uses only the official validation
partition. Maximum training is 200 epochs; conditional validation MRR is
evaluated after every epoch with patience 20 and minimum improvement `1e-5`.
The earliest epoch wins an exact within-run tie. Each arm selects one shared
configuration by mean best validation MRR across seeds 42--46; the best epoch
and checkpoint may differ by seed. Configuration ties within `1e-12` use the
first enumeration in the grid order printed above. Test data are not loaded by
the selection process, and the selected models are not retrained on
train-plus-validation.

For D2, compare raw prior, `log(prior + 1e-12)`, and the descending within-list
midrank score `1 - (midrank - 1) / (n - 1)`, with a singleton score of 1. Exact
prior ties share the midrank. Run the full 81-configuration baseline search for
each transform and select by mean official-validation MRR; ties within `1e-12`
prefer raw, then log, then rank. Freeze that one transform for both matched
arms, then give the augmented arm its own 81-configuration search. This outer
selection favours the baseline and avoids choosing a transform from the test
set or from the augmented arm.

For D3, a four-input one-hidden-layer model of width 48 has exactly 289 trainable
parameters, matching the seven-input width-32 model. The runner must assert both
counts equal 289 before training.

For D4, pin `lightgbm==4.6.0` and create one query per canonical training
product. All cap-10 candidates remain in frozen prior order; relevance is 1 for
the union of official-train reference candidates for that product and 0
otherwise. Use `objective="lambdarank"`, `label_gain=[0,1]`,
`deterministic=True`, `force_col_wise=True`, feature and bagging fractions of
1.0, and one thread. Both feature arms receive the same 27 configurations:
`num_leaves` in {7,15,31}, `min_child_samples` in {10,20,50}, and
`learning_rate` in {0.03,0.1,0.2}. Train at most 2,000 trees with patience 50 on
reaction-level official-validation conditional MRR; ties use grid enumeration
order. Validation and test reporting remain reaction-level even when a training
query contains multiple positives.

## Conformer analysis

- The common conformer pipeline is the inspected `unimol-tools==0.1.3`
  `ConformerGen(method="rdkit_random", mode="fast", remove_hs=True,
  max_atoms=256)` path with the pinned no-hydrogen checkpoint above. It parses
  SMILES, adds hydrogens, calls `AllChem.EmbedMolecule` with an explicit seed,
  applies `MMFFOptimizeMolecule` when embedding succeeds, keeps the embedded
  coordinates if MMFF fails, and uses the package's 2D fallback when embedding
  fails. No explicit ETKDG parameter object or multi-conformer pooling occurs in
  this package path. The environment lock pins the exact RDKit build.
- Conformer labels C1-C5 use seeds 42--46. Within B1/B2, only this seed changes;
  checkpoint, molecule order, canonical SMILES, package source, fallback rules,
  feature code and all ranker settings remain fixed. C1 uses the package default
  seed value explicitly rather than relying on an implicit default.
- B1 recomputes the three scalar features for every product-candidate pair.
- Flexible pairs have at least five RDKit strict rotatable bonds in the product
  or aggregate candidate; all remaining pairs are the prespecified rigid
  comparison, with zero-bond and one-to-four-bond sub-strata also reported.
  For every scalar and each of the ten C1-C5 replicate pairs, report the pooled
  absolute-difference distribution, Pearson and Spearman correlation. Per-pair
  coefficient of variation is sample SD divided by absolute mean; values with
  absolute mean at most `1e-8` are undefined and counted rather than imputed.
- B2 crosses C1-C5 with training seeds 42-46 for 25 augmented runs. The baseline
  has five runs because it does not use embeddings.
- For each conditional metric, fit the crossed random-effects model
  `gain_sc = mean + training_seed_s + conformer_c + residual_sc` by REML; the
  residual contains seed-by-conformer interaction. Report variance components
  and fractions. The conformer-marginal interval uses 10,000 bootstrap samples,
  RNG seed 2026, independently resampling canonical-product clusters, training
  seeds and conformer levels while retaining paired arms. The primary robustness
  gate is Top-1 gain above zero in at least 20/25 cells and a 95% crossed
  bootstrap interval excluding zero; MRR receives the same analysis but does
  not override the Top-1 gate.
- B3 creates 10 indexed conformers per product and per reactant fragment using
  seeds 42--51. At each index, it applies the existing fragment aggregation and
  computes all three pair-level scalars; the reported feature vector is the arithmetic mean
  of each scalar across the ten indices. Embeddings are never averaged before
  scalar computation. The pinned package fallback is retained as that replicate
  rather than silently omitted; the fallback stage and count are reported for
  every molecule and conformer index.

## Secondary workstream rules

For E, LocalRetro is trained from the verified current train partition at the
pinned code commit; its official validation partition is used for checkpoint
selection and its test partition is never used for training or selection.
Generate Top-50 candidates and preserve raw ranks. Report AiZynthFinder-only,
LocalRetro-only and canonical merged-deduplicated pools. Per-generator prior is
`1 - (rank - 1) / max(n - 1, 1)`. A merged candidate retains the maximum
normalized prior and two binary source indicators; candidates found by both
generators have both indicators set. Canonical duplicate identity is
stereochemistry-preserving and fragment-order invariant. Each pool is trained
and evaluated with matched baseline/augmented arms; no cross-pool conclusion is
attributed solely to the representation.

For F1, use the pinned Chemformer USPTO-50K forward checkpoint. Primary
round-trip success requires its beam-1 product, canonicalized with RDKit while
preserving stereochemistry, to equal the target canonical product; beam-5
success is a labeled sensitivity result. Invalid outputs are failures. Report
candidate-prior, 2D and augmented top-1 precursor sets under identical forward
settings and retain exact-match metrics.

For F2, a stereo-only reference difference is one whose stereochemistry-stripped
canonical reactant fragment set matches but stereochemistry-preserving set does
not. Count these over all 5,004 official test reactions. The sensitivity changes
only Morgan `useChirality=False` to `True` for both product and candidate
fingerprints.

For F3, ground-truth matching remains unchanged. The sensitivity changes only
the molecular input to the representation features: apply the pinned RDKit
`SaltRemover` default definition with `dontRemoveEverything=True`, preserve
stereochemistry, and retain all surviving fragments under the existing
fragmentwise concatenation rule. Every zero fallback is classified as SMILES
parse, salt-removal-empty, atom-limit, conformer/coordinate, checkpoint/model,
or cache-lookup failure. F2 and F3 are owned by the Representations workstream.

G1 uses seeds 42--61. In addition to the existing product-only interval, report
a 10,000-sample product-by-seed paired bootstrap with RNG seed 2026,
independently resampling canonical-product clusters and the 20 seed indices;
the percentile 95% interval is labeled seed-marginal. G2 defines one family of
ten class-level conditional Top-1 tests. Each raw p-value is a two-sided
canonical-product-cluster sign-flip test of seed-averaged reaction differences
with 100,000 Monte Carlo draws and RNG seed 2027. Apply Benjamini-Hochberg at
q=0.05 across exactly classes 1--10. MRR effects and intervals are reported
without creating an undeclared second p-value family.

For H2, the observation is one covered reaction after averaging frozen paired
seeds. Restrict to reactions with non-zero mean reference-rank change and model
improved versus degraded. Predictors are seed-mean within-reaction descriptor
differences between augmented and 2D top-1 candidates; continuous predictors
are standardized on the analysis set and reaction-class indicators enter the
all-class model. Report a class-3-only flexibility model only if both outcomes
have at least 20 observations. Coefficient and odds-ratio intervals use 10,000
canonical-product-cluster bootstrap refits with RNG seed 2028; failed or
separated refits are counted. All H results remain descriptive and post hoc.

## Statistical and manuscript gates

- Report every arm, not only the winner.
- Report effect sizes and intervals even when a decision rule fails.
- Use BH correction for a fixed family of the 10 reaction classes.
- The class-3 flexibility analysis is exploratory and post hoc.
- The current neutral title, *Controlled Evaluation of Uni-Mol-Derived
  Features...*, remains valid regardless of WS-C. WS-C controls the permitted
  representation claim; change the title only if a stronger causal
  3D-pretraining claim is proposed and supported.
- The abstract order is: design, coverage ceiling, effect with interval,
  conformer result, and encoder-control result. Net-flip counts are optional in
  the abstract and mandatory in Results.
- The final limitations retain only limitations not resolved by the new work.

## Blocking decisions before sign-off

- [x] Use the two-level cap-10 reproduction gate above: exact normalized
  identity/order/duplicate behavior and the frozen finite-prior tolerance.
- [x] Use `cap10-tuned-v1` as the intended revised-primary comparison after
  sign-off; retain `legacy-cap10-fixed50-v1` as sensitivity and cap-50 protocols
  as secondary expanded-pool robustness.
- [x] Use explicit `unimol-tools==0.1.3` conformer seeds 42--46, the inspected
  package fallback path, the crossed REML model and the 10,000-sample crossed
  bootstrap specified above.
- [x] For B3, compute the existing three scalars separately for each indexed
  conformer replicate after existing fragment handling, then average each
  scalar across the ten replicates; do not average embeddings.
- [x] Do not force ChemBERTa BPE tokens onto atoms. Replace C1a with GROVER-base
  2D graph atom states at the pinned code revision.
- [x] Download the official GROVER-base checkpoint and record its filename,
  source URL, size and SHA-256 before supervisor sign-off.
- [x] Freeze C1a to the concatenation of GROVER `atom_from_atom` and
  `atom_from_bond` before feature generation. GROVER features are deterministic
  and generated once; only the matched downstream ranker uses seeds 42--46.
- [x] Use deterministic radius-2, 2,048-bit per-atom Morgan vectors with
  `fromAtoms=[atom_index]`, `useChirality=False`, and the recorded RDKit
  version; do not use Mol2vec.
- [x] Use the TOST/90% interval equivalence rule and Top-1/MRR margins specified
  above for WS-C.
- [x] Treat any C2 early-stopping reference as D1, not D2; this correction is
  recorded in this plan and `AGENTS.md`.
- [x] Use D1 max 200 epochs, patience 20, `1e-5` improvement, a shared
  configuration per arm across five seeds, and the frozen enumeration tie rule.
- [x] Use D2 epsilon `1e-12`, descending normalized midranks with shared ties,
  baseline-only outer selection and one frozen transform for both arms.
- [x] Use hidden width 48 for the four-input capacity arm and assert both models
  have exactly 289 trainable parameters.
- [x] Use LightGBM 4.6.0, product query groups and the frozen 27-configuration
  deterministic budget above.
- [x] Attempt pinned rxn-ebm first and use pinned RetroRanker only as the
  documented feasibility fallback.
- [x] Use pinned LocalRetro code, train only on the verified current split,
  Top-50 candidates and the frozen merged-pool/source-indicator rules above.
- [x] Download the public Chemformer USPTO-50K forward checkpoint and record its
  filename, source URL, size and SHA-256 before supervisor sign-off.
- [x] Use the stereo-only definition and salt-removal/failure taxonomy above.
- [x] Use the frozen G1 crossed product-by-seed bootstrap and G2 ten-class
  sign-flip/BH family above.
- [x] Use one seed-aggregated covered reaction as the H2 observation and the
  canonical-product-cluster bootstrap above.
- [x] Assign F2 and F3 to the Representations owner.
- [ ] Generate and commit full transitive platform locks (including artifact
  hashes/build tags) for the pinned core stack and each separate external
  system environment before any scientific run. The checked-in top-level
  revision specs are not mislabeled as full locks.
- [x] Treat G1 as high-priority strengthening rather than an absolute P0 or
  submission-validity gate; keep B2 at 5 x 5. Expanding B2 to 20 x 5 requires a
  separate compute decision.

## Approval

Supervisor approval means the hypotheses, arms, analysis methods, and reporting
rules above are frozen before results are inspected.

- **Supervisor name:** Repository owner/user (name not supplied)
- **Approval date:** 2026-08-09
- **Commit hash containing this approved plan:**
  `c580c9a8c75aef05e87e9fee576fdc227e04ac0d`
- **Signature/recorded approval:** Explicit confirmation in the Codex task:
  "ok ký đi, tôi confirm, triển khai thôi"

The approval freezes the scientific choices above. Infrastructure timing pilots
remain non-scientific dry runs; full protocol outputs remain blocked until the
applicable platform lock is committed.

## Recorded post-gate deviation: legacy-anchored cap-50

The clean `A-CAP10-REPRO` run completed on 2026-08-14 and failed its exact
gate. Product identity/order and aligned priors were stable, but equal-prior
outcome ordering and Top-10 truncation were not exactly reproducible because
the historical full package/Python-hash environment was not retained. This
failure remains a reported result and is not relabeled as a pass.

The imported clean Top-50 nevertheless contains every one of the 393,120
normalized historical candidate identities (zero missing). On 2026-08-14 the
repository owner/user explicitly authorized continued use of this artifact if
scientifically defensible: "ns chung cái này dùng được k, nếu dùng đc thì triển
khai tiếp". The authorized deviation is `cap50-legacy-anchored-v1`: retain the
historical normalized cap-10 identities, order and priors as an immutable
anchor; append only new normalized identities from the checksummed clean
Top-50 in its frozen order, up to 50; and preserve source indicators and both
input hashes. It is secondary expanded-pool sensitivity, does not replace the
cap-10 primary, and cannot support a claim of exact candidate-generation
reproduction.

Because a clean-run extension can have a raw policy prior above the final
historical anchor candidate when equal-prior/outcome enumeration changes, the
anchor remains the immutable prefix and every such cross-boundary inversion is
counted. In WS-E, the generator prior is the already-prespecified normalized
anchored rank, `1-(rank-1)/max(n-1,1)`; raw policy priors are retained for audit
and must not silently reorder the anchor.

## Recorded post-gate preprocessing decision: LocalRetro atom mapping

The exact Chemformer source artifact `uspto_50.pickle` was downloaded from its
recorded official Box file on 2026-08-14. Its 40,245,659-byte size and SHA-256
`b069af6e459e7deead8ef3d820b5984b0379230944a3fe511aed0fc8ff8c07a2`
match the existing provenance record, but all 2,713,366 reactant/product atoms
have atom-map number zero. LocalRetro cannot extract its local templates from
that representation.

The atom-mapped GLN/Schneider files linked by LocalRetro were pinned at the
already-recorded rxn-ebm commit and audited only as a possible crosswalk. They
contain 50,016 rows with split counts 40,008/5,001/5,007, not the current
Chemformer 50,037 rows and 40,029/5,004/5,004 split. Canonical
stereochemistry-preserving product/reactant-set matching found 31,626 unique
same-split rows, 16,253 unique cross-split rows, 1,864 missing rows and 294
ambiguous mapped duplicates. Therefore GLN atom maps are not copied into the
current benchmark and its pretrained LocalRetro checkpoint is audit-only.

After this blocker and the single-mapper remedy were shown, the repository
owner/user authorized continuation on 2026-08-14 ("ok triển khai tiếp đi").
Protocol `localretro-chemformer50k-rxnmapper-v1` applies the one pinned
RXNMapper/model above to every current **train and official-validation**
reaction, validates canonical chemistry plus product-map uniqueness, and
fails if any row is invalid. Official-test reactants are never mapped or
copied into the LocalRetro training workspace. LocalRetro is trained once at
seed 42 with its official architecture and default 50-epoch/patience-5
optimization; checkpoint selection is official-validation loss only. Only
after the checkpoint hash is frozen is the 49,584-product class-unknown
inference inventory created, with no ground-truth reactants, and Top-50 is
decoded. The merged pool can contain at most 100 unique candidates; exact
normalized-prior ties use canonical candidate identity and no generator
priority. This is a preprocessing decision needed to instantiate the already
approved E arm, not a result-dependent tuning choice.

### Recorded post-gate amendment: structurally invalid mapping rows

The first retained v1 mapping attempt stopped at shard 0 as prespecified. Of
2,815 source reactions, 24 RXNMapper strings differed only in stereochemical
serialization and 21 had one to three product atom-map indices absent from all
reactants. A second diagnostic with RXNMapper canonicalization disabled gave
the same 24/21 split. Non-isomeric identities proved that all 24 serialization
differences retained the same molecular graphs; none changed non-stereo
chemistry. Inspection of the 21 remaining source rows confirmed chemically
unbalanced records for which a valid product-to-reactant atom mapping cannot
be invented.

On 2026-08-15 the repository owner explicitly approved: "xác nhận protocol v2
và loại các reaction không cân bằng có audit". Protocol
`localretro-chemformer50k-rxnmapper-filtered-v2` therefore transfers only the
pinned RXNMapper atom indices onto the exact original isomeric RDKit graphs,
so the mapper cannot alter source stereochemistry. It excludes only reactions
whose positive product atom-map indices remain absent from every reactant
after that transfer. Each exclusion records reaction ID, official split,
reason and missing-product-atom count in an immutable per-shard audit; all
other failures remain fatal. Compilation requires the union of retained and
excluded IDs to equal every current train+validation ID exactly and emits a
combined exclusion manifest. The downstream protocol is
`localretro-top50-current-split-filtered-v2`. Test remains closed until its
checkpoint freeze. This is a source-validity remedy, not result-based tuning;
the failed v1 gate and its 45-row shard-0 audit remain preserved.

The first v2 implementation attempt also stopped before writing shard 0: the
same 24 graph-identical rows failed because RDKit retained stale canonical
ranking properties after atom-map removal, yielding alternative ring-closure
serialization such as `c23` versus `c32`. Reaction 217 was inspected end to
end: original, mapper and restored molecules had the same non-stereo graph,
InChI and bidirectional chirality-aware graph match. The identity routine now
clears computed properties, sanitizes, and reassigns stereochemistry after
removing atom maps before canonical comparison. The exact reaction-217
regression passes; no chemistry gate or exclusion rule was weakened.

### Recorded compute-only decoder acceleration

The retained filtered-v2 run completed raw GPU inference for all 49,584
products, but the pinned upstream CPU decoder reached only 6,423 products
after 4 h 25 min and projected another 28 h 10 min on the rented nine-core
instance. Code inspection found that upstream calls the same deterministic
`decode_localtemplate` function twice for every proposal and keeps every
decoded product only in memory until the entire population finishes. On
2026-08-15 the repository owner authorized acceleration ("phải tăng tốc th,
nma instance này chỉ có 9 cores CPU th").

Decoder protocol `localretro-decode-resumable-v1` therefore keeps eight CPU
workers, calls the same pinned template decoder once per proposal, prepares
each product graph once, caches compiled reaction SMARTS per worker, and commits
each completed product to a resumable SQLite store. It does not change raw GPU
edits, template/site order, scores, exact duplicate handling, Top-50 stopping,
or final test-ID order. Before the full run, fixed product IDs spanning the
inventory must match the original upstream decoder exactly; any difference
stops the accelerated run. The abandoned upstream partial result had no
decoded output file and remains evidenced in its log. This is a compute-only
implementation correction under the existing filtered-v2 scientific protocol.

## Digital Discovery transfer amendment: J/K final round

- **Date drafted:** 2026-08-26
- **Target:** *Digital Discovery* (RSC); *Scientific Reports* fallback
- **Status:** APPROVED AND SIGNED FOR ISOLATED J/K/L1 EXECUTION
- **Source:** supervisor checklist `Experiment_Checklist_Final_Round.md.pdf`

This amendment extends the frozen historical-anchor analysis into the sole
active *Digital Discovery* study without revising or replacing the anchor
results. Its code and outputs use the isolated namespace
`outputs/digital_discovery_round_jk/`; the manuscript uses
`paper/digital_discovery/`. The following historical-anchor paths are read-only
inputs and must never be written by a J/K command:

- `outputs/revision_analysis/`;
- `outputs/jcheminform_revision/numerical_freeze_v2/`; and
- `paper/overleaf/`.

The re-analysis protocol is `dd-round-jk-reanalysis-v1`.  The truncation and
refit protocol is `dd-expanded-pool-cap10-d1-v1`.  Every manifest must name the
frozen comparator, the single intended change, all input fingerprints, the
candidate cap, the ordering rule, the seed set, and whether official-test
labels were loaded.  Existing candidate pools, scalar-feature shards,
predictions, model freezes, and numerical ledgers remain immutable.

### Prespecified tasks

- **J1:** prior-only Top-1/3/5/10 and MRR for historical cap-10,
  AiZynthFinder-only, LocalRetro-only, and merged pools, both within each pool's
  covered reactions and on the common four-pool covered subset.
- **J2:** within-pool headroom `1 - prior Top-1` and capture rate
  `(augmented Top-1 - matched baseline Top-1) / headroom`, with zero-headroom
  results reported as undefined rather than imputed.
- **J3:** on the 3,985 covered historical-anchor reactions, report paired
  20-seed augmented-minus-baseline Top-1 and MRR in candidate-count bins
  `{1}`, `{2-3}`, `{4-6}`, `{7-9}`, and `{10}` using 10,000
  canonical-product-clustered bootstrap draws with RNG seed 2026.  The
  singleton bin must have exactly zero effect for every seed and metric or the
  analysis fails closed.
- **J4:** stratify the same covered anchor reactions by the frozen prior+2D
  baseline reference rank bins `1`, `2`, `3`, `4-5`, and `6-10`; report
  promotions, degradations, unchanged counts, mean augmented-minus-baseline
  rank change, and rank-1 harm rate.  The exact seed aggregation used to assign
  a reaction to a baseline-rank bin must be approved below before execution.
- **K1:** truncate each frozen expanded pool to its first ten candidates in its
  existing checksummed order, refit training-only normalizers, and fit both
  frozen D1 arms for paired seeds 42--61.  Candidate generation, embedding,
  hyperparameter search, and cross-pool normalization are prohibited.  Report
  coverage@10 plus product-clustered and product-by-seed 95% intervals.
- **K2:** compare historical cap-10 with the clean regenerated cap-50 input
  truncated to ten over the 3,985 anchor-covered reactions.  Classify identical
  set/order, identical set/different order, and different-set groups; report
  all group sizes and the existing 20-seed anchor effect within the first
  group.  The legacy-anchored cap-50 artifact cannot be substituted because
  its first ten entries are the historical anchor by construction.
- **L1 (optional):** diagnose frozen RXN-EBM within-list score dispersion and
  Spearman correlation with prior without retraining or retuning.

### Frozen L1 diagnostic protocol

The repository owner authorized L1 on 2026-08-26 with the explicit instruction
"chạy L1 có lâu k, chạy đi".  L1 uses protocol
`dd-rxn-ebm-score-diagnostic-v1` and is a read-only diagnostic of the already
frozen D5 RXN-EBM checkpoints.  It may run forward scoring only; retraining,
retuning, checkpoint selection, candidate generation, fingerprint generation,
and changes to candidate identities or order are prohibited.  Its outputs must
remain under `outputs/digital_discovery_round_jk/reanalysis/l1/`.

For each of the three published D5 seeds `0`, `20210423`, and `77777777`, L1
reuses the checksummed D5 sparse test-fingerprint matrix and the corresponding
checksummed selected checkpoint.  The external higher-is-better score is the
negative frozen model energy.  The prior higher-is-better score is
`1 - (rank - 1) / max(candidate_count - 1, 1)` in the unchanged candidate
order.  For every list with at least two candidates, L1 reports population
standard deviation and range of the external score and the within-list
Spearman correlation between external and prior scores.  Average ranks are
used for ties.  Singleton lists are counted and excluded from correlation
summaries; constant-score correlations are recorded as undefined and excluded
from correlation means and quantiles rather than imputed.

The frozen summaries are the per-seed and pooled seed--reaction counts; mean,
standard deviation, minimum, 25th percentile, median, 75th percentile, and
maximum for score SD, score range, and defined Spearman correlation; and the
counts of positive, zero, negative, and undefined correlations.  No
post-result threshold or inferential significance test is authorized.  The
single intended change from D5 is retention and descriptive analysis of its
raw frozen within-list energies.

### Frozen K1 controls

The explicit D1 configuration lines in the supervisor checklist are treated as
the intended controls, pending approval:

- baseline: hidden width 128, dropout 0, learning rate `1e-3`, margin 0.3;
- augmented: hidden width 128, dropout 0.1, learning rate `1e-3`, margin 0.1;
- prior transform: raw;
- maximum 200 epochs, patience 20, minimum improvement `1e-5`;
- population-SD training-pair-only normalization with `1e-6` floor and
  clipping to `[-5, 5]`;
- at most five seeded random negative pairs per positive; and
- paired seeds 42--61.

The checklist's separate sentence that these models have 289 parameters is
incompatible with the width-128 D1 configurations.  The 289-parameter models
belong to the D-CAPACITY control (width 48 for four inputs versus width 32 for
seven inputs).  For K1, the runner must assert the width-128 configurations and
record the actual parameter counts for the expanded-pool feature dimensions;
it must not silently substitute D-CAPACITY.  This interpretation requires
explicit supervisor approval before K1 execution.

Each pool retains its already-frozen ordering.  In particular, the merged pool
uses maximum rank-normalized generator prior and canonical candidate identity
as the no-generator-priority tie-break.  K1 truncation takes the first ten rows
only and never re-sorts them.

### Decision rules

- Positive truncated AiZynthFinder effect with a product-clustered interval
  excluding zero supports list length as an explanation for the full-pool
  attenuation.
- An additional positive truncated LocalRetro effect supports transfer across
  generators at fixed list length.
- Null effects in all three truncated pools narrow the result to the historical
  artifact and make K2 the decisive fragility analysis.
- A group-(a) K2 effect close to the overall anchor effect argues against a
  tie-ordering explanation; a materially different effect must be reported as
  primary-result fragility rather than hidden in limitations.
- Much smaller LocalRetro headroom supports a partial ceiling-effect
  explanation, regardless of K1.
- Any non-zero J3 singleton-bin effect stops J3 and is reported as an evaluation
  bug before manuscript work.

### Approval record and execution gates

- [x] Repository owner approved this dated amendment on 2026-08-26 with the
  explicit instruction: "tôi đồng ý, ký và triển khai".
- [x] The width-128 D1 configurations are approved for K1.  The contradictory
  289-parameter sentence is rejected for K1; actual parameter counts must be
  calculated from each expanded-pool input dimension and recorded.
- [x] J4 assigns a reaction using the median prior+2D reference rank across the
  20 paired seeds; an exact half-rank tie is assigned to the worse-rank bin.
- [x] The amendment must be committed before scientific execution.  The commit
  identifier, this file's SHA-256, the approval date, and the platform lock are
  recorded in `docs/round_jk_approval.json` and copied into run manifests.
- [x] Repository owner separately authorized the frozen forward-only L1
  diagnostic on 2026-08-26 with the instruction "chạy L1 có lâu k, chạy đi";
  no retraining or retuning is authorized.

This approval authorizes only the isolated, staged J/K protocols and the
forward-only L1 diagnostic written above. It does not authorize modifying the
frozen historical-anchor outputs or the abandoned predecessor source, nor any
new training or retuning.

## Approved exploratory mechanism round M1/M2

- **Date approved:** 2026-08-26
- **Target:** *Digital Discovery* (exploratory strengthening only)
- **Status:** APPROVED AND SIGNED FOR FROZEN-ARTIFACT ANALYSIS
- **Authorization:** repository owner instruction "triển khai đi"

This round is post hoc and exploratory.  It may read the frozen J/K/L1 and
Journal-of-Cheminformatics artifacts but must write only under
`outputs/digital_discovery_round_jk/reanalysis/`.  It may not train or retune a
model, regenerate a candidate pool or representation, alter a frozen
prediction, select a new threshold after viewing a result, or modify either
manuscript.  A null or adverse result remains reportable evidence.

### M1: candidate-count effect heterogeneity

Protocol `dd-mechanism-heterogeneity-v1` uses the frozen historical cap-10
J3 predictions for the 3,985 covered reactions and paired seeds 42--61.  The
single intended change is exploratory inference across the already frozen
candidate-count strata; predictions, bins, and endpoints are unchanged.

For Top-1 and MRR separately, M1 first averages the paired
augmented-minus-baseline outcome over the 20 seeds for each reaction.  It then
reports:

1. a weighted between-bin omnibus statistic over all five J3 bins;
2. the same omnibus statistic over the four non-singleton bins;
3. the primary contrast between bin 4--6 and the pooled non-singleton bins
   2--3, 7--9, and 10; and
4. secondary contrasts between bin 4--6 and each of those three bins.

Omnibus and contrast p-values use 100,000 reaction-label permutations with RNG
seeds 2030 for Top-1 and 2031 for MRR.  The primary contrast also receives a
10,000-draw within-group reaction bootstrap 95% interval with RNG seeds 2040
and 2041.  Secondary pairwise permutation p-values are Holm-adjusted within
each endpoint.  The singleton bin is retained as a structural audit and must
remain exactly zero; it is not used in the primary contrast.  These p-values
are exploratory and cannot be described as prespecified confirmation of the
original effect.

### M2: frozen candidate-pool shift diagnostic

Protocol `dd-candidate-pool-shift-diagnostic-v2` compares historical cap-10,
AiZynthFinder-only cap-10, LocalRetro-only cap-10, and merged cap-10 on their
3,814-reaction common frozen covered subset.  The single intended change is
descriptive measurement of candidate-list composition and its association
with frozen within-pool effect and paired loss of transfer; no ranking model
is refit.

- **M2 amendment approved:** 2026-08-27
- **Status:** APPROVED AND SIGNED BEFORE FIRST M2 EXECUTION
- **Authorization:** repository owner instruction "Triển khai đi"

M2 must fail closed unless each supplied pool JSONL is bound by SHA-256 and
size to the corresponding frozen K1 selection/model-freeze/test-manifest
chain.  The K1 protocol, pool name, cap, candidate order, seeds 42--61, source
and metadata fingerprints, 5,004-row test count, and post-freeze test gate
must match.  On every common reaction, the truncated list length must equal
the frozen `candidate_count`, the recorded reference reactants must be
present, and their list position must equal the frozen `prior_rank`.

For each reaction and pool, M2 records candidate count, set Jaccard overlap
with the historical list, shared-candidate count, Kendall order concordance
over shared candidates when at least two shared identities exist, normalized
entropy of stored prior-feature mass, mean pairwise Morgan-Tanimoto distance,
and maximum Morgan-Tanimoto similarity between the recorded reference
reactants and any candidate not matching that reference.  Candidate
identity is canonical isomeric SMILES with fragment-order invariance.  Morgan
fingerprints use RDKit radius 2, 2,048 bits, and chirality enabled only for this
chemical-list similarity diagnostic.  Undefined entropy, concordance, or
similarity values remain missing and are never imputed.

Historical priors are raw AiZynthFinder expansion-policy values, whereas the
expanded single-generator pools use linear rank scores and the merged pool
uses the maximum generator-specific rank score.  Stored-prior entropy is
therefore a within-pool mass-dispersion descriptor, not calibrated generator
confidence or uncertainty; its absolute values must not be interpreted as a
common probability scale across pool families.

M2 links each frozen list descriptor to the corresponding reaction-level mean
20-seed augmented-minus-baseline Top-1 and MRR using Spearman correlation and
quartile summaries.  It also links frozen L1 per-list score correlation and
dispersion to the D5 external-reranker rank shift.  All associations are
descriptive/post hoc; no causal or external-validity claim is permitted.

Without replacing that approved absolute-effect analysis, M2 additionally
reports a paired loss-of-transfer diagnostic for each expanded pool.  The
outcomes are its reaction-level mean Top-1 and MRR effects minus the matching
historical effects.  Predictors are Jaccard overlap, shared-candidate count,
Kendall concordance, and the expanded-minus-historical shifts in candidate
count, mean pairwise Morgan distance, and maximum non-reference similarity.
Stored-prior entropy is excluded from this cross-pool shift analysis because
the frozen prior semantics are not commensurate.  These paired associations
use Spearman correlation and quartile summaries and remain exploratory,
post hoc, conditional on common top-10 reference coverage, and non-causal.

M2 necessarily loads the frozen official-test reference reactants after all
models and predictions are frozen, solely for common-coverage containment
checks and the label-aware non-reference similarity diagnostic.  Official
test labels are not used for training, selection, retuning, or threshold
choice.  The manifest must state this use explicitly and report the full
5,004-to-3,814 selection flow; it must not claim that test information came
only from prediction files.

### M-round gate

- [x] Repository owner authorized M1/M2 on 2026-08-26 with "triển khai đi".
- [x] Only frozen artifacts may be read and only isolated M outputs may be
  written.
- [x] The committed plan identifier and SHA-256 must be copied into
  `docs/round_jk_approval.json` before M1/M2 execution.
- [ ] Fresh external or end-to-end validation is outside this authorization and
  requires a separately selected untouched dataset and protocol.

## Approved Digital Discovery final inference/synchronization round

- **Date approved:** 2026-08-27
- **Status:** APPROVED AND SIGNED FOR FROZEN-ARTIFACT ANALYSIS
- **Authorization:** repository owner instruction "thực hiện nốt M2b, đồng bộ
  J1/J2 đi"

This round remains isolated under `outputs/digital_discovery_round_jk/` and
does not train or retune a model, regenerate a candidate pool or
representation, alter a frozen prediction, or modify either manuscript.

### M2b: direct paired transfer-loss inference

Protocol `dd-candidate-pool-transfer-inference-v1` reads only the checksummed
M2 v2 manifest and its 3,814-reaction transfer-loss table.  It fails closed
unless all three expanded pools contain the same unique reaction IDs and
canonical products.  For each pool and each of Top-1 and MRR, the frozen
reaction-level 20-seed mean transfer loss receives a 10,000-draw paired
canonical-product bootstrap percentile 95% interval and a two-sided 100,000-
draw canonical-product sign-flip test.  Bootstrap RNG seeds are 2050--2055 and
sign-flip RNG seeds are 2060--2065 in AiZynthFinder, LocalRetro, merged order
and Top-1, MRR order.  These direct contrasts are exploratory/post hoc and do
not replace the already frozen K1 intervals or M2 descriptive diagnostics.

### J1/J2 filtered-v2 synchronization

Protocol `dd-round-jk-j1-j2-filtered-v2` recomputes J1/J2 only from the frozen
historical anchor and the three checksummed final K1 cap-10 test manifests
under source-pool protocol `ws-e-localretro-three-pools-filtered-v2`.  It must
verify seeds 42--61, candidate cap 10, unchanged candidate order, 5,004 total
test reactions, own covered counts 4,012/4,462/4,584, and exactly 3,814 common
covered reactions.  Historical coverage remains 3,985.  The older 3,939-common
J1/J2 artifact is preserved as superseded exploratory output and must not be
mixed with the filtered-v2 tables.  The single intended change is alignment of
J1/J2 reporting to the final frozen K1 pool version; no scientific prediction
is recomputed.
