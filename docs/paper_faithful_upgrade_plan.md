# Paper-Faithful Agent-Friendly Upgrade Plan

Date: 2026-05-13

This plan upgrades the public reproduction repository from a
lightweight-first executable slice into a paper-faithful, agent-friendly
experiment system. The current public repo already validates the main command
surface and real-slice execution for DINO/I-JEPA across ImageNet, NYUv2,
ADE20K, and CLEVR/Count. The next step is to port the paper-scale training and
mechanism modules in a controlled way without turning the repository into a
private-workbench dump.

## Target State

The upgraded public repo should support three clearly separated modes:

| Mode | Default probe/trainer | Purpose | Required for paper-style claims? |
| --- | --- | --- | --- |
| `smoke` | lightweight / fixture / closed-form probes | Code health, CI, command-surface checks, tiny artifact contracts. | No |
| `full_paper_style` | paper-scale trainers and paper-style probe configs | Reproduce the main paper evidence chain from data/checkpoints/features. | Yes |
| `advanced_mechanism` | paper-scale trainer outputs plus specialized interventions | Robustness, layer sweeps, and native-space controls. | Used for advanced claims |

The lightweight closed-form probes remain valuable, but they should be
documented as diagnostics. The full rerun path should use paper-scale trainers
where the private experiment recipe is understood well enough to expose
portable configs, stable artifact outputs, and validation gates.

## Inclusion Boundary

### Must Enter The Public Paper-Faithful Path

These components are required for a faithful public reproduction of the main
paper chain:

- Paper-scale native probes for ImageNet-1K, NYUv2, ADE20K, and CLEVR/Count.
- Paper-scale SAE-code probes for the same four tasks.
- Module E Allocation / bucket ablation on paper-scale SAE-code probe outputs.
- Shared probe output contracts that preserve the fields needed by ranking,
  contribution scoring, ablation, artifact indexing, and table export.
- Config files that record the paper reference hyperparameters, split
  assumptions, metrics, seed policy, checkpoint-selection rule, and expected
  outputs.

### Should Enter As Robustness Or Sweep Modules

These modules strengthen the paper but should be routed through sweep configs
rather than one-off scripts:

- Alternative ranking controls:
  `probe_weight`, `validation_contribution`, and `hybrid`.
- Layer sweep:
  representative layers, second-last layers, and optional all-layer
  diagnostics should share one runner and config schema.
- Second-last comparisons:
  treated as named contrasts inside the layer sweep, not as separate scripts.

### Should Enter As Advanced Mechanism Modules

These modules are important but conceptually and computationally heavier:

- Module F native-space ablation.
- Task-conditioned native subset analysis.
- Final-layer versus second-last native intervention comparisons.

The public implementation must preserve the corrected intervention coordinate:
SAE runtime-normalized hidden states centered by `b_dec`, for example
`layer_norm(x) - b_dec` for the current layer-normalized SAEs. Random controls
must exclude task-selected top-k features when the private analysis requires
that exclusion.

### Excluded From This Upgrade Round

The following remains outside the default public upgrade unless explicitly
reopened later:

- NYUv2 perturbation-response analysis.
- Manual qualitative annotation and family-board review as a required
  executable pipeline.
- Private cluster launchers, hardcoded remote paths, and unmanaged historical
  one-off scripts.
- Final manuscript figure recreation as a required reproduction gate.

These can still be documented as optional private-workbench or future-public
extensions.

## Canonical Experiment Families

The upgraded repo should converge on these experiment families rather than
accumulating near-duplicate launchers:

| Family | Scope | Default location | Notes |
| --- | --- | --- | --- |
| `probe_train` | Paper-scale native and SAE-code probes. | `configs/probes/`, `src/feature_economy/probes/` | Full rerun default. |
| `allocation` | Bucket and subset ablation. | `configs/analyses/`, `src/feature_economy/analysis/` | Consumes paper-scale SAE probes. |
| `ranking_control` | Ranking-method robustness. | `configs/sweeps/` | Runs `probe_weight`, `validation_contribution`, `hybrid`. |
| `layer_sweep` | Representative, second-last, and all-layer diagnostics. | `configs/sweeps/` | One config family, not separate ad hoc scripts. |
| `native_ablation` | Module F native-space intervention. | `configs/advanced/` | Advanced mechanism module. |

## Proposed Public Config Shape

Paper-scale probe configs should be explicit but not hardcoded to one machine:

```yaml
probe_id: nyuv2_depth_paper_scale
task_id: nyuv2_depth
probe_family: dense_depth
input_space: native_or_sae
paper_reference:
  source_private_runner: scripts/benchmarks/run_nyuv2_depth_probe.py
  status: ported_from_private_recipe
training:
  backend: torch
  epochs: 20
  batch_size: 64
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
  scheduler: cosine
  seed: 0
selection:
  checkpoint_rule: best_validation
metrics:
  primary: rmse
  report:
    - rmse
    - abs_rel
    - delta1
outputs:
  summary: probe_summary.json
  checkpoint: probe.pt
  predictions: probe_outputs.npz
```

Exact values must come from the private canonical-trainer audit before this
schema is used as paper reference. Until then, configs should be marked
`draft_recipe`.

## Agent-Friendly Repository Contract

The public repo can be complex if each directory tells agents what belongs
there. Add nested `AGENTS.md` files as modules are ported:

| Path | Required guidance |
| --- | --- |
| `configs/AGENTS.md` | Config naming, required fields, when to add a new config versus extend an existing one. |
| `src/feature_economy/AGENTS.md` | Separation between data, extraction, probes, analyses, artifacts, and paper exports. |
| `src/feature_economy/probes/AGENTS.md` | Probe backend responsibilities, output contracts, validation rules. |
| `src/feature_economy/analysis/AGENTS.md` | Analysis-module input/output schema and table/figure consumer rules. |
| `scripts/AGENTS.md` | Launcher discipline: thin dispatch only, no hardcoded private paths. |
| `docs/AGENTS.md` | Documentation hierarchy and evidence-provenance rules. |
| `experiments/AGENTS.md` | If added, experiment manifests and runbooks live here, not arbitrary script copies. |

These files should be concise. Their purpose is to prevent agents from creating
new research ruins: duplicated scripts, unclear outputs, and undocumented
scientific distinctions.

## Upgrade Phases

### Phase 1: Write And Freeze This Plan

Status: in progress.

Deliverables:

- `docs/paper_faithful_upgrade_plan.md`.
- Link from roadmap/scope docs.
- No code migration yet.

Validation:

- Markdown link/hygiene checker.
- No generated artifacts or private paths added.

### Phase 2: Audit Private Canonical Trainers

Goal:

Identify the private scripts and configs that define the paper-scale probe
recipes.

Expected audit output:

```text
docs/private_trainer_audit_YYYYMMDD.md
```

For each task and input space, record:

- canonical private runner;
- native versus SAE-code path;
- train/validation split;
- feature/code input shape;
- model/probe architecture;
- optimizer, epochs, batch size, scheduler, seed;
- checkpoint-selection rule;
- metric implementation;
- output files consumed by ranking, ablation, tables, and paper text;
- deprecated or superseded private branches.

No public code should be ported until this audit identifies the source of
truth.

### Phase 3: Design Probe Config Schema And Output Contract

Goal:

Create stable public configs before moving trainer code.

Deliverables:

- `configs/probes/*.yaml`.
- `docs/probe_config_schema.md`.
- Updated `docs/artifact_schemas.md` if paper-scale trainers need additional
  fields.
- A probe-config validation path in `check-configs` or a dedicated checker.

Acceptance criteria:

- A config can describe both native and SAE-code probes.
- Outputs remain compatible with ranking, contribution scoring, Allocation
  ablation, artifact indexing, and table export.
- Lightweight and paper-scale probes can coexist without ambiguous names.

### Phase 4: Port Paper-Scale Trainers

Goal:

Make `full_paper_style` profiles use paper-scale trainers by default while
keeping lightweight probes for `smoke`.

Status:

- Implemented for classification/counting, dense-depth, and dense-segmentation
  saved-array probes through the public `paper-scale-torch` backend.
- `scripts/run_full_profile.sh` now defaults to `PROBE_BACKEND=paper-scale-torch`
  and extracts train/validation features, targets, and SAE codes before native
  and SAE-code probe training.
- `PROBE_BACKEND=linear-probe` remains available for lightweight diagnostics.

Deliverables:

- Torch training backends under `src/feature_economy/probes/`.
- CLI entries for paper-scale training/evaluation.
- Updated `scripts/run_full_profile.sh` or a sibling full-profile launcher that
  selects paper-scale probes by default.
- Tiny fixture or dry-run validation for every task family.

Acceptance criteria:

- ImageNet, NYUv2, ADE20K, and CLEVR/Count each have native and SAE-code
  paper-scale probe commands.
- Probe summaries clearly identify `native_backbone_probe`,
  `sae_code_probe`, `no_ablation_baseline`, or `stored_best_checkpoint`.
- Seeds and checkpoint-selection rules are recorded in run manifests.

### Phase 5: Add Ranking-Control Sweep

Goal:

Test whether Allocation conclusions depend on the default hybrid top-k ranking.

Deliverables:

- `configs/sweeps/ranking_control_paper.yaml`.
- Runner for `probe_weight`, `validation_contribution`, and `hybrid` top-k
  selections.
- Summary tables comparing selected high-usage core results across ranking
  methods.

Acceptance criteria:

- The same task/model/SAE/probe outputs can be reused across ranking methods.
- Results are reported as a robustness control, not a separate main pipeline.

### Phase 6: Add Layer Sweep

Goal:

Unify representative-layer, second-last, and optional all-layer diagnostics.

Deliverables:

- `configs/sweeps/layer_sweep_paper.yaml`.
- A runner that dispatches the same Availability/Access/Allocation analyses
  across configured model layers and SAE checkpoints.
- Named contrasts for `primary_vs_second_last` and any representative-layer
  comparison used in the paper.

Acceptance criteria:

- Second-last experiments are not separate ad hoc launchers.
- Layer outputs share artifact schemas with the main layer.
- The sweep can run partial layer subsets for affordable validation.

### Phase 7: Add Module F Native-Space Ablation

Goal:

Expose the strongest anti-artifact / native-correlate control as an advanced
module.

Deliverables:

- `configs/advanced/module_f_native_ablation_nyuv2.yaml`.
- Native subspace ablation runner.
- Documentation of the corrected intervention coordinate and random-control
  exclusion policy.

Acceptance criteria:

- The runner refuses unsupported SAE normalization modes unless implemented.
- Summary records the coordinate system, normalization mode, selected feature
  source, random pool policy, and target probe checkpoint.
- DINO and I-JEPA interpretation caveats are documented.

### Phase 8: Complete Agent Documentation And CI Gates

Goal:

Make the complex repo navigable for future agents and contributors.

Deliverables:

- Nested `AGENTS.md` files listed above.
- Updated `docs/full_reproduction.md`, `docs/extending_models_tasks.md`, and
  `docs/public_release_audit_20260513.md`.
- CI or local validation commands for new configs, trainers, sweeps, and
  advanced modules.

Acceptance criteria:

- Every new non-trivial runner has role/status/input/output/safe-to-move
  metadata.
- Every public experiment family has a runbook and expected artifact layout.
- `scripts/check_public_release.py` passes.
- Unit tests and the smoke chain still pass.

## Safety Rules

- Do not move private workbench files into the public repo before auditing their
  canonical status.
- Do not copy private hardcoded paths, remote usernames, or cluster launch
  assumptions into public runners.
- Do not overwrite remote artifacts during validation; use new directories and
  new artifact roots.
- Keep lightweight diagnostics alive. They are not paper-scale claims, but they
  are essential for CI and debugging.
- Prefer config expansion over script copying. If a third similar runner appears
  likely, stop and design a shared runner or sweep config.

## Current Open Questions

- Which private trainer is canonical for each task after later paper revisions?
- Which paper-scale hyperparameters should be public defaults versus suggested
  resource-dependent settings?
- Should the public full-profile launcher switch in place, or should it expose
  `PROBE_MODE=paper_scale|lightweight` during a transition period?
- Which Module F cases beyond NYUv2 should enter public advanced configs?
- How much full-data validation is required before tagging a paper-faithful
  public release?
