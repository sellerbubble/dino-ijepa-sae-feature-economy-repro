# Extending Models, Layers, Tasks, And Analyses

The public design goal is config-first extension.

To add a model:

1. Add a model config in `configs/models/`.
2. Add or reuse a backbone adapter. For local/I-JEPA-style checkpoints, prefer
   exporting a TorchScript feature module first; see
   `docs/export_torchscript_backbones.md`.
3. Add a tiny smoke config before running full experiments.
4. Run `feature-economy check-configs --config-root configs`.
5. Run `feature-economy plan-runs --config-root configs --output-json /tmp/run_plan.json`
   and confirm the new model appears in the intended stages.

To add an SAE:

1. Add an SAE config in `configs/saes/`.
2. Record model id, layer, top-k, expansion, training data, and activation
   normalization.
3. Validate that usage and ranking artifacts include the SAE id.
4. Run `feature-economy check-configs --config-root configs`.
5. Run `feature-economy plan-runs --config-root configs --output-json /tmp/run_plan.json`
   and confirm the new SAE appears in the intended model/task rows.

To add a task:

1. Add a task config in `configs/tasks/`.
2. Add a dataset adapter and metric definition.
3. Add native and SAE probe smoke tests before running full data.
4. Run `feature-economy check-configs --config-root configs`.
5. Run `feature-economy plan-runs --config-root configs --output-json /tmp/run_plan.json`
   and confirm the new task appears in native and SAE-chain rows.

To add an analysis module:

1. Define its input artifacts.
2. Define its output schema.
3. Add one paper-table or paper-figure consumer only after the schema is stable.
4. Decide whether the module should appear in the run plan; if yes, update the
   planner and add a test before adding private-scale launch scripts.
