"""CLI for the public Feature Economy reproduction package."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-economy",
        description="DINO/I-JEPA SAE Feature Economy reproduction commands.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    check_configs = subparsers.add_parser(
        "check-configs",
        help="Validate YAML configs without requiring datasets or checkpoints.",
    )
    check_configs.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs"),
        help="Directory containing models/, saes/, tasks/, and experiments/.",
    )
    plan_runs = subparsers.add_parser(
        "plan-runs",
        help="Expand experiment configs into a JSON/CSV reproduction run matrix.",
    )
    plan_runs.add_argument("--config-root", type=Path, default=Path("configs"))
    plan_runs.add_argument("--output-json", type=Path, required=True)
    plan_runs.add_argument("--output-csv", type=Path, default=None)
    check_bundle = subparsers.add_parser(
        "check-bundle",
        help="Check an artifact root against a reproduction run plan.",
    )
    check_bundle.add_argument("--run-plan-json", type=Path, required=True)
    check_bundle.add_argument("--artifact-root", type=Path, required=True)
    check_bundle.add_argument("--output-json", type=Path, default=None)
    check_bundle.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with an error if any planned artifact row is incomplete.",
    )
    check_models = subparsers.add_parser(
        "check-models",
        help="Validate model/SAE registry metadata and placeholder paths.",
    )
    check_models.add_argument("--config-root", type=Path, default=Path("configs"))
    check_models.add_argument(
        "--require-resolved-checkpoints",
        action="store_true",
        help="Require SAE checkpoint placeholders such as ${SAE_ROOT} to resolve.",
    )
    check_runtime = subparsers.add_parser(
        "check-runtime",
        help="Check importability of runtime dependencies.",
    )
    check_runtime.add_argument(
        "--profile",
        default="smoke",
        choices=["smoke", "base", "figures", "experiments"],
        help="Dependency profile to check.",
    )
    check_runtime.add_argument(
        "--require",
        action="store_true",
        help="Exit with an error if any dependency in the profile is missing.",
    )
    check_runtime.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable dependency report.",
    )
    validate_arrays = subparsers.add_parser(
        "validate-arrays",
        help="Validate public .npz array contracts before running probes or analysis.",
    )
    validate_arrays.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Array file to validate.",
    )
    validate_arrays.add_argument(
        "--kind",
        required=True,
        choices=["features", "codes", "probe_logits", "targets", "contribution_scores"],
        help="Public array contract to validate.",
    )
    validate_arrays.add_argument(
        "--task-type",
        default=None,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
        help="Task type, required for probe_logits and targets.",
    )
    validate_arrays.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest used to verify first-dimension alignment.",
    )
    validate_arrays.add_argument("--expected-split", default=None)
    validate_arrays.add_argument(
        "--target-key",
        default="targets",
        help="Target array key for --kind targets.",
    )
    validate_arrays.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable validation report.",
    )
    make_tables = subparsers.add_parser(
        "make-tables",
        help="Build paper-facing CSV tables from saved public artifacts.",
    )
    make_tables.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing public JSON artifacts.",
    )
    make_tables.add_argument(
        "--artifact-index",
        type=Path,
        default=None,
        help="Optional artifact_index.json. Prefer this for paper reproduction.",
    )
    make_tables.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV tables should be written.",
    )
    make_figures = subparsers.add_parser(
        "make-figures",
        help="Build overview figures from public CSV tables.",
    )
    make_figures.add_argument(
        "--table-dir",
        type=Path,
        required=True,
        help="Directory containing CSV tables from make-tables.",
    )
    make_figures.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where figures should be written.",
    )
    make_figures.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats, e.g. png,pdf,svg.",
    )
    smoke_probe = subparsers.add_parser(
        "smoke-probe",
        help="Write deterministic smoke probe artifacts from an experiment config.",
    )
    smoke_probe.add_argument(
        "--config-root",
        type=Path,
        required=True,
        help="Directory containing public configs.",
    )
    smoke_probe.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment id to materialize as smoke artifacts.",
    )
    smoke_probe.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where smoke artifacts should be written.",
    )
    smoke_analysis = subparsers.add_parser(
        "smoke-analysis",
        help="Write deterministic smoke AAA analysis artifacts from public configs.",
    )
    smoke_analysis.add_argument(
        "--config-root",
        type=Path,
        required=True,
        help="Directory containing public configs.",
    )
    smoke_analysis.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where smoke analysis artifacts should be written.",
    )
    compute_usage = subparsers.add_parser(
        "compute-usage",
        help="Compute feature usage artifacts. Currently supports smoke mode.",
    )
    compute_usage.add_argument("--config-root", type=Path, required=True)
    compute_usage.add_argument("--output-dir", type=Path, required=True)
    compute_usage.add_argument("--codes-npz", type=Path, default=None)
    compute_usage.add_argument("--model-id", default=None)
    compute_usage.add_argument("--sae-id", default=None)
    compute_usage.add_argument("--dataset-id", default=None)
    compute_usage.add_argument("--split", default="val")
    compute_usage.add_argument("--threshold", type=float, default=0.0)
    compute_usage.add_argument(
        "--smoke",
        action="store_true",
        help="Write deterministic smoke availability artifacts.",
    )
    rank_features = subparsers.add_parser(
        "rank-features",
        help="Rank task features from smoke artifacts or linear-probe weights.",
    )
    rank_features.add_argument("--config-root", type=Path, required=True)
    rank_features.add_argument("--output-dir", type=Path, required=True)
    rank_features.add_argument("--codes-npz", type=Path, default=None)
    rank_features.add_argument("--probe-logits-npz", type=Path, default=None)
    rank_features.add_argument("--task-id", default=None)
    rank_features.add_argument("--model-id", default=None)
    rank_features.add_argument("--sae-id", default=None)
    rank_features.add_argument("--top-k", type=int, default=100)
    rank_features.add_argument(
        "--ranking-method",
        default="probe_weight",
        choices=["probe_weight", "validation_contribution", "hybrid"],
        help="Feature ranking score to use for real saved-array ranking.",
    )
    rank_features.add_argument(
        "--contribution-npz",
        type=Path,
        default=None,
        help="Optional validation-contribution score array for alternative ranking controls.",
    )
    rank_features.add_argument(
        "--contribution-key",
        default="validation_contribution_score",
        help="Array key inside --contribution-npz.",
    )
    rank_features.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Probe-weight mixture weight for --ranking-method hybrid.",
    )
    rank_features.add_argument(
        "--smoke",
        action="store_true",
        help="Write deterministic smoke ranking and subset-usage artifacts.",
    )
    compute_contributions = subparsers.add_parser(
        "compute-contributions",
        help="Compute per-feature validation contribution scores from a saved linear probe.",
    )
    compute_contributions.add_argument("--config-root", type=Path, required=True)
    compute_contributions.add_argument("--output-dir", type=Path, required=True)
    compute_contributions.add_argument("--codes-npz", type=Path, required=True)
    compute_contributions.add_argument("--probe-logits-npz", type=Path, required=True)
    compute_contributions.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    compute_contributions.add_argument("--task-id", required=True)
    compute_contributions.add_argument("--model-id", required=True)
    compute_contributions.add_argument("--sae-id", required=True)
    compute_contributions.add_argument(
        "--primary-metric",
        default=None,
        help="Metric whose positive performance drop defines contribution.",
    )
    compute_contributions.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap for smoke/debug runs. Unscored feature scores remain zero.",
    )
    compute_contributions.add_argument(
        "--score-key",
        default="validation_contribution_score",
        help="Array key to write inside contribution_scores.npz.",
    )
    compute_contributions.add_argument(
        "--scoring-method",
        choices=["exact_metric_drop", "true_class_logit_drop", "weight_activation"],
        default="exact_metric_drop",
        help=(
            "Contribution score definition. exact_metric_drop reruns the metric after "
            "single-feature ablations; true_class_logit_drop is a scalable "
            "classification-only proxy for full ImageNet bundles; weight_activation "
            "is a scalable dense-task proxy using probe weight magnitude times mean activation."
        ),
    )
    subset_usage = subparsers.add_parser(
        "compute-subset-usage",
        help="Compare task-selected features with a fired-count matched random subset.",
    )
    subset_usage.add_argument("--config-root", type=Path, required=True)
    subset_usage.add_argument("--output-dir", type=Path, required=True)
    subset_usage.add_argument("--codes-npz", type=Path, required=True)
    subset_usage.add_argument("--ranking-json", type=Path, required=True)
    subset_usage.add_argument("--top-k", type=int, default=100)
    subset_usage.add_argument("--random-seed", type=int, default=0)
    subset_usage.add_argument("--threshold", type=float, default=0.0)
    subset_usage.add_argument("--high-usage-threshold", type=int, default=101)
    ablate_features = subparsers.add_parser(
        "ablate-features",
        help="Run smoke or linear-probe SAE feature ablation.",
    )
    ablate_features.add_argument("--config-root", type=Path, required=True)
    ablate_features.add_argument("--output-dir", type=Path, required=True)
    ablate_features.add_argument(
        "--backend",
        default="smoke",
        choices=["smoke", "linear-probe"],
        help="Ablation backend.",
    )
    ablate_features.add_argument("--codes-npz", type=Path, default=None)
    ablate_features.add_argument("--probe-logits-npz", type=Path, default=None)
    ablate_features.add_argument("--ranking-json", type=Path, default=None)
    ablate_features.add_argument(
        "--task-type",
        default=None,
        choices=[
            "classification",
            "count_classification",
            "dense_depth",
            "dense_segmentation",
        ],
        help="Task type for --backend linear-probe.",
    )
    ablate_features.add_argument("--top-k", type=int, default=100)
    ablate_features.add_argument("--random-seed", type=int, default=0)
    ablate_features.add_argument(
        "--smoke",
        action="store_true",
        help="Write deterministic smoke ablation artifacts.",
    )
    native_ablation = subparsers.add_parser(
        "ablate-native-subspace",
        help="Run Module F native hidden-state subspace ablation over saved arrays.",
    )
    native_ablation.add_argument("--config-root", type=Path, required=True)
    native_ablation.add_argument("--features-npz", type=Path, required=True)
    native_ablation.add_argument("--sae-checkpoint", type=Path, required=True)
    native_ablation.add_argument("--probe-logits-npz", type=Path, required=True)
    native_ablation.add_argument("--ranking-json", type=Path, required=True)
    native_ablation.add_argument("--output-dir", type=Path, required=True)
    native_ablation.add_argument(
        "--task-type",
        required=True,
        choices=[
            "classification",
            "count_classification",
            "dense_depth",
            "dense_segmentation",
        ],
    )
    native_ablation.add_argument("--top-k", type=int, default=20)
    native_ablation.add_argument("--random-seed", type=int, default=0)
    native_ablation.add_argument(
        "--random-pool",
        choices=["dictionary", "exclude_topk"],
        default="exclude_topk",
    )
    native_ablation.add_argument(
        "--normalize-activations",
        choices=["none", "layer_norm"],
        default="layer_norm",
    )
    native_ablation.add_argument("--sae-topk", type=int, default=32)
    eval_native_fixture = subparsers.add_parser(
        "eval-native-fixture",
        help="Evaluate precomputed fixture predictions as a native probe summary.",
    )
    eval_native_fixture.add_argument("--manifest", type=Path, required=True)
    eval_native_fixture.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    eval_native_fixture.add_argument("--task-id", required=True)
    eval_native_fixture.add_argument("--model-id", required=True)
    eval_native_fixture.add_argument("--output-dir", type=Path, required=True)
    eval_native_fixture.add_argument("--seed", type=int, default=0)
    eval_native_fixture.add_argument("--expected-split", default=None)
    eval_native_fixture.add_argument("--num-classes", type=int, default=None)
    eval_native_fixture.add_argument("--ignore-index", type=int, default=None)
    probe_native = subparsers.add_parser(
        "probe-native",
        help="Run the public native-probe interface.",
    )
    probe_native.add_argument(
        "--backend",
        default="fixture",
        choices=["fixture", "linear-probe", "paper-scale-torch"],
        help="Native probe backend.",
    )
    probe_native.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Manifest containing precomputed predictions for fixture evaluation.",
    )
    probe_native.add_argument(
        "--features-npz",
        type=Path,
        default=None,
        help="Saved native features for --backend linear-probe or val features for paper-scale.",
    )
    probe_native.add_argument(
        "--train-features-npz",
        type=Path,
        default=None,
        help="Saved train native features for --backend paper-scale-torch.",
    )
    probe_native.add_argument(
        "--val-features-npz",
        type=Path,
        default=None,
        help="Saved validation native features for --backend paper-scale-torch.",
    )
    probe_native.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Label manifest aligned with --features-npz for lightweight probes.",
    )
    probe_native.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Train manifest for --backend paper-scale-torch.",
    )
    probe_native.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Validation manifest for --backend paper-scale-torch.",
    )
    probe_native.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    probe_native.add_argument("--task-id", required=True)
    probe_native.add_argument("--model-id", required=True)
    probe_native.add_argument("--output-dir", type=Path, required=True)
    probe_native.add_argument("--seed", type=int, default=0)
    probe_native.add_argument("--expected-split", default=None)
    probe_native.add_argument("--num-classes", type=int, default=None)
    probe_native.add_argument("--ignore-index", type=int, default=None)
    probe_native.add_argument("--ridge", type=float, default=1e-3)
    probe_native.add_argument("--epochs", type=int, default=20)
    probe_native.add_argument("--batch-size", type=int, default=512)
    probe_native.add_argument("--lr", type=float, default=1e-3)
    probe_native.add_argument("--weight-decay", type=float, default=1e-4)
    probe_native.add_argument("--device", default="cpu")
    probe_native.add_argument(
        "--targets-npz",
        type=Path,
        default=None,
        help="Dense target array for --backend linear-probe depth/segmentation.",
    )
    probe_native.add_argument(
        "--train-targets-npz",
        type=Path,
        default=None,
        help="Train dense targets for --backend paper-scale-torch.",
    )
    probe_native.add_argument(
        "--val-targets-npz",
        type=Path,
        default=None,
        help="Validation dense targets for --backend paper-scale-torch.",
    )
    probe_native.add_argument("--target-key", default="targets")
    probe_native.add_argument("--decoder-hidden-channels", type=int, default=256)
    probe_sae = subparsers.add_parser(
        "probe-sae",
        help="Run the public SAE-code probe interface.",
    )
    probe_sae.add_argument(
        "--backend",
        default="fixture",
        choices=["fixture", "linear-probe", "paper-scale-torch"],
        help="SAE probe backend.",
    )
    probe_sae.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Manifest containing precomputed predictions for fixture evaluation.",
    )
    probe_sae.add_argument(
        "--codes-npz",
        type=Path,
        default=None,
        help="Saved SAE codes for --backend linear-probe or val codes for paper-scale.",
    )
    probe_sae.add_argument(
        "--train-codes-npz",
        type=Path,
        default=None,
        help="Saved train SAE codes for --backend paper-scale-torch.",
    )
    probe_sae.add_argument(
        "--val-codes-npz",
        type=Path,
        default=None,
        help="Saved validation SAE codes for --backend paper-scale-torch.",
    )
    probe_sae.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Label manifest aligned with --codes-npz for lightweight probes.",
    )
    probe_sae.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Train manifest for --backend paper-scale-torch.",
    )
    probe_sae.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Validation manifest for --backend paper-scale-torch.",
    )
    probe_sae.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    probe_sae.add_argument("--task-id", required=True)
    probe_sae.add_argument("--model-id", required=True)
    probe_sae.add_argument("--sae-id", required=True)
    probe_sae.add_argument("--output-dir", type=Path, required=True)
    probe_sae.add_argument("--seed", type=int, default=0)
    probe_sae.add_argument("--expected-split", default=None)
    probe_sae.add_argument("--num-classes", type=int, default=None)
    probe_sae.add_argument("--ignore-index", type=int, default=None)
    probe_sae.add_argument("--ridge", type=float, default=1e-3)
    probe_sae.add_argument("--epochs", type=int, default=20)
    probe_sae.add_argument("--batch-size", type=int, default=512)
    probe_sae.add_argument("--lr", type=float, default=1e-3)
    probe_sae.add_argument("--weight-decay", type=float, default=1e-4)
    probe_sae.add_argument("--device", default="cpu")
    probe_sae.add_argument(
        "--targets-npz",
        type=Path,
        default=None,
        help="Dense target array for --backend linear-probe depth/segmentation.",
    )
    probe_sae.add_argument(
        "--train-targets-npz",
        type=Path,
        default=None,
        help="Train dense targets for --backend paper-scale-torch.",
    )
    probe_sae.add_argument(
        "--val-targets-npz",
        type=Path,
        default=None,
        help="Validation dense targets for --backend paper-scale-torch.",
    )
    probe_sae.add_argument("--target-key", default="targets")
    probe_sae.add_argument("--decoder-hidden-channels", type=int, default=256)
    extract_features = subparsers.add_parser(
        "extract-features",
        help=(
            "Extract model features from fixture manifests or supported real backbones."
        ),
    )
    extract_features.add_argument("--config-root", type=Path, default=Path("configs"))
    extract_features.add_argument(
        "--backend",
        default="fixture",
        choices=["fixture", "huggingface", "torchscript"],
        help=(
            "Extraction backend. `huggingface` covers HF-hosted DINO-style models; "
            "`torchscript` covers exported local/I-JEPA feature modules."
        ),
    )
    extract_features.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Manifest for deterministic fixture extraction.",
    )
    extract_features.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Real dataset manifest for backbone extraction.",
    )
    extract_features.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    extract_features.add_argument("--model-id", required=True)
    extract_features.add_argument("--output-dir", type=Path, required=True)
    extract_features.add_argument("--layer", type=int, default=None)
    extract_features.add_argument("--expected-split", default=None)
    extract_features.add_argument("--feature-dim", type=int, default=8)
    extract_features.add_argument("--batch-size", type=int, default=16)
    extract_features.add_argument("--device", default="cpu")
    extract_features.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    extract_features.add_argument("--max-examples", type=int, default=None)
    extract_features.add_argument("--local-files-only", action="store_true")
    extract_features.add_argument(
        "--hf-name-or-path",
        default=None,
        help=(
            "Optional HuggingFace model id or local checkpoint directory override "
            "for --backend huggingface. Use this for offline self-provided DINO checkpoints."
        ),
    )
    extract_features.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="TorchScript module path for --backend torchscript.",
    )
    extract_features.add_argument(
        "--output-key",
        default=None,
        help="Dictionary key to select when a TorchScript module returns a dict.",
    )
    extract_features.add_argument(
        "--output-index",
        type=int,
        default=0,
        help="Tuple/list index to select when a TorchScript module returns multiple tensors.",
    )
    extract_sae_codes = subparsers.add_parser(
        "extract-sae-codes",
        help=(
            "Extract SAE codes from feature arrays using fixture or lightweight SAE backends."
        ),
    )
    extract_sae_codes.add_argument("--config-root", type=Path, default=Path("configs"))
    extract_sae_codes.add_argument(
        "--backend",
        default="fixture",
        choices=["fixture", "linear-topk"],
        help="SAE backend. `linear-topk` loads a minimal inference-only checkpoint.",
    )
    extract_sae_codes.add_argument("--features-npz", type=Path, required=True)
    extract_sae_codes.add_argument("--model-id", required=True)
    extract_sae_codes.add_argument("--sae-id", required=True)
    extract_sae_codes.add_argument("--output-dir", type=Path, required=True)
    extract_sae_codes.add_argument("--code-dim", type=int, default=16)
    extract_sae_codes.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional SAE checkpoint override for --backend linear-topk.",
    )
    extract_sae_codes.add_argument(
        "--allow-unresolved-checkpoint",
        action="store_true",
        help="Do not require config checkpoint placeholders to resolve.",
    )
    extract_sae_codes.add_argument("--topk", type=int, default=None)
    convert_sae = subparsers.add_parser(
        "convert-sae-checkpoint",
        help="Convert a full SAE checkpoint into the public lightweight .npz format.",
    )
    convert_sae.add_argument("--input-checkpoint", type=Path, required=True)
    convert_sae.add_argument("--output-checkpoint", type=Path, required=True)
    convert_sae.add_argument(
        "--allow-gated",
        action="store_true",
        help=(
            "Allow extracting W_enc from a gated checkpoint. This is not an exact "
            "gated SAE conversion and should normally be avoided."
        ),
    )
    check_manifest = subparsers.add_parser(
        "check-manifest",
        help="Validate a JSONL dataset manifest without loading image data.",
    )
    check_manifest.add_argument("--manifest", type=Path, required=True)
    check_manifest.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    check_manifest.add_argument("--expected-split", default=None)
    inspect_images = subparsers.add_parser(
        "inspect-images",
        help="Load and preprocess a small manifest slice using a model transform policy.",
    )
    inspect_images.add_argument("--config-root", type=Path, default=Path("configs"))
    inspect_images.add_argument("--manifest", type=Path, required=True)
    inspect_images.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "count_classification", "dense_depth", "dense_segmentation"],
    )
    inspect_images.add_argument("--model-id", required=True)
    inspect_images.add_argument("--expected-split", default=None)
    inspect_images.add_argument("--limit", type=int, default=8)
    inspect_images.add_argument("--output-json", type=Path, default=None)
    export_targets = subparsers.add_parser(
        "export-targets",
        help="Export dense depth or segmentation targets into the public targets.npz contract.",
    )
    export_targets.add_argument("--manifest", type=Path, required=True)
    export_targets.add_argument(
        "--task-type",
        required=True,
        choices=["dense_depth", "dense_segmentation"],
    )
    export_targets.add_argument("--output-dir", type=Path, required=True)
    export_targets.add_argument("--expected-split", default=None)
    export_targets.add_argument(
        "--features-npz",
        type=Path,
        default=None,
        help="Optional spatial features.npz used to infer dense target height/width.",
    )
    export_targets.add_argument("--height", type=int, default=None)
    export_targets.add_argument("--width", type=int, default=None)
    export_targets.add_argument("--max-examples", type=int, default=None)
    export_targets.add_argument("--target-key", default="targets")
    export_targets.add_argument(
        "--segmentation-ignore-value",
        type=int,
        default=None,
        help=(
            "Optional raw segmentation label value to map to "
            "--segmentation-output-ignore-index."
        ),
    )
    export_targets.add_argument(
        "--segmentation-output-ignore-index",
        type=int,
        default=255,
        help="Ignore index written into exported dense_segmentation targets.",
    )
    export_targets.add_argument(
        "--segmentation-label-offset",
        type=int,
        default=0,
        help=(
            "Optional offset applied to non-ignored segmentation labels. "
            "For official ADE20K masks, use --segmentation-ignore-value 0 "
            "--segmentation-label-offset -1."
        ),
    )
    index_artifacts = subparsers.add_parser(
        "index-artifacts",
        help="Build a JSON/CSV index for public reproduction artifacts.",
    )
    index_artifacts.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JSON artifacts.",
    )
    index_artifacts.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write artifact_index.json.",
    )
    index_artifacts.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write artifact_index.csv.",
    )
    index_artifacts.add_argument(
        "--require-valid",
        action="store_true",
        help="Exit with an error if any indexed artifact is invalid.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        from feature_economy import __version__

        print(__version__)
        return 0
    if args.command == "check-configs":
        from feature_economy.configs import validate_all_configs

        validated = validate_all_configs(args.config_root)
        print(f"Validated {len(validated)} config files under {args.config_root}")
        return 0
    if args.command == "plan-runs":
        from feature_economy.configs import write_reproduction_plan

        plan = write_reproduction_plan(
            config_root=args.config_root,
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
        print(f"Wrote {plan['num_rows']} planned runs to {args.output_json}")
        if args.output_csv is not None:
            print(f"Wrote run-plan CSV to {args.output_csv}")
        return 0
    if args.command == "check-bundle":
        from feature_economy.artifacts import check_artifact_bundle

        report = check_artifact_bundle(
            run_plan_json=args.run_plan_json,
            artifact_root=args.artifact_root,
            output_json=args.output_json,
        )
        print(
            "Checked artifact bundle: "
            f"{report['complete_rows']}/{report['num_rows']} rows complete"
        )
        if args.output_json is not None:
            print(f"Wrote bundle check report to {args.output_json}")
        if args.require_complete and not report["complete"]:
            parser.error(
                f"artifact bundle is incomplete: {report['missing_rows']} rows missing files"
            )
        return 0
    if args.command == "check-models":
        from feature_economy.models import ModelRegistry

        registry = ModelRegistry(args.config_root)
        for sae_id in sorted(registry.saes):
            registry.sae_checkpoint_path(
                sae_id,
                require_resolved=args.require_resolved_checkpoints,
            )
        print(
            f"Validated {len(registry.models)} models and {len(registry.saes)} SAEs "
            f"under {args.config_root}"
        )
        return 0
    if args.command == "check-runtime":
        from feature_economy.runtime import check_runtime, require_runtime, write_runtime_report

        statuses = require_runtime(args.profile) if args.require else check_runtime(args.profile)
        for status in statuses:
            if status.available:
                suffix = f"=={status.version}" if status.version else ""
                print(f"OK {status.module}{suffix}")
            else:
                print(f"MISSING {status.module}: {status.error}")
        if args.json_output is not None:
            output_path = write_runtime_report(statuses, args.json_output, profile=args.profile)
            print(f"Wrote runtime dependency report to {output_path}")
        return 0
    if args.command == "validate-arrays":
        from feature_economy.artifacts import validate_array_contract

        report = validate_array_contract(
            npz_path=args.npz,
            kind=args.kind,
            task_type=args.task_type,
            manifest_path=args.manifest,
            expected_split=args.expected_split,
            target_key=args.target_key,
            write_json=args.json_output,
        )
        shape_parts = ", ".join(
            f"{key}={shape}" for key, shape in sorted(report["arrays"].items())
        )
        print(f"Validated {args.kind} array contract for {args.npz}: {shape_parts}")
        if args.json_output is not None:
            print(f"Wrote array validation report to {args.json_output}")
        return 0
    if args.command == "make-tables":
        from feature_economy.paper import (
            write_all_tables_from_index,
            write_probe_score_table,
        )

        if args.artifact_index is not None:
            counts = write_all_tables_from_index(args.artifact_index, args.output_dir)
            print(f"Wrote tables to {args.output_dir}: {counts}")
            return 0
        elif args.input_dir is not None:
            output_csv = args.output_dir / "probe_scores.csv"
            row_count = write_probe_score_table(args.input_dir, output_csv)
            print(f"Wrote {row_count} probe-score rows to {output_csv}")
            return 0
        else:
            parser.error("make-tables requires --artifact-index or --input-dir")
    if args.command == "make-figures":
        from feature_economy.paper import write_all_figures_from_tables

        formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
        counts = write_all_figures_from_tables(
            args.table_dir,
            args.output_dir,
            formats=formats,
        )
        print(f"Wrote figures to {args.output_dir}: {counts}")
        return 0
    if args.command == "smoke-probe":
        from feature_economy.probes import write_smoke_probe_artifacts

        written = write_smoke_probe_artifacts(
            config_root=args.config_root,
            experiment_id=args.experiment_id,
            output_dir=args.output_dir,
        )
        print(f"Wrote {len(written)} smoke artifacts to {args.output_dir}")
        return 0
    if args.command == "smoke-analysis":
        from feature_economy.analysis import write_smoke_analysis_artifacts

        written = write_smoke_analysis_artifacts(
            config_root=args.config_root,
            output_dir=args.output_dir,
        )
        print(f"Wrote {len(written)} smoke analysis artifacts to {args.output_dir}")
        return 0
    if args.command == "compute-usage":
        if args.smoke:
            from feature_economy.analysis import write_smoke_analysis_artifacts

            written = write_smoke_analysis_artifacts(
                config_root=args.config_root,
                output_dir=args.output_dir,
                include={"availability"},
            )
            print(f"Wrote {len(written)} smoke usage artifacts to {args.output_dir}")
            return 0
        from feature_economy.analysis import compute_availability_from_codes

        missing = [
            name
            for name, value in {
                "--codes-npz": args.codes_npz,
                "--model-id": args.model_id,
                "--sae-id": args.sae_id,
                "--dataset-id": args.dataset_id,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(f"compute-usage requires {', '.join(missing)} unless --smoke is set")
        summary_path = compute_availability_from_codes(
            codes_npz=args.codes_npz,
            model_id=args.model_id,
            sae_id=args.sae_id,
            dataset_id=args.dataset_id,
            split=args.split,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        print(f"Wrote availability summary to {summary_path}")
        return 0
    if args.command == "rank-features":
        if args.smoke:
            from feature_economy.analysis import write_smoke_analysis_artifacts

            written = write_smoke_analysis_artifacts(
                config_root=args.config_root,
                output_dir=args.output_dir,
                include={"ranking", "subset_usage"},
            )
            print(f"Wrote {len(written)} smoke ranking/access artifacts to {args.output_dir}")
            return 0
        from feature_economy.analysis import rank_features_from_linear_probe

        missing = [
            name
            for name, value in {
                "--codes-npz": args.codes_npz,
                "--probe-logits-npz": args.probe_logits_npz,
                "--task-id": args.task_id,
                "--model-id": args.model_id,
                "--sae-id": args.sae_id,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(f"rank-features requires {', '.join(missing)} unless --smoke is set")
        ranking_path = rank_features_from_linear_probe(
            codes_npz=args.codes_npz,
            probe_logits_npz=args.probe_logits_npz,
            task_id=args.task_id,
            model_id=args.model_id,
            sae_id=args.sae_id,
            output_dir=args.output_dir,
            top_k=args.top_k,
            ranking_method=args.ranking_method,
            contribution_npz=args.contribution_npz,
            contribution_key=args.contribution_key,
            hybrid_alpha=args.hybrid_alpha,
        )
        print(f"Wrote task feature ranking to {ranking_path}")
        return 0
    if args.command == "compute-contributions":
        from feature_economy.analysis import compute_linear_probe_contribution_scores

        summary_path = compute_linear_probe_contribution_scores(
            codes_npz=args.codes_npz,
            probe_logits_npz=args.probe_logits_npz,
            task_type=args.task_type,
            task_id=args.task_id,
            model_id=args.model_id,
            sae_id=args.sae_id,
            output_dir=args.output_dir,
            primary_metric=args.primary_metric,
            max_features=args.max_features,
            score_key=args.score_key,
            scoring_method=args.scoring_method,
        )
        print(f"Wrote contribution score summary to {summary_path}")
        return 0
    if args.command == "compute-subset-usage":
        from feature_economy.analysis import compute_subset_usage_from_ranking

        subset_path = compute_subset_usage_from_ranking(
            codes_npz=args.codes_npz,
            ranking_json=args.ranking_json,
            output_dir=args.output_dir,
            top_k=args.top_k,
            random_seed=args.random_seed,
            threshold=args.threshold,
            high_usage_threshold=args.high_usage_threshold,
        )
        print(f"Wrote subset usage summary to {subset_path}")
        return 0
    if args.command == "ablate-features":
        if args.smoke or args.backend == "smoke":
            from feature_economy.analysis import write_smoke_analysis_artifacts

            written = write_smoke_analysis_artifacts(
                config_root=args.config_root,
                output_dir=args.output_dir,
                include={"ablation"},
            )
            print(f"Wrote {len(written)} smoke ablation artifacts to {args.output_dir}")
            return 0
        from feature_economy.analysis import ablate_linear_probe_features

        missing = [
            name
            for name, value in {
                "--codes-npz": args.codes_npz,
                "--probe-logits-npz": args.probe_logits_npz,
                "--ranking-json": args.ranking_json,
                "--task-type": args.task_type,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(
                "ablate-features --backend linear-probe requires "
                f"{', '.join(missing)}"
            )
        summary_path = ablate_linear_probe_features(
            codes_npz=args.codes_npz,
            probe_logits_npz=args.probe_logits_npz,
            ranking_json=args.ranking_json,
            task_type=args.task_type,
            output_dir=args.output_dir,
            top_k=args.top_k,
            random_seed=args.random_seed,
        )
        print(f"Wrote feature ablation summary to {summary_path}")
        return 0
    if args.command == "ablate-native-subspace":
        from feature_economy.analysis import ablate_native_subspace

        summary_path = ablate_native_subspace(
            features_npz=args.features_npz,
            sae_checkpoint=args.sae_checkpoint,
            probe_logits_npz=args.probe_logits_npz,
            ranking_json=args.ranking_json,
            task_type=args.task_type,
            output_dir=args.output_dir,
            top_k=args.top_k,
            random_seed=args.random_seed,
            random_pool=args.random_pool,
            normalize_activations=args.normalize_activations,
            topk=args.sae_topk,
        )
        print(f"Wrote native subspace ablation summary to {summary_path}")
        return 0
    if args.command == "eval-native-fixture":
        from feature_economy.probes import evaluate_native_fixture

        summary_path = evaluate_native_fixture(
            manifest_path=args.manifest,
            task_type=args.task_type,
            task_id=args.task_id,
            model_id=args.model_id,
            output_dir=args.output_dir,
            seed=args.seed,
            expected_split=args.expected_split,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        print(f"Wrote native fixture probe summary to {summary_path}")
        return 0
    if args.command == "probe-native":
        if args.backend == "fixture":
            from feature_economy.probes import evaluate_native_fixture

            if args.fixture_manifest is None:
                parser.error("probe-native --backend fixture requires --fixture-manifest")
            summary_path = evaluate_native_fixture(
                manifest_path=args.fixture_manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                output_dir=args.output_dir,
                seed=args.seed,
                expected_split=args.expected_split,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                command_name="feature-economy probe-native",
            )
        elif args.backend == "linear-probe":
            from feature_economy.probes import train_native_linear_probe

            if args.features_npz is None or args.manifest is None:
                parser.error("probe-native --backend linear-probe requires --features-npz and --manifest")
            summary_path = train_native_linear_probe(
                features_npz=args.features_npz,
                manifest_path=args.manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                output_dir=args.output_dir,
                expected_split=args.expected_split,
                seed=args.seed,
                ridge=args.ridge,
                targets_npz=args.targets_npz,
                target_key=args.target_key,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        elif args.backend == "paper-scale-torch":
            from feature_economy.probes import train_native_paper_scale_probe

            train_features_npz = args.train_features_npz
            val_features_npz = args.val_features_npz or args.features_npz
            train_manifest = args.train_manifest
            val_manifest = args.val_manifest or args.manifest
            if train_features_npz is None or val_features_npz is None or train_manifest is None or val_manifest is None:
                parser.error(
                    "probe-native --backend paper-scale-torch requires "
                    "--train-features-npz, --val-features-npz, --train-manifest, and --val-manifest"
                )
            summary_path = train_native_paper_scale_probe(
                train_features_npz=train_features_npz,
                train_manifest_path=train_manifest,
                val_features_npz=val_features_npz,
                val_manifest_path=val_manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                output_dir=args.output_dir,
                expected_train_split="train",
                expected_val_split=args.expected_split or "val",
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                train_targets_npz=args.train_targets_npz,
                val_targets_npz=args.val_targets_npz or args.targets_npz,
                target_key=args.target_key,
                decoder_hidden_channels=args.decoder_hidden_channels,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        else:
            parser.error(f"unsupported probe-native backend: {args.backend}")
        print(f"Wrote native probe summary to {summary_path}")
        return 0
    if args.command == "probe-sae":
        if args.backend == "fixture":
            from feature_economy.probes import evaluate_sae_fixture

            if args.fixture_manifest is None:
                parser.error("probe-sae --backend fixture requires --fixture-manifest")
            summary_path = evaluate_sae_fixture(
                manifest_path=args.fixture_manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                sae_id=args.sae_id,
                output_dir=args.output_dir,
                seed=args.seed,
                expected_split=args.expected_split,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        elif args.backend == "linear-probe":
            from feature_economy.probes import train_sae_linear_probe

            if args.codes_npz is None or args.manifest is None:
                parser.error("probe-sae --backend linear-probe requires --codes-npz and --manifest")
            summary_path = train_sae_linear_probe(
                codes_npz=args.codes_npz,
                manifest_path=args.manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                sae_id=args.sae_id,
                output_dir=args.output_dir,
                expected_split=args.expected_split,
                seed=args.seed,
                ridge=args.ridge,
                targets_npz=args.targets_npz,
                target_key=args.target_key,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        elif args.backend == "paper-scale-torch":
            from feature_economy.probes import train_sae_paper_scale_probe

            train_codes_npz = args.train_codes_npz
            val_codes_npz = args.val_codes_npz or args.codes_npz
            train_manifest = args.train_manifest
            val_manifest = args.val_manifest or args.manifest
            if train_codes_npz is None or val_codes_npz is None or train_manifest is None or val_manifest is None:
                parser.error(
                    "probe-sae --backend paper-scale-torch requires "
                    "--train-codes-npz, --val-codes-npz, --train-manifest, and --val-manifest"
                )
            summary_path = train_sae_paper_scale_probe(
                train_codes_npz=train_codes_npz,
                train_manifest_path=train_manifest,
                val_codes_npz=val_codes_npz,
                val_manifest_path=val_manifest,
                task_type=args.task_type,
                task_id=args.task_id,
                model_id=args.model_id,
                sae_id=args.sae_id,
                output_dir=args.output_dir,
                expected_train_split="train",
                expected_val_split=args.expected_split or "val",
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                train_targets_npz=args.train_targets_npz,
                val_targets_npz=args.val_targets_npz or args.targets_npz,
                target_key=args.target_key,
                decoder_hidden_channels=args.decoder_hidden_channels,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        else:
            parser.error(f"unsupported probe-sae backend: {args.backend}")
        print(f"Wrote SAE probe summary to {summary_path}")
        return 0
    if args.command == "extract-features":
        if args.backend == "fixture":
            from feature_economy.models import extract_fixture_features

            manifest_path = args.fixture_manifest or args.manifest
            if manifest_path is None:
                parser.error("extract-features --backend fixture requires --fixture-manifest")
            summary_path = extract_fixture_features(
                config_root=args.config_root,
                manifest_path=manifest_path,
                task_type=args.task_type,
                model_id=args.model_id,
                output_dir=args.output_dir,
                layer=args.layer,
                expected_split=args.expected_split,
                feature_dim=args.feature_dim,
            )
        elif args.backend == "huggingface":
            from feature_economy.models import extract_huggingface_features

            if args.manifest is None:
                parser.error("extract-features --backend huggingface requires --manifest")
            summary_path = extract_huggingface_features(
                config_root=args.config_root,
                manifest_path=args.manifest,
                task_type=args.task_type,
                model_id=args.model_id,
                output_dir=args.output_dir,
                layer=args.layer,
                expected_split=args.expected_split,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                max_examples=args.max_examples,
                local_files_only=args.local_files_only,
                hf_name_or_path=args.hf_name_or_path,
            )
        elif args.backend == "torchscript":
            from feature_economy.models import extract_torchscript_features

            if args.manifest is None:
                parser.error("extract-features --backend torchscript requires --manifest")
            if args.checkpoint is None:
                parser.error("extract-features --backend torchscript requires --checkpoint")
            summary_path = extract_torchscript_features(
                config_root=args.config_root,
                manifest_path=args.manifest,
                task_type=args.task_type,
                model_id=args.model_id,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                layer=args.layer,
                expected_split=args.expected_split,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                max_examples=args.max_examples,
                output_key=args.output_key,
                output_index=args.output_index,
            )
        else:
            parser.error(f"unsupported extract-features backend: {args.backend}")
        print(f"Wrote feature extraction summary to {summary_path}")
        return 0
    if args.command == "extract-sae-codes":
        if args.backend == "fixture":
            from feature_economy.models import extract_fixture_sae_codes

            summary_path = extract_fixture_sae_codes(
                config_root=args.config_root,
                features_npz=args.features_npz,
                model_id=args.model_id,
                sae_id=args.sae_id,
                output_dir=args.output_dir,
                code_dim=args.code_dim,
            )
        elif args.backend == "linear-topk":
            from feature_economy.models import extract_linear_topk_sae_codes

            summary_path = extract_linear_topk_sae_codes(
                config_root=args.config_root,
                features_npz=args.features_npz,
                model_id=args.model_id,
                sae_id=args.sae_id,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                require_resolved_checkpoint=not args.allow_unresolved_checkpoint,
                topk=args.topk,
            )
        else:
            parser.error(f"unsupported extract-sae-codes backend: {args.backend}")
        print(f"Wrote SAE code summary to {summary_path}")
        return 0
    if args.command == "convert-sae-checkpoint":
        from feature_economy.models import convert_sae_checkpoint_to_lightweight

        output_path = convert_sae_checkpoint_to_lightweight(
            input_checkpoint=args.input_checkpoint,
            output_checkpoint=args.output_checkpoint,
            allow_gated=args.allow_gated,
        )
        print(f"Wrote lightweight SAE checkpoint to {output_path}")
        return 0
    if args.command == "check-manifest":
        from feature_economy.data import validate_manifest

        count = validate_manifest(
            args.manifest,
            task_type=args.task_type,
            expected_split=args.expected_split,
        )
        print(f"Validated {count} records in {args.manifest}")
        return 0
    if args.command == "inspect-images":
        from feature_economy.models import inspect_manifest_images

        report = inspect_manifest_images(
            config_root=args.config_root,
            manifest_path=args.manifest,
            task_type=args.task_type,
            model_id=args.model_id,
            output_json=args.output_json,
            expected_split=args.expected_split,
            limit=args.limit,
        )
        print(
            "Inspected "
            f"{report['num_inspected']}/{report['num_manifest_records']} images "
            f"from {args.manifest}"
        )
        if args.output_json is not None:
            print(f"Wrote image inspection report to {args.output_json}")
        return 0
    if args.command == "export-targets":
        from feature_economy.data.targets import export_dense_targets

        if (args.height is None) != (args.width is None):
            parser.error("export-targets requires both --height and --width, or neither")
        target_shape = None
        if args.height is not None and args.width is not None:
            target_shape = (args.height, args.width)
        summary_path = export_dense_targets(
            manifest_path=args.manifest,
            task_type=args.task_type,
            output_dir=args.output_dir,
            expected_split=args.expected_split,
            features_npz=args.features_npz,
            target_shape=target_shape,
            max_examples=args.max_examples,
            target_key=args.target_key,
            segmentation_ignore_value=args.segmentation_ignore_value,
            segmentation_output_ignore_index=args.segmentation_output_ignore_index,
            segmentation_label_offset=args.segmentation_label_offset,
        )
        print(f"Wrote dense target export summary to {summary_path}")
        return 0
    if args.command == "index-artifacts":
        from feature_economy.artifacts import build_artifact_index

        index = build_artifact_index(
            input_dir=args.input_dir,
            output_json=args.output_json,
            output_csv=args.output_csv,
            require_valid=args.require_valid,
        )
        print(
            "Indexed "
            f"{index['num_valid']}/{index['num_records']} valid artifacts "
            f"from {args.input_dir}"
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
