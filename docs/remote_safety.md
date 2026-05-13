# Remote Safety And Non-Destructive Runs

This document defines the safety rules for running public-reproduction work on
shared remote servers.

## Core Rule

Do not overwrite, delete, or reorganize existing remote files. Use new
timestamped directories, dry-run checks, and explicit output roots.

## Recommended Servers

Based on the current SSH config and a light resource probe on 2026-05-13:

| SSH alias | Role | Notes |
| --- | --- | --- |
| `4gpu` | Primary validation server | Has shared paths and currently the best free H200 capacity. |
| `2gpu` | Backup validation server | Useful for smaller profiles or overflow. |
| `4gpu-2` | Standby | Check current GPU owners before use. |
| `8gpu` | Avoid for now | H200 GPUs were fully busy during the latest probe. |

Recheck GPU usage before launching any job:

```bash
ssh 4gpu 'nvidia-smi'
```

## Safe Directory Convention

Use shared storage, but never overwrite existing project directories.

Recommended roots:

```bash
export SHARED_WORKSPACE=/path/to/shared/workspace
export SHARED_ARTIFACTS=/path/to/shared/artifacts
export REMOTE_CODE_BASE=$SHARED_WORKSPACE/ijepa_sae_research_public_runs
export REMOTE_ARTIFACT_BASE=$SHARED_ARTIFACTS/ijepa_sae_research/artifacts/public_full_rerun
export RUN_ID=$(date +%Y%m%d_%H%M%S)_paper_full_profile
export REMOTE_CODE_DIR=$REMOTE_CODE_BASE/$RUN_ID/public_repro
export ARTIFACT_ROOT=$REMOTE_ARTIFACT_BASE/$RUN_ID
```

This keeps public-rerun work separate from:

- existing private workbench checkouts;
- prior paper artifact directories;
- any active experiment output tree.

## Safe Sync Pattern

Always dry-run before syncing:

```bash
rsync -av --dry-run --delete \
  /local/path/to/public_repro/ \
  4gpu:$REMOTE_CODE_DIR/
```

Only remove `--dry-run` after verifying the destination is a new run-specific
directory. Avoid syncing into an existing remote checkout unless the user
explicitly requests it.

## Safe Launch Pattern

Every remote command should set explicit output roots:

```bash
mkdir -p "$ARTIFACT_ROOT/logs"
cd "$REMOTE_CODE_DIR"
PYTHONPATH=src python -m feature_economy.cli.main check-runtime \
  --profile experiments \
  --require \
  --json-output "$ARTIFACT_ROOT/logs/runtime_check.json" \
  2>&1 | tee "$ARTIFACT_ROOT/logs/runtime_check.log"
```

Do not write logs or generated arrays into the code directory.

## Before Starting A Long Job

Check and record:

- `hostname`
- `nvidia-smi`
- disk usage for the selected shared workspace and artifact mount;
- public repo git commit or exported tree checksum
- `ARTIFACT_ROOT`
- command line

## Cleanup Rule

Do not delete remote outputs automatically. If cleanup is needed, first list the
exact target directory and confirm it is under the current run-specific
`ARTIFACT_ROOT`.

Safe example:

```bash
echo "$ARTIFACT_ROOT"
find "$ARTIFACT_ROOT" -maxdepth 2 -type f | head
```

Unsafe examples:

```bash
rm -rf /path/to/existing/private/workbench
rm -rf /path/to/shared/artifacts/*
rsync --delete public_repro/ 4gpu:/path/to/existing/private/workbench/
```

If there is any uncertainty, stop and ask before modifying remote files.
