# Agent Rules For Docs

Docs are part of the reproduction contract. Keep them synchronized with configs,
CLI commands, artifact schemas, and release checks.

## Source Of Truth

- `full_reproduction.md`: high-level full rerun entrypoint.
- `canonical_chain_runbook.md`: command-by-command paper chain.
- `experiment_matrix.md`: compact map of configured stages and row counts.
- `artifact_schemas.md`: human-readable artifact contract.
- `release_quickstart.md`: shortest practical path for users.
- `reproduction_status.md`: coverage and known gaps.

## Rules

Do not promise shipped data, checkpoints, or exact bitwise reproduction. The
public repo targets paper-faithful reruns with close scientific agreement.

When changing command names, config IDs, row counts, required files, or artifact
paths, update the relevant docs in the same commit.

After doc changes, run:

```bash
PYTHONPATH=src python scripts/check_public_release.py --root .
```
