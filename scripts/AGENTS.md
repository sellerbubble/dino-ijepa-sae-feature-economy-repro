# Agent Rules For Scripts

Scripts orchestrate public workflows; they should not hide scientific logic.
Core logic belongs in `src/feature_economy/`, with scripts calling CLI commands
or packaging validated artifacts.

## Script Categories

- smoke and CI checks;
- release/export helpers;
- artifact bundle planning, copying, packaging, and auditing;
- thin full-profile launchers.

## Rules

Do not add private cluster launchers or local absolute paths. Use flags and
environment variables. If a script writes artifacts, document the expected input
and output directories in the script header or nearby docs.

Prefer extending existing scripts over adding near-duplicates. If a new script
is necessary, make it composable and add a unit test or release-check coverage
when practical.

Validation:

```bash
bash -n scripts/<script>.sh
python -m py_compile scripts/<script>.py
PYTHONPATH=src python scripts/check_public_release.py --root .
```
