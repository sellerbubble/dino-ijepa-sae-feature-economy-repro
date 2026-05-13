# Agent Rules For Source Code

`src/feature_economy/` contains reusable public command implementations. Keep it
library-like: no private paths, no remote-cluster assumptions, and no large
side-effectful work at import time.

## Boundaries

- `artifacts/`: validation schemas, artifact indexing, and bundle checks.
- `configs/`: config loaders, registry helpers, and run-plan expansion.
- `models/`: feature extraction, transform policies, SAE-code conversion.
- `probes/`: native/SAE probe training and metrics.
- `analysis/`: Availability, Access, Allocation, ranking controls, ablations,
  and advanced analyses.
- `cli/`: thin argument parsing and command dispatch.

## Implementation Discipline

New public commands should:

- consume explicit file paths or config IDs;
- write a summary JSON and `run_manifest.json`;
- validate outputs through `artifacts/schemas.py`;
- have a tiny-array unit test;
- appear in `plan-runs` if they are part of the paper-style matrix.

Avoid importing private training packages in normal command paths. If an adapter
needs an optional dependency, fail with a clear message and keep smoke tests
dependency-light.

## Validation

For Python changes, run:

```bash
python -m py_compile <changed files>
PYTHONPATH=src python -m unittest discover -s tests
```
