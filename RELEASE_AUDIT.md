# Release verification

Reviewed on 26 September 2026 against the current manuscript Methods and the
published repository base `7f95490df913dbdd525cdd6852b75deabcb05a7f`.

## Change boundary

The 26 previously tracked runtime, configuration and test files are byte-for-byte
unchanged from the local pre-review snapshot. Git checkout now preserves those
bytes across platforms; three auxiliary files previously normalized by Git
retain their existing CRLF endings. The sources used by checkpoint hashing
are unchanged from both the published base and the local snapshot.
The missing `algorithm/individual_protocol.py` was restored verbatim
from the experimental source tree: `model_sha256` reads this file when saving
or loading checkpoints. Without it, the original distribution fails during
checkpoint saving. No policy architecture, loss, numerical parameter, controller,
random seed or scoring rule was modified.

Reader-facing documentation now describes the current executable behavior,
including material differences from the manuscript. A checkpoint round-trip
regression test and a standard-library file-integrity checker were added.
Existing technical comments are retained because source bytes participate in
checkpoint identity. Drafting correspondence in the documentation was replaced
with the Methods map and implementation specification.

## Validation results

- Existing mechanism tests before changes: **5 passed**.
- Tests after packaging repair, including checkpoint round trip: **6 passed**.
- Structural release validation: **6 checks passed**.
- Two deterministic five-method EEF smoke runs: completed training, checkpoint
  saving and PyBullet evaluation.
- `final_trials.csv`: **1,800 rows**, byte-identical across the two runs.
- `interface_repeatability.csv`: **60 rows**, byte-identical.
- `pybullet_trials.csv`: **40 rows**, byte-identical.
- `training_efficiency.csv`: **10 rows**; only measured `training_seconds`
  differs. Timing is not expected to be bitwise deterministic.
- Five checkpoint hashes and all corresponding state tensors are identical.
- All packaged modules import; six documented experiment/archive CLIs accept
  `--help`. This checks entry points, not archive availability.
- A wheel builds without resolving dependencies and includes the restored
  protocol source used by checkpoint hashing.

The first successful smoke run was performed immediately after restoring the
missing hash dependency; the second used the final documentation/test changes.
The uncorrected distribution cannot complete that comparison because its
dependency is absent. `docs/RUNTIME_BASELINE.json` records the preserved file
hashes and restored dependency.

Validation used Python 3.12.1, NumPy 1.26.4, PyTorch 2.14.0+cpu, PyBullet 3.2.5,
Gymnasium 1.1.1, Pillow 11.3.0 and pytest 8.4.2 on Windows. This is the validation
environment, not a claim that all supported dependency combinations were tested.

These checks establish packaging functionality and unchanged tested numerical
behavior. They do not establish equivalence to the manuscript's different
network or verification operators, or replace full multi-seed experiments.
See [Implementation scope](docs/IMPLEMENTATION_SCOPE.md) for those distinctions.
