# Implementation scope

This package provides an executable CNM simulation. Some operators and the
execution backend differ from the current manuscript. These differences affect
interpretation and predate this documentation release.

## Execution network and formation

`EEFPolicy` encodes 24 ranges and 17 state/context values with two MLP branches,
concatenates their 64-dimensional outputs, and uses a 96-unit GRU. It outputs
15 velocity/yaw-rate commands and a seven-dimensional terminal mean with a
triangular covariance factor. These differ from S2's depth CNN, 192-unit GRU,
candidate probabilities, temporal retention and reference-conditioned control.

The outcome head receives a detached recurrent feature. Its likelihood loss
trains the observer; the differentiable terminal cost supplies controller
gradients. Formation uses a point-mass surrogate; rigid-body evaluation uses
PyBullet. Baselines are literature-inspired adaptations. The S2 flight network,
flight checkpoint and flight-controller bridge are not part of this package.

## Forecasts, verification and events

`FlightSegment` accepts a terminal forecast, but it is optional. `_terminal_gate`
uses successful-acquisition-batch statistics when the forecast is absent.
The individual acquisition runner calls `from_jsonl` without a prospective
forecast. This batch-consistency gate is not held-out forecast coverage.

Position and velocity gates use Euclidean vector norms, whereas S3 specifies
componentwise normalized infinity-norm bounds. Empirical interface covariance
uses componentwise floors; prospective covariance receives only numerical
diagonal stabilization. Planner covariance regularization adds a scalar 0.025
to each of seven diagonal entries. These are different conventions.

Compilation assigns record IDs after grouping telemetry and emits events when
a record can be compiled. Event `success` is admission membership, not response
completion independently of admission. A completed response rejected by the
terminal gate therefore contributes negative evidence. The manuscript separates
completion from admission.

`ExecutionEvent` retains executor, origin, context, record, entry, command and
exit with an immutable identifier and content hash. It does not serialize the
complete S3 session/attempt/forecast/admission/simulation-origin schema. Three
admitted segments are required, but reset independence is not checked by the
compiler. Entirely rejected batches do not yield the full persistent attempt
ledger required by S3.

## Support, graph edges and evidence scores

Support samples are stored and transformed. Active entry/exit gates use a box
around the fitted center with empirical quantile radii and a Mahalanobis gate.
Connectors use combined radii. These are not the S3 union of fixed neighborhoods
around individual samples.

Placement applies rigid yaw and translation. `relative_scale` is checked as
metadata and does not scale coordinates. Support radii are copied rather than
rotated as anisotropic neighborhoods. The observed-free bridge flag defaults
to true; observed-space verification requires a caller-supplied assessment.

Evidence weighting includes `1/sqrt(n)` for successive events from one executor
in addition to context and age. That factor is absent from the manuscript's
equation. Beta quantiles and chain products are ranking scores, not calibrated
joint-success bounds. `connector_probability` is a geometric exponential score,
not a posterior based on independently identified connector executions.
Bridge actions and replan transitions are logged, but not every bridge attempt
has a standalone execution event.

The 128-record curation target protects the final record of each role and can
be exceeded when all roles are unique. There is no 24 MB byte-cap implementation.

## Collective information and recipient validation

Each robot owns a private library. For selective transfer, the runner first
plans with a temporary copy containing the pooled candidate history, then
charges and transmits missing selected records. This assumes union access
during selection and is not distributed peer-discovery/offer negotiation.

`select_for_query` checks novelty at record-ID level. Existing records with
new events are excluded by that selector, although other packet/return paths
can merge their evidence. Import checks cover schema, hashes and identity;
they do not enforce the complete S6 vehicle/placement transferability gate.

Imported support can be used by planning immediately. The schema has no
separate donor-provisional/recipient-validated state with promotion after three
recipient resets. Appending an event changes evidence scores but does not
expand recipient-local support through the compiler.

Returns use measured replan exits or a successful traversal's final measured
state. Selected records lacking such exits are skipped. Repeated uses of a
record are deduplicated within the return routine. Entry is the stored mean,
duration is apportioned from episode time, and intermediate success uses
downstream-query support. These are not complete measured events for every
attempt or interruption.

## Communication and numerical reproduction

The runner counts initial serialized record/evidence transfers. Recipient
events are merged directly into libraries; return transmission is not fully
charged. Discovery, candidate-union inspection and live peer-state traffic
are also excluded. This counter is narrower than S6's record-and-return
payload total.

The archive-level evidence bundler also requires an EEF statistical summary and
historical regression/figure files not emitted by the standalone runners.
Its aggregation and archival validation are separate from a fresh simulation
run. The README identifies these prerequisites rather than presenting the
bundler as a self-contained reproduction command.

The formation runner's native speed frontier is 0.8, 1.2, 1.6, 1.9 and 2.0 m s−1.
Figures with different speed labels require their actual source data and
transformation provenance. Plotting assets and recorded datasets are separate
from this algorithm repository. A 40-second training curve depends on hardware
and runtime as well as algorithm settings.

Lightweight release checks do not certify every published success count or
payload total. Prospective verification, recipient support promotion and
distributed return accounting need substantive implementation changes and new
experiments before they can be reported as fully reproduced here.
