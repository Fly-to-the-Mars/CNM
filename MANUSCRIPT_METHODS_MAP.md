# Methods and implementation correspondence

This map refers to Materials and Methods and Supplementary Methods S1–S8 of
the current manuscript. The implementation column describes the released
simulation. Detailed differences appear in
[Implementation scope](docs/IMPLEMENTATION_SCOPE.md).

| Mechanism | Code | Correspondence |
|---|---|---|
| Fixed execution stack | `algorithm/frozen.py`; `model_sha256` | Weights/configuration/source hashes; separate recurrent state |
| EEF formation | `algorithm/eef.py`; `run_eef_training_efficiency.py` | Privileged guidance, differentiable surrogate rollouts and outcome supervision; different execution network from S2 |
| Execution architecture, S2 | `EEFPolicy` | 24-ray MLP, 96-unit GRU, response-sequence head and terminal-distribution head |
| Record compilation, S3 | `FlightSegment`; `ExperienceCompiler.compile` | Measured interfaces and admission gates; optional forecast and batch compilation differ from prospective verification |
| Finite support, S3–S4 | `InterfaceSummary`; `entry_distance`; `connector_assessment` | Samples retained; gates use centre-based radii, not per-sample support neighborhoods |
| Evidence score | `posterior_parameters`; `beta_quantile` | Unique-event context/age weighting, plus an executor-count discount; compiler success includes admission |
| Placement/search, S4 | `CNMPlanner.place`; `compose` | Rigid yaw/translation, uncertainty gates and bounded A*; geometric connector score |
| Feedback execution | `algorithm/rollout.py` | Measured-exit replanning and bridge telemetry; incomplete standalone bridge-event ledger |
| Individual controls, S5 | `run_reconfigurable_individual_cnm.py` | 24 modular tasks, A+C+B, read-only probes, deletion/restoration; separate from competition hardware |
| Equal-information graph | `_equal_graph_path` | Minimum-duration record per role and canonical role adjacency; not full E-Graphs |
| Circulation, S6 | `select_for_query`; `merge_packet`; collective runner | Hashed payloads and deduplication; experimental selector accesses a pooled candidate union |
| Recipient return | `_return_execution_evidence` | Measured exits where available; incomplete attempt metadata and recipient-local support promotion |
| Communication | `_communicate`; return routine | Initial serialized payload counted; direct return merges are not fully charged as transmissions |

## Configuration correspondence

| Quantity | Released default | Current manuscript |
|---|---|---|
| Observation | 24 ranges + 17 state/context values | 1×12×16 depth + 10 state values + separate 6-value reference |
| Recurrent state | 96 | 192 |
| Fusion | Concatenate two 64-value branches | Add two 192-value branches |
| Outputs | 15×4 response sequence; 7-value mean and 28 Cholesky parameters | 15 scores; scalar retention; 3×2 control output |
| Teacher candidates | 9×3 moving candidates + stop | Distinct from network candidate scores |
| Active record target | 128; last record of each role protected | 512 records or 24 MB |
| Retrieval | 32 | 32 |
| Search expansions / chain length | 160 / 6 | 160 / 6 |
| Entry / connector threshold | 14.1 / 14.1 | 14.1 / 14.1 |
| Bridge duration | 0.35 s | 0.35 s |
| Position / velocity / yaw admission | 0.25 m / 0.35 m s−1 / 12°, vector norms for position/velocity | Same values in componentwise bounds |
| Clearance / admitted executions | 0.25 m / 3 | 0.25 m / 3 independent resets |

Shared numbers do not establish identical operators. Reset independence and
the 24 MB storage limit are not enforced by this implementation. The source
is preserved rather than changing defaults to match text without re-evaluation.
