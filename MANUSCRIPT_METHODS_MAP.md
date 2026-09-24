
# Manuscript-to-code traceability

| CNM mechanism | Reference implementation | Failure mode addressed | Primary evidence/control | Frozen/adaptive separation |
|---|---|---|---|---|
| Frozen execution stack | `algorithm/frozen.py` | hidden post-deployment parameter changes | source/config/checkpoint hashes | the manifest defines `W*`; its digest is fixed |
| Privileged feasibility and differentiable physical rollout | `algorithm/eef.py`, `algorithm/eef_benchmark.py` | unsafe or dynamically inconsistent local response | formation ablations | training ends before deployment |
| Terminal-outcome supervision | `EEFPolicy.outcome_head`, `ExperienceCompiler` gates | successful motion with dispersed, unusable exits | calibration, dispersion, repeatability, connector yield | predictor and gates belong to `W*` |
| Immutable execution event | `memory.ExecutionEvent` | rejected/failed attempts disappear; duplicate forwarding inflates evidence | deterministic IDs, content hash, idempotent merge | events are appended to `A` |
| Verified experience record | `memory.ExperienceRecord`, `experience.ExperienceCompiler` | open-loop trajectory replay is mistaken for reusable experience | terminal/clearance/minimum-support gates | compiler rule fixed; records evolve |
| Split/merge and provenance | `compile_variants`, `merge_compatible_records` | multimodal outcomes are averaged; origins are lost | parent IDs and operator audit | operators fixed; record set changes |
| Entry/exit interfaces | `InterfaceSummary`, `CNMPlanner.place` | geometric resemblance licenses unsupported execution | finite support, covariance and placement gates | thresholds fixed in `PlannerConfig` |
| Conservative composition | `connector_assessment`, `compose` | adjacency is treated as executability | lower reliability, stitch probability, A* budget | evidence changes; scoring/search rules do not |
| Closed-loop replanning | `algorithm/rollout.py` | a stored global chain is replayed open loop | execute first response, replan from measured exit | rollout operator belongs to `W*` |
| Individual growth | `run_reconfigurable_individual_cnm.py` | complete-route leakage or policy fine-tuning explains growth | unseen A+C+B tasks, retention, deletion/restoration | probes are read-only; weights unchanged |
| Context evidence | `posterior_parameters`, `run_context_evidence_ablation.py` | a local failure globally poisons a response | context-collapse ablation | ledger changes; kernel/decay remain fixed |
| Selective circulation | `select_for_query`, `merge_packet` | dense broadcast or best-history replication explains collective gain | unrestricted/pooling/private controls | private `A_i` states remain explicit |
| Complementary histories | `run_collective_cnm_experiment.py` | one robot already contains full capability | private/union audit, redundancy and source withholding | provenance is retained in `A_i` |
| Execution return | `_return_execution_evidence` | shared records never receive recipient evidence | feedback withholding and adverse-return probes | return changes evidence, never `W*` |

## Parameter correspondence

The reference simulator and the hardware-oriented manuscript description are
not numerically identical. The current tested simulator uses 24 range rays, a
96-unit recurrent state, 15 response steps, 28 candidates and a 128-record
budget. The manuscript text describes a four-block stereo encoder, 256-unit
recurrent module, 64 B-spline candidates, a 2 s response and 512 records.
Changing these defaults would invalidate the archived checkpoints and results,
so the release preserves the tested configuration and reports the discrepancy
instead of silently relabelling it. A hardware release must provide the stereo
model, calibration, controller bridge and matching frozen manifests before the
hardware-specific values can be claimed as code-reproduced.
