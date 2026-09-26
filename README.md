# Compositional Navigation Memory (CNM)

CNM treats executed experience as the persistent adaptive state of a robot.
The execution policy and memory operators remain fixed after training; robots
retain local experience, compose compatible responses, and revise evidence
when responses are executed again.

```text
execution → experience records → composition → new execution → evidence revision
                                   ↑
                     complementary histories from peers
```

This repository implements formation, individual composition and collective
history-sharing experiments in simulation. The code structure and workflows
below describe the released modules and how to run them locally.

## Fixed and adaptive state

The fixed stack comprises the policy, observation encoder, command interface,
compiler thresholds, graph rules and evidence operators. `FrozenStackManifest`
binds configuration and source hashes; checkpoints also bind deployment source
files and model tensors. Recurrent hidden state is transient execution state.

Each `CNMLibrary` owns records and unique execution events. Records retain
structure, context, entry/exit statistics and samples, local response commands,
and provenance. Graph queries use this state to select a response chain. The
controller follows local references under feedback and replans from measured
state. A received record preserves its source; forwarding the same event does
not create another observation.

## Code structure

```text
src/cnm_swarm_sim/
  algorithm/
    eef.py                      # policy, differentiable dynamics and checkpoints
    eef_benchmark.py             # matched response evaluations
    experience.py               # telemetry-to-record compilation
    memory.py                   # events, records, evidence, transfer and search
    frozen.py                   # frozen-stack manifests
    individual_protocol.py      # atomic protocol included in checkpoint hashes
    protocol.py                 # local module definitions
    reconfigurable_protocol.py  # 24 modular tasks, including A+C+B
    rollout.py                  # feedback execution and measured-exit replanning
  env.py                        # PyBullet environment
  config.py                     # environment and control configuration
  scenarios.py                  # obstacle layouts
  run_eef_training_efficiency.py
  run_eef_formation_ablation.py
  run_reconfigurable_individual_cnm.py
  run_context_evidence_ablation.py
  run_collective_cnm_experiment.py
  build_formal_evidence_bundle.py
tests/test_core_mechanisms.py
tools/verify_release.py
```

The simulation policy uses 24 range rays, an MLP encoder, a 96-unit GRU,
a local velocity/yaw-rate sequence, and a terminal-distribution observer.
Its 15 response steps are not the manuscript network's 15 candidate scores.

## Install

Use Python 3.10–3.12. From a fresh clone:

```powershell
git clone https://github.com/Fly-to-the-Mars/CNM.git
cd CNM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"
```

On Linux/macOS, activate with `source .venv/bin/activate`. Subsequent
`python -m ...` commands are the same on all platforms.

Verify source integrity and the core mechanisms:

```text
python tools/verify_release.py
python -m pytest
python -m cnm_swarm_sim.validate_release
```

The tests exercise record compilation, composition, targeted deletion and
restoration, idempotent transfer, tamper rejection, source-linked failure
evidence, and the unseen A+C+B task. These checks are not a replacement for
the full experiment matrix.

## First run: formation benchmark

```text
python -m cnm_swarm_sim.run_eef_training_efficiency --smoke --output runs/eef_smoke
```

This trains all five methods for a reduced budget and evaluates both surrogate
responses and PyBullet execution. Smoke checkpoints test the pipeline; they
are not sufficiently trained checkpoints for the full composition benchmark.
Each method's checkpoint is written under `training/<method>/seed_<seed>/`.

For the complete formation benchmark:

```text
python -m cnm_swarm_sim.run_eef_training_efficiency --output runs/eef_training_efficiency_v5 --iterations 700 --batch-size 128 --seeds 5 --validation-samples 512 --evaluation-samples 256 --physical-repeats 3 --device cpu
```

Methods are `legacy_student`, `agile_dagger_style`, `ppo_response`,
`diffphys_nmi_style` and `eef_full`. They adapt the training principles to a
common simulator interface. The default elapsed-training comparison uses
40 seconds per method/seed; wall-clock measurements depend on the machine.

## Figure-level formation evidence

The six analyses associated with **Fig02 EEF Training Efficiency v6** are:

| Analysis | Source table | Interpretation |
|---|---|---|
| Training efficiency | `training_efficiency.csv` | Experience yield versus generated scenes |
| Competence versus training time | `training_efficiency.csv` | Completion versus elapsed training time |
| Robustness–agility frontier | `final_trials.csv` | Completion across native commanded speeds |
| Prediction–execution alignment | `final_trials.csv` | Predicted versus realized terminal state |
| Interface-ready experience | `final_trials.csv`, `interface_repeatability.csv` | Reuse gates and repeated-response consistency |
| Rigid-body transfer | `pybullet_trials.csv` | PyBullet completion and traversal duration |

All tables are under the run's `source_data/` directory. `run_manifest.json`
records arguments, software information and evaluation boundaries;
`summary.json` contains aggregate metrics. Native frontier commands are
0.8, 1.2, 1.6, 1.9 and 2.0 m s−1. Exact archived figure reproduction requires
the figure's source data, plotting settings, checkpoints and source version.
Plotting scripts and recorded datasets are supplied in the separate data package.

## Individual composition

After completing formation, run the independent acquisition and task probes:

```text
python -m cnm_swarm_sim.run_reconfigurable_individual_cnm --eef-run runs/eef_training_efficiency_v5 --output runs/individual_cnm_reconfigurable_v2 --acquisition-attempts 5 --resume
```

The protocol crosses trained policies with independent acquisition seeds and
24 held-out modular tasks, including new ordering, placement and interface
shifts. First-attempt probes use cloned memory. Targeted deletion, matched
irrelevant deletion and exact restoration examine dependence on stored records.
The equal-information control uses canonical role adjacency and minimum-duration
records; it is a defined comparator, not a full implementation of E-Graphs.

Context-conditioned evidence is evaluated with:

```text
python -m cnm_swarm_sim.run_context_evidence_ablation --individual-run runs/individual_cnm_reconfigurable_v2 --eef-run runs/eef_training_efficiency_v5 --output runs/context_evidence_ablation_v2 --resume
```

## Collective composition

```text
python -m cnm_swarm_sim.run_collective_cnm_experiment --eef-run runs/eef_training_efficiency_v5 --output runs/collective_cnm_v5
```

Teams of 2, 4 and 6 robots receive partitioned histories across three modular
tasks. Conditions include private history, best-history replication, pooling,
unrestricted exchange, selective transfer, source withholding and feedback
withholding. Whole-team completion requires every robot to complete its task.

The selective experiment uses a pooled candidate view to choose missing
records before transfer to private libraries. Its byte counter measures the
serialized initial transfers; direct return-event merges are not fully charged
as network traffic.

Inspect `collective_trials.csv`, `complementarity_audit.csv`,
`transfer_audit.csv`, `execution_return_events.csv`, and
`adverse_return_probes.csv` under `source_data/`, together with the frozen-stack
manifests. Record/event identifiers connect decisions to their source evidence.

## Figure-level comparison evidence

First run the formation ablation:

```text
python -m cnm_swarm_sim.run_eef_formation_ablation --eef-run runs/eef_training_efficiency_v5 --output runs/eef_formation_ablation_v1 --iterations 700 --batch-size 128 --evaluation-samples 256
```

The analyses associated with **Fig05 CNM Comparison v2** use the following
derived tables in the separate data release:

| Comparison | Evidence table |
|---|---|
| Formation frontier | `comparison_eef_frontier.csv` |
| Rigid-body completion/time | `comparison_eef.csv` |
| Unseen composition, first attempt | `comparison_individual.csv` |
| Complementary histories across team sizes | `comparison_collective.csv` |
| Experience payload per episode | `comparison_collective.csv` |

These describe the component analyses; combined manuscript layouts may use
different panel letters. The workflow creates fresh simulation results and
does not substitute published numerical values for measurements.

`build_formal_evidence_bundle` is an archive-validation utility, not a standalone
step after a fresh clone. It expects an EEF `statistical_summary.json`, a
`mechanism_regression_baseline.json`, and the original figure/manuscript archive.
The formation runner currently emits `summary.json` with a different schema;
renaming it does not supply the missing statistics. The full archive must be
provided before invoking this utility. Its statistical and artifact checks
remain intact; missing archive files must not be represented as passed checks.

## Reproducibility

Keep the checkpoint, run manifest, seed, initial memory snapshot, source hashes
and event ledger together. A matching seed alone does not reproduce a result
if the checkpoint, execution stack, incoming evidence or evaluation time changes.
The record digest excludes the intervention audit log so that exact restoration
can recover the adaptive state while retaining the deletion/restoration trace.

`MANIFEST.json` and `MANIFEST.csv` identify files in this distribution.
`docs/RUNTIME_BASELINE.json` binds the unchanged runtime sources and the restored
checkpoint-hash dependency. Run `python tools/verify_release.py` to check
these files. [Third-party provenance](THIRD_PARTY_NOTICE.md) documents external
dependencies.
