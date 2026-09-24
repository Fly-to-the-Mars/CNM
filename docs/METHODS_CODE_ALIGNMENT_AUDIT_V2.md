# CNM Methods–code alignment audit (16 September 2026)

The CNM logic is coherent when interpreted as a frozen execution stack acting
through evolving verified records and unique events. Supported reachability
and capability now use the full adaptive state (records plus event ledger),
because reliability changes when new execution evidence arrives. The original manuscript
mixed intended stereo/hardware methods with completed range-ray/PyBullet
methods. SR_main_.tex Materials and Methods and Supplementary Methods S1–S9
now describe completed simulation explicitly and label unverified hardware,
mouse and competition-composition protocols. The revised Methods excerpt
compiles as a standalone 14-page PDF.

| CNM mechanism | Code and completed evidence | Failure mode addressed | Metric and causal control | Frozen/adaptive separation |
| --- | --- | --- | --- | --- |
| Embodied formation | algorithm/eef.py; five 700-update EEF-v5 seeds; Fig. 2 PyBullet trials | Earlier text falsely specified stereo CNN, Flightmare, 64 B-splines and an 8-D yaw outcome | Robustness–agility, held-out terminal coverage, exit dispersion, reusable classes and connector yield; no-filter, no-differentiable and no-outcome ablations | EEF checkpoint weights and hashed operators remain unchanged |
| Record verification | algorithm/experience.py and algorithm/memory.py; three-success admission, absolute and Mahalanobis gates | Compiler can use an acquisition-group empirical reference when no timestamped terminal prediction is present | Log the proportion admitted with prospective predictions; compare prospective and fallback gates on held-out segments | Both gates are fixed operators; new telemetry may change experience but not weights |
| Individual composition | run_reconfigurable_individual_cnm.py; 24 read-only held-out tasks including A+C+B | Earlier Methods pooled fixed-8-m/s complexity with separate speed frontiers | First-attempt unseen-order success and old-task retention; targeted/matched deletion and exact restoration | Five frozen policy seeds crossed three acquisition seeds; probes leave snapshots unchanged |
| Collective circulation and return | run_collective_cnm_experiment.py; 2/4/6 teams on three registered modular tasks | Archived runner sometimes substituted expected exit for missing measured exit and made a source-linked event | Count returns with measured exits; source and feedback withholding test distinct causal roles | Fix changes only the experiment runner, not hashed EEF, compiler, planner or rollout operators |
| Source-data reproducibility | validate_methods_contract.py; frozen manifests and archived events | Archived rollout.py hash differs from current source | Full current-code rerun must establish whole-team success and feedback effects | Do not rewrite old manifests or call a source-rebound run bitwise reproduction |

The Methods contract passes 19 configuration checks and fails two archival
integrity checks. Among 1,025 archived collective return rows, 622 have a
matched measured replan exit, 271 a measured successful final state and **132
have no identifiable measured exit**. These 132 rows cannot support a claim of
execution-grounded feedback. The archived collective manifest binds rollout.py
hash 1ab9d347..., while current source hashes to 3d9a8dfa....

The corrected runner skips selected records without a measured exit and writes
skipped_without_measured_exit in each trial. A regression test confirms that
a selected but unexecuted response creates no return event. A new smoke run
at runs/methods_return_smoke_v1 completed all three two-robot modular tasks
under selective circulation and generated 10 source-linked returns;
feedback-withholding generated none. Its current-code Methods contract passes
all 21 checks. This smoke check does not replace the protected
five-policy-seed collective result.

The Fig. 2 validator passes with 600 single-robot, 180 concurrent, 2,400
matched-entry and 19,200 perturbation simulation trials. The Fig. 3 validator
passes with 300 fixed-8-m/s comparison and 100 sequential-growth trials, and
reports no policy-weight update. Both figure datasets were left unchanged.
All 42 repository tests pass.

The next evidence-bearing implementation step is to timestamp and serialize
EEF terminal means and covariances **before** each CNM acquisition response,
then compile records against those prospective predictions and a held-out
repeatability set. This modifies record verification to remove the empirical
fallback attribution gap. Calibration validity, reusable-class yield and
connector yield should improve; a prospective-gate versus fallback ablation
will test the cause while preserving W*. After that, rerun the complete
5-seed, 3-task, 2/4/6-robot collective matrix under the corrected return rule
and current rollout source, and report old/new whole-team completion,
communication and validated return counts side by side. A competition CNM
claim additionally needs serialized precompetition memory, selected chains
and a first-attempt trace. Rosbag point cloud and trajectory alone do not
establish composition.

The AAAS Science Robotics submission template should be checked again at
submission. This Methods revision is structured for reproducibility but is
not a claim that current simulation constitutes audited physical flight.
