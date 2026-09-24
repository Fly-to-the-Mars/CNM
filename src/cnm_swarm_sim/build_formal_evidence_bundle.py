"""Audit formal CNM runs and build compact source tables for Figs. 5 and 6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .run_reconfigurable_individual_cnm import _write_csv, _write_json


def _typed(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        numeric = float(value)
        return int(numeric) if numeric.is_integer() and all(c not in value.lower() for c in (".", "e")) else numeric
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [{key: _typed(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _group(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/formal_cnm_evidence_v1"))
    parser.add_argument("--eef", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--formation", type=Path, default=Path("runs/eef_formation_ablation_v1"))
    parser.add_argument("--individual", type=Path, default=Path("runs/individual_cnm_reconfigurable_v2"))
    parser.add_argument("--collective", type=Path, default=Path("runs/collective_cnm_v5"))
    parser.add_argument("--context", type=Path, default=Path("runs/context_evidence_ablation_v2"))
    parser.add_argument("--figure-root", type=Path, default=Path("../Figure_revision"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.output / "source_data"
    eef_summary = json.loads((args.eef / "statistical_summary.json").read_text(encoding="utf-8"))
    formation_summary = json.loads((args.formation / "statistical_summary.json").read_text(encoding="utf-8"))
    individual_summary = json.loads((args.individual / "statistical_summary.json").read_text(encoding="utf-8"))
    collective_summary = json.loads((args.collective / "statistical_summary.json").read_text(encoding="utf-8"))
    context_summary = json.loads((args.context / "statistical_summary.json").read_text(encoding="utf-8"))

    eef_rows = []
    for method, metrics in eef_summary["by_method"].items():
        eef_rows.append({
            "method": method,
            "policy_seeds": 5,
            "robust_agile_area_mean": metrics["goal_safe_area"]["mean"],
            "robust_agile_area_sd": metrics["goal_safe_area"]["sd"],
            "cnm_ready_yield_mean": metrics["cnm_ready_1p6"]["mean"],
            "cnm_ready_yield_sd": metrics["cnm_ready_1p6"]["sd"],
            "terminal_error_mean": metrics["joint_terminal_error"]["mean"],
            "exit_dispersion_mean": metrics["exit_dispersion"]["mean"],
            "pybullet_completion": metrics["pybullet_completion"],
            "pybullet_time_mean_s": metrics["pybullet_time_mean_s"],
        })
    _write_csv(source / "comparison_eef.csv", eef_rows)

    eef_trials = _read_csv(args.eef / "source_data" / "final_trials.csv")
    frontier_rows = []
    nominal = [row for row in eef_trials if row["condition"] == "nominal"]
    for (method, policy_seed, speed), rows in _group(
        nominal, ("method", "policy_seed", "commanded_speed_mps")
    ).items():
        goal_safe = [
            bool(row["safe_completion"]) and float(row["interface_position_error_m"]) <= 0.35
            for row in rows
        ]
        frontier_rows.append({
            "method": method,
            "policy_seed": policy_seed,
            "commanded_speed_mps": speed,
            "goal_safe_completion": float(np.mean(goal_safe)),
            "attempts": len(rows),
        })
    _write_csv(source / "comparison_eef_frontier.csv", frontier_rows)

    individual_trials = _read_csv(args.individual / "source_data" / "heldout_first_attempts.csv")
    individual_rows = []
    for (method, policy_seed, acquisition_seed), rows in _group(
        individual_trials, ("method", "policy_seed", "acquisition_run")
    ).items():
        individual_rows.append({
            "method": method,
            "policy_seed": policy_seed,
            "acquisition_seed": acquisition_seed,
            "first_attempt_success": float(np.mean([bool(row["success"]) for row in rows])),
            "supported_fraction": float(np.mean([bool(row["supported"]) for row in rows])),
            "median_planning_time_ms": float(np.median([float(row["planning_time_ms"]) for row in rows])),
            "mean_memory_bytes": float(np.mean([float(row["memory_bytes"]) for row in rows])),
            "tasks": len(rows),
        })
    _write_csv(source / "comparison_individual.csv", individual_rows)

    collective_trials = _read_csv(args.collective / "source_data" / "collective_trials.csv")
    transfer_audit = _read_csv(args.collective / "source_data" / "transfer_audit.csv")
    collective_rows = []
    for (condition, policy_seed, team_size), rows in _group(
        collective_trials, ("condition", "policy_seed", "team_size")
    ).items():
        collective_rows.append({
            "condition": condition,
            "policy_seed": policy_seed,
            "team_size": team_size,
            "whole_team_completion": float(np.mean([bool(row["success"]) for row in rows])),
            "cnm_capability_completion": float(np.mean([bool(row["cnm_capability_success"]) for row in rows])),
            "mean_communication_kib": float(np.mean([float(row["communication_kib"]) for row in rows])),
            "episodes": len(rows),
        })
    _write_csv(source / "comparison_collective.csv", collective_rows)

    formation_rows = _read_csv(args.formation / "source_data" / "seed_summary.csv")
    _write_csv(source / "ablation_formation.csv", formation_rows)
    deletion_rows = _read_csv(args.individual / "source_data" / "deletion_interventions.csv")
    deletion_compact = []
    for (branch, policy_seed), rows in _group(deletion_rows, ("branch", "policy_seed")).items():
        deletion_compact.append({
            "branch": branch,
            "policy_seed": policy_seed,
            "success": float(np.mean([bool(row["success"]) for row in rows])),
            "supported": float(np.mean([bool(row["supported"]) for row in rows])),
            "trials": len(rows),
        })
    _write_csv(source / "ablation_deletion.csv", deletion_compact)
    context_rows = _read_csv(args.context / "source_data" / "context_ablation_trials.csv")
    _write_csv(source / "ablation_context.csv", context_rows)

    collective_ablation = []
    for condition, values in collective_summary["conditions"].items():
        collective_ablation.append({"condition": condition, **values})
    _write_csv(source / "ablation_collective.csv", collective_ablation)

    training = _read_csv(args.eef / "source_data" / "training_efficiency.csv")
    iteration_zero = [row for row in training if int(row["iteration"]) == 0]
    return_events = _read_csv(args.collective / "source_data" / "execution_return_events.csv")
    adverse_returns = _read_csv(args.collective / "source_data" / "adverse_return_probes.csv")
    _write_csv(source / "ablation_adverse_return.csv", adverse_returns)
    comp_audit = _read_csv(args.collective / "source_data" / "complementarity_audit.csv")
    formation_full = formation_summary["variants"]["eef_full"]
    formation_controls = [
        formation_summary["variants"][name]
        for name in formation_summary["variants"]
        if name != "eef_full"
    ]
    validations = {
        "eef_all_methods_start_at_zero_experience_and_time": bool(iteration_zero) and all(
            float(row["cnm_ready_yield"]) == 0.0
            and float(row["training_seconds"]) == 0.0
            and int(row["training_scenes"]) == 0
            for row in iteration_zero
        ),
        "formation_five_matched_policy_seeds": len(formation_summary["policy_seeds"]) == 5,
        "formation_full_best_combined_completion": all(
            formation_full["combined_2mps_completion"]["mean"] > item["combined_2mps_completion"]["mean"]
            for item in formation_controls
        ),
        "formation_full_best_connector_yield": all(
            formation_full["downstream_connector_yield"]["mean"] > item["downstream_connector_yield"]["mean"]
            for item in formation_controls
        ),
        "individual_design_is_5x3x24": (
            len(individual_summary["policy_seeds"]) == 5
            and len(individual_summary["acquisition_seeds"]) == 3
            and individual_summary["registered_tasks"] == 24
            and len(individual_trials) == 5 * 3 * 24 * 4
        ),
        "individual_all_tasks_unseen_and_acb_present": (
            individual_summary["all_orders_absent_from_acquisition"]
            and individual_summary["contains_explicit_A+C+B"]
            and all(not bool(row["acquisition_order_seen"]) for row in individual_trials)
        ),
        "individual_weights_frozen": all(not bool(row["policy_weights_updated"]) for row in individual_trials),
        "individual_cnm_exceeds_equal_information": (
            individual_summary["methods"]["cnm"]["first_attempt_success"]
            > individual_summary["methods"]["equal_information_graph"]["first_attempt_success"]
        ),
        "target_deletion_specific_and_restorable": (
            individual_summary["deletion_interventions"]["target_deleted"]["successes"] == 0
            and individual_summary["deletion_interventions"]["matched_deleted"]["successes"]
            == individual_summary["deletion_interventions"]["full"]["successes"]
            and individual_summary["deletion_interventions"]["restored"]["successes"]
            == individual_summary["deletion_interventions"]["full"]["successes"]
        ),
        "collective_design_is_5policy_x_3task_x_246robots_x_3episodes": (
            len(collective_summary["policy_seeds"]) == 5
            and len(collective_summary["episode_seeds"]) == 3
            and collective_summary["team_sizes"] == [2, 4, 6]
            and len(collective_summary["task_ids"]) == 3
            and len(collective_trials) == 5 * 3 * 3 * 3 * 8
        ),
        "collective_complementarity_registered": collective_summary["complementarity_verified"] and all(
            not bool(row["private_support"]) and bool(row["union_support"]) for row in comp_audit
        ),
        "feedback_control_exactly_paired": collective_summary["feedback_withholding_pairing"][
            "all_preintervention_fields_identical"
        ],
        "recipient_events_unique_and_revise_evidence": (
            len({row["event_id"] for row in return_events}) == len(return_events)
            and bool(return_events)
            and all(
                (
                    float(row["reliability_after"]) > float(row["reliability_before"])
                    if bool(row["success"])
                    else float(row["reliability_after"]) < float(row["reliability_before"])
                )
                for row in return_events
            )
        ),
        "adverse_recipient_execution_corrects_source_evidence": (
            bool(adverse_returns)
            and all(not bool(row["physical_success"]) for row in adverse_returns)
            and all(bool(row["recipient_event_added"]) and bool(row["source_event_added"]) for row in adverse_returns)
            and all(
                float(row["source_reliability_after"]) < float(row["source_reliability_before"])
                for row in adverse_returns
            )
            and all(
                float(row["withheld_source_reliability_after"])
                == float(row["withheld_source_reliability_before"])
                for row in adverse_returns
            )
            and all(bool(row["duplicate_return_idempotent"]) for row in adverse_returns)
        ),
        "selective_more_byte_efficient_than_unrestricted": (
            collective_summary["conditions"]["selective_circulation"]["admitted_records_per_kib"]
            > collective_summary["conditions"]["unrestricted_exchange"]["admitted_records_per_kib"]
        ),
        "selective_transfers_use_connector_compatible_chains": (
            bool([
                row for row in transfer_audit
                if row["condition"] in {"selective_circulation", "feedback_withholding"}
            ])
            and all(
                row.get("selection_mode") == "connector_compatible_chain"
                for row in transfer_audit
                if row["condition"] in {"selective_circulation", "feedback_withholding"}
            )
        ),
        "collective_weights_frozen": all(
            not bool(row["policy_weights_updated"]) for row in collective_trials
        ),
        "source_withholding_removes_collective_capability": (
            collective_summary["conditions"]["source_withholding"]["cnm_capability_completion"] == 0.0
        ),
        "context_conditioning_preserves_nominal_capability": (
            context_summary["conditions"]["context_conditioned"]["physical_completion"]
            > context_summary["conditions"]["context_removed"]["physical_completion"]
        ),
        "context_ablation_retains_all_adverse_attempts_and_frozen_weights": (
            context_summary["adverse_failures"] == context_summary["adverse_attempts"]
            and context_summary["adverse_attempts"] > 0
            and all(not bool(row["policy_weights_updated"]) for row in context_rows)
        ),
    }

    protected = [
        args.figure_root / "Fig02_EEF_Training_Efficiency_v5.pdf",
        args.figure_root / "Fig02_EEF_Training_Efficiency_v5.png",
        args.figure_root / "Fig03_Individual_CNM_Growth_v2.pdf",
        args.figure_root / "Fig03_Individual_CNM_Growth_v2.png",
        args.figure_root / "Fig03_Individual_CNM_Growth_v3.pdf",
    ]
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    protected_baseline = json.loads(
        (project_root / "mechanism_regression_baseline.json").read_text(encoding="utf-8")
    )["protected_artifacts"]
    protected_integrity = {
        relative: (
            (workspace_root / relative).is_file()
            and _sha256(workspace_root / relative) == expected.lower()
        )
        for relative, expected in protected_baseline.items()
    }
    validations["protected_legacy_artifacts_match_registered_hashes"] = all(
        protected_integrity.values()
    )
    input_roots = (args.eef, args.formation, args.individual, args.collective, args.context)
    manifest_paths = list(protected)
    manifest_paths.extend([
        workspace_root / "SR_main_.tex",
        project_root / "README.md",
        project_root / "CNM_FIGURE_EVIDENCE_GUIDE_V1.md",
        project_root / "FORMAL_CNM_EXPERIMENT_AUDIT_V1.md",
        project_root / "METHODS_IMPLEMENTATION_TRACEABILITY_V1.md",
        project_root / "src/cnm_swarm_sim/algorithm/eef.py",
        project_root / "src/cnm_swarm_sim/algorithm/experience.py",
        project_root / "src/cnm_swarm_sim/algorithm/memory.py",
        project_root / "src/cnm_swarm_sim/algorithm/frozen.py",
        project_root / "src/cnm_swarm_sim/algorithm/rollout.py",
        project_root / "src/cnm_swarm_sim/algorithm/reconfigurable_protocol.py",
        project_root / "src/cnm_swarm_sim/run_eef_formation_ablation.py",
        project_root / "src/cnm_swarm_sim/run_reconfigurable_individual_cnm.py",
        project_root / "src/cnm_swarm_sim/run_context_evidence_ablation.py",
        project_root / "src/cnm_swarm_sim/run_collective_cnm_experiment.py",
        project_root / "src/cnm_swarm_sim/build_formal_evidence_bundle.py",
        project_root / "src/cnm_swarm_sim/plot_formal_comparison_ablation.py",
        project_root / "src/cnm_swarm_sim/render_collective_cnm_registered_scenes.py",
    ])
    collective_scene_root = (
        args.figure_root / "FigS04_Collective_CNM_Registered_Scenes_v1"
    )
    if collective_scene_root.is_dir():
        manifest_paths.extend(sorted(collective_scene_root.glob("*.png")))
        scene_manifest = collective_scene_root / "scene_manifest.json"
        if scene_manifest.is_file():
            manifest_paths.append(scene_manifest)
    for current_figure in (
        args.figure_root / "Fig05_CNM_Comparison_v2.pdf",
        args.figure_root / "Fig05_CNM_Comparison_v2.metadata.json",
        args.figure_root / "Fig06_CNM_Ablation_v1.pdf",
        args.figure_root / "Fig06_CNM_Ablation_v1.metadata.json",
    ):
        if current_figure.is_file():
            manifest_paths.append(current_figure)
    for root in input_roots:
        manifest_paths.extend(sorted((root / "source_data").glob("*.csv")))
        manifest_paths.extend(path for path in (root / "statistical_summary.json", root / "run_manifest.json") if path.exists())
    manifest = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in dict.fromkeys(manifest_paths)
    ]
    audit = {
        "schema_version": 1,
        "status": "measured_simulation",
        "all_validations_passed": all(validations.values()),
        "validations": validations,
        "comparison_boundary": "matched local reimplementations; no numerical claims are transferred from published papers",
        "hardware_claims_supported": False,
        "protected_legacy_artifacts_modified": not all(protected_integrity.values()),
        "protected_artifact_integrity": protected_integrity,
        "input_manifest": manifest,
    }
    _write_json(args.output / "validation_report.json", audit)
    if not audit["all_validations_passed"]:
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"formal evidence validation failed: {failed}")
    print(json.dumps(audit["validations"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
