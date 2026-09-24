"""Test whether context-conditioned evidence prevents global memory poisoning."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .algorithm.eef import EEFNavigator, load_eef_checkpoint
from .algorithm.experience import FlightSegment
from .algorithm.memory import CNMLibrary, CNMPlanner, ExecutionEvent
from .algorithm.protocol import MODULES
from .algorithm.reconfigurable_protocol import reconfigurable_tasks
from .algorithm.rollout import RolloutPerturbation, run_physical_rollout
from .config import EnvConfig
from .env import CNMSwarmEnv
from .run_reconfigurable_individual_cnm import (
    ACQUISITION_SEEDS,
    POLICY_SEEDS,
    _checkpoint,
    _direct_record,
    _stable_seed,
    _write_csv,
    _write_json,
    execute_task,
    plan_task,
)


ADVERSE_CONTEXT = "east_dynamic_actuation_loss"
QUERY_CONTEXT = "east_nominal"


def _adverse_event(
    model: Any,
    library: CNMLibrary,
    record_id: str,
    policy_seed: int,
    acquisition_seed: int,
    repeat: int,
    output: Path,
    env: CNMSwarmEnv,
) -> tuple[ExecutionEvent, dict[str, Any]]:
    record = library.records[record_id]
    definition = MODULES[record.role]
    placed = CNMPlanner(library).place(record, np.asarray(definition.anchor), 1)
    seed = _stable_seed("context-ablation", policy_seed, acquisition_seed, record.role, repeat)
    log_path = output / "telemetry" / record.role / f"attempt_{repeat:02d}.jsonl"
    result = run_physical_rollout(
        env,
        EEFNavigator(model),
        [[placed]],
        speed=0.94,
        max_steps=150,
        waypoint_tolerance=0.07,
        perturbation=RolloutPerturbation(action_scale=0.0, delay_steps=2, ray_dropout=0.30),
        seed=seed,
        step_log=log_path,
        start_velocities=np.asarray((placed.entry.mean[3:6],)),
        start_yaws=np.asarray((placed.entry.mean[6],)),
        enforce_interface_terminal_state=True,
    )
    segment = FlightSegment.from_jsonl(
        log_path,
        role=record.role,
        structural_key=record.structural_key,
        context=ADVERSE_CONTEXT,
        anchor=np.asarray(definition.anchor),
        direction=1,
        origin_robot=record.origin_robot,
        immutable_hash=library.immutable_hash,
        logical_time=float(60 + repeat),
        result=result.to_dict(),
    )
    canonical_entry = segment.canonical_state(segment.entry_state)
    canonical_exit = segment.canonical_state(segment.exit_state)
    residual = canonical_exit - record.exit.mean
    residual[6] = (residual[6] + math.pi) % (2.0 * math.pi) - math.pi
    normalized_residual = float(np.sqrt(
        np.sum(np.square(residual[:3] / 0.25))
        + np.sum(np.square(residual[3:6] / 0.35))
        + (residual[6] / math.radians(12.0)) ** 2
    ))
    event = ExecutionEvent.create(
        record_id=record_id,
        origin_robot=record.origin_robot,
        executor_robot=library.owner_robot,
        context=ADVERSE_CONTEXT,
        logical_time=float(60 + repeat),
        success=bool(result.success),
        duration=float(result.completion_time),
        min_clearance=float(result.minimum_clearance),
        prediction_error=normalized_residual,
        tracking_error=float(result.mean_acceleration) / 5.0,
        reason="registered_actuation_loss_attempt",
        immutable_hash=library.immutable_hash,
        parent_event_id=record.first_event_id,
        entry_state=canonical_entry,
        command=record.command.terminal_velocity,
        exit_state=canonical_exit,
        namespace=f"context-ablation:{policy_seed}:{acquisition_seed}:{record.role}:{repeat}",
    )
    if not library.add_event(event):
        raise RuntimeError("adverse event rejected")
    return event, {
        "policy_seed": policy_seed,
        "acquisition_seed": acquisition_seed,
        "record_id": record_id,
        "role": record.role,
        "repeat": repeat,
        "context": ADVERSE_CONTEXT,
        "event_id": event.event_id,
        "event_success": event.success,
        "prediction_error": event.prediction_error,
        **result.to_dict(),
    }


def _collapse_context(library: CNMLibrary) -> CNMLibrary:
    """Remove context labels while preserving every outcome and provenance field."""

    payload = library.to_dict()
    payload["events"] = []
    for record_payload in payload["records"]:
        record_payload["evidence_event_ids"] = []
    collapsed = CNMLibrary.from_dict(payload)
    for original in sorted(library.events.values(), key=lambda item: item.event_id):
        event = ExecutionEvent.create(
            record_id=original.record_id,
            origin_robot=original.origin_robot,
            executor_robot=original.executor_robot,
            context=QUERY_CONTEXT,
            logical_time=original.logical_time,
            success=original.success,
            duration=original.duration,
            min_clearance=original.min_clearance,
            prediction_error=original.prediction_error,
            tracking_error=original.tracking_error,
            event_type=original.event_type,
            reason="context_label_removed",
            immutable_hash=original.immutable_hash,
            parent_event_id=original.parent_event_id,
            entry_state=original.entry_state,
            command=original.command,
            exit_state=original.exit_state,
            namespace=f"context-collapsed:{original.event_id}",
        )
        if not collapsed.add_event(event):
            raise RuntimeError("context-collapsed event rejected")
    return collapsed


def _evaluate_condition(
    model: Any,
    library: CNMLibrary,
    task: Any,
    condition: str,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    path, diagnostics = plan_task("cnm", library, task)
    cnm_supported = bool(path)
    execution_path = path if path else [_direct_record(task, library.immutable_hash, "context_ablation_fallback")]
    physical = execute_task(model, library, task, condition, execution_path, output, seed)
    return {
        "condition": condition,
        "cnm_supported": cnm_supported,
        "selected_chain_reliability": diagnostics.get("selected_chain_reliability"),
        "planner_reason": diagnostics.get("reason"),
        "fallback_used": not cnm_supported,
        **physical,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/context_evidence_ablation_v2"))
    parser.add_argument("--individual-run", type=Path, default=Path("runs/individual_cnm_reconfigurable_v2"))
    parser.add_argument("--eef-run", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument(
        "--adverse-repeats",
        type=int,
        default=30,
        help=(
            "Independent actuation-loss attempts per response. The registered "
            "stress dose is 30; it is evaluated against the unchanged chain-reliability gate."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-limit", type=int)
    parser.add_argument("--acquisition-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_summary_path = args.output / "statistical_summary.json"
    previous_elapsed = 0.0
    if args.resume and previous_summary_path.exists():
        previous_elapsed = float(
            json.loads(previous_summary_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0)
        )
    policies = POLICY_SEEDS[:1] if args.smoke else POLICY_SEEDS
    acquisitions = ACQUISITION_SEEDS[:1] if args.smoke else ACQUISITION_SEEDS
    adverse_repeats = 2 if args.smoke else args.adverse_repeats
    if args.policy_limit:
        policies = policies[: args.policy_limit]
    if args.acquisition_limit:
        acquisitions = acquisitions[: args.acquisition_limit]
    task = next(task for task in reconfigurable_tasks() if task.task_id == "R19_ABCD_new_placement")
    started = time.perf_counter()
    adverse_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for policy_seed in policies:
        model, _ = load_eef_checkpoint(
            _checkpoint(args.eef_run, policy_seed), device=args.device, allow_source_rebind=True
        )
        for acquisition_seed in acquisitions:
            cell_output = args.output / "cells" / f"policy_{policy_seed}" / f"acq_{acquisition_seed}"
            bundle_path = cell_output / "cell_results.json"
            if args.resume and bundle_path.exists():
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                adverse_rows.extend(bundle["adverse_attempts"])
                trial_rows.extend(bundle["evaluation"])
                continue
            source = args.individual_run / "cells" / f"policy_{policy_seed}" / f"acq_{acquisition_seed}" / "private_memory.json"
            library = CNMLibrary.from_dict(json.loads(source.read_text(encoding="utf-8")))
            cell_adverse: list[dict[str, Any]] = []
            env = CNMSwarmEnv(EnvConfig(num_drones=1, ray_count=model.config.ray_count, neighbor_k=0, episode_seconds=8.0))
            try:
                for record in sorted(library.records.values(), key=lambda item: item.role):
                    for repeat in range(adverse_repeats):
                        _, row = _adverse_event(
                            model, library, record.record_id, policy_seed, acquisition_seed,
                            repeat, cell_output, env,
                        )
                        cell_adverse.append(row)
            finally:
                env.close()
            contextual = library
            collapsed = _collapse_context(library)
            seed = _stable_seed("context-evaluation", policy_seed, acquisition_seed, task.task_id)
            cell_trials = []
            for condition, candidate in (("context_conditioned", contextual), ("context_removed", collapsed)):
                reliabilities = [candidate.reliability(record.record_id, QUERY_CONTEXT, 90.0) for record in candidate.records.values()]
                row = _evaluate_condition(model, candidate, task, condition, seed, cell_output)
                cell_trials.append({
                    "policy_seed": policy_seed,
                    "acquisition_seed": acquisition_seed,
                    "task_id": task.task_id,
                    "adverse_context": ADVERSE_CONTEXT,
                    "query_context": QUERY_CONTEXT,
                    "adverse_events": len(cell_adverse),
                    "adverse_failures": sum(not item["event_success"] for item in cell_adverse),
                    "mean_record_reliability": float(np.mean(reliabilities)),
                    "policy_weights_updated": False,
                    "frozen_stack_sha256": candidate.immutable_hash,
                    **row,
                })
            _write_json(bundle_path, {"complete": True, "adverse_attempts": cell_adverse, "evaluation": cell_trials})
            adverse_rows.extend(cell_adverse)
            trial_rows.extend(cell_trials)
            print(json.dumps({"stage": "cell", "policy_seed": policy_seed, "acquisition_seed": acquisition_seed}), flush=True)
    _write_csv(args.output / "source_data" / "adverse_attempts.csv", adverse_rows)
    _write_csv(args.output / "source_data" / "context_ablation_trials.csv", trial_rows)
    conditions: dict[str, Any] = {}
    for condition in ("context_conditioned", "context_removed"):
        rows = [row for row in trial_rows if row["condition"] == condition]
        conditions[condition] = {
            "cells": len(rows),
            "cnm_support": float(np.mean([bool(row["cnm_supported"]) for row in rows])),
            "physical_completion": float(np.mean([bool(row["success"]) for row in rows])),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in rows])),
            "mean_record_reliability": float(np.mean([float(row["mean_record_reliability"]) for row in rows])),
        }
    aggregation_elapsed = time.perf_counter() - started
    cell_bundles = list((args.output / "cells").glob("policy_*/acq_*/cell_results.json"))
    artifact_span = (
        max(path.stat().st_mtime for path in cell_bundles) - args.output.stat().st_ctime
        if cell_bundles else 0.0
    )
    summary = {
        "schema_version": 1,
        "status": "smoke_measured_simulation" if args.smoke else "measured_simulation",
        "policy_seeds": list(policies),
        "acquisition_seeds": list(acquisitions),
        "adverse_repeats_per_record": adverse_repeats,
        "registered_task": task.task_id,
        "stress_protocol": {
            "perturbation": "zero actuation with two-step delay and 30% ray dropout",
            "dose_fixed_before_full_multiseed_run": True,
            "dose_selection": (
                "A one-policy/one-acquisition mechanism calibration selected the smallest "
                "round dose tested that crossed the unchanged chain-reliability gate. "
                "The full 5x3 run is reported descriptively."
            ),
            "planner_threshold_modified": False,
        },
        "all_adverse_attempts_retained": True,
        "adverse_failures": sum(not row["event_success"] for row in adverse_rows),
        "adverse_attempts": len(adverse_rows),
        "conditions": conditions,
        "elapsed_seconds": max(aggregation_elapsed, artifact_span, previous_elapsed),
        "aggregation_elapsed_seconds": aggregation_elapsed,
    }
    _write_json(args.output / "statistical_summary.json", summary)
    _write_json(args.output / "run_manifest.json", {
        "arguments": vars(args) | {
            "output": str(args.output), "individual_run": str(args.individual_run), "eef_run": str(args.eef_run)
        },
        "summary": summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
