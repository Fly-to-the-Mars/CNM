"""Formal unseen-order and new-placement individual CNM experiment.

This runner never writes to the protected Fig. 2 v5 or Fig. 3 v2 studies.  It
uses their frozen EEF checkpoints, recompiles experience from new append-only
physical telemetry, and evaluates all registered module permutations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .algorithm.eef import EEFNavigator, load_eef_checkpoint
from .algorithm.experience import CompilerConfig, ExperienceCompiler, FlightSegment
from .algorithm.frozen import build_frozen_stack_manifest
from .algorithm.memory import (
    CNMLibrary,
    CNMPlanner,
    CommandTemplate,
    ExperienceRecord,
    InterfaceSummary,
    PlacedRecord,
    PlannerConfig,
)
from .algorithm.protocol import MODULES, ROLE_ORDER, make_verified_record
from .algorithm.reconfigurable_protocol import ReconfigurableTask, reconfigurable_tasks
from .algorithm.rollout import PhysicalRolloutResult, ReplanCallback, RolloutPerturbation, run_physical_rollout
from .config import EnvConfig
from .env import CNMSwarmEnv


METHODS = ("frozen_policy", "nearest_replay", "equal_information_graph", "cnm")
POLICY_SEEDS = (17, 114, 211, 308, 405)
ACQUISITION_SEEDS = (1301, 2603, 3907)


def _stable_seed(*parts: Any) -> int:
    value = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(value[:4], "little") % (2**31 - 1)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _checkpoint(root: Path, seed: int) -> Path:
    path = root / "training" / "eef_full" / f"seed_{seed}" / "eef_policy.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _scaffold(record: ExperienceRecord, direction: int = 1) -> list[PlacedRecord]:
    library = CNMLibrary("collection_scaffold", record.immutable_hash)
    library.add_record(record)
    definition = MODULES[record.role]
    path, diagnostics = CNMPlanner(library).compose(
        (record.role,),
        (np.asarray(definition.anchor, dtype=np.float64),),
        (record.structural_key,),
        direction=direction,
        context=record.context,
        logical_time=record.created_time,
    )
    if not path:
        raise RuntimeError(diagnostics)
    return path


def acquire_library(
    model: Any,
    frozen_hash: str,
    policy_seed: int,
    acquisition_seed: int,
    output: Path,
    *,
    attempts: int = 5,
    planner_config: PlannerConfig | None = None,
    compiler_config: CompilerConfig | None = None,
) -> tuple[CNMLibrary, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile canonical single-module flights into one private CNM."""

    planner_config = planner_config or PlannerConfig()
    compiler_config = compiler_config or CompilerConfig()
    compiler = ExperienceCompiler(compiler_config)
    owner = f"uav_p{policy_seed}_a{acquisition_seed}"
    library = CNMLibrary(owner, frozen_hash, planner_config)
    attempt_rows: list[dict[str, Any]] = []
    compiler_rows: list[dict[str, Any]] = []
    env = CNMSwarmEnv(
        EnvConfig(
            num_drones=1,
            ray_count=model.config.ray_count,
            neighbor_k=0,
            episode_seconds=24.0,
        )
    )
    try:
        for role_index, role in enumerate(ROLE_ORDER):
            definition = MODULES[role]
            context = "east_nominal"
            segments: list[FlightSegment] = []
            for attempt in range(attempts):
                logical_time = float(role_index * attempts + attempt)
                scaffold, _ = make_verified_record(
                    role,
                    1,
                    origin_robot=owner,
                    immutable_hash=frozen_hash,
                    context=context,
                    logical_time=logical_time,
                )
                # The legacy four-module PoC used +x for every interface.
                # Formal acquisition derives the entry and exit state from the
                # actual local path tangents; this is collection scaffolding,
                # while the retained interface still comes only from telemetry.
                world_points = definition.world_waypoints(1)
                entry_tangent = world_points[1] - world_points[0]
                exit_tangent = world_points[-1] - world_points[-2]
                entry_tangent /= max(np.linalg.norm(entry_tangent), 1.0e-9)
                exit_tangent /= max(np.linalg.norm(exit_tangent), 1.0e-9)
                entry_velocity = 0.68 * entry_tangent
                exit_velocity = 0.68 * exit_tangent
                entry_yaw = math.atan2(float(entry_tangent[1]), float(entry_tangent[0]))
                exit_yaw = math.atan2(float(exit_tangent[1]), float(exit_tangent[0]))
                scaffold.entry.mean[3:6] = entry_velocity
                scaffold.entry.mean[6] = entry_yaw
                scaffold.exit.mean[3:6] = exit_velocity
                scaffold.exit.mean[6] = exit_yaw
                scaffold.command.terminal_velocity = exit_velocity
                scaffold.command.terminal_yaw = exit_yaw
                path = _scaffold(scaffold)
                trial_seed = _stable_seed("formal-acquisition", policy_seed, acquisition_seed, role, attempt)
                rng = np.random.default_rng(trial_seed)
                perturbation = RolloutPerturbation(
                    action_scale=float(rng.uniform(0.98, 1.02)),
                    action_noise=float(rng.uniform(0.003, 0.012)),
                    delay_steps=int(attempt % 2),
                    ray_dropout=float(rng.uniform(0.0, 0.015)),
                    ray_noise=float(rng.uniform(0.0, 0.006)),
                    start_position_std=float(rng.uniform(0.010, 0.028)),
                )
                log_path = output / "telemetry" / f"policy_{policy_seed}" / f"acq_{acquisition_seed}" / role / f"attempt_{attempt:02d}.jsonl"
                placed = path[0]
                result = run_physical_rollout(
                    env,
                    EEFNavigator(model),
                    path and [path],
                    speed=0.94,
                    max_steps=720,
                    # Record termination is registered tightly enough that a
                    # later module placement does not inherit the rollout
                    # evaluator's looser task-completion radius as interface
                    # error.
                    waypoint_tolerance=0.07,
                    perturbation=perturbation,
                    seed=trial_seed,
                    step_log=log_path,
                    start_velocities=np.asarray((placed.entry.mean[3:6],)),
                    start_yaws=np.asarray((placed.entry.mean[6],)),
                    enforce_interface_terminal_state=True,
                )
                segment = FlightSegment.from_jsonl(
                    log_path,
                    role=role,
                    structural_key=np.asarray(definition.structural_key),
                    context=context,
                    anchor=np.asarray(definition.anchor),
                    direction=1,
                    origin_robot=owner,
                    immutable_hash=frozen_hash,
                    logical_time=logical_time,
                    result=result.to_dict(),
                )
                segments.append(segment)
                attempt_rows.append({
                    "policy_seed": policy_seed,
                    "acquisition_seed": acquisition_seed,
                    "origin_robot": owner,
                    "role": role,
                    "attempt": attempt,
                    "accepted_physical_candidate": bool(
                        segment.success and segment.minimum_clearance >= compiler_config.minimum_clearance
                    ),
                    **result.to_dict(),
                })
            compiled = compiler.compile(segments, variant=f"acq-{acquisition_seed}")
            if not library.add_record(compiled.record):
                raise RuntimeError(f"record rejected by frozen stack: {compiled.record.record_id}")
            for event in compiled.events:
                library.add_event(event)
            compiler_rows.append({
                "policy_seed": policy_seed,
                "acquisition_seed": acquisition_seed,
                "origin_robot": owner,
                "role": role,
                "record_id": compiled.record.record_id,
                "record_bytes": len(json.dumps(compiled.record.to_dict(), sort_keys=True).encode("utf-8")),
                **compiled.diagnostics.to_dict(),
            })
    finally:
        env.close()
    return library, attempt_rows, compiler_rows


def merge_acquisition_libraries(
    libraries: list[CNMLibrary], owner: str, frozen_hash: str, config: PlannerConfig
) -> CNMLibrary:
    merged = CNMLibrary(owner, frozen_hash, config)
    for source in libraries:
        packet = source.packet(source.records, event_ids=source.events)
        audit = merged.merge_packet(packet)
        if audit.get("reason") not in {"accepted", "ok", None} and not audit.get("records"):
            raise RuntimeError(audit)
    return merged


def _initial_state(task: ReconfigurableTask) -> tuple[np.ndarray, np.ndarray, float]:
    yaw = float(task.placement_yaws[0])
    definition = MODULES[task.roles[0]]
    local_points = definition.world_waypoints(1) - np.asarray(definition.anchor)
    tangent = local_points[1] - local_points[0]
    tangent /= max(np.linalg.norm(tangent), 1.0e-9)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    velocity = 0.68 * (rotation @ tangent)
    return np.concatenate((task.start, velocity, (yaw,))), velocity, yaw


def _direct_record(task: ReconfigurableTask, frozen_hash: str, label: str) -> PlacedRecord:
    _, velocity, yaw = _initial_state(task)
    entry_state = np.concatenate((task.start, velocity, (yaw,)))
    exit_state = np.concatenate((task.goal, np.zeros(3), (float(task.placement_yaws[-1]),)))
    covariance = np.eye(7) * 0.1
    support = np.ones(7)
    record = ExperienceRecord(
        record_id=f"{label}_{task.task_id}",
        structural_key=np.zeros(1),
        role=label,
        context=task.context,
        entry=InterfaceSummary(entry_state, covariance.copy(), support.copy(), 1),
        command=CommandTemplate(np.asarray((task.goal,)), np.zeros(3), float(task.placement_yaws[-1]), "progress", 12.0),
        exit=InterfaceSummary(exit_state, covariance.copy(), support.copy(), 1),
        origin_robot="none",
        first_event_id="none",
        immutable_hash=frozen_hash,
    )
    return PlacedRecord(record, np.zeros(3), 1, record.entry, record.exit, np.asarray((task.goal,)), placement_yaw=yaw)


def _nearest_path(library: CNMLibrary, task: ReconfigurableTask) -> tuple[list[PlacedRecord], dict[str, Any]]:
    started = time.perf_counter()
    planner = CNMPlanner(library)
    candidates = planner.retrieve(task.roles[0], task.structural_keys[0], task.context, 50.0)
    if not candidates:
        return [], {"supported": False, "reason": "missing_first_role", "planning_time_ms": 1000 * (time.perf_counter() - started)}
    record = min(candidates, key=lambda item: (np.linalg.norm(item.structural_key - task.structural_keys[0]), item.record_id))
    first = planner.place(record, task.anchors[0], 1, placement_yaw=task.placement_yaws[0])
    return [first, _direct_record(task, library.immutable_hash, "nearest_fallback")], {
        "supported": True,
        "reason": "one_episode_plus_memory_free_continuation",
        "connector_checks": 1,
        "connector_accepted": 0,
        "interface_rejections": 1,
        "planning_time_ms": 1000 * (time.perf_counter() - started),
    }


def _equal_graph_path(library: CNMLibrary, task: ReconfigurableTask) -> tuple[list[PlacedRecord], dict[str, Any]]:
    """Equal-record graph using position adjacency and no interface covariance."""

    started = time.perf_counter()
    planner = CNMPlanner(library)
    path: list[PlacedRecord] = []
    checks = 0
    accepted = 0
    observed_edges = set(zip(ROLE_ORDER[:-1], ROLE_ORDER[1:]))
    for role, anchor, key, yaw in zip(task.roles, task.anchors, task.structural_keys, task.placement_yaws):
        candidates = planner.retrieve(role, key, task.context, 50.0)
        if not candidates:
            return [], {"supported": False, "reason": f"missing_role:{role}", "planning_time_ms": 1000 * (time.perf_counter() - started)}
        # Topological memory has no terminal distribution.  It selects the
        # shortest observed episode and tests only scalar spatial adjacency.
        chosen = min(candidates, key=lambda item: (library.duration(item.record_id, task.context, 50.0), item.record_id))
        placed = planner.place(chosen, anchor, 1, placement_yaw=yaw)
        if path:
            checks += 1
            if (path[-1].record.role, placed.record.role) in observed_edges:
                accepted += 1
        path.append(placed)
    supported = accepted == checks
    diagnostics = {
        "supported": supported,
        "reason": "ok" if accepted == checks else "unobserved_topological_edge",
        "connector_checks": checks,
        "connector_accepted": accepted,
        "interface_rejections": checks - accepted,
        "planning_time_ms": 1000 * (time.perf_counter() - started),
        "search_algorithm": "layered_scalar_shortest_path",
    }
    return (path if supported else []), diagnostics


def plan_task(method: str, library: CNMLibrary, task: ReconfigurableTask) -> tuple[list[PlacedRecord], dict[str, Any]]:
    if method == "frozen_policy":
        return [_direct_record(task, library.immutable_hash, "frozen_policy")], {
            "supported": True, "reason": "memory_free_local_goal", "planning_time_ms": 0.0,
            "connector_checks": 0, "connector_accepted": 0, "interface_rejections": 0,
        }
    if method == "nearest_replay":
        return _nearest_path(library, task)
    if method == "equal_information_graph":
        return _equal_graph_path(library, task)
    if method == "cnm":
        initial, _, _ = _initial_state(task)
        return CNMPlanner(library).compose(
            task.roles,
            task.anchors,
            task.structural_keys,
            direction=1,
            context=task.context,
            logical_time=50.0,
            initial_state=initial,
            placement_yaws=task.placement_yaws,
        )
    raise KeyError(method)


def replan_callback(library: CNMLibrary, task: ReconfigurableTask) -> ReplanCallback:
    planner = CNMPlanner(library)

    def callback(
        _drone: int,
        executed_ids: tuple[str, ...],
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float,
    ) -> tuple[list[PlacedRecord], dict[str, Any]]:
        completed = len(executed_ids)
        if completed >= len(task.roles):
            return [], {"supported": False, "reason": "task_complete"}
        predecessor_record = library.records.get(executed_ids[-1])
        if predecessor_record is None:
            return [], {"supported": False, "reason": "predecessor_missing"}
        predecessor = planner.place(
            predecessor_record,
            task.anchors[completed - 1],
            1,
            placement_yaw=task.placement_yaws[completed - 1],
        )
        measured = np.concatenate((position, velocity, (yaw,)))
        return planner.compose(
            task.roles[completed:],
            task.anchors[completed:],
            task.structural_keys[completed:],
            direction=1,
            context=task.context,
            logical_time=50.0 + 0.1 * completed,
            predecessor=predecessor,
            predecessor_state=measured,
            placement_yaws=task.placement_yaws[completed:],
        )

    return callback


def execute_task(
    model: Any,
    library: CNMLibrary,
    task: ReconfigurableTask,
    method: str,
    path: list[PlacedRecord],
    output: Path,
    trial_seed: int,
) -> dict[str, Any]:
    if not path:
        return {"success": False, "physical_attempted": False}
    env = CNMSwarmEnv(
        EnvConfig(
            num_drones=1,
            ray_count=model.config.ray_count,
            neighbor_k=0,
            episode_seconds=60.0,
            scenario="reconfigurable_industrial",
            reconfigurable_task_id=task.task_id,
            arena_length=32.0,
            arena_width=40.0,
        )
    )
    try:
        _, velocity, yaw = _initial_state(task)
        result = run_physical_rollout(
            env,
            EEFNavigator(model),
            [path],
            speed=0.86,
            max_steps=1800,
            waypoint_tolerance=0.34,
            perturbation=RolloutPerturbation(action_noise=0.008, start_position_std=0.018),
            seed=trial_seed,
            step_log=output / "telemetry" / "evaluation" / f"{task.task_id}_{method}.jsonl",
            start_positions=np.asarray((task.start,)),
            start_velocities=np.asarray((velocity,)),
            start_yaws=np.asarray((yaw,)),
            replan_callbacks=[replan_callback(library, task) if method.startswith("cnm") else None],
            enforce_interface_terminal_state=method != "frozen_policy",
            # The bridge controller converges to the registered entry
            # velocity within the same 0.35-s horizon.  A high position gain
            # overshoots the velocity support in curved module placements.
            bridge_position_gain=1.2,
            replan_query_tolerance=0.06,
            ordered_checkpoints=[np.stack(task.checkpoints)],
            checkpoint_tolerance=0.64,
            stop_at_final_waypoint=True,
        )
        payload = result.to_dict()
        payload["physical_attempted"] = True
        payload["goal_error_m"] = float(np.linalg.norm(result.final_positions[0] - task.goal))
        return payload
    finally:
        env.close()


def evaluate_cell(
    model: Any,
    library: CNMLibrary,
    policy_seed: int,
    acquisition_run: str,
    tasks: tuple[ReconfigurableTask, ...],
    output: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    memory_bytes = len(json.dumps(library.to_dict(), sort_keys=True).encode("utf-8"))
    before = library.digest()
    for task in tasks:
        for method in METHODS:
            path, diagnostics = plan_task(method, library, task)
            # Pair the reset and perturbation realization across methods.
            seed = _stable_seed("formal-eval", policy_seed, acquisition_run, task.task_id)
            physical = execute_task(model, library, task, method, path, output, seed)
            checks = int(diagnostics.get("connector_checks", 0))
            accepted = int(diagnostics.get("connector_accepted", 0))
            rows.append({
                "policy_seed": policy_seed,
                "acquisition_run": acquisition_run,
                "task_id": task.task_id,
                "task_order": "+".join(chr(65 + ROLE_ORDER.index(role)) for role in task.roles),
                "task_length": task.length,
                "interface_shift_class": task.interface_shift_class,
                "acquisition_order_seen": task.acquisition_order_seen,
                "method": method,
                "supported": bool(diagnostics.get("supported", False)),
                "planner_reason": diagnostics.get("reason"),
                "planner_query_id": diagnostics.get("query_id"),
                "search_algorithm": diagnostics.get("search_algorithm"),
                "planning_time_ms": float(diagnostics.get("planning_time_ms", 0.0)),
                "graph_nodes_expanded": int(diagnostics.get("expanded", 0)),
                "selected_chain_reliability": diagnostics.get("selected_chain_reliability"),
                "connector_checks": checks,
                "connector_accepted": accepted,
                "interface_rejections": int(diagnostics.get("interface_rejections", checks - accepted)),
                "connector_yield": accepted / checks if checks else None,
                "selected_records": "|".join(item.record.record_id for item in path),
                "memory_bytes": 0 if method == "frozen_policy" else memory_bytes,
                "frozen_stack_sha256": library.immutable_hash,
                "policy_weights_updated": False,
                **physical,
            })
    if library.digest() != before:
        raise RuntimeError("read-only held-out probes changed the adaptive state")
    return rows


def _stage_library(library: CNMLibrary, stage: int) -> CNMLibrary:
    snapshot = CNMLibrary.from_dict(library.to_dict())
    allowed = set(ROLE_ORDER[:stage])
    snapshot.delete_records(
        [record_id for record_id, record in snapshot.records.items() if record.role not in allowed],
        reason=f"read_only_acquisition_stage:{stage}",
    )
    return snapshot


def growth_experiment(
    model: Any,
    library: CNMLibrary,
    policy_seed: int,
    acquisition_run: str,
    tasks: tuple[ReconfigurableTask, ...],
    final_rows: list[dict[str, Any]],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure first-attempt growth and retention at frozen memory snapshots."""

    final_lookup = {
        row["task_id"]: row for row in final_rows if row["method"] == "cnm"
    }
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    previous_success: set[str] = set()
    original_digest = library.digest()
    for stage in range(len(ROLE_ORDER) + 1):
        snapshot = _stage_library(library, stage)
        supported_tasks: set[str] = set()
        successful_tasks: set[str] = set()
        for task in tasks:
            if stage == len(ROLE_ORDER):
                final = final_lookup[task.task_id]
                supported = bool(final["supported"])
                success = bool(final.get("success"))
                attempted = bool(final.get("physical_attempted"))
                reason = str(final.get("planner_reason"))
                planning_ms = float(final.get("planning_time_ms", 0.0))
            else:
                path, diagnostics = plan_task("cnm", snapshot, task)
                supported = bool(path)
                reason = str(diagnostics.get("reason"))
                planning_ms = float(diagnostics.get("planning_time_ms", 0.0))
                if path:
                    result = execute_task(
                        model,
                        snapshot,
                        task,
                        f"cnm_growth_stage_{stage}",
                        path,
                        output,
                        _stable_seed("growth", policy_seed, acquisition_run, stage, task.task_id),
                    )
                    success = bool(result.get("success"))
                    attempted = True
                else:
                    success = False
                    attempted = False
            if supported:
                supported_tasks.add(task.task_id)
            if success:
                successful_tasks.add(task.task_id)
            trial_rows.append({
                "policy_seed": policy_seed,
                "acquisition_run": acquisition_run,
                "stage": stage,
                "task_id": task.task_id,
                "task_length": task.length,
                "supported": supported,
                "success": success,
                "physical_attempted": attempted,
                "planner_reason": reason,
                "planning_time_ms": planning_ms,
                "memory_records": len(snapshot.records),
                "memory_digest": snapshot.digest(),
                "frozen_stack_sha256": library.immutable_hash,
                "policy_weights_updated": False,
            })
        retained = len(previous_success & successful_tasks)
        summary_rows.append({
            "policy_seed": policy_seed,
            "acquisition_run": acquisition_run,
            "stage": stage,
            "responses_available": stage,
            "memory_records": len(snapshot.records),
            "supported_tasks": len(supported_tasks),
            "successful_tasks": len(successful_tasks),
            "coverage": len(successful_tasks) / len(tasks),
            "previous_successful_tasks": len(previous_success),
            "retained_tasks": retained,
            "retention_rate": retained / len(previous_success) if previous_success else None,
            "policy_weights_updated": False,
        })
        previous_success = successful_tasks
    if library.digest() != original_digest:
        raise RuntimeError("growth probes mutated the acquisition memory")
    return trial_rows, summary_rows


def deletion_interventions(
    model: Any,
    library: CNMLibrary,
    policy_seed: int,
    acquisition_run: str,
    tasks: tuple[ReconfigurableTask, ...],
    final_rows: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    """Causal target deletion, matched deletion and bit-exact restoration."""

    successful = {
        row["task_id"] for row in final_rows
        if row["method"] == "cnm" and bool(row.get("success"))
    }
    candidates = [task for task in tasks if task.task_id in successful and task.length <= 3]
    candidates.sort(key=lambda task: (-task.length, task.task_id))
    rows: list[dict[str, Any]] = []
    for task in candidates[:4]:
        base_digest = library.digest()
        full_path, full_diagnostics = plan_task("cnm", library, task)
        if not full_path:
            continue
        target = full_path[len(full_path) // 2].record
        target_bytes = len(json.dumps(target.to_dict(), sort_keys=True).encode("utf-8"))
        irrelevant = [record for record in library.records.values() if record.role not in task.roles]
        if not irrelevant:
            continue
        matched = min(
            irrelevant,
            key=lambda record: (
                abs(len(json.dumps(record.to_dict(), sort_keys=True).encode("utf-8")) - target_bytes),
                abs(len(record.evidence_event_ids) - len(target.evidence_event_ids)),
                record.record_id,
            ),
        )
        branch_specs = (
            ("full", None),
            ("target_deleted", target),
            ("matched_deleted", matched),
        )
        trial_seed = _stable_seed("deletion", policy_seed, acquisition_run, task.task_id)
        for branch, removed_record in branch_specs:
            removed: dict[str, ExperienceRecord] = {}
            before = library.digest()
            if removed_record is not None:
                removed = library.delete_records(
                    (removed_record.record_id,),
                    reason=branch,
                    logical_time=80.0,
                )
            path, diagnostics = plan_task("cnm", library, task)
            result = execute_task(
                model, library, task, f"cnm_intervention_{branch}", path, output, trial_seed
            ) if path else {"success": False, "physical_attempted": False}
            rows.append({
                "policy_seed": policy_seed,
                "acquisition_run": acquisition_run,
                "task_id": task.task_id,
                "branch": branch,
                "removed_record_id": None if removed_record is None else removed_record.record_id,
                "removed_role": None if removed_record is None else removed_record.role,
                "removed_bytes": 0 if removed_record is None else len(json.dumps(removed_record.to_dict(), sort_keys=True).encode("utf-8")),
                "removed_evidence_events": 0 if removed_record is None else len(removed_record.evidence_event_ids),
                "supported": bool(path),
                "success": bool(result.get("success")),
                "physical_attempted": bool(result.get("physical_attempted")),
                "planner_reason": diagnostics.get("reason"),
                "digest_before_branch": before,
                "digest_during_branch": library.digest(),
                "frozen_stack_sha256": library.immutable_hash,
                "policy_weights_updated": False,
            })
            if removed:
                library.restore_records(removed, reason=f"restore_after:{branch}", logical_time=81.0)
        restored_path, restored_diagnostics = plan_task("cnm", library, task)
        restored_result = execute_task(
            model,
            library,
            task,
            "cnm_intervention_restored",
            restored_path,
            output,
            trial_seed,
        ) if restored_path else {"success": False, "physical_attempted": False}
        rows.append({
            "policy_seed": policy_seed,
            "acquisition_run": acquisition_run,
            "task_id": task.task_id,
            "branch": "restored",
            "removed_record_id": target.record_id,
            "removed_role": target.role,
            "removed_bytes": target_bytes,
            "removed_evidence_events": len(target.evidence_event_ids),
            "supported": bool(restored_path),
            "success": bool(restored_result.get("success")),
            "physical_attempted": bool(restored_result.get("physical_attempted")),
            "planner_reason": restored_diagnostics.get("reason"),
            "digest_before_branch": base_digest,
            "digest_during_branch": library.digest(),
            "restored_digest_exact": library.digest() == base_digest,
            "frozen_stack_sha256": library.immutable_hash,
            "policy_weights_updated": False,
        })
        if library.digest() != base_digest:
            raise RuntimeError("exact restoration did not recover adaptive state")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/individual_cnm_reconfigurable_v1"))
    parser.add_argument("--eef-run", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--acquisition-attempts", type=int, default=5)
    parser.add_argument("--policy-limit", type=int)
    parser.add_argument("--acquisition-limit", type=int)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only cells with a complete typed cell_results.json bundle",
    )
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
    policy_seeds = POLICY_SEEDS[:1] if args.smoke else POLICY_SEEDS
    acquisition_seeds = ACQUISITION_SEEDS[:1] if args.smoke else ACQUISITION_SEEDS
    tasks = reconfigurable_tasks()[:3] if args.smoke else reconfigurable_tasks()
    if args.policy_limit is not None:
        policy_seeds = policy_seeds[: args.policy_limit]
    if args.acquisition_limit is not None:
        acquisition_seeds = acquisition_seeds[: args.acquisition_limit]
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]
    planner_config = PlannerConfig()
    compiler_config = CompilerConfig()
    all_attempts: list[dict[str, Any]] = []
    all_compiler: list[dict[str, Any]] = []
    all_evaluation: list[dict[str, Any]] = []
    all_growth_trials: list[dict[str, Any]] = []
    all_growth_summary: list[dict[str, Any]] = []
    all_interventions: list[dict[str, Any]] = []
    frozen_manifests: list[dict[str, Any]] = []
    started = time.perf_counter()
    for policy_seed in policy_seeds:
        model, payload = load_eef_checkpoint(
            _checkpoint(args.eef_run, policy_seed), device=args.device, allow_source_rebind=True
        )
        manifest = build_frozen_stack_manifest(
            policy_sha256=payload["immutable_sha256"],
            policy_config=model.config,
            planner_config=planner_config,
            compiler_config=compiler_config,
        )
        frozen_hash = manifest.frozen_stack_sha256
        frozen_manifests.append({"policy_seed": policy_seed, **manifest.to_dict()})
        for acquisition_seed in acquisition_seeds:
            cell_dir = args.output / "cells" / f"policy_{policy_seed}" / f"acq_{acquisition_seed}"
            bundle_path = cell_dir / "cell_results.json"
            acquisition_attempts = 3 if args.smoke else args.acquisition_attempts
            cell_signature = hashlib.sha256(json.dumps({
                "policy_seed": policy_seed,
                "acquisition_seed": acquisition_seed,
                "task_ids": [task.task_id for task in tasks],
                "acquisition_attempts": acquisition_attempts,
                "frozen_stack_sha256": frozen_hash,
            }, sort_keys=True).encode("utf-8")).hexdigest()
            if args.resume and bundle_path.exists():
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                if not bundle.get("complete") or bundle.get("cell_signature") != cell_signature:
                    raise RuntimeError(f"incompatible resume bundle: {bundle_path}")
                all_attempts.extend(bundle["acquisition_attempts"])
                all_compiler.extend(bundle["compiler_records"])
                all_evaluation.extend(bundle["heldout_first_attempts"])
                all_growth_trials.extend(bundle["growth_trials"])
                all_growth_summary.extend(bundle["growth_summary"])
                all_interventions.extend(bundle["deletion_interventions"])
                continue
            library, attempts, compiler = acquire_library(
                model,
                frozen_hash,
                policy_seed,
                acquisition_seed,
                cell_dir,
                attempts=acquisition_attempts,
                planner_config=planner_config,
                compiler_config=compiler_config,
            )
            all_attempts.extend(attempts)
            all_compiler.extend(compiler)
            _write_json(cell_dir / "private_memory.json", library.to_dict())
            # Each acquisition run is evaluated independently.  No pooled
            # audit is included in the registered 5 x 3 estimates.
            cell_rows = evaluate_cell(
                model, library, policy_seed, str(acquisition_seed), tasks, cell_dir
            )
            all_evaluation.extend(cell_rows)
            growth_trials, growth_summary = growth_experiment(
                model,
                library,
                policy_seed,
                str(acquisition_seed),
                tasks,
                cell_rows,
                cell_dir,
            )
            all_growth_trials.extend(growth_trials)
            all_growth_summary.extend(growth_summary)
            intervention_rows = deletion_interventions(
                model,
                library,
                policy_seed,
                str(acquisition_seed),
                tasks,
                cell_rows,
                cell_dir,
            )
            all_interventions.extend(intervention_rows)
            _write_json(bundle_path, {
                "complete": True,
                "cell_signature": cell_signature,
                "policy_seed": policy_seed,
                "acquisition_seed": acquisition_seed,
                "frozen_stack_sha256": frozen_hash,
                "acquisition_attempts": attempts,
                "compiler_records": compiler,
                "heldout_first_attempts": cell_rows,
                "growth_trials": growth_trials,
                "growth_summary": growth_summary,
                "deletion_interventions": intervention_rows,
            })

    _write_csv(args.output / "source_data" / "acquisition_attempts.csv", all_attempts)
    _write_csv(args.output / "source_data" / "compiler_records.csv", all_compiler)
    _write_csv(args.output / "source_data" / "heldout_first_attempts.csv", all_evaluation)
    _write_csv(args.output / "source_data" / "growth_trials.csv", all_growth_trials)
    _write_csv(args.output / "source_data" / "growth_summary.csv", all_growth_summary)
    _write_csv(args.output / "source_data" / "deletion_interventions.csv", all_interventions)
    summary: dict[str, Any] = {"methods": {}}
    for method in METHODS:
        rows = [row for row in all_evaluation if row["method"] == method]
        summary["methods"][method] = {
            "attempts": len(rows),
            "supported": sum(bool(row["supported"]) for row in rows),
            "successes": sum(bool(row.get("success")) for row in rows),
            "first_attempt_success": float(np.mean([bool(row.get("success")) for row in rows])),
            "median_planning_time_ms": float(np.median([row["planning_time_ms"] for row in rows])),
            "mean_memory_bytes": float(np.mean([row["memory_bytes"] for row in rows])),
        }
    aggregation_elapsed = time.perf_counter() - started
    cell_bundles = list((args.output / "cells").glob("policy_*/acq_*/cell_results.json"))
    artifact_span = (
        max(path.stat().st_mtime for path in cell_bundles) - args.output.stat().st_ctime
        if cell_bundles else 0.0
    )
    summary.update({
        "schema_version": 1,
        "status": "measured_simulation" if not args.smoke else "smoke_measured_simulation",
        "policy_seeds": list(policy_seeds),
        "acquisition_seeds": list(acquisition_seeds),
        "registered_tasks": len(tasks),
        "expected_cells": len(policy_seeds) * len(acquisition_seeds),
        "completed_cells": len(policy_seeds) * len(acquisition_seeds),
        "contains_explicit_A+C+B": any(task.task_id == "R08_ACB_explicit" for task in tasks),
        "all_orders_absent_from_acquisition": all(not task.acquisition_order_seen for task in tasks),
        "elapsed_seconds": max(aggregation_elapsed, artifact_span, previous_elapsed),
        "aggregation_elapsed_seconds": aggregation_elapsed,
    })
    summary["growth"] = {
        str(stage): {
            "mean_coverage": float(np.mean([
                row["coverage"] for row in all_growth_summary if row["stage"] == stage
            ])),
            "mean_retention": (
                float(np.mean([
                    row["retention_rate"] for row in all_growth_summary
                    if row["stage"] == stage and row["retention_rate"] is not None
                ]))
                if any(
                    row["stage"] == stage and row["retention_rate"] is not None
                    for row in all_growth_summary
                ) else None
            ),
        }
        for stage in range(len(ROLE_ORDER) + 1)
    }
    summary["deletion_interventions"] = {
        branch: {
            "trials": len([row for row in all_interventions if row["branch"] == branch]),
            "successes": sum(
                bool(row["success"]) for row in all_interventions if row["branch"] == branch
            ),
        }
        for branch in ("full", "target_deleted", "matched_deleted", "restored")
    }
    _write_json(args.output / "statistical_summary.json", summary)
    _write_json(args.output / "frozen_stack_manifests.json", frozen_manifests)
    _write_json(args.output / "registered_tasks.json", [
        {
            **asdict(task),
            "anchors": [item.tolist() for item in task.anchors],
            "structural_keys": [item.tolist() for item in task.structural_keys],
            "start": task.start.tolist(),
            "goal": task.goal.tolist(),
            "checkpoints": [item.tolist() for item in task.checkpoints],
            "interface_position_shifts": [item.tolist() for item in task.interface_position_shifts],
        }
        for task in tasks
    ])
    _write_json(args.output / "run_manifest.json", {
        "schema_version": 1,
        "arguments": vars(args) | {
            "output": str(args.output),
            "eef_run": str(args.eef_run),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "summary": summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
