"""Formal 2/4/6-robot complementary-history CNM experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .algorithm.eef import EEFNavigator, load_eef_checkpoint
from .algorithm.experience import CompilerConfig
from .algorithm.frozen import build_frozen_stack_manifest
from .algorithm.memory import CNMLibrary, CNMPlanner, ExecutionEvent, PlannerConfig
from .algorithm.protocol import ROLE_ORDER
from .algorithm.reconfigurable_protocol import ReconfigurableTask, reconfigurable_tasks
from .algorithm.rollout import RolloutPerturbation, run_physical_rollout
from .config import EnvConfig
from .env import CNMSwarmEnv
from .run_reconfigurable_individual_cnm import (
    POLICY_SEEDS,
    _checkpoint,
    _initial_state,
    _direct_record,
    _stable_seed,
    _write_csv,
    _write_json,
    acquire_library,
    plan_task,
    replan_callback,
)


TEAM_SIZES = (2, 4, 6)
EPISODE_SEEDS = (7103, 8209, 9311)
COLLECTIVE_TASK_IDS = (
    "R08_ACB_explicit",
    "R14_CDB_position_shift",
    "R22_ABCD_combined",
)
CONDITIONS = (
    "private_cnm",
    "best_history_replication",
    "redundant_histories",
    "one_time_pooling",
    "unrestricted_exchange",
    "feedback_withholding",
    "selective_circulation",
    "source_withholding",
)


def _clone(library: CNMLibrary) -> CNMLibrary:
    return CNMLibrary.from_dict(library.to_dict())


def _team_tasks() -> tuple[ReconfigurableTask, ...]:
    lookup = {task.task_id: task for task in reconfigurable_tasks()}
    return tuple(lookup[task_id] for task_id in COLLECTIVE_TASK_IDS)


def _shift_task(
    task: ReconfigurableTask,
    shift: np.ndarray,
) -> ReconfigurableTask:
    shift = np.asarray(shift, dtype=np.float64)
    if shift.shape != (3,):
        raise ValueError("collective task shift must be a three-dimensional vector")
    return replace(
        task,
        anchors=tuple(anchor + shift for anchor in task.anchors),
        start=task.start + shift,
        goal=task.goal + shift,
        checkpoints=tuple(point + shift for point in task.checkpoints),
    )


def _formation_offsets(team_size: int, task: ReconfigurableTask) -> np.ndarray:
    """Scene-registered lanes that respect each task's corridor width."""

    if task.task_id == "R14_CDB_position_shift":
        if team_size == 2:
            return np.asarray(((0.0, -0.40, 0.0), (0.0, 0.40, 0.0)))
        if team_size == 4:
            return np.asarray(tuple(
                (0.0, y, z) for z in (-0.40, 0.40) for y in (-0.40, 0.40)
            ))
        if team_size == 6:
            return np.asarray(tuple(
                (0.0, y, z) for z in (-0.30, 0.30, 0.90) for y in (-0.40, 0.40)
            ))

    if team_size == 2:
        return np.asarray(((0.0, -0.16, 0.0), (0.0, 0.16, 0.0)))
    if team_size == 4:
        return np.asarray(tuple(
            (0.0, y, z) for z in (-0.16, 0.16) for y in (-0.16, 0.16)
        ))
    if team_size == 6:
        return np.asarray(tuple(
            (0.0, y, z) for z in (-0.28, 0.0, 0.28) for y in (-0.16, 0.16)
        ))
    raise ValueError(f"unregistered collective team size: {team_size}")


def _partition_histories(
    sources: list[CNMLibrary], team_size: int, task: ReconfigurableTask
) -> tuple[list[CNMLibrary], list[dict[str, Any]], str]:
    """Give each actual acquisition source an incomplete private history."""

    histories: list[CNMLibrary] = []
    audit: list[dict[str, Any]] = []
    role_holders: dict[str, int] = {}
    for role_index, role in enumerate(task.roles):
        role_holders[role] = role_index % team_size
    for robot_index in range(team_size):
        source = _clone(sources[robot_index])
        allowed = {role for role, holder in role_holders.items() if holder == robot_index}
        if not allowed:
            # Extra robots retain real but task-irrelevant experience.
            allowed = {ROLE_ORDER[0]}
        source.delete_records(
            [record_id for record_id, record in source.records.items() if record.role not in allowed],
            reason="registered_complementary_partition",
            logical_time=60.0,
        )
        path, diagnostics = plan_task("cnm", source, task)
        histories.append(source)
        audit.append({
            "robot": source.owner_robot,
            "private_roles": "+".join(sorted({record.role for record in source.records.values()})),
            "private_support": bool(path),
            "private_reason": diagnostics.get("reason"),
        })
    pool = _pool(histories, "union_audit")
    union_path, union_diagnostics = plan_task("cnm", pool, task)
    if any(row["private_support"] for row in audit) or not union_path:
        raise RuntimeError({"private": audit, "union": union_diagnostics})
    withheld_role = task.roles[1]
    withheld_holder = role_holders[withheld_role]
    withheld_source = histories[withheld_holder].owner_robot
    return histories, audit, withheld_source


def _pool(libraries: list[CNMLibrary], owner: str) -> CNMLibrary:
    if not libraries:
        raise ValueError("at least one library is required")
    pool = CNMLibrary(owner, libraries[0].immutable_hash, libraries[0].config)
    for library in libraries:
        pool.merge_packet(library.packet(library.records, event_ids=library.events))
    return pool


def _redundant_histories(
    sources: list[CNMLibrary], team_size: int, task: ReconfigurableTask
) -> list[CNMLibrary]:
    """Match record counts with independent, overlapping, incomplete histories."""

    retained_role = task.roles[0]
    histories: list[CNMLibrary] = []
    role_holders = [index % team_size for index in range(len(task.roles))]
    target_counts = [max(1, role_holders.count(index)) for index in range(team_size)]
    for robot_index, source in enumerate(sources[:team_size]):
        history = CNMLibrary(source.owner_robot, source.immutable_hash, source.config)
        for donor in sources[: target_counts[robot_index]]:
            record_ids = [
                record_id for record_id, record in donor.records.items() if record.role == retained_role
            ]
            event_ids = [
                event_id for event_id, event in donor.events.items() if event.record_id in record_ids
            ]
            packet = donor.packet(record_ids, event_ids=event_ids)
            history.merge_packet(packet)
        history.operator_audit.append({
            "operator": "registered_redundant_history_control",
            "retained_role": retained_role,
            "matched_record_count": target_counts[robot_index],
            "logical_time": 60.0,
        })
        histories.append(history)
    return histories


def _prepare_condition(
    private: list[CNMLibrary],
    task: ReconfigurableTask,
    condition: str,
    withheld_source: str,
) -> tuple[list[CNMLibrary], int, list[dict[str, Any]]]:
    histories = [_clone(library) for library in private]
    transfer_rows: list[dict[str, Any]] = []
    bytes_sent = 0
    if condition == "private_cnm":
        return histories, bytes_sent, transfer_rows
    if condition == "best_history_replication":
        best = max(histories, key=lambda item: len({r.role for r in item.records.values()} & set(task.roles)))
        packet = best.packet(best.records, event_ids=best.events)
        size = len(json.dumps(packet, sort_keys=True).encode("utf-8"))
        replicated: list[CNMLibrary] = []
        for recipient in histories:
            empty = CNMLibrary(recipient.owner_robot, recipient.immutable_hash, recipient.config)
            result = empty.merge_packet(packet)
            bytes_sent += size
            transfer_rows.append({"sender": best.owner_robot, "recipient": recipient.owner_robot, "bytes": size, "condition": condition, **result})
            replicated.append(empty)
        return replicated, bytes_sent, transfer_rows

    # The feedback-withholding control must be bit-for-bit matched to selective
    # circulation up to the registered intervention.  Sharing one pool owner
    # keeps packet payloads and byte counts identical; only the recipient
    # execution event is withheld later.
    pool_owner = (
        "selective_circulation_pool"
        if condition in {"selective_circulation", "feedback_withholding"}
        else f"{condition}_pool"
    )
    pool = _pool(histories, pool_owner)
    if condition == "source_withholding":
        pool.withhold_source(withheld_source)
    if condition == "one_time_pooling":
        packet = pool.packet(pool.records, event_ids=pool.events)
        size = len(json.dumps(packet, sort_keys=True).encode("utf-8"))
        for recipient in histories:
            result = recipient.merge_packet(packet)
            bytes_sent += size
            transfer_rows.append({"sender": pool.owner_robot, "recipient": recipient.owner_robot, "bytes": size, "condition": condition, **result})
        return histories, bytes_sent, transfer_rows
    if condition == "unrestricted_exchange":
        originals = [_clone(item) for item in histories]
        for sender in originals:
            packet = sender.packet(sender.records, event_ids=sender.events)
            size = len(json.dumps(packet, sort_keys=True).encode("utf-8"))
            for recipient in histories:
                if recipient.owner_robot == sender.owner_robot:
                    continue
                result = recipient.merge_packet(packet)
                bytes_sent += size
                transfer_rows.append({"sender": sender.owner_robot, "recipient": recipient.owner_robot, "bytes": size, "condition": condition, **result})
        return histories, bytes_sent, transfer_rows

    # Selective circulation, feedback withholding, source withholding and the
    # redundant-history control use the same request path.  Feedback
    # withholding drops recipient outcomes; source withholding removes one
    # registered origin; redundant histories lack complementary roles even in
    # their union.
    for recipient in histories:
        known = set(recipient.records)
        missing = tuple(role for role in task.roles if role not in {r.role for r in recipient.records.values()})
        # Plan once over a read-only candidate union, then transmit only the
        # absent records on the selected connector-compatible chain. This is
        # the task-requested connector semantics in the manuscript. Ranking
        # records independently by role can return individually relevant yet
        # mutually brittle variants after the first measured exit.
        candidate = _clone(recipient)
        candidate.merge_packet(pool.packet(pool.records, event_ids=pool.events))
        chain, _ = plan_task("cnm", candidate, task)
        selected_ids = [
            placed.record.record_id
            for placed in chain
            if placed.record.record_id not in known and placed.record.record_id in pool.records
        ]
        if chain:
            packet = pool.packet(dict.fromkeys(selected_ids))
            size = len(json.dumps(packet, sort_keys=True).encode("utf-8"))
            if size > 48_000:
                raise RuntimeError("registered connector-compatible response exceeds byte budget")
            selection_mode = "connector_compatible_chain"
        else:
            packet, size = pool.select_for_query(
                missing,
                task.context,
                known,
                logical_time=70.0,
                byte_budget=48_000,
                structural_keys={role: key for role, key in zip(task.roles, task.structural_keys)},
            )
            selection_mode = "role_query_no_supported_chain"
        result = recipient.merge_packet(packet)
        bytes_sent += size
        transfer_rows.append({
            "sender": pool.owner_robot,
            "recipient": recipient.owner_robot,
            "bytes": size,
            "condition": condition,
            "requested_roles": "+".join(missing),
            "selection_mode": selection_mode,
            "selected_chain_records": selected_ids,
            **result,
        })
    return histories, bytes_sent, transfer_rows


def _execute_team(
    model: Any,
    histories: list[CNMLibrary],
    task: ReconfigurableTask,
    condition: str,
    team_size: int,
    policy_seed: int,
    episode_seed: int,
    output: Path,
) -> tuple[dict[str, Any], list[ReconfigurableTask]]:
    # The registered offsets remain inside the narrowest task corridor. Peer
    # separation is enforced by the same external safety layer in every
    # communication condition.
    robot_tasks = [_shift_task(task, offset) for offset in _formation_offsets(team_size, task)]
    cnm_paths = []
    diagnostics = []
    for library, robot_task in zip(histories, robot_tasks):
        path, diagnostic = plan_task("cnm", library, robot_task)
        cnm_paths.append(path)
        diagnostics.append(diagnostic)
    supported = [bool(path) for path in cnm_paths]
    # The manuscript specifies a common memory-free local navigator whenever
    # no complete supported chain exists.  We therefore retain the CNM support
    # decision and still execute every registered team episode.
    paths = [
        path if path else [_direct_record(robot_task, library.immutable_hash, "collective_fallback")]
        for path, robot_task, library in zip(cnm_paths, robot_tasks, histories)
    ]
    env = CNMSwarmEnv(
        EnvConfig(
            num_drones=team_size,
            ray_count=model.config.ray_count,
            neighbor_k=min(5, team_size - 1),
            episode_seconds=70.0,
            scenario="reconfigurable_industrial",
            reconfigurable_task_id=task.task_id,
            arena_length=32.0,
            arena_width=40.0,
        )
    )
    try:
        velocities = []
        yaws = []
        for robot_task in robot_tasks:
            _, velocity, yaw = _initial_state(robot_task)
            velocities.append(velocity)
            yaws.append(yaw)
        result = run_physical_rollout(
            env,
            EEFNavigator(model),
            paths,
            speed=0.78,
            max_steps=2000,
            waypoint_tolerance=0.42,
            perturbation=RolloutPerturbation(action_noise=0.006),
            # Every communication condition receives the same reset and
            # disturbance realization.  The manipulated variable is access to
            # past experience, not process noise.
            seed=_stable_seed("collective", policy_seed, team_size, episode_seed),
            step_log=(
                output / "telemetry" / f"policy_{policy_seed}" / f"team_{team_size}"
                / f"{task.task_id}_episode_{episode_seed}_{condition}.jsonl"
            ),
            start_positions=np.stack([item.start for item in robot_tasks]),
            start_velocities=np.stack(velocities),
            start_yaws=np.asarray(yaws),
            replan_callbacks=[
                replan_callback(library, item) if is_supported else None
                for library, item, is_supported in zip(histories, robot_tasks, supported)
            ],
            enforce_interface_terminal_state=True,
            bridge_position_gain=1.2,
            replan_query_tolerance=0.06,
            ordered_checkpoints=[np.stack(item.checkpoints) for item in robot_tasks],
            checkpoint_tolerance=0.66,
            lane_spacing=0.0,
            stop_at_final_waypoint=False,
            external_peer_safety=True,
        )
        payload = result.to_dict()
        payload["physical_attempted"] = True
        payload["supported_members"] = sum(supported)
        payload["fallback_members"] = team_size - sum(supported)
        payload["cnm_supported_team"] = all(supported)
        payload["cnm_capability_success"] = bool(result.success and all(supported))
        payload["planner_reasons"] = [item.get("reason") for item in diagnostics]
        return payload, robot_tasks
    finally:
        env.close()


def _return_execution_evidence(
    histories: list[CNMLibrary],
    result: dict[str, Any],
    robot_tasks: list[ReconfigurableTask],
    logical_time: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reliability_before: list[float] = []
    reliability_after: list[float] = []
    source_lookup = {library.owner_robot: library for library in histories}
    duplicate_stable = True
    skipped_without_measured_exit: list[dict[str, Any]] = []
    for robot_index, (library, record_ids) in enumerate(zip(histories, result.get("selected_record_ids", []))):
        robot_task = robot_tasks[robot_index]
        for record_id in dict.fromkeys(record_ids):
            record = library.records.get(record_id)
            if record is None or record.origin_robot == library.owner_robot:
                continue
            before = library.reliability(record_id, robot_task.context, logical_time)
            measured_exit = None
            boundary_supported = None
            for replan_event in result.get("replan_events", []):
                if (
                    replan_event.get("event_type") == "replan_query"
                    and int(replan_event.get("drone", -1)) == robot_index
                    and replan_event.get("completed_record_id") == record_id
                ):
                    measured_exit = np.asarray(replan_event["measured_exit"], dtype=np.float64)
                    boundary_supported = bool(replan_event.get("supported"))
                    break
            selected = list(dict.fromkeys(record_ids))
            is_final_record = bool(selected and selected[-1] == record_id)
            if measured_exit is None:
                # A successful final traversal supplies an observed terminal
                # state. For an aborted chain, selected ids alone do not show
                # that the final response was ever executed.
                final_completed = bool(
                    result.get("per_drone_success", [False] * len(histories))[robot_index]
                )
                if is_final_record and final_completed:
                    measured_exit = np.concatenate((
                        np.asarray(result["final_positions"][robot_index], dtype=np.float64),
                        np.asarray(result["final_velocities"][robot_index], dtype=np.float64),
                        (float(result["final_yaws"][robot_index]),),
                    ))
                else:
                    skipped_without_measured_exit.append({
                        "robot_index": robot_index,
                        "record_id": record_id,
                        "reason": "selected_record_has_no_measured_exit",
                    })
                    continue
            event_success = (
                bool(result.get("per_drone_success", [False] * len(histories))[robot_index])
                if is_final_record
                else bool(boundary_supported)
            )
            occurrence = robot_task.roles.index(record.role)
            anchor = robot_task.anchors[occurrence]
            placement_yaw = robot_task.placement_yaws[occurrence]
            c, s = math.cos(placement_yaw), math.sin(placement_yaw)
            inverse_rotation = np.asarray(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
            canonical_exit = np.concatenate((
                inverse_rotation @ (measured_exit[:3] - anchor),
                inverse_rotation @ measured_exit[3:6],
                ((measured_exit[6] - placement_yaw + math.pi) % (2.0 * math.pi) - math.pi,),
            ))
            residual = canonical_exit - record.exit.mean
            residual[6] = (residual[6] + math.pi) % (2.0 * math.pi) - math.pi
            prediction_error = float(np.sqrt(
                np.sum(np.square(residual[:3] / 0.25))
                + np.sum(np.square(residual[3:6] / 0.35))
                + (residual[6] / math.radians(12.0)) ** 2
            ))
            event = ExecutionEvent.create(
                record_id=record_id,
                origin_robot=record.origin_robot,
                executor_robot=library.owner_robot,
                context=robot_task.context,
                logical_time=logical_time,
                success=event_success,
                duration=float(result.get("completion_time", 0.0)) / max(len(record_ids), 1),
                min_clearance=float(result.get("minimum_clearance") or 0.0),
                prediction_error=prediction_error,
                tracking_error=float(result.get("mean_acceleration", 0.0)) / 5.0,
                # Recipient executions are ordinary physical attempts whose
                # origin differs from the executor.  Keeping the canonical
                # event type makes them enter the same fixed Beta evidence
                # update as locally acquired attempts.
                event_type="attempt",
                reason="recipient_team_execution",
                immutable_hash=library.immutable_hash,
                parent_event_id=record.first_event_id,
                entry_state=record.entry.mean,
                command=record.command.terminal_velocity,
                exit_state=canonical_exit,
                namespace=f"formal-return:{library.owner_robot}:{record_id}:{logical_time}",
            )
            library.add_event(event)
            source = source_lookup.get(record.origin_robot)
            if source is not None:
                source.add_event(event)
            after = library.reliability(record_id, robot_task.context, logical_time)
            digest = library.digest()
            library.add_event(event)
            duplicate_stable &= library.digest() == digest
            reliability_before.append(before)
            reliability_after.append(after)
            rows.append({
                "recipient": library.owner_robot,
                "origin": record.origin_robot,
                "record_id": record_id,
                "event_id": event.event_id,
                "success": event.success,
                "prediction_error": event.prediction_error,
                "measured_exit_canonical": list(event.exit_state),
                "reliability_before": before,
                "reliability_after": after,
                "parent_event_id": event.parent_event_id,
            })
    audit = {
        "returned_events": len(rows),
        "skipped_without_measured_exit": skipped_without_measured_exit,
        "duplicate_return_idempotent": duplicate_stable,
        "mean_reliability_before": float(np.mean(reliability_before)) if reliability_before else None,
        "mean_reliability_after": float(np.mean(reliability_after)) if reliability_after else None,
    }
    return rows, audit


def _feedback_pairing_audit(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify that feedback withholding changes only the return-event update."""

    identity = lambda row: (
        row["policy_seed"], row["task_id"], row["team_size"], row["episode_seed"]
    )
    selective = {
        identity(row): row for row in trials if row["condition"] == "selective_circulation"
    }
    withheld = {
        identity(row): row for row in trials if row["condition"] == "feedback_withholding"
    }
    paired_fields = (
        "communication_bytes",
        "success",
        "per_drone_success",
        "collision_events",
        "minimum_clearance",
        "minimum_separation",
        "final_positions",
        "final_velocities",
        "supported_members",
        "selected_record_ids",
    )
    mismatches = [
        {"cell": list(cell), "fields": [field for field in paired_fields if left.get(field) != withheld[cell].get(field)]}
        for cell, left in selective.items()
        if cell not in withheld
        or any(left.get(field) != withheld[cell].get(field) for field in paired_fields)
    ]
    return {
        "pairs": len(selective),
        "paired_fields": list(paired_fields),
        "all_preintervention_fields_identical": len(selective) == len(withheld) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _adverse_return_probe(
    model: Any,
    private: list[CNMLibrary],
    task: ReconfigurableTask,
    withheld_source: str,
    team_size: int,
    policy_seed: int,
    episode_seed: int,
    output: Path,
) -> dict[str, Any]:
    """Execute one transferred record under actuation loss and return its evidence.

    The recipient and source start from the same selectively shared snapshot as
    the withholding control.  Only the recipient's measured failure event is
    merged into the source in the return branch.
    """

    histories, byte_count, _ = _prepare_condition(
        private, task, "selective_circulation", withheld_source
    )
    recipient = next(
        library
        for library in histories
        if any(record.origin_robot != library.owner_robot for record in library.records.values())
    )
    record = next(
        record
        for record in sorted(recipient.records.values(), key=lambda item: item.record_id)
        if record.origin_robot != recipient.owner_robot and record.role in task.roles
    )
    source = next(library for library in histories if library.owner_robot == record.origin_robot)
    withheld_control = _clone(source)
    occurrence = task.roles.index(record.role)
    placed = CNMPlanner(recipient).place(
        record,
        task.anchors[occurrence],
        1,
        placement_yaw=task.placement_yaws[occurrence],
    )
    context = "east_dynamic_actuation_loss"
    logical_time = float(300 + episode_seed % 100)
    recipient_before = recipient.reliability(record.record_id, context, logical_time)
    source_before = source.reliability(record.record_id, context, logical_time)
    withheld_before = withheld_control.reliability(record.record_id, context, logical_time)
    env = CNMSwarmEnv(
        EnvConfig(
            num_drones=1,
            ray_count=model.config.ray_count,
            neighbor_k=0,
            episode_seconds=8.0,
            scenario="reconfigurable_industrial",
            reconfigurable_task_id=task.task_id,
            arena_length=32.0,
            arena_width=40.0,
        )
    )
    log_path = (
        output / "telemetry" / "adverse_return"
        / f"policy_{policy_seed}_{task.task_id}_team_{team_size}_episode_{episode_seed}.jsonl"
    )
    try:
        result = run_physical_rollout(
            env,
            EEFNavigator(model),
            [[placed]],
            speed=0.78,
            max_steps=150,
            waypoint_tolerance=0.07,
            perturbation=RolloutPerturbation(action_scale=0.0, delay_steps=2, ray_dropout=0.30),
            seed=_stable_seed("adverse-return", policy_seed, team_size, episode_seed),
            step_log=log_path,
            start_positions=np.asarray((placed.entry.mean[:3],)),
            start_velocities=np.asarray((placed.entry.mean[3:6],)),
            start_yaws=np.asarray((placed.entry.mean[6],)),
            enforce_interface_terminal_state=True,
        )
    finally:
        env.close()
    measured_exit = np.concatenate((
        np.asarray(result.final_positions[0], dtype=np.float64),
        np.asarray(result.final_velocities[0], dtype=np.float64),
        (float(result.final_yaws[0]),),
    ))
    yaw = task.placement_yaws[occurrence]
    c, s = math.cos(yaw), math.sin(yaw)
    inverse_rotation = np.asarray(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
    canonical_exit = np.concatenate((
        inverse_rotation @ (measured_exit[:3] - task.anchors[occurrence]),
        inverse_rotation @ measured_exit[3:6],
        ((measured_exit[6] - yaw + math.pi) % (2.0 * math.pi) - math.pi,),
    ))
    residual = canonical_exit - record.exit.mean
    residual[6] = (residual[6] + math.pi) % (2.0 * math.pi) - math.pi
    prediction_error = float(np.sqrt(
        np.sum(np.square(residual[:3] / 0.25))
        + np.sum(np.square(residual[3:6] / 0.35))
        + (residual[6] / math.radians(12.0)) ** 2
    ))
    event = ExecutionEvent.create(
        record_id=record.record_id,
        origin_robot=record.origin_robot,
        executor_robot=recipient.owner_robot,
        context=context,
        logical_time=logical_time,
        success=bool(result.success),
        duration=float(result.completion_time),
        min_clearance=float(result.minimum_clearance),
        prediction_error=prediction_error,
        tracking_error=float(result.mean_acceleration) / 5.0,
        event_type="attempt",
        reason="recipient_adverse_execution_return",
        immutable_hash=recipient.immutable_hash,
        parent_event_id=record.first_event_id,
        entry_state=record.entry.mean,
        command=record.command.terminal_velocity,
        exit_state=canonical_exit,
        namespace=f"adverse-return:{policy_seed}:{team_size}:{episode_seed}:{record.record_id}",
    )
    recipient_added = recipient.add_event(event)
    source_added = source.add_event(event)
    source_digest = source.digest()
    source.add_event(event)
    return {
        "policy_seed": policy_seed,
        "team_size": team_size,
        "episode_seed": episode_seed,
        "record_id": record.record_id,
        "origin": record.origin_robot,
        "recipient": recipient.owner_robot,
        "communication_bytes": byte_count,
        "event_id": event.event_id,
        "physical_success": bool(result.success),
        "prediction_error": prediction_error,
        "recipient_event_added": recipient_added,
        "source_event_added": source_added,
        "duplicate_return_idempotent": source.digest() == source_digest,
        "recipient_reliability_before": recipient_before,
        "recipient_reliability_after": recipient.reliability(record.record_id, context, logical_time),
        "source_reliability_before": source_before,
        "source_reliability_after": source.reliability(record.record_id, context, logical_time),
        "withheld_source_reliability_before": withheld_before,
        "withheld_source_reliability_after": withheld_control.reliability(
            record.record_id, context, logical_time
        ),
        "policy_weights_updated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/collective_cnm_v5"))
    parser.add_argument("--eef-run", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--policy-limit", type=int)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--team-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policies = POLICY_SEEDS[:1] if args.smoke else POLICY_SEEDS
    team_sizes = TEAM_SIZES[:1] if args.smoke else TEAM_SIZES
    episodes = EPISODE_SEEDS[:1] if args.smoke else EPISODE_SEEDS
    if args.policy_limit is not None:
        policies = policies[: args.policy_limit]
    if args.team_limit is not None:
        team_sizes = team_sizes[: args.team_limit]
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    tasks = _team_tasks()
    planner_config = PlannerConfig()
    compiler_config = CompilerConfig()
    trials: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    complementarity: list[dict[str, Any]] = []
    returned_events: list[dict[str, Any]] = []
    adverse_return_probes: list[dict[str, Any]] = []
    frozen_manifests: list[dict[str, Any]] = []
    started = time.perf_counter()
    for policy_seed in policies:
        model, payload = load_eef_checkpoint(
            _checkpoint(args.eef_run, policy_seed), device=args.device, allow_source_rebind=True
        )
        manifest = build_frozen_stack_manifest(
            policy_sha256=payload["immutable_sha256"],
            policy_config=model.config,
            planner_config=planner_config,
            compiler_config=compiler_config,
        )
        frozen_manifests.append({"policy_seed": policy_seed, **manifest.to_dict()})
        sources: list[CNMLibrary] = []
        for robot_index in range(max(team_sizes)):
            acquisition_seed = 5001 + 101 * robot_index
            library, _, _ = acquire_library(
                model,
                manifest.frozen_stack_sha256,
                policy_seed,
                acquisition_seed,
                args.output / "acquisition" / f"policy_{policy_seed}" / f"robot_{robot_index}",
                attempts=3,
                planner_config=planner_config,
                compiler_config=compiler_config,
            )
            sources.append(library)
        for task in tasks:
            for team_size in team_sizes:
                private, private_audit, withheld_source = _partition_histories(
                    sources[:team_size], team_size, task
                )
                for item in private_audit:
                    complementarity.append({
                        "policy_seed": policy_seed,
                        "team_size": team_size,
                        "task_id": task.task_id,
                        "interface_shift_class": task.interface_shift_class,
                        "union_support": True,
                        "withheld_source": withheld_source,
                        **item,
                    })
                for episode_seed in episodes:
                    for condition in CONDITIONS:
                        condition_private = (
                            _redundant_histories(sources, team_size, task)
                            if condition == "redundant_histories"
                            else private
                        )
                        histories, byte_count, transfer_audit = _prepare_condition(
                            condition_private, task, condition, withheld_source
                        )
                        for item in transfer_audit:
                            transfers.append({
                                "policy_seed": policy_seed,
                                "task_id": task.task_id,
                                "team_size": team_size,
                                "episode_seed": episode_seed,
                                **item,
                            })
                        result, robot_tasks = _execute_team(
                            model,
                            histories,
                            task,
                            condition,
                            team_size,
                            policy_seed,
                            episode_seed,
                            args.output,
                        )
                        return_audit = {
                            "returned_events": 0,
                            "duplicate_return_idempotent": True,
                            "mean_reliability_before": None,
                            "mean_reliability_after": None,
                        }
                        if condition == "selective_circulation" and result.get("physical_attempted"):
                            events, return_audit = _return_execution_evidence(
                                histories,
                                result,
                                robot_tasks,
                                logical_time=float(100 + episode_seed % 100),
                            )
                            returned_events.extend({
                                "policy_seed": policy_seed,
                                "task_id": task.task_id,
                                "team_size": team_size,
                                "episode_seed": episode_seed,
                                **event,
                            } for event in events)
                        trials.append({
                            "policy_seed": policy_seed,
                            "team_size": team_size,
                            "episode_seed": episode_seed,
                            "task_id": task.task_id,
                            "interface_shift_class": task.interface_shift_class,
                            "condition": condition,
                            "communication_bytes": byte_count,
                            "communication_kib": byte_count / 1024.0,
                            "frozen_stack_sha256": manifest.frozen_stack_sha256,
                            "policy_weights_updated": False,
                            **return_audit,
                            **result,
                        })
                    adverse_return_probes.append(_adverse_return_probe(
                        model,
                        private,
                        task,
                        withheld_source,
                        team_size,
                        policy_seed,
                        episode_seed,
                        args.output,
                    ))

    _write_csv(args.output / "source_data" / "collective_trials.csv", trials)
    _write_csv(args.output / "source_data" / "transfer_audit.csv", transfers)
    _write_csv(args.output / "source_data" / "complementarity_audit.csv", complementarity)
    _write_csv(args.output / "source_data" / "execution_return_events.csv", returned_events)
    _write_csv(args.output / "source_data" / "adverse_return_probes.csv", adverse_return_probes)
    summary = {
        "schema_version": 1,
        "status": "smoke_measured_simulation" if args.smoke else "measured_simulation",
        "policy_seeds": list(policies),
        "team_sizes": list(team_sizes),
        "episode_seeds": list(episodes),
        "task_ids": [task.task_id for task in tasks],
        "complementarity_verified": all(not bool(row["private_support"]) and bool(row["union_support"]) for row in complementarity),
        "conditions": {},
        "adverse_execution_return": {
            "probes": len(adverse_return_probes),
            "physical_failures": sum(not row["physical_success"] for row in adverse_return_probes),
            "all_events_source_linked": all(
                row["recipient_event_added"] and row["source_event_added"]
                for row in adverse_return_probes
            ),
            "all_failures_lower_source_reliability": all(
                row["source_reliability_after"] < row["source_reliability_before"]
                for row in adverse_return_probes if not row["physical_success"]
            ),
            "withholding_leaves_source_unchanged": all(
                row["withheld_source_reliability_after"]
                == row["withheld_source_reliability_before"]
                for row in adverse_return_probes
            ),
            "all_returns_idempotent": all(
                row["duplicate_return_idempotent"] for row in adverse_return_probes
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    pairing_audit = _feedback_pairing_audit(trials)
    if not pairing_audit["all_preintervention_fields_identical"]:
        raise RuntimeError("feedback-withholding control is not paired with selective circulation")
    summary["feedback_withholding_pairing"] = pairing_audit
    for condition in CONDITIONS:
        rows = [row for row in trials if row["condition"] == condition]
        transfer_subset = [row for row in transfers if row["condition"] == condition]
        total_bytes = sum(int(row["communication_bytes"]) for row in rows)
        admitted_records = sum(int(row.get("records", 0)) for row in transfer_subset)
        member_outcomes = [
            bool(value)
            for row in rows
            for value in row.get("per_drone_success", [])
        ]
        return_subset = returned_events if condition == "selective_circulation" else []
        summary["conditions"][condition] = {
            "episodes": len(rows),
            "whole_team_successes": sum(bool(row.get("success")) for row in rows),
            "whole_team_completion": float(np.mean([bool(row.get("success")) for row in rows])),
            "cnm_capability_successes": sum(
                bool(row.get("cnm_capability_success")) for row in rows
            ),
            "cnm_capability_completion": float(np.mean([
                bool(row.get("cnm_capability_success")) for row in rows
            ])),
            "member_completion": float(np.mean(member_outcomes)) if member_outcomes else 0.0,
            "mean_supported_fraction": float(np.mean([
                int(row.get("supported_members", 0)) / max(int(row["team_size"]), 1)
                for row in rows
            ])),
            "mean_communication_bytes": float(np.mean([row["communication_bytes"] for row in rows])),
            "admitted_records": admitted_records,
            "admitted_records_per_kib": (
                admitted_records / (total_bytes / 1024.0) if total_bytes else 0.0
            ),
            "returned_events": sum(int(row.get("returned_events", 0)) for row in rows),
            "mean_return_reliability_change": (
                float(np.mean([
                    float(row["reliability_after"]) - float(row["reliability_before"])
                    for row in return_subset
                ]))
                if return_subset else None
            ),
        }
    _write_json(args.output / "statistical_summary.json", summary)
    _write_json(args.output / "frozen_stack_manifests.json", frozen_manifests)
    _write_json(args.output / "run_manifest.json", {
        "schema_version": 1,
        "arguments": vars(args) | {"output": str(args.output), "eef_run": str(args.eef_run)},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "summary": summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
