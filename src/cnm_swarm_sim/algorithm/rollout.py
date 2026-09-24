"""Physical PyBullet rollouts driven by a frozen EEF policy and CNM paths."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..env import CNMSwarmEnv
from .eef import EEFNavigator
from .memory import InterfaceSummary, PlacedRecord


MODE_INDEX = {"progress": 0, "pass": 1, "yield": 2, "stop": 3}
ReplanDecision = tuple[list[PlacedRecord], dict[str, Any]]
ReplanCallback = Callable[
    [int, tuple[str, ...], np.ndarray, np.ndarray, float],
    ReplanDecision | list[PlacedRecord] | None,
]


@dataclass(frozen=True)
class RolloutPerturbation:
    action_scale: float = 1.0
    action_noise: float = 0.0
    delay_steps: int = 0
    ray_dropout: float = 0.0
    ray_noise: float = 0.0
    start_position_std: float = 0.0


@dataclass
class PhysicalRolloutResult:
    success: bool
    steps: int
    completion_time: float
    collision_events: int
    minimum_clearance: float
    minimum_separation: float
    mean_speed: float
    peak_speed: float
    p95_speed: float
    mean_acceleration: float
    mean_jerk: float
    reached_fraction: float
    selected_record_ids: list[list[str]]
    replan_count: int
    replan_failures: int
    replan_events: list[dict[str, Any]]
    final_positions: np.ndarray
    final_velocities: np.ndarray
    final_yaws: np.ndarray
    ordered_checkpoints_reached: list[int]
    ordered_checkpoint_totals: list[int]
    per_drone_success: list[bool]

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["final_positions"] = self.final_positions.tolist()
        payload["final_velocities"] = self.final_velocities.tolist()
        payload["final_yaws"] = self.final_yaws.tolist()
        for key, value in tuple(payload.items()):
            if isinstance(value, float) and not np.isfinite(value):
                payload[key] = None
        return payload


def flatten_composed_path(
    path: list[PlacedRecord],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    waypoints: list[np.ndarray] = []
    records: list[str] = []
    modes: list[int] = []
    target_velocities: list[np.ndarray] = []
    target_yaws: list[float] = []
    for placed in path:
        if not waypoints:
            waypoints.append(placed.entry.mean[:3].copy())
            records.append(placed.record.record_id)
            modes.append(MODE_INDEX.get(placed.record.command.mode, 0))
            target_velocities.append(placed.entry.mean[3:6].copy())
            target_yaws.append(float(placed.entry.mean[6]))
        for point_index, point in enumerate(placed.waypoints):
            if waypoints and np.linalg.norm(point - waypoints[-1]) < 1.0e-8:
                continue
            waypoints.append(point.copy())
            records.append(placed.record.record_id)
            modes.append(MODE_INDEX.get(placed.record.command.mode, 0))
            target_velocities.append(
                placed.exit.mean[3:6].copy()
                if point_index == len(placed.waypoints) - 1
                else np.full(3, math.nan, dtype=np.float64)
            )
            target_yaws.append(
                float(placed.exit.mean[6])
                if point_index == len(placed.waypoints) - 1
                else math.nan
            )
    if not waypoints:
        return (
            np.empty((0, 3)),
            [],
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.stack(waypoints),
        records,
        np.asarray(modes, dtype=np.int64),
        np.stack(target_velocities),
        np.asarray(target_yaws, dtype=np.float64),
    )


def _entry_interface_contains(
    position: np.ndarray,
    velocity: np.ndarray,
    yaw: float,
    interface: InterfaceSummary,
    *,
    covariance_floor: float = 0.025,
    threshold: float = 14.1,
) -> tuple[bool, float]:
    state = np.concatenate((position, velocity, (yaw,)))
    delta = state - interface.mean
    delta[6] = (delta[6] + math.pi) % (2.0 * math.pi) - math.pi
    support_ok = bool(np.all(np.abs(delta) <= interface.support_radius))
    covariance = interface.covariance + np.eye(7) * covariance_floor
    distance = float(delta @ np.linalg.solve(covariance, delta))
    return support_ok and distance <= threshold, distance


def run_physical_rollout(
    env: CNMSwarmEnv,
    navigator: EEFNavigator,
    paths: list[list[PlacedRecord]],
    *,
    speed: float = 1.2,
    max_steps: int = 420,
    waypoint_tolerance: float = 0.38,
    perturbation: RolloutPerturbation | None = None,
    seed: int = 0,
    step_log: str | Path | None = None,
    start_positions: np.ndarray | None = None,
    start_velocities: np.ndarray | None = None,
    start_yaws: np.ndarray | None = None,
    replan_callbacks: list[ReplanCallback | None] | None = None,
    enforce_interface_terminal_state: bool = False,
    maximum_bridge_seconds: float = 0.35,
    bridge_position_gain: float = 3.0,
    bridge_velocity_gain: float = 1.0,
    replan_query_tolerance: float = 0.16,
    ordered_checkpoints: list[np.ndarray] | None = None,
    checkpoint_tolerance: float = 0.60,
    lane_spacing: float = 0.36,
    stop_at_final_waypoint: bool = False,
    external_peer_safety: bool = False,
    peer_safety_activation: float = 0.60,
    interface_lookahead_braking: bool = False,
    interface_braking_radius: float = 4.0,
    interface_approach_speed: float = 2.0,
) -> PhysicalRolloutResult:
    """Execute one composed path per robot and retain all attempts in JSONL."""

    if len(paths) != env.config.num_drones:
        raise ValueError("one composed path is required for every drone")
    if replan_callbacks is None:
        replan_callbacks = [None] * env.config.num_drones
    if len(replan_callbacks) != env.config.num_drones:
        raise ValueError("one replan callback (or None) is required for every drone")
    if ordered_checkpoints is None:
        ordered_checkpoints = [np.empty((0, 3), dtype=np.float64) for _ in paths]
    if len(ordered_checkpoints) != env.config.num_drones:
        raise ValueError("one ordered checkpoint sequence is required for every drone")
    ordered_checkpoints = [np.asarray(points, dtype=np.float64) for points in ordered_checkpoints]
    if any(points.ndim != 2 or points.shape[1] != 3 for points in ordered_checkpoints):
        raise ValueError("ordered checkpoints must have shape N by 3")
    perturbation = perturbation or RolloutPerturbation()
    rng = np.random.default_rng(seed)
    flattened = [flatten_composed_path(path) for path in paths]
    if any(points.shape[0] == 0 for points, _, _, _, _ in flattened):
        raise ValueError("unsupported CNM path cannot be executed")
    # Preserve a small lane offset for members travelling in the same
    # direction.  This is a fixed live-traffic convention, not memory content.
    directions = np.asarray([path[0].direction for path in paths], dtype=np.int64)
    lane_offsets = np.zeros(len(paths), dtype=np.float64)
    for direction in (-1, 1):
        members = np.flatnonzero(directions == direction)
        if len(members) > 1:
            lane_offsets[members] = lane_spacing * (
                np.arange(len(members)) - 0.5 * (len(members) - 1)
            )
    flattened = [
        (
            points + np.array((0.0, lane_offsets[index], 0.0)),
            records,
            modes,
            target_velocities,
            target_yaws,
        )
        for index, (points, records, modes, target_velocities, target_yaws) in enumerate(flattened)
    ]
    starts = (
        np.stack([points[0] for points, _, _, _, _ in flattened])
        if start_positions is None
        else np.asarray(start_positions, dtype=np.float64).copy()
    )
    if starts.shape != (env.config.num_drones, 3):
        raise ValueError("start_positions must have shape (num_drones, 3)")
    entry_velocities = (
        np.zeros_like(starts)
        if start_velocities is None
        else np.asarray(start_velocities, dtype=np.float64).copy()
    )
    if entry_velocities.shape != starts.shape:
        raise ValueError("start_velocities must match start_positions")
    if perturbation.start_position_std:
        starts += rng.normal(0.0, perturbation.start_position_std, size=starts.shape)
        starts[:, 2] = np.maximum(starts[:, 2], env.config.drone_radius + 0.08)
    yaws = (
        np.asarray([0.0 if path[0].direction > 0 else np.pi for path in paths])
        if start_yaws is None
        else np.asarray(start_yaws, dtype=np.float64).copy()
    )
    if yaws.shape != (env.config.num_drones,):
        raise ValueError("start_yaws must have shape (num_drones,)")
    observation = env.set_drone_states(starts, velocities=entry_velocities, yaws=yaws)
    navigator.reset(env.config.num_drones)
    waypoint_index = np.ones(env.config.num_drones, dtype=np.int64)
    completed = np.zeros(env.config.num_drones, dtype=bool)
    collision_drones: set[int] = set()
    action_buffer = [
        np.zeros((env.config.num_drones, 4), dtype=np.float32)
        for _ in range(perturbation.delay_steps)
    ]
    velocities: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    selected_by_drone: list[list[str]] = [[] for _ in paths]
    replan_failed = np.zeros(env.config.num_drones, dtype=bool)
    bridge_attempt_started = np.full(env.config.num_drones, -1, dtype=np.int64)
    pending_entry_interfaces: list[InterfaceSummary | None] = [
        None for _ in range(env.config.num_drones)
    ]
    checkpoint_indices = np.zeros(env.config.num_drones, dtype=np.int64)
    replan_events: list[dict[str, Any]] = []
    min_clearance = float("inf")
    min_separation = float("inf")
    previous_velocity = env.vel.copy()
    previous_acceleration = np.zeros_like(previous_velocity)
    jerks: list[np.ndarray] = []
    handle = None
    if step_log is not None:
        target = Path(step_log)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("w", encoding="utf-8")
        # The initial sample is written before the first control action so an
        # experience compiler can estimate the entry distribution from actual
        # executed starts instead of reconstructing it from a route template.
        handle.write(
            json.dumps(
                {
                    "step": -1,
                    "time": 0.0,
                    "positions": env.pos.tolist(),
                    "velocities": env.vel.tolist(),
                    "yaws": env.rpy[:, 2].tolist(),
                    "actions": np.zeros((env.config.num_drones, 4), dtype=np.float32).tolist(),
                    "waypoint_index": waypoint_index.tolist(),
                    "active_record_ids": [records[0] for _, records, _, _, _ in flattened],
                    "new_events": [],
                    "immutable_policy": True,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    try:
        bridge_step_budget = int(math.ceil(
            maximum_bridge_seconds / env.config.control_timestep
        ))
        for step in range(max_steps):
            targets = np.empty_like(env.pos)
            modes = np.zeros(env.config.num_drones, dtype=np.int64)
            desired_speed_by_drone = np.full(env.config.num_drones, speed, dtype=np.float64)
            desired_terminal_velocities = np.full(
                (env.config.num_drones, 3), math.nan, dtype=np.float64
            )
            desired_terminal_yaws = np.full(env.config.num_drones, math.nan, dtype=np.float64)
            active_records: list[str] = []
            for drone_index, (
                points,
                record_ids,
                point_modes,
                point_velocities,
                point_yaws,
            ) in enumerate(flattened):
                if bridge_attempt_started[drone_index] >= 0:
                    bridge_elapsed_steps = step - bridge_attempt_started[drone_index]
                    if bridge_elapsed_steps > bridge_step_budget:
                        replan_failed[drone_index] = True
                index = min(waypoint_index[drone_index], len(points) - 1)
                boundary_entry = (
                    index < len(points) - 1
                    and record_ids[index + 1] != record_ids[index]
                    and replan_callbacks[drone_index] is not None
                    and not replan_failed[drone_index]
                )
                # This radius only decides when to query.  The planner still
                # applies the full seven-dimensional finite-support and
                # Mahalanobis entry gates to the measured state.
                target_tolerance = (
                    min(waypoint_tolerance, replan_query_tolerance)
                    if boundary_entry else waypoint_tolerance
                )
                pending_entry = pending_entry_interfaces[drone_index]
                if pending_entry is not None and index == 0:
                    target_reached, entry_distance = _entry_interface_contains(
                        env.pos[drone_index],
                        env.vel[drone_index],
                        float(env.rpy[drone_index, 2]),
                        pending_entry,
                    )
                    if target_reached:
                        pending_entry_interfaces[drone_index] = None
                        bridge_attempt_started[drone_index] = -1
                        replan_events.append(
                            {
                                "event_type": "bridge_entry_admitted",
                                "step": step,
                                "drone": drone_index,
                                "entry_mahalanobis_distance": entry_distance,
                            }
                        )
                else:
                    target_reached = (
                        np.linalg.norm(env.pos[drone_index] - points[index])
                        <= target_tolerance
                    )
                if target_reached:
                    if index == len(points) - 1:
                        completed[drone_index] = True
                    else:
                        previous_record = record_ids[index]
                        proposed_index = index + 1
                        next_record = record_ids[proposed_index]
                        callback = replan_callbacks[drone_index]
                        if callback is not None and next_record != previous_record:
                            decision = callback(
                                drone_index,
                                tuple(selected_by_drone[drone_index]),
                                env.pos[drone_index].copy(),
                                env.vel[drone_index].copy(),
                                float(env.rpy[drone_index, 2]),
                            )
                            if isinstance(decision, tuple):
                                suffix, replan_diagnostics = decision
                            else:
                                suffix, replan_diagnostics = decision, {}
                            event = {
                                "event_type": "replan_query",
                                "step": step,
                                "drone": drone_index,
                                "completed_record_id": previous_record,
                                "measured_exit": np.concatenate((
                                    env.pos[drone_index],
                                    env.vel[drone_index],
                                    (float(env.rpy[drone_index, 2]),),
                                )).tolist(),
                                "supported": bool(suffix),
                                "replacement_record_ids": [] if not suffix else [
                                    placed.record.record_id for placed in suffix
                                ],
                                "planner_query_id": replan_diagnostics.get("query_id"),
                                "planner_reason": replan_diagnostics.get("reason"),
                                "search_algorithm": replan_diagnostics.get("search_algorithm"),
                                "graph_nodes_expanded": replan_diagnostics.get("expanded"),
                                "planning_time_ms": replan_diagnostics.get("planning_time_ms"),
                                "selected_chain_reliability": replan_diagnostics.get(
                                    "selected_chain_reliability"
                                ),
                            }
                            replan_events.append(event)
                            if not suffix:
                                if bridge_attempt_started[drone_index] < 0:
                                    bridge_attempt_started[drone_index] = step
                                elapsed_steps = step - bridge_attempt_started[drone_index]
                                if elapsed_steps > bridge_step_budget:
                                    replan_failed[drone_index] = True
                            else:
                                # Enter directly when the measured state is
                                # already supported by the next record.  A
                                # physical bridge is used only for a reachable
                                # non-overlapping interface; this keeps direct
                                # compositions from paying an artificial
                                # bridge penalty.
                                direct_entry, entry_distance = _entry_interface_contains(
                                    env.pos[drone_index],
                                    env.vel[drone_index],
                                    float(env.rpy[drone_index, 2]),
                                    suffix[0].entry,
                                )
                                if direct_entry:
                                    bridge_attempt_started[drone_index] = -1
                                    pending_entry_interfaces[drone_index] = None
                                    replan_events.append(
                                        {
                                            "event_type": "direct_entry_admitted",
                                            "step": step,
                                            "drone": drone_index,
                                            "entry_mahalanobis_distance": entry_distance,
                                        }
                                    )
                                else:
                                    # Failed query time is never charged to
                                    # the bounded observed-space bridge.
                                    bridge_attempt_started[drone_index] = step
                                    pending_entry_interfaces[drone_index] = suffix[0].entry
                                flattened[drone_index] = flatten_composed_path(suffix)
                                (
                                    points,
                                    record_ids,
                                    point_modes,
                                    point_velocities,
                                    point_yaws,
                                ) = flattened[drone_index]
                                waypoint_index[drone_index] = 0
                                index = 0
                        else:
                            waypoint_index[drone_index] = proposed_index
                            index = proposed_index
                targets[drone_index] = points[index]
                modes[drone_index] = point_modes[index]
                if interface_lookahead_braking:
                    next_interface = next(
                        (
                            candidate
                            for candidate in range(index, len(points))
                            if bool(np.all(np.isfinite(point_velocities[candidate])))
                        ),
                        None,
                    )
                    if next_interface is not None:
                        interface_distance = float(
                            np.linalg.norm(env.pos[drone_index] - points[next_interface])
                        )
                        if interface_distance < interface_braking_radius:
                            phase = max(0.0, interface_distance / interface_braking_radius)
                            approach_cap = interface_approach_speed + (
                                speed - interface_approach_speed
                            ) * phase**2
                            desired_speed_by_drone[drone_index] = min(
                                desired_speed_by_drone[drone_index], approach_cap
                            )
                terminal_zone = (
                    np.linalg.norm(env.pos[drone_index] - points[index])
                    <= navigator.model.config.interface_terminal_activation_radius
                )
                if (
                    enforce_interface_terminal_state
                    and terminal_zone
                    and bool(np.all(np.isfinite(point_velocities[index])))
                ):
                    # A record preserves its demonstrated exit velocity when
                    # another response must follow.  At a registered task goal
                    # there is no downstream interface, so formal task probes
                    # may request a closed-loop stop to prevent a completed
                    # robot from drifting through neighbouring goal lanes.
                    desired_velocity = (
                        np.zeros(3, dtype=np.float64)
                        if stop_at_final_waypoint and index == len(points) - 1
                        else point_velocities[index].copy()
                    )
                    if pending_entry is not None:
                        # The connector is a short observed-space closed-loop
                        # bridge.  Track both its registered entry state and
                        # entry position; once admitted, the stored response
                        # takes over under its own terminal interface target.
                        desired_velocity += bridge_position_gain * (
                            points[index] - env.pos[drone_index]
                        )
                        desired_velocity += bridge_velocity_gain * (
                            point_velocities[index] - env.vel[drone_index]
                        )
                        velocity_norm = float(np.linalg.norm(desired_velocity))
                        if velocity_norm > navigator.model.config.max_speed:
                            desired_velocity *= navigator.model.config.max_speed / velocity_norm
                    desired_terminal_velocities[drone_index] = desired_velocity
                    desired_terminal_yaws[drone_index] = point_yaws[index]
                    desired_speed_by_drone[drone_index] = min(
                        navigator.model.config.max_speed,
                        max(0.0, float(np.linalg.norm(desired_velocity))),
                    )
                pending_entry = pending_entry_interfaces[drone_index]
                active_records.append(
                    f"bridge_to:{record_ids[index]}" if pending_entry is not None else record_ids[index]
                )
                if (
                    pending_entry is None
                    and (
                        not selected_by_drone[drone_index]
                        or selected_by_drone[drone_index][-1] != record_ids[index]
                    )
                ):
                    selected_by_drone[drone_index].append(record_ids[index])
            policy_observation = observation
            if perturbation.ray_dropout or perturbation.ray_noise:
                policy_observation = {key: value.copy() for key, value in observation.items()}
                rays = policy_observation["rays"]
                if perturbation.ray_dropout:
                    missing = rng.random(rays.shape) < perturbation.ray_dropout
                    rays[missing] = 1.0
                if perturbation.ray_noise:
                    rays += rng.normal(0.0, perturbation.ray_noise, size=rays.shape)
                    np.clip(rays, 0.0, 1.0, out=rays)
            action = navigator.act(
                policy_observation,
                env.pos.copy(),
                targets,
                modes=modes,
                desired_speeds=np.where(
                    completed | replan_failed, 0.0, desired_speed_by_drone
                ),
                desired_terminal_velocities=desired_terminal_velocities,
                desired_terminal_yaws=desired_terminal_yaws,
            )
            action[completed] = 0.0
            action[replan_failed] = 0.0
            if external_peer_safety and env.config.num_drones > 1:
                # Symmetric velocity projection shared by every memory
                # condition.  It uses only the current relative state and
                # cannot add, retrieve or modify CNM experience.
                for left in range(env.config.num_drones):
                    for right in range(left + 1, env.config.num_drones):
                        delta = env.pos[left] - env.pos[right]
                        distance = float(np.linalg.norm(delta))
                        if distance >= peer_safety_activation:
                            continue
                        if distance < 1.0e-8:
                            direction = np.asarray((0.0, 1.0, 0.0))
                        else:
                            direction = delta / distance
                        correction = 1.8 * (peer_safety_activation - distance) * direction
                        action[left, :3] += correction
                        action[right, :3] -= correction
                norms = np.linalg.norm(action[:, :3], axis=1, keepdims=True)
                action[:, :3] *= np.minimum(
                    1.0, navigator.model.config.max_speed / np.maximum(norms, 1.0e-8)
                )
                action[completed] = 0.0
                action[replan_failed] = 0.0
            action[:, :3] *= perturbation.action_scale
            if perturbation.action_noise:
                action[:, :3] += rng.normal(0.0, perturbation.action_noise, size=action[:, :3].shape)
            action_buffer.append(action.astype(np.float32))
            applied = action_buffer.pop(0)
            observation, _, terminated, truncated, info = env.step(applied)
            for drone_index, checkpoints in enumerate(ordered_checkpoints):
                while (
                    checkpoint_indices[drone_index] < len(checkpoints)
                    and np.linalg.norm(
                        env.pos[drone_index] - checkpoints[checkpoint_indices[drone_index]]
                    ) <= checkpoint_tolerance
                ):
                    checkpoint_indices[drone_index] += 1
            for event in info["new_events"]:
                if event.get("drone_a") is not None:
                    collision_drones.add(int(event["drone_a"]))
                if event.get("drone_b") is not None:
                    collision_drones.add(int(event["drone_b"]))
            if observation["rays"].size:
                min_clearance = min(min_clearance, float(observation["rays"].min() * env.config.ray_range))
            min_separation = min(min_separation, float(info["minimum_separation"]))
            velocity = env.vel.copy()
            acceleration = (velocity - previous_velocity) / env.config.control_timestep
            jerk = (acceleration - previous_acceleration) / env.config.control_timestep
            velocities.append(velocity)
            accelerations.append(acceleration)
            jerks.append(jerk)
            previous_velocity = velocity
            previous_acceleration = acceleration
            if handle is not None:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "time": info["simulation_time"],
                            "positions": env.pos.tolist(),
                            "velocities": env.vel.tolist(),
                            "yaws": env.rpy[:, 2].tolist(),
                            "actions": applied.tolist(),
                            "targets": targets.tolist(),
                            "waypoint_index": waypoint_index.tolist(),
                            "active_record_ids": active_records,
                            "new_events": info["new_events"],
                            "immutable_policy": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if bool(np.all(completed)) or bool(np.any(replan_failed)) or terminated or truncated:
                break
    finally:
        if handle is not None:
            handle.close()
    velocity_array = np.stack(velocities) if velocities else np.zeros((1, env.config.num_drones, 3))
    acceleration_array = np.stack(accelerations) if accelerations else np.zeros_like(velocity_array)
    jerk_array = np.stack(jerks) if jerks else np.zeros_like(velocity_array)
    steps = len(velocities)
    checkpoints_complete = np.asarray(
        [checkpoint_indices[index] == len(points) for index, points in enumerate(ordered_checkpoints)],
        dtype=bool,
    )
    success = bool(
        np.all(completed)
        and np.all(checkpoints_complete)
        and not collision_drones
        and not np.any(replan_failed)
    )
    per_drone_success = (
        completed
        & checkpoints_complete
        & ~replan_failed
        & np.asarray([index not in collision_drones for index in range(len(paths))])
    )
    return PhysicalRolloutResult(
        success=success,
        steps=steps,
        completion_time=steps * env.config.control_timestep,
        collision_events=len(env.events),
        minimum_clearance=min_clearance,
        minimum_separation=min_separation,
        mean_speed=float(np.linalg.norm(velocity_array, axis=-1).mean()),
        peak_speed=float(np.linalg.norm(velocity_array, axis=-1).max()),
        p95_speed=float(np.quantile(np.linalg.norm(velocity_array, axis=-1), 0.95)),
        mean_acceleration=float(np.linalg.norm(acceleration_array, axis=-1).mean()),
        mean_jerk=float(np.linalg.norm(jerk_array, axis=-1).mean()),
        reached_fraction=float(np.mean(completed & np.asarray([i not in collision_drones for i in range(len(paths))]))),
        selected_record_ids=selected_by_drone,
        replan_count=len(replan_events),
        replan_failures=int(np.sum(replan_failed)),
        replan_events=replan_events,
        final_positions=env.pos.copy(),
        final_velocities=env.vel.copy(),
        final_yaws=env.rpy[:, 2].copy(),
        ordered_checkpoints_reached=checkpoint_indices.astype(int).tolist(),
        ordered_checkpoint_totals=[len(points) for points in ordered_checkpoints],
        per_drone_success=per_drone_success.astype(bool).tolist(),
    )
