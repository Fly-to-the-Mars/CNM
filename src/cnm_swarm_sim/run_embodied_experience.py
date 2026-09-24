"""Run the registered simulation study for manuscript Fig. 02.

The study reuses the protected EEF-v5 checkpoints read-only.  PyBullet trials
measure closed-loop navigation across four scene families and concurrent flight
without CNM circulation.  A matched-entry surrogate experiment measures the
terminal properties required by downstream CNM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import CNMSwarmEnv, EnvConfig
from .algorithm.eef import (
    EEFConfig,
    EEFNavigator,
    EEFPolicy,
    SyntheticBatch,
    _features,
    _sample_batch,
    differentiable_rollout,
    load_eef_checkpoint,
)
from .algorithm.eef_benchmark import (
    BenchmarkCondition,
    _repeat_batch,
    evaluate_policy_trials,
    wilson_interval,
)


POLICY_SEEDS = (17, 114, 211, 308, 405)
SCENES = ("eef_garage", "eef_forest", "eef_indoor_static", "eef_indoor_dynamic")
MULTI_SCENES = ("eef_indoor_static", "eef_forest")
SPEEDS = (0.8, 1.1, 1.4, 1.7, 2.0)
TEAM_SIZES = (2, 4, 6)
SCENE_LABELS = {
    "eef_garage": "Garage",
    "eef_forest": "Forest",
    "eef_indoor_static": "Indoor static",
    "eef_indoor_dynamic": "Indoor dynamic",
}
CHI2_90_DF7 = 12.017036623780532
POSITION_GATE_M = 0.25
VELOCITY_GATE_MPS = 0.35
YAW_GATE_RAD = math.radians(12.0)
REPEATABILITY_RATE = 0.80
SUPPORTED_SPEED_RATE = 0.80
PERTURBATION_CONDITIONS = (
    BenchmarkCondition("nominal", "Nominal"),
    BenchmarkCondition("image_degradation", "Image degradation", observation_noise=0.040),
    BenchmarkCondition("range_dropout_10", "10% range dropout", sensor_dropout=0.10),
    BenchmarkCondition("delay_133ms", "133 ms delay", delay_extra_steps=2),
    BenchmarkCondition("payload_10", "+10% payload", response_scale=0.82, drag_multiplier=1.15),
    BenchmarkCondition("lateral_wind", "Lateral wind", wind_multiplier=1.80),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _circular_abs(values: np.ndarray) -> np.ndarray:
    return np.abs((values + np.pi) % (2.0 * np.pi) - np.pi)


def _load_models(eef_run: Path, seeds: tuple[int, ...]) -> tuple[dict[int, EEFPolicy], list[dict[str, Any]]]:
    models: dict[int, EEFPolicy] = {}
    audit: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = eef_run / "training" / "eef_full" / f"seed_{seed}" / "eef_policy.pt"
        model, payload = load_eef_checkpoint(checkpoint, allow_source_rebind=True)
        models[seed] = model.eval()
        audit.append(
            {
                "method": "eef_full",
                "policy_seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_file_sha256": _sha256(checkpoint),
                "source_checkpoint_immutable_sha256": payload.get(
                    "source_checkpoint_immutable_sha256", payload["immutable_sha256"]
                ),
                "runtime_stack_sha256": payload["immutable_sha256"],
                "weights_updated_during_evaluation": False,
            }
        )
    return models, audit


def _advance_waypoints(
    positions: np.ndarray,
    sequences: tuple[np.ndarray, ...],
    indices: np.ndarray,
    complete: np.ndarray,
    tolerance: float,
) -> None:
    for drone, points in enumerate(sequences):
        while indices[drone] < len(points):
            if np.linalg.norm(positions[drone] - points[indices[drone]]) > tolerance:
                break
            indices[drone] += 1
        complete[drone] = indices[drone] >= len(points)


def _run_episode(
    env: CNMSwarmEnv,
    model: EEFPolicy,
    *,
    speed: float,
    seed: int,
    capture_path: Path | None = None,
    capture_preset: str = "eef_scene",
    retain_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, _ = env.reset(seed=seed)
    assert env.scenario_spec is not None
    sequences = env.scenario_spec.waypoint_sequences
    if len(sequences) != env.config.num_drones:
        raise RuntimeError("EEF scene must register one waypoint sequence per drone")
    rng = np.random.default_rng(seed + 9_973)
    starts = env.scenario_spec.starts.copy()
    starts += rng.normal(0.0, (0.018, 0.018, 0.010), size=starts.shape)
    observation = env.set_drone_states(starts, yaws=env.scenario_spec.initial_yaws)
    navigator = EEFNavigator(model)
    navigator.reset(env.config.num_drones)
    indices = np.ones(env.config.num_drones, dtype=np.int64)
    complete = np.zeros(env.config.num_drones, dtype=bool)
    collision_drones: set[int] = set()
    min_clearance = float("inf")
    min_separation = float("inf")
    traveled = np.zeros(env.config.num_drones, dtype=np.float64)
    speeds: list[np.ndarray] = []
    previous = env.pos.copy()
    trace: list[dict[str, Any]] = []
    captured = False
    last_step = 0
    for step in range(env.config.horizon_steps):
        targets = np.stack(
            [
                points[min(indices[drone], len(points) - 1)]
                for drone, points in enumerate(sequences)
            ]
        )
        actions = navigator.act(
            observation,
            env.pos,
            targets,
            desired_speeds=np.full(env.config.num_drones, speed),
        )
        actions[:, :3] += rng.normal(0.0, 0.012, size=(env.config.num_drones, 3))
        actions[complete] = 0.0
        observation, _, terminated, truncated, info = env.step(actions)
        traveled += np.linalg.norm(env.pos - previous, axis=1)
        previous = env.pos.copy()
        speeds.append(np.linalg.norm(env.vel, axis=1))
        ray_clearance = float(np.min(observation["rays"]) * env.config.ray_range)
        min_clearance = min(min_clearance, ray_clearance)
        min_separation = min(min_separation, float(info["minimum_separation"]))
        for event in info["new_events"]:
            if event["drone_a"] is not None:
                collision_drones.add(int(event["drone_a"]))
            if event["drone_b"] is not None:
                collision_drones.add(int(event["drone_b"]))
        _advance_waypoints(env.pos, sequences, indices, complete, tolerance=0.42)
        progress = float(np.mean([indices[i] / len(points) for i, points in enumerate(sequences)]))
        if retain_trace and (step % 3 == 0 or np.all(complete)):
            trace.append(
                {
                    "step": step,
                    "time_s": step * env.config.control_timestep,
                    "positions": json.dumps(env.pos.tolist(), separators=(",", ":")),
                    "velocities": json.dumps(env.vel.tolist(), separators=(",", ":")),
                    "waypoint_indices": json.dumps(indices.tolist(), separators=(",", ":")),
                }
            )
        if capture_path is not None and not captured and progress >= 0.52:
            env.render_frame(capture_path, width=1120, height=680, preset=capture_preset)
            captured = True
        last_step = step
        if np.all(complete) or collision_drones or terminated or truncated:
            break
    if capture_path is not None and not captured:
        env.render_frame(capture_path, width=1120, height=680, preset=capture_preset)
    speed_array = np.stack(speeds) if speeds else np.zeros((1, env.config.num_drones))
    member_completion = complete & np.asarray(
        [index not in collision_drones for index in range(env.config.num_drones)]
    )
    duration = (last_step + 1) * env.config.control_timestep
    result = {
        "success": bool(np.all(member_completion)),
        "member_completion_rate": float(np.mean(member_completion)),
        "completed_members": int(np.sum(member_completion)),
        "collision_members": len(collision_drones),
        "collision_events": len(env.events),
        "timeout": bool(not np.all(complete) and not collision_drones),
        "duration_s": duration,
        "mean_achieved_speed_mps": float(np.mean(speed_array)),
        "peak_achieved_speed_mps": float(np.max(speed_array)),
        "mean_path_length_m": float(np.mean(traveled)),
        "minimum_clearance_m": min_clearance,
        "minimum_inter_robot_separation_m": (
            min_separation if np.isfinite(min_separation) else None
        ),
        "cnm_circulation_enabled": False,
    }
    return result, trace


def _pybullet_trials(
    output: Path,
    models: dict[int, EEFPolicy],
    *,
    trials_per_cell: int,
    multi_repeats: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    single_rows: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    speeds = SPEEDS[:2] if smoke else SPEEDS
    scenes = SCENES[:2] if smoke else SCENES
    seeds = tuple(models)[:1] if smoke else tuple(models)
    trial_count = 1 if smoke else trials_per_cell
    artifacts = output / "artifacts"
    for scene in scenes:
        env = CNMSwarmEnv(
            EnvConfig(
                num_drones=1,
                seed=0,
                scenario=scene,
                episode_seconds=28.0,
                arena_length=14.0,
                arena_width=8.0,
                arena_height=4.0,
                max_speed=2.0,
                neighbor_k=0,
                scene_decorations=False,
            )
        )
        try:
            for policy_seed in seeds:
                for speed in speeds:
                    for repeat in range(trial_count):
                        episode_seed = 21_001 + 101 * repeat + 1_009 * SCENES.index(scene)
                        capture = None
                        retain_trace = False
                        if policy_seed == seeds[0] and abs(speed - 1.4) < 1e-8 and repeat == 0:
                            capture = artifacts / f"single_{scene}.png"
                            retain_trace = True
                        result, trace = _run_episode(
                            env,
                            models[policy_seed],
                            speed=speed,
                            seed=episode_seed,
                            capture_path=capture,
                            retain_trace=retain_trace,
                        )
                        single_rows.append(
                            {
                                "scene": scene,
                                "scene_label": SCENE_LABELS[scene],
                                "policy_seed": policy_seed,
                                "episode_seed": episode_seed,
                                "repeat": repeat,
                                "commanded_speed_mps": speed,
                                "frozen_policy": True,
                                **result,
                            }
                        )
                        for row in trace:
                            row.update({"experiment": "single", "scene": scene, "team_size": 1})
                        if trace:
                            _write_csv(output / "source_data" / f"trace_single_{scene}.csv", trace)
        finally:
            env.close()
        print(json.dumps({"stage": "single_scene", "scene": scene}), flush=True)

    multi_scenes = MULTI_SCENES[:1] if smoke else MULTI_SCENES
    team_sizes = TEAM_SIZES[:1] if smoke else TEAM_SIZES
    repeats = 1 if smoke else multi_repeats
    for scene in multi_scenes:
        for team_size in team_sizes:
            representative_saved = False
            fallback_trace: list[dict[str, Any]] = []
            env = CNMSwarmEnv(
                EnvConfig(
                    num_drones=team_size,
                    seed=0,
                    scenario=scene,
                    episode_seconds=25.0,
                    arena_length=14.0,
                    arena_width=8.0,
                    arena_height=4.0,
                    max_speed=2.0,
                    neighbor_k=min(6, team_size - 1),
                    communication_radius=3.0,
                    scene_decorations=False,
                )
            )
            try:
                for policy_seed in seeds:
                    for repeat in range(repeats):
                        episode_seed = 42_001 + 101 * repeat + 1_009 * MULTI_SCENES.index(scene)
                        capture = None
                        retain_trace = False
                        if policy_seed == seeds[0] and not representative_saved:
                            capture = artifacts / f"multi_{scene}_{team_size}.png"
                            retain_trace = True
                        result, trace = _run_episode(
                            env,
                            models[policy_seed],
                            speed=1.2,
                            seed=episode_seed,
                            capture_path=capture,
                            capture_preset="eef_traffic",
                            retain_trace=retain_trace,
                        )
                        multi_rows.append(
                            {
                                "scene": scene,
                                "scene_label": SCENE_LABELS[scene],
                                "team_size": team_size,
                                "policy_seed": policy_seed,
                                "episode_seed": episode_seed,
                                "repeat": repeat,
                                "commanded_speed_mps": 1.2,
                                "frozen_policy": True,
                                **result,
                            }
                        )
                        for row in trace:
                            row.update(
                                {
                                    "experiment": "multi",
                                    "scene": scene,
                                    "team_size": team_size,
                                    "policy_seed": policy_seed,
                                    "episode_seed": episode_seed,
                                    "repeat": repeat,
                                }
                            )
                        if trace:
                            fallback_trace = trace
                            if result["success"] and not representative_saved:
                                _write_csv(
                                    output / "source_data" / f"trace_multi_{scene}_{team_size}.csv",
                                    trace,
                                )
                                representative_saved = True
            finally:
                env.close()
            if not representative_saved and fallback_trace:
                _write_csv(
                    output / "source_data" / f"trace_multi_{scene}_{team_size}.csv",
                    fallback_trace,
                )
            print(
                json.dumps({"stage": "multi_scene", "scene": scene, "team_size": team_size}),
                flush=True,
            )
    return single_rows, multi_rows


def _matched_entry_for_model(
    model: EEFPolicy,
    *,
    policy_seed: int,
    method: str,
    calibration_commands: int,
    test_commands: int,
    repeats: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = EEFConfig()
    torch_device = torch.device(device)
    command_count = calibration_commands + test_commands
    generator = torch.Generator(device=torch_device).manual_seed(71_003)
    base = _sample_batch(config, command_count, generator, torch_device)
    base.target_speed[:] = 1.6
    direction = F.normalize(base.target - base.position, dim=-1)
    initial_distance = (base.target - base.position).norm(dim=-1)
    desired_travel = torch.minimum(
        initial_distance * 0.80,
        base.target_speed[:, 0] * config.response_horizon * config.dt * 0.72,
    )
    desired_heading = torch.atan2(direction[:, 1], direction[:, 0])
    desired_exit_base = torch.cat(
        (direction * desired_travel[:, None], direction * base.target_speed, desired_heading[:, None]),
        dim=-1,
    )
    desired_exit = desired_exit_base.repeat_interleave(repeats, dim=0)
    batch: SyntheticBatch = _repeat_batch(base, repeats)
    n = command_count * repeats
    execution_generator = torch.Generator(device=torch_device).manual_seed(81_007)
    rand = lambda *shape: torch.rand(shape, generator=execution_generator, device=torch_device)
    batch.wind = (rand(n, 3) - 0.5) * torch.tensor((0.90, 0.90, 0.30), device=torch_device)
    batch.response_scale = 0.76 + 0.48 * rand(n, 1)
    batch.delay_steps = torch.randint(
        0, config.execution_delay_max_steps + 2, (n,),
        generator=execution_generator, device=torch_device,
    )
    batch.drag_scale = 0.65 + 0.90 * rand(n, 1)
    features = _features(batch, config)
    with torch.no_grad():
        response, predicted, scale_tril, _ = model.to(torch_device).eval()(features)
        executed = differentiable_rollout(
            batch.position,
            batch.velocity,
            response,
            batch.obstacles,
            batch.radii,
            config,
            wind=batch.wind,
            response_scale=batch.response_scale,
            delay_steps=batch.delay_steps,
            drag_scale=batch.drag_scale,
            decay_gradients=False,
        )
        residual = executed.final_state - predicted
        residual[:, 6] = torch.atan2(torch.sin(residual[:, 6]), torch.cos(residual[:, 6]))
        whitened = torch.linalg.solve_triangular(scale_tril, residual.unsqueeze(-1), upper=False).squeeze(-1)
        mahalanobis = whitened.square().sum(dim=-1)
    final = executed.final_state.cpu().numpy()
    desired = desired_exit.cpu().numpy()
    clearance = executed.clearance.amin(dim=1).cpu().numpy()
    mahal = mahalanobis.cpu().numpy()
    calibration_mask = np.repeat(np.arange(command_count) < calibration_commands, repeats)
    calibration_scale = float(
        math.sqrt(max(np.quantile(mahal[calibration_mask], 0.90), 1.0e-12) / CHI2_90_DF7)
    )
    coverage = mahal <= CHI2_90_DF7 * calibration_scale**2
    position_error = np.linalg.norm(final[:, :3] - desired[:, :3], axis=1)
    velocity_error = np.linalg.norm(final[:, 3:6] - desired[:, 3:6], axis=1)
    yaw_error = _circular_abs(final[:, 6] - desired[:, 6])
    connector = (
        (clearance > 0.0)
        & (position_error <= POSITION_GATE_M)
        & (velocity_error <= VELOCITY_GATE_MPS)
        & (yaw_error <= YAW_GATE_RAD)
    )
    record_admissible = connector & (clearance >= 0.25)
    trial_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    local_positions = (
        executed.positions - batch.position[:, None, :]
    ).detach().cpu().numpy()
    for command in range(calibration_commands, command_count):
        selection = np.flatnonzero(
            np.repeat(np.arange(command_count) == command, repeats)
        )
        xyz = final[selection, :3]
        vel = final[selection, 3:6]
        yaw = final[selection, 6]
        xyz_center = xyz.mean(axis=0)
        vel_center = vel.mean(axis=0)
        yaw_center = math.atan2(float(np.sin(yaw).mean()), float(np.cos(yaw).mean()))
        pos_spread = np.linalg.norm(xyz - xyz_center, axis=1)
        vel_spread = np.linalg.norm(vel - vel_center, axis=1)
        yaw_spread = _circular_abs(yaw - yaw_center)
        class_repeatable = bool(
            np.mean(clearance[selection] > 0.0) >= REPEATABILITY_RATE
            and np.quantile(pos_spread, 0.90) <= POSITION_GATE_M
            and np.quantile(vel_spread, 0.90) <= VELOCITY_GATE_MPS
            and np.quantile(yaw_spread, 0.90) <= YAW_GATE_RAD
        )
        reusable_class = bool(
            class_repeatable
            and np.mean(connector[selection]) >= REPEATABILITY_RATE
            and np.mean(record_admissible[selection]) >= REPEATABILITY_RATE
        )
        class_rows.append(
            {
                "method": method,
                "policy_seed": policy_seed,
                "command_id": command - calibration_commands,
                "repeats": repeats,
                "repeatable_class": class_repeatable,
                "cnm_reusable_class": reusable_class,
                "position_spread_p90_m": float(np.quantile(pos_spread, 0.90)),
                "velocity_spread_p90_mps": float(np.quantile(vel_spread, 0.90)),
                "yaw_spread_p90_deg": float(np.degrees(np.quantile(yaw_spread, 0.90))),
                "downstream_connector_yield": float(np.mean(connector[selection])),
                "record_admission_yield": float(np.mean(record_admissible[selection])),
                "coverage90": float(np.mean(coverage[selection])) if model.config.use_outcome_supervision else None,
                "calibration_scale": calibration_scale if model.config.use_outcome_supervision else None,
            }
        )
        for local_repeat, index in enumerate(selection):
            trial_rows.append(
                {
                    "method": method,
                    "policy_seed": policy_seed,
                    "command_id": command - calibration_commands,
                    "repeat": local_repeat,
                    "exit_dx_m": float(final[index, 0]),
                    "exit_dy_m": float(final[index, 1]),
                    "exit_dz_m": float(final[index, 2]),
                    "exit_vx_mps": float(final[index, 3]),
                    "exit_vy_mps": float(final[index, 4]),
                    "exit_vz_mps": float(final[index, 5]),
                    "exit_yaw_rad": float(final[index, 6]),
                    "desired_exit_dx_m": float(desired[index, 0]),
                    "desired_exit_dy_m": float(desired[index, 1]),
                    "desired_exit_dz_m": float(desired[index, 2]),
                    "minimum_clearance_m": float(clearance[index]),
                    "position_error_m": float(position_error[index]),
                    "velocity_error_mps": float(velocity_error[index]),
                    "yaw_error_deg": float(np.degrees(yaw_error[index])),
                    "downstream_connector_admissible": bool(connector[index]),
                    "record_admissible": bool(record_admissible[index]),
                    "coverage90": bool(coverage[index]) if model.config.use_outcome_supervision else None,
                    "calibration_scale": calibration_scale if model.config.use_outcome_supervision else None,
                }
            )
            for step, point in enumerate(local_positions[index]):
                trajectory_rows.append(
                    {
                        "method": method,
                        "policy_seed": policy_seed,
                        "command_id": command - calibration_commands,
                        "repeat": local_repeat,
                        "step": step,
                        "time_s": (step + 1) * config.dt,
                        "x_m": float(point[0]),
                        "y_m": float(point[1]),
                        "z_m": float(point[2]),
                    }
                )
    return trial_rows, class_rows, trajectory_rows


def _matched_entry_trials(
    eef_run: Path,
    eef_models: dict[int, EEFPolicy],
    *,
    device: str,
    smoke: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trials: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seeds = tuple(eef_models)[:1] if smoke else tuple(eef_models)
    for seed in seeds:
        legacy_path = eef_run / "training" / "legacy_student" / f"seed_{seed}" / "eef_policy.pt"
        legacy, payload = load_eef_checkpoint(legacy_path, allow_source_rebind=True)
        audit.append(
            {
                "method": "legacy_student",
                "policy_seed": seed,
                "checkpoint": str(legacy_path.resolve()),
                "checkpoint_file_sha256": _sha256(legacy_path),
                "source_checkpoint_immutable_sha256": payload.get(
                    "source_checkpoint_immutable_sha256", payload["immutable_sha256"]
                ),
                "runtime_stack_sha256": payload["immutable_sha256"],
                "weights_updated_during_evaluation": False,
            }
        )
        for method, model in (("legacy_student", legacy), ("eef_full", eef_models[seed])):
            trial_rows, class_rows, trajectory_rows = _matched_entry_for_model(
                model,
                policy_seed=seed,
                method=method,
                calibration_commands=3 if smoke else 36,
                test_commands=4 if smoke else 24,
                repeats=3 if smoke else 10,
                device=device,
            )
            trials.extend(trial_rows)
            classes.extend(class_rows)
            trajectories.extend(trajectory_rows)
        print(json.dumps({"stage": "matched_entry", "policy_seed": seed}), flush=True)
    return trials, classes, trajectories, audit


def _perturbation_trials(
    models: dict[int, EEFPolicy], *, device: str, smoke: bool
) -> list[dict[str, Any]]:
    """Evaluate registered sensing/dynamics perturbations with frozen weights.

    These trials use the differentiable execution surrogate because the current
    PyBullet environment does not expose calibrated image, payload, wind and
    latency interventions.  The evidence tier is carried in every source row.
    """

    rows: list[dict[str, Any]] = []
    conditions = PERTURBATION_CONDITIONS[:2] if smoke else PERTURBATION_CONDITIONS
    samples = 8 if smoke else 128
    for policy_seed, model in models.items():
        seed_rows = evaluate_policy_trials(
            {"eef_full": model},
            model.config,
            samples_per_cell=samples,
            speeds=SPEEDS[:2] if smoke else SPEEDS,
            conditions=conditions,
            seed=93_001 + policy_seed,
            device=device,
            shared_scenes_across_speeds=True,
        )
        for row in seed_rows:
            row.update(
                {
                    "policy_seed": policy_seed,
                    "frozen_policy": True,
                    "evidence_tier": "differentiable_execution_surrogate",
                }
            )
        rows.extend(seed_rows)
        print(json.dumps({"stage": "perturbation", "policy_seed": policy_seed}), flush=True)
    return rows


def _summary(
    single: list[dict[str, Any]],
    multi: list[dict[str, Any]],
    perturbation: list[dict[str, Any]],
    classes: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    single_summary: dict[str, Any] = {}
    for scene in sorted({row["scene"] for row in single}):
        scene_rows = [row for row in single if row["scene"] == scene]
        frontier = {}
        supported_speed = None
        ordered_speeds = sorted({float(row["commanded_speed_mps"]) for row in scene_rows})
        for speed_index, speed in enumerate(ordered_speeds):
            cell = [row for row in scene_rows if float(row["commanded_speed_mps"]) == speed]
            successes = sum(bool(row["success"]) for row in cell)
            paired_keys = {(int(row["policy_seed"]), int(row["repeat"])) for row in cell}
            robust = 0
            for policy_seed, repeat in paired_keys:
                earlier = [
                    row for row in scene_rows
                    if int(row["policy_seed"]) == policy_seed
                    and int(row["repeat"]) == repeat
                    and float(row["commanded_speed_mps"]) <= speed
                ]
                robust += int(len(earlier) == speed_index + 1 and all(bool(row["success"]) for row in earlier))
            frontier[str(speed)] = {
                "successes": successes,
                "attempts": len(cell),
                "success_rate": successes / len(cell),
                "wilson95": list(wilson_interval(successes, len(cell))),
                "robust_through_speed_successes": robust,
                "robust_through_speed_rate": robust / len(paired_keys),
                "robust_through_speed_wilson95": list(wilson_interval(robust, len(paired_keys))),
                "mean_achieved_speed_mps": float(np.mean([row["mean_achieved_speed_mps"] for row in cell])),
                "peak_achieved_speed_mps": float(np.max([row["peak_achieved_speed_mps"] for row in cell])),
            }
            if robust / len(paired_keys) >= SUPPORTED_SPEED_RATE:
                supported_speed = speed
        single_summary[scene] = {
            "label": SCENE_LABELS[scene],
            "maximum_supported_commanded_speed_mps": supported_speed,
            "supported_speed_success_threshold": SUPPORTED_SPEED_RATE,
            "frontier": frontier,
        }
    multi_summary: dict[str, Any] = {}
    for scene in sorted({row["scene"] for row in multi}):
        multi_summary[scene] = {}
        for team_size in sorted({int(row["team_size"]) for row in multi if row["scene"] == scene}):
            cell = [row for row in multi if row["scene"] == scene and int(row["team_size"]) == team_size]
            successes = sum(bool(row["success"]) for row in cell)
            multi_summary[scene][str(team_size)] = {
                "whole_team_successes": successes,
                "attempts": len(cell),
                "whole_team_completion": successes / len(cell),
                "member_completion": float(np.mean([row["member_completion_rate"] for row in cell])),
                "minimum_inter_robot_separation_m": float(
                    min(row["minimum_inter_robot_separation_m"] for row in cell)
                ),
                "cnm_circulation_enabled": False,
            }
    interface_summary: dict[str, Any] = {}
    for method in ("legacy_student", "eef_full"):
        selected = [row for row in classes if row["method"] == method]
        interface_summary[method] = {
            "policy_seeds": len({row["policy_seed"] for row in selected}),
            "response_classes": len(selected),
            "repeatable_response_class_fraction": float(np.mean([row["repeatable_class"] for row in selected])),
            "cnm_reusable_response_class_fraction": float(np.mean([row["cnm_reusable_class"] for row in selected])),
            "mean_exit_position_spread_p90_m": float(np.mean([row["position_spread_p90_m"] for row in selected])),
            "downstream_connector_yield": float(np.mean([row["downstream_connector_yield"] for row in selected])),
            "record_admission_yield": float(np.mean([row["record_admission_yield"] for row in selected])),
            "held_out_coverage90": (
                float(np.mean([row["coverage90"] for row in selected]))
                if method == "eef_full" else None
            ),
            "mean_frozen_calibration_scale": (
                float(np.mean([row["calibration_scale"] for row in selected]))
                if method == "eef_full" else None
            ),
        }
    perturbation_summary: dict[str, Any] = {}
    for condition in PERTURBATION_CONDITIONS:
        selected = [row for row in perturbation if row["condition"] == condition.name]
        if not selected:
            continue
        successes = sum(bool(row["safe_completion"]) for row in selected)
        by_speed = {}
        for speed in sorted({float(row["commanded_speed_mps"]) for row in selected}):
            cell = [row for row in selected if float(row["commanded_speed_mps"]) == speed]
            cell_successes = sum(bool(row["safe_completion"]) for row in cell)
            by_speed[str(speed)] = {
                "successes": cell_successes,
                "attempts": len(cell),
                "safe_completion": cell_successes / len(cell),
                "wilson95": list(wilson_interval(cell_successes, len(cell))),
            }
        perturbation_summary[condition.name] = {
            "label": condition.label,
            "successes": successes,
            "attempts": len(selected),
            "safe_completion": successes / len(selected),
            "wilson95": list(wilson_interval(successes, len(selected))),
            "mean_achieved_speed_mps": float(np.mean([row["achieved_speed_mps"] for row in selected])),
            "mean_interface_position_error_m": float(
                np.mean([row["interface_position_error_m"] for row in selected])
            ),
            "by_speed": by_speed,
        }
    return {
        "schema_version": 1,
        "status": "measured_simulation",
        "scientific_boundary": "PyBullet and differentiable execution surrogate; no hardware or stereo-image evidence",
        "policy_seeds": list(seeds),
        "all_attempts_retained": True,
        "frozen_policy_evaluation": True,
        "single_robot": single_summary,
        "concurrent_multi_robot": multi_summary,
        "controlled_perturbations": perturbation_summary,
        "matched_entry": interface_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/eef_embodied_experience_v1"))
    parser.add_argument("--eef-run", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--trials-per-cell", type=int, default=6)
    parser.add_argument("--multi-repeats", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-pybullet", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = POLICY_SEEDS[:1] if args.smoke else POLICY_SEEDS
    models, checkpoint_audit = _load_models(args.eef_run, seeds)
    if args.skip_pybullet:
        single_rows, multi_rows = [], []
    else:
        single_rows, multi_rows = _pybullet_trials(
            args.output,
            models,
            trials_per_cell=args.trials_per_cell,
            multi_repeats=args.multi_repeats,
            smoke=args.smoke,
        )
    matched_rows, class_rows, trajectory_rows, legacy_audit = _matched_entry_trials(
        args.eef_run, models, device=args.device, smoke=args.smoke
    )
    perturbation_rows = _perturbation_trials(models, device=args.device, smoke=args.smoke)
    checkpoint_audit.extend(legacy_audit)
    _write_csv(args.output / "source_data" / "single_robot_trials.csv", single_rows)
    _write_csv(args.output / "source_data" / "multi_robot_trials.csv", multi_rows)
    _write_csv(args.output / "source_data" / "matched_entry_trials.csv", matched_rows)
    _write_csv(args.output / "source_data" / "matched_entry_class_summary.csv", class_rows)
    _write_csv(args.output / "source_data" / "matched_entry_trajectories.csv", trajectory_rows)
    _write_csv(args.output / "source_data" / "perturbation_trials.csv", perturbation_rows)
    _write_csv(args.output / "source_data" / "checkpoint_audit.csv", checkpoint_audit)
    summary = _summary(single_rows, multi_rows, perturbation_rows, class_rows, seeds)
    _write_json(args.output / "statistical_summary.json", summary)
    _write_json(
        args.output / "protocol.json",
        {
            "single_robot": {
                "scenes": list(SCENES),
                "speeds_mps": list(SPEEDS),
                "trials_per_policy_seed_scene_speed": args.trials_per_cell,
                "maximum_supported_speed_rule": "largest registered speed with pooled all-attempt success >= 0.80",
            },
            "multi_robot": {
                "scenes": list(MULTI_SCENES),
                "team_sizes": list(TEAM_SIZES),
                "speed_mps": 1.2,
                "cnm_circulation_enabled": False,
            },
            "matched_entry": {
                "calibration_commands": 36,
                "held_out_commands": 24,
                "repeats_per_command": 10,
                "covariance_calibration": "single scalar fitted on calibration commands and frozen before held-out evaluation",
                "position_gate_m": POSITION_GATE_M,
                "velocity_gate_mps": VELOCITY_GATE_MPS,
                "yaw_gate_deg": math.degrees(YAW_GATE_RAD),
            },
            "controlled_perturbations": {
                "conditions": [condition.name for condition in PERTURBATION_CONDITIONS],
                "speeds_mps": list(SPEEDS),
                "samples_per_policy_seed_condition": 128,
                "evidence_tier": "differentiable execution surrogate",
            },
        },
    )
    _write_json(
        args.output / "run_manifest.json",
        {
            "arguments": vars(args) | {"output": str(args.output), "eef_run": str(args.eef_run)},
            "elapsed_seconds": time.perf_counter() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
