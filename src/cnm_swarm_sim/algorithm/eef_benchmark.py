"""Matched, all-attempt benchmarks for embodied experience formation.

The training surrogate and PyBullet remain separate.  This module evaluates
frozen policies under an execution model that contains actuation gain, delay,
drag and force residuals absent from the nominal predictor.  Every method sees
the same sampled scene and perturbation realization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .eef import (
    EEFConfig,
    EEFPolicy,
    SyntheticBatch,
    _features,
    _sample_batch,
    differentiable_rollout,
)
from .evaluation import _goal_response


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    label: str
    sensor_dropout: float = 0.0
    observation_noise: float = 0.0
    wind_multiplier: float = 1.0
    response_scale: float | None = None
    delay_extra_steps: int = 0
    drag_multiplier: float = 1.0
    reset_embodiment: bool = False


DEFAULT_CONDITIONS = (
    BenchmarkCondition("nominal", "Nominal"),
    BenchmarkCondition("depth_dropout_10", "10% depth\ndropout", sensor_dropout=0.10),
    BenchmarkCondition("depth_dropout_30", "30% depth\ndropout", sensor_dropout=0.30),
    BenchmarkCondition("delay_67ms", "67 ms\ndelay", delay_extra_steps=1),
    BenchmarkCondition("delay_133ms", "133 ms\ndelay", delay_extra_steps=2),
    BenchmarkCondition("payload_10", "+10%\npayload", response_scale=0.82, drag_multiplier=1.15),
    BenchmarkCondition(
        "combined",
        "Combined",
        sensor_dropout=0.20,
        observation_noise=0.025,
        wind_multiplier=1.6,
        response_scale=0.82,
        delay_extra_steps=1,
        drag_multiplier=1.20,
    ),
)


def _condition_batch(batch: SyntheticBatch, condition: BenchmarkCondition) -> SyntheticBatch:
    if condition.reset_embodiment:
        return replace(
            batch,
            wind=torch.zeros_like(batch.wind),
            response_scale=torch.ones_like(batch.response_scale),
            delay_steps=torch.zeros_like(batch.delay_steps),
            drag_scale=torch.ones_like(batch.drag_scale),
        )
    scale = (
        batch.response_scale
        if condition.response_scale is None
        else torch.full_like(batch.response_scale, condition.response_scale)
    )
    return replace(
        batch,
        wind=batch.wind * condition.wind_multiplier,
        response_scale=scale,
        delay_steps=batch.delay_steps + int(condition.delay_extra_steps),
        drag_scale=batch.drag_scale * condition.drag_multiplier,
    )


def _corrupt_features(
    features: torch.Tensor,
    config: EEFConfig,
    condition: BenchmarkCondition,
    generator: torch.Generator,
) -> torch.Tensor:
    result = features.clone()
    if condition.sensor_dropout:
        mask = torch.rand(
            (features.shape[0], config.ray_count),
            generator=generator,
            device=features.device,
        ) < condition.sensor_dropout
        result[:, : config.ray_count] = torch.where(
            mask,
            torch.zeros_like(result[:, : config.ray_count]),
            result[:, : config.ray_count],
        )
    if condition.observation_noise:
        noise = torch.randn(
            (features.shape[0], config.ray_count),
            generator=generator,
            device=features.device,
        ) * condition.observation_noise
        result[:, : config.ray_count] = (
            result[:, : config.ray_count] + noise
        ).clamp(0.0, 1.0)
    return result


def _repeat_batch(batch: SyntheticBatch, repeats: int) -> SyntheticBatch:
    return SyntheticBatch(
        **{
            name: getattr(batch, name).repeat_interleave(repeats, dim=0)
            for name in batch.__dataclass_fields__
        }
    )


def evaluate_policy_trials(
    models: dict[str, EEFPolicy | None],
    config: EEFConfig,
    *,
    samples_per_cell: int = 256,
    speeds: Iterable[float] = (0.8, 1.2, 1.6, 2.0, 2.4),
    conditions: Iterable[BenchmarkCondition] = DEFAULT_CONDITIONS,
    seed: int = 1103,
    device: str = "cpu",
    shared_scenes_across_speeds: bool = False,
) -> list[dict[str, float | int | str | bool | None]]:
    """Return one source-data row per policy, scene and attempted response."""

    torch_device = torch.device(device)
    rows: list[dict[str, float | int | str | bool | None]] = []
    for speed_index, speed in enumerate(speeds):
        scene_seed = seed if shared_scenes_across_speeds else seed + 7919 * speed_index
        base_generator = torch.Generator(device=torch_device).manual_seed(scene_seed)
        base = _sample_batch(config, samples_per_cell, base_generator, torch_device)
        base.target_speed[:] = float(speed)
        for condition_index, condition in enumerate(conditions):
            condition_generator = torch.Generator(device=torch_device).manual_seed(
                seed + 7919 * speed_index + 104729 * (condition_index + 1)
            )
            batch = _condition_batch(base, condition)
            features = _corrupt_features(_features(batch, config), config, condition, condition_generator)
            for method, model in models.items():
                with torch.no_grad():
                    if method == "reactive_goal":
                        response = _goal_response(batch, config)
                        predicted_exit = None
                        predicted_std = None
                    else:
                        if model is None:
                            raise ValueError(f"method {method} requires a policy")
                        model = model.to(torch_device).eval()
                        response, predicted_exit, scale_tril, _ = model(features)
                        predicted_std = torch.diagonal(
                            scale_tril @ scale_tril.transpose(-1, -2), dim1=-2, dim2=-1
                        ).sqrt()
                    nominal = differentiable_rollout(
                        batch.position,
                        batch.velocity,
                        response,
                        batch.obstacles,
                        batch.radii,
                        config,
                        decay_gradients=False,
                    )
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

                direction = F.normalize(batch.target - batch.position, dim=-1)
                initial_distance = (batch.target - batch.position).norm(dim=-1)
                desired_travel = torch.minimum(
                    initial_distance * 0.80,
                    batch.target_speed[:, 0] * config.response_horizon * config.dt * 0.72,
                )
                desired_heading = torch.atan2(direction[:, 1], direction[:, 0])
                desired_exit = torch.cat(
                    (
                        direction * desired_travel[:, None],
                        direction * batch.target_speed,
                        desired_heading[:, None],
                    ),
                    dim=-1,
                )
                displacement = executed.positions[:, -1] - batch.position
                along = (displacement * direction).sum(dim=-1)
                lateral = (displacement - along[:, None] * direction).norm(dim=-1)
                expected = torch.minimum(
                    (batch.target - batch.position).norm(dim=-1) * 0.80,
                    torch.full_like(along, float(speed) * config.response_horizon * config.dt * 0.72),
                )
                minimum_clearance = executed.clearance.amin(dim=1)
                success = (minimum_clearance > 0.0) & (along >= 0.55 * expected) & (lateral <= 1.0)
                execution_mismatch = (executed.final_state - nominal.final_state).norm(dim=-1)
                interface_position_error = (executed.final_state[:, :3] - desired_exit[:, :3]).norm(dim=-1)
                interface_velocity_error = (executed.final_state[:, 3:6] - desired_exit[:, 3:6]).norm(dim=-1)
                interface_heading_error = torch.atan2(
                    torch.sin(executed.final_state[:, 6] - desired_exit[:, 6]),
                    torch.cos(executed.final_state[:, 6] - desired_exit[:, 6]),
                ).abs()
                speed_value = executed.velocities.norm(dim=-1).mean(dim=1)
                acceleration = executed.accelerations.norm(dim=-1).mean(dim=1)
                jerk = torch.diff(executed.accelerations, dim=1).norm(dim=-1).mean(dim=1) / config.dt
                if predicted_exit is None or model is None or not model.config.use_outcome_supervision:
                    prediction_error = torch.full_like(along, torch.nan)
                    coverage90 = torch.zeros_like(success)
                else:
                    residual = executed.final_state - predicted_exit
                    prediction_error = residual[:, :3].norm(dim=-1)
                    whitened = torch.linalg.solve_triangular(
                        scale_tril,
                        residual.unsqueeze(-1),
                        upper=False,
                    ).squeeze(-1)
                    mahalanobis = whitened.square().sum(dim=-1)
                    coverage90 = mahalanobis <= 12.017
                arrays = {
                    "safe_completion": success,
                    "collision": minimum_clearance <= 0.0,
                    "progress_m": along,
                    "lateral_error_m": lateral,
                    "minimum_clearance_m": minimum_clearance,
                    "achieved_speed_mps": speed_value,
                    "mean_acceleration_mps2": acceleration,
                    "mean_jerk_mps3": jerk,
                    "prediction_execution_mismatch": execution_mismatch,
                    "interface_position_error_m": interface_position_error,
                    "interface_velocity_error_mps": interface_velocity_error,
                    "interface_heading_error_rad": interface_heading_error,
                    "predicted_position_error_m": prediction_error,
                    "coverage90": coverage90,
                }
                final = executed.final_state
                for trial in range(samples_per_cell):
                    row: dict[str, float | int | str | bool | None] = {
                        "method": method,
                        "condition": condition.name,
                        "condition_label": condition.label,
                        "commanded_speed_mps": float(speed),
                        "trial": trial,
                        "scene_seed": scene_seed,
                        "outcome_supervised": bool(
                            model is not None and model.config.use_outcome_supervision
                        ),
                        "exit_dx_m": float(final[trial, 0]),
                        "exit_dy_m": float(final[trial, 1]),
                        "exit_dz_m": float(final[trial, 2]),
                        "exit_vx_mps": float(final[trial, 3]),
                        "exit_vy_mps": float(final[trial, 4]),
                        "exit_vz_mps": float(final[trial, 5]),
                        "exit_yaw_rad": float(final[trial, 6]),
                    }
                    for key, values in arrays.items():
                        value = values[trial]
                        if value.dtype == torch.bool:
                            row[key] = bool(value)
                        else:
                            numeric = float(value)
                            row[key] = numeric if np.isfinite(numeric) else None
                    rows.append(row)
    return rows


def evaluate_interface_repeatability(
    models: dict[str, EEFPolicy | None],
    config: EEFConfig,
    *,
    commands: int = 24,
    repeats: int = 12,
    speed: float = 1.6,
    seed: int = 2207,
    device: str = "cpu",
) -> list[dict[str, float | int | str | bool]]:
    """Repeat matched local commands while varying only embodied execution."""

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    base = _sample_batch(config, commands, generator, torch_device)
    base.target_speed[:] = speed
    direction = F.normalize(base.target - base.position, dim=-1)
    initial_distance = (base.target - base.position).norm(dim=-1)
    desired_travel = torch.minimum(
        initial_distance * 0.80,
        base.target_speed[:, 0] * config.response_horizon * config.dt * 0.72,
    )
    desired_heading = torch.atan2(direction[:, 1], direction[:, 0])
    desired_exit = torch.cat(
        (direction * desired_travel[:, None], direction * base.target_speed, desired_heading[:, None]),
        dim=-1,
    ).repeat_interleave(repeats, dim=0)
    batch = _repeat_batch(base, repeats)
    n = commands * repeats
    execution_generator = torch.Generator(device=torch_device).manual_seed(seed + 1)
    rand = lambda *shape: torch.rand(shape, generator=execution_generator, device=torch_device)
    batch.wind = (rand(n, 3) - 0.5) * torch.tensor((0.90, 0.90, 0.30), device=torch_device)
    batch.response_scale = 0.76 + 0.48 * rand(n, 1)
    batch.delay_steps = torch.randint(
        0,
        config.execution_delay_max_steps + 2,
        (n,),
        generator=execution_generator,
        device=torch_device,
    )
    batch.drag_scale = 0.65 + 0.90 * rand(n, 1)
    features = _features(batch, config)
    rows: list[dict[str, float | int | str | bool]] = []
    for method, model in models.items():
        with torch.no_grad():
            if method == "reactive_goal":
                response = _goal_response(batch, config)
            else:
                if model is None:
                    raise ValueError(f"method {method} requires a policy")
                response, _, _, _ = model.to(torch_device).eval()(features)
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
        final = executed.final_state.detach().cpu().numpy()
        clearance = executed.clearance.amin(dim=1).detach().cpu().numpy()
        for index in range(n):
            rows.append(
                {
                    "method": method,
                    "command_id": index // repeats,
                    "repeat": index % repeats,
                    "speed_mps": speed,
                    "collision": bool(clearance[index] <= 0.0),
                    "minimum_clearance_m": float(clearance[index]),
                    "exit_dx_m": float(final[index, 0]),
                    "exit_dy_m": float(final[index, 1]),
                    "exit_dz_m": float(final[index, 2]),
                    "exit_vx_mps": float(final[index, 3]),
                    "exit_vy_mps": float(final[index, 4]),
                    "exit_vz_mps": float(final[index, 5]),
                    "exit_yaw_rad": float(final[index, 6]),
                    "desired_exit_dx_m": float(desired_exit[index, 0]),
                    "desired_exit_dy_m": float(desired_exit[index, 1]),
                    "desired_exit_dz_m": float(desired_exit[index, 2]),
                    "desired_exit_vx_mps": float(desired_exit[index, 3]),
                    "desired_exit_vy_mps": float(desired_exit[index, 4]),
                    "desired_exit_vz_mps": float(desired_exit[index, 5]),
                    "desired_exit_yaw_rad": float(desired_exit[index, 6]),
                }
            )
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - half), float(center + half)
