"""Registered synthetic and PyBullet-facing evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
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
    privileged_target,
)


@dataclass(frozen=True)
class EEFCondition:
    name: str
    sensor_dropout: float = 0.0
    wind_scale: float = 1.0
    response_scale: float | None = None
    delay_extra_steps: int = 0
    drag_scale: float = 1.0
    observation_noise: float = 0.0


def _goal_response(batch: SyntheticBatch, config: EEFConfig) -> torch.Tensor:
    direction = F.normalize(batch.target - batch.position, dim=-1)
    velocity = direction * batch.target_speed
    phase = torch.linspace(0.60, 1.0, config.response_horizon, device=batch.position.device)[None, :, None]
    sequence = velocity[:, None, :] * phase
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    yaw_rate = (yaw / (config.response_horizon * config.dt)).clamp(-config.max_yaw_rate, config.max_yaw_rate)
    return torch.cat((sequence, yaw_rate[:, None, None].expand(-1, config.response_horizon, 1)), dim=-1)


def evaluate_eef_models(
    models: dict[str, EEFPolicy | None],
    config: EEFConfig,
    *,
    samples: int = 256,
    speeds: Iterable[float] = (0.8, 1.2, 1.6, 2.0),
    conditions: Iterable[EEFCondition] | None = None,
    seed: int = 101,
    device: str = "cpu",
) -> list[dict[str, float | str | None]]:
    """Measure all-attempt local completion, clearance, and exit alignment."""

    if conditions is None:
        conditions = (
            EEFCondition("nominal"),
            EEFCondition("sensor_30pct", sensor_dropout=0.30, observation_noise=0.035),
            EEFCondition("wind", wind_scale=1.8),
            EEFCondition("payload", response_scale=0.82),
            EEFCondition("latency", delay_extra_steps=2),
            EEFCondition(
                "combined",
                sensor_dropout=0.20,
                wind_scale=1.6,
                response_scale=0.82,
                delay_extra_steps=1,
                drag_scale=1.25,
                observation_noise=0.025,
            ),
        )
    torch_device = torch.device(device)
    rows: list[dict[str, float | str | None]] = []
    for speed_index, speed in enumerate(speeds):
        for condition_index, condition in enumerate(conditions):
            generator = torch.Generator(device=torch_device).manual_seed(seed + speed_index * 1009 + condition_index * 9173)
            batch = _sample_batch(config, samples, generator, torch_device)
            batch.target_speed[:] = float(speed)
            features = _features(batch, config)
            if condition.sensor_dropout > 0.0:
                mask = torch.rand(
                    (samples, config.ray_count), generator=generator, device=torch_device
                ) < condition.sensor_dropout
                features[:, : config.ray_count] = torch.where(
                    mask, torch.zeros_like(features[:, : config.ray_count]), features[:, : config.ray_count]
                )
            if condition.observation_noise > 0.0:
                noise = torch.randn(
                    (samples, config.ray_count), generator=generator, device=torch_device
                ) * condition.observation_noise
                features[:, : config.ray_count] = (features[:, : config.ray_count] + noise).clamp(0.0, 1.0)
            wind = batch.wind * condition.wind_scale
            response_scale = (
                batch.response_scale
                if condition.response_scale is None
                else torch.full_like(batch.response_scale, condition.response_scale)
            )
            delay_steps = (batch.delay_steps + int(condition.delay_extra_steps)).clamp(
                max=config.execution_delay_max_steps + 3
            )
            drag_scale = batch.drag_scale * condition.drag_scale
            for method, model in models.items():
                predicted_exit = None
                predicted_std = None
                with torch.no_grad():
                    if method == "goal_only":
                        response = _goal_response(batch, config)
                    elif method == "privileged_teacher":
                        response, predicted_exit, _ = privileged_target(batch, config)
                    else:
                        if model is None:
                            raise ValueError(f"method {method} requires a model")
                        model = model.to(torch_device).eval()
                        response, predicted_exit, scale_tril, _ = model(features)
                        predicted_std = torch.diagonal(
                            scale_tril @ scale_tril.transpose(-1, -2), dim1=-2, dim2=-1
                        ).sqrt()
                    rollout = differentiable_rollout(
                        batch.position,
                        batch.velocity,
                        response,
                        batch.obstacles,
                        batch.radii,
                        config,
                        wind=wind,
                        response_scale=response_scale,
                        delay_steps=delay_steps,
                        drag_scale=drag_scale,
                        decay_gradients=False,
                    )
                initial_distance = (batch.target - batch.position).norm(dim=-1)
                final_distance = (batch.target - rollout.positions[:, -1]).norm(dim=-1)
                progress_fraction = 1.0 - final_distance / initial_distance.clamp_min(0.2)
                minimum_clearance = rollout.clearance.amin(dim=1)
                expected_progress = (
                    float(speed) * config.response_horizon * config.dt / initial_distance.clamp_min(0.2)
                ).clamp(max=0.85)
                completed = (progress_fraction >= 0.35 * expected_progress) & (minimum_clearance > 0.0)
                achieved_speed = rollout.velocities.norm(dim=-1).mean(dim=1)
                acceleration = rollout.accelerations.norm(dim=-1).mean(dim=1)
                jerk = torch.diff(rollout.accelerations, dim=1).norm(dim=-1).mean(dim=1) / config.dt
                terminal_position_error: float | None = None
                coverage90_value: float | None = None
                if predicted_exit is not None:
                    residual = rollout.final_state - predicted_exit
                    terminal_position_error = float(residual[:, :3].norm(dim=-1).mean())
                    if predicted_std is not None:
                        whitened = torch.linalg.solve_triangular(
                            scale_tril,
                            residual.unsqueeze(-1),
                            upper=False,
                        ).squeeze(-1)
                        mahalanobis = whitened.square().sum(dim=-1)
                        coverage90_value = float((mahalanobis <= 12.017).float().mean())
                rows.append(
                    {
                        "method": method,
                        "condition": condition.name,
                        "commanded_speed": float(speed),
                        "samples": float(samples),
                        "success_rate": float(completed.float().mean()),
                        "collision_rate": float((minimum_clearance <= 0.0).float().mean()),
                        "mean_progress_fraction": float(progress_fraction.mean()),
                        "mean_achieved_speed": float(achieved_speed.mean()),
                        "mean_min_clearance": float(minimum_clearance.mean()),
                        "mean_acceleration": float(acceleration.mean()),
                        "mean_jerk": float(jerk.mean()),
                        "terminal_position_error": terminal_position_error,
                        "coverage90": coverage90_value,
                    }
                )
    return rows


def robustness_agility_area(
    rows: list[dict[str, float | str | None]], method: str, condition: str = "nominal"
) -> float:
    selected = sorted(
        (row for row in rows if row["method"] == method and row["condition"] == condition),
        key=lambda row: float(row["commanded_speed"]),
    )
    if len(selected) < 2:
        return float("nan")
    speed = np.asarray([row["commanded_speed"] for row in selected], dtype=np.float64)
    success = np.asarray([row["success_rate"] for row in selected], dtype=np.float64)
    return float(np.trapz(success, speed) / (speed[-1] - speed[0]))
