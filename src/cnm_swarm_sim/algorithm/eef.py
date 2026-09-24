"""Lightweight differentiable training for embodied experience formation.

PyBullet remains the evaluation environment.  This module supplies the
training-only point-mass surrogate, privileged candidate evaluation, temporal
gradient decay, the deployed recurrent student, and immutable checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without train extra
    raise ImportError("EEF training requires the 'train' extra: pip install -e .[train]") from exc


@dataclass(frozen=True)
class EEFConfig:
    """Frozen architecture, physics, loss, and command-domain parameters."""

    ray_count: int = 24
    ray_range: float = 4.0
    obstacle_count: int = 10
    hidden_dim: int = 96
    response_horizon: int = 15
    dt: float = 1.0 / 15.0
    max_speed: float = 2.0
    max_accel: float = 5.0
    max_yaw_rate: float = 2.0
    velocity_kp: float = 3.0
    drone_radius: float = 0.115
    safety_margin: float = 0.10
    gradient_decay: float = 0.40
    covariance_floor: float = 0.025
    recurrent_retention: float = 0.0
    terminal_feedback: float = 0.35
    candidate_angles: int = 9
    candidate_vertical: int = 3
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    lambda_imitation: float = 1.5
    lambda_dynamics: float = 0.20
    lambda_nll: float = 0.04
    lambda_progress: float = 1.0
    lambda_clearance: float = 3.0
    lambda_exit: float = 0.75
    lambda_acceleration: float = 0.01
    lambda_jerk: float = 0.001
    # Registered training intervention.  The architecture and sample budget are
    # unchanged between variants; these switches isolate the source of the
    # learning signal.
    training_variant: str = "eef_full"
    teacher_embodiment: bool = True
    use_privileged_filter: bool = True
    use_differentiable_objective: bool = True
    differentiable_embodiment: bool = True
    use_outcome_supervision: bool = True
    decay_gradients: bool = True
    execution_delay_max_steps: int = 2
    drag_coefficient: float = 0.025
    # Optional fixed command-conditioned prior used by the response-level PPO
    # comparator and residual EEF variants. The trainable head predicts a
    # residual around this prior; gain can be scheduled from zero by a trainer.
    command_prior_residual_scale: float = 0.0
    command_prior_gain: float = 1.0
    command_prior_start_phase: float = 0.82
    command_prior_end_phase: float = 1.0
    outcome_feedback_gain: float = 0.0
    # Near a demonstrated moving interface, correct cross-track position while
    # preserving the demonstrated along-track terminal velocity.
    interface_position_gain: float = 1.6
    interface_terminal_feedback: float = 1.0
    interface_terminal_activation_radius: float = 1.5
    interface_yaw_gain: float = 3.0
    interface_override_clearance: float = 0.50

    @property
    def input_dim(self) -> int:
        # rays, body velocity, local target, nearest peer state, mode, speed
        return self.ray_count + 3 + 3 + 6 + 4 + 1

    @property
    def outcome_dim(self) -> int:
        # Terminal delta-position, velocity and continuous delta-yaw.
        return 7


def eef_config_for_variant(variant: str, base: EEFConfig | None = None, **overrides: Any) -> EEFConfig:
    """Return one preregistered EEF intervention with matched architecture.

    ``legacy_student`` receives only imitation targets generated with nominal
    dynamics. ``nominal_dynamics`` adds differentiable nominal rollouts. The
    complete method additionally forms targets under randomized execution and
    predicts the resulting terminal distribution. The remaining variants are
    single-factor ablations of the complete method.
    """

    cfg = base or EEFConfig()
    settings: dict[str, Any] = {
        "training_variant": variant,
        "teacher_embodiment": True,
        "use_privileged_filter": True,
        "use_differentiable_objective": True,
        "differentiable_embodiment": True,
        "use_outcome_supervision": True,
        "decay_gradients": True,
        # A 15-step local interface is short enough to retain full-horizon
        # gradients. Strong attenuation is kept as an explicit ablation.
        "gradient_decay": 1.0,
    }
    if variant == "legacy_student":
        settings.update(
            teacher_embodiment=False,
            use_differentiable_objective=False,
            differentiable_embodiment=False,
            use_outcome_supervision=False,
        )
    elif variant == "nominal_dynamics":
        settings.update(
            teacher_embodiment=False,
            differentiable_embodiment=False,
            use_outcome_supervision=False,
            gradient_decay=0.40,
        )
    elif variant == "eef_no_outcome":
        settings["use_outcome_supervision"] = False
    elif variant == "eef_no_differentiable":
        settings["use_differentiable_objective"] = False
    elif variant == "eef_no_filter":
        settings["use_privileged_filter"] = False
    elif variant in ("eef_temporal_decay", "eef_no_decay"):
        settings["gradient_decay"] = 0.40
    elif variant != "eef_full":
        raise ValueError(f"unknown EEF training variant: {variant}")
    settings.update(overrides)
    return replace(cfg, **settings)


class _TemporalGradientDecay(torch.autograd.Function):
    """Identity forward pass with a fixed multiplier on the backward pass."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, factor: float) -> Tensor:
        ctx.factor = float(factor)
        return value

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient * ctx.factor, None


def temporal_gradient_decay(value: Tensor, factor: float) -> Tensor:
    return _TemporalGradientDecay.apply(value, factor)


class EEFPolicy(nn.Module):
    """Small scan/state GRU that predicts a response sequence and exit density."""

    def __init__(self, config: EEFConfig):
        super().__init__()
        self.config = config
        state_dim = config.input_dim - config.ray_count
        self.scan_encoder = nn.Sequential(
            nn.Linear(config.ray_count, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
        )
        self.recurrent = nn.GRUCell(128, config.hidden_dim)
        self.response_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_dim, config.response_horizon * 4),
        )
        # Terminal mean followed by a full lower-triangular Cholesky factor.
        outcome_parameters = config.outcome_dim + config.outcome_dim * (config.outcome_dim + 1) // 2
        self.outcome_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, outcome_parameters),
        )

    def forward(self, features: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        if hidden is None:
            hidden = torch.zeros((features.shape[0], cfg.hidden_dim), dtype=features.dtype, device=features.device)
        scan = self.scan_encoder(features[:, : cfg.ray_count])
        state = self.state_encoder(features[:, cfg.ray_count :])
        hidden_next = self.recurrent(torch.cat((scan, state), dim=-1), hidden)
        raw = self.response_head(hidden_next).reshape(-1, cfg.response_horizon, 4)
        velocity = torch.tanh(raw[..., :3]) * cfg.max_speed
        yaw_rate = torch.tanh(raw[..., 3:4]) * cfg.max_yaw_rate
        response = torch.cat((velocity, yaw_rate), dim=-1)
        if cfg.command_prior_residual_scale > 0.0:
            target_start = cfg.ray_count + 3
            local_target = features[:, target_start : target_start + 3]
            direction = F.normalize(local_target, dim=-1)
            target_speed = features[:, -1:].clamp(0.0, 1.0) * cfg.max_speed
            phase = torch.linspace(
                cfg.command_prior_start_phase,
                cfg.command_prior_end_phase,
                cfg.response_horizon,
                device=features.device,
            )[None, :, None]
            prior_velocity = direction[:, None, :] * target_speed[:, None, :] * phase
            desired_yaw = torch.atan2(direction[:, 1], direction[:, 0])
            prior_yaw_rate = (
                desired_yaw / max(cfg.response_horizon * cfg.dt, 1.0e-6)
            ).clamp(-cfg.max_yaw_rate, cfg.max_yaw_rate)
            prior = torch.cat(
                (
                    prior_velocity,
                    prior_yaw_rate[:, None, None].expand(-1, cfg.response_horizon, 1),
                ),
                dim=-1,
            )
            response = (
                cfg.command_prior_gain * prior
                + cfg.command_prior_residual_scale * response
            )
            response = torch.cat(
                (
                    _row_clip(response[..., :3], cfg.max_speed),
                    response[..., 3:4].clamp(-cfg.max_yaw_rate, cfg.max_yaw_rate),
                ),
                dim=-1,
            )
        # The probabilistic observer is optimized from executed outcomes but its
        # gradient is stopped at the shared latent. This prevents covariance
        # fitting from trading away progress or clearance in the controller.
        # Terminal controllability is supplied by the differentiable exit term;
        # this head estimates the residual distribution for downstream CNM.
        outcome = self.outcome_head(hidden_next.detach())
        exit_mean = outcome[:, : cfg.outcome_dim]
        raw_cholesky = outcome[:, cfg.outcome_dim :]
        indices = torch.tril_indices(cfg.outcome_dim, cfg.outcome_dim, device=features.device)
        scale_tril = torch.zeros(
            (features.shape[0], cfg.outcome_dim, cfg.outcome_dim),
            dtype=features.dtype,
            device=features.device,
        )
        scale_tril[:, indices[0], indices[1]] = raw_cholesky
        diagonal = torch.arange(cfg.outcome_dim, device=features.device)
        scale_tril[:, diagonal, diagonal] = (
            F.softplus(scale_tril[:, diagonal, diagonal]) + cfg.covariance_floor
        )
        return response, exit_mean, scale_tril, hidden_next


@dataclass
class SyntheticBatch:
    position: Tensor
    velocity: Tensor
    target: Tensor
    target_speed: Tensor
    obstacles: Tensor
    radii: Tensor
    peer_state: Tensor
    mode: Tensor
    wind: Tensor
    response_scale: Tensor
    delay_steps: Tensor
    drag_scale: Tensor


@dataclass
class Rollout:
    positions: Tensor
    velocities: Tensor
    accelerations: Tensor
    clearance: Tensor
    yaws: Tensor
    final_state: Tensor


def _row_clip(value: Tensor, maximum: float) -> Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return value * torch.clamp(maximum / norm, max=1.0)


def _sample_batch(config: EEFConfig, batch_size: int, generator: torch.Generator, device: torch.device) -> SyntheticBatch:
    rand = lambda *shape: torch.rand(shape, generator=generator, device=device)
    position = torch.zeros((batch_size, 3), device=device)
    position[:, 2] = 1.35 + 0.30 * rand(batch_size)
    target = position.clone()
    target[:, 0] += 1.4 + 1.2 * rand(batch_size)
    target[:, 1] += -1.25 + 2.50 * rand(batch_size)
    target[:, 2] = (position[:, 2] - 0.45 + 0.90 * rand(batch_size)).clamp(0.55, 3.5)
    target_speed = 0.65 + (config.max_speed - 0.65) * rand(batch_size, 1)
    velocity = (rand(batch_size, 3) - 0.5) * 0.45

    obstacles = torch.empty((batch_size, config.obstacle_count, 3), device=device)
    obstacles[..., 0] = 0.25 + 2.75 * rand(batch_size, config.obstacle_count)
    obstacles[..., 1] = -2.0 + 4.0 * rand(batch_size, config.obstacle_count)
    # Horizontal deployment rays must contain enough information to explain
    # collision risk.  Training obstacles therefore intersect the vehicle's
    # sensing plane; later PyBullet tests add truly three-dimensional gates.
    obstacles[..., 2] = position[:, None, 2] - 0.18 + 0.36 * rand(batch_size, config.obstacle_count)
    radii = 0.13 + 0.34 * rand(batch_size, config.obstacle_count)
    active = rand(batch_size, config.obstacle_count) > 0.24
    radii = radii * active

    peer_state = torch.zeros((batch_size, 6), device=device)
    peer_present = rand(batch_size) < 0.35
    peer_state[peer_present, :3] = torch.stack(
        (
            0.55 + 1.8 * rand(int(peer_present.sum())),
            -1.3 + 2.6 * rand(int(peer_present.sum())),
            -0.35 + 0.70 * rand(int(peer_present.sum())),
        ),
        dim=-1,
    ) if bool(peer_present.any()) else peer_state[peer_present, :3]
    if bool(peer_present.any()):
        peer_state[peer_present, 3:] = (rand(int(peer_present.sum()), 3) - 0.5) * 1.2

    mode_index = torch.randint(0, 4, (batch_size,), generator=generator, device=device)
    mode = F.one_hot(mode_index, num_classes=4).float()
    wind = (rand(batch_size, 3) - 0.5) * torch.tensor((0.65, 0.65, 0.22), device=device)
    response_scale = 0.82 + 0.36 * rand(batch_size, 1)
    delay_steps = torch.randint(
        0,
        config.execution_delay_max_steps + 1,
        (batch_size,),
        generator=generator,
        device=device,
    )
    drag_scale = 0.72 + 0.66 * rand(batch_size, 1)
    return SyntheticBatch(
        position,
        velocity,
        target,
        target_speed,
        obstacles,
        radii,
        peer_state,
        mode,
        wind,
        response_scale,
        delay_steps,
        drag_scale,
    )


def _horizontal_rays(batch: SyntheticBatch, config: EEFConfig) -> Tensor:
    """Analytic circular ray intersections in the gravity-aligned body plane."""

    angles = torch.linspace(-math.pi, math.pi, config.ray_count + 1, device=batch.position.device)[:-1]
    directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
    centers = batch.obstacles[..., :2] - batch.position[:, None, :2]
    projection = torch.einsum("bmc,rc->bmr", centers, directions)
    center_sq = (centers * centers).sum(dim=-1, keepdim=True)
    perpendicular_sq = center_sq - projection.square()
    radius_sq = batch.radii[..., None].square()
    discriminant = radius_sq - perpendicular_sq
    valid = (projection > 0.0) & (discriminant >= 0.0) & (batch.radii[..., None] > 0.0)
    distance = projection - discriminant.clamp_min(0.0).sqrt()
    distance = torch.where(valid, distance, torch.full_like(distance, config.ray_range))
    return distance.amin(dim=1).clamp(0.0, config.ray_range) / config.ray_range


def _features(batch: SyntheticBatch, config: EEFConfig) -> Tensor:
    rays = _horizontal_rays(batch, config)
    proximity = 1.0 - rays
    local_target = (batch.target - batch.position) / config.ray_range
    local_target = local_target.clamp(-1.0, 1.0)
    peer = batch.peer_state.clone()
    peer[:, :3] /= config.ray_range
    peer[:, 3:] /= config.max_speed
    return torch.cat(
        (
            proximity,
            batch.velocity / config.max_speed,
            local_target,
            peer,
            batch.mode,
            batch.target_speed / config.max_speed,
        ),
        dim=-1,
    )


def differentiable_rollout(
    initial_position: Tensor,
    initial_velocity: Tensor,
    response: Tensor,
    obstacles: Tensor,
    radii: Tensor,
    config: EEFConfig,
    *,
    wind: Tensor | None = None,
    response_scale: Tensor | None = None,
    delay_steps: Tensor | None = None,
    drag_scale: Tensor | None = None,
    decay_gradients: bool = True,
) -> Rollout:
    """Roll a desired-velocity sequence through a differentiable point mass."""

    batch_size, horizon, _ = response.shape
    position = initial_position
    velocity = initial_velocity
    yaw = torch.zeros(batch_size, dtype=response.dtype, device=response.device)
    wind_term = torch.zeros_like(velocity) if wind is None else wind
    scale = torch.ones((batch_size, 1), device=response.device) if response_scale is None else response_scale
    delays = torch.zeros(batch_size, dtype=torch.long, device=response.device) if delay_steps is None else delay_steps.long()
    drag = torch.ones((batch_size, 1), device=response.device) if drag_scale is None else drag_scale
    positions: list[Tensor] = []
    velocities: list[Tensor] = []
    accelerations: list[Tensor] = []
    clearances: list[Tensor] = []
    yaws: list[Tensor] = []
    decay = config.gradient_decay ** config.dt
    row = torch.arange(batch_size, device=response.device)
    for step in range(horizon):
        command_index = (step - delays).clamp(0, horizon - 1)
        delayed_response = response[row, command_index]
        delayed_command = delayed_response[:, :3]
        velocity_command = torch.where(
            (step < delays)[:, None], initial_velocity, delayed_command
        )
        yaw_rate = torch.where(step < delays, torch.zeros_like(yaw), delayed_response[:, 3])
        acceleration = _row_clip(config.velocity_kp * (velocity_command - velocity), config.max_accel)
        acceleration = (
            acceleration * scale
            + wind_term
            - config.drag_coefficient * drag * velocity * velocity.norm(dim=-1, keepdim=True)
        )
        source_position = temporal_gradient_decay(position, decay) if decay_gradients else position
        source_velocity = temporal_gradient_decay(velocity, decay) if decay_gradients else velocity
        position = source_position + source_velocity * config.dt + 0.5 * acceleration * config.dt**2
        velocity = source_velocity + acceleration * config.dt
        yaw = yaw + yaw_rate * config.dt
        signed = (position[:, None, :] - obstacles).norm(dim=-1) - radii - config.drone_radius
        inactive = radii <= 0.0
        signed = torch.where(inactive, torch.full_like(signed, config.ray_range), signed)
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(acceleration)
        clearances.append(signed.amin(dim=1))
        yaws.append(yaw)
    position_history = torch.stack(positions, dim=1)
    velocity_history = torch.stack(velocities, dim=1)
    acceleration_history = torch.stack(accelerations, dim=1)
    clearance_history = torch.stack(clearances, dim=1)
    yaw_history = torch.stack(yaws, dim=1)
    final_state = torch.cat((position - initial_position, velocity, yaw[:, None]), dim=-1)
    return Rollout(
        position_history,
        velocity_history,
        acceleration_history,
        clearance_history,
        yaw_history,
        final_state,
    )


def _candidate_responses(batch: SyntheticBatch, config: EEFConfig) -> Tensor:
    """Generate deterministic, smooth response candidates in the local frame."""

    device = batch.position.device
    direction = F.normalize(batch.target - batch.position, dim=-1)
    angles = torch.linspace(-1.05, 1.05, config.candidate_angles, device=device)
    vertical = torch.linspace(-0.42, 0.42, config.candidate_vertical, device=device)
    aa, zz = torch.meshgrid(angles, vertical, indexing="ij")
    aa = aa.flatten()
    zz = zz.flatten()
    k = aa.numel()
    c, s = aa.cos()[None, :], aa.sin()[None, :]
    base_x = direction[:, 0:1]
    base_y = direction[:, 1:2]
    candidates = torch.zeros((direction.shape[0], k, 3), device=device)
    candidates[..., 0] = base_x * c - base_y * s
    candidates[..., 1] = base_x * s + base_y * c
    candidates[..., 2] = direction[:, 2:3] + zz[None, :]
    candidates = F.normalize(candidates, dim=-1)
    speeds = batch.target_speed[:, None, :] * (0.92 - 0.10 * aa.abs()[None, :, None])
    desired = candidates * speeds
    phase = torch.linspace(0.60, 1.0, config.response_horizon, device=device)[None, None, :, None]
    velocity = desired[:, :, None, :] * phase
    desired_yaw = torch.atan2(candidates[..., 1], candidates[..., 0])
    yaw_rate = (desired_yaw / max(config.response_horizon * config.dt, 1.0e-6)).clamp(
        -config.max_yaw_rate, config.max_yaw_rate
    )
    yaw_sequence = yaw_rate[:, :, None, None].expand(-1, -1, config.response_horizon, 1)
    responses = torch.cat((velocity, yaw_sequence), dim=-1)
    stop = torch.zeros((direction.shape[0], 1, config.response_horizon, 4), device=device)
    return torch.cat((responses, stop), dim=1)


def privileged_target(batch: SyntheticBatch, config: EEFConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Choose a collision-screened target using exact training-only geometry."""

    candidates = _candidate_responses(batch, config)
    b, k, h, _ = candidates.shape
    repeat = lambda x: x[:, None].expand((b, k) + x.shape[1:]).reshape((b * k,) + x.shape[1:])
    embodied = config.teacher_embodiment
    rollout = differentiable_rollout(
        repeat(batch.position),
        repeat(batch.velocity),
        candidates.reshape(b * k, h, 4),
        repeat(batch.obstacles),
        repeat(batch.radii),
        config,
        wind=repeat(batch.wind) if embodied else None,
        response_scale=repeat(batch.response_scale) if embodied else None,
        delay_steps=repeat(batch.delay_steps) if embodied else None,
        drag_scale=repeat(batch.drag_scale) if embodied else None,
        decay_gradients=False,
    )
    target = repeat(batch.target)
    initial = repeat(batch.position)
    initial_distance = (target - initial).norm(dim=-1).clamp_min(0.2)
    final_distance = (target - rollout.positions[:, -1]).norm(dim=-1)
    progress = final_distance / initial_distance
    clearance_penalty = F.softplus((config.safety_margin - rollout.clearance) * 18.0).mean(dim=1) / 18.0
    target_direction = F.normalize(target - initial, dim=-1)
    desired_terminal = target_direction * repeat(batch.target_speed)
    exit_error = (rollout.velocities[:, -1] - desired_terminal).square().sum(dim=-1)
    desired_heading = torch.atan2(target_direction[:, 1], target_direction[:, 0])
    heading_error = torch.atan2(
        torch.sin(rollout.yaws[:, -1] - desired_heading),
        torch.cos(rollout.yaws[:, -1] - desired_heading),
    ).square()
    acceleration = rollout.accelerations.square().sum(dim=-1).mean(dim=1)
    jerk = torch.diff(rollout.accelerations, dim=1).square().sum(dim=-1).mean(dim=1)
    cost = (
        config.lambda_progress * progress
        + config.lambda_clearance * clearance_penalty
        + config.lambda_exit * (exit_error + 0.25 * heading_error)
        + config.lambda_acceleration * acceleration
        + config.lambda_jerk * jerk
    ).reshape(b, k)
    feasible = (rollout.clearance.amin(dim=1) > 0.0).reshape(b, k)
    if config.use_privileged_filter:
        cost = torch.where(feasible, cost, cost + 1.0e3)
    selected = cost.argmin(dim=1)
    row = torch.arange(b, device=batch.position.device)
    target_response = candidates[row, selected]
    selected_rollout = differentiable_rollout(
        batch.position,
        batch.velocity,
        target_response,
        batch.obstacles,
        batch.radii,
        config,
        wind=batch.wind if embodied else None,
        response_scale=batch.response_scale if embodied else None,
        delay_steps=batch.delay_steps if embodied else None,
        drag_scale=batch.drag_scale if embodied else None,
        decay_gradients=False,
    )
    return target_response, selected_rollout.final_state, feasible.any(dim=1)


class EEFTrainer:
    """Deterministic CPU/GPU trainer with append-only scalar logs."""

    def __init__(self, config: EEFConfig, *, seed: int = 7, device: str = "cpu"):
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.model = EEFPolicy(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.control_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("outcome_head.")
        ]
        self.outcome_parameters = list(self.model.outcome_head.parameters())
        self.iteration = 0

    def train_step(self, batch_size: int) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        batch = _sample_batch(cfg, batch_size, self.generator, self.device)
        with torch.no_grad():
            target_response, target_exit, feasible = privileged_target(batch, cfg)
        features = _features(batch, cfg)
        random_hidden = (torch.rand(
            (batch_size, cfg.hidden_dim), generator=self.generator, device=self.device
        ) - 0.5) * 0.40
        response, exit_mean, exit_scale_tril, _ = self.model(features, random_hidden)
        rollout = differentiable_rollout(
            batch.position,
            batch.velocity,
            response,
            batch.obstacles,
            batch.radii,
            cfg,
            wind=batch.wind if cfg.differentiable_embodiment else None,
            response_scale=batch.response_scale if cfg.differentiable_embodiment else None,
            delay_steps=batch.delay_steps if cfg.differentiable_embodiment else None,
            drag_scale=batch.drag_scale if cfg.differentiable_embodiment else None,
            decay_gradients=cfg.decay_gradients,
        )
        # The observed terminal state always comes from the randomized execution
        # model, including in the imitation-only comparator. This keeps the
        # alignment metric independent of the training intervention.
        executed = differentiable_rollout(
            batch.position,
            batch.velocity,
            response,
            batch.obstacles,
            batch.radii,
            cfg,
            wind=batch.wind,
            response_scale=batch.response_scale,
            delay_steps=batch.delay_steps,
            drag_scale=batch.drag_scale,
            decay_gradients=False,
        )
        imitation = F.smooth_l1_loss(response, target_response)
        initial_distance = (batch.target - batch.position).norm(dim=-1).clamp_min(0.2)
        final_distance = (batch.target - rollout.positions[:, -1]).norm(dim=-1)
        progress = (final_distance / initial_distance).mean()
        clearance = F.softplus((cfg.safety_margin - rollout.clearance) * 18.0).mean() / 18.0
        target_direction = F.normalize(batch.target - batch.position, dim=-1)
        desired_terminal = target_direction * batch.target_speed
        desired_travel = torch.minimum(
            initial_distance * 0.80,
            batch.target_speed[:, 0] * cfg.response_horizon * cfg.dt * 0.72,
        )
        desired_displacement = target_direction * desired_travel[:, None]
        desired_heading = torch.atan2(target_direction[:, 1], target_direction[:, 0])
        heading_residual = torch.atan2(
            torch.sin(rollout.yaws[:, -1] - desired_heading),
            torch.cos(rollout.yaws[:, -1] - desired_heading),
        )
        exit_control = F.smooth_l1_loss(
            rollout.final_state[:, :3], desired_displacement
        ) + F.smooth_l1_loss(rollout.velocities[:, -1], desired_terminal) + 0.25 * F.smooth_l1_loss(
            heading_residual, torch.zeros_like(heading_residual)
        )
        acceleration = rollout.accelerations.square().sum(dim=-1).mean()
        jerk = torch.diff(rollout.accelerations, dim=1).square().sum(dim=-1).mean()
        dynamics = (
            cfg.lambda_progress * progress
            + cfg.lambda_clearance * clearance
            + cfg.lambda_exit * exit_control
            + cfg.lambda_acceleration * acceleration
            + cfg.lambda_jerk * jerk
        )
        observed_exit = executed.final_state.detach()
        whitened = torch.linalg.solve_triangular(
            exit_scale_tril,
            (observed_exit - exit_mean).unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        log_determinant = torch.log(
            torch.diagonal(exit_scale_tril, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        nll = (0.5 * whitened.square().sum(dim=-1) + log_determinant).mean() / cfg.outcome_dim
        loss = cfg.lambda_imitation * imitation
        if cfg.use_differentiable_objective:
            loss = loss + cfg.lambda_dynamics * dynamics
        if cfg.use_outcome_supervision:
            loss = loss + cfg.lambda_nll * nll
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Separate clipping preserves the stop-gradient intervention: learning
        # the uncertainty head cannot rescale the controller update.
        control_gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.control_parameters, 10.0))
        outcome_gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.outcome_parameters, 10.0))
        self.optimizer.step()
        self.iteration += 1
        return {
            "iteration": float(self.iteration),
            "loss": float(loss.detach()),
            "imitation": float(imitation.detach()),
            "dynamics": float(dynamics.detach()),
            "nll": float(nll.detach()),
            "progress_ratio": float(progress.detach()),
            "clearance_loss": float(clearance.detach()),
            "exit_velocity_loss": float(exit_control.detach()),
            "acceleration_loss": float(acceleration.detach()),
            "jerk_loss": float(jerk.detach()),
            "teacher_feasible_fraction": float(feasible.float().mean()),
            "execution_alignment_error": float(
                (executed.final_state - target_exit).norm(dim=-1).mean().detach()
            ),
            "registered_exit_error": float(
                (
                    executed.final_state
                    - torch.cat((desired_displacement, desired_terminal, desired_heading[:, None]), dim=-1)
                ).norm(dim=-1).mean().detach()
            ),
            "outcome_prediction_error": float(
                (executed.final_state - exit_mean).norm(dim=-1).mean().detach()
            ),
            "gradient_norm": control_gradient_norm,
            "outcome_gradient_norm": outcome_gradient_norm,
        }

    def fit(
        self,
        iterations: int,
        batch_size: int,
        output_dir: str | Path,
        *,
        log_every: int = 1,
    ) -> tuple[Path, list[dict[str, float]]]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        jsonl_path = target / "train_metrics.jsonl"
        csv_path = target / "train_metrics.csv"
        rows: list[dict[str, float]] = []
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for _ in range(iterations):
                metrics = self.train_step(batch_size)
                rows.append(metrics)
                if self.iteration % log_every == 0 or self.iteration == iterations:
                    jsonl.write(json.dumps(metrics, sort_keys=True) + "\n")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = self.save_checkpoint(target / "eef_policy.pt")
        return checkpoint, rows

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        immutable_hash = model_sha256(self.model, self.config)
        torch.save(
            {
                "schema_version": 1,
                "model": self.model.state_dict(),
                "config": asdict(self.config),
                "seed": self.seed,
                "iteration": self.iteration,
                "immutable_sha256": immutable_hash,
            },
            target,
        )
        target.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": self.seed,
                    "iteration": self.iteration,
                    "immutable_sha256": immutable_hash,
                    "config": asdict(self.config),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target


def model_sha256(model: nn.Module, config: EEFConfig) -> str:
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    source_root = Path(__file__).parent
    deployment_sources = [
        source_root / name
        for name in (
            "eef.py",
            "experience.py",
            "individual_protocol.py",
            "memory.py",
            "protocol.py",
            "rollout.py",
        )
    ] + [source_root.parent / name for name in ("config.py", "env.py", "scenarios.py")]
    for source in deployment_sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def load_eef_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    allow_source_rebind: bool = False,
) -> tuple[EEFPolicy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = EEFConfig(**payload["config"])
    model = EEFPolicy(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    observed = model_sha256(model, config)
    if observed != payload["immutable_sha256"] and not allow_source_rebind:
        raise ValueError("EEF checkpoint immutable hash mismatch")
    if observed != payload["immutable_sha256"]:
        payload = dict(payload)
        payload["source_checkpoint_immutable_sha256"] = payload["immutable_sha256"]
        payload["immutable_sha256"] = observed
        payload["source_rebound"] = True
    return model, payload


class EEFNavigator:
    """Stateful deployment wrapper that converts local policy output to world actions."""

    def __init__(self, model: EEFPolicy, *, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.hidden: Tensor | None = None
        self.last_exit_mean: np.ndarray | None = None
        self.last_exit_std: np.ndarray | None = None

    def reset(self, num_drones: int) -> None:
        self.hidden = torch.zeros((num_drones, self.model.config.hidden_dim), device=self.device)

    @staticmethod
    def _world_to_body(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        result = vector.copy()
        result[:, 0] = c * vector[:, 0] + s * vector[:, 1]
        result[:, 1] = -s * vector[:, 0] + c * vector[:, 1]
        return result

    def act(
        self,
        observation: dict[str, np.ndarray],
        positions: np.ndarray,
        local_targets: np.ndarray,
        *,
        modes: np.ndarray | None = None,
        desired_speeds: np.ndarray | float = 1.4,
        desired_terminal_velocities: np.ndarray | None = None,
        desired_terminal_yaws: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.model.config
        count = positions.shape[0]
        if self.hidden is None or self.hidden.shape[0] != count:
            self.reset(count)
        yaw = observation["self"][:, 5].astype(np.float64)
        local_goal = self._world_to_body(local_targets - positions, yaw)
        rays = observation["rays"]
        if rays.shape[1] != cfg.ray_count:
            raise ValueError(f"policy expects {cfg.ray_count} rays, received {rays.shape[1]}")
        proximity = 1.0 - rays
        velocity = observation["self"][:, :3] / cfg.max_speed
        goal_feature = np.clip(local_goal / cfg.ray_range, -1.0, 1.0)
        peer = np.zeros((count, 6), dtype=np.float32)
        if observation["neighbors"].shape[1]:
            peer[:, :3] = observation["neighbors"][:, 0, :3] / cfg.ray_range
            peer[:, 3:] = observation["neighbors"][:, 0, 3:] / cfg.max_speed
        if modes is None:
            modes = np.zeros(count, dtype=np.int64)
        mode_features = np.eye(4, dtype=np.float32)[np.asarray(modes, dtype=np.int64)]
        speeds = np.broadcast_to(np.asarray(desired_speeds, dtype=np.float32), (count,)).reshape(-1, 1)
        features = np.concatenate(
            (proximity, velocity, goal_feature, peer, mode_features, speeds / cfg.max_speed), axis=1
        ).astype(np.float32)
        with torch.no_grad():
            response, mean, scale_tril, hidden_next = self.model(
                torch.from_numpy(features).to(self.device), self.hidden
            )
            self.hidden = hidden_next * cfg.recurrent_retention
        body_action = response[:, 0].cpu().numpy()
        goal_norm = np.linalg.norm(local_goal, axis=1, keepdims=True)
        goal_velocity = local_goal / np.maximum(goal_norm, 1.0e-8) * speeds
        terminal_available = np.zeros(count, dtype=bool)
        if desired_terminal_velocities is not None:
            world_terminal = np.asarray(desired_terminal_velocities, dtype=np.float64)
            if world_terminal.shape != (count, 3):
                raise ValueError("desired_terminal_velocities must have shape (num_drones, 3)")
            terminal_available = np.all(np.isfinite(world_terminal), axis=1)
            if np.any(terminal_available):
                terminal_body = self._world_to_body(world_terminal.copy(), yaw)
                terminal_norm = np.linalg.norm(terminal_body, axis=1, keepdims=True)
                terminal_direction = terminal_body / np.maximum(terminal_norm, 1.0e-8)
                along_track = np.sum(local_goal * terminal_direction, axis=1, keepdims=True)
                cross_track = local_goal - along_track * terminal_direction
                stationary = terminal_norm[:, 0] < 1.0e-4
                cross_track[stationary] = local_goal[stationary]
                interface_velocity = terminal_body + cfg.interface_position_gain * cross_track
                interface_norm = np.linalg.norm(interface_velocity, axis=1, keepdims=True)
                interface_velocity *= np.minimum(
                    1.0, cfg.max_speed / np.maximum(interface_norm, 1.0e-8)
                )
                goal_velocity[terminal_available] = interface_velocity[terminal_available]
        if cfg.use_outcome_supervision and cfg.outcome_feedback_gain > 0.0:
            predicted_terminal_velocity = mean[:, 3:6].cpu().numpy()
            body_action[:, :3] += cfg.outcome_feedback_gain * (
                goal_velocity - predicted_terminal_velocity
            )
        feedback = np.full((count, 1), cfg.terminal_feedback, dtype=np.float64)
        feedback[terminal_available] = cfg.interface_terminal_feedback
        body_action[:, :3] = (1.0 - feedback) * body_action[:, :3] + feedback * goal_velocity
        if desired_terminal_yaws is not None:
            target_yaw = np.asarray(desired_terminal_yaws, dtype=np.float64)
            if target_yaw.shape != (count,):
                raise ValueError("desired_terminal_yaws must have shape (num_drones,)")
            yaw_available = np.isfinite(target_yaw)
            yaw_error = (target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
            body_action[yaw_available, 3] = np.clip(
                cfg.interface_yaw_gain * yaw_error[yaw_available],
                -cfg.max_yaw_rate,
                cfg.max_yaw_rate,
            )
        interface_action = body_action.copy()
        body_action = apply_clearance_shield(body_action, proximity, cfg)
        if np.any(terminal_available):
            observed_clearance = rays.min(axis=1) * cfg.ray_range
            interface_safe = terminal_available & (
                observed_clearance >= cfg.interface_override_clearance
            )
            body_action[interface_safe, :3] = interface_action[interface_safe, :3]
        world_action = body_action.copy()
        c, s = np.cos(yaw), np.sin(yaw)
        world_action[:, 0] = c * body_action[:, 0] - s * body_action[:, 1]
        world_action[:, 1] = s * body_action[:, 0] + c * body_action[:, 1]
        self.last_exit_mean = mean.cpu().numpy()
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        self.last_exit_std = torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt().cpu().numpy()
        return world_action.astype(np.float32)


def apply_clearance_shield(body_action: np.ndarray, proximity: np.ndarray, config: EEFConfig) -> np.ndarray:
    """Apply the fixed deployment shield shared by every policy comparator."""

    protected = np.asarray(body_action, dtype=np.float64).copy()
    angles = np.linspace(-np.pi, np.pi, config.ray_count, endpoint=False)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    close_weight = np.square(np.clip(proximity - 0.58, 0.0, None))
    repulsion = -(close_weight[..., None] * directions[None, :, :]).sum(axis=1)
    repulsion_norm = np.linalg.norm(repulsion, axis=1, keepdims=True)
    repulsion = repulsion / np.maximum(repulsion_norm, 1.0)
    protected[:, :2] += 1.15 * repulsion
    protected[:, 2] = np.clip(protected[:, 2], -0.85, 0.85)
    velocity_norm = np.linalg.norm(protected[:, :3], axis=1, keepdims=True)
    protected[:, :3] *= np.minimum(
        1.0, config.max_speed / np.maximum(velocity_norm, 1.0e-8)
    )
    return protected.astype(np.float32)
