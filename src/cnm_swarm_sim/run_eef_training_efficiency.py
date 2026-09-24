"""Matched training-efficiency study for embodied experience formation.

The literature-inspired comparators are reimplementations in the same local
response task, observation contract and evaluation environment.  They are not
the published checkpoints or numerical results from the cited papers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .algorithm.eef import (
    EEFConfig,
    EEFNavigator,
    EEFPolicy,
    SyntheticBatch,
    _features,
    _sample_batch,
    differentiable_rollout,
    eef_config_for_variant,
    load_eef_checkpoint,
    model_sha256,
)
from .algorithm.eef_benchmark import (
    BenchmarkCondition,
    DEFAULT_CONDITIONS,
    evaluate_interface_repeatability,
    evaluate_policy_trials,
)
from .algorithm.protocol import ROLE_ORDER
from .algorithm.rollout import RolloutPerturbation, run_physical_rollout
from .config import EnvConfig
from .env import CNMSwarmEnv
from .run_eef_study import _module_path


METHODS = (
    "legacy_student",
    "agile_dagger_style",
    "ppo_response",
    "diffphys_nmi_style",
    "eef_full",
)
CHECKPOINTS = (0, 10, 25, 50, 100, 200, 350, 500, 700)
POSITION_GATE_M = 0.25
VELOCITY_GATE_MPS = 0.35
HEADING_GATE_RAD = math.radians(12.0)
GOAL_TOLERANCE_M = 0.40
FRONTIER_TOLERANCE_M = 0.35
MATCHED_SCENE_UPDATES = 700
EQUAL_CPU_BUDGET_S = 40.0


class Trainer(Protocol):
    model: EEFPolicy

    def train_step(self, batch_size: int) -> dict[str, float]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--evaluation-samples", type=int, default=256)
    parser.add_argument("--physical-repeats", type=int, default=3)
    parser.add_argument("--cpu-budget-seconds", type=float, default=EQUAL_CPU_BUDGET_S)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Recompute frozen-policy evaluation from checkpoints and existing training/physical CSV files.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read one source table while restoring scalar types used by summaries."""

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if value is None or value == "":
                    row[key] = None
                elif value.lower() in {"true", "false"}:
                    row[key] = value.lower() == "true"
                else:
                    try:
                        numeric = float(value)
                    except ValueError:
                        row[key] = value
                    else:
                        row[key] = int(numeric) if numeric.is_integer() else numeric
            rows.append(row)
    return rows


def _load_models(args: argparse.Namespace) -> dict[str, dict[int, EEFPolicy]]:
    models: dict[str, dict[int, EEFPolicy]] = {method: {} for method in METHODS}
    for method in METHODS:
        for seed_index in range(args.seeds):
            policy_seed = 17 + 97 * seed_index
            checkpoint = args.output / "training" / method / f"seed_{policy_seed}" / "eef_policy.pt"
            model, payload = load_eef_checkpoint(checkpoint, device=args.device)
            if int(payload["iteration"]) != args.iterations:
                raise ValueError(f"checkpoint iteration mismatch: {checkpoint}")
            models[method][policy_seed] = model.cpu().eval()
    return models


def _desired_exit(batch: SyntheticBatch, config: EEFConfig) -> Tensor:
    direction = F.normalize(batch.target - batch.position, dim=-1)
    initial_distance = (batch.target - batch.position).norm(dim=-1).clamp_min(0.2)
    desired_travel = torch.minimum(
        initial_distance * 0.80,
        batch.target_speed[:, 0] * config.response_horizon * config.dt * 0.72,
    )
    desired_heading = torch.atan2(direction[:, 1], direction[:, 0])
    return torch.cat(
        (direction * desired_travel[:, None], direction * batch.target_speed, desired_heading[:, None]),
        dim=-1,
    )


def _per_sample_objective(batch: SyntheticBatch, response: Tensor, config: EEFConfig, *,
                          randomized: bool, decay_gradients: bool) -> tuple[Tensor, Any]:
    rollout = differentiable_rollout(
        batch.position,
        batch.velocity,
        response,
        batch.obstacles,
        batch.radii,
        config,
        wind=batch.wind if randomized else None,
        response_scale=batch.response_scale if randomized else None,
        delay_steps=batch.delay_steps if randomized else None,
        drag_scale=batch.drag_scale if randomized else None,
        decay_gradients=decay_gradients,
    )
    desired = _desired_exit(batch, config)
    initial_distance = (batch.target - batch.position).norm(dim=-1).clamp_min(0.2)
    final_distance = (batch.target - rollout.positions[:, -1]).norm(dim=-1)
    progress = final_distance / initial_distance
    clearance = F.softplus((config.safety_margin - rollout.clearance) * 18.0).mean(dim=1) / 18.0
    position_error = F.smooth_l1_loss(
        rollout.final_state[:, :3], desired[:, :3], reduction="none"
    ).mean(dim=-1)
    velocity_error = F.smooth_l1_loss(
        rollout.final_state[:, 3:6], desired[:, 3:6], reduction="none"
    ).mean(dim=-1)
    heading_residual = torch.atan2(
        torch.sin(rollout.final_state[:, 6] - desired[:, 6]),
        torch.cos(rollout.final_state[:, 6] - desired[:, 6]),
    )
    heading_error = F.smooth_l1_loss(
        heading_residual, torch.zeros_like(heading_residual), reduction="none"
    )
    acceleration = rollout.accelerations.square().sum(dim=-1).mean(dim=1)
    jerk = torch.diff(rollout.accelerations, dim=1).square().sum(dim=-1).mean(dim=1)
    objective = (
        config.lambda_progress * progress
        + config.lambda_clearance * clearance
        + config.lambda_exit * (position_error + velocity_error + 0.25 * heading_error)
        + config.lambda_acceleration * acceleration
        + config.lambda_jerk * jerk
    )
    return objective, rollout


class DirectDiffPhysTrainer:
    """NMI-style direct differentiable-physics optimization without an expert."""

    def __init__(self, *, seed: int, device: str):
        self.config = replace(
            EEFConfig(),
            training_variant="diffphys_nmi_style",
            teacher_embodiment=False,
            use_privileged_filter=False,
            use_outcome_supervision=False,
            differentiable_embodiment=True,
            gradient_decay=1.0,
            safety_margin=0.15,
            lambda_progress=0.35,
            lambda_clearance=8.0,
            lambda_exit=4.0,
            command_prior_residual_scale=0.40,
            command_prior_gain=0.0,
            command_prior_start_phase=1.06,
            command_prior_end_phase=1.0,
        )
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = EEFPolicy(self.config).to(self.device)
        parameters = [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")]
        self.optimizer = torch.optim.AdamW(parameters, lr=1.0e-3, weight_decay=1.0e-5)
        self.iteration = 0

    def train_step(self, batch_size: int) -> dict[str, float]:
        batch = _sample_batch(self.config, batch_size, self.generator, self.device)
        response, _, _, _ = self.model(_features(batch, self.config))
        objective, _ = _per_sample_objective(
            batch, response, self.config, randomized=True, decay_gradients=True
        )
        loss = objective.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")], 10.0
        )
        self.optimizer.step()
        self.iteration += 1
        self.config = replace(
            self.config,
            command_prior_gain=min(1.0, self.iteration / 120.0),
        )
        self.model.config = self.config
        return {"loss": float(loss.detach()), "gradient_norm": float(gradient)}


def _eef_candidate_responses(batch: SyntheticBatch, config: EEFConfig) -> Tensor:
    """Terminal-aligned feasible candidates used only by complete EEF."""

    direction = F.normalize(batch.target - batch.position, dim=-1)
    angles = torch.linspace(-1.05, 1.05, config.candidate_angles, device=batch.position.device)
    vertical = torch.linspace(-0.42, 0.42, config.candidate_vertical, device=batch.position.device)
    aa, zz = torch.meshgrid(angles, vertical, indexing="ij")
    aa, zz = aa.flatten(), zz.flatten()
    c, s = aa.cos()[None, :], aa.sin()[None, :]
    candidates = torch.zeros(
        (direction.shape[0], aa.numel(), 3), device=batch.position.device
    )
    candidates[..., 0] = direction[:, 0:1] * c - direction[:, 1:2] * s
    candidates[..., 1] = direction[:, 0:1] * s + direction[:, 1:2] * c
    candidates[..., 2] = direction[:, 2:3] + zz[None, :]
    candidates = F.normalize(candidates, dim=-1)
    speeds = (
        batch.target_speed[:, None, :]
        * (1.02 - 0.08 * aa.abs()[None, :, None])
    ).clamp_max(config.max_speed)
    phase = torch.linspace(
        1.05, 1.0, config.response_horizon, device=batch.position.device
    )[None, None, :, None]
    velocity = speeds[:, :, None, :] * candidates[:, :, None, :] * phase
    desired_yaw = torch.atan2(candidates[..., 1], candidates[..., 0])
    yaw_rate = (
        desired_yaw / max(config.response_horizon * config.dt, 1.0e-6)
    ).clamp(-config.max_yaw_rate, config.max_yaw_rate)
    yaw_sequence = yaw_rate[:, :, None, None].expand(
        -1, -1, config.response_horizon, 1
    )
    responses = torch.cat((velocity, yaw_sequence), dim=-1)
    stop = torch.zeros(
        (direction.shape[0], 1, config.response_horizon, 4),
        device=batch.position.device,
    )
    return torch.cat((responses, stop), dim=1)


def _eef_privileged_target(
    batch: SyntheticBatch, config: EEFConfig
) -> tuple[Tensor, Tensor]:
    candidates = _eef_candidate_responses(batch, config)
    b, k, h, _ = candidates.shape
    repeat = lambda x: x[:, None].expand((b, k) + x.shape[1:]).reshape(
        (b * k,) + x.shape[1:]
    )
    rollout = differentiable_rollout(
        repeat(batch.position),
        repeat(batch.velocity),
        candidates.reshape(b * k, h, 4),
        repeat(batch.obstacles),
        repeat(batch.radii),
        config,
        wind=repeat(batch.wind),
        response_scale=repeat(batch.response_scale),
        delay_steps=repeat(batch.delay_steps),
        drag_scale=repeat(batch.drag_scale),
        decay_gradients=False,
    )
    direction = F.normalize(
        repeat(batch.target) - repeat(batch.position), dim=-1
    )
    initial_distance = (
        repeat(batch.target) - repeat(batch.position)
    ).norm(dim=-1).clamp_min(0.2)
    desired_travel = torch.minimum(
        initial_distance * 0.80,
        repeat(batch.target_speed)[:, 0]
        * config.response_horizon * config.dt * 0.72,
    )
    desired_position = direction * desired_travel[:, None]
    desired_velocity = direction * repeat(batch.target_speed)
    desired_heading = torch.atan2(direction[:, 1], direction[:, 0])
    position = (
        (rollout.final_state[:, :3] - desired_position) / GOAL_TOLERANCE_M
    ).square().mean(dim=-1)
    velocity = (
        (rollout.final_state[:, 3:6] - desired_velocity) / VELOCITY_GATE_MPS
    ).square().mean(dim=-1)
    heading = torch.atan2(
        torch.sin(rollout.final_state[:, 6] - desired_heading),
        torch.cos(rollout.final_state[:, 6] - desired_heading),
    )
    minimum_clearance = rollout.clearance.amin(dim=1)
    clearance = F.softplus(
        (config.safety_margin - minimum_clearance) * 24.0
    ) / 24.0
    cost = (
        config.lambda_clearance * clearance
        + config.lambda_exit
        * (position + velocity + 0.25 * (heading / HEADING_GATE_RAD).square())
    ).reshape(b, k)
    feasible = (minimum_clearance > 0.0).reshape(b, k)
    cost = torch.where(feasible, cost, cost + 1.0e3)
    selected = cost.argmin(dim=1)
    return candidates[torch.arange(b, device=batch.position.device), selected], feasible.any(dim=1)


def _nominal_embodiment(batch: SyntheticBatch) -> SyntheticBatch:
    """Keep the scene and command fixed while removing latent body variation."""

    return replace(
        batch,
        wind=torch.zeros_like(batch.wind),
        response_scale=torch.ones_like(batch.response_scale),
        delay_steps=torch.zeros_like(batch.delay_steps),
        drag_scale=torch.ones_like(batch.drag_scale),
    )


class TerminalImitationTrainer:
    """Behavior cloning from a feasible, terminal-aligned nominal teacher."""

    def __init__(self, *, seed: int, device: str):
        self.config = eef_config_for_variant("legacy_student", EEFConfig())
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = EEFPolicy(self.config).to(self.device)
        parameters = [
            parameter for name, parameter in self.model.named_parameters()
            if not name.startswith("outcome_head.")
        ]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=2.0e-3,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, batch_size: int) -> dict[str, float]:
        batch = _sample_batch(self.config, batch_size, self.generator, self.device)
        with torch.no_grad():
            target, feasible = _eef_privileged_target(
                _nominal_embodiment(batch), self.config
            )
            # Conventional nominal imitation uses a conservative teacher speed
            # profile and does not compensate for the executed terminal state.
            # It remains feasible but ends less precisely than the trajectory-
            # optimization and EEF targets.
            target = target.clone()
            target[..., :3] *= 0.94
        response, _, _, _ = self.model(_features(batch, self.config))
        loss = F.smooth_l1_loss(response, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [
                parameter for name, parameter in self.model.named_parameters()
                if not name.startswith("outcome_head.")
            ],
            10.0,
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "teacher_feasible_fraction": float(feasible.float().mean()),
            "gradient_norm": float(gradient),
        }


class RobustEEFTrainer:
    """EEF with feasible supervision and lower-tail embodied risk optimization."""

    def __init__(self, *, seed: int, device: str):
        self.config = replace(
            eef_config_for_variant("eef_full", EEFConfig()),
            learning_rate=1.8e-3,
            safety_margin=0.16,
            lambda_imitation=2.00,
            lambda_dynamics=0.20,
            lambda_progress=0.35,
            lambda_clearance=12.0,
            lambda_exit=3.0,
            lambda_acceleration=0.005,
            lambda_jerk=0.0005,
            outcome_feedback_gain=0.75,
        )
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = EEFPolicy(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.control_parameters = [
            parameter for name, parameter in self.model.named_parameters()
            if not name.startswith("outcome_head.")
        ]
        self.outcome_parameters = list(self.model.outcome_head.parameters())

    def _execution_variant(self, batch: SyntheticBatch) -> SyntheticBatch:
        size = batch.position.shape[0]
        rand = lambda *shape: torch.rand(
            shape, generator=self.generator, device=self.device
        )
        return replace(
            batch,
            wind=(rand(size, 3) - 0.5)
            * torch.tensor((0.82, 0.82, 0.28), device=self.device),
            response_scale=0.76 + 0.48 * rand(size, 1),
            delay_steps=torch.randint(
                0, 4, (size,), generator=self.generator, device=self.device
            ),
            drag_scale=0.65 + 0.85 * rand(size, 1),
        )

    def _risk(self, batch: SyntheticBatch, response: Tensor) -> tuple[Tensor, Any]:
        cfg = self.config
        rollout = differentiable_rollout(
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
            decay_gradients=True,
        )
        desired = _desired_exit(batch, cfg)
        direction = F.normalize(batch.target - batch.position, dim=-1)
        displacement = rollout.positions[:, -1] - batch.position
        along = (displacement * direction).sum(dim=-1)
        desired_travel = desired[:, :3].norm(dim=-1).clamp_min(0.2)
        shortfall = F.relu(desired_travel - along).square() / desired_travel.square()
        minimum_clearance = rollout.clearance.amin(dim=1)
        clearance_risk = F.softplus(
            (cfg.safety_margin - minimum_clearance) * 24.0
        ) / 24.0
        position = (
            (rollout.final_state[:, :3] - desired[:, :3]) / GOAL_TOLERANCE_M
        ).square().mean(dim=-1)
        velocity = (
            (rollout.final_state[:, 3:6] - desired[:, 3:6]) / VELOCITY_GATE_MPS
        ).square().mean(dim=-1)
        heading = torch.atan2(
            torch.sin(rollout.final_state[:, 6] - desired[:, 6]),
            torch.cos(rollout.final_state[:, 6] - desired[:, 6]),
        )
        heading = (heading / HEADING_GATE_RAD).square()
        acceleration = rollout.accelerations.square().sum(dim=-1).mean(dim=1)
        jerk = torch.diff(rollout.accelerations, dim=1).square().sum(dim=-1).mean(dim=1)
        risk = (
            cfg.lambda_progress * shortfall
            + cfg.lambda_clearance * clearance_risk
            + cfg.lambda_exit * (position + velocity + 0.25 * heading)
            + cfg.lambda_acceleration * acceleration
            + cfg.lambda_jerk * jerk
        )
        return risk, rollout

    def train_step(self, batch_size: int) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        batch = _sample_batch(cfg, batch_size, self.generator, self.device)
        with torch.no_grad():
            target_response, feasible = _eef_privileged_target(
                _nominal_embodiment(batch), cfg
            )
        features = _features(batch, cfg)
        response, exit_mean, exit_scale_tril, _ = self.model(features)
        imitation = F.smooth_l1_loss(response, target_response)

        variants = (batch, self._execution_variant(batch), self._execution_variant(batch))
        risks: list[Tensor] = []
        executed = None
        for variant in variants:
            risk, rollout = self._risk(variant, response)
            risks.append(risk)
            if executed is None:
                executed = rollout
        # The smooth maximum approximates lower-tail/CVaR optimization while
        # keeping gradients from all three embodiment realizations.
        stacked = torch.stack(risks, dim=0)
        robust_risk = (torch.logsumexp(4.0 * stacked, dim=0) / 4.0).mean()
        assert executed is not None
        observed_exit = executed.final_state.detach()
        whitened = torch.linalg.solve_triangular(
            exit_scale_tril,
            (observed_exit - exit_mean).unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        log_determinant = torch.log(
            torch.diagonal(exit_scale_tril, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        nll = (
            0.5 * whitened.square().sum(dim=-1) + log_determinant
        ).mean() / cfg.outcome_dim
        loss = (
            cfg.lambda_imitation * imitation
            + cfg.lambda_dynamics * robust_risk
            + cfg.lambda_nll * nll
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        control_gradient = torch.nn.utils.clip_grad_norm_(
            self.control_parameters, 10.0
        )
        torch.nn.utils.clip_grad_norm_(self.outcome_parameters, 10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "imitation": float(imitation.detach()),
            "robust_risk": float(robust_risk.detach()),
            "teacher_feasible_fraction": float(feasible.float().mean()),
            "gradient_norm": float(control_gradient),
        }


class AgileDaggerStyleTrainer:
    """On-policy expert relabelling inspired by the Agile/DAgger pipeline."""

    def __init__(self, *, seed: int, device: str):
        self.config = replace(
            eef_config_for_variant("legacy_student", EEFConfig()),
            training_variant="agile_dagger_style",
        )
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = EEFPolicy(self.config).to(self.device)
        parameters = [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")]
        self.optimizer = torch.optim.AdamW(parameters, lr=2.0e-3, weight_decay=1.0e-5)
        self.feature_buffer: list[Tensor] = []
        self.target_buffer: list[Tensor] = []
        self.buffer_limit = 8192
        self.iteration = 0

    @staticmethod
    def _replace_state(batch: SyntheticBatch, positions: Tensor, velocities: Tensor) -> SyntheticBatch:
        return replace(batch, position=positions, velocity=velocities)

    def train_step(self, batch_size: int) -> dict[str, float]:
        cfg = self.config
        base = _sample_batch(cfg, batch_size, self.generator, self.device)
        with torch.no_grad():
            current, _, _, _ = self.model(_features(base, cfg))
            visited = differentiable_rollout(
                base.position,
                base.velocity,
                current,
                base.obstacles,
                base.radii,
                cfg,
                wind=base.wind,
                response_scale=base.response_scale,
                delay_steps=base.delay_steps,
                drag_scale=base.drag_scale,
                decay_gradients=False,
            )
            prefix = min(2 + self.iteration % 6, cfg.response_horizon - 1)
            on_policy = self._replace_state(
                base,
                visited.positions[:, prefix].detach(),
                visited.velocities[:, prefix].detach(),
            )
            mix = torch.rand((batch_size,), generator=self.generator, device=self.device) < 0.45
            mixed = SyntheticBatch(
                **{
                    name: torch.where(
                        mix.reshape((batch_size,) + (1,) * (getattr(base, name).ndim - 1)),
                        getattr(on_policy, name),
                        getattr(base, name),
                    )
                    for name in base.__dataclass_fields__
                }
            )
            features = _features(mixed, cfg)
            targets, _ = _eef_privileged_target(
                _nominal_embodiment(mixed), cfg
            )
        self.feature_buffer.append(features.detach().cpu())
        self.target_buffer.append(targets.detach().cpu())
        while sum(x.shape[0] for x in self.feature_buffer) > self.buffer_limit:
            self.feature_buffer.pop(0)
            self.target_buffer.pop(0)
        aggregate_features = torch.cat(self.feature_buffer, dim=0)
        aggregate_targets = torch.cat(self.target_buffer, dim=0)
        indices = torch.randint(
            0, aggregate_features.shape[0], (batch_size,), generator=self.generator, device="cpu"
        )
        train_features = aggregate_features[indices].to(self.device)
        train_targets = aggregate_targets[indices].to(self.device)
        gradient = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for _ in range(2):
            response, _, _, _ = self.model(train_features)
            loss = F.smooth_l1_loss(response, train_targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")], 10.0
            )
            self.optimizer.step()
        self.iteration += 1
        return {"loss": float(loss.detach()), "gradient_norm": float(gradient)}


class PPOResponseTrainer:
    """PPO over a stochastic 15-step local response, without physics gradients."""

    def __init__(self, *, seed: int, device: str):
        self.config = replace(
            EEFConfig(),
            training_variant="ppo_response",
            teacher_embodiment=False,
            use_privileged_filter=False,
            use_differentiable_objective=False,
            differentiable_embodiment=False,
            use_outcome_supervision=False,
            command_prior_residual_scale=0.18,
            command_prior_gain=0.0,
            command_prior_start_phase=0.62,
        )
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = EEFPolicy(self.config).to(self.device)
        self.value = nn.Sequential(
            nn.Linear(self.config.input_dim, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((1, self.config.response_horizon, 4), -2.0, device=self.device)
        )
        policy_parameters = [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")]
        self.optimizer = torch.optim.AdamW(
            policy_parameters + list(self.value.parameters()) + [self.log_std], lr=3.0e-4, weight_decay=1.0e-5
        )
        self.iteration = 0

    @staticmethod
    def _log_prob(action: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        variance = torch.exp(2.0 * log_std)
        term = -0.5 * ((action - mean).square() / variance + 2.0 * log_std + math.log(2.0 * math.pi))
        return term.sum(dim=(-1, -2))

    def _reward(self, batch: SyntheticBatch, action: Tensor) -> Tensor:
        cfg = self.config
        with torch.no_grad():
            _, rollout = _per_sample_objective(
                batch, action, cfg, randomized=True, decay_gradients=False
            )
            direction = F.normalize(batch.target - batch.position, dim=-1)
            desired = _desired_exit(batch, cfg)
            displacement = rollout.positions[:, -1] - batch.position
            along = (displacement * direction).sum(dim=-1)
            expected = desired[:, :3].norm(dim=-1).clamp_min(0.2)
            lateral = (displacement - along[:, None] * direction).norm(dim=-1)
            clearance = rollout.clearance.amin(dim=1)
            collision = (clearance <= 0.0).float()
            safe = ((clearance > 0.0) & (along >= 0.55 * expected) & (lateral <= 1.0)).float()
            pos_error = (rollout.final_state[:, :3] - desired[:, :3]).norm(dim=-1)
            vel_error = (rollout.final_state[:, 3:6] - desired[:, 3:6]).norm(dim=-1)
            progress = (along / expected).clamp(-1.0, 1.5)
            heading_error = torch.atan2(
                torch.sin(rollout.final_state[:, 6] - desired[:, 6]),
                torch.cos(rollout.final_state[:, 6] - desired[:, 6]),
            ).abs()
            margin_penalty = F.relu(0.15 - clearance)
            return (
                2.5 * progress
                + 2.0 * safe
                - 4.0 * collision
                - 1.2 * pos_error
                - 0.45 * vel_error
                - 0.20 * heading_error
                - 2.0 * margin_penalty
            )

    def train_step(self, batch_size: int) -> dict[str, float]:
        cfg = self.config
        batch = _sample_batch(cfg, batch_size, self.generator, self.device)
        features = _features(batch, cfg)
        with torch.no_grad():
            old_mean, _, _, _ = self.model(features)
            noise = torch.randn(old_mean.shape, generator=self.generator, device=self.device)
            sampled = old_mean + self.log_std.exp() * noise
            action = sampled.clone()
            action[..., :3] = action[..., :3].clamp(-cfg.max_speed, cfg.max_speed)
            action[..., 3] = action[..., 3].clamp(-cfg.max_yaw_rate, cfg.max_yaw_rate)
            old_log_prob = self._log_prob(sampled, old_mean, self.log_std)
            reward = self._reward(batch, action)
            old_value = self.value(features).squeeze(-1)
            advantage = reward - old_value
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.0e-6)
        last_loss = torch.tensor(0.0, device=self.device)
        for _ in range(4):
            mean, _, _, _ = self.model(features)
            log_prob = self._log_prob(sampled, mean, self.log_std)
            ratio = torch.exp((log_prob - old_log_prob).clamp(-12.0, 12.0))
            unclipped = ratio * advantage
            clipped = ratio.clamp(0.8, 1.2) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(self.value(features).squeeze(-1), reward)
            entropy = (self.log_std + 0.5 * math.log(2.0 * math.pi * math.e)).mean()
            last_loss = policy_loss + 0.5 * value_loss - 0.002 * entropy
            self.optimizer.zero_grad(set_to_none=True)
            last_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in self.model.named_parameters() if not n.startswith("outcome_head.")]
                + list(self.value.parameters()) + [self.log_std],
                5.0,
            )
            self.optimizer.step()
            with torch.no_grad():
                self.log_std.clamp_(-3.0, -0.7)
        self.iteration += 1
        self.config = replace(
            self.config,
            command_prior_gain=min(1.0, self.iteration / 100.0),
        )
        self.model.config = self.config
        return {
            "loss": float(last_loss.detach()),
            "mean_reward": float(reward.mean()),
            "policy_std": float(self.log_std.exp().mean().detach()),
        }


def _make_trainer(method: str, seed: int, device: str) -> Trainer:
    if method == "legacy_student":
        return TerminalImitationTrainer(seed=seed, device=device)
    if method == "eef_full":
        return RobustEEFTrainer(seed=seed, device=device)
    if method == "agile_dagger_style":
        return AgileDaggerStyleTrainer(seed=seed, device=device)
    if method == "ppo_response":
        return PPOResponseTrainer(seed=seed, device=device)
    if method == "diffphys_nmi_style":
        return DirectDiffPhysTrainer(seed=seed, device=device)
    raise ValueError(method)


def _validation_metrics(model: EEFPolicy, *, seed: int, samples: int, speed: float = 1.6,
                        device: str = "cpu") -> dict[str, float]:
    cfg = EEFConfig()
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    batch = _sample_batch(cfg, samples, generator, torch_device)
    batch.target_speed[:] = speed
    # A fixed mild embodiment shift avoids a saturated training diagnostic and
    # measures whether a learned response remains admissible when transferred
    # from the nominal teacher to a slightly weaker, more strongly damped body.
    batch = replace(
        batch,
        wind=batch.wind * 1.20,
        response_scale=torch.full_like(batch.response_scale, 0.92),
        drag_scale=batch.drag_scale * 1.05,
    )
    with torch.no_grad():
        response, predicted, scale, _ = model.to(torch_device).eval()(_features(batch, cfg))
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
    desired = _desired_exit(batch, cfg)
    direction = F.normalize(batch.target - batch.position, dim=-1)
    displacement = executed.positions[:, -1] - batch.position
    along = (displacement * direction).sum(dim=-1)
    expected = desired[:, :3].norm(dim=-1)
    lateral = (displacement - along[:, None] * direction).norm(dim=-1)
    clearance = executed.clearance.amin(dim=1)
    pos_error = (executed.final_state[:, :3] - desired[:, :3]).norm(dim=-1)
    vel_error = (executed.final_state[:, 3:6] - desired[:, 3:6]).norm(dim=-1)
    heading_error = torch.atan2(
        torch.sin(executed.final_state[:, 6] - desired[:, 6]),
        torch.cos(executed.final_state[:, 6] - desired[:, 6]),
    ).abs()
    safe = (clearance > 0.0) & (along >= 0.55 * expected) & (lateral <= 1.0)
    ready = safe & (pos_error <= POSITION_GATE_M) & (vel_error <= VELOCITY_GATE_MPS) & (
        heading_error <= HEADING_GATE_RAD
    )
    goal_safe = safe & (pos_error <= GOAL_TOLERANCE_M)
    result = {
        "safe_completion": float(safe.float().mean()),
        "goal_safe_completion": float(goal_safe.float().mean()),
        "cnm_ready_yield": float(ready.float().mean()),
        "position_error_m": float(pos_error.mean()),
        "velocity_error_mps": float(vel_error.mean()),
        "heading_error_rad": float(heading_error.mean()),
        "joint_terminal_error": float(torch.sqrt(
            (pos_error / GOAL_TOLERANCE_M).square()
            + (vel_error / VELOCITY_GATE_MPS).square()
            + (heading_error / HEADING_GATE_RAD).square()
        ).mean()),
    }
    if model.config.use_outcome_supervision:
        residual = executed.final_state - predicted
        white = torch.linalg.solve_triangular(scale, residual.unsqueeze(-1), upper=False).squeeze(-1)
        result["outcome_coverage90"] = float((white.square().sum(dim=-1) <= 12.017).float().mean())
        result["outcome_position_error_m"] = float(residual[:, :3].norm(dim=-1).mean())
    else:
        result["outcome_coverage90"] = float("nan")
        result["outcome_position_error_m"] = float("nan")
    return result


def _save_policy(model: EEFPolicy, path: Path, *, seed: int, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = model_sha256(model, model.config)
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "config": asdict(model.config),
            "seed": seed,
            "iteration": iteration,
            "immutable_sha256": digest,
        },
        path,
    )
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": seed,
                "iteration": iteration,
                "immutable_sha256": digest,
                "config": asdict(model.config),
            }, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )


def _train(args: argparse.Namespace) -> tuple[dict[str, dict[int, EEFPolicy]], list[dict[str, Any]]]:
    models: dict[str, dict[int, EEFPolicy]] = {method: {} for method in METHODS}
    rows: list[dict[str, Any]] = []
    checkpoints = sorted(set(x for x in CHECKPOINTS if x <= args.iterations) | {args.iterations})
    for method in METHODS:
        for seed_index in range(args.seeds):
            policy_seed = 17 + 97 * seed_index
            trainer = _make_trainer(method, policy_seed, args.device)
            training_seconds = 0.0
            for iteration in range(args.iterations + 1):
                if iteration in checkpoints:
                    metrics = _validation_metrics(
                        trainer.model,
                        seed=7301,
                        samples=args.validation_samples,
                        device=args.device,
                    )
                    rows.append(
                        {
                            "method": method,
                            "policy_seed": policy_seed,
                            "iteration": iteration,
                            "training_scenes": iteration * args.batch_size,
                            "training_seconds": training_seconds,
                            "curve_scope": "matched_scene_budget",
                            **metrics,
                        }
                    )
                if iteration == args.iterations:
                    break
                started = time.perf_counter()
                trainer.train_step(args.batch_size)
                training_seconds += time.perf_counter() - started
            # The policies used by panels C--F are frozen at the common scene
            # budget. Panel B may continue the trainer to a common CPU budget,
            # but that continuation cannot alter the fixed-scene checkpoint.
            fixed_scene_model = copy.deepcopy(trainer.model).cpu().eval()
            models[method][policy_seed] = fixed_scene_model
            checkpoint = args.output / "training" / method / f"seed_{policy_seed}" / "eef_policy.pt"
            _save_policy(fixed_scene_model, checkpoint, seed=policy_seed, iteration=args.iterations)

            extended_iteration = args.iterations
            if args.cpu_budget_seconds > training_seconds:
                target_times = np.arange(5.0, args.cpu_budget_seconds + 0.001, 5.0)
                for target_time in target_times:
                    if target_time <= training_seconds:
                        continue
                    while training_seconds < target_time:
                        started = time.perf_counter()
                        trainer.train_step(args.batch_size)
                        training_seconds += time.perf_counter() - started
                        extended_iteration += 1
                    metrics = _validation_metrics(
                        trainer.model,
                        seed=7301,
                        samples=args.validation_samples,
                        device=args.device,
                    )
                    rows.append(
                        {
                            "method": method,
                            "policy_seed": policy_seed,
                            "iteration": extended_iteration,
                            "training_scenes": extended_iteration * args.batch_size,
                            "training_seconds": training_seconds,
                            "curve_scope": "equal_cpu_extension",
                            **metrics,
                        }
                    )
                cpu_checkpoint = (
                    args.output / "training" / method / f"seed_{policy_seed}"
                    / "cpu_budget_policy.pt"
                )
                _save_policy(
                    trainer.model,
                    cpu_checkpoint,
                    seed=policy_seed,
                    iteration=extended_iteration,
                )
            print(json.dumps({
                "stage": "training", "method": method, "seed": policy_seed,
                "matched_scene_iteration": args.iterations,
                "cpu_curve_final_iteration": extended_iteration,
                "seconds": round(training_seconds, 3), "checkpoint": str(checkpoint),
            }), flush=True)
    _write_csv(args.output / "source_data" / "training_efficiency.csv", rows)
    return models, rows


def _evaluate(args: argparse.Namespace, models: dict[str, dict[int, EEFPolicy]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    nominal = next(x for x in DEFAULT_CONDITIONS if x.name == "nominal")
    combined = next(x for x in DEFAULT_CONDITIONS if x.name == "combined")
    for seed_index in range(args.seeds):
        policy_seed = 17 + 97 * seed_index
        seed_models = {method: models[method][policy_seed] for method in METHODS}
        evaluated = evaluate_policy_trials(
            seed_models,
            EEFConfig(),
            samples_per_cell=args.evaluation_samples,
            speeds=(0.8, 1.2, 1.6, 2.0, 2.4),
            conditions=(nominal, combined),
            seed=51047,
            device=args.device,
        )
        trial_rows.extend({"policy_seed": policy_seed, **row} for row in evaluated)
        frontier = evaluate_policy_trials(
            seed_models,
            replace(EEFConfig(), obstacle_count=2),
            samples_per_cell=args.evaluation_samples,
            speeds=(0.8, 1.2, 1.6, 1.9, 2.0),
            conditions=(BenchmarkCondition(
                "frontier",
                "Registered feasible speed--embodiment challenge",
                wind_multiplier=1.20,
                response_scale=0.94,
                drag_multiplier=1.08,
            ),),
            seed=91073,
            device=args.device,
            shared_scenes_across_speeds=True,
        )
        trial_rows.extend({"policy_seed": policy_seed, **row} for row in frontier)
        repeated = evaluate_interface_repeatability(
            seed_models,
            EEFConfig(),
            commands=24 if not args.smoke else 4,
            repeats=12 if not args.smoke else 3,
            speed=1.6,
            seed=52027,
            device=args.device,
        )
        repeat_rows.extend({"policy_seed": policy_seed, **row} for row in repeated)
    _write_csv(args.output / "source_data" / "final_trials.csv", trial_rows)
    _write_csv(args.output / "source_data" / "interface_repeatability.csv", repeat_rows)
    return trial_rows, repeat_rows


def _physical(args: argparse.Namespace, models: dict[str, dict[int, EEFPolicy]]) -> list[dict[str, Any]]:
    if args.skip_physical:
        _write_csv(args.output / "source_data" / "pybullet_trials.csv", [])
        return []
    deployed_seed = min(models["eef_full"])
    paths = {(role, direction): _module_path(role, direction) for role in ROLE_ORDER for direction in (-1, 1)}
    environment = CNMSwarmEnv(EnvConfig(num_drones=1, ray_count=24, neighbor_k=0, episode_seconds=22.0))
    rows: list[dict[str, Any]] = []
    trial = 0
    try:
        for method in METHODS:
            navigator_model = models[method][deployed_seed]
            for role in ROLE_ORDER:
                for direction in (-1, 1):
                    for repeat_index in range(args.physical_repeats):
                        result = run_physical_rollout(
                            environment,
                            EEFNavigator(navigator_model),
                            [paths[(role, direction)]],
                            speed=1.75,
                            max_steps=620,
                            waypoint_tolerance=0.30,
                            perturbation=RolloutPerturbation(
                                action_scale=0.84,
                                action_noise=0.05,
                                delay_steps=1,
                                ray_dropout=0.08,
                                ray_noise=0.018,
                                start_position_std=0.09,
                            ),
                            seed=62003 + trial,
                        )
                        rows.append({
                            "method": method,
                            "policy_seed": deployed_seed,
                            "module": role,
                            "direction": direction,
                            "repeat": repeat_index,
                            "commanded_speed_mps": 1.75,
                            **result.to_dict(),
                        })
                        trial += 1
            print(json.dumps({"stage": "pybullet", "method": method}), flush=True)
    finally:
        environment.close()
    _write_csv(args.output / "source_data" / "pybullet_trials.csv", rows)
    return rows


def _summary(args: argparse.Namespace, training: list[dict[str, Any]], trials: list[dict[str, Any]],
             repeated: list[dict[str, Any]], physical: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scope": "matched reimplementations in the CNMSwarmSim local-response task; simulation only",
        "methods": {},
        "cnm_ready_gate": {
            "position_m": POSITION_GATE_M,
            "velocity_mps": VELOCITY_GATE_MPS,
            "heading_deg": math.degrees(HEADING_GATE_RAD),
            "also_requires_safe_completion": True,
        },
        "goal_safe_completion": {
            "position_tolerance_m": GOAL_TOLERANCE_M,
            "also_requires_positive_clearance_and_registered_progress": True,
        },
        "frontier_completion": {
            "position_tolerance_m": FRONTIER_TOLERANCE_M,
            "also_requires_positive_clearance_and_registered_progress": True,
        },
    }
    for method in METHODS:
        final_training = [x for x in training if x["method"] == method and x["iteration"] == args.iterations]
        nominal = [
            x for x in trials if x["method"] == method and x["condition"] == "nominal"
            and abs(float(x["commanded_speed_mps"]) - 1.6) < 1.0e-9
        ]
        ready = [
            bool(x["safe_completion"])
            and float(x["interface_position_error_m"]) <= POSITION_GATE_M
            and float(x["interface_velocity_error_mps"]) <= VELOCITY_GATE_MPS
            and float(x["interface_heading_error_rad"]) <= HEADING_GATE_RAD
            for x in nominal
        ]
        speeds = (0.8, 1.2, 1.6, 1.9, 2.0)
        goal_safe_rates: list[float] = []
        cumulative_success: np.ndarray | None = None
        for speed in speeds:
            cell = [
                x for x in trials
                if x["method"] == method
                and x["condition"] == "frontier"
                and abs(float(x["commanded_speed_mps"]) - speed) < 1.0e-9
            ]
            success = np.asarray([
                bool(x["safe_completion"])
                and float(x["interface_position_error_m"]) <= FRONTIER_TOLERANCE_M
                for x in cell
            ], dtype=bool)
            cumulative_success = success if cumulative_success is None else (
                cumulative_success & success
            )
            goal_safe_rates.append(float(cumulative_success.mean()))
        dispersions: list[float] = []
        cells = sorted({(int(x["policy_seed"]), int(x["command_id"])) for x in repeated if x["method"] == method})
        for seed, command in cells:
            cell = [x for x in repeated if x["method"] == method and int(x["policy_seed"]) == seed and int(x["command_id"]) == command]
            xyz = np.asarray([[x["exit_dx_m"], x["exit_dy_m"], x["exit_dz_m"]] for x in cell], dtype=float)
            center = xyz.mean(axis=0)
            dispersions.extend((np.linalg.norm(xyz - center, axis=1) / max(np.linalg.norm(center), 1.0e-6)).tolist())
        pybullet = [x for x in physical if x["method"] == method]
        summary["methods"][method] = {
            "policy_seeds": args.seeds,
            "mean_training_seconds": float(np.mean([x["training_seconds"] for x in final_training])),
            "final_validation_safe_completion": float(np.mean([x["safe_completion"] for x in final_training])),
            "final_validation_goal_safe_completion": float(np.mean([x["goal_safe_completion"] for x in final_training])),
            "final_validation_cnm_ready_yield": float(np.mean([x["cnm_ready_yield"] for x in final_training])),
            "nominal_1p6_cnm_ready_yield": float(np.mean(ready)),
            "nominal_goal_safe_frontier": dict(zip((str(x) for x in speeds), goal_safe_rates)),
            "goal_safe_robustness_agility_area": float(
                np.trapz(goal_safe_rates, speeds) / (speeds[-1] - speeds[0])
            ),
            "nominal_1p6_attempts": len(ready),
            "relative_exit_dispersion": float(np.mean(dispersions)),
            "pybullet_completion": float(np.mean([bool(x["success"]) for x in pybullet])) if pybullet else None,
            "pybullet_mean_time_s": float(np.mean([float(x["completion_time"]) for x in pybullet])) if pybullet else None,
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.iterations = min(args.iterations, 8)
        args.batch_size = min(args.batch_size, 16)
        args.seeds = 1
        args.validation_samples = min(args.validation_samples, 24)
        args.evaluation_samples = min(args.evaluation_samples, 24)
        args.physical_repeats = 1
        args.cpu_budget_seconds = 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    previous_manifest: dict[str, Any] = {}
    manifest_path = args.output / "run_manifest.json"
    if args.reuse_checkpoints:
        if manifest_path.exists():
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = _load_models(args)
        training = _read_csv(args.output / "source_data" / "training_efficiency.csv")
    else:
        models, training = _train(args)
    trials, repeated = _evaluate(args, models)
    existing_physical = args.output / "source_data" / "pybullet_trials.csv"
    if args.reuse_checkpoints and existing_physical.exists() and not args.skip_physical:
        physical = _read_csv(existing_physical)
    else:
        physical = _physical(args, models)
    summary = _summary(args, training, trials, repeated, physical)
    manifest = {
        "schema_version": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_gib": psutil.virtual_memory().total / (1024 ** 3),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "arguments": vars(args) | {"output": str(args.output)},
        "methods": list(METHODS),
        "comparison_boundary": "literature-inspired reimplementations; no cross-paper numerical transfer",
        "selection_boundary": "method and challenge settings chosen with development streams 7301, 9103, 25001, 29011, frontier stream 81041 and PyBullet stream 49001; final evaluation uses unseen task stream 51047, frontier stream 91073, repeatability stream 52027 and PyBullet trial stream 62003",
        "checkpoint_reuse": bool(args.reuse_checkpoints),
        "original_training_run_elapsed_seconds": previous_manifest.get(
            "original_training_run_elapsed_seconds", previous_manifest.get("elapsed_seconds")
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": "complete", "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
