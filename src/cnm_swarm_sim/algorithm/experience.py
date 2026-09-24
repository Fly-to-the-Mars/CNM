"""Compile immutable CNM records from physical execution telemetry.

The compiler receives structural detections and local response executions.  It
does not inherit entry/exit statistics from the route used to collect those
executions: all interface means, covariances, finite supports, response
waypoints, evidence events, and diagnostics are reconstructed from the logs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .memory import CommandTemplate, ExecutionEvent, ExperienceRecord, InterfaceSummary


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _rotation(direction: int) -> np.ndarray:
    yaw = 0.0 if direction > 0 else math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


@dataclass(frozen=True)
class CompilerConfig:
    # Registered manuscript defaults.  These belong to the frozen stack W*;
    # deployment code must not relax them in response to a difficult query.
    min_successes: int = 3
    minimum_clearance: float = 0.25
    huber_k: float = 1.5
    support_quantile: float = 0.975
    covariance_shrinkage: float = 0.08
    command_points: int = 6
    entry_floor: tuple[float, ...] = (0.10, 0.10, 0.10, 0.15, 0.15, 0.15, math.radians(5.0))
    exit_floor: tuple[float, ...] = (0.10, 0.10, 0.10, 0.15, 0.15, 0.15, math.radians(5.0))
    support_floor: tuple[float, ...] = (0.10, 0.10, 0.08, 0.14, 0.14, 0.10, 0.09)
    maximum_terminal_position_error: float = 0.25
    maximum_terminal_velocity_error: float = 0.35
    maximum_terminal_yaw_error: float = math.radians(12.0)
    maximum_terminal_mahalanobis: float = 14.1
    minimum_mode_successes: int = 3
    split_bic_margin: float = 6.0


@dataclass
class FlightSegment:
    segment_id: str
    role: str
    structural_key: np.ndarray
    context: str
    anchor: np.ndarray
    direction: int
    origin_robot: str
    immutable_hash: str
    logical_time: float
    positions: np.ndarray
    velocities: np.ndarray
    yaws: np.ndarray
    actions: np.ndarray
    success: bool
    duration: float
    minimum_clearance: float
    reason: str = "completed"
    predicted_exit_mean: np.ndarray | None = None
    predicted_exit_covariance: np.ndarray | None = None
    calibration_version: str = "simulation-v1"
    hardware_class: str = "homogeneous-pybullet"

    def __post_init__(self) -> None:
        self.structural_key = np.asarray(self.structural_key, dtype=np.float64)
        self.anchor = np.asarray(self.anchor, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        self.yaws = np.asarray(self.yaws, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.float64)
        count = self.positions.shape[0]
        if self.positions.shape != (count, 3) or self.velocities.shape != (count, 3):
            raise ValueError("segment positions and velocities must be T by 3")
        if self.yaws.shape != (count,) or self.actions.shape[0] != count:
            raise ValueError("segment telemetry streams must have equal length")
        if count < 2:
            raise ValueError("at least two telemetry samples are required")
        if self.anchor.shape != (3,) or self.structural_key.ndim != 1:
            raise ValueError("invalid structural metadata")
        if self.predicted_exit_mean is not None:
            self.predicted_exit_mean = np.asarray(self.predicted_exit_mean, dtype=np.float64)
            if self.predicted_exit_mean.shape != (7,):
                raise ValueError("predicted exit mean must have seven elements")
        if self.predicted_exit_covariance is not None:
            self.predicted_exit_covariance = np.asarray(self.predicted_exit_covariance, dtype=np.float64)
            if self.predicted_exit_covariance.shape != (7, 7):
                raise ValueError("predicted exit covariance must be 7 by 7")
        if (self.predicted_exit_mean is None) != (self.predicted_exit_covariance is None):
            raise ValueError("predicted exit mean and covariance must be provided together")

    @property
    def entry_state(self) -> np.ndarray:
        return np.concatenate((self.positions[0], self.velocities[0], (self.yaws[0],)))

    @property
    def exit_state(self) -> np.ndarray:
        return np.concatenate((self.positions[-1], self.velocities[-1], (self.yaws[-1],)))

    def canonical_positions(self) -> np.ndarray:
        rotation = _rotation(self.direction)
        return (rotation.T @ (self.positions - self.anchor).T).T

    def canonical_state(self, state: np.ndarray) -> np.ndarray:
        rotation = _rotation(self.direction)
        result = np.asarray(state, dtype=np.float64).copy()
        result[:3] = rotation.T @ (result[:3] - self.anchor)
        result[3:6] = rotation.T @ result[3:6]
        result[6] = float(_wrap(result[6] - (0.0 if self.direction > 0 else math.pi)))
        return result

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        role: str,
        structural_key: np.ndarray,
        context: str,
        anchor: np.ndarray,
        direction: int,
        origin_robot: str,
        immutable_hash: str,
        logical_time: float,
        result: dict[str, Any],
        drone_index: int = 0,
    ) -> "FlightSegment":
        source = Path(path)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < 2:
            raise ValueError(f"insufficient telemetry in {source}")
        positions = np.asarray([row["positions"][drone_index] for row in rows], dtype=np.float64)
        velocities = np.asarray([row["velocities"][drone_index] for row in rows], dtype=np.float64)
        if "yaws" in rows[0]:
            yaws = np.asarray([row["yaws"][drone_index] for row in rows], dtype=np.float64)
        else:
            yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
            yaws[np.linalg.norm(velocities[:, :2], axis=1) < 1.0e-5] = 0.0 if direction > 0 else math.pi
        actions = np.asarray([row["actions"][drone_index] for row in rows], dtype=np.float64)
        identity = {
            "path": source.name,
            "role": role,
            "origin": origin_robot,
            "time": round(float(logical_time), 6),
            "samples": len(rows),
            "terminal": np.round(positions[-1], 6).tolist(),
        }
        segment_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return cls(
            segment_id=segment_id,
            role=role,
            structural_key=structural_key,
            context=context,
            anchor=anchor,
            direction=direction,
            origin_robot=origin_robot,
            immutable_hash=immutable_hash,
            logical_time=logical_time,
            positions=positions,
            velocities=velocities,
            yaws=yaws,
            actions=actions,
            success=bool(result["success"]),
            duration=float(result["completion_time"]),
            minimum_clearance=float(result.get("minimum_clearance") or 0.0),
            reason="completed" if result["success"] else "physical_failure",
            predicted_exit_mean=result.get("predicted_exit_mean"),
            predicted_exit_covariance=result.get("predicted_exit_covariance"),
            calibration_version=str(result.get("calibration_version", "simulation-v1")),
            hardware_class=str(result.get("hardware_class", "homogeneous-pybullet")),
        )


@dataclass(frozen=True)
class CompilationDiagnostics:
    attempted_segments: int
    successful_segments: int
    admitted_segments: int
    rejected_clearance: int
    rejected_outcome: int
    rejected_terminal_gate: int
    entry_generalized_dispersion: float
    exit_generalized_dispersion: float
    exit_position_p90: float
    connector_support_volume: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CompilationResult:
    record: ExperienceRecord
    events: list[ExecutionEvent]
    diagnostics: CompilationDiagnostics
    source_segment_ids: tuple[str, ...]


def _circular_reference(values: np.ndarray) -> float:
    return float(math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean())))


def _robust_interface(states: np.ndarray, floor: np.ndarray, support_floor: np.ndarray, cfg: CompilerConfig) -> InterfaceSummary:
    values = np.asarray(states, dtype=np.float64).copy()
    yaw_reference = _circular_reference(values[:, 6])
    values[:, 6] = yaw_reference + np.asarray(_wrap(values[:, 6] - yaw_reference))
    centre = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - centre), axis=0)
    scale = np.maximum(scale, floor)
    for _ in range(8):
        standardized = (values - centre) / scale
        radius = np.sqrt(np.sum(standardized**2, axis=1))
        weights = np.minimum(1.0, cfg.huber_k / np.maximum(radius, 1.0e-9))
        updated = np.average(values, axis=0, weights=weights)
        if np.max(np.abs(updated - centre)) < 1.0e-8:
            centre = updated
            break
        centre = updated
    residual = values - centre
    denominator = max(float(weights.sum() - 1.0), 1.0)
    covariance = (residual * weights[:, None]).T @ residual / denominator
    diagonal = np.maximum(np.diag(covariance), floor**2)
    covariance = (1.0 - cfg.covariance_shrinkage) * covariance + cfg.covariance_shrinkage * np.diag(diagonal)
    covariance[np.diag_indices_from(covariance)] = np.maximum(np.diag(covariance), floor**2)
    support = np.quantile(np.abs(residual), cfg.support_quantile, axis=0)
    support = np.maximum(support, support_floor)
    centre[6] = float(_wrap(centre[6]))
    finite_support = values.copy()
    finite_support[:, 6] = np.asarray(_wrap(finite_support[:, 6]))
    return InterfaceSummary(
        centre, covariance, support, sample_count=len(values), support_samples=finite_support
    )


def _terminal_gate(
    segment: FlightSegment,
    empirical_reference: InterfaceSummary,
    cfg: CompilerConfig,
) -> tuple[bool, dict[str, float]]:
    observed = segment.canonical_state(segment.exit_state)
    if segment.predicted_exit_mean is None:
        reference = empirical_reference.mean
        covariance = empirical_reference.covariance
    else:
        reference = segment.predicted_exit_mean
        covariance = segment.predicted_exit_covariance
        assert covariance is not None
    residual = observed - reference
    residual[6] = float(_wrap(residual[6]))
    regularized = covariance + np.eye(7) * 1.0e-8
    mahalanobis = float(residual @ np.linalg.solve(regularized, residual))
    diagnostics = {
        "position": float(np.linalg.norm(residual[:3])),
        "velocity": float(np.linalg.norm(residual[3:6])),
        "yaw": abs(float(residual[6])),
        "mahalanobis": mahalanobis,
    }
    accepted = (
        diagnostics["position"] <= cfg.maximum_terminal_position_error
        and diagnostics["velocity"] <= cfg.maximum_terminal_velocity_error
        and diagnostics["yaw"] <= cfg.maximum_terminal_yaw_error
        and diagnostics["mahalanobis"] <= cfg.maximum_terminal_mahalanobis
    )
    return accepted, diagnostics


def _resample_positions(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(increments)))
    if distance[-1] < 1.0e-9:
        return np.repeat(points[:1], count, axis=0)
    query = np.linspace(0.0, distance[-1], count)
    return np.stack([np.interp(query, distance, points[:, axis]) for axis in range(3)], axis=1)


def _generalized_dispersion(covariance: np.ndarray) -> float:
    position_covariance = covariance[:3, :3] + np.eye(3) * 1.0e-12
    determinant = max(float(np.linalg.det(position_covariance)), 1.0e-18)
    return determinant ** (1.0 / 6.0)


def _normal_log_likelihood(values: np.ndarray) -> float:
    variance = max(float(np.var(values)), 1.0e-8)
    residual = values - float(np.mean(values))
    return float(-0.5 * np.sum(np.log(2.0 * math.pi * variance) + residual**2 / variance))


def _two_mode_partition(states: np.ndarray, cfg: CompilerConfig) -> np.ndarray | None:
    """Return a deterministic BIC-supported two-mode partition when warranted."""

    count = len(states)
    if count < 2 * cfg.minimum_mode_successes:
        return None
    centred = states - np.median(states, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(centred), axis=0), 1.0e-6)
    whitened = centred / scale
    _, _, right = np.linalg.svd(whitened, full_matrices=False)
    projection = whitened @ right[0]
    order = np.argsort(projection, kind="stable")
    one_bic = 2.0 * math.log(count) - 2.0 * _normal_log_likelihood(projection)
    best: tuple[float, int] | None = None
    for split in range(cfg.minimum_mode_successes, count - cfg.minimum_mode_successes + 1):
        left = projection[order[:split]]
        right_values = projection[order[split:]]
        mixture_log_likelihood = (
            _normal_log_likelihood(left) + len(left) * math.log(len(left) / count)
            + _normal_log_likelihood(right_values) + len(right_values) * math.log(len(right_values) / count)
        )
        two_bic = 5.0 * math.log(count) - 2.0 * mixture_log_likelihood
        if best is None or two_bic < best[0]:
            best = (two_bic, split)
    if best is None or best[0] + cfg.split_bic_margin >= one_bic:
        return None
    labels = np.zeros(count, dtype=np.int64)
    labels[order[best[1] :]] = 1
    return labels


class ExperienceCompiler:
    """Build an executable, uncertainty-aware record from unique flight logs."""

    def __init__(self, config: CompilerConfig | None = None):
        self.config = config or CompilerConfig()

    def compile_variants(self, segments: Iterable[FlightSegment]) -> list[CompilationResult]:
        """Split a demonstrably multimodal outcome before compiling record variants."""

        samples = list(segments)
        successful = [
            segment for segment in samples
            if segment.success and segment.minimum_clearance >= self.config.minimum_clearance
        ]
        if len(successful) < 2 * self.config.minimum_mode_successes:
            return [self.compile(samples)]
        exit_states = np.stack([segment.canonical_state(segment.exit_state) for segment in successful])
        labels = _two_mode_partition(exit_states, self.config)
        if labels is None:
            return [self.compile(samples)]
        centres = np.stack([
            np.mean(exit_states[labels == label], axis=0) for label in (0, 1)
        ])
        grouped: list[list[FlightSegment]] = [[], []]
        successful_labels = {segment.segment_id: int(label) for segment, label in zip(successful, labels)}
        for segment in samples:
            label = successful_labels.get(segment.segment_id)
            if label is None:
                state = segment.canonical_state(segment.exit_state)
                residual = centres - state
                residual[:, 6] = np.asarray(_wrap(residual[:, 6]))
                label = int(np.argmin(np.linalg.norm(residual, axis=1)))
            grouped[label].append(segment)
        return [
            self.compile(group, variant=f"mode-{index}")
            for index, group in enumerate(grouped)
        ]

    def compile(self, segments: Iterable[FlightSegment], *, variant: str = "nominal") -> CompilationResult:
        samples = list(segments)
        if not samples:
            raise ValueError("at least one segment is required")
        first = samples[0]
        for segment in samples[1:]:
            identity = (segment.role, segment.context, segment.direction, segment.origin_robot, segment.immutable_hash)
            expected = (first.role, first.context, first.direction, first.origin_robot, first.immutable_hash)
            if identity != expected or not np.allclose(segment.anchor, first.anchor):
                raise ValueError("a compiled record must contain one role, context, direction, origin, and anchor")
        successful = [segment for segment in samples if segment.success]
        clearance_admitted = [
            segment for segment in successful
            if segment.minimum_clearance >= self.config.minimum_clearance
        ]
        if not clearance_admitted:
            raise ValueError(f"record {first.role}/{first.context} has no clearance-admissible executions")
        preliminary_exit = _robust_interface(
            np.stack([segment.canonical_state(segment.exit_state) for segment in clearance_admitted]),
            np.asarray(self.config.exit_floor),
            np.asarray(self.config.support_floor),
            self.config,
        )
        terminal_checks = {
            segment.segment_id: _terminal_gate(segment, preliminary_exit, self.config)
            for segment in clearance_admitted
        }
        admitted = [segment for segment in clearance_admitted if terminal_checks[segment.segment_id][0]]
        if len(admitted) < self.config.min_successes:
            raise ValueError(
                f"record {first.role}/{first.context} has {len(admitted)} admitted executions; "
                f"{self.config.min_successes} required"
            )
        entry_states = np.stack([segment.canonical_state(segment.entry_state) for segment in admitted])
        exit_states = np.stack([segment.canonical_state(segment.exit_state) for segment in admitted])
        entry = _robust_interface(
            entry_states,
            np.asarray(self.config.entry_floor),
            np.asarray(self.config.support_floor),
            self.config,
        )
        exit_interface = _robust_interface(
            exit_states,
            np.asarray(self.config.exit_floor),
            np.asarray(self.config.support_floor),
            self.config,
        )
        trajectory_samples = np.stack(
            [
                _resample_positions(segment.canonical_positions(), self.config.command_points + 1)
                for segment in admitted
            ]
        )
        median_trajectory = np.median(trajectory_samples, axis=0)
        command = CommandTemplate(
            waypoints=median_trajectory[1:],
            terminal_velocity=exit_interface.mean[3:6],
            terminal_yaw=float(exit_interface.mean[6]),
            mode="progress",
            horizon=float(np.median([segment.duration for segment in admitted])),
        )
        key = np.median(np.stack([segment.structural_key for segment in admitted]), axis=0)
        record_payload = {
            "role": first.role,
            "context": first.context,
            "direction": first.direction,
            "origin": first.origin_robot,
            "variant": variant,
            "immutable_hash": first.immutable_hash,
            "segments": sorted(segment.segment_id for segment in samples),
            "entry": np.round(entry.mean, 7).tolist(),
            "exit": np.round(exit_interface.mean, 7).tolist(),
        }
        record_id = hashlib.sha256(json.dumps(record_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        template = median_trajectory
        events: list[ExecutionEvent] = []
        admitted_ids = {segment.segment_id for segment in admitted}
        for segment in samples:
            canonical_exit = segment.canonical_state(segment.exit_state)
            prediction_error = float(np.linalg.norm(canonical_exit[:3] - exit_interface.mean[:3]))
            sampled = _resample_positions(segment.canonical_positions(), len(template))
            tracking_error = float(np.linalg.norm(sampled - template, axis=1).mean())
            event = ExecutionEvent.create(
                record_id=record_id,
                origin_robot=first.origin_robot,
                executor_robot=segment.origin_robot,
                context=segment.context,
                logical_time=segment.logical_time,
                success=segment.segment_id in admitted_ids,
                duration=segment.duration,
                min_clearance=segment.minimum_clearance,
                prediction_error=prediction_error,
                tracking_error=tracking_error,
                event_type="attempt",
                reason=(
                    "compiled_success" if segment.segment_id in admitted_ids
                    else "clearance_gate" if segment.success and segment.minimum_clearance < self.config.minimum_clearance
                    else "terminal_gate" if segment.segment_id in terminal_checks
                    else segment.reason
                ),
                immutable_hash=segment.immutable_hash,
                entry_state=segment.canonical_state(segment.entry_state),
                command=np.mean(segment.actions, axis=0),
                exit_state=canonical_exit,
                namespace=f"experience-compiler:{segment.segment_id}",
            )
            events.append(event)
        first_event = next(event for segment, event in zip(samples, events) if segment.segment_id in admitted_ids)
        record = ExperienceRecord(
            record_id=record_id,
            structural_key=key,
            role=first.role,
            context=first.context,
            entry=entry,
            command=command,
            exit=exit_interface,
            origin_robot=first.origin_robot,
            first_event_id=first_event.event_id,
            immutable_hash=first.immutable_hash,
            evidence_event_ids={event.event_id for event in events},
            created_time=min(segment.logical_time for segment in samples),
            calibration_version=first.calibration_version,
            hardware_class=first.hardware_class,
            variant_id=variant,
        )
        exit_residual = np.linalg.norm(exit_states[:, :3] - exit_interface.mean[:3], axis=1)
        diagnostics = CompilationDiagnostics(
            attempted_segments=len(samples),
            successful_segments=len(successful),
            admitted_segments=len(admitted),
            rejected_clearance=sum(segment.success and segment.minimum_clearance < self.config.minimum_clearance for segment in samples),
            rejected_outcome=sum(not segment.success for segment in samples),
            rejected_terminal_gate=sum(
                not accepted for accepted, _ in terminal_checks.values()
            ),
            entry_generalized_dispersion=_generalized_dispersion(entry.covariance),
            exit_generalized_dispersion=_generalized_dispersion(exit_interface.covariance),
            exit_position_p90=float(np.quantile(exit_residual, 0.90)),
            connector_support_volume=float(np.prod(exit_interface.support_radius)),
        )
        return CompilationResult(record, events, diagnostics, tuple(sorted(segment.segment_id for segment in samples)))
