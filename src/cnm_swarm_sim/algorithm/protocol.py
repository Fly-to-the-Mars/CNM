"""Registered module definitions and acquisition helpers for CNM experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .memory import CommandTemplate, ExecutionEvent, ExperienceRecord, InterfaceSummary


ROLE_ORDER = (
    "offset_gate_A",
    "splitter_tower_B",
    "pillar_field_C",
    "altitude_gate_D",
)


@dataclass(frozen=True)
class ModuleDefinition:
    role: str
    anchor: tuple[float, float, float]
    eastbound_waypoints: tuple[tuple[float, float, float], ...]
    westbound_waypoints: tuple[tuple[float, float, float], ...]
    structural_key: tuple[float, ...]

    def world_waypoints(self, direction: int) -> np.ndarray:
        values = self.eastbound_waypoints if direction > 0 else self.westbound_waypoints
        return np.asarray(values, dtype=np.float64)


MODULES: dict[str, ModuleDefinition] = {
    "offset_gate_A": ModuleDefinition(
        "offset_gate_A",
        (-7.4, -2.35, 1.45),
        (
            (-9.7, -3.05, 1.45),
            (-8.15, -3.05, 1.45),
            (-6.72, -3.05, 1.45),
            (-5.72, -3.04, 1.50),
            (-4.45, -2.90, 1.52),
            (-2.8, -2.25, 1.55),
        ),
        (
            (-2.8, -1.65, 1.55),
            (-4.35, -1.65, 1.50),
            (-5.72, -1.65, 1.48),
            (-6.72, -1.65, 1.45),
            (-8.15, -1.65, 1.45),
            (-9.7, -1.65, 1.45),
        ),
        (2.0, 1.0, 0.0, 1.10, 2.96, 7.4),
    ),
    "splitter_tower_B": ModuleDefinition(
        "splitter_tower_B",
        (0.0, 0.0, 1.55),
        ((-2.8, -2.25, 1.55), (-1.2, -2.25, 1.55), (1.4, -2.25, 1.55), (1.4, 1.10, 1.55)),
        ((1.4, 1.10, 1.55), (1.4, 2.25, 1.55), (-1.2, 2.25, 1.55), (-2.8, -1.65, 1.55)),
        (0.0, 1.0, 0.0, 3.1, 3.7, 0.56),
    ),
    "pillar_field_C": ModuleDefinition(
        "pillar_field_C",
        (3.3, 0.0, 1.60),
        ((1.4, 1.10, 1.55), (2.3, 1.10, 1.45), (3.6, 0.55, 1.45), (4.9, 0.0, 1.85)),
        ((4.9, 0.0, 1.85), (3.6, 0.55, 1.45), (2.3, 1.10, 1.45), (1.4, 1.10, 1.55)),
        (0.0, 0.0, 1.0, 5.0, 3.4, 0.72),
    ),
    "altitude_gate_D": ModuleDefinition(
        "altitude_gate_D",
        (6.0, 0.0, 2.05),
        ((4.9, 0.0, 1.85), (5.55, 0.0, 2.05), (6.45, 0.0, 2.05), (8.6, 0.0, 1.55)),
        ((8.6, 0.0, 1.55), (6.45, 0.0, 2.05), (5.55, 0.0, 2.05), (4.9, 0.0, 1.85)),
        (0.0, 0.0, 0.0, 8.1, 2.63, 2.05),
    ),
}


def _canonicalize(points: np.ndarray, anchor: np.ndarray, direction: int) -> np.ndarray:
    yaw = 0.0 if direction > 0 else math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    inverse = np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
    return np.stack([inverse @ (point - anchor) for point in points], axis=0)


def make_verified_record(
    role: str,
    direction: int,
    *,
    origin_robot: str,
    immutable_hash: str,
    context: str,
    logical_time: float,
    success: bool = True,
    min_clearance: float = 0.35,
    prediction_error: float = 0.08,
    tracking_error: float = 0.06,
    duration: float = 3.2,
    variant: str = "nominal",
    mode: str = "progress",
) -> tuple[ExperienceRecord, ExecutionEvent]:
    """Create a deterministic record only from a verified acquisition outcome."""

    if role not in MODULES:
        raise KeyError(role)
    definition = MODULES[role]
    anchor = np.asarray(definition.anchor, dtype=np.float64)
    world = definition.world_waypoints(direction)
    canonical = _canonicalize(world, anchor, direction)
    direction_vector = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    entry_mean = np.concatenate((canonical[0], direction_vector * 0.8, np.array((0.0,))))
    exit_mean = np.concatenate((canonical[-1], direction_vector * 0.8, np.array((0.0,))))
    covariance = np.diag((0.16**2, 0.18**2, 0.14**2, 0.22**2, 0.22**2, 0.18**2, 0.16**2))
    support = np.array((0.75, 0.85, 0.55, 0.8, 0.8, 0.65, 0.7), dtype=np.float64)
    identity_payload = {
        "role": role,
        "direction": int(direction),
        "origin": origin_robot,
        "context": context,
        "variant": variant,
        "logical_time": round(float(logical_time), 6),
        "immutable_hash": immutable_hash,
    }
    record_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    first_event = ExecutionEvent.create(
        record_id=record_id,
        origin_robot=origin_robot,
        executor_robot=origin_robot,
        context=context,
        logical_time=logical_time,
        success=success,
        duration=duration,
        min_clearance=min_clearance,
        prediction_error=prediction_error,
        tracking_error=tracking_error,
        reason="verified_acquisition" if success else "acquisition_failure",
        immutable_hash=immutable_hash,
        entry_state=entry_mean,
        command=np.mean(np.diff(canonical, axis=0), axis=0),
        exit_state=exit_mean,
    )
    record = ExperienceRecord(
        record_id=record_id,
        structural_key=np.asarray(definition.structural_key, dtype=np.float64),
        role=role,
        context=context,
        entry=InterfaceSummary(entry_mean, covariance.copy(), support.copy(), 1, entry_mean[None, :]),
        command=CommandTemplate(canonical[1:], direction_vector * 0.8, 0.0, mode, duration),
        exit=InterfaceSummary(exit_mean, covariance.copy(), support.copy(), 1, exit_mean[None, :]),
        origin_robot=origin_robot,
        first_event_id=first_event.event_id,
        immutable_hash=immutable_hash,
        evidence_event_ids={first_event.event_id},
        created_time=logical_time,
        variant_id=variant,
    )
    return record, first_event


def task_spec(direction: int = 1) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    roles = ROLE_ORDER if direction > 0 else tuple(reversed(ROLE_ORDER))
    anchors = tuple(np.asarray(MODULES[role].anchor, dtype=np.float64) for role in roles)
    keys = tuple(np.asarray(MODULES[role].structural_key, dtype=np.float64) for role in roles)
    return roles, anchors, keys
