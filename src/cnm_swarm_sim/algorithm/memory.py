"""Executable compositional navigation memory with source-linked event return."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""

    maximum_iterations = 200
    epsilon = 3.0e-12
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = floor if abs(d) < floor else d
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        d = 1.0 + coefficient * d
        d = floor if abs(d) < floor else d
        c = 1.0 + coefficient / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + coefficient * d
        d = floor if abs(d) < floor else d
        c = 1.0 + coefficient / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        update = d * c
        result *= update
        if abs(update - 1.0) < epsilon:
            break
    return result


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Dependency-free regularized incomplete beta for deterministic audits."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(probability: float, alpha: float, beta: float) -> float:
    """Return a beta posterior quantile by bounded deterministic bisection."""

    if not 0.0 < probability < 1.0:
        raise ValueError("beta quantile probability must lie strictly between zero and one")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta posterior parameters must be positive")
    lower, upper = 0.0, 1.0
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        if _regularized_incomplete_beta(midpoint, alpha, beta) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, received {result.ndim}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class InterfaceSummary:
    """Finite-support summary of [position, velocity, yaw] in a local frame."""

    mean: np.ndarray
    covariance: np.ndarray
    support_radius: np.ndarray
    sample_count: int = 1
    support_samples: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mean = _array(self.mean, ndim=1)
        self.covariance = _array(self.covariance, ndim=2)
        self.support_radius = _array(self.support_radius, ndim=1)
        if self.mean.shape != (7,) or self.covariance.shape != (7, 7) or self.support_radius.shape != (7,):
            raise ValueError("interface state must contain position(3), velocity(3), and yaw")
        if self.support_samples is None:
            self.support_samples = np.empty((0, 7), dtype=np.float64)
        else:
            self.support_samples = _array(self.support_samples, ndim=2)
            if self.support_samples.shape[1] != 7:
                raise ValueError("finite interface support samples must be N by 7")
        if self.sample_count < len(self.support_samples):
            raise ValueError("sample_count cannot be smaller than the retained finite support")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceSummary":
        return cls(
            payload["mean"], payload["covariance"], payload["support_radius"],
            payload["sample_count"], payload.get("support_samples"),
        )


@dataclass
class CommandTemplate:
    """Closed-loop local response represented by canonical-frame waypoints."""

    waypoints: np.ndarray
    terminal_velocity: np.ndarray
    terminal_yaw: float
    mode: str = "progress"
    horizon: float = 2.0

    def __post_init__(self) -> None:
        self.waypoints = _array(self.waypoints, ndim=2)
        self.terminal_velocity = _array(self.terminal_velocity, ndim=1)
        if self.waypoints.shape[1:] != (3,) or self.terminal_velocity.shape != (3,):
            raise ValueError("command waypoints and terminal velocity must be three-dimensional")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandTemplate":
        return cls(
            payload["waypoints"],
            payload["terminal_velocity"],
            float(payload["terminal_yaw"]),
            str(payload.get("mode", "progress")),
            float(payload.get("horizon", 2.0)),
        )


@dataclass
class ExperienceRecord:
    record_id: str
    structural_key: np.ndarray
    role: str
    context: str
    entry: InterfaceSummary
    command: CommandTemplate
    exit: InterfaceSummary
    origin_robot: str
    first_event_id: str
    immutable_hash: str
    parent_record_ids: tuple[str, ...] = ()
    evidence_event_ids: set[str] = field(default_factory=set)
    created_time: float = 0.0
    calibration_version: str = "simulation-v1"
    hardware_class: str = "homogeneous-pybullet"
    variant_id: str = "nominal"
    predicted_interaction_cost: float = 0.0

    def __post_init__(self) -> None:
        self.structural_key = _array(self.structural_key, ndim=1)
        self.evidence_event_ids = set(self.evidence_event_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "structural_key": self.structural_key.tolist(),
            "role": self.role,
            "context": self.context,
            "entry": self.entry.to_dict(),
            "command": self.command.to_dict(),
            "exit": self.exit.to_dict(),
            "origin_robot": self.origin_robot,
            "first_event_id": self.first_event_id,
            "immutable_hash": self.immutable_hash,
            "parent_record_ids": list(self.parent_record_ids),
            "evidence_event_ids": sorted(self.evidence_event_ids),
            "created_time": self.created_time,
            "calibration_version": self.calibration_version,
            "hardware_class": self.hardware_class,
            "variant_id": self.variant_id,
            "predicted_interaction_cost": self.predicted_interaction_cost,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        return cls(
            record_id=payload["record_id"],
            structural_key=payload["structural_key"],
            role=payload["role"],
            context=payload["context"],
            entry=InterfaceSummary.from_dict(payload["entry"]),
            command=CommandTemplate.from_dict(payload["command"]),
            exit=InterfaceSummary.from_dict(payload["exit"]),
            origin_robot=payload["origin_robot"],
            first_event_id=payload["first_event_id"],
            immutable_hash=payload["immutable_hash"],
            parent_record_ids=tuple(payload.get("parent_record_ids", ())),
            evidence_event_ids=set(payload.get("evidence_event_ids", ())),
            created_time=float(payload.get("created_time", 0.0)),
            calibration_version=str(payload.get("calibration_version", "simulation-v1")),
            hardware_class=str(payload.get("hardware_class", "homogeneous-pybullet")),
            variant_id=str(payload.get("variant_id", "nominal")),
            predicted_interaction_cost=float(payload.get("predicted_interaction_cost", 0.0)),
        )


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    record_id: str
    origin_robot: str
    executor_robot: str
    context: str
    logical_time: float
    success: bool
    duration: float
    min_clearance: float
    prediction_error: float
    tracking_error: float
    event_type: str = "attempt"
    reason: str = "completed"
    immutable_hash: str = ""
    parent_event_id: str | None = None
    entry_state: tuple[float, ...] = ()
    command: tuple[float, ...] = ()
    exit_state: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        origin_robot: str,
        executor_robot: str,
        context: str,
        logical_time: float,
        success: bool,
        duration: float,
        min_clearance: float,
        prediction_error: float,
        tracking_error: float,
        event_type: str = "attempt",
        reason: str = "completed",
        immutable_hash: str = "",
        parent_event_id: str | None = None,
        entry_state: Iterable[float] = (),
        command: Iterable[float] = (),
        exit_state: Iterable[float] = (),
        namespace: str = "cnm-event-v1",
    ) -> "ExecutionEvent":
        payload = {
            "record_id": record_id,
            "origin_robot": origin_robot,
            "executor_robot": executor_robot,
            "context": context,
            "logical_time": round(float(logical_time), 9),
            "success": bool(success),
            "duration": round(float(duration), 9),
            "min_clearance": round(float(min_clearance), 9),
            "prediction_error": round(float(prediction_error), 9),
            "tracking_error": round(float(tracking_error), 9),
            "event_type": event_type,
            "reason": reason,
            "immutable_hash": immutable_hash,
            "parent_event_id": parent_event_id,
            "entry_state": tuple(round(float(value), 9) for value in entry_state),
            "command": tuple(round(float(value), 9) for value in command),
            "exit_state": tuple(round(float(value), 9) for value in exit_state),
        }
        event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + json.dumps(payload, sort_keys=True)))
        return cls(event_id=event_id, **payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionEvent":
        return cls(**payload)


@dataclass
class PlacedRecord:
    record: ExperienceRecord
    anchor: np.ndarray
    direction: int
    entry: InterfaceSummary
    exit: InterfaceSummary
    waypoints: np.ndarray
    placement_covariance: np.ndarray = field(default_factory=lambda: np.zeros((7, 7), dtype=np.float64))
    bridge_observed_free: bool = True
    registration_residual: float = 0.0
    relative_scale: float = 1.0
    placement_yaw: float = math.nan


@dataclass(frozen=True)
class PlannerConfig:
    entry_threshold: float = 14.1
    connector_threshold: float = 14.1
    covariance_floor: float = 0.025
    retrieval_limit: int = 32
    graph_node_budget: int = 160
    max_depth: int = 6
    prior_success: float = 1.0
    prior_failure: float = 1.0
    context_decay: float = 600.0
    traffic_context_decay: float = 45.0
    static_context_decay: float = 600.0
    context_backoff: float = 0.20
    memory_budget_records: int = 128
    reliability_quantile: float = 0.10
    minimum_chain_reliability: float = 1.0e-3
    structural_distance_threshold: float = 0.35
    duration_scale: float = 3.0
    age_scale: float = 600.0
    reliability_weight: float = 1.2
    uncertainty_weight: float = 0.08
    age_weight: float = 0.15
    congestion_weight: float = 0.8
    connector_weight: float = 0.12
    bridge_weight: float = 0.08
    allowed_modes: tuple[str, ...] = ("progress", "pass", "yield", "stop")
    maximum_terminal_speed: float = 3.0
    maximum_horizon: float = 12.0
    edge_audit_limit: int = 2048
    maximum_bridge_duration: float = 0.35
    maximum_bridge_acceleration: float = 3.5
    maximum_bridge_yaw_rate: float = 6.0
    maximum_registration_residual: float = 0.12
    minimum_relative_scale: float = 0.85
    maximum_relative_scale: float = 1.15
    maximum_placement_std: float = 0.20
    merge_waypoint_rms: float = 0.25
    merge_terminal_velocity: float = 0.35
    merge_terminal_yaw: float = math.radians(12.0)
    allowed_mode_transitions: tuple[str, ...] = (
        "progress>progress", "progress>pass", "progress>yield", "progress>stop",
        "pass>progress", "pass>pass", "pass>yield", "pass>stop",
        "yield>progress", "yield>pass", "yield>yield", "yield>stop",
        "stop>progress", "stop>yield", "stop>stop",
    )


class CNMLibrary:
    """Finite record store plus a grow-only, idempotent execution-event ledger."""

    def __init__(self, owner_robot: str, immutable_hash: str, config: PlannerConfig | None = None):
        self.owner_robot = owner_robot
        self.immutable_hash = immutable_hash
        self.config = config or PlannerConfig()
        self.records: dict[str, ExperienceRecord] = {}
        self.events: dict[str, ExecutionEvent] = {}
        self.quarantined_conflicts: dict[str, list[dict[str, Any]]] = {}
        # Append-only operator trace.  It is deliberately excluded from
        # ``digest`` because deletion followed by exact restoration must
        # recover the same adaptive state while retaining an intervention log.
        self.operator_audit: list[dict[str, Any]] = []

    def add_record(self, record: ExperienceRecord) -> bool:
        if record.immutable_hash != self.immutable_hash:
            return False
        previous = self.records.get(record.record_id)
        if previous is not None:
            return json.dumps(previous.to_dict(), sort_keys=True) == json.dumps(record.to_dict(), sort_keys=True)
        self.records[record.record_id] = copy.deepcopy(record)
        self._enforce_budget()
        return record.record_id in self.records

    def add_event(self, event: ExecutionEvent) -> bool:
        if event.immutable_hash and event.immutable_hash != self.immutable_hash:
            self.quarantined_conflicts.setdefault(event.event_id, []).append(
                {"reason": "immutable_hash_mismatch", **event.to_dict()}
            )
            return False
        existing = self.events.get(event.event_id)
        if existing is not None:
            if existing.content_hash != event.content_hash:
                self.quarantined_conflicts.setdefault(event.event_id, []).append(event.to_dict())
                return False
            return True
        self.events[event.event_id] = event
        record = self.records.get(event.record_id)
        if record is not None:
            record.evidence_event_ids.add(event.event_id)
        return True

    def _context_weight(self, observed: str, requested: str) -> float:
        if observed == requested:
            return 1.0
        observed_parts = observed.split("_")
        requested_parts = requested.split("_")
        if observed_parts[0] != requested_parts[0]:
            return 0.0
        observed_tokens = set(observed_parts[1:])
        requested_tokens = set(requested_parts[1:])
        lane_tokens = {"nominal", "alternate"}
        observed_lane = observed_tokens & lane_tokens
        requested_lane = requested_tokens & lane_tokens
        if observed_lane and requested_lane and observed_lane != requested_lane:
            return 0.0
        union = observed_tokens | requested_tokens
        similarity = len(observed_tokens & requested_tokens) / len(union) if union else 1.0
        return self.config.context_backoff + (1.0 - self.config.context_backoff) * similarity

    def _decay_constant(self, *contexts: str) -> float:
        traffic_tokens = {"dense", "traffic", "dynamic", "opposing", "merge", "pass", "yield"}
        tokens = {
            token.lower()
            for context in contexts
            for token in context.replace("-", "_").split("_")
        }
        if tokens & traffic_tokens:
            return self.config.traffic_context_decay
        return self.config.static_context_decay

    def posterior_parameters(self, record_id: str, context: str, logical_time: float) -> tuple[float, float]:
        record = self.records[record_id]
        alpha = self.config.prior_success
        beta = self.config.prior_failure
        source_counts: dict[str, int] = {}
        ordered_events = sorted(
            (self.events[event_id] for event_id in record.evidence_event_ids if event_id in self.events),
            key=lambda event: (event.logical_time, event.event_id),
        )
        for event in ordered_events:
            if event.event_type != "attempt":
                continue
            context_weight = self._context_weight(event.context, context)
            if context_weight <= 0.0:
                continue
            age = max(0.0, logical_time - event.logical_time)
            decay = math.exp(-age / self._decay_constant(event.context, context))
            previous_from_source = source_counts.get(event.executor_robot, 0)
            source_discount = 1.0 / math.sqrt(previous_from_source + 1.0)
            source_counts[event.executor_robot] = previous_from_source + 1
            # Prediction residual is part of the outcome being evaluated, not
            # a reason to discard an execution.  In particular, a large miss
            # must remain strong negative evidence.  Weighting is therefore
            # limited to the registered context, age and independent-source
            # terms described by the evidence model.
            weight = context_weight * decay * source_discount
            alpha += weight * float(event.success)
            beta += weight * float(not event.success)
        return alpha, beta

    def reliability_mean(self, record_id: str, context: str, logical_time: float) -> float:
        alpha, beta = self.posterior_parameters(record_id, context, logical_time)
        return alpha / max(alpha + beta, 1.0e-12)

    def reliability(self, record_id: str, context: str, logical_time: float) -> float:
        """Conservative lower posterior quantile used for composition and sharing."""

        alpha, beta = self.posterior_parameters(record_id, context, logical_time)
        return beta_quantile(self.config.reliability_quantile, alpha, beta)

    def duration(self, record_id: str, context: str, logical_time: float) -> float:
        record = self.records[record_id]
        weighted = 0.0
        mass = 0.0
        for event_id in record.evidence_event_ids:
            event = self.events.get(event_id)
            if event is None or event.event_type != "attempt":
                continue
            context_weight = self._context_weight(event.context, context)
            if context_weight <= 0.0:
                continue
            decay = math.exp(
                -max(0.0, logical_time - event.logical_time)
                / self._decay_constant(event.context, context)
            )
            weight = context_weight * decay
            # A short collision or abort must not look faster than a completed
            # response.  Failed attempts receive the registered timeout cost.
            effective_duration = (
                event.duration
                if event.success
                else max(event.duration, 1.5 * record.command.horizon)
            )
            weighted += weight * effective_duration
            mass += weight
        return weighted / mass if mass else record.command.horizon

    def packet(self, record_ids: Iterable[str], *, event_ids: Iterable[str] | None = None) -> dict[str, Any]:
        record_set = [self.records[r].to_dict() for r in record_ids if r in self.records]
        if event_ids is None:
            attached = {eid for record in record_set for eid in record["evidence_event_ids"]}
        else:
            attached = set(event_ids)
        event_set = [self.events[e].to_dict() for e in sorted(attached) if e in self.events]
        payload = {
            "schema_version": 2,
            "sender": self.owner_robot,
            "immutable_hash": self.immutable_hash,
            "records": record_set,
            "events": event_set,
        }
        payload["payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    def merge_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Merge a signed experience packet and retain a recipient-side trace."""

        audit: dict[str, Any] = {
            "operator": "transfer_receive",
            "sender": packet.get("sender"),
            "payload_sha256": packet.get("payload_sha256"),
            "payload_bytes": len(json.dumps(packet, sort_keys=True).encode("utf-8")),
            "accepted_record_ids": [],
            "duplicate_record_ids": [],
            "accepted_event_ids": [],
            "duplicate_event_ids": [],
            "rejected_items": [],
        }

        def reject(reason: str, count: int) -> dict[str, Any]:
            audit["accepted"] = False
            audit["reason"] = reason
            audit["rejected_items"].append({"reason": reason, "count": int(count)})
            self.operator_audit.append(copy.deepcopy(audit))
            return {"records": 0, "events": 0, "rejected": int(count), "reason": reason}

        if packet.get("schema_version") not in (1, 2):
            return reject("unsupported_schema", 1)
        if packet.get("immutable_hash") != self.immutable_hash:
            return reject(
                "immutable_hash_mismatch",
                len(packet.get("records", ())) + len(packet.get("events", ())),
            )
        unsigned = {k: v for k, v in packet.items() if k != "payload_sha256"}
        observed = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if observed != packet.get("payload_sha256"):
            return reject("payload_hash_mismatch", 1)
        record_count = 0
        event_count = 0
        rejected = 0
        for item in packet.get("records", ()):
            try:
                record = ExperienceRecord.from_dict(item)
            except (KeyError, TypeError, ValueError) as exc:
                audit["rejected_items"].append(
                    {"kind": "record", "reason": "deserialization", "detail": str(exc)}
                )
                rejected += 1
                continue
            existed = record.record_id in self.records
            if self.add_record(record):
                if existed:
                    audit["duplicate_record_ids"].append(record.record_id)
                else:
                    audit["accepted_record_ids"].append(record.record_id)
                    record_count += 1
            else:
                audit["rejected_items"].append(
                    {"kind": "record", "record_id": record.record_id, "reason": "record_gate"}
                )
                rejected += 1
        for item in packet.get("events", ()):
            try:
                event = ExecutionEvent.from_dict(item)
            except (KeyError, TypeError, ValueError) as exc:
                audit["rejected_items"].append(
                    {"kind": "event", "reason": "deserialization", "detail": str(exc)}
                )
                rejected += 1
                continue
            existed = event.event_id in self.events
            if self.add_event(event):
                if existed:
                    audit["duplicate_event_ids"].append(event.event_id)
                else:
                    audit["accepted_event_ids"].append(event.event_id)
                    event_count += 1
            else:
                audit["rejected_items"].append(
                    {"kind": "event", "event_id": event.event_id, "reason": "event_gate"}
                )
                rejected += 1
        audit["accepted"] = rejected == 0
        audit["reason"] = "ok" if rejected == 0 else "partial_rejection"
        self.operator_audit.append(copy.deepcopy(audit))
        return {
            "records": record_count,
            "events": event_count,
            "rejected": rejected,
            "reason": audit["reason"],
        }

    def select_for_query(
        self,
        roles: Iterable[str],
        context: str,
        digest: set[str],
        logical_time: float,
        byte_budget: int,
        structural_keys: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], int]:
        role_set = set(roles)
        candidates = [
            record
            for record in self.records.values()
            if record.role in role_set and record.record_id not in digest
        ]
        def share_components(record: ExperienceRecord) -> dict[str, float]:
            target_key = None if structural_keys is None else structural_keys.get(record.role)
            if target_key is None:
                relevance = 1.0
            else:
                distance = float(np.linalg.norm(record.structural_key - np.asarray(target_key)))
                relevance = math.exp(-distance)
            reliability = self.reliability(record.record_id, context, logical_time)
            sources = {
                self.events[event_id].executor_robot
                for event_id in record.evidence_event_ids
                if event_id in self.events
            }
            diversity = 1.0 + math.log1p(len(sources))
            age = max(0.0, logical_time - record.created_time)
            age_factor = 1.0 + age / max(self.config.age_scale, 1.0e-9)
            record_packet = self.packet((record.record_id,))
            byte_cost = float(len(json.dumps(record_packet, sort_keys=True).encode("utf-8")))
            score = relevance * reliability * diversity / max(byte_cost * age_factor, 1.0)
            return {
                "relevance": relevance,
                "novelty": 1.0,
                "utility": reliability,
                "source_diversity": diversity,
                "byte_cost": byte_cost,
                "age_factor": age_factor,
                "score": score,
            }

        component_cache = {record.record_id: share_components(record) for record in candidates}
        candidates.sort(key=lambda record: (-component_cache[record.record_id]["score"], record.record_id))
        selected: list[str] = []
        packet = self.packet(selected)
        selection_audit: list[dict[str, Any]] = []
        for record in candidates:
            trial = self.packet(selected + [record.record_id])
            proposed_audit = selection_audit + [
                {"record_id": record.record_id, **component_cache[record.record_id]}
            ]
            trial["selection_audit"] = proposed_audit
            unsigned = {key: value for key, value in trial.items() if key != "payload_sha256"}
            trial["payload_sha256"] = hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            size = len(json.dumps(trial, sort_keys=True).encode("utf-8"))
            if size <= byte_budget:
                selected.append(record.record_id)
                packet = trial
                selection_audit = proposed_audit
        packet["selection_audit"] = selection_audit
        unsigned = {key: value for key, value in packet.items() if key != "payload_sha256"}
        packet["payload_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return packet, len(json.dumps(packet, sort_keys=True).encode("utf-8"))

    @staticmethod
    def _merge_interfaces(interfaces: list[InterfaceSummary]) -> InterfaceSummary:
        weights = np.asarray([item.sample_count for item in interfaces], dtype=np.float64)
        total = float(weights.sum())
        means = np.stack([item.mean for item in interfaces])
        yaw = math.atan2(
            float(np.sum(weights * np.sin(means[:, 6]))),
            float(np.sum(weights * np.cos(means[:, 6]))),
        )
        unwrapped = means.copy()
        unwrapped[:, 6] = yaw + (means[:, 6] - yaw + math.pi) % (2.0 * math.pi) - math.pi
        mean = np.average(unwrapped, axis=0, weights=weights)
        covariance = np.zeros((7, 7), dtype=np.float64)
        support = np.zeros(7, dtype=np.float64)
        samples: list[np.ndarray] = []
        for weight, item, item_mean in zip(weights, interfaces, unwrapped):
            delta = item_mean - mean
            covariance += weight * (item.covariance + np.outer(delta, delta))
            support = np.maximum(support, np.abs(delta) + item.support_radius)
            if len(item.support_samples):
                retained = item.support_samples.copy()
                retained[:, 6] = yaw + (retained[:, 6] - yaw + math.pi) % (2.0 * math.pi) - math.pi
                samples.append(retained)
        mean[6] = (mean[6] + math.pi) % (2.0 * math.pi) - math.pi
        finite_support = np.concatenate(samples) if samples else np.empty((0, 7), dtype=np.float64)
        if len(finite_support):
            finite_support[:, 6] = (finite_support[:, 6] + math.pi) % (2.0 * math.pi) - math.pi
        return InterfaceSummary(
            mean,
            covariance / max(total, 1.0),
            support,
            int(total),
            finite_support,
        )

    def merge_compatible_records(
        self,
        record_ids: Iterable[str],
        *,
        logical_time: float,
    ) -> tuple[ExperienceRecord | None, dict[str, Any]]:
        """Apply an explicit, auditable merge to compatible record variants."""

        identifiers = tuple(sorted(set(record_ids)))
        records = [self.records[record_id] for record_id in identifiers if record_id in self.records]
        if len(records) != len(identifiers) or len(records) < 2:
            audit = {
                "operator": "merge",
                "logical_time": float(logical_time),
                "merged": False,
                "reason": "missing_or_insufficient_records",
                "record_ids": identifiers,
            }
            self.operator_audit.append(copy.deepcopy(audit))
            return None, audit
        first = records[0]
        compatible = all(
            record.role == first.role
            and record.context == first.context
            and record.immutable_hash == self.immutable_hash
            and record.calibration_version == first.calibration_version
            and record.hardware_class == first.hardware_class
            and record.command.mode == first.command.mode
            and record.command.waypoints.shape == first.command.waypoints.shape
            and float(np.sqrt(np.mean(
                np.square(record.command.waypoints - first.command.waypoints)
            ))) <= self.config.merge_waypoint_rms
            and float(np.linalg.norm(
                record.command.terminal_velocity - first.command.terminal_velocity
            )) <= self.config.merge_terminal_velocity
            and abs(float(
                (record.command.terminal_yaw - first.command.terminal_yaw + math.pi)
                % (2.0 * math.pi) - math.pi
            )) <= self.config.merge_terminal_yaw
            and float(np.linalg.norm(record.structural_key - first.structural_key))
                <= self.config.structural_distance_threshold
            for record in records[1:]
        )
        if not compatible:
            audit = {
                "operator": "merge",
                "logical_time": float(logical_time),
                "merged": False,
                "reason": "incompatible_records",
                "record_ids": identifiers,
            }
            self.operator_audit.append(copy.deepcopy(audit))
            return None, audit
        weights = np.asarray([record.entry.sample_count for record in records], dtype=np.float64)
        waypoints = np.average(np.stack([record.command.waypoints for record in records]), axis=0, weights=weights)
        terminal_velocity = np.average(
            np.stack([record.command.terminal_velocity for record in records]), axis=0, weights=weights
        )
        terminal_yaw = math.atan2(
            float(np.sum(weights * np.sin([record.command.terminal_yaw for record in records]))),
            float(np.sum(weights * np.cos([record.command.terminal_yaw for record in records]))),
        )
        payload = {
            "operator": "merge-v1",
            "parents": identifiers,
            "immutable_hash": self.immutable_hash,
        }
        merged_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        merged = ExperienceRecord(
            record_id=merged_id,
            structural_key=np.average(
                np.stack([record.structural_key for record in records]), axis=0, weights=weights
            ),
            role=first.role,
            context=first.context,
            entry=self._merge_interfaces([record.entry for record in records]),
            command=CommandTemplate(
                waypoints,
                terminal_velocity,
                terminal_yaw,
                first.command.mode,
                float(np.average([record.command.horizon for record in records], weights=weights)),
            ),
            exit=self._merge_interfaces([record.exit for record in records]),
            origin_robot=self.owner_robot,
            first_event_id=min(record.first_event_id for record in records),
            immutable_hash=self.immutable_hash,
            parent_record_ids=identifiers,
            evidence_event_ids=set().union(*(record.evidence_event_ids for record in records)),
            created_time=logical_time,
            calibration_version=first.calibration_version,
            hardware_class=first.hardware_class,
            variant_id="merged",
        )
        removed = self.delete_records(
            identifiers,
            reason="merge_parent_records",
            logical_time=logical_time,
        )
        if not self.add_record(merged):
            self.restore_records(
                removed,
                reason="merge_rollback",
                logical_time=logical_time,
            )
            audit = {
                "operator": "merge",
                "logical_time": float(logical_time),
                "merged": False,
                "reason": "budget_or_hash_rejection",
                "record_ids": identifiers,
            }
            self.operator_audit.append(copy.deepcopy(audit))
            return None, audit
        audit = {
            "operator": "merge",
            "logical_time": float(logical_time),
            "merged": True,
            "reason": "compatible",
            "record_ids": identifiers,
            "merged_record_id": merged.record_id,
            "parent_digests": {
                record_id: hashlib.sha256(
                    json.dumps(removed[record_id].to_dict(), sort_keys=True).encode("utf-8")
                ).hexdigest()
                for record_id in identifiers
            },
        }
        self.operator_audit.append(copy.deepcopy(audit))
        return merged, audit

    def delete_records(
        self,
        record_ids: Iterable[str],
        *,
        reason: str = "unspecified",
        logical_time: float | None = None,
        audit: bool = True,
    ) -> dict[str, ExperienceRecord]:
        removed: dict[str, ExperienceRecord] = {}
        for record_id in record_ids:
            if record_id in self.records:
                removed[record_id] = self.records.pop(record_id)
        if removed and audit:
            self.operator_audit.append(
                {
                    "operator": "delete",
                    "reason": reason,
                    "logical_time": logical_time,
                    "record_ids": sorted(removed),
                    "record_sha256": {
                        record_id: hashlib.sha256(
                            json.dumps(record.to_dict(), sort_keys=True).encode("utf-8")
                        ).hexdigest()
                        for record_id, record in sorted(removed.items())
                    },
                }
            )
        return removed

    def withhold_source(self, origin_robot: str) -> dict[str, ExperienceRecord]:
        """Remove records from one registered source for a causal withholding branch."""

        return self.delete_records(
            (
                record.record_id
                for record in tuple(self.records.values())
                if record.origin_robot == origin_robot
            ),
            reason=f"source_withholding:{origin_robot}",
        )

    def restore_records(
        self,
        records: dict[str, ExperienceRecord],
        *,
        reason: str = "exact_restoration",
        logical_time: float | None = None,
        audit: bool = True,
    ) -> None:
        restored: list[str] = []
        for record in records.values():
            if self.add_record(record):
                restored.append(record.record_id)
        if restored and audit:
            self.operator_audit.append(
                {
                    "operator": "restore",
                    "reason": reason,
                    "logical_time": logical_time,
                    "record_ids": sorted(restored),
                }
            )

    def digest(self) -> str:
        payload = {
            "records": [self.records[key].to_dict() for key in sorted(self.records)],
            "events": [self.events[key].to_dict() for key in sorted(self.events)],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "owner_robot": self.owner_robot,
            "immutable_hash": self.immutable_hash,
            "planner_config": asdict(self.config),
            "records": [self.records[key].to_dict() for key in sorted(self.records)],
            "events": [self.events[key].to_dict() for key in sorted(self.events)],
            "quarantined_conflicts": self.quarantined_conflicts,
            "operator_audit": copy.deepcopy(self.operator_audit),
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CNMLibrary":
        library = cls(payload["owner_robot"], payload["immutable_hash"], PlannerConfig(**payload["planner_config"]))
        for record in payload.get("records", ()):
            library.add_record(ExperienceRecord.from_dict(record))
        for event in payload.get("events", ()):
            library.add_event(ExecutionEvent.from_dict(event))
        library.quarantined_conflicts = payload.get("quarantined_conflicts", {})
        library.operator_audit = copy.deepcopy(payload.get("operator_audit", []))
        return library

    def _enforce_budget(self) -> None:
        while len(self.records) > self.config.memory_budget_records:
            scores = []
            role_counts: dict[str, int] = {}
            for record in self.records.values():
                role_counts[record.role] = role_counts.get(record.role, 0) + 1
            for record in self.records.values():
                if role_counts[record.role] <= 1:
                    continue
                compactness = float(np.linalg.slogdet(record.exit.covariance + np.eye(7) * 1.0e-8)[1])
                source_diversity = float(record.origin_robot != self.owner_robot)
                score = 0.4 * len(record.evidence_event_ids) - 0.05 * compactness + 0.2 * source_diversity
                scores.append((score, record.created_time, record.record_id))
            if not scores:
                break
            _, _, record_id = min(scores)
            evicted = self.records.pop(record_id)
            self.operator_audit.append(
                {
                    "operator": "curate",
                    "reason": "memory_budget",
                    "record_id": record_id,
                    "record_sha256": hashlib.sha256(
                        json.dumps(evicted.to_dict(), sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                }
            )


class CNMPlanner:
    """Hard-gated placement and bounded search over executable interfaces."""

    def __init__(self, library: CNMLibrary):
        self.library = library
        self.config = library.config

    @staticmethod
    def _transform_state(
        state: np.ndarray,
        anchor: np.ndarray,
        direction: int,
        placement_yaw: float | None = None,
    ) -> np.ndarray:
        yaw = (0.0 if direction > 0 else math.pi) if placement_yaw is None else float(placement_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        result = state.copy()
        result[:3] = anchor + rotation @ state[:3]
        result[3:6] = rotation @ state[3:6]
        result[6] = (state[6] + yaw + math.pi) % (2.0 * math.pi) - math.pi
        return result

    @staticmethod
    def _transform_covariance(
        covariance: np.ndarray,
        direction: int,
        placement_yaw: float | None = None,
    ) -> np.ndarray:
        yaw = (0.0 if direction > 0 else math.pi) if placement_yaw is None else float(placement_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        r3 = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        jacobian = np.eye(7)
        jacobian[:3, :3] = r3
        jacobian[3:6, 3:6] = r3
        return jacobian @ covariance @ jacobian.T

    @classmethod
    def _transform_support(
        cls,
        samples: np.ndarray,
        anchor: np.ndarray,
        direction: int,
        placement_yaw: float | None = None,
    ) -> np.ndarray:
        if len(samples) == 0:
            return np.empty((0, 7), dtype=np.float64)
        return np.stack([
            cls._transform_state(sample, anchor, direction, placement_yaw)
            for sample in samples
        ])

    def place(
        self,
        record: ExperienceRecord,
        anchor: np.ndarray,
        direction: int,
        placement_covariance: np.ndarray | None = None,
        *,
        registration_residual: float = 0.0,
        relative_scale: float = 1.0,
        bridge_observed_free: bool = True,
        placement_yaw: float | None = None,
    ) -> PlacedRecord:
        anchor = _array(anchor, ndim=1)
        if registration_residual > self.config.maximum_registration_residual:
            raise ValueError("record placement exceeds the registered residual gate")
        if not self.config.minimum_relative_scale <= relative_scale <= self.config.maximum_relative_scale:
            raise ValueError("record placement exceeds the registered scale gate")
        placement = np.zeros((7, 7), dtype=np.float64)
        if placement_covariance is not None:
            placement = _array(placement_covariance, ndim=2)
            if placement.shape == (3, 3):
                placement_full = np.zeros((7, 7), dtype=np.float64)
                placement_full[:3, :3] = placement
                placement = placement_full
            if placement.shape != (7, 7):
                raise ValueError("placement covariance must be 3 by 3 or 7 by 7")
            placement_std = math.sqrt(max(0.0, float(np.linalg.eigvalsh(placement[:3, :3]).max())))
            if placement_std > self.config.maximum_placement_std:
                raise ValueError("record placement exceeds the registered uncertainty gate")
        entry = InterfaceSummary(
            self._transform_state(record.entry.mean, anchor, direction, placement_yaw),
            self._transform_covariance(record.entry.covariance, direction, placement_yaw) + placement,
            record.entry.support_radius.copy(),
            record.entry.sample_count,
            self._transform_support(record.entry.support_samples, anchor, direction, placement_yaw),
        )
        exit_interface = InterfaceSummary(
            self._transform_state(record.exit.mean, anchor, direction, placement_yaw),
            self._transform_covariance(record.exit.covariance, direction, placement_yaw) + placement,
            record.exit.support_radius.copy(),
            record.exit.sample_count,
            self._transform_support(record.exit.support_samples, anchor, direction, placement_yaw),
        )
        yaw = (0.0 if direction > 0 else math.pi) if placement_yaw is None else float(placement_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        waypoints = np.stack([anchor + rotation @ point for point in record.command.waypoints], axis=0)
        return PlacedRecord(
            record,
            anchor,
            direction,
            entry,
            exit_interface,
            waypoints,
            placement,
            bridge_observed_free,
            registration_residual,
            relative_scale,
            yaw,
        )

    def entry_distance(self, state: np.ndarray, placed: PlacedRecord) -> float:
        state = _array(state, ndim=1)
        delta = state - placed.entry.mean
        delta[6] = (delta[6] + math.pi) % (2.0 * math.pi) - math.pi
        covariance = placed.entry.covariance + np.eye(7) * self.config.covariance_floor
        distance = float(delta @ np.linalg.solve(covariance, delta))
        support_ok = bool(np.all(np.abs(delta) <= placed.entry.support_radius))
        return distance if support_ok else math.inf

    def exit_distance(self, state: np.ndarray, placed: PlacedRecord) -> float:
        state = _array(state, ndim=1)
        delta = state - placed.exit.mean
        delta[6] = (delta[6] + math.pi) % (2.0 * math.pi) - math.pi
        covariance = placed.exit.covariance + np.eye(7) * self.config.covariance_floor
        distance = float(delta @ np.linalg.solve(covariance, delta))
        support_ok = bool(np.all(np.abs(delta) <= placed.exit.support_radius))
        return distance if support_ok else math.inf

    def connector_distance(self, left: PlacedRecord, right: PlacedRecord) -> float:
        assessment = self.connector_assessment(left, right)
        if not assessment["support_reachable"] or not assessment["bridge_observed_free"]:
            return math.inf
        # Eq. (connector): bridge feasibility is a separate hard gate; it must
        # not erase interface mismatch from the uncertainty-aware distance.
        delta = left.exit.mean - right.entry.mean
        delta[6] = (delta[6] + math.pi) % (2.0 * math.pi) - math.pi
        covariance = left.exit.covariance + right.entry.covariance + np.eye(7) * self.config.covariance_floor
        return float(delta @ np.linalg.solve(covariance, delta))

    def connector_assessment(self, left: PlacedRecord, right: PlacedRecord) -> dict[str, Any]:
        """Assess finite support and the registered short closed-loop bridge."""

        delta = left.exit.mean - right.entry.mean
        delta[6] = (delta[6] + math.pi) % (2.0 * math.pi) - math.pi
        combined_support = left.exit.support_radius + right.entry.support_radius
        excess = np.maximum(np.abs(delta) - combined_support, 0.0)
        acceleration = max(self.config.maximum_bridge_acceleration, 1.0e-12)
        yaw_rate = max(self.config.maximum_bridge_yaw_rate, 1.0e-12)
        position_times = np.sqrt(2.0 * excess[:3] / acceleration)
        velocity_times = excess[3:6] / acceleration
        yaw_time = excess[6] / yaw_rate
        required_duration = float(max(np.max(position_times), np.max(velocity_times), yaw_time))
        transition = f"{left.record.command.mode}>{right.record.command.mode}"
        interaction_modes_compatible = transition in self.config.allowed_mode_transitions
        support_reachable = (
            required_duration <= self.config.maximum_bridge_duration
            and interaction_modes_compatible
        )
        capacity = np.concatenate((
            np.full(3, 0.5 * acceleration * self.config.maximum_bridge_duration**2),
            np.full(3, acceleration * self.config.maximum_bridge_duration),
            np.asarray((yaw_rate * self.config.maximum_bridge_duration,)),
        ))
        uncompensated = np.sign(delta) * np.maximum(excess - capacity, 0.0)
        return {
            "direct_support_overlap": bool(np.all(excess <= 1.0e-12)),
            "support_reachable": support_reachable,
            "required_bridge_duration": required_duration,
            "bridge_effort": min(1.0, required_duration / max(self.config.maximum_bridge_duration, 1.0e-12)),
            "bridge_observed_free": bool(left.bridge_observed_free and right.bridge_observed_free),
            "interaction_modes_compatible": interaction_modes_compatible,
            "mode_transition": transition,
            "uncompensated_residual": uncompensated.tolist(),
        }

    def connector_support_overlap(self, left: PlacedRecord, right: PlacedRecord) -> bool:
        return bool(self.connector_assessment(left, right)["support_reachable"])

    def connector_probability(self, distance: float, bridge_effort: float = 0.0) -> float:
        if not math.isfinite(distance):
            return 0.0
        normalized = max(0.0, distance) / max(self.config.connector_threshold, 1.0e-12)
        return math.exp(-0.5 * (normalized + bridge_effort**2))

    def _command_supported(self, record: ExperienceRecord) -> bool:
        return (
            record.command.mode in self.config.allowed_modes
            and record.command.horizon <= self.config.maximum_horizon
            and float(np.linalg.norm(record.command.terminal_velocity)) <= self.config.maximum_terminal_speed
        )

    def retrieve(self, role: str, key: np.ndarray, context: str, logical_time: float) -> list[ExperienceRecord]:
        key = _array(key, ndim=1)
        # The prefix identifies the fixed directional context family (for
        # example east_nominal and east_dense).  Opposing-direction records are
        # not executable substitutes, while density variants can still share
        # evidence through the registered context backoff.
        family = context.split("_", 1)[0]
        candidates = [
            record for record in self.library.records.values()
            if record.role == role
            and record.context.split("_", 1)[0] == family
            and self.library._context_weight(record.context, context) > 0.0
            and self._command_supported(record)
            and float(np.linalg.norm(record.structural_key - key)) <= self.config.structural_distance_threshold
        ]
        candidates.sort(
            key=lambda record: (
                record.context != context,
                float(np.linalg.norm(record.structural_key - key)),
                -self.library.reliability(record.record_id, context, logical_time),
                record.record_id,
            )
        )
        return candidates[: self.config.retrieval_limit]

    def compose(
        self,
        required_roles: tuple[str, ...],
        anchors: tuple[np.ndarray, ...],
        structural_keys: tuple[np.ndarray, ...],
        *,
        direction: int,
        context: str,
        logical_time: float,
        initial_state: np.ndarray | None = None,
        predecessor: PlacedRecord | None = None,
        predecessor_state: np.ndarray | None = None,
        placement_yaws: tuple[float, ...] | None = None,
    ) -> tuple[list[PlacedRecord], dict[str, Any]]:
        started = time.perf_counter()
        if not (len(required_roles) == len(anchors) == len(structural_keys)):
            raise ValueError("roles, anchors, and structural keys must have equal length")
        if placement_yaws is None:
            placement_yaws = tuple(0.0 if direction > 0 else math.pi for _ in required_roles)
        if len(placement_yaws) != len(required_roles):
            raise ValueError("placement_yaws must match the requested roles")
        query_payload = {
            "roles": list(required_roles),
            "anchors": [np.round(anchor, 7).tolist() for anchor in anchors],
            "keys": [np.round(key, 7).tolist() for key in structural_keys],
            "direction": direction,
            "placement_yaws": [round(float(value), 9) for value in placement_yaws],
            "context": context,
            "logical_time": round(float(logical_time), 9),
            "initial_state": None if initial_state is None else np.round(initial_state, 7).tolist(),
            "predecessor_record_id": None if predecessor is None else predecessor.record.record_id,
            "predecessor_state": (
                None if predecessor_state is None else np.round(predecessor_state, 7).tolist()
            ),
            "library_digest": self.library.digest(),
        }
        query_id = hashlib.sha256(json.dumps(query_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        if len(required_roles) > self.config.max_depth:
            return [], {
                "supported": False,
                "query_id": query_id,
                "reason": "maximum_chain_length",
                "requested_depth": len(required_roles),
                "maximum_depth": self.config.max_depth,
                "expanded": 0,
                "retrieved_candidates": 0,
                "connector_checks": 0,
                "connector_accepted": 0,
                "interface_rejections": 0,
                "edge_audit": [],
                "planning_time_ms": 1000.0 * (time.perf_counter() - started),
            }
        candidate_layers: list[list[PlacedRecord]] = []
        retrieved_candidates = 0
        for role, anchor, key, placement_yaw in zip(
            required_roles, anchors, structural_keys, placement_yaws
        ):
            candidates = [
                self.place(record, anchor, direction, placement_yaw=placement_yaw)
                for record in self.retrieve(role, key, context, logical_time)
            ]
            retrieved_candidates += len(candidates)
            if not candidates:
                return [], {
                    "supported": False,
                    "query_id": query_id,
                    "reason": f"missing_role:{role}",
                    "expanded": 0,
                    "retrieved_candidates": retrieved_candidates,
                    "connector_checks": 0,
                    "connector_accepted": 0,
                    "interface_rejections": 0,
                    "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                    "candidate_record_ids": [
                        [candidate.record.record_id for candidate in layer] for layer in candidate_layers
                    ],
                    "edge_audit": [],
                }
            candidate_layers.append(candidates)

        entry_checks = 0
        entry_rejections = 0
        predecessor_exit_checks = 0
        predecessor_exit_rejections = 0
        connector_checks = 0
        connector_accepted = 0
        edge_audit: list[dict[str, Any]] = []
        initial_connectors: dict[str, tuple[float, float, float]] = {}
        if initial_state is not None:
            eligible: list[PlacedRecord] = []
            for candidate in candidate_layers[0]:
                entry_checks += 1
                if self.entry_distance(initial_state, candidate) <= self.config.entry_threshold:
                    eligible.append(candidate)
                else:
                    entry_rejections += 1
            candidate_layers[0] = eligible
            if not eligible:
                return [], {
                    "supported": False,
                    "query_id": query_id,
                    "reason": "no_entry_interface",
                    "expanded": 0,
                    "retrieved_candidates": retrieved_candidates,
                    "entry_checks": entry_checks,
                    "entry_rejections": entry_rejections,
                    "predecessor_exit_checks": predecessor_exit_checks,
                    "predecessor_exit_rejections": predecessor_exit_rejections,
                    "connector_checks": 0,
                    "connector_accepted": 0,
                    "interface_rejections": entry_rejections,
                    "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                    "candidate_record_ids": [
                        [candidate.record.record_id for candidate in layer] for layer in candidate_layers
                    ],
                    "edge_audit": [],
                }

        if predecessor is not None:
            if predecessor_state is None:
                raise ValueError("predecessor_state is required with predecessor")
            predecessor_exit_checks = 1
            if self.exit_distance(predecessor_state, predecessor) > self.config.entry_threshold:
                predecessor_exit_rejections = 1
                return [], {
                    "supported": False,
                    "query_id": query_id,
                    "reason": "predecessor_exit_gate",
                    "expanded": 0,
                    "retrieved_candidates": retrieved_candidates,
                    "predecessor_exit_checks": predecessor_exit_checks,
                    "predecessor_exit_rejections": predecessor_exit_rejections,
                    "connector_checks": 0,
                    "connector_accepted": 0,
                    "interface_rejections": 1,
                    "candidate_record_ids": [
                        [candidate.record.record_id for candidate in layer]
                        for layer in candidate_layers
                    ],
                    "edge_audit": [],
                    "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                }
            measured_predecessor = copy.deepcopy(predecessor)
            measured_predecessor.exit.mean = _array(predecessor_state, ndim=1).copy()
            # The current state is a measured singleton.  Its estimator
            # uncertainty remains in the covariance term, while finite support
            # must be evaluated from the measured point itself rather than the
            # predecessor record's nominal mean.
            measured_predecessor.exit.support_radius = np.zeros(7, dtype=np.float64)
            measured_predecessor.exit.support_samples = np.asarray(
                (measured_predecessor.exit.mean.copy(),), dtype=np.float64
            )
            measured_predecessor.exit.sample_count = 1
            eligible = []
            for candidate in candidate_layers[0]:
                connector_checks += 1
                assessment = self.connector_assessment(measured_predecessor, candidate)
                connector = self.connector_distance(measured_predecessor, candidate)
                accepted = (
                    bool(assessment["support_reachable"])
                    and bool(assessment["bridge_observed_free"])
                    and connector <= self.config.connector_threshold
                )
                probability = self.connector_probability(
                    connector, float(assessment["bridge_effort"])
                )
                edge_audit.append({
                    "edge_type": "measured_exit_to_entry",
                    "source_record_id": predecessor.record.record_id,
                    "target_record_id": candidate.record.record_id,
                    "direct_support_overlap": bool(assessment["direct_support_overlap"]),
                    "support_reachable": bool(assessment["support_reachable"]),
                    "required_bridge_duration": float(assessment["required_bridge_duration"]),
                    "bridge_effort": float(assessment["bridge_effort"]),
                    "bridge_observed_free": bool(assessment["bridge_observed_free"]),
                    "interaction_modes_compatible": bool(assessment["interaction_modes_compatible"]),
                    "mode_transition": str(assessment["mode_transition"]),
                    "mahalanobis_distance": connector if math.isfinite(connector) else None,
                    "connector_probability": probability,
                    "accepted": accepted,
                    "rejection_reason": None if accepted else "predecessor_connector_gate",
                })
                if accepted:
                    eligible.append(candidate)
                    initial_connectors[candidate.record.record_id] = (
                        connector,
                        probability,
                        float(assessment["bridge_effort"]),
                    )
                    connector_accepted += 1
            candidate_layers[0] = eligible
            if not eligible:
                return [], {
                    "supported": False,
                    "query_id": query_id,
                    "reason": "no_predecessor_connector",
                    "expanded": 0,
                    "retrieved_candidates": retrieved_candidates,
                    "predecessor_exit_checks": predecessor_exit_checks,
                    "predecessor_exit_rejections": predecessor_exit_rejections,
                    "connector_checks": connector_checks,
                    "connector_accepted": connector_accepted,
                    "interface_rejections": connector_checks - connector_accepted,
                    "candidate_record_ids": [
                        [candidate.record.record_id for candidate in layer]
                        for layer in candidate_layers
                    ],
                    "edge_audit": edge_audit,
                    "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                }

        def node_cost(placed: PlacedRecord, reliability: float) -> tuple[float, dict[str, float]]:
            duration = self.library.duration(placed.record.record_id, context, logical_time)
            covariance = placed.exit.covariance + np.eye(7) * self.config.covariance_floor
            _, normalized_logdet = np.linalg.slogdet(
                np.eye(7) + covariance / max(self.config.covariance_floor, 1.0e-12)
            )
            age = max(0.0, logical_time - placed.record.created_time)
            terms = {
                "duration": duration / max(self.config.duration_scale, 1.0e-12),
                "reliability": -self.config.reliability_weight * math.log(max(reliability, 1.0e-12)),
                "uncertainty": self.config.uncertainty_weight * float(normalized_logdet),
                "age": self.config.age_weight * age / max(self.config.age_scale, 1.0e-12),
                "congestion": (
                    self.config.congestion_weight
                    * max(0.0, float(placed.record.predicted_interaction_cost))
                ),
            }
            return sum(terms.values()), terms

        # This is a layered graph, so an admissible A* heuristic is available
        # without learning an additional value function: for every remaining
        # layer, add the least possible record cost and assume zero connector
        # cost.  The bound cannot overestimate a completion because all actual
        # connector and bridge terms are non-negative.  Keeping g and f
        # separate is essential; the previous implementation put g directly
        # on the heap and was therefore bounded Dijkstra rather than the
        # bounded A* search specified by the method.
        layer_node_costs: list[list[float]] = []
        for candidates in candidate_layers:
            costs = []
            for placed in candidates:
                reliability = self.library.reliability(
                    placed.record.record_id, context, logical_time
                )
                total, _ = node_cost(placed, reliability)
                costs.append(total)
            layer_node_costs.append(costs)
        layer_minima = [min(costs) for costs in layer_node_costs]
        optimistic_suffix = [0.0] * len(candidate_layers)
        for layer in range(len(candidate_layers) - 2, -1, -1):
            optimistic_suffix[layer] = optimistic_suffix[layer + 1] + layer_minima[layer + 1]

        # Heap entries are (f = g + h, g, layer, candidate, path, log Q).
        frontier: list[tuple[float, float, int, int, tuple[int, ...], float]] = []
        for index, placed in enumerate(candidate_layers[0]):
            reliability = self.library.reliability(placed.record.record_id, context, logical_time)
            path_cost = layer_node_costs[0][index]
            connector_probability = 1.0
            if predecessor is not None:
                connector, connector_probability, bridge_effort = initial_connectors[
                    placed.record.record_id
                ]
                path_cost += (
                    self.config.connector_weight * connector
                    + self.config.bridge_weight * bridge_effort
                )
            heapq.heappush(
                frontier,
                (
                    path_cost + optimistic_suffix[0],
                    path_cost,
                    0,
                    index,
                    (index,),
                    math.log(max(reliability, 1.0e-12))
                    + math.log(max(connector_probability, 1.0e-12)),
                ),
            )
        expanded = 0
        while frontier and expanded < self.config.graph_node_budget:
            _, path_cost, layer, index, path, log_chain_reliability = heapq.heappop(frontier)
            expanded += 1
            if layer == len(candidate_layers) - 1:
                chain_reliability = math.exp(log_chain_reliability)
                if chain_reliability < self.config.minimum_chain_reliability:
                    continue
                result = [candidate_layers[stage][choice] for stage, choice in enumerate(path)]
                node_breakdown = []
                for placed in result:
                    reliability = self.library.reliability(
                        placed.record.record_id, context, logical_time
                    )
                    node_total, terms = node_cost(placed, reliability)
                    node_breakdown.append({
                        "record_id": placed.record.record_id,
                        "reliability_lcb": reliability,
                        "total": node_total,
                        **terms,
                    })
                return result, {
                    "supported": True,
                    "query_id": query_id,
                    "reason": "ok",
                    "search_algorithm": "bounded_a_star",
                    "graph_node_budget": self.config.graph_node_budget,
                    "expanded": expanded,
                    "cost": path_cost,
                    "retrieved_candidates": retrieved_candidates,
                    "entry_checks": entry_checks,
                    "entry_rejections": entry_rejections,
                    "predecessor_exit_checks": predecessor_exit_checks,
                    "predecessor_exit_rejections": predecessor_exit_rejections,
                    "connector_checks": connector_checks,
                    "connector_accepted": connector_accepted,
                    "interface_rejections": entry_rejections + connector_checks - connector_accepted,
                    "selected_chain": [placed.record.record_id for placed in result],
                    "selected_origins": [placed.record.origin_robot for placed in result],
                    "selected_chain_reliability": chain_reliability,
                    "minimum_chain_reliability": self.config.minimum_chain_reliability,
                    "node_cost_breakdown": node_breakdown,
                    "candidate_record_ids": [
                        [candidate.record.record_id for candidate in candidates] for candidates in candidate_layers
                    ],
                    "edge_audit": edge_audit,
                    "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                }
            left = candidate_layers[layer][index]
            for next_index, right in enumerate(candidate_layers[layer + 1]):
                connector_checks += 1
                assessment = self.connector_assessment(left, right)
                support_reachable = bool(assessment["support_reachable"])
                bridge_observed_free = bool(assessment["bridge_observed_free"])
                connector = self.connector_distance(left, right)
                accepted = (
                    support_reachable
                    and bridge_observed_free
                    and connector <= self.config.connector_threshold
                )
                connector_probability = self.connector_probability(
                    connector,
                    float(assessment["bridge_effort"]),
                )
                if len(edge_audit) < self.config.edge_audit_limit:
                    if accepted:
                        rejection_reason = None
                    elif not bool(assessment["interaction_modes_compatible"]):
                        rejection_reason = "interaction_mode"
                    elif not support_reachable:
                        rejection_reason = "finite_support_or_bridge"
                    elif not bridge_observed_free:
                        rejection_reason = "unobserved_bridge"
                    else:
                        rejection_reason = "mahalanobis_gate"
                    edge_audit.append(
                        {
                            "source_record_id": left.record.record_id,
                            "target_record_id": right.record.record_id,
                            "direct_support_overlap": bool(assessment["direct_support_overlap"]),
                            "support_reachable": support_reachable,
                            "required_bridge_duration": float(assessment["required_bridge_duration"]),
                            "bridge_effort": float(assessment["bridge_effort"]),
                            "bridge_observed_free": bridge_observed_free,
                            "interaction_modes_compatible": bool(assessment["interaction_modes_compatible"]),
                            "mode_transition": str(assessment["mode_transition"]),
                            "mahalanobis_distance": connector if math.isfinite(connector) else None,
                            "connector_probability": connector_probability,
                            "accepted": accepted,
                            "rejection_reason": rejection_reason,
                        }
                    )
                if not accepted:
                    continue
                connector_accepted += 1
                reliability = self.library.reliability(right.record.record_id, context, logical_time)
                record_cost = layer_node_costs[layer + 1][next_index]
                next_path_cost = (
                    path_cost
                    + record_cost
                    + self.config.connector_weight * connector
                    + self.config.bridge_weight * float(assessment["bridge_effort"])
                )
                next_log_reliability = (
                    log_chain_reliability
                    + math.log(max(reliability, 1.0e-12))
                    + math.log(max(connector_probability, 1.0e-12))
                )
                heapq.heappush(
                    frontier,
                    (
                        next_path_cost + optimistic_suffix[layer + 1],
                        next_path_cost,
                        layer + 1,
                        next_index,
                        path + (next_index,),
                        next_log_reliability,
                    ),
                )
        budget_exhausted = bool(frontier) and expanded >= self.config.graph_node_budget
        return [], {
            "supported": False,
            "query_id": query_id,
            "reason": "graph_node_budget_exhausted" if budget_exhausted else "no_compatible_path",
            "search_algorithm": "bounded_a_star",
            "graph_node_budget": self.config.graph_node_budget,
            "frontier_remaining": len(frontier),
            "expanded": expanded,
            "retrieved_candidates": retrieved_candidates,
            "entry_checks": entry_checks,
            "entry_rejections": entry_rejections,
            "predecessor_exit_checks": predecessor_exit_checks,
            "predecessor_exit_rejections": predecessor_exit_rejections,
            "connector_checks": connector_checks,
            "connector_accepted": connector_accepted,
            "interface_rejections": entry_rejections + connector_checks - connector_accepted,
            "minimum_chain_reliability": self.config.minimum_chain_reliability,
            "candidate_record_ids": [
                [candidate.record.record_id for candidate in candidates] for candidates in candidate_layers
            ],
            "edge_audit": edge_audit,
            "planning_time_ms": 1000.0 * (time.perf_counter() - started),
        }
