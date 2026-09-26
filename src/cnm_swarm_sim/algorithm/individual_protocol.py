"""Registered atomic responses and held-out individual CNM compositions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .memory import CommandTemplate, ExperienceRecord, InterfaceSummary, PlacedRecord


@dataclass(frozen=True)
class AtomicResponseDefinition:
    role: str
    anchor: tuple[float, float, float]
    eastbound_waypoints: tuple[tuple[float, float, float], ...]
    westbound_waypoints: tuple[tuple[float, float, float], ...]
    structural_key: tuple[float, ...]

    def world_waypoints(self, direction: int) -> np.ndarray:
        return np.asarray(self.eastbound_waypoints if direction > 0 else self.westbound_waypoints, dtype=np.float64)


@dataclass(frozen=True)
class HeldOutComposition:
    task_id: str
    direction: int
    context: str
    roles: tuple[str, ...]
    anchors: tuple[np.ndarray, ...]
    structural_keys: tuple[np.ndarray, ...]
    start: np.ndarray
    goal: np.ndarray
    checkpoints: tuple[np.ndarray, ...]
    length: int
    perturbation_class: str


# The six responses partition the physical east-west corridor.  Acquisition
# executes every item alone.  No pair or longer ordering below is ever used to
# form a record.
ATOMIC_RESPONSES: dict[str, AtomicResponseDefinition] = {
    "door_A": AtomicResponseDefinition(
        "door_A", (-7.40, -2.35, 1.45),
        ((-9.70, -3.05, 1.45), (-8.15, -3.05, 1.45), (-6.72, -3.05, 1.45), (-5.72, -3.04, 1.50)),
        ((-5.72, -1.65, 1.48), (-6.72, -1.65, 1.45), (-8.15, -1.65, 1.45), (-9.70, -1.65, 1.45)),
        (2.0, 1.0, 0.0, 1.10, 2.96, 7.40),
    ),
    "guard_B": AtomicResponseDefinition(
        "guard_B", (-4.25, -2.35, 1.52),
        ((-5.72, -3.04, 1.50), (-4.45, -2.90, 1.52), (-2.80, -2.25, 1.55)),
        ((-2.80, -1.65, 1.55), (-4.35, -1.65, 1.50), (-5.72, -1.65, 1.48)),
        (2.0, 0.0, 1.0, 3.14, 2.90, 0.68),
    ),
    "split_C": AtomicResponseDefinition(
        "split_C", (0.0, 0.0, 1.55),
        ((-2.80, -2.25, 1.55), (-1.20, -2.25, 1.55), (1.40, -2.25, 1.55), (1.40, 1.10, 1.55)),
        ((1.40, 1.10, 1.55), (1.40, 2.25, 1.55), (-1.20, 2.25, 1.55), (-2.80, -1.65, 1.55)),
        (0.0, 1.0, 0.0, 3.10, 3.70, 0.56),
    ),
    "weave_D": AtomicResponseDefinition(
        "weave_D", (2.55, 0.75, 1.50),
        ((1.40, 1.10, 1.55), (2.30, 1.10, 1.45), (3.60, 0.55, 1.45)),
        ((3.60, 0.55, 1.45), (2.30, 1.10, 1.45), (1.40, 1.10, 1.55)),
        (0.0, 0.0, 1.0, 2.20, 1.80, 0.44),
    ),
    "rise_E": AtomicResponseDefinition(
        "rise_E", (4.25, 0.25, 1.65),
        ((3.60, 0.55, 1.45), (4.25, 0.25, 1.62), (4.90, 0.00, 1.85)),
        ((4.90, 0.00, 1.85), (4.25, 0.25, 1.62), (3.60, 0.55, 1.45)),
        (0.0, 0.0, 1.0, 1.30, 1.20, 0.40),
    ),
    "window_F": AtomicResponseDefinition(
        "window_F", (6.0, 0.0, 2.05),
        ((4.90, 0.00, 1.85), (5.55, 0.00, 2.05), (6.45, 0.00, 2.05), (8.60, 0.00, 1.55)),
        ((8.60, 0.00, 1.55), (6.45, 0.00, 2.05), (5.55, 0.00, 2.05), (4.90, 0.00, 1.85)),
        (0.0, 0.0, 0.0, 8.10, 2.63, 2.05),
    ),
}

ATOM_ORDER = tuple(ATOMIC_RESPONSES)


def acquisition_world_waypoints(role: str, direction: int, route_variant: int = 0) -> np.ndarray:
    """Return one independently executable lane variant for local acquisition."""

    definition = ATOMIC_RESPONSES[role]
    nominal = definition.world_waypoints(direction).copy()
    order = ATOM_ORDER if direction > 0 else tuple(reversed(ATOM_ORDER))
    role_index = order.index(role)
    if route_variant == 2 and role_index > 0:
        # Interface-matched acquisition is still a separate episode: only the
        # current local response is executed, but it starts at the measured
        # geometric exit registered for the preceding response class.  This
        # supplies physical entry evidence without exposing a held-out chain.
        predecessor = ATOMIC_RESPONSES[order[role_index - 1]]
        nominal[0] = predecessor.world_waypoints(direction)[-1]
        return nominal
    # The first three responses have two physically valid passages in the
    # industrial scene.  Reversing the opposing-direction trace supplies the
    # complementary lane without exposing any held-out multi-response task.
    if route_variant == 1 and role in {"door_A", "guard_B", "split_C"}:
        return definition.world_waypoints(-direction)[::-1].copy()
    return nominal


def _canonicalize(points: np.ndarray, anchor: np.ndarray, direction: int) -> np.ndarray:
    yaw = 0.0 if direction > 0 else np.pi
    c, s = np.cos(yaw), np.sin(yaw)
    inverse = np.asarray(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
    return np.stack([inverse @ (point - anchor) for point in points])


def acquisition_path(
    role: str,
    direction: int,
    *,
    immutable_hash: str,
    route_id: str,
    route_variant: int = 0,
) -> list[PlacedRecord]:
    """Return a collection scaffold; none of its interface data is retained."""

    definition = ATOMIC_RESPONSES[role]
    anchor = np.asarray(definition.anchor, dtype=np.float64)
    world = acquisition_world_waypoints(role, direction, route_variant)
    canonical = _canonicalize(world, anchor, direction)
    entry_velocity = np.asarray((0.75, 0.0, 0.0), dtype=np.float64)
    state_in = np.concatenate((canonical[0], entry_velocity, (0.0,)))
    state_out = np.concatenate((canonical[-1], entry_velocity, (0.0,)))
    covariance = np.eye(7) * 0.5
    support = np.ones(7) * 2.0
    record_id = hashlib.sha256(f"collection:{role}:{direction}:{route_id}".encode()).hexdigest()[:24]
    record = ExperienceRecord(
        record_id=record_id,
        structural_key=np.asarray(definition.structural_key),
        role=role,
        context="collection_only",
        entry=InterfaceSummary(state_in, covariance.copy(), support.copy(), 1),
        command=CommandTemplate(canonical[1:], entry_velocity, 0.0, "progress", 4.0),
        exit=InterfaceSummary(state_out, covariance.copy(), support.copy(), 1),
        origin_robot="collection_scaffold",
        first_event_id="none",
        immutable_hash=immutable_hash,
    )
    # Placing the direction-specific canonical scaffold reconstructs the exact
    # registered physical route.  The compiled record is created independently.
    yaw = 0.0 if direction > 0 else np.pi
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    placed_entry = InterfaceSummary(
        np.concatenate((world[0], rotation @ entry_velocity, (yaw,))), covariance.copy(), support.copy(), 1
    )
    placed_exit = InterfaceSummary(
        np.concatenate((world[-1], rotation @ entry_velocity, (yaw,))), covariance.copy(), support.copy(), 1
    )
    return [PlacedRecord(record, anchor, direction, placed_entry, placed_exit, world[1:])]


def held_out_compositions() -> tuple[HeldOutComposition, ...]:
    """Return 30 unique multi-response tasks absent from acquisition."""

    tasks: list[HeldOutComposition] = []
    for direction, prefix in ((1, "E"), (-1, "W")):
        order = ATOM_ORDER if direction > 0 else tuple(reversed(ATOM_ORDER))
        context = "east_nominal" if direction > 0 else "west_nominal"
        task_index = 0
        for length in range(2, len(order) + 1):
            for start_index in range(0, len(order) - length + 1):
                roles = order[start_index : start_index + length]
                definitions = [ATOMIC_RESPONSES[role] for role in roles]
                first_points = definitions[0].world_waypoints(direction)
                last_points = definitions[-1].world_waypoints(direction)
                checkpoints = tuple(
                    point.copy()
                    for definition in definitions
                    for point in definition.world_waypoints(direction)[1:-1]
                ) + (last_points[-1].copy(),)
                tasks.append(
                    HeldOutComposition(
                        task_id=f"{prefix}_{'-'.join(roles)}",
                        direction=direction,
                        context=context,
                        roles=roles,
                        anchors=tuple(np.asarray(item.anchor, dtype=np.float64) for item in definitions),
                        structural_keys=tuple(np.asarray(item.structural_key, dtype=np.float64) for item in definitions),
                        start=first_points[0].copy(),
                        goal=last_points[-1].copy(),
                        checkpoints=checkpoints,
                        length=length,
                        # Sensing/dynamics perturbations belong to the EEF
                        # robustness experiment.  This registered individual
                        # growth test isolates unseen ordering/composition.
                        perturbation_class="nominal",
                    )
                )
                task_index += 1
    if len(tasks) != 30 or len({task.task_id for task in tasks}) != 30:
        raise AssertionError("the individual benchmark must contain 30 unique compositions")
    return tuple(tasks)
