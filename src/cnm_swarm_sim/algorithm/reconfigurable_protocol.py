"""Registered unseen-order and interface-shift tasks for formal CNM tests.

The acquisition scene always uses the canonical A-B-C-D ordering.  This
module deterministically constructs all 24 permutations in new placements.
No task trajectory is used to form a memory record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .protocol import MODULES, ROLE_ORDER


@dataclass(frozen=True)
class ReconfigurableTask:
    task_id: str
    roles: tuple[str, ...]
    anchors: tuple[np.ndarray, ...]
    placement_yaws: tuple[float, ...]
    structural_keys: tuple[np.ndarray, ...]
    start: np.ndarray
    goal: np.ndarray
    checkpoints: tuple[np.ndarray, ...]
    interface_shift_class: str
    interface_position_shifts: tuple[np.ndarray, ...]
    acquisition_order_seen: bool = False
    # Geometry is represented by the structural key and placement transform.
    # Context remains nominal because illumination, payload and traffic class
    # are unchanged in this individual test.
    context: str = "east_nominal"
    direction: int = 1

    @property
    def length(self) -> int:
        return len(self.roles)


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def canonical_module_points(role: str) -> np.ndarray:
    definition = MODULES[role]
    return definition.world_waypoints(1) - np.asarray(definition.anchor, dtype=np.float64)


def placed_module_points(role: str, anchor: np.ndarray, yaw: float) -> np.ndarray:
    return np.asarray(anchor, dtype=np.float64) + (
        _rotation(yaw) @ canonical_module_points(role).T
    ).T


def _task_from_order(index: int, roles: tuple[str, ...]) -> ReconfigurableTask:
    # Four registered shift families isolate translation, yaw and their joint
    # effect.  Magnitudes remain inside the frozen command and bridge domains.
    shift_class = ("new_placement", "position_shift", "yaw_shift", "combined")[index % 4]
    yaw_amplitude = 0.0 if shift_class in {"new_placement", "position_shift"} else math.radians(1.5)
    lateral_amplitude = 0.0 if shift_class in {"new_placement", "yaw_shift"} else 0.055
    vertical_amplitude = 0.0 if shift_class == "new_placement" else 0.025
    yaws: list[float] = [0.0]
    yaw_pattern = (1.0, -1.0, 0.5)
    for module_index in range(1, len(roles)):
        previous_points = canonical_module_points(roles[module_index - 1])
        next_points = canonical_module_points(roles[module_index])
        previous_tangent = previous_points[-1] - previous_points[-2]
        next_tangent = next_points[1] - next_points[0]
        previous_exit_yaw = math.atan2(float(previous_tangent[1]), float(previous_tangent[0]))
        next_entry_yaw = math.atan2(float(next_tangent[1]), float(next_tangent[0]))
        yaws.append(
            yaws[-1] + previous_exit_yaw - next_entry_yaw
            + yaw_amplitude * yaw_pattern[module_index - 1]
        )

    first_local = canonical_module_points(roles[0])[0]
    desired_entry = np.asarray((-9.65, -1.10 + 0.18 * ((index % 3) - 1), 1.52))
    first_anchor = desired_entry - _rotation(yaws[0]) @ first_local
    anchors: list[np.ndarray] = [first_anchor]
    shifts: list[np.ndarray] = []
    all_points: list[np.ndarray] = []

    for module_index, role in enumerate(roles):
        points = placed_module_points(role, anchors[-1], yaws[module_index])
        all_points.append(points)
        if module_index == len(roles) - 1:
            continue
        shift = np.asarray((
            0.025,
            lateral_amplitude * (1.0 if (index + module_index) % 2 == 0 else -1.0),
            vertical_amplitude * (1.0 if module_index % 2 == 0 else -1.0),
        ))
        if shift_class == "new_placement":
            shift[:] = 0.0
        shifts.append(shift)
        next_role = roles[module_index + 1]
        next_local_entry = canonical_module_points(next_role)[0]
        next_entry = points[-1] + shift
        anchors.append(next_entry - _rotation(yaws[module_index + 1]) @ next_local_entry)

    checkpoints = tuple(
        point.copy()
        for points in all_points
        for point in points[1:]
    )
    # Explicit labels make A+C+B identifiable in every source table.
    labels = {role: chr(65 + ROLE_ORDER.index(role)) for role in ROLE_ORDER}
    ordering = "+".join(labels[role] for role in roles)
    return ReconfigurableTask(
        task_id=f"R{index + 1:02d}_{ordering.replace('+', '')}_{shift_class}",
        roles=roles,
        anchors=tuple(np.asarray(anchor, dtype=np.float64) for anchor in anchors),
        placement_yaws=tuple(yaws),
        structural_keys=tuple(np.asarray(MODULES[role].structural_key, dtype=np.float64) for role in roles),
        start=all_points[0][0].copy(),
        goal=all_points[-1][-1].copy(),
        checkpoints=checkpoints,
        interface_shift_class=shift_class,
        interface_position_shifts=tuple(shifts),
        # Acquisition episodes contain one module each; no complete ordering,
        # including the canonical role order, is ever executed in full.
        acquisition_order_seen=False,
    )


def reconfigurable_tasks(*, include_canonical_order: bool = False) -> tuple[ReconfigurableTask, ...]:
    """Return the registered four-module permutation protocol.

    Six two-module, eight three-module and four four-module compositions probe
    capability growth (including an explicit A+C+B probe).  Six further tasks
    use canonical role order at new placements/interface shifts.
    Every complete route is absent from the single-module acquisition set.
    """

    a, b, c, d = ROLE_ORDER
    growth_orders = (
        (a, b), (a, c), (b, a), (b, c), (c, a), (c, b),
        (a, b, c), (a, c, b), (b, a, c), (b, c, a),
        (c, a, b), (c, b, a), (a, b, d), (c, d, b),
        (a, c, b, d), (b, a, d, c), (c, d, a, b), (d, c, b, a),
    )
    tasks = [_task_from_order(index, order) for index, order in enumerate(growth_orders)]
    explicit_index = next(index for index, task in enumerate(tasks) if task.roles == (a, c, b))
    explicit = tasks[explicit_index]
    tasks[explicit_index] = ReconfigurableTask(
        task_id="R08_ACB_explicit",
        roles=explicit.roles,
        anchors=explicit.anchors,
        placement_yaws=explicit.placement_yaws,
        structural_keys=explicit.structural_keys,
        start=explicit.start,
        goal=explicit.goal,
        checkpoints=explicit.checkpoints,
        interface_shift_class="explicit_A+C+B",
        interface_position_shifts=explicit.interface_position_shifts,
        acquisition_order_seen=False,
    )
    for variant in range(6):
        canonical = _task_from_order(40 + variant, ROLE_ORDER)
        tasks.append(ReconfigurableTask(
            task_id=f"R{19 + variant:02d}_ABCD_{canonical.interface_shift_class}",
            roles=canonical.roles,
            anchors=canonical.anchors,
            placement_yaws=canonical.placement_yaws,
            structural_keys=canonical.structural_keys,
            start=canonical.start,
            goal=canonical.goal,
            checkpoints=canonical.checkpoints,
            interface_shift_class=canonical.interface_shift_class,
            interface_position_shifts=canonical.interface_position_shifts,
            acquisition_order_seen=False,
        ))
    if len(tasks) != 24:
        raise AssertionError("registered reconfigurable task count changed")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise AssertionError("reconfigurable task identifiers must be unique")
    return tuple(tasks)


def task_by_id(task_id: str) -> ReconfigurableTask:
    for task in reconfigurable_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)
