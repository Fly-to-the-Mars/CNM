"""Composable continuous 3-D scenarios used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import EnvConfig


@dataclass(frozen=True)
class BoxObstacle:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    module_id: str
    color: tuple[float, float, float, float]
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: bool = True


@dataclass(frozen=True)
class CylinderObstacle:
    center: tuple[float, float, float]
    radius: float
    height: float
    module_id: str
    color: tuple[float, float, float, float]
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: bool = True


@dataclass(frozen=True)
class SphereObstacle:
    center: tuple[float, float, float]
    radius: float
    module_id: str
    color: tuple[float, float, float, float]
    visible: bool = True


ScenePrimitive = BoxObstacle | CylinderObstacle | SphereObstacle

TWIN_GATE_X = -7.40
TWIN_GATE_CENTERS = (-3.05, -1.65)
TWIN_GATE_WIDTH = 1.10

LAYOUT_VARIANTS = (
    "canonical",
    "warehouse",
    "column_grove",
    "suspended",
    "offset_bays",
    "high_bay",
)


@dataclass(frozen=True)
class ScenarioSpec:
    starts: np.ndarray
    goals: np.ndarray
    initial_yaws: np.ndarray
    group_ids: np.ndarray
    obstacles: tuple[ScenePrimitive, ...]
    decorations: tuple[ScenePrimitive, ...] = ()
    module_sequence: tuple[str, ...] = ()
    waypoint_sequences: tuple[np.ndarray, ...] = ()
    dynamic_motions: tuple["DynamicObstacleMotion", ...] = ()


@dataclass(frozen=True)
class DynamicObstacleMotion:
    """Registered sinusoidal motion for one collidable scene primitive."""

    obstacle_index: int
    axis: tuple[float, float, float]
    amplitude: float
    period_seconds: float
    phase_rad: float = 0.0


def _eef_lane_offsets(count: int) -> np.ndarray:
    """Deterministic concurrent-flight lanes with safe initial separation."""

    if count == 1:
        return np.zeros(1, dtype=np.float64)
    # Keep the inter-lane spacing fixed as team size changes.  Using fixed
    # outer bounds made the two-robot condition start on the most extreme
    # routes, confounding team size with route difficulty.
    spacing = 0.58
    return (np.arange(count, dtype=np.float64) - 0.5 * (count - 1)) * spacing


def _eef_waypoint_sequences(
    base: tuple[tuple[float, float, float], ...], count: int, *, lane_scale: float = 1.0
) -> tuple[np.ndarray, ...]:
    offsets = _eef_lane_offsets(count) * lane_scale
    return tuple(
        np.asarray(base, dtype=np.float64) + np.asarray((0.0, offset, 0.0))
        for offset in offsets
    )


def _eef_boundaries(cfg: EnvConfig, *, color: tuple[float, float, float, float]) -> list[ScenePrimitive]:
    x_half = cfg.arena_length / 2.0
    y_half = cfg.arena_width / 2.0
    z_half = cfg.arena_height / 2.0
    return [
        BoxObstacle((0.0, y_half + 0.08, z_half), (x_half, 0.08, z_half), "boundary", color),
        BoxObstacle((0.0, -y_half - 0.08, z_half), (x_half, 0.08, z_half), "boundary", color),
        BoxObstacle((x_half + 0.08, 0.0, z_half), (0.08, y_half, z_half), "boundary", color),
        BoxObstacle((-x_half - 0.08, 0.0, z_half), (0.08, y_half, z_half), "boundary", color),
    ]


def eef_indoor_static(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Structured indoor clutter with alternating free-space interfaces."""

    del rng
    count = cfg.num_drones
    lanes = _eef_lane_offsets(count) * 0.72
    starts = np.column_stack((np.full(count, -5.7), lanes - 0.55, np.full(count, 1.35)))
    goals = np.column_stack((np.full(count, 5.7), lanes + 0.45, np.full(count, 1.35)))
    base = (
        (-5.7, -0.55, 1.35), (-3.8, -0.55, 1.35), (-2.2, 0.55, 1.40),
        (-0.4, 0.55, 1.40), (1.4, -0.45, 1.32), (3.4, -0.45, 1.35),
        (5.7, 0.45, 1.35),
    )
    pale = (0.78, 0.82, 0.86, 1.0)
    steel = (0.45, 0.50, 0.55, 1.0)
    obstacles: list[ScenePrimitive] = _eef_boundaries(cfg, color=(0.88, 0.90, 0.92, 1.0))
    obstacles.extend(
        (
            BoxObstacle((-2.9, -3.02, 1.50), (0.22, 0.90, 1.50), "indoor_partition", pale),
            BoxObstacle((-2.9, 3.02, 1.50), (0.22, 0.90, 1.50), "indoor_partition", pale),
            BoxObstacle((0.55, -3.02, 1.50), (0.22, 0.90, 1.50), "indoor_partition", pale),
            BoxObstacle((0.55, 3.02, 1.50), (0.22, 0.90, 1.50), "indoor_partition", pale),
            CylinderObstacle((2.35, 1.55, 1.45), 0.32, 2.90, "indoor_column", steel),
            CylinderObstacle((3.05, -1.55, 1.45), 0.32, 2.90, "indoor_column", steel),
            SphereObstacle((4.05, 1.75, 1.55), 0.42, "indoor_clutter", (0.68, 0.73, 0.77, 1.0)),
        )
    )
    return ScenarioSpec(
        starts, goals, np.zeros(count), np.zeros(count, dtype=np.int8), tuple(obstacles),
        module_sequence=("indoor_static",),
        waypoint_sequences=_eef_waypoint_sequences(base, count, lane_scale=0.72),
    )


def eef_indoor_dynamic(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Indoor passage with independently moving lateral obstacles."""

    base_spec = eef_indoor_static(cfg, rng)
    obstacles = list(base_spec.obstacles)
    dynamic_start = len(obstacles)
    obstacles.extend(
        (
            CylinderObstacle((-1.25, -0.85, 1.35), 0.20, 2.70, "dynamic_crossing", (0.93, 0.42, 0.16, 1.0)),
            SphereObstacle((2.15, 0.75, 1.35), 0.24, "dynamic_crossing", (0.94, 0.56, 0.22, 1.0)),
        )
    )
    phase_offset = float(rng.uniform(0.0, 2.0 * np.pi))
    motions = (
        DynamicObstacleMotion(dynamic_start, (0.0, 1.0, 0.0), 0.95, 5.6, phase_offset),
        DynamicObstacleMotion(dynamic_start + 1, (0.0, 1.0, 0.0), 1.05, 4.8, phase_offset + np.pi),
    )
    return replace(
        base_spec,
        obstacles=tuple(obstacles),
        module_sequence=("indoor_dynamic",),
        dynamic_motions=motions,
    )


def eef_garage(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Low-contrast garage aisle with offset columns and overhead beams."""

    del rng
    count = cfg.num_drones
    lanes = _eef_lane_offsets(count) * 0.70
    starts = np.column_stack((np.full(count, -5.8), lanes - 0.65, np.full(count, 1.18)))
    goals = np.column_stack((np.full(count, 5.8), lanes + 0.55, np.full(count, 1.18)))
    base = (
        (-5.8, -0.65, 1.18), (-4.1, -0.65, 1.18), (-2.5, 0.55, 1.22),
        (-0.8, 0.55, 1.18), (0.9, -0.55, 1.20), (2.6, -0.55, 1.18),
        (4.2, 0.55, 1.20), (5.8, 0.55, 1.18),
    )
    concrete = (0.63, 0.65, 0.66, 1.0)
    dark = (0.36, 0.38, 0.40, 1.0)
    obstacles: list[ScenePrimitive] = _eef_boundaries(cfg, color=(0.72, 0.73, 0.74, 1.0))
    for x, y in ((-3.25, -1.75), (-2.25, 1.75), (-0.25, -1.75), (0.75, 1.75), (2.75, -1.75), (3.75, 1.75)):
        obstacles.append(BoxObstacle((x, y, 1.35), (0.34, 0.34, 1.35), "garage_column", concrete))
    for x in (-4.4, 0.0, 4.4):
        obstacles.append(BoxObstacle((x, 0.0, 2.65), (0.16, 3.6, 0.16), "garage_beam", dark))
    return ScenarioSpec(
        starts, goals, np.zeros(count), np.zeros(count, dtype=np.int8), tuple(obstacles),
        module_sequence=("garage",),
        waypoint_sequences=_eef_waypoint_sequences(base, count, lane_scale=0.70),
    )


def eef_forest(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Seeded natural-clutter surrogate with irregular trunk placement."""

    count = cfg.num_drones
    lanes = _eef_lane_offsets(count) * 0.68
    starts = np.column_stack((np.full(count, -5.8), lanes - 0.35, np.full(count, 1.45)))
    goals = np.column_stack((np.full(count, 5.8), lanes + 0.35, np.full(count, 1.45)))
    base = (
        (-5.8, -0.35, 1.45), (-4.0, -0.35, 1.45), (-2.4, 0.55, 1.48),
        (-0.7, -0.45, 1.42), (1.0, 0.50, 1.48), (2.8, -0.45, 1.43),
        (4.25, 0.35, 1.46), (5.8, 0.35, 1.45),
    )
    trunk = (0.39, 0.28, 0.20, 1.0)
    leaf = (0.34, 0.52, 0.37, 1.0)
    obstacles: list[ScenePrimitive] = _eef_boundaries(cfg, color=(0.77, 0.82, 0.75, 1.0))
    trunk_xy = [
        (-4.55, -2.20), (-3.55, 1.90), (-2.75, -1.45), (-1.75, 2.15),
        (-0.65, -1.85), (0.25, 1.85), (1.35, -1.70), (2.10, 2.15),
        (3.05, -1.75), (3.85, 1.85), (4.65, -2.10),
    ]
    for index, (x, y) in enumerate(trunk_xy):
        jitter = rng.uniform(-0.10, 0.10, size=2)
        radius = 0.20 + 0.06 * (index % 3)
        obstacles.append(CylinderObstacle((x + jitter[0], y + jitter[1], 1.65), radius, 3.30, "forest_trunk", trunk))
        if index % 2 == 0:
            obstacles.append(SphereObstacle((x + jitter[0], y + jitter[1], 3.10), 0.58, "forest_canopy", leaf))
    return ScenarioSpec(
        starts, goals, np.zeros(count), np.zeros(count, dtype=np.int8), tuple(obstacles),
        module_sequence=("forest",),
        waypoint_sequences=_eef_waypoint_sequences(base, count, lane_scale=0.68),
    )


def _vertical_grid(count: int, x: float, cfg: EnvConfig) -> np.ndarray:
    """Place a compact group without initial overlap."""

    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    n_y = min(10, max(2, int(np.ceil(np.sqrt(2.0 * count)))))
    n_z = int(np.ceil(count / n_y))
    y = np.linspace(-0.40 * cfg.arena_width, 0.40 * cfg.arena_width, n_y)
    z = 0.70 + 0.36 * np.arange(n_z)
    points = np.array([(x, yy, zz) for zz in z for yy in y], dtype=np.float64)
    return points[:count]


def bidirectional_bottleneck(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Two complementary groups exchange sides through one shared bottleneck."""

    n_left = (cfg.num_drones + 1) // 2
    n_right = cfg.num_drones - n_left
    x_spawn = 0.39 * cfg.arena_length
    left = _vertical_grid(n_left, -x_spawn, cfg)
    right = _vertical_grid(n_right, x_spawn, cfg)
    starts = np.vstack((left, right))

    # Small seeded offsets break exact symmetries without changing task identity.
    starts[:, 1:] += rng.uniform(-0.015, 0.015, size=(cfg.num_drones, 2))
    goals = starts.copy()
    goals[:n_left, 0] = x_spawn
    goals[n_left:, 0] = -x_spawn
    goals[:n_left, 1:] = left[::-1, 1:]
    goals[n_left:, 1:] = right[::-1, 1:]

    y_outer = cfg.arena_width / 2.0
    wall_half_y = (cfg.arena_width - cfg.bottleneck_width) / 4.0
    wall_center_y = cfg.bottleneck_width / 2.0 + wall_half_y
    z_half = cfg.arena_height / 2.0
    wall_color = (0.40, 0.46, 0.55, 1.0)
    boundary_color = (0.76, 0.79, 0.83, 1.0)
    obstacles = (
        BoxObstacle((0.0, wall_center_y, z_half), (0.12, wall_half_y, z_half), "bottleneck_A", wall_color),
        BoxObstacle((0.0, -wall_center_y, z_half), (0.12, wall_half_y, z_half), "bottleneck_A", wall_color),
        BoxObstacle((0.0, y_outer + 0.08, z_half), (cfg.arena_length / 2.0, 0.08, z_half), "boundary", boundary_color),
        BoxObstacle((0.0, -y_outer - 0.08, z_half), (cfg.arena_length / 2.0, 0.08, z_half), "boundary", boundary_color),
    )
    initial_yaws = np.concatenate((np.zeros(n_left), np.full(n_right, np.pi)))
    group_ids = np.concatenate((np.zeros(n_left, dtype=np.int8), np.ones(n_right, dtype=np.int8)))
    return ScenarioSpec(
        starts,
        goals,
        initial_yaws,
        group_ids,
        obstacles,
        module_sequence=("bottleneck_A",),
    )


def _industrial_decorations(cfg: EnvConfig) -> tuple[ScenePrimitive, ...]:
    """A restrained technical layer for selected publication frames.

    These parts never participate in collision or range sensing.  They add
    scale, module identity and depth without increasing the cost of training
    and benchmarking, where ``scene_decorations`` remains disabled.
    """

    pieces: list[ScenePrimitive] = []
    grid = (0.79, 0.82, 0.85, 0.56)
    grid_major = (0.66, 0.71, 0.76, 0.74)
    bay_fill = (0.83, 0.88, 0.91, 0.20)
    blue = (0.25, 0.55, 0.75, 0.82)
    cyan = (0.25, 0.66, 0.70, 0.76)
    graphite = (0.42, 0.47, 0.51, 0.86)
    silver = (0.69, 0.73, 0.76, 0.84)

    # Fine floor tiles are interrupted by a stronger central datum and by four
    # subtly tinted module bays.  All lines sit above the floor to avoid z-fight.
    for x in np.arange(-13.0, 13.01, 1.0):
        major = int(round(x)) % 4 == 0
        pieces.append(
            BoxObstacle(
                (float(x), 0.0, 0.006 if not major else 0.008),
                (0.008 if not major else 0.016, 6.82, 0.004),
                "visual_grid",
                grid_major if major else grid,
            )
        )
    for y in np.arange(-6.0, 6.01, 1.0):
        major = int(round(y)) % 4 == 0
        pieces.append(
            BoxObstacle(
                (0.0, float(y), 0.007 if not major else 0.009),
                (13.82, 0.008 if not major else 0.016, 0.004),
                "visual_grid",
                grid_major if major else grid,
            )
        )

    bay_specs = (
        (-4.95, 3.75, 5.70, blue),
        (0.0, 1.40, 4.50, cyan),
        (3.35, 1.35, 5.05, blue),
        (6.05, 1.20, 5.25, cyan),
    )
    for bay_index, (x, hx, hy, accent) in enumerate(bay_specs):
        module = "visual_bay_{0}".format(bay_index)
        pieces.append(BoxObstacle((x, 0.0, 0.010), (hx, hy, 0.003), module, bay_fill))
        pieces.extend(
            (
                BoxObstacle((x - hx, 0.0, 0.014), (0.018, hy, 0.006), module, accent),
                BoxObstacle((x + hx, 0.0, 0.014), (0.018, hy, 0.006), module, accent),
                BoxObstacle((x, -hy, 0.014), (hx, 0.018, 0.006), module, accent),
                BoxObstacle((x, hy, 0.014), (hx, 0.018, 0.006), module, accent),
            )
        )

    # Two slim data rails and repeated sensor beacons create a laboratory scale
    # reference while leaving the centre visually open.
    for side in (-1.0, 1.0):
        y = side * 6.36
        pieces.append(BoxObstacle((0.0, y, 0.16), (13.30, 0.035, 0.035), "visual_datum", graphite))
        pieces.append(BoxObstacle((0.0, y, 0.23), (13.30, 0.018, 0.012), "visual_datum", blue))
        for x in (-10.6, -7.8, -3.2, 1.8, 5.2, 9.2):
            pieces.extend(
                (
                    CylinderObstacle((x, y, 0.34), 0.035, 0.64, "visual_beacon", silver),
                    SphereObstacle((x, y, 0.69), 0.075, "visual_beacon", cyan),
                )
            )

    # Thin accent strips articulate the otherwise neutral physical modules.
    pieces.extend(
        (
            # Two 1.10 m apertures: separate blue/cyan trims make the directional
            # choices immediately legible in overview and top views.
            BoxObstacle((-7.23, -3.05, 2.98), (0.030, 0.56, 0.030), "accent_A", blue),
            BoxObstacle((-7.23, -3.60, 1.52), (0.030, 0.030, 1.43), "accent_A", blue),
            BoxObstacle((-7.23, -2.50, 1.52), (0.030, 0.030, 1.43), "accent_A", blue),
            BoxObstacle((-7.23, -1.65, 2.98), (0.030, 0.56, 0.030), "accent_A", cyan),
            BoxObstacle((-7.23, -2.20, 1.52), (0.030, 0.030, 1.43), "accent_A", cyan),
            BoxObstacle((-7.23, -1.10, 1.52), (0.030, 0.030, 1.43), "accent_A", cyan),
            BoxObstacle((-5.15, -2.35, 3.54), (1.05, 0.025, 0.025), "accent_A", blue),
            BoxObstacle((0.0, -1.55, 3.73), (0.78, 0.025, 0.025), "accent_B", cyan),
            BoxObstacle((0.0, 1.55, 3.73), (0.78, 0.025, 0.025), "accent_B", cyan),
            BoxObstacle((3.33, -4.76, 3.86), (1.58, 0.030, 0.030), "accent_C", blue),
            BoxObstacle((3.33, 4.76, 3.86), (1.58, 0.030, 0.030), "accent_C", blue),
            BoxObstacle((6.18, 0.0, 3.79), (0.030, 4.12, 0.035), "accent_D", cyan),
            BoxObstacle((6.18, 0.0, 1.06), (0.030, 4.12, 0.035), "accent_D", blue),
        )
    )

    # Quiet group-level goal regions replace saturated landing-pad graphics.
    for x, color in ((-11.6, (0.84, 0.58, 0.60, 0.50)), (11.6, (0.47, 0.65, 0.82, 0.50))):
        pieces.extend(
            (
                CylinderObstacle((x, 0.0, 0.018), 1.34, 0.026, "landing_zone", (0.88, 0.89, 0.90, 0.62)),
                CylinderObstacle((x, 0.0, 0.026), 0.56, 0.042, "landing_zone", color),
                CylinderObstacle((x, 0.0, 0.031), 0.16, 0.050, "landing_zone", (0.94, 0.95, 0.96, 0.92)),
            )
        )
    return tuple(pieces)


def _layout_context(variant: str) -> tuple[ScenePrimitive, ...]:
    """Return deterministic collidable context for controlled scene diversity.

    Registered response modules and their collision geometry are unchanged.
    Each variant changes the surrounding context while preserving conservative
    clearance around every registered task polyline.
    """

    if variant == "canonical":
        return ()
    pale = (0.78, 0.82, 0.85, 1.0)
    silver = (0.62, 0.67, 0.71, 1.0)
    graphite = (0.34, 0.39, 0.43, 1.0)
    blue = (0.34, 0.64, 0.99, 1.0)
    teal = (0.00, 0.56, 0.51, 1.0)
    warm = (0.76, 0.70, 0.63, 1.0)
    pieces: list[ScenePrimitive] = []

    if variant == "warehouse":
        for x in (-6.0, -1.8, 2.4, 6.6):
            for side in (-1.0, 1.0):
                y = side * 5.15
                pieces.extend(
                    (
                        CylinderObstacle((x - 0.65, y, 1.65), 0.055, 3.30, "context_warehouse", graphite),
                        CylinderObstacle((x + 0.65, y, 1.65), 0.055, 3.30, "context_warehouse", graphite),
                        BoxObstacle((x, y, 0.62), (0.70, 0.42, 0.055), "context_warehouse", pale),
                        BoxObstacle((x, y, 1.75), (0.70, 0.42, 0.055), "context_warehouse", silver),
                        BoxObstacle((x, y, 3.05), (0.70, 0.42, 0.055), "context_warehouse", pale),
                    )
                )
        pieces.extend(
            (
                BoxObstacle((-3.8, 4.08, 0.42), (0.52, 0.42, 0.42), "context_warehouse", warm),
                BoxObstacle((5.0, -4.12, 0.60), (0.70, 0.50, 0.60), "context_warehouse", warm),
            )
        )
    elif variant == "column_grove":
        grove = (
            (-8.8, 4.35, 0.24, 3.2), (-6.2, 5.05, 0.30, 3.8),
            (-3.6, 4.25, 0.22, 2.8), (-1.0, 5.15, 0.33, 4.2),
            (1.6, -4.35, 0.27, 3.4), (3.8, -5.10, 0.22, 3.0),
            (6.0, 4.40, 0.31, 4.0), (8.2, -4.55, 0.26, 3.5),
        )
        for index, (x, y, radius, height) in enumerate(grove):
            color = pale if index % 2 == 0 else silver
            pieces.append(CylinderObstacle((x, y, height / 2.0), radius, height, "context_column_grove", color))
            pieces.append(SphereObstacle((x, y, height + 0.12), radius * 1.55, "context_column_grove", teal if index % 3 else blue))
    elif variant == "suspended":
        for x in (-6.3, -2.2, 1.9, 6.0):
            pieces.extend(
                (
                    CylinderObstacle((x, -4.55, 2.05), 0.055, 4.10, "context_suspended", graphite),
                    CylinderObstacle((x, 4.55, 2.05), 0.055, 4.10, "context_suspended", graphite),
                    BoxObstacle((x, 0.0, 4.08), (0.055, 4.55, 0.055), "context_suspended", silver),
                    CylinderObstacle((x, 3.82, 3.33), 0.035, 1.40, "context_suspended", silver),
                    SphereObstacle((x, 3.82, 2.55), 0.27, "context_suspended", warm),
                )
            )
    elif variant == "offset_bays":
        bay_specs = (
            (-7.7, 4.75, 0.32), (-4.3, -4.65, -0.28),
            (-0.9, 4.70, -0.24), (2.5, -4.75, 0.30),
            (5.9, 4.65, 0.24), (8.5, -4.70, -0.30),
        )
        for index, (x, y, yaw) in enumerate(bay_specs):
            pieces.append(
                BoxObstacle(
                    (x, y, 1.45), (1.18, 0.10, 1.45), "context_offset_bays",
                    pale if index % 2 == 0 else silver, orientation=(0.0, 0.0, yaw),
                )
            )
            pieces.append(SphereObstacle((x + 0.55, y - np.sign(y) * 0.62, 0.55), 0.34, "context_offset_bays", blue if index % 2 == 0 else teal))
    elif variant == "high_bay":
        for x in (-8.0, -4.0, 0.0, 4.0, 8.0):
            pieces.extend(
                (
                    CylinderObstacle((x, -5.35, 2.25), 0.060, 4.50, "context_high_bay", graphite),
                    CylinderObstacle((x, 5.35, 2.25), 0.060, 4.50, "context_high_bay", graphite),
                    BoxObstacle((x, 0.0, 4.45), (0.060, 5.35, 0.060), "context_high_bay", graphite),
                    BoxObstacle((x + 0.18, 0.0, 3.90), (0.035, 5.05, 0.035), "context_high_bay", blue if x < 0 else teal),
                )
            )
        pieces.extend(
            (
                BoxObstacle((-2.2, -4.15, 0.55), (0.85, 0.45, 0.55), "context_high_bay", pale),
                BoxObstacle((2.7, 4.10, 0.78), (0.65, 0.48, 0.78), "context_high_bay", silver),
            )
        )
    else:
        raise ValueError("unknown layout variant: {0}".format(variant))
    return tuple(pieces)


def compositional_industrial(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """An open, primitive-based testbed for memory composition and swarm traffic.

    The two groups encounter the same modules in opposite orders. Each module
    isolates a reusable navigation factor. Its pale planes, cuboids, spheres,
    cylinders and sparse scaffold take visual cues from DiffPhysDrone while the
    deterministic modules retain CNM intervention semantics.
    """

    n_left = (cfg.num_drones + 1) // 2
    n_right = cfg.num_drones - n_left
    x_spawn = 0.43 * cfg.arena_length
    left = _vertical_grid(n_left, -x_spawn, cfg)
    right = _vertical_grid(n_right, x_spawn, cfg)
    starts = np.vstack((left, right))
    starts[:, 1:] += rng.uniform(-0.015, 0.015, size=(cfg.num_drones, 2))

    goals = starts.copy()
    goals[:n_left, 0] = x_spawn
    goals[n_left:, 0] = -x_spawn
    goals[:n_left, 1:] = left[::-1, 1:]
    goals[n_left:, 1:] = right[::-1, 1:]

    z_half = cfg.arena_height / 2.0
    y_half = cfg.arena_width / 2.0
    hidden = (1.0, 1.0, 1.0, 0.0)
    pale_plane = (0.79, 0.84, 0.88, 1.0)
    panel_alt = (0.73, 0.79, 0.84, 1.0)
    steel = (0.46, 0.51, 0.55, 1.0)
    dark_steel = (0.35, 0.40, 0.44, 1.0)
    pale_steel = (0.68, 0.72, 0.75, 1.0)
    warm_gray = (0.75, 0.72, 0.68, 1.0)
    cool_gray = (0.66, 0.71, 0.75, 1.0)

    gate_x = TWIN_GATE_X
    door_half = TWIN_GATE_WIDTH / 2.0
    door_centers = TWIN_GATE_CENTERS
    door_edges = tuple((center - door_half, center + door_half) for center in door_centers)
    obstacles: list[ScenePrimitive] = [
        # Invisible safety envelope: collision behavior remains bounded without
        # placing dark walls in the publication view.
        BoxObstacle((0.0, y_half + 0.10, z_half), (cfg.arena_length / 2.0, 0.10, z_half), "boundary", hidden, visible=False),
        BoxObstacle((0.0, -y_half - 0.10, z_half), (cfg.arena_length / 2.0, 0.10, z_half), "boundary", hidden, visible=False),
        BoxObstacle((cfg.arena_length / 2.0 + 0.10, 0.0, z_half), (0.10, y_half, z_half), "boundary", hidden, visible=False),
        BoxObstacle((-cfg.arena_length / 2.0 - 0.10, 0.0, z_half), (0.10, y_half, z_half), "boundary", hidden, visible=False),
    ]

    # A: two narrow directional apertures and a long guarded transition field.
    # Each 1.10 m opening gives three 0.23 m-diameter vehicles useful clearance,
    # while four cannot pass abreast without overlap.
    # Slightly alternating panel depths create readable shadows while adjacent
    # panels overlap, so the collision boundary has no unintended slit.
    wall_regions = (
        (-y_half, door_edges[0][0], 2),
        (door_edges[0][1], door_edges[1][0], 1),
        (door_edges[1][1], y_half, 5),
    )
    for region_index, (start, stop, count) in enumerate(wall_regions):
        panel_width = (stop - start) / count
        for panel_index in range(count):
            panel_center = start + panel_width * (panel_index + 0.5)
            obstacles.append(
                BoxObstacle(
                    (gate_x + 0.025 * ((panel_index + region_index) % 2), panel_center, 2.15),
                    (0.14, panel_width / 2.0 + 0.018, 2.15),
                    "offset_gate_A",
                    pale_plane if panel_index % 2 == 0 else panel_alt,
                )
            )
    obstacles.extend(
        (
        # Four jambs, two lintels and one upper band form two complete doors.
        CylinderObstacle((gate_x, door_edges[0][0], 1.48), 0.052, 2.96, "offset_gate_A", dark_steel),
        CylinderObstacle((gate_x, door_edges[0][1], 1.48), 0.052, 2.96, "offset_gate_A", steel),
        CylinderObstacle((gate_x, door_edges[1][0], 1.48), 0.052, 2.96, "offset_gate_A", steel),
        CylinderObstacle((gate_x, door_edges[1][1], 1.48), 0.052, 2.96, "offset_gate_A", dark_steel),
        BoxObstacle((gate_x, door_centers[0], 2.96), (0.15, door_half, 0.060), "offset_gate_A", steel),
        BoxObstacle((gate_x, door_centers[1], 2.96), (0.15, door_half, 0.060), "offset_gate_A", steel),
        BoxObstacle((gate_x, -2.35, 3.66), (0.15, 1.18, 0.64), "offset_gate_A", panel_alt),
        BoxObstacle((gate_x - 0.34, -2.35, 0.18), (0.48, 1.40, 0.18), "offset_gate_A", pale_steel),

        # The longer A→B interval contains a physical separator, offset columns,
        # a suspended object and a portal.  The two registered streams pass on
        # opposite sides and therefore retain distinct executable experience.
        CylinderObstacle((-5.72, -3.92, 1.75), 0.050, 3.50, "offset_gate_A", dark_steel),
        CylinderObstacle((-5.72, -0.78, 1.75), 0.050, 3.50, "offset_gate_A", dark_steel),
        BoxObstacle((-5.72, -2.35, 3.48), (0.055, 1.57, 0.055), "offset_gate_A", steel),
        SphereObstacle((-5.18, -2.35, 1.55), 0.38, "offset_gate_A", cool_gray),
        CylinderObstacle((-4.45, -4.12, 1.34), 0.25, 2.68, "offset_gate_A", pale_steel),
        CylinderObstacle((-4.22, -0.48, 1.62), 0.19, 3.24, "offset_gate_A", steel),
        SphereObstacle((-3.72, -3.28, 2.82), 0.31, "offset_gate_A", warm_gray),
        BoxObstacle((-3.58, -0.82, 0.78), (0.38, 0.44, 0.78), "offset_gate_A", cool_gray),
        BoxObstacle((-3.46, -2.35, 3.47), (0.52, 0.34, 0.18), "offset_gate_A", pale_steel),

        # B: a two-level gantry and central sphere create a bidirectional split.
        CylinderObstacle((-0.62, -1.55, 1.85), 0.050, 3.70, "splitter_tower_B", steel),
        CylinderObstacle((-0.62, 1.55, 1.85), 0.050, 3.70, "splitter_tower_B", steel),
        CylinderObstacle((0.62, -1.55, 1.85), 0.050, 3.70, "splitter_tower_B", steel),
        CylinderObstacle((0.62, 1.55, 1.85), 0.050, 3.70, "splitter_tower_B", steel),
        BoxObstacle((0.0, -1.55, 1.10), (0.68, 0.045, 0.045), "splitter_tower_B", steel),
        BoxObstacle((0.0, 1.55, 1.10), (0.68, 0.045, 0.045), "splitter_tower_B", steel),
        BoxObstacle((0.0, -1.55, 3.15), (0.68, 0.045, 0.045), "splitter_tower_B", steel),
        BoxObstacle((0.0, 1.55, 3.15), (0.68, 0.045, 0.045), "splitter_tower_B", steel),
        BoxObstacle((-0.62, 0.0, 3.67), (0.045, 1.62, 0.045), "splitter_tower_B", dark_steel),
        BoxObstacle((0.62, 0.0, 3.67), (0.045, 1.62, 0.045), "splitter_tower_B", dark_steel),
        BoxObstacle((0.0, -1.55, 2.12), (0.78, 0.032, 0.032), "splitter_tower_B", pale_steel),
        BoxObstacle((0.0, 1.55, 2.12), (0.78, 0.032, 0.032), "splitter_tower_B", pale_steel),
        SphereObstacle((0.0, 0.0, 1.55), 0.56, "splitter_tower_B", cool_gray),
        SphereObstacle((-0.62, 0.0, 3.67), 0.12, "splitter_tower_B", pale_steel),
        SphereObstacle((0.62, 0.0, 3.67), 0.12, "splitter_tower_B", pale_steel),

        # C: a denser but still legible primitive field adds occlusion, weaving
        # and height choices while preserving the registered centre corridor.
        CylinderObstacle((2.25, -3.05, 1.35), 0.32, 2.70, "pillar_field_C", pale_steel),
        SphereObstacle((2.35, 3.05, 1.28), 0.72, "pillar_field_C", warm_gray),
        BoxObstacle((3.38, -0.55, 0.92), (0.45, 0.52, 0.92), "pillar_field_C", cool_gray),
        SphereObstacle((3.62, 1.28, 3.02), 0.56, "pillar_field_C", pale_steel),
        CylinderObstacle((4.25, -2.25, 1.72), 0.20, 3.44, "pillar_field_C", steel),
        CylinderObstacle((2.85, -4.75, 1.90), 0.13, 3.80, "pillar_field_C", dark_steel),
        CylinderObstacle((4.75, 4.62, 1.35), 0.28, 2.70, "pillar_field_C", pale_steel),
        SphereObstacle((4.72, 2.78, 1.38), 0.48, "pillar_field_C", cool_gray),
        SphereObstacle((2.08, -1.25, 3.65), 0.34, "pillar_field_C", warm_gray),
        BoxObstacle((3.32, -4.76, 3.82), (1.62, 0.055, 0.055), "pillar_field_C", steel),
        BoxObstacle((3.32, 4.76, 3.82), (1.62, 0.055, 0.055), "pillar_field_C", steel),
        CylinderObstacle((1.70, -4.76, 1.91), 0.050, 3.82, "pillar_field_C", dark_steel),
        CylinderObstacle((4.94, -4.76, 1.91), 0.050, 3.82, "pillar_field_C", dark_steel),
        CylinderObstacle((1.70, 4.76, 1.91), 0.050, 3.82, "pillar_field_C", dark_steel),
        CylinderObstacle((4.94, 4.76, 1.91), 0.050, 3.82, "pillar_field_C", dark_steel),
        BoxObstacle((4.58, 3.82, 3.26), (0.055, 0.92, 0.055), "pillar_field_C", dark_steel),

        # D: two nested free-standing windows turn altitude selection into a
        # short 3-D corridor instead of one visually flat slab.
        CylinderObstacle((5.82, -4.05, 2.30), 0.070, 4.60, "altitude_gate_D", steel),
        CylinderObstacle((5.82, 4.05, 2.30), 0.070, 4.60, "altitude_gate_D", steel),
        BoxObstacle((5.82, 0.0, 0.92), (0.11, 4.05, 0.16), "altitude_gate_D", cool_gray),
        BoxObstacle((5.82, 0.0, 3.55), (0.11, 4.05, 0.16), "altitude_gate_D", cool_gray),
        CylinderObstacle((6.22, -4.05, 2.30), 0.070, 4.60, "altitude_gate_D", dark_steel),
        CylinderObstacle((6.22, 4.05, 2.30), 0.070, 4.60, "altitude_gate_D", dark_steel),
        BoxObstacle((6.22, 0.0, 0.92), (0.11, 4.05, 0.16), "altitude_gate_D", pale_steel),
        BoxObstacle((6.22, 0.0, 3.55), (0.11, 4.05, 0.16), "altitude_gate_D", pale_steel),
        CylinderObstacle((6.02, -2.15, 2.30), 0.050, 4.60, "altitude_gate_D", steel),
        CylinderObstacle((6.02, 2.15, 2.30), 0.050, 4.60, "altitude_gate_D", steel),
        BoxObstacle((6.02, -3.10, 2.65), (0.30, 0.045, 0.045), "altitude_gate_D", steel, orientation=(0.0, 0.0, 0.46)),
        BoxObstacle((6.02, 3.10, 2.65), (0.30, 0.045, 0.045), "altitude_gate_D", steel, orientation=(0.0, 0.0, -0.46)),
        )
    )
    initial_yaws = np.concatenate((np.zeros(n_left), np.full(n_right, np.pi)))
    group_ids = np.concatenate((np.zeros(n_left, dtype=np.int8), np.ones(n_right, dtype=np.int8)))
    return ScenarioSpec(
        starts=starts,
        goals=goals,
        initial_yaws=initial_yaws,
        group_ids=group_ids,
        obstacles=tuple(obstacles) + _layout_context(cfg.layout_variant),
        decorations=_industrial_decorations(cfg),
        module_sequence=("offset_gate_A", "splitter_tower_B", "pillar_field_C", "altitude_gate_D"),
    )


def _transform_primitive(
    primitive: ScenePrimitive,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
    yaw: float,
) -> ScenePrimitive:
    """Rigidly place one canonical module primitive in a registered task."""

    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    center = target_anchor + rotation @ (np.asarray(primitive.center) - source_anchor)
    if isinstance(primitive, BoxObstacle):
        roll, pitch, source_yaw = primitive.orientation
        return BoxObstacle(
            tuple(center), primitive.half_extents, primitive.module_id,
            primitive.color, (roll, pitch, source_yaw + yaw), primitive.visible,
        )
    if isinstance(primitive, CylinderObstacle):
        roll, pitch, source_yaw = primitive.orientation
        return CylinderObstacle(
            tuple(center), primitive.radius, primitive.height, primitive.module_id,
            primitive.color, (roll, pitch, source_yaw + yaw), primitive.visible,
        )
    return SphereObstacle(
        tuple(center), primitive.radius, primitive.module_id, primitive.color, primitive.visible
    )


def reconfigurable_industrial(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    """Build the exact obstacle ordering registered by a held-out CNM task."""

    if not cfg.reconfigurable_task_id:
        raise ValueError("reconfigurable_industrial requires reconfigurable_task_id")
    # Local import keeps the scenario primitives independent of planner state.
    from .algorithm.protocol import MODULES
    from .algorithm.reconfigurable_protocol import task_by_id

    task = task_by_id(cfg.reconfigurable_task_id)
    canonical_cfg = replace(
        cfg,
        scenario="compositional_industrial",
        layout_variant="canonical",
        reconfigurable_task_id="",
    )
    canonical = compositional_industrial(canonical_cfg, rng)
    boundaries = [item for item in canonical.obstacles if item.module_id == "boundary"]
    modules: list[ScenePrimitive] = []
    for role, target_anchor, yaw in zip(task.roles, task.anchors, task.placement_yaws):
        source_anchor = np.asarray(MODULES[role].anchor, dtype=np.float64)
        source_parts = [item for item in canonical.obstacles if item.module_id == role]
        if not source_parts:
            raise RuntimeError(f"canonical geometry missing for module {role}")
        modules.extend(
            _transform_primitive(item, source_anchor, target_anchor, yaw)
            for item in source_parts
        )

    count = cfg.num_drones
    offsets = np.zeros((count, 3), dtype=np.float64)
    if count > 1:
        offsets[:, 1] = 0.34 * (np.arange(count) - 0.5 * (count - 1))
        offsets[:, 2] = 0.16 * (np.arange(count) % 2)
    starts = task.start[None, :] + offsets
    goals = task.goal[None, :] + offsets[::-1]
    return ScenarioSpec(
        starts=starts,
        goals=goals,
        initial_yaws=np.full(count, task.placement_yaws[0], dtype=np.float64),
        group_ids=np.zeros(count, dtype=np.int8),
        obstacles=tuple(boundaries + modules) + _layout_context(cfg.layout_variant),
        decorations=(),
        module_sequence=task.roles,
    )


def make_scenario(cfg: EnvConfig, rng: np.random.Generator) -> ScenarioSpec:
    if cfg.scenario == "compositional_industrial":
        return compositional_industrial(cfg, rng)
    if cfg.scenario == "bidirectional_bottleneck":
        return bidirectional_bottleneck(cfg, rng)
    if cfg.scenario == "reconfigurable_industrial":
        return reconfigurable_industrial(cfg, rng)
    if cfg.scenario == "eef_indoor_static":
        return eef_indoor_static(cfg, rng)
    if cfg.scenario == "eef_indoor_dynamic":
        return eef_indoor_dynamic(cfg, rng)
    if cfg.scenario == "eef_garage":
        return eef_garage(cfg, rng)
    if cfg.scenario == "eef_forest":
        return eef_forest(cfg, rng)
    raise ValueError("unknown scenario: {0}".format(cfg.scenario))
