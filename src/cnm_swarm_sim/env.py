"""PyBullet DIRECT shared-world environment for CNM experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from ._bootstrap import prepare_windows_dll_path
from .config import EnvConfig
from .scenarios import BoxObstacle, CylinderObstacle, ScenarioSpec, ScenePrimitive, SphereObstacle, make_scenario

prepare_windows_dll_path()
import pybullet as p  # noqa: E402


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return values * scale


@dataclass
class EnvSnapshot:
    """Exact, in-process snapshot of Bullet and all Python-side experiment state."""

    bullet_state_id: int
    step_count: int
    rng_state: dict[str, Any]
    target_yaw: np.ndarray
    reached: np.ndarray
    previous_contacts: set[tuple[int, int]]
    events_length: int


class CNMSwarmEnv(gym.Env):
    """One continuous 3-D world containing all swarm members.

    Actions have shape ``(N, 4)`` and contain desired world-frame velocity
    ``vx, vy, vz`` and yaw rate. Observations expose local goal direction,
    self motion, a fixed number of nearest neighbors, and optional horizontal
    range rays. Global state remains available only through explicit diagnostic
    methods and the ``info`` dictionary used by the experiment harness.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.config.validate()
        self.client = p.connect(p.GUI if self.config.gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("PyBullet connection failed")

        n = self.config.num_drones
        k = self.config.neighbor_k
        r = self.config.ray_count
        action_high = np.tile(
            np.array([self.config.max_speed] * 3 + [self.config.max_yaw_rate], dtype=np.float32),
            (n, 1),
        )
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "self": spaces.Box(-np.inf, np.inf, shape=(n, 9), dtype=np.float32),
                "goal": spaces.Box(-np.inf, np.inf, shape=(n, 3), dtype=np.float32),
                "neighbors": spaces.Box(-np.inf, np.inf, shape=(n, k, 6), dtype=np.float32),
                "neighbor_mask": spaces.Box(0.0, 1.0, shape=(n, k), dtype=np.float32),
                "rays": spaces.Box(0.0, 1.0, shape=(n, r), dtype=np.float32),
            }
        )

        self.rng = np.random.default_rng(self.config.seed)
        self.scenario_spec: ScenarioSpec | None = None
        self.drone_ids = np.empty(0, dtype=np.int32)
        self.obstacle_ids: list[int] = []
        self.decoration_ids: list[int] = []
        self.goal_marker_ids: list[int] = []
        self.body_to_drone: dict[int, int] = {}
        self.obstacle_body_to_module: dict[int, str] = {}
        self.dynamic_obstacle_ids: dict[int, int] = {}
        self.dynamic_obstacle_origins: dict[int, np.ndarray] = {}
        self.step_count = 0
        self.target_yaw = np.zeros(n, dtype=np.float64)
        self.reached = np.zeros(n, dtype=bool)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.quat = np.zeros((n, 4), dtype=np.float64)
        self.rpy = np.zeros((n, 3), dtype=np.float64)
        self.vel = np.zeros((n, 3), dtype=np.float64)
        self.ang_vel = np.zeros((n, 3), dtype=np.float64)
        self.events: list[dict[str, Any]] = []
        self._previous_contacts: set[tuple[int, int]] = set()
        self._closed = False
        self.reset(seed=self.config.seed)

    def _configure_physics(self) -> None:
        p.setGravity(0.0, 0.0, -self.config.gravity, physicsClientId=self.client)
        p.setTimeStep(self.config.physics_timestep, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        kwargs: dict[str, Any] = {
            "numSolverIterations": 20,
            "contactBreakingThreshold": 0.001,
        }
        if self.config.deterministic:
            kwargs["deterministicOverlappingPairs"] = 1
        p.setPhysicsEngineParameter(physicsClientId=self.client, **kwargs)

    def _create_box(
        self,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        color: tuple[float, float, float, float],
        collidable: bool = True,
        orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> int:
        collision = (
            p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
            if collidable
            else -1
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color if visible else (1.0, 1.0, 1.0, 0.0),
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center,
            baseOrientation=p.getQuaternionFromEuler(orientation),
            physicsClientId=self.client,
        )

    def _create_cylinder(
        self,
        center: tuple[float, float, float],
        radius: float,
        height: float,
        color: tuple[float, float, float, float],
        collidable: bool = True,
        orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> int:
        collision = (
            p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self.client,
            )
            if collidable
            else -1
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color if visible else (1.0, 1.0, 1.0, 0.0),
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center,
            baseOrientation=p.getQuaternionFromEuler(orientation),
            physicsClientId=self.client,
        )

    def _create_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        color: tuple[float, float, float, float],
        collidable: bool = True,
        visible: bool = True,
    ) -> int:
        collision = (
            p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
            if collidable
            else -1
        )
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color if visible else (1.0, 1.0, 1.0, 0.0),
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center,
            physicsClientId=self.client,
        )

    def _create_scene_primitive(self, primitive: ScenePrimitive, collidable: bool) -> int:
        if isinstance(primitive, BoxObstacle):
            return self._create_box(
                primitive.center,
                primitive.half_extents,
                primitive.color,
                collidable=collidable,
                orientation=primitive.orientation,
                visible=primitive.visible,
            )
        if isinstance(primitive, CylinderObstacle):
            return self._create_cylinder(
                primitive.center,
                primitive.radius,
                primitive.height,
                primitive.color,
                collidable=collidable,
                orientation=primitive.orientation,
                visible=primitive.visible,
            )
        if isinstance(primitive, SphereObstacle):
            return self._create_sphere(
                primitive.center,
                primitive.radius,
                primitive.color,
                collidable=collidable,
                visible=primitive.visible,
            )
        raise TypeError("unsupported scene primitive: {0}".format(type(primitive).__name__))

    def _create_quadrotor_visual(self, accent_color: tuple[float, float, float, float]) -> int:
        """Create a light micro-quad visual while retaining one cheap collision body."""

        shell = (0.88, 0.91, 0.94, 1.0)
        graphite = (0.16, 0.19, 0.23, 1.0)
        rotor_guard = (0.58, 0.61, 0.64, 0.58)
        motor = (0.34, 0.37, 0.40, 1.0)
        battery = (0.25, 0.28, 0.31, 1.0)
        canopy = (0.10, 0.14, 0.20, 1.0)
        lens = (0.02, 0.03, 0.04, 1.0)

        shape_types: list[int] = []
        half_extents: list[list[float]] = []
        radii: list[float] = []
        lengths: list[float] = []
        positions: list[list[float]] = []
        orientations: list[tuple[float, float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []

        def add_shape(
            shape_type: int,
            half_extent: tuple[float, float, float] = (0.0, 0.0, 0.0),
            radius: float = 0.0,
            length: float = 0.0,
            position: tuple[float, float, float] = (0.0, 0.0, 0.0),
            orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
            color: tuple[float, float, float, float] = graphite,
        ) -> None:
            shape_types.append(shape_type)
            half_extents.append(list(half_extent))
            radii.append(radius)
            lengths.append(length)
            positions.append(list(position))
            orientations.append(orientation)
            colors.append(color)

        add_shape(p.GEOM_BOX, (0.075, 0.050, 0.024), position=(0.0, 0.0, 0.002), color=shell)
        add_shape(p.GEOM_BOX, (0.052, 0.037, 0.018), position=(-0.012, 0.0, -0.030), color=battery)
        add_shape(p.GEOM_SPHERE, radius=0.038, position=(0.018, 0.0, 0.026), color=canopy)
        add_shape(p.GEOM_BOX, (0.048, 0.040, 0.006), position=(0.006, 0.0, 0.035), color=accent_color)
        for yaw in (math.pi / 4.0, -math.pi / 4.0):
            add_shape(
                p.GEOM_BOX,
                (0.135, 0.012, 0.009),
                orientation=p.getQuaternionFromEuler((0.0, 0.0, yaw)),
                color=graphite,
            )
        rotor_centers = ((0.095, 0.095), (0.095, -0.095), (-0.095, 0.095), (-0.095, -0.095))
        for x, y in rotor_centers:
            add_shape(p.GEOM_CYLINDER, radius=0.016, length=0.034, position=(x, y, 0.018), color=motor)
            add_shape(p.GEOM_CYLINDER, radius=0.059, length=0.003, position=(x, y, 0.039), color=rotor_guard)
        # The forward depth-camera block makes heading legible.
        add_shape(p.GEOM_BOX, (0.018, 0.031, 0.018), position=(0.088, 0.0, -0.002), color=graphite)
        add_shape(p.GEOM_SPHERE, radius=0.012, position=(0.106, 0.0, -0.002), color=lens)
        return p.createVisualShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            radii=radii,
            lengths=lengths,
            visualFramePositions=positions,
            visualFrameOrientations=orientations,
            rgbaColors=colors,
            physicsClientId=self.client,
        )

    def _create_world(self) -> None:
        cfg = self.config
        floor = self._create_box(
            (0.0, 0.0, -0.07),
            (cfg.arena_length / 2.0 + 1.6, cfg.arena_width / 2.0 + 1.4, 0.07),
            (0.92, 0.93, 0.94, 1.0),
        )
        self.obstacle_ids = [floor]
        self.obstacle_body_to_module = {floor: "floor"}
        assert self.scenario_spec is not None
        self.dynamic_obstacle_ids = {}
        self.dynamic_obstacle_origins = {}
        dynamic_indices = {motion.obstacle_index for motion in self.scenario_spec.dynamic_motions}
        for obstacle_index, obstacle in enumerate(self.scenario_spec.obstacles):
            body = self._create_scene_primitive(obstacle, collidable=True)
            self.obstacle_ids.append(body)
            self.obstacle_body_to_module[body] = obstacle.module_id
            if obstacle_index in dynamic_indices:
                self.dynamic_obstacle_ids[obstacle_index] = body
                self.dynamic_obstacle_origins[obstacle_index] = np.asarray(
                    obstacle.center, dtype=np.float64
                )

        self.decoration_ids = []
        if cfg.scene_decorations:
            for decoration in self.scenario_spec.decorations:
                self.decoration_ids.append(self._create_scene_primitive(decoration, collidable=False))

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cfg.drone_radius,
            height=cfg.drone_height,
            physicsClientId=self.client,
        )
        visuals = [
            self._create_quadrotor_visual(color)
            for color in ((0.20, 0.50, 0.82, 1.0), (0.86, 0.34, 0.37, 1.0))
        ]
        inertia = (1.45e-5, 1.45e-5, 2.65e-5)
        ids: list[int] = []
        for idx, (xyz, yaw, group) in enumerate(
            zip(self.scenario_spec.starts, self.scenario_spec.initial_yaws, self.scenario_spec.group_ids)
        ):
            body = p.createMultiBody(
                baseMass=cfg.mass,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visuals[int(group)],
                basePosition=xyz,
                baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, float(yaw))),
                baseInertialFramePosition=(0.0, 0.0, 0.0),
                physicsClientId=self.client,
            )
            p.changeDynamics(
                body,
                -1,
                localInertiaDiagonal=inertia,
                linearDamping=0.0,
                angularDamping=0.0,
                lateralFriction=0.4,
                restitution=0.05,
                physicsClientId=self.client,
            )
            ids.append(body)
            self.body_to_drone[body] = idx
        self.drone_ids = np.asarray(ids, dtype=np.int32)

        self.goal_marker_ids = []
        if cfg.visual_markers:
            goal_visuals = [
                p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.07,
                    rgbaColor=color,
                    physicsClientId=self.client,
                )
                for color in ((0.10, 0.45, 0.75, 0.45), (0.91, 0.38, 0.20, 0.45))
            ]
            for goal, group in zip(self.scenario_spec.goals, self.scenario_spec.group_ids):
                marker = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=goal_visuals[int(group)],
                    basePosition=goal,
                    physicsClientId=self.client,
                )
                self.goal_marker_ids.append(marker)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p.resetSimulation(physicsClientId=self.client)
        self._configure_physics()
        self.scenario_spec = make_scenario(self.config, self.rng)
        self.body_to_drone = {}
        self._create_world()
        self.step_count = 0
        self.target_yaw = self.scenario_spec.initial_yaws.astype(np.float64, copy=True)
        self.reached = np.zeros(self.config.num_drones, dtype=bool)
        self.events = []
        self._previous_contacts = set()
        self._update_dynamic_obstacles(0.0)
        self._update_states()
        obs = self._compute_observation()
        return obs, self._info(new_events=[])

    def _update_states(self) -> None:
        for i, body in enumerate(self.drone_ids):
            pos, quat = p.getBasePositionAndOrientation(int(body), physicsClientId=self.client)
            vel, ang_vel = p.getBaseVelocity(int(body), physicsClientId=self.client)
            self.pos[i] = pos
            self.quat[i] = quat
            self.rpy[i] = p.getEulerFromQuaternion(quat)
            self.vel[i] = vel
            self.ang_vel[i] = ang_vel

    def _update_dynamic_obstacles(self, simulation_time: float) -> None:
        """Advance registered kinematic obstacles from absolute simulation time."""

        if self.scenario_spec is None:
            return
        for motion in self.scenario_spec.dynamic_motions:
            body = self.dynamic_obstacle_ids[motion.obstacle_index]
            origin = self.dynamic_obstacle_origins[motion.obstacle_index]
            axis = np.asarray(motion.axis, dtype=np.float64)
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            phase = 2.0 * np.pi * simulation_time / motion.period_seconds + motion.phase_rad
            position = origin + axis * motion.amplitude * np.sin(phase)
            _, orientation = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
            p.resetBasePositionAndOrientation(
                body, position, orientation, physicsClientId=self.client
            )

    def _body_frame_xy(self, vectors: np.ndarray) -> np.ndarray:
        yaw = self.rpy[:, 2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        out = vectors.copy()
        out[..., 0] = c.reshape((-1,) + (1,) * (vectors.ndim - 2)) * vectors[..., 0] + s.reshape(
            (-1,) + (1,) * (vectors.ndim - 2)
        ) * vectors[..., 1]
        out[..., 1] = -s.reshape((-1,) + (1,) * (vectors.ndim - 2)) * vectors[..., 0] + c.reshape(
            (-1,) + (1,) * (vectors.ndim - 2)
        ) * vectors[..., 1]
        return out

    def _neighbor_observation(self) -> tuple[np.ndarray, np.ndarray, float]:
        n = self.config.num_drones
        k_out = self.config.neighbor_k
        neighbors = np.zeros((n, k_out, 6), dtype=np.float32)
        mask = np.zeros((n, k_out), dtype=np.float32)
        if n <= 1 or k_out == 0:
            return neighbors, mask, math.inf

        relative_pos = self.pos[None, :, :] - self.pos[:, None, :]
        relative_vel = self.vel[None, :, :] - self.vel[:, None, :]
        distances = np.linalg.norm(relative_pos, axis=2)
        np.fill_diagonal(distances, np.inf)
        min_separation = float(np.min(distances))
        k = min(k_out, n - 1)
        indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n)[:, None]
        selected_distances = distances[row, indices]
        order = np.argsort(selected_distances, axis=1)
        indices = indices[row, order]
        rel_p = relative_pos[row, indices]
        rel_v = relative_vel[row, indices]
        neighbors[:, :k, :3] = self._body_frame_xy(rel_p).astype(np.float32)
        neighbors[:, :k, 3:] = self._body_frame_xy(rel_v).astype(np.float32)
        mask[:, :k] = (distances[row, indices] <= self.config.communication_radius).astype(np.float32)
        neighbors[:, :k, :] *= mask[:, :k, None]
        return neighbors, mask, min_separation

    def _range_observation(self) -> np.ndarray:
        n = self.config.num_drones
        r = self.config.ray_count
        if r == 0:
            return np.empty((n, 0), dtype=np.float32)
        base_angles = np.linspace(-np.pi, np.pi, r, endpoint=False)
        angles = self.rpy[:, 2, None] + base_angles[None, :]
        directions = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=2)
        starts = self.pos[:, None, :] + directions * (self.config.drone_radius + 0.015)
        ends = starts + directions * self.config.ray_range
        results = p.rayTestBatch(
            starts.reshape(-1, 3).tolist(),
            ends.reshape(-1, 3).tolist(),
            numThreads=0,
            physicsClientId=self.client,
        )
        fractions = np.fromiter((hit[2] for hit in results), dtype=np.float64, count=n * r)
        return fractions.reshape(n, r).astype(np.float32)

    def _compute_observation(self) -> dict[str, np.ndarray]:
        assert self.scenario_spec is not None
        self_velocity = self._body_frame_xy(self.vel)
        goal = self._body_frame_xy(self.scenario_spec.goals - self.pos)
        neighbors, neighbor_mask, _ = self._neighbor_observation()
        return {
            "self": np.concatenate((self_velocity, self.rpy, self.ang_vel), axis=1).astype(np.float32),
            "goal": goal.astype(np.float32),
            "neighbors": neighbors,
            "neighbor_mask": neighbor_mask,
            "rays": self._range_observation(),
        }

    def _apply_downwash(self) -> None:
        if not self.config.enable_downwash or self.config.num_drones < 2:
            return
        rel = self.pos[None, :, :] - self.pos[:, None, :]
        horizontal = np.linalg.norm(rel[..., :2], axis=2)
        dz = rel[..., 2]
        for lower in range(self.config.num_drones):
            upper = (horizontal[lower] < 0.22) & (dz[lower] > 0.0) & (dz[lower] < 0.6)
            count = int(np.sum(upper))
            if count:
                p.applyExternalForce(
                    int(self.drone_ids[lower]),
                    -1,
                    (0.0, 0.0, -0.018 * count),
                    self.pos[lower],
                    p.WORLD_FRAME,
                    physicsClientId=self.client,
                )

    def _apply_controller(self, velocity_command: np.ndarray) -> None:
        cfg = self.config
        acceleration = _clip_rows(cfg.velocity_kp * (velocity_command - self.vel), cfg.max_accel)
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        drag = -cfg.drag_coefficient * self.vel * speed
        forces = cfg.mass * (acceleration + np.array([0.0, 0.0, cfg.gravity])) + drag
        yaw_error = _wrap_angle(self.target_yaw - self.rpy[:, 2])
        torques = np.column_stack(
            (
                -cfg.attitude_kp * self.rpy[:, 0] - cfg.attitude_kd * self.ang_vel[:, 0],
                -cfg.attitude_kp * self.rpy[:, 1] - cfg.attitude_kd * self.ang_vel[:, 1],
                cfg.yaw_kp * yaw_error - cfg.attitude_kd * self.ang_vel[:, 2],
            )
        )
        torques = np.clip(torques, -0.004, 0.004)
        for i, body in enumerate(self.drone_ids):
            p.applyExternalForce(
                int(body), -1, forces[i], self.pos[i], p.WORLD_FRAME, physicsClientId=self.client
            )
            p.applyExternalTorque(int(body), -1, torques[i], p.WORLD_FRAME, physicsClientId=self.client)
        self._apply_downwash()

    def _contact_events(self) -> list[dict[str, Any]]:
        contacts = p.getContactPoints(physicsClientId=self.client)
        current: set[tuple[int, int]] = set()
        events: list[dict[str, Any]] = []
        drone_bodies = self.body_to_drone
        for contact in contacts:
            a, b = int(contact[1]), int(contact[2])
            if a not in drone_bodies and b not in drone_bodies:
                continue
            key = (min(a, b), max(a, b))
            if key in current:
                continue
            current.add(key)
            if key in self._previous_contacts:
                continue
            kind = "drone_drone" if a in drone_bodies and b in drone_bodies else "drone_obstacle"
            event = {
                "event_id": "collision:{0}:{1}:{2}".format(self.step_count, key[0], key[1]),
                "step": self.step_count,
                "kind": kind,
                "body_a": a,
                "body_b": b,
                "drone_a": drone_bodies.get(a),
                "drone_b": drone_bodies.get(b),
                "module_id": None,
            }
            if kind == "drone_obstacle":
                obstacle_body = b if a in drone_bodies else a
                event["module_id"] = self.obstacle_body_to_module.get(obstacle_body, "unknown")
            events.append(event)
        self._previous_contacts = current
        self.events.extend(events)
        return events

    def _info(self, new_events: list[dict[str, Any]]) -> dict[str, Any]:
        _, _, min_separation = self._neighbor_observation()
        return {
            "step": self.step_count,
            "simulation_time": self.step_count * self.config.control_timestep,
            "team_completion": float(np.mean(self.reached)),
            "minimum_separation": min_separation,
            "new_events": new_events,
            "collision_events_total": len(self.events),
            "shared_world": True,
            "num_drones": self.config.num_drones,
            "module_sequence": self.scenario_spec.module_sequence if self.scenario_spec else (),
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        expected = (self.config.num_drones, 4)
        if action.shape != expected:
            raise ValueError("action shape must be {0}, received {1}".format(expected, action.shape))
        velocity_command = _clip_rows(action[:, :3], self.config.max_speed)
        yaw_rate = np.clip(action[:, 3], -self.config.max_yaw_rate, self.config.max_yaw_rate)
        self.target_yaw = _wrap_angle(self.target_yaw + yaw_rate * self.config.control_timestep)

        for substep in range(self.config.physics_steps_per_control):
            simulation_time = (
                self.step_count * self.config.control_timestep
                + substep * self.config.physics_timestep
            )
            self._update_dynamic_obstacles(simulation_time)
            self._update_states()
            self._apply_controller(velocity_command)
            p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1
        self._update_states()
        new_events = self._contact_events()
        assert self.scenario_spec is not None
        goal_distance = np.linalg.norm(self.scenario_spec.goals - self.pos, axis=1)
        self.reached |= goal_distance <= self.config.goal_tolerance
        reward = (-goal_distance - 2.0 * np.array(
            [sum(e.get("drone_a") == i or e.get("drone_b") == i for e in new_events) for i in range(self.config.num_drones)]
        )).astype(np.float32)
        terminated = bool(np.all(self.reached))
        truncated = bool(self.step_count >= self.config.horizon_steps)
        observation = self._compute_observation()
        return observation, reward, terminated, truncated, self._info(new_events)

    def scripted_goal_action(self, speed: float | None = None) -> np.ndarray:
        """Deterministic baseline used only for smoke tests and throughput tests."""

        assert self.scenario_spec is not None
        delta = self.scenario_spec.goals - self.pos
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        desired_speed = min(speed if speed is not None else self.config.max_speed * 0.65, self.config.max_speed)
        velocity = delta / np.maximum(distance, 1e-9) * desired_speed
        velocity[distance[:, 0] <= self.config.goal_tolerance] = 0.0
        desired_yaw = np.arctan2(delta[:, 1], delta[:, 0])
        yaw_rate = np.clip(
            2.0 * _wrap_angle(desired_yaw - self.rpy[:, 2]),
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate,
        )
        return np.column_stack((velocity, yaw_rate)).astype(np.float32)

    def snapshot(self) -> EnvSnapshot:
        state_id = p.saveState(physicsClientId=self.client)
        return EnvSnapshot(
            bullet_state_id=state_id,
            step_count=self.step_count,
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
            target_yaw=self.target_yaw.copy(),
            reached=self.reached.copy(),
            previous_contacts=set(self._previous_contacts),
            events_length=len(self.events),
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        p.restoreState(stateId=snapshot.bullet_state_id, physicsClientId=self.client)
        self.step_count = snapshot.step_count
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        self.target_yaw = snapshot.target_yaw.copy()
        self.reached = snapshot.reached.copy()
        self._previous_contacts = set(snapshot.previous_contacts)
        del self.events[snapshot.events_length :]
        self._update_states()

    def release_snapshot(self, snapshot: EnvSnapshot) -> None:
        p.removeState(snapshot.bullet_state_id, physicsClientId=self.client)

    def save_portable_snapshot(self, path: str | Path) -> Path:
        """Save state arrays for cross-process replay from an identical reset."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            pos=self.pos,
            quat=self.quat,
            vel=self.vel,
            ang_vel=self.ang_vel,
            target_yaw=self.target_yaw,
            reached=self.reached,
            step_count=np.array([self.step_count], dtype=np.int64),
            rng_state=np.array([json.dumps(self.rng.bit_generator.state, sort_keys=True)]),
        )
        return target

    def load_portable_snapshot(self, path: str | Path) -> None:
        """Restore a serialized state after resetting the identical task and seed."""

        data = np.load(Path(path), allow_pickle=False)
        if data["pos"].shape != self.pos.shape:
            raise ValueError("snapshot swarm size does not match this environment")
        for i, body in enumerate(self.drone_ids):
            p.resetBasePositionAndOrientation(
                int(body), data["pos"][i], data["quat"][i], physicsClientId=self.client
            )
            p.resetBaseVelocity(
                int(body), data["vel"][i], data["ang_vel"][i], physicsClientId=self.client
            )
        self.target_yaw = data["target_yaw"].copy()
        self.reached = data["reached"].astype(bool, copy=True)
        self.step_count = int(data["step_count"][0])
        self.rng.bit_generator.state = json.loads(str(data["rng_state"][0]))
        self._previous_contacts = set()
        self.events = []
        self._update_states()

    def set_drone_states(
        self,
        positions: np.ndarray,
        *,
        velocities: np.ndarray | None = None,
        yaws: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Set registered episode-entry states without rebuilding the world."""

        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.config.num_drones, 3):
            raise ValueError("positions must have shape (num_drones, 3)")
        if velocities is None:
            velocities = np.zeros_like(positions)
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions")
        if yaws is None:
            yaws = np.zeros(self.config.num_drones, dtype=np.float64)
        yaws = np.asarray(yaws, dtype=np.float64)
        if yaws.shape != (self.config.num_drones,):
            raise ValueError("yaws must have shape (num_drones,)")
        for index, body in enumerate(self.drone_ids):
            quaternion = p.getQuaternionFromEuler((0.0, 0.0, float(yaws[index])))
            p.resetBasePositionAndOrientation(
                int(body), positions[index], quaternion, physicsClientId=self.client
            )
            p.resetBaseVelocity(
                int(body), velocities[index], (0.0, 0.0, 0.0), physicsClientId=self.client
            )
        self.target_yaw = yaws.copy()
        self.reached[:] = False
        self.step_count = 0
        self.events = []
        self._previous_contacts = set()
        self._update_states()
        return self._compute_observation()

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for array in (self.pos, self.quat, self.vel, self.ang_vel, self.target_yaw, self.reached):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(str(self.step_count).encode("ascii"))
        return digest.hexdigest()

    def render_frame(
        self,
        path: str | Path | None = None,
        width: int = 1600,
        height: int = 900,
        preset: str = "overview",
        focus_drone: int = 0,
    ) -> np.ndarray:
        """Render one selected frame; normal simulation remains image-free."""

        if preset == "overview":
            target_position = (0.1, 0.0, 1.35)
            distance, yaw, pitch, fov = 20.2, 39.0, -41.0, 47.0
        elif preset == "traffic":
            target_position = (0.7, 0.0, 1.60)
            distance, yaw, pitch, fov = 15.2, 31.0, -28.0, 48.0
        elif preset == "swap_top":
            target_position = (0.0, 0.0, 0.0)
            distance, yaw, pitch, fov = 17.2, 0.0, -89.5, 50.0
        elif preset == "drone_closeup":
            if not 0 <= focus_drone < self.config.num_drones:
                raise ValueError("focus_drone is outside the swarm")
            target_position = tuple(float(v) for v in self.pos[focus_drone])
            distance, yaw, pitch, fov = 0.92, 132.0, -20.0, 42.0
        elif preset == "eef_scene":
            target_position = (0.0, 0.0, 1.25)
            distance, yaw, pitch, fov = 11.8, 38.0, -34.0, 50.0
        elif preset == "eef_traffic":
            target_position = (0.4, 0.0, 1.35)
            distance, yaw, pitch, fov = 10.6, 33.0, -27.0, 52.0
        else:
            raise ValueError("unknown camera preset: {0}".format(preset))
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=fov, aspect=float(width) / float(height), nearVal=0.03, farVal=70.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            shadow=1,
            lightDirection=(-4.0, -6.0, 11.0),
            lightColor=(1.0, 1.0, 1.0),
            lightDistance=35.0,
            physicsClientId=self.client,
        )
        image = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image, mode="RGB").save(target)
        return image

    def close(self) -> None:
        if not self._closed:
            p.disconnect(physicsClientId=self.client)
            self._closed = True
