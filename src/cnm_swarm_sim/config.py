"""Configuration objects for CNMSwarmEnv."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Numerical and experimental configuration for one shared world."""

    num_drones: int = 20
    seed: int = 7
    pyb_freq: int = 120
    ctrl_freq: int = 30
    episode_seconds: float = 20.0
    gui: bool = False
    scenario: str = "compositional_industrial"
    # Controlled context geometry surrounding the registered CNM modules.
    # ``canonical`` reproduces the original benchmark exactly.
    layout_variant: str = "canonical"
    # Registered task identifier used only by ``reconfigurable_industrial``.
    reconfigurable_task_id: str = ""

    # Crazyflie-scale approximate rigid body.
    mass: float = 0.027
    drone_radius: float = 0.115
    drone_height: float = 0.040
    gravity: float = 9.81

    # Fixed low-level velocity/attitude controller.
    max_speed: float = 2.0
    max_accel: float = 5.0
    max_yaw_rate: float = 2.0
    velocity_kp: float = 3.0
    attitude_kp: float = 0.0025
    attitude_kd: float = 0.00035
    yaw_kp: float = 0.0012
    drag_coefficient: float = 0.003
    enable_downwash: bool = False

    # Local, fixed-size deployment observation.
    ray_count: int = 24
    ray_range: float = 4.0
    neighbor_k: int = 6
    communication_radius: float = 3.0

    # Geometry and reproducibility.
    arena_length: float = 28.0
    arena_width: float = 14.0
    arena_height: float = 5.0
    bottleneck_width: float = 3.0
    goal_tolerance: float = 0.30
    deterministic: bool = True
    visual_markers: bool = False
    scene_decorations: bool = False

    def validate(self) -> None:
        if self.num_drones < 1:
            raise ValueError("num_drones must be at least 1")
        if self.pyb_freq <= 0 or self.ctrl_freq <= 0:
            raise ValueError("simulation frequencies must be positive")
        if self.pyb_freq % self.ctrl_freq:
            raise ValueError("pyb_freq must be divisible by ctrl_freq")
        if self.ray_count < 0:
            raise ValueError("ray_count cannot be negative")
        if self.neighbor_k < 0:
            raise ValueError("neighbor_k cannot be negative")
        if self.bottleneck_width <= 2.0 * self.drone_radius:
            raise ValueError("bottleneck_width is too small for a drone")

    @property
    def physics_steps_per_control(self) -> int:
        return self.pyb_freq // self.ctrl_freq

    @property
    def control_timestep(self) -> float:
        return 1.0 / self.ctrl_freq

    @property
    def physics_timestep(self) -> float:
        return 1.0 / self.pyb_freq

    @property
    def horizon_steps(self) -> int:
        return int(round(self.episode_seconds * self.ctrl_freq))
