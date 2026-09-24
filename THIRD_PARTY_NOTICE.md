# Third-party provenance

CNMSwarmSim uses the PyBullet Python API and follows the lightweight, steppable,
multi-quadrotor environment design of `gym-pybullet-drones`.

- PyBullet: https://github.com/bulletphysics/bullet3 (zlib license)
- gym-pybullet-drones: https://github.com/learnsyslab/gym-pybullet-drones
  (MIT license), inspected at commit
  `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`

The initial environment implements its own compact high-level velocity
controller and approximate Crazyflie-sized collision body. It does not copy the
upstream `BaseAviary` implementation. A later motor-level validation mode may
reuse the upstream CF2X model while retaining its copyright and license notice.

