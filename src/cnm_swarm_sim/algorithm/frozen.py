"""Immutable execution-stack manifest for weight-free CNM experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experience import CompilerConfig
from .memory import PlannerConfig


def _source_sha256(filename: str) -> str:
    path = Path(__file__).with_name(filename)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenStackManifest:
    """Serializable definition of every component assigned to manuscript W*."""

    policy_sha256: str
    policy_config: dict[str, Any]
    planner_config: dict[str, Any]
    compiler_config: dict[str, Any]
    calibration_version: str
    command_vocabulary: tuple[str, ...]
    operator_source_sha256: dict[str, str]
    schema_version: int = 2

    @property
    def frozen_stack_sha256(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frozen_stack_sha256"] = self.frozen_stack_sha256
        return payload


def build_frozen_stack_manifest(
    *,
    policy_sha256: str,
    policy_config: Any,
    planner_config: PlannerConfig,
    compiler_config: CompilerConfig,
    calibration_version: str = "simulation-v1",
) -> FrozenStackManifest:
    """Bind weights, thresholds, calibration, vocabulary, and operators."""

    if hasattr(policy_config, "__dataclass_fields__"):
        policy_payload = asdict(policy_config)
    elif isinstance(policy_config, dict):
        policy_payload = dict(policy_config)
    else:
        raise TypeError("policy_config must be a dataclass or dictionary")
    return FrozenStackManifest(
        policy_sha256=policy_sha256,
        policy_config=policy_payload,
        planner_config=asdict(planner_config),
        compiler_config=asdict(compiler_config),
        calibration_version=calibration_version,
        command_vocabulary=tuple(planner_config.allowed_modes),
        operator_source_sha256={
            name: _source_sha256(name)
            for name in ("eef.py", "experience.py", "memory.py", "rollout.py")
        },
    )
