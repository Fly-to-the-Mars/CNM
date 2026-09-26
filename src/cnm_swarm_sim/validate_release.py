"""Lightweight structural validation for the clean CNM release."""

from __future__ import annotations

from dataclasses import asdict

from .algorithm.experience import CompilerConfig
from .algorithm.frozen import build_frozen_stack_manifest
from .algorithm.memory import PlannerConfig
from .algorithm.reconfigurable_protocol import reconfigurable_tasks


def main() -> None:
    planner = PlannerConfig()
    compiler = CompilerConfig()
    manifest = build_frozen_stack_manifest(
        policy_sha256="release-validation-policy",
        policy_config={"backend": "reference-simulation"},
        planner_config=planner,
        compiler_config=compiler,
    )
    tasks = reconfigurable_tasks()
    acb = [task for task in tasks if task.task_id == "R08_ACB_explicit"]
    checks = {
        "frozen_manifest": bool(manifest.frozen_stack_sha256),
        "bounded_astar": planner.graph_node_budget == 160 and planner.max_depth == 6,
        "three_success_admission": compiler.min_successes == 3,
        "registered_unseen_tasks": len(tasks) == 24 and all(not task.acquisition_order_seen for task in tasks),
        "explicit_acb": len(acb) == 1 and len(acb[0].roles) == 3,
        "serializable_configs": bool(asdict(planner)) and bool(asdict(compiler)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("release validation failed: " + ", ".join(failed))
    print(f"CNM release validation passed: {len(checks)} checks")


if __name__ == "__main__":
    main()
