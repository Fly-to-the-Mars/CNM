"""Train and evaluate the registered robust-agile EEF comparison study."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithm.eef import EEFConfig, EEFNavigator, EEFTrainer, eef_config_for_variant
from .algorithm.eef_benchmark import (
    BenchmarkCondition,
    DEFAULT_CONDITIONS,
    evaluate_interface_repeatability,
    evaluate_policy_trials,
    wilson_interval,
)
from .algorithm.memory import CNMLibrary, CNMPlanner, PlacedRecord
from .algorithm.protocol import MODULES, ROLE_ORDER, make_verified_record
from .algorithm.rollout import RolloutPerturbation, run_physical_rollout
from .config import EnvConfig
from .env import CNMSwarmEnv


PRIMARY_VARIANTS = ("legacy_student", "nominal_dynamics", "eef_full")
ABLATION_VARIANTS = (
    "eef_no_outcome",
    "eef_no_differentiable",
    "eef_no_filter",
    "eef_temporal_decay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/eef_robust_agile_v1"))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--primary-seeds", type=int, default=5)
    parser.add_argument("--ablation-seeds", type=int, default=3)
    parser.add_argument("--evaluation-samples", type=int, default=256)
    parser.add_argument("--physical-repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _module_path(role: str, direction: int) -> list[PlacedRecord]:
    context = "east_low" if direction > 0 else "west_low"
    record, _ = make_verified_record(
        role,
        direction,
        origin_robot="eef-benchmark",
        immutable_hash="eef-benchmark-path-v1",
        context=context,
        logical_time=0.0,
    )
    library = CNMLibrary("eef-benchmark", record.immutable_hash)
    library.add_record(record)
    definition = MODULES[role]
    path, diagnostics = CNMPlanner(library).compose(
        (role,),
        (np.asarray(definition.anchor),),
        (np.asarray(definition.structural_key),),
        direction=direction,
        context=context,
        logical_time=0.0,
    )
    if not diagnostics["supported"]:
        raise RuntimeError(diagnostics)
    return path


def _train_models(args: argparse.Namespace) -> tuple[dict[str, dict[int, Any]], list[dict[str, Any]]]:
    variants = PRIMARY_VARIANTS + ABLATION_VARIANTS
    trained: dict[str, dict[int, Any]] = {variant: {} for variant in variants}
    master_rows: list[dict[str, Any]] = []
    for variant in variants:
        count = args.primary_seeds if variant in PRIMARY_VARIANTS else args.ablation_seeds
        for policy_seed in range(count):
            seed = 17 + 97 * policy_seed
            config = eef_config_for_variant(variant, EEFConfig())
            trainer = EEFTrainer(config, seed=seed, device=args.device)
            run_dir = args.output / "training" / variant / f"seed_{seed}"
            started = time.perf_counter()
            checkpoint, rows = trainer.fit(args.iterations, args.batch_size, run_dir)
            elapsed = time.perf_counter() - started
            trained[variant][seed] = trainer.model.eval()
            for row in rows:
                master_rows.append(
                    {
                        "variant": variant,
                        "policy_seed": seed,
                        "elapsed_seconds_total": elapsed,
                        **row,
                    }
                )
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "variant": variant,
                        "seed": seed,
                        "seconds": round(elapsed, 3),
                        "checkpoint": str(checkpoint),
                        "final_alignment": rows[-1]["registered_exit_error"],
                    }
                ),
                flush=True,
            )
    _write_csv(args.output / "source_data" / "training_curves.csv", master_rows)
    return trained, master_rows


def _synthetic_benchmarks(
    args: argparse.Namespace,
    trained: dict[str, dict[int, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    common_seeds = sorted(set.intersection(*(set(trained[name]) for name in PRIMARY_VARIANTS)))
    for seed in common_seeds:
        models = {name: trained[name][seed] for name in PRIMARY_VARIANTS}
        trial_rows = evaluate_policy_trials(
            models,
            EEFConfig(),
            samples_per_cell=args.evaluation_samples,
            seed=4103,
            device=args.device,
        )
        for row in trial_rows:
            primary_rows.append({"policy_seed": seed, **row})
        repeated = evaluate_interface_repeatability(
            models,
            EEFConfig(),
            commands=24 if not args.smoke else 4,
            repeats=12 if not args.smoke else 3,
            seed=5107,
            device=args.device,
        )
        for row in repeated:
            repeat_rows.append({"policy_seed": seed, **row})
        print(json.dumps({"stage": "synthetic", "policy_seed": seed}), flush=True)

    ablation_rows: list[dict[str, Any]] = []
    ablation_conditions = (
        DEFAULT_CONDITIONS[0],
        next(condition for condition in DEFAULT_CONDITIONS if condition.name == "combined"),
    )
    for variant in ("eef_full",) + ABLATION_VARIANTS:
        available = sorted(trained[variant].items())
        if variant == "eef_full":
            available = available[: args.ablation_seeds]
        for seed, model in available:
            trial_rows = evaluate_policy_trials(
                {variant: model},
                EEFConfig(),
                samples_per_cell=args.evaluation_samples,
                speeds=(1.2, 2.0),
                conditions=ablation_conditions,
                seed=6101,
                device=args.device,
            )
            for row in trial_rows:
                ablation_rows.append({"policy_seed": seed, **row})
    _write_csv(args.output / "source_data" / "synthetic_trials.csv", primary_rows)
    _write_csv(args.output / "source_data" / "interface_repeatability.csv", repeat_rows)
    _write_csv(args.output / "source_data" / "ablation_trials.csv", ablation_rows)
    return primary_rows, repeat_rows, ablation_rows


def _physical_benchmark(
    args: argparse.Namespace,
    trained: dict[str, dict[int, Any]],
) -> list[dict[str, Any]]:
    if args.skip_physical:
        return []
    deployed_seed = min(trained["eef_full"])
    models = {variant: trained[variant][deployed_seed] for variant in PRIMARY_VARIANTS}
    paths = {
        (role, direction): _module_path(role, direction)
        for role in ROLE_ORDER
        for direction in (-1, 1)
    }
    speeds = (0.8, 1.2, 1.6, 2.0) if not args.smoke else (1.0,)
    repeats = args.physical_repeats if not args.smoke else 1
    perturbations = {
        "nominal": RolloutPerturbation(action_noise=0.01, start_position_std=0.025),
        "depth_dropout_10": RolloutPerturbation(ray_dropout=0.10, action_noise=0.01, start_position_std=0.025),
        "depth_dropout_30": RolloutPerturbation(ray_dropout=0.30, action_noise=0.01, start_position_std=0.025),
        "delay_67ms": RolloutPerturbation(delay_steps=2, action_noise=0.01, start_position_std=0.025),
        "delay_133ms": RolloutPerturbation(delay_steps=4, action_noise=0.01, start_position_std=0.025),
        "payload_10": RolloutPerturbation(action_scale=0.84, action_noise=0.01, start_position_std=0.025),
    }
    if args.smoke:
        perturbations = {"nominal": perturbations["nominal"]}
    rows: list[dict[str, Any]] = []
    env = CNMSwarmEnv(
        EnvConfig(num_drones=1, ray_count=24, neighbor_k=0, episode_seconds=22.0)
    )
    try:
        # The full speed frontier is evaluated nominally. Perturbation cells use
        # the registered 1.2 m/s command and do not duplicate nominal attempts.
        cells: list[tuple[float, str, RolloutPerturbation]] = [
            (speed, "nominal", perturbations["nominal"]) for speed in speeds
        ]
        if not args.smoke:
            cells.extend(
                (1.2, name, perturbation)
                for name, perturbation in perturbations.items()
                if name != "nominal"
            )
        trial_index = 0
        for method, model in models.items():
            for speed, condition, perturbation in cells:
                for role in ROLE_ORDER:
                    for direction in (-1, 1):
                        for repeat_index in range(repeats):
                            navigator = EEFNavigator(model)
                            log_path = None
                            if (
                                method in ("legacy_student", "eef_full")
                                and condition == "nominal"
                                and abs(speed - speeds[min(1, len(speeds) - 1)]) < 1.0e-8
                                and role == "pillar_field_C"
                                and direction == 1
                                and repeat_index == 0
                            ):
                                log_path = (
                                    args.output
                                    / "rollouts"
                                    / f"{method}_pillar_field_C.jsonl"
                                )
                            result = run_physical_rollout(
                                env,
                                navigator,
                                [paths[(role, direction)]],
                                speed=speed,
                                max_steps=620,
                                waypoint_tolerance=0.38,
                                perturbation=perturbation,
                                seed=8101 + trial_index,
                                step_log=log_path,
                            )
                            rows.append(
                                {
                                    "method": method,
                                    "policy_seed": deployed_seed,
                                    "condition": condition,
                                    "commanded_speed_mps": speed,
                                    "module": role,
                                    "direction": direction,
                                    "repeat": repeat_index,
                                    **result.to_dict(),
                                }
                            )
                            trial_index += 1
                print(
                    json.dumps(
                        {
                            "stage": "pybullet",
                            "method": method,
                            "condition": condition,
                            "speed": speed,
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
    _write_csv(args.output / "source_data" / "pybullet_trials.csv", rows)
    return rows


def _summarize(
    args: argparse.Namespace,
    training: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    physical: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 2,
        "scope": "PyBullet and differentiable-surrogate evidence; no hardware trials",
        "training": {},
        "synthetic": {},
        "repeatability": {},
        "pybullet": {},
    }
    for method in PRIMARY_VARIANTS:
        final_rows = [
            row
            for row in training
            if row["variant"] == method and int(row["iteration"]) == args.iterations
        ]
        summary["training"][method] = {
            "seeds": len(final_rows),
            "final_alignment_mean": float(np.mean([row["registered_exit_error"] for row in final_rows])),
            "final_alignment_sd": float(np.std([row["registered_exit_error"] for row in final_rows], ddof=1))
            if len(final_rows) > 1
            else 0.0,
            "mean_training_seconds": float(np.mean([row["elapsed_seconds_total"] for row in final_rows])),
        }
        selected = [
            row
            for row in synthetic
            if row["method"] == method
            and row["condition"] == "nominal"
            and abs(float(row["commanded_speed_mps"]) - 2.0) < 1.0e-8
        ]
        successes = sum(bool(row["safe_completion"]) for row in selected)
        low, high = wilson_interval(successes, len(selected))
        summary["synthetic"][method] = {
            "n_at_2mps": len(selected),
            "safe_completion_at_2mps": successes / len(selected),
            "wilson95": [low, high],
            "mean_interface_position_error_m": float(
                np.mean([float(row["interface_position_error_m"]) for row in selected])
            ),
        }
        method_repeat = [row for row in repeatability if row["method"] == method]
        dispersions: list[float] = []
        for policy_seed in sorted({int(row["policy_seed"]) for row in method_repeat}):
            for command_id in sorted({int(row["command_id"]) for row in method_repeat}):
                cell = [
                    row
                    for row in method_repeat
                    if int(row["policy_seed"]) == policy_seed and int(row["command_id"]) == command_id
                ]
                if not cell:
                    continue
                xyz = np.asarray(
                    [[row["exit_dx_m"], row["exit_dy_m"], row["exit_dz_m"]] for row in cell],
                    dtype=np.float64,
                )
                center = xyz.mean(axis=0)
                scale = max(float(np.linalg.norm(center)), 1.0e-6)
                dispersions.extend((np.linalg.norm(xyz - center, axis=1) / scale).tolist())
        summary["repeatability"][method] = {
            "n_executions": len(dispersions),
            "mean_relative_exit_dispersion": float(np.mean(dispersions)),
            "median_relative_exit_dispersion": float(np.median(dispersions)),
        }
        method_physical = [
            row for row in physical if row["method"] == method and row["condition"] == "nominal"
        ]
        if method_physical:
            successes = sum(bool(row["success"]) for row in method_physical)
            low, high = wilson_interval(successes, len(method_physical))
            summary["pybullet"][method] = {
                "n_all_speeds_modules_directions": len(method_physical),
                "completion": successes / len(method_physical),
                "wilson95": [low, high],
                "collisions": int(sum(int(row["collision_events"]) for row in method_physical)),
            }
    summary["design"] = {
        "primary_policy_seeds": args.primary_seeds,
        "ablation_policy_seeds": args.ablation_seeds,
        "training_iterations": args.iterations,
        "batch_size": args.batch_size,
        "synthetic_trials_per_speed_condition_seed": args.evaluation_samples,
        "physical_repeats_per_module_direction": args.physical_repeats,
        "all_attempts_retained": True,
        "matched_scenes_across_methods": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.iterations = min(args.iterations, 8)
        args.batch_size = min(args.batch_size, 16)
        args.primary_seeds = 1
        args.ablation_seeds = 1
        args.evaluation_samples = min(args.evaluation_samples, 12)
        args.physical_repeats = 1
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trained, training = _train_models(args)
    synthetic, repeatability, ablations = _synthetic_benchmarks(args, trained)
    physical = _physical_benchmark(args, trained)
    summary = _summarize(args, training, synthetic, repeatability, ablations, physical)
    manifest = {
        "schema_version": 2,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
        "arguments": vars(args) | {"output": str(args.output)},
        "summary": summary,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
