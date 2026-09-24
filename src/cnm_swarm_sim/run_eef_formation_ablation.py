"""Run matched formation ablations for the experience substrate used by CNM.

The registered full EEF checkpoints are reused read-only.  Each ablation changes
one formation mechanism while retaining architecture, data budget, optimizer,
evaluation scenes and the downstream CNM admission gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithm.eef import EEFConfig, EEFTrainer, eef_config_for_variant, load_eef_checkpoint
from .algorithm.eef_benchmark import (
    BenchmarkCondition,
    DEFAULT_CONDITIONS,
    evaluate_interface_repeatability,
    evaluate_policy_trials,
)


POLICY_SEEDS = (17, 114, 211, 308, 405)
VARIANTS = (
    "eef_full",
    "eef_no_outcome",
    "eef_no_differentiable",
    "eef_no_filter",
    "eef_temporal_decay",
)
LABELS = {
    "eef_full": "Full EEF",
    "eef_no_outcome": "No terminal supervision",
    "eef_no_differentiable": "No differentiable rollout",
    "eef_no_filter": "No feasibility guidance",
    "eef_temporal_decay": "Attenuated rollout gradient",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_or_train(
    variant: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if variant == "eef_full":
        checkpoint = args.eef_run / "training" / variant / f"seed_{seed}" / "eef_policy.pt"
        source = "protected_eef_v5"
    else:
        checkpoint = args.output / "training" / variant / f"seed_{seed}" / "eef_policy.pt"
        source = "trained_ablation"
    if checkpoint.exists() and (variant == "eef_full" or args.resume):
        model, payload = load_eef_checkpoint(checkpoint, device=args.device, allow_source_rebind=True)
        if model.config.training_variant != variant or int(payload["seed"]) != seed:
            raise ValueError(f"checkpoint registration mismatch: {checkpoint}")
        if variant != "eef_full" and int(payload["iteration"]) != args.iterations:
            raise ValueError(f"checkpoint iteration mismatch: {checkpoint}")
        return model.eval(), {
            "variant": variant,
            "policy_seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "source": source,
            "training_seconds": None,
            "iterations": int(payload["iteration"]),
        }
    config = eef_config_for_variant(variant, EEFConfig())
    trainer = EEFTrainer(config, seed=seed, device=args.device)
    started = time.perf_counter()
    checkpoint, _ = trainer.fit(args.iterations, args.batch_size, checkpoint.parent)
    elapsed = time.perf_counter() - started
    return trainer.model.eval(), {
        "variant": variant,
        "policy_seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source": source,
        "training_seconds": elapsed,
        "iterations": args.iterations,
    }


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _circular_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(values), np.cos(values)))


def _seed_summary(
    variant: str,
    seed: int,
    trials: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [r for r in trials if r["method"] == variant]
    nominal = [r for r in selected if r["condition"] == "nominal"]
    combined = [r for r in selected if r["condition"] == "combined"]
    nominal_16 = [r for r in nominal if abs(float(r["commanded_speed_mps"]) - 1.6) < 1e-9]
    combined_20 = [r for r in combined if abs(float(r["commanded_speed_mps"]) - 2.0) < 1e-9]
    cnm_ready = [
        _bool(r["safe_completion"])
        and float(r["interface_position_error_m"]) <= 0.25
        and float(r["interface_velocity_error_mps"]) <= 0.35
        and float(r["interface_heading_error_rad"]) <= np.deg2rad(12.0)
        for r in nominal_16
    ]
    supervised = [r for r in selected if _bool(r["outcome_supervised"])]
    repeat = [r for r in repeatability if r["method"] == variant]
    compact: list[bool] = []
    position_residuals: list[float] = []
    for command_id in sorted({int(r["command_id"]) for r in repeat}):
        cell = [r for r in repeat if int(r["command_id"]) == command_id]
        states = np.asarray(
            [[r["exit_dx_m"], r["exit_dy_m"], r["exit_dz_m"], r["exit_vx_mps"],
              r["exit_vy_mps"], r["exit_vz_mps"], r["exit_yaw_rad"]] for r in cell],
            dtype=np.float64,
        )
        center = states.mean(axis=0)
        center[6] = np.arctan2(np.sin(states[:, 6]).mean(), np.cos(states[:, 6]).mean())
        pos = np.linalg.norm(states[:, :3] - center[:3], axis=1)
        position_residuals.extend(pos.tolist())
        desired = np.asarray(
            [[r["desired_exit_dx_m"], r["desired_exit_dy_m"], r["desired_exit_dz_m"],
              r["desired_exit_vx_mps"], r["desired_exit_vy_mps"], r["desired_exit_vz_mps"],
              r["desired_exit_yaw_rad"]] for r in cell],
            dtype=np.float64,
        )
        connector_pos = np.linalg.norm(states[:, :3] - desired[:, :3], axis=1)
        connector_vel = np.linalg.norm(states[:, 3:6] - desired[:, 3:6], axis=1)
        connector_yaw = _circular_abs(states[:, 6] - desired[:, 6])
        compact.extend(
            ((connector_pos <= 0.25) & (connector_vel <= 0.35) &
             (connector_yaw <= np.deg2rad(12.0)) &
             ~np.asarray([_bool(r["collision"]) for r in cell])).tolist()
        )
    return {
        "variant": variant,
        "label": LABELS[variant],
        "policy_seed": seed,
        "nominal_robust_agile_area": float(np.mean([_bool(r["safe_completion"]) for r in nominal])),
        "combined_2mps_completion": float(np.mean([_bool(r["safe_completion"]) for r in combined_20])),
        "cnm_ready_yield_1p6": float(np.mean(cnm_ready)),
        "terminal_position_error_m": float(np.mean([float(r["interface_position_error_m"]) for r in nominal_16])),
        "exit_repeatability_position_m": float(np.mean(position_residuals)),
        "downstream_connector_yield": float(np.mean(compact)),
        "coverage90": float(np.mean([_bool(r["coverage90"]) for r in supervised])) if supervised else None,
        "coverage90_absolute_error": abs(float(np.mean([_bool(r["coverage90"]) for r in supervised])) - 0.9)
        if supervised else None,
        "n_policy_trials": len(selected),
        "n_repeatability_executions": len(repeat),
    }


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "nominal_robust_agile_area",
        "combined_2mps_completion",
        "cnm_ready_yield_1p6",
        "terminal_position_error_m",
        "exit_repeatability_position_m",
        "downstream_connector_yield",
        "coverage90",
        "coverage90_absolute_error",
    )
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [r for r in seed_rows if r["variant"] == variant]
        result[variant] = {"label": LABELS[variant], "policy_seeds": len(rows)}
        for metric in metrics:
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            result[variant][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/eef_formation_ablation_v1"))
    parser.add_argument("--eef-run", type=Path, default=Path("runs/eef_training_efficiency_v5"))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.iterations = min(args.iterations, 8)
        args.batch_size = min(args.batch_size, 16)
        args.evaluation_samples = min(args.evaluation_samples, 12)
        args.seed_limit = 1
    seeds = POLICY_SEEDS[: args.seed_limit] if args.seed_limit else POLICY_SEEDS
    conditions: tuple[BenchmarkCondition, ...] = (
        DEFAULT_CONDITIONS[0],
        next(c for c in DEFAULT_CONDITIONS if c.name == "combined"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in VARIANTS:
            model, checkpoint_row = _load_or_train(variant, seed, args)
            checkpoint_rows.append(checkpoint_row)
            trials = evaluate_policy_trials(
                {variant: model},
                EEFConfig(),
                samples_per_cell=args.evaluation_samples,
                speeds=(1.2, 1.6, 2.0),
                conditions=conditions,
                seed=73001,
                device=args.device,
            )
            repeatability = evaluate_interface_repeatability(
                {variant: model},
                EEFConfig(),
                commands=4 if args.smoke else 24,
                repeats=3 if args.smoke else 12,
                seed=74003,
                device=args.device,
            )
            trial_rows.extend({"policy_seed": seed, **row} for row in trials)
            repeat_rows.extend({"policy_seed": seed, **row} for row in repeatability)
            seed_rows.append(_seed_summary(variant, seed, trials, repeatability))
            print(json.dumps({"stage": "evaluated", "variant": variant, "seed": seed}), flush=True)
    _write_csv(args.output / "source_data" / "formation_trials.csv", trial_rows)
    _write_csv(args.output / "source_data" / "repeatability_trials.csv", repeat_rows)
    _write_csv(args.output / "source_data" / "seed_summary.csv", seed_rows)
    _write_csv(args.output / "source_data" / "checkpoint_audit.csv", checkpoint_rows)
    summary = {
        "schema_version": 1,
        "status": "smoke_measured_simulation" if args.smoke else "measured_simulation",
        "scope": "matched differentiable flight simulation; protected EEF v5 checkpoints reused read-only",
        "policy_seeds": list(seeds),
        "single_factor_ablations": True,
        "matched_scenes": True,
        "all_attempts_retained": True,
        "definitions": {
            "cnm_ready_yield": "safe plus terminal position <=0.25 m, velocity <=0.35 m/s and yaw <=12 deg",
            "downstream_connector_yield": "collision-free repeat whose measured exit lies inside the registered downstream entry gate: position <=0.25 m, velocity <=0.35 m/s and yaw <=12 deg",
            "coverage90": "fraction of outcomes inside the predicted seven-dimensional 90% ellipsoid",
        },
        "variants": _aggregate(seed_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
    }
    _write_json(args.output / "statistical_summary.json", summary)
    _write_json(args.output / "run_manifest.json", {
        "arguments": vars(args) | {"output": str(args.output), "eef_run": str(args.eef_run)},
        "summary": summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
