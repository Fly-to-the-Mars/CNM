from __future__ import annotations

import copy

import numpy as np

from cnm_swarm_sim.algorithm.experience import CompilerConfig, ExperienceCompiler, FlightSegment
from cnm_swarm_sim.algorithm.memory import CNMLibrary, CNMPlanner, ExecutionEvent
from cnm_swarm_sim.algorithm.protocol import ROLE_ORDER, make_verified_record, task_spec
from cnm_swarm_sim.algorithm.reconfigurable_protocol import reconfigurable_tasks


def make_library() -> tuple[CNMLibrary, dict[str, object]]:
    library = CNMLibrary("test-robot", "fixed-stack")
    records = {}
    for index, role in enumerate(ROLE_ORDER):
        record, event = make_verified_record(
            role, 1, origin_robot=f"source-{index}", immutable_hash="fixed-stack",
            context="east_low", logical_time=float(index),
        )
        assert library.add_record(record)
        assert library.add_event(event)
        records[role] = record
    return library, records


def segment(index: int, offset: float = 0.0) -> FlightSegment:
    x = np.linspace(-1.0, 1.0 + offset, 9)
    positions = np.stack((x, np.full_like(x, 0.01 * index), np.full_like(x, 1.2)), axis=1)
    return FlightSegment(
        segment_id=f"segment-{index}", role="local_response",
        structural_key=np.asarray((1.0, 2.0, 3.0)), context="nominal",
        anchor=np.zeros(3), direction=1, origin_robot="uav0",
        immutable_hash="fixed-stack", logical_time=float(index),
        positions=positions, velocities=np.gradient(positions, axis=0) * 10.0,
        yaws=np.zeros(len(x)), actions=np.zeros((len(x), 4)), success=True,
        duration=0.8, minimum_clearance=0.3,
    )


def test_verified_record_uses_measured_interfaces() -> None:
    result = ExperienceCompiler(CompilerConfig(min_successes=3)).compile(
        [segment(i, 0.01 * i) for i in range(6)]
    )
    assert result.record.entry.sample_count == 6
    assert result.record.exit.sample_count == 6
    assert len(result.events) == 6
    assert result.record.exit.support_samples.shape == (6, 7)


def test_composition_deletion_and_exact_restoration() -> None:
    library, records = make_library()
    roles, anchors, keys = task_spec(1)
    path, diagnostics = CNMPlanner(library).compose(
        roles, anchors, keys, direction=1, context="east_low", logical_time=10.0
    )
    assert path and diagnostics["supported"]
    baseline = library.digest()
    removed = library.delete_records((records[ROLE_ORDER[1]].record_id,), reason="causal-test")
    path, _ = CNMPlanner(library).compose(
        roles, anchors, keys, direction=1, context="east_low", logical_time=10.0
    )
    assert not path
    library.restore_records(removed, reason="exact-restoration")
    assert library.digest() == baseline


def test_transfer_is_idempotent_and_tampering_is_rejected() -> None:
    donor, records = make_library()
    record = records[ROLE_ORDER[0]]
    packet, _ = donor.select_for_query(
        (record.role,), "east_low", set(), logical_time=10.0,
        byte_budget=100_000, structural_keys={record.role: record.structural_key},
    )
    recipient = CNMLibrary("recipient", donor.immutable_hash)
    recipient.merge_packet(packet)
    digest = recipient.digest()
    recipient.merge_packet(packet)
    assert recipient.digest() == digest
    tampered = copy.deepcopy(packet)
    tampered["sender"] = "tampered"
    assert recipient.merge_packet(tampered)["reason"] == "payload_hash_mismatch"


def test_source_linked_failure_revises_reliability() -> None:
    library, records = make_library()
    record = records[ROLE_ORDER[0]]
    before = library.reliability(record.record_id, "east_low", 10.0)
    failure = ExecutionEvent.create(
        record_id=record.record_id, origin_robot=record.origin_robot,
        executor_robot="recipient", context="east_low", logical_time=10.0,
        success=False, duration=2.0, min_clearance=0.02,
        prediction_error=1.0, tracking_error=0.5,
        reason="recipient_execution_failure", immutable_hash=library.immutable_hash,
    )
    assert library.add_event(failure)
    assert library.reliability(record.record_id, "east_low", 10.0) < before


def test_registered_protocol_has_unseen_acb() -> None:
    tasks = reconfigurable_tasks()
    assert len(tasks) == 24
    acb = next(task for task in tasks if task.task_id == "R08_ACB_explicit")
    assert acb.roles == (ROLE_ORDER[0], ROLE_ORDER[2], ROLE_ORDER[1])
    assert not acb.acquisition_order_seen
