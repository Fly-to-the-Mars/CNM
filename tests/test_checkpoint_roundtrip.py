from __future__ import annotations

import torch

from cnm_swarm_sim.algorithm.eef import EEFConfig, EEFTrainer, load_eef_checkpoint, model_sha256


def test_checkpoint_roundtrip_preserves_hash_and_predictions(tmp_path) -> None:
    """The distributed package must contain every checkpoint-hash dependency."""
    trainer = EEFTrainer(EEFConfig(), seed=17, device="cpu")
    original = trainer.model.eval()
    expected_hash = model_sha256(original, original.config)
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    restored, payload = load_eef_checkpoint(path)
    assert payload["immutable_sha256"] == expected_hash
    assert model_sha256(restored, restored.config) == expected_hash
    features = torch.zeros((2, original.config.input_dim))
    with torch.no_grad():
        expected = original(features)
        actual = restored(features)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)
