"""Regression tests for the uncertainty-calibrated MAPPO model head."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.models import UncertaintyTransitionEnsemble


class UncertaintyTransitionEnsembleTestCase(unittest.TestCase):
    def _rows(self, count: int = 80) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(count):
            rows.append(
                {
                    "observation": np.full(9, index / count, dtype=np.float32),
                    "action": index % 5,
                    "reward": float(index % 3),
                    "next_value": 0.5,
                    "terminated": False,
                    "next_observation": np.full(9, (index + 1) / count, dtype=np.float32),
                }
            )
        return rows

    def test_fit_uses_bootstrapped_td_target_and_predicts_uncertainty(self) -> None:
        model = UncertaintyTransitionEnsemble(min_samples=64, random_seed=3)
        stats = model.fit(self._rows())
        second_stats = model.fit(self._rows())
        prediction = model.predict(np.zeros(9, dtype=np.float32), [0, 1, 2, 3, 4])

        self.assertTrue(stats["ready"])
        self.assertEqual(second_stats["sample_count"], 160)
        self.assertEqual(len(prediction["td_target_mean"]), 5)
        self.assertEqual(len(prediction["td_target_std"]), 5)
        self.assertNotIn("return_mean", prediction)
        self.assertTrue(np.isfinite(prediction["td_target_mean"]).all())

    def test_state_dict_preserves_model_contract(self) -> None:
        model = UncertaintyTransitionEnsemble(min_samples=64, discount=0.95, random_seed=5)
        model.fit(self._rows())
        restored = UncertaintyTransitionEnsemble(min_samples=64, discount=0.95, random_seed=5)
        restored.load_state_dict(model.state_dict())

        self.assertTrue(restored.ready)
        self.assertEqual(restored.discount, 0.95)
        np.testing.assert_allclose(
            model.predict(np.zeros(9, dtype=np.float32), [2])["td_target_mean"],
            restored.predict(np.zeros(9, dtype=np.float32), [2])["td_target_mean"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_state_dict_is_torch_checkpoint_serializable(self) -> None:
        model = UncertaintyTransitionEnsemble(min_samples=64, random_seed=9)
        model.fit(self._rows())
        with TemporaryDirectory() as directory:
            path = f"{directory}/model.pt"
            torch.save(model.state_dict(), path)
            payload = torch.load(path, map_location="cpu")
        self.assertIsInstance(payload["replay_inputs"], list)
        self.assertEqual(len(payload["replay_inputs"]), 80)


if __name__ == "__main__":
    unittest.main()
