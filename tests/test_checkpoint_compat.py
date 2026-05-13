"""checkpoint 向后兼容辅助逻辑测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.real_eval_support import _infer_prediction_feature_dim_from_payload


class CheckpointCompatTestCase(unittest.TestCase):
    """验证旧 checkpoint 可从权重形状恢复关键 encoder 参数。"""

    def test_infer_legacy_prediction_feature_dim(self) -> None:
        payload = {
            "network_state_dict": {
                "encoder._prediction_projection.0.weight": torch.zeros(64, 8),
            },
        }

        self.assertEqual(_infer_prediction_feature_dim_from_payload(payload), 8)

    def test_missing_prediction_projection_returns_none(self) -> None:
        self.assertIsNone(_infer_prediction_feature_dim_from_payload({"network_state_dict": {}}))


if __name__ == "__main__":
    unittest.main()
