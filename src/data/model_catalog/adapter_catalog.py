"""模型目录与 adapter cache 定义。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.envs.specs import AdapterStateBundle, CacheObject


@dataclass
class VehicleBaseModelProfile:
    """车载基础模型条目。"""

    base_model_id: str
    family: str
    memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RSUAdapterCacheProfile:
    """RSU 侧 adapter cache 条目。"""

    rsu_id: str
    cached_adapter_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCacheDatasetProfile:
    """外部 model-cache 数据源声明。"""

    dataset_id: str
    dataset_name: str
    provider: str
    download_page_url: str
    local_status: str
    usage_scope: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterCatalog:
    """统一管理车载基础模型、路侧 cache 与状态迁移包。"""

    vehicle_base_models: list[VehicleBaseModelProfile]
    rsu_adapter_caches: list[RSUAdapterCacheProfile]
    adapter_state_bundles: list[AdapterStateBundle]
    cache_objects: list[CacheObject]
    model_cache_datasets: list[ModelCacheDatasetProfile] = field(default_factory=list)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "AdapterCatalog":
        raw_data = json.loads(Path(file_path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(raw_data)

    @classmethod
    def from_dict(cls, raw_data: dict[str, Any]) -> "AdapterCatalog":
        return cls(
            vehicle_base_models=[
                VehicleBaseModelProfile(**item)
                for item in raw_data["vehicle_base_models"]
            ],
            rsu_adapter_caches=[
                RSUAdapterCacheProfile(**item)
                for item in raw_data["rsu_adapter_caches"]
            ],
            adapter_state_bundles=[
                AdapterStateBundle(**item)
                for item in raw_data["adapter_state_bundles"]
            ],
            cache_objects=[CacheObject(**item) for item in raw_data["cache_objects"]],
            model_cache_datasets=[
                ModelCacheDatasetProfile(**item)
                for item in raw_data.get("model_cache_datasets", [])
            ],
        )

    def get_vehicle_base_model_ids(self) -> list[str]:
        return [item.base_model_id for item in self.vehicle_base_models]

    def get_model_cache_dataset_ids(self) -> list[str]:
        return [item.dataset_id for item in self.model_cache_datasets]

    def get_model_cache_download_pages(self) -> dict[str, str]:
        return {
            item.dataset_id: item.download_page_url
            for item in self.model_cache_datasets
        }

    def get_initial_cached_adapters(self, rsu_id: str) -> list[str]:
        for profile in self.rsu_adapter_caches:
            if profile.rsu_id == rsu_id:
                return list(profile.cached_adapter_ids)
        return []

    def has_cached_adapter(self, rsu_id: str, adapter_id: str) -> bool:
        return adapter_id in self.get_initial_cached_adapters(rsu_id)

    def ensure_cached_adapter(self, rsu_id: str, adapter_id: str) -> bool:
        for profile in self.rsu_adapter_caches:
            if profile.rsu_id == rsu_id:
                if adapter_id not in profile.cached_adapter_ids:
                    profile.cached_adapter_ids.append(adapter_id)
                return True
        return False

    def clone_with_cache_plan(self, cache_plan: dict[str, list[str]]) -> "AdapterCatalog":
        payload = self.to_dict()
        for cache_profile in payload["rsu_adapter_caches"]:
            rsu_id = cache_profile["rsu_id"]
            if rsu_id in cache_plan:
                cache_profile["cached_adapter_ids"] = list(cache_plan[rsu_id])
        return AdapterCatalog.from_dict(payload)

    def estimate_adapter_transfer_size_mb(self, adapter_id: str | None) -> float:
        """估计 adapter 通过回传链路下发的流量成本。"""
        if adapter_id is None:
            return 0.0
        for cache_object in self.cache_objects:
            if cache_object.adapter_id == adapter_id:
                return float(cache_object.size_mb)
        return 64.0

    def estimate_bundle_transfer_size_mb(self, adapter_id: str | None) -> float:
        """估计 adapter-state bundle 的迁移开销。"""
        if adapter_id is None:
            return 0.0
        for bundle in self.adapter_state_bundles:
            if bundle.adapter_id == adapter_id:
                return 32.0
        return 16.0

    def has_state_bundle(self, adapter_id: str | None) -> bool:
        """检查是否存在可迁移状态包。"""
        if adapter_id is None:
            return False
        return any(bundle.adapter_id == adapter_id for bundle in self.adapter_state_bundles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_base_models": [item.to_dict() for item in self.vehicle_base_models],
            "rsu_adapter_caches": [item.to_dict() for item in self.rsu_adapter_caches],
            "adapter_state_bundles": [item.to_dict() for item in self.adapter_state_bundles],
            "cache_objects": [item.to_dict() for item in self.cache_objects],
            "model_cache_datasets": [item.to_dict() for item in self.model_cache_datasets],
        }
