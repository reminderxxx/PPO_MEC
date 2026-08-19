"""模型目录与 adapter cache 定义。"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.envs.specs import AdapterStateBundle, CacheObject


TYPED_MODEL_CACHE_CONTRACT_VERSION = "1.0.0"
LEGACY_MODEL_CACHE_PROFILE_ID = "legacy_adapter_only_v1"
TYPED_MODEL_CACHE_PROFILE_ID = "typed_base_adapter_state_v1"
TYPED_CACHE_OBJECT_TYPES = frozenset(
    {"base_model", "adapter", "workflow_state", "kv_prefix"}
)
ACTIVE_TYPED_CACHE_OBJECT_TYPES = frozenset(
    {"base_model", "adapter", "workflow_state"}
)


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
class RSUTypedCacheProfile:
    """Explicit initial typed residents for one RSU."""

    rsu_id: str
    resident_object_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TypedCacheObject:
    """Versioned AI model-cache object with explicit dependency semantics."""

    object_id: str
    object_type: str
    version: str
    resident_size_mb: float
    transfer_size_mb: float
    source: str
    provenance: dict[str, Any]
    base_model_family: str | None
    base_model_id: str | None
    required_base_model_id: str | None
    adapter_id: str | None
    workflow_identity: str | None
    shareability_scope: str
    mutability: str
    persistence: str
    evictability: str
    migration_semantics: str
    dependency_ids: list[str]
    dataset_profile_source: str
    license_status: str
    formal_use_status: str
    stable_fingerprint: str
    availability: str
    counts_toward_capacity: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TypedSizeResolution:
    object_id: str
    object_type: str
    resident_size_mb: float
    transfer_size_mb: float
    source: str


@dataclass(frozen=True)
class TypedPlacementPlan:
    requested_adapter_id: str
    required_base_model_id: str
    ordered_object_ids: list[str]
    missing_object_ids: list[str]
    already_resident_object_ids: list[str]
    requested_bundle_mb: float
    transfer_mb_by_type: dict[str, float]

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


@dataclass(frozen=True)
class AdapterSizeResolution:
    """Auditable resident-cache size resolution for one adapter."""

    adapter_id: str
    size_mb: float
    source: str
    object_id: str | None


@dataclass
class AdapterCatalog:
    """统一管理车载基础模型、路侧 cache 与状态迁移包。"""

    vehicle_base_models: list[VehicleBaseModelProfile]
    rsu_adapter_caches: list[RSUAdapterCacheProfile]
    adapter_state_bundles: list[AdapterStateBundle]
    cache_objects: list[CacheObject]
    model_cache_datasets: list[ModelCacheDatasetProfile] = field(default_factory=list)
    model_cache_profile_id: str = LEGACY_MODEL_CACHE_PROFILE_ID
    typed_model_cache_contract_version: str = TYPED_MODEL_CACHE_CONTRACT_VERSION
    typed_cache_objects: list[TypedCacheObject] = field(default_factory=list)
    rsu_typed_cache_profiles: list[RSUTypedCacheProfile] = field(default_factory=list)
    compatibility_map: dict[str, list[str]] = field(default_factory=dict)
    kv_prefix_enabled: bool = False
    vehicle_adapter_residency_enabled: bool = False

    @classmethod
    def from_json(cls, file_path: str | Path) -> "AdapterCatalog":
        raw_data = json.loads(Path(file_path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(raw_data)

    @classmethod
    def from_dict(cls, raw_data: dict[str, Any]) -> "AdapterCatalog":
        profile_id = str(
            raw_data.get("model_cache_profile_id") or LEGACY_MODEL_CACHE_PROFILE_ID
        )
        catalog = cls(
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
            model_cache_profile_id=profile_id,
            typed_model_cache_contract_version=str(
                raw_data.get("typed_model_cache_contract_version")
                or TYPED_MODEL_CACHE_CONTRACT_VERSION
            ),
            typed_cache_objects=[
                TypedCacheObject(**item)
                for item in raw_data.get("typed_cache_objects", [])
            ],
            rsu_typed_cache_profiles=[
                RSUTypedCacheProfile(**item)
                for item in raw_data.get("rsu_typed_cache_profiles", [])
            ],
            compatibility_map={
                str(key): [str(value) for value in values]
                for key, values in dict(raw_data.get("compatibility_map", {})).items()
            },
            kv_prefix_enabled=bool(raw_data.get("kv_prefix_enabled", False)),
            vehicle_adapter_residency_enabled=bool(
                raw_data.get("vehicle_adapter_residency_enabled", False)
            ),
        )
        catalog.validate_typed_catalog()
        return catalog

    @property
    def typed_mode_enabled(self) -> bool:
        return self.model_cache_profile_id == TYPED_MODEL_CACHE_PROFILE_ID

    def _legacy_required_base_model_id(self) -> str:
        base_ids = sorted(set(self.get_vehicle_base_model_ids()))
        if len(base_ids) != 1:
            raise ValueError(
                "legacy cache object requires exactly one unambiguous vehicle base model"
            )
        return base_ids[0]

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def compute_object_fingerprint(cls, payload: dict[str, Any]) -> str:
        normalized = dict(payload)
        normalized.pop("stable_fingerprint", None)
        return hashlib.sha256(cls._canonical_json(normalized).encode("utf-8")).hexdigest()

    def canonical_fingerprint(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(self._canonical_json(payload).encode("utf-8")).hexdigest()

    def validate_typed_catalog(self) -> dict[str, Any]:
        if self.model_cache_profile_id not in {
            LEGACY_MODEL_CACHE_PROFILE_ID,
            TYPED_MODEL_CACHE_PROFILE_ID,
        }:
            raise ValueError(f"unsupported model cache profile: {self.model_cache_profile_id}")
        if self.typed_model_cache_contract_version != TYPED_MODEL_CACHE_CONTRACT_VERSION:
            raise ValueError("unsupported typed model cache contract version")
        if not isinstance(self.compatibility_map, dict):
            raise ValueError("compatibility_map must be a mapping")
        if not self.typed_mode_enabled:
            return {
                "status": "pass",
                "profile_id": self.model_cache_profile_id,
                "typed_object_count": len(self.typed_cache_objects),
                "legacy_mapping": "cache_objects_explicitly_map_to_adapter_on_demand",
            }
        if not self.typed_cache_objects:
            raise ValueError("typed model cache profile requires typed_cache_objects")
        object_by_id: dict[str, TypedCacheObject] = {}
        identity_keys: set[tuple[str, str, str]] = set()
        adapter_ids: set[str] = set()
        base_ids: set[str] = set()
        for item in self.typed_cache_objects:
            if not item.object_id or item.object_id in object_by_id:
                raise ValueError(f"duplicate or empty typed object_id: {item.object_id!r}")
            if item.object_type not in TYPED_CACHE_OBJECT_TYPES:
                raise ValueError(f"unsupported typed object_type: {item.object_type}")
            if item.object_type == "kv_prefix" and self.kv_prefix_enabled:
                raise ValueError("kv_prefix is reserved and cannot be enabled by G13")
            for field_name, value in (
                ("resident_size_mb", item.resident_size_mb),
                ("transfer_size_mb", item.transfer_size_mb),
            ):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{item.object_id}.{field_name} must be finite and positive")
            try:
                self._canonical_json(item.provenance)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{item.object_id}.provenance must be JSON-safe") from exc
            identity = item.adapter_id or item.base_model_id or item.workflow_identity
            if not identity:
                raise ValueError(f"{item.object_id} lacks a stable type identity")
            identity_key = (item.object_type, item.version, str(identity))
            if identity_key in identity_keys:
                raise ValueError(f"duplicate typed identity: {identity_key}")
            identity_keys.add(identity_key)
            expected_fingerprint = self.compute_object_fingerprint(item.to_dict())
            if item.stable_fingerprint != expected_fingerprint:
                raise ValueError(f"stable fingerprint mismatch for {item.object_id}")
            if (
                item.dataset_profile_source == "G11_hf_metadata"
                and item.license_status == "unknown"
                and "formal" in item.formal_use_status
                and "non_formal" not in item.formal_use_status
            ):
                raise ValueError("unknown-license HF metadata cannot be formal-ready")
            if (
                item.provenance.get("size_provenance_anomaly") is True
                and item.availability != "blocked_provenance_anomaly"
            ):
                raise ValueError("size provenance anomaly must block automatic availability")
            if item.object_type == "base_model":
                if not item.base_model_id or item.required_base_model_id is not None:
                    raise ValueError("base_model requires base_model_id and no required_base_model_id")
                base_ids.add(item.base_model_id)
            elif item.object_type == "adapter":
                if not item.adapter_id or not item.required_base_model_id:
                    raise ValueError("adapter requires adapter_id and unique required_base_model_id")
                if item.adapter_id in adapter_ids:
                    raise ValueError(f"duplicate adapter mapping: {item.adapter_id}")
                adapter_ids.add(item.adapter_id)
            elif item.object_type == "workflow_state":
                if not item.workflow_identity or item.counts_toward_capacity:
                    raise ValueError(
                        "workflow_state requires workflow_identity and is migration-only in G13"
                    )
            object_by_id[item.object_id] = item
        for item in self.typed_cache_objects:
            if item.object_type == "adapter":
                if item.required_base_model_id not in base_ids:
                    raise ValueError(
                        f"adapter {item.adapter_id} references missing base {item.required_base_model_id}"
                    )
                base_object = next(
                    candidate
                    for candidate in self.typed_cache_objects
                    if candidate.object_type == "base_model"
                    and candidate.base_model_id == item.required_base_model_id
                )
                if item.base_model_family != base_object.base_model_family:
                    raise ValueError(f"adapter/base family mismatch for {item.adapter_id}")
                if item.dependency_ids != [base_object.object_id]:
                    raise ValueError(
                        f"adapter {item.adapter_id} must declare exactly its base object dependency"
                    )
            for dependency_id in item.dependency_ids:
                if dependency_id not in object_by_id:
                    raise ValueError(f"missing dependency {dependency_id} for {item.object_id}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(object_id: str) -> None:
            if object_id in visiting:
                raise ValueError("typed cache dependency cycle detected")
            if object_id in visited:
                return
            visiting.add(object_id)
            for dependency_id in object_by_id[object_id].dependency_ids:
                visit(dependency_id)
            visiting.remove(object_id)
            visited.add(object_id)

        for object_id in sorted(object_by_id):
            visit(object_id)
        for base_model_id, compatible_adapters in self.compatibility_map.items():
            if base_model_id not in base_ids:
                raise ValueError(f"compatibility map references unknown base: {base_model_id}")
            if len(compatible_adapters) != len(set(compatible_adapters)):
                raise ValueError(f"duplicate compatibility mapping for base: {base_model_id}")
            if any(adapter_id not in adapter_ids for adapter_id in compatible_adapters):
                raise ValueError(f"compatibility map references unknown adapter for {base_model_id}")
        declared_pairs = {
            (base_model_id, adapter_id)
            for base_model_id, adapter_ids_for_base in self.compatibility_map.items()
            for adapter_id in adapter_ids_for_base
        }
        actual_pairs = {
            (item.required_base_model_id, item.adapter_id)
            for item in self.typed_cache_objects
            if item.object_type == "adapter"
        }
        if declared_pairs != actual_pairs:
            raise ValueError("compatibility_map is incomplete or ambiguous")
        for profile in self.rsu_typed_cache_profiles:
            if len(profile.resident_object_ids) != len(set(profile.resident_object_ids)):
                raise ValueError(f"duplicate initial resident for {profile.rsu_id}")
            for object_id in profile.resident_object_ids:
                item = object_by_id.get(object_id)
                if item is None:
                    raise ValueError(f"unknown initial typed resident: {object_id}")
                if item.object_type == "kv_prefix" or not item.counts_toward_capacity:
                    raise ValueError(f"invalid long-lived typed resident: {object_id}")
            resident_set = set(profile.resident_object_ids)
            for object_id in resident_set:
                if not set(object_by_id[object_id].dependency_ids).issubset(resident_set):
                    raise ValueError(f"orphan initial typed resident: {profile.rsu_id}/{object_id}")
        return {
            "status": "pass",
            "profile_id": self.model_cache_profile_id,
            "typed_object_count": len(self.typed_cache_objects),
            "catalog_fingerprint": self.canonical_fingerprint(),
        }

    def typed_objects_with_legacy_projection(self) -> list[TypedCacheObject]:
        if self.typed_cache_objects:
            return list(self.typed_cache_objects)
        required_base = self._legacy_required_base_model_id()
        base_family = next(
            item.family for item in self.vehicle_base_models if item.base_model_id == required_base
        )
        projected: list[TypedCacheObject] = []
        for item in self.cache_objects:
            payload = {
                "object_id": item.object_id,
                "object_type": "adapter",
                "version": "legacy-v1",
                "resident_size_mb": float(item.size_mb),
                "transfer_size_mb": float(item.size_mb),
                "source": item.source,
                "provenance": {"projection": "legacy_cache_object", "formal_ready": False},
                "base_model_family": base_family,
                "base_model_id": None,
                "required_base_model_id": required_base,
                "adapter_id": item.adapter_id,
                "workflow_identity": None,
                "shareability_scope": "rsu",
                "mutability": "immutable",
                "persistence": "episode_resident",
                "evictability": "evictable",
                "migration_semantics": "independent_adapter_transfer",
                "dependency_ids": [],
                "dataset_profile_source": "repository_legacy_catalog",
                "license_status": "repository_native",
                "formal_use_status": "legacy_compatible_only",
                "stable_fingerprint": "",
                "availability": "available",
                "counts_toward_capacity": True,
            }
            payload["stable_fingerprint"] = self.compute_object_fingerprint(payload)
            projected.append(TypedCacheObject(**payload))
        return projected

    def get_typed_object(self, object_id: str) -> TypedCacheObject:
        matches = [item for item in self.typed_cache_objects if item.object_id == object_id]
        if len(matches) != 1:
            raise ValueError(f"typed object must resolve uniquely: {object_id}")
        return matches[0]

    def get_typed_adapter(self, adapter_id: str) -> TypedCacheObject:
        matches = [
            item for item in self.typed_cache_objects
            if item.object_type == "adapter" and item.adapter_id == adapter_id
        ]
        if len(matches) != 1:
            raise ValueError(f"typed adapter must resolve uniquely: {adapter_id}")
        return matches[0]

    def get_typed_base(self, base_model_id: str) -> TypedCacheObject:
        matches = [
            item for item in self.typed_cache_objects
            if item.object_type == "base_model" and item.base_model_id == base_model_id
        ]
        if len(matches) != 1:
            raise ValueError(f"typed base model must resolve uniquely: {base_model_id}")
        return matches[0]

    def get_initial_typed_residents(self, rsu_id: str) -> list[str]:
        for profile in self.rsu_typed_cache_profiles:
            if profile.rsu_id == rsu_id:
                return list(profile.resident_object_ids)
        return []

    def resolve_typed_size(self, object_id: str) -> TypedSizeResolution:
        item = self.get_typed_object(object_id)
        return TypedSizeResolution(
            object_id=item.object_id,
            object_type=item.object_type,
            resident_size_mb=float(item.resident_size_mb),
            transfer_size_mb=float(item.transfer_size_mb),
            source=item.source,
        )

    def resolve_typed_placement_plan(
        self, *, adapter_id: str, resident_object_ids: list[str]
    ) -> TypedPlacementPlan:
        adapter = self.get_typed_adapter(adapter_id)
        base = self.get_typed_base(str(adapter.required_base_model_id))
        ordered = [base.object_id, adapter.object_id]
        resident = set(resident_object_ids)
        missing = [object_id for object_id in ordered if object_id not in resident]
        transfer: dict[str, float] = {}
        for object_id in missing:
            item = self.get_typed_object(object_id)
            transfer[item.object_type] = transfer.get(item.object_type, 0.0) + float(
                item.transfer_size_mb
            )
        return TypedPlacementPlan(
            requested_adapter_id=adapter_id,
            required_base_model_id=str(adapter.required_base_model_id),
            ordered_object_ids=ordered,
            missing_object_ids=missing,
            already_resident_object_ids=[item for item in ordered if item in resident],
            requested_bundle_mb=sum(
                self.get_typed_object(item).resident_size_mb for item in missing
            ),
            transfer_mb_by_type={key: round(value, 6) for key, value in sorted(transfer.items())},
        )

    def resolve_workflow_state_object(self) -> TypedCacheObject | None:
        matches = [
            item for item in self.typed_cache_objects
            if item.object_type == "workflow_state" and item.availability == "available"
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("workflow state migration profile must resolve uniquely")
        return matches[0]

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

    def resolve_adapter_resident_size_mb(self, adapter_id: str | None) -> AdapterSizeResolution:
        """Resolve and validate the resident size used by cache capacity arithmetic.

        The legacy 64 MB transfer fallback remains the resident-size fallback until
        the catalog provides an explicit CacheObject. It is never interpreted as 0.
        """
        normalized_id = str(adapter_id or "").strip()
        if not normalized_id:
            raise ValueError("adapter_id is required for resident size resolution")
        cache_object = next(
            (item for item in self.cache_objects if item.adapter_id == normalized_id),
            None,
        )
        size_mb = float(cache_object.size_mb) if cache_object else float(
            self.estimate_adapter_transfer_size_mb(normalized_id)
        )
        if not math.isfinite(size_mb) or size_mb <= 0.0:
            source = "catalog_cache_object" if cache_object else "catalog_fallback"
            raise ValueError(
                f"invalid resident size_mb for adapter {normalized_id!r} from {source}: {size_mb!r}"
            )
        return AdapterSizeResolution(
            adapter_id=normalized_id,
            size_mb=size_mb,
            source="catalog_cache_object" if cache_object else "catalog_fallback",
            object_id=cache_object.object_id if cache_object else None,
        )

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
            "model_cache_profile_id": self.model_cache_profile_id,
            "typed_model_cache_contract_version": self.typed_model_cache_contract_version,
            "typed_cache_objects": [item.to_dict() for item in self.typed_cache_objects],
            "rsu_typed_cache_profiles": [
                item.to_dict() for item in self.rsu_typed_cache_profiles
            ],
            "compatibility_map": {
                key: list(values) for key, values in sorted(self.compatibility_map.items())
            },
            "kv_prefix_enabled": self.kv_prefix_enabled,
            "vehicle_adapter_residency_enabled": self.vehicle_adapter_residency_enabled,
        }
