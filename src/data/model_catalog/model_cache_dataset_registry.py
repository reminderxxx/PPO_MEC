"""Validation and artifact projection for the model-cache dataset registry."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


REGISTRY_VERSION = "1.0.0"
TAXONOMY = {
    "joint_vec_ai_cache_trace",
    "model_serving_request_trace",
    "adapter_or_lora_workload_trace",
    "kv_prefix_cache_trace",
    "model_artifact_size_metadata",
    "generic_ai_workload_trace",
    "content_dataset_not_cache_trace",
    "paper_only_or_unavailable",
    "rejected_or_unsafe",
}
FIELD_AVAILABILITY = {"present", "derivable", "absent", "unknown"}
FIELD_NAMES = {
    "timestamp",
    "request_id",
    "client_tenant_vehicle_id",
    "location_rsu",
    "model_id",
    "adapter_lora_id",
    "base_model_id",
    "object_file_size",
    "token_input_output_size",
    "request_latency",
    "loading_latency",
    "inference_latency",
    "transfer_latency",
    "cache_hit_miss",
    "eviction",
    "reuse_prefix_identity",
    "hardware_resource",
    "mobility_handoff",
    "dag_workflow",
}
SCORE_WEIGHTS = {
    "temporal_request_sequence": 20,
    "stable_model_adapter_cache_object_identity": 20,
    "bytes_size": 15,
    "reuse_cache_semantics": 10,
    "load_inference_transfer_latency": 10,
    "client_tenant_identity": 5,
    "mobility_location_rsu": 10,
    "license_provenance_reproducibility": 10,
}
RECOMMENDATIONS = {
    "direct_request_trace_candidate",
    "adapter_request_profile_candidate",
    "kv_reuse_profile_candidate",
    "model_size_profile_candidate",
    "arrival_token_profile_candidate",
    "metadata_reference_only",
    "rejected",
}
REQUIRED_IDENTITY = {
    "source_key",
    "name",
    "provider",
    "owner",
    "primary_class",
    "secondary_roles",
    "canonical_landing_page",
    "repository_or_api_url",
    "version_revision",
    "accessed_at",
    "discovery_query",
}
REQUIRED_ACCESS = {
    "public_availability",
    "authentication_required",
    "gated",
    "license",
    "license_url",
    "terms_usage_restrictions",
    "download_mechanism",
    "approximate_size",
    "file_count",
    "file_formats",
    "checksum_revision_metadata",
    "raw_download_performed",
}
REQUIRED_EVIDENCE = {
    "supporting_source_urls",
    "evidence_summary",
    "last_verified_date",
    "verification_status",
    "unresolved_questions",
}


class RegistryValidationError(ValueError):
    """Raised when a registry or a compatibility projection is invalid."""


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _is_http_url(value: Any) -> bool:
    parsed = urlparse(str(value or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _walk_numbers(value: Any, path: str = "$") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{path}[{index}]")


def _available(fields: dict[str, str], name: str) -> bool:
    return fields.get(name) in {"present", "derivable"}


def _error(errors: list[dict[str, str]], code: str, message: str, source_key: str = "registry") -> None:
    errors.append({"code": code, "source_key": source_key, "message": message})


def validate_registry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen schema, scores, taxonomy, and hard gates."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if payload.get("model_cache_dataset_registry_version") != REGISTRY_VERSION:
        _error(errors, "invalid_registry_version", f"expected {REGISTRY_VERSION}")
    if payload.get("field_availability_enum") != sorted(FIELD_AVAILABILITY):
        _error(errors, "invalid_field_availability_contract", "field availability enum differs from frozen contract")
    scoring = payload.get("scoring_contract", {})
    if scoring.get("weights") != SCORE_WEIGHTS or sum(SCORE_WEIGHTS.values()) != 100:
        _error(errors, "invalid_scoring_contract", "score weights differ from frozen 100-point contract")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        _error(errors, "missing_sources", "sources must be a non-empty list")
        sources = []
    keys: set[str] = set()
    urls: set[str] = set()
    for item in sources:
        identity = item.get("identity", {})
        access = item.get("access", {})
        evidence = item.get("evidence", {})
        fields = item.get("fields", {})
        fitness = item.get("fitness", {})
        score = item.get("qualification_score", {})
        integration = item.get("integration", {})
        source_key = str(identity.get("source_key") or "missing")

        missing_identity = sorted(REQUIRED_IDENTITY - set(identity))
        missing_access = sorted(REQUIRED_ACCESS - set(access))
        missing_evidence = sorted(REQUIRED_EVIDENCE - set(evidence))
        if missing_identity:
            _error(errors, "missing_identity_fields", str(missing_identity), source_key)
        if missing_access:
            _error(errors, "missing_access_fields", str(missing_access), source_key)
        if missing_evidence:
            _error(errors, "missing_evidence_fields", str(missing_evidence), source_key)

        if source_key in keys:
            _error(errors, "duplicate_source_key", source_key, source_key)
        keys.add(source_key)
        canonical_url = str(identity.get("canonical_landing_page") or "")
        if canonical_url in urls:
            _error(errors, "duplicate_canonical_url", canonical_url, source_key)
        urls.add(canonical_url)
        for url_field in ("canonical_landing_page", "repository_or_api_url"):
            if not _is_http_url(identity.get(url_field)):
                _error(errors, "invalid_url", f"{url_field} is not HTTP(S)", source_key)
        for supporting_url in evidence.get("supporting_source_urls", []):
            if not _is_http_url(supporting_url):
                _error(errors, "invalid_url", "supporting source URL is not HTTP(S)", source_key)
        if access.get("license_url") is not None and not _is_http_url(access.get("license_url")):
            _error(errors, "invalid_url", "license_url is not HTTP(S) or null", source_key)
        if not _is_iso_date(identity.get("accessed_at")) or not _is_iso_date(evidence.get("last_verified_date")):
            _error(errors, "invalid_date", "accessed_at/last_verified_date must be ISO dates", source_key)
        if identity.get("primary_class") not in TAXONOMY:
            _error(errors, "unknown_taxonomy", str(identity.get("primary_class")), source_key)
        if set(fields) != FIELD_NAMES:
            _error(errors, "invalid_field_set", f"expected exactly {sorted(FIELD_NAMES)}", source_key)
        for field_name, availability in fields.items():
            if availability not in FIELD_AVAILABILITY:
                _error(errors, "invalid_field_availability", f"{field_name}={availability}", source_key)

        components = score.get("components", {})
        if set(components) != set(SCORE_WEIGHTS):
            _error(errors, "invalid_score_components", "score component keys differ from contract", source_key)
        else:
            for component, value in components.items():
                if not isinstance(value, int) or value < 0 or value > SCORE_WEIGHTS[component]:
                    _error(errors, "invalid_score_component", f"{component}={value}", source_key)
            if sum(components.values()) != score.get("total"):
                _error(errors, "score_total_mismatch", "component sum does not equal total", source_key)

        recommendation = integration.get("recommendation")
        if recommendation not in RECOMMENDATIONS:
            _error(errors, "invalid_recommendation", str(recommendation), source_key)
        primary_class = identity.get("primary_class")
        if primary_class == "joint_vec_ai_cache_trace":
            if not (_available(fields, "timestamp") and _available(fields, "model_id") and _available(fields, "mobility_handoff") and _available(fields, "location_rsu")):
                _error(errors, "joint_vec_hard_gate_failed", "A class requires temporal, model, mobility, and RSU evidence", source_key)
        if primary_class == "model_serving_request_trace":
            if not (_available(fields, "timestamp") and _available(fields, "model_id")):
                _error(errors, "request_trace_hard_gate_failed", "B class requires temporal and model identity", source_key)
        if primary_class == "adapter_or_lora_workload_trace":
            if not (_available(fields, "timestamp") and _available(fields, "adapter_lora_id")):
                _error(errors, "adapter_trace_hard_gate_failed", "C class requires temporal and adapter identity", source_key)
        if primary_class == "kv_prefix_cache_trace":
            if not (_available(fields, "timestamp") and _available(fields, "reuse_prefix_identity")):
                _error(errors, "kv_trace_hard_gate_failed", "D class requires temporal and prefix identity", source_key)
        if primary_class == "model_artifact_size_metadata" and fitness.get("request_trace_ready"):
            _error(errors, "size_metadata_marked_request_ready", "E class cannot be request-trace ready", source_key)
        if str(access.get("license", "")).lower() in {"", "unknown", "none"} and fitness.get("formal_ready"):
            _error(errors, "unknown_license_marked_formal_ready", "unknown license cannot be formal-ready", source_key)
        if recommendation == "rejected" and fitness.get("live_catalog_projection"):
            _error(errors, "rejected_source_live_projection", "rejected source cannot enter live catalog projection", source_key)
        if access.get("raw_download_performed") is not False:
            _error(errors, "raw_download_boundary_violation", "G11 requires raw_download_performed=false", source_key)
        if primary_class in {"joint_vec_ai_cache_trace", "model_serving_request_trace", "adapter_or_lora_workload_trace", "kv_prefix_cache_trace"} and not _available(fields, "timestamp"):
            _error(errors, "online_policy_without_time", "online cache trace requires a time/order field", source_key)
        if not _available(fields, "mobility_handoff") and fitness.get("vec_alignment") == "direct":
            _error(errors, "vec_alignment_overclaim", "direct VEC alignment requires mobility/handoff evidence", source_key)

    for number_path, value in _walk_numbers(payload):
        if not math.isfinite(value):
            _error(errors, "non_finite_number", number_path)

    return {
        "registry_version": payload.get("model_cache_dataset_registry_version"),
        "source_count": len(sources),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_compatibility(
    registry: dict[str, Any],
    *,
    dataset_sources: dict[str, Any],
    hf_manifest: dict[str, Any],
    sample_catalog: dict[str, Any],
) -> dict[str, Any]:
    """Validate registry references and the legacy AdapterCatalog projection."""

    errors: list[dict[str, str]] = []
    source_map = {item["identity"]["source_key"]: item for item in registry.get("sources", [])}
    for declaration in dataset_sources.get("datasets", []):
        reference = declaration.get("model_cache_registry_source_key")
        if reference and reference not in source_map:
            _error(errors, "dataset_sources_missing_registry_key", str(reference), str(declaration.get("dataset_key")))

    hf_by_id = {
        str(item.get("dataset_id")): item
        for item in hf_manifest.get("sources", [])
    }
    for dataset_id, item in hf_by_id.items():
        key = item.get("registry_source_key")
        registry_item = source_map.get(str(key))
        if registry_item is None:
            _error(errors, "hf_manifest_missing_registry_key", str(key), dataset_id)
            continue
        identity = registry_item["identity"]
        if identity.get("canonical_landing_page") != item.get("download_page_url"):
            _error(errors, "hf_manifest_url_mismatch", dataset_id, str(key))
        if identity.get("name") != dataset_id:
            _error(errors, "hf_manifest_identity_mismatch", dataset_id, str(key))
        if identity.get("primary_class") != item.get("primary_class"):
            _error(errors, "hf_manifest_taxonomy_mismatch", dataset_id, str(key))
        if registry_item["integration"].get("recommendation") != item.get("recommendation"):
            _error(errors, "hf_manifest_recommendation_mismatch", dataset_id, str(key))

    catalog_ids = {
        str(item.get("dataset_id"))
        for item in sample_catalog.get("model_cache_datasets", [])
    }
    expected_catalog_ids = {
        item["identity"]["name"]
        for item in source_map.values()
        if item["fitness"].get("live_catalog_projection")
    }
    if catalog_ids != expected_catalog_ids:
        _error(
            errors,
            "catalog_projection_mismatch",
            f"expected={sorted(expected_catalog_ids)} actual={sorted(catalog_ids)}",
        )
    rejected_ids = {
        item["identity"]["name"]
        for item in source_map.values()
        if item["integration"].get("recommendation") == "rejected"
    }
    if catalog_ids & rejected_ids:
        _error(errors, "rejected_source_live_projection", str(sorted(catalog_ids & rejected_ids)))

    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "dataset_source_reference_count": sum(
            1 for item in dataset_sources.get("datasets", []) if item.get("model_cache_registry_source_key")
        ),
        "hf_manifest_source_count": len(hf_by_id),
        "catalog_projection_count": len(catalog_ids),
    }


def validate_all(
    registry: dict[str, Any],
    *,
    dataset_sources: dict[str, Any],
    hf_manifest: dict[str, Any],
    sample_catalog: dict[str, Any],
) -> dict[str, Any]:
    registry_report = validate_registry_payload(registry)
    compatibility_report = validate_compatibility(
        registry,
        dataset_sources=dataset_sources,
        hf_manifest=hf_manifest,
        sample_catalog=sample_catalog,
    )
    return {
        "validation_contract_version": REGISTRY_VERSION,
        "validated_at": registry.get("audit_metadata", {}).get("generated_at"),
        "registry": registry_report,
        "compatibility": compatibility_report,
        "valid": registry_report["valid"] and compatibility_report["valid"],
        "raw_download_performed": False,
    }


def build_artifact_payloads(registry: dict[str, Any], validation_report: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic, machine-readable G11 artifact payloads."""

    sources = registry["sources"]
    verification_rows = []
    field_rows = []
    score_rows = []
    existing_hf = []
    for item in sources:
        identity = item["identity"]
        source_key = identity["source_key"]
        verification_rows.append(
            {
                "source_key": source_key,
                "name": identity["name"],
                "primary_class": identity["primary_class"],
                "canonical_url": identity["canonical_landing_page"],
                "verification_status": item["evidence"]["verification_status"],
                "http_status": item["online_verification"]["http_status"],
                "final_url": item["online_verification"]["final_url"],
                "redirected": item["online_verification"]["redirected"],
                "verified_at": item["online_verification"]["verified_at"],
                "raw_download_performed": item["access"]["raw_download_performed"],
            }
        )
        field_rows.append({"source_key": source_key, **item["fields"]})
        score_rows.append({"source_key": source_key, **item["qualification_score"]})
        if source_key.startswith("hf_"):
            existing_hf.append(item["existing_hf_reaudit"])
    recommended = [
        item for item in sources if item["integration"]["recommendation"] != "rejected"
    ]
    rejected = [
        item for item in sources if item["integration"]["recommendation"] == "rejected"
    ]
    return {
        "candidate_registry_snapshot.json": deepcopy(registry),
        "source_verification_rows.json": {"rows": verification_rows},
        "field_coverage_matrix.json": {"field_availability_enum": sorted(FIELD_AVAILABILITY), "rows": field_rows},
        "qualification_scores.json": {"scoring_contract": registry["scoring_contract"], "rows": score_rows},
        "recommended_sources.json": {"sources": recommended},
        "rejected_sources.json": {"sources": rejected},
        "existing_hf_reaudit.json": {"sources": existing_hf},
        "integration_mapping_plan.json": {"normalized_target_fields": registry["normalized_target_fields"], "mappings": registry["integration_mapping_plan"]},
        "validation_report.json": validation_report,
        "search_query_log.json": {"literature_cutoff": registry["audit_metadata"]["literature_cutoff"], "queries": registry["search_query_log"]},
    }


def write_artifacts(output_dir: str | Path, payloads: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_rows: list[dict[str, Any]] = []
    for filename in sorted(payloads):
        data = json.dumps(payloads[filename], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path = output_path / filename
        path.write_text(data, encoding="utf-8")
        file_rows.append(
            {
                "path": filename,
                "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
                "size_bytes": len(data.encode("utf-8")),
            }
        )
    manifest = {
        "artifact_name": "model_cache_dataset_discovery_20260819_g11_v1",
        "generated_at": metadata.get("generated_at"),
        "registry_version": REGISTRY_VERSION,
        "raw_download_performed": False,
        "files": file_rows,
    }
    manifest_data = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output_path / "artifact_integrity_manifest.json").write_text(manifest_data, encoding="utf-8")
    return manifest


def assert_valid(report: dict[str, Any]) -> None:
    if not report.get("valid"):
        raise RegistryValidationError(json.dumps(report, ensure_ascii=False, indent=2))


def deterministic_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)
