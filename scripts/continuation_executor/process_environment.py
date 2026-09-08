"""Verify the actual parent interpreter and loaded critical dependencies."""
import importlib
import importlib.metadata
from pathlib import Path
import site
import sys

from .identity import ContinuationError, file_hash, read_json


def verify_process_environment(root):
    manifest=read_json(root/"configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/execution_environment_manifest.json")
    location=manifest["runtime_location"]
    if sys.executable!=location["resolved_python_absolute_path"]:
        raise ContinuationError("actual interpreter differs from frozen environment")
    if sys.prefix!=location["virtual_environment_root"] or site.ENABLE_USER_SITE is not False:
        raise ContinuationError("actual virtual environment/user-site drift")
    editable=[]
    for distribution in importlib.metadata.distributions():
        direct=distribution.read_text("direct_url.json")
        if direct:
            from .identity import strict_json
            if strict_json(direct).get("dir_info",{}).get("editable"):
                editable.append(distribution.metadata["Name"])
    if editable:raise ContinuationError("editable dependency installation forbidden")
    roots=[Path(path).resolve() for path in location["site_packages_paths"]]
    modules={"PyYAML":"yaml","numpy":"numpy","pandas":"pandas","pytest":"pytest","torch":"torch"}
    observed={}
    for distribution,expected in manifest["scientific_identity"]["critical_package_versions"].items():
        module=importlib.import_module(modules[distribution])
        path=Path(module.__file__).resolve()
        if not any(root in path.parents for root in roots):
            raise ContinuationError("critical dependency shadow import: "+distribution)
        if importlib.metadata.version(distribution)!=expected or str(module.__version__)!=expected:
            raise ContinuationError("actual critical dependency version drift: "+distribution)
        observed[distribution]={"version":expected,"path":str(path),"sha256":file_hash(path)}
    return {"python":sys.executable,"prefix":sys.prefix,"user_site_enabled":site.ENABLE_USER_SITE,
            "editable_installations":editable,"critical_modules":observed,"sys_path":list(sys.path)}
