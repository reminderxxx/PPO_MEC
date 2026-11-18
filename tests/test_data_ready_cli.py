"""Mainline data checks reject unusable files and expose reliable exit codes."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from scripts import check_data_ready


@pytest.fixture
def mainline_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(check_data_ready, "ROOT_DIR", tmp_path)
    ngsim = tmp_path / "data/raw/mobility/ngsim/trace.csv"
    alibaba = tmp_path / "data/raw/workflow/alibaba2018/batch_task.csv"
    for path in [ngsim, alibaba]:
        path.parent.mkdir(parents=True)
        path.write_text("sample\n")
    return ngsim, alibaba


def test_mainline_does_not_consult_optional_data(mainline_paths, monkeypatch, capsys):
    optional = Mock(side_effect=AssertionError("Optional sources must not block mainline"))
    for name in ["_check_lust", "_check_highd", "_check_huggingface_model_cache_sources", "_check_model_cache_dataset_registry"]:
        monkeypatch.setattr(check_data_ready, name, optional)
    assert check_data_ready.main(["--scope", "mainline", "--strict", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert [row["dataset_name"] for row in results] == ["NGSIM", "Alibaba batch task"]
    assert all(row["ready"] for row in results)
    optional.assert_not_called()


@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("unusable", ["missing", "empty", "directory"])
def test_strict_rejects_missing_empty_or_directory_csv(mainline_paths, capsys, dataset, unusable):
    path = mainline_paths[dataset]
    path.unlink()
    if unusable == "empty":
        path.touch()
    elif unusable == "directory":
        path.mkdir()
    assert check_data_ready.main(["--scope", "mainline", "--strict", "--json"]) == 1
    results = json.loads(capsys.readouterr().out)["results"]
    assert results[dataset]["ready"] is False
    assert results[dataset]["missing_items"]


def test_inspection_mode_keeps_zero_exit_for_missing_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_data_ready, "ROOT_DIR", tmp_path)
    assert check_data_ready.main(["--scope", "mainline"]) == 0
    assert "ready_count: 0/2" in capsys.readouterr().out


def test_readiness_checks_the_same_default_ngsim_file_as_the_loader(mainline_paths, capsys):
    ngsim, _ = mainline_paths
    (ngsim.parent / "aaa_empty.csv").touch()
    assert check_data_ready.main(["--scope", "mainline", "--strict", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["csv_files"][0] == "aaa_empty.csv"
    assert "aaa_empty.csv" in result["missing_items"][0]


def test_unreadable_file_is_not_ready(tmp_path, monkeypatch):
    path = tmp_path / "trace.csv"
    path.write_text("sample\n")
    monkeypatch.setattr(type(path), "open", Mock(side_effect=PermissionError("unreadable")))
    assert check_data_ready._is_readable_data_file(path) is False


@pytest.mark.parametrize("dataset", [0, 1])
def test_strict_rejects_git_lfs_pointer_files(mainline_paths, capsys, dataset):
    mainline_paths[dataset].write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'a' * 64}\nsize 12345678\n"
    )
    assert check_data_ready.main(["--scope", "mainline", "--strict", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)["results"][dataset]
    assert result["ready"] is False
    assert "Git LFS" in result["missing_items"][0]


def test_default_checks_all_six_sources(monkeypatch, capsys):
    names = ["_check_lust", "_check_ngsim", "_check_highd", "_check_alibaba_batch_task", "_check_huggingface_model_cache_sources", "_check_model_cache_dataset_registry"]
    for name in names:
        monkeypatch.setattr(check_data_ready, name, Mock(return_value={"dataset_name": name, "ready": False, "missing_items": ["missing"]}))
    assert check_data_ready.main([]) == 0
    assert "ready_count: 0/6" in capsys.readouterr().out
    for name in names:
        getattr(check_data_ready, name).assert_called_once_with()
