from __future__ import annotations

from agente_5g.settings import PROJECT_ROOT, Settings, _deep_merge, _read_yaml


def test_load_sample_mode():
    settings = Settings.load(mode="sample")
    assert settings.mode == "sample"
    assert settings.sample.max_packets_per_file == 20000


def test_load_full_mode():
    settings = Settings.load(mode="full")
    assert settings.mode == "full"
    assert settings.full.checkpointing is True


def test_load_defaults_to_base_yaml_mode_when_unspecified():
    settings = Settings.load()
    assert settings.mode in ("sample", "full")  # whatever base.yaml declares


def test_resolve_path_joins_relative_paths_against_project_root():
    settings = Settings.load(mode="sample")
    resolved = settings.resolve_path("data/raw")
    assert resolved == PROJECT_ROOT / "data" / "raw"


def test_resolve_path_returns_absolute_paths_unchanged(tmp_path):
    settings = Settings.load(mode="sample")
    resolved = settings.resolve_path(tmp_path)
    assert resolved == tmp_path


def test_read_yaml_returns_empty_dict_for_missing_file(tmp_path):
    assert _read_yaml(tmp_path / "missing.yaml") == {}


def test_read_yaml_parses_existing_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\nmode: sample\n", encoding="utf-8")
    assert _read_yaml(path) == {"seed": 7, "mode": "sample"}


def test_deep_merge_overrides_leaf_values():
    base = {"a": 1, "b": 2}
    override = {"b": 20}
    assert _deep_merge(base, override) == {"a": 1, "b": 20}


def test_deep_merge_recurses_into_nested_dicts():
    base = {"sample": {"max_packets_per_file": 20000, "max_files": None}}
    override = {"sample": {"max_packets_per_file": 500}}
    merged = _deep_merge(base, override)
    assert merged == {"sample": {"max_packets_per_file": 500, "max_files": None}}


def test_deep_merge_adds_new_keys_without_dropping_existing_ones():
    base = {"a": {"x": 1}}
    override = {"b": {"y": 2}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1}, "b": {"y": 2}}
