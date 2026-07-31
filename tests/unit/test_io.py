from __future__ import annotations

from agente_5g.settings import Settings
from agente_5g.utils.io import (
    config_hash,
    file_sha256,
    read_manifest,
    snapshot_config,
    write_manifest,
)


def test_config_hash_is_stable_for_identical_settings():
    s1 = Settings.load(mode="sample")
    s2 = Settings.load(mode="sample")
    assert config_hash(s1) == config_hash(s2)


def test_config_hash_changes_when_settings_differ():
    s1 = Settings.load(mode="sample")
    s2 = Settings.load(mode="sample")
    s2.seed = s1.seed + 1
    assert config_hash(s1) != config_hash(s2)


def test_snapshot_config_writes_yaml_under_run_id(tmp_path):
    settings = Settings.load(mode="sample")
    settings.paths.outputs = tmp_path

    snapshot_path = snapshot_config(settings, run_id="test-run-1")

    assert snapshot_path.exists()
    assert snapshot_path == tmp_path / "reports" / "test-run-1" / "config.yaml"
    assert "seed" in snapshot_path.read_text(encoding="utf-8")


def test_file_sha256_is_deterministic_and_content_sensitive(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello")
    f2.write_text("hello")
    f3 = tmp_path / "c.txt"
    f3.write_text("different content")

    assert file_sha256(f1) == file_sha256(f2)
    assert file_sha256(f1) != file_sha256(f3)


def test_read_manifest_returns_empty_dict_when_missing(tmp_path):
    assert read_manifest(tmp_path / "does_not_exist.json") == {}


def test_write_then_read_manifest_round_trips(tmp_path):
    manifest_path = tmp_path / "nested" / "manifest.json"
    manifest = {"ICMPflood_BS1.pcapng": {"status": "complete", "row_count": 100}}

    write_manifest(manifest_path, manifest)
    result = read_manifest(manifest_path)

    assert result == manifest
