"""Check the public example's configurable endpoints and output locations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.cua.harness import sweep


def test_server_urls_are_explicit_and_cli_overrides_environment(monkeypatch):
    args = SimpleNamespace(server=None, fanout=False)
    monkeypatch.delenv("BLINK_CUA_SERVERS", raising=False)
    with pytest.raises(ValueError, match="Supply --server"):
        sweep._urls(args, ["blink-mimo-9b"])
    monkeypatch.setenv("BLINK_CUA_SERVERS", "blink-mimo-9b=https://example.com:8443,"
                      "blink-4b=http://localhost:8001")
    assert sweep._urls(args, ["blink-mimo-9b", "blink-4b"]) == {
        "blink-mimo-9b": ["https://example.com:8443"], "blink-4b": ["http://localhost:8001"],
    }
    args.server = ["blink-mimo-9b=http://localhost:9000"]
    assert sweep._urls(args, ["blink-mimo-9b"]) == {"blink-mimo-9b": ["http://localhost:9000"]}
    with pytest.raises(ValueError, match="every requested model"):
        sweep._urls(args, ["blink-mimo-9b", "blink-27b"])


def test_mock_uses_environment_for_registry_apps_and_output(monkeypatch, tmp_path):
    fixtures = Path(__file__).with_name("fixtures")
    monkeypatch.setenv("BLINK_CUA_REGISTRY", str(fixtures / "stub/scenario.json"))
    monkeypatch.setenv("BLINK_CUA_APPS_ROOT", str(fixtures))
    monkeypatch.setenv("BLINK_CUA_RUNS_DIR", str(tmp_path))
    assert sweep.main(["--mock", "--run-id", "env-paths", "--no-video", "--beat", "0",
                       "--workers", "1", "--harness-sha", "test-sha"]) == 0
    output = tmp_path / "env-paths"
    assert (output / "report.md").is_file()
    assert json.loads((output / "config.json").read_text())["mock"] is True
