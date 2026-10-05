"""Unit tests for the isolated OpenCode CLI transport."""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from coworker import cli
from coworker.opencode_cli import build_opencode_prompt, complete_via_opencode_cli


def test_build_prompt_preserves_roles_and_demotes_corpus():
    prompt = build_opencode_prompt(
        [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "<corpus>DATA</corpus>"},
        ],
        321,
    )

    assert "<delegated-system>\nSYSTEM\n</delegated-system>" in prompt
    assert "<message role='user'>\n<corpus>DATA</corpus>\n</message>" in prompt
    assert "approximately 321 tokens" in prompt
    assert "untrusted data" in prompt


def test_completion_invokes_isolated_cli_and_parses_usage(monkeypatch):
    events = [
        {"type": "text", "part": {"text": "bridge-ok"}},
        {
            "type": "step_finish",
            "part": {
                "reason": "stop",
                "tokens": {
                    "input": 10,
                    "output": 3,
                    "cache": {"read": 7, "write": 0},
                },
            },
        },
    ]
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr="",
        )

    monkeypatch.setattr("coworker.opencode_cli.shutil.which", lambda _: "/bin/opencode")
    monkeypatch.setattr("coworker.opencode_cli.subprocess.run", fake_run)

    resp = complete_via_opencode_cli(
        {"transport": "opencode_cli", "cli_timeout_seconds": 42},
        "deepseek-v4-flash-free",
        [{"role": "user", "content": "hello"}],
        100,
    )

    assert captured["cmd"] == [
        "/bin/opencode",
        "run",
        "--pure",
        "--format",
        "json",
        "--title",
        "coworker",
        "--model",
        "opencode/deepseek-v4-flash-free",
    ]
    assert captured["timeout"] == 42
    assert captured["check"] is False
    assert captured["input"].endswith("<message role='user'>\nhello\n</message>")
    permissions = json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])["permission"]
    assert permissions["*"] == "deny"
    assert permissions["bash"] == "deny"
    assert permissions["write"] == "deny"
    assert resp.choices[0].message.content == "bridge-ok"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 3
    assert resp.usage.prompt_tokens_details.cached_tokens == 7


def test_completion_reports_missing_executable(monkeypatch):
    monkeypatch.setattr("coworker.opencode_cli.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="executable not found"):
        complete_via_opencode_cli({}, "model", [], 10)


def test_completion_reports_cli_failure(monkeypatch):
    monkeypatch.setattr("coworker.opencode_cli.shutil.which", lambda _: "/bin/opencode")
    monkeypatch.setattr(
        "coworker.opencode_cli.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 9, stdout="", stderr="bad"),
    )

    with pytest.raises(RuntimeError, match="exited with 9: bad"):
        complete_via_opencode_cli({}, "model", [], 10)


def test_completion_reports_timeout(monkeypatch):
    monkeypatch.setattr("coworker.opencode_cli.shutil.which", lambda _: "/bin/opencode")

    def time_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr("coworker.opencode_cli.subprocess.run", time_out)

    with pytest.raises(RuntimeError, match="timed out after 17s"):
        complete_via_opencode_cli({"cli_timeout_seconds": 17}, "model", [], 10)


def test_cli_dispatches_opencode_transport_without_api_client(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        cli,
        "complete_via_opencode_cli",
        lambda cfg, model, messages, max_tokens: expected,
    )
    monkeypatch.setattr(
        cli,
        "make_client",
        lambda _cfg: pytest.fail("API client must not be created for CLI transport"),
    )

    result = cli._create_completion(
        {"transport": "opencode_cli"},
        "model",
        [{"role": "user", "content": "hello"}],
        100,
    )

    assert result is expected


def test_cli_keeps_api_transport_behavior(monkeypatch):
    expected = object()

    def create(**kwargs):
        return expected

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    monkeypatch.setattr(cli, "make_client", lambda _cfg: client)

    result = cli._create_completion({}, "model", [], 100)

    assert result is expected


def test_completion_forwards_configured_variant(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        event = {"type": "text", "part": {"text": "ok"}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(event), stderr="")

    monkeypatch.setattr("coworker.opencode_cli.shutil.which", lambda _: "/bin/opencode")
    monkeypatch.setattr("coworker.opencode_cli.subprocess.run", fake_run)

    complete_via_opencode_cli(
        {"cli_model": "openai/gpt-6-luna", "cli_variant": "low"},
        "model",
        [{"role": "user", "content": "hello"}],
        100,
    )

    assert captured["cmd"][-4:] == ["--model", "openai/gpt-6-luna", "--variant", "low"]
