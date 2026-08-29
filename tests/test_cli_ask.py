"""Focused ask diagnostics and logging regressions."""

import json
from types import SimpleNamespace

from coworker import cli, logger


def _response(
    *,
    content,
    finish_reason,
    reasoning_content=None,
    completion_tokens=0,
    reasoning_tokens=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens,
            ),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


def _args(*, max_tokens):
    return SimpleNamespace(
        profile="code",
        provider=None,
        model=None,
        paths=[],
        question="summarize",
        max_tokens=max_tokens,
        task_id=None,
        no_log=False,
        allow_code=False,
    )


def _stub_ask(monkeypatch, response, profile=None):
    captured = {"logs": [], "max_tokens": []}
    monkeypatch.setattr(cli, "_enforce_caller_gate", lambda: True)
    monkeypatch.setattr(cli, "_apply_gate", lambda paths, allow_code: ([], []))
    monkeypatch.setattr(cli, "_build_corpus", lambda paths: "corpus")
    monkeypatch.setattr(cli, "load_providers", lambda: {"fake": {"default_model": "model"}})
    monkeypatch.setattr(
        cli,
        "load_profile",
        lambda name: profile
        or {
            "system_prompt": "system",
            "recommended_provider": "fake",
            "default_max_tokens_ask": 16384,
        },
    )

    def fake_completion(config, model, messages, max_tokens):
        captured["max_tokens"].append(max_tokens)
        return response

    monkeypatch.setattr(cli, "_create_completion", fake_completion)
    monkeypatch.setattr(
        cli,
        "log_call",
        lambda *args, **kwargs: captured["logs"].append((args, kwargs)),
    )
    return captured


def test_empty_reasoning_response_reports_diagnostics_and_exit(monkeypatch, capsys):
    raw_reasoning = "SECRET_REASONING"
    response = _response(
        content=None,
        finish_reason="length",
        reasoning_content=raw_reasoning,
        completion_tokens=16384,
        reasoning_tokens=16000,
    )
    captured = _stub_ask(monkeypatch, response)

    result = cli.cmd_ask(_args(max_tokens=1024))

    assert result == 3
    stderr = capsys.readouterr().err
    assert "finish_reason=length" in stderr
    assert "completion_tokens=16384" in stderr
    assert "reasoning_tokens=16000" in stderr
    assert "reasoning_content_present=True" in stderr
    assert f"reasoning_content_length={len(raw_reasoning)}" in stderr
    assert "Remove --max-tokens" in stderr
    assert raw_reasoning not in stderr
    assert captured["logs"][0][1]["exit_code"] == 3


def test_length_without_override_reports_profile_exhaustion(monkeypatch, capsys):
    response = _response(content="", finish_reason="length", completion_tokens=16384)
    _stub_ask(monkeypatch, response)

    result = cli.cmd_ask(_args(max_tokens=None))

    assert result == 3
    stderr = capsys.readouterr().err
    assert "Profile limit (16384 tokens) exhausted." in stderr
    assert "Remove --max-tokens" not in stderr


def test_length_with_large_override_reports_explicit_exhaustion(monkeypatch, capsys):
    response = _response(content=None, finish_reason="length", completion_tokens=32000)
    _stub_ask(monkeypatch, response)

    result = cli.cmd_ask(_args(max_tokens=32000))

    assert result == 3
    stderr = capsys.readouterr().err
    assert "Explicit limit (32000 tokens) exhausted." in stderr
    assert "Remove --max-tokens" not in stderr


def test_logger_records_safe_reasoning_diagnostics_and_exit(tmp_path):
    raw_reasoning = "SECRET_REASONING"
    response = _response(
        content=None,
        finish_reason="length",
        reasoning_content=raw_reasoning,
        completion_tokens=16384,
        reasoning_tokens=16000,
    )

    logger.log_call(
        response,
        "fake",
        {"pricing": None},
        "model",
        "code",
        "ask",
        [],
        "",
        1.0,
        None,
        log_dir=tmp_path,
        exit_code=3,
    )

    log_file = next(tmp_path.glob("*.jsonl"))
    serialized = log_file.read_text()
    record = json.loads(serialized)
    assert record["coworker.exit_code"] == 3
    assert record["gen_ai.response.finish_reason"] == "length"
    assert record["gen_ai.usage.reasoning_tokens"] == 16000
    assert "reasoning_content" not in serialized
    assert raw_reasoning not in serialized


def test_nonempty_ask_keeps_success_behavior(monkeypatch, capsys):
    response = _response(content="answer", finish_reason="stop", completion_tokens=3)
    captured = _stub_ask(monkeypatch, response)

    result = cli.cmd_ask(_args(max_tokens=None))

    output = capsys.readouterr()
    assert result == 0
    assert output.out == "answer\n"
    assert "empty response" not in output.err
    assert captured["max_tokens"] == [16384]
