"""Caller detection and allowlist policy tests — no network or config files."""

import pytest

from coworker.callers import identify_caller, is_caller_allowed

MARKERS = ("CLAUDECODE", "CURSOR_AGENT", "CURSOR_INVOKED_AS", "CODEX_HOME", "OPENCODE_CONFIG_DIR")


@pytest.fixture
def clean_caller_env(monkeypatch):
    for name in (*MARKERS, "COWORKER_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "marker, value, expected_host",
    [
        ("CLAUDECODE", "1", "claude"),
        ("CURSOR_AGENT", "1", "cursor"),
        ("CODEX_HOME", "/tmp/codex", "codex"),
        ("OPENCODE_CONFIG_DIR", "/tmp/opencode", "opencode"),
    ],
)
def test_each_marker_resolves_to_host(clean_caller_env, monkeypatch, marker, value, expected_host):
    # Disable ancestry resolution to test env marker fallback
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv(marker, value)
    host, model = identify_caller()
    assert host == expected_host
    assert model is None


def test_coworker_host_overrides_conflicting_marker(clean_caller_env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("COWORKER_HOST", "cursor:gpt-5.6-luna")
    assert identify_caller() == ("cursor", "gpt-5.6-luna")


@pytest.mark.parametrize(
    "stamp, expected",
    [
        ("claude", ("claude", None)),
        ("cursor:gpt-5.6-luna:extra", ("cursor", "gpt-5.6-luna:extra")),
    ],
)
def test_coworker_host_parses_host_and_model(clean_caller_env, monkeypatch, stamp, expected):
    monkeypatch.setenv("COWORKER_HOST", stamp)
    assert identify_caller() == expected


def test_cursor_wins_over_inherited_claude_marker(clean_caller_env, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert identify_caller() == ("cursor", None)


def test_no_markers_resolves_unknown_and_denies(clean_caller_env, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    policy = {"unknown": {"allow": False}}
    host, model = identify_caller()
    assert (host, model) == ("unknown", None)
    assert is_caller_allowed(host, model, policy) is False


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-5.6-luna-high", False),
        ("composer-2.5", False),
        ("claude-opus-5", True),
    ],
)
def test_cursor_model_deny_globs(model, expected):
    policy = {
        "cursor": {
            "allow": True,
            "deny_models": ["gpt-5.6-luna*", "composer-2.5*"],
        },
    }
    assert is_caller_allowed("cursor", model, policy) is expected


def test_unknown_model_fails_open_on_model_dimension(capsys):
    policy = {"cursor": {"allow": True, "deny_models": ["gpt-5.6-luna*"]}}
    assert is_caller_allowed("cursor", None, policy) is True
    assert capsys.readouterr().err == (
        "Model could not be verified. Set COWORKER_HOST=host:model to specify it.\n"
    )


@pytest.mark.parametrize("policy", [None, {}])
def test_missing_policy_allows_everything(policy):
    assert is_caller_allowed("unknown", None, policy) is True
    assert is_caller_allowed("opencode", "some-model", policy) is True


def test_claude_ancestor_resolves_despite_orca_env_vars(clean_caller_env, monkeypatch, tmp_path):
    """Regression: Claude ancestor resolves to claude despite CODEX_HOME and OPENCODE_CONFIG_DIR."""
    # Simulate Orca environment that sets all supervisor markers
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CODEX_HOME", "/home/che/.config/orca/codex-accounts/70dc589f/home")
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", "/home/che/.config/orca/opencode-hooks/shared")

    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is the real claude agent
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.local/bin/claude\x00--dangerously-skip-permissions\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "claude", "Ancestry should resolve to claude, not codex from env markers"
    assert model is None


def test_opencode_ancestor_resolves_despite_codex_home(clean_caller_env, monkeypatch, tmp_path):
    """OpenCode ancestor resolves to opencode despite CODEX_HOME being set."""
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex")
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.local/bin/opencode\x00--config\x00/tmp\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "opencode"

    # Verify policy denies opencode
    policy = {"opencode": {"allow": False}, "unknown": {"allow": False}}
    assert is_caller_allowed(host, model, policy) is False


def test_codex_ancestor_resolves_to_codex(clean_caller_env, monkeypatch, tmp_path):
    """Codex ancestor resolves to codex."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.local/bin/codex\x00--arg\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "codex"
    assert model is None


def test_bash_with_claude_codex_in_args_must_not_match(clean_caller_env, monkeypatch, tmp_path):
    """Bash ancestor with 'claude' and 'codex' in later args must NOT match."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    bash_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    # Current process
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{bash_pid}\n")

    # Bash with 'claude' and 'codex' in arguments (from real snapshot)
    # This should NOT match because argv[0] basename is 'bash', not 'claude' or 'codex'
    cmdline_file = proc_dir / str(bash_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(
        b"/bin/bash\x00-c\x00source /tmp/snapshot.sh && export CLAUDE_PLUGIN_DATA=/tmp/codex\x00"
    )

    # Parent of bash is init (no more ancestors)
    status_file = proc_dir / str(bash_pid) / "status"
    status_file.write_text("PPid:\t1\n")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    # Should fall back to env markers or unknown, not match the bash args
    assert host != "claude"
    assert host != "codex"


def test_ancestry_wins_over_conflicting_env_marker(clean_caller_env, monkeypatch, tmp_path):
    """Ancestry detection should win over conflicting environment marker."""
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", "/tmp/opencode")
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is claude, not opencode
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.local/bin/claude\x00--arg\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "claude", "Ancestry should win over OPENCODE_CONFIG_DIR env var"


def test_ancestry_finds_nothing_env_markers_still_work(clean_caller_env, monkeypatch, tmp_path):
    """When ancestry finds nothing, environment markers still work as fallback."""
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is unknown (not a recognized agent)
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"some-unknown-process\x00--arg\x00")

    # Parent has PPid 1 (end of chain)
    status_file = proc_dir / str(parent_pid) / "status"
    status_file.write_text("PPid:\t1\n")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "claude", "Should fallback to CLAUDECODE env marker when ancestry finds nothing"


def test_coworker_host_wins_over_ancestry(clean_caller_env, monkeypatch, tmp_path):
    """COWORKER_HOST env var still wins over ancestry."""
    monkeypatch.setenv("COWORKER_HOST", "cursor:gpt-5.6-luna")
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is claude (not cursor)
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.local/bin/claude\x00--arg\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert (host, model) == ("cursor", "gpt-5.6-luna"), "COWORKER_HOST should win over ancestry"


def test_cursor_resolves_model_at_runtime_from_proc(clean_caller_env, monkeypatch, tmp_path):
    """Test Cursor model resolution from /proc on Linux."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent\x00--model\x00gpt-5.6-luna-high\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "gpt-5.6-luna-high"


def test_cursor_resolves_model_with_model_equals_format(clean_caller_env, monkeypatch, tmp_path):
    """Test --model=value format extraction."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent\x00--model=gpt-5.6-luna-medium\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "gpt-5.6-luna-medium"


def test_cursor_fallback_to_config_file(clean_caller_env, monkeypatch, tmp_path):
    """Test fallback to ~/.cursor/cli-config.json when --model not in argv."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent\x00--help\x00")

    config_file = tmp_path / "cli-config.json"
    config_file.write_text('{"model": {"modelId": "gpt-5.6-luna-low"}}')

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        elif isinstance(path, str) and path.endswith("cli-config.json"):
            path = str(config_file)
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)
    monkeypatch.setattr("os.path.expanduser", lambda p: str(config_file) if "cli-config" in p else p)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "gpt-5.6-luna-low"


def test_cursor_model_resolver_disabled_on_non_linux(clean_caller_env, monkeypatch):
    """Test that model resolver does not run on non-Linux platforms."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")

    host, model = identify_caller()
    assert host == "cursor"
    assert model is None


def test_cursor_model_resolver_handles_missing_proc_gracefully(clean_caller_env, monkeypatch):
    """Test graceful handling when /proc is not accessible."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("os.getpid", lambda: 1000)

    def mock_open_fails(*args, **kwargs):
        raise OSError("No such file or directory")

    monkeypatch.setattr("builtins.open", mock_open_fails)

    host, model = identify_caller()
    assert host == "unknown"
    assert model is None


def test_cursor_ancestry_chain_longer_than_32_hops_terminates(clean_caller_env, monkeypatch, tmp_path):
    """Test that ancestry walk terminates after 32 hops."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    # Create a chain of 35 processes (exceeds 32 hop limit)
    for i in range(test_pid, test_pid + 35):
        status_file = proc_dir / str(i) / "status"
        status_file.parent.mkdir(parents=True)
        ppid = i + 1
        status_file.write_text(f"PPid:\t{ppid}\n")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "unknown"
    assert model is None


def test_cursor_ancestry_self_referential_ppid_terminates(clean_caller_env, monkeypatch, tmp_path):
    """Test that ancestry walk terminates on self-referential PPid."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    status_file = proc_dir / str(parent_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "unknown"
    assert model is None


def test_cursor_ancestor_two_hops_up_is_found(clean_caller_env, monkeypatch, tmp_path):
    """Test that a Cursor ancestor two or more hops up the chain is found."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    intermediate_pid = 999
    cursor_pid = 998
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{intermediate_pid}\n")

    status_file = proc_dir / str(intermediate_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{cursor_pid}\n")

    cmdline_file = proc_dir / str(intermediate_pid) / "cmdline"
    cmdline_file.write_bytes(b"some-process\x00--arg\x00")

    status_file = proc_dir / str(cursor_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("PPid:\t0\n")

    cmdline_file = proc_dir / str(cursor_pid) / "cmdline"
    cmdline_file.write_bytes(b"cursor-agent\x00--model\x00gpt-5.6-luna-high\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "gpt-5.6-luna-high"


def test_cursor_ancestor_with_agent_basename_recognized(clean_caller_env, monkeypatch, tmp_path):
    """Test that a process with /cursor-agent/versions/ path is recognized."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/cursor-agent/versions/1.0/agent\x00--model\x00composer-2.5\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "composer-2.5"


def test_opencode_with_exe_suffix_resolves(clean_caller_env, monkeypatch, tmp_path):
    """Test that argv[0] with .exe suffix (opencode.exe) resolves to opencode."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # OpenCode with .exe suffix (real case: symlink target)
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"/home/che/.npm-global/lib/node_modules/opencode-ai/bin/opencode.exe\x00--arg\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "opencode", "Should resolve opencode.exe to opencode host"
    assert model is None

    # Verify policy denies opencode
    policy = {"opencode": {"allow": False}, "unknown": {"allow": False}}
    assert is_caller_allowed(host, model, policy) is False


def test_cursor_agent_with_exe_suffix_resolves(clean_caller_env, monkeypatch, tmp_path):
    """Test that cursor-agent.exe resolves to cursor."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # cursor-agent.exe
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent.exe\x00--model\x00gpt-5.6-luna-high\x00")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    assert host == "cursor"
    assert model == "gpt-5.6-luna-high"


def test_near_miss_basenames_do_not_match(clean_caller_env, monkeypatch, tmp_path):
    """Test that near-miss names (opencoded, codexy) do NOT match any host."""
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    # Test both near-miss cases by creating a chain with fallback
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is 'opencoded' (not a real agent)
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"opencoded\x00--arg\x00")

    # Grandparent is 'codexy' (also not a real agent)
    grandparent_pid = 998
    status_file = proc_dir / str(parent_pid) / "status"
    status_file.write_text(f"PPid:\t{grandparent_pid}\n")

    cmdline_file = proc_dir / str(grandparent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"codexy\x00--arg\x00")

    # Great-grandparent is bash (also not matched)
    ggp_pid = 1
    status_file = proc_dir / str(grandparent_pid) / "status"
    status_file.write_text(f"PPid:\t{ggp_pid}\n")

    original_open = open

    def mock_open_proc(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if isinstance(path, str) and path.startswith("/proc/"):
            path = path.replace("/proc/", str(proc_dir) + "/")
        return original_open(path, *args[1:], **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_proc)

    host, model = identify_caller()
    # No ancestor matched, so should return unknown
    assert host == "unknown", "Near-miss names (opencoded, codexy) must not match any host"
    assert model is None
