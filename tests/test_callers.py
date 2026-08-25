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
    # Disable runtime model resolution to test sniffing only
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv(marker, value)
    host, model = identify_caller()
    assert host == expected_host
    if expected_host == "cursor":
        # Cursor on non-linux returns None for model
        assert model is None
    else:
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


def test_no_markers_resolves_unknown_and_denies(clean_caller_env):
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


def test_cursor_resolves_model_at_runtime_from_proc(clean_caller_env, monkeypatch, tmp_path):
    """Test Cursor model resolution from /proc on Linux."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    
    # Mock sys.platform to return linux
    monkeypatch.setattr("sys.platform", "linux")
    
    # Mock os.getpid() to return a test PID
    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)
    
    # Create mock /proc files
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    # Mock status file for current process
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"Name:\tpython\nPPid:\t{parent_pid}\n")
    
    # Mock cmdline file for parent process (cursor-agent with --model)
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent\x00--model\x00gpt-5.6-luna-high\x00")
    
    # Mock open to use our proc_dir
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
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
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
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    
    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)
    
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")
    
    # cursor-agent without --model flag
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"cursor-agent\x00--help\x00")
    
    # Create mock config file
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
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("os.getpid", lambda: 1000)
    
    def mock_open_fails(*args, **kwargs):
        raise OSError("No such file or directory")
    
    monkeypatch.setattr("builtins.open", mock_open_fails)
    
    host, model = identify_caller()
    assert host == "cursor"
    assert model is None


def test_coworker_host_takes_precedence_over_runtime_resolution(clean_caller_env, monkeypatch):
    """Test that explicit COWORKER_HOST takes precedence over runtime resolution."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setenv("COWORKER_HOST", "cursor:composer-2.5")
    monkeypatch.setattr("sys.platform", "linux")
    
    host, model = identify_caller()
    assert host == "cursor"
    assert model == "composer-2.5"


def test_cursor_ancestry_chain_longer_than_32_hops_terminates(clean_caller_env, monkeypatch, tmp_path):
    """Test that ancestry walk terminates after 32 hops."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
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
    assert host == "cursor"
    assert model is None


def test_cursor_ancestry_self_referential_ppid_terminates(clean_caller_env, monkeypatch, tmp_path):
    """Test that ancestry walk terminates on self-referential PPid."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    # Current process points to parent
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent process has self-referential PPid
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
    assert host == "cursor"
    assert model is None


def test_cursor_ancestor_two_hops_up_is_found(clean_caller_env, monkeypatch, tmp_path):
    """Test that a Cursor ancestor two or more hops up the chain is found."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    intermediate_pid = 999
    cursor_pid = 998
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    # Current process -> intermediate -> cursor-agent
    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{intermediate_pid}\n")

    status_file = proc_dir / str(intermediate_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{cursor_pid}\n")

    cmdline_file = proc_dir / str(intermediate_pid) / "cmdline"
    cmdline_file.write_bytes(b"some-process\x00--arg\x00")

    # cursor-agent is 2 hops up
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
    """Test that a process with argv[0] basename 'agent' is recognized."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Process with /cursor-agent/versions/agent in path (matches "cursor-agent")
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


def test_no_cursor_ancestor_anywhere_resolves_none(clean_caller_env, monkeypatch, tmp_path):
    """Test that when no Cursor ancestor is found, model resolves to None (not config file)."""
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.delenv("COWORKER_HOST", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    test_pid = 1000
    parent_pid = 999
    monkeypatch.setattr("os.getpid", lambda: test_pid)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    status_file = proc_dir / str(test_pid) / "status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text(f"PPid:\t{parent_pid}\n")

    # Parent is not cursor-agent
    cmdline_file = proc_dir / str(parent_pid) / "cmdline"
    cmdline_file.parent.mkdir(parents=True)
    cmdline_file.write_bytes(b"bash\x00--norc\x00")

    # Stop walking (parent has PPid 1)
    status_file = proc_dir / str(parent_pid) / "status"
    status_file.write_text("PPid:\t1\n")

    # Create a config file that would be read if the fallback wrongly triggered
    config_file = tmp_path / "cli-config.json"
    config_file.write_text('{"model": {"modelId": "gpt-5.6-luna-high"}}')

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
    # Model must be None, NOT the value from config_file
    assert model is None
