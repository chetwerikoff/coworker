"""Caller detection and policy decisions for the CLI allowlist gate."""

import fnmatch
import json
import os
import sys
from collections.abc import Mapping

_CALLER_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cursor", ("CURSOR_AGENT", "CURSOR_INVOKED_AS")),
    ("codex", ("CODEX_HOME",)),
    ("opencode", ("OPENCODE_CONFIG_DIR",)),
    ("claude", ("CLAUDECODE",)),
)


def resolve_host_and_model_from_ancestry() -> tuple[str | None, str | None]:
    """
    Resolve host and model by walking process ancestry on Linux.

    Walks the process tree up from the current PID, checking each ancestor's
    argv[0] basename and arguments against known agent patterns. Returns
    (host, model) where host is one of: cursor, claude, codex, opencode, or
    None if no match found. Model is extracted from Cursor argv --model flag
    or ~/.cursor/cli-config.json fallback.

    Returns (None, None) on non-Linux platforms or if no ancestor matches.
    """
    if not sys.platform.startswith("linux"):
        return None, None

    pid = os.getpid()
    hop_count = 0
    max_hops = 32

    while pid > 1 and hop_count < max_hops:
        hop_count += 1

        # Read parent PID from /proc/<pid>/status
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
                else:
                    return None, None
        except (OSError, ValueError):
            return None, None

        # Detect self-referential PPid
        if ppid == pid:
            break

        # Read command line from /proc/<ppid>/cmdline
        try:
            with open(f"/proc/{ppid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="ignore").split("\0")
        except (OSError, UnicodeDecodeError):
            pid = ppid
            continue

        if not cmdline or not cmdline[0]:
            pid = ppid
            continue

        # Get basename of argv[0] only, never scan substrings across all args
        argv0_basename = os.path.basename(cmdline[0])
        # Normalize by stripping .exe suffix (Windows launcher compatibility)
        if argv0_basename.endswith(".exe"):
            argv0_basename = argv0_basename[:-4]

        # Check against known agent patterns (match basename only, except for path check)
        host = None
        if argv0_basename == "cursor-agent" or any("/cursor-agent/versions/" in arg for arg in cmdline):
            host = "cursor"
        elif argv0_basename == "claude":
            host = "claude"
        elif argv0_basename == "codex":
            host = "codex"
        elif argv0_basename == "opencode":
            host = "opencode"

        if host:
            model = None
            # For Cursor, extract model from argv
            if host == "cursor":
                for i, arg in enumerate(cmdline):
                    if arg == "--model" and i + 1 < len(cmdline):
                        model = cmdline[i + 1]
                        break
                if not model:
                    for arg in cmdline:
                        if arg.startswith("--model="):
                            model = arg.split("=", 1)[1]
                            break
                # Fallback to config file if model still not found
                if not model:
                    config_path = os.path.expanduser("~/.cursor/cli-config.json")
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                            model = config.get("model", {}).get("modelId")
                    except (OSError, json.JSONDecodeError, KeyError, TypeError):
                        pass

            return host, model

        pid = ppid

    return None, None


def identify_caller(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str | None]:
    """Return the caller host and optional explicitly stamped model.

    Detection precedence:
    1. COWORKER_HOST environment variable (authoritative override)
    2. Process ancestry via /proc (primary on Linux)
    3. Environment markers (fallback for non-Linux or when ancestry finds nothing)
    4. unknown
    """
    env = os.environ if environ is None else environ
    stamped = env.get("COWORKER_HOST")
    if stamped is not None:
        host, _, model = stamped.partition(":")
        return host or "unknown", model or None

    # Try ancestry-based detection first (primary method on Linux)
    host, model = resolve_host_and_model_from_ancestry()
    if host:
        return host, model

    # Fallback to environment markers for non-Linux or when ancestry finds nothing
    for host, markers in _CALLER_MARKERS:
        if any(marker in env for marker in markers):
            return host, None

    return "unknown", None


def is_caller_allowed(
    host: str,
    model: str | None,
    policy: dict | None,
) -> bool:
    """Return whether a caller identity is allowed by the supplied policy."""
    if not policy:
        return True

    rule = policy.get(host) or policy.get("unknown") or {"allow": False}
    if not rule.get("allow", False):
        return False

    deny_models = rule.get("deny_models") or []
    if model is None and deny_models:
        print(
            "Model could not be verified. Set COWORKER_HOST=host:model to specify it.",
            file=sys.stderr,
        )
        return True
    return not any(fnmatch.fnmatchcase(model, pattern) for pattern in deny_models)


def caller_label(host: str, model: str | None) -> str:
    """Render a caller identity for diagnostics."""
    return f"{host}:{model}" if model else host
