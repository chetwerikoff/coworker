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


def resolve_cursor_model_at_runtime() -> str | None:
    """
    Detect the Cursor model at runtime by walking process ancestry on Linux.

    Returns the model id if found via --model flag in cursor-agent process,
    falls back to ~/.cursor/cli-config.json value if available, else None.
    Only runs on Linux; returns None on other platforms or if detection fails.
    """
    if not sys.platform.startswith("linux"):
        return None

    pid = os.getpid()
    cursor_ancestor_found = False
    hop_count = 0
    max_hops = 32

    # Walk up process tree looking for cursor-agent
    while pid > 1 and hop_count < max_hops:
        hop_count += 1

        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
                else:
                    return None
        except (OSError, ValueError):
            return None

        # Detect self-referential PPid
        if ppid == pid:
            break

        # Check if this process is cursor-agent
        try:
            with open(f"/proc/{ppid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="ignore").split("\0")
        except (OSError, UnicodeDecodeError):
            pid = ppid
            continue

        # Look for cursor-agent in the command
        if any("cursor-agent" in arg for arg in cmdline):
            cursor_ancestor_found = True
            # Extract --model flag
            for i, arg in enumerate(cmdline):
                if arg == "--model" and i + 1 < len(cmdline):
                    return cmdline[i + 1]
            # Try --model=value format
            for arg in cmdline:
                if arg.startswith("--model="):
                    return arg.split("=", 1)[1]
            # Found cursor-agent but no --model; try config file
            break

        pid = ppid

    # Only read config if we found a cursor-agent ancestor with no --model flag
    if cursor_ancestor_found:
        config_path = os.path.expanduser("~/.cursor/cli-config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("model", {}).get("modelId")
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    return None


def identify_caller(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str | None]:
    """Return the caller host and optional explicitly stamped model."""
    env = os.environ if environ is None else environ
    stamped = env.get("COWORKER_HOST")
    if stamped is not None:
        host, _, model = stamped.partition(":")
        return host or "unknown", model or None

    for host, markers in _CALLER_MARKERS:
        if any(marker in env for marker in markers):
            # Try to resolve Cursor model at runtime
            model = None
            if host == "cursor":
                model = resolve_cursor_model_at_runtime()
            return host, model
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
