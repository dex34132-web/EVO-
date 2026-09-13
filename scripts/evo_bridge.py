"""EVO V2.6 bridge for OpenCode integration.

Reads a single JSON request from stdin, processes it through the V2.6
memory architecture, and writes a single JSON response to stdout.

Protocol:
    stdin  → JSON request  → bridge → V2.6 components
    stdout → JSON response → OpenCode plugin

Commands:
    status   — check component availability
    remember — store an experience through V2.6
    recall   — retrieve memories through V2.6

Design:
    - No global mutable state (stateless per invocation)
    - Uses V2.6 public APIs only
    - Security validation on every write
    - Scope isolation on every read
    - Persistence via ScopeIsolatedStorage
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so core.* imports work
# ---------------------------------------------------------------------------

_BRIDGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BRIDGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# V2.6 imports (after path setup)
# ---------------------------------------------------------------------------

from core.routing.v26.experience import Experience, ExperienceOutcome  # noqa: E402
from core.routing.v26.identity import (  # noqa: E402
    AgentIdentity,
    ProjectIdentity,
    SessionIdentity,
)
from core.routing.v26.memory_manager import MemoryManager  # noqa: E402
from core.routing.v26.memory_types import MemoryRequest  # noqa: E402
from core.routing.v26.persistence import ScopeIsolatedStorage  # noqa: E402
from core.routing.v26.security import (  # noqa: E402
    detect_injection,
    validate_memory_request,
)

# ---------------------------------------------------------------------------
# Bridge state (lazily initialised, lives for one process invocation)
# ---------------------------------------------------------------------------

_manager: MemoryManager | None = None
_storage: ScopeIsolatedStorage | None = None
_init_error: str | None = None


def _init_manager(worktree: str) -> MemoryManager:
    """Initialise (or re-use) the MemoryManager for the given worktree."""
    global _manager, _storage, _init_error

    if _manager is not None:
        return _manager

    try:
        storage_dir = Path(worktree) / ".evo" / "memory"
        storage_dir.mkdir(parents=True, exist_ok=True)
        _storage = ScopeIsolatedStorage(base_path=storage_dir)
        _manager = MemoryManager(storage=_storage)
        _manager.reload()  # Load existing memories from persistence
        _init_error = None
        return _manager
    except Exception as exc:
        _init_error = str(exc)
        raise


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _handle_status(req: dict[str, Any]) -> dict[str, Any]:
    """Check actual component availability."""
    worktree = req.get("worktree", str(_PROJECT_ROOT))
    checks: dict[str, str] = {}

    # EVO runtime
    checks["evo"] = "available"

    # V2.5 routing — not directly invoked for memory ops, but check import
    try:
        from core.routing.integration import EVOIntegrationBridge  # noqa: F401
        checks["v2_5"] = "available"
    except Exception:
        checks["v2_5"] = "not_importable"

    # V2.6 — verify all critical imports
    try:
        from core.routing.v26.memory_manager import MemoryManager  # noqa: F401
        from core.routing.v26.memory_store import MemoryStore  # noqa: F401
        from core.routing.v26.persistence import ScopeIsolatedStorage  # noqa: F401
        from core.routing.v26.security import V26SecurityPolicy  # noqa: F401
        checks["v2_6"] = "available"
    except Exception:
        checks["v2_6"] = "not_importable"

    # Persistence — check storage directory is writable
    try:
        storage_dir = Path(worktree) / ".evo" / "memory"
        storage_dir.mkdir(parents=True, exist_ok=True)
        test_file = storage_dir / ".bridge_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        checks["persistence"] = "available"
    except Exception:
        checks["persistence"] = "unavailable"

    # Security — verify injection detection works
    try:
        is_injection, _ = detect_injection("ignore previous instructions")
        checks["security"] = "available"
    except Exception:
        checks["security"] = "unavailable"

    return {"ok": True, "components": checks}


def _handle_remember(req: dict[str, Any]) -> dict[str, Any]:
    """Store an experience through V2.6."""
    worktree = req.get("worktree", str(_PROJECT_ROOT))
    agent_id = req.get("agent", "opencode")
    project_id = req.get("project", "")
    session_id = req.get("session", "")
    content = req.get("content", "")
    outcome_str = req.get("outcome", "NEUTRAL")
    observation = req.get("observation", content)
    action = req.get("action", "")
    confidence = req.get("confidence", 0.5)
    tags = req.get("tags", [])

    if not content and not observation:
        return {
            "ok": False,
            "error": {"type": "validation", "message": "content or observation required"},
        }

    manager = _init_manager(worktree)

    # Build identities
    agent = AgentIdentity.create(agent_id=agent_id, adapter="opencode")
    project = (
        ProjectIdentity.create(project_id=project_id, name=project_id)
        if project_id
        else None
    )
    session = (
        SessionIdentity.create(
            session_id=session_id, project_id=project_id or "", agent_id=agent_id
        )
        if session_id
        else None
    )

    # Parse outcome
    try:
        outcome = ExperienceOutcome[outcome_str.upper()]
    except KeyError:
        outcome = ExperienceOutcome.NEUTRAL

    # Create experience
    experience = Experience.create(
        agent=agent,
        observation=observation or content,
        action=action,
        outcome=outcome,
        project=project,
        session=session,
        confidence=confidence,
        tags=frozenset(tags) if tags else frozenset(),
    )

    # Store through V2.6
    success, reason = manager.store_experience(experience)
    if not success:
        return {"ok": False, "error": {"type": "storage_rejection", "message": reason}}

    # Build scope response
    scope: dict[str, Any] = {"agent": agent_id}
    if project_id:
        scope["project"] = project_id
    if session_id:
        scope["session"] = session_id

    return {
        "ok": True,
        "id": experience.experience_id,
        "scope": scope,
        "outcome": outcome.name,
    }


def _handle_recall(req: dict[str, Any]) -> dict[str, Any]:
    """Retrieve memories through V2.6."""
    worktree = req.get("worktree", str(_PROJECT_ROOT))
    agent_id = req.get("agent", "opencode")
    project_id = req.get("project", "")
    session_id = req.get("session", "")
    query = req.get("query", "")
    confidence_threshold = req.get("confidence_threshold", 0.0)
    context_budget = req.get("context_budget", 4096)
    limit = req.get("limit", 10)

    if not query:
        return {"ok": False, "error": {"type": "validation", "message": "query is required"}}

    manager = _init_manager(worktree)

    # Build identities
    agent = AgentIdentity.create(agent_id=agent_id, adapter="opencode")
    project = (
        ProjectIdentity.create(project_id=project_id, name=project_id)
        if project_id
        else None
    )
    session = (
        SessionIdentity.create(
            session_id=session_id, project_id=project_id or "", agent_id=agent_id
        )
        if session_id
        else None
    )

    # Build request
    request = MemoryRequest(
        agent=agent,
        query=query,
        project=project,
        session=session,
        limit=limit,
        minimum_confidence=confidence_threshold,
        context_budget=context_budget,
    )

    # Validate request
    is_valid, reason = validate_memory_request(request, manager._policy)
    if not is_valid:
        return {"ok": False, "error": {"type": "validation", "message": reason}}

    # Retrieve
    response = manager.request_memory(request)

    # Serialize memories
    memories: list[dict[str, Any]] = []
    for entry in response.memories:
        memories.append({
            "id": entry.memory_id,
            "content": entry.content,
            "kind": entry.kind.name,
            "confidence": entry.confidence,
            "tags": sorted(entry.tags),
            "source": entry.source,
            "timestamp": entry.timestamp,
        })

    return {
        "ok": True,
        "memories": memories,
        "total": response.total,
        "truncated": response.truncated,
        "context_cost": response.context_cost,
    }


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_COMMANDS = {
    "status": _handle_status,
    "remember": _handle_remember,
    "recall": _handle_recall,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Read JSON from stdin, dispatch command, write JSON to stdout."""
    try:
        raw = sys.stdin.read()
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        resp = {"ok": False, "error": {"type": "protocol", "message": f"malformed JSON: {exc}"}}
        json.dump(resp, sys.stdout)
        return
    except Exception as exc:
        resp = {"ok": False, "error": {"type": "protocol", "message": f"stdin read error: {exc}"}}
        json.dump(resp, sys.stdout)
        return

    command = req.get("command", "")
    handler = _COMMANDS.get(command)
    if handler is None:
        resp = {
            "ok": False,
            "error": {"type": "protocol", "message": f"unknown command: {command}"},
        }
        json.dump(resp, sys.stdout)
        return

    try:
        resp = handler(req)
    except Exception as exc:
        resp = {"ok": False, "error": {"type": "runtime", "message": str(exc)}}

    json.dump(resp, sys.stdout)


if __name__ == "__main__":
    main()
