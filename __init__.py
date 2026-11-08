"""Branchpoint: undo-and-explore for Hermes sessions.

/branchpoint fork "<goal>"   explore N distinct directions before building
/branchpoint rewind <N>      undo what you built since prompt N and try other ways
Routes run in isolated git worktrees; your working tree is untouched until you adopt a route.
"""
from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def register(ctx) -> None:
    # Lazy, relative imports: the directory-plugin loader mounts this file as a package and the host's
    # capability probe runs register() in a recording context, so nothing heavy happens at import time.
    from . import commands as _commands
    from .hooks import Hooks

    def _get(key, default):
        try:
            value = ctx.get_config(key, default)
        except Exception:
            return default
        return default if value is None else value

    service = _commands.BranchpointService(ctx, get_setting=_get)
    hooks = Hooks(ctx, _get)

    def handler(raw_args: str = "", *, session_id: str = "", **kwargs) -> str:
        return service.handle(raw_args, session_id=session_id, **kwargs)

    ctx.register_command(
        "branchpoint", handler=handler,
        description="Rewind and explore alternative builds in isolated worktrees; a blind council picks one.",
        args_hint='fork "<goal>" | rewind <N> | status | view | council | adopt <route> [merge|replay|adr]',
    )

    ctx.register_hook("pre_llm_call", hooks.pre_llm_call)
    ctx.register_hook("pre_tool_call", hooks.pre_tool_call)
    ctx.register_hook("post_tool_call", hooks.post_tool_call)
    ctx.register_hook("transform_tool_result", hooks.transform_tool_result)
    ctx.register_hook("post_api_request", hooks.post_api_request)

    schemas = _commands.tool_schemas()
    for name, fn in (("branchpoint_status", service.tool_status), ("branchpoint_decision", service.tool_decision),
                     ("branchpoint_fork", service.tool_fork)):
        schema = schemas[name]
        ctx.register_tool(name=name, toolset="branchpoint", schema=schema, handler=fn, description=schema["description"], emoji="🔱")

    skill = Path(__file__).parent / "skills" / "branchpoint" / "SKILL.md"
    if skill.exists() and hasattr(ctx, "register_skill"):
        ctx.register_skill("branchpoint", skill, description="When and how to use Branchpoint's fork/rewind exploration.")
