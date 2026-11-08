"""Where Branchpoint keeps its state: ``<hermes home>/plugin-data/branchpoint/`` (never the install dir)."""
from __future__ import annotations

from pathlib import Path

from .util import BranchpointError

PLUGIN_NAME = "branchpoint"
_override: Path | None = None


def set_data_dir_override(path: str | Path | None) -> None:
    """Tests and the standalone CLI point state somewhere explicit; the plugin path never calls this."""
    global _override
    _override = Path(path) if path else None


def hermes_home() -> Path | None:
    try:
        from hermes_constants import get_hermes_home  # public plugin API; resolved per call
        return Path(get_hermes_home())
    except Exception:
        return None


def data_dir() -> Path:
    if _override is not None:
        _override.mkdir(parents=True, exist_ok=True)
        return _override
    try:
        from plugins.plugin_storage import plugin_data_dir  # public plugin API
        return Path(plugin_data_dir(PLUGIN_NAME))
    except Exception:
        pass
    home = hermes_home()
    if home is None:
        raise BranchpointError("Cannot locate the Hermes plugin data directory (not running inside Hermes, and no explicit state dir was set).")
    d = home / "plugin-data" / PLUGIN_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def standalone_data_dir() -> Path:
    return Path.home() / ".local" / "state" / "hermes-branchpoint"
