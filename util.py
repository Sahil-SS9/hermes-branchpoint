"""Small, host-free helpers shared by every Branchpoint module (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class BranchpointError(Exception):
    """A user-facing failure: the message is shown as-is."""


# --------------------------------------------------------------------------- paths / secrets

# Files that must never be read into a model prompt or a repo map. Broad on purpose: a denylist that
# misses `.envrc`, `.git-credentials` or `kubeconfig` reads as "secrets are protected" while handing the
# model the keys. Repos add their own through ``extra_secret_globs``.
_SECRET_RE = re.compile(
    r"(^|/)("
    r"\.env(\..*)?|\.envrc|\.netrc|\.npmrc|\.pypirc|\.git-credentials|\.htpasswd|auth\.json|"
    r"hosts\.ya?ml|kubeconfig|\.kube/.*|"
    r"id_(rsa|dsa|ecdsa|ed25519)[^/]*|credentials(\.[^/]*)?|service-account[^/]*\.json|"
    r"terraform\.tfstate[^/]*|[^/]*\.(tfvars|pem|key|p12|pfx|keystore|jks|ovpn|mobileprovision)|"
    r"\.docker/config\.json|"
    r"[^/]*secrets?\.(json|ya?ml|toml|env)"
    r")$",
    re.IGNORECASE,
)

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build", "target", ".next", ".nuxt",
    ".branchpoint", "coverage",
}


def is_secret_path(rel: str) -> bool:
    return bool(_SECRET_RE.search(rel.replace("\\", "/")))


def in_skipped_dir(rel: str) -> bool:
    return any(part in _SKIP_DIRS for part in Path(rel).parts[:-1])


def within(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """True when ``path`` resolves (symlinks included) to ``root`` or something below it."""
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
    except (OSError, RuntimeError):
        return False
    return p == r or r in p.parents


# --------------------------------------------------------------------------- time / text

def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else now()))


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].strip("-")) or "decision"


def short(text: Any, n: int = 200) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def tail(text: str, n: int = 2000) -> str:
    return text if len(text) <= n else "…" + text[-n:]


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------- JSON

def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model reply (tolerates fences and prose)."""
    if not text:
        raise ValueError("empty model reply")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(s[i:])
                return obj
            except ValueError:
                continue
    raise ValueError("no JSON found in model reply")


# --------------------------------------------------------------------------- subprocess

@dataclass
class CmdResult:
    rc: int
    out: str
    err: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


# Variables that would point a child at the host's interpreter instead of the project's.
_SCRUB_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")


def clean_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["CI"] = "1"
    if extra:
        env.update(extra)
    return env


def hermes_child_env(cwd: str | os.PathLike | None = None) -> dict[str, str]:
    """Environment for a headless ``hermes`` child: nothing inherited from the parent's session, gateway, YOLO or
    approval state (every ``HERMES_*`` variable except HERMES_HOME is dropped), working directory pinned."""
    env = clean_env()
    for k in list(env):
        if k.startswith("HERMES_") and k != "HERMES_HOME":
            del env[k]
    try:  # the calling session's profile (a multiplexed host does not export it to os.environ)
        from hermes_constants import get_hermes_home  # public plugin API
        env["HERMES_HOME"] = str(get_hermes_home())
    except Exception:
        pass
    if cwd is not None:
        env["TERMINAL_CWD"] = str(cwd)   # a gateway-inherited cwd must never point back at the user's tree
        env["PWD"] = str(cwd)
    return env


def run(cmd: Sequence[str], cwd: str | os.PathLike, *, env: Mapping[str, str] | None = None,
        timeout: float = 120, stdin: str | None = None) -> CmdResult:
    """Run an argv list (never through a shell) and capture its output."""
    t0 = time.monotonic()
    try:
        cp = subprocess.run(
            list(cmd), cwd=str(cwd), env=dict(env) if env is not None else None, input=stdin,
            capture_output=True, text=True, timeout=timeout, errors="replace",
        )
        return CmdResult(cp.returncode, cp.stdout, cp.stderr, time.monotonic() - t0)
    except subprocess.TimeoutExpired as exc:
        def _s(v: Any) -> str:
            return v.decode("utf-8", "replace") if isinstance(v, bytes) else (v or "")
        return CmdResult(124, _s(exc.stdout), _s(exc.stderr) + f"\n[timed out after {timeout:.0f}s]",
                         time.monotonic() - t0, timed_out=True)
    except FileNotFoundError as exc:
        return CmdResult(127, "", f"command not found: {exc.filename}", time.monotonic() - t0)
    except OSError as exc:
        return CmdResult(126, "", str(exc), time.monotonic() - t0)


def git(args: Sequence[str], cwd: str | os.PathLike, *, env: Mapping[str, str] | None = None,
        timeout: float = 120, check: bool = True, stdin: str | None = None) -> str:
    """Run ``git <args>`` and return stdout. ``check`` raises BranchpointError on failure."""
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", **(env or {})}
    res = run(["git", *args], cwd, env=full_env, timeout=timeout, stdin=stdin)
    if check and not res.ok:
        raise BranchpointError(f"git {' '.join(args[:3])} failed: {short(res.err or res.out, 300)}")
    return res.out


def git_ok(args: Sequence[str], cwd: str | os.PathLike, **kw: Any) -> bool:
    try:
        git(args, cwd, **kw)
        return True
    except BranchpointError:
        return False


# Flags that stop a repository's own configuration from running code inside a *parent* git call.
# ``core.fsmonitor`` is the sharp one: it is not a hook, so ``--no-verify`` does not disable it, and a
# rewritten ``<wt>/.git`` pointing at a gitdir the build agent controls is enough to make ``git add``
# execute that command as the user. Attributes-based ``filter``/``textconv`` drivers are shut off with
# ``--no-ext-diff``/``--no-textconv`` and by pointing the global attributes file at /dev/null.
GIT_HARDEN_FLAGS = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "diff.external=",
    "-c", "core.attributesFile=/dev/null",
    "-c", "core.pager=cat",
    "-c", "core.sshCommand=",
)
GIT_HARDEN_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}


def worktree_gitdir(repo: str | os.PathLike, worktree: str | os.PathLike) -> Path:
    """The ``<common>/worktrees/<name>`` git dir for ``worktree``, found from the *repo* side.

    Deliberately does not read ``<wt>/.git`` to decide anything: that file lives in the worktree, which the
    build agent can write. Each ``worktrees/<name>/gitdir`` file is authoritative and outside the worktree.
    """
    common = Path(git(["rev-parse", "--git-common-dir"], repo).strip())
    common = common if common.is_absolute() else Path(repo) / common
    want = (Path(worktree) / ".git").resolve()
    matches: list[Path] = []
    for candidate in (common / "worktrees").glob("*"):
        pointer = candidate / "gitdir"
        try:
            target = Path(pointer.read_text(encoding="utf-8").strip())
        except OSError:
            continue
        if target.resolve() == want:
            matches.append(candidate)
    if len(matches) != 1:
        raise BranchpointError(f"cannot identify the private git dir for worktree {worktree} "
                               f"({len(matches)} candidates): refusing to run git in a worktree we cannot account for")
    return matches[0]


def dotgit_intact(worktree: str | os.PathLike, gitdir: str | os.PathLike) -> bool:
    """True when ``<wt>/.git`` still holds the one pointer line git wrote for this worktree."""
    try:
        text = (Path(worktree) / ".git").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not text.startswith("gitdir:"):
        return False
    target = Path(text.split(":", 1)[1].strip())
    if not target.is_absolute():
        target = (Path(worktree) / target).resolve()
    return target.resolve() == Path(gitdir).resolve()


def git_wt(repo: str | os.PathLike, worktree: str | os.PathLike, args: Sequence[str], *,
           env: Mapping[str, str] | None = None, timeout: float = 120, check: bool = True,
           stdin: str | None = None) -> str:
    """Run git against a route worktree with an explicit git dir, never by discovering it from the cwd.

    Raises ``BranchpointError`` when the worktree's ``.git`` no longer matches the pointer git wrote, so a
    redirected git dir stops the call instead of redirecting it.
    """
    gitdir = worktree_gitdir(repo, worktree)
    if not dotgit_intact(worktree, gitdir):
        raise BranchpointError(
            f"{worktree}/.git no longer points at this worktree's git dir. A route may not redirect git, "
            "so this call was refused.")
    full_env = {**GIT_HARDEN_ENV, **(dict(env) if env else {})}
    res = run(["git", *GIT_HARDEN_FLAGS, f"--git-dir={gitdir}", f"--work-tree={worktree}", *args],
              worktree, env=full_env, timeout=timeout, stdin=stdin)
    if check and not res.ok:
        raise BranchpointError(f"git {' '.join(args[:3])} failed: {short(res.err or res.out, 300)}")
    return res.out


def allow_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """An allowlisted environment for an *unsandboxed* command: nothing the caller's shell held.

    Only used when the user explicitly accepts unsandboxed checks (``--unsafe-checks``). Unlike
    ``clean_env`` this starts empty, so a provider key or an SSH agent cannot reach the command.
    """
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM")
           if os.environ.get(k)}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CI"] = "1"
    env.update(extra or {})
    return env


GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Branchpoint", "GIT_AUTHOR_EMAIL": "branchpoint@localhost",
    "GIT_COMMITTER_NAME": "Branchpoint", "GIT_COMMITTER_EMAIL": "branchpoint@localhost",
}


@contextmanager
def worktree_lock(repo: str | Path, *, timeout: float = 180):
    """Serialize our Git worktree mutations across threads/processes sharing one common directory.

    Git publishes worktree metadata in multiple steps; its own ref locks do not protect readers of
    a sibling's not-yet-created commondir. Never unlink this lock file: waiters must share its inode.
    """
    try:
        import fcntl
    except ImportError as exc:
        raise BranchpointError("Worktree management requires POSIX file locking (Linux/macOS; use WSL on Windows).") from exc
    common = Path(git(["rev-parse", "--git-common-dir"], repo).strip())
    common = common if common.is_absolute() else Path(repo) / common
    deadline = time.monotonic() + timeout
    with (common.resolve() / "branchpoint-worktrees.lock").open("a") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BranchpointError(f"Timed out waiting for this repository's worktree mutation lock ({timeout}s).")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def repo_root(cwd: str | os.PathLike) -> Path:
    res = run(["git", "rev-parse", "--show-toplevel"], cwd, timeout=15)
    if not res.ok:
        raise BranchpointError(f"{cwd} is not inside a git repository (Branchpoint needs git worktrees). Run it from your project, "
                               "pass --repo <path>, or set terminal.cwd in your Hermes config.")
    return Path(res.out.strip()).resolve()


def snapshot_tree(repo: str | os.PathLike, *, parent: str | None = None, message: str = "branchpoint snapshot",
                  timeout: float = 60) -> str:
    """Commit the *current working tree* (tracked + untracked, .gitignore respected) as a dangling
    commit, without touching the user's index, HEAD, branches or files. Returns the commit SHA."""
    repo = Path(repo)
    gitdir = Path(git(["rev-parse", "--absolute-git-dir"], repo).strip())
    tmpdir = tempfile.mkdtemp(prefix="bp-index-")
    idx = os.path.join(tmpdir, "index")
    try:
        real = gitdir / "index"
        if real.exists():
            try:
                Path(idx).write_bytes(real.read_bytes())
            except OSError:
                pass
        env = {"GIT_INDEX_FILE": idx, **GIT_IDENTITY}
        git(["add", "-A", "--", "."], repo, env=env, timeout=timeout)
        tree = git(["write-tree"], repo, env=env).strip()
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args[2:2] = ["-p", parent]
        return git(args, repo, env=env).strip()
    finally:
        try:
            for f in Path(tmpdir).iterdir():
                f.unlink()
            os.rmdir(tmpdir)
        except OSError:
            pass


def tree_of(repo: str | os.PathLike, rev: str) -> str:
    return git(["rev-parse", f"{rev}^{{tree}}"], repo).strip()


def head_sha(repo: str | os.PathLike) -> str | None:
    res = run(["git", "rev-parse", "--verify", "-q", "HEAD^{commit}"], repo, timeout=15)
    return res.out.strip() if res.ok and res.out.strip() else None


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
