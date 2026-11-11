"""Blast-radius tracer: what does a change (or a plan) reach?

Static only: an import graph (Python via ``ast``; JS/TS via a relative-import scan, since madge and
ts-morph are not guaranteed to be installed), public-API surface comparison, schema/migration and
config/env detection, and dependency-manifest diffs. Every fact records how it was established so
the council can tell traced facts from model-declared ones.
"""
from __future__ import annotations

import ast
import json
import posixpath
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .ledger import Blast
from .protocol import normalise_rel
from .util import git, in_skipped_dir, is_secret_path, run, short

PY_EXT = (".py",)
JS_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
MAX_GRAPH_FILES = 4000
MAX_FILE_BYTES = 400_000

SCHEMA_PATH_RE = re.compile(
    r"(^|/)(migrations?|alembic|db/migrate|prisma)/|\.sql$|(^|/)schema\.(py|prisma|graphql|sql)$|(^|/)models?/migrations/",
    re.I,
)
SCHEMA_LINE_RE = re.compile(
    r"\b(CREATE|ALTER|DROP)\s+(TABLE|INDEX|COLUMN)\b|\bop\.(add_column|create_table|drop_column|drop_table|alter_column)\b",
    re.I,
)
CONFIG_PATH_RE = re.compile(
    r"(^|/)(docker-compose[^/]*\.ya?ml|Dockerfile[^/]*|\.env\.example|\.env\.sample|Procfile|Makefile|"
    r"[^/]*\.(ini|cfg|conf)|\.github/workflows/[^/]+|config/[^/]+\.(ya?ml|json|toml))$",
    re.I,
)
ENV_KEY_RES = (
    re.compile(r"os\.environ(?:\.get)?\s*[\[(]\s*[\"']([A-Za-z_][A-Za-z0-9_]{2,})[\"']"),
    re.compile(r"os\.getenv\(\s*[\"']([A-Za-z_][A-Za-z0-9_]{2,})[\"']"),
    re.compile(r"process\.env\.([A-Za-z_][A-Za-z0-9_]{2,})"),
    re.compile(r"\bgetenv\(\s*[\"']([A-Za-z_][A-Za-z0-9_]{2,})[\"']"),
)
DEP_MANIFESTS = {"requirements.txt", "pyproject.toml", "package.json", "go.mod", "Cargo.toml", "Gemfile", "Pipfile"}
LOCKFILES = {"uv.lock", "poetry.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "go.sum", "Gemfile.lock", "Pipfile.lock"}
INFRA_DEPS = {
    "redis", "aioredis", "hiredis", "pymemcache", "pylibmc", "python-memcached", "celery", "kombu", "pika", "aio-pika",
    "kafka-python", "confluent-kafka", "aiokafka", "pymongo", "motor", "psycopg2", "psycopg2-binary", "psycopg", "asyncpg",
    "mysqlclient", "pymysql", "aiomysql", "elasticsearch", "opensearch-py", "boto3", "rq", "dramatiq", "ioredis", "memcached",
    "pg", "mysql2", "mongodb", "mongoose", "amqplib", "kafkajs", "bullmq", "bull",
}
CONTRACT_TOKENS = {
    "response", "jsonresponse", "plaintextresponse", "header", "headers", "status_code", "httpexception", "etag",
    "cache-control", "if-none-match", "last-modified", "304", "request", "redirectresponse", "streamingresponse",
}
ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "route", "websocket", "api_route"}


# --------------------------------------------------------------------------- import graph

@dataclass
class ImportGraph:
    files: set[str] = field(default_factory=set)
    forward: dict[str, set[str]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)
    languages: set[str] = field(default_factory=set)
    truncated: bool = False

    def add_edge(self, src: str, dst: str) -> None:
        if src == dst:
            return
        self.forward.setdefault(src, set()).add(dst)
        self.reverse.setdefault(dst, set()).add(src)


def _list_files(root: Path) -> list[str]:
    res = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], root, timeout=30)
    names = [n for n in res.out.split("\0") if n] if res.ok else []
    if not names:  # not a git checkout (or empty): walk
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if not in_skipped_dir(rel):
                    names.append(rel)
    return sorted(n for n in names if not in_skipped_dir(n) and not is_secret_path(n))


def _read(root: Path, rel: str) -> str:
    p = root / rel
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _py_module_map(py_files: Iterable[str]) -> dict[str, str]:
    mods: dict[str, str] = {}
    for f in py_files:
        parts = f[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        variants = {".".join(parts)}
        if parts[0] in ("src", "lib") and len(parts) > 1:
            variants.add(".".join(parts[1:]))
        for v in variants:
            mods.setdefault(v, f)
    return mods


def _resolve_py(candidate: str, mods: dict[str, str]) -> str | None:
    parts = candidate.split(".")
    while parts:
        key = ".".join(parts)
        if key in mods:
            return mods[key]
        parts.pop()
    return None


def _py_imports(src: str, file_rel: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    parts = file_rel[:-3].split("/")
    is_init = parts[-1] == "__init__"
    pkg_parts = parts[:-1]
    cands: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            cands.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = node.level - 1
                base = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
                prefixes = [".".join(base)]
                if base and base[0] in ("src", "lib"):
                    prefixes.append(".".join(base[1:]))
            else:
                prefixes = [""]
            mod = node.module or ""
            for pre in prefixes:
                head = ".".join(x for x in (pre, mod) if x)
                if head:
                    cands.append(head)
                for a in node.names:
                    if a.name != "*":
                        cands.append(".".join(x for x in (head, a.name) if x))
    _ = is_init
    return cands


_JS_IMPORT_RE = re.compile(
    r"""(?:import|export)\s[^'"`;]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)|import\(\s*['"]([^'"]+)['"]\s*\)"""
)


def _resolve_js(spec: str, file_rel: str, files: set[str]) -> str | None:
    if not spec.startswith("."):
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(file_rel), spec))
    cands = [base] + [base + e for e in JS_EXT] + [posixpath.join(base, "index" + e) for e in JS_EXT]
    for c in cands:
        if c in files:
            return c
    return None


def build_graph(root: str | Path) -> ImportGraph:
    root = Path(root)
    files = _list_files(root)
    g = ImportGraph()
    if len(files) > MAX_GRAPH_FILES:
        files = files[:MAX_GRAPH_FILES]
        g.truncated = True
    g.files = set(files)
    py = [f for f in files if f.endswith(PY_EXT)]
    js = [f for f in files if f.endswith(JS_EXT)]
    if py:
        g.languages.add("python")
    if js:
        g.languages.add("javascript/typescript")
    mods = _py_module_map(py)
    for f in py:
        for cand in _py_imports(_read(root, f), f):
            dst = _resolve_py(cand, mods)
            if dst:
                g.add_edge(f, dst)
    for f in js:
        for m in _JS_IMPORT_RE.finditer(_read(root, f)):
            spec = next((x for x in m.groups() if x), "")
            dst = _resolve_js(spec, f, g.files)
            if dst:
                g.add_edge(f, dst)
    return g


def dependents_of(graph: ImportGraph, changed: Iterable[str], depth: int = 4) -> set[str]:
    """Files that (transitively, up to ``depth`` hops) import any of ``changed`` (excluding them)."""
    changed = set(changed)
    seen: set[str] = set()
    frontier = {c for c in changed if c in graph.files}
    for _ in range(depth):
        nxt: set[str] = set()
        for f in frontier:
            for d in graph.reverse.get(f, ()):
                if d not in seen and d not in changed:
                    seen.add(d)
                    nxt.add(d)
        if not nxt:
            break
        frontier = nxt
    return seen


def is_test_path(rel: str) -> bool:
    name = posixpath.basename(rel)
    return bool(re.search(r"(^|/)(tests?|__tests__|spec)/", rel)) or name.startswith("test_") or name.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", ".test.tsx"))


# --------------------------------------------------------------------------- python public surface

def _sig(node: ast.AST) -> str:
    try:
        args = ast.unparse(node.args)  # type: ignore[attr-defined]
        returns = getattr(node, "returns", None)
        ret = f" -> {ast.unparse(returns)}" if returns is not None else ""
        return f"({args}){ret}"
    except Exception:  # pragma: no cover - defensive
        return "(?)"


def _route_of(fn: ast.AST) -> tuple[str, str] | None:
    for dec in getattr(fn, "decorator_list", []):
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in ROUTE_METHODS:
            path = ""
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                path = dec.args[0].value
            return dec.func.attr.upper(), path
    return None


def _contract_tokens(fn: ast.AST) -> set[str]:
    toks: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            toks.add(n.id.lower())
        elif isinstance(n, ast.Attribute):
            toks.add(n.attr.lower())
        elif isinstance(n, ast.arg):
            toks.add(n.arg.lower())
            if n.annotation is not None:
                toks.update(t.lower() for t in re.findall(r"[A-Za-z_]+", ast.unparse(n.annotation)))
        elif isinstance(n, ast.Constant) and isinstance(n.value, (str, int)):
            toks.add(str(n.value).lower())
    return toks & CONTRACT_TOKENS


@dataclass
class PySurface:
    symbols: dict[str, str] = field(default_factory=dict)
    routes: dict[str, dict[str, Any]] = field(default_factory=dict)  # "GET /x" -> {name, sig, tokens}
    all_names: list[str] | None = None


def py_surface(src: str) -> PySurface:
    out = PySurface()
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    out.all_names = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            route = _route_of(node)
            if route:
                out.routes[f"{route[0]} {route[1]}"] = {"name": node.name, "sig": _sig(node), "tokens": sorted(_contract_tokens(node))}
            if not node.name.startswith("_"):
                out.symbols[node.name] = _sig(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = sorted(m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"))
            out.symbols[node.name] = f"class({','.join(ast.unparse(b) for b in node.bases)}) methods={methods}"
    if out.all_names is not None:
        out.symbols = {k: v for k, v in out.symbols.items() if k in out.all_names or k in {r['name'] for r in out.routes.values()}}
    return out


def compare_surfaces(path: str, base: PySurface | None, head: PySurface | None, has_dependents: bool) -> list[str]:
    """Human-readable public-API changes between two versions of one Python module."""
    notes: list[str] = []
    base = base or PySurface()
    head = head or PySurface()
    for key, h in head.routes.items():
        b = base.routes.get(key)
        if b is None:
            notes.append(f"{path}: new route {key} ({h['name']})")
            continue
        added = set(h["tokens"]) - set(b["tokens"])
        if b["sig"] != h["sig"]:
            notes.append(f"{path}: route {key} signature changed {b['sig']} -> {h['sig']}")
        elif added:
            notes.append(f"{path}: route {key} response contract changed (now uses {', '.join(sorted(added))})")
    for key in base.routes:
        if key not in head.routes:
            notes.append(f"{path}: route {key} removed")
    if has_dependents or base.routes or head.routes:
        for name, sig in base.symbols.items():
            if name not in head.symbols:
                notes.append(f"{path}: public symbol {name} removed")
            elif head.symbols[name] != sig and name not in {r["name"] for r in base.routes.values()}:
                notes.append(f"{path}: public symbol {name} signature changed")
    return notes


def js_exports(src: str) -> set[str]:
    names = set(re.findall(r"export\s+(?:async\s+)?(?:function|class|const|let|var|interface|type|enum)\s+([A-Za-z_$][\w$]*)", src))
    if re.search(r"export\s+default\b", src):
        names.add("default")
    for grp in re.findall(r"export\s*\{([^}]*)\}", src):
        names.update(x.strip().split(" as ")[-1].strip() for x in grp.split(",") if x.strip())
    return names


# --------------------------------------------------------------------------- dependency manifests

def _norm_dep(name: str) -> str:
    return re.split(r"[<>=!~;\[ @(]", name.strip(), maxsplit=1)[0].strip().lower().replace("_", "-")


def deps_in(manifest: str, text: str) -> set[str]:
    base = posixpath.basename(manifest)
    out: set[str] = set()
    try:
        if base == "package.json":
            data = json.loads(text or "{}")
            for sect in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                out.update((data.get(sect) or {}).keys())
        elif base == "pyproject.toml":
            data = tomllib.loads(text or "")
            proj = data.get("project") or {}
            for d in proj.get("dependencies") or []:
                out.add(_norm_dep(d))
            for lst in (proj.get("optional-dependencies") or {}).values():
                out.update(_norm_dep(d) for d in lst)
            for lst in (data.get("dependency-groups") or {}).values():
                out.update(_norm_dep(d) for d in lst if isinstance(d, str))
            poetry = (data.get("tool") or {}).get("poetry") or {}
            out.update(k.lower() for k in (poetry.get("dependencies") or {}) if k.lower() != "python")
        elif base == "Cargo.toml":
            data = tomllib.loads(text or "")
            for sect in ("dependencies", "dev-dependencies", "build-dependencies"):
                out.update((data.get(sect) or {}).keys())
        elif base == "go.mod":
            out.update(m.group(1) for m in re.finditer(r"^\s*(?:require\s+)?([\w.\-/]+\.[\w.\-/]+)\s+v[\w.\-+]+", text or "", re.M))
        elif base == "Gemfile":
            out.update(re.findall(r"^\s*gem\s+['\"]([^'\"]+)['\"]", text or "", re.M))
        else:  # requirements*.txt / Pipfile-ish
            for line in (text or "").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith(("-", "git+", "http")):
                    out.add(_norm_dep(line))
    except (ValueError, tomllib.TOMLDecodeError):
        return out
    return {d for d in out if d}


def is_dep_manifest(path: str) -> bool:
    base = posixpath.basename(path)
    return base in DEP_MANIFESTS or bool(re.fullmatch(r"requirements[\w.\-]*\.txt", base))


# --------------------------------------------------------------------------- results

@dataclass
class BlastFacts:
    changed_files: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    dependent_tests: list[str] = field(default_factory=list)
    public_api: list[str] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)
    deps_added: list[str] = field(default_factory=list)
    infra: bool = False
    basis: str = ""
    notes: list[str] = field(default_factory=list)

    def score(self) -> int:
        s = 0
        s += 2 if self.public_api else 0
        s += 2 if self.schema else 0
        s += 2 if len(self.dependents) + len(self.dependent_tests) > 10 else 0
        s += 1 if self.config else 0
        s += 1 if self.deps_added else 0
        s += 1 if len(self.changed_files) >= 5 else 0
        return max(0, min(10, s))

    def to_blast(self, evidence_ref: str = "") -> Blast:
        return Blast(
            dependents=len(self.dependents) + len(self.dependent_tests),
            dependents_list=(self.dependents + self.dependent_tests)[:40],
            public_api=self.public_api[:20], schema=self.schema[:20], config=self.config[:20],
            score=self.score(), basis=self.basis, evidence_ref=evidence_ref,
        )

    def trace_text(self) -> str:
        lines = [f"basis: {self.basis}", f"files changed ({len(self.changed_files)}): " + ", ".join(self.changed_files[:30])]
        lines.append(f"dependents ({len(self.dependents)} source, {len(self.dependent_tests)} tests): " + (", ".join((self.dependents + self.dependent_tests)[:30]) or "none"))
        lines.append("public API: " + ("; ".join(self.public_api) or "no change detected"))
        lines.append("schema/migrations: " + ("; ".join(self.schema) or "none"))
        lines.append("config/env: " + ("; ".join(self.config) or "none"))
        lines.append("new dependencies: " + (", ".join(self.deps_added) or "none"))
        lines.append(f"score: {self.score()}/10 (+2 public API, +2 schema, +2 >10 dependents, +1 config, +1 new dep, +1 >=5 files)")
        lines.extend(self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- diff analysis (execute / baseline)

def _show(repo: Path, rev: str, path: str) -> str | None:
    res = run(["git", "show", f"{rev}:{path}"], repo, timeout=30)
    return res.out if res.ok else None


def analyze_diff(repo: str | Path, base: str, head: str, base_graph: ImportGraph | None = None) -> BlastFacts:
    """Blast radius of the real diff ``base..head`` (any commit-ish in ``repo``)."""
    repo = Path(repo)
    facts = BlastFacts(basis="traced on the real diff")
    out = git(["diff", "--name-status", "-M", "--no-renames", base, head], repo, check=False)
    changed: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            changed.append((parts[0][0], parts[-1]))
    facts.changed_files = [p for _, p in changed if not is_secret_path(p)]
    graph = base_graph
    if graph is None:
        # Build from the head tree materialised lazily: cheaper to reuse caller's graph, else skip.
        graph = ImportGraph()
        facts.notes.append("no base import graph supplied; dependents not traced")
    deps = dependents_of(graph, facts.changed_files)
    facts.dependent_tests = sorted(d for d in deps if is_test_path(d))
    facts.dependents = sorted(d for d in deps if not is_test_path(d))
    if graph.truncated:
        facts.notes.append(f"import graph truncated at {MAX_GRAPH_FILES} files")
    if not (graph.languages):
        facts.notes.append("no Python/JS import graph available for this repo's languages")

    added_lines = git(["diff", "-U0", "--no-color", base, head], repo, check=False)
    # Public API (Python)
    for status, path in changed:
        if path.endswith(".py") and not is_test_path(path):
            b_src = _show(repo, base, path) if status != "A" else None
            h_src = _show(repo, head, path) if status != "D" else None
            facts.public_api.extend(compare_surfaces(
                path, py_surface(b_src) if b_src is not None else None, py_surface(h_src) if h_src is not None else None,
                has_dependents=bool([d for d in graph.reverse.get(path, ()) if not is_test_path(d)]),
            ))
        elif path.endswith(JS_EXT) and not is_test_path(path):
            b_src = _show(repo, base, path) if status != "A" else ""
            h_src = _show(repo, head, path) if status != "D" else ""
            gone = js_exports(b_src or "") - js_exports(h_src or "")
            if gone and graph.reverse.get(path):
                facts.public_api.append(f"{path}: exports removed: {', '.join(sorted(gone))}")
    # Schema / config
    for _, path in changed:
        if SCHEMA_PATH_RE.search(path):
            facts.schema.append(f"{path}: schema/migration file")
        elif CONFIG_PATH_RE.search(path) and not is_dep_manifest(path):
            facts.config.append(f"{path}: config file changed")
    if SCHEMA_LINE_RE.search("\n".join(l for l in added_lines.splitlines() if l.startswith("+") and not l.startswith("+++"))) and not facts.schema:
        facts.schema.append("added lines contain DDL / migration operations")
    envkeys: set[str] = set()
    for line in added_lines.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            for rx in ENV_KEY_RES:
                envkeys.update(rx.findall(line))
    for k in sorted(envkeys):
        facts.config.append(f"env/config key {k}")
    # Dependencies
    for status, path in changed:
        if is_dep_manifest(path):
            b = deps_in(path, _show(repo, base, path) or "") if status != "A" else set()
            h = deps_in(path, _show(repo, head, path) or "") if status != "D" else set()
            facts.deps_added.extend(sorted(h - b))
        elif posixpath.basename(path) in LOCKFILES:
            facts.notes.append(f"{path}: lockfile changed")
    facts.deps_added = sorted(set(facts.deps_added))
    facts.infra = any(_norm_dep(d) in INFRA_DEPS for d in facts.deps_added) or any("docker-compose" in c for c in facts.config)
    return facts


# --------------------------------------------------------------------------- plan analysis (simulate)

def analyze_plan(root: str | Path, plan: dict[str, Any], base_graph: ImportGraph | None = None) -> BlastFacts:
    """Blast radius of a *planned* change, traced statically on the base checkout ``root``."""
    root = Path(root)
    graph = base_graph or build_graph(root)
    facts = BlastFacts(basis="static trace of the planned files on the base import graph + model-declared facts")
    planned = [f for f in plan.get("files") or [] if isinstance(f, dict) and f.get("path")]
    paths = [normalise_rel(f["path"]) for f in planned]
    facts.changed_files = paths
    existing = [p for p in paths if p in graph.files]
    deps = dependents_of(graph, existing)
    facts.dependent_tests = sorted(d for d in deps if is_test_path(d))
    facts.dependents = sorted(d for d in deps if not is_test_path(d))
    for f in planned:
        path = normalise_rel(f["path"])
        symbols = [str(s) for s in f.get("symbols") or []]
        contract = bool(f.get("contract_change"))
        if path in graph.files and path.endswith(".py"):
            surf = py_surface(_read(root, path))
            route_names = {r["name"]: k for k, r in surf.routes.items()}
            for s in symbols:
                if s in route_names and contract:
                    facts.public_api.append(f"{path}: route {route_names[s]} ({s}) contract change [static: handler exists; contract change declared by plan]")
                elif s in surf.symbols and contract and (graph.reverse.get(path) or surf.routes):
                    facts.public_api.append(f"{path}: public symbol {s} changes [static: symbol exists; change declared by plan]")
        if SCHEMA_PATH_RE.search(path):
            facts.schema.append(f"{path}: schema/migration file")
        if CONFIG_PATH_RE.search(path) and not is_dep_manifest(path):
            facts.config.append(f"{path}: config file")
    for claim in plan.get("public_api_changes") or []:
        facts.public_api.append(f"claimed: {short(claim, 160)}")
    for claim in plan.get("schema_changes") or []:
        facts.schema.append(f"claimed: {short(claim, 160)}")
    for key in plan.get("config_keys") or []:
        facts.config.append(f"env/config key {short(key, 80)}")
    base_deps: set[str] = set()
    for m in graph.files:
        if is_dep_manifest(m):
            base_deps |= deps_in(m, _read(root, m))
    new = [str(d) for d in plan.get("new_deps") or [] if _norm_dep(str(d)) not in base_deps]
    facts.deps_added = sorted(set(new))
    facts.infra = bool(plan.get("new_service")) or any(_norm_dep(d) in INFRA_DEPS for d in facts.deps_added)
    facts.public_api = list(dict.fromkeys(facts.public_api))
    facts.config = list(dict.fromkeys(facts.config))
    if not graph.languages:
        facts.notes.append("no Python/JS import graph available for this repo's languages")
    return facts
