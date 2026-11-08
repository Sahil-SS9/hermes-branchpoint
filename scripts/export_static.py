#!/usr/bin/env python3
"""Export a finished run as ONE self-contained HTML file (no server, no network): open it in any browser.

    python scripts/export_static.py <state-dir>/runs/<run-id> docs/demo.html
The page replays the run with a play/pause button, a scrubber, speed and three looks; evidence refs open their snippets.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(run_dir: str, out: str) -> int:
    rd = Path(run_dir)
    meta = json.loads((rd / "run.json").read_text())
    events = [json.loads(l) for l in (rd / "events.jsonl").read_text().splitlines() if l.strip()]
    evidence = {}
    for p in (rd / "routes").glob("*/evidence.json"):
        evidence.update(json.loads(p.read_text()))
    decision = (rd / "decision.md").read_text() if (rd / "decision.md").exists() else ""
    # nothing machine-specific leaves the file: drop absolute paths from the embedded metadata
    safe_meta = {k: v for k, v in meta.items() if k not in ("repo", "context", "view_url", "anchor_notes")}
    sys.path.insert(0, str(ROOT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("hermes_branchpoint", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["hermes_branchpoint"] = pkg
    spec.loader.exec_module(pkg)
    from hermes_branchpoint.detail import build_detail
    payload = {"state": {"meta": safe_meta, "events": events}, "decision": decision, "evidence": evidence, "detail": build_detail(rd)}
    html = (ROOT / "view" / "index.html").read_text(encoding="utf-8")
    blob = json.dumps(payload)
    blob = re.sub(r"(?:/tmp|/home|/Users)/[^\"\s\\]*/([^/\"\s\\]+)", r"…/\1", blob)     # no machine paths in a shareable file
    blob = blob.replace("<", "\\u003c")   # JSON-safe, and keeps "<!--" / "<script" in diffs from confusing the HTML parser
    html = html.replace("<script>\n", f"<script>\nwindow.__BP_STATIC__={blob};\n", 1) if "<script>\n" in html else html.replace("<script>", f"<script>window.__BP_STATIC__={blob};", 1)
    html = html.replace("__RUN_ID__", meta["run_id"]).replace("__TOKEN__", "static")
    Path(out).write_text(html, encoding="utf-8")
    print(f"wrote {out} ({Path(out).stat().st_size // 1024} KB, {len(events)} events, {len(evidence)} evidence snippets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
