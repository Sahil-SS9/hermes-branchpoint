"""The Decision Record's detail view is assembled from run files only: comments must come from real evidence."""
from __future__ import annotations

import json

from hermes_branchpoint.detail import build_detail


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _run(tmp_path):
    rd = tmp_path / "run"
    ev = {
        "C:diff:h1": {"kind": "diff", "title": "app/main.py @@ -1,3 +1,6", "text": "@@\n-a\n+b"},
        "C:diff:h2": {"kind": "diff", "title": "app/main.py @@ -9,3 +12,6", "text": "@@\n-c\n+d"},
        "C:trace:blast": {"kind": "trace", "title": "blast", "text": "x"},
    }
    _w(rd / "routes/C/evidence.json", ev)
    _w(rd / "routes/C/ledger.json", {"route": "C", "blast": {"public_api": ["app/main.py: route GET /r signature changed"]}, "issues": [], "deps_added": []})
    _w(rd / "decision.json", {"winner": "C", "narrative": {
        "winner_why": [{"text": "Small blast radius [C:trace:blast].", "refs": ["C:trace:blast"]}],
        "losers": {}, "risks_carried": [{"text": "Cites something that does not exist.", "refs": ["C:diff:h99"]}], "borrow": []}})
    (rd / "decision.md").write_text("# Decision", encoding="utf-8")
    return rd


def test_comments_attach_to_the_evidence_they_cite(tmp_path):
    d = build_detail(_run(tmp_path))
    texts = [c["text"] for c in d["comments"]["C:trace:blast"]]
    assert texts == ["Small blast radius."]          # inline ref stripped; rendered once as a chip instead


def test_refs_that_do_not_exist_get_no_comment(tmp_path):
    assert "C:diff:h99" not in build_detail(_run(tmp_path))["comments"]


def test_tracer_facts_attach_to_every_hunk_of_the_file_they_concern(tmp_path):
    c = build_detail(_run(tmp_path))["comments"]
    for ref in ("C:diff:h1", "C:diff:h2"):
        assert any(n["source"].startswith("Tracer") and "signature changed" in n["text"] for n in c[ref])


def test_missing_run_files_do_not_crash(tmp_path):
    d = build_detail(tmp_path / "nothing-here")
    assert d["ledgers"] == {} and d["comments"] == {}
