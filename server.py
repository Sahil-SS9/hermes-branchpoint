"""Live multiverse view: a tiny loopback-only HTTP server streaming run events over SSE.

Stdlib only (no aiohttp), bound to 127.0.0.1, every request needs the per-server token and a loopback
Host header (DNS-rebinding guard). It serves one static page plus read-only JSON/SSE endpoints.
"""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from .detail import build_detail
from .ledger import RunEvidence

VIEW_HTML = Path(__file__).parent / "view" / "index.html"


class ViewServer:
    def __init__(self, get_run: Callable[[str], object], default_run: Callable[[], str | None], port: int = 0):
        self.get_run = get_run
        self.default_run = default_run
        self.token = secrets.token_urlsafe(16)
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/?t={self.token}"

    def run_url(self, run_id: str) -> str:
        return f"{self.url}&run={run_id}"

    def start(self) -> str:
        if self._httpd is not None:
            return self.url
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a, **k):  # silence
                return

            def _deny(self, code: int = 403, msg: str = "forbidden") -> None:
                body = msg.encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200) -> None:
                body = json.dumps(obj, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                host = (self.headers.get("Host") or "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    return self._deny()
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if q.get("t", [""])[0] != outer.token:
                    return self._deny(401, "token required")
                run_id = q.get("run", [""])[0] or outer.default_run() or ""
                if u.path == "/":
                    html = VIEW_HTML.read_text(encoding="utf-8").replace("__RUN_ID__", run_id).replace("__TOKEN__", outer.token)
                    body = html.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                try:
                    run = outer.get_run(run_id)
                except Exception as exc:
                    return self._json({"error": str(exc)}, 404)
                if u.path == "/state":
                    return self._json({"meta": run.meta, "events": run.bus.all()})
                if u.path == "/detail":
                    return self._json(build_detail(run.dir))
                if u.path == "/decision":
                    p = run.dir / "decision.md"
                    return self._json({"markdown": p.read_text(encoding="utf-8") if p.exists() else ""})
                if u.path == "/evidence":
                    item = RunEvidence(run.dir).resolve(q.get("ref", [""])[0])
                    return self._json(item or {"error": "unknown ref"}, 200 if item else 404)
                if u.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    seq = int(self.headers.get("Last-Event-ID") or q.get("since", ["0"])[0] or 0)
                    try:
                        while True:
                            evs = run.bus.wait(seq, 12.0)
                            if not evs:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                continue
                            for ev in evs:
                                seq = ev["seq"]
                                self.wfile.write(f"id: {seq}\ndata: {json.dumps(ev, default=str)}\n\n".encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                return self._deny(404, "not found")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="branchpoint-view", daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
