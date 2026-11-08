"""A deterministic OpenAI-compatible chat endpoint for real-process tests (no credentials, no cost).

The script is a list of steps; the step used for a request is the number of tool results already in the
conversation, so each model turn does exactly one scripted tool call, then the final text.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModel:
    def __init__(self, steps, final_text="Built the route. Trade-off: none. Nothing left undone."):
        self.steps = steps            # [(tool_name, args_dict | callable(tool_results)->args_dict)]
        self.final_text = final_text
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a, **k):
                return

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    return self._send(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
                self._send(404, {"error": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(req)
                msg = outer._reply(req)
                usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
                if req.get("stream"):
                    chunks = []
                    base = {"id": "cmpl-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake-model"}
                    delta = {"role": "assistant", "content": msg.get("content") or ""}
                    if msg.get("tool_calls"):
                        delta["tool_calls"] = [{"index": 0, **tc} for tc in msg["tool_calls"]]
                    chunks.append({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}], "usage": usage})
                    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    return self._send(200, body.encode(), "text/event-stream")
                self._send(200, {"id": "cmpl-1", "object": "chat.completion", "created": int(time.time()), "model": "fake-model",
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}], "usage": usage})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def _reply(self, req):
        if not req.get("tools"):
            return {"role": "assistant", "content": "ok"}
        msgs = req.get("messages") or []
        results = [m for m in msgs if m.get("role") == "tool"]
        i = len(results)
        if i < len(self.steps):
            name, args, *rest = self.steps[i]
            if rest:
                time.sleep(float(rest[0]))   # lets background work (the plugin's checks) finish deterministically
            if callable(args):
                args = args([m.get("content") for m in results])
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return {"role": "assistant", "content": self.final_text}

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
