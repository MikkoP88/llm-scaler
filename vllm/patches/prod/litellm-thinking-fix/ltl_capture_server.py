#!/usr/bin/env python3
"""ltl_capture_server — logs the exact JSON body LiteLLM puts on the wire,
returns a minimal valid chat.completion so the proxy call completes."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "captured.jsonl")
RESP = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
    "model": "q",
    "choices": [{"index": 0, "message": {"role": "assistant",
                                         "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
              "total_tokens": 2},
}).encode()


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"__raw": body.decode("utf-8", "replace")}
        with open(LOG, "ab") as f:
            f.write(json.dumps({"path": self.path,
                                "body": parsed}).encode() + b"\n")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(RESP)))
        self.end_headers()
        self.wfile.write(RESP)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8901), H).serve_forever()
