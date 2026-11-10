#!/usr/bin/env python3
"""Ollama-API bridge in front of a vLLM OpenAI server.

PAIR (NVIDIA Personal AI Router) only recognises Ollama, LM Studio and
llama.cpp. This makes vLLM look like Ollama so PAIR can adopt it and route
cluster traffic to the real vLLM process on :8888 -- no second copy of the
model is loaded.

Binds 127.0.0.1:11434 (Ollama's default port, which PAIR probes) and forwards:
    /api/version, /api/tags, /api/chat, /api/generate, /api/embed, /health
    /v1/*  (passthrough, streaming SSE preserved)
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8888")
LISTEN_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("BRIDGE_PORT", "11434"))
TIMEOUT = float(os.environ.get("BRIDGE_TIMEOUT", "600"))


def upstream_json(path, payload=None, stream=False):
    url = UPSTREAM.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def model_list():
    """Ask vLLM what it serves; empty list while vLLM is down."""
    try:
        with upstream_json("/v1/models") as r:
            body = json.loads(r.read().decode() or "{}")
        out = []
        for m in body.get("data", []):
            mid = m.get("id") or "model"
            fam = mid.split(":")[0].split("/")[-1].split("-")[0].lower() or "unknown"
            out.append({
                "name": mid,
                "model": mid,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime()),
                "size": 0,
                "digest": "vllm-" + mid.replace(":", "-").replace("/", "-"),
                "details": {
                    "parent_model": "",
                    "format": "safetensors",
                    "family": fam,
                    "families": [fam],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            })
        return out
    except Exception:
        return []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------- helpers ----------
    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode() or "{}")
        except Exception:
            return {}

    def _proxy(self, path, payload=None):
        """Stream an upstream response back verbatim (SSE-safe)."""
        try:
            with upstream_json(path, payload) as r:
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._send(503, {"error": "upstream vLLM unreachable on %s: %s" % (UPSTREAM, e)})

    # ---------- routes ----------
    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            self._send(200, {"status": "ok", "upstream": UPSTREAM})
        elif self.path == "/api/version":
            self._send(200, {"version": "0.40.16"})
        elif self.path.startswith("/api/tags") or self.path.startswith("/api/ps"):
            self._send(200, {"models": model_list()})
        elif self.path.startswith("/api/show"):
            models = model_list()
            name = ""
            if "name=" in self.path:
                name = self.path.split("name=")[1].split("&")[0]
            hit = next((m for m in models if m["name"] == name), models[0] if models else None)
            self._send(200, {"details": (hit or {}).get("details", {}),
                             "model_info": {}, "capabilities": ["completion", "tools"]})
        elif self.path.startswith("/v1/") or self.path.startswith("/api/"):
            self._proxy(self.path)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        payload = self._body()
        if self.path.startswith("/api/chat"):
            msgs = payload.get("messages") or []
            v = {
                "model": payload.get("model") or "",
                "messages": msgs,
                "temperature": payload.get("options", {}).get("temperature", 0.7),
                "max_tokens": payload.get("options", {}).get("num_predict", 512),
                "stream": bool(payload.get("stream")),
            }
            try:
                if not v["stream"]:
                    with upstream_json("/v1/chat/completions", v) as r:
                        body = json.loads(r.read().decode() or "{}")
                    ch = (body.get("choices") or [{}])[0]
                    msg = ch.get("message", {})
                    usage = body.get("usage", {})
                    self._send(200, {
                        "model": body.get("model", v["model"]),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime()),
                        "message": {"role": msg.get("role", "assistant"),
                                    "content": msg.get("content", "")},
                        "done": True,
                        "done_reason": ch.get("finish_reason", "stop"),
                        "prompt_eval_count": usage.get("prompt_tokens", 0),
                        "eval_count": usage.get("completion_tokens", 0),
                    })
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    with upstream_json("/v1/chat/completions", v) as r:
                        for line in r:
                            line = line.strip()
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                break
                            try:
                                d = json.loads(data.decode())
                            except Exception:
                                continue
                            delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
                            piece = json.dumps({"model": v["model"], "message": {"role": "assistant",
                                                                                "content": delta},
                                                "done": False}).encode() + b"\n"
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                            self.wfile.flush()
                    end = json.dumps({"model": v["model"], "message": {"role": "assistant", "content": ""},
                                      "done": True, "done_reason": "stop"}).encode() + b"\n"
                    self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(end), end))
                    self.wfile.flush()
            except Exception as e:
                self._send(503, {"error": "upstream vLLM unreachable on %s: %s" % (UPSTREAM, e)})
        elif self.path.startswith("/api/generate"):
            prompt = payload.get("prompt") or ""
            v = {"model": payload.get("model") or "", "prompt": prompt,
                 "max_tokens": payload.get("options", {}).get("num_predict", 512),
                 "stream": False}
            try:
                with upstream_json("/v1/completions", v) as r:
                    body = json.loads(r.read().decode() or "{}")
                text = (body.get("choices") or [{}])[0].get("text", "")
                self._send(200, {"model": v["model"], "response": text, "done": True,
                                 "done_reason": "stop"})
            except Exception as e:
                self._send(503, {"error": "upstream vLLM unreachable on %s: %s" % (UPSTREAM, e)})
        elif self.path.startswith("/api/embed") or self.path.startswith("/api/embeddings"):
            hint = self.path.rsplit("/", 1)[-1]
            self._proxy("/v1/embeddings", {"model": payload.get("model", ""),
                                           "input": payload.get("input") or payload.get("prompt") or ""})
        elif self.path.startswith("/v1/") or self.path.startswith("/api/"):
            self._proxy(self.path, payload)
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    sys.stderr.write("vLLM->Ollama bridge on %s:%d -> %s\n" % (LISTEN_HOST, LISTEN_PORT, UPSTREAM))
    srv.serve_forever()
