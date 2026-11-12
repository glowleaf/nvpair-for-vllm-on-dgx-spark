"""PAIR engine bridge: llama.cpp router identity in front of ANY OpenAI server.

Two contracts, both learned the hard way against PAIR v1.0.0:

  1. GET /props -> {"role": "router"}      makes PAIR call the engine HEALTHY
  2. GET /models -> llama.cpp list shape   makes PAIR ADVERTISE the models

vLLM/SGLang serve /v1/models and 404 on /models. With a bare reverse proxy the engine shows
healthy and the node advertises nothing, so every cluster request comes back as
{"error":"no available node advertises the requested model"}. This bridge answers /models by
reshaping the upstream's /v1/models into llama.cpp's shape, and proxies everything else
(Appending "Server: llama.cpp" on responses, because PAIR sniffs the header.)

usage:
  python3 pair_llamacpp_bridge.py [listen] [upstream] [alias] [model_path] [n_params] [ftype] [n_ctx]

  PAIR_BRIDGE_BIND=0.0.0.0   # default 127.0.0.1; keep loopback unless PAIR runs elsewhere
"""
import http.client
import http.server
import json
import os
import sys
import time

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
UPSTREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
ALIAS = sys.argv[3] if len(sys.argv) > 3 else "model"
MODEL_PATH = sys.argv[4] if len(sys.argv) > 4 else "unknown"
N_PARAMS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
FTYPE = sys.argv[6] if len(sys.argv) > 6 else "unknown"
N_CTX = int(sys.argv[7]) if len(sys.argv) > 7 else 8192
BIND = os.environ.get("PAIR_BRIDGE_BIND", "127.0.0.1")
LOG = os.path.expanduser("~/pair_llamacpp_bridge.log")

PROPS = {
    "role": "router",
    "max_instances": 4,
    "models_autoload": True,
    "model_alias": ALIAS,
    "model_path": MODEL_PATH,
    "default_generation_settings": {"params": None, "n_ctx": N_CTX},
    "ui_settings": {},
    "build_info": "b12262",
    "cors_proxy_enabled": False,
}


def log(line):
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
    except OSError:
        pass


def upstream_models():
    """Ask the real server what it serves."""
    ids = []
    try:
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=6)
        conn.request("GET", "/v1/models")
        body = conn.getresponse().read()
        conn.close()
        for m in json.loads(body.decode()).get("data", []):
            if m.get("id"):
                ids.append(m["id"])
    except Exception as exc:  # noqa: BLE001
        log("models: upstream error %r" % (exc,))
    return ids or [ALIAS]


class Bridge(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Server", "llama.cpp")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _model_list(self):
        now = int(time.time())
        data = [{
            "id": mid,
            "object": "model",
            "created": now,
            "owned_by": "llamacpp",
            "meta": {"n_vocab": 0, "n_ctx": N_CTX, "n_params": N_PARAMS,
                     "size": 0, "ftype": FTYPE, "n_embd": 0},
        } for mid in upstream_models()]
        return json.dumps({"object": "list", "data": data}).encode()

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=1800)
        try:
            conn.request(method, self.path, body=body, headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
        except Exception as exc:  # noqa: BLE001
            log("ERROR %s %s %r" % (method, self.path, exc))
            self._send(502, b'{"error":"upstream unreachable"}')
            return
        finally:
            conn.close()
        log("%s %s -> %s (%d B)" % (method, self.path, resp.status, len(data)))
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            # hop-by-hop and identity headers must not be copied: PAIR reads Server
            if k.lower() in ("transfer-encoding", "content-length", "connection", "server"):
                continue
            self.send_header(k, v)
        self.send_header("Server", "llama.cpp")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/props":
            self._send(200, json.dumps(PROPS).encode())
        elif path in ("/models", "/v1/models"):
            body = self._model_list()
            log("GET %s -> %d models" % (path, body.count(b'"id"')))
            self._send(200, body)
        else:
            self._relay("GET")

    def do_HEAD(self):
        path = self.path.split("?")[0]
        if path in ("/props", "/models", "/v1/models"):
            self._send(200, self._model_list() if path != "/props" else json.dumps(PROPS).encode())
        else:
            self._relay("HEAD")

    def do_POST(self):
        self._relay("POST")


if __name__ == "__main__":
    open(LOG, "w", encoding="utf-8").close()
    srv = http.server.ThreadingHTTPServer((BIND, LISTEN), Bridge)
    print(json.dumps({"bridge": "%s:%d -> 127.0.0.1:%d (%s, n_ctx %d)"
                      % (BIND, LISTEN, UPSTREAM, ALIAS, N_CTX)}), flush=True)
    srv.serve_forever()
