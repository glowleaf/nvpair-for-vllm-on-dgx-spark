# NVPAIR for vLLM / SGLang on DGX Spark

Make [NVIDIA PAIR (Personal AI Router)](https://github.com/NVIDIA/Personal-AI-Router) route cluster requests to a **vLLM or SGLang** server on a DGX Spark — without loading a second copy of the model.

PAIR in v1.0.0 only recognises three engines: **Ollama, LM Studio, llama.cpp**. There is no vLLM engine and no custom/OpenAI-compatible engine type, so a server like `vllm serve` or SGLang's OpenAI server on `:8888` is invisible to it: the node joins the cluster but advertises nothing, and requests come back as

```
{"error":"no available node advertises the requested model"}
```

This repo fixes that with a tiny façade: a process that speaks the **Ollama API** on Ollama's own port and translates everything to your existing OpenAI server.

```
PAIR (engine: "Ollama")  ->  127.0.0.1:11434  ->  vllm_ollama_bridge.py  ->  127.0.0.1:8888
                              (Ollama API)          (facade)                  (real vLLM/SGLang)
```

No model is re-downloaded, no GGUF conversion, no second load in memory. The model list PAIR shows is exactly what your vLLM advertises, so it updates by itself when you switch `~/models/current`.

## Why not just use llama.cpp / Ollama natively?

Measurements from the shipped PAIR v1.0.0 binaries (`nvpair-engine-manager`, `-workload-manager`, `-cluster-manager`, `-tui`, service zips, 2026-11-10):

| Check | Result |
|---|---|
| `grep -a -o -i "vllm"` on all four binaries | **0 hits** |
| Engine identifiers present in `nvpair-engine-manager` | `ollama` (59), `lmstudio` (28), `llama.cpp` (15), `llamacpp` (11) |
| `custom` / `base_url` / `openai-compatible` engine hooks | none (only Go stdlib symbols) |
| Engine probe endpoints | `api/version`, `api/tags` (Ollama), `/props` (llama.cpp), `/v1/models`, `/health` |

Adoption is documented for "Ollama or LM Studio"; **llama.cpp is always a PAIR-managed install**. So on a DGX Spark where the 27B lives in NVFP4/SGLang format, the only ways to get PAIR serving it are (a) load a second GGUF copy in RAM, or (b) this façade. (b) costs a few MB of RSS.

## The second contract: PAIR enumerates with `GET /models`

The Ollama façade above works, but it is not the only way in, and on a DGX Spark it is not the
cheapest. PAIR's **llama.cpp** engine can be pointed at a bridge on `:8081` that keeps the
upstream OpenAI server exactly where it is. Getting that to advertise models needs **two**
contracts, and the second one is the one nobody documents:

| Contract | Endpoint PAIR probes | What it decides |
|---|---|---|
| readiness | `GET /props` | engine shows **HEALTHY** |
| enumeration | `GET /models` (llama.cpp router shape) | node **advertises** its models |

`GET /props` returning `{"role":"router"}` hardcodes llama.cpp. That alone is not enough:
vLLM and SGLang answer `/v1/models` and **404 on `/models`**, so the engine goes green and the
node advertises nothing. Measured on a DGX Spark with PAIR v1.0.0 and SGLang on `:8888`:

```
$ journalctl -u pair-bridge | tail
GET /models -> 404          # repeated, every refresh
$ curl -s localhost:8080/v1/models        # PAIR's own proxy
{"error":"no available node advertises the requested model"}
```

`pair_llamacpp_bridge.py` fixes that: it answers `/models` itself by reshaping the upstream's
`/v1/models` into llama.cpp's shape, and proxies everything else untouched (including
`Server: llama.cpp`, which PAIR sniffs).

```bash
python3 pair_llamacpp_bridge.py 8081 8888 qwen3.8-27b-sglang /home/me/models/current 27000000000 NVFP4 262144
#                        listen upstream alias              model_path               n_params ftype  n_ctx
```

Then point PAIR's llama.cpp engine port at `8081` and restart the node's registration:

```
ENGINE     INSTALLED  RUNNING  HEALTHY  ENGINE PORT  PROXY PORT
llama.cpp  yes        yes      yes      8081         8082
Models
  qwen3.8-27b-sglang
```

Verified end to end: `curl -s localhost:8082/v1/chat/completions -d '{"model":"qwen3.8-27b-sglang",...}'`
returned a completion, and the cluster inventory listed the model within 15 s of registration.
`n_ctx` must be passed explicitly: PAIR reads it from the `/models` metadata, so a hardcoded
8192 advertises 8192 even when the model serves 262144.

### Which bridge to use

- **Ollama façade** (`vllm_ollama_bridge.py`, port `11434`) when PAIR must treat the node as an
  Ollama node, or when you want `/api/chat` translation.
- **llama.cpp bridge** (`pair_llamacpp_bridge.py`, port `8081`) when the node is joining a
  cluster of llama.cpp nodes. It is a pass-through for `/v1/*`, so streaming, tool calls and
  long contexts behave exactly as the upstream does.

## Requirements

- NVIDIA PAIR v1.0.0+ installed on the node (`nvpair` on PATH).
  On a DGX Spark / GB10 use the **arm64** package — the amd64 `.deb` fails with `libuuid1:amd64 is not installable`:
  ```bash
  wget https://github.com/NVIDIA/Personal-AI-Router/releases/download/v1.0.0/NVPAIR-Setup-1.0.0-arm64.deb
  sudo apt install ./NVPAIR-Setup-1.0.0-arm64.deb
  ```
- Any **OpenAI-compatible** server on `127.0.0.1:8888` (vLLM, SGLang, TGI, LMDeploy …). Nothing else.
- Python 3 (stdlib only — no pip packages).
- The host must already have an Ollama install PAIR can see (`/usr/local/bin/ollama` and/or `~/.ollama`), otherwise PAIR lists the engine as `INSTALLED no`. The façade only needs the *detection*, not the running ollama.

## Install

```bash
git clone https://github.com/glowleaf/nvpair-for-vllm-on-dgx-spark.git
cd nvpair-for-vllm-on-dgx-spark
sudo ./install.sh                 # copies the bridge + unit, enables and starts it
```

`install.sh` honours three overrides:

```bash
sudo UPSTREAM=http://127.0.0.1:8888 PORT=11434 USER_NAME=machinegeorge ./install.sh
```

Then start PAIR and open your own node's Engines pane (`enter` on the row):

```
ENGINE      INSTALLED  RUNNING  HEALTHY  ENGINE PORT  PROXY PORT
llama.cpp   no         no       no       8081         8080
LM Studio   no         no       no       1235         1234
Ollama      yes        yes      yes      11434        11435
Models
  <whatever :8888 serves>
```

PAIR now routes to it like any other Ollama node, including across the cluster. Clients keep using PAIR's proxy port (`11435` locally, or each node's own proxy port).

## Verify

```bash
# façade alone
curl -s http://127.0.0.1:11434/api/version          # {"version": "0.40.16"}
curl -s http://127.0.0.1:11434/api/tags             # models mirrored from :8888
curl -s -X POST http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id from /api/tags>","messages":[{"role":"user","content":"hi"}],"stream":false}'

# through PAIR's proxy for that engine
curl -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

If `/api/tags` returns an empty list, the upstream on `:8888` is not running — the façade deliberately reports zero models rather than inventing any.

## Behaviour notes / gotchas

- **The façade owns `11434`.** If real Ollama is installed on the same host it can no longer bind there. Either stop it, or set `PORT=` in `install.sh` and change PAIR's Ollama *engine port* (`e` in the Engines pane) to match.
- **Model list = whatever `/v1/models` says.** No caching, no renaming: switching the served model on `:8888` shows up on the next PAIR refresh.
- **SGLang on DGX Spark:** `qwen27b.service`-style units often do `ExecStartPre=systemctl stop comfyui-clean.service` to get KV room. Check ComfyUI's queue (`curl http://127.0.0.1:8188/queue`) before restarting the LLM, or you kill a running render.
- **Streaming:** `/v1/*` is proxied byte-for-byte with chunked encoding; `/api/chat` with `"stream": true` is translated to NDJSON.
- **PAIR's scheduler ignores VRAM** and weighs queued work + GPU utilisation, so it will happily send work to a node whose GPU is busy with ComfyUI.

## Verified on

DGX Spark `spark-a714` (GB10, aarch64, 128 GB unified) — PAIR v1.0.0 + `NVPAIR-Setup-1.0.0-arm64.deb`, upstream SGLang/vLLM on `:8888`. Bridge translation also verified against an unrelated OpenAI-compatible upstream (a Raspberry Pi running Ollama), `/api/chat` round-trip returning the expected completion.

## Licence

MIT — see `LICENSE`.
