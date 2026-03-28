#!/usr/bin/env python3
import asyncio, os, json, time, logging, hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    import orjson
    _dumps = lambda o: orjson.dumps(o).decode()
    _loads = orjson.loads
except ImportError:
    _dumps = json.dumps
    _loads = json.loads

import httpx, uvicorn
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from mcp.server.fastmcp import FastMCP
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

HF_TOKEN          = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY      = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_UPLOAD_TIMEOUT = int(os.environ.get("HF_UPLOAD_TIMEOUT", "300"))
MCP_HOST          = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT          = int(os.environ.get("MCP_PORT", "8000"))
HF_CACHE_TTL      = int(os.environ.get("HF_CACHE_TTL", "180"))
WEB_CONCURRENCY   = int(os.environ.get("WEB_CONCURRENCY", "2"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hf-mcp")

api = HfApi(token=HF_TOKEN or None)
_executor = ThreadPoolExecutor(max_workers=16)
_http_headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
_http_client: Optional[httpx.AsyncClient] = None
_HEALTH_BODY = _dumps({"status": "ok", "v": "6.0.0"}).encode()
_whoami_cache: Optional[dict] = None

def _get_whoami() -> dict:
    global _whoami_cache
    if _whoami_cache is None and HF_TOKEN:
        try:
            info = api.whoami()
            _whoami_cache = {"name": info.get("name", ""), "type": info.get("type", "user")}
        except Exception as e:
            return {"error": str(e)}
    return _whoami_cache or {"error": "no token"}

def get_http() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            headers={**_http_headers, "Accept-Encoding": "gzip, br"},
            timeout=httpx.Timeout(connect=6.0, read=60.0, write=60.0, pool=4.0),
            follow_redirects=True, http2=True,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=20, keepalive_expiry=60.0),
        )
    return _http_client

_cache: dict = {}

def _ck(repo_id, fn, rt): return hashlib.md5(f"{rt}/{repo_id}/{fn}".encode()).hexdigest()
def _cget(k):
    e = _cache.get(k)
    return e[1] if e and (time.monotonic() - e[0]) < HF_CACHE_TTL else None
def _cset(k, repo_id, rt, v):
    v["_r"] = f"{rt}/{repo_id}"
    _cache[k] = (time.monotonic(), v)
    if len(_cache) > 400:
        cut = time.monotonic() - HF_CACHE_TTL
        for x in [x for x, (ts, _) in list(_cache.items()) if ts < cut]: _cache.pop(x, None)
def _cinv(repo_id, rt):
    tag = f"{rt}/{repo_id}"
    for k in [k for k, (_, v) in list(_cache.items()) if v.get("_r") == tag]: _cache.pop(k, None)

async def _fetch(repo_id, fn, rt, max_sz=500_000):
    k = _ck(repo_id, fn, rt)
    c = _cget(k)
    if c: return {**c, "_cached": True}
    try:
        r = await get_http().get(f"https://huggingface.co/{rt}s/{repo_id}/resolve/main/{fn}")
        r.raise_for_status()
        raw = r.text
        res = {"content": raw[:max_sz], "size": len(raw), "truncated": len(raw) > max_sz}
        _cset(k, repo_id, rt, res)
        return res
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

async def _run(fn, *a, **kw):
    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(_executor, lambda: fn(*a, **kw)),
            timeout=HF_UPLOAD_TIMEOUT)
    except asyncio.TimeoutError: return {"error": f"timeout {HF_UPLOAD_TIMEOUT}s"}
    except Exception as e: return {"error": str(e), "type": type(e).__name__}

def _safe(fn, *a, **kw):
    try: return fn(*a, **kw)
    except Exception as e: return {"error": str(e)}

def _rguard():
    if HF_READ_ONLY: raise PermissionError("READ-ONLY mode")

mcp = FastMCP("hf-mcp", stateless_http=True, json_response=True)

# ─────────────────────── TOOLS ────────────────────────

@mcp.tool()
def hf_whoami() -> dict:
    """Returns your HF username. Call once when you need username/repo_id."""
    return _get_whoami()


@mcp.tool()
async def hf_list_files(repo_id: str, repo_type: str = "space") -> dict:
    """List all files in a HF repo. Call this before reading any files."""
    try:
        r = await get_http().get(
            f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/main?recursive=true&expand=false")
        r.raise_for_status()
        items = _loads(r.text)
        return {"files": [{
            "path": it["path"],
            "size": it.get("size"),
            "type": it.get("type", "file"),
        } for it in items]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def hf_read_file(repo_id: str, path: str, repo_type: str = "space") -> dict:
    """Read a single file from a HF repo. Returns {content, size}."""
    return await _fetch(repo_id, path, repo_type)


@mcp.tool()
async def hf_write_file(
    repo_id: str, path: str, content: str,
    repo_type: str = "space", commit_message: str = "",
) -> dict:
    """Write (create or overwrite) a single file in a HF repo."""
    _rguard()
    r = await _run(api.upload_file,
                   path_or_fileobj=content.encode(),
                   path_in_repo=path,
                   repo_id=repo_id,
                   repo_type=repo_type,
                   commit_message=commit_message or f"update {path}")
    _cinv(repo_id, repo_type)
    return {"ok": True, "result": str(r)} if not isinstance(r, dict) else r


@mcp.tool()
async def hf_write_many(
    repo_id: str,
    files: list,
    commit_message: str,
    repo_type: str = "space",
) -> dict:
    """
    Write multiple files to a HF repo in one commit.
    files: [{"path": "index.html", "content": "..."}]
    """
    _rguard()
    ops = [CommitOperationAdd(path_in_repo=f["path"],
                              path_or_fileobj=f["content"].encode())
           for f in files]
    r = await _run(api.create_commit,
                   repo_id=repo_id, repo_type=repo_type,
                   operations=ops, commit_message=commit_message)
    _cinv(repo_id, repo_type)
    return {"ok": True, "files": len(files), "commit": str(r)} if not isinstance(r, dict) else r


@mcp.tool()
async def hf_delete_file(
    repo_id: str, path: str,
    repo_type: str = "space", commit_message: str = "",
) -> dict:
    """Delete a file from a HF repo."""
    _rguard()
    r = await _run(api.delete_file,
                   path_in_repo=path, repo_id=repo_id, repo_type=repo_type,
                   commit_message=commit_message or f"delete {path}")
    _cinv(repo_id, repo_type)
    return {"ok": True, "deleted": path} if not isinstance(r, dict) else r


@mcp.tool()
async def hf_create_space(
    space_name: str,
    files: list,
    sdk: str = "static",
    private: bool = False,
) -> dict:
    """
    Create a HF Space and upload files in one call.
    space_name: just the name, e.g. "my-portfolio" (no username prefix).
    files: [{"path": "index.html", "content": "..."}]
    sdk: static | gradio | streamlit | docker
    Returns {repo_id, url}.
    """
    _rguard()
    whoami = _get_whoami()
    if "error" in whoami: return whoami
    repo_id = f"{whoami['name']}/{space_name}"
    cr = _safe(api.create_repo, repo_id=repo_id, repo_type="space",
               space_sdk=sdk, private=private, exist_ok=True)
    if isinstance(cr, dict) and "error" in cr: return cr
    if not files:
        return {"repo_id": repo_id, "url": f"https://huggingface.co/spaces/{repo_id}"}
    ops = [CommitOperationAdd(path_in_repo=f["path"],
                              path_or_fileobj=f["content"].encode()) for f in files]
    r = await _run(api.create_commit, repo_id=repo_id, repo_type="space",
                   operations=ops, commit_message="Initial upload")
    if isinstance(r, dict) and "error" in r:
        return {"repo_id": repo_id, "created": True, "upload_error": r["error"]}
    return {"repo_id": repo_id, "url": f"https://huggingface.co/spaces/{repo_id}",
            "files_uploaded": len(files)}


@mcp.tool()
def hf_create_repo(name: str, repo_type: str = "model", private: bool = False) -> dict:
    """Create a HF repo (model/dataset). For spaces use hf_create_space instead."""
    _rguard()
    whoami = _get_whoami()
    if "error" in whoami: return whoami
    repo_id = f"{whoami['name']}/{name}"
    r = _safe(api.create_repo, repo_id=repo_id, repo_type=repo_type,
              private=private, exist_ok=True)
    return {"repo_id": repo_id, "result": str(r)}


@mcp.tool()
def hf_repo_info(repo_id: str, repo_type: str = "space") -> dict:
    """Get info about a HF repo."""
    r = _safe(api.repo_info, repo_id=repo_id, repo_type=repo_type)
    return r if isinstance(r, dict) else r.__dict__


@mcp.tool()
def hf_restart_space(space_id: str) -> dict:
    """Restart a HF Space. Call after every deployment."""
    _rguard()
    return {"restarted": str(_safe(api.restart_space, repo_id=space_id))}


@mcp.tool()
def hf_search(content_type: str, query: str = "", author: str = "", limit: int = 20) -> dict:
    """Search HF Hub. content_type: models | datasets | spaces"""
    try:
        kw = dict(search=query or None, author=author or None, limit=limit)
        if content_type == "models":     items = api.list_models(**kw)
        elif content_type == "datasets": items = api.list_datasets(**kw)
        else:                            items = api.list_spaces(**kw)
        return {"results": [{"id": getattr(i, "id", None) or getattr(i, "modelId", None),
                             "author": getattr(i, "author", None)} for i in items]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def hf_system_info() -> dict:
    """Server status + whoami."""
    return {"v": "6.0.0", "ro": HF_READ_ONLY, "token": bool(HF_TOKEN),
            "cache": len(_cache), "whoami": _get_whoami()}


# ─────────────────────── ASGI ─────────────────────────

class FastMiddleware:
    __slots__ = ("app", "_lh")
    def __init__(self, app: ASGIApp, port: int = 8000):
        self.app = app
        self._lh = f"localhost:{port}".encode()
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            if scope.get("path", "") in ("/health", "/health/"):
                await Response(_HEALTH_BODY, media_type="application/json")(scope, receive, send)
                return
            lh = self._lh
            scope["headers"] = [
                (b"host", lh) if n == b"host"
                else (b"origin", b"http://" + lh) if n == b"origin"
                else (n, v)
                for n, v in scope.get("headers", [])
            ]
        await self.app(scope, receive, send)

app = GZipMiddleware(FastMiddleware(mcp.streamable_http_app(), port=MCP_PORT), minimum_size=400)

if __name__ == "__main__":
    uvicorn.run("server:app", host=MCP_HOST, port=MCP_PORT, workers=WEB_CONCURRENCY,
                log_level="warning", access_log=False, loop="uvloop", http="httptools",
                timeout_keep_alive=45, timeout_graceful_shutdown=3,
                backlog=256, limit_concurrency=100)
