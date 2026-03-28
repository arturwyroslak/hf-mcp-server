#!/usr/bin/env python3
import asyncio, os, re, json, time, logging, hashlib
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
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

HF_TOKEN          = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY      = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_ADMIN_MODE     = os.environ.get("HF_ADMIN_MODE", "false").lower() == "true"
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
_HEALTH_BODY = _dumps({"status": "ok", "v": "5.3.0"}).encode()

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
_flight: dict = {}

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
    if k in _flight:
        try: return await asyncio.shield(_flight[k])
        except: pass
    fut = asyncio.get_event_loop().create_future()
    _flight[k] = fut
    try:
        r = await get_http().get(f"https://huggingface.co/{rt}s/{repo_id}/resolve/main/{fn}")
        r.raise_for_status()
        raw = r.text
        res = {"content": raw[:max_sz], "size": len(raw), "truncated": len(raw) > max_sz}
        _cset(k, repo_id, rt, res); fut.set_result(res); return res
    except httpx.HTTPStatusError as e:
        err = {"error": f"HTTP {e.response.status_code}"}; fut.set_exception(Exception(err["error"])); return err
    except Exception as e:
        err = {"error": str(e)}; fut.set_exception(e); return err
    finally: _flight.pop(k, None)

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

def _edits(text, edits):
    log_e = []
    for i, e in enumerate(edits):
        m = e.get("mode", "replace")
        try:
            if m == "overwrite":
                text = e["content"]; log_e.append({"i": i, "ok": True})
            elif m == "replace":
                if e["old"] not in text: log_e.append({"i": i, "ok": False, "reason": "not_found"}); continue
                n = text.count(e["old"]); text = text.replace(e["old"], e["new"]); log_e.append({"i": i, "ok": True, "n": n})
            elif m == "regex":
                fl = 0
                for f in e.get("flags","").upper().split("|"):
                    if f=="IGNORECASE": fl|=re.IGNORECASE
                    elif f=="MULTILINE": fl|=re.MULTILINE
                    elif f=="DOTALL": fl|=re.DOTALL
                text, n = re.subn(e["pattern"], e["replacement"], text, flags=fl); log_e.append({"i": i, "ok": True, "n": n})
            elif m == "insert_after":
                lines, out, found = text.splitlines(keepends=True), [], False
                for ln in lines:
                    out.append(ln)
                    if e["after_pattern"] in ln: ins=e["insert"]; out.append(ins if ins.endswith("\n") else ins+"\n"); found=True
                text="".join(out); log_e.append({"i":i,"ok":found})
            elif m == "insert_before":
                lines, out, found = text.splitlines(keepends=True), [], False
                for ln in lines:
                    if e["before_pattern"] in ln: ins=e["insert"]; out.append(ins if ins.endswith("\n") else ins+"\n"); found=True
                    out.append(ln)
                text="".join(out); log_e.append({"i":i,"ok":found})
            elif m == "delete_lines":
                old_lines = text.splitlines(keepends=True)
                new_lines = [l for l in old_lines if e["line_pattern"] not in l]
                text="".join(new_lines); log_e.append({"i":i,"ok":True,"deleted":len(old_lines)-len(new_lines)})
            else: log_e.append({"i":i,"ok":False,"reason":"unknown_mode"})
        except Exception as ex: log_e.append({"i":i,"ok":False,"reason":str(ex)})
    return text, log_e

def _rguard():
    if HF_READ_ONLY: raise PermissionError("READ-ONLY mode")
def _aguard():
    if not HF_ADMIN_MODE: raise PermissionError("Admin mode disabled")

mcp = FastMCP("hf-mcp", stateless_http=True, json_response=True)

# ── TOOLS ── keep descriptions SHORT (fewer tokens = faster AI decisions) ────

@mcp.tool()
def hf_system_info() -> dict:
    """Server status and cache stats."""
    return {"v":"5.3.0","ro":HF_READ_ONLY,"admin":HF_ADMIN_MODE,
            "token":bool(HF_TOKEN),"cache":len(_cache),"flight":len(_flight)}

@mcp.tool()
async def hf_list_files(repo_id: str, repo_type: str = "space") -> dict:
    """
    Returns file tree (path, size, type) for a HF repo.
    STEP 1: Always call this first. Then use hf_read_many for only relevant files.
    Skip: *.png *.jpg *.woff *.lock node_modules __pycache__
    """
    try:
        r = await get_http().get(
            f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/main?recursive=true&expand=false")
        r.raise_for_status()
        items = _loads(r.text)
        tree = [{"path":it["path"],"type":it.get("type","file"),"size":it.get("size")} for it in items]
        return {"tree":tree,"files":[x["path"] for x in tree if x["type"]=="file"]}
    except Exception as e: return {"error": str(e)}

@mcp.tool()
async def hf_read_many(repo_id: str, filenames: list, repo_type: str = "space",
                       max_size_per_file: int = 200_000) -> dict:
    """
    Read multiple files in parallel. STEP 2 after hf_list_files.
    Returns {filename: {content,size,truncated}}. Cached 3min.
    """
    results = await asyncio.gather(*[_fetch(repo_id, f, repo_type, max_size_per_file) for f in filenames])
    return dict(zip(filenames, results))

@mcp.tool()
async def hf_smart_edit(repo_id: str, filename: str, edits: list,
                        repo_type: str = "space", commit_message: str = "") -> dict:
    """
    Edit one file with surgical ops in one commit. Cache auto-cleared.
    modes: replace|regex|insert_after|insert_before|delete_lines|overwrite
    replace: {mode,old,new} regex: {mode,pattern,replacement,flags}
    """
    _rguard()
    res = await _fetch(repo_id, filename, repo_type, 2_000_000)
    if "error" in res: return res
    text, elog = _edits(res["content"], edits)
    if text == res["content"]: return {"status":"no_changes","log":elog}
    r = await _run(api.upload_file, path_or_fileobj=text.encode(),
                   path_in_repo=filename, repo_id=repo_id, repo_type=repo_type,
                   commit_message=commit_message or f"edit {filename}")
    _cinv(repo_id, repo_type)
    return {"status":"ok","result":str(r),"log":elog}

@mcp.tool()
async def hf_atomic_commit(repo_id: str, files: list, commit_message: str,
                           repo_type: str = "space", create_pr: bool = False) -> dict:
    """
    Commit multiple files atomically. Prefer over multiple hf_smart_edit calls.
    ops: add{op,path,content} delete{op,path} rename{op,from,to} edit{op,path,edits[]}
    """
    _rguard()
    edit_items = [(i,x) for i,x in enumerate(files) if x.get("op")=="edit"]
    fetched = {}
    if edit_items:
        res_list = await asyncio.gather(*[_fetch(repo_id,x["path"],repo_type,2_000_000) for _,x in edit_items])
        fetched = {x["path"]:r for (_,x),r in zip(edit_items,res_list)}
    ops, elogs = [], {}
    for item in files:
        op = item.get("op","add")
        if op=="add": ops.append(CommitOperationAdd(path_in_repo=item["path"],path_or_fileobj=item["content"].encode()))
        elif op=="delete": ops.append(CommitOperationDelete(path_in_repo=item["path"]))
        elif op=="rename":
            ops.append(CommitOperationCopy(src_path_in_repo=item["from"],dest_path_in_repo=item["to"]))
            ops.append(CommitOperationDelete(path_in_repo=item["from"]))
        elif op=="copy": ops.append(CommitOperationCopy(src_path_in_repo=item["from"],dest_path_in_repo=item["to"]))
        elif op=="edit":
            fr = fetched.get(item["path"],{})
            if "error" in fr: elogs[item["path"]]={"error":fr}; continue
            text, el = _edits(fr["content"], item.get("edits",[]))
            elogs[item["path"]] = el
            ops.append(CommitOperationAdd(path_in_repo=item["path"],path_or_fileobj=text.encode()))
    if not ops: return {"error":"no valid ops","logs":elogs}
    r = await _run(api.create_commit, repo_id=repo_id, repo_type=repo_type,
                   operations=ops, commit_message=commit_message, create_pr=create_pr)
    _cinv(repo_id, repo_type)
    return {"status":"ok","commit":str(r),"ops":len(ops),"logs":elogs} if not isinstance(r,dict) else r

@mcp.tool()
def hf_repository_manager(action: str, repo_id: str, repo_type: str = "model",
                          private: bool = False, space_sdk: str = "") -> dict:
    """Repo CRUD. action: create|delete|info"""
    if action=="info":
        r=_safe(api.repo_info,repo_id=repo_id,repo_type=repo_type)
        return r if isinstance(r,dict) else r.__dict__
    if action=="create":
        _rguard()
        return {"created":str(_safe(api.create_repo,repo_id=repo_id,repo_type=repo_type,private=private,space_sdk=space_sdk or None,exist_ok=True))}
    if action=="delete":
        _rguard();_aguard();_safe(api.delete_repo,repo_id=repo_id,repo_type=repo_type)
        return {"deleted":repo_id}
    return {"error":f"unknown: {action}"}

@mcp.tool()
async def hf_file_operations(action: str, repo_id: str, filename: str = "",
                             repo_type: str = "space", content: str = "",
                             commit_message: str = "", old_text: str = "",
                             new_text: str = "", max_size: int = 500_000) -> dict:
    """File CRUD fallback. action: read|write|edit|delete|validate|backup"""
    if action=="read": return await _fetch(repo_id,filename,repo_type,max_size)
    if action=="write":
        _rguard()
        r=await _run(api.upload_file,path_or_fileobj=content.encode(),path_in_repo=filename,
                     repo_id=repo_id,repo_type=repo_type,commit_message=commit_message or f"upload {filename}")
        _cinv(repo_id,repo_type); return {"uploaded":str(r)}
    if action=="edit":
        _rguard()
        res=await _fetch(repo_id,filename,repo_type,2_000_000)
        if "error" in res: return res
        if old_text not in res["content"]: return {"error":"old_text not found"}
        up=res["content"].replace(old_text,new_text)
        r=await _run(api.upload_file,path_or_fileobj=up.encode(),path_in_repo=filename,
                     repo_id=repo_id,repo_type=repo_type,commit_message=commit_message or f"edit {filename}")
        _cinv(repo_id,repo_type); return {"edited":str(r)}
    if action=="delete":
        _rguard()
        r=await _run(api.delete_file,path_in_repo=filename,repo_id=repo_id,repo_type=repo_type,
                     commit_message=commit_message or f"delete {filename}")
        _cinv(repo_id,repo_type); return {"deleted":str(r)}
    if action=="validate":
        res=await _fetch(repo_id,filename,repo_type)
        if "error" in res: return res
        ext=filename.rsplit(".",1)[-1].lower() if "." in filename else "text"
        try:
            if ext=="json": _loads(res["content"])
            return {"valid":True,"format":ext,"size":res.get("size",0)}
        except Exception as e: return {"valid":False,"error":str(e)}
    if action=="backup":
        _rguard()
        res=await _fetch(repo_id,filename,repo_type,2_000_000)
        if "error" in res: return res
        r=await _run(api.upload_file,path_or_fileobj=res["content"].encode(),
                     path_in_repo=f"{filename}.backup",repo_id=repo_id,repo_type=repo_type,
                     commit_message=f"backup {filename}")
        return {"backup":f"{filename}.backup","result":str(r)}
    return {"error":f"unknown: {action}"}

@mcp.tool()
def hf_search_hub(content_type: str, query: str = "", author: str = "",
                 filter_tag: str = "", limit: int = 20) -> dict:
    """Search HF Hub. content_type: models|datasets|spaces"""
    try:
        kw=dict(search=query or None,author=author or None,filter=filter_tag or None,limit=limit)
        if content_type=="models": items=api.list_models(**kw)
        elif content_type=="datasets": items=api.list_datasets(**kw)
        else: items=api.list_spaces(**kw)
        return {"results":[{"id":getattr(i,"id",None) or getattr(i,"modelId",None),
                            "author":getattr(i,"author",None),"likes":getattr(i,"likes",None)} for i in items]}
    except Exception as e: return {"error":str(e)}

@mcp.tool()
def hf_space_management(action: str, space_id: str, to_id: str = "",
                        sleep_time: int = 300) -> dict:
    """Space ops. action: runtime_info|restart|pause|set_sleep_time|duplicate"""
    if action=="runtime_info":
        r=_safe(api.get_space_runtime,repo_id=space_id)
        return r if isinstance(r,dict) else r.__dict__
    if action=="restart": _rguard(); return {"restarted":str(_safe(api.restart_space,repo_id=space_id))}
    if action=="pause": _rguard(); return {"paused":str(_safe(api.pause_space,repo_id=space_id))}
    if action=="set_sleep_time": _rguard(); return {"set":str(_safe(api.set_space_sleep_time,repo_id=space_id,sleep_time=sleep_time))}
    if action=="duplicate": _rguard(); return {"to":to_id,"result":str(_safe(api.duplicate_space,from_id=space_id,to_id=to_id,exist_ok=True))}
    return {"error":f"unknown: {action}"}

@mcp.tool()
def hf_community_features(action: str, repo_id: str = "", repo_type: str = "model",
                          title: str = "", description: str = "") -> dict:
    """Community. action: like|unlike|get_likes|create_discussion|get_commits|get_refs"""
    if action=="like": _rguard(); _safe(api.like,repo_id=repo_id,repo_type=repo_type); return {"liked":repo_id}
    if action=="unlike": _rguard(); _safe(api.unlike,repo_id=repo_id,repo_type=repo_type); return {"unliked":repo_id}
    if action=="get_likes": return {"repos":[str(x) for x in (_safe(api.list_liked_repos) or [])]}
    if action=="create_discussion":
        _rguard()
        r=_safe(api.create_discussion,repo_id=repo_id,repo_type=repo_type,title=title or "Discussion",description=description,pull_request=False)
        return {"discussion":str(r)}
    if action=="get_commits":
        r=_safe(api.list_repo_commits,repo_id=repo_id,repo_type=repo_type)
        return r if isinstance(r,dict) else {"commits":[{"id":c.commit_id,"msg":c.title} for c in list(r)[:50]]}
    if action=="get_refs":
        r=_safe(api.list_repo_refs,repo_id=repo_id,repo_type=repo_type)
        return r if isinstance(r,dict) else r.__dict__
    return {"error":f"unknown: {action}"}

# ── ASGI ──────────────────────────────────────────────────────────────────────

class FastMiddleware:
    """Health endpoint + Host/Origin header fix for MCP 421 bypass."""
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
