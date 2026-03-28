#!/usr/bin/env python3
import asyncio
import os
import re
import json
import time
import fnmatch
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import uvicorn
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from mcp.server.fastmcp import FastMCP
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

HF_TOKEN             = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY         = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_ADMIN_MODE        = os.environ.get("HF_ADMIN_MODE", "false").lower() == "true"
HF_UPLOAD_TIMEOUT    = int(os.environ.get("HF_UPLOAD_TIMEOUT", "300"))
MCP_HOST             = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT             = int(os.environ.get("MCP_PORT", "8000"))
HF_CACHE_TTL         = int(os.environ.get("HF_CACHE_TTL", "120"))  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hf-mcp")

api = HfApi(token=HF_TOKEN or None, endpoint="https://huggingface.co")
_executor = ThreadPoolExecutor(max_workers=16)

# Shared async httpx client — reused across all requests, avoids connection overhead
_http_headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
_http_client: Optional[httpx.AsyncClient] = None

# In-memory file content cache: key -> (timestamp, content_dict)
_file_cache: dict = {}

mcp = FastMCP("hf-mcp-server", stateless_http=True, json_response=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            headers=_http_headers,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
            follow_redirects=True,
            http2=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


def _cache_key(repo_id: str, filename: str, repo_type: str) -> str:
    return hashlib.md5(f"{repo_type}/{repo_id}/{filename}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    entry = _file_cache.get(key)
    if entry and (time.time() - entry[0]) < HF_CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: dict):
    _file_cache[key] = (time.time(), value)
    # Evict old entries if cache grows large
    if len(_file_cache) > 200:
        now = time.time()
        expired = [k for k, (ts, _) in _file_cache.items() if now - ts > HF_CACHE_TTL]
        for k in expired:
            _file_cache.pop(k, None)


def _cache_invalidate(repo_id: str, repo_type: str = "space"):
    """Invalidate all cached files for a repo after a write."""
    prefix = hashlib.md5(f"{repo_type}/{repo_id}/".encode()).hexdigest()[:8]
    to_del = [k for k in _file_cache if _file_cache[k][1].get("_repo") == f"{repo_type}/{repo_id}"]
    for k in to_del:
        _file_cache.pop(k, None)


async def fetch_file_http(repo_id: str, filename: str, repo_type: str,
                          max_size: int = 500_000) -> dict:
    """Fetch file content directly via HTTP — faster than hf_hub_download (no disk I/O)."""
    cache_key = _cache_key(repo_id, filename, repo_type)
    cached = _cache_get(cache_key)
    if cached:
        log.info(f"Cache hit: {repo_id}/{filename}")
        return cached

    client = get_http_client()
    url = f"https://huggingface.co/{repo_type}s/{repo_id}/resolve/main/{filename}"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.text
        result = {
            "content": raw[:max_size],
            "size": len(raw),
            "truncated": len(raw) > max_size,
            "_repo": f"{repo_type}/{repo_id}",
        }
        _cache_set(cache_key, result)
        return result
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: fn(*args, **kwargs)),
            timeout=HF_UPLOAD_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {"error": f"Timed out after {HF_UPLOAD_TIMEOUT}s"}
    except Exception as e:
        log.error(f"{getattr(fn, '__name__', str(fn))} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(f"{fn.__name__} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


def _is_text_file(filename: str) -> bool:
    TEXT_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".txt",
        ".sh", ".bash", ".env", ".dockerfile", ".xml", ".svg", ".rst",
        ".go", ".rs", ".java", ".c", ".cpp", ".h", ".rb", ".php",
        ".vue", ".svelte", ".graphql", ".sql", ".csv",
    }
    ext = os.path.splitext(filename.lower())[1]
    return ext in TEXT_EXTS or filename in ("Dockerfile", ".gitignore", ".gitattributes", "Makefile")


def _apply_edits(text: str, edits: list) -> tuple[str, list]:
    """Apply a list of edit operations to text. Returns (new_text, edit_log)."""
    edit_log = []
    for i, edit in enumerate(edits):
        mode = edit.get("mode", "replace")
        try:
            if mode == "overwrite":
                text = edit["content"]
                edit_log.append({"edit": i, "mode": mode, "status": "ok"})
            elif mode == "replace":
                old = edit["old"]
                if old not in text:
                    edit_log.append({"edit": i, "mode": mode, "status": "not_found"})
                    continue
                count = text.count(old)
                text = text.replace(old, edit["new"])
                edit_log.append({"edit": i, "mode": mode, "status": "ok", "replacements": count})
            elif mode == "regex":
                flags = 0
                for f in edit.get("flags", "").split("|"):
                    n = f.strip().upper()
                    if n == "IGNORECASE": flags |= re.IGNORECASE
                    elif n == "MULTILINE": flags |= re.MULTILINE
                    elif n == "DOTALL": flags |= re.DOTALL
                text, count = re.subn(edit["pattern"], edit["replacement"], text, flags=flags)
                edit_log.append({"edit": i, "mode": mode, "status": "ok", "substitutions": count})
            elif mode == "insert_after":
                lines = text.splitlines(keepends=True)
                out, found = [], False
                for line in lines:
                    out.append(line)
                    if edit["after_pattern"] in line:
                        ins = edit["insert"]
                        out.append(ins if ins.endswith("\n") else ins + "\n")
                        found = True
                text = "".join(out)
                edit_log.append({"edit": i, "mode": mode, "status": "ok" if found else "not_found"})
            elif mode == "insert_before":
                lines = text.splitlines(keepends=True)
                out, found = [], False
                for line in lines:
                    if edit["before_pattern"] in line:
                        ins = edit["insert"]
                        out.append(ins if ins.endswith("\n") else ins + "\n")
                        found = True
                    out.append(line)
                text = "".join(out)
                edit_log.append({"edit": i, "mode": mode, "status": "ok" if found else "not_found"})
            elif mode == "delete_lines":
                lines = text.splitlines(keepends=True)
                new_lines = [l for l in lines if edit["line_pattern"] not in l]
                text = "".join(new_lines)
                edit_log.append({"edit": i, "mode": mode, "status": "ok",
                                  "lines_deleted": len(lines) - len(new_lines)})
            else:
                edit_log.append({"edit": i, "mode": mode, "status": "unknown_mode"})
        except Exception as e:
            edit_log.append({"edit": i, "mode": mode, "status": "error", "detail": str(e)})
    return text, edit_log


def read_only_guard():
    if HF_READ_ONLY:
        raise PermissionError("Server is in READ-ONLY mode.")

def admin_guard():
    if not HF_ADMIN_MODE:
        raise PermissionError("Admin mode is disabled.")


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def hf_system_info() -> dict:
    """Return server status, version and HF connectivity."""
    return {
        "version": "5.0.0",
        "read_only": HF_READ_ONLY,
        "admin_mode": HF_ADMIN_MODE,
        "token_set": bool(HF_TOKEN),
        "upload_timeout": HF_UPLOAD_TIMEOUT,
        "cache_ttl": HF_CACHE_TTL,
        "cache_entries": len(_file_cache),
        "whoami": safe_run(api.whoami) if HF_TOKEN else None,
    }


@mcp.tool()
async def hf_read_many(
    repo_id: str,
    filenames: list,
    repo_type: str = "space",
    max_size_per_file: int = 200_000,
) -> dict:
    """
    Read multiple files from a HF repo IN PARALLEL — fast, uses HTTP + cache.
    Returns {filename: {content, size, truncated} | {error}}.
    Use this before modifying specific files you already know.
    """
    tasks = [
        fetch_file_http(repo_id, fname, repo_type, max_size_per_file)
        for fname in filenames
    ]
    results_list = await asyncio.gather(*tasks)
    return dict(zip(filenames, results_list))


@mcp.tool()
async def hf_repo_snapshot(
    repo_id: str,
    repo_type: str = "space",
    include_patterns: list = None,
    exclude_patterns: list = None,
    max_size_per_file: int = 100_000,
    max_total_size: int = 1_500_000,
) -> dict:
    """
    Full repository snapshot — returns structure + content of ALL text files IN PARALLEL.
    Use as FIRST step before any modification to get full codebase context.
    include_patterns: e.g. ["*.py", "*.html"] — only include matching files.
    exclude_patterns: skip matching files (binaries excluded by default).
    Returns: tree (all files), files (content dict), stats.
    """
    if include_patterns is None:
        include_patterns = []
    if exclude_patterns is None:
        exclude_patterns = [
            "*.lock", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico",
            "*.woff", "*.woff2", "*.ttf", "*.eot", "*.zip", "*.tar",
            "*.gz", "*.bin", "*.pkl", "*.pt", "*.pth", "*.ckpt",
            "*.safetensors", ".git/*", "node_modules/*", "__pycache__/*",
        ]

    # Get file tree via HTTP — faster than list_repo_tree blocking call
    client = get_http_client()
    tree_url = (f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/main"
                f"?recursive=true&expand=false")
    try:
        resp = await client.get(tree_url)
        resp.raise_for_status()
        items = resp.json()
        tree = [{"path": it["path"], "size": it.get("size"), "type": it.get("type", "file")}
                for it in items]
        all_paths = [it["path"] for it in items if it.get("type", "file") == "file"]
    except Exception as e:
        return {"error": f"Failed to get file tree: {e}"}

    def should_include(fname: str) -> bool:
        if not _is_text_file(fname):
            return False
        if any(fnmatch.fnmatch(fname, p) for p in exclude_patterns):
            return False
        if include_patterns and not any(fnmatch.fnmatch(fname, p) for p in include_patterns):
            return False
        return True

    to_read = [p for p in all_paths if should_include(p)]
    skipped = [p for p in all_paths if not should_include(p)]

    # Fetch ALL files in parallel
    tasks = [fetch_file_http(repo_id, fname, repo_type, max_size_per_file) for fname in to_read]
    results_list = await asyncio.gather(*tasks)

    files_content = {}
    total_read = 0
    for fname, result in zip(to_read, results_list):
        if total_read >= max_total_size:
            skipped.append(fname)
            continue
        files_content[fname] = result
        total_read += result.get("size", 0)

    return {
        "tree": tree,
        "files": files_content,
        "stats": {
            "total_files_in_repo": len(all_paths),
            "files_read": len(files_content),
            "files_skipped": len(skipped),
            "total_bytes_read": total_read,
        },
    }


@mcp.tool()
async def hf_smart_edit(
    repo_id: str,
    filename: str,
    edits: list,
    repo_type: str = "space",
    commit_message: str = "",
) -> dict:
    """
    Intelligent single-file editor — applies multiple edits in one commit.
    Edit modes: replace | regex | insert_after | insert_before | delete_lines | overwrite
    Each edit: {"mode": "replace", "old": "...", "new": "..."}
    Regex: {"mode": "regex", "pattern": "...", "replacement": "...", "flags": "IGNORECASE|MULTILINE"}
    Edits applied sequentially. Cache invalidated after write.
    """
    read_only_guard()
    result = await fetch_file_http(repo_id, filename, repo_type, max_size=2_000_000)
    if "error" in result:
        return result
    original = result["content"]
    text, edit_log = _apply_edits(original, edits)
    if text == original:
        return {"status": "no_changes", "edit_log": edit_log}
    r = await run_in_thread(
        api.upload_file,
        path_or_fileobj=text.encode(),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Smart edit {filename}",
    )
    _cache_invalidate(repo_id, repo_type)
    return {"status": "ok", "uploaded": str(r), "edit_log": edit_log}


@mcp.tool()
async def hf_atomic_commit(
    repo_id: str,
    files: list,
    commit_message: str,
    repo_type: str = "space",
    create_pr: bool = False,
    pr_description: str = "",
) -> dict:
    """
    Commit multiple file changes atomically in ONE HF commit. Preferred for multi-file changes.
    ops: add {op,path,content} | delete {op,path} | rename {op,from,to} | copy {op,from,to}
         edit {op,path,edits:[...]} — same edit format as hf_smart_edit
    "edit" ops fetch files IN PARALLEL before building commit.
    Cache invalidated after commit.
    """
    read_only_guard()

    # Collect all "edit" ops that need file fetching, fetch in parallel
    edit_items = [(i, item) for i, item in enumerate(files) if item.get("op") == "edit"]
    if edit_items:
        fetch_tasks = [
            fetch_file_http(repo_id, item["path"], repo_type, max_size=2_000_000)
            for _, item in edit_items
        ]
        fetched = await asyncio.gather(*fetch_tasks)
    else:
        fetched = []

    fetched_map = {item["path"]: res for (_, item), res in zip(edit_items, fetched)}

    operations = []
    edit_logs = {}

    for item in files:
        op = item.get("op", "add")
        if op == "add":
            operations.append(CommitOperationAdd(
                path_in_repo=item["path"],
                path_or_fileobj=item["content"].encode(),
            ))
        elif op == "delete":
            operations.append(CommitOperationDelete(path_in_repo=item["path"]))
        elif op == "rename":
            operations.append(CommitOperationCopy(
                src_path_in_repo=item["from"], dest_path_in_repo=item["to"]))
            operations.append(CommitOperationDelete(path_in_repo=item["from"]))
        elif op == "copy":
            operations.append(CommitOperationCopy(
                src_path_in_repo=item["from"], dest_path_in_repo=item["to"]))
        elif op == "edit":
            fname = item["path"]
            fetch_result = fetched_map.get(fname, {})
            if "error" in fetch_result:
                edit_logs[fname] = {"error": fetch_result}
                continue
            text, log_entries = _apply_edits(fetch_result["content"], item.get("edits", []))
            edit_logs[fname] = log_entries
            operations.append(CommitOperationAdd(
                path_in_repo=fname, path_or_fileobj=text.encode()))

    if not operations:
        return {"error": "No valid operations", "edit_logs": edit_logs}

    extra = {"pr_description": pr_description} if create_pr and pr_description else {}
    r = await run_in_thread(
        api.create_commit,
        repo_id=repo_id, repo_type=repo_type,
        operations=operations, commit_message=commit_message,
        create_pr=create_pr, **extra,
    )
    _cache_invalidate(repo_id, repo_type)
    if isinstance(r, dict):
        return r
    return {"status": "ok", "commit": str(r), "operations": len(operations), "edit_logs": edit_logs}


@mcp.tool()
def hf_repository_manager(
    action: str, repo_id: str, repo_type: str = "model",
    private: bool = False, description: str = "", space_sdk: str = "",
) -> dict:
    """Manage HF repositories. action: create | delete | info | list_files"""
    if action == "info":
        r = safe_run(api.repo_info, repo_id=repo_id, repo_type=repo_type)
        return r if isinstance(r, dict) else r.__dict__
    if action == "list_files":
        files = safe_run(api.list_repo_files, repo_id=repo_id, repo_type=repo_type)
        return {"files": list(files) if not isinstance(files, dict) else files}
    if action == "create":
        read_only_guard()
        r = safe_run(api.create_repo, repo_id=repo_id, repo_type=repo_type,
                     private=private, space_sdk=space_sdk or None, exist_ok=True)
        return {"created": str(r)}
    if action == "delete":
        read_only_guard(); admin_guard()
        safe_run(api.delete_repo, repo_id=repo_id, repo_type=repo_type)
        return {"deleted": repo_id}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
async def hf_file_operations(
    action: str, repo_id: str, filename: str = "", repo_type: str = "space",
    content: str = "", commit_message: str = "", old_text: str = "",
    new_text: str = "", max_size: int = 500_000,
) -> dict:
    """Single-file CRUD. action: read | write | edit | delete | validate | backup"""
    if action == "read":
        return await fetch_file_http(repo_id, filename, repo_type, max_size)
    if action == "write":
        read_only_guard()
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=content.encode(), path_in_repo=filename,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Upload {filename}")
        _cache_invalidate(repo_id, repo_type)
        return {"uploaded": str(r)}
    if action == "edit":
        read_only_guard()
        result = await fetch_file_http(repo_id, filename, repo_type, max_size=2_000_000)
        if "error" in result: return result
        original = result["content"]
        if old_text not in original:
            return {"error": "old_text not found in file"}
        updated = original.replace(old_text, new_text)
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=updated.encode(), path_in_repo=filename,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Edit {filename}")
        _cache_invalidate(repo_id, repo_type)
        return {"edited": str(r)}
    if action == "delete":
        read_only_guard()
        r = await run_in_thread(api.delete_file,
            path_in_repo=filename, repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Delete {filename}")
        _cache_invalidate(repo_id, repo_type)
        return {"deleted": str(r)}
    if action == "validate":
        result = await fetch_file_http(repo_id, filename, repo_type)
        if "error" in result: return result
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "text"
        try:
            if ext == "json": json.loads(result["content"])
            return {"valid": True, "format": ext, "size": result.get("size", 0)}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    if action == "backup":
        read_only_guard()
        result = await fetch_file_http(repo_id, filename, repo_type, max_size=2_000_000)
        if "error" in result: return result
        bak = f"{filename}.backup"
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=result["content"].encode(), path_in_repo=bak,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=f"Backup {filename}")
        return {"backup_created": bak, "result": str(r)}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
def hf_search_hub(content_type: str, query: str = "", author: str = "",
                  filter_tag: str = "", limit: int = 20) -> dict:
    """Search HF Hub. content_type: models | datasets | spaces"""
    try:
        kw = dict(search=query or None, author=author or None,
                  filter=filter_tag or None, limit=limit)
        if content_type == "models": items = api.list_models(**kw)
        elif content_type == "datasets": items = api.list_datasets(**kw)
        else: items = api.list_spaces(**kw)
        return {"results": [{"id": getattr(i, "id", None) or getattr(i, "modelId", None),
                             "author": getattr(i, "author", None),
                             "likes": getattr(i, "likes", None),
                             "tags": getattr(i, "tags", [])} for i in items]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def hf_space_management(action: str, space_id: str, to_id: str = "",
                        sleep_time: int = 300) -> dict:
    """Manage HF Spaces. action: runtime_info|restart|pause|set_sleep_time|duplicate"""
    if action == "runtime_info":
        r = safe_run(api.get_space_runtime, repo_id=space_id)
        return r if isinstance(r, dict) else r.__dict__
    if action == "restart":
        read_only_guard(); return {"restarted": str(safe_run(api.restart_space, repo_id=space_id))}
    if action == "pause":
        read_only_guard(); return {"paused": str(safe_run(api.pause_space, repo_id=space_id))}
    if action == "set_sleep_time":
        read_only_guard()
        return {"sleep_time_set": str(safe_run(api.set_space_sleep_time,
            repo_id=space_id, sleep_time=sleep_time))}
    if action == "duplicate":
        read_only_guard()
        return {"duplicated_to": to_id, "result": str(safe_run(api.duplicate_space,
            from_id=space_id, to_id=to_id, exist_ok=True))}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
def hf_community_features(action: str, repo_id: str = "", repo_type: str = "model",
                          title: str = "", description: str = "") -> dict:
    """Community. action: like|unlike|get_likes|create_discussion|get_commits|get_refs"""
    if action == "like":
        read_only_guard(); safe_run(api.like, repo_id=repo_id, repo_type=repo_type)
        return {"liked": repo_id}
    if action == "unlike":
        read_only_guard(); safe_run(api.unlike, repo_id=repo_id, repo_type=repo_type)
        return {"unliked": repo_id}
    if action == "get_likes":
        return {"liked_repos": [str(x) for x in (safe_run(api.list_liked_repos) or [])]}
    if action == "create_discussion":
        read_only_guard()
        r = safe_run(api.create_discussion, repo_id=repo_id, repo_type=repo_type,
                     title=title or "New Discussion", description=description, pull_request=False)
        return {"discussion": str(r)}
    if action == "get_commits":
        r = safe_run(api.list_repo_commits, repo_id=repo_id, repo_type=repo_type)
        if isinstance(r, dict): return r
        return {"commits": [{"id": c.commit_id, "message": c.title} for c in list(r)[:50]]}
    if action == "get_refs":
        r = safe_run(api.list_repo_refs, repo_id=repo_id, repo_type=repo_type)
        return r if isinstance(r, dict) else r.__dict__
    return {"error": f"Unknown action: {action}"}


# ── ASGI middleware ────────────────────────────────────────────────────────────

class FixHeadersMiddleware:
    def __init__(self, app: ASGIApp, port: int = 8000):
        self.app = app
        self._localhost = f"localhost:{port}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/health", "/health/"):
                await JSONResponse({"status": "ok", "version": "5.0.0",
                                    "cache_entries": len(_file_cache)})(scope, receive, send)
                return
            headers = []
            for name, value in scope.get("headers", []):
                if name.lower() == b"host":
                    headers.append((b"host", self._localhost))
                elif name.lower() == b"origin":
                    headers.append((b"origin", b"http://" + self._localhost))
                else:
                    headers.append((name, value))
            scope["headers"] = headers
        await self.app(scope, receive, send)


_mcp_asgi = mcp.streamable_http_app()
app = FixHeadersMiddleware(_mcp_asgi, port=MCP_PORT)

if __name__ == "__main__":
    log.info(f"Starting HF MCP Server v5.0.0 on {MCP_HOST}:{MCP_PORT}")
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
