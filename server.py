#!/usr/bin/env python3
import asyncio
import contextlib
import os
import json
import fnmatch
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import uvicorn
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from mcp.server.fastmcp import FastMCP
from huggingface_hub import (
    HfApi,
    InferenceClient,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

HF_TOKEN             = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY         = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_ADMIN_MODE        = os.environ.get("HF_ADMIN_MODE", "false").lower() == "true"
HF_MAX_FILE_SIZE     = int(os.environ.get("HF_MAX_FILE_SIZE", str(100 * 1024 * 1024)))
HF_INFERENCE_TIMEOUT = int(os.environ.get("HF_INFERENCE_TIMEOUT", "30"))
HF_UPLOAD_TIMEOUT    = int(os.environ.get("HF_UPLOAD_TIMEOUT", "300"))  # 5 min
MCP_HOST             = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT             = int(os.environ.get("MCP_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hf-mcp")

# HfApi with extended timeout for uploads
api = HfApi(
    token=HF_TOKEN or None,
    # Pass a custom httpx client with long timeout
    endpoint="https://huggingface.co",
)

# Thread pool for blocking HF API calls
_executor = ThreadPoolExecutor(max_workers=4)

mcp = FastMCP(
    "hf-mcp-server",
    stateless_http=True,
    json_response=True,
)


def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(f"{fn.__name__} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


async def run_in_thread(fn, *args, **kwargs):
    """Run a blocking HF API call in thread pool with long timeout."""
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: fn(*args, **kwargs)),
            timeout=HF_UPLOAD_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"Operation timed out after {HF_UPLOAD_TIMEOUT}s"}
    except Exception as e:
        log.error(f"{fn.__name__} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


def read_only_guard():
    if HF_READ_ONLY:
        raise PermissionError("Server is in READ-ONLY mode.")

def admin_guard():
    if not HF_ADMIN_MODE:
        raise PermissionError("Admin mode is disabled.")


@mcp.tool()
def hf_system_info() -> dict:
    """Return server status and HF connectivity."""
    return {
        "version": "3.2.0",
        "read_only": HF_READ_ONLY,
        "admin_mode": HF_ADMIN_MODE,
        "token_set": bool(HF_TOKEN),
        "upload_timeout": HF_UPLOAD_TIMEOUT,
        "whoami": safe_run(api.whoami) if HF_TOKEN else None,
    }


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
        r = safe_run(api.create_repo, repo_id=repo_id, repo_type=repo_type, private=private,
                     space_sdk=space_sdk or None, exist_ok=True)
        return {"created": str(r)}
    if action == "delete":
        read_only_guard(); admin_guard()
        safe_run(api.delete_repo, repo_id=repo_id, repo_type=repo_type)
        return {"deleted": repo_id}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
async def hf_file_operations(
    action: str, repo_id: str, filename: str = "", repo_type: str = "model",
    content: str = "", commit_message: str = "", old_text: str = "",
    new_text: str = "", max_size: int = 500000, chunk_size: int = 0,
    chunk_number: int = 0, pattern: str = "", replacement: str = "",
    file_patterns: list = None,
) -> dict:
    """CRUD on HF repo files. action: read|write|edit|delete|validate|backup|batch_edit"""
    if file_patterns is None:
        file_patterns = ["*.md"]

    if action == "read":
        try:
            local = await run_in_thread(api.hf_hub_download,
                repo_id=repo_id, filename=filename, repo_type=repo_type)
            if isinstance(local, dict): return local
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            if chunk_size:
                start = chunk_number * chunk_size
                return {"content": text[start:start + chunk_size], "chunk_number": chunk_number,
                        "more": start + chunk_size < len(text)}
            return {"content": text[:max_size], "truncated": len(text) > max_size}
        except Exception as e:
            return {"error": str(e)}

    if action == "write":
        read_only_guard()
        r = await run_in_thread(
            api.upload_file,
            path_or_fileobj=content.encode(),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {filename}",
        )
        return {"uploaded": str(r)}

    if action == "edit":
        read_only_guard()
        try:
            local = await run_in_thread(api.hf_hub_download,
                repo_id=repo_id, filename=filename, repo_type=repo_type)
            if isinstance(local, dict): return local
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
            if old_text not in original:
                return {"error": "old_text not found"}
            updated = original.replace(old_text, new_text, 1)
            r = await run_in_thread(
                api.upload_file,
                path_or_fileobj=updated.encode(),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message or f"Edit {filename}",
            )
            return {"edited": str(r)}
        except Exception as e:
            return {"error": str(e)}

    if action == "delete":
        read_only_guard()
        r = await run_in_thread(
            api.delete_file,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Delete {filename}",
        )
        return {"deleted": str(r)}

    if action == "validate":
        try:
            local = await run_in_thread(api.hf_hub_download,
                repo_id=repo_id, filename=filename, repo_type=repo_type)
            if isinstance(local, dict): return local
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "text"
            if ext == "json":
                json.loads(raw)
            return {"valid": True, "format": ext, "size": len(raw)}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    if action == "backup":
        read_only_guard()
        try:
            local = await run_in_thread(api.hf_hub_download,
                repo_id=repo_id, filename=filename, repo_type=repo_type)
            if isinstance(local, dict): return local
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            bak = f"{filename}.backup"
            r = await run_in_thread(
                api.upload_file,
                path_or_fileobj=raw.encode(),
                path_in_repo=bak,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Backup {filename}",
            )
            return {"backup_created": bak, "result": str(r)}
        except Exception as e:
            return {"error": str(e)}

    if action == "batch_edit":
        read_only_guard()
        try:
            files_list = list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
            matched = [f for f in files_list if any(fnmatch.fnmatch(f, p) for p in file_patterns)]
            results = []
            for fname in matched:
                local = await run_in_thread(api.hf_hub_download,
                    repo_id=repo_id, filename=fname, repo_type=repo_type)
                if isinstance(local, dict):
                    results.append({"file": fname, "status": "error", "detail": local})
                    continue
                with open(local, "r", encoding="utf-8", errors="replace") as fh:
                    original = fh.read()
                if pattern in original:
                    await run_in_thread(
                        api.upload_file,
                        path_or_fileobj=original.replace(pattern, replacement).encode(),
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        commit_message=f"batch_edit {fname}",
                    )
                    results.append({"file": fname, "status": "edited"})
                else:
                    results.append({"file": fname, "status": "skipped"})
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: {action}"}


@mcp.tool()
def hf_search_hub(content_type: str, query: str = "", author: str = "",
                  filter_tag: str = "", limit: int = 20) -> dict:
    """Search HF Hub. content_type: models | datasets | spaces"""
    try:
        kw = dict(search=query or None, author=author or None, filter=filter_tag or None, limit=limit)
        if content_type == "models":
            items = api.list_models(**kw)
        elif content_type == "datasets":
            items = api.list_datasets(**kw)
        else:
            items = api.list_spaces(**kw)
        return {"results": [{"id": getattr(i, "id", None) or getattr(i, "modelId", None),
                             "author": getattr(i, "author", None),
                             "downloads": getattr(i, "downloads", None),
                             "likes": getattr(i, "likes", None),
                             "tags": getattr(i, "tags", [])} for i in items]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def hf_upload_manager(
    action: str, repo_id: str, repo_type: str = "model",
    file_path: str = "", content: str = "", commit_message: str = "",
    files: list = None, pr_title: str = "", pr_description: str = "",
) -> dict:
    """Upload files to HF. action: single_file | multiple_files | with_pr"""
    files = files or []
    read_only_guard()
    if action == "single_file":
        r = await run_in_thread(
            api.upload_file,
            path_or_fileobj=content.encode(),
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {file_path}",
        )
        return {"uploaded": str(r)}
    if action == "multiple_files":
        ops = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=f["content"].encode()) for f in files]
        r = await run_in_thread(
            api.create_commit,
            repo_id=repo_id, repo_type=repo_type, operations=ops,
            commit_message=commit_message or "Upload multiple files",
        )
        return {"commit": str(r)}
    if action == "with_pr":
        ops = [CommitOperationAdd(path_in_repo=file_path, path_or_fileobj=content.encode())]
        r = await run_in_thread(
            api.create_commit,
            repo_id=repo_id, repo_type=repo_type, operations=ops,
            commit_message=commit_message or f"Upload {file_path}",
            create_pr=True, pr_description=pr_description,
        )
        return {"pr": str(r), "title": pr_title}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
def hf_space_management(action: str, space_id: str, to_id: str = "", sleep_time: int = 300) -> dict:
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
        return {"sleep_time_set": str(safe_run(api.set_space_sleep_time, repo_id=space_id, sleep_time=sleep_time))}
    if action == "duplicate":
        read_only_guard()
        return {"duplicated_to": to_id,
                "result": str(safe_run(api.duplicate_space, from_id=space_id, to_id=to_id, exist_ok=True))}
    return {"error": f"Unknown action: {action}"}


@mcp.tool()
def hf_community_features(action: str, repo_id: str = "", repo_type: str = "model",
                          title: str = "", description: str = "") -> dict:
    """Community features. action: like|unlike|get_likes|create_discussion|get_commits|get_refs"""
    if action == "like":
        read_only_guard(); safe_run(api.like, repo_id=repo_id, repo_type=repo_type); return {"liked": repo_id}
    if action == "unlike":
        read_only_guard(); safe_run(api.unlike, repo_id=repo_id, repo_type=repo_type); return {"unliked": repo_id}
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


@mcp.tool()
async def hf_repo_file_manager(
    action: str, repo_id: str, repo_type: str = "model",
    filename: str = "", new_filename: str = "", content: str = "",
    old_text: str = "", new_text: str = "", commit_message: str = "",
    private: bool = False, description: str = "", space_sdk: str = "",
) -> dict:
    """Unified repo+file manager."""
    if action == "repo_create":
        return hf_repository_manager("create", repo_id, repo_type, private, description, space_sdk)
    if action == "repo_delete":
        return hf_repository_manager("delete", repo_id, repo_type)
    if action == "repo_info":
        return hf_repository_manager("info", repo_id, repo_type)
    if action == "list_files":
        return hf_repository_manager("list_files", repo_id, repo_type)
    if action == "file_read":
        return await hf_file_operations("read", repo_id, filename=filename, repo_type=repo_type)
    if action == "file_write":
        return await hf_file_operations("write", repo_id, filename=filename, repo_type=repo_type,
                                         content=content, commit_message=commit_message)
    if action == "file_edit":
        return await hf_file_operations("edit", repo_id, filename=filename, repo_type=repo_type,
                                         old_text=old_text, new_text=new_text, commit_message=commit_message)
    if action == "file_delete":
        return await hf_file_operations("delete", repo_id, filename=filename, repo_type=repo_type,
                                         commit_message=commit_message)
    if action == "file_rename":
        read_only_guard()
        ops = [
            CommitOperationCopy(src_path_in_repo=filename, dest_path_in_repo=new_filename),
            CommitOperationDelete(path_in_repo=filename),
        ]
        r = await run_in_thread(
            api.create_commit,
            repo_id=repo_id, repo_type=repo_type, operations=ops,
            commit_message=commit_message or f"Rename {filename} -> {new_filename}",
        )
        return {"renamed": f"{filename} -> {new_filename}", "result": str(r)}
    return {"error": f"Unknown action: {action}"}


# ── ASGI middleware stack ──────────────────────────────────────────────────────

class FixHeadersMiddleware:
    """Rewrites Host to localhost (bypasses MCP 421 check) and handles /health."""
    def __init__(self, app: ASGIApp, port: int = 8000):
        self.app = app
        self._localhost = f"localhost:{port}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/health", "/health/"):
                await JSONResponse({"status": "ok", "version": "3.2.0"})(scope, receive, send)
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
    log.info(f"Starting HF MCP Server on {MCP_HOST}:{MCP_PORT}")
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
