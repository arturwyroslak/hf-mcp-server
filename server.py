#!/usr/bin/env python3
"""
HuggingFace MCP Server — Streamable HTTP transport
Endpoint: http://0.0.0.0:8000/mcp
"""

import os
import json
import fnmatch

from mcp.server.fastmcp import FastMCP
from huggingface_hub import (
    HfApi,
    InferenceClient,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_ADMIN_MODE = os.environ.get("HF_ADMIN_MODE", "false").lower() == "true"
HF_MAX_FILE_SIZE = int(os.environ.get("HF_MAX_FILE_SIZE", str(100 * 1024 * 1024)))
HF_INFERENCE_TIMEOUT = int(os.environ.get("HF_INFERENCE_TIMEOUT", "30"))
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")

api = HfApi(token=HF_TOKEN or None)
mcp = FastMCP("hf-mcp-server", stateless_http=True)


def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def read_only_guard():
    if HF_READ_ONLY:
        raise PermissionError("READ-ONLY mode active. Set HF_READ_ONLY=false to enable writes.")


def admin_guard():
    if not HF_ADMIN_MODE:
        raise PermissionError("Admin action blocked. Set HF_ADMIN_MODE=true to enable.")


# ── 1. SYSTEM INFO ────────────────────────────────────────
@mcp.tool()
def hf_system_info() -> dict:
    """Return server status, configuration, and HF connectivity test."""
    whoami = safe_run(api.whoami) if HF_TOKEN else None
    return {
        "server": "hf-mcp-server",
        "version": "2.0.0-http",
        "transport": "streamable-http",
        "endpoint": f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}",
        "read_only": HF_READ_ONLY,
        "admin_mode": HF_ADMIN_MODE,
        "token_set": bool(HF_TOKEN),
        "max_file_size_bytes": HF_MAX_FILE_SIZE,
        "whoami": whoami,
    }


# ── 2. REPOSITORY MANAGER ─────────────────────────────────
@mcp.tool()
def hf_repository_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    private: bool = False,
    description: str = "",
    space_sdk: str = "",
) -> dict:
    """
    Manage HF repositories.
    action: create | delete (admin) | info | list_files
    repo_type: model | dataset | space
    space_sdk: gradio | streamlit | docker | static (for spaces)
    """
    if action == "info":
        r = safe_run(api.repo_info, repo_id=repo_id, repo_type=repo_type)
        return r if isinstance(r, dict) else r.__dict__

    if action == "list_files":
        files = safe_run(api.list_repo_files, repo_id=repo_id, repo_type=repo_type)
        return {"files": list(files) if not isinstance(files, dict) else files}

    if action == "create":
        read_only_guard()
        r = safe_run(
            api.create_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            space_sdk=space_sdk or None,
            exist_ok=True,
        )
        return {"created": str(r)}

    if action == "delete":
        read_only_guard()
        admin_guard()
        safe_run(api.delete_repo, repo_id=repo_id, repo_type=repo_type)
        return {"deleted": repo_id}

    return {"error": f"Unknown action: {action}"}


# ── 3. FILE OPERATIONS ────────────────────────────────────
@mcp.tool()
def hf_file_operations(
    action: str,
    repo_id: str,
    filename: str = "",
    repo_type: str = "model",
    content: str = "",
    commit_message: str = "",
    old_text: str = "",
    new_text: str = "",
    max_size: int = 500000,
    chunk_size: int = 0,
    chunk_number: int = 0,
    pattern: str = "",
    replacement: str = "",
    file_patterns: list = None,
) -> dict:
    """
    Full CRUD on files in HF repos.
    action: read | write | edit | delete | validate | backup | batch_edit
    """
    if file_patterns is None:
        file_patterns = ["*.md"]

    if action == "read":
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            if chunk_size:
                start = chunk_number * chunk_size
                chunk = text[start: start + chunk_size]
                return {"content": chunk, "chunk_number": chunk_number, "more": start + chunk_size < len(text)}
            return {"content": text[:max_size], "truncated": len(text) > max_size}
        except Exception as e:
            return {"error": str(e)}

    if action == "write":
        read_only_guard()
        r = safe_run(
            api.upload_file,
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {filename}",
        )
        return {"uploaded": str(r)}

    if action == "edit":
        read_only_guard()
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
            if old_text not in original:
                return {"error": "old_text not found in file"}
            updated = original.replace(old_text, new_text, 1)
            r = safe_run(
                api.upload_file,
                path_or_fileobj=updated.encode("utf-8"),
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
        r = safe_run(
            api.delete_file,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Delete {filename}",
        )
        return {"deleted": str(r)}

    if action == "validate":
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
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
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            bak = f"{filename}.backup"
            r = safe_run(
                api.upload_file,
                path_or_fileobj=raw.encode("utf-8"),
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
            files = list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
            matched = [f for f in files if any(fnmatch.fnmatch(f, p) for p in file_patterns)]
            results = []
            for fname in matched:
                local = api.hf_hub_download(repo_id=repo_id, filename=fname, repo_type=repo_type)
                with open(local, "r", encoding="utf-8", errors="replace") as fh:
                    original = fh.read()
                if pattern in original:
                    updated = original.replace(pattern, replacement)
                    api.upload_file(
                        path_or_fileobj=updated.encode("utf-8"),
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


# ── 4. SEARCH HUB ─────────────────────────────────────────
@mcp.tool()
def hf_search_hub(
    content_type: str,
    query: str = "",
    author: str = "",
    filter_tag: str = "",
    limit: int = 20,
) -> dict:
    """Search HF Hub for models, datasets, or spaces."""
    try:
        kw = dict(search=query or None, author=author or None, filter=filter_tag or None, limit=limit)
        if content_type == "models":
            items = api.list_models(**kw)
        elif content_type == "datasets":
            items = api.list_datasets(**kw)
        else:
            items = api.list_spaces(**kw)
        return {"results": [{"id": getattr(i, "id", None) or getattr(i, "modelId", None), "author": getattr(i, "author", None), "downloads": getattr(i, "downloads", None), "likes": getattr(i, "likes", None), "tags": getattr(i, "tags", [])} for i in items]}
    except Exception as e:
        return {"error": str(e)}


# ── 5. COLLECTIONS ────────────────────────────────────────
@mcp.tool()
def hf_collections(
    action: str,
    title: str = "",
    namespace: str = "",
    description: str = "",
    private: bool = False,
    collection_slug: str = "",
    item_id: str = "",
    item_type: str = "model",
    note: str = "",
) -> dict:
    """Manage HF Collections. action: create | add_item | info"""
    if action == "create":
        read_only_guard()
        r = safe_run(api.create_collection, title=title, namespace=namespace or None, description=description, private=private, exists_ok=True)
        return {"slug": getattr(r, "slug", str(r))}
    if action == "add_item":
        read_only_guard()
        r = safe_run(api.add_collection_item, collection_slug=collection_slug, item_id=item_id, item_type=item_type, note=note, exists_ok=True)
        return {"added": str(r)}
    if action == "info":
        r = safe_run(api.get_collection, collection_slug=collection_slug)
        return r if isinstance(r, dict) else r.__dict__
    return {"error": f"Unknown action: {action}"}


# ── 6. PULL REQUESTS ──────────────────────────────────────
@mcp.tool()
def hf_pull_requests(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    title: str = "",
    description: str = "",
    status: str = "open",
    author: str = "",
    pr_number: int = 0,
    files: list = None,
    commit_message: str = "",
    pr_title: str = "",
    pr_description: str = "",
) -> dict:
    """Manage HF Pull Requests. action: create | list | details | create_with_files"""
    files = files or []
    if action == "create":
        read_only_guard()
        r = safe_run(api.create_discussion, repo_id=repo_id, repo_type=repo_type, title=title or "New PR", description=description, pull_request=True)
        return {"pr": str(r)}
    if action == "list":
        discussions = safe_run(api.get_repo_discussions, repo_id=repo_id, repo_type=repo_type)
        if isinstance(discussions, dict):
            return discussions
        prs = [{"num": d.num, "title": d.title, "status": d.status, "author": d.author} for d in discussions if getattr(d, "is_pull_request", False) and (status == "all" or getattr(d, "status", None) == status) and (not author or getattr(d, "author", None) == author)]
        return {"pull_requests": prs}
    if action == "details":
        r = safe_run(api.get_discussion_details, repo_id=repo_id, repo_type=repo_type, discussion_num=pr_number)
        return r if isinstance(r, dict) else r.__dict__
    if action == "create_with_files":
        read_only_guard()
        ops = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=f["content"].encode("utf-8")) for f in files]
        r = safe_run(api.create_commit, repo_id=repo_id, repo_type=repo_type, operations=ops, commit_message=commit_message or "Update files", create_pr=True, pr_description=pr_description)
        return {"pr": str(r), "title": pr_title}
    return {"error": f"Unknown action: {action}"}


# ── 7. UPLOAD MANAGER ─────────────────────────────────────
@mcp.tool()
def hf_upload_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    file_path: str = "",
    content: str = "",
    commit_message: str = "",
    files: list = None,
    pr_title: str = "",
    pr_description: str = "",
) -> dict:
    """Upload files to HF. action: single_file | multiple_files | with_pr"""
    files = files or []
    read_only_guard()
    if action == "single_file":
        r = safe_run(api.upload_file, path_or_fileobj=content.encode("utf-8"), path_in_repo=file_path, repo_id=repo_id, repo_type=repo_type, commit_message=commit_message or f"Upload {file_path}")
        return {"uploaded": str(r)}
    if action == "multiple_files":
        ops = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=f["content"].encode("utf-8")) for f in files]
        r = safe_run(api.create_commit, repo_id=repo_id, repo_type=repo_type, operations=ops, commit_message=commit_message or "Upload multiple files")
        return {"commit": str(r)}
    if action == "with_pr":
        ops = [CommitOperationAdd(path_in_repo=file_path, path_or_fileobj=content.encode("utf-8"))]
        r = safe_run(api.create_commit, repo_id=repo_id, repo_type=repo_type, operations=ops, commit_message=commit_message or f"Upload {file_path}", create_pr=True, pr_description=pr_description)
        return {"pr": str(r), "title": pr_title}
    return {"error": f"Unknown action: {action}"}


# ── 8. BATCH OPERATIONS ───────────────────────────────────
@mcp.tool()
def hf_batch_operations(operation_type: str, operations: list) -> dict:
    """Batch ops: search | info | files across multiple repos."""
    results = []
    for op in operations:
        if operation_type == "search":
            results.append(hf_search_hub(**op))
        elif operation_type == "info":
            results.append(hf_repository_manager(action="info", **op))
        elif operation_type == "files":
            results.append(hf_repository_manager(action="list_files", **op))
        else:
            results.append({"error": f"Unknown operation_type: {operation_type}"})
    return {"results": results}


# ── 9. SPACE MANAGEMENT ───────────────────────────────────
@mcp.tool()
def hf_space_management(
    action: str,
    space_id: str,
    to_id: str = "",
    sleep_time: int = 300,
) -> dict:
    """Manage HF Spaces. action: runtime_info | restart | pause | set_sleep_time | duplicate"""
    if action == "runtime_info":
        r = safe_run(api.get_space_runtime, repo_id=space_id)
        return r if isinstance(r, dict) else r.__dict__
    if action == "restart":
        read_only_guard()
        return {"restarted": str(safe_run(api.restart_space, repo_id=space_id))}
    if action == "pause":
        read_only_guard()
        return {"paused": str(safe_run(api.pause_space, repo_id=space_id))}
    if action == "set_sleep_time":
        read_only_guard()
        return {"sleep_time_set": str(safe_run(api.set_space_sleep_time, repo_id=space_id, sleep_time=sleep_time))}
    if action == "duplicate":
        read_only_guard()
        return {"duplicated_to": to_id, "result": str(safe_run(api.duplicate_space, from_id=space_id, to_id=to_id, exist_ok=True))}
    return {"error": f"Unknown action: {action}"}


# ── 10. COMMUNITY FEATURES ────────────────────────────────
@mcp.tool()
def hf_community_features(
    action: str,
    repo_id: str = "",
    repo_type: str = "model",
    title: str = "",
    description: str = "",
) -> dict:
    """Community features. action: like | unlike | get_likes | create_discussion | get_commits | get_refs"""
    if action == "like":
        read_only_guard()
        safe_run(api.like, repo_id=repo_id, repo_type=repo_type)
        return {"liked": repo_id}
    if action == "unlike":
        read_only_guard()
        safe_run(api.unlike, repo_id=repo_id, repo_type=repo_type)
        return {"unliked": repo_id}
    if action == "get_likes":
        r = safe_run(api.list_liked_repos)
        return {"liked_repos": [str(x) for x in (r or [])]}
    if action == "create_discussion":
        read_only_guard()
        r = safe_run(api.create_discussion, repo_id=repo_id, repo_type=repo_type, title=title or "New Discussion", description=description, pull_request=False)
        return {"discussion": str(r)}
    if action == "get_commits":
        r = safe_run(api.list_repo_commits, repo_id=repo_id, repo_type=repo_type)
        if isinstance(r, dict):
            return r
        return {"commits": [{"id": c.commit_id, "message": c.title} for c in list(r)[:50]]}
    if action == "get_refs":
        r = safe_run(api.list_repo_refs, repo_id=repo_id, repo_type=repo_type)
        return r if isinstance(r, dict) else r.__dict__
    return {"error": f"Unknown action: {action}"}


# ── 11. INFERENCE TOOLS ───────────────────────────────────
@mcp.tool()
def hf_inference_tools(
    action: str,
    repo_id: str,
    inputs: list = None,
    parameters: dict = None,
) -> dict:
    """Test model inference. action: check_endpoints | test_inference"""
    inputs = inputs or ["Hello"]
    parameters = parameters or {}
    if action == "check_endpoints":
        info = safe_run(api.repo_info, repo_id=repo_id, repo_type="model")
        if isinstance(info, dict):
            return info
        return {"repo_id": repo_id, "pipeline_tag": getattr(info, "pipeline_tag", None), "library_name": getattr(info, "library_name", None), "tags": getattr(info, "tags", [])}
    if action == "test_inference":
        try:
            client = InferenceClient(model=repo_id, token=HF_TOKEN or None, timeout=HF_INFERENCE_TIMEOUT)
            results = []
            for inp in inputs:
                try:
                    out = client.text_generation(inp, **parameters)
                    results.append({"input": inp, "output": str(out)})
                except Exception as e:
                    results.append({"input": inp, "error": str(e)})
            return {"inference_results": results}
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Unknown action: {action}"}


# ── 12. REPO FILE MANAGER ─────────────────────────────────
@mcp.tool()
def hf_repo_file_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    filename: str = "",
    new_filename: str = "",
    content: str = "",
    old_text: str = "",
    new_text: str = "",
    commit_message: str = "",
    private: bool = False,
    description: str = "",
    space_sdk: str = "",
) -> dict:
    """
    Unified repo + file manager with rename support.
    action: repo_create | repo_delete | repo_info | list_files
            file_read | file_write | file_edit | file_delete | file_rename
    """
    if action == "repo_create":
        return hf_repository_manager("create", repo_id, repo_type, private, description, space_sdk)
    if action == "repo_delete":
        return hf_repository_manager("delete", repo_id, repo_type)
    if action == "repo_info":
        return hf_repository_manager("info", repo_id, repo_type)
    if action == "list_files":
        return hf_repository_manager("list_files", repo_id, repo_type)
    if action == "file_read":
        return hf_file_operations("read", repo_id, filename=filename, repo_type=repo_type)
    if action == "file_write":
        return hf_file_operations("write", repo_id, filename=filename, repo_type=repo_type, content=content, commit_message=commit_message)
    if action == "file_edit":
        return hf_file_operations("edit", repo_id, filename=filename, repo_type=repo_type, old_text=old_text, new_text=new_text, commit_message=commit_message)
    if action == "file_delete":
        return hf_file_operations("delete", repo_id, filename=filename, repo_type=repo_type, commit_message=commit_message)
    if action == "file_rename":
        read_only_guard()
        ops = [
            CommitOperationCopy(src_path_in_repo=filename, dest_path_in_repo=new_filename),
            CommitOperationDelete(path_in_repo=filename),
        ]
        r = safe_run(api.create_commit, repo_id=repo_id, repo_type=repo_type, operations=ops, commit_message=commit_message or f"Rename {filename} -> {new_filename}")
        return {"renamed": f"{filename} -> {new_filename}", "result": str(r)}
    return {"error": f"Unknown action: {action}"}


# ── ENTRYPOINT ────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("hf-mcp-server").info(
        f"Starting HF MCP Server | HTTP transport | {MCP_HOST}:{MCP_PORT}{MCP_PATH}"
    )
    mcp.run(transport="http", host=MCP_HOST, port=MCP_PORT, path=MCP_PATH)
