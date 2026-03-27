#!/usr/bin/env python3
import asyncio
import os
import re
import json
import fnmatch
import logging
from concurrent.futures import ThreadPoolExecutor

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
HF_MAX_FILE_SIZE     = int(os.environ.get("HF_MAX_FILE_SIZE", str(100 * 1024 * 1024)))
HF_INFERENCE_TIMEOUT = int(os.environ.get("HF_INFERENCE_TIMEOUT", "30"))
HF_UPLOAD_TIMEOUT    = int(os.environ.get("HF_UPLOAD_TIMEOUT", "300"))
MCP_HOST             = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT             = int(os.environ.get("MCP_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hf-mcp")

api = HfApi(token=HF_TOKEN or None, endpoint="https://huggingface.co")
_executor = ThreadPoolExecutor(max_workers=8)

mcp = FastMCP("hf-mcp-server", stateless_http=True, json_response=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(f"{fn.__name__} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


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
        log.error(f"{fn.__name__} failed: {e}")
        return {"error": str(e), "type": type(e).__name__}


def _read_local(path: str, max_size: int = 500_000) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return {"content": text[:max_size], "size": len(text), "truncated": len(text) > max_size}
    except Exception as e:
        return {"error": str(e)}


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
        "version": "4.0.0",
        "read_only": HF_READ_ONLY,
        "admin_mode": HF_ADMIN_MODE,
        "token_set": bool(HF_TOKEN),
        "upload_timeout": HF_UPLOAD_TIMEOUT,
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
    Read multiple files from a HF repo in one call.
    Returns a dict {filename: {content, size, truncated} | {error}}.
    Perfect for getting full context of a Space before modifying it.
    """
    results = {}
    for fname in filenames:
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=fname, repo_type=repo_type)
        if isinstance(local, dict):
            results[fname] = local
        else:
            results[fname] = _read_local(local, max_size_per_file)
    return results


@mcp.tool()
async def hf_repo_snapshot(
    repo_id: str,
    repo_type: str = "space",
    include_patterns: list = None,
    exclude_patterns: list = None,
    max_size_per_file: int = 100_000,
    max_total_size: int = 1_000_000,
) -> dict:
    """
    Full repository snapshot — returns structure + content of all text files.
    include_patterns: e.g. ["*.py", "*.html"] — only include matching files.
    exclude_patterns: e.g. ["*.lock", "node_modules/*"] — skip matching files.
    Use this as the FIRST step before any modification to understand the full codebase.
    Returns:
      - tree: list of all files with sizes
      - files: {filename: content} for all readable text files
      - stats: total files, total size read, skipped
    """
    if include_patterns is None:
        include_patterns = []
    if exclude_patterns is None:
        exclude_patterns = ["*.lock", "*.png", "*.jpg", "*.jpeg", "*.gif",
                             "*.ico", "*.woff", "*.woff2", "*.ttf", "*.eot",
                             "*.zip", "*.tar", "*.gz", "*.bin", "*.pkl",
                             "*.pt", "*.pth", "*.ckpt", "*.safetensors",
                             ".git/*", "node_modules/*", "__pycache__/*"]

    # Get file list
    raw_files = safe_run(api.list_repo_tree, repo_id=repo_id, repo_type=repo_type, recursive=True)
    if isinstance(raw_files, dict) and "error" in raw_files:
        # Fallback to list_repo_files
        raw_files = safe_run(api.list_repo_files, repo_id=repo_id, repo_type=repo_type)
        if isinstance(raw_files, dict):
            return raw_files
        tree = [{"path": f, "size": None} for f in raw_files]
        all_paths = list(raw_files)
    else:
        tree = []
        all_paths = []
        for item in (raw_files or []):
            if hasattr(item, "path"):
                size = getattr(item, "size", None)
                tree.append({"path": item.path, "size": size,
                              "type": getattr(item, "type", "file")})
                if getattr(item, "type", "file") == "file":
                    all_paths.append(item.path)

    # Filter
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

    # Read files, stop at total size limit
    files_content = {}
    total_read = 0
    for fname in to_read:
        if total_read >= max_total_size:
            skipped.append(fname)
            continue
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=fname, repo_type=repo_type)
        if isinstance(local, dict):
            files_content[fname] = local
        else:
            result = _read_local(local, max_size_per_file)
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
    Intelligent file editor — applies multiple edits to a single file in one commit.
    Each edit in `edits` is a dict with one of these modes:

    1. Replace exact text:
       {"mode": "replace", "old": "exact old text", "new": "new text"}

    2. Regex substitution:
       {"mode": "regex", "pattern": "regex pattern", "replacement": "new text", "flags": "IGNORECASE"}
       flags: IGNORECASE, MULTILINE, DOTALL (combine with |)

    3. Insert after line containing pattern:
       {"mode": "insert_after", "after_pattern": "text to find", "insert": "lines to insert"}

    4. Insert before line containing pattern:
       {"mode": "insert_before", "before_pattern": "text to find", "insert": "lines to insert"}

    5. Replace entire file:
       {"mode": "overwrite", "content": "full new file content"}

    6. Delete lines matching pattern:
       {"mode": "delete_lines", "line_pattern": "pattern to match lines for deletion"}

    Edits are applied sequentially — the output of each edit is the input of the next.
    Returns the final content and a log of each edit result.
    """
    read_only_guard()

    # Download current version
    local = await run_in_thread(api.hf_hub_download,
        repo_id=repo_id, filename=filename, repo_type=repo_type)
    if isinstance(local, dict):
        return local

    with open(local, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    original_text = text
    edit_log = []

    for i, edit in enumerate(edits):
        mode = edit.get("mode", "replace")
        try:
            if mode == "overwrite":
                text = edit["content"]
                edit_log.append({"edit": i, "mode": mode, "status": "ok"})

            elif mode == "replace":
                old = edit["old"]
                new = edit["new"]
                if old not in text:
                    edit_log.append({"edit": i, "mode": mode, "status": "not_found",
                                     "detail": f"old text not found in file"})
                    continue
                count = text.count(old)
                text = text.replace(old, new)
                edit_log.append({"edit": i, "mode": mode, "status": "ok",
                                  "replacements": count})

            elif mode == "regex":
                pattern = edit["pattern"]
                replacement = edit["replacement"]
                flag_str = edit.get("flags", "")
                flags = 0
                for flag_name in flag_str.split("|"):
                    flag_name = flag_name.strip().upper()
                    if flag_name == "IGNORECASE": flags |= re.IGNORECASE
                    elif flag_name == "MULTILINE": flags |= re.MULTILINE
                    elif flag_name == "DOTALL": flags |= re.DOTALL
                new_text, count = re.subn(pattern, replacement, text, flags=flags)
                text = new_text
                edit_log.append({"edit": i, "mode": mode, "status": "ok",
                                  "substitutions": count})

            elif mode == "insert_after":
                after_pattern = edit["after_pattern"]
                insert = edit["insert"]
                lines = text.splitlines(keepends=True)
                new_lines = []
                found = False
                for line in lines:
                    new_lines.append(line)
                    if after_pattern in line:
                        new_lines.append(insert if insert.endswith("\n") else insert + "\n")
                        found = True
                if not found:
                    edit_log.append({"edit": i, "mode": mode, "status": "not_found"})
                    continue
                text = "".join(new_lines)
                edit_log.append({"edit": i, "mode": mode, "status": "ok"})

            elif mode == "insert_before":
                before_pattern = edit["before_pattern"]
                insert = edit["insert"]
                lines = text.splitlines(keepends=True)
                new_lines = []
                found = False
                for line in lines:
                    if before_pattern in line:
                        new_lines.append(insert if insert.endswith("\n") else insert + "\n")
                        found = True
                    new_lines.append(line)
                if not found:
                    edit_log.append({"edit": i, "mode": mode, "status": "not_found"})
                    continue
                text = "".join(new_lines)
                edit_log.append({"edit": i, "mode": mode, "status": "ok"})

            elif mode == "delete_lines":
                line_pattern = edit["line_pattern"]
                lines = text.splitlines(keepends=True)
                new_lines = [l for l in lines if line_pattern not in l]
                deleted = len(lines) - len(new_lines)
                text = "".join(new_lines)
                edit_log.append({"edit": i, "mode": mode, "status": "ok",
                                  "lines_deleted": deleted})

            else:
                edit_log.append({"edit": i, "mode": mode, "status": "unknown_mode"})

        except Exception as e:
            edit_log.append({"edit": i, "mode": mode, "status": "error", "detail": str(e)})

    if text == original_text:
        return {"status": "no_changes", "edit_log": edit_log}

    r = await run_in_thread(
        api.upload_file,
        path_or_fileobj=text.encode(),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Smart edit {filename}",
    )
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
    Commit multiple file changes atomically in a single HF commit.
    This is the preferred way to make multi-file changes — avoids partial states.

    `files` is a list of operation dicts:

    Add / overwrite a file:
      {"op": "add", "path": "app.py", "content": "file content as string"}

    Delete a file:
      {"op": "delete", "path": "old_file.py"}

    Rename / move a file:
      {"op": "rename", "from": "old.py", "to": "new.py"}

    Copy a file:
      {"op": "copy", "from": "src.py", "to": "dst.py"}

    Smart-edit a file (apply edits list, same format as hf_smart_edit):
      {"op": "edit", "path": "app.py", "edits": [{"mode": "replace", "old": "x", "new": "y"}]}

    Returns commit URL or PR URL if create_pr=True.
    """
    read_only_guard()

    operations = []
    edit_logs = {}

    for item in files:
        op = item.get("op", "add")

        if op == "add":
            content = item["content"]
            operations.append(
                CommitOperationAdd(
                    path_in_repo=item["path"],
                    path_or_fileobj=content.encode(),
                )
            )

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
            # Download, apply edits in memory, then add as modified content
            fname = item["path"]
            local = await run_in_thread(api.hf_hub_download,
                repo_id=repo_id, filename=fname, repo_type=repo_type)
            if isinstance(local, dict):
                edit_logs[fname] = {"error": local}
                continue
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            log_entries = []
            for i, edit in enumerate(item.get("edits", [])):
                mode = edit.get("mode", "replace")
                try:
                    if mode == "overwrite":
                        text = edit["content"]
                        log_entries.append({"edit": i, "mode": mode, "status": "ok"})
                    elif mode == "replace":
                        old = edit["old"]
                        if old not in text:
                            log_entries.append({"edit": i, "mode": mode, "status": "not_found"})
                            continue
                        text = text.replace(old, edit["new"])
                        log_entries.append({"edit": i, "mode": mode, "status": "ok"})
                    elif mode == "regex":
                        flags = 0
                        for flag_name in edit.get("flags", "").split("|"):
                            n = flag_name.strip().upper()
                            if n == "IGNORECASE": flags |= re.IGNORECASE
                            elif n == "MULTILINE": flags |= re.MULTILINE
                            elif n == "DOTALL": flags |= re.DOTALL
                        text, count = re.subn(edit["pattern"], edit["replacement"], text, flags=flags)
                        log_entries.append({"edit": i, "mode": mode, "status": "ok", "subs": count})
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
                        log_entries.append({"edit": i, "mode": mode,
                                            "status": "ok" if found else "not_found"})
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
                        log_entries.append({"edit": i, "mode": mode,
                                            "status": "ok" if found else "not_found"})
                    elif mode == "delete_lines":
                        lines = text.splitlines(keepends=True)
                        new_lines = [l for l in lines if edit["line_pattern"] not in l]
                        deleted = len(lines) - len(new_lines)
                        text = "".join(new_lines)
                        log_entries.append({"edit": i, "mode": mode, "status": "ok",
                                            "lines_deleted": deleted})
                except Exception as e:
                    log_entries.append({"edit": i, "mode": mode, "status": "error", "detail": str(e)})

            edit_logs[fname] = log_entries
            operations.append(
                CommitOperationAdd(path_in_repo=fname, path_or_fileobj=text.encode())
            )

    if not operations:
        return {"error": "No valid operations to commit", "edit_logs": edit_logs}

    r = await run_in_thread(
        api.create_commit,
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=commit_message,
        create_pr=create_pr,
        **(({"pr_description": pr_description}) if create_pr and pr_description else {}),
    )
    if isinstance(r, dict):
        return r
    return {
        "status": "ok",
        "commit": str(r),
        "operations": len(operations),
        "edit_logs": edit_logs,
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
    action: str, repo_id: str, filename: str = "", repo_type: str = "space",
    content: str = "", commit_message: str = "", old_text: str = "",
    new_text: str = "", max_size: int = 500_000,
) -> dict:
    """Single-file CRUD. action: read | write | edit | delete | validate | backup"""
    if action == "read":
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=filename, repo_type=repo_type)
        if isinstance(local, dict): return local
        return _read_local(local, max_size)

    if action == "write":
        read_only_guard()
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=content.encode(), path_in_repo=filename,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Upload {filename}")
        return {"uploaded": str(r)}

    if action == "edit":
        read_only_guard()
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=filename, repo_type=repo_type)
        if isinstance(local, dict): return local
        with open(local, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
        if old_text not in original:
            return {"error": "old_text not found in file"}
        updated = original.replace(old_text, new_text)
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=updated.encode(), path_in_repo=filename,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Edit {filename}")
        return {"edited": str(r)}

    if action == "delete":
        read_only_guard()
        r = await run_in_thread(api.delete_file,
            path_in_repo=filename, repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Delete {filename}")
        return {"deleted": str(r)}

    if action == "validate":
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=filename, repo_type=repo_type)
        if isinstance(local, dict): return local
        with open(local, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "text"
        try:
            if ext == "json": json.loads(raw)
            return {"valid": True, "format": ext, "size": len(raw)}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    if action == "backup":
        read_only_guard()
        local = await run_in_thread(api.hf_hub_download,
            repo_id=repo_id, filename=filename, repo_type=repo_type)
        if isinstance(local, dict): return local
        with open(local, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        bak = f"{filename}.backup"
        r = await run_in_thread(api.upload_file,
            path_or_fileobj=raw.encode(), path_in_repo=bak,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=f"Backup {filename}")
        return {"backup_created": bak, "result": str(r)}

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
        return {"sleep_time_set": str(safe_run(api.set_space_sleep_time,
            repo_id=space_id, sleep_time=sleep_time))}
    if action == "duplicate":
        read_only_guard()
        return {"duplicated_to": to_id,
                "result": str(safe_run(api.duplicate_space,
                    from_id=space_id, to_id=to_id, exist_ok=True))}
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


# ── ASGI middleware ────────────────────────────────────────────────────────────

class FixHeadersMiddleware:
    """Rewrites Host to localhost (bypasses MCP 421) and handles /health."""
    def __init__(self, app: ASGIApp, port: int = 8000):
        self.app = app
        self._localhost = f"localhost:{port}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/health", "/health/"):
                await JSONResponse({"status": "ok", "version": "4.0.0"})(scope, receive, send)
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
