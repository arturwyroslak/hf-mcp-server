#!/usr/bin/env python3
"""
HuggingFace MCP Server
Core tools: repository management, file operations, spaces, collections, pull requests, uploads, community features.
No analytics/AI workflow tools.
"""

import os
import sys
import json
import logging
import traceback
from typing import Any, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError as e:
    print(f"ERROR: mcp package not installed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
except ImportError as e:
    print(f"ERROR: huggingface_hub not installed: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("hf-mcp-server")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_READ_ONLY = os.environ.get("HF_READ_ONLY", "false").lower() == "true"
HF_ADMIN_MODE = os.environ.get("HF_ADMIN_MODE", "false").lower() == "true"
HF_MAX_FILE_SIZE = int(os.environ.get("HF_MAX_FILE_SIZE", str(100 * 1024 * 1024)))
HF_INFERENCE_TIMEOUT = int(os.environ.get("HF_INFERENCE_TIMEOUT", "30"))

api = HfApi(token=HF_TOKEN if HF_TOKEN else None)
server = Server("hf-mcp-server")


def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(traceback.format_exc())
        return {"error": str(e), "type": type(e).__name__}


def read_only_guard():
    if HF_READ_ONLY:
        raise PermissionError("Server is running in READ-ONLY mode. Set HF_READ_ONLY=false to enable writes.")


def admin_guard():
    if not HF_ADMIN_MODE:
        raise PermissionError("This action requires ADMIN mode. Set HF_ADMIN_MODE=true to enable.")


# ──────────────────────────────────────────
# TOOL DEFINITIONS
# ──────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="hf_system_info",
            description="Get server status, configuration summary and connectivity test.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="hf_repository_manager",
            description=(
                "Manage HuggingFace repositories. Actions: create, delete (admin), info, list_files.\n"
                "For Spaces use space_sdk: gradio | streamlit | docker | static."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "delete", "info", "list_files"]},
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "private": {"type": "boolean", "default": False},
                    "description": {"type": "string"},
                    "space_sdk": {"type": "string", "enum": ["gradio", "streamlit", "docker", "static"]},
                },
                "required": ["action", "repo_id"],
            },
        ),
        types.Tool(
            name="hf_file_operations",
            description=(
                "Full CRUD on files in any HF repository.\n"
                "Actions: read, write, edit (find&replace with auto-backup), delete, validate, backup, batch_edit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "edit", "delete", "validate", "backup", "batch_edit"],
                    },
                    "repo_id": {"type": "string"},
                    "filename": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "content": {"type": "string"},
                    "commit_message": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "max_size": {"type": "integer", "default": 500000},
                    "chunk_size": {"type": "integer"},
                    "chunk_number": {"type": "integer", "default": 0},
                    "pattern": {"type": "string"},
                    "replacement": {"type": "string"},
                    "file_patterns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["action", "repo_id"],
            },
        ),
        types.Tool(
            name="hf_search_hub",
            description="Search HuggingFace Hub for models, datasets, or spaces.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content_type": {"type": "string", "enum": ["models", "datasets", "spaces"]},
                    "query": {"type": "string"},
                    "author": {"type": "string"},
                    "filter_tag": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["content_type"],
            },
        ),
        types.Tool(
            name="hf_collections",
            description="Manage HuggingFace Collections. Actions: create, add_item, info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "add_item", "info"]},
                    "title": {"type": "string"},
                    "namespace": {"type": "string"},
                    "description": {"type": "string"},
                    "private": {"type": "boolean", "default": False},
                    "collection_slug": {"type": "string"},
                    "item_id": {"type": "string"},
                    "item_type": {"type": "string", "enum": ["model", "dataset", "space"]},
                    "note": {"type": "string"},
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="hf_pull_requests",
            description=(
                "Manage Pull Requests on HuggingFace Hub.\n"
                "Actions: create, list, details, create_with_files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "list", "details", "create_with_files"]},
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                    "author": {"type": "string"},
                    "pr_number": {"type": "integer"},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        },
                    },
                    "commit_message": {"type": "string"},
                    "pr_title": {"type": "string"},
                    "pr_description": {"type": "string"},
                },
                "required": ["action", "repo_id"],
            },
        ),
        types.Tool(
            name="hf_upload_manager",
            description=(
                "Upload files to HuggingFace repositories.\n"
                "Actions: single_file, multiple_files, with_pr."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["single_file", "multiple_files", "with_pr"]},
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "commit_message": {"type": "string"},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        },
                    },
                    "pr_title": {"type": "string"},
                    "pr_description": {"type": "string"},
                },
                "required": ["action", "repo_id"],
            },
        ),
        types.Tool(
            name="hf_batch_operations",
            description="Execute batch operations: search, info, files across multiple repos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation_type": {"type": "string", "enum": ["search", "info", "files"]},
                    "operations": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["operation_type", "operations"],
            },
        ),
        types.Tool(
            name="hf_space_management",
            description=(
                "Advanced Hugging Face Spaces management.\n"
                "Actions: runtime_info, restart, pause, set_sleep_time, duplicate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["runtime_info", "restart", "pause", "set_sleep_time", "duplicate"],
                    },
                    "space_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "sleep_time": {"type": "integer"},
                },
                "required": ["action", "space_id"],
            },
        ),
        types.Tool(
            name="hf_community_features",
            description=(
                "Community & social features for HF repositories.\n"
                "Actions: like, unlike, get_likes, create_discussion, get_commits, get_refs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["like", "unlike", "get_likes", "create_discussion", "get_commits", "get_refs"],
                    },
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="hf_inference_tools",
            description="Test model inference and check available endpoints.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["test_inference", "check_endpoints"]},
                    "repo_id": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "parameters": {"type": "object"},
                },
                "required": ["action", "repo_id"],
            },
        ),
        types.Tool(
            name="hf_repo_file_manager",
            description=(
                "Unified repo + file manager with rename support.\n"
                "Actions: repo_create, repo_delete, repo_info, list_files, "
                "file_read, file_write, file_edit, file_delete, file_rename."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "repo_create", "repo_delete", "repo_info", "list_files",
                            "file_read", "file_write", "file_edit", "file_delete", "file_rename",
                        ],
                    },
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "filename": {"type": "string"},
                    "new_filename": {"type": "string"},
                    "content": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "commit_message": {"type": "string"},
                    "private": {"type": "boolean", "default": False},
                    "description": {"type": "string"},
                    "space_sdk": {"type": "string"},
                },
                "required": ["action", "repo_id"],
            },
        ),
    ]


# ──────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ──────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    log.info(f"Tool call: {name} args={list(arguments.keys())}")

    result = await _dispatch(name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _dispatch(name: str, args: dict) -> Any:
    if name == "hf_system_info":
        return _system_info()
    if name == "hf_repository_manager":
        return _repository_manager(args)
    if name == "hf_file_operations":
        return _file_operations(args)
    if name == "hf_search_hub":
        return _search_hub(args)
    if name == "hf_collections":
        return _collections(args)
    if name == "hf_pull_requests":
        return _pull_requests(args)
    if name == "hf_upload_manager":
        return _upload_manager(args)
    if name == "hf_batch_operations":
        return _batch_operations(args)
    if name == "hf_space_management":
        return _space_management(args)
    if name == "hf_community_features":
        return _community_features(args)
    if name == "hf_inference_tools":
        return _inference_tools(args)
    if name == "hf_repo_file_manager":
        return _repo_file_manager(args)
    return {"error": f"Unknown tool: {name}"}


# ── 1. SYSTEM INFO ────────────────────────
def _system_info() -> dict:
    whoami = None
    if HF_TOKEN:
        whoami = safe_run(api.whoami)
    return {
        "server": "hf-mcp-server",
        "version": "1.0.0",
        "read_only": HF_READ_ONLY,
        "admin_mode": HF_ADMIN_MODE,
        "token_set": bool(HF_TOKEN),
        "max_file_size_bytes": HF_MAX_FILE_SIZE,
        "whoami": whoami,
        "tools": 12,
    }


# ── 2. REPOSITORY MANAGER ────────────────
def _repository_manager(args: dict) -> dict:
    action = args["action"]
    repo_id = args["repo_id"]
    repo_type = args.get("repo_type", "model")

    if action == "info":
        return safe_run(lambda: api.repo_info(repo_id=repo_id, repo_type=repo_type).__dict__)

    if action == "list_files":
        files = safe_run(api.list_repo_files, repo_id=repo_id, repo_type=repo_type)
        return {"files": list(files) if not isinstance(files, dict) else files}

    if action == "create":
        read_only_guard()
        result = safe_run(
            api.create_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            private=args.get("private", False),
            space_sdk=args.get("space_sdk"),
            exist_ok=True,
        )
        return {"created": str(result)}

    if action == "delete":
        read_only_guard()
        admin_guard()
        safe_run(api.delete_repo, repo_id=repo_id, repo_type=repo_type)
        return {"deleted": repo_id}

    return {"error": f"Unknown action: {action}"}


# ── 3. FILE OPERATIONS ───────────────────
def _file_operations(args: dict) -> dict:
    action = args["action"]
    repo_id = args["repo_id"]
    filename = args.get("filename", "")
    repo_type = args.get("repo_type", "model")

    if action == "read":
        max_size = args.get("max_size", 500000)
        chunk_size = args.get("chunk_size")
        chunk_number = args.get("chunk_number", 0)
        try:
            content_bytes = api.hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type=repo_type
            )
            with open(content_bytes, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            if chunk_size:
                start = chunk_number * chunk_size
                text = text[start: start + chunk_size]
            else:
                text = text[:max_size]
            return {"content": text, "truncated": len(text) >= max_size}
        except Exception as e:
            return {"error": str(e)}

    if action == "write":
        read_only_guard()
        content = args.get("content", "")
        commit_msg = args.get("commit_message", f"Upload {filename}")
        result = safe_run(
            api.upload_file,
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_msg,
        )
        return {"uploaded": str(result)}

    if action == "edit":
        read_only_guard()
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        commit_msg = args.get("commit_message", f"Edit {filename}")
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
            if old_text not in original:
                return {"error": f"old_text not found in {filename}"}
            updated = original.replace(old_text, new_text, 1)
            result = safe_run(
                api.upload_file,
                path_or_fileobj=updated.encode("utf-8"),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_msg,
            )
            return {"edited": str(result)}
        except Exception as e:
            return {"error": str(e)}

    if action == "delete":
        read_only_guard()
        commit_msg = args.get("commit_message", f"Delete {filename}")
        result = safe_run(
            api.delete_file,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_msg,
        )
        return {"deleted": str(result)}

    if action == "validate":
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext == "json":
                json.loads(content)
                return {"valid": True, "format": "json"}
            return {"valid": True, "format": ext, "size": len(content)}
        except json.JSONDecodeError as e:
            return {"valid": False, "error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    if action == "backup":
        read_only_guard()
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            backup_name = f"{filename}.backup"
            result = safe_run(
                api.upload_file,
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=backup_name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Backup of {filename}",
            )
            return {"backup_created": backup_name, "result": str(result)}
        except Exception as e:
            return {"error": str(e)}

    if action == "batch_edit":
        read_only_guard()
        pattern = args.get("pattern", "")
        replacement = args.get("replacement", "")
        file_patterns = args.get("file_patterns", ["*.md"])
        import fnmatch
        files = list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        matched = [f for f in files if any(fnmatch.fnmatch(f, p) for p in file_patterns)]
        results = []
        for fname in matched:
            try:
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
                        commit_message=f"batch_edit: replace pattern in {fname}",
                    )
                    results.append({"file": fname, "status": "edited"})
                else:
                    results.append({"file": fname, "status": "skipped (pattern not found)"})
            except Exception as e:
                results.append({"file": fname, "status": "error", "error": str(e)})
        return {"results": results}

    return {"error": f"Unknown action: {action}"}


# ── 4. SEARCH HUB ────────────────────────
def _search_hub(args: dict) -> dict:
    content_type = args["content_type"]
    query = args.get("query")
    author = args.get("author")
    filter_tag = args.get("filter_tag")
    limit = args.get("limit", 20)

    def _fmt(item):
        return {
            "id": getattr(item, "id", None) or getattr(item, "modelId", None),
            "author": getattr(item, "author", None),
            "downloads": getattr(item, "downloads", None),
            "likes": getattr(item, "likes", None),
            "tags": getattr(item, "tags", []),
        }

    try:
        if content_type == "models":
            items = api.list_models(search=query, author=author, filter=filter_tag, limit=limit)
        elif content_type == "datasets":
            items = api.list_datasets(search=query, author=author, filter=filter_tag, limit=limit)
        else:
            items = api.list_spaces(search=query, author=author, filter=filter_tag, limit=limit)
        return {"results": [_fmt(i) for i in items]}
    except Exception as e:
        return {"error": str(e)}


# ── 5. COLLECTIONS ───────────────────────
def _collections(args: dict) -> dict:
    action = args["action"]
    if action == "create":
        read_only_guard()
        result = safe_run(
            api.create_collection,
            title=args["title"],
            namespace=args.get("namespace"),
            description=args.get("description", ""),
            private=args.get("private", False),
            exist_ok=True,
        )
        return {"slug": getattr(result, "slug", str(result))}

    if action == "add_item":
        read_only_guard()
        result = safe_run(
            api.add_collection_item,
            collection_slug=args["collection_slug"],
            item_id=args["item_id"],
            item_type=args["item_type"],
            note=args.get("note", ""),
            exists_ok=True,
        )
        return {"added": str(result)}

    if action == "info":
        result = safe_run(api.get_collection, collection_slug=args["collection_slug"])
        return result if isinstance(result, dict) else result.__dict__

    return {"error": f"Unknown action: {action}"}


# ── 6. PULL REQUESTS ─────────────────────
def _pull_requests(args: dict) -> dict:
    action = args["action"]
    repo_id = args["repo_id"]
    repo_type = args.get("repo_type", "model")

    if action == "create":
        read_only_guard()
        title = args.get("title", "New PR")
        result = safe_run(
            api.create_discussion,
            repo_id=repo_id,
            repo_type=repo_type,
            title=title,
            description=args.get("description", ""),
            pull_request=True,
        )
        return {"pr": str(result)}

    if action == "list":
        status = args.get("status", "open")
        discussions = safe_run(
            api.get_repo_discussions,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        if isinstance(discussions, dict):
            return discussions
        prs = [
            {"num": d.num, "title": d.title, "status": d.status, "author": d.author}
            for d in discussions
            if d.is_pull_request and (status == "all" or d.status == status)
        ]
        return {"pull_requests": prs}

    if action == "details":
        pr_number = args["pr_number"]
        result = safe_run(
            api.get_discussion_details,
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=pr_number,
        )
        return result if isinstance(result, dict) else result.__dict__

    if action == "create_with_files":
        read_only_guard()
        files = args.get("files", [])
        commit_msg = args.get("commit_message", "Update files")
        pr_title = args.get("pr_title", "File updates")
        ops = [
            {
                "operationId": f"upload_{i}",
                "key": f["path"],
                "value": f["content"].encode("utf-8"),
            }
            for i, f in enumerate(files)
        ]
        result = safe_run(
            api.create_commit,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[
                api.CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=f["content"].encode("utf-8"))
                for f in files
            ],
            commit_message=commit_msg,
            create_pr=True,
            pr_description=args.get("pr_description", ""),
        )
        return {"pr": str(result)}

    return {"error": f"Unknown action: {action}"}


# ── 7. UPLOAD MANAGER ────────────────────
def _upload_manager(args: dict) -> dict:
    read_only_guard()
    action = args["action"]
    repo_id = args["repo_id"]
    repo_type = args.get("repo_type", "model")

    if action == "single_file":
        result = safe_run(
            api.upload_file,
            path_or_fileobj=args["content"].encode("utf-8"),
            path_in_repo=args["file_path"],
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=args.get("commit_message", f"Upload {args['file_path']}"),
        )
        return {"uploaded": str(result)}

    if action == "multiple_files":
        from huggingface_hub import CommitOperationAdd
        files = args.get("files", [])
        operations = [
            CommitOperationAdd(
                path_in_repo=f["path"],
                path_or_fileobj=f["content"].encode("utf-8"),
            )
            for f in files
        ]
        result = safe_run(
            api.create_commit,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=args.get("commit_message", "Upload multiple files"),
        )
        return {"commit": str(result)}

    if action == "with_pr":
        from huggingface_hub import CommitOperationAdd
        operations = [
            CommitOperationAdd(
                path_in_repo=args["file_path"],
                path_or_fileobj=args["content"].encode("utf-8"),
            )
        ]
        result = safe_run(
            api.create_commit,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=args.get("commit_message", f"Upload {args['file_path']}"),
            create_pr=True,
        )
        return {"pr": str(result)}

    return {"error": f"Unknown action: {action}"}


# ── 8. BATCH OPERATIONS ──────────────────
def _batch_operations(args: dict) -> dict:
    op_type = args["operation_type"]
    operations = args["operations"]
    results = []
    for op in operations:
        if op_type == "search":
            results.append(_search_hub(op))
        elif op_type == "info":
            results.append(_repository_manager({"action": "info", **op}))
        elif op_type == "files":
            results.append(_repository_manager({"action": "list_files", **op}))
        else:
            results.append({"error": f"Unknown op_type: {op_type}"})
    return {"results": results}


# ── 9. SPACE MANAGEMENT ──────────────────
def _space_management(args: dict) -> dict:
    action = args["action"]
    space_id = args["space_id"]

    if action == "runtime_info":
        result = safe_run(api.get_space_runtime, repo_id=space_id)
        return result if isinstance(result, dict) else result.__dict__

    if action == "restart":
        read_only_guard()
        result = safe_run(api.restart_space, repo_id=space_id)
        return {"restarted": str(result)}

    if action == "pause":
        read_only_guard()
        result = safe_run(api.pause_space, repo_id=space_id)
        return {"paused": str(result)}

    if action == "set_sleep_time":
        read_only_guard()
        result = safe_run(api.set_space_sleep_time, repo_id=space_id, sleep_time=args.get("sleep_time", 300))
        return {"sleep_time_set": str(result)}

    if action == "duplicate":
        read_only_guard()
        result = safe_run(
            api.duplicate_space,
            from_id=space_id,
            to_id=args["to_id"],
            exist_ok=True,
        )
        return {"duplicated_to": args["to_id"], "result": str(result)}

    return {"error": f"Unknown action: {action}"}


# ── 10. COMMUNITY FEATURES ───────────────
def _community_features(args: dict) -> dict:
    action = args["action"]
    repo_id = args.get("repo_id", "")
    repo_type = args.get("repo_type", "model")

    if action == "like":
        read_only_guard()
        safe_run(api.like, repo_id=repo_id, repo_type=repo_type)
        return {"liked": repo_id}

    if action == "unlike":
        read_only_guard()
        safe_run(api.unlike, repo_id=repo_id, repo_type=repo_type)
        return {"unliked": repo_id}

    if action == "get_likes":
        result = safe_run(api.list_liked_repos)
        return {"liked_repos": [str(r) for r in (result or [])]}

    if action == "create_discussion":
        read_only_guard()
        result = safe_run(
            api.create_discussion,
            repo_id=repo_id,
            repo_type=repo_type,
            title=args.get("title", "New Discussion"),
            description=args.get("description", ""),
            pull_request=False,
        )
        return {"discussion": str(result)}

    if action == "get_commits":
        result = safe_run(api.list_repo_commits, repo_id=repo_id, repo_type=repo_type)
        if isinstance(result, dict):
            return result
        return {"commits": [{"id": c.commit_id, "message": c.title} for c in list(result)[:50]]}

    if action == "get_refs":
        result = safe_run(api.list_repo_refs, repo_id=repo_id, repo_type=repo_type)
        return result if isinstance(result, dict) else result.__dict__

    return {"error": f"Unknown action: {action}"}


# ── 11. INFERENCE TOOLS ──────────────────
def _inference_tools(args: dict) -> dict:
    action = args["action"]
    repo_id = args["repo_id"]

    if action == "check_endpoints":
        info = safe_run(api.repo_info, repo_id=repo_id, repo_type="model")
        if isinstance(info, dict):
            return info
        return {
            "repo_id": repo_id,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": getattr(info, "tags", []),
        }

    if action == "test_inference":
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=repo_id, token=HF_TOKEN or None, timeout=HF_INFERENCE_TIMEOUT)
            inputs = args.get("inputs", ["Hello"])
            parameters = args.get("parameters", {})
            results = []
            for inp in inputs:
                try:
                    out = client.text_generation(inp, **parameters)
                    results.append({"input": inp, "output": str(out)})
                except Exception as e:
                    results.append({"input": inp, "error": str(e)})
            return {"inference_results": results}
        except ImportError:
            return {"error": "InferenceClient not available in this huggingface_hub version"}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: {action}"}


# ── 12. REPO FILE MANAGER ────────────────
def _repo_file_manager(args: dict) -> dict:
    action = args["action"]
    repo_id = args["repo_id"]
    repo_type = args.get("repo_type", "model")
    filename = args.get("filename", "")

    if action == "repo_create":
        return _repository_manager({"action": "create", "repo_id": repo_id, "repo_type": repo_type, **args})
    if action == "repo_delete":
        return _repository_manager({"action": "delete", "repo_id": repo_id, "repo_type": repo_type})
    if action == "repo_info":
        return _repository_manager({"action": "info", "repo_id": repo_id, "repo_type": repo_type})
    if action == "list_files":
        return _repository_manager({"action": "list_files", "repo_id": repo_id, "repo_type": repo_type})
    if action == "file_read":
        return _file_operations({"action": "read", "repo_id": repo_id, "filename": filename, "repo_type": repo_type})
    if action == "file_write":
        return _file_operations({"action": "write", "repo_id": repo_id, "filename": filename, "repo_type": repo_type, **args})
    if action == "file_edit":
        return _file_operations({"action": "edit", "repo_id": repo_id, "filename": filename, "repo_type": repo_type, **args})
    if action == "file_delete":
        return _file_operations({"action": "delete", "repo_id": repo_id, "filename": filename, "repo_type": repo_type, **args})

    if action == "file_rename":
        read_only_guard()
        new_filename = args["new_filename"]
        commit_msg = args.get("commit_message", f"Rename {filename} -> {new_filename}")
        try:
            from huggingface_hub import CommitOperationCopy, CommitOperationDelete
            operations = [
                CommitOperationCopy(src_path_in_repo=filename, dest_path_in_repo=new_filename),
                CommitOperationDelete(path_in_repo=filename),
            ]
            result = safe_run(
                api.create_commit,
                repo_id=repo_id,
                repo_type=repo_type,
                operations=operations,
                commit_message=commit_msg,
            )
            return {"renamed": f"{filename} -> {new_filename}", "result": str(result)}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: {action}"}


# ──────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────

async def main():
    log.info("Starting HF MCP Server (stdio transport)")
    log.info(f"read_only={HF_READ_ONLY} admin_mode={HF_ADMIN_MODE} token_set={bool(HF_TOKEN)}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
