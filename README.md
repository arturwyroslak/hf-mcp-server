# hf-mcp-server

Produkcyjny serwer MCP (Model Context Protocol) do zarządzania HuggingFace Hub — uruchamiany przez Docker.

[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://hub.docker.com)
[![MCP](https://img.shields.io/badge/MCP-1.0-green)](https://modelcontextprotocol.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-yellow?logo=huggingface)](https://huggingface.co)

## 🚀 Szybki start

```bash
git clone https://github.com/arturwyroslak/hf-mcp-server
cd hf-mcp-server
cp .env.example .env
# Edytuj .env — wpisz swój HF_TOKEN
docker compose build
docker compose run --rm hf-mcp
```

## 🛠️ Narzędzia (12)

Szczegółowa dokumentacja wszystkich narzędzi → [`TOOLS.md`](./TOOLS.md)

| # | Narzędzie | Opis |
|---|-----------|------|
| 1 | `hf_system_info` | Status serwera i test łączności |
| 2 | `hf_repository_manager` | Tworzenie/usuwanie/info repozytoriów i Spaces |
| 3 | `hf_file_operations` | Pełny CRUD plików (read/write/edit/delete/batch_edit) |
| 4 | `hf_search_hub` | Wyszukiwanie modeli, datasetów, Spaces |
| 5 | `hf_collections` | Zarządzanie kolekcjami |
| 6 | `hf_pull_requests` | Tworzenie i zarządzanie PR-ami |
| 7 | `hf_upload_manager` | Upload plików (single/batch/with PR) |
| 8 | `hf_batch_operations` | Operacje batch na wielu repozytoriach |
| 9 | `hf_space_management` | Zarządzanie Spaces (restart/pause/duplicate) |
| 10 | `hf_community_features` | Like, dyskusje, historia commitów |
| 11 | `hf_inference_tools` | Test inferencji i sprawdzanie endpointów |
| 12 | `hf_repo_file_manager` | Unified manager z obsługą rename |

## ⚙️ Konfiguracja

Skopiuj `.env.example` do `.env` i ustaw zmienne:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # wymagany
HF_READ_ONLY=false                            # true = tylko odczyt
HF_ADMIN_MODE=false                           # true = włącza delete repo
HF_MAX_FILE_SIZE=104857600                    # 100 MB
HF_INFERENCE_TIMEOUT=30                       # sekundy
```

## 🔌 Podłączenie klientów MCP

### Claude Desktop

Edytuj `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) lub `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "hf-mcp": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/hf-mcp-server/docker-compose.yml",
        "run", "--rm", "-i", "hf-mcp"
      ]
    }
  }
}
```

### VS Code (Copilot / Continue)

Dodaj do `.vscode/mcp.json` lub `settings.json`:

```json
{
  "mcp": {
    "servers": {
      "hf-mcp": {
        "type": "stdio",
        "command": "docker",
        "args": [
          "compose", "-f", "/absolute/path/to/hf-mcp-server/docker-compose.yml",
          "run", "--rm", "-i", "hf-mcp"
        ]
      }
    }
  }
}
```

### Cursor

Edytuj `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hf-mcp": {
      "command": "docker",
      "args": [
        "compose", "-f", "/absolute/path/to/hf-mcp-server/docker-compose.yml",
        "run", "--rm", "-i", "hf-mcp"
      ]
    }
  }
}
```

### Gemini CLI / inne klienty

Każdy klient obsługujący transport `stdio` może podłączyć się przez:

```bash
docker compose -f /path/to/hf-mcp-server/docker-compose.yml run --rm -i hf-mcp
```

## 🐳 Docker commands

```bash
# Zbuduj obraz
docker compose build

# Test — ręczny stdin/stdout
docker compose run --rm -i hf-mcp

# Rebuild z czystym cache
docker compose build --no-cache

# Sprawdź logi
docker compose logs hf-mcp
```

## 📄 Licencja

MIT
