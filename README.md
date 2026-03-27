# hf-mcp-server

Produkcyjny serwer MCP do zarządzania HuggingFace Hub, wystawiony przez **Streamable HTTP** na porcie **8000**.

[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://hub.docker.com)
[![MCP](https://img.shields.io/badge/MCP-1.10-green)](https://modelcontextprotocol.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-yellow?logo=huggingface)](https://huggingface.co)

## 🚀 Quickstart

```bash
git clone https://github.com/arturwyroslak/hf-mcp-server
cd hf-mcp-server
cp .env.example .env
# wpisz swój HF_TOKEN w .env

docker compose build
docker compose up -d
```

Serwer będzie dostępny pod:

```
http://localhost:8000/mcp
```

## ⚙️ Konfiguracja

```env
HF_TOKEN=hf_xxxxxxxx          # wymagany
HF_READ_ONLY=false            # true = tylko odczyt
HF_ADMIN_MODE=false           # true = włącza delete repo
MCP_HOST=0.0.0.0              # adres nasłuchu
MCP_PORT=8000                 # port HTTP
MCP_PATH=/mcp                 # ścieżka endpointu
```

## 🌍 Ustawienie domeny

**Port:** `8000`  
**Endpoint:** `/mcp`  
**Docelowy URL:** `https://mcp.twojadomena.pl/mcp`

### Nginx reverse proxy

```nginx
server {
    server_name mcp.twojadomena.pl;

    location /mcp {
        proxy_pass         http://127.0.0.1:8000/mcp;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_buffering    off;
        proxy_read_timeout 300s;
    }
}
```

```bash
certbot --nginx -d mcp.twojadomena.pl
```

### Caddy (automatyczny SSL)

```caddyfile
mcp.twojadomena.pl {
    reverse_proxy /mcp localhost:8000
}
```

## 🔌 Podłączenie klientów MCP

### Claude Desktop

Edytuj `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hf-mcp": {
      "type": "http",
      "url": "https://mcp.twojadomena.pl/mcp"
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
      "type": "http",
      "url": "https://mcp.twojadomena.pl/mcp"
    }
  }
}
```

### VS Code / Continue

```json
{
  "mcp": {
    "servers": {
      "hf-mcp": {
        "type": "http",
        "url": "https://mcp.twojadomena.pl/mcp"
      }
    }
  }
}
```

### Python (OpenAI Agents SDK)

```python
from agents.mcp import MCPServerStreamableHttp

server = MCPServerStreamableHttp(
    name="hf-mcp",
    params={"url": "https://mcp.twojadomena.pl/mcp", "timeout": 30},
)
```

## 🐳 Docker commands

```bash
# Zbuduj
docker compose build

# Uruchom w tle
docker compose up -d

# Sprawdź logi
docker compose logs -f hf-mcp

# Rebuild bez cache
docker compose build --no-cache

# Zatrzymaj
docker compose down
```

## 🛠️ Narzędzia (12)

Szczegółowa dokumentacja → [`TOOLS.md`](./TOOLS.md)

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

## 📄 Licencja

MIT
