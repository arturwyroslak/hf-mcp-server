# HF MCP Server — Lista Narzędzi

Serwer wystawia **12 narzędzi** przez **Streamable HTTP** na:

```
http://localhost:8000/mcp          ← lokalnie
https://mcp.twojadomena.pl/mcp     ← z domeną
```

---

## 1. `hf_system_info`
**Opis:** Status serwera, konfiguracja i test łączności z HF Hub.  
**Parametry:** brak  
**Zwraca:** wersję serwera, endpoint URL, flagi read_only/admin_mode, dane zalogowanego użytkownika.

---

## 2. `hf_repository_manager`
**Opis:** Pełne zarządzanie repozytoriami (modele, datasety, Spaces).

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `create` \| `delete` \| `info` \| `list_files` |
| `repo_id` | string ✳ | np. `username/repo-name` |
| `repo_type` | string | `model` \| `dataset` \| `space` (domyślnie: `model`) |
| `private` | boolean | Prywatne repo |
| `description` | string | Opis |
| `space_sdk` | string | `gradio` \| `streamlit` \| `docker` \| `static` |

> ⚠️ `delete` wymaga `HF_ADMIN_MODE=true`

---

## 3. `hf_file_operations`
**Opis:** Pełny CRUD na plikach z walidacją, backupem i batch edycją.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `read` \| `write` \| `edit` \| `delete` \| `validate` \| `backup` \| `batch_edit` |
| `repo_id` | string ✳ | Identyfikator repo |
| `filename` | string | Ścieżka pliku w repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `content` | string | Treść pliku (write) |
| `commit_message` | string | Wiadomość commita |
| `old_text` | string | Tekst do zastąpienia (edit) |
| `new_text` | string | Nowy tekst (edit) |
| `max_size` | integer | Max znaków do odczytu (domyślnie: 500000) |
| `chunk_size` | integer | Rozmiar chunka |
| `chunk_number` | integer | Numer chunka (domyślnie: 0) |
| `pattern` | string | Wzorzec (batch_edit) |
| `replacement` | string | Zamiennik (batch_edit) |
| `file_patterns` | array | Wzorce plików np. `["*.md"]` (batch_edit) |

---

## 4. `hf_search_hub`
**Opis:** Wyszukiwanie modeli, datasetów i Spaces z filtrami.

| Parametr | Typ | Opis |
|----------|-----|------|
| `content_type` | string ✳ | `models` \| `datasets` \| `spaces` |
| `query` | string | Fraza wyszukiwania |
| `author` | string | Filtruj po autorze |
| `filter_tag` | string | Filtruj po tagu |
| `limit` | integer | Liczba wyników (domyślnie: 20) |

---

## 5. `hf_collections`
**Opis:** Zarządzanie kolekcjami na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `create` \| `add_item` \| `info` |
| `title` | string | Tytuł kolekcji (create) |
| `namespace` | string | Namespace (domyślnie: zalogowany użytkownik) |
| `description` | string | Opis |
| `private` | boolean | Prywatna kolekcja |
| `collection_slug` | string | Identyfikator kolekcji |
| `item_id` | string | ID elementu |
| `item_type` | string | `model` \| `dataset` \| `space` |
| `note` | string | Notatka do elementu |

---

## 6. `hf_pull_requests`
**Opis:** Zarządzanie Pull Requestami na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `create` \| `list` \| `details` \| `create_with_files` |
| `repo_id` | string ✳ | Identyfikator repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `title` | string | Tytuł PR |
| `description` | string | Opis PR |
| `status` | string | `open` \| `closed` \| `all` |
| `pr_number` | integer | Numer PR (details) |
| `files` | array | `[{path, content}]` (create_with_files) |
| `commit_message` | string | Wiadomość commita |
| `pr_title` | string | Tytuł PR (create_with_files) |
| `pr_description` | string | Opis PR |

---

## 7. `hf_upload_manager`
**Opis:** Upload plików — pojedynczy, batch i z PR.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `single_file` \| `multiple_files` \| `with_pr` |
| `repo_id` | string ✳ | Identyfikator repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `file_path` | string | Ścieżka w repo |
| `content` | string | Treść pliku |
| `commit_message` | string | Wiadomość commita |
| `files` | array | `[{path, content}]` (multiple_files) |
| `pr_title` | string | Tytuł PR (with_pr) |
| `pr_description` | string | Opis PR |

---

## 8. `hf_batch_operations`
**Opis:** Równoległe operacje na wielu repozytoriach.

| Parametr | Typ | Opis |
|----------|-----|------|
| `operation_type` | string ✳ | `search` \| `info` \| `files` |
| `operations` | array ✳ | Lista operacji jako obiekty z parametrami |

---

## 9. `hf_space_management`
**Opis:** Zaawansowane zarządzanie HF Spaces.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `runtime_info` \| `restart` \| `pause` \| `set_sleep_time` \| `duplicate` |
| `space_id` | string ✳ | np. `username/space-name` |
| `to_id` | string | Docelowe ID (duplicate) |
| `sleep_time` | integer | Czas uśpienia w sekundach |

---

## 10. `hf_community_features`
**Opis:** Funkcje społecznościowe dla repozytoriów HF.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `like` \| `unlike` \| `get_likes` \| `create_discussion` \| `get_commits` \| `get_refs` |
| `repo_id` | string | Identyfikator repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `title` | string | Tytuł dyskusji |
| `description` | string | Treść dyskusji |

---

## 11. `hf_inference_tools`
**Opis:** Testowanie inferencji modeli i sprawdzanie endpointów.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `check_endpoints` \| `test_inference` |
| `repo_id` | string ✳ | ID modelu na HF Hub |
| `inputs` | array | Teksty wejściowe |
| `parameters` | object | Parametry inferencji |

---

## 12. `hf_repo_file_manager`
**Opis:** Ujednolicony manager plików i repozytoriów z obsługą rename.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string ✳ | `repo_create` \| `repo_delete` \| `repo_info` \| `list_files` \| `file_read` \| `file_write` \| `file_edit` \| `file_delete` \| `file_rename` |
| `repo_id` | string ✳ | Identyfikator repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `filename` | string | Ścieżka pliku |
| `new_filename` | string | Nowa ścieżka (file_rename) |
| `content` | string | Treść pliku |
| `old_text` | string | Tekst do zastąpienia |
| `new_text` | string | Nowy tekst |
| `commit_message` | string | Wiadomość commita |

> ✳ = parametr wymagany

---

## Uprawnienia

| Zmienna env | Domyślnie | Efekt |
|-------------|-----------|-------|
| `HF_READ_ONLY=true` | false | Blokuje wszystkie zapisy |
| `HF_ADMIN_MODE=true` | false | Odblokowuje `repo delete` |
| `HF_TOKEN` | — | Wymagany dla operacji zapisu |
