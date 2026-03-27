# HF MCP Server — Lista Narzędzi

Serwer udostępnia **12 narzędzi** przez protokół MCP (stdio transport).

---

## 1. `hf_system_info`
**Opis:** Zwraca status serwera, konfigurację i wynik testu łączności z HF Hub.

**Parametry:** brak

**Zwraca:** wersję serwera, flagi (read_only, admin_mode), informacje o zalogowanym użytkowniku.

---

## 2. `hf_repository_manager`
**Opis:** Pełne zarządzanie repozytoriami na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `create` \| `delete` \| `info` \| `list_files` |
| `repo_id` | string (wymagany) | np. `username/repo-name` |
| `repo_type` | string | `model` \| `dataset` \| `space` (domyślnie: `model`) |
| `private` | boolean | Prywatne repo (domyślnie: `false`) |
| `description` | string | Opis repozytorium |
| `space_sdk` | string | `gradio` \| `streamlit` \| `docker` \| `static` |

> ⚠️ `delete` wymaga `HF_ADMIN_MODE=true`

---

## 3. `hf_file_operations`
**Opis:** Pełny CRUD na plikach w repozytoriach HF.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `read` \| `write` \| `edit` \| `delete` \| `validate` \| `backup` \| `batch_edit` |
| `repo_id` | string (wymagany) | Identyfikator repozytorium |
| `filename` | string | Ścieżka pliku w repo |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `content` | string | Treść pliku (dla write) |
| `commit_message` | string | Wiadomość commita |
| `old_text` | string | Tekst do zastąpienia (dla edit) |
| `new_text` | string | Nowy tekst (dla edit) |
| `max_size` | integer | Max znaki do odczytu (domyślnie: 500000) |
| `chunk_size` | integer | Rozmiar chunka do odczytu |
| `chunk_number` | integer | Numer chunka (domyślnie: 0) |
| `pattern` | string | Wzorzec tekstu (dla batch_edit) |
| `replacement` | string | Zamiennik (dla batch_edit) |
| `file_patterns` | array | Wzorce plików (np. `["*.md"]`) |

---

## 4. `hf_search_hub`
**Opis:** Wyszukiwanie modeli, datasetów i Spaces na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `content_type` | string (wymagany) | `models` \| `datasets` \| `spaces` |
| `query` | string | Fraza wyszukiwania |
| `author` | string | Filtruj po autorze |
| `filter_tag` | string | Filtruj po tagu |
| `limit` | integer | Liczba wyników (domyślnie: 20) |

---

## 5. `hf_collections`
**Opis:** Zarządzanie kolekcjami na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `create` \| `add_item` \| `info` |
| `title` | string | Tytuł kolekcji (dla create) |
| `namespace` | string | Namespace (domyślnie: zalogowany użytkownik) |
| `description` | string | Opis kolekcji |
| `private` | boolean | Prywatna kolekcja |
| `collection_slug` | string | Identyfikator kolekcji |
| `item_id` | string | ID elementu do dodania |
| `item_type` | string | `model` \| `dataset` \| `space` |
| `note` | string | Notatka do elementu |

---

## 6. `hf_pull_requests`
**Opis:** Zarządzanie Pull Requestami na HF Hub.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `create` \| `list` \| `details` \| `create_with_files` |
| `repo_id` | string (wymagany) | Identyfikator repozytorium |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `title` | string | Tytuł PR |
| `description` | string | Opis PR |
| `status` | string | `open` \| `closed` \| `all` |
| `pr_number` | integer | Numer PR (dla details) |
| `files` | array | `[{path, content}]` (dla create_with_files) |
| `commit_message` | string | Wiadomość commita |
| `pr_title` | string | Tytuł PR (dla create_with_files) |
| `pr_description` | string | Opis PR |

---

## 7. `hf_upload_manager`
**Opis:** Upload plików do repozytoriów HF.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `single_file` \| `multiple_files` \| `with_pr` |
| `repo_id` | string (wymagany) | Identyfikator repozytorium |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `file_path` | string | Ścieżka w repo |
| `content` | string | Treść pliku |
| `commit_message` | string | Wiadomość commita |
| `files` | array | `[{path, content}]` (dla multiple_files) |
| `pr_title` | string | Tytuł PR (dla with_pr) |
| `pr_description` | string | Opis PR |

---

## 8. `hf_batch_operations`
**Opis:** Wykonywanie operacji batch na wielu repozytoriach jednocześnie.

| Parametr | Typ | Opis |
|----------|-----|------|
| `operation_type` | string (wymagany) | `search` \| `info` \| `files` |
| `operations` | array (wymagany) | Lista operacji (każda jako obiekt z parametrami) |

---

## 9. `hf_space_management`
**Opis:** Zaawansowane zarządzanie HuggingFace Spaces.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `runtime_info` \| `restart` \| `pause` \| `set_sleep_time` \| `duplicate` |
| `space_id` | string (wymagany) | np. `username/space-name` |
| `to_id` | string | Docelowe ID (dla duplicate) |
| `sleep_time` | integer | Czas uśpienia w sekundach |

---

## 10. `hf_community_features`
**Opis:** Funkcje społecznościowe dla repozytoriów HF.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `like` \| `unlike` \| `get_likes` \| `create_discussion` \| `get_commits` \| `get_refs` |
| `repo_id` | string | Identyfikator repozytorium |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `title` | string | Tytuł dyskusji |
| `description` | string | Treść dyskusji |

---

## 11. `hf_inference_tools`
**Opis:** Testowanie inferencji i sprawdzanie endpointów modeli.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `test_inference` \| `check_endpoints` |
| `repo_id` | string (wymagany) | ID modelu na HF Hub |
| `inputs` | array | Lista tekstów wejściowych |
| `parameters` | object | Parametry inferencji (np. `max_length`) |

---

## 12. `hf_repo_file_manager`
**Opis:** Ujednolicony manager plików i repozytoriów z obsługą zmiany nazwy.

| Parametr | Typ | Opis |
|----------|-----|------|
| `action` | string (wymagany) | `repo_create` \| `repo_delete` \| `repo_info` \| `list_files` \| `file_read` \| `file_write` \| `file_edit` \| `file_delete` \| `file_rename` |
| `repo_id` | string (wymagany) | Identyfikator repozytorium |
| `repo_type` | string | `model` \| `dataset` \| `space` |
| `filename` | string | Ścieżka pliku w repo |
| `new_filename` | string | Nowa ścieżka (dla file_rename) |
| `content` | string | Treść pliku |
| `old_text` | string | Tekst do zastąpienia |
| `new_text` | string | Nowy tekst |
| `commit_message` | string | Wiadomość commita |

---

## Uprawnienia

| Flaga env | Domyślnie | Efekt |
|-----------|-----------|-------|
| `HF_READ_ONLY=true` | false | Blokuje wszystkie operacje zapisu |
| `HF_ADMIN_MODE=true` | false | Odblokowuje `repo delete` |
| `HF_TOKEN` | — | Wymagany dla operacji zapisu |
