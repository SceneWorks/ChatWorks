use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Manager};

use crate::fsutil::{now_secs, write_json_atomic};

const PREVIEW_MAX_CHARS: usize = 80;
/// Filename suffix for the small per-conversation metadata sidecar.
///
/// The sidecar holds the derived `ConversationMetadata` so `list_conversations` can render the
/// history without ever opening a full transcript file. `.meta` is deliberately a separate
/// extension from the body `.json` so the two never collide: for any id, `<id>.json` is the body
/// and `<id>.meta` is the sidecar — even when an id itself contains `.meta` or `.json`.
const META_SUFFIX: &str = ".meta";
const JSON_SUFFIX: &str = ".json";

/// Per-conversation sampling overrides. Mirrors the in-app sampling defaults shape so a
/// conversation carries the exact params it was run with and round-trips untouched.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationParams {
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: f32,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub disable_thinking: bool,
}

/// Full conversation transcript + params. `messages` is kept as a flexible
/// `serde_json::Value` so the role/content/thinking/tool_calls/images/videos/denied/isError
/// shape round-trips untouched and survives future message-field additions.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub updated_at: u64,
    #[serde(default)]
    pub params: ConversationParams,
    #[serde(default)]
    pub messages: Value,
}

/// Lightweight view returned by `list_conversations` — metadata + derived preview only.
/// Message bodies are deliberately absent so listing never returns transcript data. It is also
/// the on-disk shape of the `.meta` sidecar written alongside each conversation file.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMetadata {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
    pub preview: String,
}

pub fn list_conversations(app: &AppHandle) -> Result<Vec<ConversationMetadata>, String> {
    list_conversations_in_dir(&conversations_dir(app)?)
}

pub fn get_conversation(app: &AppHandle, id: &str) -> Result<Conversation, String> {
    let dir = conversations_dir(app)?;
    get_conversation_in_dir(&dir, id)
}

pub fn save_conversation(app: &AppHandle, conversation: Conversation) -> Result<Conversation, String> {
    save_conversation_in_dir(&conversations_dir(app)?, conversation)
}

pub fn rename_conversation(
    app: &AppHandle,
    id: &str,
    title: &str,
) -> Result<ConversationMetadata, String> {
    rename_conversation_in_dir(&conversations_dir(app)?, id, title)
}

pub fn delete_conversation(app: &AppHandle, id: &str) -> Result<(), String> {
    delete_conversation_in_dir(&conversations_dir(app)?, id)
}

fn conversations_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("conversations"))
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir().map_err(|error| error.to_string())
}

fn conversation_file_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}{JSON_SUFFIX}"))
}

fn conversation_meta_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}{META_SUFFIX}"))
}

/// Validates the conversation id is non-empty and path-safe, returning the canonical **trimmed**
/// id so callers build the storage key from the same value that was checked. Canonicalizing once at
/// the boundary fixes the contract mismatch where `validate_id` trimmed for its checks but callers
/// then used the raw id on disk — so an id like `"  abc  "` validated, then wrote `"  abc  .json"`,
/// which a later `get_conversation("abc")` could not find (code-review F-009). Whitespace-padded
/// dangerous values (e.g. `" .. "`) are rejected rather than slipping past the `.`/`..`/segment
/// checks.
fn validate_id(id: &str) -> Result<String, String> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return Err("conversation id is required".to_string());
    }
    if trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains('\0')
        || trimmed == "."
        || trimmed == ".."
        || trimmed.split('/').any(|segment| segment == "..")
    {
        return Err(format!("invalid conversation id {trimmed:?}"));
    }
    Ok(trimmed.to_string())
}

/// Lists conversations reading ONLY the small `.meta` sidecar files — full transcript bodies are
/// never opened on the steady-state list path. A body file with no sidecar (legacy/old file) is
/// opened once, derived into metadata, and a sidecar is persisted so subsequent lists stay cheap.
fn list_conversations_in_dir(dir: &Path) -> Result<Vec<ConversationMetadata>, String> {
    let mut items = Vec::new();
    if !dir.is_dir() {
        return Ok(items);
    }
    // First pass: collect sidecar + body file ids so order on disk can't affect resolution.
    let mut sidecar_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut sidecar_paths: Vec<PathBuf> = Vec::new();
    let mut body_ids_without_sidecar: Vec<String> = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(id) = file_name.strip_suffix(META_SUFFIX) {
            sidecar_ids.insert(id.to_string());
            sidecar_paths.push(path);
        } else if let Some(id) = file_name.strip_suffix(JSON_SUFFIX) {
            body_ids_without_sidecar.push(id.to_string());
        }
    }

    // Cheap path: read only the small sidecar files. Skip corrupt sidecars so one bad entry never
    // blanks the whole history list.
    for path in sidecar_paths {
        if let Ok(meta) = read_meta_file(&path) {
            items.push(meta);
        }
    }
    // Legacy fallback: a body with no sidecar gets opened once, derived, and a sidecar persisted
    // so the next list does not touch the body again.
    for id in body_ids_without_sidecar {
        if sidecar_ids.contains(&id) {
            continue;
        }
        let path = conversation_file_path(dir, &id);
        let Ok(conversation) = read_conversation_file(&path) else {
            continue;
        };
        let meta = metadata_from(&conversation);
        // Persisting the sidecar is best-effort; the derived metadata is still returned either way.
        let _ = write_json_atomic(&conversation_meta_path(dir, &id), &meta);
        items.push(meta);
    }

    items.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then_with(|| a.id.cmp(&b.id)));
    Ok(items)
}

fn get_conversation_in_dir(dir: &Path, id: &str) -> Result<Conversation, String> {
    let id = validate_id(id)?;
    let path = conversation_file_path(dir, &id);
    if !path.exists() {
        return Err(format!("conversation {id:?} was not found"));
    }
    read_conversation_file(&path)
}

fn save_conversation_in_dir(dir: &Path, mut conversation: Conversation) -> Result<Conversation, String> {
    // Canonicalize the id once at the storage boundary so the safety check and the storage key agree
    // on the id (F-009): the key is built from the validated, trimmed id, not the raw input.
    conversation.id = validate_id(&conversation.id)?;
    let path = conversation_file_path(dir, &conversation.id);
    let now = now_secs();
    if conversation.created_at == 0 {
        // Preserve the original creation time on update when the caller omits it; stamp `now`
        // only for genuinely new conversations.
        let existing = if path.exists() {
            read_conversation_file(&path).ok().map(|stored| stored.created_at).unwrap_or(0)
        } else {
            0
        };
        conversation.created_at = if existing > 0 { existing } else { now };
    }
    conversation.updated_at = now;
    write_json_atomic(&path, &conversation)?;
    // Persist a metadata sidecar so list never needs to open the full transcript.
    let meta = metadata_from(&conversation);
    write_json_atomic(&conversation_meta_path(dir, &conversation.id), &meta)?;
    Ok(conversation)
}

fn rename_conversation_in_dir(dir: &Path, id: &str, title: &str) -> Result<ConversationMetadata, String> {
    let id = validate_id(id)?;
    let mut conversation = get_conversation_in_dir(dir, &id)?;
    conversation.title = title.to_string();
    conversation.updated_at = now_secs();
    let path = conversation_file_path(dir, &id);
    write_json_atomic(&path, &conversation)?;
    let meta = metadata_from(&conversation);
    write_json_atomic(&conversation_meta_path(dir, &id), &meta)?;
    Ok(meta)
}

fn delete_conversation_in_dir(dir: &Path, id: &str) -> Result<(), String> {
    let id = validate_id(id)?;
    let path = conversation_file_path(dir, &id);
    // Best-effort: drop the sidecar first, then the body. Idempotent for missing ids.
    let _ = fs::remove_file(conversation_meta_path(dir, &id));
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn read_conversation_file(path: &Path) -> Result<Conversation, String> {
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str::<Conversation>(&body).map_err(|error| error.to_string())
}

fn read_meta_file(path: &Path) -> Result<ConversationMetadata, String> {
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str::<ConversationMetadata>(&body).map_err(|error| error.to_string())
}

fn metadata_from(conversation: &Conversation) -> ConversationMetadata {
    let message_count = conversation
        .messages
        .as_array()
        .map(|messages| messages.len())
        .unwrap_or(0);
    let preview = first_user_message_text(&conversation.messages)
        .map(|text| truncate_text(&text, PREVIEW_MAX_CHARS))
        .unwrap_or_default();
    let title = if conversation.title.trim().is_empty() {
        preview.clone()
    } else {
        conversation.title.clone()
    };
    ConversationMetadata {
        id: conversation.id.clone(),
        title,
        created_at: conversation.created_at,
        updated_at: conversation.updated_at,
        message_count,
        preview,
    }
}

/// Extracts the text of the first message whose `role == "user"`, handling string content
/// and array-of-parts (the first `text` part) without panicking on unexpected shapes.
fn first_user_message_text(messages: &Value) -> Option<String> {
    let messages = messages.as_array()?;
    for message in messages {
        let role = message.get("role").and_then(|value| value.as_str());
        if role == Some("user") {
            return Some(extract_text(message.get("content")));
        }
    }
    None
}

fn extract_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            for part in parts {
                if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                    if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                        return text.to_string();
                    }
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}\u{2026}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil::{TempDir, TEMP_FILE_SUFFIX};
    use serde_json::json;

    /// Alias for the on-disk temp-suffix `fsutil` appends, so the "ignore temp files" test tracks
    /// the real value instead of re-declaring a string that could drift (PR #30 review).
    const TMP_SUFFIX: &str = TEMP_FILE_SUFFIX;

    #[test]
    fn save_then_get_round_trips_messages_and_params() {
        let dir = TempDir::new("conversations-round-trip");
        let messages = json!([
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!", "thinking": "reasoning"}
        ]);
        let params = ConversationParams {
            system_prompt: "sys".to_string(),
            temperature: 0.5,
            top_p: 0.8,
            max_tokens: 128,
            disable_thinking: false,
        };
        let conversation = Conversation {
            id: "abc".to_string(),
            title: "Hello".to_string(),
            created_at: 0,
            updated_at: 0,
            params: params.clone(),
            messages: messages.clone(),
        };
        let saved = save_conversation_in_dir(dir.path(), conversation).unwrap();
        assert!(saved.created_at > 0);
        assert!(saved.updated_at >= saved.created_at);

        let got = get_conversation_in_dir(dir.path(), "abc").unwrap();
        assert_eq!(got.id, "abc");
        assert_eq!(got.title, "Hello");
        assert_eq!(got.messages, messages);
        assert_eq!(got.params.temperature, params.temperature);
        assert_eq!(got.params.max_tokens, params.max_tokens);
        assert_eq!(got.created_at, saved.created_at);
        assert_eq!(got.updated_at, saved.updated_at);
    }

    #[test]
    fn save_rejects_empty_id() {
        let dir = test_dir("empty-id");
        let conversation = Conversation {
            id: "   ".to_string(),
            ..empty_conversation("x")
        };
        let error = save_conversation_in_dir(dir.path(), conversation).unwrap_err();
        assert!(error.contains("id"), "unexpected error: {error}");
    }

    #[test]
    fn save_rejects_path_traversal_id() {
        let dir = test_dir("traversal-id");
        let cases = ["../escape", "a/b", "a\\b", ".", "..", "bad\u{0000}id", " .. ", "  .  "];
        for id in cases {
            let conversation = Conversation {
                id: id.to_string(),
                ..empty_conversation(id)
            };
            assert!(
                save_conversation_in_dir(dir.path(), conversation).is_err(),
                "id {id:?} should be rejected"
            );
        }
    }

    #[test]
    fn validate_id_trims_before_path_checks() {
        // Whitespace-padded dangerous ids must be rejected, not slip through because the raw
        // string is not literally "." or "..".
        for id in [" .. ", "  .  ", "  ..  ", " / ", " \\ "] {
            assert!(
                validate_id(id).is_err(),
                "id {id:?} should be rejected after trimming"
            );
        }
        // validate_id returns the canonical trimmed form (F-009), so the safety check and the
        // storage key agree on what the id is.
        assert_eq!(validate_id("  abc  ").unwrap(), "abc");
    }

    #[test]
    fn save_canonicalizes_padded_id_so_get_finds_it() {
        // F-009: a padded id is stored under its trimmed key, so a later get with the trimmed id
        // (or the padded id) resolves to the same conversation rather than writing "  abc  .json".
        let dir = test_dir("canonical-id");
        let saved = save_conversation_in_dir(
            dir.path(),
            Conversation {
                id: "  abc  ".to_string(),
                title: "T".to_string(),
                created_at: 0,
                updated_at: 0,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "hi"}]),
            },
        )
        .unwrap();
        assert_eq!(saved.id, "abc");
        // The body + sidecar are written under the trimmed id.
        assert!(conversation_file_path(dir.path(), "abc").exists());
        assert!(conversation_meta_path(dir.path(), "abc").exists());
        assert!(!conversation_file_path(dir.path(), "  abc  ").exists());
        // Both the trimmed and the padded id resolve to the same stored conversation.
        let got_trimmed = get_conversation_in_dir(dir.path(), "abc").unwrap();
        let got_padded = get_conversation_in_dir(dir.path(), "  abc  ").unwrap();
        assert_eq!(got_trimmed.id, "abc");
        assert_eq!(got_padded.id, "abc");
    }

    #[test]
    fn save_sets_created_at_when_new_and_preserves_it_on_update() {
        let dir = test_dir("timestamps");
        let conversation = Conversation {
            id: "ts".to_string(),
            title: "T".to_string(),
            created_at: 0,
            updated_at: 0,
            params: ConversationParams::default(),
            messages: json!([{"role": "user", "content": "first"}]),
        };
        let saved = save_conversation_in_dir(dir.path(), conversation).unwrap();
        let created_at = saved.created_at;
        assert!(created_at > 0);

        // Re-save with createdAt == 0: the original creation time must be preserved.
        let update = Conversation {
            id: "ts".to_string(),
            title: "T2".to_string(),
            created_at: 0,
            updated_at: 0,
            params: ConversationParams::default(),
            messages: json!([
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"}
            ]),
        };
        let updated = save_conversation_in_dir(dir.path(), update).unwrap();
        assert_eq!(updated.created_at, created_at);
        assert!(updated.updated_at >= saved.updated_at);
    }

    #[test]
    fn save_upserts_existing_conversation() {
        let dir = test_dir("upsert");
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "u".to_string(),
                title: "Old".to_string(),
                created_at: 100,
                updated_at: 100,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "a"}]),
            },
        )
        .unwrap();
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "u".to_string(),
                title: "New".to_string(),
                created_at: 100,
                updated_at: 100,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "b"}]),
            },
        )
        .unwrap();

        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        let got = get_conversation_in_dir(dir.path(), "u").unwrap();
        assert_eq!(got.title, "New");
        assert_eq!(got.messages, json!([{"role": "user", "content": "b"}]));
    }

    #[test]
    fn save_writes_metadata_sidecar() {
        let dir = test_dir("sidecar-write");
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "sc".to_string(),
                title: "Sidecar".to_string(),
                created_at: 0,
                updated_at: 0,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "hello"}]),
            },
        )
        .unwrap();
        assert!(conversation_file_path(dir.path(), "sc").exists(), "body file must exist");
        assert!(
            conversation_meta_path(dir.path(), "sc").exists(),
            "sidecar file must exist after save"
        );
        let meta = read_meta_file(&conversation_meta_path(dir.path(), "sc")).unwrap();
        assert_eq!(meta.id, "sc");
        assert_eq!(meta.title, "Sidecar");
        assert_eq!(meta.preview, "hello");
        assert_eq!(meta.message_count, 1);
    }

    #[test]
    fn list_sorts_by_updated_at_desc_and_omits_bodies() {
        let dir = test_dir("list");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "a".to_string(),
                title: "A".to_string(),
                created_at: 100,
                updated_at: 100,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "first A"}]),
            },
        );
        write_raw_conversation(
            &dir,
            Conversation {
                id: "b".to_string(),
                title: "B".to_string(),
                created_at: 200,
                updated_at: 300,
                params: ConversationParams::default(),
                messages: json!([
                    {"role": "user", "content": "first B"},
                    {"role": "assistant", "content": "body-secret"}
                ]),
            },
        );
        write_raw_conversation(
            &dir,
            Conversation {
                id: "c".to_string(),
                title: String::new(),
                created_at: 50,
                updated_at: 50,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "first C"}]),
            },
        );

        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].id, "b");
        assert_eq!(list[1].id, "a");
        assert_eq!(list[2].id, "c");

        // message_count derived without returning bodies
        assert_eq!(list[0].message_count, 2);
        assert_eq!(list[1].message_count, 1);
        assert_eq!(list[2].message_count, 1);

        // preview derived from the first user message
        assert_eq!(list[0].preview, "first B");
        assert_eq!(list[1].preview, "first A");

        // title derived from first user message when not user-set
        assert_eq!(list[2].title, "first C");

        // Bodies are absent from the serialized metadata (the response carries no transcript).
        let serialized = serde_json::to_string(&list[0]).unwrap();
        assert!(!serialized.contains("body-secret"));
        assert!(!serialized.contains("assistant"));
        assert!(!serialized.contains("messages"));
    }

    #[test]
    fn list_preview_truncates_long_first_user_message() {
        let dir = test_dir("preview");
        let long_text = "x".repeat(200);
        write_raw_conversation(
            &dir,
            Conversation {
                id: "long".to_string(),
                title: String::new(),
                created_at: 1,
                updated_at: 1,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": long_text}]),
            },
        );
        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        // 80 chars + the ellipsis
        assert_eq!(list[0].preview.chars().count(), PREVIEW_MAX_CHARS + 1);
        assert!(list[0].preview.ends_with('\u{2026}'));
    }

    #[test]
    fn list_handles_part_content_arrays_and_missing_text() {
        let dir = test_dir("parts");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "p".to_string(),
                title: String::new(),
                created_at: 1,
                updated_at: 1,
                params: ConversationParams::default(),
                messages: json!([
                    {"role": "user", "content": [
                        {"type": "image", "url": "data:..."},
                        {"type": "text", "text": "describe this"}
                    ]}
                ]),
            },
        );
        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list[0].preview, "describe this");
        assert_eq!(list[0].title, "describe this");
    }

    #[test]
    fn list_ignores_temp_and_non_json_files() {
        let dir = test_dir("ignore");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "keep".to_string(),
                title: "K".to_string(),
                created_at: 1,
                updated_at: 1,
                params: ConversationParams::default(),
                messages: json!([]),
            },
        );
        // stale temp files + unrelated files should not be listed or crash
        fs::write(dir.join(format!("keep{JSON_SUFFIX}{TMP_SUFFIX}")), "garbage").unwrap();
        fs::write(dir.join(format!("keep{META_SUFFIX}{TMP_SUFFIX}")), "garbage").unwrap();
        fs::write(dir.join("notes.md"), "nope").unwrap();
        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "keep");
    }

    #[test]
    fn list_does_not_read_message_bodies_in_steady_state() {
        // After save, both <id>.json and <id>.meta exist. Corrupt the body file: list must still
        // succeed purely from the sidecar, proving it never opens the full transcript.
        let dir = test_dir("no-body-read");
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "nb".to_string(),
                title: "NoBody".to_string(),
                created_at: 0,
                updated_at: 0,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "preview text"}]),
            },
        )
        .unwrap();

        // Corrupt the body AFTER the sidecar is written. If list reads bodies, this breaks it.
        fs::write(
            conversation_file_path(dir.path(), "nb"),
            "THIS IS NOT VALID JSON {{{ { { { garbage",
        )
        .unwrap();

        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1, "list must find the conversation via the sidecar");
        assert_eq!(list[0].id, "nb");
        assert_eq!(list[0].title, "NoBody");
        assert_eq!(list[0].preview, "preview text");
        assert_eq!(list[0].message_count, 1);

        // get_conversation reads the body and must now fail — proving the corruption is real and
        // the list path genuinely avoided opening the body file.
        assert!(
            get_conversation_in_dir(dir.path(), "nb").is_err(),
            "get should fail on the corrupted body, confirming list did not use it"
        );
    }

    #[test]
    fn list_handles_bodies_with_huge_messages_without_loading_them() {
        // A saved conversation with a very large body must not be read by list. We confirm by
        // checking list is fast/small and still returns the correct sidecar-derived count even
        // when the body dwarfs the sidecar.
        let dir = test_dir("huge-body");
        let huge = "Z".repeat(2_000_000);
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "big".to_string(),
                title: "Big".to_string(),
                created_at: 0,
                updated_at: 0,
                params: ConversationParams::default(),
                messages: json!([
                    {"role": "user", "content": "tiny preview"},
                    {"role": "assistant", "content": huge}
                ]),
            },
        )
        .unwrap();
        let body_size = fs::metadata(conversation_file_path(dir.path(), "big")).unwrap().len();
        let meta_size = fs::metadata(conversation_meta_path(dir.path(), "big")).unwrap().len();
        assert!(
            meta_size < body_size / 1000,
            "sidecar ({meta_size}B) should be far smaller than body ({body_size}B)"
        );

        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].message_count, 2);
        assert_eq!(list[0].preview, "tiny preview");
    }

    #[test]
    fn list_migrates_legacy_body_without_sidecar() {
        // A body file written with no sidecar (legacy/old format) is read once, derived, and a
        // sidecar is persisted so the next list does not touch the body again.
        let dir = test_dir("legacy");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "legacy".to_string(),
                title: "Legacy".to_string(),
                created_at: 5,
                updated_at: 7,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "legacy preview"}]),
            },
        );
        assert!(!conversation_meta_path(dir.path(), "legacy").exists());

        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "legacy");
        assert_eq!(list[0].preview, "legacy preview");
        assert_eq!(list[0].updated_at, 7);
        assert!(
            conversation_meta_path(dir.path(), "legacy").exists(),
            "sidecar must be written after the legacy fallback list"
        );

        // Now corrupt the body: the second list must still succeed via the freshly-written sidecar.
        fs::write(conversation_file_path(dir.path(), "legacy"), "GARBAGE").unwrap();
        let list2 = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list2.len(), 1);
        assert_eq!(list2[0].id, "legacy");
        assert_eq!(list2[0].preview, "legacy preview");
    }

    #[test]
    fn rename_bumps_updated_at_and_keeps_created_at() {
        let dir = test_dir("rename");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "r1".to_string(),
                title: "Old".to_string(),
                created_at: 1000,
                updated_at: 1000,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "hi"}]),
            },
        );
        let meta = rename_conversation_in_dir(dir.path(), "r1", "New Title").unwrap();
        assert_eq!(meta.title, "New Title");
        assert_eq!(meta.created_at, 1000);
        assert!(meta.updated_at > 1000);

        let stored = get_conversation_in_dir(dir.path(), "r1").unwrap();
        assert_eq!(stored.title, "New Title");
        assert_eq!(stored.updated_at, meta.updated_at);
        // rename also refreshes the sidecar so list reflects the new title without reading the body.
        let list = list_conversations_in_dir(dir.path()).unwrap();
        assert_eq!(list[0].title, "New Title");
    }

    #[test]
    fn rename_missing_id_errors() {
        let dir = test_dir("rename-missing");
        assert!(rename_conversation_in_dir(dir.path(), "nope", "X").is_err());
    }

    #[test]
    fn delete_removes_conversation_and_sidecar() {
        let dir = test_dir("delete");
        save_conversation_in_dir(
            &dir,
            Conversation {
                id: "d1".to_string(),
                title: "D".to_string(),
                created_at: 0,
                updated_at: 0,
                params: ConversationParams::default(),
                messages: json!([{"role": "user", "content": "x"}]),
            },
        )
        .unwrap();
        assert!(conversation_file_path(dir.path(), "d1").exists());
        assert!(conversation_meta_path(dir.path(), "d1").exists());

        delete_conversation_in_dir(dir.path(), "d1").unwrap();
        assert!(!conversation_file_path(dir.path(), "d1").exists(), "body must be removed");
        assert!(!conversation_meta_path(dir.path(), "d1").exists(), "sidecar must be removed");
        assert!(get_conversation_in_dir(dir.path(), "d1").is_err());
        assert!(list_conversations_in_dir(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn delete_is_idempotent_for_missing_id() {
        let dir = test_dir("delete-idempotent");
        assert!(delete_conversation_in_dir(dir.path(), "ghost").is_ok());
    }

    #[test]
    fn get_missing_id_errors() {
        let dir = test_dir("get-missing");
        let error = get_conversation_in_dir(dir.path(), "missing").unwrap_err();
        assert!(error.contains("was not found"), "unexpected error: {error}");
    }

    fn empty_conversation(id: &str) -> Conversation {
        Conversation {
            id: id.to_string(),
            title: String::new(),
            created_at: 0,
            updated_at: 0,
            params: ConversationParams::default(),
            messages: json!([]),
        }
    }

    /// Writes a conversation body directly, bypassing `save_conversation_in_dir` (so no sidecar
    /// is created). Used to exercise the legacy-fallback list path and to seed pre-sidecar state.
    fn write_raw_conversation(dir: &Path, conversation: Conversation) {
        let path = conversation_file_path(dir, &conversation.id);
        write_json_atomic(&path, &conversation).unwrap();
    }

    fn test_dir(label: &str) -> TempDir {
        TempDir::new(&format!("conversations-{label}"))
    }
}
