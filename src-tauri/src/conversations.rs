use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Manager};

const PREVIEW_MAX_CHARS: usize = 80;

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
/// Message bodies are deliberately absent so listing never returns transcript data.
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
    dir.join(format!("{id}.json"))
}

fn validate_id(id: &str) -> Result<(), String> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return Err("conversation id is required".to_string());
    }
    if id.contains('/')
        || id.contains('\\')
        || id.contains('\0')
        || id == "."
        || id == ".."
        || id.split('/').any(|segment| segment == "..")
    {
        return Err(format!("invalid conversation id {id:?}"));
    }
    Ok(())
}

fn list_conversations_in_dir(dir: &Path) -> Result<Vec<ConversationMetadata>, String> {
    let mut items = Vec::new();
    if !dir.is_dir() {
        return Ok(items);
    }
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        // Skip corrupt files so one bad entry never blanks the whole history list.
        let Ok(conversation) = read_conversation_file(&path) else {
            continue;
        };
        items.push(metadata_from(&conversation));
    }
    items.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then_with(|| a.id.cmp(&b.id)));
    Ok(items)
}

fn get_conversation_in_dir(dir: &Path, id: &str) -> Result<Conversation, String> {
    validate_id(id)?;
    let path = conversation_file_path(dir, id);
    if !path.exists() {
        return Err(format!("conversation {id:?} was not found"));
    }
    read_conversation_file(&path)
}

fn save_conversation_in_dir(dir: &Path, mut conversation: Conversation) -> Result<Conversation, String> {
    validate_id(&conversation.id)?;
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
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
    write_conversation_atomic(&path, &conversation)?;
    Ok(conversation)
}

fn rename_conversation_in_dir(dir: &Path, id: &str, title: &str) -> Result<ConversationMetadata, String> {
    let mut conversation = get_conversation_in_dir(dir, id)?;
    conversation.title = title.to_string();
    conversation.updated_at = now_secs();
    let path = conversation_file_path(dir, id);
    write_conversation_atomic(&path, &conversation)?;
    Ok(metadata_from(&conversation))
}

fn delete_conversation_in_dir(dir: &Path, id: &str) -> Result<(), String> {
    validate_id(id)?;
    let path = conversation_file_path(dir, id);
    // Idempotent: removing a conversation that no longer exists is not an error.
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn write_conversation_atomic(path: &Path, conversation: &Conversation) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(
        &tmp,
        serde_json::to_string_pretty(conversation).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(tmp, path).map_err(|error| error.to_string())
}

fn read_conversation_file(path: &Path) -> Result<Conversation, String> {
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str::<Conversation>(&body).map_err(|error| error.to_string())
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

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn save_then_get_round_trips_messages_and_params() {
        let dir = test_dir("round-trip");
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
        let saved = save_conversation_in_dir(&dir, conversation).unwrap();
        assert!(saved.created_at > 0);
        assert!(saved.updated_at >= saved.created_at);

        let got = get_conversation_in_dir(&dir, "abc").unwrap();
        assert_eq!(got.id, "abc");
        assert_eq!(got.title, "Hello");
        assert_eq!(got.messages, messages);
        assert_eq!(got.params.temperature, params.temperature);
        assert_eq!(got.params.max_tokens, params.max_tokens);
        assert_eq!(got.created_at, saved.created_at);
        assert_eq!(got.updated_at, saved.updated_at);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_rejects_empty_id() {
        let dir = test_dir("empty-id");
        let conversation = Conversation {
            id: "   ".to_string(),
            ..empty_conversation("x")
        };
        let error = save_conversation_in_dir(&dir, conversation).unwrap_err();
        assert!(error.contains("id"), "unexpected error: {error}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_rejects_path_traversal_id() {
        let dir = test_dir("traversal-id");
        let cases = ["../escape", "a/b", "a\\b", ".", "..", "bad\u{0000}id"];
        for id in cases {
            let conversation = Conversation {
                id: id.to_string(),
                ..empty_conversation(id)
            };
            assert!(
                save_conversation_in_dir(&dir, conversation).is_err(),
                "id {id:?} should be rejected"
            );
        }
        let _ = fs::remove_dir_all(dir);
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
        let saved = save_conversation_in_dir(&dir, conversation).unwrap();
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
        let updated = save_conversation_in_dir(&dir, update).unwrap();
        assert_eq!(updated.created_at, created_at);
        assert!(updated.updated_at >= saved.updated_at);
        let _ = fs::remove_dir_all(dir);
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

        let list = list_conversations_in_dir(&dir).unwrap();
        assert_eq!(list.len(), 1);
        let got = get_conversation_in_dir(&dir, "u").unwrap();
        assert_eq!(got.title, "New");
        assert_eq!(got.messages, json!([{"role": "user", "content": "b"}]));
        let _ = fs::remove_dir_all(dir);
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

        let list = list_conversations_in_dir(&dir).unwrap();
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
        let _ = fs::remove_dir_all(dir);
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
        let list = list_conversations_in_dir(&dir).unwrap();
        assert_eq!(list.len(), 1);
        // 80 chars + the ellipsis
        assert_eq!(list[0].preview.chars().count(), PREVIEW_MAX_CHARS + 1);
        assert!(list[0].preview.ends_with('\u{2026}'));
        let _ = fs::remove_dir_all(dir);
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
        let list = list_conversations_in_dir(&dir).unwrap();
        assert_eq!(list[0].preview, "describe this");
        assert_eq!(list[0].title, "describe this");
        let _ = fs::remove_dir_all(dir);
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
        // stale temp file + unrelated file should not be listed or crash
        fs::write(dir.join("keep.json.tmp"), "garbage").unwrap();
        fs::write(dir.join("notes.md"), "nope").unwrap();
        let list = list_conversations_in_dir(&dir).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "keep");
        let _ = fs::remove_dir_all(dir);
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
        let meta = rename_conversation_in_dir(&dir, "r1", "New Title").unwrap();
        assert_eq!(meta.title, "New Title");
        assert_eq!(meta.created_at, 1000);
        assert!(meta.updated_at > 1000);

        let stored = get_conversation_in_dir(&dir, "r1").unwrap();
        assert_eq!(stored.title, "New Title");
        assert_eq!(stored.updated_at, meta.updated_at);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_missing_id_errors() {
        let dir = test_dir("rename-missing");
        assert!(rename_conversation_in_dir(&dir, "nope", "X").is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_removes_conversation() {
        let dir = test_dir("delete");
        write_raw_conversation(
            &dir,
            Conversation {
                id: "d1".to_string(),
                title: "D".to_string(),
                created_at: 1,
                updated_at: 1,
                params: ConversationParams::default(),
                messages: json!([]),
            },
        );
        delete_conversation_in_dir(&dir, "d1").unwrap();
        assert!(get_conversation_in_dir(&dir, "d1").is_err());
        assert!(list_conversations_in_dir(&dir).unwrap().is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_is_idempotent_for_missing_id() {
        let dir = test_dir("delete-idempotent");
        assert!(delete_conversation_in_dir(&dir, "ghost").is_ok());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn get_missing_id_errors() {
        let dir = test_dir("get-missing");
        let error = get_conversation_in_dir(&dir, "missing").unwrap_err();
        assert!(error.contains("was not found"), "unexpected error: {error}");
        let _ = fs::remove_dir_all(dir);
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

    fn write_raw_conversation(dir: &Path, conversation: Conversation) {
        let path = conversation_file_path(dir, &conversation.id);
        write_conversation_atomic(&path, &conversation).unwrap();
    }

    fn test_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("chatworks-conversations-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
