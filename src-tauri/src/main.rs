#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chatworks::app_settings::{
    clear_api_auth_token, load_app_settings as load_app_settings_inner, read_api_auth_token,
    resolve_api_auth_token, save_api_auth_token, save_app_settings as save_app_settings_inner,
    AppSettings,
};
use chatworks::conversations::{
    delete_conversation as delete_conversation_inner, get_conversation as get_conversation_inner,
    list_conversations as list_conversations_inner,
    rename_conversation as rename_conversation_inner, save_conversation as save_conversation_inner,
    Conversation, ConversationMetadata,
};
use chatworks::engine::{
    prepare_remote_media as prepare_remote_media_inner, EngineHandle, EngineStatus,
    GenerateRequest, GenerateResponse, LoadModelRequest, PreparedMedia,
};
use chatworks::model_registry::{
    adopt_cached_hf_model as adopt_cached_hf_model_inner, clear_hf_token as clear_hf_token_inner,
    hf_token_status as hf_token_status_inner, import_hf_model as import_hf_model_inner,
    list_cached_hf_models as list_cached_hf_models_inner,
    list_registered_models as list_registered_models_inner,
    load_registered_model as load_registered_model_inner, set_hf_token as set_hf_token_inner,
    AdoptCachedModelRequest, CachedModelCandidate, HfTokenStatus, ImportHfModelRequest,
    ModelRegistry, SetHfTokenRequest,
};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle, OpenAiServerStatus};
use tauri::{AppHandle, Emitter, Manager, State};

type ApiAuthState = std::sync::Mutex<Result<Option<String>, String>>;

#[tauri::command]
fn load_model(
    engine: State<'_, EngineHandle>,
    request: LoadModelRequest,
) -> Result<EngineStatus, String> {
    engine.load_model(request)
}

#[tauri::command]
fn unload_model(engine: State<'_, EngineHandle>) -> Result<EngineStatus, String> {
    engine.unload_model()
}

#[tauri::command]
fn engine_status(engine: State<'_, EngineHandle>) -> Result<EngineStatus, String> {
    engine.status()
}

#[tauri::command]
fn stream_completion(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    request: GenerateRequest,
) -> Result<GenerateResponse, String> {
    engine.generate(request, |event| {
        let _ = app.emit("engine://stream", event);
    })
}

/// Request cancellation of the in-flight generation (code-review F-004). Returns `true` when a
/// generation was in flight and its cancel flag was tripped; the provider stops promptly and the
/// in-progress `stream_completion` resolves with a `cancelled` finish reason.
#[tauri::command]
fn stop_generation(engine: State<'_, EngineHandle>) -> bool {
    engine.cancel()
}

type MediaPreparations =
    std::sync::Mutex<std::collections::HashMap<String, chatworks::core_llm::CancelFlag>>;

#[tauri::command]
fn begin_media_preparation(preparations: State<'_, MediaPreparations>) -> Result<String, String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .to_string();
    preparations
        .lock()
        .map_err(|e| e.to_string())?
        .insert(id.clone(), chatworks::core_llm::CancelFlag::new());
    Ok(id)
}

#[tauri::command]
fn cancel_media_preparation(
    id: String,
    preparations: State<'_, MediaPreparations>,
) -> Result<(), String> {
    if let Some(flag) = preparations.lock().map_err(|e| e.to_string())?.remove(&id) {
        flag.cancel();
    }
    Ok(())
}

#[tauri::command]
async fn prepare_remote_media(
    id: String,
    source: String,
    kind: String,
    preparations: State<'_, MediaPreparations>,
) -> Result<PreparedMedia, String> {
    let flag = preparations
        .lock()
        .map_err(|e| e.to_string())?
        .get(&id)
        .cloned()
        .ok_or("media preparation cancelled")?;
    let result = tauri::async_runtime::spawn_blocking(move || {
        prepare_remote_media_inner(source, kind, flag)
    })
    .await
    .map_err(|error| error.to_string());
    preparations.lock().map_err(|e| e.to_string())?.remove(&id);
    result?
}

#[tauri::command]
fn stop_openai_server(server: State<'_, OpenAiServerHandle>) -> Result<OpenAiServerStatus, String> {
    server.stop()
}

#[tauri::command]
fn openai_server_status(
    server: State<'_, OpenAiServerHandle>,
) -> Result<OpenAiServerStatus, String> {
    server.status()
}

#[tauri::command]
fn load_app_settings(app: AppHandle) -> Result<AppSettings, String> {
    load_app_settings_inner(&app)
}

fn save_with_credential_change(
    settings: &mut AppSettings,
    provided_token: &str,
    auth: &ApiAuthState,
    stop_server: impl FnOnce() -> Result<(), String>,
    clear_token: impl FnOnce() -> Result<(), String>,
    save_token: impl FnOnce(&str) -> Result<(), String>,
    write_settings: impl FnOnce(&AppSettings) -> Result<(), String>,
) -> Result<Option<String>, String> {
    // A failed settings write must never leave the previous bearer token live after Keychain
    // deletion or rotation. Stop first so even a failed credential operation cannot race it.
    stop_server()?;
    let trimmed = provided_token.trim();
    let token = if trimmed.is_empty() {
        clear_token()?;
        settings.server.auth_enabled = false;
        None
    } else {
        save_token(trimmed)?;
        Some(trimmed.to_string())
    };
    // The Keychain mutation has completed even if the following settings write fails.
    *auth.lock().map_err(|error| error.to_string())? = Ok(token.clone());
    write_settings(settings)?;
    Ok(token)
}

#[tauri::command]
fn save_app_settings(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    server: State<'_, OpenAiServerHandle>,
    auth: State<'_, ApiAuthState>,
    settings: AppSettings,
    api_auth_token: Option<String>,
) -> Result<(AppSettings, OpenAiServerStatus, Option<String>), String> {
    let mut settings = settings.normalized()?;
    let credential_change = api_auth_token.is_some();
    let token = if let Some(token) = api_auth_token {
        save_with_credential_change(
            &mut settings,
            &token,
            auth.inner(),
            || server.stop().map(|_| ()),
            clear_api_auth_token,
            save_api_auth_token,
            |settings| save_app_settings_inner(&app, settings),
        )?
    } else if settings.server.auth_enabled {
        let cached = auth.lock().map_err(|error| error.to_string())?.clone();
        match cached {
            Ok(Some(token)) => Some(token),
            Err(error) => {
                server.stop()?;
                return Err(error);
            }
            Ok(None) => match resolve_api_auth_token(true, read_api_auth_token) {
                Ok(token) => token,
                Err(error) => {
                    server.stop()?;
                    *auth.lock().map_err(|error| error.to_string())? = Err(error.clone());
                    return Err(error);
                }
            },
        }
    } else {
        None
    };

    if !credential_change {
        save_app_settings_inner(&app, &settings)?;
    }
    *auth.lock().map_err(|error| error.to_string())? = Ok(token.clone());
    let status = start_server_from_settings(&settings, token.clone(), &engine, &server)?;
    Ok((settings, status, token))
}

#[tauri::command]
fn api_auth_token(auth: State<'_, ApiAuthState>) -> Result<Option<String>, String> {
    auth.lock().map_err(|error| error.to_string())?.clone()
}

#[tauri::command]
fn list_registered_models(app: AppHandle) -> Result<ModelRegistry, String> {
    list_registered_models_inner(&app)
}

#[tauri::command]
async fn import_hf_model(
    app: AppHandle,
    request: ImportHfModelRequest,
) -> Result<ModelRegistry, String> {
    import_hf_model_inner(app, request).await
}

#[tauri::command]
fn list_cached_hf_models() -> Result<Vec<CachedModelCandidate>, String> {
    list_cached_hf_models_inner()
}

#[tauri::command]
fn adopt_cached_hf_model(
    app: AppHandle,
    request: AdoptCachedModelRequest,
) -> Result<ModelRegistry, String> {
    adopt_cached_hf_model_inner(&app, request)
}

#[tauri::command]
fn load_registered_model(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    model_id: String,
    projector_source: Option<String>,
) -> Result<EngineStatus, String> {
    load_registered_model_inner(&app, &engine, model_id, projector_source)
}

#[tauri::command]
fn list_builtin_tools() -> Vec<serde_json::Value> {
    chatworks::tools::builtin_tool_specs()
}

#[tauri::command]
fn execute_tool(name: String, arguments: serde_json::Value) -> Result<String, String> {
    chatworks::tools::execute_builtin_tool(&name, &arguments)
}

#[tauri::command]
fn hf_token_status() -> Result<HfTokenStatus, String> {
    hf_token_status_inner()
}

#[tauri::command]
fn set_hf_token(request: SetHfTokenRequest) -> Result<HfTokenStatus, String> {
    set_hf_token_inner(request)
}

#[tauri::command]
fn clear_hf_token() -> Result<HfTokenStatus, String> {
    clear_hf_token_inner()
}

#[tauri::command]
fn list_conversations(app: AppHandle) -> Result<Vec<ConversationMetadata>, String> {
    list_conversations_inner(&app)
}

#[tauri::command]
fn get_conversation(app: AppHandle, id: String) -> Result<Conversation, String> {
    get_conversation_inner(&app, &id)
}

#[tauri::command]
fn save_conversation(app: AppHandle, conversation: Conversation) -> Result<Conversation, String> {
    save_conversation_inner(&app, conversation)
}

#[tauri::command]
fn rename_conversation(
    app: AppHandle,
    id: String,
    title: String,
) -> Result<ConversationMetadata, String> {
    rename_conversation_inner(&app, &id, &title)
}

#[tauri::command]
fn delete_conversation(app: AppHandle, id: String) -> Result<(), String> {
    delete_conversation_inner(&app, &id)
}

fn server_config_from_settings(
    settings: &AppSettings,
    token: Option<String>,
) -> Result<OpenAiServerConfig, String> {
    if settings.server.auth_enabled && token.as_deref().unwrap_or("").trim().is_empty() {
        return Err("API auth token must be saved before enabling auth".to_string());
    }
    Ok(OpenAiServerConfig {
        host: settings.server.host.clone(),
        port: settings.server.port,
        allow_lan: settings.server.allow_lan,
        allow_local_files: settings.server.allow_local_files,
        auth_token: if settings.server.auth_enabled {
            token
        } else {
            None
        },
        sampling_defaults: settings.sampling.clone(),
    })
}

fn start_server_from_settings(
    settings: &AppSettings,
    token: Option<String>,
    engine: &EngineHandle,
    server: &OpenAiServerHandle,
) -> Result<OpenAiServerStatus, String> {
    server.start(
        server_config_from_settings(settings, token)?,
        engine.clone(),
    )
}

fn main() {
    let mut context = tauri::generate_context!();
    chatworks::profile::isolate_webviews(context.config_mut()).expect("invalid acceptance profile");
    tauri::Builder::default()
        .manage(MediaPreparations::default())
        .setup(|app| {
            let engine = EngineHandle::spawn();
            // Every finished generation (desktop or API client) pushes the served model's decode
            // status, so the decode-path view updates without polling (sc-24139).
            let handle = app.handle().clone();
            engine.observe_generations(move |status| {
                let _ = handle.emit("engine://decode", status);
            });
            let server = OpenAiServerHandle::new();
            let settings = load_app_settings_inner(app.handle());
            let token = settings
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|settings| {
                    resolve_api_auth_token(settings.server.auth_enabled, read_api_auth_token)
                });
            if let Err(error) = token.as_ref() {
                eprintln!("ChatWorks settings or API auth unavailable: {error}");
            }
            if let (Ok(settings), Ok(token)) = (settings, token.clone()) {
                if let Err(error) = start_server_from_settings(&settings, token, &engine, &server) {
                    eprintln!("ChatWorks OpenAI server failed to start: {error}");
                }
            }
            app.manage(ApiAuthState::new(token));
            app.manage(engine);
            app.manage(server);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_model,
            unload_model,
            engine_status,
            stream_completion,
            stop_generation,
            begin_media_preparation,
            cancel_media_preparation,
            prepare_remote_media,
            stop_openai_server,
            openai_server_status,
            load_app_settings,
            save_app_settings,
            api_auth_token,
            list_registered_models,
            import_hf_model,
            list_cached_hf_models,
            adopt_cached_hf_model,
            load_registered_model,
            list_builtin_tools,
            execute_tool,
            hf_token_status,
            set_hf_token,
            clear_hf_token,
            list_conversations,
            get_conversation,
            save_conversation,
            rename_conversation,
            delete_conversation,
        ])
        .run(context)
        .expect("error while running the ChatWorks desktop shell");
}

#[cfg(test)]
mod credential_tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    fn enabled_auth_cannot_build_an_unauthenticated_server() {
        for allow_local_files in [false, true] {
            let mut settings = AppSettings::default();
            settings.server.auth_enabled = true;
            settings.server.allow_local_files = allow_local_files;
            assert!(server_config_from_settings(&settings, None).is_err());
            let config = server_config_from_settings(&settings, Some("secret".into())).unwrap();
            assert_eq!(config.auth_token.as_deref(), Some("secret"));
        }
    }

    #[test]
    fn failed_settings_write_after_clear_or_rotation_leaves_server_stopped() {
        for (provided, expected) in [("", None), ("new", Some("new"))] {
            let running = Cell::new(true);
            let stored = RefCell::new(Some("old".to_string()));
            let auth = ApiAuthState::new(Ok(Some("old".to_string())));
            let mut settings = AppSettings::default();
            settings.server.auth_enabled = true;

            let result = save_with_credential_change(
                &mut settings,
                provided,
                &auth,
                || {
                    running.set(false);
                    Ok(())
                },
                || {
                    assert!(!running.get());
                    *stored.borrow_mut() = None;
                    Ok(())
                },
                |token| {
                    assert!(!running.get());
                    *stored.borrow_mut() = Some(token.to_string());
                    Ok(())
                },
                |_| {
                    assert!(!running.get());
                    assert_eq!(stored.borrow().as_deref(), expected);
                    Err("settings write failed".to_string())
                },
            );

            assert_eq!(result.unwrap_err(), "settings write failed");
            assert!(!running.get());
            assert_eq!(stored.borrow().as_deref(), expected);
            assert_eq!(auth.lock().unwrap().as_ref().unwrap().as_deref(), expected);
        }
    }

    #[test]
    fn failed_server_stop_prevents_credential_mutation() {
        let auth = ApiAuthState::new(Ok(Some("old".to_string())));
        let mut settings = AppSettings::default();
        let result = save_with_credential_change(
            &mut settings,
            "new",
            &auth,
            || Err("server stop failed".to_string()),
            || panic!("credential must not be cleared"),
            |_| panic!("credential must not be replaced"),
            |_| panic!("settings must not be written"),
        );
        assert_eq!(result.unwrap_err(), "server stop failed");
        assert_eq!(
            auth.lock().unwrap().as_ref().unwrap().as_deref(),
            Some("old")
        );
    }
}
