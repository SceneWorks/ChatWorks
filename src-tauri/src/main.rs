#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chatworks::app_settings::{
    api_auth_token_present, clear_api_auth_token, load_app_settings as load_app_settings_inner,
    read_api_auth_token, save_api_auth_token, save_app_settings as save_app_settings_inner,
    AppSettings,
};
use chatworks::conversations::{
    delete_conversation as delete_conversation_inner, get_conversation as get_conversation_inner,
    list_conversations as list_conversations_inner,
    rename_conversation as rename_conversation_inner, save_conversation as save_conversation_inner,
    Conversation, ConversationMetadata,
};
use chatworks::engine::KvCacheQuantRequest;
use chatworks::engine::{
    EngineHandle, EngineStatus, GenerateRequest, GenerateResponse, LoadModelRequest,
};
use chatworks::model_registry::{
    adopt_cached_hf_model as adopt_cached_hf_model_inner, clear_hf_token as clear_hf_token_inner,
    hf_token_status as hf_token_status_inner, import_hf_model as import_hf_model_inner,
    list_cached_hf_models as list_cached_hf_models_inner,
    list_registered_models as list_registered_models_inner,
    load_registered_model as load_registered_model_inner, set_hf_token as set_hf_token_inner,
    set_model_kv_cache_quant as set_model_kv_cache_quant_inner, AdoptCachedModelRequest,
    CachedModelCandidate, HfTokenStatus, ImportHfModelRequest, ModelRegistry, SetHfTokenRequest,
};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle, OpenAiServerStatus};
use tauri::{AppHandle, Emitter, Manager, State};

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

#[tauri::command]
fn start_openai_server(
    engine: State<'_, EngineHandle>,
    server: State<'_, OpenAiServerHandle>,
    config: OpenAiServerConfig,
) -> Result<OpenAiServerStatus, String> {
    server.start(config, engine.inner().clone())
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

#[tauri::command]
fn save_app_settings(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    server: State<'_, OpenAiServerHandle>,
    settings: AppSettings,
    api_auth_token: Option<String>,
) -> Result<(AppSettings, OpenAiServerStatus), String> {
    let mut settings = settings.normalized()?;
    if let Some(token) = api_auth_token {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            clear_api_auth_token()?;
            settings.server.auth_enabled = false;
        } else {
            save_api_auth_token(trimmed)?;
        }
    } else if settings.server.auth_enabled && !api_auth_token_present() {
        return Err("API auth token must be saved before enabling auth".to_string());
    }

    save_app_settings_inner(&app, &settings)?;
    let status = start_server_from_settings(&settings, &engine, &server)?;
    Ok((settings, status))
}

#[tauri::command]
fn clear_api_auth() -> Result<bool, String> {
    clear_api_auth_token()?;
    Ok(api_auth_token_present())
}

#[tauri::command]
fn api_auth_status() -> bool {
    api_auth_token_present()
}

#[tauri::command]
fn api_auth_token() -> Result<Option<String>, String> {
    read_api_auth_token().map_err(|error| error.to_string())
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
) -> Result<EngineStatus, String> {
    load_registered_model_inner(&app, &engine, model_id)
}

/// Set (or clear) a model's KV-cache quantization (sc-8533). `kv_cache_quant: null` ⇒ dense. If the
/// model is currently loaded it reloads with the new setting; an unsupported backend/model returns
/// an error string the UI surfaces as a "not supported" state.
#[tauri::command]
fn set_model_kv_cache_quant(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    model_id: String,
    kv_cache_quant: Option<KvCacheQuantRequest>,
) -> Result<ModelRegistry, String> {
    set_model_kv_cache_quant_inner(&app, &engine, model_id, kv_cache_quant)
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
fn hf_token_status() -> HfTokenStatus {
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

fn server_config_from_settings(settings: &AppSettings) -> OpenAiServerConfig {
    OpenAiServerConfig {
        host: settings.server.host.clone(),
        port: settings.server.port,
        allow_lan: settings.server.allow_lan,
        auth_token: if settings.server.auth_enabled {
            read_api_auth_token().ok().flatten()
        } else {
            None
        },
        sampling_defaults: settings.sampling.clone(),
    }
}

fn start_server_from_settings(
    settings: &AppSettings,
    engine: &EngineHandle,
    server: &OpenAiServerHandle,
) -> Result<OpenAiServerStatus, String> {
    server.start(server_config_from_settings(settings), engine.clone())
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let engine = EngineHandle::spawn();
            let server = OpenAiServerHandle::new();
            let settings = load_app_settings_inner(app.handle()).unwrap_or_else(|error| {
                eprintln!("ChatWorks settings failed to load: {error}");
                AppSettings::default()
            });
            if let Err(error) = start_server_from_settings(&settings, &engine, &server) {
                eprintln!("ChatWorks OpenAI server failed to start: {error}");
            }
            app.manage(engine);
            app.manage(server);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_model,
            unload_model,
            engine_status,
            stream_completion,
            start_openai_server,
            stop_openai_server,
            openai_server_status,
            load_app_settings,
            save_app_settings,
            clear_api_auth,
            api_auth_status,
            api_auth_token,
            list_registered_models,
            import_hf_model,
            list_cached_hf_models,
            adopt_cached_hf_model,
            load_registered_model,
            set_model_kv_cache_quant,
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
        .run(tauri::generate_context!())
        .expect("error while running the ChatWorks desktop shell");
}
