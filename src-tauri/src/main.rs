#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chatworks::app_settings::{
    api_auth_token_present, clear_api_auth_token, load_app_settings as load_app_settings_inner,
    read_api_auth_token, save_api_auth_token, save_app_settings as save_app_settings_inner,
    AppSettings,
};
use chatworks::engine::{
    EngineHandle, EngineStatus, GenerateRequest, GenerateResponse, LoadModelRequest,
};
use chatworks::model_registry::{
    clear_hf_token as clear_hf_token_inner, hf_token_status as hf_token_status_inner,
    import_hf_model as import_hf_model_inner,
    list_registered_models as list_registered_models_inner,
    load_registered_model as load_registered_model_inner, set_hf_token as set_hf_token_inner,
    HfTokenStatus, ImportHfModelRequest, ModelRegistry, SetHfTokenRequest,
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
fn load_registered_model(
    app: AppHandle,
    engine: State<'_, EngineHandle>,
    model_id: String,
) -> Result<EngineStatus, String> {
    load_registered_model_inner(&app, &engine, model_id)
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
            load_registered_model,
            hf_token_status,
            set_hf_token,
            clear_hf_token,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the ChatWorks desktop shell");
}
