#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chatworks::engine::{
    EngineHandle, EngineStatus, GenerateRequest, GenerateResponse, LoadModelRequest,
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

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let engine = EngineHandle::spawn();
            let server = OpenAiServerHandle::new();
            if let Err(error) = server.start(OpenAiServerConfig::default(), engine.clone()) {
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running the ChatWorks desktop shell");
}
