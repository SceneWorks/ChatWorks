#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chatworks::engine::{
    EngineHandle, EngineStatus, GenerateRequest, GenerateResponse, LoadModelRequest,
};
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

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(EngineHandle::spawn());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_model,
            unload_model,
            engine_status,
            stream_completion,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the ChatWorks desktop shell");
}
