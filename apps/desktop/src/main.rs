#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use reqwest::blocking::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{path::PathBuf, process::{Child, Command, Stdio}, sync::Mutex};
use tauri::{Manager, State};

struct AgentState { child: Option<Child>, port: u16, token: String }
struct AppState { db: Mutex<Option<Connection>>, agent: Mutex<AgentState> }

#[derive(Debug, Serialize, Deserialize)]
struct RunRequest { kind: String, circuit: Value, config: Value }

fn init_db(path: PathBuf) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(r#"
      PRAGMA foreign_keys = ON;
      CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS circuit_versions (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, qasm TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, version_id INTEGER, kind TEXT NOT NULL, status TEXT NOT NULL, request TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, finished_at TEXT);
      CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS experiment_groups (id INTEGER PRIMARY KEY, name TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS experiment_items (id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL, run_id INTEGER, request TEXT NOT NULL, status TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, project_id INTEGER, body TEXT NOT NULL, updated_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS hardware_snapshots (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
    "#)?;
    Ok(conn)
}

fn now() -> String { chrono::Utc::now().to_rfc3339() }

#[tauri::command]
fn get_app_status(state: State<'_, AppState>) -> Value {
    let agent = state.agent.lock().unwrap();
    json!({"ok": true, "agent_port": agent.port, "agent_running": agent.child.is_some(), "token_configured": !agent.token.is_empty()})
}

#[tauri::command]
fn start_compute_agent(state: State<'_, AppState>) -> Result<Value, String> {
    let mut agent = state.agent.lock().map_err(|e| e.to_string())?;
    if agent.child.is_some() { return Ok(json!({"running": true, "port": agent.port})); }
    let port = portpicker::pick_unused_port().ok_or("no free local port")?;
    let command = std::env::var("QC_AGENT_COMMAND").unwrap_or_else(|_| "python".into());
    let mut cmd = Command::new(command);
    if std::env::var("QC_AGENT_COMMAND").is_err() { cmd.args(["agent/app.py"]); }
    let child = cmd.env("QC_AGENT_PORT", port.to_string()).env("QC_AGENT_TOKEN", &agent.token).stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e| format!("agent start failed: {e}"))?;
    agent.port = port;
    agent.child = Some(child);
    Ok(json!({"running": true, "port": port}))
}

#[tauri::command]
fn stop_compute_agent(state: State<'_, AppState>) -> Result<Value, String> {
    let mut agent = state.agent.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = agent.child.take() { let _ = child.kill(); let _ = child.wait(); }
    Ok(json!({"running": false}))
}

fn agent_get(state: &State<'_, AppState>, path: &str) -> Result<Value, String> {
    let agent = state.agent.lock().map_err(|e| e.to_string())?;
    let url = format!("http://127.0.0.1:{}{}", agent.port, path);
    Client::new().get(url).header("x-qc-agent-token", &agent.token).send().map_err(|e| e.to_string())?.json().map_err(|e| e.to_string())
}

fn agent_post(state: &State<'_, AppState>, path: &str, body: Value) -> Result<Value, String> {
    let agent = state.agent.lock().map_err(|e| e.to_string())?;
    let url = format!("http://127.0.0.1:{}{}", agent.port, path);
    Client::new().post(url).header("x-qc-agent-token", &agent.token).json(&body).send().map_err(|e| e.to_string())?.json().map_err(|e| e.to_string())
}

#[tauri::command]
fn get_hardware_capabilities(state: State<'_, AppState>) -> Result<Value, String> { agent_get(&state, "/capabilities") }

#[tauri::command]
fn run_preflight(state: State<'_, AppState>, payload: Value) -> Result<Value, String> { agent_post(&state, "/jobs/preflight", payload) }

#[tauri::command]
fn submit_run(state: State<'_, AppState>, request: RunRequest) -> Result<Value, String> {
    let mut body = request.circuit.as_object().cloned().ok_or("circuit must be an object")?;
    if let Some(config) = request.config.as_object() { for (k, v) in config { body.insert(k.clone(), v.clone()); } }
    body.insert("backend".into(), json!(if request.kind == "sample" { "reference" } else { "auto" }));
    agent_post(&state, "/jobs/run", Value::Object(body))
}

#[tauri::command]
fn get_run_status(state: State<'_, AppState>, job_id: String) -> Result<Value, String> { agent_get(&state, &format!("/async/jobs/{job_id}")) }

#[tauri::command]
fn cancel_run(state: State<'_, AppState>, job_id: String) -> Result<Value, String> { agent_post(&state, &format!("/async/jobs/{job_id}/cancel"), json!({})) }

#[tauri::command]
fn save_circuit(state: State<'_, AppState>, circuit: Value) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let created = now();
    conn.execute("INSERT OR IGNORE INTO projects(id,name,created_at) VALUES(1,'Default Project',?1)", params![created]).map_err(|e| e.to_string())?;
    conn.execute("INSERT INTO circuit_versions(project_id,qasm,metadata,created_at) VALUES(1,?1,?2,?3)", params![circuit.get("qasm").and_then(Value::as_str).unwrap_or(""), circuit.to_string(), created]).map_err(|e| e.to_string())?;
    Ok(json!({"id": conn.last_insert_rowid(), "project_id": 1, "created_at": created}))
}

#[tauri::command]
fn load_circuit(state: State<'_, AppState>, version_id: Option<i64>) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let mut stmt = conn.prepare("SELECT id,qasm,metadata,created_at FROM circuit_versions WHERE (?1 IS NULL OR id=?1) ORDER BY id DESC LIMIT 1").map_err(|e| e.to_string())?;
    stmt.query_row(params![version_id], |r| Ok(json!({"id":r.get::<_,i64>(0)?,"qasm":r.get::<_,String>(1)?,"metadata":serde_json::from_str::<Value>(&r.get::<_,String>(2)?).unwrap_or(json!({})),"created_at":r.get::<_,String>(3)?}))).map_err(|e| e.to_string())
}

#[tauri::command]
fn create_run(state: State<'_, AppState>, request: Value) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let created = now();
    conn.execute("INSERT INTO runs(kind,status,request,result,error,created_at) VALUES(?1,'queued',?2,'{}','',?3)", params![request.get("kind").and_then(Value::as_str).unwrap_or("unknown"), request.to_string(), created]).map_err(|e| e.to_string())?;
    Ok(json!({"id": conn.last_insert_rowid(), "status": "queued", "created_at": created}))
}

#[tauri::command]
fn list_runs(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let mut stmt = conn.prepare("SELECT id,kind,status,request,result,error,created_at FROM runs ORDER BY id DESC").map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| Ok(json!({"id":r.get::<_,i64>(0)?,"kind":r.get::<_,String>(1)?,"status":r.get::<_,String>(2)?,"request":r.get::<_,String>(3)?,"result":r.get::<_,String>(4)?,"error":r.get::<_,String>(5)?,"created_at":r.get::<_,String>(6)?}))).map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())
}

#[tauri::command]
fn export_project(state: State<'_, AppState>) -> Result<Value, String> { Ok(json!({"format":"qc-project-v1","runs":list_runs(state)?})) }

fn main() {
    tauri::Builder::default().setup(|app| {
        let dir = app.path().app_data_dir()?;
        std::fs::create_dir_all(&dir)?;
        let db = init_db(dir.join("quantum-circuit.sqlite"))?;
        let token = format!("qc-{}", std::process::id());
        app.manage(AppState { db: Mutex::new(Some(db)), agent: Mutex::new(AgentState { child: None, port: 8788, token }) });
        Ok(())
    }).invoke_handler(tauri::generate_handler![get_app_status, start_compute_agent, stop_compute_agent, get_hardware_capabilities, run_preflight, submit_run, get_run_status, cancel_run, save_circuit, load_circuit, create_run, list_runs, export_project]).run(tauri::generate_context!()).expect("error while running Quantum Circuit Desktop");
}
