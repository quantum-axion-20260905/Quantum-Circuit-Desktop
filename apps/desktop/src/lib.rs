use reqwest::blocking::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{path::PathBuf, process::{Child, Command, Stdio}, sync::Mutex, time::Duration};
use tauri::{Manager, RunEvent, State};

struct AgentState { child: Option<Child>, port: u16, token: String }
struct AppState { db: Mutex<Option<Connection>>, agent: Mutex<AgentState> }

#[allow(dead_code)]
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
      CREATE TABLE IF NOT EXISTS studies (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL, request TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
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

fn find_python_command(repo_root: &std::path::Path) -> (String, Vec<String>) {
    if let Ok(cmd) = std::env::var("QC_AGENT_COMMAND") {
        return (cmd, vec![]);
    }
    // Check local virtual environments first
    let win_venv = repo_root.join(".venv").join("Scripts").join("python.exe");
    if win_venv.exists() {
        return (win_venv.to_string_lossy().to_string(), vec!["agent/app.py".into()]);
    }
    let unix_venv = repo_root.join(".venv").join("bin").join("python");
    if unix_venv.exists() {
        return (unix_venv.to_string_lossy().to_string(), vec!["agent/app.py".into()]);
    }
    // Fall back to system python
    ("python".into(), vec!["agent/app.py".into()])
}

#[tauri::command]
fn start_compute_agent(state: State<'_, AppState>) -> Result<Value, String> {
    let mut agent = state.agent.lock().map_err(|e| e.to_string())?;
    if let Some(is_running) = agent.child.as_mut().map(|child| matches!(child.try_wait(), Ok(None))) {
        if is_running {
            return Ok(json!({"running": true, "port": agent.port}));
        }
        agent.child = None;
    }
    let port = portpicker::pick_unused_port().ok_or("no free local port")?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (program, args) = find_python_command(&repo_root);

    let mut cmd = Command::new(&program);
    cmd.current_dir(&repo_root);
    if !args.is_empty() {
        cmd.args(&args);
    }
    let child = cmd
        .env("QC_AGENT_PORT", port.to_string())
        .env("QC_AGENT_TOKEN", &agent.token)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python compute agent ('{program}'): {e}. Please ensure Python 3.11+ is installed or set QC_AGENT_COMMAND."))?;
    agent.child = Some(child);
    agent.port = port;
    let client = Client::builder().timeout(Duration::from_millis(500)).build().map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(12) {
        if client.get(format!("http://127.0.0.1:{port}/health")).send().is_ok() {
            return Ok(json!({"running": true, "port": port}));
        }
        // Check if child exited prematurely
        if let Some(child_ref) = agent.child.as_mut() {
            if let Ok(Some(status)) = child_ref.try_wait() {
                return Err(format!("Python agent exited unexpectedly with status {status}. Verify Python environment and dependencies: pip install -r agent/requirements.txt"));
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err("Compute agent startup timed out. Run 'python agent/app.py' in terminal to view startup logs.".into())
}


#[tauri::command]
fn stop_compute_agent(state: State<'_, AppState>) -> Result<Value, String> {
    let mut agent = state.agent.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = agent.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    Ok(json!({"stopped": true}))
}

#[tauri::command]
fn agent_request(state: State<'_, AppState>, method: String, path: String, payload: Option<Value>) -> Result<Value, String> {
    let agent = state.agent.lock().map_err(|e| e.to_string())?;
    let client = Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| e.to_string())?;
    let url = format!("http://127.0.0.1:{}{path}", agent.port);
    let mut req = match method.to_uppercase().as_str() {
        "POST" => client.post(&url),
        "GET" => client.get(&url),
        other => return Err(format!("unsupported method: {other}")),
    };
    if !agent.token.is_empty() { req = req.header("x-qc-agent-token", &agent.token); }
    if let Some(data) = payload { req = req.json(&data); }
    let res = req.send().map_err(|e| format!("request to agent failed: {e}"))?;
    let status = res.status();
    let body: Value = res.json().unwrap_or_else(|_| json!({"error": "invalid json from agent"}));
    if !status.is_success() { return Err(format!("agent responded with status {status}: {body}")); }
    Ok(body)
}

#[tauri::command]
fn get_hardware_capabilities(state: State<'_, AppState>) -> Result<Value, String> {
    agent_request(state, "GET".into(), "/capabilities".into(), None)
}

#[tauri::command]
fn run_preflight(state: State<'_, AppState>, payload: Value) -> Result<Value, String> {
    agent_request(state, "POST".into(), "/jobs/preflight".into(), Some(payload))
}

#[tauri::command]
fn submit_run(state: State<'_, AppState>, payload: Value) -> Result<Value, String> {
    agent_request(state, "POST".into(), "/async/jobs".into(), Some(payload))
}

#[tauri::command]
fn get_run_status(state: State<'_, AppState>, job_id: String) -> Result<Value, String> {
    agent_request(state, "GET".into(), format!("/async/jobs/{job_id}"), None)
}

#[tauri::command]
fn cancel_run(state: State<'_, AppState>, job_id: String) -> Result<Value, String> {
    agent_request(state, "POST".into(), format!("/async/jobs/{job_id}/cancel"), None)
}

#[tauri::command]
fn save_circuit(state: State<'_, AppState>, project_name: String, qasm: String, metadata: Value) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let created = now();
    let project_id: i64 = conn.query_row("SELECT id FROM projects WHERE name = ?1", params![project_name], |r| r.get(0)).unwrap_or_else(|_| {
        conn.execute("INSERT INTO projects (name, created_at) VALUES (?1, ?2)", params![project_name, created]).unwrap();
        conn.last_insert_rowid()
    });
    conn.execute("INSERT INTO circuit_versions (project_id, qasm, metadata, created_at) VALUES (?1, ?2, ?3, ?4)", params![project_id, qasm, metadata.to_string(), created]).map_err(|e| e.to_string())?;
    let version_id = conn.last_insert_rowid();
    Ok(json!({"project_id": project_id, "version_id": version_id, "saved_at": created}))
}

#[tauri::command]
fn load_circuit(state: State<'_, AppState>, project_name: String) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let row = conn.query_row("SELECT cv.qasm, cv.metadata, cv.created_at, cv.id FROM circuit_versions cv JOIN projects p ON p.id = cv.project_id WHERE p.name = ?1 ORDER BY cv.id DESC LIMIT 1", params![project_name], |r| {
        Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,i64>(3)?))
    }).map_err(|e| e.to_string())?;
    let metadata: Value = serde_json::from_str(&row.1).map_err(|e| e.to_string())?;
    Ok(json!({"version_id": row.3, "qasm": row.0, "metadata": metadata, "created_at": row.2}))
}

#[tauri::command]
fn create_run(state: State<'_, AppState>, version_id: Option<i64>, kind: String, request: Value) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let created = now();
    conn.execute("INSERT INTO runs (version_id, kind, status, request, created_at) VALUES (?1, ?2, 'queued', ?3, ?4)", params![version_id, kind, request.to_string(), created]).map_err(|e| e.to_string())?;
    Ok(json!({"run_id": conn.last_insert_rowid(), "status": "queued", "created_at": created}))
}

#[tauri::command]
fn record_run(state: State<'_, AppState>, run_id: i64, status: String, result: Option<Value>, error: Option<String>) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let finished = now();
    let res_text = result.map(|v| v.to_string()).unwrap_or_else(|| "{}".into());
    let err_text = error.unwrap_or_default();
    conn.execute("UPDATE runs SET status = ?1, result = ?2, error = ?3, finished_at = ?4 WHERE id = ?5", params![status, res_text, err_text, finished, run_id]).map_err(|e| e.to_string())?;
    Ok(json!({"updated": true, "finished_at": finished}))
}

#[tauri::command]
fn list_runs(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let mut stmt = conn.prepare("SELECT id, version_id, kind, status, request, result, error, created_at, finished_at FROM runs ORDER BY id DESC LIMIT 50").map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| Ok(json!({
        "id": r.get::<_,i64>(0)?, "version_id": r.get::<_,Option<i64>>(1)?, "kind": r.get::<_,String>(2)?,
        "status": r.get::<_,String>(3)?, "request": r.get::<_,String>(4)?, "result": r.get::<_,String>(5)?,
        "error": r.get::<_,String>(6)?, "created_at": r.get::<_,String>(7)?, "finished_at": r.get::<_,Option<String>>(8)?
    }))).map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())
}

#[tauri::command]
fn record_study(
    state: State<'_, AppState>,
    kind: String,
    label: String,
    status: String,
    request: Value,
    result: Option<Value>,
    error: Option<String>,
    started_at: Option<String>,
) -> Result<Value, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let created = now();
    let finished = if status == "running" { None } else { Some(now()) };
    let res_text = result.map(|v| v.to_string()).unwrap_or_else(|| "{}".into());
    let err_text = error.unwrap_or_default();
    conn.execute(
        "INSERT INTO studies (kind, label, status, request, result, error, created_at, started_at, finished_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            kind,
            label,
            status,
            request.to_string(),
            res_text,
            err_text,
            created,
            started_at,
            finished,
        ],
    ).map_err(|e| e.to_string())?;
    Ok(json!({"id": conn.last_insert_rowid(), "status": status, "created_at": created}))
}

#[tauri::command]
fn list_studies(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let conn = db.as_ref().ok_or("database unavailable")?;
    let mut stmt = conn.prepare("SELECT id,kind,label,status,request,result,error,created_at,started_at,finished_at FROM studies ORDER BY id DESC").map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| Ok(json!({
        "id": r.get::<_,i64>(0)?, "kind": r.get::<_,String>(1)?, "label": r.get::<_,String>(2)?,
        "status": r.get::<_,String>(3)?, "request": r.get::<_,String>(4)?, "result": r.get::<_,String>(5)?,
        "error": r.get::<_,String>(6)?, "created_at": r.get::<_,String>(7)?,
        "started_at": r.get::<_,Option<String>>(8)?, "finished_at": r.get::<_,Option<String>>(9)?
    }))).map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())
}

#[tauri::command]
fn export_project(state: State<'_, AppState>) -> Result<Value, String> { Ok(json!({"format":"qc-project-v1","runs":list_runs(state.clone())?,"studies":list_studies(state)?})) }

pub fn run() {
    tauri::Builder::default().setup(|app| {
        let dir = app.path().app_data_dir()?;
        std::fs::create_dir_all(&dir)?;
        let db = init_db(dir.join("quantum-circuit.sqlite"))?;
        let token = format!("qc-{}", std::process::id());
        app.manage(AppState { db: Mutex::new(Some(db)), agent: Mutex::new(AgentState { child: None, port: 8788, token }) });
        Ok(())
    }).invoke_handler(tauri::generate_handler![get_app_status, start_compute_agent, stop_compute_agent, agent_request, get_hardware_capabilities, run_preflight, submit_run, get_run_status, cancel_run, save_circuit, load_circuit, create_run, record_run, record_study, list_runs, list_studies, export_project]).build(tauri::generate_context!()).expect("error while building Quantum Circuit Desktop").run(|app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<AppState>() {
                if let Ok(mut agent) = state.agent.lock() {
                    if let Some(mut child) = agent.child.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        }
    });
}
