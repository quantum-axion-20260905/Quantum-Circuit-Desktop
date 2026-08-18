# Tauri setup (Windows)

The Tauri shell is now scaffolded under `apps/desktop`. Rust is still required on the development machine to compile it.

## Install Rust

Install via rustup, then restart the terminal.

## Initialize Tauri

From repo root:

```powershell
$npmCli = Join-Path (Split-Path (Get-Command node).Source) 'node_modules\npm\bin\npm-cli.js'
node $npmCli -w apps/web run build
```

The shell starts the local agent, allocates a free port, stores projects and runs in SQLite, and exposes typed command boundaries to the React renderer. The renderer should use `src/lib/desktop.ts`; direct agent calls remain only as browser/dev fallback.
