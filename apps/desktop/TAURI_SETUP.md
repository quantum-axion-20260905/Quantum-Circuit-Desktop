# Tauri setup (Windows)

The Tauri shell is implemented under `apps/desktop`. Rust is required on the
development machine to compile or package it.

## Install Rust

Install via rustup, then restart the terminal.

## Build the desktop shell

From repo root:

```powershell
$npmCli = Join-Path (Split-Path (Get-Command node).Source) 'node_modules\npm\bin\npm-cli.js'
node $npmCli -w apps/web run build
node $npmCli -w apps/desktop run build
```

The shell starts the local agent, allocates a free port, stores projects and runs in SQLite, and exposes typed command boundaries to the React renderer. The renderer should use `src/lib/desktop.ts`; direct agent calls remain only as browser/dev fallback.
