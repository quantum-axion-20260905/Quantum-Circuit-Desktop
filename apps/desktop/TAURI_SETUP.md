# Tauri setup (Windows)

Rust is not installed in this environment yet, so the desktop shell is scaffolded as docs only.

## Install Rust

Install via rustup, then restart the terminal.

## Initialize Tauri

From repo root:

```powershell
$npmCli = Join-Path (Split-Path (Get-Command node).Source) 'node_modules\npm\bin\npm-cli.js'
node $npmCli -w apps/web run build
```

Then create a new Tauri app that points at the web build output (or dev server), and add IPC calls to:

- check `http://127.0.0.1:8788/health`
- submit jobs to `http://127.0.0.1:8788/jobs/*`
