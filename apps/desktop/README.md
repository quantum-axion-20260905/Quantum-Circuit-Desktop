# Desktop (Tauri)

This folder contains the Tauri shell that embeds `apps/web`, starts the local
compute agent on an available loopback port, and stores circuit versions/runs
and convergence-study manifests in an app-local SQLite database. Compute jobs
remain in the Python agent; the Tauri layer owns local transport and durable
desktop history.

Prerequisites: Rust toolchain (`rustup`) and WebView2. The shell is source-
complete, but packaging still requires a machine with Rust installed and either
the development Python environment or a configured `QC_AGENT_COMMAND` binary.

