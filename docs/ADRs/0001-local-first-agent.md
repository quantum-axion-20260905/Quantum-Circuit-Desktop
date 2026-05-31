# ADR-0001: Local-first compute via Agent

## Context
Server-side GPU compute xarajati tez oshadi va ilmiy workflow’da ko‘p joblar bo‘ladi.

## Decision
Default compute local Agent’da bo‘ladi (RTX/GPU). Server compute optional/self-host.

## Alternatives
- Full server compute
- Full browser compute (WASM/WebGPU)

## Consequences
- Desktop/Tauri va agent lifecycle muhim bo‘ladi.
- Reproducibility loglar client-side ham to‘planadi.

