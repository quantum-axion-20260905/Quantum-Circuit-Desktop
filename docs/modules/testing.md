# Testing strategy

## Acceptance tests (system-level)
- Agent GPU detection ok
- TN amplitude sanity (Bell, GHZ)
- Web can call agent and render results
- IR round-trip (export/import)

## Unit tests (module-level)
- IR validators and converters
- Agent backend determinism with fixed seeds
- Editor model operations (undo/redo, move, insert)

