# DataLoader-RS Torch Parity Roadmap

Goal: close the highest-value gaps with PyTorch `DataLoader`, while keeping this crate's strengths (typed API, low overhead, predictable performance).

How to use this roadmap:
- Each section below is a standalone case.
- We implement and release each case independently.
- Every case has scope, design direction, and clear acceptance checks.

---

## Case 1 - Iterable Dataset in DataLoader

### Why
Current `DataLoader` is map-style (`get(index)` + `Sampler`). Torch supports iterable-style loading for streams and sequential data sources.

### Scope
- Add `IterableDataLoader` (or a mode on `DataLoader`) consuming `IterableDataset::iter()`.
- Support batching, `drop_last`, `collator`, `prefetch_depth`, and `num_workers`.
- Keep map-style and iterable-style APIs explicit and type-safe.

### Design direction
- Avoid forcing `Sampler` for iterable datasets.
- For `num_workers > 0`, shard or partition stream consumption deterministically (worker-id based strategy).
- Preserve bounded-channel backpressure behavior.

### Acceptance checks
- Stream dataset yields full epoch batches correctly.
- `drop_last` behavior matches map-style semantics.
- Parallel iterable loading is deterministic when seeded.
- No deadlocks on early iterator drop.

---

## Case 3 - Worker Lifecycle and Seeding Hooks

### Why
Torch offers worker init controls and reproducibility ergonomics.

### Scope
- Add worker initialization hook support (e.g. callback receiving worker_id).
- Add deterministic per-worker seed derivation API.
- Expose optional persistent worker mode for repeated epochs.

### Design direction
- Worker config object:
  - `base_seed`
  - `seed_strategy`
  - `init_hook`
  - `persistent_workers`
- Keep defaults simple (off/none).

### Acceptance checks
- Reproducible multi-worker runs with fixed seed.
- Different workers receive non-overlapping seeds.
- Persistent workers reduce repeated epoch startup cost.

---

## Case 4 - Timeout and Failure Policies

### Why
Operational robustness requires explicit behavior when workers stall or fail.

### Scope
- Add optional batch receive timeout.
- Add configurable on-error behavior:
  - fail-fast
  - skip batch
  - stop epoch gracefully

### Design direction
- Policy enum in builder:
  - `ErrorPolicy::FailFast`
  - `ErrorPolicy::SkipBatch`
  - `ErrorPolicy::StopEpoch`
- Timeout is measured at iterator receive boundary.

### Acceptance checks
- Timeout surfaces deterministic error type.
- Policies produce expected iteration outcomes.
- Early-drop and timeout paths do not leak threads.

---

## Case 5 - Built-in Advanced Collation Suite

Status: Completed

### Implemented and removed from backlog
- Core `DefaultCollator` / `DefaultCollate` architecture.
- Split collator implementations into per-type files.
- Primitive / tuple / sequence / array / map support.
- Feature-gated third-party modules:
  - ndarray support
  - torch-rs support (fallible `f_stack` path).

### Why
Torch has strong default collation for nested and heterogeneous structures.

### Scope
- Completed scope:
  - advanced default collator with split per-type modules
  - tuple support extended (up to arity 6)
  - sequence, array, map, primitive coverage
  - feature-gated ndarray and torch-rs collation modules
  - integration coverage added for default and feature-gated paths

### Design direction
- Mirror the style used in `ai-dataloader` collate modules.
- Keep collators explicit and opt-in (no magic runtime reflection).

### Acceptance checks
- End-to-end tests for each collator type.
- Throughput overhead remains acceptable vs `VecCollator`.

---

## Case 6 - Device Handoff / Pinned Memory Path

### Why
Torch `pin_memory` helps CPU->GPU transfer throughput. This is key for training loops.

### Scope
- Add optional "host memory strategy" abstraction.
- Provide a pinned-host implementation behind a feature flag.
- Expose batch handoff API compatible with common Rust ML runtimes.

### Design direction
- Keep core crate portable; gate OS/device-specific behavior via features.
- Design trait-based memory handoff to avoid locking into one backend.

### Acceptance checks
- Feature-gated build compiles cleanly with and without pinned support.
- Benchmarks show transfer improvements in GPU pipelines.

---

## Case 7 - Distributed Epoch Ergonomics

### Why
Torch workflows rely on easy per-epoch reseeding for distributed samplers.

### Scope
- Extend distributed sampler with epoch setter semantics (`set_epoch` style).
- Provide helper APIs for rank/world integration in training loops.

### Design direction
- Keep deterministic shuffling:
  - seed = f(base_seed, epoch, rank)
- Maintain equal-per-rank guarantees with padding behavior.

### Acceptance checks
- Different epochs produce different rank-local orders.
- Same epoch+seed reproduces exact order.
- Coverage and balance tests pass for divisible and non-divisible dataset sizes.

---

## Case 8 - Python/PyO3 Compatibility Layer

### Why
Torch parity includes interoperability and easy adoption from Python stacks.

### Scope
- Add type-erased Python-facing loader wrapper.
- Provide Python callback-based dataset and collator adapters.
- Keep Rust-native generic API untouched.

### Design direction
- Separate crate/module for bindings (`dataloader-rs-py`).
- Strict boundary between Python object world and Rust generic core.

### Acceptance checks
- Python examples for map-style and iterable-style loading.
- Error propagation across FFI boundary is clear and stable.
- No regressions in Rust-native benchmarks when Python feature is off.

---

## Recommended implementation order

1. Case 1 - Iterable Dataset in DataLoader  
2. Case 3 - Worker Lifecycle and Seeding Hooks  
3. Case 4 - Timeout and Failure Policies  
4. Case 7 - Distributed Epoch Ergonomics  
5. Case 6 - Device Handoff / Pinned Memory Path  
6. Case 8 - Python/PyO3 Compatibility Layer

Rationale:
- Start with API and batching control (high user impact, low platform coupling).
- Then streaming and worker behavior.
- Finally platform-specific and binding-heavy work.

---

## Definition of done per case

- Public API docs updated.
- Unit + integration tests added.
- Benchmarks updated where relevant.
- Changelog entry added.
- Migration notes included for any breaking API changes.
