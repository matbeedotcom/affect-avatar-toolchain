# Spike A — `llama-cpp-4` Control Vector Binding Verification

**Status**: Resolved (2026-05-04). **Off the Phase 5 critical path as of 2026-05-05** —
[Spike G](spike-g-mlx-target-llm.md) pivoted the calibration target to
LFM2-Audio-1.5B (MLX), so `set_adapter_cvec` is no longer the primary
steering primitive. The finding remains valid: `llama-cpp-4` v0.2.50 exposes
the API, and this becomes the **fallback path** if the MLX pivot fails
validation in §4.6. Phase 4's declaration-tightening recommendation (R1
below) still applies whenever the llama.cpp path is exercised.

**Date**: 2026-05-04 (original); 2026-05-05 (status note added)

---

## Question

Does the version of `llama-cpp-4` actually used by this workspace expose
`LlamaContext::set_adapter_cvec`, or do we need a version bump before
Phase 4 can wire activation steering through `llama_set_adapter_cvec`?

## Answer

**Yes — `set_adapter_cvec` is available now.** No bump required. A small
defensive change is recommended (see Recommendations).

---

## Findings

### F1 — Declared vs resolved version

| Source | Version | Notes |
|---|---|---|
| Workspace `Cargo.toml:168` | `"0.2.13"` | Interpreted as `^0.2.13` per Cargo semver (any `0.2.x` with `x ≥ 13`). |
| `crates/core/Cargo.toml:66` | inherits via `workspace = true` | Adds `optional = true, features = ["cuda"]`. |
| `Cargo.lock` resolved | **`0.2.50`** | This is what `cargo build` actually pulls. Checksum `7dcf0cd0...afb194`. |

The `^0.2.13` declaration permits any `0.2.x` release; the resolver picked
`0.2.50` (the latest in the `0.2.x` line at lock time). **The version
shown in `Cargo.toml` is misleading on its own** — the lock file is
authoritative for what links into the binary.

### F2 — API location and signature (v0.2.50)

Cached source: `/Users/mathieugosbee/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/llama-cpp-4-0.2.50/src/context.rs:688-710`.

```rust
/// data: The control vector data (embedding values). Pass an empty slice to clear.
/// n_embd: The embedding dimension.
/// il_start: The starting layer index (inclusive).
/// il_end: The ending layer index (exclusive).
pub fn set_adapter_cvec(
    &mut self,
    data: &[f32],
    n_embd: i32,
    il_start: i32,
    il_end: i32,
) -> Result<(), i32> {
    let ret = unsafe {
        llama_cpp_sys_4::llama_set_adapter_cvec(
            self.context.as_ptr(),
            data.as_ptr(),
            data.len(),
            n_embd,
            il_start,
            il_end,
        )
    };
    if ret != 0 { Err(ret) } else { Ok(()) }
}
```

Key properties for Phase 4:

- **Public, safe** wrapper on `LlamaContext` — no `unsafe extern "C"` shim
  needed.
- Takes a flat `&[f32]` of length `n_layers × n_embd` (per the underlying C
  contract); the wrapper itself does no length validation, so the
  `ControlVectorBuffer` builder in
  [`IMPLEMENTATION_PLAN.md` §1.2](../IMPLEMENTATION_PLAN.md) must produce a
  correctly sized buffer.
- **Empty slice clears the cvec** (per doc comment) — useful for the
  prefill→decode swap in Phase 4 / Workstream B.
- Return is `Result<(), i32>` — error code passes through from C.
  `IMPLEMENTATION_PLAN.md`'s sketch should map this to
  `Error::Execution(format!("set_adapter_cvec failed: {}", e))`.

### F3 — Underlying C symbol exposed by `llama-cpp-sys-4`

The unsafe FFI call site is `llama_cpp_sys_4::llama_set_adapter_cvec`. The
prior exploration confirmed `llama-cpp-sys-4-0.2.50` exposes this symbol via
its generated `bindings.rs` (re-exported through `pub use bindings::*`).

If, in some hypothetical future scenario, the safe wrapper were removed from
`llama-cpp-4` while the C symbol remained in `llama-cpp-sys-4`, a fallback
path would be to call `llama_set_adapter_cvec` directly from an
`unsafe { ... }` block in a Phase 4 helper. We do **not** need to do this
today.

### F4 — Smoke binary (deferred, not blocking)

The plan called for a throwaway smoke binary (~50 LoC) that loads a GGUF,
calls `set_adapter_cvec` with random data, and generates 20 tokens. This
was **not executed in Phase 0** because:

1. The API existence is verified by source inspection; the smoke binary
   would only verify *runtime* behavior (CUDA link, valid memory layout).
2. Phase 4's planned `llama_cpp_steer_smoke` example
   ([`IMPLEMENTATION_PLAN.md` §1.6](../IMPLEMENTATION_PLAN.md)) will exercise
   the API end-to-end with assertions on output divergence; it supersedes
   the throwaway.

If runtime verification is needed earlier (e.g. before Phase 4 starts), the
smoke binary template is straightforward: copy
[`crates/core/examples/llama_cpp_chat_smoke.rs`](../../../../crates/core/examples/llama_cpp_chat_smoke.rs),
add a `set_adapter_cvec` call after model load, generate, exit. Pin model
path via `LLAMA_TEST_MODEL`.

---

## Decision gate evaluation

Per the plan's three branches:

- **(a) v0.2.13 has it → unblocked, no bump.** Strictly speaking the
  declared minimum (0.2.13) was not directly inspected; we know only that
  the resolved version (0.2.50) has it. For project planning purposes, the
  *practically relevant* answer is (a): every `cargo build` from now on
  picks up an API-supporting version, because the lockfile pins 0.2.50.
- **(b) Bump needed without breaks.** Not applicable — the lockfile
  already effectively bumped us.
- **(c) Breaking-change migration.** Not applicable.

**Resolved as (a).**

---

## Recommendations (for Phase 4, not Phase 0)

### R1 — Tighten the declared version

Change [`Cargo.toml:168`](../../../../Cargo.toml) from `"0.2.13"` to one of:

- `"0.2.50"` (semantically: `^0.2.50`, allows `0.2.x ≥ 50`) — **recommended**.
- `"=0.2.50"` (exact pin) — most defensive but blocks patch updates.

Rationale: prevents `cargo update -p llama-cpp-4 --precise 0.2.13` from
silently downgrading the workspace below the API floor. Should land as the
first PR of Phase 4 (Workstream A), alongside introducing the `cvec.rs`
module that depends on the API.

### R2 — Buffer-length validation in `ControlVectorBuffer`

The safe wrapper does *not* validate `data.len() == n_embd × (il_end - il_start)`
(or against `n_layers`). Mismatches likely cause C-side undefined behavior or
a non-zero return code. The Phase 4 builder must validate, as planned in
[`IMPLEMENTATION_PLAN.md` §1.2](../IMPLEMENTATION_PLAN.md):

```rust
ControlVectorBuffer::single_layer(layer, vec, n_embd, n_layers) // returns Err on mismatch
```

### R3 — Use empty-slice clearing semantics in the phase-aware path

Workstream B's `run_generation_with_phase_cvec` can pass an empty slice
between prefill and decode if the assistant-side cvec is not configured —
cheaper than building a zero-buffer.

### R4 — No `--precise` or `--locked` change in Phase 0

Do not run `cargo update --precise 0.2.50` now. The lockfile is already
in the right state; touching it in Phase 0 invites unrelated transitive
updates and cargo-lock churn that's better contained in Phase 4's
boundary-defined PR.

---

## Files referenced

- Workspace declaration: [`Cargo.toml:168`](../../../../Cargo.toml)
- Per-crate consumer: [`crates/core/Cargo.toml:66`](../../../../crates/core/Cargo.toml)
- Resolved version: `Cargo.lock` (entry `name = "llama-cpp-4"`, `version = "0.2.50"`)
- API definition: `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/llama-cpp-4-0.2.50/src/context.rs:688`
- Existing usage of `llama-cpp-4`: [`crates/core/src/nodes/llama_cpp/inference.rs`](../../../../crates/core/src/nodes/llama_cpp/inference.rs)
- Steering stub awaiting this API: [`crates/core/src/nodes/llama_cpp/steer.rs:308-313`](../../../../crates/core/src/nodes/llama_cpp/steer.rs#L308)

---

## Open questions

- **Q-A1**: Did v0.2.13 specifically have `set_adapter_cvec`? Not confirmed
  by source inspection (v0.2.13 was not cached and was not fetched). Low
  priority because the lockfile resolves to 0.2.50 anyway. If this question
  ever matters (e.g. someone reverts the lockfile), inspect by:

      cargo download --output /tmp llama-cpp-4@0.2.13
      tar -xzf /tmp/llama-cpp-4-0.2.13.crate -C /tmp
      grep -n 'set_adapter_cvec\|adapter_cvec' /tmp/llama-cpp-4-0.2.13/src/context.rs

  (`cargo download` is provided by the `cargo-download` cargo subcommand;
  not strictly necessary for this project.)

- **Q-A2**: Does `llama-cpp-4` have any *related* APIs we should also map
  while we're touching this area — e.g., loading a serialized control
  vector from a GGUF, or per-token cvec scaling? Not required for Phase 4
  but worth a 5-minute look during the first Workstream A PR.
