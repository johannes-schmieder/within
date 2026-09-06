# MultiwayMG modified-LSMR workspace increment

This owner-controlled fork starts its `multiwaymg` branch at upstream
`b7779cbab7a3116be56aae4389fde1f6e6a99a9f`. Upstream `main` is preserved separately;
its later statistical-screen and dependency changes are outside this increment.
The scope is standalone MultiwayMG M3, before CPU scheduling (M7).

## Implemented boundary

- `OperatorMut` permits caller-owned mutable scratch without `Sync`, locks or
  interior mutability. Borrowed immutable `Operator` instances adapt directly.
- `MlsmrWorkspace` retains all modified bidiagonalization, solution, warm-start
  and local-reorthogonalization vectors. Required/retained byte queries use
  checked arithmetic and construction uses fallible exact reservations.
- `mlsmr_with_workspace` uses serial internal kernels, with no global Rayon entry.
  Operators and escalation handlers control their own execution/allocation.
  Shape/options validate before actions or scratch changes. Callback failures
  may dirty scratch, but no result is published and the workspace is reusable.
- The workspace is shape-bound, not numerical-owner-bound. Every run initializes
  all state it reads; prior basis/history/recurrence/images are not reused. Reset
  ring indices make old slots unread until overwritten. The caller must hold a
  fixed operator within each run and validate external hierarchy generations.
- A result borrows the solution vector and carries native scalar diagnostics.
  Rust prevents reuse while that result is still borrowed. Native convergence
  remains a candidate for the caller's independent certification.
- Optional escalation takes a caller-created mutable handler; callers provide
  fresh/reset per-run handler state. Warm starts keep upstream correction and
  original-RHS tolerance semantics.
- Existing allocating APIs and workspace solves share one numerical recurrence.
  Standard LSMR and allocating modified LSMR retain the original legacy Rayon
  threshold behavior. The prepared API is serial in this increment; controlled
  parallel reductions/scheduling are deliberately a later qualified change.

For m observations, n coefficients and k=min(requested window,m,n), requested
payload is `8*(3m + 6n + 2kn)` bytes. The extra observation vector always reserves
warm-start capacity, even for cold solves. The two MGS windows are retained;
there is no algorithmic reduction in reorthogonalization quality. Inline state,
allocator overhead, caller data, operator memory and callback scratch are excluded.
No construction-peak or total-process claim may silently use only this payload.

## Validation and compatibility

The frozen upstream crate is a development-only Git dependency in comparison
tests. It runs the original code, with its own distinct operator trait; no
baseline numerical recurrence is copied into this fork. Tested small over- and
underdetermined/rank-deficient matrices, warm starts, zero/iteration-cap paths,
escalation, repeated changing RHS and windows None/0/1/3/8/capped-large match
upstream bits. Serial versus legacy-parallel configurations require numerical
agreement rather than a cross-configuration bitwise promise.

Other tests cover mutable changed operators, callback errors and reuse, invalid
shape before action calls, zero and exact warm starts, checked-size overflow,
all 11 reservation-failure boundaries, and the borrowed-result lifetime.
The isolated allocator executable measures a very first call and 16 repeats at
n=12,000 (above the legacy parallel threshold): zero internal solve allocations,
2,400,000 retained bytes for window 8, and exact release on drop. This is a
workspace allocation result, not a competitive solve-time claim.

Use pinned Rust 1.85 and locked dependencies: format, strict workspace Clippy,
all/minimal tests and warning-free documentation. The dedicated permanent
`multiwaymg-workspace.yml` workflow also tests Linux/macOS/Windows in debug/
release and all/minimal configurations. Two existing `within` domain-test
vectors were changed to equivalent arrays to satisfy Rust 1.85 strict Clippy;
no statistical solver/domain algorithm was changed.

Adding the baseline as a dev dependency makes `cargo -p schwarz-precond`
ambiguous. Use `--manifest-path crates/schwarz-precond/Cargo.toml` for that local
package, or workspace commands. The baseline is not a runtime dependency of
downstream consumers. Both MultiwayMG within/schwarz pins must move together
after this fork PR qualifies; immutable performance baselines stay unchanged.

## Delivery ledger

- Implemented: caller-owned serial recurrence and mutable action API.
- Local upstream and direct baseline compatibility/allocation tests pass.
- Full local Rust 1.85 format, strict workspace Clippy, all/minimal tests and
  warning-free documentation passed. Exact-source GitHub qualification pending.
- Dependency pin integration into MultiwayMG: pending qualified fork merge.
