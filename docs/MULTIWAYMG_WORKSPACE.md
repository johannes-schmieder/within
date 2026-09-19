# MultiwayMG modified-LSMR workspace increment

This owner-controlled fork starts its `multiwaymg` branch at upstream
`b7779cbab7a3116be56aae4389fde1f6e6a99a9f`. Upstream `main` is preserved separately;
its later statistical-screen and dependency changes are outside this increment.
The original scope is standalone MultiwayMG M3, with the M4 candidate-gate
extension below; CPU scheduling remains M7.

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
- M3 local and exact-source/PR checks passed before delivery; the new M4
  candidate-gate qualification is recorded below.
- M3 merged in fork PR #1 as `2e7d5ec935b4846b369430ff00deb59f90e7d2d5`
  and integrated into MultiwayMG PR #37 after green source/PR checks.

The first inherited CI source-policy check correctly rejected the new Git test
baseline. `deny.toml` now explicitly allows only the upstream within repository
and requires revision pins for Git sources. The baseline remains the exact
`b7779cb` revision; unknown Git sources still fail. This is the explicit source
registration required by the new compatibility test, not removal of the check.

## M4 candidate-gate extension

MultiwayMG's committed serial baseline (`9b4f174`, PR #43) found reproducible
native normal-equation stops that missed the independent original-operator
certificate. This fork extension adds `LsmrCandidateGate` and
`mlsmr_with_workspace_and_candidate_gate`, leaving the old APIs and M3 memory
layout unchanged. It does not retune native tolerances or modify the fixed
bidiagonalization/rotation/reorthogonalization arithmetic.

At a native tolerance candidate with remaining iteration budget and a nonzero
next bidiagonal direction, call the user's gate before the native terminal audit.
A veto continues the same recurrence, including its local MGS history. This order
matters: the terminal modified-LSMR audit overwrites the current `v` vector with
a preconditioned true gradient. A veto must avoid that destructive audit.
The existing per-iteration escalation check still runs after a veto.

The gate receives the correction and optional warm-start offset separately;
it must check their sum when certifying a warm-start candidate. It owns any
certificate arrays and action/work counters. No extra fork-owned storage is
reserved; serial required/retained payload remains `8*(3m+6n+2kn)`.
Operator/preconditioner state stays fixed throughout the call. Callback errors
publish no candidate; a later call reinitializes the same workspace.

A gate is not final acceptance. Zero/initial-normal-zero exits, exact breakdown,
iteration limits and escalation cannot necessarily continue and may return a
candidate without a gate call. Native diagnostics preserve their meaning.
Every returned candidate still needs the caller's independent final certificate;
it may be rejected even when native convergence is true. A gate that always
permits stopping matches the existing serial route, and the ungated APIs retain
the frozen upstream comparison tests. There is no implicit restart or fallback.

The existing three-platform workspace target now also tests uninterrupted
recurrence bit equality under repeated vetoes, exact forward/adjoint counts,
cold/warm candidates, continued original-gradient convergence, native diagnostics,
callback failure/recovery, invalid shapes, non-resumable exits and escalation.
The isolated allocator gate covers the first and repeated n=12,000 guarded solves,
multiple candidate vetoes, callback failure and subsequent native recovery with
zero internal allocations, unchanged 11-array setup and exact release.
Full local and exact-source/PR CI qualification is required before moving both
MultiwayMG dependency pins to the new fork merge.

M4 candidate-gate local Rust 1.85 format, strict workspace Clippy, all-feature
and minimal-feature workspace tests, warning-free rustdoc and release all/minimal
workspace/allocation tests pass. The frozen-upstream comparisons remain intact.
Exact-source and PR qualification is pending; both downstream pins remain at the
qualified M3 merge until this incremental fork PR merges.


## M5 prepared norm rejection

M4 merged in fork PR #2 as `cb20b27a7137804202be39976415618686a144de`.
M5's stronger complete-solve allocator test exposed a 100-byte formatted error
when a finite RHS had an unrepresentable Euclidean norm. The prepared workspace
now returns `SolveError::NonFiniteResidualNorm { value_bits }` at the same boundary.
The owning `mlsmr` wrapper translates it to the historical `InvalidInput` variant
and exact text. No successful arithmetic, scan, work count, or storage changes.
This covers the numerical norm failure; static malformed-input diagnostics in
the standalone dependency remain allocating. MultiwayMG validates those before
entering the driver. The isolated allocator test covers gated/native cold norm
overflow, warm residual overflow, and same-workspace successful recovery.
Rust 1.85 formatting, strict Clippy, all/minimal workspace tests, warning-free
rustdoc and release all/minimal workspace/allocation checks pass locally.
Exact-source and PR CI qualification remains before downstream pin integration.


## M6o serial Schwarz application

M5's norm-error increment merged as `fad1d462d44e7d5b5226370021d69dab4e854669`.
The new `SchwarzPreconditioner::try_serial_workspace` borrows its exact immutable
owner and keeps one array for a V-element transaction result plus two maximum
local scratch vectors. The maximum includes augmented ground/cover dimensions.
Requested bytes are `8*(V+2*Smax)`, checked against overflow/Vec representability
and the caller's requested-payload limit before one fallible reservation. Actual
capacity is separately reported; owner storage, old pooled buffers, allocator
rounding/headers and opaque local-factor internals are excluded. This is not yet
a full terminal budget. Construction of the preconditioner remains unchanged.

`SerialSchwarzWorkspace::apply` visits entries in stored order, passes exact local
slices and disables the inner-parallelism hint. Immutable core indices validate
once at binding; each call checks shapes and local scratch bounds before touching
scratch. It never enters Rayon or the old buffer pool. User output is published
only after every domain succeeds. Typed static failures allocate no messages;
local errors retain their existing types and may allocate text. Dirty scratch
from a failure is reusable. The borrowed owner cannot be substituted or outlived;
generic local implementations must keep interior numerical state fixed and obey
their scratch contract. A generic outer executor cannot forbid its local solvers
from allocating or ignoring the parallelism hint.

`within::SerialPreconditionerWorkspace` hides the private concrete pair solver,
exposes the same requested/retained payload boundary and implements `OperatorMut`
for direct prepared-LSMR use. Diagonal actions reserve no array. Existing
allocating/pooled APIs, factor setup, numerical kernels and solver recurrences
are unchanged; no dependency version or serialized owner layout changes.

Independent dense assembly and cancellation-order controls cover two-sided
partition weights, overlapping/uncovered/empty domains, augmentation, static
rejection, late local failure, dirty-buffer recovery, changed generic scratch,
checked-size overflow and reservation failure/unwind. Real dense/approximate
pair tests use unit/dyadic weights, connected, nested, disconnected interleaved
and unbalanced inputs; compare with one-worker pooled actions, fixed results
across caller pools and independent concurrent workspaces. The existing 33,792-
coordinate local fixture also checks the serial action above concrete parallel
thresholds inside a four-worker caller pool. Isolated executables
measure first and repeated actions with active allocator controls. Concrete
setup uses an explicit pool and joins each OS worker handle before the first measured action;
this avoids unrelated asynchronous worker TLS allocations contaminating a
process-wide allocator measurement. No action warmup hides first-call costs.
The initial global-pool and scoped-pool harness failures are retained in external
qualification logs; explicit OS-thread joins remove the delayed setup activity.
The small connected recipe was relabeled from the inaccurate "tensor" to "latin"
without changing its tuples; earlier raw log labels remain untouched.

### Remaining terminal requirements

The pinned approx-chol0.5.0 factor can still allocate an n-vector when it carries
a permutation; its nested capacities and build/fill peak are not exposed. Passing
these particular real-factor fixtures does not establish allocation freedom for
all valid factor provenance. Serial setup, permutation scratch, factor capacity
statistics and checked Schur/fill admission require a separate dependency change.

Full numerical range is also distinct from structural factor-shift projection.
An external MultiwayMG audit found dimension12/rank7 support whose two-sided
structurally projected Schwarz action is symmetric/positive on the quotient,
yet leaks 0.252389 (unit) / 0.176210 (dyadic weights) outside the numerical range.
That input requires extra treatment or rejection under the accepted terminal
gate. A workspace does not resolve this eligibility issue, and certified existing
LSMR results are not invalidated by it. No sparse terminal, route/default change,
competitive claim or holdout is introduced by this fork increment.


M6o local Rust1.85 format, strict all-target/all-feature Clippy, all/minimal
workspace tests, warning-free rustdoc and release generic/all/minimal/concrete
contracts pass. Sixteen real-factor configurations record one exact outer
reservation (288/360/632 bytes by fixture), zero first/repeated/static-rejection
allocations and exact drop. The generic n=12,000 action retains 288,016 bytes,
with zero first/repeated/static/local-error/recovery allocations for its tested
allocation-free local solver. Source/PR CI qualification precedes fork merge;
both downstream pins remain unchanged until that merge qualifies.
