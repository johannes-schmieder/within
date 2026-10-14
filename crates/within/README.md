# within

Solves linear fixed-effects models **y = D x + e** where D is a sparse
categorical design matrix. Modified LSMR with a domain-decomposition
(Schwarz) preconditioner backed by approximate Cholesky local solvers.
Designed for econometric panel data with millions of observations and
multiple high-dimensional fixed effects.

## Install

```
cargo add within
```

## Quick example

```rust
use ndarray::Array2;
use within::{solve, LsmrOptions, PreconditionerConfig};

// Two factors: 100 levels each, 10 000 observations
let n_obs = 10_000usize;
let mut categories = Array2::<u32>::zeros((n_obs, 2));
for i in 0..n_obs {
    categories[[i, 0]] = (i % 100) as u32;
    categories[[i, 1]] = (i / 100) as u32;
}
let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();

// Solve with library defaults: LSMR + additive Schwarz
let result = solve(categories.view(), &y, None, &LsmrOptions::default(), None)
    .expect("solve should succeed");
assert!(result.converged);
println!("LSMR converged in {} iterations", result.iterations);

// Tighter tolerance with an explicit preconditioner config
let lsmr = LsmrOptions { tol: 1e-10, ..LsmrOptions::default() };
let precond = PreconditionerConfig::default();
let result = solve(categories.view(), &y, None, &lsmr, &precond)
    .expect("solve should succeed");
assert!(result.converged);

// Or opt into a diagonal/Jacobi preconditioner.
let diagonal = PreconditionerConfig::Diagonal;
let result = solve(categories.view(), &y, None, &lsmr, &diagonal)
    .expect("solve should succeed");
assert!(result.converged);
```

## Feature flags

| Feature   | Default | Effect                                                                 |
|-----------|---------|------------------------------------------------------------------------|
| `ndarray` | yes     | Enables `from_array` constructors on observation stores for interop with `ndarray::ArrayView2`. |

## Architecture

The crate is organized in four layers:

1. **`observation`** — Row-aligned observation columns via the
   `ObservationFrame` (categorical level codes plus continuous loadings,
   each borrowed or owned per column). Observation weights are not owned
   here — they flow as `Option<&[f64]>` to the operator layer.

2. **`domain`** — Domain decomposition. `Design` wraps a frame with
   factor metadata; `build_local_domains` constructs factor-pair subdomains
   with partition-of-unity weights for the Schwarz preconditioner.

3. **`operator`** — Linear algebra primitives. Internal rectangular
   `sqrt(W) D` operator for LSMR and Schwarz preconditioner builders
   that wire approximate Cholesky local solvers into the generic
   `schwarz-precond` framework.

4. **`orchestrate`** — End-to-end solve entry points (`solve`, `solve_batch`)
   with typed configuration (`LsmrOptions`, `PreconditionerConfig`).

## Weighted assembly memory

Weighted intercept-pair construction uses compensated sums for both cross-tab
cells and independently accumulated diagonals. The Laplacian consistency
budget is unchanged. Structural unit weights and pairs containing slope
channels retain their existing accumulation paths. For a weighted intercept
pair with `R` and `C` active levels, dense construction temporarily adds
`8 * (R*C + R + C)` bytes; sparse construction adds `8 * (R + C)` bytes and
reuses the column corrections for edge sums. These buffers are not retained
by the solver. Concurrent pair construction must account for each live pair.

## License

MIT

## References

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.

### Optional fixed pair backends and solve sessions

`Solver::pair_nomination` selects a categorical pair by its largest retained
component dimension, then total local dimension, then original factor indices.
Only components strictly above the configured dense threshold are eligible.
`build_pair_replacement` accepts a `FixedReducedInverseFactory` and constructs
all eligible components transactionally. Unchanged components share their
incumbent factors. A failed factory or construction checkpoint leaves the
incumbent usable. The factory admits its resource envelope and supplies bounded
work/cancellation checks through `checkpoint(work_units)`.

A `FixedReducedInverse` must supply one fixed linear, symmetric positive
semidefinite action in the Euclidean retained-coordinate metric, with exactly
the constant kernel. A fixed symmetric multigrid cycle is appropriate; CG with
either a tolerance or a fixed number of iterations is not linear. `within`
keeps the incumbent sign transform, full-component mean projections, triangular
elimination/recovery and two-sided partition weights. The full wrapped action
retains the incumbent component kernel even though the unprojected triangular
congruence generally does not. This local construction requires no global
full-rank coefficient basis.

`Solver::start_session` retains an original weighted RHS and an internal
coefficient iterate. `run_action` with the same backend resumes all existing
Golub–Kahan vectors, rotations and reorthogonalization state. After rejected
setup, continuing with the incumbent therefore gives the same recurrence as
an uninterrupted solve. Accepting a different replacement, or calling
`restart_correction` at an equal or tighter tolerance, discards the entire
recurrence and recomputes `b - A x`. Both stopping references use the original
RHS under the current fixed preconditioner; progress ratios from different
preconditioners are not comparable. All actions share the original iteration
budget. `snapshot` returns a recoverable output copy without changing state;
`finish` returns the final result. Final application-specific scientific
certification remains the caller's responsibility.

Replacement handles are owned, cheaply cloneable, and tied to their exact
originating solver through a private identity token, including after that solver
moves. They are runtime-only. Existing preconditioner wire formats and ordinary
`solve`/`solve_batch` execution remain unchanged; external reduced factors cannot
be serialized. Only callers opting into the new action can change their
numerical solve path.
