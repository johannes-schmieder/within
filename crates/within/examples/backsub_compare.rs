//! Fixed-work comparison of public block-elimination applications and complete solves.
//!
//! Build this same file against a frozen base and base plus the back-substitution patch.
//! Run each executable with identical arguments, retaining every observation. No sampling,
//! best-of selection, or automatic retry is performed. The default is science-only; timers
//! are enabled only by `--measure`. External drivers own pairing and the bounded schedule.
//!
//! Example: `backsub_compare --artifact BASE --size memory --threads 4 --weighted`
//! Add `--mode solve` for complete LSMR solves. `--warmups N --measure` records one
//! group after exactly N warmup groups. Hashes are FNV-1a over little-endian words;
//! they are exact comparison signatures, not cryptographic artifact identities.
//!
//! Both cases have one connected factor pair and four distinct neighbors per left level.
//! A 127-level exact reduced factor keeps its error independently checkable. The memory
//! case grows the eliminated factor itself: its two CSR orientations, inverse diagonal,
//! and four RHS/output banks exceed 172 MiB, before solver scratch and observations.
//! Thus repeated applications cannot be mistaken for a large input with tiny factors.

use std::hint::black_box;
use std::time::Instant;

use within::{
    Design, Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy,
    SchurMode, Solver,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const RIGHT: usize = 127;
const DEGREE: usize = 4;
const TOLERANCE: f64 = 1e-11;
const WITNESS_TOLERANCE: f64 = 1e-8;

struct Args {
    artifact: String,
    size: String,
    left: usize,
    threads: usize,
    rhs: usize,
    weighted: bool,
    solve: bool,
    warmups: usize,
    measure: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut out = Self {
            artifact: "development".into(),
            size: "cache".into(),
            left: 4096,
            threads: 1,
            rhs: 4,
            weighted: false,
            solve: false,
            warmups: 0,
            measure: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => out.artifact = args.next().ok_or("missing artifact")?,
                "--size" => {
                    out.size = args.next().ok_or("missing size")?;
                    out.left = match out.size.as_str() {
                        "cache" => 4096,
                        "memory" => 1_048_576,
                        _ => return Err("size must be cache or memory".into()),
                    };
                }
                "--left" => {
                    out.left = args.next().ok_or("missing left")?.parse()?;
                    out.size = "custom".into();
                }
                "--threads" => out.threads = args.next().ok_or("missing threads")?.parse()?,
                "--rhs" => out.rhs = args.next().ok_or("missing rhs")?.parse()?,
                "--weighted" => out.weighted = true,
                "--mode" => {
                    out.solve = match args.next().ok_or("missing mode")?.as_str() {
                        "apply" => false,
                        "solve" => true,
                        _ => return Err("mode must be apply or solve".into()),
                    }
                }
                "--warmups" => out.warmups = args.next().ok_or("missing warmups")?.parse()?,
                "--measure" => out.measure = true,
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        if out.left < RIGHT
            || out.left > u32::MAX as usize / DEGREE
            || out.threads == 0
            || !(1..=16).contains(&out.rhs)
            || out.warmups > 16
            || !out
                .artifact
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        {
            return Err("invalid shape, worker count, warmup count, or artifact label".into());
        }
        Ok(out)
    }
}

fn hash_word(state: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *state = (*state ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}
fn signature(values: &[f64]) -> u64 {
    let mut state = 0xcbf29ce484222325;
    for value in values {
        hash_word(&mut state, value.to_bits());
    }
    state
}
fn coefficients(left: usize, rhs: usize) -> Vec<f64> {
    (0..left + RIGHT)
        .map(|i| {
            let value = if i < left {
                ((i * 13 + rhs * 7) % 31) as i64 - 15
            } else {
                (((i - left) * 17 + rhs * 11) % 37) as i64 - 18
            };
            value as f64 / 16.0
        })
        .collect()
}
fn response(x: &[f64], left: usize, q: &[u32], r: &[u32]) -> Vec<f64> {
    q.iter()
        .zip(r)
        .map(|(&q, &r)| x[q as usize] + x[left + r as usize])
        .collect()
}
fn check_solution(x: &[f64], expected: &[f64], left: usize, q: &[u32], r: &[u32]) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for (&q, &r) in q.iter().zip(r) {
        let (q, r) = (q as usize, left + r as usize);
        maximum = maximum.max(((x[q] + x[r]) - (expected[q] + expected[r])).abs());
    }
    if !x.iter().all(|v| v.is_finite()) || maximum > WITNESS_TOLERANCE {
        return Err(format!("independent fitted-value witness failed: {maximum:e}").into());
    }
    Ok(maximum)
}

#[derive(Default)]
struct Outcome {
    hash: u64,
    iterations: usize,
    maximum_error: f64,
    maximum_residual: f64,
    elapsed_ns: u128,
}

fn run(args: &Args) -> Result<()> {
    let nobs = args.left * DEGREE;
    let ndofs = args.left + RIGHT;
    let mut q = Vec::with_capacity(nobs);
    let mut r = Vec::with_capacity(nobs);
    let mut weights = Vec::with_capacity(nobs);
    let mut input_hash = 0xcbf29ce484222325;
    for i in 0..args.left {
        // Adjacent neighbors connect all right levels; labels cover 0..RIGHT.
        for offset in [0, 1, 5, 13] {
            let j = (i * 17 + offset) % RIGHT;
            let weight = if args.weighted {
                1.0 + ((i + j) % 7) as f64 / 8.0
            } else {
                1.0
            };
            q.push(i as u32);
            r.push(j as u32);
            weights.push(weight);
            hash_word(&mut input_hash, i as u64);
            hash_word(&mut input_hash, j as u64);
            hash_word(&mut input_hash, weight.to_bits());
        }
    }
    let config = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            schur: SchurMode::Exact,
            dense_threshold: RIGHT + 1,
            ..Default::default()
        },
        // One connected pair gives one subdomain, with no cross-subdomain reduction ambiguity.
        reduction: ReductionStrategy::ParallelReduction,
    };
    let started = args.measure.then(Instant::now);
    let design = Design::new([Effect::new(&q, true, [])?, Effect::new(&r, true, [])?])?;
    let solver = Solver::new(design, args.weighted.then_some(weights.as_slice()), &config)?;
    let build_ns = started.map(|t| t.elapsed().as_nanos());
    if solver.n_dofs() != ndofs || !solver.warnings().is_empty() {
        return Err("unexpected coefficient layout or construction warning".into());
    }
    let preconditioner = solver
        .preconditioner()
        .ok_or("additive preconditioner missing")?;
    if preconditioner.variant_name() != "Additive" {
        return Err(format!(
            "unexpected preconditioner: {}",
            preconditioner.variant_name()
        )
        .into());
    }
    let options = LsmrOptions {
        tol: TOLERANCE,
        maxiter: 200,
        local_size: None,
    };
    let expected: Vec<Vec<f64>> = (0..args.rhs).map(|k| coefficients(args.left, k)).collect();
    let mut inputs = Vec::with_capacity(args.rhs);
    for x in &expected {
        let y = response(x, args.left, &q, &r);
        if args.solve {
            inputs.push(y);
        } else {
            // Independent row-wise D' W D x, exact dyadic sums at both fixed sizes.
            let mut b = vec![0.0; ndofs];
            for (((&q, &r), &w), &y) in q.iter().zip(&r).zip(&weights).zip(&y) {
                b[q as usize] += w * y;
                b[args.left + r as usize] += w * y;
            }
            inputs.push(b);
        }
    }
    for input in &inputs {
        hash_word(&mut input_hash, signature(input));
    }
    let mut outputs = vec![vec![0.0; ndofs]; args.rhs];
    let mut previous = None;
    let mut final_outcome = Outcome::default();
    for group in 0..=args.warmups {
        let measure = args.measure && group == args.warmups;
        let mut outcome = Outcome {
            hash: 0xcbf29ce484222325,
            ..Default::default()
        };
        if args.solve {
            for ((input, expected), output) in inputs.iter().zip(&expected).zip(&mut outputs) {
                let started = measure.then(Instant::now);
                let solved = solver.solve(black_box(input), &options)?;
                outcome.elapsed_ns += started.map_or(0, |t| t.elapsed().as_nanos());
                if !solved.converged
                    || !solved.residual.is_finite()
                    || !solved.unidentified.is_empty()
                    || !solved.warnings.is_empty()
                {
                    return Err("complete solve did not converge cleanly".into());
                }
                outcome.iterations += solved.iterations;
                outcome.maximum_residual = outcome.maximum_residual.max(solved.residual);
                outcome.maximum_error = outcome
                    .maximum_error
                    .max(check_solution(&solved.x, expected, args.left, &q, &r)?);
                if solved
                    .demeaned
                    .iter()
                    .any(|r| !r.is_finite() || r.abs() > WITNESS_TOLERANCE)
                {
                    return Err("independent zero-demeaned witness failed".into());
                }
                hash_word(&mut outcome.hash, signature(&solved.demeaned));
                hash_word(&mut outcome.hash, solved.iterations as u64);
                hash_word(&mut outcome.hash, solved.residual.to_bits());
                output.copy_from_slice(&solved.x);
            }
        } else {
            let started = measure.then(Instant::now);
            for (input, output) in inputs.iter().zip(&mut outputs) {
                preconditioner.apply(black_box(input), black_box(output))?;
            }
            outcome.elapsed_ns = started.map_or(0, |t| t.elapsed().as_nanos());
            for (output, expected) in outputs.iter().zip(&expected) {
                outcome.maximum_error = outcome
                    .maximum_error
                    .max(check_solution(output, expected, args.left, &q, &r)?);
            }
        }
        for output in &outputs {
            hash_word(&mut outcome.hash, signature(output));
        }
        if let Some(hash) = previous {
            if hash != outcome.hash {
                return Err(format!(
                    "fixed-worker scientific signatures differ across groups: {hash:016x} versus {:016x}; maximum fitted error {:e}",
                    outcome.hash, outcome.maximum_error,
                ).into());
            }
        }
        previous = Some(outcome.hash);
        final_outcome = outcome;
    }
    // Count only known retained arrays; excludes dense factor, maps, scratch and input observations.
    let csr_bytes = 2 * nobs * 12 + 4 * (ndofs + 2);
    let factor_rhs_lower_bound = csr_bytes + 8 * args.left + 16 * ndofs * args.rhs;
    let ns = |v: Option<u128>| v.map_or_else(|| "null".into(), |v| v.to_string());
    println!(
        concat!("{{\"artifact\":\"{}\",\"size\":\"{}\",\"mode\":\"{}\",",
        "\"threads\":{},\"weighted\":{},\"left\":{},\"right\":{},\"nobs\":{},\"rhs\":{},",
        "\"warmups\":{},\"explicit_apply_calls\":{},\"iterations\":{},\"factor_rhs_lower_bound\":{},",
        "\"backsub_rows_per_apply\":{},\"backsub_nnz_per_apply\":{},",
        "\"input_fnv64\":\"{:016x}\",\"science_fnv64\":\"{:016x}\",",
        "\"maximum_error\":{:e},\"recurrence_residual\":{:e},\"build_ns\":{},\"elapsed_ns\":{}}}"),
        args.artifact,
        args.size,
        if args.solve { "solve" } else { "apply" },
        args.threads,
        args.weighted,
        args.left,
        RIGHT,
        nobs,
        args.rhs,
        args.warmups,
        if args.solve { 0 } else { args.rhs },
        final_outcome.iterations,
        factor_rhs_lower_bound,
        args.left,
        nobs,
        input_hash,
        final_outcome.hash,
        final_outcome.maximum_error,
        final_outcome.maximum_residual,
        ns(build_ns),
        ns(args.measure.then_some(final_outcome.elapsed_ns))
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    // Keep the caller's worker ceiling fixed for construction and every apply/solve.
    pool.install(|| run(&args).map_err(|e| e.to_string()))
        .map_err(Into::into)
}
