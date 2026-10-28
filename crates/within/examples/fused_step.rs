//! Step-kernel timing on a sparse three-FE panel: 5 periods per worker, a
//! firm per worker with 2% annual mobility (workers / 20 firms) and a
//! 32-level third effect, solving a batch of RHS columns concurrently.
//!
//! Usage:
//!   cargo run --release --example fused_step -p within -- \
//!       <rows> <rhs> <threads> <fused|separate> <diagonal|additive> <iterations> <repetitions>
//!
//! Every solve runs exactly `iterations` steps (tolerance 0). Prints one
//! line per repetition with the batch solve seconds (preconditioner build
//! excluded) and a digest of the solution bits.

use std::time::Instant;

use within::{
    Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy, Solver,
    StepKernel,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.into());
    let rows: usize = arg(1, "1000000").parse().unwrap();
    let rhs: usize = arg(2, "16").parse().unwrap();
    let threads: usize = arg(3, "8").parse().unwrap();
    let kernel = match arg(4, "fused").as_str() {
        "separate" => StepKernel::Separate,
        _ => StepKernel::Fused,
    };
    let additive = arg(5, "diagonal") == "additive";
    let iterations: usize = arg(6, "23").parse().unwrap();
    let repetitions: usize = arg(7, "3").parse().unwrap();

    let mut state = 0x5851_f42d_4c95_7f2du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let periods = 5;
    let workers = rows.div_ceil(periods);
    let firms = (workers / 20).max(8);
    let mut firm = Vec::with_capacity(rows);
    for w in 0..workers {
        let mut current = (w % firms) as u32;
        for t in 0..periods {
            if t > 0 && next() % 50 == 0 {
                current = (next() % firms as u64) as u32;
            }
            firm.push(current);
        }
    }
    firm.truncate(rows);
    let worker: Vec<u32> = (0..rows).map(|i| (i / periods) as u32).collect();
    let third: Vec<u32> = (0..rows)
        .map(|i| (firm[i] + 17 * worker[i] + (i % periods) as u32) % 32)
        .collect();
    let columns: Vec<Vec<f64>> = (0..rhs)
        .map(|_| {
            (0..rows)
                .map(|_| (next() % 1_000_000) as f64 / 1e6 - 0.5)
                .collect()
        })
        .collect();
    let columns: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    let preconditioner = if additive {
        PreconditionerConfig::Additive {
            local_solver: LocalSolverConfig::default(),
            reduction: ReductionStrategy::Auto,
        }
    } else {
        PreconditionerConfig::Diagonal
    };
    let options = LsmrOptions {
        tol: 0.0,
        maxiter: iterations,
        local_size: None,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let none = std::iter::empty::<&[f64]>;
        let effects = vec![
            Effect::new(&worker, true, none()).unwrap(),
            Effect::new(&firm, true, none()).unwrap(),
            Effect::new(&third, true, none()).unwrap(),
        ];
        let solver = Solver::new(effects, None, preconditioner)
            .unwrap()
            .with_step_kernel(kernel)
            .unwrap();
        for repetition in 0..repetitions {
            let started = Instant::now();
            let result = solver.solve_batch(&columns, &options).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let mut digest = 0xcbf2_9ce4_8422_2325u64;
            for value in result.x.iter().chain(&result.demeaned) {
                digest = (digest ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3);
            }
            println!(
                "rows={rows} rhs={rhs} threads={threads} kernel={kernel:?} additive={additive} \
                 repetition={repetition} iterations={} seconds={seconds:.6} digest={digest:016x}",
                result.iterations.iter().max().unwrap_or(&0)
            );
        }
    });
}
