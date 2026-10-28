//! Default-mode solves repeat exactly at every worker count when terms have
//! at least 100,000 levels, where the separate kernel scatters atomically.

use within::{Effect, LsmrOptions, PreconditionerConfig, Solver, StepKernel};

const ROWS: usize = 600_000;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn default_mode_large_terms_repeat_across_worker_counts() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // 200,000 sorted workers, 120,000 unsorted firms and a 32-level term.
    let worker: Vec<u32> = (0..ROWS).map(|i| (i / 3) as u32).collect();
    let firm: Vec<u32> = (0..ROWS).map(|_| (next() % 120_000) as u32).collect();
    let third: Vec<u32> = (0..ROWS).map(|i| ((i * 17 + i / 7) % 32) as u32).collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| f64::from(firm[i] % 101) * 0.01 + (next() % 1_000) as f64 / 1_000.0)
        .collect();
    let weights: Vec<f64> = (0..ROWS).map(|_| 0.5 + (next() % 8) as f64).collect();
    let options = LsmrOptions {
        tol: 1e-10,
        maxiter: 12,
        local_size: None,
    };
    for weighted in [false, true] {
        for precond in [PreconditionerConfig::Off, PreconditionerConfig::Diagonal] {
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        let none = std::iter::empty::<&[f64]>;
                        let effects = vec![
                            Effect::new(&worker, true, none()).unwrap(),
                            Effect::new(&firm, true, none()).unwrap(),
                            Effect::new(&third, true, none()).unwrap(),
                        ];
                        let solver = Solver::new(
                            effects,
                            weighted.then(|| weights.clone()),
                            precond.clone(),
                        )
                        .unwrap();
                        assert!(!solver.reproducible());
                        assert_eq!(solver.step_kernel(), StepKernel::Fused);
                        let solved = solver.solve(&y, &options).unwrap();
                        let mut session = solver.start_session(&y, &options).unwrap();
                        session.run_action(None, 5, None).unwrap();
                        session.run_action(None, usize::MAX, None).unwrap();
                        let continued = session.finish();
                        (
                            bits(&solved.x),
                            bits(&solved.demeaned),
                            bits(&continued.x),
                            bits(&continued.demeaned),
                            solved.residual.to_bits(),
                        )
                    })
            };
            let reference = run(1);
            for threads in [2, 4, 8, 8] {
                assert!(
                    run(threads) == reference,
                    "weighted={weighted} {precond:?}: {threads} workers changed bits"
                );
            }
        }
    }
}
