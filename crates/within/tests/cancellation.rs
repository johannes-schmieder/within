use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use within::{
    BuildError, Cancellation, Design, Effect, IntoDesign, LsmrOptions, PreconditionerConfig,
    ReductionStrategy, SolveError, Solver, WithinError,
};

fn additive() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: Default::default(),
        reduction: ReductionStrategy::ParallelReduction,
    }
}

struct Fixture {
    f: Vec<u32>,
    g: Vec<u32>,
    z: Vec<f64>,
    weights: Vec<f64>,
    y: Vec<f64>,
}

fn fixture() -> Fixture {
    let n = 800;
    Fixture {
        f: (0..n).map(|i| (i % 20) as u32).collect(),
        g: (0..n).map(|i| ((i / 20) % 10) as u32).collect(),
        z: (0..n)
            .map(|i| 0.5 + ((i * 13) % 100) as f64 / 100.0)
            .collect(),
        weights: (0..n).map(|i| (1 + i % 3) as f64).collect(),
        y: (0..n).map(|i| (i as f64 * 0.123).sin()).collect(),
    }
}

#[test]
fn untriggered_signal_keeps_weighted_slope_and_categorical_results_exact() {
    let Fixture {
        f,
        g,
        z,
        weights,
        y,
    } = fixture();
    let y2: Vec<_> = y.iter().map(|v| v * 3.0 + 0.2).collect();
    for workers in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                for sloped in [false, true] {
                    for weights in [None, Some(&weights[..])] {
                        let design = Design::new(vec![
                            if sloped {
                                Effect::new(&f, false, [&z[..]]).unwrap()
                            } else {
                                Effect::new(&f, true, []).unwrap()
                            },
                            Effect::new(&g, true, []).unwrap(),
                        ])
                        .unwrap();
                        for config in [
                            PreconditionerConfig::Off,
                            PreconditionerConfig::Diagonal,
                            additive(),
                            PreconditionerConfig::default(),
                        ] {
                            let solver = Solver::new(&design, weights, &config).unwrap();
                            let query = || false;
                            let cancellable = Solver::new_cancellable(
                                &design,
                                weights,
                                &config,
                                Cancellation::new(&query),
                            )
                            .unwrap();
                            let baseline = solver.solve_batch(&[&y, &y2], None).unwrap();
                            let actual = cancellable
                                .solve_batch_cancellable(
                                    &[&y, &y2],
                                    None,
                                    Cancellation::new(&query),
                                )
                                .unwrap();
                            assert_eq!(baseline.x, actual.x);
                            assert_eq!(baseline.demeaned, actual.demeaned);
                            assert_eq!(baseline.iterations, actual.iterations);
                            assert_eq!(baseline.converged, actual.converged);
                            assert_eq!(baseline.residual, actual.residual);
                            assert_eq!(baseline.unidentified, actual.unidentified);
                            assert_eq!(baseline.warnings, actual.warnings);
                        }
                    }
                }
            });
    }
}

#[test]
fn every_construction_and_solve_checkpoint_can_cancel() {
    let Fixture {
        f, g, weights, y, ..
    } = fixture();
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let design = Design::new(vec![
                Effect::new(&f, true, []).unwrap(),
                Effect::new(&g, true, []).unwrap(),
            ])
            .unwrap();
            let polls = AtomicUsize::new(0);
            let count = || {
                polls.fetch_add(1, Ordering::Relaxed);
                false
            };
            let solver = Solver::new_cancellable(
                &design,
                Some(&weights),
                additive(),
                Cancellation::new(&count),
            )
            .unwrap();
            let total = polls.load(Ordering::Relaxed);
            assert!(
                total > 8,
                "must reach pair construction and factorization checkpoints"
            );
            for stop in 1..=total {
                let polls = AtomicUsize::new(0);
                let query = || polls.fetch_add(1, Ordering::Relaxed) + 1 >= stop;
                assert!(
                    matches!(
                        Solver::new_cancellable(
                            &design,
                            Some(&weights),
                            additive(),
                            Cancellation::new(&query)
                        ),
                        Err(BuildError::Cancelled)
                    ),
                    "construction stop {stop}/{total}"
                );
            }
            polls.store(0, Ordering::Relaxed);
            let options = LsmrOptions {
                tol: 1e-12,
                ..Default::default()
            };
            let _ = solver
                .solve_cancellable(&y, &options, Cancellation::new(&count))
                .unwrap();
            let total = polls.load(Ordering::Relaxed);
            for stop in 1..=total {
                let polls = AtomicUsize::new(0);
                let query = || polls.fetch_add(1, Ordering::Relaxed) + 1 >= stop;
                assert!(
                    matches!(
                        solver.solve_cancellable(&y, &options, Cancellation::new(&query)),
                        Err(WithinError::Solve(SolveError::Cancelled))
                    ),
                    "solve stop {stop}/{total}"
                );
            }
            assert!(
                solver.solve(&y, &options).unwrap().converged,
                "cancelled solves leave the solver reusable"
            );
        });
}

#[test]
fn cancellation_after_design_error_is_distinct() {
    struct Invalid<'a>(&'a AtomicBool);
    impl<'a> IntoDesign<'a> for Invalid<'a> {
        fn into_design(self) -> Result<Design<'a>, BuildError> {
            self.0.store(true, Ordering::Relaxed);
            Err(BuildError::EmptyObservations)
        }
    }
    let cancelled = AtomicBool::new(false);
    let query = || cancelled.load(Ordering::Relaxed);
    assert!(matches!(
        Solver::new_cancellable(Invalid(&cancelled), None, None, Cancellation::new(&query)),
        Err(BuildError::Cancelled)
    ));
}

#[test]
fn cancellation_during_preconditioner_conversion_precedes_dimension_error() {
    struct Input<'a>(&'a AtomicBool, within::Preconditioner);
    impl From<Input<'_>> for within::PreconditionerInput {
        fn from(input: Input<'_>) -> Self {
            input.0.store(true, Ordering::Relaxed);
            Self::Prebuilt(input.1)
        }
    }
    let levels = [0u32, 0];
    let source = Solver::new(
        vec![Effect::new(&levels, true, []).unwrap()],
        None,
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    let preconditioner = source.preconditioner().unwrap().clone();
    let cancelled = AtomicBool::new(false);
    let query = || cancelled.load(Ordering::Relaxed);
    let target = [0u32, 1];
    assert!(matches!(
        Solver::new_cancellable(
            vec![Effect::new(&target, true, []).unwrap()],
            None,
            Input(&cancelled, preconditioner),
            Cancellation::new(&query),
        ),
        Err(BuildError::Cancelled)
    ));
}

#[test]
fn cancelled_batch_joins_in_flight_workers_and_returns_no_partial_batch() {
    let Fixture { f, g, y, .. } = fixture();
    let design = Design::new(vec![
        Effect::new(&f, true, []).unwrap(),
        Effect::new(&g, true, []).unwrap(),
    ])
    .unwrap();
    let solver = Solver::new(design, None, PreconditionerConfig::Diagonal).unwrap();
    let entered = AtomicUsize::new(0);
    let active = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let query = || {
        active.fetch_add(1, Ordering::SeqCst);
        let poll = entered.fetch_add(1, Ordering::SeqCst);
        if poll == 1 {
            std::thread::sleep(Duration::from_millis(20));
        }
        if poll >= 2 {
            cancelled.store(true, Ordering::SeqCst);
        }
        let result = cancelled.load(Ordering::SeqCst);
        active.fetch_sub(1, Ordering::SeqCst);
        result
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let outcome = pool.install(|| {
        solver.solve_batch_cancellable(&[&y, &y, &y, &y], None, Cancellation::new(&query))
    });
    assert!(matches!(
        outcome,
        Err(WithinError::Solve(SolveError::Cancelled))
    ));
    assert_eq!(active.load(Ordering::SeqCst), 0);
    let polls = entered.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(entered.load(Ordering::SeqCst), polls);
    assert!(solver.solve(&y, None).unwrap().converged);
}
