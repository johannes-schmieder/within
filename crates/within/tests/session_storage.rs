//! Session storage changes must not move a bit: the separate step kernel keeps
//! the digests pinned at `efe35ee`. The fused kernel must repeat exactly and
//! agree across worker counts in default and reproducible mode, and agree
//! with the separate kernel within the solve tolerance.
//!
//! Separate reproducible solvers run at 1, 2, 4 and 8 workers and must match
//! the pinned digests; separate default solvers are checked at one worker,
//! where their reductions are serial.

use within::{
    Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy, SolveResult,
    Solver, StepKernel,
};

const ROWS: usize = 3 * 4096 + 7_001;

struct Data {
    worker: Vec<u32>,
    firm: Vec<u32>,
    third: Vec<u32>,
    y: Vec<f64>,
    weights: Vec<f64>,
}

fn data() -> Data {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let worker: Vec<u32> = (0..ROWS).map(|i| (i / 5) as u32).collect();
    let mut firm = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let moved = i % 5 == 0 || next() % 7 == 0;
        firm.push(if moved {
            (next() % 311) as u32
        } else {
            firm[i - 1]
        });
    }
    let third: Vec<u32> = (0..ROWS)
        .map(|i| (firm[i] + 17 * worker[i] + (i % 5) as u32) % 32)
        .collect();
    let y = (0..ROWS)
        .map(|i| {
            let noise = (next() % 10_000) as f64 / 10_000.0 - 0.5;
            0.01 * f64::from(worker[i] % 97) - 0.03 * f64::from(firm[i] % 13)
                + 0.2 * f64::from(third[i] % 3)
                + noise
        })
        .collect();
    let weights = (0..ROWS)
        .map(|i| 0.25 + (next() % 1_000) as f64 / 250.0 + if i % 11 == 0 { 0.0 } else { 0.5 })
        .collect();
    Data {
        worker,
        firm,
        third,
        y,
        weights,
    }
}

fn digest(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn result_digest(result: &SolveResult) -> u64 {
    digest(
        result
            .x
            .iter()
            .chain(&result.demeaned)
            .map(|value| value.to_bits())
            .chain([
                result.iterations as u64,
                u64::from(result.converged),
                result.residual.to_bits(),
            ]),
    )
}

fn preconditioner(name: &str) -> PreconditionerConfig {
    match name {
        "off" => PreconditionerConfig::Off,
        "diagonal" => PreconditionerConfig::Diagonal,
        _ => PreconditionerConfig::Additive {
            local_solver: LocalSolverConfig::default(),
            reduction: ReductionStrategy::Ordered,
        },
    }
}

/// One session flow per case: an unseeded or seeded start, a capped first
/// action, a continuation, a tighter residual correction and a snapshot.
fn run(
    data: &Data,
    weighted: bool,
    precond: &str,
    seeded: bool,
    reproducible: bool,
    kernel: StepKernel,
) -> ([u64; 3], SolveResult) {
    let effects = vec![
        Effect::new(&data.worker, true, std::iter::empty::<&[f64]>()).unwrap(),
        Effect::new(&data.firm, true, std::iter::empty::<&[f64]>()).unwrap(),
        Effect::new(&data.third, true, std::iter::empty::<&[f64]>()).unwrap(),
    ];
    let weights = weighted.then(|| data.weights.clone());
    let mut solver = Solver::new(effects, weights, preconditioner(precond))
        .unwrap()
        .with_step_kernel(kernel)
        .unwrap();
    if reproducible {
        solver = solver.with_reproducible().unwrap();
    }
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 400,
        local_size: None,
    };
    let mut session = if seeded {
        let seed: Vec<f64> = (0..solver.n_dofs())
            .map(|j| ((j * 37) % 23) as f64 / 230.0 - 0.05)
            .collect();
        solver
            .start_session_categorical_seeded(&data.y, &seed, &options)
            .unwrap()
    } else {
        solver.start_session(&data.y, &options).unwrap()
    };
    session.run_action(None, 3, None).unwrap();
    let snapshot = result_digest(&session.snapshot());
    session.run_action(None, usize::MAX, None).unwrap();
    let converged = result_digest(&session.snapshot());
    session.restart_correction(1e-11).unwrap();
    session.run_action(None, usize::MAX, None).unwrap();
    let finished = session.finish();
    ([snapshot, converged, result_digest(&finished)], finished)
}

fn at(
    threads: usize,
    work: impl FnOnce() -> ([u64; 3], SolveResult) + Send,
) -> ([u64; 3], SolveResult) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

/// `(weighted, preconditioner, seeded, reproducible)` and its pinned digests.
type Case = (bool, &'static str, bool, bool, [u64; 3]);

const PINNED: &[Case] = &[
    (
        false,
        "off",
        false,
        false,
        [0xe28b6e8dfebaacf0, 0x0c69abda9f12024b, 0x40558012b58fd450],
    ),
    (
        false,
        "off",
        false,
        true,
        [0xe28b6e8dfebaacf0, 0x0c69abda9f12024b, 0x40558012b58fd450],
    ),
    (
        false,
        "off",
        true,
        false,
        [0xdde22648961c9cbd, 0x79ae64c72c015b13, 0x8e760df4cd8709ba],
    ),
    (
        false,
        "off",
        true,
        true,
        [0xdde22648961c9cbd, 0x79ae64c72c015b13, 0x8e760df4cd8709ba],
    ),
    (
        false,
        "diagonal",
        false,
        false,
        [0xf0fabda98ab8eacc, 0x91f7fd183fc71307, 0x5d3c4098f3ccca16],
    ),
    (
        false,
        "diagonal",
        false,
        true,
        [0xf0fabda98ab8eacc, 0x91f7fd183fc71307, 0x5d3c4098f3ccca16],
    ),
    (
        false,
        "diagonal",
        true,
        false,
        [0xa3dcfccf4c31dde2, 0x7cfb2dbee80d23a8, 0xf17397cdc4c7db74],
    ),
    (
        false,
        "diagonal",
        true,
        true,
        [0xa3dcfccf4c31dde2, 0x7cfb2dbee80d23a8, 0xf17397cdc4c7db74],
    ),
    (
        false,
        "additive",
        false,
        false,
        [0x76f4cb3d452b53d5, 0x51d26ab08100e4ab, 0x2f212b83b8bfe97e],
    ),
    (
        false,
        "additive",
        false,
        true,
        [0x76f4cb3d452b53d5, 0x51d26ab08100e4ab, 0x2f212b83b8bfe97e],
    ),
    (
        false,
        "additive",
        true,
        false,
        [0xb1b3325b6bba088d, 0x420dff9149551a70, 0xa9b1dd8586714d01],
    ),
    (
        false,
        "additive",
        true,
        true,
        [0xb1b3325b6bba088d, 0x420dff9149551a70, 0xa9b1dd8586714d01],
    ),
    (
        true,
        "off",
        false,
        false,
        [0x266cc7952a5f33ea, 0x7d50fe68952470c3, 0x55d89bae18899c93],
    ),
    (
        true,
        "off",
        false,
        true,
        [0x266cc7952a5f33ea, 0x7d50fe68952470c3, 0x55d89bae18899c93],
    ),
    (
        true,
        "off",
        true,
        false,
        [0x50509f0af4280b62, 0xd06932dbade71e0d, 0xa169c34d7a2ad69e],
    ),
    (
        true,
        "off",
        true,
        true,
        [0x50509f0af4280b62, 0xd06932dbade71e0d, 0xa169c34d7a2ad69e],
    ),
    (
        true,
        "diagonal",
        false,
        false,
        [0xd0fadbfd4685f3db, 0xb1f44db48516aad1, 0xd587c4e50d7c6aea],
    ),
    (
        true,
        "diagonal",
        false,
        true,
        [0xd0fadbfd4685f3db, 0xb1f44db48516aad1, 0xd587c4e50d7c6aea],
    ),
    (
        true,
        "diagonal",
        true,
        false,
        [0xa3bd614af9e34160, 0x7bc369efd81f5d16, 0x8c37b3271bb539f5],
    ),
    (
        true,
        "diagonal",
        true,
        true,
        [0xa3bd614af9e34160, 0x7bc369efd81f5d16, 0x8c37b3271bb539f5],
    ),
    (
        true,
        "additive",
        false,
        false,
        [0x7d0ade5924498154, 0x04dd3616ff748455, 0xaac1677cb4e01b8b],
    ),
    (
        true,
        "additive",
        false,
        true,
        [0x7d0ade5924498154, 0x04dd3616ff748455, 0xaac1677cb4e01b8b],
    ),
    (
        true,
        "additive",
        true,
        false,
        [0x7d5d0a2b7269d6e6, 0xce96244eb6c67bc0, 0x6d9f160255a96437],
    ),
    (
        true,
        "additive",
        true,
        true,
        [0x7d5d0a2b7269d6e6, 0xce96244eb6c67bc0, 0x6d9f160255a96437],
    ),
];

#[test]
fn separate_kernel_sessions_match_pinned_digests() {
    let data = data();
    let mut observed = Vec::new();
    let mut failures = Vec::new();
    for weighted in [false, true] {
        for precond in ["off", "diagonal", "additive"] {
            for seeded in [false, true] {
                for reproducible in [false, true] {
                    let workers: &[usize] = if reproducible { &[1, 2, 4, 8] } else { &[1] };
                    let digests: Vec<[u64; 3]> = workers
                        .iter()
                        .map(|&threads| {
                            at(threads, || {
                                let kernel = StepKernel::Separate;
                                run(&data, weighted, precond, seeded, reproducible, kernel)
                            })
                            .0
                        })
                        .collect();
                    let case = (weighted, precond, seeded, reproducible);
                    if digests.iter().any(|d| *d != digests[0]) {
                        failures.push(format!("{case:?} differs across workers: {digests:x?}"));
                    }
                    match PINNED.iter().find(|p| (p.0, p.1, p.2, p.3) == case) {
                        Some(pinned) if pinned.4 == digests[0] => {}
                        Some(pinned) => failures.push(format!(
                            "{case:?} pinned {:x?}, observed {:x?}",
                            pinned.4, digests[0]
                        )),
                        None => failures.push(format!("{case:?} has no pinned digest")),
                    }
                    observed.push(format!(
                        "    ({weighted}, {precond:?}, {seeded}, {reproducible}, [{:#018x}, {:#018x}, {:#018x}]),",
                        digests[0][0], digests[0][1], digests[0][2]
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\nobserved:\n{}",
        failures.join("\n"),
        observed.join("\n")
    );
}

/// The fused kernel repeats exactly and agrees across worker counts in both
/// modes, and matches the separate kernel within the solve tolerance.
#[test]
fn fused_kernel_sessions_repeat_across_worker_counts_and_match_separate() {
    let data = data();
    let mut failures = Vec::new();
    for weighted in [false, true] {
        for precond in ["off", "diagonal", "additive"] {
            for seeded in [false, true] {
                for reproducible in [false, true] {
                    let case = (weighted, precond, seeded, reproducible);
                    let fused = |threads| {
                        at(threads, || {
                            run(
                                &data,
                                weighted,
                                precond,
                                seeded,
                                reproducible,
                                StepKernel::Fused,
                            )
                        })
                    };
                    let (reference, result) = fused(1);
                    for threads in [1, 2, 4, 8, 8] {
                        let digest = fused(threads).0;
                        if digest != reference {
                            failures.push(format!(
                                "{case:?} at {threads} workers: {digest:x?} vs {reference:x?}"
                            ));
                        }
                    }
                    let separate = at(1, || {
                        run(
                            &data,
                            weighted,
                            precond,
                            seeded,
                            reproducible,
                            StepKernel::Separate,
                        )
                    })
                    .1;
                    let gap = result
                        .demeaned
                        .iter()
                        .zip(&separate.demeaned)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    if !(result.converged && separate.converged && gap <= 1e-8) {
                        failures.push(format!("{case:?} demeaned gap {gap:e}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
