use within::{
    ApproxCholConfig, Effect, LocalSolverConfig, Preconditioner, PreconditionerConfig,
    ReductionStrategy, SchurMode, Solver,
};
pub fn build(case: &str, approximate: bool, weighted: bool) -> Preconditioner {
    let mut levels = [Vec::new(), Vec::new(), Vec::new()];
    let (a_count, b_count) = match case {
        "unbalanced" => (17, 3),
        "disconnected-interleaved" => (6, 6),
        _ => (5, 5),
    };
    for a in 0..a_count {
        for b in 0..b_count {
            if case == "disconnected-interleaved" && a % 2 != b % 2 {
                continue;
            }
            let c = match case {
                "nested" => a,
                "disconnected-interleaved" => (a + 2 * b) % 6,
                _ => (a + 2 * b) % 5,
            };
            for (v, id) in levels.iter_mut().zip([a, b, c]) {
                v.push(id);
            }
        }
    }
    let effects = levels
        .iter()
        .map(|v| Effect::new(v, true, std::iter::empty::<&[f64]>()).unwrap())
        .collect::<Vec<_>>();
    let weights = weighted.then(|| {
        (0..levels[0].len())
            .map(|i| 2.0_f64.powi((i % 7) as i32 - 3))
            .collect()
    });
    let config = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            dense_threshold: if approximate { 0 } else { 256 },
            schur: SchurMode::Exact,
            approx_chol: ApproxCholConfig {
                seed: 17,
                ..Default::default()
            },
            ..Default::default()
        },
        reduction: ReductionStrategy::AtomicScatter,
    };
    Solver::new(effects, weights, config)
        .unwrap()
        .preconditioner()
        .unwrap()
        .clone()
}
