//! Deterministic intercept-only inputs and configurations for checked-setup qualification.
#![allow(dead_code)]
use within::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, ReductionStrategy, SchurMode,
};

pub struct Case {
    pub name: String,
    /// One level column per factor, in caller observation order.
    pub levels: Vec<Vec<u32>>,
    pub weights: Option<Vec<f64>>,
    /// Too large for the configurations that factor every kept block exactly.
    pub large: bool,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }

    fn dyadic(&mut self) -> f64 {
        2.0_f64.powi(self.next(9) as i32 - 4)
    }
}

fn case(name: impl Into<String>, levels: Vec<Vec<u32>>, weights: Option<Vec<f64>>) -> Case {
    Case {
        name: name.into(),
        levels,
        weights,
        large: false,
    }
}

pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let mut random = Lcg(0x7769_7468);
    for &n in &[60usize, 600, 3000] {
        let a = (n / 10).max(3);
        let b = (n / 25).max(3);
        let levels = vec![
            (0..n).map(|_| random.next(a) as u32).collect(),
            (0..n).map(|_| random.next(b) as u32).collect(),
            (0..n).map(|_| random.next(7) as u32).collect(),
        ];
        let dyadic: Vec<f64> = (0..n).map(|_| random.dyadic()).collect();
        // A zero weight can leave a level active but dead.
        let zeros: Vec<f64> = (0..n)
            .map(|i| if i % 11 == 3 { 0.0 } else { random.dyadic() })
            .collect();
        out.push(case(format!("random3-{n}-unit"), levels.clone(), None));
        out.push(case(
            format!("random3-{n}-dyadic"),
            levels.clone(),
            Some(dyadic),
        ));
        out.push(case(format!("random3-{n}-zeros"), levels, Some(zeros)));
    }

    // Interleaved components: factor B only meets factor A levels of one residue mod 4.
    let n = 800;
    let a: Vec<u32> = (0..n).map(|_| random.next(80) as u32).collect();
    let b: Vec<u32> = a
        .iter()
        .map(|&a| (a % 4) + 4 * random.next(10) as u32)
        .collect();
    let c: Vec<u32> = (0..n).map(|_| random.next(5) as u32).collect();
    out.push(case("interleaved", vec![a.clone(), b.clone(), c], None));
    out.push(case(
        "interleaved-two-factor",
        vec![a, b],
        Some((0..n).map(|_| random.dyadic()).collect()),
    ));

    // B nested in A, as tuples sorted canonically (the MultiwayMG input shape).
    let mut tuples: Vec<[u32; 3]> = (0..900)
        .map(|_| {
            let a = random.next(30) as u32;
            [a, 3 * a + random.next(3) as u32, random.next(12) as u32]
        })
        .collect();
    tuples.sort_unstable();
    tuples.dedup();
    let sorted_levels: Vec<Vec<u32>> = (0..3)
        .map(|f| tuples.iter().map(|t| t[f]).collect())
        .collect();
    let sorted_weights: Vec<f64> = (0..tuples.len()).map(|_| random.dyadic()).collect();
    out.push(case("nested-sorted", sorted_levels, Some(sorted_weights)));

    // Unused codes inflate the level count without becoming active.
    let n = 500;
    out.push(case(
        "unused-levels",
        vec![
            (0..n).map(|_| 5 * random.next(40) as u32).collect(),
            (0..n).map(|_| 3 * random.next(20) as u32 + 1).collect(),
            (0..n).map(|_| random.next(9) as u32).collect(),
        ],
        Some((0..n).map(|_| random.dyadic()).collect()),
    ));

    // Four factors and heterogeneous extreme weights.
    let n = 700;
    out.push(case(
        "four-factor-extreme",
        vec![
            (0..n).map(|_| random.next(50) as u32).collect(),
            (0..n).map(|_| random.next(20) as u32).collect(),
            (0..n).map(|_| random.next(8) as u32).collect(),
            (0..n).map(|_| random.next(4) as u32).collect(),
        ],
        Some(
            (0..n)
                .map(|i| 2.0_f64.powi(if i % 2 == 0 { 30 } else { -30 }) * random.dyadic())
                .collect(),
        ),
    ));

    // Near-cancelling weights, the inputs most likely to defeat an exact pivot.
    let n = 400;
    out.push(case(
        "tiny-weights",
        vec![
            (0..n).map(|_| random.next(30) as u32).collect(),
            (0..n).map(|_| random.next(12) as u32).collect(),
        ],
        Some(
            (0..n)
                .map(|i| if i % 3 == 0 { 1e-20 } else { 1.0 })
                .collect(),
        ),
    ));

    // Each A level joins two B levels, so the kept Schur complement is the floating path
    // b0 -1- b1 -1e-20- b2 -1- b3, whose dense exact factor meets a non-positive pivot
    // and falls back. A repeated copy keeps the fallback inside a larger pair.
    let pivot_a = [0u32, 0, 1, 1, 2, 2, 3];
    let pivot_b = [0u32, 1, 1, 2, 2, 3, 3];
    let pivot_w = [2.0, 2.0, 2e-20, 2e-20, 2.0, 2.0, 1.0];
    out.push(case(
        "pivot-cancellation",
        vec![pivot_a.to_vec(), pivot_b.to_vec()],
        Some(pivot_w.to_vec()),
    ));
    out.push(case(
        "pivot-cancellation-twice",
        vec![
            pivot_a
                .iter()
                .chain(&pivot_a)
                .enumerate()
                .map(|(i, &a)| a + 4 * u32::from(i >= 7))
                .collect(),
            pivot_b
                .iter()
                .chain(&pivot_b)
                .enumerate()
                .map(|(i, &b)| b + 4 * u32::from(i >= 7))
                .collect(),
        ],
        Some(pivot_w.iter().chain(&pivot_w).copied().collect()),
    ));

    // A pair whose table exceeds the dense cap with few observations: sparse accumulation.
    let n = 12_000;
    let mut sparse = case(
        "sparse-accumulation",
        vec![
            (0..n).map(|_| random.next(2_500) as u32).collect(),
            (0..n).map(|_| random.next(2_400) as u32).collect(),
        ],
        Some((0..n).map(|_| random.dyadic()).collect()),
    );
    sparse.large = true;
    out.push(sparse);
    out
}

pub fn configs() -> Vec<(&'static str, LocalSolverConfig, ReductionStrategy)> {
    let base = LocalSolverConfig::default();
    vec![
        ("default", base.clone(), ReductionStrategy::Auto),
        (
            "exact-schur",
            LocalSolverConfig {
                schur: SchurMode::Exact,
                ..base.clone()
            },
            ReductionStrategy::AtomicScatter,
        ),
        (
            "no-dense",
            LocalSolverConfig {
                dense_threshold: 0,
                ..base.clone()
            },
            ReductionStrategy::ParallelReduction,
        ),
        (
            "all-dense",
            LocalSolverConfig {
                dense_threshold: 100_000,
                ..base.clone()
            },
            ReductionStrategy::Auto,
        ),
        (
            "ac-split3",
            LocalSolverConfig {
                approx_chol: ApproxCholConfig {
                    seed: 5,
                    split_merge: None,
                },
                schur: SchurMode::Approximate(ApproxSchurConfig { seed: 9, split: 3 }),
                ..base
            },
            ReductionStrategy::Auto,
        ),
    ]
}

/// Configurations that factor every kept block exactly are skipped on large cases.
pub fn applies(case: &Case, config: &str) -> bool {
    !(case.large && matches!(config, "exact-schur" | "all-dense"))
}
