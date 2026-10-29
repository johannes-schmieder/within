//! The caller's staging policy, coefficient frame, and independent least-squares witnesses.
use std::sync::{Arc, Mutex};
use within::{
    Effect, EscalationHandler, EscalationPolicy, LsmrOptions, LsmrStopReason, PreconditionerConfig,
    Progress, SolveControl, Solver,
};

struct StopAfter {
    iterations: usize,
    seen: Arc<Mutex<Vec<usize>>>,
}
impl EscalationPolicy for StopAfter {
    fn handler(&self) -> Box<dyn EscalationHandler> {
        Box::new(Self {
            iterations: self.iterations,
            seen: self.seen.clone(),
        })
    }
}
impl EscalationHandler for StopAfter {
    fn should_escalate(&mut self, progress: Progress) -> bool {
        self.seen.lock().unwrap().push(progress.iteration);
        progress.iteration >= self.iterations
    }
}

fn options(maxiter: usize) -> LsmrOptions {
    LsmrOptions {
        tol: 1e-12,
        maxiter,
        local_size: None,
    }
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 3e-9 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn caller_handoff_keeps_shared_budget_original_response_and_total_coefficients() {
    let n = 900;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 31) as u32).collect();
    let g: Vec<_> = (0..n).map(|i| ((i / 3 + i % 7) % 23) as u32).collect();
    let y: Vec<_> = (0..n)
        .map(|i| (i as f64 * 0.7).sin() + (i % 19) as f64)
        .collect();
    let w: Vec<_> = (0..n).map(|i| 0.1 + (i % 9) as f64).collect();
    let effects = || {
        vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ]
    };
    let diagonal = Solver::new(effects(), Some(&w), PreconditionerConfig::Diagonal).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stop = StopAfter {
        iterations: 2,
        seen: seen.clone(),
    };
    let first = diagonal
        .solve_controlled(
            &y,
            &options(100),
            SolveControl {
                initial_coefficients: None,
                escalation: Some(&stop),
                ..Default::default()
