use schwarz_precond::{lsmr, mlsmr, MlsmrOptions, Operator, SolveError};

struct RectIdentity;

impl Operator for RectIdentity {
    fn nrows(&self) -> usize {
        3
    }

    fn ncols(&self) -> usize {
        2
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(&[x[0], x[1], 0.0]);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(&x[..2]);
        Ok(())
    }
}

struct DeflatingMetric;

impl Operator for DeflatingMetric {
    fn nrows(&self) -> usize {
        2
    }

    fn ncols(&self) -> usize {
        2
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y[0] = x[0];
        y[1] = 1e-20 * x[1];
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

struct DenseFourByTwo;

impl Operator for DenseFourByTwo {
    fn nrows(&self) -> usize {
        4
    }

    fn ncols(&self) -> usize {
        2
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(&[x[0], x[0] + x[1], x[1], x[0] - x[1]]);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(&[x[0] + x[1] + x[3], x[1] + x[2] - x[3]]);
        Ok(())
    }
}

struct IdentityMetric;

impl Operator for IdentityMetric {
    fn nrows(&self) -> usize {
        2
    }

    fn ncols(&self) -> usize {
        2
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }
}

#[test]
fn plain_exit_audit_refuses_a_metric_deflated_false_certificate() {
    let result = mlsmr(
        &RectIdentity,
        &[1.0, 1.0, 1.0],
        &DeflatingMetric,
        1e-8,
        20,
        MlsmrOptions::default(),
    )
    .expect("solve succeeds");

    assert!(
        !result.converged || (result.x[1] - 1.0).abs() <= 1e-8,
        "a converged solve left the deflated coordinate at {} after {} iterations",
        result.x[1],
        result.iterations,
    );
}

#[test]
fn warm_start_uses_the_original_rhs_for_the_plain_exit_audit() {
    let operator = DenseFourByTwo;
    let rhs = [1.0, 2.0, 3.0, 5.0];
    let solution = lsmr(&operator, &rhs, 1e-14, 100, None)
        .expect("cold solve succeeds")
        .x;
    let delta = 1e-11;
    let warm_start = [solution[0] + delta, solution[1] - 2.0 * delta];

    let result = mlsmr(
        &operator,
        &rhs,
        &IdentityMetric,
        1e-8,
        20,
        MlsmrOptions {
            warm_start: Some(&warm_start),
            ..Default::default()
        },
    )
    .expect("warm-start solve succeeds");

    assert!(result.converged, "warm-start solve must remain certified");
}
