//! The checked serial build reproduces the parallel build exactly, honours its
//! limit inclusively and reports every refused reservation.

use super::*;
use crate::block_elim::solver::CHECKED_EXACT_FALLBACKS;
use crate::observation::ObservationFrame;
use crate::setup::Ledger;
use crate::{Effect, SetupLimit, Solver};

#[path = "../../../tests/checked_support/mod.rs"]
mod support;
use support::{applies, cases, configs, Case};

fn design(case: &Case) -> Design<'static> {
    let columns = case.levels.iter().map(|c| c.clone().into()).collect();
    Design::from_frame(ObservationFrame::new(columns, Vec::new()).expect("valid frame"))
        .expect("valid design")
}

/// The parallel build, given weights in internal order as [`Solver::new`] passes them.
fn reference(
    design: &Design<'_>,
    case: &Case,
    local_solver: &LocalSolverConfig,
    reduction: schwarz_precond::ReductionStrategy,
) -> Option<Preconditioner> {
    let weights = case
        .weights
        .as_deref()
        .map(|w| design.permute_obs_in(w).into_owned());
    let config = PreconditionerConfig::Additive {
        local_solver: local_solver.clone(),
        reduction,
    };
    build_preconditioner(design, weights.as_deref(), Some(&config))
        .expect("reference build")
        .0
}

fn wire(preconditioner: &Option<Preconditioner>) -> Option<Vec<u8>> {
    preconditioner
        .as_ref()
        .map(|p| postcard::to_stdvec(&p.inner).expect("serialize"))
}

#[test]
fn checked_builds_match_the_parallel_build_on_the_wire() {
    let mut compared = 0;
    let mut fallbacks_before = CHECKED_EXACT_FALLBACKS.with(|count| count.get());
    let mut fallback_cases = Vec::new();
    for case in cases() {
        let design = design(&case);
        for (name, local_solver, reduction) in configs() {
            if !applies(&case, name) {
                continue;
            }
            let label = format!("{} {name}", case.name);
            let expected = wire(&reference(&design, &case, &local_solver, reduction));
            let (checked, report) = Preconditioner::build_checked(
                &design,
                case.weights.as_deref(),
                &local_solver,
                reduction,
                SetupLimit::UNLIMITED,
            )
            .expect("checked build");
            assert_eq!(wire(&checked), expected, "{label}");
            let checked = checked.expect("every case has pair subdomains");
            assert!(report.peak_payload_bytes >= report.retained_payload_bytes);
            assert!(
                report.retained_payload_bytes
                    > checked
                        .local_factor_retained_heap_bytes()
                        .expect("factor capacity fits usize"),
                "{label}: the retained payload includes the factors and more"
            );
            let fallbacks = CHECKED_EXACT_FALLBACKS.with(|count| count.get());
            if fallbacks > fallbacks_before {
                fallback_cases.push(label);
                fallbacks_before = fallbacks;
            }
            compared += 1;
        }
    }
    let applicable = cases()
        .iter()
        .map(|case| {
            configs()
                .iter()
                .filter(|(name, ..)| applies(case, name))
                .count()
        })
        .sum::<usize>();
    assert_eq!(compared, applicable);
    assert!(
        fallback_cases.len() >= 2,
        "a failed exact attempt and its fallback are exercised"
    );
    println!("checked wire identity: {compared} builds, exact fallbacks in {fallback_cases:?}");
}

#[test]
fn checked_builds_match_the_public_solver_route() {
    for case in cases().iter().filter(|case| !case.large) {
        let (name, local_solver, reduction) = configs().swap_remove(0);
        let checked = Preconditioner::build_checked(
            &design(case),
            case.weights.as_deref(),
            &local_solver,
            reduction,
            SetupLimit::UNLIMITED,
        )
        .expect("checked build")
        .0;
        let effects: Vec<Effect<'_>> = case
            .levels
            .iter()
            .map(|levels| Effect::new(levels, true, []).expect("intercept effect"))
            .collect();
        let solver = Solver::new(
            effects,
            case.weights.clone(),
            PreconditionerConfig::Additive {
                local_solver,
                reduction,
            },
        )
        .expect("solver");
        assert_eq!(
            wire(&checked),
            wire(&solver.preconditioner().cloned()),
            "{} {name}",
            case.name
        );
    }
}

#[test]
fn single_threaded_actions_agree_bit_for_bit() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    for case in cases().iter().filter(|case| !case.large) {
        let design = design(case);
        for (name, local_solver, reduction) in configs() {
            let expected = reference(&design, case, &local_solver, reduction).expect("domains");
            let (checked, _) = Preconditioner::build_checked(
                &design,
                case.weights.as_deref(),
                &local_solver,
                reduction,
                SetupLimit::UNLIMITED,
            )
            .expect("checked build");
            let checked = checked.expect("domains");
            let x: Vec<f64> = (0..design.n_dofs)
                .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
                .collect();
            let (mut a, mut b) = (vec![0.0; x.len()], vec![0.0; x.len()]);
            pool.install(|| {
                expected.apply(&x, &mut a).expect("reference action");
                checked.apply(&x, &mut b).expect("checked action");
            });
            for (u, v) in a.iter().zip(&b) {
                assert_eq!(u.to_bits(), v.to_bits(), "{} {name}", case.name);
            }
        }
    }
}

#[test]
fn limits_are_inclusive_and_never_change_the_result() {
    for case in cases().iter().filter(|case| !case.large) {
        let design = design(case);
        for (name, local_solver, reduction) in configs().into_iter().take(2) {
            let label = format!("{} {name}", case.name);
            let build = |limit| {
                Preconditioner::build_checked(
                    &design,
                    case.weights.as_deref(),
                    &local_solver,
                    reduction,
                    SetupLimit {
                        maximum_payload_bytes: limit,
                    },
                )
            };
            let (unlimited, report) = build(usize::MAX).expect("unlimited");
            let peak = report.peak_payload_bytes;
            let (at_peak, again) = build(peak).expect("the peak is admissible");
            assert_eq!(again, report, "{label}: accounting is deterministic");
            assert_eq!(wire(&at_peak), wire(&unlimited), "{label}");
            for limit in [peak - 1, peak / 2, report.retained_payload_bytes / 2, 0] {
                match build(limit) {
                    Err(BuildError::SetupBudgetExceeded {
                        required,
                        limit: refused,
                        ..
                    }) => {
                        assert_eq!(refused, limit, "{label}");
                        assert!(required > limit && required <= peak, "{label} at {limit}");
                    }
                    Err(other) => panic!("{label} at {limit}: {other}"),
                    Ok(_) => panic!("{label} at {limit}: admitted below the peak"),
                }
            }
        }
    }
}

#[test]
fn every_reservation_failure_is_reported() {
    let mut injected = 0;
    for case in cases().iter().filter(|case| {
        matches!(
            case.name.as_str(),
            "random3-60-zeros"
                | "interleaved-two-factor"
                | "nested-sorted"
                | "pivot-cancellation-twice"
        )
    }) {
        let design = design(case);
        for (name, local_solver, reduction) in configs() {
            let build = |ledger: &mut Ledger| {
                Preconditioner::build_checked_in(
                    &design,
                    case.weights.as_deref(),
                    &local_solver,
                    reduction,
                    ledger,
                )
            };
            let mut ledger = Ledger::new(SetupLimit::UNLIMITED);
            build(&mut ledger).expect("unfaulted build");
            let reservations = ledger.reservations();
            assert!(reservations > 10, "{} {name}", case.name);
            for at in 1..=reservations {
                let mut ledger = Ledger::failing_at(SetupLimit::UNLIMITED, at);
                match build(&mut ledger) {
                    Err(BuildError::SetupAllocationFailed { .. }) => injected += 1,
                    Err(other) => panic!("{} {name} at {at}: {other}", case.name),
                    Ok(_) => panic!("{} {name}: reservation {at} failed silently", case.name),
                }
            }
        }
    }
    assert!(injected > 1000, "{injected} injected failures");
}

#[test]
fn slope_designs_are_refused_before_reserving() {
    let levels = [0u32, 1, 2, 0, 1, 2];
    let other = [0u32, 1, 0, 1, 0, 1];
    let z = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5];
    let design = Design::new(vec![
        Effect::new(&levels, true, [&z[..]]).expect("slope effect"),
        Effect::new(&other, true, []).expect("intercept effect"),
    ])
    .expect("design");
    let mut ledger = Ledger::new(SetupLimit::UNLIMITED);
    let refused = Preconditioner::build_checked_in(
        &design,
        None,
        &LocalSolverConfig::default(),
        schwarz_precond::ReductionStrategy::Auto,
        &mut ledger,
    );
    assert!(matches!(
        refused,
        Err(BuildError::CheckedSetupUnsupported { .. })
    ));
    assert_eq!(ledger.report().peak_payload_bytes, 0);
    assert_eq!(ledger.reservations(), 0);
}

#[test]
fn designs_without_pairs_build_nothing_and_retain_nothing() {
    let case = Case {
        name: "single".into(),
        levels: vec![vec![0, 1, 2, 1]],
        weights: None,
        large: false,
    };
    let design = design(&case);
    let local_solver = LocalSolverConfig::default();
    let reduction = schwarz_precond::ReductionStrategy::Auto;
    assert!(reference(&design, &case, &local_solver, reduction).is_none());
    let (checked, report) = Preconditioner::build_checked(
        &design,
        None,
        &local_solver,
        reduction,
        SetupLimit::UNLIMITED,
    )
    .expect("checked build");
    assert!(checked.is_none());
    assert_eq!(report.retained_payload_bytes, 0);
}

#[test]
fn invalid_inputs_are_rejected_as_by_the_solver() {
    let case = &cases()[0];
    let design = design(case);
    let local_solver = LocalSolverConfig::default();
    let reduction = schwarz_precond::ReductionStrategy::Auto;
    let short = vec![1.0; design.n_obs() - 1];
    assert!(matches!(
        Preconditioner::build_checked(
            &design,
            Some(&short),
            &local_solver,
            reduction,
            SetupLimit::UNLIMITED
        ),
        Err(BuildError::WeightCountMismatch { .. })
    ));
    let mut negative = vec![1.0; design.n_obs()];
    negative[3] = -1.0;
    assert!(matches!(
        Preconditioner::build_checked(
            &design,
            Some(&negative),
            &local_solver,
            reduction,
            SetupLimit::UNLIMITED
        ),
        Err(BuildError::InvalidWeight { index: 3, .. })
    ));
    let ridged = LocalSolverConfig {
        ridge: f64::NAN,
        ..local_solver
    };
    assert!(matches!(
        Preconditioner::build_checked(&design, None, &ridged, reduction, SetupLimit::UNLIMITED),
        Err(BuildError::InvalidRidge { .. })
    ));
}
