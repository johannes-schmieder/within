//! Isolated allocator proof that the checked setup ledger charges every allocation.
//!
//! A counting allocator records live bytes and their peak on this thread. Every
//! checked build must stay within its reported peak and retain exactly its reported
//! payload, dropping the preconditioner must release it, and a refused build must
//! stay within its limit and release everything.
mod checked_support;
use checked_support::{applies, cases, configs};
use within::{BuildError, Design, Effect, Preconditioner, SetupLimit};

/// Peak and net live bytes of `run` on this thread. The counting allocator's
/// default reallocation allocates the new buffer before freeing the old one, which
/// is the overlap the ledger charges.
fn measure<R>(run: impl FnOnce() -> R) -> (R, usize, i64) {
    let mut result = None;
    let info = allocation_counter::measure(|| result = Some(run()));
    (result.unwrap(), info.bytes_max as usize, info.bytes_current)
}

fn main() {
    let mut builds = 0;
    let mut slack_max = 0usize;
    for case in cases() {
        let effects: Vec<Effect<'_>> = case
            .levels
            .iter()
            .map(|levels| Effect::new(levels, true, []).expect("intercept effect"))
            .collect();
        let design = Design::new(effects).expect("design");
        for (name, local_solver, reduction) in configs() {
            if !applies(&case, name) {
                continue;
            }
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
            let (built, peak, retained) = measure(|| build(usize::MAX));
            let (preconditioner, report) = built.expect("checked build");
            assert!(
                peak <= report.peak_payload_bytes,
                "{label}: allocator peak {peak} exceeds ledger peak {}",
                report.peak_payload_bytes
            );
            assert_eq!(
                retained, report.retained_payload_bytes as i64,
                "{label}: retained payload"
            );
            slack_max = slack_max.max(report.peak_payload_bytes - peak);
            let (_, _, released) = measure(|| drop(preconditioner));
            assert_eq!(released, -retained, "{label}: drop releases the payload");

            for limit in [report.peak_payload_bytes - 1, report.peak_payload_bytes / 3] {
                let (refused, peak, retained) = measure(|| build(limit));
                assert!(
                    matches!(refused, Err(BuildError::SetupBudgetExceeded { .. })),
                    "{label} at {limit}"
                );
                assert!(
                    peak <= limit,
                    "{label}: refused build peaked at {peak} > {limit}"
                );
                // The error itself owns no heap payload.
                assert_eq!(retained, 0, "{label}: refused build leaked");
            }
            println!(
                "CHECKED_SETUP case={} config={name} peak={} allocator_peak={peak} retained={retained}",
                case.name, report.peak_payload_bytes
            );
            builds += 1;
        }
    }
    assert!(builds >= 80, "{builds} builds");
    println!("CHECKED_SETUP_SUMMARY builds={builds} maximum_conservative_slack_bytes={slack_max}");
}
