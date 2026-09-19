//! Measured concrete pair actions; does not generalize to every opaque factor provenance.
mod serial_support;
use stats_alloc::{StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let before = GLOBAL.stats();
    let control = black_box(vec![0.; 123]);
    assert_eq!((GLOBAL.stats() - before).allocations, 1);
    drop(control);
    for case in ["latin", "nested", "disconnected-interleaved", "unbalanced"] {
        for approximate in [false, true] {
            for weighted in [false, true] {
                // Setup is still parallel. Join every setup worker before
                // measuring the first serial action, so asynchronous worker TLS
                // activity cannot contaminate the process-wide allocator log.
                let mut setup_threads = Vec::new();
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .spawn_handler(|thread| {
                        setup_threads
                            .push(std::thread::Builder::new().spawn(move || thread.run())?);
                        Ok(())
                    })
                    .build()?;
                let owner = pool.install(|| serial_support::build(case, approximate, weighted));
                drop(pool);
                for thread in setup_threads {
                    thread.join().expect("setup worker completed");
                }
                let n = owner.nrows();
                let rhs: Vec<_> = (0..n).map(|i| (i % 7) as f64 / 8. - 0.5).collect();
                let mut out = vec![0.; n];
                let bytes = owner.serial_workspace_required_payload_bytes()?;
                let before = GLOBAL.stats();
                assert!(owner.try_serial_workspace(bytes - 1).is_err());
                let d = GLOBAL.stats() - before;
                assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 0));
                let before = GLOBAL.stats();
                let mut w = owner.try_serial_workspace(bytes)?;
                let d = GLOBAL.stats() - before;
                assert_eq!((d.allocations, d.reallocations, d.deallocations), (1, 0, 0));
                assert_eq!(d.bytes_allocated, bytes);
                assert_eq!(w.retained_payload_bytes(), bytes);
                let before = GLOBAL.stats();
                w.apply(black_box(&rhs), black_box(&mut out))?;
                let d = GLOBAL.stats() - before;
                assert_eq!(
                    (d.allocations, d.reallocations, d.deallocations),
                    (0, 0, 0),
                    "first {case} approximate={approximate} weighted={weighted}"
                );
                assert!(out.iter().all(|v| v.is_finite()));
                let expected = out.clone();
                let before = GLOBAL.stats();
                for _ in 0..64 {
                    w.apply(black_box(&rhs), black_box(&mut out))?;
                    assert_eq!(out, expected);
                }
                out.fill(9.);
                assert!(w.apply(&rhs[..n - 1], &mut out).is_err());
                assert!(out.iter().all(|&v| v == 9.));
                w.apply(&rhs, &mut out)?;
                assert_eq!(out, expected);
                let d = GLOBAL.stats() - before;
                assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 0));
                let before = GLOBAL.stats();
                drop(w);
                let d = GLOBAL.stats() - before;
                assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 1));
                assert_eq!(d.bytes_deallocated, bytes);
                println!("{case} approximate={approximate} weighted={weighted}: n={n} outer_bytes={bytes}; first+64 repeat/static rejection/reuse=0 allocations; exact drop");
            }
        }
    }
    Ok(())
}
