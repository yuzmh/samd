#![cfg(target_arch = "x86_64")]

use samd_x86::{X86Backend, best_x86_backend};
use std::arch::is_x86_feature_detected;

#[test]
fn detected_backend_matches_cpu_and_os_feature_support() {
    let backend = best_x86_backend();
    let avx2_available = is_x86_feature_detected!("avx2");
    let sse2_available = is_x86_feature_detected!("sse2");

    match backend {
        X86Backend::Avx2 => assert!(avx2_available),
        X86Backend::Sse2 => {
            assert!(!avx2_available);
            assert!(sse2_available);
        }
    }
}
