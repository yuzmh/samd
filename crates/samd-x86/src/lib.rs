#![no_std]
//! Native `x86_64` SIMD backends and runtime feature selection for SAMD.

#[cfg(target_arch = "x86_64")]
use core::arch::asm;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__cpuid, __cpuid_count};
#[cfg(target_arch = "x86_64")]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod sse2;

/// The `x86_64` backend selected from the current CPU and operating-system state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86Backend {
    /// AVX2 is supported by the CPU and the OS saves/restores YMM state.
    Avx2,
    /// SSE2 is supported by the CPU.
    Sse2,
}

#[cfg(target_arch = "x86_64")]
static DETECTED_BACKEND: AtomicU8 = AtomicU8::new(0);

/// Detects the strongest implemented x86 backend available to this process.
///
/// AVX2 selection checks CPU support, OSXSAVE, and XCR0 YMM state before it can
/// select code that uses AVX instructions. The function uses no allocation and
/// does not require `std`.
#[must_use]
#[cfg(target_arch = "x86_64")]
pub fn best_x86_backend() -> X86Backend {
    match DETECTED_BACKEND.load(Ordering::Relaxed) {
        1 => return X86Backend::Sse2,
        2 => return X86Backend::Avx2,
        _ => {}
    }

    let backend = detect_x86_backend();
    let encoded = match backend {
        X86Backend::Sse2 => 1,
        X86Backend::Avx2 => 2,
    };
    let _ = DETECTED_BACKEND.compare_exchange(0, encoded, Ordering::Relaxed, Ordering::Relaxed);
    backend
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_backend() -> X86Backend {
    let max_leaf = __cpuid(0).eax;
    if max_leaf < 1 {
        return X86Backend::Sse2;
    }

    let leaf1 = __cpuid(1);
    let has_avx = (leaf1.ecx & (1 << 28)) != 0;
    let has_osxsave = (leaf1.ecx & (1 << 27)) != 0;

    if has_avx && has_osxsave && max_leaf >= 7 && ymm_state_enabled() {
        let leaf7 = __cpuid_count(7, 0);
        if (leaf7.ebx & (1 << 5)) != 0 {
            return X86Backend::Avx2;
        }
    }

    // SSE2 is part of the x86_64 target baseline.
    X86Backend::Sse2
}

#[cfg(target_arch = "x86_64")]
fn ymm_state_enabled() -> bool {
    let low: u32;
    let high: u32;
    // SAFETY: called only after CPUID reports OSXSAVE, which guarantees XGETBV
    // is enabled by the operating system. XCR0 index zero reports enabled state.
    unsafe {
        asm!(
            "xgetbv",
            in("ecx") 0_u32,
            out("eax") low,
            out("edx") high,
            options(nomem, nostack, preserves_flags),
        );
    }
    (u64::from(high) << 32 | u64::from(low)) & 0x6 == 0x6
}
