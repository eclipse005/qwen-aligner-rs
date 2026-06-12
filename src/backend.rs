//! Backend selection — public tag enum + private resolved form.
//!
//! `DeviceRequest` is the user-facing tag passed to `Qwen3ForcedAligner::load`.
//! `resolve()` produces a `ResolvedBackend` carrying any runtime state (GPU handle).

#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Pick which engine backend powers a `Qwen3ForcedAligner`.
///
/// `Auto` probes the most capable backend the binary was compiled with and
/// the host actually supports — currently CUDA first, then CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// Pure-CPU engine (gemm + rayon, no GPU dependencies).
    Cpu,
    /// CUDA engine on the given device ordinal (typically `Cuda(0)`).
    #[cfg(feature = "cuda")]
    Cuda(usize),
    /// Probe and pick the best available backend at load time.
    Auto,
}

impl Default for DeviceRequest {
    fn default() -> Self {
        DeviceRequest::Auto
    }
}

/// Internal resolved backend — carries `Arc<CudaState>` when CUDA is selected.
/// Never exposed in the public API.
pub(crate) enum ResolvedBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Arc<crate::cudarc_engine::CudaState>),
}

impl DeviceRequest {
    /// Resolve `Auto` to a concrete backend; leave explicit choices unchanged.
    pub(crate) fn resolve(self) -> anyhow::Result<ResolvedBackend> {
        match self {
            DeviceRequest::Cpu => Ok(ResolvedBackend::Cpu),
            #[cfg(feature = "cuda")]
            DeviceRequest::Cuda(ordinal) => {
                let state = crate::cudarc_engine::CudaState::new(ordinal)
                    .map_err(|e| anyhow::anyhow!("CUDA init failed (device {}): {}", ordinal, e))?;
                Ok(ResolvedBackend::Cuda(Arc::new(state)))
            }
            DeviceRequest::Auto => {
                #[cfg(feature = "cuda")]
                {
                    match crate::cudarc_engine::CudaState::new(0) {
                        Ok(state) => {
                            log::info!("Auto: selected CUDA device 0");
                            return Ok(ResolvedBackend::Cuda(Arc::new(state)));
                        }
                        Err(e) => {
                            log::warn!("Auto: CUDA init failed ({e}); falling back to CPU");
                        }
                    }
                }
                #[cfg(not(feature = "cpu"))]
                {
                    anyhow::bail!(
                        "Auto: no available backend (CPU engine not compiled in; \
                         CUDA backend either disabled at build time or unavailable at runtime)"
                    )
                }
                #[cfg(feature = "cpu")]
                {
                    log::info!("Auto: using CPU backend");
                    Ok(ResolvedBackend::Cpu)
                }
            }
        }
    }

    /// Short human label — useful for logs.
    pub fn tag(&self) -> &'static str {
        match self {
            DeviceRequest::Auto => "auto",
            DeviceRequest::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            DeviceRequest::Cuda(_) => "cuda",
        }
    }
}
