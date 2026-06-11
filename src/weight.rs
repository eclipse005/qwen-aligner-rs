//! CPU-side weight container shared by every engine backend.
//!
//! Lives outside `cudarc_engine` so the non-cuda build still has it (the CPU
//! engine, when it lands, also needs to consume `HashMap<String, WeightTensor>`
//! produced by the safetensors loader in `inference.rs`).

/// Minimal CPU-side weight container: f32 data + shape.  Produced by the
/// safetensors loader, consumed by whichever engine backend `load_model`
/// resolves at runtime.
pub struct WeightTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl WeightTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "WeightTensor data len mismatch");
        Self { data, shape }
    }
}
