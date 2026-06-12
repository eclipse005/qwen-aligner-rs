//! Safetensors weight loading — single-file and sharded index.

use std::collections::HashMap;
use std::path::Path;

use crate::raw_tensor::RawTensor;

pub(crate) fn load_weights(model_dir: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let wm = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid index.json"))?;
        let mut sf: std::collections::HashSet<String> = std::collections::HashSet::new();
        for v in wm.values() {
            if let Some(s) = v.as_str() {
                sf.insert(s.to_string());
            }
        }
        let mut all = HashMap::new();
        for s in sf {
            all.extend(load_safetensors_file(&model_dir.join(&s))?);
        }
        return Ok(all);
    }
    load_safetensors_file(&model_dir.join("model.safetensors"))
}

fn load_safetensors_file(path: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let buf = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| anyhow::anyhow!("safetensors: {}", e))?;
    let names = st.names();
    let tensors = st.tensors();
    let mut weights = HashMap::new();
    for i in 0..names.len() {
        let name = names[i];
        let view = &tensors[i];
        let data: Vec<u8> = view.1.data().to_vec();
        let shape: Vec<usize> = view.1.shape().to_vec();
        let dtype = view.1.dtype();
        weights.insert(
            name.to_string(),
            RawTensor { data, shape, dtype },
        );
    }
    Ok(weights)
}
