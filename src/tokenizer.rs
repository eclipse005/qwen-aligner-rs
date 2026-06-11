use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    added_tokens_decoder: BTreeMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    configured_added_token_ids: HashMap<String, u32>,
}

impl QwenTokenizer {
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.configured_added_token_ids
            .get(token)
            .copied()
            .or_else(|| self.tokenizer.token_to_id(token))
    }
}

pub fn load_qwen_tokenizer(model_dir: &Path) -> Result<QwenTokenizer> {
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let config_path = model_dir.join("tokenizer_config.json");

    let bpe = BPE::from_file(
        vocab.to_string_lossy().as_ref(),
        merges.to_string_lossy().as_ref(),
    )
    .build()
    .map_err(|err| anyhow::anyhow!("failed to build BPE from {}: {err}", model_dir.display()))?;

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));

    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: TokenizerConfig = serde_json::from_str(&config_text)
        .with_context(|| format!("invalid json {}", config_path.display()))?;

    let mut normal_tokens = Vec::new();
    let mut special_tokens = Vec::new();
    let mut configured_added_token_ids = HashMap::new();
    for (id, item) in &config.added_tokens_decoder {
        let token = AddedToken::from(item.content.clone(), item.special)
            .lstrip(item.lstrip)
            .rstrip(item.rstrip)
            .single_word(item.single_word)
            .normalized(item.normalized);
        configured_added_token_ids.insert(
            item.content.clone(),
            id.parse::<u32>()
                .with_context(|| format!("invalid added token id {id}"))?,
        );
        if item.special {
            special_tokens.push(token);
        } else {
            normal_tokens.push(token);
        }
    }
    tokenizer.add_special_tokens(&special_tokens);
    tokenizer.add_tokens(&normal_tokens);

    Ok(QwenTokenizer {
        tokenizer,
        configured_added_token_ids,
    })
}

pub fn encode_to_ids(tokenizer: &QwenTokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .tokenizer
        .encode(text, false)
        .map_err(|err| anyhow::anyhow!("tokenizer encode failed: {err}"))?;
    Ok(encoding
        .get_tokens()
        .iter()
        .zip(encoding.get_ids())
        .map(|(token, id)| {
            tokenizer
                .configured_added_token_ids
                .get(token)
                .copied()
                .unwrap_or(*id)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_token_ids_are_preserved() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("Qwen3-ForcedAligner-0.6B");
        let tokenizer = load_qwen_tokenizer(&root).unwrap();
        assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(151669));
        assert_eq!(tokenizer.token_to_id("<|audio_end|>"), Some(151670));
        assert_eq!(tokenizer.token_to_id("<|audio_pad|>"), Some(151676));
        assert_eq!(tokenizer.token_to_id("<timestamp>"), Some(151705));
    }
}
