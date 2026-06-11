use anyhow::{Context, Result};
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use std::collections::HashSet;
use std::sync::OnceLock;

const KOREAN_DICT: &str = include_str!("../assets/korean_dict_jieba.dict");

fn is_kept_char(ch: char) -> bool {
    ch == '\'' || ch.is_alphanumeric()
}

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

fn clean_token(token: &str) -> String {
    token.chars().filter(|&ch| is_kept_char(ch)).collect()
}

fn split_segment_with_chinese(seg: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();

    for ch in seg.chars() {
        if is_cjk_char(ch) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }

    if !buf.is_empty() {
        tokens.push(buf);
    }

    tokens
}

pub fn tokenize_space_lang(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for seg in text.split_whitespace() {
        let cleaned = clean_token(seg);
        if !cleaned.is_empty() {
            tokens.extend(split_segment_with_chinese(&cleaned));
        }
    }
    tokens
}

fn korean_scores() -> &'static HashSet<String> {
    static SCORES: OnceLock<HashSet<String>> = OnceLock::new();
    SCORES.get_or_init(|| {
        KOREAN_DICT
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .map(str::to_string)
            .collect()
    })
}

fn split_korean_lr(token: &str) -> (String, String) {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 2 {
        return (token.to_string(), String::new());
    }

    let scores = korean_scores();
    let mut best_left = token.to_string();
    let mut best_right = String::new();
    let mut best_score = i32::MIN;
    let mut best_left_len = 0usize;

    for end in 2..=chars.len() {
        let left: String = chars[..end].iter().collect();
        let right: String = chars[end..].iter().collect();
        let score = if scores.contains(&left) { 1 } else { 0 };
        if score > best_score || (score == best_score && end > best_left_len) {
            best_left = left;
            best_right = right;
            best_score = score;
            best_left_len = end;
        }
    }

    (best_left, best_right)
}

pub fn tokenize_korean(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for token in text.split_whitespace() {
        let (left, right) = split_korean_lr(token);
        for part in [left, right] {
            if part.is_empty() {
                continue;
            }
            let cleaned = clean_token(&part);
            if !cleaned.is_empty() {
                tokens.push(cleaned);
            }
        }
    }
    tokens
}

fn tokenize_japanese(text: &str) -> Result<Vec<String>> {
    static TOKENIZER: OnceLock<Result<LinderaTokenizer, String>> = OnceLock::new();
    let tokenizer = TOKENIZER
        .get_or_init(|| {
            let dictionary = load_dictionary("embedded://ipadic")
                .map_err(|err| format!("failed to load embedded Lindera IPADIC: {err}"))?;
            let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
            Ok(LinderaTokenizer::new(segmenter))
        })
        .as_ref()
        .map_err(|err| anyhow::anyhow!(err.clone()))?;

    let mut words = Vec::new();
    for token in tokenizer
        .tokenize(text)
        .context("failed to tokenize Japanese text with Lindera")?
    {
        let cleaned = clean_token(token.surface.as_ref());
        if !cleaned.is_empty() {
            words.push(cleaned);
        }
    }
    Ok(words)
}

pub fn encode_timestamp(text: &str, language: &str) -> Result<(Vec<String>, String)> {
    let lang = language.to_lowercase();
    let words = if lang == "japanese" {
        tokenize_japanese(text)?
    } else if lang == "korean" {
        tokenize_korean(text)
    } else {
        tokenize_space_lang(text)
    };

    let mut input = String::from("<|audio_start|><|audio_pad|><|audio_end|>");
    for word in &words {
        input.push_str(word);
        input.push_str("<timestamp><timestamp>");
    }
    Ok((words, input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_timestamp_text_matches_python_shape() {
        let (words, input) = encode_timestamp("All right, we purse.", "English").unwrap();
        assert_eq!(words, vec!["All", "right", "we", "purse"]);
        assert_eq!(
            input,
            "<|audio_start|><|audio_pad|><|audio_end|>All<timestamp><timestamp>right<timestamp><timestamp>we<timestamp><timestamp>purse<timestamp><timestamp>"
        );
    }

    #[test]
    fn keeps_apostrophe_and_splits_cjk_inside_space_language() {
        let (words, _) = encode_timestamp("don't ABC中国 123.", "English").unwrap();
        assert_eq!(words, vec!["don't", "ABC", "中", "国", "123"]);
    }

    #[test]
    fn japanese_timestamp_text_handles_basic_lindera_tokens() {
        let (words, _) =
            encode_timestamp("女子アナの仕事に耐える。辛抱大工です。", "Japanese").unwrap();
        assert_eq!(
            words,
            vec![
                "女子",
                "アナ",
                "の",
                "仕事",
                "に",
                "耐える",
                "辛抱",
                "大工",
                "です"
            ]
        );
    }

    #[test]
    fn japanese_timestamp_text_uses_lindera_ipadic_boundaries() {
        let (words, _) = encode_timestamp(
            "本日は島根県にある有名な人気ラーメン店にやってきました。",
            "Japanese",
        )
        .unwrap();
        assert_eq!(
            words,
            vec![
                "本日",
                "は",
                "島根",
                "県",
                "に",
                "ある",
                "有名",
                "な",
                "人気",
                "ラーメン",
                "店",
                "に",
                "やってき",
                "まし",
                "た"
            ]
        );
    }

    #[test]
    fn korean_timestamp_text_matches_soynlp_ltokenizer_reference() {
        let (words, _) = encode_timestamp("안녕하세요 여러분 나는 학교에 간다", "Korean").unwrap();
        assert_eq!(
            words,
            vec!["안녕", "하세요", "여러분", "나는", "학교", "에", "간다"]
        );
    }
}
