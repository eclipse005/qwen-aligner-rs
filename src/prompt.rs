#![cfg_attr(
    not(feature = "cuda"),
    allow(dead_code)
)]
//! Rationale: audio-pad expansion is part of the alignment pipeline; with the
//! cuda feature off the alignment pipeline itself is unreachable, so these
//! helpers come along for the ride.  Once the CPU engine lands, the CPU
//! backend will call them and this allow-attr will go away.

use anyhow::Result;

pub(crate) fn feature_extract_output_len(input_len: i64) -> i64 {
    let input_len_leave = input_len % 100;
    let feat_len = (input_len_leave - 1).div_euclid(2) + 1;
    (((feat_len - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1) + (input_len / 100) * 13
}

pub(crate) fn expand_audio_pad_once(input: &str, count: usize) -> Result<String> {
    const AUDIO_PAD: &str = "<|audio_pad|>";
    if count == 0 {
        anyhow::bail!("audio pad expansion count must be positive");
    }
    let Some(pos) = input.find(AUDIO_PAD) else {
        anyhow::bail!("aligner input does not contain {AUDIO_PAD}");
    };

    let mut output = String::with_capacity(input.len() + AUDIO_PAD.len() * count.saturating_sub(1));
    output.push_str(&input[..pos]);
    for _ in 0..count {
        output.push_str(AUDIO_PAD);
    }
    output.push_str(&input[pos + AUDIO_PAD.len()..]);
    Ok(output)
}

#[cfg(test)]
mod tests {
    #[test]
    fn feature_len_matches_python_processor_formula() {
        assert_eq!(super::feature_extract_output_len(0), 0);
        assert_eq!(super::feature_extract_output_len(1), 1);
        assert_eq!(super::feature_extract_output_len(100), 13);
        assert_eq!(super::feature_extract_output_len(1500), 195);
    }

    #[test]
    fn expands_first_audio_pad_only() {
        let input = "<|audio_start|><|audio_pad|><|audio_end|> <|audio_pad|>";
        let output = super::expand_audio_pad_once(input, 3).unwrap();
        assert_eq!(
            output,
            "<|audio_start|><|audio_pad|><|audio_pad|><|audio_pad|><|audio_end|> <|audio_pad|>"
        );
    }
}
