use anyhow::{Context, Result};
use std::path::Path;

const SAMPLE_RATE: u32 = 16_000;

pub(crate) fn load_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        anyhow::bail!("wav has zero channels: {}", path.display());
    }
    let channels = spec.channels as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => {
            let raw = reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read float samples from {}", path.display()))?;
            downmix_channels(&raw, channels)
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
            let raw = reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read int16 samples from {}", path.display()))?;
            downmix_int_channels_to_f32(&raw, channels, 32768.0)
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 32 => {
            let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
            let raw = reader
                .samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read int32 samples from {}", path.display()))?;
            downmix_int_channels_to_f32(&raw, channels, scale)
        }
        _ => anyhow::bail!(
            "unsupported wav format: sample_format={:?} bits={}",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(samples);
    }
    // TODO: replace with libsoxr-rs (pure-Rust port of libsoxr) to match
    // librosa's soxr_hq numerically. This linear-interpolation path lacks
    // anti-aliasing and is the sole remaining source of divergence from the
    // Python reference. See docs/ALIGNMENT_HANDOFF.md for the full diagnosis.
    Ok(resample_linear(&samples, spec.sample_rate, SAMPLE_RATE))
}

fn downmix_channels(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn downmix_int_channels_to_f32<T>(samples: &[T], channels: usize, scale: f32) -> Vec<f32>
where
    T: Copy + Into<i64>,
{
    if channels == 1 {
        return samples
            .iter()
            .map(|sample| (*sample).into() as f32 / scale)
            .collect();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame.iter().map(|sample| (*sample).into()).sum::<i64>();
            floor_div(sum, channels as i64) as f32 / scale
        })
        .collect()
}

fn floor_div(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && (remainder > 0) != (divisor > 0) {
        quotient - 1
    } else {
        quotient
    }
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    if source_rate.is_multiple_of(target_rate) {
        let step = (source_rate / target_rate) as usize;
        return samples.iter().step_by(step).copied().collect();
    }
    let target_len = ((samples.len() as u64 * target_rate as u64 + source_rate as u64 / 2)
        / source_rate as u64) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(target_len);
    for index in 0..target_len {
        let pos = index as f64 * ratio;
        let left = pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (pos - left as f64) as f32;
        output.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }
    output
}
