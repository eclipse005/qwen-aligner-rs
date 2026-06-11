use anyhow::Result;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const MEL_BINS: usize = 128;
const FREQ_BINS: usize = N_FFT / 2 + 1;

pub struct LogMelFeatures {
    pub values: Vec<f32>,
    pub mel_bins: usize,
    pub frames: usize,
    pub attention_mask: Vec<i32>,
}

pub fn extract_log_mel_features(waveform: &[f32]) -> Result<LogMelFeatures> {
    if waveform.len() <= N_FFT {
        anyhow::bail!("waveform is too short: {} samples", waveform.len());
    }

    let frames = waveform.len() / HOP_LENGTH;
    let padded = reflect_pad(waveform, N_FFT / 2)?;
    let window = hann_window(N_FFT);
    let mel_filters = mel_filter_bank();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut fft_input = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0_f32; FREQ_BINS];
    let mut features = vec![0.0_f32; MEL_BINS * frames];

    for frame in 0..frames {
        let offset = frame * HOP_LENGTH;
        for i in 0..N_FFT {
            fft_input[i] = Complex32::new(padded[offset + i] * window[i], 0.0);
        }
        fft.process(&mut fft_input);

        for bin in 0..FREQ_BINS {
            power[bin] = fft_input[bin].norm_sqr();
        }

        for mel in 0..MEL_BINS {
            let mut value = 0.0_f32;
            for bin in 0..FREQ_BINS {
                value += mel_filters[bin * MEL_BINS + mel] * power[bin];
            }
            features[mel * frames + frame] = value.max(1e-10).log10();
        }
    }

    let max_value = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_value - 8.0;
    for value in &mut features {
        *value = (value.max(floor) + 4.0) / 4.0;
    }

    Ok(LogMelFeatures {
        values: features,
        mel_bins: MEL_BINS,
        frames,
        attention_mask: vec![1; frames],
    })
}

fn reflect_pad(waveform: &[f32], pad: usize) -> Result<Vec<f32>> {
    if waveform.len() <= pad {
        anyhow::bail!(
            "waveform length {} must be greater than reflect pad {}",
            waveform.len(),
            pad
        );
    }

    let len = waveform.len() as isize;
    let mut output = Vec::with_capacity(waveform.len() + pad * 2);
    for padded_idx in 0..(waveform.len() + pad * 2) {
        let idx = padded_idx as isize - pad as isize;
        let reflected = if idx < 0 {
            -idx
        } else if idx >= len {
            2 * len - 2 - idx
        } else {
            idx
        };
        output.push(waveform[reflected as usize]);
    }
    Ok(output)
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = std::f32::consts::TAU * i as f32 / size as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

fn mel_filter_bank() -> Vec<f32> {
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(8_000.0);
    let mel_freqs = linspace(mel_min, mel_max, MEL_BINS + 2);
    let filter_freqs: Vec<f32> = mel_freqs.iter().map(|mel| mel_to_hertz(*mel)).collect();
    let fft_freqs = linspace(0.0, (SAMPLE_RATE / 2) as f32, FREQ_BINS);

    let mut filters = vec![0.0_f32; FREQ_BINS * MEL_BINS];
    for bin in 0..FREQ_BINS {
        for mel in 0..MEL_BINS {
            let down =
                (fft_freqs[bin] - filter_freqs[mel]) / (filter_freqs[mel + 1] - filter_freqs[mel]);
            let up = (filter_freqs[mel + 2] - fft_freqs[bin])
                / (filter_freqs[mel + 2] - filter_freqs[mel + 1]);
            let enorm = 2.0 / (filter_freqs[mel + 2] - filter_freqs[mel]);
            filters[bin * MEL_BINS + mel] = down.min(up).max(0.0) * enorm;
        }
    }
    filters
}

fn linspace(start: f32, end: f32, count: usize) -> Vec<f32> {
    if count == 1 {
        return vec![start];
    }
    let step = (end - start) / (count - 1) as f32;
    (0..count).map(|i| start + step * i as f32).collect()
}

fn hertz_to_mel(freq: f64) -> f32 {
    let freq = freq as f32;
    let min_log_hertz = 1000.0_f32;
    let min_log_mel = 15.0_f32;
    let logstep = 27.0 / 6.4_f32.ln();
    let mel = 3.0 * freq / 200.0;
    if freq >= min_log_hertz {
        min_log_mel + (freq / min_log_hertz).ln() * logstep
    } else {
        mel
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    let min_log_hertz = 1000.0_f32;
    let min_log_mel = 15.0_f32;
    let logstep = 6.4_f32.ln() / 27.0;
    let freq = 200.0 * mel / 3.0;
    if mel >= min_log_mel {
        min_log_hertz * (logstep * (mel - min_log_mel)).exp()
    } else {
        freq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_frame_count_matches_whisper_mask_rescale() {
        let waveform = vec![0.0; 240_000];
        let features = extract_log_mel_features(&waveform).unwrap();
        assert_eq!(features.mel_bins, 128);
        assert_eq!(features.frames, 1500);
        assert_eq!(features.attention_mask.len(), 1500);
    }

    #[test]
    fn slaney_mel_filter_shape_is_stable() {
        let filters = mel_filter_bank();
        assert_eq!(filters.len(), 201 * 128);
        assert!((filters[128] - 0.01237399).abs() < 1e-7);
        assert!((filters[129] - 0.03039256).abs() < 1e-7);
    }
}
