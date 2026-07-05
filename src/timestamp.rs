#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairItem {
    pub raw_start_time: i64,
    pub raw_end_time: i64,
    pub start_time: i64,
    pub end_time: i64,
    pub repair_reasons: Vec<String>,
}

/// Repair a flat timestamp sequence via LIS-based anomaly filling.
///
/// This is a line-for-line port of the reference Python implementation
/// (`qwen_asr.Qwen3ForceAlignProcessor.fix_timestamp`,
/// `qwen3_forced_aligner.py:147-234`). The earlier Rust version diverged in
/// two places that changed downstream output:
///
/// 1. it pre-flattened inverted `(start > end)` pairs before LIS — Python
///    does NOT do this, and it reshaped which tokens the LIS treated as
///    anomalies (the main cause of the large multi-second drifts on
///    ko_4m / ja_1m / zh_180s).
/// 2. in the `anomaly_count <= 2` branch it guarded the left/right choice
///    with `i > 0`, whereas Python evaluates `(k - (i - 1))` with signed
///    arithmetic and lets negative values through.
///
/// Both are removed here so output matches the reference bit-for-bit given
/// identical raw input.
pub fn fix_timestamp(data: &[i64]) -> Vec<i64> {
    let n = data.len();
    let mut dp = vec![1usize; n];
    let mut parent = vec![None; n];

    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }

    // `dp.index(max(dp))` — first index achieving the maximum.
    let mut max_idx = 0usize;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut lis_indices = Vec::new();
    let mut idx = Some(max_idx);
    while let Some(i) = idx {
        lis_indices.push(i);
        idx = parent[i];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for idx in lis_indices {
        is_normal[idx] = true;
    }

    let mut result = data.to_vec();
    let mut i = 0usize;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }

        let mut j = i;
        while j < n && !is_normal[j] {
            j += 1;
        }
        let anomaly_count = j - i;

        let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
        let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

        if anomaly_count <= 2 {
            // Python:  result[k] = left if (k - (i - 1)) <= (j - k) else right
            // i may be 0, in which case (k - (i-1)) == k + 1 — compute in i64.
            for k in i..j {
                result[k] = match (left_val, right_val) {
                    (None, Some(r)) => r,
                    (Some(l), None) => l,
                    (Some(l), Some(r)) => {
                        let dist_left = k as i64 - (i as i64 - 1);
                        let dist_right = j as i64 - k as i64;
                        if dist_left <= dist_right { l } else { r }
                    }
                    (None, None) => result[k],
                };
            }
        } else if let (Some(l), Some(r)) = (left_val, right_val) {
            let step = (r - l) as f64 / (anomaly_count + 1) as f64;
            for k in i..j {
                result[k] = (l as f64 + step * (k - i + 1) as f64) as i64;
            }
        } else if let Some(l) = left_val {
            for k in i..j {
                result[k] = l;
            }
        } else if let Some(r) = right_val {
            for k in i..j {
                result[k] = r;
            }
        }

        i = j;
    }

    result
}

pub fn repair_timestamps(data: &[i64]) -> Vec<RepairItem> {
    let fixed = fix_timestamp(data);
    let mut out = Vec::new();

    for i in (0..data.len().saturating_sub(1)).step_by(2) {
        let raw_start = data[i];
        let raw_end = data[i + 1];
        let start = fixed[i];
        let end = fixed[i + 1];
        let mut reasons = Vec::new();

        if raw_start > raw_end {
            reasons.push("inverted_pair_start_clamped_to_end".to_string());
        }
        if raw_start == raw_end {
            reasons.push("raw_zero_duration".to_string());
        }
        if start != raw_start || end != raw_end {
            reasons.push("timestamp_repaired".to_string());
        }
        if start == end {
            reasons.push("zero_duration".to_string());
        }
        if reasons.is_empty() {
            reasons.push("unchanged".to_string());
        }

        out.push(RepairItem {
            raw_start_time: raw_start,
            raw_end_time: raw_end,
            start_time: start,
            end_time: end,
            repair_reasons: reasons,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression input for the LIS-based timestamp repair. Expected output
    /// is computed by the reference Python `Qwen3ForceAlignProcessor.fix_timestamp`
    /// (`qwen3_forced_aligner.py:147`); this test pins the Rust port to it.
    #[test]
    fn fix_timestamp_matches_python_reference() {
        let raw = vec![
            71920, 72240, 72240, 72400, 72400, 72640, 72640, 72880, 72880, 73120, 73120, 73200,
            73840, 74000, 78400, 76960, 78400, 78800,
        ];

        let fixed = fix_timestamp(&raw);
        // Python reference output (no pre-flattening of inverted pairs).
        assert_eq!(
            fixed,
            vec![
                71920, 72240, 72240, 72400, 72400, 72640, 72640, 72880, 72880, 73120, 73120, 73200,
                73840, 74000, 78400, 78400, 78400, 78800,
            ]
        );
        // The inverted pair (78400, 76960) at indices 14-15 is NOT pre-clamped:
        // both raw values reach LIS normally; index 15 is the single anomaly
        // and gets filled to 78400 by the anomaly_count==1 path.
        assert_eq!(fixed[14], 78400);
        assert_eq!(fixed[15], 78400);
        assert_eq!(fixed[16], 78400);
        assert_eq!(fixed[17], 78800);
    }
}
