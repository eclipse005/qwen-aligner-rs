#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairItem {
    pub raw_start_time: i64,
    pub raw_end_time: i64,
    pub start_time: i64,
    pub end_time: i64,
    pub repair_reasons: Vec<String>,
}

pub fn fix_timestamp(data: &[i64]) -> Vec<i64> {
    let mut data = data.to_vec();
    let n = data.len();

    for i in (0..n.saturating_sub(1)).step_by(2) {
        if data[i] > data[i + 1] {
            data[i] = data[i + 1];
        }
    }

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

    let mut result = data.clone();
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
            for (k, item) in result.iter_mut().enumerate().take(j).skip(i) {
                *item = match (left_val, right_val) {
                    (None, Some(r)) => r,
                    (Some(l), None) => l,
                    (Some(l), Some(r)) => {
                        if i > 0 && (k - (i - 1)) <= (j - k) {
                            l
                        } else {
                            r
                        }
                    }
                    (None, None) => *item,
                };
            }
        } else if let (Some(l), Some(r)) = (left_val, right_val) {
            let step = (r - l) as f64 / (anomaly_count + 1) as f64;
            for (k, item) in result.iter_mut().enumerate().take(j).skip(i) {
                *item = (l as f64 + step * (k - i + 1) as f64) as i64;
            }
        } else if let Some(l) = left_val {
            for item in result.iter_mut().take(j).skip(i) {
                *item = l;
            }
        } else if let Some(r) = right_val {
            for item in result.iter_mut().take(j).skip(i) {
                *item = r;
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

    #[test]
    fn inverted_token_pair_keeps_raw_end_time() {
        let raw = vec![
            71920, 72240, 72240, 72400, 72400, 72640, 72640, 72880, 72880, 73120, 73120, 73200,
            73840, 74000, 78400, 76960, 78400, 78800,
        ];

        let fixed = fix_timestamp(&raw);
        let debug = repair_timestamps(&raw);

        assert_eq!(fixed[14], 76960);
        assert_eq!(fixed[15], 76960);
        assert_eq!(fixed[16], 78400);
        assert_eq!(fixed[17], 78800);
        assert_eq!(debug[7].raw_start_time, 78400);
        assert_eq!(debug[7].raw_end_time, 76960);
        assert_eq!(debug[7].start_time, 76960);
        assert_eq!(debug[7].end_time, 76960);
        assert!(debug[7]
            .repair_reasons
            .contains(&"inverted_pair_start_clamped_to_end".to_string()));
    }
}
