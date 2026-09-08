//! SIMD reductions change floating-point addition order.

use crate::scoring::SortedIndices;
use wide::f64x4;

/// Two-pass variance, like `Statistic::stat`; do not use E[x²] - E[x]².
pub(crate) fn mean_std(values: &[f64], ddof: usize) -> (f64, f64) {
    let mut sum = f64x4::ZERO;
    let (chunks, tail) = values.as_chunks::<4>();
    for &chunk in chunks {
        sum += f64x4::new(chunk);
    }
    let sum = tail.iter().fold(sum.reduce_add(), |s, x| s + x);
    let mean = sum / values.len() as f64;
    let center = f64x4::splat(mean);
    let mut squared = f64x4::ZERO;
    for &chunk in chunks {
        let diff = center - f64x4::new(chunk);
        squared += diff * diff;
    }
    let squared = tail.iter().fold(squared.reduce_add(), |s, x| {
        let diff = mean - x;
        s + diff * diff
    });
    (mean, (squared / (values.len() - ddof) as f64).sqrt())
}

/// Accumulate ssGSEA's three sums in one pass.
pub(crate) fn ssgsea_score(metric: &[f64], hits: SortedIndices<'_>) -> f64 {
    let hits = hits.as_slice();
    let n = metric.len() as f64;
    let mut weights = f64x4::ZERO;
    let mut distances = f64x4::ZERO;
    let mut weighted = f64x4::ZERO;
    let (chunks, tail) = hits.as_chunks::<4>();
    for &chunk in chunks {
        let w = f64x4::new(chunk.map(|i| metric[i]));
        let d = f64x4::splat(n) - f64x4::new(chunk.map(|i| i as f64));
        weights += w;
        distances += d;
        weighted += w * d;
    }
    let (mut weights, mut distances, mut weighted) = (
        weights.reduce_add(),
        distances.reduce_add(),
        weighted.reduce_add(),
    );
    for &i in tail {
        let d = n - i as f64;
        weights += metric[i];
        distances += d;
        weighted += metric[i] * d;
    }
    weighted / weights - (n * (n + 1.0) / 2.0 - distances) / (n - hits.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::ssgsea_score_scalar;

    fn assert_close(a: f64, b: f64, scale: f64) {
        assert!(
            a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * scale,
            "{a} != {b} (scale {scale})"
        );
    }

    #[test]
    fn mean_std_matches_two_pass_reference_including_tails() {
        for len in (0..70).chain([127, 128, 129, 1024]) {
            for offset in [0.0, 1e6] {
                let values: Vec<f64> = (0..len).map(|i| offset + (i as f64 * 1.7).sin()).collect();
                for ddof in 0..=len.min(1) {
                    let mean = values.iter().sum::<f64>() / len as f64;
                    let std = (values.iter().map(|x| (mean - x) * (mean - x)).sum::<f64>()
                        / (len - ddof) as f64)
                        .sqrt();
                    let (a, b) = mean_std(&values, ddof);
                    assert_close(a, mean, mean.abs().max(1.0));
                    assert_close(b, std, std.abs().max(1.0));
                }
            }
        }
        for value in [0.0, -0.0, 3.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (mean, std) = mean_std(&[value; 19], 1);
            assert_close(mean, value, 1.0);
            assert_close(std, if value.is_finite() { 0.0 } else { f64::NAN }, 1.0);
        }
    }

    #[test]
    fn ssgsea_matches_scalar_reference_including_tails() {
        let mut metric: Vec<f64> = (0..137).map(|i| (i as f64 + 0.5).powf(0.25)).collect();
        for len in 0..=metric.len() {
            let hits: Vec<usize> = (0..len).collect();
            let hits = SortedIndices::new(&hits).unwrap();
            assert_close(
                ssgsea_score(&metric, hits),
                ssgsea_score_scalar(&metric, hits),
                metric.len() as f64,
            );
        }
        let hits: Vec<usize> = (0..metric.len()).step_by(3).collect();
        let hits = SortedIndices::new(&hits).unwrap();
        for value in [0.0, 1.0, f64::NAN, f64::INFINITY] {
            metric.fill(value);
            assert_close(
                ssgsea_score(&metric, hits),
                ssgsea_score_scalar(&metric, hits),
                metric.len() as f64,
            );
        }
    }
}
