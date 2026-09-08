//! Score kernels and their input constraints.

/// Increasing, unique indices. Callers must ensure they are in bounds.
#[derive(Clone, Copy)]
pub(crate) struct SortedIndices<'a>(&'a [usize]);

impl<'a> SortedIndices<'a> {
    pub fn new(indices: &'a [usize]) -> Option<Self> {
        indices
            .windows(2)
            .all(|p| p[0] < p[1])
            .then_some(Self(indices))
    }

    pub fn sort_unique(indices: &'a mut Vec<usize>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        Self(indices)
    }

    pub fn as_slice(self) -> &'a [usize] {
        self.0
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RankWeights<'a> {
    values: &'a [f64],
    finite: bool,
}

impl<'a> RankWeights<'a> {
    pub fn new(values: &'a [f64]) -> Self {
        Self {
            values,
            finite: values.iter().all(|x| x.is_finite()),
        }
    }
}

pub(crate) fn gsea_score(weights: RankWeights<'_>, hits: SortedIndices<'_>) -> f64 {
    let metric = weights.values;
    let hits = hits.as_slice();
    let ns: f64 = if weights.finite {
        hits.iter().map(|&i| metric[i]).sum()
    } else {
        // A non-finite value at a miss contributes 0 * NaN/Inf, not zero.
        let mut next = hits.iter().peekable();
        metric
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let tag = if next.peek().is_some_and(|&&hit| hit == i) {
                    next.next();
                    1.0
                } else {
                    0.0
                };
                tag * value
            })
            .sum()
    };
    let q1 = 1.0 / (metric.len() as f64 - hits.len() as f64);
    let q2 = 1.0 / ns;
    let mut res: f64 = 0.0;
    let mut cur: f64 = 0.0;
    let mut last = -1.0;
    for &i in hits {
        let pos = i as f64;
        cur -= q1 * (pos - last - 1.0);
        if cur.abs() > res.abs() {
            res = cur;
        }
        cur += q2 * metric[i];
        if cur.abs() > res.abs() {
            res = cur;
        }
        last = pos;
    }
    res
}

pub(crate) fn ssgsea_score(metric: &[f64], hits: SortedIndices<'_>) -> f64 {
    #[cfg(feature = "simd")]
    if hits.as_slice().len() >= 16 {
        return crate::simd::ssgsea_score(metric, hits);
    }
    ssgsea_score_scalar(metric, hits)
}

pub(crate) fn ssgsea_score_scalar(metric: &[f64], hits: SortedIndices<'_>) -> f64 {
    let hits = hits.as_slice();
    let n = metric.len() as f64;
    let step_cdf_in = hits
        .iter()
        .map(|&i| metric[i] * (n - i as f64))
        .sum::<f64>()
        / hits.iter().map(|&i| metric[i]).sum::<f64>();
    let step_cdf_out = (n * (n + 1.0) / 2.0 - hits.iter().map(|&i| n - i as f64).sum::<f64>())
        / (n - hits.len() as f64);
    step_cdf_in - step_cdf_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_indices_reject_invalid_order_and_duplicates() {
        assert!(SortedIndices::new(&[3, 1]).is_none());
        assert!(SortedIndices::new(&[1, 1]).is_none());
        assert!(SortedIndices::new(&[]).is_some());
        let mut indices = vec![3, 1, 3, 0];
        assert_eq!(
            SortedIndices::sort_unique(&mut indices).as_slice(),
            &[0, 1, 3]
        );
    }

    #[test]
    fn scores_cover_finite_and_nonfinite_weights() {
        let hits = SortedIndices::new(&[0, 3]).unwrap();
        assert_eq!(
            gsea_score(RankWeights::new(&[4.0, 3.0, 2.0, 1.0]), hits),
            0.8
        );
        assert!((ssgsea_score(&[4.0, 3.0, 2.0, 1.0], hits) - 0.9).abs() < 1e-14);
        let hits = SortedIndices::new(&[1, 3]).unwrap();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                gsea_score(RankWeights::new(&[value, 3.0, 2.0, 1.0]), hits),
                -0.5
            );
        }
        assert_eq!(
            gsea_score(RankWeights::new(&[]), SortedIndices::new(&[]).unwrap()),
            0.0
        );
    }
}
