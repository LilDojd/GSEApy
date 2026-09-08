use gse::GSEAResult;
use std::collections::BTreeMap;

#[test]
fn prerank_matches_upstream_python() {
    // Reference: unmodified prerank_rs at upstream
    // 6e6f0e29ce3b407a7fb19bc6a9a73ee0015263fa (seed 42, 64 permutations).
    let genes: Vec<String> = (0..10).map(|i| format!("g{i}")).collect();
    let scores = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, -1.5, -3.0, -5.0];
    let top = ["g0", "g1", "g2"].map(String::from);
    let bottom = ["g7", "g8", "g9"].map(String::from);
    let mixed = ["g0", "g4", "g8"].map(String::from);
    let sets = BTreeMap::from([
        ("top", top.as_slice()),
        ("bottom", bottom.as_slice()),
        ("mixed", mixed.as_slice()),
        ("too_small", &top[..1]),
    ]);
    let expected = [
        (
            "bottom",
            [
                -1.0,
                -1.7581395348837208,
                0.07407407407407407,
                0.028169014084507043,
                0.0,
            ],
            [7, 8, 9],
        ),
        (
            "mixed",
            [
                0.5882352941176471,
                0.9557469901062483,
                0.47619047619047616,
                0.5289256198347108,
                0.78125,
            ],
            [0, 4, 8],
        ),
        (
            "top",
            [
                1.0,
                1.5921306139727538,
                0.023809523809523808,
                0.03305785123966942,
                0.03125,
            ],
            [0, 1, 2],
        ),
    ];
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut result = GSEAResult::new(1.0, 5, 2, 64, 42);
        pool.install(|| result.prerank(&genes, &scores, &sets));
        assert_eq!(result.rankings, vec![scores.to_vec()]);
        assert_eq!(result.indices, vec![(0..10).collect::<Vec<_>>()]);
        assert_eq!(result.summaries.len(), expected.len());
        for (actual, (term, values, hits)) in result.summaries.iter().zip(&expected) {
            assert_eq!(&actual.term, term);
            assert_eq!(&actual.hits, hits);
            assert_eq!(actual.esnull.len(), 64);
            assert_eq!(actual.run_es.len(), genes.len());
            for (a, b) in [actual.es, actual.nes, actual.pval, actual.fdr, actual.fwerp]
                .iter()
                .zip(values)
            {
                assert!((a - b).abs() < 1e-12, "{term}: {a} != {b}");
            }
        }
    }
}
