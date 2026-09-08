use criterion::{measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId, Criterion};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

fn measure<T>(
    group: Option<&mut BenchmarkGroup<'_, WallTime>>,
    name: &str,
    config: &str,
    repeats: usize,
    run: impl Fn() -> T,
) {
    if let Some(group) = group {
        group.bench_function(BenchmarkId::new(name, config), |b| {
            b.iter_batched(|| (), |_| run(), BatchSize::PerIteration);
        });
        return;
    }
    let mut previous = black_box(run());
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let next = black_box(run());
        times.push(start.elapsed().as_secs_f64());
        previous = next;
    }
    println!("{name}: seconds={times:?}");
    times.sort_by(f64::total_cmp);
    println!(
        "{name}: median={:.6}s",
        (times[(repeats - 1) / 2] + times[repeats / 2]) / 2.0
    );
    black_box(previous);
}

fn check_floats(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        let close = cfg!(feature = "simd") && (x - y).abs() <= 1e-9 + 1e-10 * x.abs().max(y.abs());
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()) || close,
            "{x} != {y}"
        );
    }
}

fn check_results(a: &gse::GSEAResult, b: &gse_original::GSEAResult) {
    assert_eq!(
        (a.weight, a.min_size, a.max_size, a.nperm, a.seed),
        (b.weight, b.min_size, b.max_size, b.nperm, b.seed)
    );
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.rankings.len(), b.rankings.len());
    for (x, y) in a.rankings.iter().zip(&b.rankings) {
        check_floats(x, y);
    }
    assert_eq!(a.summaries.len(), b.summaries.len());
    for (x, y) in a.summaries.iter().zip(&b.summaries) {
        assert_eq!((&x.term, &x.hits, x.index), (&y.term, &y.hits, y.index));
        check_floats(
            &[x.es, x.nes, x.pval, x.fwerp, x.fdr, x.log2err],
            &[y.es, y.nes, y.pval, y.fwerp, y.fdr, y.log2err],
        );
        check_floats(&x.run_es, &y.run_es);
        check_floats(&x.esnull, &y.esnull);
    }
}

fn main() {
    let profile = std::env::args().any(|s| s == "--profile");
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|s| s != "--bench" && s != "--profile")
        .collect();
    let backend = args.first().map(String::as_str).unwrap_or("compare");
    let workflow = args.get(1).map(String::as_str).unwrap_or("prerank");
    let number = |i: usize, default| {
        args.get(i)
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let (n, sets, permutations, samples, threads, repeats) = (
        number(2, 5000),
        number(3, 100),
        number(4, 256),
        number(5, 12),
        number(6, 1),
        number(7, 5),
    );
    assert!(n >= 10 && samples >= 4 && repeats > 0);
    let mut rng = SmallRng::seed_from_u64(2026);
    let genes: Vec<String> = (0..n).map(|i| format!("G{i:05}")).collect();
    let mut scores: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    scores.sort_by(|a, b| b.total_cmp(a));
    let expression: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..samples).map(|_| rng.random_range(0.1..10.0)).collect())
        .collect();
    let group: Vec<bool> = (0..samples).map(|i| i < samples / 2).collect();
    let mut order: Vec<usize> = (0..n).collect();
    let pathways: BTreeMap<String, Vec<String>> = (0..sets)
        .map(|i| {
            order.shuffle(&mut rng);
            (
                format!("set_{i:04}"),
                order[..(15 + i % 186).min(n / 2)]
                    .iter()
                    .map(|&j| genes[j].clone())
                    .collect(),
            )
        })
        .collect();
    let gmt = pathways
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let weight = if workflow.starts_with("ssgsea") {
        0.25
    } else {
        1.0
    };

    // The two crates have distinct result and enum types but the same API.
    macro_rules! run {
        ($core:ident) => {{
            let mut result = $core::GSEAResult::new(weight, 500, 5, permutations, 42);
            match workflow {
                "prerank" => result.prerank(&genes, &scores, &gmt),
                "multilevel" => result.prerank_multilevel(&genes, &scores, &gmt, 101, 1e-10),
                "gsea" => result.gsea(
                    &genes,
                    &group,
                    &expression,
                    &gmt,
                    $core::Metric::Signal2Noise,
                ),
                "ssgsea" => result.ss_gsea(&genes, &expression, &gmt, $core::CorrelType::Rank),
                "ssgsea-permutation" => {
                    result.ss_gsea_permuate(&genes, &expression, &gmt, $core::CorrelType::Rank)
                }
                _ => panic!("unknown workflow: {workflow}"),
            }
            result
        }};
    }
    let original = || pool.install(|| run!(gse_original));
    let optimized = || pool.install(|| run!(gse));
    println!("{workflow}: genes={n} sets={sets} permutations={permutations} samples={samples} threads={threads}");
    let config =
        format!("{n}genes-{sets}sets-{permutations}perms-{samples}samples-{threads}threads");
    let feature = if cfg!(feature = "simd") {
        "simd"
    } else {
        "scalar"
    };
    let mut criterion = (!profile).then(|| Criterion::default().sample_size(repeats.max(10)));
    let mut group = criterion
        .as_mut()
        .map(|c| c.benchmark_group(format!("{workflow}-{feature}")));
    match backend {
        "compare" => {
            check_results(&optimized(), &original());
            println!(
                "outputs match (SIMD tolerances: {})",
                cfg!(feature = "simd")
            );
            measure(group.as_mut(), "original", &config, repeats, original);
            measure(group.as_mut(), "optimized", &config, repeats, optimized);
        }
        "original" => measure(group.as_mut(), "original", &config, repeats, original),
        "optimized" => measure(group.as_mut(), "optimized", &config, repeats, optimized),
        _ => panic!("backend must be compare, original, or optimized"),
    }
    drop(group);
    if let Some(criterion) = criterion {
        criterion.final_summary();
    }
}
