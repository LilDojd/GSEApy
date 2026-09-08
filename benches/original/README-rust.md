# Rust library

This fork exposes GSEApy's existing core without Python. Package: `gseapy`;
Rust import: `gse`. Pin a commit from the `rust-lib` branch:

```toml
[dependencies]
gse = { package = "gseapy", git = "https://github.com/LilDojd/GSEApy", rev = "FULL_COMMIT_SHA" }
```

```rust
use gse::GSEAResult;
use std::collections::BTreeMap;

let genes = ["A", "B", "C", "D"].map(String::from);
let scores = [4.0, 2.0, -2.0, -4.0];
let pathway = ["A", "B"].map(String::from);
let sets = BTreeMap::from([("example", pathway.as_slice())]);
// weight, MAX size, MIN size, permutations, seed
let mut result = GSEAResult::new(1.0, 500, 2, 1000, 42);
result.prerank(&genes, &scores, &sets);
assert!((result.summaries[0].es - 1.0).abs() < 1e-12);
```

This is an **unchecked computational API**, not Python's preprocessing pipeline.
Validate requests: at least two unique gene IDs, equally sized finite scores
in rank order, finite nonnegative weight, `1 <= min_size <= max_size`, and no
retained gene set matching every ranked gene. Weighted scores/sums must remain
finite; degenerate inputs/null distributions can yield non-finite statistics.
Bound input sizes and permutations in your service. Use a fresh result per run.
Zero permutations computes ES only, not meaningful NES/p-values/FDR.

Results include ES, NES, p-value, FDR, FWER, hit indices and running curves;
`log2err` is not meaningful for permutation prerank. Results implement Serde.
The core uses Rayon; use `ThreadPool::install` to control CPU usage and keep
blocking analyses off an async executor.

## Development

```sh
devenv shell
cargo test --locked
cargo test --locked --features python
```

Default builds require no Python. `python` enables bindings with interpreter
linking for tests; `extension-module` enables wheel builds (selected by
`setup.py`). Keep using upstream's uv/setuptools workflow for Python packaging.
Commit your application's `Cargo.lock` and use `--locked` in CI.
