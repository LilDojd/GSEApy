#![doc = include_str!("../README-rust.md")]

mod algorithm;
mod fgsea;
mod gsva;
mod scoring;
mod stats;
mod utils;

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "python")]
mod python;

pub use stats::{GSEAResult, GSEASummary};
pub use utils::{CorrelType, Metric};
