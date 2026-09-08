#![doc = include_str!("../README-rust.md")]

mod algorithm;
mod fgsea;
mod gsva;
mod stats;
mod utils;

#[cfg(feature = "python")]
mod python;

pub use stats::{GSEAResult, GSEASummary};
pub use utils::{CorrelType, Metric};
