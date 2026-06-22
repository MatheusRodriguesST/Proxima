//! Proxima core — base types for the vector engine: the [`Vector`] type and the
//! [`Metric`] trait with two implementations ([`L2`], [`Cosine`]).

mod metric;
mod vector;

pub use metric::{Cosine, Metric, L2};
pub use vector::Vector;
