//! The [`Vector`] type: an owned, dense, single-precision embedding.
//!
//! `f32` (not `f64`) to halve memory and match what FAISS/hnswlib store; dense
//! (not sparse) because neural embeddings carry signal in every dimension. The
//! dimension is `len()` — not encoded in the type, since it is a per-collection
//! runtime property validated at the insert/query boundary.

use std::ops::Deref;

/// An owned dense `f32` embedding. Thin wrapper over `Vec<f32>`.
///
/// Derefs to `[f32]`, so `&Vector` coerces to `&[f32]` — exactly what the
/// [`crate::Metric`] trait consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector(Vec<f32>);

impl Vector {
    /// Build a `Vector` from its components.
    pub fn new(data: Vec<f32>) -> Self {
        Vector(data)
    }

    /// The dimension of the vector (number of components).
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Borrow the underlying components as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl Deref for Vector {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for Vector {
    fn from(data: Vec<f32>) -> Self {
        Vector(data)
    }
}

impl<const N: usize> From<[f32; N]> for Vector {
    fn from(data: [f32; N]) -> Self {
        Vector(data.to_vec())
    }
}
