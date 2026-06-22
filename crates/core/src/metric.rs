//! Distance metrics: L2 (Euclidean) and cosine. Both implement [`Metric`], so
//! the index is generic over the notion of "closeness" it is built around.

/// A way to measure distance between two vectors. Smaller means more similar.
pub trait Metric {
    /// Distance between `a` and `b`.
    ///
    /// Precondition: `a.len() == b.len()`. Dimension agreement is validated once
    /// at the insert/query boundary, not on this hot path; debug builds assert it.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Euclidean (L2) distance: `sqrt(Σ (aᵢ − bᵢ)²)`.
pub struct L2;

impl Metric for L2 {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "L2: dimension mismatch");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

/// Cosine distance: `1 − (a·b) / (‖a‖·‖b‖)`.
///
/// Identical direction → 0, orthogonal → 1, opposite → 2. A zero-norm input has
/// no direction and is treated as maximally dissimilar (distance 1), never NaN.
pub struct Cosine;

impl Metric for Cosine {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Cosine: dimension mismatch");
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (&x, &y) in a.iter().zip(b) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // equal within float tolerance
    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    #[test]
    fn l2_classic_3_4_5_triangle() {
        // (0,0) to (3,4): the 3-4-5 right triangle. Distance is exactly 5.
        approx(L2.distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn l2_to_self_is_zero() {
        approx(L2.distance(&[1.5, -2.0, 7.0], &[1.5, -2.0, 7.0]), 0.0);
    }

    #[test]
    fn l2_is_symmetric() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, -1.0];
        approx(L2.distance(&a, &b), L2.distance(&b, &a));
    }

    #[test]
    fn cosine_identical_direction_is_zero() {
        // Same direction (even at different magnitudes) → distance 0.
        approx(Cosine.distance(&[1.0, 0.0], &[1.0, 0.0]), 0.0);
        approx(Cosine.distance(&[1.0, 1.0], &[5.0, 5.0]), 0.0);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        approx(Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
    }

    #[test]
    fn cosine_opposite_is_two() {
        approx(Cosine.distance(&[1.0, 0.0], &[-1.0, 0.0]), 2.0);
    }

    #[test]
    fn cosine_zero_vector_is_defined() {
        // No NaN/inf: the zero vector is treated as maximally dissimilar.
        let d = Cosine.distance(&[0.0, 0.0], &[1.0, 2.0]);
        assert!(d.is_finite(), "zero vector produced a non-finite distance");
        approx(d, 1.0);
    }
}
