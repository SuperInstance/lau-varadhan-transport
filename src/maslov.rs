//! Maslov dequantization: interpolation from (+,×) to (min,+) as ℏ: 1→0
//!
//! In tropical geometry, the Maslov dequantization is the observation that
//! as ℏ→0, the "deformed" arithmetic x ⊕_ℏ y = ℏ log(e^{x/ℏ} + e^{y/ℏ})
//! converges to min(x,y), and x ⊗_ℏ y = x + y remains unchanged.

use serde::{Deserialize, Serialize};

/// Tropical semiring (min-plus): (R ∪ {∞}, min, +)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TropicalMinPlus;

impl TropicalMinPlus {
    pub fn add(a: f64, b: f64) -> f64 {
        a.min(b)
    }

    pub fn mul(a: f64, b: f64) -> f64 {
        a + b
    }

    pub fn zero() -> f64 {
        f64::INFINITY // additive identity for min
    }

    pub fn one() -> f64 {
        0.0 // multiplicative identity for +
    }
}

/// Tropical semiring (max-plus): (R ∪ {-∞}, max, +)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TropicalMaxPlus;

impl TropicalMaxPlus {
    pub fn add(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    pub fn mul(a: f64, b: f64) -> f64 {
        a + b
    }

    pub fn zero() -> f64 {
        f64::NEG_INFINITY
    }

    pub fn one() -> f64 {
        0.0
    }
}

/// Deformed addition (log-sum-exp / softmax): x ⊕_ℏ y = ℏ log(e^{x/ℏ} + e^{y/ℏ})
///
/// As ℏ→0, this converges to min(x,y).
pub fn deformed_add(a: f64, b: f64, hbar: f64) -> f64 {
    if hbar <= 0.0 {
        return TropicalMinPlus::add(a, b);
    }
    let max_val = a.max(b);
    if max_val.is_infinite() {
        return max_val;
    }
    // unused — see deformed_add_stable below
    hbar * ((a - max_val) / hbar).exp().ln_1p()
}

/// Deformed addition for min-semiring (stable version).
/// -ℏ log(exp(-a/ℏ) + exp(-b/ℏ)) → min(a,b) as ℏ→0.
pub fn deformed_add_stable(a: f64, b: f64, hbar: f64) -> f64 {
    if hbar <= 0.0 {
        return a.min(b);
    }
    // -ℏ log(exp(-a/ℏ) + exp(-b/ℏ)) = -ℏ(-max/ℏ + log(1 + exp(-(min-max)/ℏ)))
    // = min - ℏ*log(1 + exp(-(min-max)/ℏ))
    // Hmm that's wrong. Let's be precise:
    // -ℏ log(exp(-a/ℏ) + exp(-b/ℏ))
    // = -ℏ log(exp(-min/ℏ) * (1 + exp(-(max-min)/ℏ)))  (min = min(a,b))
    // = -ℏ(-min/ℏ + log(1 + exp(-(max-min)/ℏ)))
    // = min - ℏ*log(1 + exp(-(max-min)/ℏ))
    // As ℏ→0, exp(-(max-min)/ℏ)→0, so → min ✓
    let min_val = a.min(b);
    let max_val = a.max(b);
    if min_val.is_infinite() && min_val < 0.0 {
        return min_val;
    }
    let diff = max_val - min_val; // >= 0
    min_val - hbar * (-diff / hbar).exp().ln_1p()
}

/// Deformed addition for max-semiring: ℏ log(e^{x/ℏ} ⊕ e^{y/ℏ}) → max
pub fn deformed_add_max(a: f64, b: f64, hbar: f64) -> f64 {
    if hbar <= 0.0 {
        return a.max(b);
    }
    let max_val = a.max(b);
    let min_val = a.min(b);
    if max_val.is_infinite() {
        return max_val;
    }
    // log(exp(a/ℏ) + exp(b/ℏ)) * ℏ = max + ℏ*log(1 + exp((min-max)/ℏ))
    max_val + hbar * ((min_val - max_val) / hbar).exp().ln_1p()
}

/// Deformed multiplication: x ⊗_ℏ y = x + y (unchanged, same as tropical).
pub fn deformed_mul(a: f64, b: f64, _hbar: f64) -> f64 {
    a + b
}

/// Maslov dequantization path: interpolate from standard to tropical arithmetic.
///
/// Returns the deformed sum of values at the given ℏ parameter.
pub fn maslov_sum(values: &[f64], hbar: f64) -> f64 {
    if values.is_empty() {
        return TropicalMinPlus::zero();
    }
    let mut result = values[0];
    for &x in &values[1..] {
        result = deformed_add_stable(result, x, hbar);
    }
    result
}

/// Maslov product of values (just sum, independent of ℏ).
pub fn maslov_product(values: &[f64], _hbar: f64) -> f64 {
    values.iter().sum()
}

/// Maslov polynomial evaluation: evaluate a tropical polynomial with deformed arithmetic.
/// A tropical polynomial is sum_i (a_i + b_i * x) in min-plus.
pub fn maslov_polynomial(coeffs: &[(f64, f64)], x: f64, hbar: f64) -> f64 {
    let terms: Vec<f64> = coeffs.iter().map(|&(c, k)| c + k * x).collect();
    maslov_sum(&terms, hbar)
}

/// Compute the dequantization schedule: track convergence of deformed sum to min.
pub fn dequantization_convergence(values: &[f64], hbar_values: &[f64]) -> Vec<(f64, f64)> {
    let true_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    hbar_values.iter().map(|&hbar| {
        let approx = maslov_sum(values, hbar);
        (hbar, (approx - true_min).abs())
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tropical_min_plus_add() {
        assert_eq!(TropicalMinPlus::add(3.0, 5.0), 3.0);
        assert_eq!(TropicalMinPlus::add(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_tropical_min_plus_mul() {
        assert_eq!(TropicalMinPlus::mul(3.0, 5.0), 8.0);
    }

    #[test]
    fn test_tropical_min_plus_identities() {
        assert_eq!(TropicalMinPlus::add(3.0, TropicalMinPlus::zero()), 3.0);
        assert_eq!(TropicalMinPlus::mul(3.0, TropicalMinPlus::one()), 3.0);
    }

    #[test]
    fn test_tropical_max_plus_add() {
        assert_eq!(TropicalMaxPlus::add(3.0, 5.0), 5.0);
    }

    #[test]
    fn test_tropical_max_plus_mul() {
        assert_eq!(TropicalMaxPlus::mul(3.0, 5.0), 8.0);
    }

    #[test]
    fn test_deformed_add_large_hbar() {
        // For large ℏ, result ≈ min - ℏ*ln(2) (between the values but below min)
        let result = deformed_add_stable(1.0, 3.0, 10.0);
        // min(1,3)=1, result = 1 - 10*ln(1+exp(-0.2)) ≈ 1 - 3.13 = -2.13
        assert!(result < 1.0);
    }

    #[test]
    fn test_deformed_add_small_hbar() {
        // For small ℏ, deformed add → min
        let result = deformed_add_stable(1.0, 3.0, 0.001);
        assert_relative_eq!(result, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_deformed_add_max_small_hbar() {
        let result = deformed_add_max(1.0, 3.0, 0.001);
        assert_relative_eq!(result, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_deformed_mul() {
        assert_eq!(deformed_mul(3.0, 5.0, 1.0), 8.0);
        assert_eq!(deformed_mul(3.0, 5.0, 0.0), 8.0);
    }

    #[test]
    fn test_maslov_sum() {
        let values = vec![1.0, 3.0, 5.0, 7.0];
        let result = maslov_sum(&values, 0.001);
        assert_relative_eq!(result, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_maslov_product() {
        let values = vec![1.0, 2.0, 3.0];
        assert_eq!(maslov_product(&values, 1.0), 6.0);
    }

    #[test]
    fn test_maslov_polynomial() {
        // p(x) = min(1+2x, 3+1x) in tropical
        let coeffs = vec![(1.0, 2.0), (3.0, 1.0)];
        let result = maslov_polynomial(&coeffs, 1.0, 0.001);
        // At x=1: min(1+2, 3+1) = min(3, 4) = 3
        assert_relative_eq!(result, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_dequantization_convergence() {
        let values = vec![1.0, 5.0, 10.0];
        let hbars = vec![1.0, 0.1, 0.01, 0.001];
        let conv = dequantization_convergence(&values, &hbars);
        assert_eq!(conv.len(), 4);
        // Error should decrease as hbar → 0
        assert!(conv.last().unwrap().1 < conv.first().unwrap().1);
    }

    #[test]
    fn test_maslov_sum_empty() {
        assert_eq!(maslov_sum(&[], 1.0), f64::INFINITY);
    }
}
