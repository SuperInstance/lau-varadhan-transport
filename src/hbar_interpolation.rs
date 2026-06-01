//! ℏ-interpolation between linear spectral and piecewise-linear tropical.
//!
//! This module provides the continuous interpolation controlled by ℏ:
//! - ℏ = 1: standard linear spectral world (heat equation, Fourier analysis)
//! - ℏ → 0: tropical world (Hamilton-Jacobi, min-plus algebra)
//! - Intermediate ℏ: smoothed transport problems (entropic regularization)
//!
//! The key insight: optimal transport with entropic regularization at parameter ℏ
//! interpolates between the spectral (ℏ=1) and tropical (ℏ→0) regimes.

use crate::heat_kernel::HeatKernel;
use crate::maslov::DeformedSemiring;
use crate::cole_hopf::{cole_hopf_transform, cole_hopf_inverse};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// ℏ-interpolation state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HbarInterpolation {
    pub hbar: f64,
}

impl HbarInterpolation {
    pub fn new(hbar: f64) -> Self {
        Self { hbar }
    }

    /// Interpolated addition: transitions from standard (+) to tropical (min) as ℏ→0.
    /// ⊕_ℏ(a,b) = ℏ log(exp(a/ℏ) + exp(b/ℏ))
    pub fn interp_add(&self, a: f64, b: f64) -> f64 {
        let sr = DeformedSemiring::new(self.hbar);
        sr.log_sum_exp(a, b)
    }

    /// Interpolated kernel: exp(-d²/(ℏ)) — the Gibbs kernel.
    pub fn gibbs_kernel(&self, distance: f64) -> f64 {
        (-distance / self.hbar).exp()
    }

    /// Interpolated kernel matrix.
    pub fn gibbs_kernel_matrix(&self, distances: &DMatrix<f64>) -> DMatrix<f64> {
        distances.map(|d| self.gibbs_kernel(d))
    }

    /// The ℏ-regularized optimal transport cost.
    /// Uses the Sinkhorn algorithm with parameter ℏ.
    pub fn regularized_ot_cost(
        &self,
        mu: &DVector<f64>,
        nu: &DVector<f64>,
        cost_matrix: &DMatrix<f64>,
        n_iter: usize,
    ) -> f64 {
        let n = mu.len();
        let epsilon = self.hbar;

        // Kernel K = exp(-C/ε)
        let k = cost_matrix.map(|c| (-c / epsilon).exp());

        let mut a = DVector::from_element(n, 1.0);
        let mut b = DVector::from_element(n, 1.0);

        for _ in 0..n_iter {
            let kb = &k * &b;
            for i in 0..n {
                if kb[i] > 1e-30 {
                    a[i] = mu[i] / kb[i];
                }
            }
            let kta = &k.transpose() * &a;
            for j in 0..n {
                if kta[j] > 1e-30 {
                    b[j] = nu[j] / kta[j];
                }
            }
        }

        // Dual potentials
        let f = epsilon * a.map(|x| if x > 0.0 { x.ln() } else { -50.0 });
        let g = epsilon * b.map(|x| if x > 0.0 { x.ln() } else { -50.0 });

        // Transport plan
        let mut cost = 0.0;
        for i in 0..n {
            for j in 0..n {
                let pi_ij = a[i] * k[(i, j)] * b[j];
                cost += cost_matrix[(i, j)] * pi_ij;
            }
        }
        cost
    }

    /// Compute the ℏ-interpolation path for the heat → HJ transition.
    pub fn spectral_to_tropical_path(
        &self,
        hk: &HeatKernel,
        initial: &DVector<f64>,
        t: f64,
        hbar_values: &[f64],
    ) -> Vec<DVector<f64>> {
        hbar_values.iter().map(|&h| {
            let u0 = cole_hopf_inverse(initial, h);
            let pt = hk.matrix(t);
            let u_t = &pt * &u0;
            cole_hopf_transform(&u_t, h)
        }).collect()
    }
}

/// Compute the ℏ-regularized Wasserstein distance using dual formulation.
pub fn hbar_wasserstein(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    distances: &DMatrix<f64>,
    hbar: f64,
    n_iter: usize,
) -> f64 {
    let interp = HbarInterpolation::new(hbar);
    interp.regularized_ot_cost(mu, nu, &distances.map(|d| d * d), n_iter).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::*;
    use approx::*;

    #[test]
    fn test_interp_add_large_hbar() {
        let interp = HbarInterpolation::new(100.0);
        let result = interp.interp_add(3.0, 7.0);
        // With large ℏ, should be close to (3+7)/2 = 5 (average-like)
        assert!(result > 3.0 && result < 7.0);
    }

    #[test]
    fn test_interp_add_small_hbar() {
        let interp = HbarInterpolation::new(0.001);
        let result = interp.interp_add(3.0, 7.0);
        // With small ℏ, should converge to min(3,7) = 3
        assert_relative_eq!(result, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_gibbs_kernel() {
        let interp = HbarInterpolation::new(1.0);
        let k0 = interp.gibbs_kernel(0.0);
        assert_relative_eq!(k0, 1.0);
        let k_inf = interp.gibbs_kernel(100.0);
        assert!(k_inf < 0.01);
    }

    #[test]
    fn test_gibbs_kernel_matrix() {
        let dist = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let interp = HbarInterpolation::new(1.0);
        let k = interp.gibbs_kernel_matrix(&dist);
        assert_relative_eq!(k[(0, 0)], 1.0);
        assert_relative_eq!(k[(0, 1)], (-1.0_f64).exp());
    }

    #[test]
    fn test_regularized_ot_self() {
        let mu = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let cost = DMatrix::from_row_slice(4, 4, &[
            0.0, 1.0, 4.0, 9.0,
            1.0, 0.0, 1.0, 4.0,
            4.0, 1.0, 0.0, 1.0,
            9.0, 4.0, 1.0, 0.0,
        ]);
        let interp = HbarInterpolation::new(0.1);
        let ot_cost = interp.regularized_ot_cost(&mu, &mu, &cost, 50);
        // Transporting to self should have small cost
        assert!(ot_cost < 2.0);
    }

    #[test]
    fn test_hbar_wasserstein_positive() {
        let mu = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        let dist = DMatrix::from_row_slice(4, 4, &[
            0.0, 1.0, 2.0, 3.0,
            1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0,
            3.0, 2.0, 1.0, 0.0,
        ]);
        // Normalize
        let mu_n = &mu / mu.sum();
        let nu_n = &nu / nu.sum();
        let w = hbar_wasserstein(&mu_n, &nu_n, &dist, 0.1, 50);
        assert!(w > 0.0);
    }

    #[test]
    fn test_spectral_to_tropical_path() {
        let g = path_graph(4);
        let hk = HeatKernel::from_graph(&g);
        let interp = HbarInterpolation::new(1.0);
        let initial = DVector::from_vec(vec![0.0, 1.0, 2.0, 1.0]);
        let path = interp.spectral_to_tropical_path(&hk, &initial, 0.1, &[1.0, 0.1, 0.01]);
        assert_eq!(path.len(), 3);
        for p in &path {
            assert_eq!(p.len(), 4);
        }
    }

    #[test]
    fn test_hbar_interpolation_monotonicity() {
        // As hbar decreases, OT cost should increase (less entropic blurring → sharper transport)
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let mu_n = &mu / mu.sum();
        let nu_n = &nu / nu.sum();
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);

        let interp_large = HbarInterpolation::new(1.0);
        let interp_small = HbarInterpolation::new(0.01);
        let cost_large = interp_large.regularized_ot_cost(&mu_n, &nu_n, &cost, 100);
        let cost_small = interp_small.regularized_ot_cost(&mu_n, &nu_n, &cost, 100);
        assert!(cost_small >= cost_large - 0.1);
    }
}
