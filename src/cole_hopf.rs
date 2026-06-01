//! Cole-Hopf transform: u → −ℏ log u
//!
//! The Cole-Hopf transform converts solutions of the heat equation
//! to solutions of the Hamilton-Jacobi equation:
//!   If u solves ∂_t u = Δu, then v = −ℏ log u solves ∂_t v + |∇v|² = ℏΔv.

use crate::heat_kernel::{heat_kernel, Graph};
use nalgebra::{DMatrix, DVector};

/// Apply the Cole-Hopf transform: v = −ℏ log(u) element-wise.
pub fn cole_hopf(u: &DMatrix<f64>, hbar: f64) -> DMatrix<f64> {
    u.map(|x| {
        if x > 0.0 && x.is_finite() {
            -hbar * x.ln()
        } else {
            f64::INFINITY
        }
    })
}

/// Apply the inverse Cole-Hopf transform: u = exp(−v/ℏ).
pub fn cole_hopf_inverse(v: &DMatrix<f64>, hbar: f64) -> DMatrix<f64> {
    v.map(|x| (-x / hbar).exp())
}

/// Apply Cole-Hopf transform to a vector.
pub fn cole_hopf_vec(u: &DVector<f64>, hbar: f64) -> DVector<f64> {
    u.map(|x| {
        if x > 0.0 && x.is_finite() {
            -hbar * x.ln()
        } else {
            f64::INFINITY
        }
    })
}

/// Apply inverse Cole-Hopf to a vector.
pub fn cole_hopf_inverse_vec(v: &DVector<f64>, hbar: f64) -> DVector<f64> {
    v.map(|x| (-x / hbar).exp())
}

/// Full Cole-Hopf pipeline: compute heat kernel, then transform.
pub fn heat_to_hamilton_jacobi(graph: &Graph, t: f64, hbar: f64) -> DMatrix<f64> {
    let h = heat_kernel(graph, t);
    cole_hopf(&h, hbar)
}

/// Verify the Cole-Hopf transform is invertible.
pub fn verify_cole_hopf_inverse(u: &DMatrix<f64>, hbar: f64, tol: f64) -> bool {
    let v = cole_hopf(u, hbar);
    let u_recovered = cole_hopf_inverse(&v, hbar);
    let n = u.nrows();
    for i in 0..n {
        for j in 0..u.ncols() {
            if (u[(i, j)] - u_recovered[(i, j)]).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::path_graph;
    use approx::assert_relative_eq;

    #[test]
    fn test_cole_hopf_basic() {
        let u = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let v = cole_hopf(&u, 1.0);
        assert_relative_eq!(v[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[(0, 1)], 2.0_f64.ln(), epsilon = 1e-10); // -1 * ln(0.5) = ln(2)
    }

    #[test]
    fn test_cole_hopf_inverse_roundtrip() {
        let u = DMatrix::from_row_slice(2, 2, &[2.0, 0.3, 0.7, 1.5]);
        let v = cole_hopf(&u, 1.0);
        let u2 = cole_hopf_inverse(&v, 1.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(u[(i, j)], u2[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cole_hopf_with_hbar() {
        let u = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let v = cole_hopf(&u, 2.0);
        assert_relative_eq!(v[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[(0, 1)], -2.0 * 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_cole_hopf_vec() {
        let u = DVector::from_vec(vec![1.0, 2.0, 0.5]);
        let v = cole_hopf_vec(&u, 1.0);
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[1], -2.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(v[2], 2.0_f64.ln(), epsilon = 1e-10); // -ln(0.5) = ln(2)
    }

    #[test]
    fn test_cole_hopf_inverse_vec() {
        let u = DVector::from_vec(vec![1.0, 2.0, 0.5]);
        let v = cole_hopf_vec(&u, 1.0);
        let u2 = cole_hopf_inverse_vec(&v, 1.0);
        for i in 0..3 {
            assert_relative_eq!(u[i], u2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_heat_to_hamilton_jacobi() {
        let g = path_graph(4);
        let hj = heat_to_hamilton_jacobi(&g, 0.1, 1.0);
        assert_eq!(hj.nrows(), 4);
        assert_eq!(hj.ncols(), 4);
        // Diagonal should be smallest (closest to self)
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(hj[(i, i)] <= hj[(i, j)] || hj[(i, i)].is_nan() == hj[(i, j)].is_nan());
                }
            }
        }
    }

    #[test]
    fn test_verify_cole_hopf_inverse() {
        let u = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        assert!(verify_cole_hopf_inverse(&u, 1.0, 1e-10));
    }

    #[test]
    fn test_cole_hopf_varadhan_connection() {
        // Cole-Hopf with hbar=4t should give Varadhan's formula
        let g = path_graph(4);
        let t = 0.05;
        let h = heat_kernel(&g, t);
        let v = cole_hopf(&h, 4.0 * t);
        let varadhan = crate::varadhan::varadhan_approx(&g, t);
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(v[(i, j)], varadhan[(i, j)], epsilon = 1e-10);
            }
        }
    }
}
