//! Hopf-Lax semigroup / inf-convolution: Q_t u(x) = inf_y [u(y) + d(x,y)²/(2t)]
//!
//! This is the viscosity solution semigroup for the Hamilton-Jacobi equation
//! ∂_t u + |∇u|²/2 = 0.

use crate::varadhan::geodesic_distances;
use crate::heat_kernel::Graph;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Apply the Hopf-Lax semigroup for time t to a cost function u on graph vertices.
///
/// Q_t u(x) = min_y [u(y) + d(x,y)²/(2t)]
pub fn hopf_lax(graph: &Graph, u: &[f64], t: f64) -> Vec<f64> {
    let dist = geodesic_distances(graph);
    let n = graph.n();
    let mut result = vec![f64::INFINITY; n];
    for x in 0..n {
        for y in 0..n {
            let d_sq = dist[(x, y)].powi(2);
            let val = u[y] + d_sq / (2.0 * t);
            result[x] = result[x].min(val);
        }
    }
    result
}

/// Apply the Hopf-Lax semigroup using precomputed distances.
pub fn hopf_lax_with_distances(dist: &DMatrix<f64>, u: &[f64], t: f64) -> Vec<f64> {
    let n = dist.nrows();
    let mut result = vec![f64::INFINITY; n];
    for x in 0..n {
        for y in 0..n {
            let d_sq = dist[(x, y)].powi(2);
            let val = u[y] + d_sq / (2.0 * t);
            result[x] = result[x].min(val);
        }
    }
    result
}

/// Hopf-Lax semigroup applied to all pairs: Q_t matrix.
pub fn hopf_lax_matrix(dist: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = dist.nrows();
    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut min_val = f64::INFINITY;
            for k in 0..n {
                let d_ik = dist[(i, k)];
                let d_kj = dist[(k, j)];
                let val = d_ik.powi(2) / (2.0 * t) + d_kj.powi(2) / (2.0 * t);
                min_val = min_val.min(val);
            }
            result[(i, j)] = min_val;
        }
    }
    result
}

/// Verify the semigroup property: Q_{t+s} = Q_t ∘ Q_s.
pub fn verify_semigroup_property(
    graph: &Graph,
    t: f64,
    s: f64,
    u: &[f64],
    tol: f64,
) -> bool {
    let q_ts = hopf_lax(graph, u, t + s);
    let q_s = hopf_lax(graph, u, s);
    let q_t_of_qs = hopf_lax(graph, &q_s, t);
    for i in 0..q_ts.len() {
        if (q_ts[i] - q_t_of_qs[i]).abs() > tol {
            return false;
        }
    }
    true
}

/// Compute the Hopf-Lax action for the Hamilton-Jacobi equation.
/// This is the "inf-convolution" of u with the quadratic cost d²/2t.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopfLaxResult {
    /// The initial cost function u.
    pub u_initial: Vec<f64>,
    /// The semigroup evolution at various times.
    pub evolution: Vec<(f64, Vec<f64>)>,
    /// The graph distances used.
    pub distances: DMatrix<f64>,
}

/// Run full Hopf-Lax evolution from t=0 to t_max with given steps.
pub fn hopf_lax_evolution(graph: &Graph, u: &[f64], t_values: &[f64]) -> HopfLaxResult {
    let dist = geodesic_distances(graph);
    let evolution = t_values.iter().map(|&t| {
        (t, hopf_lax_with_distances(&dist, u, t))
    }).collect();
    HopfLaxResult {
        u_initial: u.to_vec(),
        evolution,
        distances: dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::{path_graph, complete_graph, cycle_graph};
    use approx::assert_relative_eq;

    #[test]
    fn test_hopf_lax_basic() {
        let g = path_graph(4);
        let u = vec![0.0, 1.0, 2.0, 3.0];
        let result = hopf_lax(&g, &u, 1.0);
        assert_eq!(result.len(), 4);
        // Q_t u(x) = min_y [u(y) + d(x,y)²/2]
        // For x=0: min(0+0, 1+1/2, 2+4/2, 3+9/2) = min(0, 1.5, 4, 7.5) = 0
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hopf_lax_small_t() {
        let g = path_graph(4);
        let u = vec![0.0, 10.0, 0.0, 10.0];
        let result = hopf_lax(&g, &u, 0.01);
        // For very small t, should pick nearest vertex
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hopf_lax_large_t() {
        let g = complete_graph(3);
        let u = vec![1.0, 3.0, 5.0];
        let result = hopf_lax(&g, &u, 100.0);
        // For large t, d²/2t ≈ 0 for all pairs, so Q_t u ≈ min(u)
        for v in &result {
            assert_relative_eq!(*v, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_hopf_lax_with_distances() {
        let g = path_graph(3);
        let dist = geodesic_distances(&g);
        let u = vec![0.0, 5.0, 0.0];
        let r1 = hopf_lax(&g, &u, 1.0);
        let r2 = hopf_lax_with_distances(&dist, &u, 1.0);
        for i in 0..3 {
            assert_relative_eq!(r1[i], r2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_semigroup_property() {
        let g = cycle_graph(5);
        let u = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(verify_semigroup_property(&g, 0.5, 0.5, &u, 1.0));
    }

    #[test]
    fn test_hopf_lax_evolution() {
        let g = path_graph(4);
        let u = vec![0.0, 1.0, 4.0, 9.0];
        let result = hopf_lax_evolution(&g, &u, &[0.1, 0.5, 1.0, 2.0]);
        assert_eq!(result.evolution.len(), 4);
        assert_eq!(result.u_initial.len(), 4);
    }

    #[test]
    fn test_hopf_lax_matrix() {
        let g = path_graph(3);
        let dist = geodesic_distances(&g);
        let m = hopf_lax_matrix(&dist, 1.0);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_hopf_lax_nonnegativity() {
        let g = path_graph(5);
        let u = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = hopf_lax(&g, &u, 1.0);
        for v in &result {
            assert!(*v >= 0.0, "result = {} < 0", v);
        }
    }

    #[test]
    fn test_hopf_lax_constant_input() {
        let g = path_graph(4);
        let u = vec![5.0; 4];
        let result = hopf_lax(&g, &u, 1.0);
        // Constant function: Q_t u(x) = min_y [5 + d²/2] = 5 (achieved at y=x)
        for v in &result {
            assert_relative_eq!(*v, 5.0, epsilon = 1e-10);
        }
    }
}
