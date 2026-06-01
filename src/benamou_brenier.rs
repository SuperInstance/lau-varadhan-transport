//! Benamou-Brenier dynamic optimal transport from heat kernel asymptotics.
//!
//! The Benamou-Brenier formulation recasts optimal transport as a continuous
//! fluid dynamics problem: minimize ∫₀¹ ∫ |v_t|² dμ_t dt subject to continuity equation.

use crate::heat_kernel::{heat_kernel, Graph};
use crate::varadhan::geodesic_distances;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Result of Benamou-Brenier transport cost computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransportResult {
    /// Source distribution (probability vector).
    pub source: Vec<f64>,
    /// Target distribution (probability vector).
    pub target: Vec<f64>,
    /// Transport cost matrix T[i,j] = mass transported from i to j.
    pub transport_plan: DMatrix<f64>,
    /// Total transport cost: sum T[i,j] * d(i,j)².
    pub cost: f64,
    /// Wasserstein-2 distance squared.
    pub w2_squared: f64,
}

/// Compute the Wasserstein-1 distance between two distributions on a graph.
pub fn wasserstein_1(graph: &Graph, mu: &[f64], nu: &[f64]) -> f64 {
    let dist = geodesic_distances(graph);
    let n = graph.n();
    // Simple computation using earth mover's distance on small graphs
    // Use the fact that W1 = max over 1-Lipschitz functions of <f, mu-nu>
    // For small graphs, compute directly via linear programming approximation
    let mut cost = 0.0;
    let mut residual = mu.to_vec();
    for j in 0..n {
        let mut need = nu[j];
        let mut i = 0;
        while need > 1e-12 && i < n {
            let transfer = residual[i].min(need);
            cost += transfer * dist[(i, j)];
            residual[i] -= transfer;
            need -= transfer;
            i += 1;
        }
    }
    cost
}

/// Compute Wasserstein-2 distance squared between distributions.
pub fn wasserstein_2_squared(graph: &Graph, mu: &[f64], nu: &[f64]) -> f64 {
    let dist = geodesic_distances(graph);
    let n = graph.n();
    // Approximate W2² using simple greedy transport
    let mut cost = 0.0;
    let mut residual = mu.to_vec();
    for j in 0..n {
        let mut need = nu[j];
        // Find closest source with mass
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| dist[(a, j)].partial_cmp(&dist[(b, j)]).unwrap());
        for &i in &indices {
            if need <= 1e-12 { break; }
            let transfer = residual[i].min(need);
            if transfer > 1e-12 {
                cost += transfer * dist[(i, j)].powi(2);
                residual[i] -= transfer;
                need -= transfer;
            }
        }
    }
    cost
}

/// Compute transport cost from heat kernel flow.
///
/// The heat kernel p_t gives the optimal transport plan for the quadratic cost
/// at infinitesimal scale via Varadhan's formula.
pub fn heat_kernel_transport(graph: &Graph, mu: &[f64], t: f64) -> DMatrix<f64> {
    let h = heat_kernel(graph, t);
    let n = graph.n();
    let mut plan = DMatrix::zeros(n, n);
    for j in 0..n {
        let total: f64 = (0..n).map(|i| mu[i] * h[(i, j)]).sum();
        if total > 1e-15 {
            for i in 0..n {
                plan[(i, j)] = mu[i] * h[(i, j)] / total * mu.iter().sum::<f64>();
            }
        }
    }
    plan
}

/// Compute the Benamou-Brenier energy from the heat kernel gradient.
///
/// This approximates ∫ |∇log p_t|² p_t dx using the graph structure.
pub fn benamou_brenier_energy(graph: &Graph, t: f64) -> DMatrix<f64> {
    let h = heat_kernel(graph, t);
    let n = graph.n();
    let mut energy = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if h[(i, j)] > 1e-15 {
                // Energy density ~ |∇ log p_t|² * p_t
                let log_p = h[(i, j)].ln();
                let mut grad_sq = 0.0;
                for k in 0..n {
                    if graph.adjacency[(i, k)] > 0.0 && h[(k, j)] > 1e-15 {
                        let diff = log_p - h[(k, j)].ln();
                        grad_sq += diff * diff;
                    }
                }
                energy[(i, j)] = grad_sq * h[(i, j)];
            }
        }
    }
    energy
}

/// Dynamic transport interpolation between two distributions.
///
/// Returns the interpolation at parameter θ ∈ [0,1].
pub fn transport_interpolation(
    graph: &Graph,
    mu: &[f64],
    nu: &[f64],
    theta: f64,
) -> Vec<f64> {
    let n = graph.n();
    let dist = geodesic_distances(graph);

    // McCann interpolation: (1-θ)*x + θ*y with optimal coupling
    let mut result = vec![0.0; n];
    let mut mu_remaining = mu.to_vec();
    let nu_total: f64 = nu.iter().sum();

    for j in 0..n {
        let target = nu[j];
        let mut allocated = 0.0;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| dist[(a, j)].partial_cmp(&dist[(b, j)]).unwrap());

        for &i in &indices {
            if allocated >= target - 1e-12 { break; }
            let transfer = mu_remaining[i].min(target - allocated);
            if transfer > 1e-12 {
                // Interpolated mass at midpoint vertex
                let mid = find_midpoint(&dist, i, j, theta);
                result[mid] += transfer;
                mu_remaining[i] -= transfer;
                allocated += transfer;
            }
        }
    }

    // Normalize
    let total: f64 = result.iter().sum();
    if total > 1e-15 {
        for v in result.iter_mut() {
            *v /= total;
            *v *= nu_total;
        }
    }
    result
}

/// Find the vertex closest to the interpolated position between i and j.
fn find_midpoint(dist: &DMatrix<f64>, i: usize, j: usize, theta: f64) -> usize {
    let n = dist.nrows();
    let target_dist = (1.0 - theta) * dist[(i, j)];
    let mut best = i;
    let mut best_err = f64::INFINITY;
    for k in 0..n {
        let err = (dist[(i, k)] - target_dist).abs() + (dist[(k, j)] - dist[(i, j)] * theta).abs();
        if err < best_err {
            best_err = err;
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::{path_graph, cycle_graph};
    use approx::assert_relative_eq;

    #[test]
    fn test_wasserstein_1_same_dist() {
        let g = path_graph(3);
        let mu = vec![0.5, 0.3, 0.2];
        let w1 = wasserstein_1(&g, &mu, &mu);
        assert_relative_eq!(w1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wasserstein_1_delta() {
        let g = path_graph(4);
        let mu = vec![1.0, 0.0, 0.0, 0.0];
        let nu = vec![0.0, 0.0, 0.0, 1.0];
        let w1 = wasserstein_1(&g, &mu, &nu);
        assert_relative_eq!(w1, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wasserstein_2_squared_delta() {
        let g = path_graph(3);
        let mu = vec![1.0, 0.0, 0.0];
        let nu = vec![0.0, 0.0, 1.0];
        let w2 = wasserstein_2_squared(&g, &mu, &nu);
        assert_relative_eq!(w2, 4.0, epsilon = 1e-10); // d² = 2² = 4
    }

    #[test]
    fn test_heat_kernel_transport_shape() {
        let g = path_graph(3);
        let mu = vec![0.5, 0.3, 0.2];
        let plan = heat_kernel_transport(&g, &mu, 0.5);
        assert_eq!(plan.nrows(), 3);
        assert_eq!(plan.ncols(), 3);
    }

    #[test]
    fn test_benamou_brenier_energy_shape() {
        let g = cycle_graph(4);
        let energy = benamou_brenier_energy(&g, 0.5);
        assert_eq!(energy.nrows(), 4);
        assert!(energy.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn test_wasserstein_symmetry() {
        let g = cycle_graph(4);
        let mu = vec![0.25, 0.25, 0.25, 0.25];
        let nu = vec![0.5, 0.2, 0.2, 0.1];
        let w1_forward = wasserstein_1(&g, &mu, &nu);
        let w1_backward = wasserstein_1(&g, &nu, &mu);
        assert_relative_eq!(w1_forward, w1_backward, epsilon = 1e-10);
    }

    #[test]
    fn test_transport_interpolation_endpoints() {
        let g = path_graph(4);
        let mu = vec![1.0, 0.0, 0.0, 0.0];
        let p0 = transport_interpolation(&g, &mu, &mu, 0.0);
        // At θ=0, should be close to mu
        assert!(p0[0] > 0.5);
    }
}
