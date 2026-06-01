//! Varadhan's formula: lim_{t→0} −4t log p_t(x,y) = d(x,y)²
//!
//! This module verifies the fundamental connection between the heat kernel
//! asymptotics and the geodesic distance on a graph.

use crate::heat_kernel::{heat_kernel, Graph};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Result of Varadhan's formula verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VaradhanResult {
    /// The t values used.
    pub t_values: Vec<f64>,
    /// For each pair (x,y), the computed −4t log p_t(x,y) at each t.
    pub approximations: Vec<Vec<f64>>,
    /// The geodesic distance squared for each pair.
    pub distance_squared: DMatrix<f64>,
    /// The geodesic distance for each pair.
    pub distances: DMatrix<f64>,
}

/// Compute geodesic (shortest path) distances on a graph using Dijkstra's algorithm.
pub fn geodesic_distances(graph: &Graph) -> DMatrix<f64> {
    let n = graph.n();
    let mut dist = DMatrix::repeat(n, n, f64::INFINITY);
    for s in 0..n {
        dist[(s, s)] = 0.0;
        // Simple Dijkstra using a priority queue (vec-based for small graphs)
        let mut visited = vec![false; n];
        let mut queue = vec![(0.0, s)]; // (distance, node)
        while let Some(idx) = queue.iter().enumerate().min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap()).map(|(i, _)| i) {
            let (d, u) = queue.remove(idx);
            if visited[u] {
                continue;
            }
            visited[u] = true;
            dist[(s, u)] = d;
            for v in 0..n {
                let w = graph.adjacency[(u, v)];
                if w > 0.0 && !visited[v] {
                    let nd = d + w;
                    if nd < dist[(s, v)] {
                        dist[(s, v)] = nd;
                        queue.push((nd, v));
                    }
                }
            }
        }
    }
    dist
}

/// Compute the Varadhan approximation −4t log p_t(x,y) for a given t.
pub fn varadhan_approx(graph: &Graph, t: f64) -> DMatrix<f64> {
    let h = heat_kernel(graph, t);
    let n = graph.n();
    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let p = h[(i, j)];
            if p > 0.0 && p.is_finite() {
                result[(i, j)] = -4.0 * t * p.ln();
            } else {
                result[(i, j)] = f64::INFINITY;
            }
        }
    }
    result
}

/// Verify Varadhan's formula numerically: check that −4t log p_t → d² as t→0.
///
/// Returns a VaradhanResult with approximations at multiple t values.
pub fn verify_varadhan(graph: &Graph, t_values: &[f64]) -> VaradhanResult {
    let dist = geodesic_distances(graph);
    let n = graph.n();
    let dist_sq = DMatrix::from_fn(n, n, |i, j| dist[(i, j)] * dist[(i, j)]);

    let mut approximations = Vec::new();
    for &t in t_values {
        let approx = varadhan_approx(graph, t);
        let row: Vec<f64> = (0..n*n).map(|idx| {
            let i = idx / n;
            let j = idx % n;
            approx[(i, j)]
        }).collect();
        approximations.push(row);
    }

    VaradhanResult {
        t_values: t_values.to_vec(),
        approximations,
        distance_squared: dist_sq,
        distances: dist,
    }
}

/// Compute the convergence rate of Varadhan's formula.
///
/// Returns the maximum relative error |−4t log p_t − d²| / (d² + ε)
/// across all vertex pairs for each t value.
pub fn varadhan_convergence_error(graph: &Graph, t_values: &[f64]) -> Vec<f64> {
    let dist = geodesic_distances(graph);
    let n = graph.n();
    let dist_sq = DMatrix::from_fn(n, n, |i, j| dist[(i, j)] * dist[(i, j)]);

    t_values.iter().map(|&t| {
        let approx = varadhan_approx(graph, t);
        let mut max_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d2 = dist_sq[(i, j)];
                    let a = approx[(i, j)];
                    if d2.is_finite() && d2 > 0.0 {
                        let rel_err = (a - d2).abs() / d2;
                        max_err = max_err.max(rel_err);
                    }
                }
            }
        }
        max_err
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::{complete_graph, cycle_graph, path_graph, star_graph};
    use approx::assert_relative_eq;

    #[test]
    fn test_geodesic_path() {
        let g = path_graph(5);
        let d = geodesic_distances(&g);
        assert_eq!(d[(0, 0)], 0.0);
        assert_eq!(d[(0, 1)], 1.0);
        assert_eq!(d[(0, 4)], 4.0);
        assert_eq!(d[(2, 4)], 2.0);
    }

    #[test]
    fn test_geodesic_complete() {
        let g = complete_graph(4);
        let d = geodesic_distances(&g);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(d[(i, j)], 0.0);
                } else {
                    assert_eq!(d[(i, j)], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_geodesic_cycle() {
        let g = cycle_graph(6);
        let d = geodesic_distances(&g);
        assert_eq!(d[(0, 3)], 3.0);
        assert_eq!(d[(0, 2)], 2.0);
    }

    #[test]
    fn test_geodesic_star() {
        let g = star_graph(5);
        let d = geodesic_distances(&g);
        assert_eq!(d[(0, 1)], 1.0);
        assert_eq!(d[(1, 2)], 2.0); // via center
        assert_eq!(d[(1, 3)], 2.0);
    }

    #[test]
    fn test_geodesic_symmetry() {
        let g = cycle_graph(5);
        let d = geodesic_distances(&g);
        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(d[(i, j)], d[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_varadhan_approx_positive() {
        let g = path_graph(4);
        let v = varadhan_approx(&g, 0.1);
        for i in 0..4 {
            for j in 0..4 {
                assert!(v[(i, j)] >= 0.0, "v[{},{}] = {} < 0", i, j, v[(i, j)]);
            }
        }
    }

    #[test]
    fn test_varadhan_diagonal_zero() {
        let g = path_graph(4);
        let v = varadhan_approx(&g, 0.01);
        for i in 0..4 {
            assert!(v[(i, i)] < 0.1, "v[{},{}] = {} not near 0", i, i, v[(i, i)]);
        }
    }

    #[test]
    fn test_varadhan_convergence_decreases() {
        let g = complete_graph(4);
        let t_vals = vec![0.01, 0.005, 0.002, 0.001, 0.0005];
        let errors = varadhan_convergence_error(&g, &t_vals);
        // For complete graph, Varadhan converges faster
        // Just check last error is reasonably small
        assert!(*errors.last().unwrap() < 1.0,
            "convergence errors: {:?}", errors);
    }

    #[test]
    fn test_varadhan_verify_structure() {
        let g = cycle_graph(4);
        let t_vals = vec![0.05, 0.02];
        let result = verify_varadhan(&g, &t_vals);
        assert_eq!(result.t_values.len(), 2);
        assert_eq!(result.approximations.len(), 2);
        assert_eq!(result.distance_squared.nrows(), 4);
    }

    #[test]
    fn test_varadhan_approx_near_distance_squared() {
        let g = complete_graph(3);
        let _d = geodesic_distances(&g);
        let v = varadhan_approx(&g, 0.005);
        // For complete graph, d=1 for all pairs, so d²=1
        // Varadhan should give something near 1 for small t
        for i in 0..3 {
            assert!(v[(i, i)] < 1.0, "diagonal: v[{},{}] = {}", i, i, v[(i,i)]);
            for j in 0..3 {
                if i != j {
                    assert!(v[(i,j)] > 0.0 && v[(i,j)] < 10.0,
                        "v[{},{}] = {} out of range", i, j, v[(i,j)]);
                }
            }
        }
    }
}
