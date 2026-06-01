//! Heat kernel on discrete graphs via matrix exponential of negative Laplacian.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A weighted undirected graph with adjacency matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    /// Weighted adjacency matrix (symmetric for undirected graphs).
    pub adjacency: DMatrix<f64>,
    /// Optional vertex labels.
    pub labels: Vec<String>,
}

impl Graph {
    /// Create a graph from a weighted adjacency matrix.
    pub fn new(adjacency: DMatrix<f64>) -> Self {
        let n = adjacency.nrows();
        Self {
            adjacency,
            labels: (0..n).map(|i| format!("v{}", i)).collect(),
        }
    }

    /// Create a graph with labels.
    pub fn with_labels(adjacency: DMatrix<f64>, labels: Vec<String>) -> Self {
        Self { adjacency, labels }
    }

    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.adjacency.nrows()
    }

    /// Compute the graph Laplacian: L = D - A.
    pub fn laplacian(&self) -> DMatrix<f64> {
        let n = self.n();
        let mut degree = DMatrix::zeros(n, n);
        for i in 0..n {
            let d: f64 = (0..n).map(|j| self.adjacency[(i, j)]).sum();
            degree[(i, i)] = d;
        }
        degree - &self.adjacency
    }

    /// Compute the normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}.
    pub fn normalized_laplacian(&self) -> DMatrix<f64> {
        let n = self.n();
        let lap = self.laplacian();
        let mut d_inv_sqrt = DMatrix::zeros(n, n);
        for i in 0..n {
            let d: f64 = (0..n).map(|j| self.adjacency[(i, j)]).sum();
            d_inv_sqrt[(i, i)] = if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 };
        }
        &d_inv_sqrt * &lap * &d_inv_sqrt
    }

    /// Check if the graph is connected (simple BFS check).
    pub fn is_connected(&self) -> bool {
        let n = self.n();
        if n == 0 {
            return true;
        }
        let mut visited = vec![false; n];
        let mut stack = vec![0];
        visited[0] = true;
        let mut count = 1;
        while let Some(v) = stack.pop() {
            for u in 0..n {
                if !visited[u] && self.adjacency[(v, u)] > 0.0 {
                    visited[u] = true;
                    count += 1;
                    stack.push(u);
                }
            }
        }
        count == n
    }
}

/// Compute the heat kernel p_t(x,y) = [exp(-tL)]_{x,y}.
///
/// For a graph Laplacian L, the heat kernel is the matrix exponential of -tL.
pub fn heat_kernel(graph: &Graph, t: f64) -> DMatrix<f64> {
    let lap = graph.laplacian();
    matrix_exp_neg_scaled(&lap, t)
}

/// Compute heat kernel using normalized Laplacian.
pub fn heat_kernel_normalized(graph: &Graph, t: f64) -> DMatrix<f64> {
    let lap = graph.normalized_laplacian();
    matrix_exp_neg_scaled(&lap, t)
}

/// Compute p_t(x,y) for specific vertices.
pub fn heat_kernel_entry(graph: &Graph, t: f64, x: usize, y: usize) -> f64 {
    let h = heat_kernel(graph, t);
    h[(x, y)]
}

/// Scale matrix exponential of -t*M using eigendecomposition.
fn matrix_exp_neg_scaled(m: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = m.nrows();
    if n == 0 {
        return DMatrix::zeros(0, 0);
    }
    // Use eigendecomposition: M = P D P^{-1}, exp(-tM) = P exp(-tD) P^{-1}
    // For symmetric matrices, P is orthogonal.
    let sym = (m + &m.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();

    let mut exp_d = DMatrix::zeros(n, n);
    for i in 0..n {
        exp_d[(i, i)] = (-t * eigen.eigenvalues[i]).exp();
    }

    &eigen.eigenvectors * &exp_d * eigen.eigenvectors.transpose()
}

/// Create a path graph on n vertices with unit weights.
pub fn path_graph(n: usize) -> Graph {
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n.saturating_sub(1) {
        adj[(i, i + 1)] = 1.0;
        adj[(i + 1, i)] = 1.0;
    }
    Graph::new(adj)
}

/// Create a complete graph K_n with unit weights.
pub fn complete_graph(n: usize) -> Graph {
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                adj[(i, j)] = 1.0;
            }
        }
    }
    Graph::new(adj)
}

/// Create a cycle graph on n vertices.
pub fn cycle_graph(n: usize) -> Graph {
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n {
        adj[(i, (i + 1) % n)] = 1.0;
        let ni = (i + 1) % n;
        adj[(ni, i)] = 1.0;
    }
    Graph::new(adj)
}

/// Create a star graph with center vertex 0.
pub fn star_graph(n: usize) -> Graph {
    let mut adj = DMatrix::zeros(n, n);
    for i in 1..n {
        adj[(0, i)] = 1.0;
        adj[(i, 0)] = 1.0;
    }
    Graph::new(adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_graph_construction() {
        let g = path_graph(4);
        assert_eq!(g.n(), 4);
        assert_eq!(g.adjacency[(0, 1)], 1.0);
        assert_eq!(g.adjacency[(0, 2)], 0.0);
        assert_eq!(g.adjacency[(1, 3)], 0.0);
    }

    #[test]
    fn test_laplacian_path() {
        let g = path_graph(3);
        let lap = g.laplacian();
        assert_eq!(lap[(0, 0)], 1.0);
        assert_eq!(lap[(1, 1)], 2.0);
        assert_eq!(lap[(0, 1)], -1.0);
        assert_eq!(lap[(1, 0)], -1.0);
    }

    #[test]
    fn test_laplacian_complete() {
        let g = complete_graph(4);
        let lap = g.laplacian();
        assert_eq!(lap[(0, 0)], 3.0);
        assert_eq!(lap[(0, 1)], -1.0);
    }

    #[test]
    fn test_heat_kernel_t0_is_identity() {
        let g = path_graph(4);
        let h = heat_kernel(&g, 1e-12);
        for i in 0..4 {
            assert_relative_eq!(h[(i, i)], 1.0, epsilon = 1e-6);
            for j in 0..4 {
                if i != j {
                    assert!(h[(i, j)].abs() < 1e-4, "h[{},{}] = {}", i, j, h[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn test_heat_kernel_symmetry() {
        let g = path_graph(5);
        let h = heat_kernel(&g, 0.5);
        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(h[(i, j)], h[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_heat_kernel_positive() {
        let g = cycle_graph(5);
        let h = heat_kernel(&g, 0.1);
        for i in 0..5 {
            for j in 0..5 {
                assert!(h[(i, j)] > 0.0, "h[{},{}] = {} not positive", i, j, h[(i, j)]);
            }
        }
    }

    #[test]
    fn test_heat_kernel_row_sums() {
        let g = complete_graph(4);
        let h = heat_kernel(&g, 1.0);
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| h[(i, j)]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_connected_path() {
        assert!(path_graph(5).is_connected());
    }

    #[test]
    fn test_connected_complete() {
        assert!(complete_graph(5).is_connected());
    }

    #[test]
    fn test_star_graph() {
        let g = star_graph(5);
        assert_eq!(g.adjacency[(0, 1)], 1.0);
        assert_eq!(g.adjacency[(1, 2)], 0.0);
        assert!(g.is_connected());
    }

    #[test]
    fn test_normalized_laplacian() {
        let g = path_graph(3);
        let nl = g.normalized_laplacian();
        assert_relative_eq!(nl[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_heat_kernel_entry() {
        let g = path_graph(4);
        let val = heat_kernel_entry(&g, 0.5, 0, 1);
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}
