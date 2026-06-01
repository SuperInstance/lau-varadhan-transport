//! Spectral-to-transport bridge.

use crate::heat_kernel::{heat_kernel, Graph};
use crate::varadhan::geodesic_distances;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Spectral embedding of a graph using the first k eigenvectors of the Laplacian.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralEmbedding {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: DMatrix<f64>,
    pub coordinates: DMatrix<f64>,
}

/// Compute the spectral embedding using the first k non-trivial eigenvectors.
pub fn spectral_embed(graph: &Graph, k: usize) -> SpectralEmbedding {
    let lap = graph.laplacian();
    let sym = (&lap + &lap.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();

    let n = graph.n();
    let all_vals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    let all_vecs = eigen.eigenvectors;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| all_vals[a].partial_cmp(&all_vals[b]).unwrap());

    let start = if !indices.is_empty() && all_vals[indices[0]].abs() < 1e-10 { 1 } else { 0 };
    let k = k.min(n.saturating_sub(start));
    let selected: Vec<usize> = indices.iter().skip(start).take(k).copied().collect();

    let eigenvalues: Vec<f64> = selected.iter().map(|&i| all_vals[i]).collect();
    let mut coordinates = DMatrix::zeros(n, k);
    for (j, &idx) in selected.iter().enumerate() {
        for i in 0..n {
            coordinates[(i, j)] = all_vecs[(i, idx)];
        }
    }

    SpectralEmbedding {
        eigenvalues,
        eigenvectors: all_vecs,
        coordinates,
    }
}

/// Compute spectral distance: Euclidean distance in spectral embedding space.
pub fn spectral_distance(embedding: &SpectralEmbedding) -> DMatrix<f64> {
    let n = embedding.coordinates.nrows();
    let mut dist = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let d: f64 = (0..embedding.coordinates.ncols())
                .map(|k| {
                    let diff = embedding.coordinates[(i, k)] - embedding.coordinates[(j, k)];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            dist[(i, j)] = d;
        }
    }
    dist
}

/// Compute the diffusion distance at time t from the heat kernel.
pub fn diffusion_distance(graph: &Graph, t: f64) -> DMatrix<f64> {
    let h = heat_kernel(graph, t);
    let n = graph.n();
    let mut dist = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let d: f64 = (0..n)
                .map(|k| {
                    let diff = h[(i, k)] - h[(j, k)];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            dist[(i, j)] = d;
        }
    }
    dist
}

/// Spectral-to-transport bridge data.
#[derive(Clone, Debug)]
pub struct SpectralTransportBridge {
    pub n: usize,
    pub eigenvalues: Vec<f64>,
    pub geodesic_dist: DMatrix<f64>,
    pub spectral_dist: DMatrix<f64>,
    pub diffusion_distances: Vec<(f64, DMatrix<f64>)>,
    pub varadhan_approx: Vec<(f64, DMatrix<f64>)>,
}

/// Build the full spectral-to-transport bridge.
pub fn build_bridge(graph: &Graph, k: usize, t_values: &[f64]) -> SpectralTransportBridge {
    let n = graph.n();
    let embedding = spectral_embed(graph, k);
    let spectral_dist = spectral_distance(&embedding);
    let geodesic_dist = geodesic_distances(graph);

    let diffusion_distances: Vec<(f64, DMatrix<f64>)> = t_values
        .iter()
        .map(|&t| (t, diffusion_distance(graph, t)))
        .collect();

    let varadhan_approx: Vec<(f64, DMatrix<f64>)> = t_values
        .iter()
        .map(|&t| (t, crate::varadhan::varadhan_approx(graph, t)))
        .collect();

    SpectralTransportBridge {
        n,
        eigenvalues: embedding.eigenvalues,
        geodesic_dist,
        spectral_dist,
        diffusion_distances,
        varadhan_approx,
    }
}

/// Compute the spectral gap (second smallest eigenvalue of the Laplacian).
pub fn spectral_gap(graph: &Graph) -> f64 {
    let lap = graph.laplacian();
    let sym = (&lap + &lap.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();
    let mut vals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if vals.len() < 2 { return 0.0; }
    vals[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::{path_graph, complete_graph, cycle_graph};
    use approx::assert_relative_eq;

    #[test]
    fn test_spectral_embed_dimensions() {
        let g = path_graph(5);
        let emb = spectral_embed(&g, 2);
        assert_eq!(emb.coordinates.nrows(), 5);
        assert_eq!(emb.coordinates.ncols(), 2);
        assert_eq!(emb.eigenvalues.len(), 2);
    }

    #[test]
    fn test_spectral_embed_eigenvalues() {
        let g = path_graph(3);
        let emb = spectral_embed(&g, 2);
        for &v in &emb.eigenvalues {
            assert!(v >= -1e-10, "eigenvalue {} < 0", v);
        }
    }

    #[test]
    fn test_spectral_distance_symmetry() {
        let g = cycle_graph(5);
        let emb = spectral_embed(&g, 2);
        let dist = spectral_distance(&emb);
        for i in 0..5 {
            assert_relative_eq!(dist[(i, i)], 0.0, epsilon = 1e-10);
            for j in 0..5 {
                assert_relative_eq!(dist[(i, j)], dist[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_diffusion_distance_symmetry() {
        let g = path_graph(4);
        let dd = diffusion_distance(&g, 0.5);
        for i in 0..4 {
            assert_relative_eq!(dd[(i, i)], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spectral_gap_path() {
        let g = path_graph(4);
        let gap = spectral_gap(&g);
        assert!(gap > 0.0);
        assert!(gap < 10.0);
    }

    #[test]
    fn test_spectral_gap_complete() {
        let g = complete_graph(5);
        let gap = spectral_gap(&g);
        assert!(gap > 0.0);
        assert_relative_eq!(gap, 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_build_bridge() {
        let g = cycle_graph(4);
        let bridge = build_bridge(&g, 2, &[0.1, 0.5]);
        assert_eq!(bridge.n, 4);
        assert_eq!(bridge.eigenvalues.len(), 2);
        assert_eq!(bridge.diffusion_distances.len(), 2);
        assert_eq!(bridge.varadhan_approx.len(), 2);
    }
}
