//! Tropical attention: softmax → hardmax as ℏ→0
//!
//! Standard attention uses softmax which is the log-sum-exp / Maslov dequantized
//! version of hardmax. As ℏ→0, the attention mechanism becomes tropical:
//! attention(Q,K,V) with temperature ℏ interpolates between soft and hard attention.

// unused import removed
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Tropical attention configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TropicalAttentionConfig {
    /// Temperature parameter (ℏ). As ℏ→0, becomes hard attention.
    pub temperature: f64,
    /// Whether to use min-plus (true) or max-plus (false).
    pub min_plus: bool,
}

impl Default for TropicalAttentionConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_plus: true,
        }
    }
}

/// Compute tropical attention scores.
///
/// In standard attention: score(i,j) = softmax(Q·K^T / sqrt(d))
/// In tropical attention: score(i,j) = ℏ log Σ_k exp((Q·K^T)_k / ℏ) → min/max as ℏ→0
pub fn tropical_attention_scores(
    queries: &DMatrix<f64>,
    keys: &DMatrix<f64>,
    config: &TropicalAttentionConfig,
) -> DMatrix<f64> {
    let hbar = config.temperature;
    let scores = queries * keys.transpose();

    // Apply tropical softmax row-wise
    let n = scores.nrows();
    let m = scores.ncols();
    let mut result = DMatrix::zeros(n, m);

    for i in 0..n {
        // Compute tropical normalization
        let row: Vec<f64> = (0..m).map(|j| scores[(i, j)]).collect();
        let row_min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if hbar < 1e-10 {
            // Hard attention: all mass on the min (or max)
            let target = if config.min_plus { row_min } else { row_max };
            for j in 0..m {
                result[(i, j)] = if (scores[(i, j)] - target).abs() < 1e-10 { 1.0 } else { 0.0 };
            }
            // Normalize
            let total: f64 = (0..m).map(|j| result[(i, j)]).sum();
            if total > 0.0 {
                for j in 0..m {
                    result[(i, j)] /= total;
                }
            }
        } else {
            // Soft attention with temperature
            let _center = if config.min_plus { row_min } else { row_max };
            let mut exp_sum = 0.0;
            let mut exps = vec![0.0; m];
            for j in 0..m {
                if config.min_plus {
                    exps[j] = ((row_min - scores[(i, j)]) / hbar).exp();
                } else {
                    exps[j] = ((scores[(i, j)] - row_max) / hbar).exp();
                }
                exp_sum += exps[j];
            }
            for j in 0..m {
                result[(i, j)] = exps[j] / exp_sum;
            }
        }
    }
    result
}

/// Apply tropical attention: output = attention_scores · values.
pub fn tropical_attention(
    queries: &DMatrix<f64>,
    keys: &DMatrix<f64>,
    values: &DMatrix<f64>,
    config: &TropicalAttentionConfig,
) -> DMatrix<f64> {
    let scores = tropical_attention_scores(queries, keys, config);
    scores * values
}

/// Compute the entropy of attention scores at a given temperature.
/// As ℏ→0, entropy → 0 (hard attention).
pub fn attention_entropy(
    queries: &DMatrix<f64>,
    keys: &DMatrix<f64>,
    config: &TropicalAttentionConfig,
) -> Vec<f64> {
    let scores = tropical_attention_scores(queries, keys, config);
    let n = scores.nrows();
    let m = scores.ncols();
    let mut entropy = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            let p = scores[(i, j)];
            if p > 1e-15 {
                entropy[i] -= p * p.ln();
            }
        }
    }
    entropy
}

/// Track the softening path: how attention changes from hard (ℏ=0) to soft (ℏ=∞).
pub struct AttentionSofteningPath {
    pub temperatures: Vec<f64>,
    pub attention_matrices: Vec<DMatrix<f64>>,
    pub entropies: Vec<Vec<f64>>,
}

/// Compute the full softening path from hard to soft attention.
pub fn attention_softening_path(
    queries: &DMatrix<f64>,
    keys: &DMatrix<f64>,
    temperatures: &[f64],
) -> AttentionSofteningPath {
    let mut attention_matrices = Vec::new();
    let mut entropies = Vec::new();

    for &t in temperatures {
        let config = TropicalAttentionConfig {
            temperature: t,
            min_plus: false,
        };
        attention_matrices.push(tropical_attention_scores(queries, keys, &config));
        entropies.push(attention_entropy(queries, keys, &config));
    }

    AttentionSofteningPath {
        temperatures: temperatures.to_vec(),
        attention_matrices,
        entropies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tropical_attention_scores_soft() {
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let config = TropicalAttentionConfig {
            temperature: 1.0,
            min_plus: false,
        };
        let scores = tropical_attention_scores(&q, &k, &config);
        assert_eq!(scores.nrows(), 2);
        assert_eq!(scores.ncols(), 2);
        // Rows should sum to 1
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| scores[(i, j)]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tropical_attention_scores_hard() {
        let q = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let config = TropicalAttentionConfig {
            temperature: 0.001,
            min_plus: false,
        };
        let scores = tropical_attention_scores(&q, &k, &config);
        // Should concentrate on max similarity
        assert!(scores[(0, 0)] > 0.99);
    }

    #[test]
    fn test_tropical_attention_output() {
        let q = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let v = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let config = TropicalAttentionConfig::default();
        let output = tropical_attention(&q, &k, &v, &config);
        assert_eq!(output.nrows(), 1);
        assert_eq!(output.ncols(), 2);
    }

    #[test]
    fn test_attention_entropy_soft() {
        let q = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let config = TropicalAttentionConfig {
            temperature: 1.0,
            min_plus: false,
        };
        let ent = attention_entropy(&q, &k, &config);
        // Uniform attention should have high entropy
        assert!(ent[0] > 0.5);
    }

    #[test]
    fn test_attention_entropy_hard() {
        let q = DMatrix::from_row_slice(1, 2, &[10.0, 0.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let config = TropicalAttentionConfig {
            temperature: 0.001,
            min_plus: false,
        };
        let ent = attention_entropy(&q, &k, &config);
        // Hard attention should have near-zero entropy
        assert!(ent[0] < 0.1);
    }

    #[test]
    fn test_attention_softening_path() {
        let q = DMatrix::from_row_slice(1, 2, &[1.0, 0.5]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let temps = vec![0.01, 0.1, 1.0, 10.0];
        let path = attention_softening_path(&q, &k, &temps);
        assert_eq!(path.attention_matrices.len(), 4);
        assert_eq!(path.entropies.len(), 4);
        // Entropy should increase with temperature
        assert!(path.entropies[0][0] <= path.entropies[3][0] + 0.01);
    }

    #[test]
    fn test_min_plus_attention() {
        let q = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let config = TropicalAttentionConfig {
            temperature: 0.001,
            min_plus: true,
        };
        let scores = tropical_attention_scores(&q, &k, &config);
        // Min-plus: should concentrate on minimum score
        let row_sum: f64 = (0..2).map(|j| scores[(0, j)]).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_attention_row_sums_one() {
        let q = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
        let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        for temp in &[0.01, 0.1, 1.0, 10.0] {
            let config = TropicalAttentionConfig {
                temperature: *temp,
                min_plus: false,
            };
            let scores = tropical_attention_scores(&q, &k, &config);
            for i in 0..3 {
                let row_sum: f64 = (0..2).map(|j| scores[(i, j)]).sum();
                assert_relative_eq!(row_sum, 1.0, epsilon = 1e-8);
            }
        }
    }
}
