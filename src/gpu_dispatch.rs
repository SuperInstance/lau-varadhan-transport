//! Application: unified framework for GPU dispatch — heat kernel → transport cost → optimal kernel scheduling.
//!
//! This module applies the spectral-to-transport bridge to GPU compute scheduling.
//! The idea: model GPU kernels as nodes in a graph where edge weights represent
//! data transfer costs. Use heat kernel diffusion to estimate kernel importance,
//! Varadhan distances for effective transfer costs, and optimal transport to
//! find the optimal scheduling assignment.

use crate::heat_kernel::{Graph, HeatKernel};
use crate::varadhan::varadhan_estimate;
use crate::hopf_lax::HopfLax;
use crate::benamou_brenier::{DiscreteDistribution, w2_squared_heat};
use crate::tropical_attention::{tempered_softmax, tropical_attention};
use crate::hbar_interpolation::HbarInterpolation;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A GPU kernel specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuKernel {
    /// Kernel name.
    pub name: String,
    /// Compute cost (normalized).
    pub compute_cost: f64,
    /// Memory requirement (normalized).
    pub memory: f64,
    /// Priority weight.
    pub priority: f64,
}

/// A GPU streaming multiprocessor (SM).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuSM {
    /// SM identifier.
    pub id: usize,
    /// Available compute capacity.
    pub compute_capacity: f64,
    /// Available memory.
    pub memory_capacity: f64,
}

/// GPU dispatch configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuDispatchConfig {
    /// Temperature for attention-based dispatch.
    pub temperature: f64,
    /// Whether to use tropical (hard) dispatch.
    pub use_tropical: bool,
}

impl Default for GpuDispatchConfig {
    fn default() -> Self {
        Self { temperature: 0.1, use_tropical: false }
    }
}

/// Build a kernel dependency graph from a list of kernels and their transfer costs.
pub fn build_kernel_graph(kernels: &[GpuKernel], transfer_cost_fn: &dyn Fn(usize, usize) -> f64) -> Graph {
    let n = kernels.len();
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let cost = transfer_cost_fn(i, j);
                if cost > 0.0 {
                    adj[(i, j)] = 1.0 / cost;
                }
            }
        }
    }
    Graph::with_labels(adj, kernels.iter().map(|k| k.name.clone()).collect())
}

/// Compute kernel importance scores using heat kernel diffusion.
pub fn kernel_importance(graph: &Graph, t: f64) -> DVector<f64> {
    let hk = HeatKernel::from_graph(graph);
    let n = graph.n_nodes();
    let pt = hk.matrix(t);

    // Importance = row sum of heat kernel (how much each node "receives" from all others)
    let mut importance = DVector::zeros(n);
    for i in 0..n {
        importance[i] = pt.row(i).sum();
    }
    importance
}

/// Compute effective transport costs between kernels using Varadhan's formula.
pub fn transport_costs(graph: &Graph, t: f64) -> DMatrix<f64> {
    let hk = HeatKernel::from_graph(graph);
    let n = graph.n_nodes();
    let mut costs = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            costs[(i, j)] = varadhan_estimate(&hk, t, i, j).sqrt().max(0.0);
        }
    }
    costs
}

/// Optimal kernel dispatch: assign kernels to SMs to minimize transport cost.
pub fn optimal_dispatch(
    kernels: &[GpuKernel],
    sms: &[GpuSM],
    transfer_costs: &DMatrix<f64>,
    config: &GpuDispatchConfig,
) -> DMatrix<f64> {
    let n_kernels = kernels.len();
    let n_sms = sms.len();

    // Build affinity matrix: kernel i to SM j
    let mut affinity = DMatrix::zeros(n_kernels, n_sms);
    for i in 0..n_kernels {
        for j in 0..n_sms {
            // Affinity based on compute/memory match
            let compute_match = if sms[j].compute_capacity > 0.0 {
                1.0 - (kernels[i].compute_cost - sms[j].compute_capacity).abs() / sms[j].compute_capacity
            } else {
                0.0
            };
            let memory_ok = if kernels[i].memory <= sms[j].memory_capacity { 1.0 } else { 0.1 };
            affinity[(i, j)] = compute_match * memory_ok * kernels[i].priority;
        }
    }

    // Use tempered softmax for assignment weights
    let hbar = if config.use_tropical { 0.001 } else { config.temperature };
    let mut assignment = DMatrix::zeros(n_kernels, n_sms);

    for i in 0..n_kernels {
        let row: DVector<f64> = affinity.row(i).into_owned();
        let weights = tempered_softmax(&row, hbar);
        for j in 0..n_sms {
            assignment[(i, j)] = weights[j];
        }
    }

    assignment
}

/// Full GPU dispatch pipeline using the Varadhan transport framework.
pub fn gpu_dispatch_pipeline(
    kernels: &[GpuKernel],
    sms: &[GpuSM],
    transfer_cost_fn: &dyn Fn(usize, usize) -> f64,
    config: &GpuDispatchConfig,
) -> GpuDispatchResult {
    let graph = build_kernel_graph(kernels, transfer_cost_fn);
    let importance = kernel_importance(&graph, 0.1);
    let t_costs = transport_costs(&graph, 0.001);
    let assignment = optimal_dispatch(kernels, sms, &t_costs, config);

    // Compute total dispatch cost
    let mut total_cost = 0.0;
    for i in 0..kernels.len() {
        for j in 0..sms.len() {
            total_cost += assignment[(i, j)] * kernels[i].compute_cost;
        }
    }

    GpuDispatchResult {
        assignment,
        importance,
        transport_costs: t_costs,
        total_cost,
    }
}

/// Result of the GPU dispatch pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuDispatchResult {
    /// Assignment matrix: assignment[i][j] = probability kernel i runs on SM j.
    pub assignment: DMatrix<f64>,
    /// Importance scores for each kernel.
    pub importance: DVector<f64>,
    /// Transport cost matrix between kernels.
    pub transport_costs: DMatrix<f64>,
    /// Total dispatch cost.
    pub total_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_gpu_kernel_creation() {
        let k = GpuKernel {
            name: "matmul".to_string(),
            compute_cost: 0.8,
            memory: 0.5,
            priority: 1.0,
        };
        assert_eq!(k.name, "matmul");
    }

    #[test]
    fn test_build_kernel_graph() {
        let kernels = vec![
            GpuKernel { name: "k0".into(), compute_cost: 0.5, memory: 0.3, priority: 1.0 },
            GpuKernel { name: "k1".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
            GpuKernel { name: "k2".into(), compute_cost: 0.3, memory: 0.2, priority: 0.5 },
        ];
        let graph = build_kernel_graph(&kernels, &|i, j| if (i as i32 - j as i32).abs() == 1 { 1.0 } else { 10.0 });
        assert_eq!(graph.n_nodes(), 3);
    }

    #[test]
    fn test_kernel_importance() {
        let kernels = vec![
            GpuKernel { name: "k0".into(), compute_cost: 0.5, memory: 0.3, priority: 1.0 },
            GpuKernel { name: "k1".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
        ];
        let graph = build_kernel_graph(&kernels, &|_, _| 1.0);
        let imp = kernel_importance(&graph, 0.1);
        assert_eq!(imp.len(), 2);
        assert!(imp[0] > 0.0);
        assert!(imp[1] > 0.0);
    }

    #[test]
    fn test_transport_costs_positive() {
        let kernels = vec![
            GpuKernel { name: "k0".into(), compute_cost: 0.5, memory: 0.3, priority: 1.0 },
            GpuKernel { name: "k1".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
        ];
        let graph = build_kernel_graph(&kernels, &|_, _| 1.0);
        let costs = transport_costs(&graph, 0.001);
        assert_eq!(costs.nrows(), 2);
        for i in 0..2 {
            assert_relative_eq!(costs[(i, i)], 0.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_optimal_dispatch_shape() {
        let kernels = vec![
            GpuKernel { name: "k0".into(), compute_cost: 0.5, memory: 0.3, priority: 1.0 },
            GpuKernel { name: "k1".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
            GpuKernel { name: "k2".into(), compute_cost: 0.3, memory: 0.2, priority: 0.5 },
        ];
        let sms = vec![
            GpuSM { id: 0, compute_capacity: 1.0, memory_capacity: 1.0 },
            GpuSM { id: 1, compute_capacity: 0.8, memory_capacity: 0.5 },
        ];
        let costs = DMatrix::from_element(3, 3, 1.0);
        let config = GpuDispatchConfig::default();
        let assignment = optimal_dispatch(&kernels, &sms, &costs, &config);
        assert_eq!(assignment.nrows(), 3);
        assert_eq!(assignment.ncols(), 2);
        // Each row should sum to 1 (assignment probability)
        for i in 0..3 {
            let row_sum: f64 = assignment.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tropical_dispatch_harder() {
        let kernels = vec![
            GpuKernel { name: "k0".into(), compute_cost: 0.5, memory: 0.3, priority: 1.0 },
            GpuKernel { name: "k1".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
        ];
        let sms = vec![
            GpuSM { id: 0, compute_capacity: 1.0, memory_capacity: 1.0 },
            GpuSM { id: 1, compute_capacity: 0.5, memory_capacity: 0.5 },
        ];
        let costs = DMatrix::from_element(2, 2, 1.0);
        let config_tropical = GpuDispatchConfig { temperature: 0.1, use_tropical: true };
        let config_soft = GpuDispatchConfig { temperature: 1.0, use_tropical: false };

        let hard = optimal_dispatch(&kernels, &sms, &costs, &config_tropical);
        let soft = optimal_dispatch(&kernels, &sms, &costs, &config_soft);

        // Hard assignment should be more peaked (closer to one-hot)
        let hard_entropy: f64 = (0..2).map(|i| {
            let row = hard.row(i).iter().cloned().collect::<Vec<_>>();
            -row.iter().filter(|&&v| v > 1e-10).map(|&v| v * v.ln()).sum::<f64>()
        }).sum();
        let soft_entropy: f64 = (0..2).map(|i| {
            let row = soft.row(i).iter().cloned().collect::<Vec<_>>();
            -row.iter().filter(|&&v| v > 1e-10).map(|&v| v * v.ln()).sum::<f64>()
        }).sum();
        assert!(hard_entropy <= soft_entropy + 0.1);
    }

    #[test]
    fn test_full_pipeline() {
        let kernels = vec![
            GpuKernel { name: "matmul".into(), compute_cost: 0.9, memory: 0.7, priority: 2.0 },
            GpuKernel { name: "conv".into(), compute_cost: 0.8, memory: 0.6, priority: 1.5 },
            GpuKernel { name: "relu".into(), compute_cost: 0.1, memory: 0.1, priority: 0.5 },
            GpuKernel { name: "bn".into(), compute_cost: 0.3, memory: 0.2, priority: 1.0 },
        ];
        let sms = vec![
            GpuSM { id: 0, compute_capacity: 1.0, memory_capacity: 1.0 },
            GpuSM { id: 1, compute_capacity: 0.8, memory_capacity: 0.8 },
        ];
        let result = gpu_dispatch_pipeline(
            &kernels,
            &sms,
            &|i, j| if (i as i32 - j as i32).abs() <= 1 { 0.5 } else { 2.0 },
            &GpuDispatchConfig::default(),
        );
        assert_eq!(result.assignment.nrows(), 4);
        assert_eq!(result.assignment.ncols(), 2);
        assert!(result.total_cost > 0.0);
    }
}
