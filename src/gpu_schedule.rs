//! GPU kernel scheduling via heat kernel → transport cost.
//!
//! This application models GPU kernel scheduling as an optimal transport problem
//! where the heat kernel on a task dependency graph determines scheduling costs.

use crate::heat_kernel::{heat_kernel, Graph};
use crate::varadhan::geodesic_distances;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A GPU kernel task with compute cost and memory requirements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelTask {
    pub id: usize,
    pub name: String,
    pub compute_cost: f64,
    pub memory_mb: f64,
    pub dependencies: Vec<usize>,
}

/// A GPU kernel schedule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelSchedule {
    /// Task execution order.
    pub order: Vec<usize>,
    /// Start time for each task.
    pub start_times: Vec<f64>,
    /// End time for each task.
    pub end_times: Vec<f64>,
    /// Total schedule makespan.
    pub makespan: f64,
    /// Transport cost of the schedule.
    pub transport_cost: f64,
}

/// Build a dependency graph from kernel tasks.
pub fn build_dependency_graph(tasks: &[KernelTask]) -> Graph {
    let n = tasks.len();
    let mut adj = DMatrix::zeros(n, n);
    for task in tasks {
        for &dep in &task.dependencies {
            // Edge weight = compute cost of dependency
            adj[(dep, task.id)] = tasks[dep].compute_cost;
            adj[(task.id, dep)] = tasks[dep].compute_cost;
        }
    }
    Graph::new(adj)
}

/// Compute scheduling priority using heat kernel diffusion.
///
/// Tasks with higher heat kernel centrality get higher priority.
pub fn heat_kernel_priority(tasks: &[KernelTask], t: f64) -> Vec<(usize, f64)> {
    let graph = build_dependency_graph(tasks);
    let h = heat_kernel(&graph, t);
    let n = tasks.len();

    let mut priorities: Vec<(usize, f64)> = (0..n).map(|i| {
        // Priority = inverse of average heat kernel distance to all others
        let avg_h: f64 = (0..n).map(|j| h[(i, j)]).sum::<f64>() / n as f64;
        (i, 1.0 / (avg_h + 1e-10))
    }).collect();

    priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    priorities
}

/// Schedule kernels using heat kernel-based optimal transport.
///
/// Uses Varadhan's formula to estimate transport costs between task distributions
/// and finds a near-optimal schedule.
pub fn schedule_kernels(tasks: &[KernelTask], t: f64) -> KernelSchedule {
    let n = tasks.len();
    if n == 0 {
        return KernelSchedule {
            order: vec![],
            start_times: vec![],
            end_times: vec![],
            makespan: 0.0,
            transport_cost: 0.0,
        };
    }

    let graph = build_dependency_graph(tasks);
    let priorities = heat_kernel_priority(tasks, t);
    let _dist = geodesic_distances(&graph);

    // Schedule in priority order, respecting dependencies
    let mut scheduled = vec![false; n];
    let mut end_time = vec![0.0; n];
    let mut order = Vec::new();

    for _ in 0..n {
        // Find highest priority unscheduled task with all deps met
        let next = priorities.iter().find(|&&(idx, _)| {
            !scheduled[idx] && tasks[idx].dependencies.iter().all(|&d| scheduled[d])
        });

        if let Some(&(idx, _)) = next {
            scheduled[idx] = true;
            let dep_end = tasks[idx].dependencies.iter()
                .map(|&d| end_time[d])
                .fold(0.0f64, f64::max);
            let start = dep_end;
            let end = start + tasks[idx].compute_cost;
            end_time[idx] = end;
            order.push(idx);
        }
    }

    let makespan = end_time.iter().cloned().fold(0.0f64, f64::max);

    // Compute transport cost using Varadhan's formula
    let varadhan = crate::varadhan::varadhan_approx(&graph, t);
    let mut transport_cost = 0.0;
    for task in tasks {
        for &dep in &task.dependencies {
            transport_cost += varadhan[(dep, task.id)];
        }
    }

    let start_times = order.iter().map(|&i| end_time[i] - tasks[i].compute_cost).collect();

    KernelSchedule {
        order,
        start_times,
        end_times: end_time,
        makespan,
        transport_cost,
    }
}

/// Evaluate a schedule's quality using transport-theoretic metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduleEvaluation {
    pub makespan: f64,
    pub transport_cost: f64,
    pub avg_latency: f64,
    pub critical_path_length: f64,
}

/// Evaluate a kernel schedule.
pub fn evaluate_schedule(tasks: &[KernelTask], schedule: &KernelSchedule) -> ScheduleEvaluation {
    let n = tasks.len();
    let makespan = schedule.makespan;
    let avg_latency = if n > 0 {
        schedule.start_times.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    // Compute critical path length
    let mut cp = vec![0.0; n];
    for &task_id in &schedule.order {
        let dep_max = tasks[task_id].dependencies.iter()
            .map(|&d| cp[d])
            .fold(0.0f64, f64::max);
        cp[task_id] = dep_max + tasks[task_id].compute_cost;
    }
    let critical_path = cp.iter().cloned().fold(0.0f64, f64::max);

    ScheduleEvaluation {
        makespan,
        transport_cost: schedule.transport_cost,
        avg_latency,
        critical_path_length: critical_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tasks() -> Vec<KernelTask> {
        vec![
            KernelTask { id: 0, name: "load".into(), compute_cost: 1.0, memory_mb: 100.0, dependencies: vec![] },
            KernelTask { id: 1, name: "compute_a".into(), compute_cost: 3.0, memory_mb: 200.0, dependencies: vec![0] },
            KernelTask { id: 2, name: "compute_b".into(), compute_cost: 2.0, memory_mb: 150.0, dependencies: vec![0] },
            KernelTask { id: 3, name: "merge".into(), compute_cost: 1.5, memory_mb: 300.0, dependencies: vec![1, 2] },
            KernelTask { id: 4, name: "store".into(), compute_cost: 0.5, memory_mb: 50.0, dependencies: vec![3] },
        ]
    }

    #[test]
    fn test_build_dependency_graph() {
        let tasks = make_test_tasks();
        let g = build_dependency_graph(&tasks);
        assert_eq!(g.n(), 5);
        assert!(g.adjacency[(0, 1)] > 0.0);
        assert!(g.adjacency[(1, 3)] > 0.0);
    }

    #[test]
    fn test_heat_kernel_priority() {
        let tasks = make_test_tasks();
        let priorities = heat_kernel_priority(&tasks, 0.5);
        assert_eq!(priorities.len(), 5);
        // All priorities should be positive
        for (_, p) in &priorities {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_schedule_kernels() {
        let tasks = make_test_tasks();
        let schedule = schedule_kernels(&tasks, 0.5);
        assert_eq!(schedule.order.len(), 5);
        assert!(schedule.makespan > 0.0);
        // Task 0 should come first
        assert_eq!(schedule.order[0], 0);
    }

    #[test]
    fn test_schedule_respects_dependencies() {
        let tasks = make_test_tasks();
        let schedule = schedule_kernels(&tasks, 0.5);
        let mut position = vec![0; 5];
        for (i, &task_id) in schedule.order.iter().enumerate() {
            position[task_id] = i;
        }
        for task in &tasks {
            for &dep in &task.dependencies {
                assert!(position[dep] < position[task.id],
                    "dep {} scheduled after task {}", dep, task.id);
            }
        }
    }

    #[test]
    fn test_schedule_makespan() {
        let tasks = make_test_tasks();
        let schedule = schedule_kernels(&tasks, 0.5);
        // Critical path: load(1) + compute_a(3) + merge(1.5) + store(0.5) = 6.0
        // or: load(1) + compute_b(2) + merge(1.5) + store(0.5) = 5.0
        // But compute_a and compute_b can overlap after load
        // So: load(1) + max(compute_a(3), compute_b(2)) + merge(1.5) + store(0.5)
        // = 1 + 3 + 1.5 + 0.5 = 6.0
        assert!(schedule.makespan >= 5.5 && schedule.makespan <= 7.0);
    }

    #[test]
    fn test_evaluate_schedule() {
        let tasks = make_test_tasks();
        let schedule = schedule_kernels(&tasks, 0.5);
        let eval = evaluate_schedule(&tasks, &schedule);
        assert!(eval.makespan > 0.0);
        assert!(eval.avg_latency >= 0.0);
        assert!(eval.critical_path_length > 0.0);
    }

    #[test]
    fn test_schedule_empty() {
        let schedule = schedule_kernels(&[], 0.5);
        assert_eq!(schedule.order.len(), 0);
        assert_eq!(schedule.makespan, 0.0);
    }

    #[test]
    fn test_schedule_single() {
        let tasks = vec![
            KernelTask { id: 0, name: "solo".into(), compute_cost: 2.0, memory_mb: 100.0, dependencies: vec![] },
        ];
        let schedule = schedule_kernels(&tasks, 0.5);
        assert_eq!(schedule.order, vec![0]);
        assert_eq!(schedule.makespan, 2.0);
    }

    #[test]
    fn test_schedule_serialization() {
        let tasks = make_test_tasks();
        let schedule = schedule_kernels(&tasks, 0.5);
        let json = serde_json::to_string(&schedule).unwrap();
        let deserialized: KernelSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.order, schedule.order);
    }
}
