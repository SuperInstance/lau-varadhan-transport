# lau-varadhan-transport

**Varadhan's formula and Maslov dequantization** — spectral theory through tropical geometry to optimal transport.

This crate bridges three mathematical worlds:
1. **Spectral theory**: heat kernels on graphs via Laplacian eigendecomposition
2. **Tropical geometry**: Maslov dequantization, where (+,×) → (min,+) as ℏ → 0
3. **Optimal transport**: Wasserstein distances, Benamou-Brenier dynamics, Sinkhorn regularization

The unifying theorem is **Varadhan's formula**: `lim_{t→0} −4t log p_t(x,y) = d(x,y)²`, connecting the heat kernel (spectral) to geodesic distance (metric). The **Cole-Hopf transform** converts between heat and Hamilton-Jacobi equations, while **ℏ-interpolation** smoothly connects the linear spectral world to the piecewise-linear tropical world.

---

## What This Does

- **Heat kernel computation**: matrix exponential of the graph Laplacian via eigendecomposition, with graph constructors (path, complete, cycle, star).
- **Varadhan's formula verification**: numerically confirm that `−4t log p_t(x,y) → d(x,y)²` as `t → 0`, with convergence rate tracking.
- **Cole-Hopf transform**: `v = −ℏ log u` converts heat solutions to Hamilton-Jacobi solutions; invertible roundtrip.
- **Maslov dequantization**: deformed addition `x ⊕_ℏ y → min(x,y)` as `ℏ→0`, tropical semirings (min-plus and max-plus), dequantization convergence tracking.
- **Hopf-Lax semigroup**: `Q_t u(x) = min_y [u(y) + d(x,y)²/(2t)]` — the viscosity solution semigroup for Hamilton-Jacobi, with verified semigroup property.
- **Benamou-Brenier transport**: Wasserstein-1 and W₂² distances, heat-kernel transport plans, McCann interpolation between distributions.
- **Tropical attention**: softmax → hardmax as temperature → 0, with entropy tracking and softening paths — a tropical interpretation of transformer attention.
- **Spectral transport bridge**: spectral embedding, diffusion distance, spectral gap, and the full bridge from eigenvalues to geodesic distances.
- **ℏ-interpolation**: Sinkhorn-regularized optimal transport via the Gibbs kernel `exp(−d²/ℏ)`, with ℏ controlling the spectral↔tropical transition.
- **GPU kernel scheduling**: two application modules applying the framework to real-world GPU dispatch — heat kernel priority, Varadhan transport costs, tropical attention-based assignment.

---

## Key Idea

> **Varadhan's formula**: `lim_{t→0} −4t log p_t(x,y) = d(x,y)²`
>
> The heat kernel `p_t(x,y) = [exp(−tL)]_{x,y}` encodes the geometry of the underlying space. As `t → 0`, the logarithm of the heat kernel recovers the squared geodesic distance. This connects:
> - **Spectral** (heat = Laplacian exponential)
> - **Geometric** (distance = shortest path)
> - **Tropical** (log transforms exp to +, and exp to min/max)

The **Maslov dequantization** `ℏ: 1 → 0` interpolates:
- `ℏ = 1`: standard arithmetic (+, ×), heat equation, Fourier analysis
- `ℏ → 0`: tropical arithmetic (min, +), Hamilton-Jacobi equation, shortest paths

---

## Install

```toml
[dependencies]
lau-varadhan-transport = "0.1"
```

Or via git:

```toml
[dependencies]
lau-varadhan-transport = { git = "https://github.com/SuperInstance/lau-varadhan-transport" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (linear algebra + serde), `serde` + `serde_json` (serialization), `approx` (dev-only, test assertions).

---

## Quick Start

### Heat Kernel and Varadhan's Formula

```rust
use lau_varadhan_transport::*;

// Build a graph
let graph = cycle_graph(6);

// Compute heat kernel at time t
let h = heat_kernel(&graph, 0.1);
println!("p_t(0,3) = {}", h[(0, 3)]);

// Verify Varadhan's formula
let t_values = vec![0.1, 0.05, 0.01, 0.005, 0.001];
let errors = varadhan_convergence_error(&graph, &t_values);
for (t, err) in t_values.iter().zip(errors.iter()) {
    println!("t={:.4}  max_rel_error={:.4}", t, err);
}
```

### Cole-Hopf Transform

```rust
use lau_varadhan_transport::*;

let graph = path_graph(5);
let h = heat_kernel(&graph, 0.1);

// Heat → Hamilton-Jacobi via Cole-Hopf
let hj = cole_hopf(&h, 1.0);

// Invertible
assert!(verify_cole_hopf_inverse(&h, 1.0, 1e-10));

// Full pipeline
let hj_direct = heat_to_hamilton_jacobi(&graph, 0.1, 1.0);
```

### Maslov Dequantization

```rust
use lau_varadhan_transport::*;

// Tropical semiring
assert_eq!(TropicalMinPlus::add(3.0, 5.0), 3.0);  // min
assert_eq!(TropicalMinPlus::mul(3.0, 5.0), 8.0);  // +

// Deformed addition converges to min as ℏ→0
let large_hbar = deformed_add_stable(1.0, 3.0, 10.0);  // ≈ -2.13
let small_hbar = deformed_add_stable(1.0, 3.0, 0.001); // ≈ 1.0 = min(1,3)

// Track convergence of dequantization
let values = vec![1.0, 5.0, 10.0];
let hbars = vec![1.0, 0.1, 0.01, 0.001];
let convergence = dequantization_convergence(&values, &hbars);
```

### Optimal Transport

```rust
use lau_varadhan_transport::*;

let graph = path_graph(4);
let mu = vec![1.0, 0.0, 0.0, 0.0];  // delta at vertex 0
let nu = vec![0.0, 0.0, 0.0, 1.0];  // delta at vertex 3

// Wasserstein distances
let w1 = wasserstein_1(&graph, &mu, &nu);          // = 3.0
let w2_sq = wasserstein_2_squared(&graph, &mu, &nu); // = 9.0

// Heat kernel transport plan
let plan = heat_kernel_transport(&graph, &mu, 0.5);
```

### Tropical Attention

```rust
use lau_varadhan_transport::*;
use nalgebra::DMatrix;

let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
let k = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
let v = DMatrix::from_row_slice(2, 2, &[10.0, 20.0, 30.0, 40.0]);

// Soft attention (temperature=1.0)
let soft = tropical_attention(&q, &k, &v, &TropicalAttentionConfig {
    temperature: 1.0, min_plus: false,
});

// Hard attention (temperature→0)
let hard = tropical_attention(&q, &k, &v, &TropicalAttentionConfig {
    temperature: 0.001, min_plus: false,
});
```

---

## API Reference

### Heat Kernel (`heat_kernel`)

| Type / Function | Description |
|---|---|
| `Graph` | Weighted undirected graph with adjacency matrix |
| `heat_kernel(graph, t)` | `p_t = exp(−tL)` via eigendecomposition |
| `heat_kernel_normalized(graph, t)` | Using normalized Laplacian |
| `heat_kernel_entry(graph, t, x, y)` | Single entry |
| `path_graph(n)`, `complete_graph(n)`, `cycle_graph(n)`, `star_graph(n)` | Graph constructors |
| `Graph::laplacian()`, `Graph::normalized_laplacian()` | `L = D − A` |
| `Graph::is_connected()` | BFS connectivity check |

### Varadhan's Formula (`varadhan`)

| Type / Function | Description |
|---|---|
| `geodesic_distances(graph)` | Dijkstra shortest paths |
| `varadhan_approx(graph, t)` | `−4t log p_t(x,y)` matrix |
| `verify_varadhan(graph, t_values)` | Full verification with multiple t values |
| `varadhan_convergence_error(graph, t_values)` | Max relative error per t |
| `VaradhanResult` | Structured result with distances and approximations |

### Cole-Hopf Transform (`cole_hopf`)

| Function | Description |
|---|---|
| `cole_hopf(u, ℏ)` | `v = −ℏ log u` (heat → HJ) |
| `cole_hopf_inverse(v, ℏ)` | `u = exp(−v/ℏ)` (HJ → heat) |
| `heat_to_hamilton_jacobi(graph, t, ℏ)` | Full pipeline |
| `verify_cole_hopf_inverse(u, ℏ, tol)` | Check invertibility |

### Maslov Dequantization (`maslov`)

| Type / Function | Description |
|---|---|
| `TropicalMinPlus` | `(ℝ∪{∞}, min, +)` semiring |
| `TropicalMaxPlus` | `(ℝ∪{−∞}, max, +)` semiring |
| `deformed_add_stable(a, b, ℏ)` | `−ℏ log(exp(−a/ℏ) + exp(−b/ℏ))` → min |
| `deformed_add_max(a, b, ℏ)` | Max-plus variant → max |
| `maslov_sum(values, ℏ)` | Deformed sum of multiple values |
| `maslov_polynomial(coeffs, x, ℏ)` | Tropical polynomial evaluation |
| `dequantization_convergence(values, hbars)` | Track ℏ → min convergence |

### Hopf-Lax Semigroup (`hopf_lax`)

| Function | Description |
|---|---|
| `hopf_lax(graph, u, t)` | `Q_t u(x) = min_y [u(y) + d²/2t]` |
| `hopf_lax_with_distances(dist, u, t)` | Precomputed distances |
| `verify_semigroup_property(graph, t, s, u, tol)` | Check `Q_{t+s} = Q_t ∘ Q_s` |
| `hopf_lax_evolution(graph, u, t_values)` | Full time evolution |

### Benamou-Brenier Transport (`benamou_brenier`)

| Function | Description |
|---|---|
| `wasserstein_1(graph, μ, ν)` | W₁ distance (earth mover's) |
| `wasserstein_2_squared(graph, μ, ν)` | W₂² distance |
| `heat_kernel_transport(graph, μ, t)` | Transport plan from heat flow |
| `benamou_brenier_energy(graph, t)` | Energy density `|∇log p_t|² p_t` |
| `transport_interpolation(graph, μ, ν, θ)` | McCann interpolation at θ ∈ [0,1] |

### Tropical Attention (`tropical_attention`)

| Type / Function | Description |
|---|---|
| `TropicalAttentionConfig` | Temperature `ℏ` and min/max-plus toggle |
| `tropical_attention_scores(Q, K, config)` | Attention score matrix |
| `tropical_attention(Q, K, V, config)` | Full attention: scores × V |
| `attention_entropy(Q, K, config)` | Per-row entropy |
| `attention_softening_path(Q, K, temps)` | Track soft → hard transition |

### Spectral Transport Bridge (`spectral_transport`)

| Type / Function | Description |
|---|---|
| `SpectralEmbedding` | Eigenvector coordinates |
| `spectral_embed(graph, k)` | First k non-trivial eigenvectors |
| `spectral_distance(embedding)` | Euclidean distance in spectral space |
| `diffusion_distance(graph, t)` | Heat-kernel-based distance |
| `spectral_gap(graph)` | Second-smallest Laplacian eigenvalue |
| `build_bridge(graph, k, t_values)` | Full spectral ↔ transport bridge |

### ℏ-Interpolation (`hbar_interpolation`)

| Type / Function | Description |
|---|---|
| `HbarInterpolation` | Interpolation state with ℏ parameter |
| `interp_add(a, b)` | Deformed addition |
| `gibbs_kernel(distance)` | `exp(−d/ℏ)` |
| `regularized_ot_cost(μ, ν, C, n_iter)` | Sinkhorn OT cost |
| `hbar_wasserstein(μ, ν, dist, ℏ, n_iter)` | ℏ-regularized W distance |

### GPU Scheduling (`gpu_schedule`, `gpu_dispatch`)

| Type / Function | Description |
|---|---|
| `KernelTask` | GPU kernel with compute cost and dependencies |
| `KernelSchedule` | Execution order, start/end times, transport cost |
| `schedule_kernels(tasks, t)` | Heat-kernel-priority scheduling |
| `heat_kernel_priority(tasks, t)` | Kernel importance via diffusion |
| `GpuKernel`, `GpuSM`, `GpuDispatchConfig` | Full GPU dispatch types |
| `gpu_dispatch_pipeline(kernels, sms, cost_fn, config)` | End-to-end dispatch |

---

## How It Works

### The Chain of Connections

```
Heat equation (∂_t u = Δu)
    ↓  Cole-Hopf: v = −ℏ log u
Hamilton-Jacobi (∂_t v + |∇v|² = ℏΔv)
    ↓  ℏ → 0 (Maslov dequantization)
Tropical HJ (∂_t v + |∇v|² = 0)
    ↓  Hopf-Lax semigroup
Optimal transport (min-cost flow)
```

### Numerical Pipeline

1. **Build a graph** → compute Laplacian `L = D − A`
2. **Heat kernel** → eigendecompose `L`, compute `exp(−tL)`
3. **Varadhan** → verify `−4t log p_t → d²` as `t → 0`
4. **Cole-Hopf** → transform heat kernel to distance-like function
5. **Maslov** → interpolate between smooth (ℏ=1) and tropical (ℏ→0)
6. **Transport** → compute Wasserstein distances, run Sinkhorn

### GPU Application

The framework is applied to GPU kernel scheduling:
- Model kernels as graph nodes with transfer-cost edges
- Heat kernel diffusion gives kernel importance scores
- Varadhan distances give effective transport costs
- Tropical attention assigns kernels to streaming multiprocessors
- ℏ parameter controls soft vs. hard assignment

---

## The Math

### Varadhan's Formula

**Theorem** (Varadhan, 1967): On a Riemannian manifold (or graph), the heat kernel satisfies:

```
lim_{t→0} −4t log p_t(x,y) = d(x,y)²
```

where `p_t(x,y) = [exp(−tL)]_{x,y}` is the heat kernel and `d(x,y)` is the geodesic distance.

### Cole-Hopf Transform

If `u` solves the heat equation `∂_t u = Δu`, then `v = −ℏ log u` solves:

```
∂_t v + |∇v|² = ℏ Δv
```

As `ℏ → 0`, this becomes the inviscid Hamilton-Jacobi equation `∂_t v + |∇v|² = 0`.

### Maslov Dequantization

The deformed addition:

```
x ⊕_ℏ y = −ℏ log(exp(−x/ℏ) + exp(−y/ℏ))
```

converges to `min(x, y)` as `ℏ → 0`. This gives a continuous family:

| ℏ | Addition | Multiplication | Algebra |
|---|----------|----------------|---------|
| 1 | `log(exp(a) + exp(b))` | `a + b` | Standard (up to shift) |
| → 0 | `min(a, b)` | `a + b` | Tropical (min-plus) |

### Hopf-Lax Formula

The viscosity solution of `∂_t v + |∇v|²/2 = 0` with initial condition `v(0) = u` is:

```
Q_t u(x) = inf_y { u(y) + |x − y|² / (2t) }
```

This satisfies the semigroup property: `Q_{t+s} = Q_t ∘ Q_s`.

### Benamou-Brenier Formulation

Optimal transport as a continuous fluid dynamics problem:

```
minimize  ∫₀¹ ∫ |v_t|² dμ_t dt
subject to  ∂_t μ_t + ∇ · (μ_t v_t) = 0  (continuity equation)
            μ_0 = μ, μ_1 = ν
```

The Wasserstein-2 distance is the square root of this minimum.

### ℏ-Regularized Optimal Transport (Sinkhorn)

Replace the hard min with a deformed min:

```
OT_ℏ(μ, ν) = min_π { ⟨C, π⟩ + ℏ KL(π || μ⊗ν) }
```

Solved via the Sinkhorn algorithm with kernel `K = exp(−C/ℏ)`. As `ℏ → 0`, converges to the true OT cost.

---

## Testing

**99 tests** across 12 modules covering:
- Heat kernel: construction, Laplacian, symmetry, positivity, row sums, connectivity
- Varadhan: geodesic distances (path, complete, cycle, star), convergence, symmetry
- Cole-Hopf: roundtrip, vector variants, Varadhan connection
- Maslov: tropical semiring laws, deformed addition limits, polynomial evaluation
- Hopf-Lax: basic, small/large t, semigroup property, constant input, non-negativity
- Benamou-Brenier: W₁, W₂², symmetry, heat kernel transport, interpolation
- Tropical attention: soft/hard scores, entropy, row sums, softening path
- Spectral transport: embedding, distances, spectral gap, bridge construction
- ℏ-interpolation: Gibbs kernel, Sinkhorn OT, monotonicity
- GPU scheduling: dependency graphs, priority, scheduling, makespan, serialization
- GPU dispatch: kernel importance, transport costs, assignment, full pipeline

---

## License

MIT
