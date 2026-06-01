//! # lau-varadhan-transport
//!
//! Varadhan's formula and Maslov dequantization: the bridge from spectral theory
//! through tropical geometry to optimal transport.
//!
//! Key theorem: `lim_{t→0} −4t log p_t(x,y) = d(x,y)²`
//!
//! The Cole-Hopf transform sends heat → Hamilton-Jacobi, and Maslov dequantization
//! interpolates from (+,×) to (min,+) as ℏ: 1→0.

pub mod heat_kernel;
pub mod varadhan;
pub mod cole_hopf;
pub mod maslov;
pub mod hopf_lax;
pub mod benamou_brenier;
pub mod tropical_attention;
pub mod spectral_transport;
pub mod gpu_schedule;

pub use heat_kernel::*;
pub use varadhan::*;
pub use cole_hopf::*;
pub use maslov::*;
pub use hopf_lax::*;
pub use benamou_brenier::*;
pub use tropical_attention::*;
pub use spectral_transport::*;
pub use gpu_schedule::*;
