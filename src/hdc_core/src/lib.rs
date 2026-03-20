use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIndexError};
use core::arch::x86_64::*;
use std::ptr;

/// Zero-cost abstraction Memory Arena for complex FHRR hypervectors built for PyO3.
///
/// Supports dynamic dimension expansion (Meta-Grammar Emergence) and
/// algebraic fusion operators (bind ∘ bundle) for topological phase transitions.
#[pyclass]
pub struct FhrrArena {
    buffer: Vec<f32>,
    dimension: usize,
    capacity: usize,
    head: usize,
    /// Tracks cumulative bind/bundle operations per handle for thermodynamic cost.
    op_counts: Vec<(u32, u32)>,  // (bind_count, bundle_count) per handle
}

#[pymethods]
impl FhrrArena {
    #[new]
    pub fn new(capacity: usize, dimension: usize) -> Self {
        let floats_per_vec = dimension * 2;
        let total_floats = capacity * floats_per_vec;
        
        let mut buffer = Vec::with_capacity(total_floats);
        
        unsafe {
            ptr::write_bytes(buffer.as_mut_ptr(), 0, total_floats);
            buffer.set_len(total_floats);
        }

        let op_counts = vec![(0u32, 0u32); capacity];

        Self {
            buffer,
            dimension,
            capacity,
            head: 0,
            op_counts,
        }
    }

    pub fn allocate(&mut self) -> PyResult<usize> {
        if self.head >= self.capacity {
            return Err(PyValueError::new_err("Arena capacity exhausted."));
        }
        let id = self.head;
        self.head += 1;
        Ok(id)
    }

    pub fn reset(&mut self) {
        self.head = 0;
    }

    pub fn inject_phases(&mut self, handle: usize, phases: Vec<f32>) -> PyResult<()> {
        if handle >= self.head { return Err(PyIndexError::new_err("Invalid Handle")); }
        if phases.len() != self.dimension { return Err(PyValueError::new_err("Phase length mismatch")); }
        
        let offset = handle * self.dimension * 2;
        let p_data = self.buffer.as_mut_ptr();
        unsafe {
            for (i, &theta) in phases.iter().enumerate() {
                let (sin_t, cos_t) = theta.sin_cos();
                *p_data.add(offset + 2 * i) = cos_t;         
                *p_data.add(offset + 2 * i + 1) = sin_t;     
            }
        }
        Ok(())
    }

    pub fn bind(&mut self, h1: usize, h2: usize, out_handle: usize) -> PyResult<()> {
        unsafe { self.bind_simd(h1, h2, out_handle)?; }
        // Track thermodynamic cost: each bind adds ln(2) entropy
        if out_handle < self.capacity {
            self.op_counts[out_handle].0 += 1;
        }
        Ok(())
    }

    pub fn bundle(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        let fan_in = handles.len() as u32;
        unsafe { self.bundle_simd(handles, out_handle)?; }
        // Track thermodynamic cost: bundle of k handles adds k×ln(2) entropy
        if out_handle < self.capacity {
            self.op_counts[out_handle].1 += fan_in;
        }
        Ok(())
    }

    pub fn compute_correlation(&self, h1: usize, h2: usize) -> PyResult<f32> {
        self.validate_handles(&[h1, h2])?;
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_ptr();
        let mut sum_real = 0.0;

        unsafe {
            let mut p1 = p_buf.add(h1 * floats);
            let mut p2 = p_buf.add(h2 * floats);
            let end = p1.add(floats);

            while p1 < end {
                let re1 = *p1; let im1 = *p1.add(1);
                let re2 = *p2; let im2 = *p2.add(1);

                sum_real += re1 * re2 + im1 * im2;

                p1 = p1.add(2);
                p2 = p2.add(2);
            }
        }
        Ok(sum_real / (self.dimension as f32))
    }

    // ── Meta-Grammar Emergence: Dimension & Operator Expansion ───

    /// Returns the current hypervector dimension.
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the current number of allocated handles.
    pub fn get_head(&self) -> usize {
        self.head
    }

    /// Returns the arena capacity.
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    /// Deterministically expand the arena dimension from d to new_dim (new_dim > d).
    ///
    /// For each existing vector, the original d components are preserved.
    /// New components [d, new_dim) are derived deterministically:
    ///   new[j] = conjugate(existing[j % d])  (phase negation = algebraic reflection)
    ///
    /// This is O(head × new_dim) — linear in the number of live vectors.
    pub fn expand_dimension(&mut self, new_dim: usize) -> PyResult<()> {
        if new_dim <= self.dimension {
            return Err(PyValueError::new_err(
                "New dimension must be strictly greater than current dimension."
            ));
        }

        let old_dim = self.dimension;
        let old_floats = old_dim * 2;
        let new_floats = new_dim * 2;
        let total_new = self.capacity * new_floats;

        let mut new_buffer: Vec<f32> = Vec::with_capacity(total_new);
        unsafe {
            ptr::write_bytes(new_buffer.as_mut_ptr(), 0, total_new);
            new_buffer.set_len(total_new);
        }

        // Copy and extend each live vector
        for h in 0..self.head {
            let old_offset = h * old_floats;
            let new_offset = h * new_floats;

            // Copy original components
            unsafe {
                ptr::copy_nonoverlapping(
                    self.buffer.as_ptr().add(old_offset),
                    new_buffer.as_mut_ptr().add(new_offset),
                    old_floats,
                );
            }

            // Deterministic extension: conjugate reflection of existing components
            // new[j] for j in [old_dim, new_dim): (re_{j%d}, -im_{j%d})
            for j in old_dim..new_dim {
                let src_j = j % old_dim;
                unsafe {
                    let src_re = *self.buffer.as_ptr().add(old_offset + src_j * 2);
                    let src_im = *self.buffer.as_ptr().add(old_offset + src_j * 2 + 1);
                    *new_buffer.as_mut_ptr().add(new_offset + j * 2) = src_re;
                    *new_buffer.as_mut_ptr().add(new_offset + j * 2 + 1) = -src_im;
                }
            }
        }

        self.buffer = new_buffer;
        self.dimension = new_dim;
        Ok(())
    }

    /// Algebraic fusion operator: out = bind(h_bind, bundle(handles))
    ///
    /// This is a new topological grammar rule synthesized from ⊗ and ⊕:
    ///   fuse(A, {B₁,...,Bₖ}) = A ⊗ (B₁ ⊕ ... ⊕ Bₖ)
    ///
    /// Executed as a single fused pass to avoid intermediate allocation.
    /// The bundle is computed into a temporary region, then bound with h_bind.
    pub fn bind_bundle_fusion(
        &mut self, h_bind: usize, handles_bundle: Vec<usize>, out_handle: usize,
    ) -> PyResult<()> {
        if handles_bundle.is_empty() {
            return Err(PyValueError::new_err("Bundle handles must be non-empty."));
        }
        self.validate_handles(&[h_bind, out_handle])?;
        self.validate_handles(&handles_bundle)?;

        let floats = self.dimension * 2;

        // Step 1: Compute bundle into out_handle (temporary use)
        unsafe { self.bundle_simd(handles_bundle, out_handle)?; }

        // Step 2: Bind h_bind with bundled result in-place
        // We need a temporary copy of the bundle result
        let bundle_copy: Vec<f32> = self.buffer
            [out_handle * floats..(out_handle + 1) * floats]
            .to_vec();

        // Perform complex multiply: out = h_bind ⊗ bundle_result
        let p_buf = self.buffer.as_mut_ptr();
        unsafe {
            let p_bind = p_buf.add(h_bind * floats);
            let p_out = p_buf.add(out_handle * floats);

            for i in 0..self.dimension {
                let re1 = *p_bind.add(i * 2);
                let im1 = *p_bind.add(i * 2 + 1);
                let re2 = bundle_copy[i * 2];
                let im2 = bundle_copy[i * 2 + 1];
                *p_out.add(i * 2) = re1 * re2 - im1 * im2;
                *p_out.add(i * 2 + 1) = re1 * im2 + im1 * re2;
            }
        }

        // Track op costs: fusion = 1 bind + 1 bundle
        self.op_counts[out_handle].0 += 1;
        self.op_counts[out_handle].1 += 1;
        Ok(())
    }

    // ── Topological Thermodynamics: Operation Cost Tracking ──────

    /// Record a bind operation on the output handle.
    pub fn record_bind_cost(&mut self, handle: usize) -> PyResult<()> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        self.op_counts[handle].0 += 1;
        Ok(())
    }

    /// Record a bundle operation on the output handle.
    pub fn record_bundle_cost(&mut self, handle: usize, fan_in: usize) -> PyResult<()> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        self.op_counts[handle].1 += fan_in as u32;
        Ok(())
    }

    /// Retrieve the thermodynamic cost (bind_count, bundle_count) for a handle.
    pub fn get_op_counts(&self, handle: usize) -> PyResult<(u32, u32)> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        Ok(self.op_counts[handle])
    }

    /// Compute the thermodynamic entropy penalty for a handle.
    ///
    /// S(h) = bind_count × ln(2) + bundle_count × ln(bundle_fan_in)
    ///
    /// This is a deterministic, O(1) complexity metric.
    pub fn compute_entropy(&self, handle: usize) -> PyResult<f32> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        let (binds, bundles) = self.op_counts[handle];
        let ln2: f32 = 2.0_f32.ln();
        // Each bind costs ln(2), each bundle fan-in costs ln(fan_in) ≈ ln(2) per unit
        let entropy = (binds as f32) * ln2 + (bundles as f32) * ln2;
        Ok(entropy)
    }
}

impl FhrrArena {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn bind_simd(&mut self, h1: usize, h2: usize, out_handle: usize) -> PyResult<()> {
        self.validate_handles(&[h1, h2, out_handle])?;
        
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        
        let mut p1 = p_buf.add(h1 * floats);
        let mut p2 = p_buf.add(h2 * floats);
        let mut p_out = p_buf.add(out_handle * floats);
        let end = p1.add(floats);
        
        while p1 < end.sub(7) {
            let a = _mm256_loadu_ps(p1);
            let b = _mm256_loadu_ps(p2);

            let a_real = _mm256_moveldup_ps(a);
            let a_imag = _mm256_movehdup_ps(a);

            let b_shuf = _mm256_shuffle_ps(b, b, 0xb1);

            let res1 = _mm256_mul_ps(a_real, b);
            let res2 = _mm256_mul_ps(a_imag, b_shuf);

            let result = _mm256_addsub_ps(res1, res2);

            _mm256_storeu_ps(p_out, result);
            
            p1 = p1.add(8);
            p2 = p2.add(8);
            p_out = p_out.add(8);
        }
        
        while p1 < end {
            let re1 = *p1; let im1 = *p1.add(1);
            let re2 = *p2; let im2 = *p2.add(1);
            *p_out = re1 * re2 - im1 * im2;
            *p_out.add(1) = re1 * im2 + im1 * re2;
            p1 = p1.add(2);
            p2 = p2.add(2);
            p_out = p_out.add(2);
        }

        Ok(())
    }

    #[target_feature(enable = "avx2")]
    unsafe fn bundle_simd(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        if handles.is_empty() { return Ok(()); }
        self.validate_handles(&handles)?;
        self.validate_handles(&[out_handle])?;
        
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        let mut p_out = p_buf.add(out_handle * floats);
        
        ptr::write_bytes(p_out, 0, floats);

        for h in handles {
            let mut p_curr = p_buf.add(h * floats);
            let mut p_out_curr = p_out;
            let end = p_curr.add(floats);

            while p_curr < end.sub(7) {
                let v_out = _mm256_loadu_ps(p_out_curr);
                let v_curr = _mm256_loadu_ps(p_curr);
                let summed = _mm256_add_ps(v_out, v_curr);
                _mm256_storeu_ps(p_out_curr, summed);
                p_curr = p_curr.add(8);
                p_out_curr = p_out_curr.add(8);
            }

            while p_curr < end {
                *p_out_curr += *p_curr;
                p_curr = p_curr.add(1);
                p_out_curr = p_out_curr.add(1);
            }
        }

        self.normalize_simd(out_handle)?;
        Ok(())
    }

    #[target_feature(enable = "avx2")]
    unsafe fn normalize_simd(&mut self, handle: usize) -> PyResult<()> {
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        let mut p_vec = p_buf.add(handle * floats);
        let end = p_vec.add(floats);

        while p_vec < end.sub(7) {
            let v = _mm256_loadu_ps(p_vec);
            let sq = _mm256_mul_ps(v, v);
            
            let shuf = _mm256_shuffle_ps(sq, sq, 0xb1);
            let mags_sq = _mm256_add_ps(sq, shuf);
            
            let inv_mags = _mm256_rsqrt_ps(mags_sq);
            let normalized = _mm256_mul_ps(v, inv_mags);
            
            _mm256_storeu_ps(p_vec, normalized);
            p_vec = p_vec.add(8);
        }

        while p_vec < end {
            let re = *p_vec;
            let im = *p_vec.add(1);
            let mag_inv = 1.0 / (re*re + im*im).sqrt();
            if mag_inv.is_finite() {
                *p_vec *= mag_inv;
                *p_vec.add(1) *= mag_inv;
            }
            p_vec = p_vec.add(2);
        }
        Ok(())
    }

    fn validate_handles(&self, handles: &[usize]) -> PyResult<()> {
        for &h in handles {
            if h >= self.head {
                return Err(PyIndexError::new_err(format!("Handle {} out of bounds.", h)));
            }
        }
        Ok(())
    }
}

#[pymodule]
fn hdc_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FhrrArena>()?;
    Ok(())
}
