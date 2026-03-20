use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIndexError};
use core::arch::x86_64::*;
use std::ptr;

/// Zero-cost abstraction Memory Arena for complex FHRR hypervectors built for PyO3.
#[pyclass]
pub struct FhrrArena {
    buffer: Vec<f32>,
    dimension: usize,
    capacity: usize,
    head: usize,
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

        Self {
            buffer,
            dimension,
            capacity,
            head: 0,
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
        unsafe { self.bind_simd(h1, h2, out_handle) }
    }

    pub fn bundle(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        unsafe { self.bundle_simd(handles, out_handle) }
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
