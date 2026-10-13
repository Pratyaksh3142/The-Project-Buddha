use crate::ast::OdeSystem;
use crate::wrapper::{generate_cpp_source, SimConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub status: String,
    pub integrator: String,
    pub final_time: f64,
    pub diverged_at_t: f64, // 0 if not diverged
    pub steps_taken: usize,
    pub throttled_steps: usize,
    pub max_denominator: f64,
    pub execution_time_seconds: f64,
    pub final_state: Vec<f64>,
}

#[repr(C)]
struct SimulationResultC {
    status: std::os::raw::c_int,
    final_time: std::os::raw::c_double,
    steps_taken: usize,
    throttled_steps: usize,
    max_denominator: std::os::raw::c_double,
    execution_time_sec: std::os::raw::c_double,
}

pub struct OmniEngine {
    build_dir: String,
}

impl OmniEngine {
    pub fn new() -> Self {
        let build_dir = "./build".to_string();
        fs::create_dir_all(&build_dir).expect("Failed to create build directory");
        Self { build_dir }
    }

    /// Compiles the AST to a shared library and executes it via FFI for a batch of initial conditions.
    pub fn compile_and_run_batch(
        &self,
        system: &OdeSystem,
        config: &SimConfig,
        y0_batch: &[Vec<f64>],
    ) -> Result<Vec<SimulationResult>, String> {
        let code = generate_cpp_source(system, config);
        let cpp_path = format!("{}/sim_{}.cpp", self.build_dir, system.name);
        
        // Define shared library name based on OS
        let lib_ext = std::env::consts::DLL_EXTENSION;
        let lib_path = format!("{}/libsim_{}.{}", self.build_dir, system.name, lib_ext);

        fs::write(&cpp_path, code).map_err(|e| format!("Failed to write C++ source: {}", e))?;

        // 1. Compile as Shared Library
        let compile_status = Command::new("clang++")
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-shared")
            .arg("-fPIC")
            .arg(&cpp_path)
            .arg("-o")
            .arg(&lib_path)
            .status()
            .map_err(|e| format!("Failed to invoke clang++: {}", e))?;

        if !compile_status.success() {
            return Err("C++ Compilation failed.".to_string());
        }

        // 2. Load Shared Library and Execute via FFI
        let num_ics = y0_batch.len();
        let dim = system.dimension;
        
        let mut flat_y0 = Vec::with_capacity(num_ics * dim);
        for y0 in y0_batch {
            if y0.len() != dim {
                return Err(format!("Initial condition length {} does not match system dimension {}", y0.len(), dim));
            }
            flat_y0.extend_from_slice(y0);
        }

        let mut final_y_batch = vec![0.0; num_ics * dim];
        let mut results_c = Vec::with_capacity(num_ics);
        for _ in 0..num_ics {
            results_c.push(SimulationResultC {
                status: 0,
                final_time: 0.0,
                steps_taken: 0,
                throttled_steps: 0,
                max_denominator: 0.0,
                execution_time_sec: 0.0,
            });
        }

        let integrator_choice = match config.integrator.as_str() {
            "pratyaksh_b" => 0,
            "pratyaksh_a" => 1,
            "rk4" => 2,
            _ => 0,
        };

        unsafe {
            let lib = libloading::Library::new(Path::new(&lib_path))
                .map_err(|e| format!("Failed to load shared library: {}", e))?;
            
            type RunBatchFunc = unsafe extern "C" fn(
                usize, *const f64, usize, f64, f64, f64, f64, std::os::raw::c_int, *mut f64, *mut SimulationResultC
            );

            let func: libloading::Symbol<RunBatchFunc> = lib.get(b"run_simulation_batch\0")
                .map_err(|e| format!("Failed to find symbol 'run_simulation_batch': {}", e))?;

            func(
                num_ics,
                flat_y0.as_ptr(),
                dim,
                config.t_start,
                config.t_end,
                config.dt,
                config.beta,
                integrator_choice,
                final_y_batch.as_mut_ptr(),
                results_c.as_mut_ptr(),
            );
        }

        // 3. Convert C structs back to Rust Output
        let mut final_results = Vec::new();
        for b in 0..num_ics {
            let rc = &results_c[b];
            
            let status_str = match rc.status {
                0 => "SUCCESS",
                1 => "DIVERGED_NAN",
                2 => "ABORTED_UNPHYSICAL",
                _ => "UNKNOWN",
            };

            let mut final_state = Vec::with_capacity(dim);
            for d in 0..dim {
                final_state.push(final_y_batch[b * dim + d]);
            }

            final_results.push(SimulationResult {
                status: status_str.to_string(),
                integrator: config.integrator.clone(),
                final_time: rc.final_time,
                diverged_at_t: if rc.status == 1 { rc.final_time } else { 0.0 },
                steps_taken: rc.steps_taken,
                throttled_steps: rc.throttled_steps,
                max_denominator: rc.max_denominator,
                execution_time_seconds: rc.execution_time_sec,
                final_state,
            });
        }

        Ok(final_results)
    }

    /// Backwards compatibility method for running a single trajectory
    pub fn compile_and_run(&self, system: &OdeSystem, config: &SimConfig) -> Result<SimulationResult, String> {
        let batch = vec![system.default_y0.clone()];
        let mut results = self.compile_and_run_batch(system, config, &batch)?;
        Ok(results.remove(0))
    }

    pub fn run_showdown(&self, system: &OdeSystem, config: &SimConfig) -> Result<(SimulationResult, SimulationResult), String> {
        let mut config_rk4 = config.clone();
        config_rk4.integrator = "rk4".to_string();

        let res_rk4 = self.compile_and_run(system, &config_rk4)?;

        let mut config_prat = config_rk4.clone();
        config_prat.integrator = "pratyaksh_b".to_string();

        let res_prat = self.compile_and_run(system, &config_prat)?;

        Ok((res_prat, res_rk4))
    }
}
