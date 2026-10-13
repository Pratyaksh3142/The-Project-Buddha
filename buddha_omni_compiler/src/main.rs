use buddha_omni_compiler::ast::OdeSystem;
use buddha_omni_compiler::engine::OmniEngine;
use buddha_omni_compiler::wrapper::{generate_cpp_source, SimConfig};
use std::env;
use std::fs;

fn print_banner() {
    println!("================================================================================");
    println!("           PROJECT BUDDHA: OMNI-COMPILER & PRATYAKSH EXECUTION ENGINE           ");
    println!("     High-Speed AST -> C++ Transpiler Wrapped in Explicit TVD Integration       ");
    println!("================================================================================");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let engine = OmniEngine::new();

    if args.len() < 2 {
        print_banner();
        println!("Usage: buddha_omni_compiler <command> [args...]");
        println!();
        println!("Commands:");
        println!("  test-harmonic       Run Harmonic Oscillator verification test");
        println!("  test-vanderpol      Run Stiff Van der Pol (mu=100) Showdown: Pratyaksh vs RK4");
        println!("  test-robertson      Run Robertson Chemical Kinetics Showdown: Pratyaksh vs RK4");
        println!("  emit <file.json>    Transpile ODE system JSON to C++ source");
        println!("  run <file.json>     Compile and execute ODE system JSON");
        println!("  showdown <file.json> Run side-by-side Pratyaksh vs RK4 on system JSON");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "test-harmonic" => {
            print_banner();
            println!("\n[1/3] Benchmarking 2D Harmonic Oscillator (Exact Conservation Test)...");
            let sys = OdeSystem::harmonic_oscillator(1.0);
            let mut cfg = SimConfig::default();
            cfg.t_start = 0.0;
            cfg.t_end = 20.0;
            cfg.dt = 0.01;
            cfg.record_trajectory = false;

            println!(" -> Emitting C++ and Compiling via clang++ -O3...");
            let res = engine.compile_and_run(&sys, &cfg)?;
            println!(" -> Status:                {}", res.status);
            println!(" -> Integrator:            {}", res.integrator);
            println!(" -> Steps Taken:           {}", res.steps_taken);
            println!(" -> Final Time:            {:.4}", res.final_time);
            println!(" -> Final State:           {:?}", res.final_state);
            println!(" -> Max Denominator:       {:.6}", res.max_denominator);
            println!(" -> Execution Time:        {:.6} s", res.execution_time_seconds);

            // Verify analytical property: at t = 20.0 (near 20 rad = 3.18 periods)
            // Energy E = 0.5 * (y0^2 + y1^2) should be conserved = 0.5
            let energy = 0.5 * (res.final_state[0].powi(2) + res.final_state[1].powi(2));
            println!(" -> Conserved Energy:      {:.8} (Initial: 0.50000000)", energy);
            if (energy - 0.5_f64).abs() < 1e-3 {
                println!(" [PASSED] Harmonic Oscillator verified within tight tolerances!\n");
            } else {
                println!(" [WARNING] Energy drift detected: {}\n", energy);
            }
        }

        "test-vanderpol" => {
            print_banner();
            println!("\n[2/3] Benchmarking Van der Pol Relaxation Oscillator (mu = 100)...");
            let sys = OdeSystem::vanderpol(100.0);
            let mut cfg = SimConfig::default();
            cfg.t_start = 0.0;
            cfg.t_end = 10.0;
            cfg.dt = 0.001; // Step size where standard RK4 faces steep gradient shocks

            println!(" -> Running Showdown: Pratyaksh-II vs Classical RK4...");
            let (res_prat, res_rk4) = engine.run_showdown(&sys, &cfg)?;

            println!("--------------------------------------------------------------------------------");
            println!("{:<20} | {:<15} | {:<15} | {:<15}", "Solver", "Status", "Steps Taken", "Max Denom / NaN");
            println!("--------------------------------------------------------------------------------");
            println!(
                "{:<20} | {:<15} | {:<15} | {:.6}",
                res_prat.integrator, res_prat.status, res_prat.steps_taken, res_prat.max_denominator
            );
            println!(
                "{:<20} | {:<15} | {:<15} | {:.6}",
                res_rk4.integrator, res_rk4.status, res_rk4.steps_taken, res_rk4.max_denominator
            );
            println!("--------------------------------------------------------------------------------");
            println!(" -> Pratyaksh Final State: {:?}", res_prat.final_state);
            println!(" -> RK4 Final State:       {:?}", res_rk4.final_state);
            println!(" [PASSED] Van der Pol showdown executed successfully.\n");
        }

        "test-robertson" => {
            print_banner();
            println!("\n[3/3] Benchmarking Robertson Chemical Kinetics (Extreme Stiffness 10^7)...");
            let sys = OdeSystem::robertson();
            let mut cfg = SimConfig::default();
            cfg.t_start = 0.0;
            cfg.t_end = 1.0;
            cfg.dt = 1e-4;

            println!(" -> Running Showdown: Pratyaksh-II vs Classical RK4...");
            let (res_prat, res_rk4) = engine.run_showdown(&sys, &cfg)?;

            println!("--------------------------------------------------------------------------------");
            println!("{:<20} | {:<15} | {:<15} | {:<15}", "Solver", "Status", "Steps Taken", "Max Denom / NaN");
            println!("--------------------------------------------------------------------------------");
            println!(
                "{:<20} | {:<15} | {:<15} | {:.6}",
                res_prat.integrator, res_prat.status, res_prat.steps_taken, res_prat.max_denominator
            );
            println!(
                "{:<20} | {:<15} | {:<15} | {:.6}",
                res_rk4.integrator, res_rk4.status, res_rk4.steps_taken, res_rk4.max_denominator
            );
            println!("--------------------------------------------------------------------------------");
            println!(" -> Pratyaksh Final State: {:?}", res_prat.final_state);
            println!(" -> RK4 Final State:       {:?}", res_rk4.final_state);
            println!(" [PASSED] Robertson benchmark executed successfully.\n");
        }

        "emit" => {
            if args.len() < 3 {
                eprintln!("Error: Missing json file path");
                std::process::exit(1);
            }
            let path = &args[2];
            let data = fs::read_to_string(path)?;
            let sys: OdeSystem = serde_json::from_str(&data)?;
            let cfg = SimConfig::default();
            let cpp = generate_cpp_source(&sys, &cfg);
            println!("{}", cpp);
        }

        "run" => {
            if args.len() < 3 {
                eprintln!("Error: Missing json file path");
                std::process::exit(1);
            }
            let path = &args[2];
            let data = fs::read_to_string(path)?;
            let sys: OdeSystem = serde_json::from_str(&data)?;
            let cfg = SimConfig::default();
            let res = engine.compile_and_run(&sys, &cfg)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }

        "showdown" => {
            if args.len() < 3 {
                eprintln!("Error: Missing json file path");
                std::process::exit(1);
            }
            let path = &args[2];
            let data = fs::read_to_string(path)?;
            let sys: OdeSystem = serde_json::from_str(&data)?;
            let cfg = SimConfig::default();
            let (res_prat, res_rk4) = engine.run_showdown(&sys, &cfg)?;
            println!("=== PRATYAKSH-II RESULT ===");
            println!("{}", serde_json::to_string_pretty(&res_prat)?);
            println!("\n=== CLASSICAL RK4 RESULT ===");
            println!("{}", serde_json::to_string_pretty(&res_rk4)?);
        }

        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }

    Ok(())
}
