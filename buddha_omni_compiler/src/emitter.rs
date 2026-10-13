use crate::ast::{Expr, OdeSystem};

/// Converts an AST `Expr` into a valid, optimized C++ expression string.
pub fn emit_expr_cpp(expr: &Expr, is_pde: bool, offset: i32) -> String {
    match expr {
        Expr::Constant(val) => {
            if val.is_nan() {
                "std::numeric_limits<double>::quiet_NaN()".to_string()
            } else if val.is_infinite() {
                if *val > 0.0 {
                    "std::numeric_limits<double>::infinity()".to_string()
                } else {
                    "-std::numeric_limits<double>::infinity()".to_string()
                }
            } else {
                format!("{:.16e}", val)
            }
        }
        Expr::State(idx) => {
            if is_pde {
                if offset == 0 {
                    "y[i]".to_string()
                } else if offset > 0 {
                    format!("y[i + {}]", offset)
                } else {
                    format!("y[i - {}]", -offset)
                }
            } else {
                format!("y[{}]", idx)
            }
        }
        Expr::Time => "t".to_string(),
        Expr::Param(name) => format!("params.{}", name),
        Expr::Neg(inner) => format!("(-{})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Add(lhs, rhs) => format!("({} + {})", emit_expr_cpp(lhs, is_pde, offset), emit_expr_cpp(rhs, is_pde, offset)),
        Expr::Sub(lhs, rhs) => format!("({} - {})", emit_expr_cpp(lhs, is_pde, offset), emit_expr_cpp(rhs, is_pde, offset)),
        Expr::Mul(lhs, rhs) => format!("({} * {})", emit_expr_cpp(lhs, is_pde, offset), emit_expr_cpp(rhs, is_pde, offset)),
        Expr::Div(lhs, rhs) => format!("({} / {})", emit_expr_cpp(lhs, is_pde, offset), emit_expr_cpp(rhs, is_pde, offset)),
        Expr::Pow(base, exp) => format!("std::pow({}, {})", emit_expr_cpp(base, is_pde, offset), emit_expr_cpp(exp, is_pde, offset)),
        Expr::Sin(inner) => format!("std::sin({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Cos(inner) => format!("std::cos({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Tan(inner) => format!("std::tan({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Exp(inner) => format!("std::exp({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Ln(inner) => format!("std::log({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Sqrt(inner) => format!("std::sqrt({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Abs(inner) => format!("std::fabs({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Tanh(inner) => format!("std::tanh({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Sinh(inner) => format!("std::sinh({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::Cosh(inner) => format!("std::cosh({})", emit_expr_cpp(inner, is_pde, offset)),
        Expr::SpatialDerivative1D(inner) => {
            let right = emit_expr_cpp(inner, is_pde, offset + 1);
            let left = emit_expr_cpp(inner, is_pde, offset - 1);
            format!("(({}) - ({})) / (2.0 * params.dx)", right, left)
        }
        Expr::Laplacian1D(inner) => {
            let right = emit_expr_cpp(inner, is_pde, offset + 1);
            let center = emit_expr_cpp(inner, is_pde, offset);
            let left = emit_expr_cpp(inner, is_pde, offset - 1);
            format!("(({}) - 2.0 * ({}) + ({})) / (params.dx * params.dx)", right, center, left)
        }
    }
}

pub fn emit_params_struct(system: &OdeSystem) -> String {
    let mut out = String::new();
    out.push_str("struct SystemParams {\n");
    let mut has_dx = false;
    for (name, val) in &system.params {
        if name == "dx" { has_dx = true; }
        out.push_str(&format!("    double {} = {:.16e};\n", name, val));
    }
    if !has_dx {
        out.push_str("    double dx = 1.0;\n");
    }
    out.push_str("};\n");
    out
}

pub fn emit_derivatives_function(system: &OdeSystem) -> String {
    let mut out = String::new();
    out.push_str("// System RHS Evaluator: dy/dt = f(t, y)\n");
    out.push_str("inline void compute_derivatives(\n");
    out.push_str("    double t,\n");
    out.push_str("    const std::vector<double>& y,\n");
    out.push_str("    std::vector<double>& dy,\n");
    out.push_str("    const SystemParams& params\n");
    out.push_str(") {\n");

    if system.is_pde {
        let n_eqs = system.equations.len();
        if n_eqs == 1 {
            let expr_str = emit_expr_cpp(&system.equations[0], true, 0);
            out.push_str("    if (y.size() >= 3) {\n");
            out.push_str("        for (size_t i = 1; i < y.size() - 1; ++i) {\n");
            out.push_str(&format!("            dy[i] = {};\n", expr_str));
            out.push_str("        }\n");
            out.push_str("        dy[0] = dy[1];\n");
            out.push_str("        dy[y.size() - 1] = dy[y.size() - 2];\n");
            out.push_str("    }\n");
        } else {
            out.push_str(&format!("    size_t n_grid = y.size() / {};\n", n_eqs));
            out.push_str("    if (n_grid >= 3) {\n");
            for (comp, eq) in system.equations.iter().enumerate() {
                let expr_str = emit_expr_cpp(eq, true, 0);
                out.push_str("        for (size_t i = 1; i < n_grid - 1; ++i) {\n");
                out.push_str(&format!("            dy[{} * n_grid + i] = {};\n", comp, expr_str));
                out.push_str("        }\n");
                out.push_str(&format!("        dy[{} * n_grid] = dy[{} * n_grid + 1];\n", comp, comp));
                out.push_str(&format!("        dy[{} * n_grid + n_grid - 1] = dy[{} * n_grid + n_grid - 2];\n", comp, comp));
            }
            out.push_str("    }\n");
        }
    } else {
        for (i, eq) in system.equations.iter().enumerate() {
            let expr_str = emit_expr_cpp(eq, system.is_pde, 0);
            out.push_str(&format!("    dy[{}] = {};\n", i, expr_str));
        }
    }

    out.push_str("}\n");
    out
}
