use serde::{Deserialize, Serialize};

/// Mathematical Abstract Syntax Tree (AST) node representing continuous operators and expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Expr {
    /// Scalar numerical constant, e.g. 3.14159, -0.04
    Constant(f64),
    /// State variable indexed by dimension, e.g. y[0], y[1], y[2]
    State(usize),
    /// Independent time variable t
    Time,
    /// Named physical parameter or constant, e.g. "omega", "mu", "k"
    Param(String),
    /// Unary negation: -x
    Neg(Box<Expr>),
    /// Addition: lhs + rhs
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: lhs - rhs
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: lhs * rhs
    Mul(Box<Expr>, Box<Expr>),
    /// Division: lhs / rhs
    Div(Box<Expr>, Box<Expr>),
    /// Power: base ^ exponent (e.g. base^2, base^0.5)
    Pow(Box<Expr>, Box<Expr>),
    /// Sine: sin(x)
    Sin(Box<Expr>),
    /// Cosine: cos(x)
    Cos(Box<Expr>),
    /// Tangent: tan(x)
    Tan(Box<Expr>),
    /// Exponential: exp(x)
    Exp(Box<Expr>),
    /// Natural logarithm: ln(x)
    Ln(Box<Expr>),
    /// Square root: sqrt(x)
    Sqrt(Box<Expr>),
    /// Absolute value: abs(x)
    Abs(Box<Expr>),
    /// Hyperbolic tangent: tanh(x)
    Tanh(Box<Expr>),
    /// Hyperbolic sine: sinh(x)
    Sinh(Box<Expr>),
    /// Hyperbolic cosine: cosh(x)
    Cosh(Box<Expr>),
    /// First spatial derivative: u_x
    SpatialDerivative1D(Box<Expr>),
    /// Second spatial derivative (Laplacian): u_xx
    Laplacian1D(Box<Expr>),
}

impl Expr {
    pub fn constant(val: f64) -> Self {
        Expr::Constant(val)
    }

    pub fn state(idx: usize) -> Self {
        Expr::State(idx)
    }

    pub fn time() -> Self {
        Expr::Time
    }

    pub fn param(name: impl Into<String>) -> Self {
        Expr::Param(name.into())
    }

    pub fn neg(expr: Expr) -> Self {
        Expr::Neg(Box::new(expr))
    }

    pub fn add(lhs: Expr, rhs: Expr) -> Self {
        Expr::Add(Box::new(lhs), Box::new(rhs))
    }

    pub fn sub(lhs: Expr, rhs: Expr) -> Self {
        Expr::Sub(Box::new(lhs), Box::new(rhs))
    }

    pub fn mul(lhs: Expr, rhs: Expr) -> Self {
        Expr::Mul(Box::new(lhs), Box::new(rhs))
    }

    pub fn div(lhs: Expr, rhs: Expr) -> Self {
        Expr::Div(Box::new(lhs), Box::new(rhs))
    }

    pub fn pow(base: Expr, exponent: Expr) -> Self {
        Expr::Pow(Box::new(base), Box::new(exponent))
    }

    pub fn sin(expr: Expr) -> Self {
        Expr::Sin(Box::new(expr))
    }

    pub fn cos(expr: Expr) -> Self {
        Expr::Cos(Box::new(expr))
    }

    pub fn exp(expr: Expr) -> Self {
        Expr::Exp(Box::new(expr))
    }

    pub fn ln(expr: Expr) -> Self {
        Expr::Ln(Box::new(expr))
    }

    pub fn sqrt(expr: Expr) -> Self {
        Expr::Sqrt(Box::new(expr))
    }

    pub fn abs(expr: Expr) -> Self {
        Expr::Abs(Box::new(expr))
    }

    pub fn tanh(expr: Expr) -> Self {
        Expr::Tanh(Box::new(expr))
    }

    pub fn spatial_deriv_1d(expr: Expr) -> Self {
        Expr::SpatialDerivative1D(Box::new(expr))
    }

    pub fn laplacian_1d(expr: Expr) -> Self {
        Expr::Laplacian1D(Box::new(expr))
    }
}

/// Represents a system of Ordinary Differential Equations (ODEs):
/// dy_i / dt = equations[i](t, y, params)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OdeSystem {
    pub name: String,
    pub dimension: usize,
    pub is_pde: bool,
    /// System derivative expressions for each dimension: dy_0/dt, dy_1/dt, ...
    pub equations: Vec<Expr>,
    /// System parameters with default values, e.g. [("omega", 1.0)]
    pub params: Vec<(String, f64)>,
    /// Default initial conditions y(0)
    pub default_y0: Vec<f64>,
    /// Default time span (t_start, t_end)
    pub default_t_span: (f64, f64),
    /// Default time step dt
    pub default_dt: f64,
}

impl OdeSystem {
    pub fn new(name: impl Into<String>, equations: Vec<Expr>) -> Self {
        let dim = equations.len();
        Self {
            name: name.into(),
            dimension: dim,
            is_pde: false,
            equations,
            params: Vec::new(),
            default_y0: vec![0.0; dim],
            default_t_span: (0.0, 10.0),
            default_dt: 0.01,
        }
    }

    pub fn with_is_pde(mut self, is_pde: bool) -> Self {
        self.is_pde = is_pde;
        self
    }

    /// Factory for the classical 2D Harmonic Oscillator:
    /// y0' = y1
    /// y1' = -omega^2 * y0
    pub fn harmonic_oscillator(omega: f64) -> Self {
        let mut sys = Self::new(
            "HarmonicOscillator",
            vec![
                Expr::state(1),
                Expr::mul(
                    Expr::neg(Expr::constant(omega * omega)),
                    Expr::state(0),
                ),
            ],
        );
        sys.default_y0 = vec![1.0, 0.0];
        sys.default_t_span = (0.0, 10.0);
        sys.default_dt = 0.01;
        sys
    }

    /// Factory for the Robertson chemical kinetics problem (extreme stiffness benchmark):
    /// y0' = -0.04 * y0 + 1e4 * y1 * y2
    /// y1' =  0.04 * y0 - 1e4 * y1 * y2 - 3e7 * y1^2
    /// y2' =  3e7 * y1^2
    pub fn robertson() -> Self {
        let eq0 = Expr::add(
            Expr::mul(Expr::constant(-0.04), Expr::state(0)),
            Expr::mul(
                Expr::constant(1e4),
                Expr::mul(Expr::state(1), Expr::state(2)),
            ),
        );

        let eq1 = Expr::sub(
            Expr::sub(
                Expr::mul(Expr::constant(0.04), Expr::state(0)),
                Expr::mul(
                    Expr::constant(1e4),
                    Expr::mul(Expr::state(1), Expr::state(2)),
                ),
            ),
            Expr::mul(
                Expr::constant(3e7),
                Expr::pow(Expr::state(1), Expr::constant(2.0)),
            ),
        );

        let eq2 = Expr::mul(
            Expr::constant(3e7),
            Expr::pow(Expr::state(1), Expr::constant(2.0)),
        );

        let mut sys = Self::new("Robertson", vec![eq0, eq1, eq2]);
        sys.default_y0 = vec![1.0, 0.0, 0.0];
        sys.default_t_span = (0.0, 10.0);
        sys.default_dt = 1e-4;
        sys
    }

    /// Factory for the Van der Pol oscillator (relaxation oscillations, boundary layer stiffness):
    /// y0' = y1
    /// y1' = mu * (1 - y0^2) * y1 - y0
    pub fn vanderpol(mu: f64) -> Self {
        let eq0 = Expr::state(1);
        let eq1 = Expr::sub(
            Expr::mul(
                Expr::mul(
                    Expr::constant(mu),
                    Expr::sub(
                        Expr::constant(1.0),
                        Expr::pow(Expr::state(0), Expr::constant(2.0)),
                    ),
                ),
                Expr::state(1),
            ),
            Expr::state(0),
        );

        let mut sys = Self::new(format!("VanDerPol_mu{mu}"), vec![eq0, eq1]);
        sys.default_y0 = vec![2.0, 0.0];
        sys.default_t_span = (0.0, 20.0);
        sys.default_dt = 0.005;
        sys
    }
}
