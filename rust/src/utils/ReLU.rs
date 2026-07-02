use crate::utils::matrix::Matrix;

pub fn forward(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn backward(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/*
Element-wise ReLU: max(0, x)
*/
pub fn relu(x: &Matrix) -> Matrix {
    x.map(|v| if v > 0.0 { v } else { 0.0 })
}

/*
ReLU gradient: 1 if x > 0, else 0
*/
pub fn relu_grad(x: &Matrix) -> Matrix {
    x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
}
