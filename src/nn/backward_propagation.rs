use crate::utils::matrix::Matrix;
use crate::utils::ReLU;
use crate::nn::forward_propagation::Cache;

pub struct Grads {
    pub dW1: Matrix,
    pub db1: Matrix,
    pub dW2: Matrix,
    pub db2: Matrix,
}

pub fn backward_propagation(
    _W1: &Matrix,
    W2: &Matrix,
    cache: &Cache,
    X: &Matrix,
    Y: &Matrix,
) -> Grads {
    /*
    Arguments:
        W1    -- weight matrix (n_h, n_x)
        W2    -- weight matrix (n_y, n_h)
        cache -- Cache from forward_propagation (Z1, A1, Z2, A2)
        X     -- input data, shape (n_x, m)
        Y     -- true labels (one-hot), shape (n_y, m)

    Returns:
        grads -- Grads containing dW1, db1, dW2, db2
    */

    let m = X.cols as f64;

    // softmax + cross-entropy gradient
    let dZ2 = cache.A2.subtract(Y);
    let dW2 = dZ2.dot(&cache.A1.transpose()).scalar_mul(1.0 / m);
    let db2 = dZ2.sum_axis(1).scalar_mul(1.0 / m);

    // ReLU backprop
    let dZ1 = W2.transpose().dot(&dZ2).mul_elementwise(&ReLU::relu_grad(&cache.Z1));
    let dW1 = dZ1.dot(&X.transpose()).scalar_mul(1.0 / m);
    let db1 = dZ1.sum_axis(1).scalar_mul(1.0 / m);

    Grads { dW1, db1, dW2, db2 }
}
