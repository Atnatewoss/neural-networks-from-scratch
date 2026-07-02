use crate::utils::matrix::Matrix;
use crate::nn::backward_propagation::Grads;

pub fn update_parameters(
    W1: &mut Matrix,
    b1: &mut Matrix,
    W2: &mut Matrix,
    b2: &mut Matrix,
    grads: &Grads,
    learning_rate: f64,
) {
    /*
    Arguments:
        W1, b1, W2, b2 -- parameters (mutated in place)
        grads -- Grads containing dW1, db1, dW2, db2
        learning_rate -- step size for gradient descent
    */

    for i in 0..W1.data.len() {
        W1.data[i] -= learning_rate * grads.dW1.data[i];
    }
    for i in 0..b1.data.len() {
        b1.data[i] -= learning_rate * grads.db1.data[i];
    }
    for i in 0..W2.data.len() {
        W2.data[i] -= learning_rate * grads.dW2.data[i];
    }
    for i in 0..b2.data.len() {
        b2.data[i] -= learning_rate * grads.db2.data[i];
    }
}
