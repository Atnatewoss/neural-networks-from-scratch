use crate::utils::matrix::Matrix;
use crate::utils::ReLU;
use crate::utils::softmax;

pub struct Cache {
    pub Z1: Matrix,
    pub A1: Matrix,
    pub Z2: Matrix,
    pub A2: Matrix,
}

pub fn forward_propagation(
    X: &Matrix,
    W1: &Matrix,
    b1: &Matrix,
    W2: &Matrix,
    b2: &Matrix,
) -> (Matrix, Cache) {
    /*
    Arguments:
        X  -- input data of shape (n_x, m)
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)

    Returns:
        A2 -- softmax output, shape (n_y, m)
        cache -- Cache containing Z1, A1, Z2, A2 for backprop
    */

    let Z1 = W1.dot(X).add(b1);
    let A1 = ReLU::relu(&Z1);
    let Z2 = W2.dot(&A1).add(b2);
    let A2 = softmax::forward(&Z2);

    let cache = Cache {
        Z1,
        A1,
        Z2,
        A2: A2.clone(),
    };

    (A2, cache)
}
