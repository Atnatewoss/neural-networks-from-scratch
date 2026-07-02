use crate::utils::matrix::Matrix;
use crate::utils::ReLU;
use crate::utils::softmax;

pub fn predict(X: &Matrix, W1: &Matrix, b1: &Matrix, W2: &Matrix, b2: &Matrix) -> Matrix {
    /*
    Arguments:
        X  -- input data, shape (n_x, m)
        W1 -- weight matrix (n_h, n_x)
        b1 -- bias vector (n_h, 1)
        W2 -- weight matrix (n_y, n_h)
        b2 -- bias vector (n_y, 1)

    Returns:
        predictions -- class predictions (0-9) as row vector, shape (1, m)
    */

    let Z1 = W1.dot(X).add(b1);
    let A1 = ReLU::relu(&Z1);
    let Z2 = W2.dot(&A1).add(b2);
    let A2 = softmax::forward(&Z2);

    let pred_indices = A2.argmax_columns();
    let mut data = vec![0.0; pred_indices.len()];
    for (j, &p) in pred_indices.iter().enumerate() {
        data[j] = p as f64;
    }
    Matrix::from_vec(data, 1, pred_indices.len())
}
