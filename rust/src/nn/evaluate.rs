use crate::utils::matrix::Matrix;
use crate::nn::predict;

/*
Arguments:
    X  -- input data, shape (n_x, m)
    Y  -- true labels (one-hot), shape (n_y, m)
    W1 -- weight matrix (n_h, n_x)
    b1 -- bias vector (n_h, 1)
    W2 -- weight matrix (n_y, n_h)
    b2 -- bias vector (n_y, 1)

Returns:
    acc -- percentage of correct predictions (0.0 to 100.0)
*/
pub fn accuracy(X: &Matrix, Y: &Matrix, W1: &Matrix, b1: &Matrix, W2: &Matrix, b2: &Matrix) -> f64 {
    let preds = predict::predict(X, W1, b1, W2, b2);
    let m = X.cols;
    let mut correct = 0.0;
    for j in 0..m {
        if Y.data[preds.data[j] as usize * Y.cols + j] > 0.5 {
            correct += 1.0;
        }
    }
    correct / m as f64 * 100.0
}
