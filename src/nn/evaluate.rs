use crate::utils::matrix::Matrix;
use crate::nn::forward_propagation;

/*
Predict class labels and compute accuracy against ground truth.

Standard approach: forward propagation → softmax → argmax → compare.
Softmax produces a probability distribution over classes; argmax
selects the most likely class; accuracy is the fraction correct.

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
    // Step 1: Forward propagation (includes softmax) → class probabilities A2
    let (A2, _) = forward_propagation::forward_propagation(X, W1, b1, W2, b2);

    // Step 2: Argmax → predicted class index for each example
    let pred_indices = A2.argmax_columns();

    // Step 3: Count correct predictions
    let m = X.cols;
    let mut correct = 0.0;
    for j in 0..m {
        if Y.data[pred_indices[j] * Y.cols + j] > 0.5 {
            correct += 1.0;
        }
    }

    // Step 4: Return accuracy as percentage
    correct / m as f64 * 100.0
}
