use crate::utils::matrix::Matrix;

pub fn compute_cost(A2: &Matrix, Y: &Matrix) -> f64 {
    /*
    Arguments:
        A2 -- predicted probabilities, shape (n_y, m)
        Y  -- true labels (one-hot), shape (n_y, m)

    Returns:
        cost -- categorical cross-entropy cost
    */

    let m = A2.cols as f64;
    let mut total = 0.0;
    for i in 0..A2.data.len() {
        total += Y.data[i] * A2.data[i].ln();
    }
    -total / m
}
