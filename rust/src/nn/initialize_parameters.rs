use rand::Rng;
use crate::utils::matrix::Matrix;

pub fn initialize_parameters(n_x: usize, n_h: usize, n_y: usize) -> (Matrix, Matrix, Matrix, Matrix) {
    /*
    Arguments:
        n_x -- size of the input layer (784 for MNIST)
        n_h -- size of the hidden layer (128)
        n_y -- size of the output layer (10)

    Returns:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    */

    let mut rng = rand::thread_rng();

    // He (Kaiming) initialization for ReLU: uniform [-sqrt(6/n_in), sqrt(6/n_in)]
    let he_scale1 = (6.0 / n_x as f64).sqrt();
    let w1_data: Vec<f64> = (0..n_h * n_x).map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * he_scale1).collect();
    let W1 = Matrix::from_vec(w1_data, n_h, n_x);

    let b1 = Matrix::zeros(n_h, 1);

    // He init for output layer as well: uniform [-sqrt(6/n_h), sqrt(6/n_h)]
    let he_scale2 = (6.0 / n_h as f64).sqrt();
    let w2_data: Vec<f64> = (0..n_y * n_h).map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * he_scale2).collect();
    let W2 = Matrix::from_vec(w2_data, n_y, n_h);

    let b2 = Matrix::zeros(n_y, 1);

    (W1, b1, W2, b2)
}
