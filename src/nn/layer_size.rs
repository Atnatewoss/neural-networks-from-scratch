use crate::utils::matrix::Matrix;

pub fn layer_size(X: &Matrix, Y: &Matrix, n_h: usize) -> (usize, usize, usize) {
    /*
    Arguments:
        X -- input data of shape (input size or number of features, number of examples)
        Y -- true labels shape (output size, number of examples)
        n_h -- size of hidden layer
    
    Returns:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
    */

    let n_x = X.rows;
    let n_y = Y.rows;
    (n_x, n_h, n_y)
}
