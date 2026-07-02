use rand::Rng;
use crate::config::Config;
use crate::nn::backward_propagation;
use crate::nn::compute_cost;
use crate::nn::evaluate;
use crate::nn::forward_propagation;
use crate::nn::initialize_parameters;
use crate::nn::update_parameters;
use crate::utils::matrix::Matrix;

pub fn nn_model(
    X_train: &Matrix,
    Y_train: &Matrix,
    X_val: &Matrix,
    Y_val: &Matrix,
    config: &Config,
) -> (Matrix, Matrix, Matrix, Matrix, Vec<f64>) {
    /*
    Arguments:
        X_train -- training input, shape (n_x, m_train)
        Y_train -- training labels (one-hot), shape (n_y, m_train)
        X_val   -- validation input, shape (n_x, m_val)
        Y_val   -- validation labels (one-hot), shape (n_y, m_val)
        config  -- Config with n_h, learning_rate, num_epochs, batch_size, print_cost

    Returns:
        W1, b1, W2, b2 -- trained parameters
        costs         -- average cross-entropy cost per epoch
    */

    let n_x = X_train.rows;
    let n_y = Y_train.rows;
    let m = X_train.cols;

    let (mut W1, mut b1, mut W2, mut b2) = initialize_parameters::initialize_parameters(n_x, config.n_h, n_y);

    println!("{:>5}  {:>12}  {:>10}  {:>10}", "Epoch", "Cost", "Train", "Val");

    let mut rng = rand::thread_rng();
    let mut costs = Vec::new();

    for epoch in 0..config.num_epochs {
        let mut indices: Vec<usize> = (0..m).collect();
        for i in (1..m).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let mut epoch_cost = 0.0;
        let mut num_batches = 0;

        for batch_start in (0..m).step_by(config.batch_size) {
            let batch_end = std::cmp::min(batch_start + config.batch_size, m);
            let batch_size_actual = batch_end - batch_start;

            let mut X_batch = Matrix::new(n_x, batch_size_actual);
            let mut Y_batch = Matrix::new(n_y, batch_size_actual);
            for (k, &idx) in indices[batch_start..batch_end].iter().enumerate() {
                for i in 0..n_x {
                    X_batch.data[i * batch_size_actual + k] = X_train.data[i * m + idx];
                }
                for i in 0..n_y {
                    Y_batch.data[i * batch_size_actual + k] = Y_train.data[i * m + idx];
                }
            }

            let (A2, cache) = forward_propagation::forward_propagation(
                &X_batch, &W1, &b1, &W2, &b2,
            );

            let cost = compute_cost::compute_cost(&A2, &Y_batch);
            epoch_cost += cost;
            num_batches += 1;

            let grads = backward_propagation::backward_propagation(
                &W1, &W2, &cache, &X_batch, &Y_batch,
            );

            update_parameters::update_parameters(
                &mut W1, &mut b1, &mut W2, &mut b2,
                &grads, config.learning_rate,
            );
        }

        let avg_cost = epoch_cost / num_batches as f64;
        costs.push(avg_cost);

        if config.print_cost {
            let train_acc = evaluate::accuracy(X_train, Y_train, &W1, &b1, &W2, &b2);
            let val_acc = evaluate::accuracy(X_val, Y_val, &W1, &b1, &W2, &b2);
            println!("{:>5}  {:>12.6}  {:>9.2}%  {:>9.2}%",
                epoch, avg_cost, train_acc, val_acc);
        }
    }

    (W1, b1, W2, b2, costs)
}
