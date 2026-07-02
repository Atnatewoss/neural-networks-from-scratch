#![allow(non_snake_case)]

use nnfs::config::Config;
use nnfs::data::mnist;
use nnfs::nn::evaluate;
use nnfs::nn::nn_model;

fn main() {
    let config = Config::default();
    let data = mnist::load(&config);

    let X_train = &data.train_images;
    let Y_train = &data.train_labels;
    let X_val = &data.val_images;
    let Y_val = &data.val_labels;
    let X_test = &data.test_images;
    let Y_test = &data.test_labels;

    println!("Architecture: {} -> {} -> {}", X_train.rows, config.n_h, Y_train.rows);
    println!("Epochs: {}, Batch size: {}, Learning rate: {}\n",
        config.num_epochs, config.batch_size, config.learning_rate);

    let (W1, b1, W2, b2, _costs) =
        nn_model::nn_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, &config);

    let test_acc = evaluate::accuracy(X_test, Y_test, &W1, &b1, &W2, &b2);
    println!("\nTest accuracy: {:.2}%", test_acc);
}
