#![allow(non_snake_case)]

use std::fs;
use std::path::Path;
use nnfs::config::Config;
use nnfs::data::mnist;
use nnfs::nn::evaluate;
use nnfs::nn::nn_model;
use nnfs::nn::save_load;

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
        nn_model::nn_model(X_train, Y_train, X_val, Y_val, &config);

    let test_acc = evaluate::accuracy(X_test, Y_test, &W1, &b1, &W2, &b2);
    println!("\nTest accuracy: {:.2}%\n", test_acc);

    let model_dir = Path::new("models");
    fs::create_dir_all(model_dir).expect("failed to create models/");
    let model_path = model_dir.join("mnist.json");
    save_load::save(&model_path, &W1, &b1, &W2, &b2, config.n_h);

    // round-trip: load back and verify accuracy matches
    let (W1_l, b1_l, W2_l, b2_l, _n_h_l) = save_load::load(&model_path);
    let loaded_acc = evaluate::accuracy(X_test, Y_test, &W1_l, &b1_l, &W2_l, &b2_l);
    println!("Reloaded accuracy: {:.2}%  (match: {})", loaded_acc,
        if (test_acc - loaded_acc).abs() < 0.01 { "yes" } else { "no" });
}
