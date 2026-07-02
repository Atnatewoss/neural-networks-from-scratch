#![allow(non_snake_case)]

use std::fs;
use std::path::Path;
use nnfs::config::Config;
use nnfs::data::mnist;
use nnfs::nn::evaluate;
use nnfs::nn::nn_model;
use nnfs::nn::save_load;

fn main() {
    // 1. Load dataset
    let config = Config::default();
    let data = mnist::load(&config);
    let X_test = &data.test_images;
    let Y_test = &data.test_labels;

    // 2. Load saved model or train from scratch
    let model_path = Path::new("models/mnist.json");
    let (W1, b1, W2, b2) = if model_path.exists() {
        let (W1, b1, W2, b2) = save_load::load(model_path);
        println!();
        (W1, b1, W2, b2)
    } else {
        let X_train = &data.train_images;
        let Y_train = &data.train_labels;
        let X_val = &data.val_images;
        let Y_val = &data.val_labels;

        println!("Architecture: {} -> {} -> {}", X_train.rows, config.n_h, Y_train.rows);
        println!("Epochs: {}, Batch size: {}, Learning rate: {}\n",
            config.num_epochs, config.batch_size, config.learning_rate);

        // 3. Train network
        let (W1, b1, W2, b2, _costs) =
            nn_model::nn_model(X_train, Y_train, X_val, Y_val, &config);

        // 4. Save trained model
        fs::create_dir_all("models").expect("failed to create models/");
        save_load::save(model_path, &W1, &b1, &W2, &b2, config.n_h);
        println!();

        (W1, b1, W2, b2)
    };

    // 5. Evaluate on test set
    let test_acc = evaluate::accuracy(X_test, Y_test, &W1, &b1, &W2, &b2);
    println!("Test accuracy: {:.2}%", test_acc);
}
