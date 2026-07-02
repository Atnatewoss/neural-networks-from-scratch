# Trained Models

Trained model parameters are saved here as JSON files (e.g. `mnist.json`) after running `cargo run --release`.

Each file contains the weight matrices (W1, W2), bias vectors
(b1, b2), and the hidden-layer size (n_h) serialised via
serde_json.

These `.json` files are **not tracked by Git** — they change
every retraining and can be large. To share a checkpoint,
upload the file separately or copy it out of the directory.

To use a saved model:

```rust
use nnfs::nn::save_load;
let (W1, b1, W2, b2, n_h) = save_load::load("models/mnist.json");
```
