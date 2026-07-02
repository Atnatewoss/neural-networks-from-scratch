# NN from Scratch

A minimal neural network framework built from scratch in Rust.
Educational -- implements forward propagation, backpropagation, and mini-batch gradient descent without any ML framework dependencies.

## Dataset

**MNIST** -- handwritten digit classification (0-9).
- 60,000 training images (split: 50k train / 10k validation)
- 10,000 test images
- 28x28 grayscale pixels (784 features), normalized to [0, 1]

Downloaded automatically on first run via HTTPS (ureq + flate2).

## Architecture

```
784 (input) -> 128 (ReLU hidden) -> 10 (softmax output)
```

- He (Kaiming) weight initialization
- ReLU activation (hidden layer)
- Softmax + categorical cross-entropy (output layer)
- Mini-batch SGD optimizer

## Usage

```
cargo run --release
```

Training progress is printed per epoch:

```
Epoch          Cost       Train         Val        Test
    0      2.347987      10.21%       9.67%       9.98%
```

Hyperparameters are set in `src/config.rs` and can be overridden in `src/main.rs`:

```rust
let config = Config { learning_rate: 0.1, ..Config::default() };
```

## Project Structure

```
src/
  main.rs              -- entry point, loads MNIST, runs training
  lib.rs               -- crate root, module declarations
  config.rs            -- central Config struct (hyperparameters + defaults)
  nn/
    mod.rs             -- nn module declaration
    nn_model.rs        -- training loop (shuffle, batch, forward/backward/update)
    forward_propagation.rs  -- feed-forward: Z = WX + b, ReLU, softmax
    backward_propagation.rs -- gradient computation (dW, db)
    compute_cost.rs         -- categorical cross-entropy
    initialize_parameters.rs -- He initialization
    update_parameters.rs    -- SGD weight update
    predict.rs              -- inference (argmax over softmax)
    evaluate.rs             -- accuracy computation
    layer_size.rs           -- dimension extraction
  utils/
    mod.rs             -- utils module declaration
    matrix.rs          -- Matrix struct (dot, transpose, add, map, ...)
    ReLU.rs            -- ReLU activation (scalar and matrix)
    softmax.rs         -- stable softmax (max subtraction)
  data/
    mod.rs             -- data module declaration
    mnist.rs           -- module root: MnistData struct + load() orchestration
    mnist/
      download.rs      -- HTTPS download + gzip decompression
      parse.rs         -- IDX binary parser
      preprocess.rs    -- flatten, normalise, one-hot, shuffle, split
```
