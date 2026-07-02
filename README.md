# NN from Scratch

A minimal neural network framework built from scratch in Rust. Educational -- implements forward propagation, backpropagation, and mini-batch gradient descent without any ML framework dependencies.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (edition 2021)

### Run

```sh
cargo run --release
```

On first run, MNIST is downloaded automatically (~12 MB) and the model is trained from scratch. On subsequent runs, the saved model is loaded directly so training is skipped. Training progress is printed per epoch:

```
Epoch          Cost       Train         Val
    0      2.347987      10.21%       9.67%
```

Hyperparameters can be overridden in `src/main.rs`:

```rust
let config = Config { learning_rate: 0.1, ..Config::default() };
```

### Model Persistence

After training, the model is automatically saved to `models/mnist.json` as JSON.
On subsequent runs `main.rs` checks if this file exists — if so, it loads the
saved model instead of retraining. To use the API elsewhere:

```rust
use nnfs::nn::save_load;
use nnfs::nn::evaluate;

let (W1, b1, W2, b2) = save_load::load("models/mnist.json");
let acc = evaluate::accuracy(X_test, Y_test, &W1, &b1, &W2, &b2);
println!("Test accuracy: {:.2}%", acc);
```

## Pipeline

```
                         ┌─────────────────────┐
                         │   MNIST on disk      │
                         │  (4 IDX gzip files)  │
                         └──────────┬──────────┘
                                    │ download + decompress
                                    ▼
                         ┌─────────────────────┐
                         │   Raw u8 bytes       │
                         │  60k train + 10k test│
                         └──────────┬──────────┘
                                    │ flatten + /255 one-hot encode
                                    ▼
                         ┌─────────────────────┐
                         │   Normalised data    │
                         │  (784, m) + (10, m) │
                         └──────────┬──────────┘
                                    │ shuffle + split
                                    ▼
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  (784, 50000)               (784, 10000)              (784, 10000)
   train_images               val_images                test_images
  + (10, 50000)              + (10, 10000)             + (10, 10000)
   train_labels               val_labels                test_labels
         │
         │ mini-batch SGD (batch_size=64)
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    Training Loop (per epoch)                 │
  │                                                             │
  │  1. Shuffle training set (Fisher-Yates)                     │
  │  2. For each mini-batch:                                    │
  │       ┌──────────────────┐                                  │
  │       │ Forward prop      │  X → Z1 → ReLU → A1 →          │
  │       │                   │  Z2 → softmax → A2             │
  │       └────────┬─────────┘                                  │
  │                ▼                                            │
  │       ┌──────────────────┐                                  │
  │       │ Compute cost     │  CE = -1/m Σ Y·ln(A2)           │
  │       └────────┬─────────┘                                  │
  │                ▼                                            │
  │       ┌──────────────────┐                                  │
  │       │ Backward prop    │  dZ2 = A2 - Y                    │
  │       │                  │  dW2 = (1/m)·dZ2·A1ᵀ            │
  │       │                  │  dZ1 = W2ᵀ·dZ2 ⊙ ReLU'(Z1)     │
  │       │                  │  dW1 = (1/m)·dZ1·Xᵀ             │
  │       └────────┬─────────┘                                  │
  │                ▼                                            │
  │       ┌──────────────────┐                                  │
  │       │ Parameter update │  W ← W - lr · dW                │
  │       └──────────────────┘                                  │
  │                                                             │
  │  3. Evaluate: accuracy on train / val                       │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────────────────────┐
                          │  models/mnist.json exists?          │
                          │  ├── yes → load model               │
                          │  └── no  → train → save model      │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
                               ┌─────────────────────┐
                               │  Evaluate on test   │
                               └──────────┬──────────┘
                                          │ predict: forward(A2) → argmax
                                          ▼
                               ┌─────────────────────┐
                               │   Predicted digit   │
                               │       (0-9)          │
                               └─────────────────────┘
```

## Architecture

```
784 (input) -> 128 (ReLU hidden) -> 10 (softmax output)
```

- He (Kaiming) weight initialization
- ReLU activation (hidden layer)
- Softmax + categorical cross-entropy (output layer)
- Mini-batch SGD optimizer

## Dataset

**MNIST** -- handwritten digit classification (0-9).
- 60,000 training images (split: 50k train / 10k validation)
- 10,000 test images
- 28x28 grayscale pixels (784 features), normalized to [0, 1]

Downloaded automatically on first run via HTTPS (ureq + flate2) and cached in `data/`.

## Project Structure

```
src/
  main.rs              -- entry point: loads MNIST, loads saved model or trains, evaluates
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
    predict.rs              -- inference (forward + argmax)
    save_load.rs            -- JSON serialization (save / load)
    evaluate.rs             -- accuracy computation
    layer_size.rs           -- dimension extraction
  utils/
    mod.rs             -- utils module declaration
    matrix.rs          -- Matrix struct (dot, transpose, add, map, ...)
    ReLU.rs            -- ReLU activation (scalar and matrix)
    softmax.rs         -- stable softmax (max subtraction)
  data/
    .gitkeep           -- dataset directory
  models/
    .gitkeep           -- saved model parameters go here
    mod.rs             -- data module declaration
    mnist.rs           -- module root: MnistData struct + load() orchestration
    mnist/
      download.rs      -- HTTPS download + gzip decompression
      parse.rs         -- IDX binary parser
      preprocess.rs    -- flatten, normalise, one-hot, shuffle, split
```
