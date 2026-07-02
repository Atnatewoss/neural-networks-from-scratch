pub mod download;
pub mod parse;
pub mod preprocess;

use crate::config::Config;
use crate::utils::matrix::Matrix;

pub struct MnistData {
    pub train_images: Matrix,
    pub train_labels: Matrix,
    pub val_images: Matrix,
    pub val_labels: Matrix,
    pub test_images: Matrix,
    pub test_labels: Matrix,
}

// ---------------------------------------------------------------------------
// Public API
//
// MNIST has two official splits: 60k training + 10k test.
// The 10k validation set is derived by splitting the training set
// (50k train / 10k val) after a random shuffle.
// ---------------------------------------------------------------------------

pub fn load(config: &Config) -> MnistData {
    /*
    Download (if needed), parse, and preprocess MNIST.
    config.val_size controls how many training samples are held out
    for validation.
    Returns a MnistData struct with train/val/test splits.
    */

    download::download_if_needed();

    let dir = download::data_dir();

    // ---- read raw bytes ---------------------------------------------------
    println!("Loading MNIST training images...");
    let (raw_train_imgs, num_train, rows, cols) = parse::read_raw_images(&dir.join(download::filenames()[0]));
    let pixels_per_image = rows * cols;
    println!("  raw: {} images, {}x{} = {} pixels each", num_train, rows, cols, pixels_per_image);

    println!("Loading MNIST training labels...");
    let (raw_train_labels, _) = parse::read_raw_labels(&dir.join(download::filenames()[1]));

    println!("Loading MNIST test images...");
    let (raw_test_imgs, num_test, _, _) = parse::read_raw_images(&dir.join(download::filenames()[2]));

    println!("Loading MNIST test labels...");
    let (raw_test_labels, _) = parse::read_raw_labels(&dir.join(download::filenames()[3]));

    // ---- preprocessing pipeline -------------------------------------------
    // step 1: flatten + normalise + stack -> (784, m), pixels mapped to [0, 1]
    println!("\nPreprocessing...");
    println!("  step 1: flatten images -> (784, m), normalise to [0, 1]");
    let train_images_full = preprocess::flatten_images(&raw_train_imgs, num_train, pixels_per_image);
    let test_images = preprocess::flatten_images(&raw_test_imgs, num_test, pixels_per_image);
    train_images_full.print_shape("  flattened train");
    test_images.print_shape("  flattened test");

    // labels: one-hot encode
    let train_labels_full = preprocess::one_hot(&raw_train_labels, num_train);
    let test_labels = preprocess::one_hot(&raw_test_labels, num_test);

    // ---- shuffle training set (MNIST is ordered by class on disk) ---------
    println!("  step 2: shuffle training set (ordered by class on disk)");
    let (train_shuffled, train_labels_shuffled) = preprocess::shuffle_cols(&train_images_full, &train_labels_full);

    // ---- split 60k train into train + validation --------------------------
    let split_idx = num_train - config.val_size;
    let (train_images, val_images) = preprocess::split_cols(&train_shuffled, split_idx);
    let (train_labels, val_labels) = preprocess::split_cols(&train_labels_shuffled, split_idx);

    println!("  step 3: split -> train ({}), val ({}), test ({})",
        train_images.cols, val_images.cols, test_images.cols);
    println!("  (MNIST official: 60k train + 10k test; val derived from train)");
    println!();

    train_images.print_shape("train_images");
    val_images.print_shape("val_images");
    test_images.print_shape("test_images");

    MnistData { train_images, train_labels, val_images, val_labels, test_images, test_labels }
}
