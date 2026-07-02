use rand::Rng;

use crate::utils::matrix::Matrix;

// Preprocessing pipeline

/*
Step 1: flatten each image into a column vector and stack into (features, m).
For MNIST: each image is 28x28 = 784 pixels.
The output matrix has shape (pixels_per_image, num_images) so that each column
is one flattened sample -- this is the convention used by the nn module.
*/
pub fn flatten_images(raw: &[u8], num_images: usize, pixels_per_image: usize) -> Matrix {
    /*
    Arguments:
        raw              -- raw pixel bytes, length = num_images * pixels_per_image
        num_images       -- number of samples (m)
        pixels_per_image -- features per sample (e.g. 784 for MNIST)

    Returns:
        Matrix of shape (pixels_per_image, num_images)
    */

    let mut data = Vec::with_capacity(num_images * pixels_per_image);
    for i in 0..num_images {
        let offset = i * pixels_per_image;
        for j in 0..pixels_per_image {
            data.push(raw[offset + j] as f64 / 255.0);
        }
    }
    Matrix::from_vec(data, pixels_per_image, num_images)
}

/*
One-hot encode labels: class label k -> row vector with 1.0 at position k.
*/
pub fn one_hot(raw: &[u8], num_labels: usize) -> Matrix {
    /*
    Arguments:
        raw        -- raw label bytes, each byte is the class index (0-9)
        num_labels -- number of samples

    Returns:
        Matrix of shape (10, num_labels), one column per sample
    */

    let mut data = vec![0.0; num_labels * 10];
    for i in 0..num_labels {
        let label = raw[i] as usize;
        data[i * 10 + label] = 1.0;
    }
    Matrix::from_vec(data, 10, num_labels)
}

// ---------------------------------------------------------------------------
// Shuffle before split
//
// MNIST on disk is ordered by class (all 0s, then 1s, then 2s, ...).
// Splitting without shuffling produces a biased validation set.
// We must apply the same random column permutation to both images and labels.
// ---------------------------------------------------------------------------

/*
Apply the same random column permutation to X and Y.
Returns (shuffled_X, shuffled_Y) with shapes unchanged.
*/
pub fn shuffle_cols(X: &Matrix, Y: &Matrix) -> (Matrix, Matrix) {
    /*
    Arguments:
        X -- input matrix, shape (n_x, m)
        Y -- label matrix, shape (n_y, m)

    Returns:
        (shuffled_X, shuffled_Y) -- same shapes, columns permuted identically
    */

    let m = X.cols;
    let mut perm: Vec<usize> = (0..m).collect();
    let mut rng = rand::thread_rng();
    for i in (1..m).rev() {
        let j = rng.gen_range(0..=i);
        perm.swap(i, j);
    }

    let mut Xs = Matrix::new(X.rows, m);
    let mut Ys = Matrix::new(Y.rows, m);
    for k in 0..m {
        let src = perm[k];
        for i in 0..X.rows {
            Xs.data[i * m + k] = X.data[i * m + src];
        }
        for i in 0..Y.rows {
            Ys.data[i * m + k] = Y.data[i * m + src];
        }
    }
    (Xs, Ys)
}

// ---------------------------------------------------------------------------
// Splitting helpers
// ---------------------------------------------------------------------------

/*
Split columns [0..split) as left, [split..cols) as right.
*/
pub fn split_cols(m: &Matrix, split: usize) -> (Matrix, Matrix) {
    let left_cols = split;
    let right_cols = m.cols - split;

    let mut left = Matrix::new(m.rows, left_cols);
    let mut right = Matrix::new(m.rows, right_cols);

    for i in 0..m.rows {
        for j in 0..left_cols {
            left.data[i * left_cols + j] = m.data[i * m.cols + j];
        }
        for j in 0..right_cols {
            right.data[i * right_cols + j] = m.data[i * m.cols + (split + j)];
        }
    }

    (left, right)
}
